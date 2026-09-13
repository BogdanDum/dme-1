"""Parse and check the Google hardware surface-code experiments.

Each experiment directory is one configuration; each of its 50,000 shots is one
execution of the circuit. A shot only exists as the *alignment* of several
companion files: record ``n`` of ``measurements.b8`` belongs with record ``n`` of
``detection_events.b8`` and line ``n+1`` of ``obs_flips_actual.01``. Nothing in
the files themselves records that correspondence, so every companion length is
validated before any row is published.

Packed bits stay packed. ``measurement_bits``, ``sweep_bits`` and
``detector_bits`` are stored as the supplied Stim ``b8`` bytes; unpacking them
into one column per bit is Part II's problem and would multiply the data by
eight.

Five concepts are kept apart, because conflating them is an explicit grade cap:
a measurement bit, a detector event, an actual logical flip, a decoder
prediction, and a decoder *mistake*. Only the first four are stored. A mistake is
``prediction != actual``, which is derived, so it is computed for evidence and
never written to Silver as if it were source data.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import pyarrow as pa
import yaml

from .. import ids
from ..formats import b8_record_bytes, iter_b8_records, parse_01_records
from ..ml import GOOGLE_META_PREDICTION_COLUMNS
from ..runcontext import RunContext
from ..schemas import DECODER_PREDICTION_COLUMNS, GOOGLE_EXPERIMENT, GOOGLE_SHOT

SOURCE_NAME = "google_qec"
SHOT_BATCH = 25_000

# The decoder file names and the Silver column names come from the same tuple, so
# Silver, Gold and the ML export cannot drift apart.
DECODERS = tuple(
    column.removesuffix("_prediction") for column in GOOGLE_META_PREDICTION_COLUMNS
)

PROPERTIES_MEMBER = "properties.yml"
MEASUREMENTS_MEMBER = "measurements.b8"
SWEEP_MEMBER = "sweep.b8"
DETECTORS_MEMBER = "detection_events.b8"
ACTUAL_MEMBER = "obs_flips_actual.01"
PREDICTION_MEMBERS = tuple(
    f"obs_flips_predicted_by_{decoder}.01" for decoder in DECODERS
)

# Every member a shot is assembled from, in sorted order for stable tracing.
SHOT_MEMBERS = tuple(
    sorted(
        (
            MEASUREMENTS_MEMBER,
            SWEEP_MEMBER,
            DETECTORS_MEMBER,
            ACTUAL_MEMBER,
            *PREDICTION_MEMBERS,
        )
    )
)
REQUIRED_MEMBERS = (PROPERTIES_MEMBER, *SHOT_MEMBERS)

DIRECTORY = re.compile(
    r"^surface_code_b(?P<basis>[A-Z])_d(?P<distance>\d+)_r(?P<rounds>\d+)"
    r"_center_(?P<row>\d+)_(?P<col>\d+)$"
)


@dataclass(frozen=True)
class Properties:
    """The declared shape of one experiment, read from ``properties.yml``."""

    basis: str
    rounds: int
    distance: int
    data_qubits: int
    measure_qubits: int
    shots: int
    center_row: int
    center_col: int
    measurement_count: int
    sweep_bits: int
    detector_count: int
    observables: int
    qubits: int

    @classmethod
    def from_yaml(cls, text: str) -> Properties:
        document = yaml.safe_load(text) or {}
        return cls(
            basis=str(document["basis"]),
            rounds=int(document["rounds"]),
            distance=int(document["distance"]),
            data_qubits=int(document["data_qubits"]),
            measure_qubits=int(document["measure_qubits"]),
            shots=int(document["shots"]),
            center_row=int(document["center_data_qubit_row"]),
            center_col=int(document["center_data_qubit_col"]),
            measurement_count=int(document["circuit_measurements"]),
            sweep_bits=int(document["circuit_sweep_bits"]),
            detector_count=int(document["circuit_detectors"]),
            observables=int(document["circuit_observables"]),
            qubits=int(document["circuit_qubits"]),
        )


@dataclass
class ExperimentStats:
    experiment_id: str
    shots_read: int = 0
    shots_accepted: int = 0
    shots_rejected: int = 0
    detector_count: int = 0
    measurement_count: int = 0
    sweep_bits: int = 0
    detector_events_total: int = 0
    actual_flip_count: int = 0
    decoder_mistakes: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "shots_read": self.shots_read,
            "shots_accepted": self.shots_accepted,
            "shots_rejected": self.shots_rejected,
            "detector_count": self.detector_count,
            "measurement_count": self.measurement_count,
            "sweep_bits": self.sweep_bits,
            "detector_events_total": self.detector_events_total,
            "actual_flip_count": self.actual_flip_count,
            "decoder_mistakes": dict(sorted(self.decoder_mistakes.items())),
        }


@dataclass
class GoogleOutcome:
    experiment_rows: int = 0
    shot_rows: int = 0
    per_experiment: dict[str, dict[str, object]] = field(default_factory=dict)


def padding_mask(bits_per_record: int) -> int:
    """Byte mask selecting the unused high bits of a record's final byte.

    Bits inside a byte are little-endian, so the valid bits of the last byte are
    the low ones and any padding sits above them. Returns 0 when the record is a
    whole number of bytes, as it is for 200 and 600 detectors.
    """
    remainder = bits_per_record % 8
    return 0 if remainder == 0 else (0xFF << remainder) & 0xFF


def count_set_bits(records: np.ndarray, bits_per_record: int) -> np.ndarray:
    """Number of set bits per record, honouring little-endian bit order."""
    unpacked = np.unpackbits(
        records, axis=1, bitorder="little", count=bits_per_record
    )
    return unpacked.sum(axis=1, dtype=np.int64)


def build(
    context: RunContext, key: str, bronze_object: str, sha256: str
) -> GoogleOutcome:
    """Parse every experiment directory and stage both Google tables."""
    outcome = GoogleOutcome()

    with context.lake.open_bronze_zip(key) as archive:
        names = set(archive.namelist())
        directories = _discover(names)

        described = []
        for directory in directories:
            properties = _load_properties(context, archive, directory, names)
            described.append((directory, properties))

        experiment_rows = [
            _experiment_row(directory, properties)
            for directory, properties in described
        ]
        for row in experiment_rows:
            context.trace.add(
                source_record_id=row["source_record_id"],
                source_name=SOURCE_NAME,
                bronze_object=bronze_object,
                archive_member=f"{row['experiment_id']}/{PROPERTIES_MEMBER}",
                record_locator=ids.member_locator(PROPERTIES_MEMBER),
                input_sha256=sha256,
            )
        table = pa.table(
            {
                name: [row[name] for row in experiment_rows]
                for name in GOOGLE_EXPERIMENT.column_names
            },
            schema=GOOGLE_EXPERIMENT.schema,
        )
        outcome.experiment_rows = context.lake.stage_table(
            GOOGLE_EXPERIMENT, table
        ).row_count
        context.quality.count(
            GOOGLE_EXPERIMENT.name,
            read=len(directories),
            accepted=outcome.experiment_rows,
        )

        outcome.shot_rows = context.lake.stage_batches(
            GOOGLE_SHOT,
            _iter_shot_batches(
                context,
                archive,
                described,
                bronze_object=bronze_object,
                sha256=sha256,
                outcome=outcome,
            ),
        ).row_count

    total = sum(
        int(entry["shots_read"]) for entry in outcome.per_experiment.values()
    )
    rejected = sum(
        int(entry["shots_rejected"]) for entry in outcome.per_experiment.values()
    )
    context.quality.count(
        GOOGLE_SHOT.name,
        read=total,
        accepted=outcome.shot_rows,
        rejected=rejected,
    )
    return outcome


def _discover(names: set[str]) -> tuple[str, ...]:
    directories = {
        name.split("/", 1)[0]
        for name in names
        if "/" in name and DIRECTORY.match(name.split("/", 1)[0])
    }
    return tuple(sorted(directories))


def _load_properties(
    context: RunContext, archive, directory: str, names: set[str]
) -> Properties:
    for member in REQUIRED_MEMBERS:
        if f"{directory}/{member}" not in names:
            context.quality.fatal(
                "google_qec.experiment.companion_file_missing",
                source_name=SOURCE_NAME,
                record_locator=f"{directory}/{member}",
                observed_value=member,
                reason=(
                    f"{directory} is missing required companion file {member}. "
                    "Shots cannot be aligned, so the run stops rather than "
                    "publishing a partial experiment."
                ),
            )

    try:
        properties = Properties.from_yaml(
            archive.read(f"{directory}/{PROPERTIES_MEMBER}").decode("utf-8")
        )
    except (yaml.YAMLError, KeyError, TypeError, ValueError) as error:
        context.quality.fatal(
            "google_qec.experiment.properties_unparsable",
            source_name=SOURCE_NAME,
            record_locator=f"{directory}/{PROPERTIES_MEMBER}",
            observed_value=str(error),
            reason=f"properties.yml for {directory} is unusable: {error}",
        )
        raise  # unreachable; fatal() raises

    _check_directory_agreement(context, directory, properties)
    _check_derived_invariants(context, directory, properties)
    return properties


def _check_directory_agreement(
    context: RunContext, directory: str, properties: Properties
) -> None:
    matched = DIRECTORY.match(directory)
    if not matched:
        return
    expected = {
        "basis": matched["basis"],
        "distance": int(matched["distance"]),
        "rounds": int(matched["rounds"]),
        "center_row": int(matched["row"]),
        "center_col": int(matched["col"]),
    }
    actual = {
        "basis": properties.basis,
        "distance": properties.distance,
        "rounds": properties.rounds,
        "center_row": properties.center_row,
        "center_col": properties.center_col,
    }
    differences = {
        field_name: (value, actual[field_name])
        for field_name, value in expected.items()
        if actual[field_name] != value
    }
    if differences:
        context.quality.record(
            "google_qec.experiment.directory_name_mismatch",
            source_name=SOURCE_NAME,
            record_locator=f"{directory}/{PROPERTIES_MEMBER}",
            observed_value="; ".join(
                f"{name}: directory={pair[0]} properties={pair[1]}"
                for name, pair in sorted(differences.items())
            ),
            reason=(
                "The directory name and properties.yml disagree. properties.yml "
                "is treated as authoritative and no value was changed."
            ),
        )


def _check_derived_invariants(
    context: RunContext, directory: str, properties: Properties
) -> None:
    """Confirm the counts in properties.yml are mutually consistent."""
    expectations = (
        (
            "circuit_detectors == rounds * measure_qubits",
            properties.detector_count,
            properties.rounds * properties.measure_qubits,
        ),
        (
            "circuit_measurements == rounds * measure_qubits + data_qubits",
            properties.measurement_count,
            properties.rounds * properties.measure_qubits + properties.data_qubits,
        ),
        (
            "circuit_sweep_bits == data_qubits",
            properties.sweep_bits,
            properties.data_qubits,
        ),
        (
            "data_qubits == distance ** 2",
            properties.data_qubits,
            properties.distance**2,
        ),
        (
            "measure_qubits == distance ** 2 - 1",
            properties.measure_qubits,
            properties.distance**2 - 1,
        ),
    )
    for description, observed, expected in expectations:
        if observed != expected:
            context.quality.record(
                "google_qec.experiment.derived_invariant_mismatch",
                source_name=SOURCE_NAME,
                record_locator=f"{directory}/{PROPERTIES_MEMBER}",
                observed_value=f"{description}: {observed} != {expected}",
                reason=(
                    f"{description} does not hold for {directory}; the declared "
                    "counts are used as given and the discrepancy is reported."
                ),
            )


def _experiment_row(directory: str, properties: Properties) -> dict[str, object]:
    return {
        "source_record_id": ids.google_experiment_record_id(directory),
        "experiment_id": directory,
        "basis": properties.basis,
        "distance": properties.distance,
        "rounds": properties.rounds,
        "shots": properties.shots,
        "center_row": properties.center_row,
        "center_col": properties.center_col,
        "measurement_count": properties.measurement_count,
        "detector_count": properties.detector_count,
    }


def _iter_shot_batches(
    context: RunContext,
    archive,
    described: list[tuple[str, Properties]],
    *,
    bronze_object: str,
    sha256: str,
    outcome: GoogleOutcome,
) -> Iterator[pa.RecordBatch]:
    """Yield shot batches in ``(experiment_id, shot_index)`` order.

    Experiments are visited in sorted order and shots in ascending index, so the
    stream already matches the table's declared sort key and never needs a
    post-sort of 250,000 rows.
    """
    for directory, properties in described:
        yield from _iter_experiment_shots(
            context,
            archive,
            directory,
            properties,
            bronze_object=bronze_object,
            sha256=sha256,
            outcome=outcome,
        )


def _iter_experiment_shots(
    context: RunContext,
    archive,
    directory: str,
    properties: Properties,
    *,
    bronze_object: str,
    sha256: str,
    outcome: GoogleOutcome,
) -> Iterator[pa.RecordBatch]:
    shots = properties.shots
    stats = ExperimentStats(
        experiment_id=directory,
        shots_read=shots,
        detector_count=properties.detector_count,
        measurement_count=properties.measurement_count,
        sweep_bits=properties.sweep_bits,
    )

    measurement_bytes = b8_record_bytes(properties.measurement_count)
    detector_bytes = b8_record_bytes(properties.detector_count)
    sweep_bytes = (
        b8_record_bytes(properties.sweep_bits) if properties.sweep_bits else 0
    )

    measurements = _read_b8(
        context, archive, directory, MEASUREMENTS_MEMBER, shots, measurement_bytes
    )
    detectors = _read_b8(
        context, archive, directory, DETECTORS_MEMBER, shots, detector_bytes
    )
    sweeps = _read_b8(
        context, archive, directory, SWEEP_MEMBER, shots, sweep_bytes
    )

    actual = _read_01(context, archive, directory, ACTUAL_MEMBER, shots)
    predictions = {
        column: _read_01(context, archive, directory, member, shots)
        for column, member in zip(
            DECODER_PREDICTION_COLUMNS, PREDICTION_MEMBERS, strict=True
        )
    }

    detector_grid = np.frombuffer(detectors, dtype=np.uint8).reshape(
        shots, detector_bytes
    )
    event_counts = count_set_bits(detector_grid, properties.detector_count)
    _verify_fast_path(detectors, detector_bytes, properties.detector_count, event_counts)

    excluded = _padding_failures(
        context,
        directory,
        shots,
        (
            (MEASUREMENTS_MEMBER, measurements, measurement_bytes, properties.measurement_count),
            (DETECTORS_MEMBER, detectors, detector_bytes, properties.detector_count),
            (SWEEP_MEMBER, sweeps, sweep_bytes, properties.sweep_bits),
        ),
    )

    actual_array = np.asarray(actual, dtype=np.int8)
    stats.actual_flip_count = int(actual_array.sum())
    stats.detector_events_total = int(event_counts.sum())
    for column, values in predictions.items():
        # A decoder mistake is derived, not supplied. It is counted here as
        # evidence that predictions and mistakes are different concepts, and is
        # deliberately absent from the Silver schema.
        stats.decoder_mistakes[column.removesuffix("_prediction")] = int(
            (np.asarray(values, dtype=np.int8) != actual_array).sum()
        )

    accepted_ids: list[str] = []
    accepted_locators: list[str] = []

    for start in range(0, shots, SHOT_BATCH):
        stop = min(start + SHOT_BATCH, shots)
        indices = [index for index in range(start, stop) if index not in excluded]
        if not indices:
            continue
        record_ids = [
            ids.google_shot_record_id(directory, index) for index in indices
        ]
        accepted_ids.extend(record_ids)
        accepted_locators.extend(ids.shot_locator(index) for index in indices)

        batch = pa.record_batch(
            {
                "source_record_id": pa.array(record_ids, type=pa.string()),
                "experiment_id": pa.array(
                    [directory] * len(indices), type=pa.string()
                ),
                "shot_index": pa.array(indices, type=pa.int64()),
                "measurement_bits": pa.array(
                    [
                        measurements[i * measurement_bytes : (i + 1) * measurement_bytes]
                        for i in indices
                    ],
                    type=pa.binary(),
                ),
                "sweep_bits": pa.array(
                    [
                        sweeps[i * sweep_bytes : (i + 1) * sweep_bytes]
                        if sweep_bytes
                        else b""
                        for i in indices
                    ],
                    type=pa.binary(),
                ),
                "detector_bits": pa.array(
                    [
                        detectors[i * detector_bytes : (i + 1) * detector_bytes]
                        for i in indices
                    ],
                    type=pa.binary(),
                ),
                "detector_event_count": pa.array(
                    [int(event_counts[i]) for i in indices], type=pa.int32()
                ),
                "actual_observable_flip": pa.array(
                    [bool(actual[i]) for i in indices], type=pa.bool_()
                ),
                **{
                    column: pa.array(
                        [bool(predictions[column][i]) for i in indices],
                        type=pa.bool_(),
                    )
                    for column in DECODER_PREDICTION_COLUMNS
                },
            },
            schema=GOOGLE_SHOT.schema,
        )
        stats.shots_accepted += batch.num_rows
        yield batch

    stats.shots_rejected = len(excluded)
    context.trace.add_aligned_block(
        source_record_ids=accepted_ids,
        record_locators=accepted_locators,
        source_name=SOURCE_NAME,
        bronze_object=bronze_object,
        archive_members=tuple(f"{directory}/{member}" for member in SHOT_MEMBERS),
        input_sha256=sha256,
    )
    outcome.per_experiment[directory] = stats.as_dict()


def _read_b8(
    context: RunContext,
    archive,
    directory: str,
    member: str,
    shots: int,
    record_bytes: int,
) -> bytes:
    payload = archive.read(f"{directory}/{member}")
    expected = shots * record_bytes
    if len(payload) != expected:
        context.quality.fatal(
            "google_qec.b8.record_length_mismatch",
            source_name=SOURCE_NAME,
            record_locator=f"{directory}/{member}",
            observed_value=str(len(payload)),
            reason=(
                f"{member} is {len(payload)} bytes; {shots} shots at "
                f"{record_bytes} bytes per record require {expected}. Shots "
                "cannot be aligned, so the run stops."
            ),
        )
    return payload


def _read_01(
    context: RunContext, archive, directory: str, member: str, shots: int
) -> list[int]:
    payload = archive.read(f"{directory}/{member}")
    try:
        values = parse_01_records(payload)
    except ValueError as error:
        context.quality.fatal(
            "google_qec.01.value_domain",
            source_name=SOURCE_NAME,
            record_locator=f"{directory}/{member}",
            observed_value=str(error),
            reason=f"{member} is not a binary-per-line file: {error}",
        )
        raise  # unreachable
    if len(values) != shots:
        context.quality.fatal(
            "google_qec.01.line_count_mismatch",
            source_name=SOURCE_NAME,
            record_locator=f"{directory}/{member}",
            observed_value=str(len(values)),
            reason=(
                f"{member} has {len(values)} lines but the experiment declares "
                f"{shots} shots, so outcomes cannot be aligned to shots."
            ),
        )
    return values


def _padding_failures(
    context: RunContext,
    directory: str,
    shots: int,
    members: tuple[tuple[str, bytes, int, int], ...],
) -> set[int]:
    """Find shots whose packed records set bits above the declared bit count."""
    excluded: set[int] = set()
    for member, payload, record_bytes, bits in members:
        mask = padding_mask(bits) if record_bytes else 0
        if not mask:
            continue
        grid = np.frombuffer(payload, dtype=np.uint8).reshape(shots, record_bytes)
        offenders = np.nonzero(grid[:, -1] & mask)[0]
        for index in offenders.tolist():
            excluded.add(int(index))
            context.quality.record(
                "google_qec.b8.padding_bits_set",
                source_name=SOURCE_NAME,
                record_locator=f"{directory}/{member}#{ids.shot_locator(int(index))}",
                observed_value=f"0x{int(grid[index, -1]):02x}",
                reason=(
                    f"Shot {index} of {member} sets padding bits above bit "
                    f"{bits - 1}; the shot is excluded rather than masked."
                ),
                source_record_id=ids.google_shot_record_id(directory, int(index)),
            )
    return excluded


def _verify_fast_path(
    payload: bytes, record_bytes: int, bits: int, counts: np.ndarray
) -> None:
    """Cross-check the vectorised bit count against the supplied reader.

    A disagreement means this module's bit handling is wrong, not that the data
    is bad, so it raises rather than recording a data issue.
    """
    sample = payload[:record_bytes]
    if not sample:
        return
    reference = sum(next(iter_b8_records(sample, bits_per_record=bits)))
    if int(counts[0]) != reference:
        raise AssertionError(
            f"vectorised bit count {int(counts[0])} disagrees with "
            f"iter_b8_records {reference}"
        )
