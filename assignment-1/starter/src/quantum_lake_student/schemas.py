"""Declared Arrow schemas for every published table.

The six Silver schemas are transcribed from ``assignment/silver-tables.md`` and
are the contract the pipeline is graded against. They are declared explicitly,
rather than inferred from a dataframe, so that a wrong dtype fails loudly at
write time instead of silently reaching a Parquet file.

Each :class:`TableSpec` also carries the sort key used before publishing. Source
row order never reaches an output: sorting on content-derived columns is what
makes a rerun on unchanged input produce byte-identical files.
"""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa


@dataclass(frozen=True)
class TableSpec:
    """A published table: where it goes, its columns, and its row order."""

    name: str
    relative_path: str
    schema: pa.Schema
    sort_by: tuple[str, ...]
    row_meaning: str

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(self.schema.names)


SOURCE_NAMES = ("qec_syndromes", "google_qec", "qasmbench")

# The four decoders ship one `01` prediction file each. The column names are
# fixed by ml.GOOGLE_META_PREDICTION_COLUMNS, so Silver uses them verbatim and
# the supplied model-input helper works unmodified downstream.
DECODER_PREDICTION_COLUMNS = (
    "belief_matching_prediction",
    "correlated_matching_prediction",
    "pymatching_prediction",
    "tensor_network_contraction_prediction",
)


SYNDROME_OBSERVATION = TableSpec(
    name="syndrome_observation",
    relative_path="silver/qec_syndromes/syndrome_observation.parquet",
    row_meaning="one original aggregate CSV row",
    sort_by=("experiment_id", "source_record_id"),
    schema=pa.schema(
        [
            pa.field("source_record_id", pa.string(), nullable=False),
            pa.field("experiment_id", pa.string(), nullable=False),
            pa.field("physical_fault_rate", pa.float64(), nullable=False),
            # Exactly 16 one-byte binary values, round first and check second.
            pa.field("syndrome_bits", pa.binary(), nullable=False),
            pa.field("round_count", pa.int32(), nullable=False),
            pa.field("check_count", pa.int32(), nullable=False),
            pa.field("logical_error_label", pa.bool_(), nullable=False),
            pa.field("quantity", pa.int64(), nullable=False),
        ]
    ),
)


GOOGLE_EXPERIMENT = TableSpec(
    name="experiment",
    relative_path="silver/google_qec/experiment.parquet",
    row_meaning="one hardware experiment directory",
    sort_by=("experiment_id",),
    schema=pa.schema(
        [
            pa.field("source_record_id", pa.string(), nullable=False),
            pa.field("experiment_id", pa.string(), nullable=False),
            pa.field("basis", pa.string(), nullable=False),
            pa.field("distance", pa.int32(), nullable=False),
            pa.field("rounds", pa.int32(), nullable=False),
            pa.field("shots", pa.int64(), nullable=False),
            pa.field("center_row", pa.int32(), nullable=False),
            pa.field("center_col", pa.int32(), nullable=False),
            pa.field("measurement_count", pa.int32(), nullable=False),
            pa.field("detector_count", pa.int32(), nullable=False),
        ]
    ),
)


GOOGLE_SHOT = TableSpec(
    name="shot",
    relative_path="silver/google_qec/shot.parquet",
    row_meaning="one aligned hardware shot",
    sort_by=("experiment_id", "shot_index"),
    schema=pa.schema(
        [
            pa.field("source_record_id", pa.string(), nullable=False),
            pa.field("experiment_id", pa.string(), nullable=False),
            pa.field("shot_index", pa.int64(), nullable=False),
            # Packed Stim b8 rows, stored exactly as supplied. Unpacking to one
            # column per bit belongs to Part II, not to Silver.
            pa.field("measurement_bits", pa.binary(), nullable=False),
            pa.field("sweep_bits", pa.binary(), nullable=False),
            pa.field("detector_bits", pa.binary(), nullable=False),
            pa.field("detector_event_count", pa.int32(), nullable=False),
            # Five distinct concepts. An actual flip is not a decoder mistake,
            # and a decoder mistake is derived, so it is not stored here.
            pa.field("actual_observable_flip", pa.bool_(), nullable=False),
            *(
                pa.field(column, pa.bool_(), nullable=False)
                for column in DECODER_PREDICTION_COLUMNS
            ),
        ]
    ),
)


QASM_CIRCUIT = TableSpec(
    name="circuit",
    relative_path="silver/qasmbench/circuit.parquet",
    row_meaning="one parsed circuit variant",
    sort_by=("circuit_id",),
    schema=pa.schema(
        [
            pa.field("source_record_id", pa.string(), nullable=False),
            pa.field("circuit_id", pa.string(), nullable=False),
            pa.field("benchmark_name", pa.string(), nullable=False),
            pa.field("variant", pa.string(), nullable=False),
            pa.field("register_declarations", pa.string(), nullable=False),
            pa.field("qubit_count", pa.int32(), nullable=False),
            pa.field("measurement_count", pa.int32(), nullable=False),
            pa.field("two_qubit_gate_count", pa.int32(), nullable=False),
        ]
    ),
)


QASM_STABILIZER_CHECK = TableSpec(
    name="stabilizer_check",
    relative_path="silver/qasmbench/stabilizer_check.parquet",
    row_meaning="one parity/stabilizer check identified in a circuit",
    sort_by=("circuit_id", "check_id"),
    schema=pa.schema(
        [
            pa.field("source_record_id", pa.string(), nullable=False),
            pa.field("circuit_id", pa.string(), nullable=False),
            pa.field("check_id", pa.string(), nullable=False),
            pa.field("ancilla_qubit", pa.string(), nullable=False),
            # Plain `list<string>`, as the contract writes it. Parquet's LIST
            # annotation renames the element field to "element" on round-trip;
            # that is the format's own naming, not a schema change.
            pa.field("data_qubits", pa.list_(pa.string()), nullable=False),
            pa.field("syndrome_bit", pa.string(), nullable=False),
        ]
    ),
)


QASM_CONDITIONAL_CORRECTION = TableSpec(
    name="conditional_correction",
    relative_path="silver/qasmbench/conditional_correction.parquet",
    row_meaning="one recovery operation controlled by a measured syndrome",
    sort_by=("circuit_id", "source_record_id"),
    schema=pa.schema(
        [
            pa.field("source_record_id", pa.string(), nullable=False),
            pa.field("circuit_id", pa.string(), nullable=False),
            pa.field("condition_register", pa.string(), nullable=False),
            pa.field("condition_value", pa.int64(), nullable=False),
            pa.field("gate", pa.string(), nullable=False),
            pa.field("target_qubit", pa.string(), nullable=False),
        ]
    ),
)


SILVER_TABLES: tuple[TableSpec, ...] = (
    SYNDROME_OBSERVATION,
    GOOGLE_EXPERIMENT,
    GOOGLE_SHOT,
    QASM_CIRCUIT,
    QASM_STABILIZER_CHECK,
    QASM_CONDITIONAL_CORRECTION,
)

SILVER_TABLES_BY_SOURCE: dict[str, tuple[TableSpec, ...]] = {
    "qec_syndromes": (SYNDROME_OBSERVATION,),
    "google_qec": (GOOGLE_EXPERIMENT, GOOGLE_SHOT),
    "qasmbench": (
        QASM_CIRCUIT,
        QASM_STABILIZER_CHECK,
        QASM_CONDITIONAL_CORRECTION,
    ),
}


# `archive_member` is nullable because the contract says "member path inside an
# archive, if applicable"; every Bronze object here is an archive, but an
# object-level trace row is still representable.
#
# ``sort_by`` is empty on purpose. This table has one row per (Silver record,
# contributing Bronze member) pair, which is two million rows for the Google
# shots alone, so it is streamed to Parquet rather than sorted in memory. Its
# order is still fully reproducible: sources are processed in a fixed order,
# experiment directories and archive members are visited sorted, and shots
# ascend, so the emission order itself is the deterministic order.
SOURCE_TRACE = TableSpec(
    name="source_trace",
    relative_path="part1/source_trace.parquet",
    row_meaning="one Bronze location contributing to one Silver record",
    sort_by=(),
    schema=pa.schema(
        [
            pa.field("source_record_id", pa.string(), nullable=False),
            pa.field("source_name", pa.string(), nullable=False),
            pa.field("bronze_object", pa.string(), nullable=False),
            pa.field("archive_member", pa.string(), nullable=True),
            pa.field("record_locator", pa.string(), nullable=False),
            pa.field("input_sha256", pa.string(), nullable=False),
        ]
    ),
)


# `run_id` is the only per-run field in any published table, which is why the
# rerun test compares issue_id sets here rather than file bytes.
DATA_ISSUES = TableSpec(
    name="data_issues",
    relative_path="part1/data_issues.parquet",
    row_meaning="one check outcome that excluded a record or warranted a warning",
    sort_by=("rule_id", "issue_id"),
    schema=pa.schema(
        [
            pa.field("issue_id", pa.string(), nullable=False),
            pa.field("run_id", pa.string(), nullable=False),
            pa.field("source_record_id", pa.string(), nullable=True),
            pa.field("rule_id", pa.string(), nullable=False),
            pa.field("severity", pa.string(), nullable=False),
            pa.field("observed_value", pa.string(), nullable=True),
            pa.field("action", pa.string(), nullable=False),
            pa.field("reason", pa.string(), nullable=False),
        ]
    ),
)
