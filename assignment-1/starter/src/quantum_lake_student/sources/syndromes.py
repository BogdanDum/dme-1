"""Parse and check the simulated surface-code syndrome CSVs.

One row of one CSV is one *aggregate* observation: a distinct
(logical-error label, 4x4 syndrome pattern) pair, with ``quantity`` recording how
many simulated shots produced it. The rows are never expanded into individual
shots -- 70 million rows would carry no more information than the 75,598
supplied ones, and expanding them is an explicit anti-pattern.

Two encodings meet here and must not be confused. This source stores
``syndrome_bits`` as 16 one-byte values, round first and check second, because
that is what the Silver contract and the supplied ``syndrome_model_input`` helper
expect. The Google source stores *bit-packed* Stim rows instead.
"""

from __future__ import annotations

import ast
import csv
import io
import re
from dataclasses import dataclass, field

import pyarrow as pa

from .. import ids
from ..quality import QualityLog
from ..runcontext import RunContext, TraceCollector
from ..schemas import SYNDROME_OBSERVATION

SOURCE_NAME = "qec_syndromes"
README_MEMBER = "README.txt"
EXPECTED_HEADER = ("labels", "syndromes", "quantity")
EXPECTED_ROUNDS = 4
EXPECTED_CHECKS = 4

FILENAME = re.compile(
    r"^d-(?P<distance>\d+)_pfr-(?P<rate>\d\.\d+)_nb-(?P<nominal>\d+)(?P<unit>[KMG]?)$"
)

# One strict pattern is both the shape check and the bit-domain check: exactly
# four parenthesised groups of exactly four 0/1 values. Whitespace is tolerated
# because the release writes ", " between values.
_ROUND = r"\(\s*[01]\s*(?:,\s*[01]\s*){%d}\)" % (EXPECTED_CHECKS - 1)
SYNDROME_TEXT = re.compile(
    r"^\(\s*%(round)s\s*(?:,\s*%(round)s\s*){%(extra)d}\)$"
    % {"round": _ROUND, "extra": EXPECTED_ROUNDS - 1}
)

_UNIT_SCALE = {"": 1, "K": 1_000, "M": 1_000_000, "G": 1_000_000_000}

# The README's column list, e.g. "- label: binary label (0: no error...)".
_DOCUMENTED_COLUMN = re.compile(r"^-\s+(\w+)\s*:", re.MULTILINE)


@dataclass
class FileStats:
    """Reconciliation counters for one fault-rate file."""

    member: str
    read: int = 0
    accepted: int = 0
    rejected: int = 0
    nominal_from_filename: int = 0
    weighted_total_read: int = 0
    weighted_total_accepted: int = 0
    syndromes_with_both_labels: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "read": self.read,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "nominal_from_filename": self.nominal_from_filename,
            "weighted_total_read": self.weighted_total_read,
            "weighted_total_accepted": self.weighted_total_accepted,
            "syndromes_with_both_labels": self.syndromes_with_both_labels,
        }


@dataclass
class SyndromeOutcome:
    row_count: int = 0
    per_file: dict[str, dict[str, int]] = field(default_factory=dict)

    def totals(self) -> dict[str, int]:
        keys = (
            "read",
            "accepted",
            "rejected",
            "weighted_total_read",
            "weighted_total_accepted",
        )
        return {
            key: sum(entry[key] for entry in self.per_file.values()) for key in keys
        }


def documented_columns(readme_text: str) -> tuple[str, ...]:
    """Read the column names the bundled README claims the CSVs have.

    Deriving this from the shipped documentation, rather than hardcoding the
    known-wrong ``label``, means the check reports what the release actually
    says and keeps working if the release is corrected.
    """
    return tuple(_DOCUMENTED_COLUMN.findall(readme_text))


def parse_syndrome_text(text: str) -> tuple[int, ...] | None:
    """Return the 16 bits round-major, or ``None`` if the cell is not valid.

    The outer tuple is rounds and the inner tuples are checks, so reading the
    digits in source order already yields round-first, check-second ordering.
    """
    if not SYNDROME_TEXT.match(text):
        return None
    return tuple(int(digit) for digit in re.findall(r"[01]", text))


def describe_bad_syndrome(text: str) -> str:
    """Explain *why* a syndrome cell was rejected, for the issue reason.

    Uses :func:`ast.literal_eval`, which evaluates literals only and cannot
    execute code -- ``eval`` is never used on source data.
    """
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return "the syndromes cell is not a readable nested tuple"
    if not isinstance(value, tuple):
        return f"expected a tuple of rounds, found {type(value).__name__}"
    if len(value) != EXPECTED_ROUNDS:
        return f"expected {EXPECTED_ROUNDS} rounds, found {len(value)}"
    for index, entry in enumerate(value):
        if not isinstance(entry, tuple):
            return f"round {index} is {type(entry).__name__}, not a tuple of checks"
        if len(entry) != EXPECTED_CHECKS:
            return (
                f"round {index} has {len(entry)} checks, "
                f"expected {EXPECTED_CHECKS}"
            )
        for check, bit in enumerate(entry):
            if bit not in (0, 1):
                return f"round {index} check {check} is {bit!r}, not 0 or 1"
    return "the syndromes cell did not match the expected 4x4 binary shape"


def syndrome_blob(bits: tuple[int, ...]) -> bytes:
    """Pack the 16 values as 16 one-byte binary values, per the contract."""
    return bytes(bits)


def parse_filename(stem: str) -> tuple[int, str, int] | None:
    """Return ``(distance, rate_text, nominal_shots)`` from a CSV filename."""
    matched = FILENAME.match(stem)
    if not matched:
        return None
    nominal = int(matched["nominal"]) * _UNIT_SCALE[matched["unit"]]
    return int(matched["distance"]), matched["rate"], nominal


def build(context: RunContext, key: str, bronze_object: str, sha256: str) -> SyndromeOutcome:
    """Parse every CSV in the archive and stage the Silver table."""
    outcome = SyndromeOutcome()
    rows: list[dict[str, object]] = []

    with context.lake.open_bronze_zip(key) as archive:
        members = sorted(
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv") and not name.endswith("/")
        )
        documented = _documented_columns(context, archive)
        for member in members:
            with archive.open(member) as handle:
                stats = _read_member(
                    context,
                    handle,
                    member=member,
                    documented=documented,
                    bronze_object=bronze_object,
                    sha256=sha256,
                    rows=rows,
                )
            outcome.per_file[member] = stats.as_dict()

    table = pa.table(
        {
            name: [row[name] for row in rows]
            for name in SYNDROME_OBSERVATION.column_names
        },
        schema=SYNDROME_OBSERVATION.schema,
    )
    published = context.lake.stage_table(SYNDROME_OBSERVATION, table)
    outcome.row_count = published.row_count
    return outcome


def _documented_columns(context: RunContext, archive) -> tuple[str, ...]:
    try:
        readme = archive.read(README_MEMBER).decode("utf-8", errors="replace")
    except KeyError:
        return ()
    return documented_columns(readme)


def _read_member(
    context: RunContext,
    handle,
    *,
    member: str,
    documented: tuple[str, ...],
    bronze_object: str,
    sha256: str,
    rows: list[dict[str, object]],
) -> FileStats:
    quality: QualityLog = context.quality
    trace: TraceCollector = context.trace

    stem = member.rsplit("/", 1)[-1].removesuffix(".csv")
    parsed_name = parse_filename(stem)
    if parsed_name is None:
        quality.fatal(
            "qec_syndromes.file.filename_unparsable",
            source_name=SOURCE_NAME,
            record_locator=member,
            observed_value=stem,
            reason=(
                f"{stem!r} does not match the documented "
                "d-<distance>_pfr-<rate>_nb-<count> pattern, so the fault rate "
                "cannot be recovered."
            ),
        )
    distance, rate_text, nominal = parsed_name
    experiment_id = ids.syndrome_experiment_id(distance, rate_text)
    physical_fault_rate = float(rate_text)

    stats = FileStats(member=member, nominal_from_filename=nominal)
    reader = csv.reader(io.TextIOWrapper(handle, encoding="utf-8", newline=""))

    header = next(reader, None)
    if header is None or tuple(header) != EXPECTED_HEADER:
        quality.fatal(
            "qec_syndromes.csv.header_unexpected",
            source_name=SOURCE_NAME,
            record_locator=member,
            observed_value=",".join(header or ()),
            reason=(
                f"Expected header {','.join(EXPECTED_HEADER)}; the columns "
                "cannot be interpreted otherwise."
            ),
        )
    if documented and tuple(documented) != EXPECTED_HEADER:
        quality.record(
            "qec_syndromes.readme.documented_header_mismatch",
            source_name=SOURCE_NAME,
            record_locator=member,
            observed_value=(
                f"documented={','.join(documented)} actual={','.join(header)}"
            ),
            reason=(
                f"The bundled README documents columns {','.join(documented)} "
                f"while the CSV header is {','.join(header)}. The actual header "
                "is authoritative and no value was changed."
            ),
        )

    labels_by_syndrome: dict[str, set[bool]] = {}

    for row_number, raw in enumerate(reader, start=1):
        stats.read += 1
        locator = f"{member}#{ids.csv_row_locator(row_number)}"
        record = _parse_row(
            quality,
            raw,
            locator=locator,
            experiment_id=experiment_id,
            physical_fault_rate=physical_fault_rate,
        )
        if record is None:
            stats.rejected += 1
            continue

        stats.weighted_total_read += record["quantity"]
        stats.accepted += 1
        stats.weighted_total_accepted += record["quantity"]
        rows.append(record)
        labels_by_syndrome.setdefault(
            record["syndrome_bits"].hex(), set()
        ).add(record["logical_error_label"])

        trace.add(
            source_record_id=record["source_record_id"],
            source_name=SOURCE_NAME,
            bronze_object=bronze_object,
            archive_member=member,
            record_locator=ids.csv_row_locator(row_number),
            input_sha256=sha256,
        )

    stats.syndromes_with_both_labels = sum(
        1 for labels in labels_by_syndrome.values() if len(labels) > 1
    )
    _reconcile_file(quality, stats, member)
    context.quality.count(
        SYNDROME_OBSERVATION.name,
        read=stats.read,
        accepted=stats.accepted,
        rejected=stats.rejected,
    )
    return stats


def _parse_row(
    quality: QualityLog,
    raw: list[str],
    *,
    locator: str,
    experiment_id: str,
    physical_fault_rate: float,
) -> dict[str, object] | None:
    if len(raw) != len(EXPECTED_HEADER):
        quality.record(
            "qec_syndromes.row.syndrome_unparsable",
            source_name=SOURCE_NAME,
            record_locator=locator,
            observed_value=",".join(raw),
            reason=f"Expected {len(EXPECTED_HEADER)} fields, found {len(raw)}.",
        )
        return None

    label_text, syndrome_text, quantity_text = raw

    if label_text not in ("0", "1"):
        quality.record(
            "qec_syndromes.row.label_domain",
            source_name=SOURCE_NAME,
            record_locator=locator,
            observed_value=label_text,
            reason=f"labels is {label_text!r}; only 0 and 1 are meaningful.",
        )
        return None

    bits = parse_syndrome_text(syndrome_text)
    if bits is None:
        reason = describe_bad_syndrome(syndrome_text)
        rule = (
            "qec_syndromes.row.syndrome_bit_domain"
            if "not 0 or 1" in reason
            else "qec_syndromes.row.syndrome_shape_invalid"
            if "round" in reason or "expected" in reason
            else "qec_syndromes.row.syndrome_unparsable"
        )
        quality.record(
            rule,
            source_name=SOURCE_NAME,
            record_locator=locator,
            observed_value=syndrome_text,
            reason=reason,
        )
        return None

    try:
        quantity = int(quantity_text)
    except ValueError:
        quality.record(
            "qec_syndromes.row.quantity_not_positive",
            source_name=SOURCE_NAME,
            record_locator=locator,
            observed_value=quantity_text,
            reason=f"quantity {quantity_text!r} is not an integer.",
        )
        return None
    if quantity <= 0:
        quality.record(
            "qec_syndromes.row.quantity_not_positive",
            source_name=SOURCE_NAME,
            record_locator=locator,
            observed_value=quantity_text,
            reason=(
                f"quantity is {quantity}; a weight must represent at least one "
                "physical observation."
            ),
        )
        return None

    syndrome_bits = syndrome_blob(bits)
    logical_error_label = label_text == "1"
    return {
        "source_record_id": ids.syndrome_record_id(
            experiment_id, syndrome_bits, logical_error_label
        ),
        "experiment_id": experiment_id,
        "physical_fault_rate": physical_fault_rate,
        "syndrome_bits": syndrome_bits,
        "round_count": EXPECTED_ROUNDS,
        "check_count": EXPECTED_CHECKS,
        "logical_error_label": logical_error_label,
        "quantity": quantity,
    }


def _reconcile_file(quality: QualityLog, stats: FileStats, member: str) -> None:
    """Compare the weighted total to the count encoded in the filename.

    ``weighted_total_read`` is the right side of this comparison: once a row is
    rejected the accepted sum is legitimately below the nominal count, so
    comparing that instead would raise a false alarm.
    """
    if stats.weighted_total_read != stats.nominal_from_filename:
        quality.record(
            "qec_syndromes.file.weighted_total_mismatch",
            source_name=SOURCE_NAME,
            record_locator=member,
            observed_value=str(stats.weighted_total_read),
            reason=(
                f"Parsed quantities sum to {stats.weighted_total_read:,} but the "
                f"filename declares {stats.nominal_from_filename:,}. Individually "
                "valid rows are kept and the difference is reported."
            ),
        )
    if stats.syndromes_with_both_labels:
        quality.record(
            "qec_syndromes.observation.same_syndrome_both_labels",
            source_name=SOURCE_NAME,
            record_locator=member,
            observed_value=str(stats.syndromes_with_both_labels),
            reason=(
                f"{stats.syndromes_with_both_labels:,} syndromes occur under both "
                "logical-error labels. The label is part of the record key, so "
                "both rows are valid and both are preserved."
            ),
        )
