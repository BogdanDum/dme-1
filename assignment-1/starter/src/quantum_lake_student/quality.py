"""The check catalogue, and the log that turns findings into evidence.

Every check the pipeline applies is declared once in :data:`RULES`, so the set of
implemented rules is readable in one place and can be asserted against by tests.

Two dispositions exist, matching the brief:

* a rule marked ``fatal`` raises :class:`FatalDataError` and stops the run before
  anything is published — a missing companion file or an unsafe archive member
  means the input cannot be trusted at all;
* every other failing rule records a finding, excludes the record from Silver,
  and leaves the original value visible in ``data_issues.parquet``. Values are
  never silently repaired.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum

import pyarrow as pa

from . import ids
from .models import QualityFinding, Severity
from .schemas import DATA_ISSUES


class Action(StrEnum):
    """What the pipeline did about a finding."""

    EXCLUDED = "excluded"
    ACCEPTED_WITH_WARNING = "accepted_with_warning"
    RUN_ABORTED = "run_aborted"


@dataclass(frozen=True)
class Rule:
    rule_id: str
    severity: Severity
    action: Action
    description: str
    fatal: bool = False


def _rule(
    rule_id: str,
    severity: Severity,
    action: Action,
    description: str,
    *,
    fatal: bool = False,
) -> Rule:
    return Rule(rule_id, severity, action, description, fatal)


RULES: dict[str, Rule] = {
    rule.rule_id: rule
    for rule in (
        # -- Bronze: the supplied bytes must be exactly what was released ----
        _rule(
            "bronze.object.missing",
            Severity.ERROR,
            Action.RUN_ABORTED,
            "A source object named in the release manifest is not in Bronze.",
            fatal=True,
        ),
        _rule(
            "bronze.object.size_mismatch",
            Severity.ERROR,
            Action.RUN_ABORTED,
            "A Bronze object's byte size differs from the release manifest.",
            fatal=True,
        ),
        _rule(
            "bronze.object.sha256_mismatch",
            Severity.ERROR,
            Action.RUN_ABORTED,
            "A Bronze object's SHA-256 differs from the release manifest.",
            fatal=True,
        ),
        _rule(
            "bronze.archive.unsafe_member_path",
            Severity.ERROR,
            Action.RUN_ABORTED,
            "An archive member path is absolute or escapes the archive root.",
            fatal=True,
        ),
        _rule(
            "bronze.archive.corrupt",
            Severity.ERROR,
            Action.RUN_ABORTED,
            "An archive failed its internal integrity test.",
            fatal=True,
        ),
        # -- qec_syndromes ---------------------------------------------------
        _rule(
            "qec_syndromes.readme.documented_header_mismatch",
            Severity.WARNING,
            Action.ACCEPTED_WITH_WARNING,
            "The bundled README documents 'label' while the CSV header is "
            "'labels'. The actual header is authoritative.",
        ),
        _rule(
            "qec_syndromes.csv.header_unexpected",
            Severity.ERROR,
            Action.RUN_ABORTED,
            "A syndrome CSV header is not 'labels,syndromes,quantity'.",
            fatal=True,
        ),
        _rule(
            "qec_syndromes.file.filename_unparsable",
            Severity.ERROR,
            Action.RUN_ABORTED,
            "A syndrome filename does not encode distance and fault rate.",
            fatal=True,
        ),
        _rule(
            "qec_syndromes.row.syndrome_unparsable",
            Severity.ERROR,
            Action.EXCLUDED,
            "The syndromes cell is not a readable nested tuple.",
        ),
        _rule(
            "qec_syndromes.row.syndrome_shape_invalid",
            Severity.ERROR,
            Action.EXCLUDED,
            "The syndrome sequence is not 4 rounds of 4 checks.",
        ),
        _rule(
            "qec_syndromes.row.syndrome_bit_domain",
            Severity.ERROR,
            Action.EXCLUDED,
            "A syndrome value is not 0 or 1.",
        ),
        _rule(
            "qec_syndromes.row.label_domain",
            Severity.ERROR,
            Action.EXCLUDED,
            "The labels value is not 0 or 1.",
        ),
        _rule(
            "qec_syndromes.row.quantity_not_positive",
            Severity.ERROR,
            Action.EXCLUDED,
            "The quantity weight is not greater than zero.",
        ),
        _rule(
            "qec_syndromes.file.weighted_total_mismatch",
            Severity.ERROR,
            Action.ACCEPTED_WITH_WARNING,
            "Accepted quantities do not sum to the sample count in the "
            "filename. Individually valid rows are kept and the shortfall is "
            "reported rather than invented.",
        ),
        _rule(
            "qec_syndromes.observation.same_syndrome_both_labels",
            Severity.INFO,
            Action.ACCEPTED_WITH_WARNING,
            "A syndrome occurs under both logical-error labels. This is valid "
            "and both rows are preserved.",
        ),
        # -- google_qec ------------------------------------------------------
        _rule(
            "google_qec.experiment.companion_file_missing",
            Severity.ERROR,
            Action.RUN_ABORTED,
            "A required companion file is absent from an experiment directory.",
            fatal=True,
        ),
        _rule(
            "google_qec.experiment.properties_unparsable",
            Severity.ERROR,
            Action.RUN_ABORTED,
            "properties.yml is unreadable or lacks a required key.",
            fatal=True,
        ),
        _rule(
            "google_qec.experiment.directory_name_mismatch",
            Severity.ERROR,
            Action.ACCEPTED_WITH_WARNING,
            "The directory name disagrees with properties.yml. properties.yml "
            "is treated as authoritative and the disagreement is reported.",
        ),
        _rule(
            "google_qec.experiment.derived_invariant_mismatch",
            Severity.WARNING,
            Action.ACCEPTED_WITH_WARNING,
            "A derived relation between properties.yml counts does not hold.",
        ),
        _rule(
            "google_qec.b8.record_length_mismatch",
            Severity.ERROR,
            Action.RUN_ABORTED,
            "A b8 file length is not shots times the byte-aligned record size, "
            "so shots cannot be aligned.",
            fatal=True,
        ),
        _rule(
            "google_qec.b8.padding_bits_set",
            Severity.ERROR,
            Action.EXCLUDED,
            "A b8 record sets bits above the declared bit count; those bits "
            "are padding and must be zero.",
        ),
        _rule(
            "google_qec.01.line_count_mismatch",
            Severity.ERROR,
            Action.RUN_ABORTED,
            "A 01 file does not contain exactly one line per declared shot.",
            fatal=True,
        ),
        _rule(
            "google_qec.01.value_domain",
            Severity.ERROR,
            Action.RUN_ABORTED,
            "A 01 file contains a value other than 0 or 1.",
            fatal=True,
        ),
        # -- qasmbench -------------------------------------------------------
        _rule(
            "qasmbench.qasm.register_declaration_unparsable",
            Severity.ERROR,
            Action.EXCLUDED,
            "A qreg/creg declaration could not be read.",
        ),
        _rule(
            "qasmbench.qasm.gate_call_unresolved",
            Severity.WARNING,
            Action.ACCEPTED_WITH_WARNING,
            "A gate call names no known definition or library gate, so its "
            "expanded operation count is unknown.",
        ),
        _rule(
            "qasmbench.qasm.no_explicit_parity_check",
            Severity.INFO,
            Action.ACCEPTED_WITH_WARNING,
            "No parity check is explicit in this circuit, so no stabilizer "
            "rows are produced. Structure is not invented where absent.",
        ),
        _rule(
            "qasmbench.qasm.no_conditional_correction",
            Severity.INFO,
            Action.ACCEPTED_WITH_WARNING,
            "The circuit contains no syndrome-controlled recovery operation.",
        ),
    )
}


class FatalDataError(RuntimeError):
    """Raised by a fatal rule; the run stops before anything is published."""

    def __init__(
        self,
        rule_id: str,
        source_name: str,
        record_locator: str,
        message: str,
        observed_value: str | None = None,
    ) -> None:
        super().__init__(f"[{rule_id}] {record_locator}: {message}")
        self.rule_id = rule_id
        self.source_name = source_name
        self.record_locator = record_locator
        self.observed_value = observed_value


@dataclass
class TableCounts:
    """Rows read, kept, and rejected for one output table."""

    read: int = 0
    accepted: int = 0
    rejected: int = 0

    def reconciles(self) -> bool:
        return self.read == self.accepted + self.rejected

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "read": self.read,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "reconciles": self.reconciles(),
        }


@dataclass
class QualityLog:
    """Collects findings and per-table counts for one run."""

    run_id: str
    _issues: list[dict[str, str | None]] = field(default_factory=list, init=False)
    _seen: set[str] = field(default_factory=set, init=False)
    counts: dict[str, TableCounts] = field(
        default_factory=lambda: defaultdict(TableCounts), init=False
    )

    def record(
        self,
        rule_id: str,
        *,
        source_name: str,
        record_locator: str,
        observed_value: str | None = None,
        reason: str | None = None,
        source_record_id: str | None = None,
    ) -> QualityFinding:
        """Log one non-fatal finding, keeping the observed value unmodified."""
        rule = self._rule(rule_id)
        if rule.fatal:
            raise AssertionError(
                f"{rule_id} is a fatal rule; call fatal() so the run stops."
            )
        finding = QualityFinding(
            rule_id=rule.rule_id,
            severity=rule.severity,
            source_system=source_name,
            source_record_locator=record_locator,
            message=reason or rule.description,
            observed_value=observed_value,
        )
        identifier = ids.issue_id(
            rule.rule_id, source_name, record_locator, observed_value
        )
        # The same rule can fire on the same record twice within a run only if
        # the caller double-checks it; one evidence row per finding is enough.
        if identifier not in self._seen:
            self._seen.add(identifier)
            self._issues.append(
                {
                    "issue_id": identifier,
                    "run_id": self.run_id,
                    "source_record_id": source_record_id,
                    "rule_id": rule.rule_id,
                    "severity": str(rule.severity),
                    "observed_value": observed_value,
                    "action": str(rule.action),
                    "reason": finding.message,
                }
            )
        return finding

    def fatal(
        self,
        rule_id: str,
        *,
        source_name: str,
        record_locator: str,
        observed_value: str | None = None,
        reason: str | None = None,
    ) -> None:
        """Abort the run. Declared fatal rules must reach this method."""
        rule = self._rule(rule_id)
        if not rule.fatal:
            raise AssertionError(f"{rule_id} is not declared fatal.")
        raise FatalDataError(
            rule.rule_id,
            source_name,
            record_locator,
            reason or rule.description,
            observed_value,
        )

    def count(
        self, table: str, *, read: int = 0, accepted: int = 0, rejected: int = 0
    ) -> None:
        entry = self.counts[table]
        entry.read += read
        entry.accepted += accepted
        entry.rejected += rejected

    @property
    def issue_count(self) -> int:
        return len(self._issues)

    def issue_ids(self) -> frozenset[str]:
        return frozenset(self._seen)

    def to_table(self) -> pa.Table:
        columns = {
            name: [issue[name] for issue in self._issues]
            for name in DATA_ISSUES.column_names
        }
        return pa.table(columns, schema=DATA_ISSUES.schema)

    def counts_payload(self) -> dict[str, dict[str, int | bool]]:
        return {
            table: self.counts[table].as_dict() for table in sorted(self.counts)
        }

    @staticmethod
    def _rule(rule_id: str) -> Rule:
        try:
            return RULES[rule_id]
        except KeyError:
            raise KeyError(f"Undeclared rule_id {rule_id!r}") from None


def is_unsafe_member(member: str) -> bool:
    """Reject absolute paths, parent traversal, and drive-relative members."""
    if not member or member.startswith(("/", "\\")):
        return True
    if ":" in member.split("/", 1)[0]:
        return True
    return any(part == ".." for part in member.replace("\\", "/").split("/"))
