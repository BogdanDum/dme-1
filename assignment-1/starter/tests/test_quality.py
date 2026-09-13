"""Invariants of the check catalogue and the evidence it produces."""

from __future__ import annotations

import pytest

from quantum_lake_student.models import Severity
from quantum_lake_student.quality import (
    RULES,
    Action,
    FatalDataError,
    QualityLog,
    TableCounts,
    is_unsafe_member,
)

# Each bullet of the required-check list in brief.md, mapped to the rules that
# implement it. A missing entry here means a required check has no rule.
BRIEF_CHECKS: dict[str, tuple[str, ...]] = {
    "documented label versus actual labels": (
        "qec_syndromes.readme.documented_header_mismatch",
    ),
    "four rounds with four binary values": (
        "qec_syndromes.row.syndrome_shape_invalid",
        "qec_syndromes.row.syndrome_bit_domain",
    ),
    "positive quantity and ten million weighted per file": (
        "qec_syndromes.row.quantity_not_positive",
        "qec_syndromes.file.weighted_total_mismatch",
    ),
    "one syndrome under both labels is valid": (
        "qec_syndromes.observation.same_syndrome_both_labels",
    ),
    "b8 length, alignment, bit order and padding": (
        "google_qec.b8.record_length_mismatch",
        "google_qec.b8.padding_bits_set",
    ),
    "01 row count and binary values": (
        "google_qec.01.line_count_mismatch",
        "google_qec.01.value_domain",
    ),
    "alignment of actual and predicted flips": (
        "google_qec.01.line_count_mismatch",
    ),
    "QASM registers, operations, checks and corrections": (
        "qasmbench.qasm.register_declaration_unparsable",
        "qasmbench.qasm.no_explicit_parity_check",
        "qasmbench.qasm.no_conditional_correction",
    ),
    "required companion files and safe archive members": (
        "google_qec.experiment.companion_file_missing",
        "bronze.archive.unsafe_member_path",
    ),
}


def test_every_required_check_has_a_declared_rule():
    for description, rule_ids in BRIEF_CHECKS.items():
        for rule_id in rule_ids:
            assert rule_id in RULES, f"{description}: {rule_id} is not declared"


def test_rule_ids_follow_the_naming_convention():
    for rule_id in RULES:
        assert rule_id.count(".") >= 2, rule_id
        assert rule_id.islower()


def test_fatal_rules_abort_the_run_rather_than_recording():
    for rule in RULES.values():
        if rule.fatal:
            assert rule.action is Action.RUN_ABORTED, rule.rule_id
            assert rule.severity is Severity.ERROR, rule.rule_id
        else:
            assert rule.action is not Action.RUN_ABORTED, rule.rule_id


def test_recording_a_fatal_rule_is_a_programming_error():
    log = QualityLog(run_id="r")
    with pytest.raises(AssertionError):
        log.record(
            "bronze.archive.unsafe_member_path",
            source_name="s",
            record_locator="l",
        )


def test_raising_a_non_fatal_rule_is_a_programming_error():
    log = QualityLog(run_id="r")
    with pytest.raises(AssertionError):
        log.fatal(
            "qec_syndromes.row.label_domain",
            source_name="s",
            record_locator="l",
        )


def test_unknown_rule_ids_are_rejected():
    log = QualityLog(run_id="r")
    with pytest.raises(KeyError):
        log.record("not.a.rule", source_name="s", record_locator="l")


def test_fatal_carries_the_rule_and_the_location():
    log = QualityLog(run_id="r")
    with pytest.raises(FatalDataError) as raised:
        log.fatal(
            "google_qec.experiment.companion_file_missing",
            source_name="google_qec",
            record_locator="exp/obs_flips_actual.01",
            observed_value="obs_flips_actual.01",
        )
    assert raised.value.rule_id == "google_qec.experiment.companion_file_missing"
    assert raised.value.record_locator == "exp/obs_flips_actual.01"


def test_issue_ids_are_stable_across_runs():
    first = QualityLog(run_id="run-a")
    second = QualityLog(run_id="run-b")
    for log in (first, second):
        log.record(
            "qec_syndromes.row.label_domain",
            source_name="qec_syndromes",
            record_locator="f.csv#row=3",
            observed_value="7",
        )
    left = first.to_table().to_pylist()[0]
    right = second.to_table().to_pylist()[0]
    assert left["issue_id"] == right["issue_id"]
    assert left["run_id"] != right["run_id"]


def test_the_same_finding_is_recorded_once():
    log = QualityLog(run_id="r")
    for _ in range(3):
        log.record(
            "qec_syndromes.row.label_domain",
            source_name="qec_syndromes",
            record_locator="f.csv#row=3",
            observed_value="7",
        )
    assert log.issue_count == 1


def test_issue_rows_carry_value_rule_action_and_reason():
    log = QualityLog(run_id="r")
    log.record(
        "qec_syndromes.row.quantity_not_positive",
        source_name="qec_syndromes",
        record_locator="f.csv#row=9",
        observed_value="-3",
        reason="quantity is -3",
    )
    row = log.to_table().to_pylist()[0]
    assert row["observed_value"] == "-3"
    assert row["rule_id"] == "qec_syndromes.row.quantity_not_positive"
    assert row["action"] == "excluded"
    assert row["reason"] == "quantity is -3"
    assert row["severity"] == "error"


def test_counts_reconcile_only_when_they_balance():
    counts = TableCounts(read=10, accepted=8, rejected=2)
    assert counts.reconciles()
    assert not TableCounts(read=10, accepted=8, rejected=1).reconciles()


def test_count_accumulates_per_table():
    log = QualityLog(run_id="r")
    log.count("shot", read=5, accepted=4, rejected=1)
    log.count("shot", read=5, accepted=5)
    assert log.counts["shot"].read == 10
    assert log.counts["shot"].accepted == 9
    assert log.counts["shot"].rejected == 1
    assert log.counts["shot"].reconciles()


@pytest.mark.parametrize(
    ("member", "unsafe"),
    [
        ("small/qec_sm_n5/qec_sm_n5.qasm", False),
        ("a/b/c.txt", False),
        ("/etc/passwd", True),
        ("../escape", True),
        ("a/../../escape", True),
        ("C:/windows", True),
        ("\\\\server\\share", True),
        ("", True),
    ],
)
def test_unsafe_member_paths_are_detected(member, unsafe):
    assert is_unsafe_member(member) is unsafe
