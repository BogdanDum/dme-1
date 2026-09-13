"""Syndrome cleaning rules: shape, bit domain, weights, and the valid duplicate."""

from __future__ import annotations

import pytest

from quantum_lake_student.quality import QualityLog
from quantum_lake_student.sources import syndromes


def test_parses_the_release_spacing():
    bits = syndromes.parse_syndrome_text(
        "((0, 0, 1, 0), (0, 0, 1, 0), (0, 0, 0, 0), (0, 0, 0, 0))"
    )
    assert bits == (0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)


def test_blob_is_sixteen_one_byte_values_round_major():
    bits = syndromes.parse_syndrome_text(
        "((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))"
    )
    blob = syndromes.syndrome_blob(bits)
    assert len(blob) == 16
    assert set(blob) <= {0, 1}
    # Round first, check second: the set bit walks one position per round.
    assert blob == bytes([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])


@pytest.mark.parametrize(
    "text",
    [
        "((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))",          # three rounds
        "((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))",        # three checks
        "((0, 0, 2, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))",  # not binary
        "not a tuple",
        "",
    ],
)
def test_rejects_malformed_syndromes(text):
    assert syndromes.parse_syndrome_text(text) is None


def test_rejection_reasons_are_specific():
    assert "3" in syndromes.describe_bad_syndrome(
        "((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))"
    )
    assert "not 0 or 1" in syndromes.describe_bad_syndrome(
        "((0, 0, 2, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))"
    )
    assert "readable" in syndromes.describe_bad_syndrome("nonsense(")


def test_documented_columns_come_from_the_readme():
    readme = (
        "columns:\n"
        "- label: binary label\n"
        "- syndromes: sequence\n"
        "- quantity: number of samples\n"
    )
    assert syndromes.documented_columns(readme) == ("label", "syndromes", "quantity")


def test_documented_columns_disagree_with_the_actual_header():
    # This is the mismatch the brief asks to detect: 'label' versus 'labels'.
    documented = syndromes.documented_columns(
        "- label: x\n- syndromes: y\n- quantity: z\n"
    )
    assert documented != syndromes.EXPECTED_HEADER
    assert documented[0] == "label"
    assert syndromes.EXPECTED_HEADER[0] == "labels"


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("d-3_pfr-0.000010_nb-10M", (3, "0.000010", 10_000_000)),
        ("d-5_pfr-0.010000_nb-1K", (5, "0.010000", 1_000)),
        ("d-3_pfr-0.005000_nb-500", (3, "0.005000", 500)),
    ],
)
def test_filename_encodes_distance_rate_and_nominal_count(stem, expected):
    assert syndromes.parse_filename(stem) == expected


def test_filename_must_be_recognisable():
    assert syndromes.parse_filename("syndromes") is None


def _row(quality, raw):
    return syndromes._parse_row(
        quality,
        raw,
        locator="file.csv#row=1",
        experiment_id="exp",
        physical_fault_rate=0.005,
    )


def test_quantity_must_be_positive():
    quality = QualityLog(run_id="r")
    zeroes = "((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))"
    assert _row(quality, ["0", zeroes, "0"]) is None
    assert _row(quality, ["0", zeroes, "-5"]) is None
    assert _row(quality, ["0", zeroes, "not-a-number"]) is None
    rules = {issue["rule_id"] for issue in quality.to_table().to_pylist()}
    assert rules == {"qec_syndromes.row.quantity_not_positive"}


def test_label_must_be_binary():
    quality = QualityLog(run_id="r")
    zeroes = "((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))"
    assert _row(quality, ["2", zeroes, "1"]) is None
    issues = quality.to_table().to_pylist()
    assert issues[0]["rule_id"] == "qec_syndromes.row.label_domain"
    assert issues[0]["observed_value"] == "2"
    assert issues[0]["action"] == "excluded"


def test_rejected_row_keeps_its_original_value_visible():
    quality = QualityLog(run_id="r")
    bad = "((0, 0, 2, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))"
    assert _row(quality, ["0", bad, "5"]) is None
    issue = quality.to_table().to_pylist()[0]
    assert issue["observed_value"] == bad
    assert issue["action"] == "excluded"
    assert issue["reason"]


def test_accepted_row_carries_the_contract_values():
    quality = QualityLog(run_id="r")
    record = _row(
        quality,
        ["1", "((1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))", "42"],
    )
    assert record is not None
    assert record["round_count"] == 4
    assert record["check_count"] == 4
    assert record["logical_error_label"] is True
    assert record["quantity"] == 42
    assert len(record["syndrome_bits"]) == 16
    assert quality.issue_count == 0
