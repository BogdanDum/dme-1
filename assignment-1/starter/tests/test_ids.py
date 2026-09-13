"""Identifiers must depend on content only, never on order or on the clock."""

from __future__ import annotations

import random

from quantum_lake_student import ids


def test_syndrome_record_id_is_stable_for_equal_content():
    first = ids.syndrome_record_id("exp", bytes([0, 1] * 8), True)
    second = ids.syndrome_record_id("exp", bytes([0, 1] * 8), True)
    assert first == second


def test_syndrome_record_id_separates_label_and_pattern():
    pattern = bytes([0, 1] * 8)
    other = bytes([1, 0] * 8)
    assert ids.syndrome_record_id("exp", pattern, True) != ids.syndrome_record_id(
        "exp", pattern, False
    )
    assert ids.syndrome_record_id("exp", pattern, True) != ids.syndrome_record_id(
        "exp", other, True
    )
    assert ids.syndrome_record_id("a", pattern, True) != ids.syndrome_record_id(
        "b", pattern, True
    )


def test_syndrome_ids_do_not_depend_on_row_order():
    records = [
        ("exp", bytes([index % 2] * 16), index % 2 == 0) for index in range(50)
    ]
    forward = {ids.syndrome_record_id(*record) for record in records}
    shuffled = records[:]
    random.Random(7).shuffle(shuffled)
    assert {ids.syndrome_record_id(*record) for record in shuffled} == forward


def test_experiment_id_keeps_the_padded_fault_rate():
    # 0.000010 and 1e-05 are the same number but different strings; keeping the
    # filename form is what makes ids group reliably.
    assert ids.syndrome_experiment_id(3, "0.000010").endswith("pfr=0.000010")
    assert ids.syndrome_experiment_id(3, "0.000010") != ids.syndrome_experiment_id(
        3, "1e-05"
    )


def test_google_shot_id_depends_on_experiment_and_index():
    assert ids.google_shot_record_id("a", 1) != ids.google_shot_record_id("a", 2)
    assert ids.google_shot_record_id("a", 1) != ids.google_shot_record_id("b", 1)
    assert ids.google_shot_record_id("a", 1) == ids.google_shot_record_id("a", 1)


def test_correction_ordinal_separates_repeated_statements():
    common = ("circuit", "syn", 1, "x", "q[0]")
    first = ids.conditional_correction_record_id(
        common[0], 0, *common[1:]
    )
    second = ids.conditional_correction_record_id(
        common[0], 1, *common[1:]
    )
    assert first != second


def test_issue_id_excludes_the_run_id():
    # Two runs that find the same problem must agree on its issue id.
    first = ids.issue_id("rule", "source", "row=1", "bad")
    second = ids.issue_id("rule", "source", "row=1", "bad")
    assert first == second
    assert first != ids.issue_id("rule", "source", "row=2", "bad")


def test_ids_contain_no_timestamp_or_random_component():
    generated = [
        ids.syndrome_record_id("exp", bytes(16), False),
        ids.google_shot_record_id("exp", 0),
        ids.circuit_record_id("a.qasm", "0" * 64),
        ids.issue_id("rule", "source", "row=1", None),
    ]
    for value in generated:
        assert len(value) == 64
        assert all(character in "0123456789abcdef" for character in value)
