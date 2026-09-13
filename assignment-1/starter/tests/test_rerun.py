"""A second run over unchanged input must add nothing and change nothing.

This is the evidence for the rubric's "proves that a second run adds no
duplicates or unstable identifiers". Silver and the trace are compared byte for
byte; ``data_issues`` is compared after dropping ``run_id``, which the contract
requires it to carry and which therefore must be its only difference.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from quantum_lake_student import pipeline
from quantum_lake_student.schemas import SILVER_TABLES

BYTE_IDENTICAL = tuple(spec.relative_path for spec in SILVER_TABLES) + (
    "part1/source_trace.parquet",
)


def _run(release, tmp_path: Path, label: str):
    return pipeline.run_part1(
        settings=release.settings,
        results_root=tmp_path / f"results-{label}",
        repository=tmp_path,
        run_id=f"run-{label}",
    )


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _published_path(release, tmp_path: Path, label: str, relative: str) -> Path:
    if relative.startswith("silver/"):
        return release.settings.local_lake_root / relative
    return tmp_path / f"results-{label}" / relative


def test_silver_and_trace_are_byte_identical_on_a_second_run(release, tmp_path):
    first = _run(release, tmp_path, "one")
    digests = {
        relative: first.published[relative].sha256 for relative in BYTE_IDENTICAL
    }
    second = _run(release, tmp_path, "two")

    for relative in BYTE_IDENTICAL:
        assert second.published[relative].sha256 == digests[relative], relative


def test_row_counts_do_not_grow_on_a_second_run(release, tmp_path):
    first = _run(release, tmp_path, "one")
    second = _run(release, tmp_path, "two")
    assert {k: v.row_count for k, v in first.published.items()} == {
        k: v.row_count for k, v in second.published.items()
    }


def test_record_ids_are_unchanged_and_unique(release, tmp_path):
    _run(release, tmp_path, "one")
    first_ids = _all_record_ids(release)
    _run(release, tmp_path, "two")
    assert _all_record_ids(release) == first_ids

    for relative, identifiers in first_ids.items():
        assert len(identifiers) == len(set(identifiers)), relative


def _all_record_ids(release) -> dict[str, list[str]]:
    return {
        spec.relative_path: pq.read_table(
            release.settings.local_lake_root / spec.relative_path
        )
        .column("source_record_id")
        .to_pylist()
        for spec in SILVER_TABLES
    }


def test_data_issues_differ_only_by_run_id(release, tmp_path):
    _run(release, tmp_path, "one")
    _run(release, tmp_path, "two")

    left = pq.read_table(tmp_path / "results-one" / "part1" / "data_issues.parquet")
    right = pq.read_table(tmp_path / "results-two" / "part1" / "data_issues.parquet")

    assert set(left.column("run_id").to_pylist()) != set(
        right.column("run_id").to_pylist()
    )
    assert left.drop_columns(["run_id"]).equals(right.drop_columns(["run_id"]))
    assert set(left.column("issue_id").to_pylist()) == set(
        right.column("issue_id").to_pylist()
    )


def test_no_extra_outputs_appear_on_a_second_run(release, tmp_path):
    first = _run(release, tmp_path, "one")
    lake_files = sorted(
        path.relative_to(release.settings.local_lake_root).as_posix()
        for path in (release.settings.local_lake_root / "silver").rglob("*.parquet")
    )
    second = _run(release, tmp_path, "two")
    assert sorted(
        path.relative_to(release.settings.local_lake_root).as_posix()
        for path in (release.settings.local_lake_root / "silver").rglob("*.parquet")
    ) == lake_files
    assert set(first.published) == set(second.published)


def test_no_incomplete_temporary_files_are_left_behind(release, tmp_path):
    _run(release, tmp_path, "one")
    leftovers = [
        path
        for path in release.settings.local_lake_root.rglob("*")
        if path.is_file() and path.name.startswith(".")
    ]
    assert leftovers == []


def test_every_silver_record_id_appears_in_the_trace(release, tmp_path):
    _run(release, tmp_path, "one")
    trace = pq.read_table(
        tmp_path / "results-one" / "part1" / "source_trace.parquet"
    )
    traced = set(trace.column("source_record_id").to_pylist())
    for relative, identifiers in _all_record_ids(release).items():
        missing = set(identifiers) - traced
        assert not missing, f"{relative}: {len(missing)} untraced record(s)"


def test_reconciliation_passes_in_the_run_record(release, tmp_path):
    _run(release, tmp_path, "one")
    counts = json.loads(
        (tmp_path / "results-one" / "part1" / "row_counts.json").read_text()
    )
    assert counts["reconciliation"]
    for check in counts["reconciliation"]:
        assert check["passed"], check["check"]


def test_run_record_captures_input_hashes_and_revision(release, tmp_path):
    _run(release, tmp_path, "one")
    record = json.loads((tmp_path / "results-one" / "part1" / "run.json").read_text())
    assert record["run_id"] == "run-one"
    assert record["code_revision"]
    assert record["started_at"] and record["finished_at"]
    assert set(record["inputs"]) == {"qec_syndromes", "google_qec", "qasmbench"}
    for source_name, entry in record["inputs"].items():
        assert entry["sha256"] == release.sha256(source_name)
        assert entry["bytes"] > 0
    # Every output table's row count is recorded.
    assert record["outputs"]


def test_trace_examples_demonstrate_both_required_traces(release, tmp_path):
    _run(release, tmp_path, "one")
    examples = json.loads(
        (tmp_path / "results-one" / "part1" / "trace_examples.json").read_text()
    )
    syndrome = examples["syndrome"]
    assert syndrome["record"]["source_record_id"]
    assert syndrome["bronze_trace"], "a syndrome must resolve to a Bronze member"
    assert syndrome["occurs_under_both_labels"] is True

    shot = examples["google"]
    assert shot["record"]["source_record_id"]
    # One shot is assembled from several aligned companion members.
    assert len(shot["bronze_trace"]) == 8
    assert all(row["input_sha256"] for row in shot["bronze_trace"])


def test_a_failed_run_leaves_the_previous_outputs_intact(release, tmp_path):
    """A fatal rule must not publish, so the last good Silver survives."""
    import conftest
    import pytest

    from quantum_lake_student.quality import FatalDataError

    _run(release, tmp_path, "one")
    before = {
        spec.relative_path: _digest(
            release.settings.local_lake_root / spec.relative_path
        )
        for spec in SILVER_TABLES
    }

    # Break the release: drop a required companion file.
    members = conftest.google_members(omit="obs_flips_actual.01")
    payload = conftest.zip_bytes(members)
    relative = "raw/source=google_qec/google-surface-code-curated.zip"
    (release.settings.local_lake_root / relative).write_bytes(payload)
    manifest_path = (
        release.settings.local_lake_root / "metadata" / "bundle-manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["objects"]:
        if entry["source"] == "google_qec":
            entry["bytes"] = len(payload)
            entry["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2))

    with pytest.raises(FatalDataError):
        _run(release, tmp_path, "broken")

    after = {
        spec.relative_path: _digest(
            release.settings.local_lake_root / spec.relative_path
        )
        for spec in SILVER_TABLES
    }
    assert after == before
