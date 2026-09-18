"""Google checks: byte lengths, bit order, padding, alignment, companion files.

Several of these rules never fire on the supplied release -- its lengths are
correct and its padding bits are already zero -- so they are proven against
synthetic archives built here. A rule that cannot be shown to fire is not
evidence of anything.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import conftest
import numpy as np
import pytest

from quantum_lake_student.formats import b8_record_bytes, iter_b8_records
from quantum_lake_student.ml import unpack_little_endian_bits
from quantum_lake_student.quality import FatalDataError
from quantum_lake_student.runcontext import RunContext
from quantum_lake_student.schemas import DECODER_PREDICTION_COLUMNS
from quantum_lake_student.sources import google
from quantum_lake_student.stages.register_sources import register


def _context(release, tmp_path: Path) -> RunContext:
    return RunContext.create(
        settings=release.settings,
        results_root=tmp_path / "results",
        repository=tmp_path,
        run_id="run-test",
        bronze_root=release.root,
    )


def _build(release, tmp_path: Path):
    context = _context(release, tmp_path)
    registry, _ = register(context)
    entry = registry["google_qec"]
    outcome = google.build(
        context, entry.key, entry.bronze_object, entry.sha256
    )
    return context, outcome


# --------------------------------------------------------------------- bit order


def test_padding_mask_selects_only_the_unused_high_bits():
    assert google.padding_mask(200) == 0  # 25 whole bytes, nothing to mask
    assert google.padding_mask(600) == 0
    assert google.padding_mask(209) == 0b1111_1110  # one valid bit in byte 27
    assert google.padding_mask(9) == 0b1111_1110
    assert google.padding_mask(25) == 0b1111_1110


def test_vectorised_bit_count_matches_the_supplied_reader():
    rows = [
        [1, 0, 1, 0, 0, 0, 0, 1, 1, 1],
        [0] * 10,
        [1] * 10,
    ]
    payload = conftest.bits_to_b8(rows, 10)
    record_bytes = b8_record_bytes(10)
    grid = np.frombuffer(payload, dtype=np.uint8).reshape(len(rows), record_bytes)

    fast = google.count_set_bits(grid, 10).tolist()
    reference = [sum(record) for record in iter_b8_records(payload, bits_per_record=10)]
    assert fast == reference == [5, 0, 10]


def test_bit_order_agrees_with_the_ml_helper():
    rows = [[1, 0, 1, 0, 0, 0, 0, 1, 1, 1]]
    payload = conftest.bits_to_b8(rows, 10)
    assert unpack_little_endian_bits(payload, 10) == tuple(rows[0])


# ------------------------------------------------------------------- happy path


def test_publishes_one_row_per_shot(release, tmp_path):
    context, outcome = _build(release, tmp_path)
    assert outcome.experiment_rows == 1
    assert outcome.shot_rows == conftest.GOOGLE_SHOTS


def test_detector_event_counts_are_the_number_of_set_bits(release, tmp_path):
    import io

    import pyarrow.parquet as pq

    context, _ = _build(release, tmp_path)
    payload = context.lake.staged_bytes("silver/google_qec/shot.parquet")
    table = pq.read_table(io.BytesIO(payload)).sort_by("shot_index")
    assert table.column("detector_event_count").to_pylist() == [1, 3, 0]
    # Recomputing from the stored packed bytes must give the same answer.
    for row in table.to_pylist():
        recomputed = sum(
            unpack_little_endian_bits(row["detector_bits"], conftest.GOOGLE_DETECTORS)
        )
        assert recomputed == row["detector_event_count"]


def test_packed_columns_keep_their_declared_widths(release, tmp_path):
    import io

    import pyarrow.parquet as pq

    context, _ = _build(release, tmp_path)
    table = pq.read_table(
        io.BytesIO(context.lake.staged_bytes("silver/google_qec/shot.parquet"))
    )
    for row in table.to_pylist():
        assert len(row["detector_bits"]) == b8_record_bytes(conftest.GOOGLE_DETECTORS)
        assert len(row["measurement_bits"]) == b8_record_bytes(
            conftest.GOOGLE_MEASUREMENTS
        )
        assert len(row["sweep_bits"]) == b8_record_bytes(conftest.GOOGLE_SWEEP_BITS)


def test_a_decoder_mistake_is_derived_not_a_stored_flip(release, tmp_path):
    _, outcome = _build(release, tmp_path)
    stats = outcome.per_experiment[conftest.GOOGLE_DIRECTORY]
    # The fixture makes each decoder wrong on a different shot, so mistakes are
    # non-zero while the actual flip count is a different number entirely.
    assert stats["actual_flip_count"] == 2
    assert all(count == 1 for count in stats["decoder_mistakes"].values())
    # A mistake count is not the flip count: conflating the two is the trap.
    assert stats["actual_flip_count"] != sum(stats["decoder_mistakes"].values())


def test_shot_rows_carry_no_decoder_error_column(release, tmp_path):
    import io

    import pyarrow.parquet as pq

    context, _ = _build(release, tmp_path)
    names = pq.read_table(
        io.BytesIO(context.lake.staged_bytes("silver/google_qec/shot.parquet"))
    ).schema.names
    assert not any("mistake" in name or "error" in name for name in names)


def test_clean_experiment_records_no_issues(release, tmp_path):
    context, _ = _build(release, tmp_path)
    assert context.quality.issue_count == 0


# ------------------------------------------------------------------ fatal rules


def test_missing_companion_file_stops_the_run(tmp_path):
    release = conftest.write_release(
        tmp_path / "bronze", omit="obs_flips_predicted_by_pymatching.01"
    )
    with pytest.raises(FatalDataError) as raised:
        _build(release, tmp_path)
    assert raised.value.rule_id == "google_qec.experiment.companion_file_missing"


def test_truncated_b8_stops_the_run(tmp_path):
    root = tmp_path / "bronze"
    conftest.write_release(root)
    # Rewrite the archive with a detector file one record short.
    members = conftest.google_members()
    key = f"{conftest.GOOGLE_DIRECTORY}/detection_events.b8"
    members[key] = members[key][:-2]
    _rewrite(root, members)

    with pytest.raises(FatalDataError) as raised:
        _build(_reload(root), tmp_path)
    assert raised.value.rule_id == "google_qec.b8.record_length_mismatch"


def test_short_prediction_file_stops_the_run(tmp_path):
    root = tmp_path / "bronze"
    conftest.write_release(root)
    members = conftest.google_members()
    members[f"{conftest.GOOGLE_DIRECTORY}/obs_flips_actual.01"] = (
        conftest.zero_one_file([1, 0])  # two lines for three shots
    )
    _rewrite(root, members)

    with pytest.raises(FatalDataError) as raised:
        _build(_reload(root), tmp_path)
    assert raised.value.rule_id == "google_qec.01.line_count_mismatch"


def test_non_binary_prediction_value_stops_the_run(tmp_path):
    root = tmp_path / "bronze"
    conftest.write_release(root)
    members = conftest.google_members()
    members[f"{conftest.GOOGLE_DIRECTORY}/obs_flips_actual.01"] = b"1\n2\n0\n"
    _rewrite(root, members)

    with pytest.raises(FatalDataError) as raised:
        _build(_reload(root), tmp_path)
    assert raised.value.rule_id == "google_qec.01.value_domain"


# ---------------------------------------------------------------- padding rule


def test_non_zero_padding_bit_excludes_only_that_shot(tmp_path):
    release = conftest.write_release(tmp_path / "bronze", measurement_padding=True)
    context, outcome = _build(release, tmp_path)

    assert outcome.shot_rows == conftest.GOOGLE_SHOTS - 1
    issues = [
        issue
        for issue in context.quality.to_table().to_pylist()
        if issue["rule_id"] == "google_qec.b8.padding_bits_set"
    ]
    assert len(issues) == 1
    assert issues[0]["action"] == "excluded"
    assert issues[0]["observed_value"].startswith("0x")
    stats = outcome.per_experiment[conftest.GOOGLE_DIRECTORY]
    assert stats["shots_read"] == conftest.GOOGLE_SHOTS
    assert stats["shots_accepted"] == conftest.GOOGLE_SHOTS - 1
    assert stats["shots_rejected"] == 1


def test_experiment_with_no_sweep_bits_stores_an_empty_value(tmp_path):
    """The contract allows an empty sweep value; prove that path works.

    The supplied release always has 9 or 25 sweep bits, so this branch can only
    be reached with a synthetic experiment.
    """
    import io

    import pyarrow.parquet as pq

    release = conftest.write_release(tmp_path / "bronze", sweep_bits=0)
    context, outcome = _build(release, tmp_path)

    assert outcome.shot_rows == conftest.GOOGLE_SHOTS
    table = pq.read_table(
        io.BytesIO(context.lake.staged_bytes("silver/google_qec/shot.parquet"))
    )
    assert set(table.column("sweep_bits").to_pylist()) == {b""}
    # Empty, not null: the column stays non-nullable.
    assert table.column("sweep_bits").null_count == 0
    # circuit_sweep_bits == data_qubits no longer holds, so the invariant fires.
    rules = {issue["rule_id"] for issue in context.quality.to_table().to_pylist()}
    assert "google_qec.experiment.derived_invariant_mismatch" in rules


def test_out_of_order_batches_are_rejected():
    """The streamed shot table is never sorted after the fact, so order is checked."""
    import pyarrow as pa

    from quantum_lake_student.lake import assert_sorted
    from quantum_lake_student.schemas import GOOGLE_SHOT

    def batch(experiment: str, indexes: list[int]) -> pa.RecordBatch:
        rows = len(indexes)
        return pa.record_batch(
            {
                "source_record_id": pa.array(["x"] * rows, type=pa.string()),
                "experiment_id": pa.array([experiment] * rows, type=pa.string()),
                "shot_index": pa.array(indexes, type=pa.int64()),
                "measurement_bits": pa.array([b""] * rows, type=pa.binary()),
                "sweep_bits": pa.array([b""] * rows, type=pa.binary()),
                "detector_bits": pa.array([b""] * rows, type=pa.binary()),
                "detector_event_count": pa.array([0] * rows, type=pa.int32()),
                "actual_observable_flip": pa.array([False] * rows, type=pa.bool_()),
                **{
                    column: pa.array([False] * rows, type=pa.bool_())
                    for column in DECODER_PREDICTION_COLUMNS
                },
            },
            schema=GOOGLE_SHOT.schema,
        )

    highest = assert_sorted(GOOGLE_SHOT, batch("a", [0, 1, 2]))
    assert highest == ("a", 2)
    # A batch that goes backwards relative to the previous one is caught.
    with pytest.raises(ValueError, match="out of order"):
        assert_sorted(GOOGLE_SHOT, batch("a", [1, 2]), highest)
    # A batch that is internally unsorted is caught.
    with pytest.raises(ValueError, match="not sorted"):
        assert_sorted(GOOGLE_SHOT, batch("a", [5, 3]))


# ------------------------------------------------------------------- utilities


def _rewrite(root: Path, members: dict[str, bytes]) -> None:
    """Replace the Google archive and its manifest entry."""
    import hashlib
    import json

    payload = conftest.zip_bytes(members)
    relative = "raw/source=google_qec/google-surface-code-curated.zip"
    (root / relative).write_bytes(payload)

    manifest_path = root / "metadata" / "bundle-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["objects"]:
        if entry["source"] == "google_qec":
            entry["bytes"] = len(payload)
            entry["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2))


def _reload(root: Path):
    """A Release view of an already-written root, after _rewrite."""
    from quantum_lake_student.config import Settings

    settings = Settings(
        lake_backend="local",
        local_lake_root=root,
        s3_endpoint="http://minio:9000",
        s3_access_key="test",
        s3_secret_key="test",
        s3_bucket="quantum-lake",
        postgres_host="postgres",
        postgres_port=5432,
        postgres_db="quantum_lake",
        postgres_user="quantum",
        postgres_password="test",
    )
    return conftest.Release(root=root, settings=settings, objects={})


def test_archive_with_an_unsafe_member_stops_the_run(tmp_path):
    root = tmp_path / "bronze"
    conftest.write_release(root)
    members = conftest.google_members()
    members["../escape.b8"] = b"\x00"
    _rewrite(root, members)

    with pytest.raises(FatalDataError) as raised:
        context = _context(_reload(root), tmp_path)
        register(context)
    assert raised.value.rule_id == "bronze.archive.unsafe_member_path"


def test_tampered_object_stops_the_run(tmp_path):
    root = tmp_path / "bronze"
    conftest.write_release(root)
    target = root / "raw/source=qasmbench/qasmbench-qec.zip"
    payload = bytearray(target.read_bytes())
    payload[-1] ^= 0xFF  # same length, different bytes
    target.write_bytes(bytes(payload))

    with pytest.raises(FatalDataError) as raised:
        context = _context(_reload(root), tmp_path)
        register(context)
    assert raised.value.rule_id in {
        "bronze.object.sha256_mismatch",
        "bronze.archive.corrupt",
    }
