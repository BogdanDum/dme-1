"""The published tables must match silver-tables.md exactly, column for column.

The expectations below are transcribed from the assignment contract rather than
generated from ``schemas.py``, so that editing the implementation cannot quietly
edit the thing it is checked against.
"""

from __future__ import annotations

import io

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quantum_lake_student import schemas

CONTRACT: dict[str, list[tuple[str, pa.DataType]]] = {
    "silver/qec_syndromes/syndrome_observation.parquet": [
        ("source_record_id", pa.string()),
        ("experiment_id", pa.string()),
        ("physical_fault_rate", pa.float64()),
        ("syndrome_bits", pa.binary()),
        ("round_count", pa.int32()),
        ("check_count", pa.int32()),
        ("logical_error_label", pa.bool_()),
        ("quantity", pa.int64()),
    ],
    "silver/google_qec/experiment.parquet": [
        ("source_record_id", pa.string()),
        ("experiment_id", pa.string()),
        ("basis", pa.string()),
        ("distance", pa.int32()),
        ("rounds", pa.int32()),
        ("shots", pa.int64()),
        ("center_row", pa.int32()),
        ("center_col", pa.int32()),
        ("measurement_count", pa.int32()),
        ("detector_count", pa.int32()),
    ],
    "silver/google_qec/shot.parquet": [
        ("source_record_id", pa.string()),
        ("experiment_id", pa.string()),
        ("shot_index", pa.int64()),
        ("measurement_bits", pa.binary()),
        ("sweep_bits", pa.binary()),
        ("detector_bits", pa.binary()),
        ("detector_event_count", pa.int32()),
        ("actual_observable_flip", pa.bool_()),
        ("belief_matching_prediction", pa.bool_()),
        ("correlated_matching_prediction", pa.bool_()),
        ("pymatching_prediction", pa.bool_()),
        ("tensor_network_contraction_prediction", pa.bool_()),
    ],
    "silver/qasmbench/circuit.parquet": [
        ("source_record_id", pa.string()),
        ("circuit_id", pa.string()),
        ("benchmark_name", pa.string()),
        ("variant", pa.string()),
        ("register_declarations", pa.string()),
        ("qubit_count", pa.int32()),
        ("measurement_count", pa.int32()),
        ("two_qubit_gate_count", pa.int32()),
    ],
    "silver/qasmbench/stabilizer_check.parquet": [
        ("source_record_id", pa.string()),
        ("circuit_id", pa.string()),
        ("check_id", pa.string()),
        ("ancilla_qubit", pa.string()),
        ("data_qubits", pa.list_(pa.string())),
        ("syndrome_bit", pa.string()),
    ],
    "silver/qasmbench/conditional_correction.parquet": [
        ("source_record_id", pa.string()),
        ("circuit_id", pa.string()),
        ("condition_register", pa.string()),
        ("condition_value", pa.int64()),
        ("gate", pa.string()),
        ("target_qubit", pa.string()),
    ],
}

TRACE_COLUMNS = [
    "source_record_id",
    "source_name",
    "bronze_object",
    "archive_member",
    "record_locator",
    "input_sha256",
]

ISSUE_COLUMNS = [
    "issue_id",
    "run_id",
    "source_record_id",
    "rule_id",
    "severity",
    "observed_value",
    "action",
    "reason",
]


def test_all_six_silver_tables_are_declared():
    assert {spec.relative_path for spec in schemas.SILVER_TABLES} == set(CONTRACT)


@pytest.mark.parametrize("relative_path", sorted(CONTRACT))
def test_declared_schema_matches_the_contract(relative_path):
    spec = next(
        item
        for item in schemas.SILVER_TABLES
        if item.relative_path == relative_path
    )
    expected = CONTRACT[relative_path]
    assert list(spec.schema.names) == [name for name, _ in expected]
    for name, dtype in expected:
        assert spec.schema.field(name).type == dtype, name


def test_source_trace_has_the_required_columns():
    assert list(schemas.SOURCE_TRACE.schema.names) == TRACE_COLUMNS
    # "member path inside an archive, if applicable"
    assert schemas.SOURCE_TRACE.schema.field("archive_member").nullable


def test_data_issues_has_the_required_columns():
    assert list(schemas.DATA_ISSUES.schema.names) == ISSUE_COLUMNS
    assert schemas.DATA_ISSUES.schema.field("source_record_id").nullable
    assert schemas.DATA_ISSUES.schema.field("observed_value").nullable


def test_prediction_columns_match_the_supplied_helper():
    from quantum_lake_student.ml import GOOGLE_META_PREDICTION_COLUMNS

    assert schemas.DECODER_PREDICTION_COLUMNS == GOOGLE_META_PREDICTION_COLUMNS


def test_no_silver_table_carries_a_run_id_or_timestamp():
    for spec in schemas.SILVER_TABLES:
        for name in spec.schema.names:
            assert "run_id" not in name
            assert "timestamp" not in name
            assert not name.endswith("_at")


def test_every_silver_table_carries_the_trace_id():
    for spec in schemas.SILVER_TABLES:
        assert "source_record_id" in spec.schema.names


def test_no_silver_table_mixes_two_sources():
    """Silver stays source-separated; integration belongs to Gold."""
    for source_name, specs in schemas.SILVER_TABLES_BY_SOURCE.items():
        for spec in specs:
            assert f"silver/{source_name}/" in spec.relative_path


@pytest.mark.usefixtures("database")
def test_published_tables_read_back_with_the_contract_schema(release, tmp_path):
    from quantum_lake_student import pipeline

    result = pipeline.run_part1(
        settings=release.settings,
        results_root=tmp_path / "results",
        repository=tmp_path,
    )
    for relative_path, expected in CONTRACT.items():
        payload = result.published[relative_path]
        assert payload.row_count >= 0
        table = pq.read_table(
            tmp_path / "results" / relative_path
            if not relative_path.startswith("silver/")
            else release.settings.local_lake_root / relative_path
        )
        assert list(table.schema.names) == [name for name, _ in expected]
        for name, dtype in expected:
            assert table.schema.field(name).type == dtype, (relative_path, name)


@pytest.mark.usefixtures("database")
def test_trace_reads_back_as_plain_strings(release, tmp_path):
    from quantum_lake_student import pipeline

    pipeline.run_part1(
        settings=release.settings,
        results_root=tmp_path / "results",
        repository=tmp_path,
    )
    table = pq.read_table(tmp_path / "results" / "part1" / "source_trace.parquet")
    assert list(table.schema.names) == TRACE_COLUMNS
    for name in TRACE_COLUMNS:
        assert table.schema.field(name).type == pa.string()


def test_written_parquet_carries_no_pandas_metadata():
    table = pa.table(
        {name: [] for name in schemas.QASM_CIRCUIT.schema.names},
        schema=schemas.QASM_CIRCUIT.schema,
    )
    sink = io.BytesIO()
    pq.write_table(table, sink)
    metadata = pq.read_table(io.BytesIO(sink.getvalue())).schema.metadata or {}
    assert b"pandas" not in metadata
