"""Real PostgreSQL tests using course settings or TEST_POSTGRES_DSN.

The role must permit CREATE DATABASE. Each test owns a disposable database,
never the configured course Gold schema.
"""

from pathlib import Path

import psycopg
from psycopg import sql
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quantum_lake_student import pipeline
from quantum_lake_student.schemas import SILVER_TABLES
from quantum_lake_student.stages import load_postgres
from quantum_lake_student.stages.postgres_model import GOLD_TABLES


QUERIES = Path(__file__).resolve().parents[1] / "results/part1/analysis/queries"
DECODERS = (
    "belief_matching", "correlated_matching", "pymatching",
    "tensor_network_contraction",
)


def query(connection, filename):
    return connection.execute((QUERIES / filename).read_text()).fetchall()


def snapshot(connect):
    with connect() as connection:
        return {
            name: connection.execute(
                sql.SQL("SELECT * FROM gold.{} ORDER BY 1, 2").format(sql.Identifier(name))
                if name != "decoder" else sql.SQL("SELECT * FROM gold.decoder ORDER BY 1")
            ).fetchall()
            for name in GOLD_TABLES
        }


@pytest.fixture
def silver(release):
    rows = {
        "syndrome_observation": [
            ("o0", "synthetic", .005, bytes(16), 4, 4, False, 90),
            ("o1", "synthetic", .005, bytes(16), 4, 4, True, 10),
        ],
        "experiment": [("meta", "hardware", "X", 3, 25, 1, 3, 5, 8, 8)],
        "shot": [("shot0", "hardware", 0, b"\x81", b"", b"\x02", 1,
                  True, True, False, True, False)],
        "circuit": [("source", "circuit", "qec_sm_n5", "source", "[]", 5, 5, 4)],
        "stabilizer_check": [("check", "circuit", "s0", "a[0]", ["q[0]", "q[1]"], "syn[0]")],
        "conditional_correction": [("correction", "circuit", "syn", 1, "x", "q[0]")],
    }
    for spec in SILVER_TABLES:
        path = release.settings.local_lake_root / spec.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist([
            dict(zip(spec.column_names, row)) for row in rows[spec.name]
        ], schema=spec.schema), path)
    return release.settings


def test_load_preserves_records_and_is_repeatable(database, silver, monkeypatch):
    monkeypatch.setattr(load_postgres, "BATCH_SIZE", 1)
    first = load_postgres.load("first", silver)
    before = snapshot(database)
    second = load_postgres.load("second", silver)
    assert snapshot(database) == before
    assert first.input_count == second.input_count == 7
    assert first.output_count == second.output_count == 16
    assert before["syndrome_observation"] == [
        ("o0", "synthetic", bytes(16), False, 90),
        ("o1", "synthetic", bytes(16), True, 10),
    ]
    assert before["shot"] == [("shot0", "hardware", 0, b"\x81", b"", b"\x02", 1, True)]
    assert {(r[1], r[2]) for r in before["decoder_prediction"]} == {
        (name, prediction) for name, prediction in zip(DECODERS, (True, False, True, False))
    }
    with database() as connection:
        assert connection.execute(
            "SELECT decoder_name, decoder_correctness FROM gold.decoder_correctness ORDER BY decoder_name"
        ).fetchall() == [(name, value) for name, value in zip(DECODERS, (True, False, True, False))]


@pytest.mark.parametrize("failure", ["constraint", "reconciliation", "missing_file"])
def test_failed_load_retains_previous_gold(database, silver, failure):
    load_postgres.load("good", silver)
    before = snapshot(database)
    spec = next(s for s in SILVER_TABLES if s.name == "experiment")
    path = silver.local_lake_root / spec.relative_path
    if failure == "missing_file":
        path.unlink()
        error = FileNotFoundError
    else:
        rows = pq.read_table(path).to_pylist()
        rows[0]["distance" if failure == "constraint" else "shots"] = -1 if failure == "constraint" else 2
        pq.write_table(pa.Table.from_pylist(rows, schema=spec.schema), path)
        error = psycopg.errors.CheckViolation if failure == "constraint" else ValueError
    with pytest.raises(error):
        load_postgres.load("bad", silver)
    assert snapshot(database) == before


def test_pipeline_loads_fresh_silver_using_supplied_settings(database, release, tmp_path, monkeypatch):
    monkeypatch.setenv("LAKE_BACKEND", "local")
    monkeypatch.setenv("LOCAL_LAKE_ROOT", str(tmp_path / "wrong-lake"))
    pipeline.run_part1(
        settings=release.settings,
        results_root=tmp_path / "results",
        repository=tmp_path,
        bronze_root=release.root,
    )
    with database() as connection:
        for spec in SILVER_TABLES:
            expected = pq.read_table(release.settings.local_lake_root / spec.relative_path)
            assert connection.execute(
                sql.SQL("SELECT count(*) FROM gold.{}").format(sql.Identifier(spec.name))
            ).fetchone()[0] == expected.num_rows
        assert connection.execute("SELECT count(*) FROM gold.shot").fetchone()[0] == 3


def test_partial_pipeline_does_not_refresh_gold(database, silver, release, tmp_path):
    load_postgres.load("previous", silver)
    before = snapshot(database)
    result = pipeline.run_part1(
        settings=release.settings,
        results_root=tmp_path / "results",
        repository=tmp_path,
        bronze_root=release.root,
        sources=("qec_syndromes",),
    )
    assert result.partial
    assert snapshot(database) == before


@pytest.mark.parametrize("statement,error", [
    ("INSERT INTO gold.syndrome_observation SELECT * FROM gold.syndrome_observation LIMIT 1",
     psycopg.errors.UniqueViolation),
    ("UPDATE gold.shot SET experiment_id = 'missing'", psycopg.errors.ForeignKeyViolation),
    ("UPDATE gold.decoder_prediction SET decoder_name = 'missing' WHERE decoder_name = 'pymatching'",
     psycopg.errors.ForeignKeyViolation),
    ("UPDATE gold.stabilizer_check SET circuit_id = 'missing'", psycopg.errors.ForeignKeyViolation),
    ("UPDATE gold.syndrome_observation SET sample_weight = 0", psycopg.errors.CheckViolation),
    ("UPDATE gold.syndrome_observation SET syndrome_bits = decode('02', 'hex')",
     psycopg.errors.CheckViolation),
    ("UPDATE gold.syndrome_observation SET syndrome_bits = decode(repeat('02', 16), 'hex')",
     psycopg.errors.CheckViolation),
    ("UPDATE gold.stabilizer_check SET data_qubits = '{}'", psycopg.errors.CheckViolation),
    ("UPDATE gold.conditional_correction SET condition_value = -1", psycopg.errors.CheckViolation),
])
def test_gold_rejects_invalid_records(database, silver, statement, error):
    load_postgres.load("good", silver)
    before = snapshot(database)
    with pytest.raises(error):
        with database() as connection:
            connection.execute(statement)
    assert snapshot(database) == before


def test_weighted_syndrome_query(database):
    with database() as connection:
        connection.execute("INSERT INTO gold.syndrome_experiment VALUES ('a', .001, 4, 4), ('b', .01, 4, 4)")
        for row in [
            ("a0", "a", bytes(16), False, 60),
            ("a1", "a", bytes(16), True, 20),
            ("a2", "a", bytes([1]) * 16, True, 20),
            ("b0", "b", bytes(16), False, 1),
            ("b1", "b", bytes([1]) * 16, True, 3),
        ]:
            connection.execute("INSERT INTO gold.syndrome_observation VALUES (%s,%s,%s,%s,%s)", row)
        result = query(connection, "query1.sql")
    expected = [(.001, bytes(16), 80, .8, .25, .4),
                (.001, bytes([1]) * 16, 20, .2, 1, .4),
                (.01, bytes([1]) * 16, 3, .75, 1, .75),
                (.01, bytes(16), 1, .25, 0, .75)]
    assert len(result) == len(expected)
    for actual, wanted in zip(result, expected):
        assert actual[:3] == wanted[:3]
        assert tuple(map(float, actual[3:])) == pytest.approx(wanted[3:])
    for rate in (.001, .01):
        assert sum(float(row[3]) for row in result if row[0] == rate) == pytest.approx(1)


def test_decoder_queries_pool_shots_and_separate_distance_location_basis_rounds(database):
    experiments = [
        ("a", 3, "X", 25, 3, 5, [False, True, True]),
        ("b", 3, "X", 25, 5, 5, [False]),
        ("c", 5, "X", 25, 3, 5, [True, True]),
        ("d", 3, "Z", 25, 3, 5, [True]),
        ("e", 3, "X", 10, 3, 5, [False]),
    ]
    with database() as connection:
        for decoder in DECODERS:
            connection.execute("INSERT INTO gold.decoder VALUES (%s)", (decoder,))
        for name, distance, basis, rounds, row, col, outcomes in experiments:
            connection.execute("INSERT INTO gold.experiment VALUES (%s,%s,%s,%s,%s,%s,%s,%s,8,8)",
                               (name, name, basis, distance, rounds, len(outcomes), row, col))
            for index, outcome in enumerate(outcomes):
                shot = f"{name}-{index}"
                connection.execute("INSERT INTO gold.shot VALUES (%s,%s,%s,%s,%s,%s,0,%s)",
                                   (shot, name, index, b"\0", b"", b"\0", outcome))
                for decoder, prediction in zip(DECODERS, (outcome, not outcome, False, True)):
                    connection.execute("INSERT INTO gold.decoder_prediction VALUES (%s,%s,%s)",
                                       (shot, decoder, prediction))
        by_distance = {r[:4]: r[4:] for r in query(connection, "query2a.sql")}
        by_location = {r[:5]: r[5:] for r in query(connection, "query2b.sql")}
    assert len(by_distance) == 16
    assert len(by_location) == 16
    for key, counts in [((3, "X", 25), (4, 2)), ((5, "X", 25), (2, 2)),
                        ((3, "Z", 25), (1, 1)), ((3, "X", 10), (1, 0))]:
        total, flips = counts
        for decoder, errors in zip(DECODERS, (0, total, flips, total - flips)):
            result = by_distance[(*key, decoder)]
            assert result[:2] == (total, errors)
            assert float(result[2]) == pytest.approx(errors / total)
    for _, distance, basis, rounds, row, col, outcomes in experiments:
        if distance != 3:
            continue
        total, flips = len(outcomes), sum(outcomes)
        for decoder, errors in zip(DECODERS, (0, total, flips, total - flips)):
            result = by_location[(row, col, basis, rounds, decoder)]
            assert result[:2] == (total, errors)
            assert float(result[2]) == pytest.approx(errors / total)


def test_circuit_query_scopes_circuit_variant_and_condition_register(database):
    with database() as connection:
        for circuit, benchmark, variant in [
            ("selected", "qec_sm_n5", "source"),
            ("transpiled", "qec_sm_n5", "transpiled"),
            ("other", "other", "source"),
        ]:
            connection.execute("INSERT INTO gold.circuit VALUES (%s,%s,%s,%s,'[]',5,5,4)",
                               (circuit, circuit, benchmark, variant))
            for index, register in enumerate(("syn", "syn", "unused")):
                connection.execute("INSERT INTO gold.stabilizer_check VALUES (%s,%s,%s,%s,%s,%s)",
                                   (f"{circuit}-{index}", circuit, f"s{index}", f"a[{index}]",
                                    [f"q[{index}]", f"q[{index + 1}]"], f"{register}[{index}]"))
            for value, target in ((1, "q[0]"), (2, "q[2]"), (3, "q[1]")):
                connection.execute("INSERT INTO gold.conditional_correction VALUES (%s,%s,'syn',%s,'x',%s)",
                                   (f"{circuit}-r{value}", circuit, value, target))
        result = query(connection, "query3.sql")
    assert result == [
        ("qec_sm_n5", "source", "syn", value, "x", target, f"s{index}",
         [f"q[{index}]", f"q[{index + 1}]"], f"a[{index}]", f"syn[{index}]")
        for value, target in ((1, "q[0]"), (2, "q[2]"), (3, "q[1]"))
        for index in range(2)
    ]