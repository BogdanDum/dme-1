"""Relational integration stage.

Create the student-designed PostgreSQL tables and load them as one all-or-
nothing update. Primary/foreign keys, value checks, and indexes are part of the
deliverable. Repeated loads must not create duplicate records.
"""

from __future__ import annotations

from contextlib import contextmanager
from tempfile import TemporaryFile

import pyarrow as pa
import pyarrow.parquet as pq
from psycopg import sql

from ..config import Settings
from ..connections import minio_client, postgres_connection
from ..models import StageResult
from ..schemas import SILVER_TABLES, TableSpec
from .postgres_model import DDL, GOLD_TABLES, INSERTS

BATCH_SIZE = 25_000


@contextmanager
def _parquet(settings: Settings, spec: TableSpec):
    if settings.lake_backend == "local":
        with pq.ParquetFile(settings.local_lake_root / spec.relative_path) as parquet:
            yield parquet
        return

    response = minio_client(settings).get_object(
        settings.s3_bucket, spec.relative_path
    )
    try:
        with TemporaryFile(dir=".") as stream:
            for chunk in response.stream(1024 * 1024):
                stream.write(chunk)
            stream.seek(0)
            with pq.ParquetFile(stream) as parquet:
                yield parquet
    finally:
        response.close()
        response.release_conn()


def _sql_type(dtype: pa.DataType) -> str:
    if pa.types.is_string(dtype):
        return "text"
    if pa.types.is_binary(dtype):
        return "bytea"
    if pa.types.is_boolean(dtype):
        return "boolean"
    if pa.types.is_int32(dtype):
        return "integer"
    if pa.types.is_int64(dtype):
        return "bigint"
    if pa.types.is_float64(dtype):
        return "double precision"
    if pa.types.is_list(dtype) and pa.types.is_string(dtype.value_type):
        return "text[]"
    raise TypeError(f"Unsupported Silver type: {dtype}")


def _stage(cursor, settings: Settings, spec: TableSpec) -> int:
    with _parquet(settings, spec) as parquet:
        name = sql.Identifier("silver_" + spec.name)
        columns = sql.SQL(", ").join(
            sql.SQL("{} {} NOT NULL").format(
                sql.Identifier(field.name), sql.SQL(_sql_type(field.type))
            )
            for field in spec.schema
        )
        cursor.execute(
            sql.SQL("CREATE TEMP TABLE {} ({}) ON COMMIT DROP").format(name, columns)
        )
        with cursor.copy(sql.SQL("COPY {} FROM STDIN").format(name)) as copy:
            for batch in parquet.iter_batches(
                batch_size=BATCH_SIZE, columns=list(spec.column_names)
            ):
                for row in batch.to_pylist():
                    copy.write_row(tuple(row[column] for column in spec.column_names))
        return parquet.metadata.num_rows


def postgres_run(run_id: str) -> StageResult:
    return load(run_id, Settings.from_environment())


def load(run_id: str, settings: Settings) -> StageResult:
    """Replace Gold atomically from all six Silver files, retaining source IDs.

    Input count is Silver rows; output count is physical Gold rows (including
    normalized experiments and predictions, excluding views). No files or
    platform metadata are changed. Missing files and constraint failures abort.
    """
    result = StageResult(stage="load_postgres", run_id=run_id)
    with postgres_connection(settings) as connection:
        with connection.cursor() as cursor:
            # Serialize loaders, including first-time schema creation.
            cursor.execute("SELECT pg_advisory_xact_lock(716203941)")
            counts = {spec.name: _stage(cursor, settings, spec) for spec in SILVER_TABLES}

            cursor.execute(DDL)
            # DELETE preserves MVCC snapshots and existing downstream view dependencies.
            for name in reversed(GOLD_TABLES):
                cursor.execute(sql.SQL("DELETE FROM gold.{}").format(sql.Identifier(name)))
            cursor.execute(INSERTS)
            actual = {}
            for name in GOLD_TABLES:
                cursor.execute(sql.SQL("SELECT count(*) FROM gold.{}").format(sql.Identifier(name)))
                actual[name] = cursor.fetchone()[0]
            for name, expected in counts.items():
                if actual[name] != expected:
                    raise ValueError(f"Gold row count mismatch for {name}")
            if actual["decoder_prediction"] != 4 * counts["shot"]:
                raise ValueError("Gold decoder prediction count mismatch")
            cursor.execute("""
                SELECT e.experiment_id FROM gold.experiment e
                LEFT JOIN gold.shot s USING (experiment_id)
                GROUP BY e.experiment_id, e.shots
                HAVING count(s.source_record_id_shot) <> e.shots
                    OR min(s.shot_index) <> 0 OR max(s.shot_index) <> e.shots - 1
            """)
            if cursor.fetchone() is not None:
                raise ValueError("Google experiment shot counts or indices do not reconcile")
            result.input_count = sum(counts.values())
            result.output_count = sum(actual.values())
    result.finish()
    return result
