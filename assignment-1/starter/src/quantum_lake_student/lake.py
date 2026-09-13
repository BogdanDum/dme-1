"""Reading Bronze and publishing Silver, over MinIO or a local directory.

Bronze is three ZIP objects that are never unpacked on disk. They are read into
memory and opened with :mod:`zipfile`, so the supplied bytes are never rewritten,
renamed, or corrected in place.

Publishing is staged: every table is serialised to Parquet in memory first, and
only once all of them succeed are they written out. A run that fails partway
therefore leaves the previous complete set of Silver files untouched, instead of
mixing new and stale tables.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from minio.error import S3Error

from .config import Settings
from .connections import bronze_inventory, minio_client
from .schemas import TableSpec

# Fixed so that identical rows produce identical bytes across runs and machines.
COMPRESSION = "zstd"
ROW_GROUP_SIZE = 20_000
SILVER_PREFIX = "silver/"


@dataclass(frozen=True)
class BronzeObject:
    """One supplied archive, addressed by its stored key."""

    source_name: str
    key: str
    byte_size: int

    @property
    def object_name(self) -> str:
        return self.key.rsplit("/", 1)[-1]


@dataclass(frozen=True)
class PublishResult:
    relative_path: str
    row_count: int
    byte_size: int
    sha256: str


def bronze_root_from_environment(settings: Settings) -> Path | None:
    """Where to read Bronze from on the local backend.

    Bronze is supplied read-only -- the container mounts it at ``/course-data``
    with ``:ro`` -- so it must not double as the write root. Returning ``None``
    means "same as the lake root", which is what the tests use.
    """
    configured = os.getenv("LOCAL_BRONZE_ROOT")
    if configured:
        return Path(configured).resolve()
    mounted = Path("/course-data")
    if mounted.is_dir():
        return mounted
    return None


@dataclass
class Lake:
    """Bronze reads and staged Silver writes for one run.

    ``bronze_root`` is kept separate from ``settings.local_lake_root`` so that
    published tables can never land inside the supplied source tree.
    """

    settings: Settings
    results_root: Path
    bronze_root: Path | None = None
    _staged: dict[str, bytes] = field(default_factory=dict, init=False)
    _published: dict[str, PublishResult] = field(default_factory=dict, init=False)

    # ------------------------------------------------------------------ Bronze

    @property
    def _bronze_settings(self) -> Settings:
        if self.bronze_root is None:
            return self.settings
        return replace(self.settings, local_lake_root=self.bronze_root)

    def bronze_objects(self) -> tuple[BronzeObject, ...]:
        """Discover the supplied archives via the provided inventory helper."""
        found: list[BronzeObject] = []
        for key, size in bronze_inventory(self._bronze_settings):
            source_name = _source_from_key(key)
            if source_name is None or not key.endswith(".zip"):
                continue
            found.append(BronzeObject(source_name, key, size))
        return tuple(sorted(found, key=lambda item: item.source_name))

    def read_bronze(self, key: str) -> bytes:
        if self.settings.lake_backend == "local":
            return (self._bronze_settings.local_lake_root / key).read_bytes()
        response = minio_client(self.settings).get_object(self.settings.s3_bucket, key)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def open_bronze_zip(self, key: str) -> zipfile.ZipFile:
        """Open a Bronze archive from memory, leaving the stored bytes alone."""
        return zipfile.ZipFile(io.BytesIO(self.read_bronze(key)))

    def read_release_manifest(self) -> dict:
        """Load the course release manifest.

        The seeding step mirrors the bundle's ``metadata/`` directory to
        ``metadata/course-release/``, so the key differs between backends while
        the bytes are the same.
        """
        candidates = (
            "metadata/course-release/bundle-manifest.json",
            "metadata/bundle-manifest.json",
        )
        for key in candidates:
            try:
                return json.loads(self.read_bronze(key))
            except (OSError, S3Error):
                continue
        raise FileNotFoundError(
            f"No release manifest found at any of {candidates}"
        )

    # ----------------------------------------------------------------- staging

    def stage_table(self, spec: TableSpec, table: pa.Table) -> PublishResult:
        """Sort, validate against the contract, and serialise one table."""
        prepared = _sorted_to_contract(spec, table)
        return self._stage_bytes(spec, _serialise(spec, prepared), prepared.num_rows)

    def stage_batches(
        self, spec: TableSpec, batches: Iterable[pa.RecordBatch]
    ) -> PublishResult:
        """Serialise a table that is too large to hold as one Arrow table.

        Batches must already arrive in ``spec.sort_by`` order; the caller knows
        the source iteration order, and :func:`assert_sorted` proves it.
        """
        sink = io.BytesIO()
        rows = 0
        writer = pq.ParquetWriter(
            sink,
            spec.schema,
            compression=COMPRESSION,
            version="2.6",
            write_statistics=True,
        )
        try:
            for batch in batches:
                writer.write_batch(batch, row_group_size=ROW_GROUP_SIZE)
                rows += batch.num_rows
        finally:
            writer.close()
        return self._stage_bytes(spec, sink.getvalue(), rows)

    def open_stream(self, spec: TableSpec) -> ParquetStream:
        """Stage a table incrementally, so its rows are never all in memory.

        Used for ``source_trace``, which has two million rows: materialising it
        as one Arrow table costs about 530 MB because every row repeats the
        source name, object name and input hash, while the compressed Parquet is
        around 11 MB.
        """
        return ParquetStream(spec, self)

    def _stage_bytes(
        self, spec: TableSpec, payload: bytes, row_count: int
    ) -> PublishResult:
        result = PublishResult(
            relative_path=spec.relative_path,
            row_count=row_count,
            byte_size=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        self._staged[spec.relative_path] = payload
        self._published[spec.relative_path] = result
        return result

    # ------------------------------------------------------------------ commit

    def commit(self) -> dict[str, PublishResult]:
        """Write every staged table, replacing any previous version atomically.

        The staged bytes are kept afterwards so the evidence writer can read the
        published tables back without a round trip to storage.
        """
        for relative_path, payload in sorted(self._staged.items()):
            if relative_path.startswith(SILVER_PREFIX):
                self._write_lake(relative_path, payload)
            else:
                self._write_results(relative_path, payload)
        return dict(self._published)

    def staged_bytes(self, relative_path: str) -> bytes | None:
        return self._staged.get(relative_path)

    def _write_lake(self, relative_path: str, payload: bytes) -> None:
        if self.settings.lake_backend == "local":
            _replace_file(self.settings.local_lake_root / relative_path, payload)
            return
        # A single PUT is atomic: readers see either the old object or the new
        # one, never a partial write.
        minio_client(self.settings).put_object(
            self.settings.s3_bucket,
            relative_path,
            io.BytesIO(payload),
            length=len(payload),
            content_type="application/vnd.apache.parquet",
        )

    def _write_results(self, relative_path: str, payload: bytes) -> None:
        _replace_file(self.results_root / relative_path, payload)

    def write_results_text(self, relative_path: str, text: str) -> PublishResult:
        """Write a JSON/Markdown evidence file under ``results/``."""
        payload = text.encode("utf-8")
        _replace_file(self.results_root / relative_path, payload)
        result = PublishResult(
            relative_path=relative_path,
            row_count=0,
            byte_size=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        self._published[relative_path] = result
        return result


class ParquetStream:
    """An open Parquet sink that accepts tables and stages the result."""

    def __init__(self, spec: TableSpec, lake: Lake) -> None:
        self._spec = spec
        self._lake = lake
        self._sink = io.BytesIO()
        self._rows = 0
        self._writer = pq.ParquetWriter(
            self._sink,
            spec.schema,
            compression=COMPRESSION,
            version="2.6",
            write_statistics=True,
        )

    def write(self, table: pa.Table) -> None:
        if not table.num_rows:
            return
        self._writer.write_table(table.cast(self._spec.schema), ROW_GROUP_SIZE)
        self._rows += table.num_rows

    @property
    def row_count(self) -> int:
        return self._rows

    def close(self) -> PublishResult:
        self._writer.close()
        return self._lake._stage_bytes(self._spec, self._sink.getvalue(), self._rows)


def assert_sorted(spec: TableSpec, rows: list[dict[str, object]]) -> None:
    """Fail loudly if a streamed table is not in its declared order."""
    keys = [tuple(row[column] for column in spec.sort_by) for row in rows]
    if keys != sorted(keys):  # type: ignore[type-var]
        raise ValueError(f"{spec.name} batch is not sorted by {spec.sort_by}")


def _sorted_to_contract(spec: TableSpec, table: pa.Table) -> pa.Table:
    if tuple(table.schema.names) != spec.column_names:
        raise ValueError(
            f"{spec.name} has columns {tuple(table.schema.names)}, "
            f"contract requires {spec.column_names}"
        )
    if not spec.sort_by:
        # An empty sort key means the caller emits rows in a deterministic order
        # already; see SOURCE_TRACE.
        return table.cast(spec.schema)
    ordered = table.sort_by([(column, "ascending") for column in spec.sort_by])
    return ordered.cast(spec.schema)


def _serialise(spec: TableSpec, table: pa.Table) -> bytes:
    sink = io.BytesIO()
    pq.write_table(
        table,
        sink,
        compression=COMPRESSION,
        row_group_size=ROW_GROUP_SIZE,
        version="2.6",
        write_statistics=True,
    )
    return sink.getvalue()


def _replace_file(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _source_from_key(key: str) -> str | None:
    for part in key.split("/"):
        if part.startswith("source="):
            return part.removeprefix("source=")
    return None
