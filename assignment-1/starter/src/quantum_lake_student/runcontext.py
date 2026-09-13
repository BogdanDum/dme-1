"""Per-run state: identifiers, the clock, code revision, and the trace spine.

One :class:`RunContext` is threaded through both stages so that the quality log,
the lineage trace, and the publisher are shared rather than rebuilt per source.

The trace deliberately offers two ways to add rows. Small sources add one row at
a time. The Google shots do not: 250,000 shots aligned from eight companion
members would be two million Python dictionaries, so those rows are built
directly as Arrow arrays instead.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa

from .config import Settings
from .lake import Lake, bronze_root_from_environment
from .quality import QualityLog
from .schemas import SOURCE_TRACE


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_run_id(clock: datetime | None = None) -> str:
    """A readable, sortable run id. Never reaches a Silver column."""
    return (clock or utc_now()).strftime("run-%Y%m%dT%H%M%SZ")


def code_revision(repository: Path) -> str:
    """The commit the run executed, or ``unversioned`` outside a checkout."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unversioned"
    revision = completed.stdout.strip()
    return revision if completed.returncode == 0 and revision else "unversioned"


TRACE_FLUSH_ROWS = 50_000
TRACE_CHUNK_RECORDS = 8_192


@dataclass
class TraceCollector:
    """Streams ``source_trace`` rows out instead of holding them all.

    When a :class:`~.lake.ParquetStream` is attached, rows are flushed to it as
    they accumulate, which keeps peak memory at one batch. Without a stream the
    rows are kept in memory, which is what the tests use.
    """

    stream: object | None = None
    _chunks: list[pa.Table] = field(default_factory=list, init=False)
    _pending: list[dict[str, str | None]] = field(default_factory=list, init=False)
    _row_count: int = field(default=0, init=False)

    def add(
        self,
        *,
        source_record_id: str,
        source_name: str,
        bronze_object: str,
        archive_member: str | None,
        record_locator: str,
        input_sha256: str,
    ) -> None:
        self._pending.append(
            {
                "source_record_id": source_record_id,
                "source_name": source_name,
                "bronze_object": bronze_object,
                "archive_member": archive_member,
                "record_locator": record_locator,
                "input_sha256": input_sha256,
            }
        )
        self._row_count += 1
        if len(self._pending) >= TRACE_FLUSH_ROWS:
            self._flush_pending()

    def add_aligned_block(
        self,
        *,
        source_record_ids: list[str],
        record_locators: list[str],
        source_name: str,
        bronze_object: str,
        archive_members: tuple[str, ...],
        input_sha256: str,
    ) -> None:
        """Add one row per (record, member) pair without materialising dicts.

        Used for Google shots: each shot is assembled from several aligned
        companion members, and the tracing contract explicitly allows several
        trace rows to share one ``source_record_id``.
        """
        if len(source_record_ids) != len(record_locators):
            raise ValueError("record ids and locators must be the same length")
        records = len(source_record_ids)
        if not records or not archive_members:
            return
        # Rows are emitted record-major: a record's member rows are adjacent, so
        # its repeated id compresses away and the file stays small. Emitting
        # member-major instead scatters them and costs roughly seven times the
        # bytes. Written in chunks so the block is never fully materialised.
        for start in range(0, records, TRACE_CHUNK_RECORDS):
            stop = min(start + TRACE_CHUNK_RECORDS, records)
            width = stop - start
            total = width * len(archive_members)
            self._write(
                pa.table(
                    {
                        "source_record_id": pa.array(
                            [
                                identifier
                                for identifier in source_record_ids[start:stop]
                                for _ in archive_members
                            ],
                            type=pa.string(),
                        ),
                        "source_name": pa.array(
                            [source_name] * total, type=pa.string()
                        ),
                        "bronze_object": pa.array(
                            [bronze_object] * total, type=pa.string()
                        ),
                        "archive_member": pa.array(
                            list(archive_members) * width, type=pa.string()
                        ),
                        "record_locator": pa.array(
                            [
                                locator
                                for locator in record_locators[start:stop]
                                for _ in archive_members
                            ],
                            type=pa.string(),
                        ),
                        "input_sha256": pa.array(
                            [input_sha256] * total, type=pa.string()
                        ),
                    },
                    schema=SOURCE_TRACE.schema,
                )
            )
            self._row_count += total

    @property
    def row_count(self) -> int:
        return self._row_count

    def _pending_table(self) -> pa.Table:
        return pa.table(
            {
                name: [row[name] for row in self._pending]
                for name in SOURCE_TRACE.column_names
            },
            schema=SOURCE_TRACE.schema,
        )

    def _flush_pending(self) -> None:
        if not self._pending:
            return
        table = self._pending_table()
        self._pending.clear()
        self._write(table)

    def _write(self, table: pa.Table) -> None:
        if self.stream is not None:
            self.stream.write(table)  # type: ignore[attr-defined]
        else:
            self._chunks.append(table)

    def finish(self) -> None:
        """Flush any buffered rows. Call once, after every source has run."""
        self._flush_pending()

    def to_table(self) -> pa.Table:
        """Materialise the trace. Only valid when no stream is attached."""
        chunks = list(self._chunks)
        if self._pending:
            chunks.append(self._pending_table())
        if not chunks:
            return SOURCE_TRACE.schema.empty_table()
        return pa.concat_tables(chunks)


@dataclass
class RunContext:
    """Everything a stage needs, created once per run."""

    run_id: str
    settings: Settings
    lake: Lake
    quality: QualityLog
    trace: TraceCollector
    code_revision: str
    started_at: datetime
    bronze_hashes: dict[str, str] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        settings: Settings,
        results_root: Path,
        repository: Path,
        run_id: str | None = None,
        bronze_root: Path | None = None,
    ) -> RunContext:
        started_at = utc_now()
        identifier = run_id or new_run_id(started_at)
        return cls(
            run_id=identifier,
            settings=settings,
            lake=Lake(
                settings=settings,
                results_root=results_root,
                bronze_root=bronze_root or bronze_root_from_environment(settings),
            ),
            quality=QualityLog(run_id=identifier),
            trace=TraceCollector(),
            code_revision=code_revision(repository),
            started_at=started_at,
        )
