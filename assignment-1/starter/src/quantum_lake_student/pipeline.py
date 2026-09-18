"""The one documented command for Part I.

Runs Bronze registration then Silver preparation against a single shared run
context, publishes Silver, refreshes Gold, and writes the Part I evidence files.

Nothing is published until every source has parsed successfully. A fatal rule --
a missing companion file or an unsafe archive member -- aborts before the commit,
so the previous complete set of outputs survives a failed run intact.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .lake import PublishResult
from .quality import FatalDataError
from .results import write_all
from .runcontext import RunContext
from .stages import load_postgres
from .stages.prepare_data import SOURCE_ORDER, SilverOutcome, prepare
from .stages.register_sources import register

EXIT_OK = 0
EXIT_DATA_STOPPED = 3


@dataclass
class Part1Result:
    run_id: str
    outcome: SilverOutcome
    published: dict[str, PublishResult]
    results_root: Path
    partial: bool = False


def default_results_root() -> Path:
    """``results/`` beside the package root, i.e. ``/workspace/results``."""
    return Path(__file__).resolve().parents[2] / "results"


def default_repository() -> Path:
    return Path(__file__).resolve().parents[2]


def run_part1(
    *,
    settings: Settings | None = None,
    results_root: Path | None = None,
    repository: Path | None = None,
    run_id: str | None = None,
    sources: tuple[str, ...] | None = None,
    bronze_root: Path | None = None,
) -> Part1Result:
    """Execute Bronze and Silver, publish, refresh Gold, and write evidence.

    ``sources`` limits the run to a subset, for iterating on one parser. Such a
    run is *partial*: its ``source_trace`` covers only the sources it processed,
    so its evidence goes under ``results/partial/<sources>/`` instead of over the
    canonical ``results/part1/``. Otherwise the Silver tables an earlier full run
    left in place would be left untraced, breaking the tracing contract.
    Partial runs do not refresh Gold from a mixture of fresh and stale Silver.
    """
    if sources:
        unknown = set(sources) - set(SOURCE_ORDER)
        if unknown:
            raise ValueError(f"Unknown source(s): {', '.join(sorted(unknown))}")

    canonical = results_root or default_results_root()
    effective = canonical / "partial" / "-".join(sorted(sources)) if sources else canonical

    context = RunContext.create(
        settings=settings or Settings.from_environment(),
        results_root=effective,
        repository=repository or default_repository(),
        run_id=run_id,
        bronze_root=bronze_root,
    )

    registry, register_result = register(context)
    if sources:
        registry = {
            name: entry for name, entry in registry.items() if name in sources
        }

    outcome, prepare_result = prepare(context, registry)
    published = context.lake.commit()
    stage_results = [register_result, prepare_result]
    if not sources:
        load_result = load_postgres.load(context.run_id, context.settings)
        stage_results.append(load_result)
    write_all(
        context,
        registry=registry,
        outcome=outcome,
        stage_results=stage_results,
        published=published,
    )
    return Part1Result(
        run_id=context.run_id,
        outcome=outcome,
        published=published,
        results_root=effective,
        partial=bool(sources),
    )


def main(
    *, sources: tuple[str, ...] | None = None, run_id: str | None = None
) -> tuple[int, Part1Result | FatalDataError]:
    """Run Part I and translate a fatal data rule into an exit code."""
    try:
        return EXIT_OK, run_part1(sources=sources, run_id=run_id)
    except FatalDataError as error:
        return EXIT_DATA_STOPPED, error
