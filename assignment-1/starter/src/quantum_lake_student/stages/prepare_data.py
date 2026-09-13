"""Parse, check, and publish the Silver tables.

Apply documented checks, normalize nested/wide data, connect valid
relationships, and write prepared Parquet tables with chosen column names and
types. Write invalid records and the reason for exclusion to a machine-readable
data-issues output. The project README defines where generated files live.

Sources are processed in a fixed order and each one only ever touches its own
archive. The three sources are never joined here: they share QEC vocabulary but
no row-level identifier, and integrating them is Gold's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import StageResult
from ..runcontext import RunContext
from ..schemas import DATA_ISSUES, SOURCE_TRACE
from ..sources import google, qasm, syndromes
from .register_sources import RegisteredSource

STAGE = "prepare_data"

# Fixed order, so the streamed trace is emitted reproducibly.
SOURCE_ORDER = ("google_qec", "qasmbench", "qec_syndromes")


@dataclass
class SilverOutcome:
    """Row counts and per-source reconciliation detail for one run."""

    table_rows: dict[str, int] = field(default_factory=dict)
    sources: dict[str, dict[str, object]] = field(default_factory=dict)
    trace_rows: int = 0
    issue_rows: int = 0


def run(run_id: str) -> StageResult:
    """Entry point kept for the supplied stage contract."""
    from pathlib import Path

    from ..config import Settings
    from .register_sources import register

    context = RunContext.create(
        settings=Settings.from_environment(),
        results_root=Path("results"),
        repository=Path.cwd(),
        run_id=run_id,
    )
    registry, _ = register(context)
    _, result = prepare(context, registry)
    context.lake.commit()
    return result


def prepare(
    context: RunContext, registry: dict[str, RegisteredSource]
) -> tuple[SilverOutcome, StageResult]:
    """Build every Silver table and stage the Part I evidence tables."""
    result = StageResult(stage=STAGE, run_id=context.run_id)
    outcome = SilverOutcome()

    trace_stream = context.lake.open_stream(SOURCE_TRACE)
    context.trace.stream = trace_stream
    try:
        for source_name in SOURCE_ORDER:
            registered = registry.get(source_name)
            if registered is None:
                continue
            _build_source(context, source_name, registered, outcome)
    finally:
        context.trace.finish()
        outcome.trace_rows = trace_stream.close().row_count

    issues = context.lake.stage_table(DATA_ISSUES, context.quality.to_table())
    outcome.issue_rows = issues.row_count

    result.input_count = len(registry)
    result.output_count = sum(outcome.table_rows.values())
    result.issue_count = context.quality.issue_count
    result.finish()
    return outcome, result


def _build_source(
    context: RunContext,
    source_name: str,
    registered: RegisteredSource,
    outcome: SilverOutcome,
) -> None:
    key = registered.key
    bronze_object = registered.bronze_object
    sha256 = registered.sha256

    if source_name == "qec_syndromes":
        built = syndromes.build(context, key, bronze_object, sha256)
        outcome.table_rows["syndrome_observation"] = built.row_count
        outcome.sources[source_name] = {
            "totals": built.totals(),
            "per_file": built.per_file,
        }
        return

    if source_name == "google_qec":
        built = google.build(context, key, bronze_object, sha256)
        outcome.table_rows["experiment"] = built.experiment_rows
        outcome.table_rows["shot"] = built.shot_rows
        outcome.sources[source_name] = {
            "experiments": built.experiment_rows,
            "shots": built.shot_rows,
            "per_experiment": built.per_experiment,
        }
        return

    if source_name == "qasmbench":
        counts = qasm.build(context, key, bronze_object, sha256)
        outcome.table_rows.update(counts)
        outcome.sources[source_name] = {"tables": counts}
        return

    raise ValueError(f"No parser is registered for source {source_name!r}")
