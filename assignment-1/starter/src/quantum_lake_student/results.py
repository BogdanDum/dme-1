"""Write the Part I evidence files under ``results/part1``.

These are the machine-readable record of what a run did: which bytes went in,
which rules fired, how the counts reconcile, and how a published record is
traced back to Bronze.

The two demonstrated traces are chosen by rule rather than by a hardcoded id, so
they are reproducible and so each one is actually interesting: the syndrome
example is a pattern that occurs under *both* labels, and the Google example is a
shot where at least one supplied decoder was wrong.
"""

from __future__ import annotations

import io
import json
from typing import Any

import pyarrow.compute as pc
import pyarrow.parquet as pq

from .lake import Lake, PublishResult
from .runcontext import RunContext, utc_now
from .schemas import (
    DECODER_PREDICTION_COLUMNS,
    GOOGLE_SHOT,
    SILVER_TABLES,
    SOURCE_TRACE,
    SYNDROME_OBSERVATION,
)
from .stages.prepare_data import SilverOutcome
from .stages.register_sources import RegisteredSource

RUN_RECORD = "part1/run.json"
ROW_COUNTS = "part1/row_counts.json"
TRACE_EXAMPLES = "part1/trace_examples.json"

# The fault rate the supplied helper assigns to the test split. Choosing the
# syndrome example from here makes the trace demonstrate a record Part II will
# actually evaluate on.
EXAMPLE_FAULT_RATE = 0.005


def write_all(
    context: RunContext,
    *,
    registry: dict[str, RegisteredSource],
    outcome: SilverOutcome,
    stage_results: list[Any],
    published: dict[str, PublishResult],
) -> None:
    """Write run.json, row_counts.json and trace_examples.json."""
    _write_json(
        context.lake,
        RUN_RECORD,
        _run_record(context, registry, outcome, stage_results, published),
    )
    _write_json(
        context.lake, ROW_COUNTS, _row_counts(context, outcome, published)
    )
    _write_json(context.lake, TRACE_EXAMPLES, _trace_examples(context))


def _write_json(lake: Lake, relative_path: str, payload: dict[str, Any]) -> None:
    lake.write_results_text(
        relative_path, json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def _run_record(
    context: RunContext,
    registry: dict[str, RegisteredSource],
    outcome: SilverOutcome,
    stage_results: list[Any],
    published: dict[str, PublishResult],
) -> dict[str, Any]:
    return {
        "run_id": context.run_id,
        "code_revision": context.code_revision,
        "started_at": context.started_at.isoformat(),
        "finished_at": utc_now().isoformat(),
        "lake_backend": context.settings.lake_backend,
        "inputs": {
            source_name: {
                "bronze_object": registered.bronze_object,
                "key": registered.key,
                "bytes": registered.byte_size,
                "sha256": registered.sha256,
                "archive_members": registered.member_count,
            }
            for source_name, registered in sorted(registry.items())
        },
        "stages": [
            {
                "stage": stage.stage,
                "input_count": stage.input_count,
                "output_count": stage.output_count,
                "issue_count": stage.issue_count,
                "started_at": stage.started_at.isoformat(),
                "finished_at": (
                    stage.finished_at.isoformat() if stage.finished_at else None
                ),
            }
            for stage in stage_results
        ],
        "outputs": {
            relative_path: {
                "rows": result.row_count,
                "bytes": result.byte_size,
                "sha256": result.sha256,
            }
            for relative_path, result in sorted(published.items())
        },
        "issues": {
            "total": outcome.issue_rows,
            "by_rule": _issue_counts_by_rule(context),
        },
        # Reserved so the Gold stage extends this record instead of restructuring it.
        "silver_to_gold": {"status": "pending"},
    }


def _row_counts(
    context: RunContext, outcome: SilverOutcome, published: dict[str, PublishResult]
) -> dict[str, Any]:
    reconciliation = [
        {
            "check": f"{table}: read == accepted + rejected",
            "read": counts.read,
            "accepted": counts.accepted,
            "rejected": counts.rejected,
            "passed": counts.reconciles(),
        }
        for table, counts in sorted(context.quality.counts.items())
    ]
    for spec in SILVER_TABLES:
        entry = context.quality.counts.get(spec.name)
        result = published.get(spec.relative_path)
        if entry is None or result is None:
            continue
        reconciliation.append(
            {
                "check": f"{spec.name}: published rows == accepted rows",
                "published": result.row_count,
                "accepted": entry.accepted,
                "passed": result.row_count == entry.accepted,
            }
        )
    return {
        "run_id": context.run_id,
        "tables": {
            table: counts.as_dict()
            for table, counts in sorted(context.quality.counts.items())
        },
        "silver": {
            spec.relative_path: published[spec.relative_path].row_count
            for spec in SILVER_TABLES
            if spec.relative_path in published
        },
        "results": {
            "source_trace.parquet": outcome.trace_rows,
            "data_issues.parquet": outcome.issue_rows,
        },
        "sources": outcome.sources,
        "reconciliation": reconciliation,
        # Populated once the PostgreSQL load exists.
        "loaded": {},
    }


def _trace_examples(context: RunContext) -> dict[str, Any]:
    trace = _read(context, SOURCE_TRACE.relative_path)
    return {
        "run_id": context.run_id,
        "syndrome": _syndrome_example(context, trace),
        "google": _google_example(context, trace),
    }


def _syndrome_example(context: RunContext, trace: Any) -> dict[str, Any]:
    table = _read(context, SYNDROME_OBSERVATION.relative_path)
    if table is None or not table.num_rows:
        return {"selected": None, "reason": "no syndrome rows were published"}

    subset = table.filter(
        pc.equal(table.column("physical_fault_rate"), EXAMPLE_FAULT_RATE)
    )
    if not subset.num_rows:
        subset = table

    # Prefer a syndrome that occurs under both labels: it demonstrates the case
    # the assignment calls out as valid and easy to destroy by deduplicating.
    seen: dict[bytes, set[bool]] = {}
    for bits, label in zip(
        subset.column("syndrome_bits").to_pylist(),
        subset.column("logical_error_label").to_pylist(),
        strict=True,
    ):
        seen.setdefault(bits, set()).add(label)
    both = sorted(bits for bits, labels in seen.items() if len(labels) > 1)
    target = both[0] if both else sorted(seen)[0]

    chosen = subset.filter(pc.equal(subset.column("syndrome_bits"), target))
    row = chosen.slice(0, 1).to_pylist()[0]
    return {
        "selection_rule": (
            "lowest syndrome_bits at physical_fault_rate "
            f"{EXAMPLE_FAULT_RATE} that occurs under both logical-error labels"
        ),
        "occurs_under_both_labels": bool(both),
        "silver_table": SYNDROME_OBSERVATION.relative_path,
        "record": {
            "source_record_id": row["source_record_id"],
            "experiment_id": row["experiment_id"],
            "physical_fault_rate": row["physical_fault_rate"],
            "syndrome_bits_hex": row["syndrome_bits"].hex(),
            "round_count": row["round_count"],
            "check_count": row["check_count"],
            "logical_error_label": row["logical_error_label"],
            "quantity": row["quantity"],
        },
        "bronze_trace": _trace_rows_for(trace, row["source_record_id"]),
        "planned_gold_and_ml": {"status": "pending"},
    }


def _google_example(context: RunContext, trace: Any) -> dict[str, Any]:
    table = _read(context, GOOGLE_SHOT.relative_path)
    if table is None or not table.num_rows:
        return {"selected": None, "reason": "no shot rows were published"}

    actual = table.column("actual_observable_flip")
    wrong = None
    for column in DECODER_PREDICTION_COLUMNS:
        mismatch = pc.not_equal(table.column(column), actual)
        wrong = mismatch if wrong is None else pc.or_(wrong, mismatch)
    candidates = table.filter(wrong)
    if not candidates.num_rows:
        candidates = table

    row = candidates.slice(0, 1).to_pylist()[0]
    mistakes = sorted(
        column.removesuffix("_prediction")
        for column in DECODER_PREDICTION_COLUMNS
        if row[column] != row["actual_observable_flip"]
    )
    return {
        "selection_rule": (
            "first shot, in (experiment_id, shot_index) order, where at least "
            "one supplied decoder prediction differs from the actual flip"
        ),
        "silver_table": GOOGLE_SHOT.relative_path,
        "record": {
            "source_record_id": row["source_record_id"],
            "experiment_id": row["experiment_id"],
            "shot_index": row["shot_index"],
            "detector_event_count": row["detector_event_count"],
            "detector_bits_bytes": len(row["detector_bits"]),
            "measurement_bits_bytes": len(row["measurement_bits"]),
            "sweep_bits_bytes": len(row["sweep_bits"]),
            "actual_observable_flip": row["actual_observable_flip"],
            "predictions": {
                column: row[column] for column in DECODER_PREDICTION_COLUMNS
            },
        },
        # A mistake is derived, never stored in Silver. It is spelled out here so
        # the distinction between a flip and a decoder error is on the record.
        "decoders_that_were_wrong": mistakes,
        "bronze_trace": _trace_rows_for(trace, row["source_record_id"]),
        "planned_gold_and_ml": {"status": "pending"},
    }


def _trace_rows_for(trace: Any, source_record_id: str) -> list[dict[str, Any]]:
    if trace is None:
        return []
    matched = trace.filter(
        pc.equal(trace.column("source_record_id"), source_record_id)
    )
    return [
        {
            "bronze_object": row["bronze_object"],
            "archive_member": row["archive_member"],
            "record_locator": row["record_locator"],
            "input_sha256": row["input_sha256"],
        }
        for row in matched.to_pylist()
    ]


def _read(context: RunContext, relative_path: str) -> Any:
    payload = context.lake.staged_bytes(relative_path)
    if payload is None:
        return None
    return pq.read_table(io.BytesIO(payload))


def _issue_counts_by_rule(context: RunContext) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rule_id in context.quality.to_table().column("rule_id").to_pylist():
        counts[rule_id] = counts.get(rule_id, 0) + 1
    return dict(sorted(counts.items()))
