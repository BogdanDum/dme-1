"""Command-line entry point for the student workspace."""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.table import Table

from .config import Settings
from .connections import bronze_inventory, check_platform


console = Console()


def command_check(settings: Settings) -> int:
    result = check_platform(settings)
    for service, message in result.items():
        console.print(f"[green]OK[/green] {service}: {message}")
    return 0


def command_inventory(settings: Settings) -> int:
    table = Table(title="Supplied course data files (kept unchanged)")
    table.add_column("Stored path")
    table.add_column("Bytes", justify="right")
    for key, size in bronze_inventory(settings):
        table.add_row(key, f"{size:,}")
    console.print(table)
    return 0


def command_run(settings: Settings, arguments: argparse.Namespace) -> int:
    """Run Part I: verify Bronze, build Silver, write the evidence files."""
    from .pipeline import EXIT_DATA_STOPPED, run_part1
    from .quality import FatalDataError

    sources = tuple(arguments.source) if arguments.source else None
    try:
        result = run_part1(sources=sources, run_id=arguments.run_id)
    except FatalDataError as error:
        console.print(f"[red]Run stopped by rule {error.rule_id}[/red]")
        console.print(f"  at {error.record_locator}")
        console.print(f"  {error}")
        console.print("[yellow]Nothing was published.[/yellow]")
        return EXIT_DATA_STOPPED

    table = Table(title=f"Part I outputs ({result.run_id})")
    table.add_column("Path")
    table.add_column("Rows", justify="right")
    table.add_column("Bytes", justify="right")
    for relative_path, published in sorted(result.published.items()):
        table.add_row(
            relative_path, f"{published.row_count:,}", f"{published.byte_size:,}"
        )
    console.print(table)

    failed = [
        name
        for name, counts in result.outcome.sources.items()
        if isinstance(counts, dict) and counts.get("reconciles") is False
    ]
    if failed:
        console.print(f"[red]Counts did not reconcile for: {', '.join(failed)}[/red]")
        return 1
    console.print(
        f"[green]OK[/green] {result.outcome.issue_rows} issue(s) recorded, "
        f"{result.outcome.trace_rows:,} trace row(s)"
    )
    return 0


def command_train(_: Settings) -> int:
    console.print(
        "[yellow]The AI/ML stage is intentionally unimplemented.[/yellow]\n"
        "Consume the required ML input tables through the supplied helpers and "
        "write model files and the required results/part2 files."
    )
    return 2


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "command",
        choices=("check", "inventory", "run", "train"),
        help="Action to perform",
    )
    result.add_argument(
        "--run-id",
        default=None,
        help="Reuse a specific run id instead of deriving one from the clock",
    )
    result.add_argument(
        "--source",
        action="append",
        choices=("qec_syndromes", "google_qec", "qasmbench"),
        help="Limit `run` to one source; repeatable. Useful while iterating.",
    )
    return result


def main() -> None:
    arguments = parser().parse_args()
    settings = Settings.from_environment()
    if arguments.command == "run":
        raise SystemExit(command_run(settings, arguments))
    commands = {
        "check": command_check,
        "inventory": command_inventory,
        "train": command_train,
    }
    raise SystemExit(commands[arguments.command](settings))


if __name__ == "__main__":
    main()
