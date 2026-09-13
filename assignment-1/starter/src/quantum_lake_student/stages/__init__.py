"""Pipeline stages to be designed and implemented by the student team.

``register_sources`` covers Bronze: it verifies the supplied bytes against the
release manifest and lists archive members safely. ``prepare_data`` covers
Silver: it parses each source, applies the documented checks, and publishes the
six required Parquet tables.

The remaining three stages are still the supplied placeholders.
"""

from . import (
    build_ml_tables,
    load_postgres,
    prepare_data,
    register_sources,
    train,
)

__all__ = [
    "build_ml_tables",
    "load_postgres",
    "prepare_data",
    "register_sources",
    "train",
]
