"""Register and verify the original source files.

Student responsibilities:

- verify checksums from the release description;
- list source files and archive members safely;
- keep the supplied bytes unchanged;
- report missing or unexpected source objects;
- make a second run safe: no duplicate source records.

Bronze is read-only by construction here: archives are read into memory, hashed,
and listed. Nothing is written back, extracted to disk, or renamed. Corrections
happen later, while Silver is built.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..lake import BronzeObject
from ..models import StageResult
from ..quality import is_unsafe_member
from ..runcontext import RunContext
from ..schemas import SOURCE_NAMES

STAGE = "register_sources"
EXPECTED_BUNDLE_VERSION = 3


@dataclass(frozen=True)
class RegisteredSource:
    """One verified Bronze archive and its safe member list."""

    source_name: str
    bronze_object: str
    key: str
    byte_size: int
    sha256: str
    members: tuple[str, ...]

    @property
    def member_count(self) -> int:
        return len(self.members)


def run(run_id: str) -> StageResult:
    """Entry point kept for the supplied stage contract.

    The orchestrated pipeline calls :func:`register` with a shared
    :class:`RunContext`; this wrapper builds a throwaway one so the stage can
    still be invoked on its own.
    """
    from pathlib import Path

    from ..config import Settings

    context = RunContext.create(
        settings=Settings.from_environment(),
        results_root=Path("results"),
        repository=Path.cwd(),
        run_id=run_id,
    )
    _, result = register(context)
    return result


def register(context: RunContext) -> tuple[dict[str, RegisteredSource], StageResult]:
    """Verify every supplied object and return the registry keyed by source."""
    result = StageResult(stage=STAGE, run_id=context.run_id)
    manifest = context.lake.read_release_manifest()
    _check_bundle_version(context, manifest)

    expected = {entry["source"]: entry for entry in manifest.get("objects", [])}
    available = {item.source_name: item for item in context.lake.bronze_objects()}
    result.input_count = len(available)

    registry: dict[str, RegisteredSource] = {}
    for source_name in SOURCE_NAMES:
        entry = expected.get(source_name)
        if entry is None:
            context.quality.fatal(
                "bronze.object.missing",
                source_name=source_name,
                record_locator=f"manifest:source={source_name}",
                reason=f"The release manifest declares no object for {source_name}.",
            )
        found = available.get(source_name)
        if found is None:
            context.quality.fatal(
                "bronze.object.missing",
                source_name=source_name,
                record_locator=f"bronze/source={source_name}",
                reason=(
                    f"No Bronze object for {source_name}. "
                    "Run `make seed` to populate the object store."
                ),
            )
        registry[source_name] = _verify(context, source_name, found, entry)

    result.output_count = len(registry)
    result.issue_count = context.quality.issue_count
    result.finish()
    return registry, result


def _check_bundle_version(context: RunContext, manifest: dict) -> None:
    """Refuse to run against a release the checks were not written for."""
    version = manifest.get("bundle_version")
    if version != EXPECTED_BUNDLE_VERSION:
        context.quality.fatal(
            "bronze.object.missing",
            source_name="release",
            record_locator="metadata/bundle-manifest.json",
            observed_value=str(version),
            reason=(
                f"Release bundle_version is {version!r}; these checks target "
                f"version {EXPECTED_BUNDLE_VERSION}."
            ),
        )


def _verify(
    context: RunContext,
    source_name: str,
    found: BronzeObject,
    entry: dict,
) -> RegisteredSource:
    payload = context.lake.read_bronze(found.key)

    declared_bytes = int(entry["bytes"])
    if len(payload) != declared_bytes:
        context.quality.fatal(
            "bronze.object.size_mismatch",
            source_name=source_name,
            record_locator=found.key,
            observed_value=str(len(payload)),
            reason=(
                f"{found.object_name} is {len(payload)} bytes; the manifest "
                f"declares {declared_bytes}."
            ),
        )

    digest = hashlib.sha256(payload).hexdigest()
    declared_sha256 = str(entry["sha256"])
    if digest != declared_sha256:
        context.quality.fatal(
            "bronze.object.sha256_mismatch",
            source_name=source_name,
            record_locator=found.key,
            observed_value=digest,
            reason=(
                f"{found.object_name} hashes to {digest}; the manifest declares "
                f"{declared_sha256}."
            ),
        )

    members = _safe_members(context, source_name, found)
    context.bronze_hashes[found.key] = digest
    context.quality.count(
        f"bronze:{source_name}", read=1, accepted=1
    )
    return RegisteredSource(
        source_name=source_name,
        bronze_object=found.object_name,
        key=found.key,
        byte_size=len(payload),
        sha256=digest,
        members=members,
    )


def _safe_members(
    context: RunContext, source_name: str, found: BronzeObject
) -> tuple[str, ...]:
    """List archive members, refusing to read an archive with an unsafe path."""
    with context.lake.open_bronze_zip(found.key) as archive:
        if archive.testzip() is not None:
            context.quality.fatal(
                "bronze.archive.corrupt",
                source_name=source_name,
                record_locator=found.key,
                observed_value=archive.testzip(),
                reason=f"{found.object_name} failed its integrity test.",
            )
        names = tuple(sorted(archive.namelist()))

    for member in names:
        if is_unsafe_member(member):
            context.quality.fatal(
                "bronze.archive.unsafe_member_path",
                source_name=source_name,
                record_locator=f"{found.key}!{member}",
                observed_value=member,
                reason=(
                    f"Archive member {member!r} is absolute or escapes the "
                    "archive root, so the archive is not read."
                ),
            )
    return names
