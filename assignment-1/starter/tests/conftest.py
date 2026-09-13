"""Shared fixtures: a miniature Bronze release built in a temporary directory.

The fixtures mirror the real release's structure -- three archives under
``raw/source=<name>/`` plus a manifest -- at a size the tests can run in
milliseconds. Archive entries are written with a fixed timestamp so the archives
hash identically on every run, which is what lets the rerun test compare bytes.

No binary fixtures are committed; everything here is generated.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from quantum_lake_student.config import Settings

FIXED_TIMESTAMP = (2026, 1, 1, 0, 0, 0)

SYNDROME_README = """The file names: d-<surface_code_distance>_pfr-<physical_fault_rate>_nb-<number_of_samples>

The file format is a csv file with the following columns:
- label: binary label (0: no error, 1: error)
- syndromes: syndrome measurement sequence (tuples of the form (round, syndromes))
- quantity: number of samples for this label + syndrome sequence
"""

# 1,000 weighted observations, matching the nb-1K in the filename. The last two
# rows are the same syndrome under both labels, which must survive.
SYNDROME_CSV = """labels,syndromes,quantity
0,"((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))",900
1,"((1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))",60
0,"((0, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))",30
1,"((0, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))",10
"""

QASM_PARITY = """// Repetition code syndrome measurement
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
qreg a[2];
creg c[3];
creg syn[2];
gate syndrome d1,d2,d3,a1,a2
{
  cx d1,a1; cx d2,a1;
  cx d2,a2; cx d3,a2;
}
x q[0]; // error
barrier q;
syndrome q[0],q[1],q[2],a[0],a[1];
measure a -> syn;
if(syn==1) x q[0];
if(syn==2) x q[2];
if(syn==3) x q[1];
measure q -> c;
"""

QASM_NO_PARITY = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q -> c;
"""

# A small but internally consistent experiment: distance 3, 2 rounds, 3 shots.
GOOGLE_DIRECTORY = "surface_code_bX_d3_r2_center_3_5"
GOOGLE_SHOTS = 3
GOOGLE_ROUNDS = 2
GOOGLE_DATA_QUBITS = 9
GOOGLE_MEASURE_QUBITS = 8
GOOGLE_DETECTORS = GOOGLE_ROUNDS * GOOGLE_MEASURE_QUBITS          # 16 bits, 2 bytes
GOOGLE_MEASUREMENTS = GOOGLE_DETECTORS + GOOGLE_DATA_QUBITS       # 25 bits, 4 bytes
GOOGLE_SWEEP_BITS = GOOGLE_DATA_QUBITS                            # 9 bits, 2 bytes

GOOGLE_PROPERTIES = f"""type: surface_code_memory_experiment
basis: X
rounds: {GOOGLE_ROUNDS}
distance: 3
data_qubits: {GOOGLE_DATA_QUBITS}
measure_qubits: {GOOGLE_MEASURE_QUBITS}
shots: {GOOGLE_SHOTS}
center_data_qubit_row: 3
center_data_qubit_col: 5
circuit_measurements: {GOOGLE_MEASUREMENTS}
circuit_sweep_bits: {GOOGLE_SWEEP_BITS}
circuit_detectors: {GOOGLE_DETECTORS}
circuit_observables: 1
circuit_qubits: 17
"""

DECODER_FILES = (
    "obs_flips_predicted_by_belief_matching.01",
    "obs_flips_predicted_by_correlated_matching.01",
    "obs_flips_predicted_by_pymatching.01",
    "obs_flips_predicted_by_tensor_network_contraction.01",
)


def zip_bytes(members: dict[str, bytes]) -> bytes:
    """Build a ZIP whose bytes depend only on its contents."""
    sink = io.BytesIO()
    with zipfile.ZipFile(sink, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, date_time=FIXED_TIMESTAMP)
            info.external_attr = 0o644 << 16
            archive.writestr(info, members[name])
    return sink.getvalue()


def bits_to_b8(bit_rows: list[list[int]], bits_per_record: int) -> bytes:
    """Pack rows of bits little-endian within each byte, as Stim ``b8`` does."""
    record_bytes = (bits_per_record + 7) // 8
    out = bytearray()
    for row in bit_rows:
        buffer = bytearray(record_bytes)
        for index, bit in enumerate(row[:bits_per_record]):
            if bit:
                buffer[index // 8] |= 1 << (index % 8)
        out.extend(buffer)
    return bytes(out)


def zero_one_file(values: list[int]) -> bytes:
    return b"".join(b"%d\n" % value for value in values)


def google_members(
    *,
    detector_rows: list[list[int]] | None = None,
    actual: list[int] | None = None,
    omit: str | None = None,
    measurement_padding: bool = False,
) -> dict[str, bytes]:
    """Members of one synthetic Google experiment directory."""
    detector_rows = detector_rows or [
        [1] + [0] * (GOOGLE_DETECTORS - 1),
        [1, 1, 1] + [0] * (GOOGLE_DETECTORS - 3),
        [0] * GOOGLE_DETECTORS,
    ]
    actual = actual or [1, 0, 1]

    measurements = bytearray(
        bits_to_b8(
            [[1] + [0] * (GOOGLE_MEASUREMENTS - 1)] * GOOGLE_SHOTS,
            GOOGLE_MEASUREMENTS,
        )
    )
    if measurement_padding:
        # Set a bit above the declared measurement count in the last shot's
        # final byte. 25 bits occupy 4 bytes, so bits 25..31 are padding.
        record_bytes = (GOOGLE_MEASUREMENTS + 7) // 8
        measurements[GOOGLE_SHOTS * record_bytes - 1] |= 0b1000_0000

    members = {
        f"{GOOGLE_DIRECTORY}/properties.yml": GOOGLE_PROPERTIES.encode(),
        f"{GOOGLE_DIRECTORY}/measurements.b8": bytes(measurements),
        f"{GOOGLE_DIRECTORY}/detection_events.b8": bits_to_b8(
            detector_rows, GOOGLE_DETECTORS
        ),
        f"{GOOGLE_DIRECTORY}/sweep.b8": bits_to_b8(
            [[0] * GOOGLE_SWEEP_BITS] * GOOGLE_SHOTS, GOOGLE_SWEEP_BITS
        ),
        f"{GOOGLE_DIRECTORY}/obs_flips_actual.01": zero_one_file(actual),
    }
    for index, name in enumerate(DECODER_FILES):
        # Each decoder is wrong on a different shot, so predictions and mistakes
        # are distinguishable in tests.
        predictions = [
            value if position != index % GOOGLE_SHOTS else 1 - value
            for position, value in enumerate(actual)
        ]
        members[f"{GOOGLE_DIRECTORY}/{name}"] = zero_one_file(predictions)

    if omit is not None:
        members.pop(f"{GOOGLE_DIRECTORY}/{omit}", None)
    return members


@dataclass
class Release:
    """A miniature Bronze release on disk, plus its settings."""

    root: Path
    settings: Settings
    objects: dict[str, bytes]

    def sha256(self, source_name: str) -> str:
        return hashlib.sha256(self.objects[source_name]).hexdigest()


def write_release(root: Path, **google_kwargs) -> Release:
    """Materialise a miniature release under ``root`` and return its settings."""
    archives = {
        "qec_syndromes": zip_bytes(
            {
                "README.txt": SYNDROME_README.encode(),
                "d-3_pfr-0.005000_nb-1K.csv": SYNDROME_CSV.encode(),
            }
        ),
        "qasmbench": zip_bytes(
            {
                "small/qec_sm_n5/qec_sm_n5.qasm": QASM_PARITY.encode(),
                "small/plain_n2/plain_n2.qasm": QASM_NO_PARITY.encode(),
            }
        ),
        "google_qec": zip_bytes(google_members(**google_kwargs)),
    }
    filenames = {
        "qec_syndromes": "syndromes_dataset.zip",
        "qasmbench": "qasmbench-qec.zip",
        "google_qec": "google-surface-code-curated.zip",
    }

    objects = []
    for source_name, payload in archives.items():
        relative = f"raw/source={source_name}/{filenames[source_name]}"
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        objects.append(
            {
                "bytes": len(payload),
                "mandatory": True,
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "source": source_name,
                "source_member": None,
                "teaching_role": "test fixture",
            }
        )

    manifest = root / "metadata" / "bundle-manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "bundle_version": 3,
                "objects": objects,
                "release_date": "2026-01-01",
                "release_name": "quantum-data-core-test",
            },
            indent=2,
        )
    )

    settings = Settings(
        lake_backend="local",
        local_lake_root=root,
        s3_endpoint="http://minio:9000",
        s3_access_key="test",
        s3_secret_key="test",
        s3_bucket="quantum-lake",
        postgres_host="postgres",
        postgres_port=5432,
        postgres_db="quantum_lake",
        postgres_user="quantum",
        postgres_password="test",
    )
    return Release(root=root, settings=settings, objects=archives)


@pytest.fixture
def release(tmp_path: Path) -> Release:
    return write_release(tmp_path / "bronze")
