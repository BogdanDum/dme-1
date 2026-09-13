"""Every identifier the pipeline mints, in one place.

Two kinds of identifier are used deliberately:

* Business ids (``experiment_id``, ``circuit_id``, ``check_id``) are readable
  composites, so a human can recognise a record in Adminer or a Parquet dump.
* ``source_record_id`` is an opaque SHA-256 over the fields that define the
  record, built with the supplied :func:`models.stable_record_hash`.

Both kinds are derived only from record content and from repeatable positions in
the source. Nothing here reads the clock, a run id, or a row's arrival order, so
a second run over unchanged input mints exactly the same identifiers.
"""

from __future__ import annotations

from .models import stable_record_hash

SYNDROMES = "qec_syndromes"
GOOGLE = "google_qec"
QASMBENCH = "qasmbench"


def syndrome_experiment_id(distance: int, fault_rate_text: str) -> str:
    """Identify one simulated fault-rate experiment.

    ``fault_rate_text`` keeps the zero-padded form found in the filename
    (``0.000010``). Reformatting it as a float first would make ``0.00001``,
    ``1e-05`` and ``0.000010`` fail to group with each other.
    """
    return f"{SYNDROMES}:d{distance}:pfr={fault_rate_text}"


def syndrome_record_id(
    experiment_id: str, syndrome_bits: bytes, logical_error_label: bool
) -> str:
    """Identify one aggregate syndrome row.

    ``(labels, syndromes)`` is unique within every supplied CSV, so the content
    alone identifies the row and the CSV line number is not needed. That keeps
    the id independent of row order, as the tracing rule requires.
    """
    return stable_record_hash(
        {
            "source": SYNDROMES,
            "experiment_id": experiment_id,
            "syndrome_bits": syndrome_bits.hex(),
            "logical_error_label": logical_error_label,
        }
    )


def google_experiment_record_id(experiment_id: str) -> str:
    """Identify one hardware experiment directory."""
    return stable_record_hash({"source": GOOGLE, "experiment_id": experiment_id})


def google_shot_record_id(experiment_id: str, shot_index: int) -> str:
    """Identify one aligned hardware shot.

    A shot is assembled from several Bronze members, so this one id maps to
    several ``source_trace`` rows.
    """
    return stable_record_hash(
        {
            "source": GOOGLE,
            "experiment_id": experiment_id,
            "shot_index": shot_index,
        }
    )


def circuit_id(benchmark_name: str, variant: str) -> str:
    """Identify one circuit variant, e.g. ``qasmbench:qec_sm_n5:transpiled``."""
    return f"{QASMBENCH}:{benchmark_name}:{variant}"


def circuit_record_id(archive_member: str, member_sha256: str) -> str:
    """Identify the QASM member a circuit row was parsed from."""
    return stable_record_hash(
        {
            "source": QASMBENCH,
            "archive_member": archive_member,
            "member_sha256": member_sha256,
        }
    )


def check_id(circuit: str, ancilla_qubit: str) -> str:
    """Identify one parity check within a circuit by its ancilla qubit."""
    return f"{circuit}#check={ancilla_qubit}"


def stabilizer_check_record_id(
    circuit: str, ancilla_qubit: str, data_qubits: tuple[str, ...], syndrome_bit: str
) -> str:
    """Identify the statements that define one parity check."""
    return stable_record_hash(
        {
            "source": QASMBENCH,
            "circuit_id": circuit,
            "ancilla_qubit": ancilla_qubit,
            "data_qubits": list(data_qubits),
            "syndrome_bit": syndrome_bit,
        }
    )


def conditional_correction_record_id(
    circuit: str,
    ordinal: int,
    condition_register: str,
    condition_value: int,
    gate: str,
    target_qubit: str,
) -> str:
    """Identify one syndrome-controlled correction statement.

    ``ordinal`` is the statement's position among the corrections of this
    circuit. It is included because a circuit may legitimately repeat the same
    ``(register, value, gate, target)`` combination, which would otherwise
    collapse two distinct statements onto one id.
    """
    return stable_record_hash(
        {
            "source": QASMBENCH,
            "circuit_id": circuit,
            "ordinal": ordinal,
            "condition_register": condition_register,
            "condition_value": condition_value,
            "gate": gate,
            "target_qubit": target_qubit,
        }
    )


def issue_id(
    rule_id: str,
    source_name: str,
    record_locator: str,
    observed_value: str | None,
) -> str:
    """Identify one issue independently of the run that found it.

    ``run_id`` is deliberately excluded so that two runs over unchanged input
    produce the same set of issue ids.
    """
    return stable_record_hash(
        {
            "rule_id": rule_id,
            "source_name": source_name,
            "record_locator": record_locator,
            "observed_value": observed_value,
        }
    )


def csv_row_locator(row_number: int) -> str:
    """Repeatable position of a CSV data row, 1-based and excluding the header."""
    return f"row={row_number}"


def shot_locator(shot_index: int) -> str:
    """Repeatable position of a shot inside its companion files."""
    return f"shot={shot_index}"


def statement_locator(statement_index: int) -> str:
    """Repeatable position of a statement inside a QASM member."""
    return f"stmt={statement_index}"


def member_locator(archive_member: str) -> str:
    """Repeatable position when the whole archive member is the record."""
    return f"member={archive_member}"
