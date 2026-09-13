"""Parse the QASMBench QEC circuits into circuits, checks, and corrections.

This is a reader, not a simulator. It answers structural questions only: which
registers exist, which operations actually execute, which qubit collects a parity
into which syndrome bit, and which recovery operations a measured syndrome
controls.

Three distinctions matter and are all graded:

* a ``gate`` body is a *definition*. Its operations do not execute until the gate
  is called, so counting ``cx`` lines with a grep overcounts.
* ``measure a -> syn`` is one statement but several executed measurements, one
  per register bit.
* ``q[0]`` and ``a[0]`` are different qubits. Every qubit is identified by
  ``(register, index)``, never by a bare integer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Literal

import pyarrow as pa

from .. import ids
from ..quality import QualityLog
from ..runcontext import RunContext
from ..schemas import (
    QASM_CIRCUIT,
    QASM_CONDITIONAL_CORRECTION,
    QASM_STABILIZER_CHECK,
)

SOURCE_NAME = "qasmbench"

# Operations that carry no quantum action for counting purposes.
NON_OPERATIONS = frozenset({"barrier", "reset", "id", "delay"})

_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT = re.compile(r"//[^\n]*")
_GATE_DEF = re.compile(
    r"\bgate\s+(?P<name>\w+)\s*(?:\((?P<params>[^)]*)\))?\s*"
    r"(?P<qubits>[\w\s,]*?)\s*\{(?P<body>[^}]*)\}",
    re.DOTALL,
)
_OPERAND = re.compile(r"(?P<name>\w+)\s*(?:\[\s*(?P<index>\d+)\s*\])?")
_REGISTER = re.compile(r"^(?P<kind>qreg|creg)\s+(?P<name>\w+)\s*\[\s*(?P<size>\d+)\s*\]$")
_MEASURE = re.compile(r"^measure\s+(?P<source>[\w\[\]\s]+?)\s*->\s*(?P<target>[\w\[\]\s]+)$")
_CONDITION = re.compile(r"^if\s*\(\s*(?P<register>\w+)\s*==\s*(?P<value>\d+)\s*\)\s*(?P<inner>.+)$")
_VERSION = re.compile(r"^OPENQASM\s+(?P<version>[\d.]+)$")
_CALL = re.compile(r"^(?P<name>\w+)\s*(?:\((?P<params>[^)]*)\))?\s*(?P<operands>.*)$")


@dataclass(frozen=True)
class Register:
    kind: Literal["qreg", "creg"]
    name: str
    size: int


@dataclass(frozen=True)
class Operand:
    """A register reference, either one bit or the whole register."""

    register: str
    index: int | None

    def resolve(self, size: int) -> tuple[str, ...]:
        if self.index is not None:
            return (f"{self.register}[{self.index}]",)
        return tuple(f"{self.register}[{position}]" for position in range(size))


@dataclass(frozen=True)
class GateDefinition:
    name: str
    qubits: tuple[str, ...]
    body: tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True)
class ExecutedOperation:
    gate: str
    qubits: tuple[str, ...]
    statement_index: int

    @property
    def is_two_qubit(self) -> bool:
        return len(self.qubits) == 2


@dataclass(frozen=True)
class ExecutedMeasurement:
    qubit: str
    clbit: str
    statement_index: int


@dataclass(frozen=True)
class Correction:
    condition_register: str
    condition_value: int
    gate: str
    target_qubit: str
    statement_index: int


@dataclass(frozen=True)
class StabilizerCheck:
    ancilla_qubit: str
    data_qubits: tuple[str, ...]
    syndrome_bit: str


@dataclass
class Program:
    version: str | None = None
    registers: dict[str, Register] = field(default_factory=dict)
    gates: dict[str, GateDefinition] = field(default_factory=dict)
    operations: list[ExecutedOperation] = field(default_factory=list)
    measurements: list[ExecutedMeasurement] = field(default_factory=list)
    corrections: list[Correction] = field(default_factory=list)
    unresolved_gates: list[str] = field(default_factory=list)

    @property
    def qubit_count(self) -> int:
        return sum(
            register.size
            for register in self.registers.values()
            if register.kind == "qreg"
        )

    def register_declarations(self) -> str:
        """A stable, repeatable serialisation of the declared registers."""
        entries = [
            {"kind": register.kind, "name": register.name, "size": register.size}
            for register in sorted(
                self.registers.values(), key=lambda item: (item.kind, item.name)
            )
        ]
        return json.dumps(entries, sort_keys=True, separators=(",", ":"))


def strip_comments(text: str) -> str:
    return _LINE_COMMENT.sub("", _BLOCK_COMMENT.sub("", text))


def parse_program(text: str) -> Program:
    """Parse one OpenQASM 2.0 member into declarations and executed operations."""
    program = Program()
    body = strip_comments(text)

    # Gate definitions are lifted out first: their bodies contain semicolons and
    # would otherwise be mistaken for top-level statements.
    for matched in _GATE_DEF.finditer(body):
        definition = _parse_gate_definition(matched)
        program.gates[definition.name] = definition
    body = _GATE_DEF.sub("", body)

    for statement_index, raw in enumerate(body.split(";")):
        statement = " ".join(raw.split())
        if not statement:
            continue
        _consume_statement(program, statement, statement_index)
    return program


def _parse_gate_definition(matched: re.Match[str]) -> GateDefinition:
    qubits = tuple(
        part.strip() for part in matched["qubits"].split(",") if part.strip()
    )
    body: list[tuple[str, tuple[str, ...]]] = []
    for raw in matched["body"].split(";"):
        statement = " ".join(raw.split())
        if not statement:
            continue
        call = _CALL.match(statement)
        if not call:
            continue
        operands = tuple(
            part.strip() for part in (call["operands"] or "").split(",") if part.strip()
        )
        body.append((call["name"], operands))
    return GateDefinition(matched["name"], qubits, tuple(body))


def _consume_statement(program: Program, statement: str, statement_index: int) -> None:
    version = _VERSION.match(statement)
    if version:
        program.version = version["version"]
        return
    if statement.startswith("include"):
        return

    register = _REGISTER.match(statement)
    if register:
        program.registers[register["name"]] = Register(
            kind=register["kind"],  # type: ignore[arg-type]
            name=register["name"],
            size=int(register["size"]),
        )
        return

    condition = _CONDITION.match(statement)
    if condition:
        _consume_conditional(program, condition, statement_index)
        return

    measure = _MEASURE.match(statement)
    if measure:
        _consume_measure(program, measure, statement_index)
        return

    _consume_operation(program, statement, statement_index)


def _consume_conditional(
    program: Program, condition: re.Match[str], statement_index: int
) -> None:
    """Record a syndrome-controlled recovery operation.

    A conditional operation does not execute unconditionally, so it is not added
    to the executed-operation counts; it is fully represented as a correction.
    """
    inner = condition["inner"].strip()
    call = _CALL.match(inner)
    if not call:
        return
    targets = _resolve_operands(program, call["operands"])
    for target in targets:
        program.corrections.append(
            Correction(
                condition_register=condition["register"],
                condition_value=int(condition["value"]),
                gate=call["name"],
                target_qubit=target,
                statement_index=statement_index,
            )
        )


def _consume_measure(
    program: Program, measure: re.Match[str], statement_index: int
) -> None:
    """Expand ``measure a -> syn`` into one executed measurement per bit."""
    sources = _resolve_operands(program, measure["source"])
    targets = _resolve_operands(program, measure["target"])
    if len(sources) != len(targets):
        raise ValueError(
            f"measure operands have different widths: "
            f"{len(sources)} qubits into {len(targets)} classical bits"
        )
    for qubit, clbit in zip(sources, targets, strict=True):
        program.measurements.append(
            ExecutedMeasurement(qubit, clbit, statement_index)
        )


def _consume_operation(
    program: Program, statement: str, statement_index: int
) -> None:
    call = _CALL.match(statement)
    if not call:
        return
    name = call["name"]
    if name in NON_OPERATIONS:
        return
    operands = _resolve_operand_groups(program, call["operands"])
    if not operands:
        return
    if name in program.gates:
        _expand_gate_call(program, name, operands, statement_index, depth=0)
        return
    for qubits in _broadcast(operands):
        program.operations.append(ExecutedOperation(name, qubits, statement_index))


def _expand_gate_call(
    program: Program,
    name: str,
    operands: list[tuple[str, ...]],
    statement_index: int,
    *,
    depth: int,
) -> None:
    """Substitute a custom gate's arguments into its body and count the result."""
    if depth > 16:
        program.unresolved_gates.append(name)
        return
    definition = program.gates[name]
    for qubits in _broadcast(operands):
        binding = dict(zip(definition.qubits, qubits, strict=False))
        for inner_name, inner_operands in definition.body:
            if inner_name in NON_OPERATIONS:
                continue
            resolved = tuple(
                binding.get(operand, operand) for operand in inner_operands
            )
            if inner_name in program.gates:
                _expand_gate_call(
                    program,
                    inner_name,
                    [(operand,) for operand in resolved],
                    statement_index,
                    depth=depth + 1,
                )
                continue
            program.operations.append(
                ExecutedOperation(inner_name, resolved, statement_index)
            )


def _resolve_operand_groups(program: Program, text: str | None) -> list[tuple[str, ...]]:
    groups: list[tuple[str, ...]] = []
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        matched = _OPERAND.fullmatch(part)
        if not matched:
            continue
        operand = Operand(
            matched["name"],
            int(matched["index"]) if matched["index"] is not None else None,
        )
        register = program.registers.get(operand.register)
        groups.append(operand.resolve(register.size if register else 1))
    return groups


def _resolve_operands(program: Program, text: str | None) -> tuple[str, ...]:
    groups = _resolve_operand_groups(program, text)
    return tuple(name for group in groups for name in group)


def _broadcast(groups: list[tuple[str, ...]]) -> list[tuple[str, ...]]:
    """Apply OpenQASM register broadcasting across operand groups."""
    width = max((len(group) for group in groups), default=0)
    if width == 0:
        return []
    for group in groups:
        if len(group) not in (1, width):
            raise ValueError("operand register sizes do not broadcast")
    return [
        tuple(group[position] if len(group) == width else group[0] for group in groups)
        for position in range(width)
    ]


def find_stabilizer_checks(program: Program) -> tuple[StabilizerCheck, ...]:
    """Identify parity checks that are *explicit* in the circuit.

    A qubit is treated as a parity-check ancilla only when all four conditions
    hold, which is the strict reading of "where explicit":

    1. it receives no single-qubit operation anywhere in the circuit;
    2. it is never the control of a two-qubit operation;
    3. it is the target of at least two two-qubit operations, with distinct
       controls; and
    4. it is measured exactly once, unconditionally, into one classical bit.

    Condition 1 is what makes the parity accumulation trustworthy. Circuits that
    interleave single-qubit rotations on the candidate qubit yield no checks
    rather than an invented one.
    """
    single_qubit: set[str] = set()
    controls: set[str] = set()
    accumulated: dict[str, list[str]] = {}

    for operation in program.operations:
        if len(operation.qubits) == 1:
            single_qubit.add(operation.qubits[0])
        elif operation.is_two_qubit:
            control, target = operation.qubits
            controls.add(control)
            accumulated.setdefault(target, []).append(control)
        else:
            single_qubit.update(operation.qubits)

    measured: dict[str, list[str]] = {}
    for measurement in program.measurements:
        measured.setdefault(measurement.qubit, []).append(measurement.clbit)

    checks: list[StabilizerCheck] = []
    for ancilla, contributors in accumulated.items():
        if ancilla in single_qubit or ancilla in controls:
            continue
        clbits = measured.get(ancilla, [])
        if len(clbits) != 1:
            continue
        distinct: list[str] = []
        for control in contributors:
            if control not in distinct:
                distinct.append(control)
        if len(distinct) < 2:
            continue
        checks.append(StabilizerCheck(ancilla, tuple(distinct), clbits[0]))

    return tuple(sorted(checks, key=lambda check: check.ancilla_qubit))


def build(
    context: RunContext, key: str, bronze_object: str, sha256: str
) -> dict[str, int]:
    """Parse every QASM member and stage the three QASMBench tables."""
    circuit_rows: list[dict[str, object]] = []
    check_rows: list[dict[str, object]] = []
    correction_rows: list[dict[str, object]] = []
    rejected = 0

    with context.lake.open_bronze_zip(key) as archive:
        members = sorted(
            name
            for name in archive.namelist()
            if name.lower().endswith(".qasm") and not name.endswith("/")
        )
        for member in members:
            payload = archive.read(member)
            outcome = _build_member(
                context,
                member=member,
                payload=payload,
                bronze_object=bronze_object,
                sha256=sha256,
            )
            if outcome is None:
                rejected += 1
                continue
            circuit, checks, corrections = outcome
            circuit_rows.append(circuit)
            check_rows.extend(checks)
            correction_rows.extend(corrections)

    context.quality.count(
        QASM_CIRCUIT.name,
        read=len(members),
        accepted=len(circuit_rows),
        rejected=rejected,
    )
    context.quality.count(
        QASM_STABILIZER_CHECK.name,
        read=len(check_rows),
        accepted=len(check_rows),
    )
    context.quality.count(
        QASM_CONDITIONAL_CORRECTION.name,
        read=len(correction_rows),
        accepted=len(correction_rows),
    )

    counts: dict[str, int] = {}
    for spec, rows in (
        (QASM_CIRCUIT, circuit_rows),
        (QASM_STABILIZER_CHECK, check_rows),
        (QASM_CONDITIONAL_CORRECTION, correction_rows),
    ):
        table = pa.table(
            {name: [row[name] for row in rows] for name in spec.column_names},
            schema=spec.schema,
        )
        counts[spec.name] = context.lake.stage_table(spec, table).row_count
    return counts


def _build_member(
    context: RunContext,
    *,
    member: str,
    payload: bytes,
    bronze_object: str,
    sha256: str,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]] | None:
    quality: QualityLog = context.quality
    benchmark_name, variant = _identify(member)

    try:
        program = parse_program(payload.decode("utf-8", errors="replace"))
    except ValueError as error:
        quality.record(
            "qasmbench.qasm.register_declaration_unparsable",
            source_name=SOURCE_NAME,
            record_locator=member,
            observed_value=str(error),
            reason=f"{member} could not be parsed: {error}",
        )
        return None

    circuit = ids.circuit_id(benchmark_name, variant)
    record_id = ids.circuit_record_id(member, sha256)

    if program.unresolved_gates:
        quality.record(
            "qasmbench.qasm.gate_call_unresolved",
            source_name=SOURCE_NAME,
            record_locator=member,
            observed_value=",".join(sorted(set(program.unresolved_gates))),
            reason="A gate call could not be expanded to executed operations.",
            source_record_id=record_id,
        )

    checks = find_stabilizer_checks(program)
    if not checks:
        quality.record(
            "qasmbench.qasm.no_explicit_parity_check",
            source_name=SOURCE_NAME,
            record_locator=member,
            observed_value="0",
            reason=(
                f"{member} contains no qubit that accumulates a parity through "
                "two-qubit operations only and is then measured, so no "
                "stabilizer check is recorded rather than inventing one."
            ),
            source_record_id=record_id,
        )
    if not program.corrections:
        quality.record(
            "qasmbench.qasm.no_conditional_correction",
            source_name=SOURCE_NAME,
            record_locator=member,
            observed_value="0",
            reason=f"{member} contains no syndrome-controlled recovery operation.",
            source_record_id=record_id,
        )

    context.trace.add(
        source_record_id=record_id,
        source_name=SOURCE_NAME,
        bronze_object=bronze_object,
        archive_member=member,
        record_locator=ids.member_locator(member),
        input_sha256=sha256,
    )

    circuit_row = {
        "source_record_id": record_id,
        "circuit_id": circuit,
        "benchmark_name": benchmark_name,
        "variant": variant,
        "register_declarations": program.register_declarations(),
        "qubit_count": program.qubit_count,
        "measurement_count": len(program.measurements),
        "two_qubit_gate_count": sum(
            1 for operation in program.operations if operation.is_two_qubit
        ),
    }

    check_rows: list[dict[str, object]] = []
    for check in checks:
        identifier = ids.check_id(circuit, check.ancilla_qubit)
        check_record_id = ids.stabilizer_check_record_id(
            circuit, check.ancilla_qubit, check.data_qubits, check.syndrome_bit
        )
        check_rows.append(
            {
                "source_record_id": check_record_id,
                "circuit_id": circuit,
                "check_id": identifier,
                "ancilla_qubit": check.ancilla_qubit,
                "data_qubits": list(check.data_qubits),
                "syndrome_bit": check.syndrome_bit,
            }
        )
        context.trace.add(
            source_record_id=check_record_id,
            source_name=SOURCE_NAME,
            bronze_object=bronze_object,
            archive_member=member,
            record_locator=ids.member_locator(member),
            input_sha256=sha256,
        )

    correction_rows: list[dict[str, object]] = []
    for ordinal, correction in enumerate(program.corrections):
        correction_record_id = ids.conditional_correction_record_id(
            circuit,
            ordinal,
            correction.condition_register,
            correction.condition_value,
            correction.gate,
            correction.target_qubit,
        )
        correction_rows.append(
            {
                "source_record_id": correction_record_id,
                "circuit_id": circuit,
                "condition_register": correction.condition_register,
                "condition_value": correction.condition_value,
                "gate": correction.gate,
                "target_qubit": correction.target_qubit,
            }
        )
        context.trace.add(
            source_record_id=correction_record_id,
            source_name=SOURCE_NAME,
            bronze_object=bronze_object,
            archive_member=member,
            record_locator=ids.statement_locator(correction.statement_index),
            input_sha256=sha256,
        )

    return circuit_row, check_rows, correction_rows


def _identify(member: str) -> tuple[str, str]:
    """Split a member path into its benchmark name and its variant."""
    stem = member.rsplit("/", 1)[-1].removesuffix(".qasm")
    if stem.endswith("_transpiled"):
        return stem.removesuffix("_transpiled"), "transpiled"
    return stem, "source"
