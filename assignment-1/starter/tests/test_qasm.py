"""OpenQASM parsing: definitions vs executions, register expansion, parity checks."""

from __future__ import annotations

import json

from conftest import QASM_NO_PARITY, QASM_PARITY

from quantum_lake_student.sources import qasm

# The transpiled form of the same circuit: the custom gate is inlined and the
# register measurements are split into one statement per bit.
QASM_PARITY_TRANSPILED = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
qreg a[2];
creg c[3];
creg syn[2];
x q[0];
barrier q[0],q[1],q[2];
cx q[0],a[0];
cx q[1],a[0];
cx q[1],a[1];
cx q[2],a[1];
measure a[0] -> syn[0];
measure a[1] -> syn[1];
if(syn==1) x q[0];
measure q[0] -> c[0];
if(syn==2) x q[2];
if(syn==3) x q[1];
measure q[1] -> c[1];
measure q[2] -> c[2];
"""

DEFINITION_ONLY = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
qreg a[2];
gate syndrome d1,d2,d3,a1,a2
{
  cx d1,a1; cx d2,a1;
  cx d2,a2; cx d3,a2;
}
"""


def test_registers_are_read_with_their_sizes():
    program = qasm.parse_program(QASM_PARITY)
    assert {name: register.size for name, register in program.registers.items()} == {
        "q": 3,
        "a": 2,
        "c": 3,
        "syn": 2,
    }
    assert program.qubit_count == 5  # qreg only, matching the n5 in the name


def test_register_declarations_serialise_deterministically():
    first = qasm.parse_program(QASM_PARITY).register_declarations()
    second = qasm.parse_program(QASM_PARITY).register_declarations()
    assert first == second
    assert json.loads(first) == [
        {"kind": "creg", "name": "c", "size": 3},
        {"kind": "creg", "name": "syn", "size": 2},
        {"kind": "qreg", "name": "a", "size": 2},
        {"kind": "qreg", "name": "q", "size": 3},
    ]


def test_a_gate_body_is_a_definition_not_an_execution():
    program = qasm.parse_program(DEFINITION_ONLY)
    assert "syndrome" in program.gates
    # The four cx in the body must not count until the gate is called.
    assert program.operations == []


def test_register_measurement_expands_to_one_measurement_per_bit():
    program = qasm.parse_program(QASM_PARITY)
    # measure a -> syn is 2 measurements, measure q -> c is 3, so 5 in total
    # from only two statements.
    assert len(program.measurements) == 5
    pairs = {(m.qubit, m.clbit) for m in program.measurements}
    assert ("a[0]", "syn[0]") in pairs
    assert ("a[1]", "syn[1]") in pairs
    assert ("q[2]", "c[2]") in pairs


def test_custom_gate_call_is_expanded_into_two_qubit_operations():
    program = qasm.parse_program(QASM_PARITY)
    two_qubit = [op for op in program.operations if op.is_two_qubit]
    assert len(two_qubit) == 4
    assert {op.qubits for op in two_qubit} == {
        ("q[0]", "a[0]"),
        ("q[1]", "a[0]"),
        ("q[1]", "a[1]"),
        ("q[2]", "a[1]"),
    }


def test_source_and_transpiled_variants_agree():
    """The transpiled file is the same circuit written out flat."""
    source = qasm.parse_program(QASM_PARITY)
    transpiled = qasm.parse_program(QASM_PARITY_TRANSPILED)

    assert source.qubit_count == transpiled.qubit_count
    assert len(source.measurements) == len(transpiled.measurements)
    assert sum(1 for op in source.operations if op.is_two_qubit) == sum(
        1 for op in transpiled.operations if op.is_two_qubit
    )
    assert qasm.find_stabilizer_checks(source) == qasm.find_stabilizer_checks(
        transpiled
    )
    assert len(source.corrections) == len(transpiled.corrections)


def test_parity_checks_are_identified_with_their_data_qubits():
    checks = qasm.find_stabilizer_checks(qasm.parse_program(QASM_PARITY))
    assert [
        (check.ancilla_qubit, check.data_qubits, check.syndrome_bit)
        for check in checks
    ] == [
        ("a[0]", ("q[0]", "q[1]"), "syn[0]"),
        ("a[1]", ("q[1]", "q[2]"), "syn[1]"),
    ]


def test_register_local_indexes_are_distinct_qubits():
    checks = qasm.find_stabilizer_checks(qasm.parse_program(QASM_PARITY))
    ancillas = {check.ancilla_qubit for check in checks}
    # a[0] is a check qubit; q[0] is a data qubit. A bare 0 would conflate them.
    assert "a[0]" in ancillas
    assert "q[0]" not in ancillas


def test_no_check_is_invented_where_none_is_explicit():
    # One cx onto q[1] and a single-qubit h: not a parity accumulation.
    assert qasm.find_stabilizer_checks(qasm.parse_program(QASM_NO_PARITY)) == ()


def test_single_qubit_gate_on_the_candidate_disqualifies_it():
    interleaved = """OPENQASM 2.0;
qreg q[3];
creg c[1];
cx q[0],q[2];
h q[2];
cx q[1],q[2];
measure q[2] -> c[0];
"""
    assert qasm.find_stabilizer_checks(qasm.parse_program(interleaved)) == ()


def test_conditional_corrections_are_read_with_register_and_value():
    program = qasm.parse_program(QASM_PARITY)
    assert [
        (c.condition_register, c.condition_value, c.gate, c.target_qubit)
        for c in program.corrections
    ] == [
        ("syn", 1, "x", "q[0]"),
        ("syn", 2, "x", "q[2]"),
        ("syn", 3, "x", "q[1]"),
    ]


def test_conditional_operations_are_not_counted_as_executed():
    program = qasm.parse_program(QASM_PARITY)
    # The three conditional x gates live in corrections, not in operations.
    single = [op for op in program.operations if len(op.qubits) == 1]
    assert [op.gate for op in single] == ["x"]  # only the unconditional error


def test_stray_whitespace_before_the_semicolon_is_tolerated():
    program = qasm.parse_program(
        "OPENQASM 2.0;\nqreg q[2];\ncx q[0],q[1] ;\n"
    )
    assert len([op for op in program.operations if op.is_two_qubit]) == 1


def test_comments_are_ignored():
    program = qasm.parse_program(
        "OPENQASM 2.0;\nqreg q[2];\n// cx q[0],q[1];\n/* cx q[0],q[1]; */\nx q[0];\n"
    )
    assert [op.gate for op in program.operations] == ["x"]


def test_barrier_and_reset_are_not_operations():
    program = qasm.parse_program(
        "OPENQASM 2.0;\nqreg q[2];\nbarrier q;\nreset q[0];\nid q[1];\n"
    )
    assert program.operations == []
