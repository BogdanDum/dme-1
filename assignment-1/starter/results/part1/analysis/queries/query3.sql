SELECT
    c.benchmark_name,
    c.variant,
    r.condition_register,
    r.condition_value,
    r.gate AS correction_gate,
    r.target_qubit AS correction_target,
    s.check_id,
    s.data_qubits,
    s.ancilla_qubit,
    s.syndrome_bit
FROM gold.circuit AS c
         JOIN gold.stabilizer_check AS s USING (circuit_id)
         JOIN gold.conditional_correction AS r
              ON r.circuit_id = c.circuit_id
                  AND r.condition_register = split_part(s.syndrome_bit, '[', 1)
WHERE c.benchmark_name = 'qec_sm_n5'
  AND c.variant = 'source'
ORDER BY r.condition_value, r.source_record_id, s.check_id;