## Query Notes

#### Analysis 1

>How do weighted syndrome frequency and logical-error labels change with physical fault rate?

We first count the errors as the sum of sample weights per each syndrome and physical fault rate:

```sql
WITH pattern_counts AS (  
    SELECT  
        e.physical_fault_rate,  
        o.syndrome_bits,  
        SUM(o.sample_weight) AS weighted_count,  
        SUM(CASE WHEN o.logical_error_label  
                     THEN o.sample_weight ELSE 0 END) AS error_count  
    FROM gold.syndrome_observation AS o  
             JOIN gold.syndrome_experiment AS e USING (experiment_id)  
    GROUP BY e.physical_fault_rate, o.syndrome_bits  
)
```

Then we divide the counts by the total weighted count for each physical fault rate:

```SQL
SELECT  
    physical_fault_rate,  
    syndrome_bits,  
    weighted_count,  
    weighted_count / SUM(weighted_count) OVER (  
        PARTITION BY physical_fault_rate  
        ) AS syndrome_frequency,  
    error_count / weighted_count AS logical_error_rate_within_pattern,  
    SUM(error_count) OVER (  
        PARTITION BY physical_fault_rate  
        ) / SUM(weighted_count) OVER (  
        PARTITION BY physical_fault_rate  
        ) AS logical_error_rate_at_fault_rate  
FROM pattern_counts  
ORDER BY physical_fault_rate, syndrome_frequency DESC, syndrome_bits;
```

#### Analysis 2

>How do the supplied decoder logical-error rates compare by code distance and distance-three processor location?

We count the errors as the occurrences where predicted observable flip does not match the actual observable flip and average them as logical error rate. Then we see how they relate to the distance.

```SQL
SELECT
    e.distance,
    e.basis,
    e.rounds,
    p.decoder_name,
    COUNT(*) AS evaluated_shots,
    COUNT(*) FILTER (
        WHERE p.predicted_observable_flip <> s.actual_observable_flip
        ) AS logical_errors,
    AVG((p.predicted_observable_flip <> s.actual_observable_flip)::int)
             AS logical_error_rate
FROM gold.decoder_prediction AS p
         JOIN gold.shot AS s USING (source_record_id_shot)
         JOIN gold.experiment AS e USING (experiment_id)
GROUP BY e.distance, e.basis, e.rounds, p.decoder_name
ORDER BY e.distance, e.basis, e.rounds, logical_error_rate;
```

This query extracts the same data, but compares the logical error rate for different locations at distance 3.

```SQL
SELECT
    e.center_row,
    e.center_col,
    e.basis,
    e.rounds,
    p.decoder_name,
    COUNT(*) AS evaluated_shots,
    COUNT(*) FILTER (
        WHERE p.predicted_observable_flip <> s.actual_observable_flip
        ) AS logical_errors,
    AVG((p.predicted_observable_flip <> s.actual_observable_flip)::int)
             AS logical_error_rate
FROM gold.decoder_prediction AS p
         JOIN gold.shot AS s USING (source_record_id_shot)
         JOIN gold.experiment AS e USING (experiment_id)
WHERE e.distance = 3
GROUP BY
    e.center_row, e.center_col, e.basis, e.rounds, p.decoder_name
ORDER BY
    e.center_row, e.center_col, e.basis, e.rounds, logical_error_rate;
```

#### Analysis 3

>How does the repetition-code circuit map data qubits to parity-check ancillas, syndrome bits, and conditional corrections?

As the [quantum data primer](../../../../../assignment/quantum-data-primer.md) suggests: "In the supplied repetition-code circuit, one ancilla checks data qubits 0 and 1, while another checks data qubits 1 and 2.", the repetition-code circuit is the one where data_qubits is `{q[0],q[1]}` and `{q[1],q[2]}`, which is the circuit with benchmark_name is `qec_sm_n5` and variant is `source`. These conditions are considered in the `WHERE` statement.

```SQL
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
```