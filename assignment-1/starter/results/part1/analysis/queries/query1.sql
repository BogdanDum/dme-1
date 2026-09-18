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