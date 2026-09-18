SELECT
    e.center_row,
    e.center_col,
    e.basis,
    e.rounds,
    p.decoder_name,
    COUNT(*) AS evaluated_shots,
    COUNT(*) FILTER (WHERE NOT p.decoder_correctness) AS logical_errors,
    AVG((NOT p.decoder_correctness)::int) AS logical_error_rate
FROM gold.decoder_correctness AS p
         JOIN gold.experiment AS e USING (experiment_id)
WHERE e.distance = 3
GROUP BY
    e.center_row, e.center_col, e.basis, e.rounds, p.decoder_name
ORDER BY
    e.center_row, e.center_col, e.basis, e.rounds, logical_error_rate;