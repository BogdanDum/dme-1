"""Gold SQL model and Silver-to-Gold transformations.

Syndrome experiments and decoder predictions are normalized; packed Google
bytes stay lossless without expanding millions of detector events. Source IDs
resolve through the existing source_trace Parquet evidence. QASMBench circuits
are deliberately not joined to experiments: no shared row identifier exists.
"""

GOLD_TABLES = (
    "syndrome_experiment", "syndrome_observation", "experiment", "shot",
    "decoder", "decoder_prediction", "circuit", "stabilizer_check",
    "conditional_correction",
)

DDL = """
CREATE SCHEMA IF NOT EXISTS gold;
CREATE TABLE IF NOT EXISTS gold.syndrome_experiment (
    experiment_id text PRIMARY KEY,
    physical_fault_rate double precision NOT NULL CHECK (physical_fault_rate BETWEEN 0 AND 1),
    round_count integer NOT NULL CHECK (round_count = 4),
    check_count integer NOT NULL CHECK (check_count = 4)
);
COMMENT ON TABLE gold.syndrome_experiment IS 'One source-specific synthetic fault-rate experiment.';
CREATE TABLE IF NOT EXISTS gold.syndrome_observation (
    source_record_id text PRIMARY KEY,
    experiment_id text NOT NULL REFERENCES gold.syndrome_experiment,
    syndrome_bits bytea NOT NULL CHECK (
        octet_length(syndrome_bits) = 16 AND encode(syndrome_bits, 'hex') ~ '^(00|01){16}$'
    ),
    logical_error_label boolean NOT NULL,
    sample_weight bigint NOT NULL CHECK (sample_weight > 0)
);
COMMENT ON TABLE gold.syndrome_observation IS 'One weighted source observation; identical patterns may have different labels.';
CREATE INDEX IF NOT EXISTS syndrome_experiment_idx ON gold.syndrome_observation (experiment_id);
CREATE TABLE IF NOT EXISTS gold.experiment (
    source_record_id text NOT NULL UNIQUE,
    experiment_id text PRIMARY KEY,
    basis text NOT NULL CHECK (basis IN ('X', 'Z')),
    distance integer NOT NULL CHECK (distance > 0),
    rounds integer NOT NULL CHECK (rounds > 0),
    shots bigint NOT NULL CHECK (shots >= 0),
    center_row integer NOT NULL,
    center_col integer NOT NULL,
    measurement_count integer NOT NULL CHECK (measurement_count >= 0),
    detector_count integer NOT NULL CHECK (detector_count >= 0)
);
COMMENT ON TABLE gold.experiment IS 'One Google hardware experiment directory.';
CREATE TABLE IF NOT EXISTS gold.shot (
    source_record_id text PRIMARY KEY,
    experiment_id text NOT NULL REFERENCES gold.experiment,
    shot_index bigint NOT NULL CHECK (shot_index >= 0),
    measurement_bits bytea NOT NULL,
    sweep_bits bytea NOT NULL,
    detector_bits bytea NOT NULL,
    detector_event_count integer NOT NULL CHECK (detector_event_count >= 0),
    actual_observable_flip boolean NOT NULL,
    UNIQUE (experiment_id, shot_index)
);
COMMENT ON TABLE gold.shot IS 'One aligned Google shot with packed bits and a detector summary.';
CREATE TABLE IF NOT EXISTS gold.decoder (
    decoder_name text PRIMARY KEY
);
COMMENT ON TABLE gold.decoder IS 'One decoder algorithm.';
CREATE TABLE IF NOT EXISTS gold.decoder_prediction (
    source_record_id_shot text NOT NULL REFERENCES gold.shot,
    decoder_name text NOT NULL REFERENCES gold.decoder,
    predicted_observable_flip boolean NOT NULL,
    PRIMARY KEY (source_record_id_shot, decoder_name)
);
COMMENT ON TABLE gold.decoder_prediction IS 'One predicted observable flip by one decoder for one shot.';
CREATE INDEX IF NOT EXISTS prediction_decoder_idx ON gold.decoder_prediction (decoder_name);
CREATE TABLE IF NOT EXISTS gold.circuit (
    source_record_id text NOT NULL UNIQUE,
    circuit_id text PRIMARY KEY,
    benchmark_name text NOT NULL,
    variant text NOT NULL,
    register_declarations text NOT NULL,
    qubit_count integer NOT NULL CHECK (qubit_count > 0),
    measurement_count integer NOT NULL CHECK (measurement_count >= 0),
    two_qubit_gate_count integer NOT NULL CHECK (two_qubit_gate_count >= 0)
);
COMMENT ON TABLE gold.circuit IS 'One independent QASMBench circuit variant.';
CREATE TABLE IF NOT EXISTS gold.stabilizer_check (
    source_record_id text NOT NULL UNIQUE,
    circuit_id text NOT NULL REFERENCES gold.circuit,
    check_id text NOT NULL,
    ancilla_qubit text NOT NULL,
    data_qubits text[] NOT NULL CHECK (cardinality(data_qubits) > 0),
    syndrome_bit text NOT NULL,
    PRIMARY KEY (circuit_id, check_id)
);
COMMENT ON TABLE gold.stabilizer_check IS 'One parity check mapping data qubits to an ancilla and syndrome bit.';
CREATE TABLE IF NOT EXISTS gold.conditional_correction (
    source_record_id text PRIMARY KEY,
    circuit_id text NOT NULL REFERENCES gold.circuit,
    condition_register text NOT NULL,
    condition_value bigint NOT NULL CHECK (condition_value >= 0),
    gate text NOT NULL,
    target_qubit text NOT NULL
);
COMMENT ON TABLE gold.conditional_correction IS 'One syndrome-controlled recovery operation in a circuit.';
CREATE INDEX IF NOT EXISTS correction_circuit_idx ON gold.conditional_correction (circuit_id);
CREATE OR REPLACE VIEW gold.syndrome_value AS
SELECT source_record_id, i / 4 AS round_index, i % 4 AS check_index,
       get_byte(syndrome_bits, i) AS value
FROM gold.syndrome_observation CROSS JOIN generate_series(0, 15) AS position(i);
CREATE OR REPLACE VIEW gold.decoder_error AS
SELECT p.*, s.experiment_id, s.shot_index,
       p.predicted_observable_flip <> s.actual_observable_flip AS decoder_error
FROM gold.decoder_prediction p JOIN gold.shot s USING (source_record_id);
"""

INSERTS = """
INSERT INTO gold.syndrome_experiment
SELECT DISTINCT experiment_id, physical_fault_rate, round_count, check_count
FROM silver_syndrome_observation;
INSERT INTO gold.syndrome_observation
SELECT source_record_id, experiment_id, syndrome_bits, logical_error_label, sample_weight
FROM silver_syndrome_observation;
INSERT INTO gold.experiment SELECT * FROM silver_experiment;
INSERT INTO gold.shot
SELECT source_record_id, experiment_id, shot_index, measurement_bits, sweep_bits,
       detector_bits, detector_event_count, actual_observable_flip
FROM silver_shot;
INSERT INTO gold.decoder VALUES
    ('belief_matching'), ('correlated_matching'), ('pymatching'), ('tensor_network_contraction');
INSERT INTO gold.decoder_prediction
SELECT s.source_record_id, p.decoder_name, p.prediction
FROM silver_shot s CROSS JOIN LATERAL (VALUES
    ('belief_matching', s.belief_matching_prediction),
    ('correlated_matching', s.correlated_matching_prediction),
    ('pymatching', s.pymatching_prediction),
    ('tensor_network_contraction', s.tensor_network_contraction_prediction)
) p(decoder_name, prediction);
INSERT INTO gold.circuit SELECT * FROM silver_circuit;
INSERT INTO gold.stabilizer_check SELECT * FROM silver_stabilizer_check;
INSERT INTO gold.conditional_correction SELECT * FROM silver_conditional_correction;
"""