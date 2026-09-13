# dme-1 — DSAIT4000 Project 1: QEC data pipeline

Team repository for the QEC data-pipeline assignment. The course package is
extracted at [`assignment-1/`](assignment-1/); all our code lives inside
[`assignment-1/starter/`](assignment-1/starter/).

## Status

| Part I stage | State | Where |
| --- | --- | --- |
| **Bronze** — verify supplied objects | **done** | `stages/register_sources.py` |
| **Silver** — six required Parquet tables | **done** | `stages/prepare_data.py` + `sources/` |
| **results/part1** — issues, trace, run evidence | **done** | `results.py` |
| Gold — PostgreSQL model and load | not started | `stages/load_postgres.py` (stub) |
| Gold→ML export — the two ML tables | not started | `stages/build_ml_tables.py` (stub) |
| Part I SQL analyses | not started | `results/part1/analysis/` |
| Part II — all three tasks | not started | `stages/train.py` (stub) |

Bronze and Silver are complete, tested, and reconciling. Everything below the
line is still open.

## First-time setup

Needs Docker with Compose, ~8 GB of memory available to Docker, ~5 GB of disk,
and GNU Make. Full instructions in
[`assignment-1/assignment/getting-started.md`](assignment-1/assignment/getting-started.md).

```bash
brew install --cask docker      # then launch it and set memory to 8 GB
cd assignment-1
make bootstrap                  # pulls images, starts services, seeds Bronze
```

That brings up PostgreSQL, MinIO, Adminer and JupyterLab, and loads the three
supplied archives into the Bronze area. Then:

- JupyterLab <http://localhost:8888/lab?token=quantum-course>
- MinIO console <http://localhost:9001> (`quantum` / `quantum-course-only`)
- Adminer <http://localhost:8080>

The datasets are **not** in this repo — the submission checklist forbids
committing course archives. Get `assignment-1.zip` from Brightspace and extract
its `datasets/` directory into `assignment-1/`, or copy it from a teammate.

## Running it

From inside the workspace container (`docker compose exec workspace bash`), in
`/workspace`:

```bash
make run        # Part I: verify Bronze, build Silver, write results/part1
make test       # 115 tests
make check      # platform connectivity
make inventory  # lists the three Bronze objects

make syndromes  # re-run one source only, while iterating
make google
make qasm
```

`make run` is the single documented Part I command. It is safe to re-run: a
second run over unchanged input rewrites the same bytes rather than appending.

The single-source targets are for iteration only. Their lineage covers just the
source they processed, so they write evidence to
`starter/results/partial/<source>/` and leave the canonical `results/part1/`
alone — otherwise the Silver tables from the last full run would be left
untraced. Always finish with a full `make run` before committing evidence.

### Running on the host, without Docker

Useful for fast iteration on the parsers. Bronze is read from the extracted
bundle and Silver is written to a local directory:

```bash
cd assignment-1/starter
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'

LAKE_BACKEND=local \
LOCAL_BRONZE_ROOT=../datasets/student-bundle/core \
LOCAL_LAKE_ROOT=lake \
PYTHONPATH=src .venv/bin/python -m quantum_lake_student.cli run
```

`LOCAL_BRONZE_ROOT` (where Bronze is read) is deliberately separate from
`LOCAL_LAKE_ROOT` (where Silver is written). Do not point them at the same
directory — Silver would land inside the read-only supplied bundle.

## Where the code is

All paths relative to `assignment-1/starter/`.

**Supplied by the course — do not rewrite:** `config.py`, `connections.py`,
`formats.py` (Stim `b8`/`01` readers), `ml.py` (splits, bit unpacking, weighted
LER), `models.py` (`stable_record_hash`, `QualityFinding`, `StageResult`).

**Our code**, under `src/quantum_lake_student/`:

| File | What it owns |
| --- | --- |
| `pipeline.py` | `run_part1()` — the orchestrator behind `make run` |
| `schemas.py` | the six Silver schemas, transcribed from `silver-tables.md` |
| `ids.py` | every `source_record_id` and business-id constructor |
| `lake.py` | Bronze reads, staged all-or-nothing publish, atomic replace |
| `quality.py` | the 27-rule check catalogue, `FatalDataError`, issue log |
| `runcontext.py` | per-run state and the streaming lineage trace |
| `results.py` | `run.json`, `row_counts.json`, `trace_examples.json` |
| `sources/syndromes.py` | the seven syndrome CSVs |
| `sources/google.py` | the five hardware experiments and their 250k shots |
| `sources/qasm.py` | the OpenQASM 2.0 parser |
| `stages/register_sources.py` | Bronze: hashes, member safety |
| `stages/prepare_data.py` | Silver: runs the parsers, publishes |
| `cli.py` | extended with `run`, `--run-id`, `--source` |

Tests are in `tests/` (8 files, 115 tests). Fixtures are generated in
`conftest.py` — no binary test data is committed.

## Where the artifacts are

**Silver is not in git.** It goes to the object store, or to `starter/lake/`
locally, both of which are gitignored — it is regenerated by `make run`.

| Artifact | Location | Rows | In git? |
| --- | --- | --- | --- |
| `syndrome_observation.parquet` | `silver/qec_syndromes/` | 75,598 | no |
| `experiment.parquet` | `silver/google_qec/` | 5 | no |
| `shot.parquet` | `silver/google_qec/` | 250,000 | no |
| `circuit.parquet` | `silver/qasmbench/` | 6 | no |
| `stabilizer_check.parquet` | `silver/qasmbench/` | 4 | no |
| `conditional_correction.parquet` | `silver/qasmbench/` | 6 | no |
| `source_trace.parquet` | `starter/results/part1/` | 2,075,619 | **yes** |
| `data_issues.parquet` | `starter/results/part1/` | 21 | **yes** |
| `run.json`, `row_counts.json`, `trace_examples.json` | `starter/results/part1/` | — | **yes** |

On the MinIO backend Silver is at `s3://quantum-lake/silver/...`; browse it in
the console or read it with `pyarrow`/`duckdb`. Bronze stays at
`s3://quantum-lake/bronze/source=<name>/*.zip`, untouched.

To inspect a table:

```python
import duckdb
duckdb.sql("SELECT * FROM 'lake/silver/qasmbench/circuit.parquet'").show()
```

## Decisions worth not undoing

These are load-bearing for the rubric. If you change one, change the tests and
the design report with it.

- **`source_record_id` is a content hash, with no row index.** `(labels,
  syndromes)` is unique inside every supplied CSV, so nothing depends on row
  order — which the tracing contract requires.
- **Google bits stay packed.** `measurement_bits`, `sweep_bits` and
  `detector_bits` hold the supplied `b8` bytes. Unpacking is Part II's job.
- **A decoder mistake is never stored.** It is derived (`prediction != actual`)
  and counted only as evidence. Measurements, detector events, actual flips,
  predictions and mistakes stay five distinct concepts — conflating them caps
  the grade at 70.
- **`quantity` is a weight, never expanded.** 75,598 rows represent 70,000,000
  observations. Expanding them also caps the grade at 70.
- **Sources are never joined in Silver.** QASMBench has no row-level match to
  the experiments; integration belongs in Gold, and the rejected relationship is
  a required part of the report.
- **A parity check is only recorded where explicit.** The ancilla must
  accumulate through two-qubit gates alone, then be measured. `qec_en_n5` and
  `error_correctiond3_n5` therefore yield zero checks, recorded as `info`
  findings rather than invented structure.
- **Reruns are byte-identical.** Silver and `source_trace` hash the same on a
  second run; `data_issues` differs only in `run_id`, which its contract
  requires it to carry. `tests/test_rerun.py` enforces this.

## Findings for the design report

Two things worth writing up, both verified against the raw files:

- `sweep.b8` contains only 10 distinct patterns, in 10 contiguous blocks of
  5,000 shots, so `shot_index` correlates with the initialisation state. The
  prescribed odd/even + `%10 == 8` split interleaves and is safe; any contiguous
  split would leak.
- `belief_matching` and `tensor_network_contraction` were configured from
  `pij_from_even_for_odd.dem` / `pij_from_odd_for_even.dem` depending on shot
  parity, while `pymatching` and `correlated_matching` used a single error
  model. The prescribed test split is exactly the odd shots, so decoder
  predictions are not exchangeable across parity.

## Picking up the next piece

Gold is the largest remaining design task and the assignment's main design
marks. Start from [`assignment-1/assignment/brief.md`](assignment-1/assignment/brief.md)
("Gold design requirements") and
[`required-ml-tables.md`](assignment-1/assignment/required-ml-tables.md).

Three things Silver already sets up for you:

1. Silver column names match `ml.GOOGLE_META_PREDICTION_COLUMNS` exactly, so the
   supplied helpers work unmodified downstream.
2. The syndrome Silver grain maps 1:1 to `ml_syndrome_decoder_example`, so no
   aggregation is needed on that path.
3. `run.json` and `row_counts.json` already carry `silver_to_gold` and `loaded`
   placeholders — extend them rather than restructuring the files.

Gold must live in its own PostgreSQL schema; `course_admin` is reserved for
platform metadata.
