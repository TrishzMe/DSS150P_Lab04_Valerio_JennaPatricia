# Reproducibility Report — DSS150P Laboratory Activity #4

| Field | Entry |
|---|---|
| Name | Jenna Patricia G. Valerio |
| Student Number | 2024102708 |
| Section | DSS150P_CM17 |
| Previous project | *Predicting PM2.5 Air Quality Exceedances Using Machine Learning: A Co-Pollutant Chemistry Approach* (Predictive Analytics group project: S. Grino, B. J. Perez, J. P. Valerio) |
| Dataset | World Air Quality Data 2024 (Updated), Kaggle user kanchana1990, CC BY 4.0, SHA-256 `2b2202a0fa1cffdeddf803e54dd7f822812d86de3a6449c03798bb872b272f10` |
| Predictive task | Classification: station-day PM2.5 > 15 µg/m³ (WHO 24-h guideline), *Unhealthy* vs *Safe* |
| Repository | https://github.com/TrishzMe/DSS150P_Lab04_Valerio_JennaPatricia |

## 1. Summary

The original notebook was rebuilt as a layered, orchestrated pipeline:
- **Layers:** source → raw → staging → curated → model input → outputs.
- **Runtimes:** Python modules, PostgreSQL 16 and Airflow 3.3.2, all in Docker Compose.
- **Safeguards:** checksummed source, data contract, quality checks, rerun-safe loads, tests and evidence.

**Result: the reproduction matches the baseline within tolerance on every platform, and exactly on Windows.**

| What was compared | Result |
|---|---|
| Baseline values checked automatically | All 29: 18 test/CV metrics, 3 confusion-matrix counts, 8 Random Forest feature importances |
| Agreement, Windows (Python 3.12.10, local venv) | Identical to the 6 decimals the notebook printed; maximum absolute difference **0** |
| Agreement, Linux containers (Python 3.12.14; Docker pipeline image, Airflow, clean clone) | 27 of 29 identical. Gradient Boosting CV F1 mean 0.711814 (diff **0.000396**) and std 0.006152 (diff 0.000250); all within tolerance (section 3, deviation 9) |
| Declared tolerance | ±0.0005 (3 decimal places) |
| Train/test split | Exactly the same 2,102 / 526 station-days as the original notebook (equal key checksums), on both platforms |
| Model-ready table | Bit-identical to the notebook's frame (2,628 rows, same order, same float values) |
| Held-out test predictions | Identical on both platforms (same confusion matrices for all three models) |

Outcome category (lab section 10.2): **Matches within tolerance**. The difference is zero on Windows, and on Linux the
only non-zero differences are explained in section 3.

## 2. Prior project vs. engineered reproduction

| Aspect | Original group project | This reproduction |
|---|---|---|
| Execution | One notebook; cells with shared globals, run order unknown (execution counts cleared) | `python -m src.cli <stage>` modules; `run-all`; Airflow DAG calling the same commands |
| Source | `world_air_quality.csv` next to the notebook, plus duplicate copies and zips | Single committed copy in `data/source/`, SHA-256 verified on every run (`source_manifest.yml`) |
| Environment | Kernel Python 3.13.9; library versions not recorded | Fully pinned `requirements.txt` lock; Docker images `python:3.12.14-slim-bookworm`, `apache/airflow:3.3.2-python3.12`, `postgres:16.15` |
| Layers | All in memory | Raw (full copy + 5 checkpointed batches), staging, quarantine, curated Parquet, PostgreSQL schemas |
| Data rules | Implicit pandas defaults (NA tokens, file order) | Explicit `null_tokens`, explicit order on `source_row_key`, data contract with fail / quarantine / exclude / warn |
| Split | `train_test_split(..., random_state=42)`, never saved | Same call over the same row order; `split_manifest.json` with key checksums; compared with the notebook's split |
| Metrics | Printed in the notebook, copied into the paper | `outputs/metrics.json`, compared automatically with `metadata/expected_metrics.json` |
| Models / predictions | In memory only | `outputs/model/*.joblib` + `model_card.json`, `predictions.csv`, `curated.predictions` table |
| Observability | None | JSON logs with `pipeline_run_id`, run records, `audit.*` tables, quality reports per layer |

### Baseline vs. reproduced metrics (held-out test set, n = 526; 5-fold CV)

| Model | Metric | Baseline | Reproduced | Abs. diff |
|---|---|---|---|---|
| Logistic Regression | Accuracy | 0.673004 | 0.673004 | 0 |
| | Precision (w) | 0.673589 | 0.673589 | 0 |
| | Recall (w) | 0.673004 | 0.673004 | 0 |
| | F1 (w) | 0.656994 | 0.656994 | 0 |
| | CV F1 mean ± std | 0.672786 ± 0.023194 | 0.672786 ± 0.023194 | 0 |
| Random Forest | Accuracy | 0.716730 | 0.716730 | 0 |
| | Precision (w) | 0.715967 | 0.715967 | 0 |
| | Recall (w) | 0.716730 | 0.716730 | 0 |
| | F1 (w) | 0.716283 | 0.716283 | 0 |
| | CV F1 mean ± std | 0.707718 ± 0.008398 | 0.707718 ± 0.008398 | 0 |
| Gradient Boosting | Accuracy | 0.705323 | 0.705323 | 0 |
| | Precision (w) | 0.703983 | 0.703983 | 0 |
| | Recall (w) | 0.705323 | 0.705323 | 0 |
| | F1 (w) | 0.704445 | 0.704445 | 0 |
| | **CV F1 mean (primary)** | **0.712210** ± 0.006402 | **0.712210** ± 0.006402 | **0** |

The reproduced values above are from the Windows run. In the Linux containers every value is the same except Gradient
Boosting's CV F1: 0.711814 ± 0.006152 (difference 0.000396 / 0.000250).

**Other values checked:**
- Confusion-matrix true-Safe counts: 98 / 146 / 141, all equal to the paper.
- Random Forest importances: O3 0.183439, Ox 0.180247, NO2/O3 ratio 0.179825, CO 0.157234, NO2 0.157030, SO2 0.120666, CO flag 0.012007, SO2 flag 0.009552, all equal.
- Best model by CV F1: Gradient Boosting, same as the original.

Machine-readable comparison: `outputs/reproducibility_comparison.json`.

## 3. Deviations from the original and why they do not change the result

| # | Deviation | Reason | Effect on this source version |
|---|---|---|---|
| 1 | Python 3.12 with pinned pandas 3.0.6 / numpy 2.5.4 / scikit-learn 1.9.1, instead of an unrecorded 3.13 environment | Reproducible, lockable environment; the Airflow image uses Python 3.12 | None: metrics identical |
| 2 | snake_case column names (`PM2.5` → `pm25`, `Ox_total_oxidant` → `ox_total_oxidant`, …) | SQL-friendly, consistent names | None: values and feature order unchanged |
| 3 | Explicit total order `(location, reading_date, country_label, city)` instead of `sort_values(['Location','Date'])` on a non-unique key | Deterministic ordering before the split (row order decides which rows go to test) | None: the order is identical (verified row by row) |
| 4 | Unparseable values/timestamps and unknown units are quarantined as invalid, with a 1% limit | The notebook would crash or silently drop them | None: 0 such rows in this file |
| 5 | Imputation moved from feature engineering to the model-input stage | It is a fitted step and does not belong in the curated layer | None in `baseline` scope: same medians on the same rows |
| 6 | Optional `impute_scope: train_only` (medians from training rows only; imputer refitted inside every CV fold) | Removes the original leakage | **None observed:** the medians are SO2 = 6.0 and CO = 600.0 whether computed on all rows or on training rows (values are heavily rounded), so all metrics are identical. Results: `outputs/variants/train_only/` |
| 7 | Model fitting and CV run single-threaded (`threadpoolctl` limit 1) | Removes thread-count-dependent floating-point reductions | None |
| 8 | Extra columns (`station_day_key`, `n_readings`, audit columns) | Lineage and keys | Not used as features |
| 9 | Execution on Linux (Docker image and Airflow) as well as Windows | Containerised, reviewer-friendly runtime | **One CV fold of Gradient Boosting differs** (fold 3: 0.714993 on Linux vs 0.716972 on Windows), giving CV mean 0.711814 vs 0.712210. The other folds, all test metrics and all predictions are identical. See below. |

**Investigation of deviation 9** (`docs/evidence/platform_difference.txt`):
- **Inputs are identical:** the curated table, split and wheels (numpy 2.5.4, scikit-learn 1.9.1) are the same on both platforms, and fitting is single-threaded and seeded.
- **numpy is ruled out:** switching off numpy's AVX2/FMA SIMD dispatch (`NPY_DISABLE_CPU_FEATURES=X86_V3`) does not change the Linux result.
- **Most likely cause:** scikit-learn's compiled tree and loss code is built with different compilers and linked against different C math libraries (GCC/glibc vs MSVC/UCRT). Last-bit differences in `exp`/`log` or in summation order can change which of two nearly equal splits a boosting stage picks in one fold.
- **Decision:** this is an environment-level non-determinism, not a data, preprocessing, split or parameter difference, and it stays inside the declared tolerance. The tolerance was therefore kept at the lab's default 3-decimal rule (0.0005); the margin is 0.0001.

## 4. Deterministic controls

- **Data identity:** SHA-256, size, header and row count are checked against the manifest before every ingest. `.gitattributes`
  marks the CSV `-text`, so git cannot rewrite its CRLF line endings.
- **Order:** the immutable `source_row_key` drives deduplication ("keep first"); curated data has a total order on
  its unique key (checked by DET-01).
- **Parsing:** values use the pandas parser (`pd.to_numeric`), which is identical to the notebook's `read_csv`. Python's
  `float()` differs in the last bit for some source values, and this was checked.
- **Seeds:** split `random_state=42`, CV `random_state=42`, Random Forest and Gradient Boosting `random_state=42`, plus `random`,
  `numpy` and `PYTHONHASHSEED`, all set from `config/settings.yml`.
- **Split proof:** `baseline/replay_notebook.py` re-executes the notebook's own code and records its train/test key
  checksums in `metadata/split_manifest.json`; every run must reproduce them (PRD-04).
- **Environment:** fully pinned lock (`uv pip compile --universal`); the same lock is used by the pipeline image and by
  Airflow's isolated `/opt/pipeline-venv`, so Airflow's own dependencies cannot change the model libraries.
- **Tolerance rule:** each metric must satisfy |reproduced − expected| ≤ 0.0005; counts and checksums must be exact. A
  failing comparison exits non-zero and blocks `publish_outputs`.

## 5. Evidence index

| Requirement | Evidence |
|---|---|
| Source identity | `metadata/source_manifest.yml`; `python -m src.cli verify-source`; test `test_verify_source_detects_a_changed_byte` |
| Profile / contract | `metadata/source_profile.md`, `metadata/data_contract.yml` |
| Incremental ingestion: interrupt, resume, rerun | `docs/evidence/ingestion_recovery_demo.txt` (54,255 rows, 54,255 distinct keys, 0 duplicates) |
| Rerun-safe load / second run | `docs/evidence/rerun_no_duplicates.txt` (0 inserted / 0 updated / 0 deleted; 0 duplicate keys) |
| Storage benchmark and partition | `docs/benchmark_report.md`, `outputs/benchmark/benchmark_results.json` |
| Airflow success | `docs/evidence/airflow_success.txt` (dependency graph, 11/11 tasks) |
| Airflow failure and recovery | `docs/evidence/airflow_db_outage.txt` (warehouse down → `load_postgres` failed after 4 tries, nothing published → DB restarted, task cleared → same run succeeded) |
| Failure injection | `docs/evidence/failure_drills.txt` (tampered source, missing column, 2% invalid rows, 0.2% invalid rows, invalid config) |
| Tests | `pytest`: 20 tests (unit, integration incl. PostgreSQL, data quality) |
| Metric comparison | `outputs/reproducibility_comparison.json`, `outputs/metrics.json` |
| Same rows / same split as the notebook | `docs/evidence/notebook_replay.json` (`python baseline/replay_notebook.py --compare-curated`) |
| Clean clone → containers → reproduction | `docs/evidence/clean_clone_check.txt` (fresh `git clone`, `.env` from template, all 11 stages, 20 tests, rerun without duplicates) |
| Windows vs Linux difference | `docs/evidence/platform_difference.txt` |

## 6. Known limitations

- **Precision of the baseline:** baseline metrics are known only to 6 decimals (the notebook's printed outputs), so "identical"
  means identical to 6 decimals.
- **Platform dependence:** exact reproduction of Gradient Boosting's CV score depends on the platform. Linux differs by
  0.000396, close to the 0.0005 limit. A future library or OS change could exceed it, and then the tolerance would need
  re-justifying (for example ±0.001 for CV metrics only) or the reference platform would need fixing.
- **Simulated incremental arrival:** the dataset is a single static snapshot, so incremental arrival is simulated with
  sequence batches (see Q4). The checkpoint is not a business watermark.
- **Outliers kept:** for example, CO up to 3,198,676 µg/m³. The original project did not cap them, and changing that would change the
  baseline.
- **SQL vs Python averages:** the SQL rebuild of station-day means differs from pandas by up to 7×10⁻¹⁵ because of summation order. The
  check tolerates 10⁻⁶ and the model always reads the Parquet (pandas) values.
- **Mutable image tags:** Docker image tags are pinned but not digests; an upstream rebuild of a tag could change OS packages
  (not Python packages, which come from the lock).
- **Dev-mode Airflow:** Airflow runs in `standalone` mode (single container), which suits a lab, not production.
- **Single-machine benchmark:** benchmark timings come from one laptop and are not general (see `docs/benchmark_report.md`).
- **Windows-specific issues, handled and documented in the runbook:** `localhost` resolving to IPv6 added about 10 s per DB
  connection, and antivirus briefly locking new files.

## 7. Design decision I would change for a real production system

The ingestion checkpoint and the run records are **JSON files on a bind-mounted folder**, protected by a file lock.
This is simple and transparent for a lab, but in production I would keep them in the warehouse, in tables such as
`audit.ingest_checkpoint` and `audit.pipeline_runs`, updated in the same transaction as the data they describe:
- the checkpoint could never disagree with the committed data;
- concurrent workers would be safe without file locks;
- the state would survive the loss of a worker's disk.

The data layers would move from the local disk to object storage, so that Airflow workers on different machines share them.

## 8. Answers to the analysis and reflection questions

**1. What prevented another person from reproducing the previous project immediately?**
- **No environment record:** only the kernel version (Python 3.13.9) was saved, with no library versions.
- **Unclear source:** the dataset was loaded from the notebook's working folder, with several copies of the file and no checksum.
- **Hidden state:** all execution counts were cleared, so the run order is unknown, and cells depend on globals defined earlier.
- **Never persisted:** neither the split nor the models nor the predictions were saved.
- **Metrics only as text:** they existed only as printed notebook text and numbers pasted into the paper.
- **Implicit cleaning rules:** pandas' default NA handling turned `N/A`/`None` city names into missing values, and deduplication depended on file order.

**2. Which artifact is the true authoritative source?**
- **Authoritative: the original file in `data/source/`.** It is immutable and identified by its SHA-256, and every other dataset can be regenerated from it.
- **Raw:** a faithful, verifiable copy with lineage columns (which source record, which batch, which run). It exists so that processing never touches the source, and so that ingestion is restartable.
- **Curated:** the authoritative *business* definition for consumers (station-day means, WHO label), but it is derived and rebuildable.
- **Model-ready:** the model's view, derived from curated (imputation, split). It is never edited by hand.

If a derived layer disagrees with the source, the derived layer is wrong.

**3. What identifier or rule prevents duplicates, and what assumption makes it valid?**
- **Raw: `source_row_key`**, the record's ordinal in the file, together with a checkpoint of committed batches. This is valid because the file is immutable and checksum-pinned, so the ordinal always names the same record. `record_hash` additionally detects altered content.
- **Staging: `(location, pollutant, last_updated_utc)`**, keeping the first record in source order. This assumes a station reports one value per pollutant per timestamp; extra rows are duplicate sensor readings (8,074 were removed, the same rule as the original notebook).
- **Curated: `station_day_key = sha256(country_label, city, location, reading_date)`**. This assumes the four fields identify a station-day; `location` alone is not unique across countries.
- **PostgreSQL:** primary keys and unique constraints enforce each rule, and loads use `ON CONFLICT`.

**4. How was incremental ingestion simulated without pretending the checkpoint is a business timestamp?**
- **Why not `Last Updated`:** it is a per-station *last reported* time spanning 2014–2024 (78% of stations report only one date). It says nothing about when a row arrived, so it cannot be a watermark.
- **What was done instead:** the immutable file is split into 5 contiguous batches of `source_row_key`. The checkpoint stores committed batch IDs and is labelled `"checkpoint_type": "simulated_sequence_checkpoint (not a business watermark)"`.
- **Where this is documented:** the README and the `ingest.py` docstring.

**5. Which transformations belong in staging versus curated, and why?**
- **Staging: record-level standardisation that is true regardless of the use case.** This covers null tokens, type conversion, UTC timestamps, deduplication of sensor readings, unit consistency, and invalid (negative) values, with every rejected row quarantined with a reason.
- **Curated: business meaning for this data product.** This covers aggregation to station-day means, the model-ready selection (PM2.5, NO2 and O3 present), row-level derived features, and the WHO label.
- **Neither: imputation.** It is fitted on data, so it lives in the model-input stage, where it can be fitted on the right rows (all rows in the baseline, training rows in the leakage-free variant).

**6. Which storage format performed best, and why should the result not be generalised?**

Parquet performed best on every measure for this workload:

| Measure | Parquet | Others |
|---|---|---|
| Size | 2.9 MB | CSV 7.6 MB, PostgreSQL 11.0 MB, JSONL 15.2 MB |
| Full-read median | 0.008 s | CSV 0.147 s, PostgreSQL 0.344 s, JSONL 0.367 s |
| Selective read | 0.009 s | PostgreSQL with its index 0.014 s |

It also preserves types exactly, which the bit-identical reproduction depends on.

These results should not be generalised:
- the data is small (about 30k rows);
- every read loads all columns into pandas;
- the database runs on the same laptop and returns rows through a Python driver.

With larger data, concurrent users, many selective queries, joins or updates, PostgreSQL (or a warehouse engine) wins. A different storage medium or column subsets would change the ranking too.

**7. What happens if Airflow retries the load task after the database committed but the task never received confirmation?**

The retry runs the whole load again: COPY the same snapshot into a temporary table, then merge with `ON CONFLICT`. Every key already exists with an identical `row_hash`, so it inserts 0, updates 0 and deletes 0 rows. `audit.load_log` records a second, empty load, and the tables are unchanged.

**8. How does the design make that retry safe?**
- **One transaction:** COPY, upsert and snapshot delete either commit together or not at all.
- **Business-key merge:** primary keys on business keys plus `INSERT … ON CONFLICT DO UPDATE … WHERE row_hash IS DISTINCT FROM EXCLUDED.row_hash`.
- **Deterministic snapshot:** `row_hash` excludes the audit columns, so a rerun of the same snapshot has the same hashes.
- **Publication:** predictions are keyed on `(station_day_key, model_name)` and metrics on `(run, model, metric)`.

Proven by the integration test `test_postgres_load_is_rerun_safe_and_matches_parquet` and by `docs/evidence/rerun_no_duplicates.txt`.

**9. Which data-quality check is most likely to detect a change that would silently damage model accuracy?**

**APP-01**, the expected row counts per layer for the registered source version, together with **INV-01/VAL-03** (unknown or unparseable values above 1%).

The most dangerous silent change found during profiling is the unit filter: it matches the exact string `µg/m³`. A re-encoded file would make the original notebook drop almost every row without any error, and the model would train on a tiny or biased sample.

In this pipeline:
- unknown units are classified as invalid, not "excluded", so INV-01 fails;
- APP-01 fails as soon as the staging, station-day or model-ready counts differ from the declared 30,545 / 11,558 / 2,628;
- CON-01 (label consistency) and PRD-04 (split checksums) protect the model side.

**10. How do you know the training and test rows are the same logical rows used in the baseline?**
- **Independent replay:** `baseline/replay_notebook.py` re-runs the original notebook code verbatim on the same checksummed file. It reproduces the saved metrics, then records the SHA-256 of the station-day keys in its train and test sets (`metadata/split_manifest.json`).
- **Pipeline check:** every pipeline run computes the same checksums from its own split, and the reproducibility check requires them to be equal. They are.
- **Input check:** `--compare-curated` compares the curated model-ready table with the notebook's frame before imputation; it is identical in row order and float values (`docs/evidence/notebook_replay.json`). The same input, the same seed and the same `train_test_split` call therefore select the same rows.

**11. What sources of non-determinism remain?**
- **Operating system / compiled math libraries (observed):** the same wheels give Gradient Boosting CV F1 0.712210 on Windows and 0.711814 on Linux, because one CV fold builds a slightly different boosting sequence (deviation 9). The likely cause is platform-specific floating-point behaviour in scikit-learn's compiled code. Single-threading removes thread-order effects but not this.
- **CPU architecture:** other CPU architectures (for example ARM) could add similar last-bit differences.
- **Mutable image tags:** Docker base images are pinned by tag, not digest.
- **Audit metadata:** run IDs, timestamps and durations differ by design; `row_hash` excludes them.
- **Benchmark timings:** these naturally vary, which is why medians are reported.
- **Upstream changes:** the Kaggle "Updated" dataset may change upstream. The checksum turns that into an explicit new version instead of silent drift.

**12. If the reproduced metric differed, how would you find whether the cause is data, preprocessing, split, dependencies, parameters or non-determinism?**

Isolate one layer at a time (runbook section 6):
1. **Data:** `verify-source` and the per-layer counts (APP-01).
2. **Preprocessing:** compare the curated model-ready table with `baseline/replay_notebook.py`'s frame.
3. **Split:** compare the key checksums in `reproducibility_comparison.json`.
4. **Dependencies:** compare the `model_card.json` environment with the lock.
5. **Parameters:** compare `model_card.json` params with `settings.yml`.
6. **Non-determinism:** run twice in the same environment. If the same inputs give different outputs, look for unseeded or multi-threaded steps.

Because each artefact is persisted with a checksum, the first layer whose checksum differs identifies the cause.

**13. What would you monitor if this pipeline ran daily?**
- **Runs:** status, duration per task and retries (Airflow, `audit.task_runs`).
- **Row counts per layer and quarantine rate per reason**, compared with recent runs (thresholds plus anomaly alerts).
- **Source version changes and data-quality check failures.**
- **Load statistics** (inserted / updated / deleted) for unexpected churn.
- **Model metrics:** the class balance and predicted exceedance rate (drift), and model metrics against the baseline when labels arrive.
- **Infrastructure:** database size, connection errors and disk space for the data layers.

**14. What becomes the first bottleneck at 100× data?**

At about 5.4 million rows (650 MB of CSV), the bottleneck is the **in-memory pandas processing on a single machine**:
- ingestion reads the whole file and hashes every row in Python;
- staging and the pivot hold everything in memory;
- the PostgreSQL load writes rows one by one through COPY from Python;
- the CON-04 consistency check reads entire tables back into pandas.

Fixes:
- process per batch or partition;
- move the pivot into SQL or DuckDB/Polars (or Spark);
- compare aggregate hashes instead of full tables;
- use server-side `COPY FROM` file loads.

**15. What would change if the dataset became sensitive or regulated?**
- **Source data:** stop committing it; store it in access-controlled, encrypted storage, keeping only the manifest (checksum, schema) and acquisition instructions in git.
- **Identifiers:** pseudonymise station identifiers and coordinates at ingestion.
- **Access:** use least-privilege database roles (separate loader, reader and model users) and secrets from a secret manager instead of `.env`.
- **Quarantine and logs:** treat quarantine files and logs as sensitive too; they contain raw rows, so mask values in logs.
- **Governance:** add retention and deletion policies and access auditing, and check the licence and consent conditions before publishing predictions.
