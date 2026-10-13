# Phase I — Lifecycle Mapping, Baseline Capture and Engineering Gaps

This document reconstructs what the original Predictive Analytics group project
(notebook `Predictive_Final_Project_code.ipynb` and the project paper; not included in this repository,
cited in README section 14)
actually did, records its result as the baseline to reproduce, and lists the engineering gaps
this repository closes.

## 1. Lifecycle of the original project

| Lifecycle stage | What the original project did | Where |
|---|---|---|
| Generation | Government/research monitors report pollutant readings; OpenAQ aggregates them; a Kaggle user publishes a snapshot CSV | upstream, outside the project |
| Ingestion | Manual download from Kaggle; the CSV (and a zip copy) placed next to the notebook | manual |
| Storage | One local CSV, read straight into memory; no intermediate files | `pd.read_csv('world_air_quality.csv', sep=';')` (cell 1) |
| Transformation | Filter 6 pollutants → parse timestamps (UTC) → drop duplicate readings → keep µg/m³ → drop negatives → fill missing geography → pivot to station-day means | cells 3 and 5 |
| Feature engineering | Require PM2.5/NO2/O3; median-impute SO2/CO plus missing flags; add `Ox_total_oxidant`, `NO2_O3_ratio` | cell 9 |
| Model training | WHO threshold label; stratified 80/20 split (seed 42); Logistic Regression, Random Forest(200), Gradient Boosting(200) | cell 11 |
| Evaluation | Test-set accuracy / weighted precision, recall, F1; 5-fold stratified CV (F1-weighted); confusion matrices; RF importances | cell 14 |
| Serving / consumer | `predict_air_quality()` function inside the notebook; results pasted into the paper and slides | cells 16–18, paper |

The engineered lifecycle implemented in this repository is drawn in `docs/architecture.png`
(components and data movement) and `docs/lineage.png` (dataset- and field-level flow).

```
data/source (immutable CSV, SHA-256)  ->  ingest  ->  data/raw (full copy + 5 checkpointed batches)
  -> transform_staging -> data/staging (typed, deduplicated, quarantine)  -> build_curated
  -> data/curated (station-day table, model-ready table)  -> load_postgres (staging/curated/audit schemas)
  -> train (model input: imputation + deterministic split) -> evaluate -> reproducibility_check
  -> publish (outputs/, curated.predictions, curated.model_metrics) -> consumers (report, analysts)
```

## 2. Manual, hidden, non-deterministic and environment-specific parts

| Type | Observation in the original project |
|---|---|
| Manual | Dataset downloaded and unzipped by hand; two identical copies of the notebook and CSV (plus zips) exist, with nothing saying which is authoritative. Figures and metric tables were copied into the paper by hand. |
| Notebook-only | All logic lived in notebook cells; there was no script or command that runs the pipeline end to end. |
| Hidden state | Saved outputs show `execution_count = None` for every cell, so the run order is unknown; later cells depend on globals (`df_model_ready`, `X`, `best_model`) created earlier. |
| Hidden assumptions | `read_csv` defaults silently turned the text values `N/A` and `None` (City/Location) into missing values; the unit filter depends on the exact string `µg/m³`, so a re-encoded file would silently drop every row. |
| Order-dependent | `drop_duplicates(keep='first')` keeps whichever duplicate comes first in the file; it runs before the unit filter, so the result depends on source row order. `sort_values(['Location','Date'])` sorts on a non-unique key. |
| Non-deterministic risk | Split, RF and GB were seeded (`random_state=42`), but the split was never persisted; library versions were not recorded, so a library upgrade could silently change results. |
| Leakage | SO2/CO medians were computed on all 2,628 rows before the train/test split, and CV was run on the already-imputed data. |
| Environment-specific | Relative path that only works when Jupyter starts in the notebook's folder; kernel Python 3.13.9; no dependency list. |

## 3. Baseline record (what must be reproduced)

| Item | Baseline value |
|---|---|
| Model family | Logistic Regression (`max_iter=1000`), Random Forest (`n_estimators=200, random_state=42`), Gradient Boosting (`n_estimators=200, random_state=42`) |
| Target | `Air_Quality` = Unhealthy if station-day mean PM2.5 > 15.0 µg/m³ else Safe (1,513 Unhealthy / 1,115 Safe) |
| Features (ordered) | NO2, O3, SO2, CO, SO2_missing, CO_missing, Ox_total_oxidant, NO2_O3_ratio |
| Preprocessing | filter 6 pollutants → UTC timestamps → dedup (Location, Pollutant, Last Updated) → µg/m³ only → value ≥ 0 → fill City/Location/Country Label with "Unknown" → station-day mean pivot → require PM2.5/NO2/O3 → median-impute SO2/CO + flags → two engineered features |
| Train/test strategy | `train_test_split(test_size=0.2, random_state=42, stratify=y, shuffle=True)` → 2,102 / 526 rows |
| Validation | `StratifiedKFold(5, shuffle=True, random_state=42)`, `f1_weighted`, over all model-ready rows |
| Random seed | 42 (split, CV, RF, GB) |
| Best model | Gradient Boosting (highest CV F1 = 0.712) |

| Model | Accuracy | Precision (w) | Recall (w) | F1 (w) | CV F1 mean | CV F1 std |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.673004 | 0.673589 | 0.673004 | 0.656994 | 0.672786 | 0.023194 |
| Random Forest | 0.716730 | 0.715967 | 0.716730 | 0.716283 | 0.707718 | 0.008398 |
| Gradient Boosting | 0.705323 | 0.703983 | 0.705323 | 0.704445 | 0.712210 | 0.006402 |

Machine-readable form: `metadata/expected_metrics.json`.

## 4. Engineering-gap table

| # | Current behaviour (original project) | Risk | DSS150P concept | Remediation in this repository |
|---|---|---|---|---|
| 1 | CSV read from the notebook's working folder; several copies of the file and zips | Cannot run elsewhere; unclear which copy is the source | Source systems; reproducible environments | One copy in `data/source/`; relative paths in `config/settings.yml`; SHA-256 in `metadata/source_manifest.yml`; `verify-source` check |
| 2 | No dependency versions recorded | A library upgrade can change splits or metrics | Reproducible environments | Fully pinned `requirements.txt` lock, `.python-version`, Dockerfile on a fixed Python image |
| 3 | Notebook cells with shared globals, unknown run order | Hidden execution state | Modular code; orchestration | `src/` modules, `python -m src.cli` entry point, Airflow DAG calling the same commands |
| 4 | Implicit pandas NA parsing (`N/A`, `None` → missing) | Results change if reader options change | Schemas and data contracts | Raw layer keeps original text; explicit `null_tokens` list in config and contract |
| 5 | Duplicate removal depends on file order | Re-exported or re-sorted data keeps different rows | Ingestion; deduplication | Immutable `source_row_key`; dedup keeps the lowest key; duplicates archived in quarantine |
| 6 | Sort on non-unique (Location, Date) | Tie order could move rows between train and test | Deterministic transformations | Total ordering on the unique station-day key before the split |
| 7 | Imputation fitted on all rows before the split | Test information leaks into training | Predictive integrity | Baseline-faithful default (to reproduce) + `impute_scope: train_only` variant, both reported |
| 8 | Seeded split never saved | Cannot prove which rows were in the test set | Reproducibility | `split_manifest.json` with row counts and key checksums, compared on every run |
| 9 | Metrics only printed in the notebook / pasted in the paper | No machine-readable output | Observability; delivery | `outputs/metrics.json` + automatic comparison with `metadata/expected_metrics.json` |
| 10 | Models and predictions only in memory | Nothing to audit or serve | Delivery | `outputs/model/` artefacts + model card, `outputs/predictions.csv`, PostgreSQL tables |
| 11 | No data validation; rows silently dropped | Upstream changes go unnoticed | Data quality | Contract checks (fail / quarantine / exclude / warn), quarantine with reasons, row-count reconciliation |
| 12 | Cleaning in memory only, no intermediate layers | Cannot inspect, restart or reuse | Storage and data organisation | Raw / staging / curated Parquet layers and PostgreSQL schemas |
| 13 | No run identity or logs | Cannot trace which run produced which output | Observability; lineage | `pipeline_run_id` on every row, JSON logs, run records, `audit` schema |
