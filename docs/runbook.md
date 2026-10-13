# Runbook — PM2.5 exceedance reproducibility pipeline

Operational guide for running, checking, restarting and recovering the pipeline. Commands are run
from the repository root. "CLI" means `python -m src.cli` (local virtual environment) or
`docker compose run --rm pipeline` (container). Both run the same code.

## 1. Normal execution

| Goal | Command |
|---|---|
| Start services (warehouse DB, Airflow and its metadata DB) | `docker compose up -d --build` |
| Check services | `docker compose ps` (all three must show `healthy`) |
| Run the DAG (UI) | http://localhost:8081 → `predictive_repro_pipeline` → **Trigger** (keep the defaults) |
| Run the DAG (CLI) | `docker compose exec airflow airflow dags trigger predictive_repro_pipeline` |
| Run without Airflow (container) | `docker compose run --rm pipeline run-all` |
| Run without Airflow (local venv) | `python -m src.cli run-all` |
| Run one stage | `python -m src.cli --run-id <id> <stage>`: `ingest`, `validate-raw`, `transform-staging`, `build-curated`, `partition`, `load-postgres`, `validate-curated`, `train`, `evaluate`, `reproducibility-check`, `publish` |
| Leakage-free variant | add `--impute-scope train_only` (results go to `outputs/variants/train_only/`) |

**A run is healthy when:**

- every Airflow task is `success`, or `run-all` exits with code 0;
- `outputs/run_summary.json` has `"status": "success"` and `"reproducibility_outcome": "matches_within_tolerance"`;
- `outputs/quality/{raw,staging,curated}.json` all have `"status": "passed"`;
- `python -m src.cli db-stats` shows `duplicate_business_keys: 0` for every table.

Expected counts for source SHA-256 `2b2202a0…72f10`: raw 54,255; staging 30,545; quarantine 23,710;
station-days 11,558; model-ready 2,628 (1,513 Unhealthy / 1,115 Safe); train 2,102; test 526;
predictions 1,578 (526 × 3 models).

## 2. Where to look

| Question | Location |
|---|---|
| What happened in run X? | `outputs/runs/<run>/run_record.json` (status, attempt, duration, row counts, error per stage); `audit.v_run_overview` in PostgreSQL |
| Detailed logs | `logs/pipeline.jsonl`: one JSON object per event with `pipeline_run_id`, `stage`, counts, `duration_s`, `error`. Example: `grep '"stage_failed"' logs/pipeline.jsonl` |
| Airflow task log | UI → run → task → Logs, or `docker compose logs airflow` |
| Which rows were rejected and why | `data/quarantine/staging_rejects.parquet` (`reject_type`, `reject_reason`, `check_id`) |
| Data-quality results | `outputs/quality/<layer>.json` (latest) and `outputs/runs/<run>/quality/` |
| Load statistics | `SELECT * FROM audit.load_log ORDER BY load_id DESC;` |

Connect to the warehouse: `docker compose exec postgres psql -U repro -d air_quality`.

## 3. Common failures and what to do

| Symptom (log `error_type` / message) | Cause | Action |
|---|---|---|
| `SourceIdentityError: ... sha256 ... != manifest` | Source file changed, re-saved by a spreadsheet tool, or line endings converted by git | Restore the original file (`git checkout -- data/source/world_air_quality.csv`) or download it again. Never "fix" the manifest to match an unknown file. |
| `SourceIdentityError: Source dataset not found` | File missing | Place it as described in `data/source/README.md`. |
| `ContractViolation: SCH-01 schema mismatch` | Columns renamed, removed or reordered upstream | Treat it as a new source version: update the contract and transformations deliberately, then run with `--mode full`. |
| `CheckpointMismatch` | Source version or `ingestion.num_batches` changed since the last run | `python -m src.cli ingest --mode full` (rebuilds every batch). |
| `ContractViolation: Raw layer incomplete: k/5 batches` | An earlier ingest was interrupted | `python -m src.cli ingest`; it resumes at the first uncommitted batch. |
| `DataQualityError ... INV-01` | More than 1% of in-scope rows are unparseable | Inspect `data/quarantine/staging_rejects.parquet`. Staging was **not** updated, so nothing downstream changed. Fix the source, then rerun. |
| `DataQualityError ... APP-01` | Layer counts differ from the expected counts for this source version | A transformation or source change altered the data. Investigate before retraining. |
| `DatabaseUnavailable` | Warehouse container down or wrong host/port | `docker compose ps`; `docker compose start postgres`; locally use `POSTGRES_HOST=127.0.0.1`, `POSTGRES_PORT=5434`. Then clear or rerun the task. |
| Each DB step takes about 10 s on Windows | `POSTGRES_HOST=localhost` resolves to IPv6 first | Use `127.0.0.1` in `.env`. |
| `ReproducibilityError: REP-01 ...` | A metric or the split differs from the baseline beyond 0.0005 | See section 6 below. Outputs were written but **not published**. |
| `PredictiveIntegrityError: PRD-0x` | Leakage column in the features, train/test overlap, or wrong feature order | Fix `config/settings.yml` `model.features` / `leakage_columns`. |
| `CONFIG ERROR: Invalid configuration ...` (exit code 2) | Invalid value in `settings.yml` | The message names the key; nothing was executed. |
| `PermissionError [WinError 5]` on a rename | Antivirus or indexer holding a new file (Windows) | Retried automatically (10 attempts). If it persists, exclude the repository folder from real-time scanning. |
| Port already allocated (5434 / 8081) | Another project uses the port | Change `POSTGRES_HOST_PORT` / `AIRFLOW_HOST_PORT` in `.env`, then `docker compose up -d`. |

Exit codes: `0` success, `1` failed stage, `2` invalid configuration, `3` simulated interruption.

## 4. Restart and rerun procedures

Every stage is **rerun-safe**, so re-running a stage that already succeeded does not create duplicates.

- **Ingest** skips committed batches (`batch_skipped_already_committed` in the log).
- **Staging, curated, partitions and model outputs** are rebuilt and replace their files atomically
  (written to a temp file, then renamed).
- **The PostgreSQL load** is one transaction with a `row_hash` upsert and a snapshot delete. Rerunning it inserts, updates
  and deletes 0 rows.
- **Publishing predictions** is an upsert on `(station_day_key, model_name)`.

**Airflow:**
- Retry a failed run: UI → failed task → **Clear** (with downstream).
- Script: `python scripts/airflow_demo.py db-outage` shows the full procedure via the REST API.
- The run keeps its `run_id`, so its run record shows the failed attempt and the successful retry.

**CLI:** rerun from the failed stage with the same `--run-id`, or rerun `run-all` (rerun-safe).

## 5. Checkpoint recovery (raw layer)

The checkpoint is `data/raw/_state/ingest_checkpoint.json`; `checkpoint_history.jsonl` in the same
folder keeps every change.

- **After an interruption:** run `python -m src.cli ingest`. Committed batches are verified by SHA-256 and skipped.
- **If a batch file is deleted or altered:** the next ingest recommits that batch, and `validate-raw`
  (check INT-01) reports any tampering.
- **To reset the raw layer completely:** `python -m src.cli ingest --mode full`. This deletes the batches and checkpoint, makes a
  fresh run-specific copy and rebuilds all batches.
- **Demonstration:** `python scripts/demo_ingest_recovery.py` writes `docs/evidence/ingestion_recovery_demo.txt`.

## 6. When the reproducibility check fails

Work through these in order. Each step isolates one cause (lab question 12).

1. **Data:** `python -m src.cli verify-source`; check `outputs/quality/*.json` counts (APP-01).
2. **Preprocessing:** compare `data/curated/model_ready.parquet` with the notebook's frame:
   `python baseline/replay_notebook.py` re-runs the original notebook code.
3. **Split:** look at the `split` section of `outputs/reproducibility_comparison.json`. Equal key
   checksums mean the same rows are in the same split.
4. **Dependencies:** compare `outputs/model/model_card.json` → `environment` with `requirements.txt`.
5. **Parameters:** compare `model_card.json` → `models.*.params` with `config/settings.yml`.
6. **Non-determinism:** run twice and compare. Fitting is single-threaded and seeded, so identical
   inputs must give identical metrics.

## 7. Database reset

- **Warehouse only** (data and audit history are lost):
  ```
  docker compose stop postgres
  docker compose rm -f postgres
  docker volume rm dss150p_lab04_valerio_jennapatricia_pgdata
  docker compose up -d postgres
  python -m src.cli init-db
  ```
- **Rebuild the warehouse contents** without dropping the volume: `python -m src.cli load-postgres`.
  It resynchronises the tables with the current Parquet snapshots.
- **Airflow history:** `docker volume rm dss150p_lab04_valerio_jennapatricia_airflow-db-data`
  (after `docker compose down`).

## 8. Rollback and cleanup

- **Generated layers are disposable:** delete `data/raw/*`, `data/staging/*`, `data/curated/*`,
  `data/partitioned/*`, `data/quarantine/*`, `outputs/runs/` and `logs/`, then run `run-all`. The `.gitkeep` files keep the folders.
- **Roll back code or configuration:** `git checkout <commit>`, then `docker compose build`, then run
  with `--mode full`. Committed `outputs/*.json` show what each version produced.
- **Stop everything:** `docker compose down` (keeps volumes) or `docker compose down -v` (also deletes the
  database volumes).
- **Failure drills** write to `outputs/failure_drills/` (git-ignored); it is safe to delete.
