"""
### predictive_repro_pipeline

Reproducible PM2.5 WHO-exceedance pipeline (DSS150P Lab 04).

The DAG only **orchestrates**: every task runs one command of the repository's CLI
(`python -m src.cli <command>`), so all transformation and modelling logic lives in `src/`
and is identical whether it is run by hand, by `run-all`, or by Airflow.

**Run identity:** every task receives the Airflow `run_id` as `PIPELINE_RUN_ID`, so all logs,
files (`outputs/runs/<run>/`), PostgreSQL rows and audit records of one DAG run share one id.

**Schedule:** `schedule=None` (manual trigger). The source is a static, checksum-pinned snapshot,
so there is nothing new to process on a timer; a run is triggered when code, configuration or
the registered source version changes. `catchup=False` so no historical runs are back-filled.

**Parameters:** `ingest_mode` (incremental | full), `impute_scope` (baseline | train_only),
`selected_partition` (pollutant partition loaded and verified by `partition_staging`).
"""
from __future__ import annotations

import logging
import os
from datetime import timedelta

import pendulum
from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG, Param

PROJECT_DIR = os.environ.get("REPRO_PROJECT_ROOT", "/opt/project")
PIPELINE_PYTHON = os.environ.get("PIPELINE_PYTHON", "/opt/pipeline-venv/bin/python")
POLLUTANTS = ["PM2.5", "PM10", "NO2", "O3", "SO2", "CO"]

log = logging.getLogger("predictive_repro_pipeline")


def log_failure(context) -> None:
    """Failure callback: one structured line with everything needed to start debugging."""
    ti = context["task_instance"]
    log.error(
        "TASK FAILED dag_id=%s task_id=%s run_id=%s try_number=%s max_tries=%s exception=%r "
        "-> see outputs/runs/<run>/run_record.json and logs/pipeline.jsonl; recover with "
        "`airflow tasks clear %s -t %s` once the cause is fixed (tasks are rerun-safe).",
        ti.dag_id, ti.task_id, ti.run_id, ti.try_number, ti.max_tries, context.get("exception"),
        ti.dag_id, ti.task_id,
    )


def log_retry(context) -> None:
    ti = context["task_instance"]
    log.warning("TASK RETRY task_id=%s run_id=%s try_number=%s exception=%r",
                ti.task_id, ti.run_id, ti.try_number, context.get("exception"))


def cli_task(task_id: str, command: str, **kwargs) -> BashOperator:
    """BashOperator that runs one pipeline CLI command for this DAG run."""
    return BashOperator(
        task_id=task_id,
        bash_command=(
            f'cd "{PROJECT_DIR}" && "{PIPELINE_PYTHON}" -m src.cli '
            '--run-id "$PIPELINE_RUN_ID" --impute-scope "{{ params.impute_scope }}" '
            f"{command}"
        ),
        env={"PIPELINE_RUN_ID": "{{ run_id }}"},
        append_env=True,  # keep the container's POSTGRES_* settings
        **kwargs,
    )


default_args = {
    "owner": "jenna.valerio",
    "retries": 2,
    "retry_delay": timedelta(seconds=20),
    "on_failure_callback": log_failure,
    "on_retry_callback": log_retry,
}

with DAG(
    dag_id="predictive_repro_pipeline",
    description="Reproducible PM2.5 exceedance pipeline: ingest -> validate -> transform -> load -> train -> verify -> publish",
    doc_md=__doc__,
    schedule=None,
    start_date=pendulum.datetime(2026, 10, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,  # runs share the raw checkpoint and the warehouse tables
    default_args=default_args,
    params={
        "ingest_mode": Param("incremental", enum=["incremental", "full"],
                             description="incremental skips committed batches; full rebuilds raw from a fresh copy"),
        "impute_scope": Param("baseline", enum=["baseline", "train_only"],
                              description="baseline reproduces the original notebook; train_only is leakage-free"),
        "selected_partition": Param("PM2.5", enum=POLLUTANTS,
                                    description="partition loaded on its own and verified"),
    },
    tags=["dss150p", "lab04", "reproducibility"],
) as dag:
    # Safe to retry: committed batches are skipped, the checkpoint advances only after validation.
    ingest_source = cli_task("ingest_source", 'ingest --mode "{{ params.ingest_mode }}"',
                             execution_timeout=timedelta(minutes=10))
    # Deterministic checks: a retry cannot change the answer, so fail immediately.
    validate_raw = cli_task("validate_raw", "validate-raw", retries=0)
    transform_staging = cli_task("transform_staging", "transform-staging", execution_timeout=timedelta(minutes=10))
    build_curated = cli_task("build_curated", "build-curated", execution_timeout=timedelta(minutes=10))
    partition_staging = cli_task("partition_staging", 'partition --select "{{ params.selected_partition }}"')
    # Retry-safe: one transaction + row_hash upsert; a retry after a lost commit ack is a no-op.
    load_postgres = cli_task("load_postgres", "load-postgres", retries=3, retry_delay=timedelta(seconds=30),
                             retry_exponential_backoff=True, max_retry_delay=timedelta(minutes=5),
                             execution_timeout=timedelta(minutes=10))
    validate_curated = cli_task("validate_curated", "validate-curated", retries=1)
    train_model = cli_task("train_model", "train", execution_timeout=timedelta(minutes=15))
    evaluate_model = cli_task("evaluate_model", "evaluate", execution_timeout=timedelta(minutes=20))
    reproducibility_check = cli_task("reproducibility_check", "reproducibility-check", retries=0)
    publish_outputs = cli_task("publish_outputs", "publish", retries=3, retry_delay=timedelta(seconds=30),
                               retry_exponential_backoff=True)

    ingest_source >> validate_raw >> transform_staging >> build_curated >> load_postgres >> validate_curated
    transform_staging >> partition_staging
    validate_curated >> train_model >> evaluate_model >> reproducibility_check
    [reproducibility_check, partition_staging] >> publish_outputs
