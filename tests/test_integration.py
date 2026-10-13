"""Integration tests: several modules plus file and database boundaries."""
from __future__ import annotations

import os
import uuid

import pandas as pd
import pytest

from src.ingest import SimulatedInterruption, ingest, load_checkpoint, raw_summary
from src.transform import build_curated, build_staging, model_ready_path, station_day_path
from src.validate import validate_curated, validate_raw


def test_ingest_interrupt_resume_and_rerun_never_duplicate(mini_settings):
    from tests.conftest import synthetic_rows

    total = len(synthetic_rows())
    with pytest.raises(SimulatedInterruption):
        ingest(mini_settings, "run-1", mode="full", stop_after_batches=1)
    cp = load_checkpoint(mini_settings)
    assert cp["last_committed_batch_id"] == 1 and len(cp["committed_batches"]) == 1

    second = ingest(mini_settings, "run-2", mode="incremental")
    assert second["batches_skipped_already_committed"] == [1]
    assert second["batches_committed_this_run"] == [2, 3]

    third = ingest(mini_settings, "run-3", mode="incremental")
    assert third["batches_committed_this_run"] == [] and third["batches_skipped_already_committed"] == [1, 2, 3]

    stats = raw_summary(mini_settings)
    assert stats["raw_rows"] == total and stats["distinct_source_row_keys"] == total
    assert stats["duplicate_source_row_keys"] == 0
    batch_runs = {b["batch_id"]: b["pipeline_run_id"] for b in load_checkpoint(mini_settings)["committed_batches"].values()}
    assert batch_runs == {1: "run-1", 2: "run-2", 3: "run-2"}   # lineage: which run committed each batch


def test_file_pipeline_end_to_end_is_valid_and_deterministic(mini_settings):
    ingest(mini_settings, "e2e-1", mode="full")
    assert validate_raw(mini_settings, "e2e-1")["status"] == "passed"
    funnel = build_staging(mini_settings, "e2e-1")
    # 63 rows - 1 out-of-scope (NO) - 1 duplicate - 1 ppm - 1 negative = 59
    assert funnel["reconciles"] and funnel["staging_rows"] == 59
    result = build_curated(mini_settings, "e2e-1")
    assert result["model_ready_rows"] == 13          # 12 stations + Santiago Centro (Nowhere lacks NO2/O3)
    report = validate_curated(mini_settings, "e2e-1", include_database=False)
    failed = [c["check_id"] for c in report["checks"] if c["status"] == "fail"]
    assert failed == []

    audit = ["pipeline_run_id", "processed_at_utc"]
    first = pd.read_parquet(model_ready_path(mini_settings)).drop(columns=audit)
    first_sd = pd.read_parquet(station_day_path(mini_settings)).drop(columns=audit)
    build_staging(mini_settings, "e2e-2")
    build_curated(mini_settings, "e2e-2")
    assert first.equals(pd.read_parquet(model_ready_path(mini_settings)).drop(columns=audit))
    assert first_sd.equals(pd.read_parquet(station_day_path(mini_settings)).drop(columns=audit))


def _database_available(settings) -> bool:
    try:
        import psycopg

        with psycopg.connect(**{**settings.database().connect_kwargs(), "connect_timeout": 3}):
            return True
    except Exception:  # noqa: BLE001
        return False


@pytest.fixture
def scratch_database(mini_settings, monkeypatch):
    """A throw-away database so the test never touches the real warehouse tables."""
    import psycopg

    if not _database_available(mini_settings):
        pytest.skip("PostgreSQL not reachable (start it with `docker compose up -d postgres`)")
    admin = mini_settings.database().connect_kwargs()
    name = f"repro_test_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(**admin, autocommit=True) as conn:
        conn.execute(f'CREATE DATABASE "{name}"')
    monkeypatch.setenv("POSTGRES_DB", name)
    yield name
    with psycopg.connect(**admin, autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


def test_postgres_load_is_rerun_safe_and_matches_parquet(mini_settings, scratch_database):
    from src.storage import database_consistency_check, db_stats, load_postgres

    ingest(mini_settings, "pg-1", mode="full")
    build_staging(mini_settings, "pg-1")
    build_curated(mini_settings, "pg-1")
    first = {r["table"]: r for r in load_postgres(mini_settings, "pg-1")["loads"]}
    assert first["curated.model_ready"]["rows_inserted"] == 13

    # Simulate an Airflow retry after a commit whose acknowledgement was lost: same snapshot again.
    build_staging(mini_settings, "pg-2")
    build_curated(mini_settings, "pg-2")
    second = load_postgres(mini_settings, "pg-2")["loads"]
    assert all(r["rows_inserted"] == r["rows_updated"] == r["rows_deleted"] == 0 for r in second)
    assert all(r["rows_unchanged"] == r["snapshot_rows"] == r["table_rows_after"] for r in second)

    assert all(t["duplicate_business_keys"] == 0 for t in db_stats(mini_settings)["duplicate_business_keys"])
    assert database_consistency_check(mini_settings).passed
    assert os.environ["POSTGRES_DB"] == scratch_database
