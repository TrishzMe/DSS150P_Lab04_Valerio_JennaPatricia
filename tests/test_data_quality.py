"""Data-quality tests: the contract checks must pass on real outputs and must catch bad data."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import REPO_ROOT
from src.validate import curated_checks, staging_checks

REAL_MODEL_READY = REPO_ROOT / "data" / "curated" / "model_ready.parquet"
needs_real_outputs = pytest.mark.skipif(not REAL_MODEL_READY.exists(),
                                        reason="run the pipeline first (python -m src.cli run-all)")


def _failed(results):
    return sorted({r.check_id for r in results if r.status == "fail"})


@needs_real_outputs
def test_real_curated_layer_passes_every_contract_check(repo_settings):
    from src.transform import read_model_ready, read_staging, station_day_path

    station_day = pd.read_parquet(station_day_path(repo_settings))
    results = curated_checks(repo_settings, station_day, read_model_ready(repo_settings),
                             staging_rows=len(read_staging(repo_settings)))
    assert _failed(results) == []
    applicability = next(r for r in results if r.check_id == "APP-01")
    assert applicability.passed, applicability.observed  # 11,558 station-days / 2,628 model-ready rows


@needs_real_outputs
def test_quality_checks_catch_injected_defects(repo_settings):
    """A14: schema, completeness, uniqueness, validity and business-rule violations are all detected."""
    from src.transform import read_model_ready, read_staging, station_day_path

    station_day = pd.read_parquet(station_day_path(repo_settings))
    ready = read_model_ready(repo_settings)
    bad = ready.copy()
    bad.loc[0, "air_quality"] = "Unhealthy" if bad.loc[0, "air_quality"] == "Safe" else "Safe"  # business rule
    bad.loc[1, "no2"] = np.nan                                                                  # completeness
    bad = pd.concat([bad, bad.iloc[[2]]], ignore_index=True)                                     # uniqueness
    bad.loc[3, "so2_missing"] = 7                                                                # validity
    failed = _failed(curated_checks(repo_settings, station_day, bad))
    for check in ("CON-01", "CMP-02", "UNQ-03", "VAL-04", "CON-03"):
        assert check in failed

    staging = read_staging(repo_settings)
    broken = staging.copy()
    broken["value"] = broken["value"].astype(str)                                                # schema
    broken.loc[0, "value"] = "-1"
    broken = pd.concat([broken, broken.iloc[[5]]], ignore_index=True)
    funnel = {"structural_invalid_rate": 0.0, "reconciles": True, "raw_rows": 1, "staging_rows": 1, "rejected_by_type": {}}
    failed = _failed(staging_checks(broken, funnel, repo_settings))
    assert "SCH-02" in failed and "UNQ-02" in failed


@needs_real_outputs
def test_model_rule_no_leakage_and_disjoint_split(repo_settings):
    """Project-specific model rule: no target-derived feature, train/test disjoint, stable column order."""
    from src.train import prepare_model_input

    mi = prepare_model_input(repo_settings)
    assert list(mi.X.columns) == repo_settings["model"]["features"]
    assert not set(mi.X.columns) & {"pm25", "pm10", "air_quality"}
    assert not set(mi.keys.iloc[mi.train_idx]) & set(mi.keys.iloc[mi.test_idx])
    assert len(mi.train_idx) == 2102 and len(mi.test_idx) == 526


def test_reproduced_metrics_within_declared_tolerance():
    """The committed comparison must show every baseline metric within tolerance (A17)."""
    comparison = json.loads((REPO_ROOT / "outputs" / "reproducibility_comparison.json").read_text(encoding="utf-8"))
    assert comparison["impute_scope"] == "baseline" and comparison["enforced"]
    assert comparison["passed"] and comparison["outcome"] == "matches_within_tolerance"
    assert all(r["within_tolerance"] for r in comparison["metrics"])
    assert all(v["equal"] for v in comparison["split"].values())


def test_dag_orchestrates_only_and_has_the_required_structure():
    """The DAG file must not contain business logic and must keep the lab's task order."""
    source = (REPO_ROOT / "dags" / "predictive_repro_pipeline.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {n.names[0].name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import)}
    imported |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert not imported & {"src", "pandas", "sklearn", "numpy", "psycopg"}
    task_ids = {n.args[0].value for n in ast.walk(tree)
                if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "cli_task"}
    assert task_ids == {"ingest_source", "validate_raw", "transform_staging", "build_curated", "partition_staging",
                        "load_postgres", "validate_curated", "train_model", "evaluate_model",
                        "reproducibility_check", "publish_outputs"}
    assert "schedule=None" in source and "catchup=False" in source
    chain = "ingest_source >> validate_raw >> transform_staging >> build_curated >> load_postgres >> validate_curated"
    assert chain in source and "validate_curated >> train_model >> evaluate_model >> reproducibility_check" in source


def test_no_absolute_user_paths_in_executable_code():
    """A3: no machine-specific absolute paths in code or configuration."""
    offenders = []
    for path in [*REPO_ROOT.glob("src/*.py"), *REPO_ROOT.glob("dags/*.py"), REPO_ROOT / "config" / "settings.yml"]:
        text = path.read_text(encoding="utf-8")
        if "C:\\Users" in text or "C:/Users" in text or "/Users/" in text or "Downloads" in text:
            offenders.append(path.name)
    assert offenders == []
