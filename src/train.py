"""Model input preparation and training (curated -> model input -> fitted models).

Model input = deterministic matrices built from data/curated/model_ready.parquet:
  * features in the contract order, target from the curated label;
  * deterministic stratified split (train_test_split, seed from config) over the curated row
    order, which is the same row order as the original notebook;
  * median imputation of SO2/CO, fitted either on all model-ready rows (``impute_scope:
    baseline``, the original notebook's behaviour) or on the training rows only
    (``train_only``, the leakage-free variant).

Outputs (baseline scope -> outputs/, other scopes -> outputs/variants/<scope>/):
  model/<model>.joblib, model/model_card.json, split_manifest.json, split_assignment.csv
"""
from __future__ import annotations

import importlib
import os
import platform
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from .config import Settings
from .io_utils import atomic_write_csv, atomic_write_json, digest_of_keys, read_json, sha256_file
from .observability import StageContext, utc_now_iso
from .source import load_manifest
from .transform import read_model_ready


class PredictiveIntegrityError(RuntimeError):
    """Leakage, overlapping splits or unstable feature layout detected."""


@dataclass
class ModelInput:
    frame: pd.DataFrame          # curated model-ready rows (curated order)
    X: pd.DataFrame              # imputed feature matrix, contract column order
    X_raw: pd.DataFrame          # features before imputation (NaN in imputed columns)
    y: pd.Series
    train_idx: np.ndarray
    test_idx: np.ndarray
    medians: dict[str, float]
    impute_scope: str

    @property
    def keys(self) -> pd.Series:
        return self.frame["station_day_key"]


def output_dir(settings: Settings) -> Path:
    scope = settings["model"]["impute_scope"]
    base = settings.path("outputs_dir")
    return base if scope == "baseline" else base / "variants" / scope


def set_global_seeds(seed: int) -> None:
    """Seed every random source the project uses (estimators also get explicit random_state)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def build_estimator(spec: dict[str, Any]):
    module_name, _, class_name = spec["estimator"].rpartition(".")
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls(**spec.get("params", {}))


def integrity_checks(settings: Settings, frame: pd.DataFrame, features: list[str]) -> None:
    m = settings["model"]
    leaking = sorted(set(features) & set(m["leakage_columns"]))
    if leaking:  # PRD-01
        raise PredictiveIntegrityError(f"PRD-01: features include target/leakage columns {leaking}")
    missing = [f for f in features if f not in frame.columns]
    if missing:  # PRD-03
        raise PredictiveIntegrityError(f"PRD-03: curated data lacks feature columns {missing}")


def prepare_model_input(settings: Settings) -> ModelInput:
    m = settings["model"]
    set_global_seeds(int(m["global_seed"]))
    frame = read_model_ready(settings).reset_index(drop=True)
    features = list(m["features"])
    integrity_checks(settings, frame, features)
    X_raw = frame[features].copy()
    y = frame[m["target_column"]].astype(str)

    split = m["split"]
    positions = np.arange(len(frame))
    train_idx, test_idx = train_test_split(
        positions,
        test_size=split["test_size"],
        random_state=split["random_state"],
        stratify=y if split["method"] == "stratified_random" else None,
        shuffle=split["shuffle"],
    )
    if set(frame["station_day_key"].iloc[train_idx]) & set(frame["station_day_key"].iloc[test_idx]):
        raise PredictiveIntegrityError("PRD-02: train and test share station_day_key values")

    scope = m["impute_scope"]
    fit_rows = X_raw if scope == "baseline" else X_raw.iloc[train_idx]
    medians = {c: float(fit_rows[c].median()) for c in m["imputed_columns"]}
    X = X_raw.copy()
    for col, value in medians.items():
        X[col] = X[col].fillna(value)
    if list(X.columns) != features:  # PRD-03: column order is part of the model contract
        raise PredictiveIntegrityError(f"PRD-03: feature order {list(X.columns)} != {features}")
    return ModelInput(frame, X, X_raw, y, train_idx, test_idx, medians, scope)


def split_manifest(settings: Settings, mi: ModelInput) -> dict[str, Any]:
    split = settings["model"]["split"]
    keys = mi.keys
    y_train, y_test = mi.y.iloc[mi.train_idx], mi.y.iloc[mi.test_idx]
    return {
        "method": "sklearn.model_selection.train_test_split",
        "stratified": split["method"] == "stratified_random",
        "shuffle": split["shuffle"],
        "test_size": split["test_size"],
        "random_state": split["random_state"],
        "row_order": "curated order: location, reading_date, country_label, city (same as the original notebook)",
        "identifier": "station_day_key = sha256(country_label, city, location, reading_date)",
        "model_ready_rows": int(len(keys)),
        "train_rows": int(len(mi.train_idx)),
        "test_rows": int(len(mi.test_idx)),
        "train_class_counts": {k: int(v) for k, v in y_train.value_counts().sort_index().items()},
        "test_class_counts": {k: int(v) for k, v in y_test.value_counts().sort_index().items()},
        "model_ready_key_checksum": digest_of_keys(keys),
        "train_key_checksum": digest_of_keys(keys.iloc[mi.train_idx]),
        "test_key_checksum": digest_of_keys(keys.iloc[mi.test_idx]),
    }


def environment_info() -> dict[str, str]:
    import scipy

    return {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "scipy": scipy.__version__}


def train(settings: Settings, run_id: str, ctx: StageContext | None = None) -> dict[str, Any]:
    m = settings["model"]
    mi = prepare_model_input(settings)
    out = output_dir(settings)
    model_dir = out / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = mi.X.iloc[mi.train_idx], mi.y.iloc[mi.train_idx]
    artifacts = {}
    # Single-threaded BLAS/OpenMP so floating-point reduction order cannot vary between machines.
    with threadpool_limits(limits=1):
        for name, spec in m["models"].items():
            estimator = build_estimator(spec)
            estimator.fit(X_train, y_train)
            path = model_dir / f"{name}.joblib"
            joblib.dump({"estimator": estimator, "features": list(mi.X.columns), "medians": mi.medians,
                         "impute_scope": mi.impute_scope, "pipeline_run_id": run_id}, path)
            artifacts[name] = {"file": path.relative_to(settings.root).as_posix(), "sha256": sha256_file(path),
                               "estimator": spec["estimator"], "params": estimator.get_params()}

    manifest = split_manifest(settings, mi)
    atomic_write_json(out / "split_manifest.json", {"pipeline_run_id": run_id, **manifest})
    assignment = pd.DataFrame({"station_day_key": mi.keys, "split": "train"})
    assignment.loc[mi.test_idx, "split"] = "test"
    atomic_write_csv(out / "split_assignment.csv", assignment)

    card = {
        "pipeline_run_id": run_id,
        "trained_at_utc": utc_now_iso(),
        "task": "binary classification: air_quality (Unhealthy if station-day PM2.5 > 15 µg/m³)",
        "training_data": {"file": "data/curated/model_ready.parquet",
                          "source_sha256": load_manifest(settings)["file"]["sha256"],
                          "model_ready_key_checksum": manifest["model_ready_key_checksum"],
                          "train_key_checksum": manifest["train_key_checksum"]},
        "features": list(mi.X.columns),
        "target": m["target_column"],
        "impute_scope": mi.impute_scope,
        "imputation_medians": mi.medians,
        "seeds": {"global_seed": m["global_seed"], "split_random_state": m["split"]["random_state"],
                  "cv_random_state": m["cv"]["random_state"]},
        "threading": "threadpoolctl limits=1 during fit and cross-validation",
        "environment": environment_info(),
        "models": artifacts,
    }
    atomic_write_json(model_dir / "model_card.json", card)
    result = {"train_rows": manifest["train_rows"], "test_rows": manifest["test_rows"],
              "impute_scope": mi.impute_scope, "medians": mi.medians, "models": list(artifacts)}
    if ctx is not None:
        ctx.counts.update(train_rows=manifest["train_rows"], test_rows=manifest["test_rows"])
        ctx.details.update(impute_scope=mi.impute_scope, medians=mi.medians)
    return result


def load_models(settings: Settings) -> dict[str, dict[str, Any]]:
    model_dir = output_dir(settings) / "model"
    card = read_json(model_dir / "model_card.json")
    loaded = {}
    for name, info in card["models"].items():
        path = settings.root / info["file"]
        if sha256_file(path) != info["sha256"]:
            raise PredictiveIntegrityError(f"Model artefact {path} does not match its model card checksum")
        loaded[name] = joblib.load(path)
    return loaded
