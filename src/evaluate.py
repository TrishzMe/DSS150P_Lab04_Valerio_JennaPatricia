"""Evaluation, reproducibility check and publication of model outputs.

evaluate              -> metrics.json, predictions.csv (held-out test set), using exactly the
                         notebook's metric definitions and 5-fold stratified CV.
reproducibility_check -> reproducibility_comparison.json: every baseline metric side by side with
                         the reproduced value, absolute difference and tolerance verdict; split
                         checksums compared with metadata/split_manifest.json. Fails the run (exit
                         code != 0) when the baseline-scope reproduction is outside tolerance.
publish               -> curated.predictions / curated.model_metrics in PostgreSQL, published CSV,
                         run summary; the run record is closed with status "success".
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from .config import Settings
from .io_utils import atomic_write_csv, atomic_write_json, read_json
from .observability import RunRecord, StageContext, log_event, utc_now_iso
from .source import load_manifest
from .train import environment_info, load_models, output_dir, prepare_model_input, split_manifest

METRIC_KEYS = ["test_accuracy", "test_precision_weighted", "test_recall_weighted", "test_f1_weighted",
               "cv_f1_weighted_mean", "cv_f1_weighted_std"]


class ReproducibilityError(RuntimeError):
    """Reproduced metrics or split differ from the baseline beyond the declared tolerance."""


def evaluate(settings: Settings, run_id: str, ctx: StageContext | None = None) -> dict[str, Any]:
    m = settings["model"]
    mi = prepare_model_input(settings)
    out = output_dir(settings)
    trained_split = read_json(out / "split_manifest.json")
    current_split = split_manifest(settings, mi)
    for key in ("train_key_checksum", "test_key_checksum", "model_ready_key_checksum"):
        if trained_split[key] != current_split[key]:
            raise ReproducibilityError(f"Curated data changed between train and evaluate ({key} differs); retrain.")

    models = load_models(settings)
    X_test, y_test = mi.X.iloc[mi.test_idx], mi.y.iloc[mi.test_idx]
    labels = [m["negative_label"], m["positive_label"]]
    cv_cfg = m["cv"]
    cv = StratifiedKFold(n_splits=cv_cfg["n_splits"], shuffle=cv_cfg["shuffle"], random_state=cv_cfg["random_state"])

    results: dict[str, Any] = {}
    prediction_parts = []
    with threadpool_limits(limits=1):
        for name, bundle in models.items():
            est = bundle["estimator"]
            pred = est.predict(X_test)
            proba = est.predict_proba(X_test)[:, list(est.classes_).index(m["positive_label"])]
            cm = confusion_matrix(y_test, pred, labels=labels)
            if mi.impute_scope == "baseline":
                # Original notebook: CV over the already-imputed full matrix.
                scores = cross_val_score(clone(est), mi.X, mi.y, cv=cv, scoring=cv_cfg["scoring"])
            else:
                # Leakage-free: the imputer is re-fitted inside every training fold.
                pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("model", clone(est))])
                scores = cross_val_score(pipe, mi.X_raw, mi.y, cv=cv, scoring=cv_cfg["scoring"])
            results[name] = {
                "display_name": m["models"][name]["display_name"],
                "test_accuracy": float(accuracy_score(y_test, pred)),
                "test_precision_weighted": float(precision_score(y_test, pred, average="weighted", zero_division=0)),
                "test_recall_weighted": float(recall_score(y_test, pred, average="weighted", zero_division=0)),
                "test_f1_weighted": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
                "cv_f1_weighted_mean": float(scores.mean()),
                "cv_f1_weighted_std": float(scores.std()),
                "cv_fold_scores": [float(s) for s in scores],
                "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
                "test_true_safe": int(cm[0][0]),
            }
            prediction_parts.append(pd.DataFrame({
                "station_day_key": mi.keys.iloc[mi.test_idx].values,
                "country_label": mi.frame["country_label"].iloc[mi.test_idx].values,
                "location": mi.frame["location"].iloc[mi.test_idx].values,
                "reading_date": mi.frame["reading_date"].iloc[mi.test_idx].values,
                "model_name": name,
                "split": "test",
                "actual": y_test.values,
                "predicted": pred,
                "probability_unhealthy": proba,
            }))

    best = max(results, key=lambda n: results[n][m["selection_metric"]])
    rf = models.get("random_forest")
    importances = ({f: float(v) for f, v in zip(rf["features"], rf["estimator"].feature_importances_)}
                   if rf else {})
    metrics = {
        "pipeline_run_id": run_id,
        "evaluated_at_utc": utc_now_iso(),
        "impute_scope": mi.impute_scope,
        "imputation_medians": mi.medians,
        "source_sha256": load_manifest(settings)["file"]["sha256"],
        "row_counts": {"model_ready_rows": int(len(mi.frame)), "train_rows": int(len(mi.train_idx)),
                       "test_rows": int(len(mi.test_idx)),
                       "class_counts": {k: int(v) for k, v in mi.y.value_counts().sort_index().items()},
                       "test_safe_rows": int((y_test == m["negative_label"]).sum())},
        "split": {k: current_split[k] for k in ("train_key_checksum", "test_key_checksum", "model_ready_key_checksum")},
        "selection_metric": m["selection_metric"],
        "best_model": best,
        "models": results,
        "random_forest_feature_importances": dict(sorted(importances.items(), key=lambda kv: -kv[1])),
        "environment": environment_info(),
    }
    predictions = pd.concat(prediction_parts, ignore_index=True)
    predictions["is_selected_model"] = predictions["model_name"] == best
    predictions["impute_scope"] = mi.impute_scope
    predictions["pipeline_run_id"] = run_id
    atomic_write_json(out / "metrics.json", metrics)
    atomic_write_csv(out / "predictions.csv", predictions)
    if ctx is not None:
        ctx.counts.update(prediction_rows=int(len(predictions)), test_rows=int(len(mi.test_idx)))
        ctx.details.update(best_model=best, **{f"{n}_test_accuracy": round(r["test_accuracy"], 6)
                                               for n, r in results.items()})
    return {"best_model": best, **{n: {k: round(r[k], 6) for k in METRIC_KEYS} for n, r in results.items()}}


def _verdict(diff: float, tolerance: float) -> bool:
    return abs(diff) <= tolerance + 1e-12


def reproducibility_check(settings: Settings, run_id: str, ctx: StageContext | None = None) -> dict[str, Any]:
    rep = settings["reproducibility"]
    tol = float(rep["tolerance_abs"])
    out = output_dir(settings)
    actual = read_json(out / "metrics.json")
    expected = read_json(settings.rel(rep["expected_metrics_file"]))
    expected_split = read_json(settings.rel(rep["expected_split_file"]))
    enforced = actual["impute_scope"] == "baseline"

    rows = []
    for model, exp in expected["models"].items():
        got = actual["models"].get(model, {})
        for metric in [*METRIC_KEYS, "test_true_safe"]:
            if metric not in exp:
                continue
            a = got.get(metric)
            diff = None if a is None else a - exp[metric]
            exact = metric == "test_true_safe"
            ok = a is not None and (diff == 0 if exact else _verdict(diff, tol))
            rows.append({"model": model, "metric": metric, "expected": exp[metric],
                         "reproduced": None if a is None else (a if exact else round(a, 6)),
                         "abs_diff": None if diff is None else round(abs(diff), 6),
                         "tolerance": 0 if exact else tol, "within_tolerance": ok})
    for feature, exp in expected["random_forest_feature_importances"].items():
        a = actual["random_forest_feature_importances"].get(feature)
        diff = None if a is None else a - exp
        rows.append({"model": "random_forest", "metric": f"feature_importance.{feature}", "expected": exp,
                     "reproduced": None if a is None else round(a, 6),
                     "abs_diff": None if diff is None else round(abs(diff), 6), "tolerance": tol,
                     "within_tolerance": a is not None and _verdict(diff, tol)})

    count_checks = {
        "model_ready_rows": (actual["row_counts"]["model_ready_rows"], expected["row_counts"]["model_ready_rows"]),
        "train_rows": (actual["row_counts"]["train_rows"], expected["row_counts"]["train_rows"]),
        "test_rows": (actual["row_counts"]["test_rows"], expected["row_counts"]["test_rows"]),
        "class_counts": (actual["row_counts"]["class_counts"], expected["row_counts"]["class_counts"]),
        "test_safe_rows": (actual["row_counts"]["test_safe_rows"], expected["row_counts"]["test_safe_rows"]),
        "source_sha256": (actual["source_sha256"], expected["source_sha256"]),
        "best_model": (actual["best_model"], expected["best_model"]),
    }
    split_checks = {k: (actual["split"][k], expected_split[k])
                    for k in ("model_ready_key_checksum", "train_key_checksum", "test_key_checksum")}

    primary_model, primary_metric = rep["primary_metric"].split(".")
    primary = next(r for r in rows if r["model"] == primary_model and r["metric"] == primary_metric)
    metrics_ok = all(r["within_tolerance"] for r in rows)
    counts_ok = all(a == b for a, b in count_checks.values())
    split_ok = all(a == b for a, b in split_checks.values())
    if metrics_ok and counts_ok and split_ok:
        outcome = "matches_within_tolerance"
    elif counts_ok and split_ok and primary["within_tolerance"]:
        outcome = "primary_within_tolerance_secondary_differences"
    else:
        signed = actual["models"][primary_model][primary_metric] - expected["models"][primary_model][primary_metric]
        outcome = "better_than_baseline" if signed > tol else "materially_different"

    comparison = {
        "pipeline_run_id": run_id,
        "generated_at_utc": utc_now_iso(),
        "impute_scope": actual["impute_scope"],
        "enforced": enforced,
        "tolerance_rule": f"|reproduced - expected| <= {tol} for every metric (3 decimal places); counts exact",
        "primary_metric": {"name": rep["primary_metric"], **primary},
        "outcome": outcome,
        "passed": bool(metrics_ok and counts_ok and split_ok),
        "metrics": rows,
        "counts": {k: {"reproduced": a, "expected": b, "equal": a == b} for k, (a, b) in count_checks.items()},
        "split": {k: {"reproduced": a, "expected": b, "equal": a == b} for k, (a, b) in split_checks.items()},
        "max_abs_diff": max((r["abs_diff"] or 0) for r in rows),
    }
    atomic_write_json(out / "reproducibility_comparison.json", comparison)
    log_event("reproducibility_result", pipeline_run_id=run_id, stage="reproducibility_check", outcome=outcome,
              passed=comparison["passed"], enforced=enforced, max_abs_diff=comparison["max_abs_diff"])
    if ctx is not None:
        ctx.counts.update(metrics_compared=len(rows), metrics_within_tolerance=sum(r["within_tolerance"] for r in rows))
        ctx.details.update(outcome=outcome)
    if enforced and not comparison["passed"]:
        failed = [f"{r['model']}.{r['metric']} {r['reproduced']} vs {r['expected']}" for r in rows if not r["within_tolerance"]]
        failed += [f"{k}: {a} vs {b}" for k, (a, b) in {**count_checks, **split_checks}.items() if a != b]
        raise ReproducibilityError(f"REP-01 reproduction outside tolerance ({outcome}): " + "; ".join(failed[:10]))
    return {"outcome": outcome, "passed": comparison["passed"], "enforced": enforced,
            "max_abs_diff": comparison["max_abs_diff"]}


def publish(settings: Settings, run_id: str, ctx: StageContext | None = None) -> dict[str, Any]:
    """Deliver outputs: PostgreSQL tables, published CSV, run summary; close the run record."""
    from .storage import TableSpec, connect, prepare_frame
    from .transform import read_model_ready

    out = output_dir(settings)
    metrics = read_json(out / "metrics.json")
    comparison = read_json(out / "reproducibility_comparison.json")
    predictions = pd.read_csv(out / "predictions.csv")
    expected = {(r["model"], r["metric"]): r for r in comparison["metrics"]}

    pred_cols = ["station_day_key", "model_name", "split", "actual", "predicted", "probability_unhealthy",
                 "is_selected_model", "impute_scope"]
    pred_spec = TableSpec("curated.predictions", "station_day_key", [*pred_cols, "row_hash", "pipeline_run_id"],
                          pred_cols)
    metric_rows = []
    for model, values in metrics["models"].items():
        for metric in [*METRIC_KEYS, "test_true_safe"]:
            ref = expected.get((model, metric), {})
            metric_rows.append((run_id, model, metric, float(values[metric]), ref.get("expected"),
                                ref.get("abs_diff"), ref.get("within_tolerance"), metrics["impute_scope"]))

    with connect(settings) as conn, conn.transaction():
        # Predictions are keyed by (station_day_key, model_name): merge per model so a rerun with the
        # same predictions changes nothing and a model's old rows are replaced, never duplicated.
        loads = []
        for model in sorted(predictions["model_name"].unique()):
            part = predictions[predictions["model_name"] == model]
            conn.execute("DELETE FROM curated.predictions WHERE model_name = %s AND station_day_key <> ALL(%s)",
                         (model, list(part["station_day_key"])))
            loads.append(_merge_predictions(conn, pred_spec, prepare_frame(pred_spec, part), run_id))
        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO curated.model_metrics (pipeline_run_id, model_name, metric_name, metric_value,
                       expected_value, abs_diff, within_tolerance, impute_scope)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (pipeline_run_id, model_name, metric_name) DO UPDATE SET
                       metric_value = EXCLUDED.metric_value, expected_value = EXCLUDED.expected_value,
                       abs_diff = EXCLUDED.abs_diff, within_tolerance = EXCLUDED.within_tolerance""",
                metric_rows,
            )
        pg_predictions = conn.execute("SELECT count(*) FROM curated.predictions").fetchone()[0]

    published_dir = settings.path("outputs_dir") / "published"
    curated = read_model_ready(settings).drop(columns=["pipeline_run_id", "processed_at_utc"])
    if metrics["impute_scope"] == "baseline":
        atomic_write_csv(published_dir / "curated_model_ready.csv", curated)

    summary = write_run_summary(settings, run_id, metrics, comparison, predictions, int(pg_predictions))
    for load in loads:
        log_event("predictions_loaded", pipeline_run_id=run_id, stage="publish_outputs", **load)
    if ctx is not None:
        ctx.counts.update(prediction_rows=int(len(predictions)), postgres_prediction_rows=int(pg_predictions))
        ctx.details.update(prediction_loads=loads)
    return {"loads": loads, "summary": summary}


def write_run_summary(settings: Settings, run_id: str, metrics: dict[str, Any] | None = None,
                      comparison: dict[str, Any] | None = None, predictions: pd.DataFrame | None = None,
                      pg_predictions: int | None = None) -> dict[str, Any]:
    """Run summary (start/end, status, counts per layer) from the run record and model outputs."""
    out = output_dir(settings)
    metrics = metrics or read_json(out / "metrics.json")
    comparison = comparison or read_json(out / "reproducibility_comparison.json")
    if predictions is None:
        predictions = pd.read_csv(out / "predictions.csv")
    record = RunRecord(settings, run_id).load()
    summary = {
        "pipeline_run_id": run_id,
        "status": record["status"],
        "started_at_utc": record["started_at_utc"],
        "ended_at_utc": record["ended_at_utc"],
        "impute_scope": metrics["impute_scope"],
        "counts": {
            "raw_rows": record["counts"].get("raw_rows"),
            "staging_rows": record["counts"].get("staging_rows"),
            "quarantined_rows": record["counts"].get("quarantined_rows"),
            "curated_station_day_rows": record["counts"].get("curated_station_day_rows"),
            "curated_model_ready_rows": record["counts"].get("curated_model_ready_rows"),
            "train_rows": metrics["row_counts"]["train_rows"],
            "test_rows": metrics["row_counts"]["test_rows"],
            "prediction_rows": int(len(predictions)),
            "postgres_prediction_rows": pg_predictions,
        },
        "best_model": metrics["best_model"],
        "reproducibility_outcome": comparison["outcome"],
        "stage_status": {k: v["status"] for k, v in record["stages"].items()},
    }
    atomic_write_json(out / "run_summary.json", summary)
    return summary


def _merge_predictions(conn, spec, frame: pd.DataFrame, run_id: str) -> dict[str, Any]:
    """Upsert one model's predictions keyed by (station_day_key, model_name)."""
    from psycopg import sql

    cols = spec.columns
    tmp = "tmp_predictions"
    conn.execute("DROP TABLE IF EXISTS tmp_predictions")
    conn.execute("CREATE TEMP TABLE tmp_predictions (LIKE curated.predictions INCLUDING DEFAULTS) ON COMMIT DROP")
    from .storage import _records

    with conn.cursor().copy(sql.SQL("COPY {} ({}) FROM STDIN").format(
            sql.Identifier(tmp), sql.SQL(", ").join(sql.Identifier(c) for c in cols))) as copy:
        for row in _records(frame.assign(pipeline_run_id=run_id), cols):
            copy.write_row(row)
    update = sql.SQL(", ").join(sql.SQL("{0} = EXCLUDED.{0}").format(sql.Identifier(c))
                                for c in cols if c not in ("station_day_key", "model_name"))
    flags = conn.execute(sql.SQL(
        "INSERT INTO curated.predictions AS t ({cols}) SELECT {cols} FROM tmp_predictions "
        "ON CONFLICT (station_day_key, model_name) DO UPDATE SET {update}, loaded_at_utc = now() "
        "WHERE t.row_hash IS DISTINCT FROM EXCLUDED.row_hash RETURNING (xmax = 0)").format(
        cols=sql.SQL(", ").join(sql.Identifier(c) for c in cols), update=update)).fetchall()
    inserted = sum(1 for (f,) in flags if f)
    return {"model": frame["model_name"].iloc[0], "snapshot_rows": int(len(frame)), "rows_inserted": inserted,
            "rows_updated": len(flags) - inserted, "rows_unchanged": int(len(frame)) - len(flags)}
