"""Replay the original group notebook's modelling logic and capture its split.

The original notebook never saved which rows were in its training and test sets. This script runs
the notebook's own code (cells 1, 3, 5, 9, 11 and 14 of
baseline/original/Predictive_Final_Project_code.ipynb, verbatim apart from removed print/plot
statements) and records:
  * the metrics it produces (to confirm the replay matches the saved notebook outputs), and
  * checksums of the station-day keys in its train and test sets.

Those checksums become metadata/split_manifest.json, the reference the engineered pipeline's split
must match (check PRD-04). The script does not use any pipeline code except the key-hash helper,
so it is an independent witness of the baseline.

Usage:  python baseline/replay_notebook.py [--write-manifest]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.io_utils import digest_of_keys, hash_fields  # noqa: E402  (key definition shared with the pipeline)


def replay(source: Path) -> dict:
    # ---- cell 1 -------------------------------------------------------------------------------
    df = pd.read_csv(source, sep=';')
    # ---- cell 3 -------------------------------------------------------------------------------
    target_indicators = ['PM2.5', 'PM10', 'NO2', 'O3', 'SO2', 'CO']
    df_filtered = df[df['Pollutant'].isin(target_indicators)].copy()
    df_filtered['Last Updated'] = pd.to_datetime(df_filtered['Last Updated'], utc=True)
    df_cleaned = df_filtered.drop_duplicates(subset=['Location', 'Pollutant', 'Last Updated'], keep='first').copy()
    df_cleaned = df_cleaned[df_cleaned['Unit'] == 'µg/m³']
    df_cleaned = df_cleaned[df_cleaned['Value'] >= 0]
    df_cleaned['City'] = df_cleaned['City'].fillna('Unknown')
    df_cleaned['Location'] = df_cleaned['Location'].fillna('Unknown')
    df_cleaned['Country Label'] = df_cleaned['Country Label'].fillna('Unknown')
    df_cleaned = df_cleaned.drop(columns=['Coordinates'])
    # ---- cell 5 -------------------------------------------------------------------------------
    df_cleaned['Date'] = df_cleaned['Last Updated'].dt.date
    df_wide = df_cleaned.pivot_table(index=['Country Label', 'City', 'Location', 'Date'], columns='Pollutant',
                                     values='Value', aggfunc='mean').reset_index()
    df_wide = df_wide.sort_values(by=['Location', 'Date'])
    # ---- cell 9 -------------------------------------------------------------------------------
    df_model_ready = df_wide.dropna(subset=['PM2.5', 'NO2', 'O3']).copy()
    before_imputation = df_model_ready.copy()  # (not in the notebook) kept for --compare-curated
    for col in ['SO2', 'CO']:
        df_model_ready[f'{col}_missing'] = df_model_ready[col].isna().astype(int)
        df_model_ready[col] = df_model_ready[col].fillna(df_model_ready[col].median())
    df_model_ready['Ox_total_oxidant'] = df_model_ready['NO2'] + df_model_ready['O3']
    df_model_ready['NO2_O3_ratio'] = df_model_ready['NO2'] / (df_model_ready['O3'] + 1.0)
    final_features = ['NO2', 'O3', 'SO2', 'CO', 'SO2_missing', 'CO_missing', 'Ox_total_oxidant', 'NO2_O3_ratio']
    # ---- cell 11 ------------------------------------------------------------------------------
    df_model_ready['Air_Quality'] = np.where(df_model_ready['PM2.5'] > 15.0, 'Unhealthy', 'Safe')
    features = final_features
    X = df_model_ready[features]
    y_class = df_model_ready['Air_Quality']
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class, shuffle=True
    )
    log_model = LogisticRegression(max_iter=1000)
    rfc_model = RandomForestClassifier(n_estimators=200, random_state=42)
    gbc_model = GradientBoostingClassifier(n_estimators=200, random_state=42)
    log_model.fit(X_train_c, y_train_c)
    rfc_model.fit(X_train_c, y_train_c)
    gbc_model.fit(X_train_c, y_train_c)
    # ---- cell 14 ------------------------------------------------------------------------------
    trained_models = {
        "logistic_regression": log_model,
        "random_forest": rfc_model,
        "gradient_boosting": gbc_model,
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics = {}
    for name, model in trained_models.items():
        preds = model.predict(X_test_c)
        scores = cross_val_score(model, X, y_class, cv=cv, scoring='f1_weighted')
        metrics[name] = {
            "test_accuracy": round(accuracy_score(y_test_c, preds), 6),
            "test_precision_weighted": round(precision_score(y_test_c, preds, average='weighted', zero_division=0), 6),
            "test_recall_weighted": round(recall_score(y_test_c, preds, average='weighted', zero_division=0), 6),
            "test_f1_weighted": round(f1_score(y_test_c, preds, average='weighted', zero_division=0), 6),
            "cv_f1_weighted_mean": round(scores.mean(), 6),
            "cv_f1_weighted_std": round(scores.std(), 6),
        }

    # ---- split capture (not in the notebook) --------------------------------------------------
    def keys(frame: pd.DataFrame) -> list[str]:
        return [hash_fields([c, ci, loc, d.isoformat()]) for c, ci, loc, d in
                frame[['Country Label', 'City', 'Location', 'Date']].itertuples(index=False, name=None)]

    all_keys = keys(df_model_ready)
    train_keys = keys(df_model_ready.loc[X_train_c.index])
    test_keys = keys(df_model_ready.loc[X_test_c.index])
    return {
        "before_imputation": before_imputation.reset_index(drop=True),
        "metrics": metrics,
        "split": {
            "method": "sklearn.model_selection.train_test_split",
            "stratified": True,
            "shuffle": True,
            "test_size": 0.2,
            "random_state": 42,
            "row_order": "notebook order: pivot_table index order, then sort_values(['Location', 'Date'])",
            "identifier": "station_day_key = sha256(country_label, city, location, reading_date)",
            "model_ready_rows": len(all_keys),
            "train_rows": len(train_keys),
            "test_rows": len(test_keys),
            "train_class_counts": {k: int(v) for k, v in y_train_c.value_counts().sort_index().items()},
            "test_class_counts": {k: int(v) for k, v in y_test_c.value_counts().sort_index().items()},
            "model_ready_key_checksum": digest_of_keys(all_keys),
            "train_key_checksum": digest_of_keys(train_keys),
            "test_key_checksum": digest_of_keys(test_keys),
        },
    }


def compare_curated(notebook_frame: pd.DataFrame, curated_path: Path) -> dict:
    """Is the pipeline's curated model-ready table identical to the notebook's frame (order + values)?"""
    ours = pd.read_parquet(curated_path)
    same_rows = len(ours) == len(notebook_frame)
    keys_equal = same_rows and all(
        (notebook_frame[a].astype(str).values == ours[b].astype(str).values).all()
        for a, b in [('Country Label', 'country_label'), ('City', 'city'), ('Location', 'location'),
                     ('Date', 'reading_date')])
    values = {b: bool(same_rows and np.array_equal(notebook_frame[a].to_numpy(), ours[b].to_numpy(), equal_nan=True))
              for a, b in [('PM2.5', 'pm25'), ('PM10', 'pm10'), ('NO2', 'no2'), ('O3', 'o3'), ('SO2', 'so2'), ('CO', 'co')]}
    return {"rows": [len(notebook_frame), len(ours)], "same_row_order_and_keys": bool(keys_equal),
            "bit_identical_values": values, "identical": bool(keys_equal and all(values.values()))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=str(ROOT / "data" / "source" / "world_air_quality.csv"))
    parser.add_argument("--write-manifest", action="store_true", help="write metadata/split_manifest.json")
    parser.add_argument("--compare-curated", action="store_true",
                        help="compare data/curated/model_ready.parquet with the notebook's pre-imputation frame")
    args = parser.parse_args()
    result = replay(Path(args.source))
    notebook_frame = result.pop("before_imputation")
    if args.compare_curated:
        result["curated_comparison"] = compare_curated(notebook_frame, ROOT / "data" / "curated" / "model_ready.parquet")
    expected = json.loads((ROOT / "metadata" / "expected_metrics.json").read_text(encoding="utf-8"))
    mismatches = [f"{m}.{k}: {v} != {expected['models'][m][k]}"
                  for m, vals in result["metrics"].items() for k, v in vals.items()
                  if abs(v - expected["models"][m][k]) > 5e-7]
    result["matches_saved_notebook_outputs"] = not mismatches
    result["mismatches"] = mismatches
    print(json.dumps(result, indent=2))
    if args.write_manifest:
        if mismatches:
            print("Refusing to write the split manifest: replay does not match the notebook outputs.", file=sys.stderr)
            return 1
        manifest = {
            "description": "Baseline train/test split of the original notebook, captured by baseline/replay_notebook.py. "
                           "The pipeline's split must reproduce these key checksums (check PRD-04).",
            **result["split"],
        }
        path = ROOT / "metadata" / "split_manifest.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"Wrote {path.relative_to(ROOT)}", file=sys.stderr)
    return 0 if not mismatches else 1


if __name__ == "__main__":
    sys.exit(main())
