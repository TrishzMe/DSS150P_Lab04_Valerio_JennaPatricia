"""Staging and curated transformations (raw -> staging -> curated).

The steps and their ORDER reproduce the original notebook exactly, because the order changes the
result (deduplication keeps the first reading in source order and runs before the unit filter).
What changed is engineering, not semantics: explicit typing, explicit null tokens, quarantine
with reasons instead of silent drops, audit columns, deterministic ordering and atomic writes.

Staging  (data/staging/air_quality_readings.parquet): one valid reading per
         (location, pollutant, last_updated_utc), typed, in-scope pollutants, µg/m³, value >= 0.
Curated  (data/curated/station_day_pollutants.parquet): station-day mean per pollutant.
         (data/curated/model_ready.parquet): rows with PM2.5, NO2, O3 present, plus the derived
         features and the target label. Imputation is NOT done here: it is a fitted step and
         belongs to the model-input stage (src/train.py).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Settings
from .ingest import read_raw
from .io_utils import atomic_write_parquet, hash_rows
from .observability import StageContext, utc_now_iso
from .source import RAW_COLUMNS, ContractViolation, load_contract

STAGING_NAMES = {
    "Country Code": "country_code",
    "City": "city",
    "Location": "location",
    "Coordinates": "coordinates",
    "Pollutant": "pollutant",
    "Source Name": "source_name",
    "Unit": "unit",
    "Value": "value",
    "Last Updated": "last_updated_utc",
    "Country Label": "country_label",
}
POLLUTANT_COLUMNS = {"PM2.5": "pm25", "PM10": "pm10", "NO2": "no2", "O3": "o3", "SO2": "so2", "CO": "co"}
STATION_DAY_KEY_COLUMNS = ["country_label", "city", "location", "reading_date"]
# Total order on the unique station-day key. Equivalent to the notebook's pivot order followed by
# a stable sort on (Location, Date), but no longer dependent on sort-algorithm tie behaviour.
CURATED_ORDER = ["location", "reading_date", "country_label", "city"]

STAGING_COLUMNS = [
    "source_row_key", "source_batch_id", "record_hash", "country_code", "city", "location", "coordinates",
    "pollutant", "source_name", "unit", "value", "last_updated_utc", "reading_date", "country_label",
    "pipeline_run_id", "processed_at_utc",
]
QUARANTINE_COLUMNS = [
    "source_row_key", "source_batch_id", "record_hash", *RAW_COLUMNS,
    "reject_type", "reject_reason", "check_id", "pipeline_run_id", "rejected_at_utc",
]


def staging_path(settings: Settings) -> Path:
    return settings.path("staging_dir") / "air_quality_readings.parquet"


def quarantine_path(settings: Settings) -> Path:
    return settings.path("quarantine_dir") / "staging_rejects.parquet"


def station_day_path(settings: Settings) -> Path:
    return settings.path("curated_dir") / "station_day_pollutants.parquet"


def model_ready_path(settings: Settings) -> Path:
    return settings.path("curated_dir") / "model_ready.parquet"


class _Rejects:
    """Collects rejected rows with a reason; nothing is dropped silently."""

    def __init__(self, raw: pd.DataFrame, run_id: str):
        self.raw = raw.set_index("source_row_key", drop=False)
        self.run_id = run_id
        self.parts: list[pd.DataFrame] = []
        self.counts: dict[str, int] = {}

    def add(self, keys: pd.Series, reject_type: str, reason: str, check_id: str) -> None:
        keys = keys.astype("int64")
        self.counts[reason] = self.counts.get(reason, 0) + int(len(keys))
        if len(keys) == 0:
            return
        part = self.raw.loc[keys.values, ["source_row_key", "source_batch_id", "record_hash", *RAW_COLUMNS]].copy()
        part["reject_type"] = reject_type
        part["reject_reason"] = reason
        part["check_id"] = check_id
        self.parts.append(part)

    def frame(self) -> pd.DataFrame:
        if not self.parts:
            return pd.DataFrame(columns=QUARANTINE_COLUMNS)
        out = pd.concat(self.parts, ignore_index=True).sort_values("source_row_key", kind="stable")
        out["pipeline_run_id"] = self.run_id
        out["rejected_at_utc"] = utc_now_iso()
        return out[QUARANTINE_COLUMNS].reset_index(drop=True)


def standardize_nulls(frame: pd.DataFrame, columns: list[str], tokens: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        out[col] = out[col].mask(out[col].isin(tokens))
    return out


def clean_readings(raw: pd.DataFrame, settings: Settings, run_id: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Raw (text) -> staging (typed, valid, deduplicated). Returns (staging, quarantine, funnel)."""
    cfg = settings["transform"]
    contract = {c["name"]: c for c in load_contract(settings)["columns"]}
    allowed_pollutants = set(contract["Pollutant"]["allowed_values"])
    allowed_units = set(contract["Unit"]["allowed_values"])
    rejects = _Rejects(raw, run_id)
    funnel: dict[str, Any] = {"raw_rows": int(len(raw))}

    df = raw.sort_values("source_row_key", kind="stable")
    df = standardize_nulls(df, RAW_COLUMNS, settings["source"]["null_tokens"]).rename(columns=STAGING_NAMES)

    # 1. Scope: keep the six pollutants used by the project (others are valid but out of scope).
    null_pollutant = df["pollutant"].isna()
    rejects.add(df.loc[null_pollutant, "source_row_key"], "invalid", "null_pollutant", "CMP-01")
    unknown_pollutant = ~null_pollutant & ~df["pollutant"].isin(allowed_pollutants)
    rejects.add(df.loc[unknown_pollutant, "source_row_key"], "invalid", "unknown_pollutant", "VAL-03")
    in_scope = df["pollutant"].isin(cfg["target_pollutants"])
    rejects.add(df.loc[~in_scope & ~null_pollutant & ~unknown_pollutant, "source_row_key"],
                "excluded", "out_of_scope_pollutant", "SCOPE")
    df = df.loc[in_scope].copy()
    funnel["in_scope_rows"] = int(len(df))

    # 2. Types: value -> float (pandas parser, identical to the notebook's read_csv), timestamp -> UTC.
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["last_updated_utc"] = pd.to_datetime(df["last_updated_utc"], utc=True, errors="coerce")
    bad_value = df["value"].isna()
    bad_ts = df["last_updated_utc"].isna() & ~bad_value
    null_unit = df["unit"].isna() & ~bad_value & ~bad_ts
    rejects.add(df.loc[bad_value, "source_row_key"], "invalid", "null_or_unparseable_value", "SCH-02")
    rejects.add(df.loc[bad_ts, "source_row_key"], "invalid", "null_or_unparseable_timestamp", "SCH-03")
    rejects.add(df.loc[null_unit, "source_row_key"], "invalid", "null_unit", "CMP-01")
    df = df.loc[~(bad_value | bad_ts | null_unit)]
    funnel["typed_rows"] = int(len(df))

    # 3. Duplicate sensor readings: keep the first in source order (notebook: keep='first').
    dup = df.duplicated(subset=cfg["dedup_key"], keep="first")
    rejects.add(df.loc[dup, "source_row_key"], "duplicate", "duplicate_reading", "UNQ-02")
    df = df.loc[~dup]
    funnel["deduplicated_rows"] = int(len(df))

    # 4. Units: only µg/m³ (ppm rows are valid but excluded; unknown units are invalid).
    unknown_unit = ~df["unit"].isin(allowed_units)
    rejects.add(df.loc[unknown_unit, "source_row_key"], "invalid", "unknown_unit", "VAL-03")
    other_unit = ~unknown_unit & (df["unit"] != cfg["accepted_unit"])
    rejects.add(df.loc[other_unit, "source_row_key"], "excluded", "non_ugm3_unit", "VAL-02")
    df = df.loc[df["unit"] == cfg["accepted_unit"]]
    funnel["ugm3_rows"] = int(len(df))

    # 5. Negative concentrations are sensor errors.
    negative = df["value"] < cfg["min_valid_value"]
    rejects.add(df.loc[negative, "source_row_key"], "invalid", "negative_value", "VAL-01")
    df = df.loc[~negative].copy()

    # 6. Missing geography -> placeholder (must happen before grouping; NaN keys would be dropped).
    for col in ("city", "location", "country_label"):
        df[col] = df[col].fillna(cfg["geo_fill_value"])

    df["reading_date"] = df["last_updated_utc"].dt.strftime("%Y-%m-%d")
    df["pipeline_run_id"] = run_id
    df["processed_at_utc"] = utc_now_iso()
    staging = df[STAGING_COLUMNS].reset_index(drop=True)
    quarantine = rejects.frame()
    funnel["staging_rows"] = int(len(staging))
    funnel["rejected"] = dict(sorted(rejects.counts.items()))
    funnel["rejected_by_type"] = {t: int((quarantine["reject_type"] == t).sum())
                                  for t in ("invalid", "excluded", "duplicate")}
    # Every raw row is either in staging or in quarantine with a reason.
    funnel["reconciles"] = funnel["staging_rows"] + len(quarantine) == funnel["raw_rows"]
    return staging, quarantine, funnel


STRUCTURAL_REASONS = ("null_pollutant", "unknown_pollutant", "null_or_unparseable_value",
                      "null_or_unparseable_timestamp", "null_unit", "unknown_unit")


def build_staging(settings: Settings, run_id: str, ctx: StageContext | None = None) -> dict[str, Any]:
    """Build staging; data-quality checks run BEFORE the staging file is replaced."""
    from .validate import finalize_report, staging_checks

    raw = read_raw(settings, require_complete=True)
    staging, quarantine, funnel = clean_readings(raw, settings, run_id)
    structural = sum(v for k, v in funnel["rejected"].items() if k in STRUCTURAL_REASONS)
    funnel["structural_invalid_rows"] = structural
    funnel["structural_invalid_rate"] = round(structural / max(funnel["in_scope_rows"], 1), 6)
    # Quarantine is always written so rejected rows can be inspected, even when the checks fail.
    atomic_write_parquet(quarantine_path(settings), quarantine)
    finalize_report(settings, run_id, "staging", staging_checks(staging, funnel, settings), ctx)
    atomic_write_parquet(staging_path(settings), staging)
    if ctx is not None:
        ctx.counts.update(staging_rows=funnel["staging_rows"], quarantined_rows=int(len(quarantine)),
                          quarantined_invalid_rows=funnel["rejected_by_type"]["invalid"])
        ctx.details.update(funnel=funnel)
    return funnel


def read_staging(settings: Settings) -> pd.DataFrame:
    path = staging_path(settings)
    if not path.exists():
        raise FileNotFoundError(f"Staging data not found at {path}: run transform-staging first.")
    return pd.read_parquet(path)


def pivot_station_day(staging: pd.DataFrame) -> pd.DataFrame:
    """Station-day mean per pollutant (the notebook's pivot_table), deterministically ordered."""
    wide = staging.pivot_table(
        index=STATION_DAY_KEY_COLUMNS, columns="pollutant", values="value", aggfunc="mean"
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(columns=POLLUTANT_COLUMNS)
    for col in POLLUTANT_COLUMNS.values():
        if col not in wide:
            wide[col] = np.nan
    counts = staging.groupby(STATION_DAY_KEY_COLUMNS).size().rename("n_readings").reset_index()
    wide = wide.merge(counts, on=STATION_DAY_KEY_COLUMNS, how="left", validate="one_to_one")
    wide = wide.sort_values(CURATED_ORDER, kind="mergesort").reset_index(drop=True)
    wide.insert(0, "station_day_key", hash_rows(wide, STATION_DAY_KEY_COLUMNS).astype(str).values)
    return wide[["station_day_key", *STATION_DAY_KEY_COLUMNS, *POLLUTANT_COLUMNS.values(), "n_readings"]]


def derive_model_ready(station_day: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Rows with the required pollutants, row-level derived features and the target label."""
    m = settings["model"]
    ready = station_day.dropna(subset=m["required_columns"]).copy()
    for col in m["imputed_columns"]:
        ready[f"{col}_missing"] = ready[col].isna().astype("int64")
    ready["ox_total_oxidant"] = ready["no2"] + ready["o3"]
    ready["no2_o3_ratio"] = ready["no2"] / (ready["o3"] + 1.0)
    ready[m["target_column"]] = np.where(ready[m["target_source_column"]] > m["threshold_ugm3"],
                                         m["positive_label"], m["negative_label"])
    cols = ["station_day_key", *STATION_DAY_KEY_COLUMNS, "pm25", "pm10", "no2", "o3", "so2", "co",
            "so2_missing", "co_missing", "ox_total_oxidant", "no2_o3_ratio", m["target_column"]]
    return ready[cols].reset_index(drop=True)


def build_curated(settings: Settings, run_id: str, ctx: StageContext | None = None) -> dict[str, Any]:
    staging = read_staging(settings)
    station_day = pivot_station_day(staging)
    model_ready = derive_model_ready(station_day, settings)
    if not station_day["station_day_key"].is_unique:
        raise ContractViolation("UNQ-03: station_day_key is not unique")
    stamp = utc_now_iso()
    for frame in (station_day, model_ready):
        frame["pipeline_run_id"] = run_id
        frame["processed_at_utc"] = stamp
    atomic_write_parquet(station_day_path(settings), station_day)
    atomic_write_parquet(model_ready_path(settings), model_ready)
    target = settings["model"]["target_column"]
    result = {
        "station_day_rows": int(len(station_day)),
        "model_ready_rows": int(len(model_ready)),
        "class_counts": {k: int(v) for k, v in model_ready[target].value_counts().sort_index().items()},
    }
    if ctx is not None:
        ctx.counts.update(curated_station_day_rows=result["station_day_rows"],
                          curated_model_ready_rows=result["model_ready_rows"])
        ctx.details.update(class_counts=result["class_counts"])
    return result


def read_model_ready(settings: Settings) -> pd.DataFrame:
    path = model_ready_path(settings)
    if not path.exists():
        raise FileNotFoundError(f"Curated model-ready data not found at {path}: run build-curated first.")
    return pd.read_parquet(path)
