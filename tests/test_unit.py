"""Unit tests for reusable pipeline functions."""
from __future__ import annotations

import pandas as pd
import pytest

from src.config import ConfigError, load_settings
from src.ingest import plan_batches
from src.io_utils import digest_of_keys, hash_fields, hash_rows
from src.source import SourceIdentityError, verify_source
from src.transform import clean_readings, derive_model_ready, pivot_station_day, standardize_nulls


def test_plan_batches_is_contiguous_complete_and_deterministic():
    plan = plan_batches(10, 3)
    assert [(b["row_start"], b["row_end"]) for b in plan] == [(1, 4), (5, 7), (8, 10)]
    assert sum(b["row_count"] for b in plan) == 10
    assert plan == plan_batches(10, 3)
    with pytest.raises(ValueError):
        plan_batches(2, 3)


def test_hash_rows_matches_hash_fields_and_separates_fields():
    frame = pd.DataFrame({"a": ["ab", "a", None], "b": ["c", "bc", "x"]})
    hashes = hash_rows(frame, ["a", "b"])
    assert hashes.iloc[0] == hash_fields(["ab", "c"])
    assert hashes.iloc[0] != hashes.iloc[1]          # ("ab","c") must not collide with ("a","bc")
    assert hashes.iloc[2] == hash_fields([None, "x"])  # None is hashed as empty text
    assert digest_of_keys(["b", "a"]) == digest_of_keys(["a", "b"])  # order-independent set checksum


def test_standardize_nulls_uses_declared_tokens_only():
    frame = pd.DataFrame({"city": ["N/A", "None", "", "Unknown", "Namur"]})
    out = standardize_nulls(frame, ["city"], ["", "N/A", "None"])
    assert out["city"].isna().tolist() == [True, True, True, False, False]


def test_clean_readings_applies_rules_in_notebook_order(mini_settings):
    from src.source import read_source_text

    raw = read_source_text(mini_settings.source_path, mini_settings)
    raw.insert(0, "source_row_key", range(1, len(raw) + 1))
    raw["source_batch_id"] = 1
    raw["record_hash"] = "h"
    staging, quarantine, funnel = clean_readings(raw, mini_settings, "unit-test")
    reasons = quarantine["reject_reason"].value_counts().to_dict()
    assert reasons == {"duplicate_reading": 1, "non_ugm3_unit": 1, "negative_value": 1, "out_of_scope_pollutant": 1}
    # The FIRST reading in source order survives deduplication (notebook keep='first').
    st00 = staging[(staging["location"] == "Station 00") & (staging["pollutant"] == "PM2.5")]
    assert st00["value"].tolist() == [8.0]
    assert "Unknown" in set(staging["city"]) and "Unknown" in set(staging["country_label"])
    assert funnel["reconciles"] and len(staging) + len(quarantine) == len(raw)


def test_derive_model_ready_features_and_label_threshold(mini_settings):
    staging = pd.DataFrame({
        "country_label": ["X"] * 8, "city": ["c"] * 8, "location": ["L1"] * 4 + ["L2"] * 4,
        "reading_date": ["2024-01-01"] * 8,
        "pollutant": ["PM2.5", "NO2", "O3", "SO2", "PM2.5", "NO2", "O3", "CO"],
        "value": [15.0, 10.0, 39.0, 3.0, 15.01, 20.0, 9.0, 400.0],
    })
    ready = derive_model_ready(pivot_station_day(staging), mini_settings).set_index("location")
    assert ready.loc["L1", "air_quality"] == "Safe"          # exactly 15.0 is not an exceedance
    assert ready.loc["L2", "air_quality"] == "Unhealthy"
    assert ready.loc["L1", "ox_total_oxidant"] == 49.0
    assert ready.loc["L1", "no2_o3_ratio"] == pytest.approx(10.0 / 40.0)
    assert (ready.loc["L1", "so2_missing"], ready.loc["L1", "co_missing"]) == (0, 1)
    assert (ready.loc["L2", "so2_missing"], ready.loc["L2", "co_missing"]) == (1, 0)


@pytest.mark.parametrize("override, message", [
    ({"model": {"split": {"test_size": 1.5}}}, "test_size"),
    ({"model": {"impute_scope": "everything"}}, "impute_scope"),
    ({"model": {"features": ["no2", "pm25"]}}, "leakage"),
    ({"ingestion": {"num_batches": 2}}, "num_batches"),
    ({"paths": {"raw_dir": "C:/Users/someone/raw"}}, "relative"),
])
def test_invalid_configuration_is_rejected(override, message):
    with pytest.raises(ConfigError, match=message):
        load_settings(overrides=override)


def test_verify_source_detects_a_changed_byte(mini_settings):
    verify_source(mini_settings)  # pristine copy passes
    data = bytearray(mini_settings.source_path.read_bytes())
    data[-5] = ord("9") if data[-5] != ord("9") else ord("8")
    mini_settings.source_path.write_bytes(bytes(data))
    with pytest.raises(SourceIdentityError, match="sha256"):
        verify_source(mini_settings)
