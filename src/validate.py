"""Data-quality checks driven by metadata/data_contract.yml.

Each check returns a CheckResult tagged with its contract id and category. A report is written
per layer to ``outputs/runs/<run>/quality/<layer>.json`` and ``outputs/quality/<layer>.json``
(latest). Any failed check whose contract behaviour is ``fail`` raises DataQualityError, which
stops the pipeline before bad data reaches downstream layers or model outputs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import pandas as pd

from .config import Settings
from .io_utils import atomic_write_json, hash_rows, sha256_file
from .observability import StageContext, log_event, run_slug, utc_now_iso
from .source import RAW_COLUMNS, load_contract, load_manifest, verify_source


class DataQualityError(RuntimeError):
    """At least one `fail`-severity data-quality check failed."""

    def __init__(self, layer: str, failed: list["CheckResult"]):
        self.layer = layer
        self.failed = failed
        detail = "; ".join(f"{c.check_id} {c.name}: observed={c.observed!r} expected={c.expected!r}" for c in failed)
        super().__init__(f"{len(failed)} data-quality check(s) failed on layer '{layer}': {detail}")


@dataclass
class CheckResult:
    check_id: str
    name: str
    category: str
    layer: str
    passed: bool
    on_violation: str
    observed: Any
    expected: Any
    message: str = ""

    @property
    def status(self) -> str:
        if self.passed:
            return "pass"
        return "fail" if self.on_violation == "fail" else "warn"


def _check(check_id: str, name: str, category: str, layer: str, passed: bool, observed: Any, expected: Any,
           on_violation: str = "fail", message: str = "") -> CheckResult:
    return CheckResult(check_id, name, category, layer, bool(passed), on_violation, observed, expected, message)


def _guard(check_id: str, name: str, category: str, layer: str, fn: Callable[[], CheckResult]) -> CheckResult:
    """Run a check; an exception inside the check is itself a failed check (never a silent pass)."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        return _check(check_id, name, category, layer, False, f"{type(exc).__name__}: {exc}", "check runs", "fail")


def finalize_report(settings: Settings, run_id: str, layer: str, results: list[CheckResult],
                    ctx: StageContext | None = None, raise_on_fail: bool = True) -> dict[str, Any]:
    failed = [r for r in results if r.status == "fail"]
    warned = [r for r in results if r.status == "warn"]
    report = {
        "pipeline_run_id": run_id,
        "layer": layer,
        "generated_at_utc": utc_now_iso(),
        "status": "failed" if failed else "passed",
        "summary": {"checks": len(results), "passed": sum(r.passed for r in results),
                    "failed": len(failed), "warnings": len(warned)},
        "checks": [{**asdict(r), "status": r.status} for r in results],
    }
    out = settings.path("outputs_dir")
    atomic_write_json(out / "runs" / run_slug(run_id) / "quality" / f"{layer}.json", report)
    atomic_write_json(out / "quality" / f"{layer}.json", report)
    for r in results:
        log_event("dq_check", level=20 if r.passed else (40 if r.status == "fail" else 30),
                  pipeline_run_id=run_id, stage=f"validate_{layer}", check_id=r.check_id, check=r.name,
                  category=r.category, status=r.status, observed=r.observed, expected=r.expected)
    if ctx is not None:
        ctx.counts.update({f"dq_{layer}_checks": len(results), f"dq_{layer}_failed": len(failed),
                           f"dq_{layer}_warnings": len(warned)})
    if failed and raise_on_fail:
        raise DataQualityError(layer, failed)
    return report


# --------------------------------------------------------------------------------------- raw layer

def raw_checks(settings: Settings) -> list[CheckResult]:
    from .ingest import AUDIT_COLUMNS, committed_batch_files, load_checkpoint, read_raw

    layer = "raw"
    manifest = load_manifest(settings)
    expected_rows = int(manifest["file"]["row_count"])
    results: list[CheckResult] = []

    def src01() -> CheckResult:
        facts = verify_source(settings)
        return _check("SRC-01", "source checksum matches manifest", "freshness_applicability", layer,
                      True, facts["sha256"], manifest["file"]["sha256"])

    results.append(_guard("SRC-01", "source checksum matches manifest", "freshness_applicability", layer, src01))

    checkpoint = load_checkpoint(settings)
    committed = len(checkpoint["committed_batches"]) if checkpoint else 0
    planned = checkpoint["num_batches"] if checkpoint else int(settings["ingestion"]["num_batches"])
    results.append(_check("CMP-00a", "all planned batches committed", "completeness", layer,
                          committed == planned, committed, planned))
    if committed != planned:
        return results

    raw = read_raw(settings)
    expected_cols = ["source_row_key", *RAW_COLUMNS, *AUDIT_COLUMNS[1:]]
    results.append(_check("SCH-01", "raw columns match contract", "schema", layer,
                          list(raw.columns) == expected_cols, list(raw.columns), expected_cols))
    results.append(_check("CMP-00", "raw row count equals manifest", "completeness", layer,
                          len(raw) == expected_rows, len(raw), expected_rows))
    keys = raw["source_row_key"]
    contiguous = keys.is_unique and len(keys) > 0 and int(keys.min()) == 1 and int(keys.max()) == len(keys)
    results.append(_check("UNQ-01", "source_row_key unique and contiguous", "uniqueness", layer,
                          contiguous, {"rows": len(keys), "distinct": int(keys.nunique()),
                                       "min": int(keys.min()), "max": int(keys.max())},
                          {"distinct": len(keys), "min": 1, "max": len(keys)}))

    def int01() -> CheckResult:
        mismatched = []
        for path in committed_batch_files(settings):
            bid = str(int(path.stem.split("_")[1]))
            if sha256_file(path) != checkpoint["committed_batches"][bid]["file_sha256"]:
                mismatched.append(path.name)
        recomputed = hash_rows(raw, RAW_COLUMNS)
        bad_hashes = int((recomputed.to_numpy(dtype=object) != raw["record_hash"].to_numpy(dtype=object)).sum())
        ok = not mismatched and bad_hashes == 0
        return _check("INT-01", "batch files and record hashes untampered", "consistency", layer, ok,
                      {"files_changed": mismatched, "record_hash_mismatches": bad_hashes},
                      {"files_changed": [], "record_hash_mismatches": 0})

    results.append(_guard("INT-01", "batch files and record hashes untampered", "consistency", layer, int01))
    dup_hash = int(raw["record_hash"].duplicated().sum())
    results.append(_check("UNQ-01b", "no identical duplicate source records", "uniqueness", layer,
                          dup_hash == 0, dup_hash, 0, on_violation="warn",
                          message="identical records are kept in raw (faithful copy); staging deduplicates"))
    return results


def validate_raw(settings: Settings, run_id: str, ctx: StageContext | None = None) -> dict[str, Any]:
    return finalize_report(settings, run_id, "raw", raw_checks(settings), ctx)


# ----------------------------------------------------------------------------------- staging layer

def staging_checks(staging: pd.DataFrame, funnel: dict[str, Any], settings: Settings) -> list[CheckResult]:
    layer = "staging"
    cfg = settings["transform"]
    key = cfg["dedup_key"]
    critical = ["pollutant", "unit", "value", "last_updated_utc", "location", "city", "country_label"]
    nulls = {c: int(staging[c].isna().sum()) for c in critical}
    dup = int(staging.duplicated(subset=key).sum())
    # Coerce so a wrongly typed column is reported by SCH-02 instead of crashing this check.
    numeric_value = pd.to_numeric(staging["value"], errors="coerce")
    negatives = int((numeric_value < cfg["min_valid_value"]).sum())
    bad_units = sorted(set(staging["unit"].dropna()) - {cfg["accepted_unit"]})
    bad_pollutants = sorted(set(staging["pollutant"].dropna()) - set(cfg["target_pollutants"]))
    ts_type = str(staging["last_updated_utc"].dtype)
    limit = cfg["max_invalid_row_rate"]
    return [
        _check("SCH-02", "value is float", "schema", layer, pd.api.types.is_float_dtype(staging["value"]),
               str(staging["value"].dtype), "float64"),
        _check("SCH-03", "last_updated_utc is a UTC timestamp", "schema", layer,
               ts_type.startswith("datetime64") and "UTC" in ts_type, ts_type, "datetime64[*, UTC]"),
        _check("CMP-01", "critical fields not null", "completeness", layer, not any(nulls.values()), nulls,
               {c: 0 for c in critical}),
        _check("UNQ-02", "(location, pollutant, last_updated_utc) unique", "uniqueness", layer, dup == 0, dup, 0),
        _check("VAL-01", "value >= 0", "validity", layer, negatives == 0, negatives, 0),
        _check("VAL-02", "only µg/m³ and in-scope pollutants", "validity", layer,
               not bad_units and not bad_pollutants, {"units": bad_units, "pollutants": bad_pollutants},
               {"units": [], "pollutants": []}),
        _check("INV-01", "structurally invalid share within limit", "validity", layer,
               funnel.get("structural_invalid_rate", 0) <= limit, funnel.get("structural_invalid_rate"), f"<= {limit}"),
        _check("REC-01", "every raw row is in staging or quarantine", "consistency", layer,
               bool(funnel.get("reconciles")),
               {"raw": funnel.get("raw_rows"), "staging": funnel.get("staging_rows"),
                "quarantine": sum(funnel.get("rejected_by_type", {}).values())}, "raw == staging + quarantine"),
    ]


# ----------------------------------------------------------------------------------- curated layer

def _expected_counts_applicable(settings: Settings) -> tuple[bool, dict[str, Any]]:
    contract = load_contract(settings)
    manifest = load_manifest(settings)
    return contract.get("source_version_sha256") == manifest["file"]["sha256"], contract.get("expected_counts", {})


def curated_checks(settings: Settings, station_day: pd.DataFrame, model_ready: pd.DataFrame,
                   staging_rows: int | None = None) -> list[CheckResult]:
    import numpy as np

    from .transform import CURATED_ORDER

    layer = "curated"
    m = settings["model"]
    target = m["target_column"]
    results = [
        _check("UNQ-03", "station_day_key unique", "uniqueness", layer,
               station_day["station_day_key"].is_unique and model_ready["station_day_key"].is_unique,
               {"station_day_dups": int(station_day["station_day_key"].duplicated().sum()),
                "model_ready_dups": int(model_ready["station_day_key"].duplicated().sum())}, 0),
    ]
    req_nulls = {c: int(model_ready[c].isna().sum()) for c in m["required_columns"]}
    results.append(_check("CMP-02", "model-ready required pollutants not null", "completeness", layer,
                          not any(req_nulls.values()), req_nulls, {c: 0 for c in m["required_columns"]}))
    labels = sorted(model_ready[target].dropna().unique().tolist())
    allowed = sorted([m["negative_label"], m["positive_label"]])
    flags_ok = all(set(model_ready[f"{c}_missing"].unique()) <= {0, 1} for c in m["imputed_columns"])
    pollutant_cols = ["pm25", "pm10", "no2", "o3", "so2", "co"]
    negatives = int((station_day[pollutant_cols] < 0).sum().sum())
    results.append(_check("VAL-04", "labels, flags and concentrations valid", "validity", layer,
                          set(labels) <= set(allowed) and model_ready[target].notna().all() and flags_ok and negatives == 0,
                          {"labels": labels, "flags_binary": flags_ok, "negative_values": negatives},
                          {"labels": allowed, "flags_binary": True, "negative_values": 0}))
    expected_label = np.where(model_ready[m["target_source_column"]] > m["threshold_ugm3"], m["positive_label"],
                              m["negative_label"])
    mislabelled = int((model_ready[target].to_numpy(dtype=object) != expected_label).sum())
    results.append(_check("CON-01", f"air_quality consistent with pm25 > {m['threshold_ugm3']}", "consistency",
                          layer, mislabelled == 0, mislabelled, 0))
    ox_bad = int((~np.isclose(model_ready["ox_total_oxidant"], model_ready["no2"] + model_ready["o3"],
                              rtol=0, atol=1e-9)).sum())
    ratio_bad = int((~np.isclose(model_ready["no2_o3_ratio"], model_ready["no2"] / (model_ready["o3"] + 1.0),
                                 rtol=0, atol=1e-12)).sum())
    results.append(_check("CON-02", "derived features equal their formulas", "consistency", layer,
                          ox_bad == 0 and ratio_bad == 0, {"ox_total_oxidant": ox_bad, "no2_o3_ratio": ratio_bad},
                          {"ox_total_oxidant": 0, "no2_o3_ratio": 0}))
    flag_bad = {c: int((model_ready[f"{c}_missing"] != model_ready[c].isna().astype(int)).sum())
                for c in m["imputed_columns"]}
    results.append(_check("CON-03", "missing flags match nulls", "consistency", layer,
                          not any(flag_bad.values()), flag_bad, {c: 0 for c in m["imputed_columns"]}))
    for name, frame in (("station_day", station_day), ("model_ready", model_ready)):
        ordered = frame[CURATED_ORDER].astype(str)
        expected = ordered.sort_values(CURATED_ORDER, kind="mergesort")
        results.append(_check("DET-01", f"{name} rows in declared total order {CURATED_ORDER}", "consistency",
                              layer, ordered.index.equals(expected.index), "sorted" if ordered.index.equals(expected.index)
                              else "not sorted", "sorted"))

    applicable, expected = _expected_counts_applicable(settings)
    observed = {
        "station_day_rows": int(len(station_day)),
        "model_ready_rows": int(len(model_ready)),
        "model_ready_class_counts": {k: int(v) for k, v in model_ready[target].value_counts().sort_index().items()},
    }
    if staging_rows is not None:
        observed["staging_rows"] = int(staging_rows)
    wanted = {k: expected.get(k) for k in observed}
    if applicable:
        results.append(_check("APP-01", "layer row counts equal expected counts for this source version",
                              "freshness_applicability", layer, observed == wanted, observed, wanted))
    else:
        results.append(_check("APP-01", "expected counts not applicable (different source version)",
                              "freshness_applicability", layer, False, observed, wanted, on_violation="warn",
                              message="source checksum differs from the contract's source_version_sha256"))
    return results


def validate_curated(settings: Settings, run_id: str, ctx: StageContext | None = None,
                     include_database: bool = True) -> dict[str, Any]:
    from .transform import read_model_ready, read_staging, station_day_path

    station_day = pd.read_parquet(station_day_path(settings))
    model_ready = read_model_ready(settings)
    results = curated_checks(settings, station_day, model_ready, staging_rows=len(read_staging(settings)))
    if include_database:
        from .storage import database_consistency_check

        results.append(_guard("CON-04", "PostgreSQL tables equal Parquet files", "consistency", "curated",
                              lambda: database_consistency_check(settings)))
    return finalize_report(settings, run_id, "curated", results, ctx)


def contract_check_ids(settings: Settings) -> set[str]:
    return {c["id"] for c in load_contract(settings).get("checks", [])}
