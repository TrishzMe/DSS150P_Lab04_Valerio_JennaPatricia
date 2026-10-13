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


def contract_check_ids(settings: Settings) -> set[str]:
    return {c["id"] for c in load_contract(settings).get("checks", [])}
