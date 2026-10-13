"""Failure-injection drills: safe, reversible failures run against a DISPOSABLE copy of the project.

Each drill copies config/ and metadata/ plus a (possibly corrupted) copy of the source into
outputs/failure_drills/<drill>/workspace and runs the real CLI there (REPRO_PROJECT_ROOT), so the
real data layers, outputs and database are never touched. Database stages are skipped (the
Airflow database-outage drill covers PostgreSQL failures).

  python scripts/failure_drills.py      -> docs/evidence/failure_drills.txt (+ .json)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.source import inspect_file  # noqa: E402

DRILLS_DIR = ROOT / "outputs" / "failure_drills"
FILE_STAGES = ["ingest --mode full", "validate-raw", "transform-staging", "build-curated",
               "validate-curated --skip-database", "train", "evaluate", "reproducibility-check"]
SOURCE_REL = Path("data/source/world_air_quality.csv")


def make_workspace(name: str, source_bytes: bytes, register: bool, settings_patch: dict | None = None) -> Path:
    ws = DRILLS_DIR / name / "workspace"
    if ws.exists():
        shutil.rmtree(ws)
    shutil.copytree(ROOT / "config", ws / "config")
    shutil.copytree(ROOT / "metadata", ws / "metadata")
    (ws / SOURCE_REL).parent.mkdir(parents=True)
    (ws / SOURCE_REL).write_bytes(source_bytes)
    if register:  # simulate a new source version that someone registered in the manifest
        facts = inspect_file(ws / SOURCE_REL)
        manifest_path = ws / "metadata" / "source_manifest.yml"
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        manifest["file"].update(sha256=facts["sha256"], size_bytes=facts["size_bytes"], row_count=facts["row_count"],
                                column_count=facts["column_count"])
        manifest["schema"]["columns"] = facts["header"]
        manifest_path.write_text(yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False), encoding="utf-8")
    if settings_patch:
        path = ws / "config" / "settings.yml"
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        for dotted, value in settings_patch.items():
            node = cfg
            *parents, leaf = dotted.split(".")
            for p in parents:
                node = node[p]
            node[leaf] = value
        path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return ws


def run_cli(ws: Path, command: str, run_id: str) -> tuple[int, str]:
    env = {**os.environ, "REPRO_PROJECT_ROOT": str(ws), "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run([sys.executable, "-m", "src.cli", "--run-id", run_id, *command.split()], cwd=ROOT,
                          env=env, capture_output=True, text=True, encoding="utf-8")
    events = []
    for line in (proc.stdout + "\n" + proc.stderr).splitlines():
        if line.startswith("CONFIG ERROR"):
            events.append(line)
        if not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("event") in ("stage_failed", "command_failed") or (
                rec.get("event") == "dq_check" and rec.get("status") != "pass"):
            keep = {k: rec.get(k) for k in ("event", "stage", "check_id", "status", "error_type", "error", "observed")
                    if rec.get(k) is not None}
            events.append(json.dumps(keep, ensure_ascii=False)[:600])
        if rec.get("event") == "stage_succeeded" and rec.get("stage") == "transform_staging":
            events.append(json.dumps({"event": "stage_succeeded", "stage": "transform_staging",
                                      "row_counts": rec.get("row_counts")}))
    return proc.returncode, "\n".join(events)


def run_until_failure(ws: Path, run_id: str, stages: list[str], lines: list[str]) -> tuple[str | None, int]:
    for command in stages:
        code, events = run_cli(ws, command, run_id)
        lines.append(f"$ python -m src.cli {command}   -> exit {code}")
        if events:
            lines.append("    " + events.replace("\n", "\n    "))
        if code != 0:
            return command, code
    return None, 0


def model_outputs_exist(ws: Path) -> bool:
    return (ws / "outputs" / "metrics.json").exists() or (ws / "outputs" / "predictions.csv").exists()


def corrupt_rows(text: str, every: int) -> str:
    lines = text.split("\r\n")
    out = [lines[0]]
    for i, line in enumerate(lines[1:], start=1):
        if line and i % every == 0:
            fields = line.split(";")
            fields[7] = "not-a-number" if (i // every) % 2 else fields[7]
            fields[8] = fields[8] if (i // every) % 2 else "yesterday"
            line = ";".join(fields)
        out.append(line)
    return "\r\n".join(out)


def main() -> int:
    source = (ROOT / SOURCE_REL).read_bytes()
    text = source.decode("utf-8-sig")
    lines = ["Failure-injection drills (disposable workspaces under outputs/failure_drills/)",
             f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}", "=" * 100]
    results = {}

    # 1. Tampered source: a single changed byte.
    tampered = bytearray(source)
    tampered[1000] = ord("7") if tampered[1000] != ord("7") else ord("8")
    ws = make_workspace("tampered_source", bytes(tampered), register=False)
    lines += ["", "## Drill 1 - tampered source (one byte changed, manifest NOT updated)"]
    failed_at, code = run_until_failure(ws, "drill_tampered", FILE_STAGES, lines)
    results["tampered_source"] = {"failed_at": failed_at, "exit_code": code,
                                  "model_outputs_written": model_outputs_exist(ws),
                                  "raw_batches_written": (ws / "data" / "raw" / "batches").exists()}

    # 2. Missing required column (registered as a new version, so only the contract can catch it).
    header, *rest = text.split("\r\n")
    drop = header.split(";").index("Unit")
    no_unit = "\r\n".join(";".join(f for j, f in enumerate(l.split(";")) if j != drop) if l else l for l in [header, *rest])
    ws = make_workspace("missing_column", b"\xef\xbb\xbf" + no_unit.encode("utf-8"), register=True)
    lines += ["", "## Drill 2 - required column 'Unit' removed (manifest re-registered, contract unchanged)"]
    failed_at, code = run_until_failure(ws, "drill_missing_column", FILE_STAGES, lines)
    results["missing_column"] = {"failed_at": failed_at, "exit_code": code,
                                 "model_outputs_written": model_outputs_exist(ws)}

    # 3. Invalid rows above the quarantine threshold, then recovery with the correct source.
    bad = corrupt_rows(text, every=50)  # 2% of rows: unparseable value or timestamp
    ws = make_workspace("invalid_rows", b"\xef\xbb\xbf" + bad.encode("utf-8"), register=True)
    lines += ["", "## Drill 3 - 2% of rows with unparseable Value / Last Updated (threshold 1%)"]
    failed_at, code = run_until_failure(ws, "drill_invalid_rows", FILE_STAGES, lines)
    q = ws / "data" / "quarantine" / "staging_rejects.parquet"
    import pandas as pd

    quarantine = pd.read_parquet(q)["reject_reason"].value_counts().to_dict() if q.exists() else {}
    results["invalid_rows"] = {"failed_at": failed_at, "exit_code": code,
                               "staging_written": (ws / "data" / "staging" / "air_quality_readings.parquet").exists(),
                               "model_outputs_written": model_outputs_exist(ws), "quarantine_reasons": quarantine}
    lines += ["", f"Quarantine after the failure: {quarantine}",
              f"Staging written: {results['invalid_rows']['staging_written']}; model outputs written: "
              f"{results['invalid_rows']['model_outputs_written']}",
              "", "### Correct the problem: restore the authentic source and its manifest, rerun the same stages"]
    shutil.copy(ROOT / SOURCE_REL, ws / SOURCE_REL)
    shutil.copy(ROOT / "metadata" / "source_manifest.yml", ws / "metadata" / "source_manifest.yml")
    failed_again, code = run_until_failure(ws, "drill_invalid_rows_fixed", FILE_STAGES, lines)
    comparison = json.loads((ws / "outputs" / "reproducibility_comparison.json").read_text(encoding="utf-8")) \
        if failed_again is None else {}
    results["invalid_rows"]["recovery"] = {"failed_at": failed_again, "exit_code": code,
                                           "reproducibility_outcome": comparison.get("outcome")}
    lines.append(f"Recovery result: failed_at={failed_again}, reproducibility outcome={comparison.get('outcome')}")

    # 4. A few invalid rows below the threshold: quarantined, pipeline continues.
    few = corrupt_rows(text, every=500)  # 0.2%
    ws = make_workspace("few_invalid_rows", b"\xef\xbb\xbf" + few.encode("utf-8"), register=True)
    lines += ["", "## Drill 4 - 0.2% invalid rows (below the 1% threshold): quarantined, run continues"]
    failed_at, code = run_until_failure(ws, "drill_few_invalid", FILE_STAGES[:5], lines)
    q = ws / "data" / "quarantine" / "staging_rejects.parquet"
    reasons = pd.read_parquet(q)["reject_reason"].value_counts().to_dict() if q.exists() else {}
    results["few_invalid_rows"] = {"failed_at": failed_at, "exit_code": code, "quarantine_reasons": reasons}
    lines.append(f"Quarantine reasons: {reasons}")

    # 5. Invalid configuration value.
    ws = make_workspace("invalid_config", source, register=False, settings_patch={"model.split.test_size": 1.5})
    lines += ["", "## Drill 5 - invalid configuration value (model.split.test_size = 1.5)"]
    failed_at, code = run_until_failure(ws, "drill_bad_config", FILE_STAGES, lines)
    results["invalid_config"] = {"failed_at": failed_at, "exit_code": code,
                                 "anything_written": (ws / "data" / "raw").exists()}

    expectations = {
        "tampered_source": results["tampered_source"]["failed_at"] == "ingest --mode full"
        and not results["tampered_source"]["raw_batches_written"] and not results["tampered_source"]["model_outputs_written"],
        "missing_column": results["missing_column"]["failed_at"] == "ingest --mode full"
        and not results["missing_column"]["model_outputs_written"],
        "invalid_rows": results["invalid_rows"]["failed_at"] == "transform-staging"
        and not results["invalid_rows"]["staging_written"] and not results["invalid_rows"]["model_outputs_written"]
        and results["invalid_rows"]["recovery"]["reproducibility_outcome"] == "matches_within_tolerance",
        "few_invalid_rows": results["few_invalid_rows"]["failed_at"] is None
        and sum(results["few_invalid_rows"]["quarantine_reasons"].get(k, 0) for k in
                ("null_or_unparseable_value", "null_or_unparseable_timestamp")) > 0,
        "invalid_config": results["invalid_config"]["exit_code"] == 2 and not results["invalid_config"]["anything_written"],
    }
    lines += ["", "=" * 100, "SUMMARY"]
    lines += [f"  {name:<18} expected behaviour observed: {ok}   {json.dumps(results[name], ensure_ascii=False)}"
              for name, ok in expectations.items()]
    ok = all(expectations.values())
    lines.append(f"RESULT: {'PASS' if ok else 'FAIL'}")
    evidence = ROOT / "docs" / "evidence"
    (evidence / "failure_drills.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    (evidence / "failure_drills.json").write_text(json.dumps({"results": results, "expectations": expectations},
                                                             indent=2, ensure_ascii=False) + "\n",
                                                  encoding="utf-8", newline="\n")
    print("\n".join(lines[-8:]))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
