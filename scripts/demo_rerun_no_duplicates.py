"""Evidence for acceptance test A15: a second end-to-end run adds no duplicate logical records.

Records table sizes and duplicate business-key counts in PostgreSQL, runs the whole pipeline
again (`run-all`, incremental), records them again, and shows the load statistics of that run.

  python scripts/demo_rerun_no_duplicates.py   -> docs/evidence/rerun_no_duplicates.txt
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import load_settings  # noqa: E402
from src.storage import connect, db_stats  # noqa: E402


def table_state(settings) -> dict:
    return {r["table_name"]: {"rows": r["total_rows"], "duplicate_business_keys": r["duplicate_business_keys"]}
            for r in db_stats(settings)["duplicate_business_keys"]}


def main() -> int:
    settings = load_settings()
    run_id = f"rerun_check__{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    before = table_state(settings)
    proc = subprocess.run([sys.executable, "-m", "src.cli", "--run-id", run_id, "run-all"], cwd=ROOT,
                          capture_output=True, text=True, encoding="utf-8")
    after = table_state(settings)
    with connect(settings) as conn:
        cur = conn.execute("SELECT table_name, snapshot_rows, rows_inserted, rows_updated, rows_deleted, rows_unchanged, "
                           "table_rows_after FROM audit.load_log WHERE pipeline_run_id = %s ORDER BY load_id", (run_id,))
        loads = [dict(zip([d.name for d in cur.description], r)) for r in cur.fetchall()]
    skipped = [json.loads(l)["batch_id"] for l in proc.stdout.splitlines()
               if l.startswith("{") and '"batch_skipped_already_committed"' in l]
    summary = json.loads((ROOT / "outputs" / "run_summary.json").read_text(encoding="utf-8"))
    ok = (proc.returncode == 0 and before == after
          and all(v["duplicate_business_keys"] == 0 for v in after.values())
          and all(l["rows_inserted"] == l["rows_updated"] == l["rows_deleted"] == 0 for l in loads)
          and summary["pipeline_run_id"] == run_id and summary["reproducibility_outcome"] == "matches_within_tolerance")
    lines = [
        "Second end-to-end run: no duplicate logical records (acceptance test A15)",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "=" * 100,
        "", "## PostgreSQL tables BEFORE the run", json.dumps(before, indent=2),
        "", f"## python -m src.cli --run-id {run_id} run-all  -> exit {proc.returncode}",
        f"raw batches skipped (already committed): {skipped}",
        "", "## Load statistics of this run (audit.load_log)",
        *[f"  {l['table_name']:<34} snapshot={l['snapshot_rows']:>6} inserted={l['rows_inserted']} "
          f"updated={l['rows_updated']} deleted={l['rows_deleted']} unchanged={l['rows_unchanged']:>6} "
          f"after={l['table_rows_after']:>6}" for l in loads],
        "", "## PostgreSQL tables AFTER the run", json.dumps(after, indent=2),
        "", f"## Reproducibility outcome of the rerun: {summary['reproducibility_outcome']}",
        "", f"RESULT: {'PASS' if ok else 'FAIL'} (table sizes unchanged, 0 duplicate business keys, 0 rows written)",
    ]
    out = ROOT / "docs" / "evidence" / "rerun_no_duplicates.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines[-1:]))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
