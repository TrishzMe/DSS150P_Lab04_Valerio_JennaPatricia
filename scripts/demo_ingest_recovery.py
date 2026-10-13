"""Evidence run for checkpointed ingestion: interrupted run -> resumed run -> rerun -> verification.

Usage (from the repository root):  python scripts/demo_ingest_recovery.py
Writes the full transcript to docs/evidence/ingestion_recovery_demo.txt and exits non-zero if
any expectation is not met.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "evidence" / "ingestion_recovery_demo.txt"
STAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

STEPS = [
    ("1. First run (full refresh) - deliberately interrupted after 2 committed batches",
     ["--run-id", f"demo_{STAMP}_run1", "ingest", "--mode", "full", "--stop-after-batches", "2"], 3),
    ("2. Checkpoint state after the interruption", ["raw-stats"], 0),
    ("3. Resumed run (incremental) - must skip batches 1-2 and commit 3-5",
     ["--run-id", f"demo_{STAMP}_run2", "ingest", "--mode", "incremental"], 0),
    ("4. Rerun (incremental) - nothing new, every batch skipped",
     ["--run-id", f"demo_{STAMP}_run3", "ingest", "--mode", "incremental"], 0),
    ("5. Raw-layer data-quality checks", ["--run-id", f"demo_{STAMP}_run3", "validate-raw"], 0),
    ("6. Final duplicate-count / row-count verification", ["raw-stats"], 0),
]


def main() -> int:
    lines = [
        "Checkpointed ingestion recovery demonstration",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Python: {sys.version.split()[0]}  Command prefix: python -m src.cli",
        "=" * 100,
    ]
    ok = True
    last_stats = None
    for title, cli_args, expected_code in STEPS:
        proc = subprocess.run([sys.executable, "-m", "src.cli", *cli_args], cwd=ROOT,
                              capture_output=True, text=True, encoding="utf-8")
        status = "OK" if proc.returncode == expected_code else "UNEXPECTED"
        ok &= proc.returncode == expected_code
        lines += ["", f"## {title}", f"$ python -m src.cli {' '.join(cli_args)}",
                  f"exit code: {proc.returncode} (expected {expected_code}) -> {status}", proc.stdout.rstrip()]
        if proc.stderr.strip():
            lines.append("[stderr]\n" + proc.stderr.rstrip())
        if cli_args == ["raw-stats"]:
            last_stats = json.loads(proc.stdout[proc.stdout.index("{"):])
            if title.startswith("2."):
                ok &= last_stats["committed_batches"] == 2 and last_stats["last_committed_batch_id"] == 2

    assert last_stats is not None
    final_ok = (last_stats["raw_rows"] == 54255 and last_stats["distinct_source_row_keys"] == 54255
                and last_stats["duplicate_source_row_keys"] == 0 and last_stats["committed_batches"] == 5)
    ok &= final_ok
    lines += ["", "=" * 100,
              f"VERIFICATION: raw_rows={last_stats['raw_rows']} distinct_keys={last_stats['distinct_source_row_keys']} "
              f"duplicate_keys={last_stats['duplicate_source_row_keys']} committed_batches={last_stats['committed_batches']}",
              f"RESULT: {'PASS' if ok else 'FAIL'}"]
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-2:]))
    print(f"Transcript written to {EVIDENCE.relative_to(ROOT)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
