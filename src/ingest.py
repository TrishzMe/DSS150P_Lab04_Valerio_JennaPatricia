"""Raw-layer ingestion: full snapshot copy and checkpointed incremental batches.

The source has no reliable arrival timestamp (``Last Updated`` is a per-station "last reported"
time, see metadata/source_profile.md), so incremental arrival is SIMULATED: the immutable file
is split into ``ingestion.num_batches`` contiguous batches by ``source_row_key`` (the 1-based
record ordinal). The checkpoint stores the batches that were committed; it is a laboratory
sequence checkpoint, not a business watermark.

Guarantees:
* source bytes are never modified (the full copy is verified by SHA-256);
* a batch is written to a temporary file, read back and validated, renamed into place, and
  only then recorded in the checkpoint (``data/raw/_state/ingest_checkpoint.json``);
* re-running skips committed batches, so no logical record is appended twice;
* ``--mode full`` (full refresh) rebuilds every batch from a fresh run-specific copy.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Settings
from .io_utils import atomic_write_json, digest_of_keys, hash_rows, read_json, sha256_file
from .observability import StageContext, log_event, run_slug, utc_now_iso
from .source import RAW_COLUMNS, ContractViolation, load_manifest, read_source_text, verify_source

AUDIT_COLUMNS = [
    "source_row_key",
    "source_batch_id",
    "record_hash",
    "pipeline_run_id",
    "ingested_at_utc",
    "source_file",
    "source_checksum",
]
CHECKPOINT_VERSION = 1


class SimulatedInterruption(RuntimeError):
    """Raised on purpose by ``--stop-after-batches`` to demonstrate crash recovery."""


class CheckpointMismatch(RuntimeError):
    """The checkpoint belongs to a different source version or batch plan."""


def batches_dir(settings: Settings) -> Path:
    return settings.path("raw_dir") / "batches"


def batch_file(settings: Settings, batch_id: int) -> Path:
    return batches_dir(settings) / f"batch_{batch_id:04d}.parquet"


def plan_batches(row_count: int, num_batches: int) -> list[dict[str, int]]:
    """Contiguous, near-equal, deterministic batches over source_row_key 1..row_count."""
    if row_count < num_batches:
        raise ValueError(f"Cannot split {row_count} rows into {num_batches} batches")
    bounds = np.array_split(np.arange(1, row_count + 1), num_batches)
    return [
        {"batch_id": i, "row_start": int(b[0]), "row_end": int(b[-1]), "row_count": int(len(b))}
        for i, b in enumerate(bounds, start=1)
    ]


def load_checkpoint(settings: Settings) -> dict[str, Any] | None:
    path = settings.checkpoint_path
    return read_json(path) if path.exists() else None


def _new_checkpoint(source_checksum: str, plan: list[dict[str, int]]) -> dict[str, Any]:
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_type": "simulated_sequence_checkpoint (not a business watermark)",
        "source_checksum": source_checksum,
        "num_batches": len(plan),
        "total_rows": sum(b["row_count"] for b in plan),
        "last_committed_batch_id": 0,
        "last_committed_source_row_key": 0,
        "committed_batches": {},
        "updated_at_utc": utc_now_iso(),
    }


def _save_checkpoint(settings: Settings, checkpoint: dict[str, Any]) -> None:
    checkpoint["updated_at_utc"] = utc_now_iso()
    committed = sorted(int(k) for k in checkpoint["committed_batches"])
    # The checkpoint advances only through an unbroken prefix of committed batches.
    last = 0
    for bid in committed:
        if bid != last + 1:
            break
        last = bid
    checkpoint["last_committed_batch_id"] = last
    checkpoint["last_committed_source_row_key"] = (
        checkpoint["committed_batches"][str(last)]["row_end"] if last else 0
    )
    atomic_write_json(settings.checkpoint_path, checkpoint)
    history = settings.checkpoint_path.parent / "checkpoint_history.jsonl"
    with history.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"at_utc": checkpoint["updated_at_utc"], "last_committed_batch_id": last,
                             "committed": committed}) + "\n")


def _reset_raw_batches(settings: Settings) -> None:
    if batches_dir(settings).exists():
        shutil.rmtree(batches_dir(settings))
    if settings.checkpoint_path.exists():
        settings.checkpoint_path.unlink()


def copy_full_snapshot(settings: Settings, run_id: str) -> dict[str, Any]:
    """Byte-for-byte copy of the source into a run-specific raw folder, verified by SHA-256."""
    facts = verify_source(settings)
    target_dir = settings.path("raw_dir") / "full" / f"run={run_slug(run_id)}"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / settings["source"]["filename"]
    tmp = target.with_name(f".{target.name}.tmp")
    shutil.copyfile(settings.source_path, tmp)
    os.replace(tmp, target)
    copy_checksum = sha256_file(target)
    if copy_checksum != facts["sha256"]:
        raise ContractViolation(f"Raw copy checksum {copy_checksum} differs from source {facts['sha256']}")
    meta = {
        "pipeline_run_id": run_id,
        "ingested_at_utc": utc_now_iso(),
        "source_file": load_manifest(settings)["file"]["relative_path"],
        "source_checksum": facts["sha256"],
        "raw_file": target.relative_to(settings.root).as_posix(),
        "raw_checksum": copy_checksum,
        "row_count": facts["row_count"],
        "size_bytes": facts["size_bytes"],
    }
    atomic_write_json(target_dir / "_ingest_metadata.json", meta)
    return meta


def _validate_batch_file(path: Path, spec: dict[str, int], expected_columns: list[str]) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    problems = []
    if list(frame.columns) != expected_columns:
        problems.append(f"columns {list(frame.columns)} != {expected_columns}")
    if len(frame) != spec["row_count"]:
        problems.append(f"row_count {len(frame)} != {spec['row_count']}")
    keys = frame["source_row_key"]
    if not keys.is_unique:
        problems.append("duplicate source_row_key values")
    if len(frame) and (int(keys.min()) != spec["row_start"] or int(keys.max()) != spec["row_end"]):
        problems.append(f"key range {keys.min()}..{keys.max()} != {spec['row_start']}..{spec['row_end']}")
    if problems:
        raise ContractViolation(f"Batch {spec['batch_id']} failed validation: " + "; ".join(problems))
    return frame


def ingest(
    settings: Settings,
    run_id: str,
    mode: str = "incremental",
    stop_after_batches: int | None = None,
    ctx: StageContext | None = None,
) -> dict[str, Any]:
    """Ingest the source into the raw layer. ``mode`` is ``incremental`` or ``full``."""
    if mode not in ("incremental", "full"):
        raise ValueError(f"mode must be 'incremental' or 'full', got {mode!r}")

    snapshot = None
    if mode == "full":
        snapshot = copy_full_snapshot(settings, run_id)
        log_event("full_snapshot_copied", pipeline_run_id=run_id, stage="ingest", raw_file=snapshot["raw_file"],
                  raw_checksum=snapshot["raw_checksum"], row_count=snapshot["row_count"])
        _reset_raw_batches(settings)
        read_from = settings.root / snapshot["raw_file"]
        source_checksum = snapshot["source_checksum"]
    else:
        facts = verify_source(settings)
        read_from = settings.source_path
        source_checksum = facts["sha256"]

    manifest = load_manifest(settings)
    text = read_source_text(read_from, settings)
    if list(text.columns) != RAW_COLUMNS:  # contract SCH-01
        raise ContractViolation(f"SCH-01 schema mismatch: columns {list(text.columns)} != {RAW_COLUMNS}")

    plan = plan_batches(len(text), int(settings["ingestion"]["num_batches"]))
    checkpoint = load_checkpoint(settings)
    if checkpoint is None:
        checkpoint = _new_checkpoint(source_checksum, plan)
    elif checkpoint["source_checksum"] != source_checksum or checkpoint["num_batches"] != len(plan):
        raise CheckpointMismatch(
            "Checkpoint was built for a different source version or batch plan "
            f"(checkpoint {checkpoint['source_checksum'][:12]}/{checkpoint['num_batches']} batches, "
            f"now {source_checksum[:12]}/{len(plan)} batches). Run ingest with --mode full to rebuild."
        )

    text.insert(0, "source_row_key", np.arange(1, len(text) + 1, dtype="int64"))
    record_hash = hash_rows(text, RAW_COLUMNS)
    expected_columns = ["source_row_key", *RAW_COLUMNS, *AUDIT_COLUMNS[1:]]
    source_file = manifest["file"]["relative_path"]

    committed_now, skipped = [], []
    for spec in plan:
        bid = spec["batch_id"]
        entry = checkpoint["committed_batches"].get(str(bid))
        path = batch_file(settings, bid)
        if entry and path.exists() and sha256_file(path) == entry["file_sha256"]:
            skipped.append(bid)
            log_event("batch_skipped_already_committed", pipeline_run_id=run_id, stage="ingest", batch_id=bid,
                      committed_by_run=entry["pipeline_run_id"])
            continue
        if entry:
            log_event("batch_recommitting_missing_or_changed_file", level=30, pipeline_run_id=run_id,
                      stage="ingest", batch_id=bid)

        rows = slice(spec["row_start"] - 1, spec["row_end"])
        batch = text.iloc[rows].copy()
        batch["source_batch_id"] = bid
        batch["record_hash"] = record_hash.iloc[rows].values
        batch["pipeline_run_id"] = run_id
        batch["ingested_at_utc"] = utc_now_iso()
        batch["source_file"] = source_file
        batch["source_checksum"] = source_checksum
        batch = batch[expected_columns]

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        batch.to_parquet(tmp, index=False)
        _validate_batch_file(tmp, spec, expected_columns)  # validate BEFORE committing
        os.replace(tmp, path)
        checkpoint["committed_batches"][str(bid)] = {
            **spec,
            "file": path.relative_to(settings.root).as_posix(),
            "file_sha256": sha256_file(path),
            "record_hash_digest": digest_of_keys(batch["record_hash"]),
            "pipeline_run_id": run_id,
            "committed_at_utc": utc_now_iso(),
        }
        _save_checkpoint(settings, checkpoint)  # advance the checkpoint only after the batch is safe
        committed_now.append(bid)
        log_event("batch_committed", pipeline_run_id=run_id, stage="ingest", batch_id=bid,
                  row_count=spec["row_count"], row_start=spec["row_start"], row_end=spec["row_end"])

        if stop_after_batches is not None and len(committed_now) >= stop_after_batches and bid < len(plan):
            raise SimulatedInterruption(
                f"Simulated interruption after committing {len(committed_now)} batch(es) "
                f"(last batch {bid}); rerun ingest to resume from batch {bid + 1}."
            )

    summary = raw_summary(settings)
    result = {
        "mode": mode,
        "batches_planned": len(plan),
        "batches_committed_this_run": committed_now,
        "batches_skipped_already_committed": skipped,
        "full_snapshot": snapshot,
        **summary,
    }
    if ctx is not None:
        ctx.counts.update(raw_rows=summary["raw_rows"], raw_batches_committed=summary["committed_batches"])
        ctx.details.update(batches_committed_this_run=committed_now, batches_skipped=skipped, mode=mode)
    return result


def committed_batch_files(settings: Settings) -> list[Path]:
    checkpoint = load_checkpoint(settings)
    if not checkpoint:
        return []
    ids = sorted(int(k) for k in checkpoint["committed_batches"])
    return [settings.root / checkpoint["committed_batches"][str(i)]["file"] for i in ids]


def read_raw(settings: Settings, require_complete: bool = True) -> pd.DataFrame:
    """Union of committed raw batches in source order."""
    checkpoint = load_checkpoint(settings)
    if checkpoint is None:
        raise FileNotFoundError("No ingestion checkpoint found: run `python -m src.cli ingest` first.")
    if require_complete and len(checkpoint["committed_batches"]) != checkpoint["num_batches"]:
        raise ContractViolation(
            f"Raw layer incomplete: {len(checkpoint['committed_batches'])}/{checkpoint['num_batches']} "
            "batches committed. Rerun ingest to resume."
        )
    frames = [pd.read_parquet(p) for p in committed_batch_files(settings)]
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return raw.sort_values("source_row_key", kind="stable").reset_index(drop=True)


def raw_summary(settings: Settings) -> dict[str, Any]:
    """Row/duplicate counts over the committed raw batches (used as rerun evidence)."""
    checkpoint = load_checkpoint(settings) or {"committed_batches": {}, "num_batches": 0}
    files = committed_batch_files(settings)
    if not files:
        return {"committed_batches": 0, "raw_rows": 0, "distinct_source_row_keys": 0,
                "duplicate_source_row_keys": 0, "duplicate_record_hashes": 0}
    cols = pd.concat([pd.read_parquet(p, columns=["source_row_key", "record_hash"]) for p in files])
    return {
        "committed_batches": len(checkpoint["committed_batches"]),
        "num_batches": checkpoint["num_batches"],
        "last_committed_batch_id": checkpoint.get("last_committed_batch_id"),
        "raw_rows": int(len(cols)),
        "distinct_source_row_keys": int(cols["source_row_key"].nunique()),
        "duplicate_source_row_keys": int(cols["source_row_key"].duplicated().sum()),
        "duplicate_record_hashes": int(cols["record_hash"].duplicated().sum()),
    }
