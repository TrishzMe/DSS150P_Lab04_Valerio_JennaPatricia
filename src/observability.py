"""Structured logging, pipeline run identifiers and per-run records.

Every log line is a JSON object carrying ``pipeline_run_id`` and ``stage`` so one run can be
traced across CLI calls and Airflow tasks. Each stage also updates a run record
(``outputs/runs/<run>/run_record.json``) with status, duration, row counts and errors; the
record is later mirrored into PostgreSQL (``audit.pipeline_runs`` / ``audit.task_runs``).
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .config import Settings
from .io_utils import atomic_write_json, read_json

LOGGER_NAME = "repro"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat(timespec="milliseconds")


def new_run_id(prefix: str = "manual") -> str:
    return f"{prefix}__{utc_now():%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:6]}"


def resolve_run_id(cli_value: str | None = None) -> str:
    return cli_value or os.environ.get("PIPELINE_RUN_ID") or new_run_id()


def run_slug(run_id: str) -> str:
    """Filesystem-safe form of a run id (Airflow run ids contain ':' and '+')."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id).strip("_")[:150]


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        if record.exc_info:
            payload["error_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
            payload["traceback"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(settings: Settings, level: int = logging.INFO) -> logging.Logger:
    """JSON lines to stdout (visible in Airflow task logs) and to logs/pipeline.jsonl."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    formatter = JsonFormatter()
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    log_dir = settings.path("logs_dir")
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "pipeline.jsonl", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def log_event(event: str, level: int = logging.INFO, exc_info: bool = False, **fields: Any) -> None:
    get_logger().log(level, event, extra={"fields": fields}, exc_info=exc_info)


@contextmanager
def file_lock(path: Path, timeout_s: float = 60.0, stale_s: float = 300.0) -> Iterator[None]:
    """Cross-process lock (parallel Airflow tasks update the same run record)."""
    lock = path.with_name(path.name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > stale_s:  # holder crashed
                    lock.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() > deadline:
                raise TimeoutError(f"Could not acquire {lock} within {timeout_s}s")
            time.sleep(0.05)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


class RunRecord:
    """JSON run record shared by all stages of one pipeline run (updates are lock-protected)."""

    def __init__(self, settings: Settings, run_id: str):
        self.settings = settings
        self.run_id = run_id
        self.path = settings.path("outputs_dir") / "runs" / run_slug(run_id) / "run_record.json"

    def _update(self, mutate) -> dict[str, Any]:
        with file_lock(self.path):
            record = self.load()
            mutate(record)
            self.save(record)
            return record

    def load(self) -> dict[str, Any]:
        if self.path.exists():
            return read_json(self.path)
        return {
            "pipeline_run_id": self.run_id,
            "started_at_utc": utc_now_iso(),
            "ended_at_utc": None,
            "status": "running",
            "parameters": {},
            "counts": {},
            "stages": {},
        }

    def save(self, record: dict[str, Any]) -> None:
        atomic_write_json(self.path, record)

    def update_parameters(self, **params: Any) -> None:
        self._update(lambda r: r["parameters"].update({k: v for k, v in params.items() if v is not None}))

    def start_stage(self, name: str) -> None:
        def mutate(record: dict[str, Any]) -> None:
            previous = record["stages"].get(name, {})
            record["stages"][name] = {
                "status": "running",
                "attempt": int(previous.get("attempt", 0)) + 1,
                "started_at_utc": utc_now_iso(),
                "ended_at_utc": None,
                "duration_s": None,
                "row_counts": {},
                "error": None,
            }
            record["status"] = "running"
            record["ended_at_utc"] = None

        self._update(mutate)

    def finish_stage(
        self, name: str, status: str, duration_s: float, counts: dict[str, Any], error: str | None = None
    ) -> dict[str, Any]:
        def mutate(record: dict[str, Any]) -> None:
            stage = record["stages"].setdefault(name, {"attempt": 1, "started_at_utc": utc_now_iso()})
            stage.update(status=status, ended_at_utc=utc_now_iso(), duration_s=round(duration_s, 4),
                         row_counts=counts, error=error)
            if status == "success":
                record["counts"].update({k: v for k, v in counts.items() if isinstance(v, (int, float))})
            if status == "failed":
                record["status"] = "failed"
                record["ended_at_utc"] = utc_now_iso()

        return self._update(mutate)

    def finalize(self, status: str) -> dict[str, Any]:
        def mutate(record: dict[str, Any]) -> None:
            record["status"] = status
            record["ended_at_utc"] = utc_now_iso()

        return self._update(mutate)


@dataclass
class StageContext:
    run_id: str
    name: str
    counts: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


@contextmanager
def stage(settings: Settings, run_id: str, name: str) -> Iterator[StageContext]:
    """Time a pipeline stage, log start/end/failure, and update the run record."""
    run_record = RunRecord(settings, run_id)
    run_record.start_stage(name)
    ctx = StageContext(run_id=run_id, name=name)
    log_event("stage_started", pipeline_run_id=run_id, stage=name)
    started = time.perf_counter()
    try:
        yield ctx
    except BaseException as exc:
        duration = time.perf_counter() - started
        message = f"{type(exc).__name__}: {exc}"
        run_record.finish_stage(name, "failed", duration, ctx.counts, error=message)
        log_event(
            "stage_failed",
            level=logging.ERROR,
            pipeline_run_id=run_id,
            stage=name,
            duration_s=round(duration, 4),
            row_counts=ctx.counts,
            error=message,
            traceback=traceback.format_exc(limit=5),
        )
        raise
    duration = time.perf_counter() - started
    run_record.finish_stage(name, "success", duration, ctx.counts)
    log_event(
        "stage_succeeded",
        pipeline_run_id=run_id,
        stage=name,
        duration_s=round(duration, 4),
        row_counts=ctx.counts,
        **ctx.details,
    )
