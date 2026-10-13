"""Command-line entry point: ``python -m src.cli <command> [options]``.

Every pipeline stage is a separate command so it can be run by hand, by ``run-all`` or by an
Airflow task. Exit status is 0 on success and non-zero on any unrecoverable error.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Callable

from .config import ConfigError, Settings, load_settings
from .observability import RunRecord, configure_logging, log_event, resolve_run_id, stage

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_CONFIG = 2
EXIT_INTERRUPTED = 3


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


# ------------------------------------------------------------------------------------- commands

def cmd_verify_source(settings: Settings, args: argparse.Namespace) -> Any:
    from .source import verify_source

    with stage(settings, args.run_id, "verify_source") as ctx:
        facts = verify_source(settings)
        ctx.counts["source_rows"] = facts["row_count"]
        ctx.details["sha256"] = facts["sha256"]
    return facts


def cmd_profile(settings: Settings, args: argparse.Namespace) -> Any:
    from .profile_source import write_profile

    with stage(settings, args.run_id, "profile") as ctx:
        profile = write_profile(settings)
        ctx.counts["source_rows"] = profile["file"]["row_count"]
    return {"profile": "metadata/source_profile.md", "rows": profile["file"]["row_count"]}


def cmd_ingest(settings: Settings, args: argparse.Namespace) -> Any:
    from .ingest import ingest

    RunRecord(settings, args.run_id).update_parameters(ingest_mode=args.mode)
    with stage(settings, args.run_id, "ingest_source") as ctx:
        return ingest(settings, args.run_id, mode=args.mode, stop_after_batches=args.stop_after_batches, ctx=ctx)


def cmd_validate_raw(settings: Settings, args: argparse.Namespace) -> Any:
    from .validate import validate_raw

    with stage(settings, args.run_id, "validate_raw") as ctx:
        report = validate_raw(settings, args.run_id, ctx)
    return report["summary"]


def cmd_transform_staging(settings: Settings, args: argparse.Namespace) -> Any:
    from .transform import build_staging

    with stage(settings, args.run_id, "transform_staging") as ctx:
        return build_staging(settings, args.run_id, ctx)


def cmd_build_curated(settings: Settings, args: argparse.Namespace) -> Any:
    from .transform import build_curated

    with stage(settings, args.run_id, "build_curated") as ctx:
        return build_curated(settings, args.run_id, ctx)


def cmd_validate_curated(settings: Settings, args: argparse.Namespace) -> Any:
    from .validate import validate_curated

    with stage(settings, args.run_id, "validate_curated") as ctx:
        report = validate_curated(settings, args.run_id, ctx, include_database=not args.skip_database)
    return report["summary"]


def cmd_init_db(settings: Settings, args: argparse.Namespace) -> Any:
    from .storage import init_db

    return init_db(settings)


def cmd_load_postgres(settings: Settings, args: argparse.Namespace) -> Any:
    from .storage import load_postgres

    with stage(settings, args.run_id, "load_postgres") as ctx:
        return load_postgres(settings, args.run_id, ctx)


def cmd_db_stats(settings: Settings, args: argparse.Namespace) -> Any:
    from .storage import db_stats

    return db_stats(settings)


def cmd_sync_audit(settings: Settings, args: argparse.Namespace) -> Any:
    from .storage import sync_run_record

    sync_run_record(settings, args.run_id)
    return {"synced_run": args.run_id}


def cmd_partition(settings: Settings, args: argparse.Namespace) -> Any:
    from .benchmark import verify_partition, write_partitions

    with stage(settings, args.run_id, "partition_staging") as ctx:
        result = write_partitions(settings)
        ctx.counts["partitions"] = len(result["partitions"])
        if args.select:
            result["selected_partition_check"] = check = verify_partition(settings, args.select)
            if not (check["row_count_matches"] and check["same_records"]):
                raise RuntimeError(f"Selected partition check failed: {check}")
    return result


def cmd_benchmark(settings: Settings, args: argparse.Namespace) -> Any:
    from .benchmark import run_benchmark

    with stage(settings, args.run_id, "storage_benchmark") as ctx:
        report = run_benchmark(settings, args.run_id, ctx)
    return {fmt: {"size_bytes": v["size_bytes"], "read_median_s": v["read_full"]["median_s"],
                  "equivalent": v["equivalent_to_staging"]} for fmt, v in report["formats"].items()}


def cmd_train(settings: Settings, args: argparse.Namespace) -> Any:
    from .train import train

    RunRecord(settings, args.run_id).update_parameters(impute_scope=settings["model"]["impute_scope"])
    with stage(settings, args.run_id, "train_model") as ctx:
        return train(settings, args.run_id, ctx)


def cmd_evaluate(settings: Settings, args: argparse.Namespace) -> Any:
    from .evaluate import evaluate

    with stage(settings, args.run_id, "evaluate_model") as ctx:
        return evaluate(settings, args.run_id, ctx)


def cmd_reproducibility_check(settings: Settings, args: argparse.Namespace) -> Any:
    from .evaluate import reproducibility_check

    with stage(settings, args.run_id, "reproducibility_check") as ctx:
        return reproducibility_check(settings, args.run_id, ctx)


def cmd_publish(settings: Settings, args: argparse.Namespace) -> Any:
    from .evaluate import publish, write_run_summary
    from .storage import sync_run_record

    with stage(settings, args.run_id, "publish_outputs") as ctx:
        result = publish(settings, args.run_id, ctx)
    RunRecord(settings, args.run_id).finalize("success")
    summary = write_run_summary(settings, args.run_id,
                                pg_predictions=result["summary"]["counts"]["postgres_prediction_rows"])
    sync_run_record(settings, args.run_id)
    return summary


PIPELINE_SEQUENCE = [
    ("ingest", "ingest_source"),
    ("validate-raw", "validate_raw"),
    ("transform-staging", "transform_staging"),
    ("build-curated", "build_curated"),
    ("partition", "partition_staging"),
    ("load-postgres", "load_postgres"),
    ("validate-curated", "validate_curated"),
    ("train", "train_model"),
    ("evaluate", "evaluate_model"),
    ("reproducibility-check", "reproducibility_check"),
    ("publish", "publish_outputs"),
]


def cmd_run_all(settings: Settings, args: argparse.Namespace) -> Any:
    """Run every stage in order with one pipeline_run_id; stop at the first failure."""
    record = RunRecord(settings, args.run_id)
    record.update_parameters(orchestrator="cli run-all", ingest_mode=args.mode,
                             impute_scope=settings["model"]["impute_scope"])
    defaults = argparse.Namespace(run_id=args.run_id, mode=args.mode, stop_after_batches=None, force=False,
                                  skip_database=False, select=settings["storage"]["benchmark_filter"]["pollutant"])
    for command, _ in PIPELINE_SEQUENCE:
        handler, _ = COMMANDS[command]
        try:
            handler(settings, defaults)
        except Exception:
            record.finalize("failed")
            try:
                from .storage import sync_run_record

                sync_run_record(settings, args.run_id)
            except Exception as sync_exc:  # noqa: BLE001 - the database may be the thing that failed
                log_event("audit_sync_failed", level=logging.WARNING, pipeline_run_id=args.run_id,
                          error=str(sync_exc))
            raise
    from .io_utils import read_json
    from .train import output_dir

    return read_json(output_dir(settings) / "run_summary.json")


def cmd_raw_stats(settings: Settings, args: argparse.Namespace) -> Any:
    from .ingest import load_checkpoint, raw_summary

    checkpoint = load_checkpoint(settings) or {}
    return {
        **raw_summary(settings),
        "checkpoint": {k: checkpoint.get(k) for k in
                       ("checkpoint_type", "last_committed_batch_id", "last_committed_source_row_key", "updated_at_utc")},
        "batches": {bid: {k: b[k] for k in ("row_start", "row_end", "row_count", "pipeline_run_id", "committed_at_utc")}
                    for bid, b in sorted(checkpoint.get("committed_batches", {}).items(), key=lambda kv: int(kv[0]))},
    }


Handler = Callable[[Settings, argparse.Namespace], Any]
COMMANDS: dict[str, tuple[Handler, str]] = {
    "verify-source": (cmd_verify_source, "Verify the source file against metadata/source_manifest.yml"),
    "profile": (cmd_profile, "Regenerate metadata/source_profile.md and .json"),
    "ingest": (cmd_ingest, "Ingest the source into the raw layer (full or incremental)"),
    "validate-raw": (cmd_validate_raw, "Run raw-layer data-quality checks"),
    "raw-stats": (cmd_raw_stats, "Show committed raw batches, row and duplicate counts"),
    "transform-staging": (cmd_transform_staging, "Raw -> staging (typed, deduplicated, quarantine)"),
    "build-curated": (cmd_build_curated, "Staging -> curated station-day and model-ready tables"),
    "validate-curated": (cmd_validate_curated, "Run curated-layer data-quality checks"),
    "init-db": (cmd_init_db, "Create PostgreSQL schemas, tables and views (idempotent)"),
    "load-postgres": (cmd_load_postgres, "Rerun-safe load of staging and curated tables into PostgreSQL"),
    "db-stats": (cmd_db_stats, "Show duplicate business keys and recent load statistics"),
    "sync-audit": (cmd_sync_audit, "Copy the run record of --run-id into audit.pipeline_runs/task_runs"),
    "partition": (cmd_partition, "Write pollutant-partitioned Parquet; --select VALUE loads one partition"),
    "benchmark": (cmd_benchmark, "CSV / JSONL / Parquet / PostgreSQL storage benchmark + report"),
    "train": (cmd_train, "Build model input (imputation + deterministic split) and fit the models"),
    "evaluate": (cmd_evaluate, "Evaluate on the held-out test set + 5-fold CV; write metrics/predictions"),
    "reproducibility-check": (cmd_reproducibility_check, "Compare metrics/split with the baseline (tolerance)"),
    "publish": (cmd_publish, "Publish predictions/metrics to PostgreSQL and close the run record"),
    "run-all": (cmd_run_all, "Run the whole pipeline once with a single run id"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.cli", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="Path to settings.yml (default: config/settings.yml)")
    parser.add_argument("--run-id", help="Pipeline run id shared by all stages (default: $PIPELINE_RUN_ID or new)")
    parser.add_argument("--impute-scope", choices=["baseline", "train_only"],
                        help="override model.impute_scope (baseline = original notebook behaviour)")
    sub = parser.add_subparsers(dest="command", required=True)
    parsers = {name: sub.add_parser(name, help=help_text) for name, (_, help_text) in COMMANDS.items()}

    for name in ("ingest", "run-all"):
        p = parsers[name]
        p.add_argument("--mode", choices=["incremental", "full"], default="incremental",
                       help="incremental: append uncommitted batches only; full: fresh run-specific copy + rebuild all batches")
        p.add_argument("--force", "--full-refresh", dest="force", action="store_true", help="alias for --mode full")
    parsers["ingest"].add_argument("--stop-after-batches", type=int, default=None,
                                   help="simulate a crash after committing N batches (recovery demonstration)")
    parsers["validate-curated"].add_argument("--skip-database", action="store_true",
                                             help="skip the PostgreSQL consistency check (CON-04)")
    parsers["partition"].add_argument("--select", help="partition value to load and verify, e.g. PM2.5")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "force", False):
        args.mode = "full"
    overrides = {"model": {"impute_scope": args.impute_scope}} if args.impute_scope else None
    try:
        settings = load_settings(config_path=args.config, overrides=overrides)
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    configure_logging(settings)
    args.run_id = resolve_run_id(args.run_id)
    handler, _ = COMMANDS[args.command]
    try:
        result = handler(settings, args)
    except Exception as exc:  # noqa: BLE001 - report every failure with context and a non-zero code
        from .ingest import SimulatedInterruption

        log_event("command_failed", level=logging.ERROR, command=args.command, pipeline_run_id=args.run_id,
                  error_type=type(exc).__name__, error=str(exc))
        return EXIT_INTERRUPTED if isinstance(exc, SimulatedInterruption) else EXIT_FAILED
    if result is not None:
        _print(result)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
