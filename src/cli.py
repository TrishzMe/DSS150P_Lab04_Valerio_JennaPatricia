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
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.cli", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="Path to settings.yml (default: config/settings.yml)")
    parser.add_argument("--run-id", help="Pipeline run id shared by all stages (default: $PIPELINE_RUN_ID or new)")
    sub = parser.add_subparsers(dest="command", required=True)
    parsers = {name: sub.add_parser(name, help=help_text) for name, (_, help_text) in COMMANDS.items()}

    p = parsers["ingest"]
    p.add_argument("--mode", choices=["incremental", "full"], default="incremental",
                   help="incremental: append uncommitted batches only; full: fresh run-specific copy + rebuild all batches")
    p.add_argument("--force", "--full-refresh", dest="force", action="store_true",
                   help="alias for --mode full")
    p.add_argument("--stop-after-batches", type=int, default=None,
                   help="simulate a crash after committing N batches (recovery demonstration)")
    parsers["validate-curated"].add_argument("--skip-database", action="store_true",
                                             help="skip the PostgreSQL consistency check (CON-04)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "force", False):
        args.mode = "full"
    try:
        settings = load_settings(config_path=args.config)
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
