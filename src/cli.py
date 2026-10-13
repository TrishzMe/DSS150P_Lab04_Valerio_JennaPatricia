"""Command-line entry point: ``python -m src.cli <command> [options]``.

Every pipeline stage is a separate command so it can be run by hand, by ``run-all`` or by an
Airflow task. Exit status is 0 on success and non-zero on any unrecoverable error.
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, Callable

from .config import ConfigError, Settings, load_settings
from .observability import configure_logging, log_event


def cmd_verify_source(settings: Settings, args: argparse.Namespace) -> dict[str, Any]:
    from .source import verify_source

    facts = verify_source(settings)
    log_event("source_verified", stage="verify_source", sha256=facts["sha256"], row_count=facts["row_count"])
    return facts


def cmd_profile(settings: Settings, args: argparse.Namespace) -> dict[str, Any]:
    from .profile_source import write_profile

    profile = write_profile(settings)
    log_event("source_profiled", stage="profile", rows=profile["file"]["row_count"])
    return {"rows": profile["file"]["row_count"]}


COMMANDS: dict[str, tuple[Callable[[Settings, argparse.Namespace], Any], str]] = {
    "verify-source": (cmd_verify_source, "Verify the source file against metadata/source_manifest.yml"),
    "profile": (cmd_profile, "Regenerate metadata/source_profile.md and .json"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.cli", description=__doc__)
    parser.add_argument("--config", help="Path to settings.yml (default: config/settings.yml)")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (_, help_text) in COMMANDS.items():
        sub.add_parser(name, help=help_text)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = load_settings(config_path=args.config)
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 2
    configure_logging(settings)
    handler, _ = COMMANDS[args.command]
    try:
        handler(settings, args)
    except Exception as exc:  # noqa: BLE001 - report every failure with context and a non-zero code
        log_event("command_failed", level=logging.ERROR, command=args.command, error=f"{type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
