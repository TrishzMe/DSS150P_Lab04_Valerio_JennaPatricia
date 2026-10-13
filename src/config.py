"""Configuration loading and validation.

Settings come from ``config/settings.yml``; credentials and machine-specific values come from
environment variables (optionally loaded from ``.env``). Every path is resolved against the
project root, which defaults to this repository and can be redirected with
``REPRO_PROJECT_ROOT`` (tests and failure drills use this to run in an isolated workspace).
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "settings.yml"

VALID_IMPUTE_SCOPES = ("baseline", "train_only")
REQUIRED_SECTIONS = ("paths", "source", "ingestion", "transform", "model", "reproducibility", "storage", "database")


class ConfigError(ValueError):
    """Raised when configuration is missing or invalid."""


@dataclass(frozen=True)
class DatabaseSettings:
    host: str
    port: int
    dbname: str
    user: str
    password: str
    connect_timeout_s: int

    def connect_kwargs(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "dbname": self.dbname,
            "user": self.user,
            "password": self.password,
            "connect_timeout": self.connect_timeout_s,
        }


@dataclass(frozen=True)
class Settings:
    root: Path
    data: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def path(self, key: str) -> Path:
        """Resolve an entry of the ``paths`` section against the project root."""
        return self.root / self.data["paths"][key]

    def rel(self, relative: str) -> Path:
        return self.root / relative

    @property
    def source_path(self) -> Path:
        return self.path("source_dir") / self.data["source"]["filename"]

    @property
    def manifest_path(self) -> Path:
        return self.rel(self.data["source"]["manifest_file"])

    @property
    def contract_path(self) -> Path:
        return self.rel(self.data["source"]["contract_file"])

    @property
    def checkpoint_path(self) -> Path:
        return self.rel(self.data["ingestion"]["checkpoint_file"])

    def database(self) -> DatabaseSettings:
        """Database settings from the environment; raises ConfigError if a value is missing."""
        missing = [k for k in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB") if not os.environ.get(k)]
        if missing:
            raise ConfigError(
                f"Missing database environment variables: {', '.join(missing)}. "
                "Copy .env.example to .env and fill in the values."
            )
        port = os.environ.get("POSTGRES_PORT", "5432")
        try:
            port_int = int(port)
        except ValueError as exc:
            raise ConfigError(f"POSTGRES_PORT must be an integer, got {port!r}") from exc
        return DatabaseSettings(
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=port_int,
            dbname=os.environ["POSTGRES_DB"],
            user=os.environ["POSTGRES_USER"],
            password=os.environ["POSTGRES_PASSWORD"],
            connect_timeout_s=int(self.data["database"].get("connect_timeout_s", 10)),
        )


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_dotenv(root: Path) -> None:
    env_file = root / ".env"
    if env_file.exists():
        from dotenv import load_dotenv

        # Real environment variables (e.g. set by docker-compose) win over the .env file.
        load_dotenv(env_file, override=False)


def validate_settings(data: dict[str, Any]) -> None:
    """Fail fast with a readable message instead of failing deep inside a pipeline stage."""
    errors: list[str] = []
    for section in REQUIRED_SECTIONS:
        if not isinstance(data.get(section), dict):
            errors.append(f"missing section '{section}'")
    if errors:
        raise ConfigError("Invalid configuration: " + "; ".join(errors))

    model = data["model"]
    split = model.get("split", {})
    test_size = split.get("test_size")
    if not isinstance(test_size, (int, float)) or not 0 < test_size < 1:
        errors.append(f"model.split.test_size must be between 0 and 1, got {test_size!r}")
    if model.get("impute_scope") not in VALID_IMPUTE_SCOPES:
        errors.append(f"model.impute_scope must be one of {VALID_IMPUTE_SCOPES}, got {model.get('impute_scope')!r}")
    features = model.get("features") or []
    if not features:
        errors.append("model.features must not be empty")
    leaking = sorted(set(features) & set(model.get("leakage_columns") or []))
    if leaking:
        errors.append(f"model.features contains leakage columns {leaking}")
    if len(set(features)) != len(features):
        errors.append("model.features contains duplicates")
    if not model.get("models"):
        errors.append("model.models must define at least one estimator")
    for key in ("random_state",):
        if not isinstance(split.get(key), int):
            errors.append(f"model.split.{key} must be an integer seed")
    cv = model.get("cv", {})
    if not isinstance(cv.get("n_splits"), int) or cv.get("n_splits", 0) < 2:
        errors.append("model.cv.n_splits must be an integer >= 2")

    num_batches = data["ingestion"].get("num_batches")
    if not isinstance(num_batches, int) or num_batches < 3:
        errors.append(f"ingestion.num_batches must be an integer >= 3, got {num_batches!r}")

    tol = data["reproducibility"].get("tolerance_abs")
    if not isinstance(tol, (int, float)) or not 0 <= tol <= 0.01:
        errors.append(f"reproducibility.tolerance_abs must be within [0, 0.01], got {tol!r}")

    rate = data["transform"].get("max_invalid_row_rate")
    if not isinstance(rate, (int, float)) or not 0 <= rate < 1:
        errors.append(f"transform.max_invalid_row_rate must be within [0, 1), got {rate!r}")

    repeats = data["storage"].get("benchmark_repeats")
    if not isinstance(repeats, int) or repeats < 5:
        errors.append(f"storage.benchmark_repeats must be an integer >= 5, got {repeats!r}")

    for key, rel in data["paths"].items():
        if Path(rel).is_absolute():
            errors.append(f"paths.{key} must be relative to the project root, got {rel!r}")

    if errors:
        raise ConfigError("Invalid configuration: " + "; ".join(errors))


def load_settings(
    config_path: str | Path | None = None,
    root: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Settings:
    """Load, merge and validate settings.

    ``root`` defaults to ``REPRO_PROJECT_ROOT`` or the repository root. ``config_path``
    defaults to ``REPRO_CONFIG`` or ``<root>/config/settings.yml``.
    """
    root_path = Path(root or os.environ.get("REPRO_PROJECT_ROOT") or REPO_ROOT).resolve()
    cfg_path = Path(config_path or os.environ.get("REPRO_CONFIG") or root_path / "config" / "settings.yml")
    if not cfg_path.is_absolute():
        cfg_path = root_path / cfg_path
    if not cfg_path.exists():
        raise ConfigError(f"Configuration file not found: {cfg_path}")
    with cfg_path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Configuration file {cfg_path} must contain a mapping")
    if overrides:
        data = _deep_merge(data, overrides)
    validate_settings(data)
    _load_dotenv(root_path)
    return Settings(root=root_path, data=data)
