"""Shared fixtures: an isolated mini project (synthetic source + real config/contract) per test."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from src.config import REPO_ROOT, Settings, load_settings
from src.source import RAW_COLUMNS, inspect_file

HEADER = ";".join(RAW_COLUMNS)


def _row(code, city, loc, pollutant, unit, value, ts, label):
    return ";".join([code, city, loc, "1.0, 2.0", pollutant, "test-source", unit, str(value), ts, label])


def synthetic_rows() -> list[str]:
    """Small source covering every staging rule (scope, dedup, units, negatives, null tokens)."""
    rows = []
    ts = "2024-03-10T13:30:00+05:30"   # 08:00 UTC, same date
    # 12 complete station-days: PM2.5 alternates Safe / Unhealthy, NO2 and O3 always present.
    for i in range(12):
        loc = f"Station {i:02d}"
        pm25 = 8.0 if i % 2 == 0 else 30.0
        rows += [
            _row("JP", "Tokyo", loc, "PM2.5", "µg/m³", pm25, ts, "Japan"),
            _row("JP", "Tokyo", loc, "NO2", "µg/m³", 10 + i, ts, "Japan"),
            _row("JP", "Tokyo", loc, "O3", "µg/m³", 50 + i, ts, "Japan"),
            _row("JP", "Tokyo", loc, "PM10", "µg/m³", pm25 * 2, ts, "Japan"),
        ]
        if i % 3 == 0:
            rows.append(_row("JP", "Tokyo", loc, "SO2", "µg/m³", 4, ts, "Japan"))
        if i % 4 == 0:
            rows.append(_row("JP", "Tokyo", loc, "CO", "µg/m³", 500, ts, "Japan"))
    rows += [
        _row("JP", "Tokyo", "Station 00", "PM2.5", "µg/m³", 99.0, ts, "Japan"),   # duplicate reading -> later one dropped
        _row("JP", "Tokyo", "Station 01", "NO2", "ppm", 0.02, "2024-03-10T14:30:00+05:30", "Japan"),  # ppm -> excluded
        _row("JP", "Tokyo", "Station 02", "O3", "µg/m³", -5, "2024-03-10T15:30:00+05:30", "Japan"),  # negative -> quarantine
        _row("JP", "Tokyo", "Station 03", "NO", "µg/m³", 3, ts, "Japan"),        # out-of-scope pollutant
        _row("CL", "N/A", "Santiago Centro", "PM2.5", "µg/m³", 40, ts, "Chile"),  # 'N/A' city -> Unknown
        _row("CL", "N/A", "Santiago Centro", "NO2", "µg/m³", 30, ts, "Chile"),
        _row("CL", "N/A", "Santiago Centro", "O3", "µg/m³", 20, ts, "Chile"),
        _row("XX", "", "Nowhere", "PM2.5", "µg/m³", 12, ts, ""),                 # empty country label -> Unknown
    ]
    return rows


def write_source(path: Path, rows: list[str]) -> None:
    text = "\r\n".join([HEADER, *rows]) + "\r\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))


def register_manifest(root: Path, source: Path) -> None:
    facts = inspect_file(source)
    manifest = yaml.safe_load((REPO_ROOT / "metadata" / "source_manifest.yml").read_text(encoding="utf-8"))
    manifest["file"].update(size_bytes=facts["size_bytes"], row_count=facts["row_count"], sha256=facts["sha256"],
                            column_count=facts["column_count"])
    (root / "metadata" / "source_manifest.yml").write_text(yaml.safe_dump(manifest, allow_unicode=True), encoding="utf-8")


@pytest.fixture
def mini_root(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "config").mkdir(parents=True)
    (root / "metadata").mkdir()
    cfg = yaml.safe_load((REPO_ROOT / "config" / "settings.yml").read_text(encoding="utf-8"))
    cfg["ingestion"]["num_batches"] = 3
    (root / "config" / "settings.yml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    for name in ("data_contract.yml", "expected_metrics.json", "split_manifest.json"):
        shutil.copy(REPO_ROOT / "metadata" / name, root / "metadata" / name)
    source = root / "data" / "source" / "world_air_quality.csv"
    write_source(source, synthetic_rows())
    register_manifest(root, source)
    return root


@pytest.fixture
def mini_settings(mini_root: Path) -> Settings:
    return load_settings(root=mini_root, config_path=mini_root / "config" / "settings.yml")


@pytest.fixture
def repo_settings() -> Settings:
    return load_settings(root=REPO_ROOT)
