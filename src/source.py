"""The immutable source dataset: manifest/contract access, identity checks and raw reading."""
from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .config import Settings
from .io_utils import sha256_file

RAW_COLUMNS = [
    "Country Code",
    "City",
    "Location",
    "Coordinates",
    "Pollutant",
    "Source Name",
    "Unit",
    "Value",
    "Last Updated",
    "Country Label",
]


class SourceIdentityError(RuntimeError):
    """The source file is missing or is not the snapshot declared in the manifest."""


class ContractViolation(RuntimeError):
    """Data violates a rule declared as `fail` in the data contract."""


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required metadata file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_manifest(settings: Settings) -> dict[str, Any]:
    return load_yaml(settings.manifest_path)


def load_contract(settings: Settings) -> dict[str, Any]:
    return load_yaml(settings.contract_path)


def inspect_file(path: Path, delimiter: str = ";") -> dict[str, Any]:
    """Byte-level facts about a delimited text file (no pandas involved)."""
    raw = path.read_bytes()
    has_bom = raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig")
    crlf = raw.count(b"\r\n")
    lf_only = raw.count(b"\n") - crlf
    header = next(csv.reader(io.StringIO(text.split("\n", 1)[0].rstrip("\r")), delimiter=delimiter))
    # Count data records with the csv module so quoted newlines would not be miscounted.
    record_count = sum(1 for _ in csv.reader(io.StringIO(text), delimiter=delimiter)) - 1
    return {
        "size_bytes": len(raw),
        "sha256": sha256_file(path),
        "encoding": "utf-8-sig" if has_bom else "utf-8",
        "has_utf8_bom": has_bom,
        "line_terminator": "CRLF" if crlf and not lf_only else ("LF" if lf_only and not crlf else "mixed"),
        "header": header,
        "column_count": len(header),
        "row_count": record_count,
    }


def verify_source(settings: Settings, path: Path | None = None) -> dict[str, Any]:
    """Prove the source file is exactly the snapshot recorded in source_manifest.yml.

    Raises SourceIdentityError on any mismatch (checksum, size, header or row count).
    """
    path = path or settings.source_path
    if not path.exists():
        raise SourceIdentityError(
            f"Source dataset not found at {path}. Place the file as described in data/source/README.md."
        )
    manifest = load_manifest(settings)
    expected = manifest["file"]
    facts = inspect_file(path, settings["source"]["delimiter"])
    problems = []
    if facts["sha256"] != expected["sha256"]:
        problems.append(f"sha256 {facts['sha256']} != manifest {expected['sha256']}")
    if facts["size_bytes"] != expected["size_bytes"]:
        problems.append(f"size {facts['size_bytes']} != manifest {expected['size_bytes']}")
    if facts["header"] != manifest["schema"]["columns"]:
        problems.append(f"header {facts['header']} != manifest {manifest['schema']['columns']}")
    if facts["row_count"] != expected["row_count"]:
        problems.append(f"row_count {facts['row_count']} != manifest {expected['row_count']}")
    if problems:
        raise SourceIdentityError("Source identity check failed: " + "; ".join(problems))
    return facts


def read_source_text(path: Path, settings: Settings) -> pd.DataFrame:
    """Read the source exactly as text: no type inference and no NA conversion.

    Null standardisation and typing happen later in staging, so raw keeps the original values.
    """
    src = settings["source"]
    frame = pd.read_csv(
        path,
        sep=src["delimiter"],
        encoding=src["encoding"],
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    return frame
