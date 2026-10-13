"""Small I/O helpers: checksums, deterministic hashing and crash-safe (atomic) writes."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

HASH_FIELD_SEPARATOR = "\x1f"  # ASCII unit separator: cannot appear in the CSV text fields


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_fields(values: Iterable[Any]) -> str:
    """Deterministic SHA-256 of an ordered list of field values (None -> empty string)."""
    return sha256_text(HASH_FIELD_SEPARATOR.join("" if v is None else str(v) for v in values))


def hash_rows(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Row-wise deterministic hash over ``columns`` (used for record_hash / business keys).

    Equivalent to ``hash_fields(row[columns])`` for every row, but vectorised.
    """
    parts = [frame[c].astype("string").fillna("") for c in columns]
    joined = parts[0]
    for part in parts[1:]:
        joined = joined + HASH_FIELD_SEPARATOR + part
    return pd.Series([sha256_text(s) for s in joined], index=frame.index, dtype="string")


def digest_of_keys(keys: Iterable[str]) -> str:
    """Order-independent checksum of a set of keys (sorted, newline-joined, SHA-256)."""
    return sha256_text("\n".join(sorted(keys)))


def _tmp_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp-{os.getpid()}")


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    with tmp.open("wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)


def atomic_write_parquet(path: str | Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    frame.to_parquet(tmp, index=False, engine="pyarrow")
    os.replace(tmp, path)


def atomic_write_csv(path: str | Path, frame: pd.DataFrame, **kwargs: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    frame.to_csv(tmp, index=False, lineterminator="\n", **kwargs)
    os.replace(tmp, path)
