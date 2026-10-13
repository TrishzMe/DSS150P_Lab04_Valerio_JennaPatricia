"""PostgreSQL storage: schema setup, rerun-safe snapshot loads, consistency checks and audit sync.

Load algorithm (one transaction for all tables, so a load is all-or-nothing):
  1. COPY the new snapshot into a temporary table shaped like the target;
  2. INSERT ... ON CONFLICT (key) DO UPDATE ... WHERE row_hash changed   (sql/transformations.sql);
  3. DELETE target rows whose key is not in the snapshot;
  4. record inserted / updated / deleted / unchanged counts in audit.load_log.
Running it again with the same data inserts 0, updates 0 and deletes 0 rows. If Airflow retries
after the database committed but the task never received the acknowledgement, the retry sees
identical row hashes and changes nothing, so no duplicate logical records can appear.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from .config import Settings
from .io_utils import digest_of_keys, hash_rows, read_json
from .observability import RunRecord, StageContext, log_event

SQL_DIR = Path(__file__).resolve().parents[1] / "sql"


@dataclass(frozen=True)
class TableSpec:
    name: str              # schema-qualified table
    key: str               # primary-key column used for the merge
    columns: list[str]     # columns written by the pipeline (in COPY order)
    hash_columns: list[str]


STAGING_BUSINESS = ["source_row_key", "source_batch_id", "record_hash", "country_code", "city", "location",
                    "coordinates", "pollutant", "source_name", "unit", "value", "last_updated_utc",
                    "reading_date", "country_label"]
STATION_DAY_BUSINESS = ["station_day_key", "country_label", "city", "location", "reading_date",
                        "pm25", "pm10", "no2", "o3", "so2", "co", "n_readings"]
MODEL_READY_BUSINESS = ["station_day_key", "country_label", "city", "location", "reading_date", "pm25", "pm10",
                        "no2", "o3", "so2", "co", "so2_missing", "co_missing", "ox_total_oxidant",
                        "no2_o3_ratio", "air_quality"]
AUDIT = ["row_hash", "pipeline_run_id", "processed_at_utc"]

TABLES = {
    "staging": TableSpec("staging.air_quality_readings", "source_row_key", STAGING_BUSINESS + AUDIT, STAGING_BUSINESS),
    "station_day": TableSpec("curated.station_day_pollutants", "station_day_key", STATION_DAY_BUSINESS + AUDIT,
                             STATION_DAY_BUSINESS),
    "model_ready": TableSpec("curated.model_ready", "station_day_key", MODEL_READY_BUSINESS + AUDIT,
                             MODEL_READY_BUSINESS),
}
LOAD_ORDER = ["staging", "station_day", "model_ready"]  # parents before children (foreign keys)


class DatabaseUnavailable(RuntimeError):
    """PostgreSQL could not be reached after retries."""


def sql_blocks(path: Path | None = None) -> dict[str, str]:
    """Split a .sql file into named blocks introduced by '-- name: <id>'."""
    text = (path or SQL_DIR / "transformations.sql").read_text(encoding="utf-8")
    blocks: dict[str, str] = {}
    for part in re.split(r"^-- name: ", text, flags=re.MULTILINE)[1:]:
        name, _, body = part.partition("\n")
        blocks[name.strip()] = body.strip()
    return blocks


def connect(settings: Settings, attempts: int = 5, delay_s: float = 2.0):
    import psycopg

    db = settings.database()
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return psycopg.connect(**db.connect_kwargs())
        except psycopg.OperationalError as exc:
            last_error = exc
            log_event("db_connect_failed", level=30, attempt=attempt, host=db.host, port=db.port,
                      error=str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__)
            if attempt < attempts:
                time.sleep(delay_s * attempt)
    raise DatabaseUnavailable(
        f"PostgreSQL at {db.host}:{db.port}/{db.dbname} unavailable after {attempts} attempts: {last_error}"
    )


def init_db(settings: Settings) -> dict[str, Any]:
    schema_sql = (SQL_DIR / "schema.sql").read_text(encoding="utf-8")
    with connect(settings) as conn:
        conn.execute(schema_sql)
        conn.execute(sql_blocks()["create_views"])
    return {"schema": "applied", "views": "created"}


def _records(frame: pd.DataFrame, columns: list[str]) -> Iterator[tuple]:
    """DataFrame rows as Python objects with NaN/NaT -> None (COPY-friendly)."""
    sub = frame[columns].astype(object)
    sub = sub.where(pd.notna(sub), None)
    return sub.itertuples(index=False, name=None)


def prepare_frame(spec: TableSpec, frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["row_hash"] = hash_rows(out, spec.hash_columns).astype(str).values
    return out


def load_table(conn, spec: TableSpec, frame: pd.DataFrame, run_id: str) -> dict[str, Any]:
    """Merge one snapshot into its table. Must be called inside an open transaction."""
    from psycopg import sql

    started = time.perf_counter()
    blocks = sql_blocks()
    tmp = f"tmp_{spec.name.replace('.', '_')}"
    target = sql.SQL(".").join(sql.Identifier(p) for p in spec.name.split("."))
    tmp_ident = sql.Identifier(tmp)
    cols = sql.SQL(", ").join(sql.Identifier(c) for c in spec.columns)
    conn.execute(sql.SQL("CREATE TEMP TABLE {} (LIKE {} INCLUDING DEFAULTS) ON COMMIT DROP").format(tmp_ident, target))
    with conn.cursor().copy(sql.SQL("COPY {} ({}) FROM STDIN").format(tmp_ident, cols)) as copy:
        for row in _records(frame, spec.columns):
            copy.write_row(row)
    update_set = sql.SQL(", ").join(
        sql.SQL("{} = EXCLUDED.{}").format(sql.Identifier(c), sql.Identifier(c)) for c in spec.columns if c != spec.key
    )
    merge = sql.SQL(blocks["merge_snapshot"]).format(
        target=target, source=tmp_ident, columns=cols, key=sql.Identifier(spec.key), update_set=update_set)
    inserted_flags = [r[0] for r in conn.execute(merge).fetchall()]
    deleted = conn.execute(sql.SQL(blocks["delete_missing"]).format(
        target=target, source=tmp_ident, key=sql.Identifier(spec.key))).rowcount
    total_after = conn.execute(sql.SQL("SELECT count(*) FROM {}").format(target)).fetchone()[0]
    inserted = sum(1 for f in inserted_flags if f)
    updated = len(inserted_flags) - inserted
    stats = {
        "table": spec.name,
        "snapshot_rows": int(len(frame)),
        "rows_inserted": inserted,
        "rows_updated": updated,
        "rows_deleted": int(deleted),
        "rows_unchanged": int(len(frame)) - len(inserted_flags),
        "table_rows_after": int(total_after),
        "duration_s": round(time.perf_counter() - started, 4),
    }
    conn.execute(
        "INSERT INTO audit.load_log (pipeline_run_id, table_name, snapshot_rows, rows_inserted, rows_updated, "
        "rows_deleted, rows_unchanged, table_rows_after, duration_s) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        (run_id, spec.name, stats["snapshot_rows"], inserted, updated, stats["rows_deleted"],
         stats["rows_unchanged"], stats["table_rows_after"], stats["duration_s"]),
    )
    return stats


def snapshot_frames(settings: Settings) -> dict[str, pd.DataFrame]:
    from .transform import read_model_ready, read_staging, station_day_path

    return {
        "staging": read_staging(settings),
        "station_day": pd.read_parquet(station_day_path(settings)),
        "model_ready": read_model_ready(settings),
    }


def load_postgres(settings: Settings, run_id: str, ctx: StageContext | None = None) -> dict[str, Any]:
    init_db(settings)
    frames = snapshot_frames(settings)
    results = []
    with connect(settings) as conn:
        with conn.transaction():
            for name in LOAD_ORDER:
                spec = TABLES[name]
                stats = load_table(conn, spec, prepare_frame(spec, frames[name]), run_id)
                results.append(stats)
                log_event("table_loaded", pipeline_run_id=run_id, stage="load_postgres", **stats)
    sync_run_record(settings, run_id)
    if ctx is not None:
        ctx.counts.update({f"pg_{r['table'].split('.')[1]}_rows": r["table_rows_after"] for r in results})
        ctx.details.update(loads=results)
    return {"loads": results}


def sync_run_record(settings: Settings, run_id: str) -> None:
    """Mirror outputs/runs/<run>/run_record.json into audit.pipeline_runs / audit.task_runs."""
    path = RunRecord(settings, run_id).path
    if not path.exists():
        return
    record = read_json(path)
    from .source import load_manifest

    checksum = load_manifest(settings)["file"]["sha256"]
    with connect(settings) as conn, conn.transaction():
        conn.execute(
            """INSERT INTO audit.pipeline_runs (pipeline_run_id, status, started_at_utc, ended_at_utc,
                   source_checksum, parameters, counts, updated_at_utc)
               VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, now())
               ON CONFLICT (pipeline_run_id) DO UPDATE SET status = EXCLUDED.status,
                   started_at_utc = EXCLUDED.started_at_utc, ended_at_utc = EXCLUDED.ended_at_utc,
                   parameters = EXCLUDED.parameters, counts = EXCLUDED.counts, updated_at_utc = now()""",
            (run_id, record["status"], record["started_at_utc"], record["ended_at_utc"], checksum,
             json.dumps(record.get("parameters", {})), json.dumps(record.get("counts", {}))),
        )
        for task, info in record["stages"].items():
            conn.execute(
                """INSERT INTO audit.task_runs (pipeline_run_id, task_name, attempt, status, started_at_utc,
                       ended_at_utc, duration_s, row_counts, error, updated_at_utc)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, now())
                   ON CONFLICT (pipeline_run_id, task_name) DO UPDATE SET attempt = EXCLUDED.attempt,
                       status = EXCLUDED.status, started_at_utc = EXCLUDED.started_at_utc,
                       ended_at_utc = EXCLUDED.ended_at_utc, duration_s = EXCLUDED.duration_s,
                       row_counts = EXCLUDED.row_counts, error = EXCLUDED.error, updated_at_utc = now()""",
                (run_id, task, info.get("attempt", 1), info.get("status"), info.get("started_at_utc"),
                 info.get("ended_at_utc"), info.get("duration_s"), json.dumps(info.get("row_counts", {})),
                 info.get("error")),
            )


def read_table(settings: Settings, table: str, columns: list[str], order_by: str) -> pd.DataFrame:
    from psycopg import sql

    with connect(settings) as conn:
        query = sql.SQL("SELECT {} FROM {} ORDER BY {}").format(
            sql.SQL(", ").join(sql.Identifier(c) for c in columns),
            sql.SQL(".").join(sql.Identifier(p) for p in table.split(".")),
            sql.Identifier(order_by),
        )
        cur = conn.execute(query)
        return pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Comparable representation for Parquet vs PostgreSQL values (types differ, values must not)."""
    out = pd.DataFrame(index=range(len(frame)))
    for col in frame.columns:
        s = frame[col].reset_index(drop=True)
        if col == "reading_date":
            out[col] = pd.to_datetime(s.astype(str)).dt.strftime("%Y-%m-%d")
        elif col == "last_updated_utc":
            out[col] = pd.to_datetime(s, utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        elif pd.api.types.is_numeric_dtype(s) or s.map(lambda v: isinstance(v, (int, float)) or v is None).all():
            out[col] = pd.to_numeric(s).astype("float64")
        else:
            out[col] = s.astype(object).where(s.notna(), None).map(lambda v: None if v is None else str(v).strip())
    return out


def database_consistency_check(settings: Settings):
    """CON-04: every PostgreSQL table equals its Parquet snapshot (counts, key checksum, values)."""
    from .validate import _check

    frames = snapshot_frames(settings)
    observed: dict[str, Any] = {}
    ok = True
    for name in LOAD_ORDER:
        spec = TABLES[name]
        parquet = frames[name].sort_values(spec.key, kind="stable")[spec.hash_columns]
        pg = read_table(settings, spec.name, spec.hash_columns, spec.key)
        # Keys are compared as text: CHAR(64)/BIGINT in PostgreSQL vs string/int64 in Parquet.
        same_keys = digest_of_keys(parquet[spec.key].astype(str)) == digest_of_keys(pg[spec.key].astype(str).str.strip())
        values_equal = len(parquet) == len(pg) and _normalise(parquet).equals(_normalise(pg))
        observed[spec.name] = {"parquet_rows": int(len(parquet)), "postgres_rows": int(len(pg)),
                               "key_checksum_equal": same_keys, "values_equal": values_equal}
        ok &= len(parquet) == len(pg) and same_keys and values_equal
    with connect(settings) as conn:
        cur = conn.execute(sql_blocks()["elt_consistency"])
        elt = dict(zip([d.name for d in cur.description], cur.fetchone()))
    elt = {k: float(v) if k == "max_abs_diff" else int(v) for k, v in elt.items()}
    elt_ok = (elt["only_in_sql_view"] == 0 and elt["only_in_curated_table"] == 0 and elt["null_mismatches"] == 0
              and elt["n_readings_mismatches"] == 0 and elt["max_abs_diff"] <= 1e-6)
    observed["sql_elt_vs_python_station_day"] = elt
    return _check("CON-04", "PostgreSQL tables equal Parquet files; SQL ELT matches Python", "consistency",
                  "curated", ok and elt_ok, observed,
                  {"rows/keys/values": "equal", "elt_max_abs_diff": "<= 1e-6", "elt_mismatches": 0})


def db_stats(settings: Settings) -> dict[str, Any]:
    with connect(settings) as conn:
        cur = conn.execute(sql_blocks()["duplicate_business_keys"])
        dup = [dict(zip([d.name for d in cur.description], r)) for r in cur.fetchall()]
        cur = conn.execute(
            "SELECT pipeline_run_id, table_name, snapshot_rows, rows_inserted, rows_updated, rows_deleted, "
            "rows_unchanged, table_rows_after, loaded_at_utc FROM audit.load_log ORDER BY load_id DESC LIMIT 6")
        loads = [dict(zip([d.name for d in cur.description], r)) for r in cur.fetchall()]
        runs = conn.execute("SELECT count(*) FROM audit.pipeline_runs").fetchone()[0]
    return {"duplicate_business_keys": dup, "recent_loads": loads, "audited_runs": runs}
