"""Storage experiment: the same logical staging records as CSV, JSONL, Parquet and PostgreSQL,
plus a pollutant-partitioned Parquet dataset.

Method
* Dataset: data/staging/air_quality_readings.parquet (all columns), identical for every format.
* Every write and read is timed ``storage.benchmark_repeats`` times (>= 5); the median is reported.
  One extra untimed-for-median "cold" read happens first and is reported separately.
* Each format is read back and compared with the original (row count, key checksum, value sum)
  so the comparison is between equivalent logical records.
* PostgreSQL: COPY into benchmark.staging_copy (write), a full-table SELECT and a selective
  SELECT on (pollutant, country_label), which has an index.
"""
from __future__ import annotations

import os
import platform
import shutil
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .config import Settings
from .io_utils import atomic_write_json, atomic_write_text, digest_of_keys, replace_file
from .observability import StageContext, log_event, utc_now_iso
from .transform import read_staging

PARTITION_DATASET = "staging_by_pollutant"


def _timed(fn: Callable[[], Any], repeats: int) -> tuple[list[float], Any]:
    times, result = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return times, result


def _summary(times: list[float]) -> dict[str, Any]:
    return {"median_s": round(statistics.median(times), 5), "min_s": round(min(times), 5),
            "max_s": round(max(times), 5), "runs": len(times)}


def _fingerprint(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "key_checksum": digest_of_keys(frame["source_row_key"].astype("int64").astype(str)),
        "record_hash_checksum": digest_of_keys(frame["record_hash"].astype(str).str.strip()),
        "value_sum": round(float(pd.to_numeric(frame["value"]).sum()), 6),
    }


# ------------------------------------------------------------------------------------ partitions

def partition_dir(settings: Settings) -> Path:
    return settings.path("partitioned_dir") / PARTITION_DATASET


def write_partitions(settings: Settings, staging: pd.DataFrame | None = None) -> dict[str, Any]:
    """Hive-style Parquet partitions by pollutant; the whole dataset is replaced on each run."""
    staging = read_staging(settings) if staging is None else staging
    column = settings["storage"]["partition_column"]
    target = partition_dir(settings)
    tmp = target.with_name(f".{target.name}.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    table = pa.Table.from_pandas(staging.sort_values("source_row_key"), preserve_index=False)
    pq.write_to_dataset(table, root_path=str(tmp), partition_cols=[column],
                        basename_template="part-{i}.parquet", existing_data_behavior="delete_matching")
    if target.exists():
        shutil.rmtree(target)
    replace_file(tmp, target)  # replacement-partition strategy: readers never see a half-written dataset
    counts = staging[column].value_counts().sort_index()
    return {"dataset": target.relative_to(settings.root).as_posix(), "partition_column": column,
            "partitions": {str(k): int(v) for k, v in counts.items()}}


def read_partition(settings: Settings, value: str) -> pd.DataFrame:
    """Load ONE partition directory only (no other partition files are opened)."""
    column = settings["storage"]["partition_column"]
    path = partition_dir(settings) / f"{column}={value}"
    if not path.exists():
        available = sorted(p.name for p in partition_dir(settings).glob(f"{column}=*"))
        raise FileNotFoundError(f"Partition {path.name} not found; available: {available}")
    frame = pd.read_parquet(path)
    frame[column] = value  # the partition value lives in the directory name, not in the files
    return frame


def verify_partition(settings: Settings, value: str) -> dict[str, Any]:
    column = settings["storage"]["partition_column"]
    t0 = time.perf_counter()
    part = read_partition(settings, value)
    elapsed = time.perf_counter() - t0
    files = sorted(p.name for p in (partition_dir(settings) / f"{column}={value}").glob("*.parquet"))
    staging = read_staging(settings)
    expected = staging[staging[column] == value]
    return {
        "partition": f"{column}={value}",
        "files_read": files,
        "rows_loaded": int(len(part)),
        "rows_expected_from_staging_filter": int(len(expected)),
        "row_count_matches": int(len(part)) == int(len(expected)),
        "same_records": _fingerprint(part) == _fingerprint(expected),
        "read_seconds": round(elapsed, 5),
    }


# ------------------------------------------------------------------------------------ benchmark

def run_benchmark(settings: Settings, run_id: str, ctx: StageContext | None = None) -> dict[str, Any]:
    from .storage import connect

    repeats = int(settings["storage"]["benchmark_repeats"])
    flt = settings["storage"]["benchmark_filter"]
    staging = read_staging(settings).sort_values("source_row_key").reset_index(drop=True)
    reference = _fingerprint(staging)
    selective_expected = int(((staging["pollutant"] == flt["pollutant"]) &
                              (staging["country_label"] == flt["country_label"])).sum())
    bench_dir = settings.path("benchmark_dir")
    bench_dir.mkdir(parents=True, exist_ok=True)
    paths = {"csv": bench_dir / "staging.csv", "jsonl": bench_dir / "staging.jsonl",
             "parquet": bench_dir / "staging.parquet"}

    writers = {
        "csv": lambda: staging.to_csv(paths["csv"], index=False),
        "jsonl": lambda: staging.to_json(paths["jsonl"], orient="records", lines=True, date_format="iso",
                                         force_ascii=False),
        "parquet": lambda: staging.to_parquet(paths["parquet"], index=False),
    }
    # Hash/text columns are declared as strings so type inference cannot mangle them.
    text_cols = {"record_hash": str, "pipeline_run_id": str}
    readers = {
        "csv": lambda: pd.read_csv(paths["csv"], dtype=text_cols),
        "jsonl": lambda: pd.read_json(paths["jsonl"], orient="records", lines=True, dtype=False, convert_dates=False),
        "parquet": lambda: pd.read_parquet(paths["parquet"]),
    }
    selective = {
        "csv": lambda: (lambda f: f[(f["pollutant"] == flt["pollutant"]) & (f["country_label"] == flt["country_label"])])(
            readers["csv"]()),
        "jsonl": lambda: (lambda f: f[(f["pollutant"] == flt["pollutant"]) & (f["country_label"] == flt["country_label"])])(
            readers["jsonl"]()),
        "parquet": lambda: pd.read_parquet(paths["parquet"], filters=[("pollutant", "==", flt["pollutant"]),
                                                                      ("country_label", "==", flt["country_label"])]),
    }

    results: dict[str, Any] = {}
    for fmt in ("csv", "jsonl", "parquet"):
        write_times, _ = _timed(writers[fmt], repeats)
        cold_t0 = time.perf_counter()
        cold = readers[fmt]()
        cold_s = time.perf_counter() - cold_t0
        read_times, frame = _timed(readers[fmt], repeats)
        sel_times, sel = _timed(selective[fmt], repeats)
        results[fmt] = {
            "size_bytes": paths[fmt].stat().st_size,
            "write": _summary(write_times),
            "read_full_cold_s": round(cold_s, 5),
            "read_full": _summary(read_times),
            "read_selective": _summary(sel_times),
            "selective_rows": int(len(sel)),
            "equivalent_to_staging": _fingerprint(frame) == reference and _fingerprint(cold) == reference,
        }
        log_event("benchmark_format_done", pipeline_run_id=run_id, stage="benchmark", format=fmt,
                  size_bytes=results[fmt]["size_bytes"], read_median_s=results[fmt]["read_full"]["median_s"])

    # PostgreSQL: dedicated benchmark table so the production tables are never truncated.
    from .storage import TABLES, _records

    spec_cols = [c for c in staging.columns]
    with connect(settings) as conn:
        conn.execute("CREATE SCHEMA IF NOT EXISTS benchmark")
        conn.execute("DROP TABLE IF EXISTS benchmark.staging_copy")
        conn.execute("CREATE TABLE benchmark.staging_copy (LIKE staging.air_quality_readings INCLUDING ALL)")
        conn.execute("ALTER TABLE benchmark.staging_copy DROP COLUMN row_hash, DROP COLUMN loaded_at_utc")
        conn.commit()

        def pg_write() -> None:
            with conn.transaction():
                conn.execute("TRUNCATE benchmark.staging_copy")
                cols = ", ".join(f'"{c}"' for c in spec_cols)
                with conn.cursor().copy(f"COPY benchmark.staging_copy ({cols}) FROM STDIN") as copy:
                    for row in _records(staging, spec_cols):
                        copy.write_row(row)

        def pg_query(query: str, params: tuple = ()) -> pd.DataFrame:
            cur = conn.execute(query, params)
            return pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])

        write_times, _ = _timed(pg_write, repeats)
        conn.execute("ANALYZE benchmark.staging_copy")
        full_sql = "SELECT * FROM benchmark.staging_copy"
        sel_sql = "SELECT * FROM benchmark.staging_copy WHERE pollutant = %s AND country_label = %s"
        cold_t0 = time.perf_counter()
        cold = pg_query(full_sql)
        cold_s = time.perf_counter() - cold_t0
        read_times, frame = _timed(lambda: pg_query(full_sql), repeats)
        sel_times, sel = _timed(lambda: pg_query(sel_sql, (flt["pollutant"], flt["country_label"])), repeats)
        plan = [r[0] for r in conn.execute("EXPLAIN " + sel_sql, (flt["pollutant"], flt["country_label"])).fetchall()]
        sizes = conn.execute("SELECT pg_relation_size('benchmark.staging_copy'), "
                             "pg_total_relation_size('benchmark.staging_copy')").fetchone()
        version = conn.execute("SHOW server_version").fetchone()[0]
        conn.commit()
    results["postgresql"] = {
        "size_bytes": int(sizes[1]),
        "table_heap_bytes": int(sizes[0]),
        "write": _summary(write_times),
        "read_full_cold_s": round(cold_s, 5),
        "read_full": _summary(read_times),
        "read_selective": _summary(sel_times),
        "selective_rows": int(len(sel)),
        "selective_plan": plan,
        "equivalent_to_staging": _fingerprint(frame) == reference and _fingerprint(cold) == reference,
        "server_version": version,
    }
    _ = TABLES  # production table specs are untouched by the benchmark

    partitions = write_partitions(settings, staging)
    selected = verify_partition(settings, flt["pollutant"])
    report = {
        "pipeline_run_id": run_id,
        "generated_at_utc": utc_now_iso(),
        "dataset": "data/staging/air_quality_readings.parquet",
        "reference": reference,
        "repeats": repeats,
        "timing_method": f"time.perf_counter; median of {repeats} runs; first cold full read reported separately",
        "selective_filter": flt,
        "selective_rows_expected": selective_expected,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "postgres": results["postgresql"]["server_version"],
            "in_container": Path("/.dockerenv").exists(),
        },
        "formats": results,
        "partitioning": {**partitions, "selected_partition_check": selected},
    }
    out = settings.path("outputs_dir") / "benchmark" / "benchmark_results.json"
    atomic_write_json(out, report)
    atomic_write_text(settings.root / "docs" / "benchmark_report.md", render_report(report))
    if ctx is not None:
        ctx.counts.update(benchmark_rows=reference["rows"])
    return report


# ------------------------------------------------------------------------------------ report

def _kb(n: int) -> str:
    return f"{n / 1024:,.0f} KiB"


def render_report(r: dict[str, Any]) -> str:
    f = r["formats"]
    order = ["csv", "jsonl", "parquet", "postgresql"]
    names = {"csv": "CSV", "jsonl": "JSON Lines", "parquet": "Parquet (snappy)", "postgresql": "PostgreSQL table"}
    rows = []
    for k in order:
        x = f[k]
        rows.append(f"| {names[k]} | {_kb(x['size_bytes'])} | {x['write']['median_s']:.4f} | {x['read_full_cold_s']:.4f} | "
                    f"{x['read_full']['median_s']:.4f} | {x['read_selective']['median_s']:.4f} | {x['selective_rows']:,} | "
                    f"{'yes' if x['equivalent_to_staging'] else 'NO'} |")
    by_read = sorted(order, key=lambda k: f[k]["read_full"]["median_s"])
    by_sel = sorted(order, key=lambda k: f[k]["read_selective"]["median_s"])
    by_size = sorted(order, key=lambda k: f[k]["size_bytes"])
    p = r["partitioning"]
    sel = p["selected_partition_check"]
    env = r["environment"]
    lines = [
        "# Storage Benchmark Report",
        "",
        f"_Generated by `python -m src.cli benchmark` (run `{r['pipeline_run_id']}`, {r['generated_at_utc']})._",
        "_Numbers are regenerated on every benchmark run; the conclusions explain how to read them._",
        "",
        "## Setup",
        "",
        f"- **Dataset:** `{r['dataset']}`: {r['reference']['rows']:,} staging readings, all columns. Every format",
        "  holds the same logical records; each is read back and compared with the original by row count,",
        "  `source_row_key` checksum, `record_hash` checksum and the sum of `value` (column *Equivalent*).",
        f"- **Timing:** {r['timing_method']}. Times are wall-clock seconds and include parsing into a pandas",
        "  DataFrame (for PostgreSQL: query execution and transfer through the driver).",
        f"- **Selective read:** `pollutant = '{r['selective_filter']['pollutant']}' AND country_label = "
        f"'{r['selective_filter']['country_label']}'` ({r['selective_rows_expected']:,} rows). Files are read and then",
        "  filtered (Parquet pushes the filter into the reader); PostgreSQL uses its index.",
        f"- **Environment:** {env['platform']}, Python {env['python']}, {env['cpu_count']} logical CPUs, pandas "
        f"{env['pandas']}, pyarrow {env['pyarrow']}, PostgreSQL {env['postgres']} (Docker container on the same "
        f"machine, reached over localhost). Pipeline in a container: {env['in_container']}.",
        "",
        "## Results",
        "",
        "| Format | Size | Write median (s) | Full read, cold (s) | Full read median (s) | Selective read median (s) | Selective rows | Equivalent |",
        "|---|---|---|---|---|---|---|---|",
        *rows,
        "",
        f"PostgreSQL size is `pg_total_relation_size` (heap {_kb(f['postgresql']['table_heap_bytes'])} plus primary-key,",
        "unique and filter indexes). Selective query plan:",
        "",
        "```",
        *f["postgresql"]["selective_plan"],
        "```",
        "",
        f"Ranking: full read {' < '.join(names[k] for k in by_read)}; selective read "
        f"{' < '.join(names[k] for k in by_sel)}; size {' < '.join(names[k] for k in by_size)}.",
        "",
        "## Partitioning",
        "",
        f"Natural partition key: **`{p['partition_column']}`**. It is a low-cardinality category that every",
        "downstream step filters or pivots on (the model needs PM2.5/NO2/O3 per station-day), and the partitions",
        "are reasonably balanced. Dates were rejected as a key: 78% of stations report a single date and the dates",
        "span ten years, so date partitions would be thousands of tiny files.",
        "",
        "| Partition | Rows |",
        "|---|---|",
        *[f"| `{p['partition_column']}={k}` | {v:,} |" for k, v in p["partitions"].items()],
        "",
        f"**Selected-partition load:** only `{p['dataset']}/{sel['partition']}/` was opened "
        f"(files: {', '.join(sel['files_read'])}). Rows loaded: **{sel['rows_loaded']:,}**; rows expected from filtering",
        f"staging: **{sel['rows_expected_from_staging_filter']:,}**; match: **{sel['row_count_matches']}**; identical records:",
        f"**{sel['same_records']}**; read time {sel['read_seconds']:.4f} s.",
        "",
        "## Conclusions for this project (not a universal ranking)",
        "",
        "| Use | Choice | Why, in this project |",
        "|---|---|---|",
        "| Source exchange | CSV (as delivered) | The publisher ships `;`-separated CSV; it is human-readable and universally importable. It is also the most fragile (encoding, delimiter, type inference), which is why the source is checksum-pinned and read as text. |",
        "| Pipeline intermediate storage | Parquet | Smallest files, fastest full reads here, types preserved (timestamps stay UTC, floats bit-exact), atomic file replacement makes reruns safe, and partitions allow loading one pollutant. |",
        "| Model input | Parquet (`data/curated/model_ready.parquet`) | Types and row order survive the round-trip exactly, which the bit-identical reproduction depends on. Reading the same table from PostgreSQL would need an explicit `ORDER BY ... COLLATE \"C\"` to keep the same order. |",
        "| Analytical querying / serving | PostgreSQL | Selective, indexed queries; constraints and keys enforce uniqueness and validity; concurrent consumers and SQL views (`curated.v_exceedance_by_country`). Its full-table reads pay for row-by-row transfer through the driver. |",
        "| Interchange with web/JS tools | JSON Lines | Self-describing records, but the largest and slowest here; timestamps become strings and types must be re-inferred. |",
        "",
        "**Why this must not be generalised:** the dataset is small (~30k rows, ~10 MB in memory), every read loads",
        "all columns into pandas, and the database runs in a container on the same laptop. With larger data,",
        "column subsets, many concurrent users, remote object storage or a tuned server, the ranking can change;",
        "timings also vary between runs because of OS caching and background load, so medians are reported.",
        "",
    ]
    return "\n".join(lines)
