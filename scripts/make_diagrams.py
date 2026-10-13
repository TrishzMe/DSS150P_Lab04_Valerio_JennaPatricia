"""Generate docs/architecture.png and docs/lineage.png (python scripts/make_diagrams.py).

Diagrams are drawn from code so they stay in sync with the implementation and can be regenerated.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

INK = "#1f2933"
MUTED = "#52606d"
COLORS = {
    "source": ("#fdf2e9", "#c05621"),
    "raw": ("#fefcbf", "#b7791f"),
    "staging": ("#e6fffa", "#2c7a7b"),
    "curated": ("#ebf8ff", "#2b6cb0"),
    "model": ("#faf5ff", "#6b46c1"),
    "output": ("#f0fff4", "#2f855a"),
    "infra": ("#f7fafc", "#4a5568"),
    "quarantine": ("#fff5f5", "#c53030"),
}


def box(ax, x, y, w, h, title, body="", kind="infra", title_size=11, body_size=8.6, mono=False):
    fill, edge = COLORS[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.003,rounding_size=0.01",
                                fc=fill, ec=edge, lw=1.6))
    ax.text(x + 0.008, y + h - 0.012, title, ha="left", va="top", fontsize=title_size, fontweight="bold", color=edge)
    if body:
        ax.text(x + 0.008, y + h - 0.042, body, ha="left", va="top", fontsize=body_size, color=INK,
                family="monospace" if mono else None, linespacing=1.35)


def arrow(ax, start, end, label="", color=INK, style="-|>", ls="-", rad=0.0, label_offset=(0, 0.012), size=8.2):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=14, color=color, lw=1.4,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}"))
    if label:
        mx, my = (start[0] + end[0]) / 2 + label_offset[0], (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="bottom", fontsize=size, color=color,
                bbox=dict(fc="white", ec="none", pad=1.2, alpha=0.9))


def canvas(width, height, title, subtitle):
    fig = plt.figure(figsize=(width, height), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.012, 0.985, title, ha="left", va="top", fontsize=17, fontweight="bold", color=INK)
    ax.text(0.012, 0.952, subtitle, ha="left", va="top", fontsize=10, color=MUTED)
    return fig, ax


def stage_label(ax, x, y, text, color):
    ax.text(x, y, text, ha="center", va="bottom", fontsize=8.6, color=color, fontweight="bold",
            bbox=dict(fc="white", ec="none", pad=1.0))


def architecture() -> None:
    fig, ax = canvas(20, 11, "Architecture: reproducible PM2.5 exceedance pipeline (DSS150P Lab 04)",
                     "Top: data movement through the layers (solid arrows, labelled with the Airflow task that moves it). "
                     "Bottom: the components that run it (dashed arrows = control).")
    w, top_y, top_h = 0.138, 0.62, 0.25
    xs = [0.008 + i * 0.1655 for i in range(6)]
    tops = [
        ("Source system", "source",
         "Kaggle: World Air Quality\nData 2024 (CC BY 4.0)\n\ndata/source/\n world_air_quality.csv\n54,255 rows x 10 cols\n\nSHA-256 pinned in\nmetadata/source_manifest.yml"),
        ("Raw layer", "raw",
         "data/raw/\n full/run=<id>/  byte copy\n batches/batch_000N.parquet\n   5 x 10,851 rows\n _state/\n   ingest_checkpoint.json\n\n+ source_row_key,\n  record_hash, run id"),
        ("Staging layer", "staging",
         "data/staging/\n air_quality_readings\n   .parquet\n30,545 typed readings\n\none row per\n(location, pollutant,\n last_updated_utc)"),
        ("Curated layer", "curated",
         "data/curated/\n station_day_pollutants\n   11,558 station-days\n model_ready\n   2,628 rows + label\n\ndata/partitioned/\n pollutant=<value>/"),
        ("PostgreSQL 16.15 warehouse", "curated",
         "staging.air_quality_\n  readings\ncurated.station_day_\n  pollutants\ncurated.model_ready\ncurated.predictions\ncurated.model_metrics\naudit.pipeline_runs,\n  task_runs, load_log"),
        ("Consumers", "output",
         "Reproducibility report\n (expected vs actual)\nAnalysts: SQL views\n v_exceedance_by_country\nOperations:\n audit.v_run_overview\nModel users:\n predictions + model card"),
    ]
    for x, (title, kind, body) in zip(xs, tops):
        box(ax, x, top_y, w, top_h, title, body, kind, body_size=8.6, mono=kind in ("raw", "staging", "curated"))

    hop_labels = ["ingest_source\nvalidate_raw", "transform_\nstaging", "build_curated\npartition_staging",
                  "load_postgres\nvalidate_curated", "SQL views"]
    hop_colors = [COLORS["raw"][1], COLORS["staging"][1], COLORS["curated"][1], COLORS["curated"][1], COLORS["output"][1]]
    for i, (label, color) in enumerate(zip(hop_labels, hop_colors)):
        x0, x1 = xs[i] + w, xs[i + 1]
        arrow(ax, (x0, top_y + 0.13), (x1, top_y + 0.13), "", color)
        stage_label(ax, (x0 + x1) / 2, top_y + top_h + 0.008, label, color)

    # Middle row: quarantine and model outputs.
    mid_y, mid_h = 0.39, 0.16
    box(ax, xs[2], mid_y, w, mid_h, "Quarantine", "data/quarantine/\n staging_rejects.parquet\n23,710 rows with\nreject_type + reason\n(nothing dropped silently)",
        "quarantine", body_size=8.4)
    arrow(ax, (xs[2] + w / 2, top_y), (xs[2] + w / 2, mid_y + mid_h), "", COLORS["quarantine"][1])
    box(ax, xs[3] + 0.04, mid_y, 0.27, mid_h, "Model input & artefacts (outputs/)",
        "model/*.joblib + model_card.json      split_manifest.json\nmetrics.json   predictions.csv       split_assignment.csv\nreproducibility_comparison.json        run_summary.json\nquality/*.json  (data-quality reports per layer)",
        "model", body_size=8.2, mono=True)
    arrow(ax, (xs[3] + 0.075, top_y), (xs[3] + 0.075, mid_y + mid_h), "", COLORS["model"][1])
    stage_label(ax, xs[3] + 0.075, mid_y + mid_h + 0.012, "train_model / evaluate_model /\nreproducibility_check",
                COLORS["model"][1])
    arrow(ax, (xs[4] + 0.11, mid_y + mid_h), (xs[4] + 0.11, top_y), "", COLORS["output"][1])
    stage_label(ax, xs[4] + 0.11, mid_y + mid_h + 0.012, "publish_outputs", COLORS["output"][1])

    # Bottom band: runtime components.
    ax.add_patch(FancyBboxPatch((0.004, 0.035), 0.992, 0.27, boxstyle="round,pad=0.004,rounding_size=0.012",
                                fc="#fbfcfd", ec="#cbd2d9", lw=1.2, ls="--"))
    ax.text(0.012, 0.296, "Runtime (docker compose project dss150p_lab04_valerio_jennapatricia)", fontsize=10,
            color=MUTED, va="top")
    bot_y, bot_h = 0.05, 0.22
    box(ax, 0.012, bot_y, 0.14, bot_h, "Airflow metadata DB",
        "[container] postgres:16.15\n(airflow-db)\n\nseparate from the\nwarehouse: a warehouse\n"
        "outage cannot take the\norchestrator down", "infra", body_size=8.4)
    box(ax, 0.168, bot_y, 0.215, bot_h, "Apache Airflow 3.3.2",
        "[container] airflow standalone\nDAG predictive_repro_pipeline\nschedule=None, catchup=False\n"
        "params: ingest_mode, impute_scope,\n  selected_partition\nretries + exponential backoff,\n"
        "timeouts, failure/retry callbacks", "infra", body_size=8.4)
    box(ax, 0.415, bot_y, 0.25, bot_h, "Pipeline code src/ (python -m src.cli)",
        "ingest  validate  transform  storage\nbenchmark  train  evaluate  cli\n\nruns in the pipeline image\n"
        "(docker compose run pipeline run-all)\nor in Airflow's /opt/pipeline-venv;\nboth from the same pinned\n"
        "requirements.txt (Python 3.12)", "infra", body_size=8.4)
    box(ax, 0.68, bot_y, 0.145, bot_h, "Configuration",
        "config/settings.yml\n (paths, seeds, params,\n  tolerance, batches)\n.env  (git-ignored)\nmetadata/\n"
        " data_contract.yml\n expected_metrics.json", "infra", body_size=8.4)
    box(ax, 0.84, bot_y, 0.148, bot_h, "Observability",
        "written by every stage:\nlogs/pipeline.jsonl\n (ts, pipeline_run_id,\n  stage, row counts,\n"
        "  duration, errors)\noutputs/runs/<run>/\n  run_record.json", "infra", body_size=8.4)
    arrow(ax, (0.152, 0.16), (0.168, 0.16), "", MUTED, style="<|-|>", ls="--")
    arrow(ax, (0.383, 0.16), (0.415, 0.16), "", MUTED, ls="--")
    ax.text(0.399, 0.168, "runs", fontsize=8, color=MUTED, ha="center")
    arrow(ax, (0.68, 0.16), (0.665, 0.16), "", MUTED, ls="--")
    arrow(ax, (0.54, bot_y + bot_h), (0.54, mid_y - 0.005), "", MUTED, ls="--")
    ax.text(0.545, 0.33, "executes every stage above", fontsize=8.4, color=MUTED, va="center")
    fig.savefig(DOCS / "architecture.png", facecolor="white")
    plt.close(fig)


def lineage() -> None:
    fig, ax = canvas(22, 11, "Data lineage: source fields -> raw -> staging -> curated -> model features -> model -> outputs",
                     "Dataset-level flow with the field mappings and transformation rules applied at each hop "
                     "(row counts for source SHA-256 2b2202a0...).")
    col_w, h = 0.148, 0.33
    y = 0.45
    cols = [
        (0.008, "SOURCE  (54,255 rows)", "source",
         "world_air_quality.csv\n';' UTF-8-BOM CRLF\n\nCountry Code\nCity\nLocation\nCoordinates\nPollutant\nSource Name\nUnit\nValue\nLast Updated\nCountry Label"),
        (0.172, "RAW  (54,255 rows, 5 batches)", "raw",
         "all 10 source fields as TEXT\n(unchanged values)\n\n+ source_row_key  (1..N)\n+ source_batch_id\n+ record_hash     (SHA-256)\n+ pipeline_run_id\n+ ingested_at_utc\n+ source_file\n+ source_checksum\n\ncheckpoint advances only\nafter batch validation"),
        (0.336, "STAGING  (30,545 rows)", "staging",
         "country_code, city*, location*\ncoordinates, pollutant\nsource_name, unit\nvalue       FLOAT\nlast_updated_utc TIMESTAMPTZ\nreading_date DATE (UTC)\ncountry_label*\n+ row_hash, pipeline_run_id,\n  processed_at_utc\n\n* null tokens / NULL -> 'Unknown'\nkey: (location, pollutant,\n      last_updated_utc)"),
        (0.500, "CURATED  (11,558 / 2,628)", "curated",
         "station_day_pollutants\n key = sha256(country_label,\n   city, location, reading_date)\n pm25 pm10 no2 o3 so2 co\n   = mean(value) per pollutant\n n_readings\n\nmodel_ready (pm25,no2,o3 present)\n so2_missing, co_missing\n ox_total_oxidant = no2 + o3\n no2_o3_ratio = no2/(o3+1)\n air_quality = Unhealthy if\n   pm25 > 15 else Safe"),
        (0.664, "MODEL INPUT  (2,102 / 526)", "model",
         "features (ordered):\n no2, o3, so2, co,\n so2_missing, co_missing,\n ox_total_oxidant,\n no2_o3_ratio\ntarget: air_quality\n\nso2/co median-imputed\n (baseline: all rows;\n  train_only: train rows)\nstratified split, seed 42\nEXCLUDED: pm25, pm10\n (target / leakage)"),
        (0.828, "MODELS -> OUTPUTS", "output",
         "LogisticRegression(max_iter=1000)\nRandomForest(200, rs=42)\nGradientBoosting(200, rs=42)\n\npredictions.csv\n -> curated.predictions\nmetrics.json\n -> curated.model_metrics\n -> reproducibility_\n    comparison.json\n    vs expected_metrics.json\nrun_summary.json -> audit.*"),
    ]
    for x, title, kind, body in cols:
        box(ax, x, y, col_w + (0.006 if x > 0.8 else 0), h, title, body, kind, title_size=10.5, body_size=8.5, mono=True)
    hops = [
        (0.156, "ingest\nbyte copy +\n5 batches"),
        (0.320, "standardise NULLs,\nscope 6 pollutants,\ntype, dedup (keep\nfirst), ug/m3 only,\nvalue >= 0"),
        (0.484, "pivot mean per\nstation-day;\nderive features\n+ WHO label"),
        (0.648, "impute,\nsplit,\norder check"),
        (0.812, "fit /\npredict /\nevaluate"),
    ]
    for x, label in hops:
        arrow(ax, (x, 0.70), (x + 0.016, 0.70), "", INK)
        ax.text(x + 0.008, 0.935, label, ha="center", va="top", fontsize=8.0, color=INK,
                bbox=dict(fc="white", ec="#cbd2d9", pad=2.0))
        ax.plot([x + 0.008, x + 0.008], [0.855, 0.715], color="#cbd2d9", lw=1)

    box(ax, 0.336, 0.11, 0.31, 0.22, "QUARANTINE  (23,710 rows, with reason)",
        "out_of_scope_pollutant  6,385   (NO, NOX, BC, ...)\nduplicate_reading       8,074   (same location/pollutant/time)\nnon_ugm3_unit           8,943   (ppm etc.)\nnegative_value            308   (sensor errors)\nraw rows = staging rows + quarantined rows (REC-01)", "quarantine", body_size=8.5, mono=True)
    arrow(ax, (0.41, 0.45), (0.41, 0.335), "", COLORS["quarantine"][1])
    box(ax, 0.664, 0.11, 0.328, 0.22, "Field lineage of the key columns",
        "PM2.5 'Value' -> value -> pm25 (mean) -> air_quality (target)\nPM10  'Value' -> pm10  (kept, never a feature: leakage)\nNO2/O3 'Value' -> no2, o3 -> features + ox/ratio\nSO2/CO 'Value' -> so2, co (+ _missing) -> imputed\nLast Updated (+05:30) -> UTC -> reading_date\nCoordinates -> staging only (unused downstream)", "infra", body_size=8.5, mono=True)
    box(ax, 0.008, 0.11, 0.312, 0.22, "Identity carried through every layer",
        "source_checksum   which source version\nsource_row_key    which source record\nrecord_hash       raw content identity\nstation_day_key   curated business key\npipeline_run_id   which run produced it\nsplit_manifest    train/test key checksums", "infra", body_size=8.5, mono=True)
    fig.savefig(DOCS / "lineage.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    lineage()
    print("wrote docs/architecture.png and docs/lineage.png")
