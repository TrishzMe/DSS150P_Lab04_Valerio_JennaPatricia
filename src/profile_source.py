"""Profile the immutable source dataset and write metadata/source_profile.{md,json}.

The profile is generated from the data (not typed by hand) so it can be regenerated and
diffed: ``python -m src.cli profile``.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from .config import Settings
from .io_utils import atomic_write_json, atomic_write_text
from .source import RAW_COLUMNS, inspect_file, load_manifest, read_source_text

CANDIDATE_KEYS = [
    ["Location", "Pollutant", "Last Updated"],
    ["Location", "Pollutant", "Unit", "Last Updated"],
    ["Location", "Pollutant", "Source Name", "Last Updated"],
    ["Country Code", "Location", "Coordinates", "Pollutant", "Unit", "Last Updated"],
    RAW_COLUMNS,
]
DECLARED_TYPES = {
    "Country Code": "string (ISO-3166 alpha-2)",
    "City": "string, nullable",
    "Location": "string (station name/address)",
    "Coordinates": "string 'lat, lon'",
    "Pollutant": "categorical string",
    "Source Name": "categorical string",
    "Unit": "categorical string",
    "Value": "float",
    "Last Updated": "timestamp with UTC offset (ISO-8601)",
    "Country Label": "string, nullable",
}
TOP_N_COLUMNS = ["Country Code", "Pollutant", "Unit", "Source Name", "Country Label", "City"]


def _column_profile(text: pd.Series, std: pd.Series, name: str) -> dict[str, Any]:
    empty = int((text == "").sum())
    null_total = int(std.isna().sum())
    token_counts = text[(text != "") & std.isna()].value_counts().to_dict()
    profile: dict[str, Any] = {
        "column": name,
        "declared_type": DECLARED_TYPES[name],
        "non_null": int(std.notna().sum()),
        "empty_strings": empty,
        "null_tokens": {str(k): int(v) for k, v in token_counts.items()},
        "null_total": null_total,
        "null_rate": round(null_total / len(std), 6),
        "distinct_non_null": int(std.nunique(dropna=True)),
    }
    if name == "Value":
        numeric = pd.to_numeric(std, errors="coerce")
        profile.update(
            inferred_type="float64",
            unparseable=int(numeric.isna().sum() - null_total),
            min=float(numeric.min()),
            max=float(numeric.max()),
            median=float(numeric.median()),
            negatives=int((numeric < 0).sum()),
            zeros=int((numeric == 0).sum()),
        )
    elif name == "Last Updated":
        ts = pd.to_datetime(std, utc=True, errors="coerce")
        profile.update(
            inferred_type="datetime64[ns, UTC]",
            unparseable=int(ts.isna().sum() - null_total),
            min=str(std.min()),
            max=str(std.max()),
            min_utc=str(ts.min()),
            max_utc=str(ts.max()),
            utc_offsets={k: int(v) for k, v in std.str[-6:].value_counts().items()},
            distinct_utc_dates=int(ts.dt.date.nunique()),
        )
    else:
        profile["inferred_type"] = "string"
    if name in TOP_N_COLUMNS:
        profile["top_values"] = {str(k): int(v) for k, v in std.value_counts().head(8).items()}
    return profile


def build_profile(settings: Settings) -> dict[str, Any]:
    path = settings.source_path
    manifest = load_manifest(settings)
    facts = inspect_file(path, settings["source"]["delimiter"])
    text = read_source_text(path, settings)
    tokens = settings["source"]["null_tokens"]
    std = text.mask(text.isin(tokens))

    columns = [_column_profile(text[c], std[c], c) for c in RAW_COLUMNS]
    value = pd.to_numeric(std["Value"], errors="coerce")

    keys = []
    for key in CANDIDATE_KEYS:
        dup_rows = int(std.duplicated(subset=key, keep="first").sum())
        keys.append({"columns": key, "duplicate_rows": dup_rows, "is_unique": dup_rows == 0})

    pollutant_unit = (
        std.groupby(["Pollutant", "Unit"], dropna=False).size().rename("rows").reset_index()
        .sort_values(["Pollutant", "Unit"]).to_dict(orient="records")
    )
    in_scope = settings["transform"]["target_pollutants"]
    unit = settings["transform"]["accepted_unit"]

    pm25 = value[(std["Pollutant"] == "PM2.5") & (std["Unit"] == unit) & (value >= 0)]
    threshold = settings["model"]["threshold_ugm3"]
    loc = std.dropna(subset=["Location"])
    dates_per_location = (
        pd.to_datetime(loc["Last Updated"], utc=True).dt.date.groupby(loc["Location"]).nunique()
    )
    code_label = std.dropna(subset=["Country Code", "Country Label"]).drop_duplicates(["Country Code", "Country Label"])
    coords_ok = std["Coordinates"].str.match(r"^-?\d+(\.\d+)?, -?\d+(\.\d+)?$", na=False)
    non_ascii = std["Location"].fillna("").str.contains(r"[^\x00-\x7F]", regex=True)

    return {
        "dataset_id": manifest["dataset_id"],
        "source_file": manifest["file"]["relative_path"],
        "file": facts,
        "columns": columns,
        "candidate_keys": keys,
        "duplicates": {
            "full_row_duplicates": int(text.duplicated().sum()),
            "business_key": ["Location", "Pollutant", "Last Updated"],
            "business_key_duplicate_rows": int(std.duplicated(["Location", "Pollutant", "Last Updated"]).sum()),
            "business_key_duplicate_rows_in_scope": int(
                std[std["Pollutant"].isin(in_scope)].duplicated(["Location", "Pollutant", "Last Updated"]).sum()
            ),
        },
        "pollutant_unit_matrix": pollutant_unit,
        "in_scope_rows": int(std["Pollutant"].isin(in_scope).sum()),
        "target_reading_level": {
            "pm25_ugm3_valid_readings": int(len(pm25)),
            "above_threshold": int((pm25 > threshold).sum()),
            "at_or_below_threshold": int((pm25 <= threshold).sum()),
            "share_above_threshold": round(float((pm25 > threshold).mean()), 6),
            "median": float(pm25.median()),
            "p95": float(pm25.quantile(0.95)),
            "max": float(pm25.max()),
        },
        "temporal": {
            "locations": int(dates_per_location.size),
            "median_dates_per_location": float(dates_per_location.median()),
            "max_dates_per_location": int(dates_per_location.max()),
            "share_locations_single_date": round(float((dates_per_location == 1).mean()), 6),
        },
        "malformed": {
            "negative_values": int((value < 0).sum()),
            "empty_or_malformed_coordinates": int((~coords_ok).sum()),
            "country_code_label_pairs": int(len(code_label)),
            "country_codes_with_multiple_labels": int((code_label.groupby("Country Code").size() > 1).sum()),
            "rows_with_non_ascii_location": int(non_ascii.sum()),
            "climate_readings": {
                "TEMPERATURE": int((std["Pollutant"] == "TEMPERATURE").sum()),
                "RELATIVEHUMIDITY": int((std["Pollutant"] == "RELATIVEHUMIDITY").sum()),
            },
        },
    }


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |")
    return "\n".join(out)


def render_markdown(profile: dict[str, Any], manifest: dict[str, Any]) -> str:
    f = profile["file"]
    lines = [
        "# Source Profile — World Air Quality Data 2024",
        "",
        "_Generated by `python -m src.cli profile` from `data/source/world_air_quality.csv`._",
        "_Do not edit by hand; regenerate instead._",
        "",
        "## 1. File identity and structure",
        "",
        _md_table(
            ["Property", "Value"],
            [
                ["Source location", f"`{profile['source_file']}` (from {manifest['provenance']['landing_page']})"],
                ["Format", "Delimited text (CSV)"],
                ["Delimiter", "`;` (semicolon)"],
                ["Encoding", f"{f['encoding']} (UTF-8 with BOM: {f['has_utf8_bom']})"],
                ["Line terminator", f["line_terminator"]],
                ["Size", f"{f['size_bytes']:,} bytes"],
                ["Rows (excluding header)", f"{f['row_count']:,}"],
                ["Columns", f["column_count"]],
                ["SHA-256", f"`{f['sha256']}`"],
            ],
        ),
        "",
        "## 2. Column profile",
        "",
        "Nulls are counted after standardising the text null tokens listed in `config/settings.yml`",
        "(`source.null_tokens`, pandas' default list, which the original notebook applied implicitly).",
        "",
    ]
    rows = []
    for c in profile["columns"]:
        if "min" in c:
            rng = f"{c['min']} → {c['max']}"
        elif "top_values" in c:
            rng = ", ".join(f"{k} ({v})" for k, v in list(c["top_values"].items())[:3])
        else:
            rng = "free text"
        tokens = ", ".join(f"'{k}'×{v}" for k, v in c["null_tokens"].items()) or "–"
        rows.append([
            c["column"], c["declared_type"], c["inferred_type"], c["null_total"],
            f"{c['null_rate']:.4f}", tokens, c["distinct_non_null"], rng,
        ])
    lines += [
        _md_table(
            ["Column", "Declared type", "Inferred type", "Nulls", "Null rate", "Null tokens (non-empty)", "Distinct", "Range / top values"],
            rows,
        ),
        "",
    ]
    value = next(c for c in profile["columns"] if c["column"] == "Value")
    ts = next(c for c in profile["columns"] if c["column"] == "Last Updated")
    lines += [
        f"- `Value`: median {value['median']}, {value['negatives']} negative readings, {value['zeros']} zeros, "
        f"{value['unparseable']} unparseable.",
        f"- `Last Updated`: {ts['min']} → {ts['max']} (UTC {ts['min_utc']} → {ts['max_utc']}); "
        f"UTC offsets {ts['utc_offsets']}; {ts['unparseable']} unparseable; {ts['distinct_utc_dates']} distinct UTC dates.",
        "",
        "## 3. Keys and duplicates",
        "",
        f"- Full-row duplicates: **{profile['duplicates']['full_row_duplicates']}**.",
        f"- Business key `(Location, Pollutant, Last Updated)` duplicate rows: "
        f"**{profile['duplicates']['business_key_duplicate_rows']:,}** "
        f"({profile['duplicates']['business_key_duplicate_rows_in_scope']:,} among in-scope pollutants).",
        "- No natural column combination is unique, so the pipeline assigns `source_row_key` (1-based record",
        "  ordinal in the immutable file) as the raw-layer key, and `record_hash` (SHA-256 of all fields).",
        "",
        _md_table(
            ["Candidate key", "Duplicate rows", "Unique?"],
            [[", ".join(k["columns"]), f"{k['duplicate_rows']:,}", k["is_unique"]] for k in profile["candidate_keys"]],
        ),
        "",
        "## 4. Pollutants and units",
        "",
        f"In-scope pollutants for the model (PM2.5, PM10, NO2, O3, SO2, CO): **{profile['in_scope_rows']:,}** rows.",
        "Several gases are reported in both `ppm` and `µg/m³`; only `µg/m³` is kept (as in the original project).",
        "",
        _md_table(["Pollutant", "Unit", "Rows"], [[r["Pollutant"], r["Unit"], r["rows"]] for r in profile["pollutant_unit_matrix"]]),
        "",
        "## 5. Target variable",
        "",
        "The target `air_quality` is derived, not stored: the station-day mean PM2.5 (µg/m³) is labelled",
        "`Unhealthy` when it is strictly greater than the WHO 24-hour guideline of 15 µg/m³, else `Safe`.",
        "",
    ]
    t = profile["target_reading_level"]
    lines += [
        _md_table(
            ["Reading-level PM2.5 statistic (µg/m³, non-negative)", "Value"],
            [
                ["Valid PM2.5 readings", f"{t['pm25_ugm3_valid_readings']:,}"],
                ["> 15 µg/m³ (Unhealthy)", f"{t['above_threshold']:,} ({t['share_above_threshold']:.1%})"],
                ["≤ 15 µg/m³ (Safe)", f"{t['at_or_below_threshold']:,}"],
                ["Median / P95 / Max", f"{t['median']} / {t['p95']:.2f} / {t['max']}"],
            ],
        ),
        "",
        "At the model-ready station-day level the original project reported 1,513 Unhealthy (57.6%) vs",
        "1,115 Safe (42.4%): a moderate imbalance, handled with a stratified split and weighted metrics.",
        "",
        "## 6. Risks: leakage, identifiers, timestamps, malformed values, schema",
        "",
    ]
    m = profile["malformed"]
    tp = profile["temporal"]
    lines += [
        "- **Leakage:** `PM10` physically contains the PM2.5 fraction (r = 0.72 at station-day level in the",
        "  original EDA), so it is excluded from the features. `PM2.5` itself only defines the target.",
        "- **Identifiers:** `Location` (station name/address) and `Coordinates` identify stations; they are",
        "  grouping keys, never features. `Coordinates` is empty or malformed in "
        f"{m['empty_or_malformed_coordinates']} rows (unused downstream).",
        f"- **Timestamps:** every `Last Updated` value carries the same offset (+05:30, the publisher's",
        "  timezone), so it is converted to UTC before taking the date. It is a per-station *last reported*",
        f"  time spanning ten years, not an arrival time: {tp['share_locations_single_date']:.1%} of "
        f"{tp['locations']:,} locations report on a single date (median {tp['median_dates_per_location']:.0f}).",
        "  It therefore cannot serve as a business watermark (see ingestion design).",
        f"- **Malformed values:** {m['negative_values']} negative concentrations (sensor errors, removed in staging);",
        "  null tokens `N/A` / `None` inside `City` and `Location`; empty `Country Label` values.",
        f"- **Extreme outliers:** `Value` reaches {value['max']:,} (CO, µg/m³). The original project did not cap",
        "  outliers, so the reproduction keeps them (changing this would change the baseline) and flags them",
        "  as a known limitation.",
        f"- **Categories:** {m['country_code_label_pairs']} country code/label pairs, "
        f"{m['country_codes_with_multiple_labels']} codes map to more than one label.",
        f"- **Sparse climate fields:** TEMPERATURE {m['climate_readings']['TEMPERATURE']} rows, RELATIVEHUMIDITY "
        f"{m['climate_readings']['RELATIVEHUMIDITY']} rows: the reason the original project changed predictors.",
        f"- **Schema/encoding risks:** UTF-8 BOM in the header, CRLF line endings, `;` delimiter, "
        f"{m['rows_with_non_ascii_location']:,} rows with non-ASCII station names, and the unit filter",
        "  depends on the exact string `µg/m³`. Opening and re-saving the file in a spreadsheet tool would",
        "  silently change these and is caught by the checksum check.",
        "",
        "## 7. Ownership, licence and availability",
        "",
        f"- Publisher: {manifest['provenance']['publisher']} — {manifest['provenance']['landing_page']}",
        f"- Upstream origin: {manifest['provenance']['upstream_origin']}",
        f"- Licence: {manifest['license']['name']} ({manifest['license']['url']}); redistribution allowed with attribution.",
        f"- Update frequency: {manifest['provenance']['update_frequency']}",
        f"- Availability: {manifest['provenance']['expected_availability']}",
        f"- Privacy: {manifest['privacy']['notes']}",
        "",
    ]
    return "\n".join(lines)


def write_profile(settings: Settings) -> dict[str, Any]:
    profile = build_profile(settings)
    manifest = load_manifest(settings)
    meta = settings.path("metadata_dir")
    atomic_write_json(meta / "source_profile.json", profile)
    atomic_write_text(meta / "source_profile.md", render_markdown(profile, manifest))
    return profile
