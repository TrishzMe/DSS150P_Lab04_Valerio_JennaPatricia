# data/source — immutable source dataset

| Item | Value |
|---|---|
| File | `world_air_quality.csv` |
| Dataset | World Air Quality Data 2024 (Updated) |
| Publisher | Kaggle user kanchana1990 — https://www.kaggle.com/datasets/kanchana1990/world-air-quality-data-2024-updated |
| Licence | CC BY 4.0 — https://creativecommons.org/licenses/by/4.0/ |
| SHA-256 | `2b2202a0fa1cffdeddf803e54dd7f822812d86de3a6449c03798bb872b272f10` |
| Size | 6,472,005 bytes, 54,255 data rows, 10 columns, `;`-delimited, UTF-8 with BOM, CRLF |

**Attribution:** "World Air Quality Data 2024 (Updated)" by kanchana1990 (Kaggle), licensed
under CC BY 4.0. The file in this folder is an unmodified copy. Every other dataset in this
repository is derived from it by the pipeline.

## Rules

- Never edit, re-save, re-encode or re-sort this file. Spreadsheet tools change the delimiter,
  encoding and line endings, which breaks the checksum.
- `.gitattributes` marks it `-text` so git never converts its CRLF line endings.
- If the file is missing (for example, a partial clone), download it from the Kaggle page above,
  place it here with exactly this name, and run `python -m src.cli verify-source`. A different
  checksum means a different source version: it must be registered as a new version in
  `metadata/source_manifest.yml`, not silently accepted.
