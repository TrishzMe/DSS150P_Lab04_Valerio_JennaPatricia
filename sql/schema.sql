-- PostgreSQL schema for the PM2.5 exceedance data product.
-- Idempotent: safe to run on every pipeline start (python -m src.cli init-db).
--
--   staging.*  typed, validated, deduplicated readings (one row per business key)
--   curated.*  station-day tables, model-ready table and published model outputs
--   audit.*    pipeline runs, task runs and load statistics (observability)
--
-- Every data table carries:
--   row_hash         SHA-256 of the business columns (excludes audit columns); the rerun-safe
--                    upsert only rewrites a row when its row_hash changed
--   pipeline_run_id  run that last inserted or changed the row
--   processed_at_utc when the pipeline produced the row; loaded_at_utc when it reached the table

CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS curated;
CREATE SCHEMA IF NOT EXISTS audit;

-- ---------------------------------------------------------------------------------- staging
CREATE TABLE IF NOT EXISTS staging.air_quality_readings (
    source_row_key    BIGINT           PRIMARY KEY,
    source_batch_id   INTEGER          NOT NULL,
    record_hash       CHAR(64)         NOT NULL,
    country_code      TEXT             NOT NULL,
    city              TEXT             NOT NULL,
    location          TEXT             NOT NULL,
    coordinates       TEXT,
    pollutant         TEXT             NOT NULL CHECK (pollutant IN ('PM2.5', 'PM10', 'NO2', 'O3', 'SO2', 'CO')),
    source_name       TEXT             NOT NULL,
    unit              TEXT             NOT NULL CHECK (unit = 'µg/m³'),
    value             DOUBLE PRECISION NOT NULL CHECK (value >= 0),
    last_updated_utc  TIMESTAMPTZ      NOT NULL,
    reading_date      DATE             NOT NULL,
    country_label     TEXT             NOT NULL,
    row_hash          CHAR(64)         NOT NULL,
    pipeline_run_id   TEXT             NOT NULL,
    processed_at_utc  TIMESTAMPTZ      NOT NULL,
    loaded_at_utc     TIMESTAMPTZ      NOT NULL DEFAULT now(),
    CONSTRAINT uq_staging_business_key UNIQUE (location, pollutant, last_updated_utc)
);
CREATE INDEX IF NOT EXISTS ix_staging_pollutant_country
    ON staging.air_quality_readings (pollutant, country_label);

-- ---------------------------------------------------------------------------------- curated
CREATE TABLE IF NOT EXISTS curated.station_day_pollutants (
    station_day_key   CHAR(64)         PRIMARY KEY,
    country_label     TEXT             NOT NULL,
    city              TEXT             NOT NULL,
    location          TEXT             NOT NULL,
    reading_date      DATE             NOT NULL,
    pm25              DOUBLE PRECISION CHECK (pm25 >= 0),
    pm10              DOUBLE PRECISION CHECK (pm10 >= 0),
    no2               DOUBLE PRECISION CHECK (no2 >= 0),
    o3                DOUBLE PRECISION CHECK (o3 >= 0),
    so2               DOUBLE PRECISION CHECK (so2 >= 0),
    co                DOUBLE PRECISION CHECK (co >= 0),
    n_readings        INTEGER          NOT NULL CHECK (n_readings > 0),
    row_hash          CHAR(64)         NOT NULL,
    pipeline_run_id   TEXT             NOT NULL,
    processed_at_utc  TIMESTAMPTZ      NOT NULL,
    loaded_at_utc     TIMESTAMPTZ      NOT NULL DEFAULT now(),
    CONSTRAINT uq_station_day UNIQUE (country_label, city, location, reading_date)
);

CREATE TABLE IF NOT EXISTS curated.model_ready (
    station_day_key   CHAR(64)         PRIMARY KEY
                      REFERENCES curated.station_day_pollutants (station_day_key) ON DELETE CASCADE,
    country_label     TEXT             NOT NULL,
    city              TEXT             NOT NULL,
    location          TEXT             NOT NULL,
    reading_date      DATE             NOT NULL,
    pm25              DOUBLE PRECISION NOT NULL,
    pm10              DOUBLE PRECISION,
    no2               DOUBLE PRECISION NOT NULL,
    o3                DOUBLE PRECISION NOT NULL,
    so2               DOUBLE PRECISION,
    co                DOUBLE PRECISION,
    so2_missing       SMALLINT         NOT NULL CHECK (so2_missing IN (0, 1)),
    co_missing        SMALLINT         NOT NULL CHECK (co_missing IN (0, 1)),
    ox_total_oxidant  DOUBLE PRECISION NOT NULL,
    no2_o3_ratio      DOUBLE PRECISION NOT NULL,
    air_quality       TEXT             NOT NULL CHECK (air_quality IN ('Safe', 'Unhealthy')),
    row_hash          CHAR(64)         NOT NULL,
    pipeline_run_id   TEXT             NOT NULL,
    processed_at_utc  TIMESTAMPTZ      NOT NULL,
    loaded_at_utc     TIMESTAMPTZ      NOT NULL DEFAULT now(),
    CONSTRAINT ck_missing_flags CHECK ((so2_missing = 1) = (so2 IS NULL) AND (co_missing = 1) = (co IS NULL))
);

-- Published model outputs: latest prediction per (station-day, model) on the held-out test set.
CREATE TABLE IF NOT EXISTS curated.predictions (
    station_day_key        CHAR(64)         NOT NULL
                           REFERENCES curated.model_ready (station_day_key) ON DELETE CASCADE,
    model_name             TEXT             NOT NULL,
    split                  TEXT             NOT NULL CHECK (split = 'test'),
    actual                 TEXT             NOT NULL CHECK (actual IN ('Safe', 'Unhealthy')),
    predicted              TEXT             NOT NULL CHECK (predicted IN ('Safe', 'Unhealthy')),
    probability_unhealthy  DOUBLE PRECISION CHECK (probability_unhealthy BETWEEN 0 AND 1),
    is_selected_model      BOOLEAN          NOT NULL,
    impute_scope           TEXT             NOT NULL,
    row_hash               CHAR(64)         NOT NULL,
    pipeline_run_id        TEXT             NOT NULL,
    loaded_at_utc          TIMESTAMPTZ      NOT NULL DEFAULT now(),
    PRIMARY KEY (station_day_key, model_name)
);

-- Metric history: one row per run, model and metric (a run log, so run id is part of the key).
CREATE TABLE IF NOT EXISTS curated.model_metrics (
    pipeline_run_id   TEXT             NOT NULL,
    model_name        TEXT             NOT NULL,
    metric_name       TEXT             NOT NULL,
    metric_value      DOUBLE PRECISION NOT NULL,
    expected_value    DOUBLE PRECISION,
    abs_diff          DOUBLE PRECISION,
    within_tolerance  BOOLEAN,
    impute_scope      TEXT             NOT NULL,
    loaded_at_utc     TIMESTAMPTZ      NOT NULL DEFAULT now(),
    PRIMARY KEY (pipeline_run_id, model_name, metric_name)
);

-- ------------------------------------------------------------------------------------ audit
CREATE TABLE IF NOT EXISTS audit.pipeline_runs (
    pipeline_run_id   TEXT        PRIMARY KEY,
    status            TEXT        NOT NULL,
    started_at_utc    TIMESTAMPTZ,
    ended_at_utc      TIMESTAMPTZ,
    source_checksum   CHAR(64),
    parameters        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    counts            JSONB       NOT NULL DEFAULT '{}'::jsonb,
    updated_at_utc    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS audit.task_runs (
    pipeline_run_id   TEXT             NOT NULL REFERENCES audit.pipeline_runs (pipeline_run_id) ON DELETE CASCADE,
    task_name         TEXT             NOT NULL,
    attempt           INTEGER          NOT NULL,
    status            TEXT             NOT NULL,
    started_at_utc    TIMESTAMPTZ,
    ended_at_utc      TIMESTAMPTZ,
    duration_s        DOUBLE PRECISION,
    row_counts        JSONB            NOT NULL DEFAULT '{}'::jsonb,
    error             TEXT,
    updated_at_utc    TIMESTAMPTZ      NOT NULL DEFAULT now(),
    PRIMARY KEY (pipeline_run_id, task_name)
);

CREATE TABLE IF NOT EXISTS audit.load_log (
    load_id           BIGSERIAL   PRIMARY KEY,
    pipeline_run_id   TEXT        NOT NULL,
    table_name        TEXT        NOT NULL,
    snapshot_rows     INTEGER     NOT NULL,
    rows_inserted     INTEGER     NOT NULL,
    rows_updated      INTEGER     NOT NULL,
    rows_deleted      INTEGER     NOT NULL,
    rows_unchanged    INTEGER     NOT NULL,
    table_rows_after  INTEGER     NOT NULL,
    duration_s        DOUBLE PRECISION NOT NULL,
    loaded_at_utc     TIMESTAMPTZ NOT NULL DEFAULT now()
);
