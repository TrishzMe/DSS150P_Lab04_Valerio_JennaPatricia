-- SQL transformations used by src/storage.py. Each block starts with "-- name: <id>".
-- Blocks with {placeholders} are composed with psycopg.sql (identifiers are quoted safely).

-- name: merge_snapshot
-- Rerun-safe snapshot merge, executed inside ONE transaction together with the COPY into the
-- temporary table and the delete below. Re-running it with the same snapshot changes nothing:
-- rows are inserted when the key is new, updated only when row_hash changed, otherwise skipped.
-- If a retry happens after a commit whose acknowledgement was lost, the retry finds every key
-- with an identical row_hash and is a no-op.
INSERT INTO {target} AS t ({columns})
SELECT {columns} FROM {source}
ON CONFLICT ({key}) DO UPDATE
    SET {update_set}, loaded_at_utc = now()
    WHERE t.row_hash IS DISTINCT FROM EXCLUDED.row_hash
RETURNING (xmax = 0) AS inserted;

-- name: delete_missing
-- Keys that disappeared from the new snapshot are removed so the table mirrors the snapshot.
DELETE FROM {target} AS t
WHERE NOT EXISTS (SELECT 1 FROM {source} AS s WHERE s.{key} = t.{key});

-- name: create_views
-- ELT cross-check: rebuild the station-day table inside PostgreSQL from staging.
CREATE OR REPLACE VIEW curated.v_station_day_from_staging AS
SELECT country_label,
       city,
       location,
       reading_date,
       avg(value) FILTER (WHERE pollutant = 'PM2.5') AS pm25,
       avg(value) FILTER (WHERE pollutant = 'PM10')  AS pm10,
       avg(value) FILTER (WHERE pollutant = 'NO2')   AS no2,
       avg(value) FILTER (WHERE pollutant = 'O3')    AS o3,
       avg(value) FILTER (WHERE pollutant = 'SO2')   AS so2,
       avg(value) FILTER (WHERE pollutant = 'CO')    AS co,
       count(*)                                      AS n_readings
FROM staging.air_quality_readings
GROUP BY country_label, city, location, reading_date;

-- Consumer view: WHO exceedance share per country on the model-ready station-days.
CREATE OR REPLACE VIEW curated.v_exceedance_by_country AS
SELECT country_label,
       count(*)                                                   AS station_days,
       sum((air_quality = 'Unhealthy')::int)                      AS unhealthy_station_days,
       round(avg((air_quality = 'Unhealthy')::int)::numeric, 4)   AS unhealthy_share,
       round(avg(pm25)::numeric, 2)                               AS mean_pm25
FROM curated.model_ready
GROUP BY country_label;

-- Operations view: one line per pipeline run with task outcomes.
CREATE OR REPLACE VIEW audit.v_run_overview AS
SELECT r.pipeline_run_id,
       r.status,
       r.started_at_utc,
       r.ended_at_utc,
       count(t.task_name)                                AS tasks,
       count(t.task_name) FILTER (WHERE t.status = 'success') AS tasks_succeeded,
       count(t.task_name) FILTER (WHERE t.status = 'failed')  AS tasks_failed,
       max(t.attempt)                                    AS max_attempt,
       r.counts
FROM audit.pipeline_runs r
LEFT JOIN audit.task_runs t USING (pipeline_run_id)
GROUP BY r.pipeline_run_id, r.status, r.started_at_utc, r.ended_at_utc, r.counts;

-- name: elt_consistency
-- Compares the SQL-built station-day view with the Python-built curated table.
SELECT count(*) FILTER (WHERE t.station_day_key IS NULL)  AS only_in_sql_view,
       count(*) FILTER (WHERE v.location IS NULL)         AS only_in_curated_table,
       count(*) FILTER (WHERE (v.pm25 IS NULL) <> (t.pm25 IS NULL) OR (v.no2 IS NULL) <> (t.no2 IS NULL)
                           OR (v.o3 IS NULL) <> (t.o3 IS NULL) OR (v.so2 IS NULL) <> (t.so2 IS NULL)
                           OR (v.co IS NULL) <> (t.co IS NULL) OR (v.pm10 IS NULL) <> (t.pm10 IS NULL)) AS null_mismatches,
       coalesce(max(greatest(abs(v.pm25 - t.pm25), abs(v.pm10 - t.pm10), abs(v.no2 - t.no2),
                             abs(v.o3 - t.o3), abs(v.so2 - t.so2), abs(v.co - t.co))), 0) AS max_abs_diff,
       count(*) FILTER (WHERE v.n_readings <> t.n_readings) AS n_readings_mismatches
FROM curated.v_station_day_from_staging v
FULL JOIN curated.station_day_pollutants t USING (country_label, city, location, reading_date);

-- name: duplicate_business_keys
-- Rerun evidence: duplicate business keys per table (must stay 0 after any number of runs).
SELECT 'staging.air_quality_readings' AS table_name, count(*) AS total_rows,
       count(*) - count(DISTINCT (location, pollutant, last_updated_utc)) AS duplicate_business_keys
FROM staging.air_quality_readings
UNION ALL
SELECT 'curated.station_day_pollutants', count(*),
       count(*) - count(DISTINCT (country_label, city, location, reading_date))
FROM curated.station_day_pollutants
UNION ALL
SELECT 'curated.model_ready', count(*), count(*) - count(DISTINCT station_day_key)
FROM curated.model_ready
UNION ALL
SELECT 'curated.predictions', count(*), count(*) - count(DISTINCT (station_day_key, model_name))
FROM curated.predictions;
