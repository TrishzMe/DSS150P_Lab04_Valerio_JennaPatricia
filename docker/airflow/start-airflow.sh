#!/usr/bin/env bash
# Start Airflow in standalone mode (api-server, scheduler, dag-processor, triggerer in one container).
# The admin password comes from AIRFLOW_ADMIN_PASSWORD (.env) instead of a generated file.
set -euo pipefail

python - <<'PY'
import json, os
path = os.environ["AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_PASSWORDS_FILE"]
with open(path, "w", encoding="utf-8") as fh:
    json.dump({"admin": os.environ["AIRFLOW_ADMIN_PASSWORD"]}, fh)
PY

airflow db migrate
exec airflow standalone
