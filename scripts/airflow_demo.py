"""Airflow evidence runs through the Airflow REST API (the same calls the web UI makes).

  python scripts/airflow_demo.py success      # trigger one run, wait, record task states
  python scripts/airflow_demo.py db-outage    # warehouse DB stopped -> load_postgres fails after
                                              # its retries -> DB restarted -> failed tasks cleared
                                              # -> the SAME run completes successfully

Requires `docker compose up -d` and the values in .env. Transcripts are written to
docs/evidence/airflow_<scenario>.txt (+ .json).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DAG_ID = "predictive_repro_pipeline"
EVIDENCE = ROOT / "docs" / "evidence"


def env_value(key: str, default: str | None = None) -> str:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip()
    if default is None:
        raise SystemExit(f"{key} missing from .env")
    return default


BASE = f"http://127.0.0.1:{env_value('AIRFLOW_HOST_PORT', '8081')}"


class Api:
    def __init__(self) -> None:
        body = json.dumps({"username": "admin", "password": env_value("AIRFLOW_ADMIN_PASSWORD")}).encode()
        req = urllib.request.Request(f"{BASE}/auth/token", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            self.token = json.loads(resp.read())["access_token"]

    def call(self, method: str, path: str, payload: dict | None = None, raw: bool = False):
        req = urllib.request.Request(
            f"{BASE}/api/v2{path}", method=method,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json",
                     "Accept": "text/plain" if raw else "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
            return data.decode("utf-8", "replace") if raw else json.loads(data or b"{}")


class Transcript:
    def __init__(self, scenario: str) -> None:
        self.scenario = scenario
        self.lines = [f"Airflow evidence: {scenario}", f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
                      f"DAG: {DAG_ID}   API: {BASE}/api/v2", "=" * 100]
        self.data: dict = {"scenario": scenario, "steps": []}

    def add(self, title: str, text: str = "") -> None:
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self.lines += ["", f"## [{stamp}] {title}"] + ([text] if text else [])
        print(f"[{stamp}] {title}")

    def save(self) -> None:
        EVIDENCE.mkdir(parents=True, exist_ok=True)
        (EVIDENCE / f"airflow_{self.scenario}.txt").write_text("\n".join(self.lines) + "\n", encoding="utf-8", newline="\n")
        (EVIDENCE / f"airflow_{self.scenario}.json").write_text(json.dumps(self.data, indent=2) + "\n",
                                                                encoding="utf-8", newline="\n")


def compose(*args: str) -> str:
    proc = subprocess.run(["docker", "compose", *args], cwd=ROOT, capture_output=True, text=True)
    return (proc.stdout + proc.stderr).strip()


def wait_healthy(service: str, timeout_s: int = 120) -> str:
    name = f"dss150p_lab04_valerio_jennapatricia-{service}-1"
    deadline = time.time() + timeout_s
    status = "unknown"
    while time.time() < deadline:
        status = subprocess.run(["docker", "inspect", "-f", "{{.State.Health.Status}}", name],
                                capture_output=True, text=True).stdout.strip()
        if status == "healthy":
            break
        time.sleep(3)
    return status


def task_table(api: Api, run_id: str) -> tuple[str, list[dict]]:
    tis = api.call("GET", f"/dags/{DAG_ID}/dagRuns/{urllib.request.quote(run_id, safe='')}/taskInstances")["task_instances"]
    rows = sorted(({"task_id": t["task_id"], "state": t["state"], "try_number": t["try_number"],
                    "duration_s": None if t.get("duration") is None else round(t["duration"], 2)} for t in tis),
                  key=lambda r: r["task_id"])
    text = "\n".join(f"  {r['task_id']:<24} {str(r['state']):<16} try={r['try_number']}  duration_s={r['duration_s']}"
                     for r in rows)
    return text, rows


def run_state(api: Api, run_id: str) -> str:
    return api.call("GET", f"/dags/{DAG_ID}/dagRuns/{urllib.request.quote(run_id, safe='')}")["state"]


def trigger(api: Api, conf: dict) -> str:
    run = api.call("POST", f"/dags/{DAG_ID}/dagRuns", {"logical_date": None, "conf": conf})
    return run["dag_run_id"]


def wait_for(api: Api, run_id: str, done: callable, timeout_s: int = 900, every_s: int = 10) -> str:
    deadline = time.time() + timeout_s
    state = run_state(api, run_id)
    while time.time() < deadline and not done(api, run_id):
        time.sleep(every_s)
    return run_state(api, run_id)


def _seconds(value) -> str | None:
    if isinstance(value, dict):
        return f"{int(value.get('days', 0)) * 86400 + int(value.get('seconds', 0))}s"
    return value


def graph_text(api: Api) -> tuple[str, list[dict]]:
    tasks = api.call("GET", f"/dags/{DAG_ID}/tasks")["tasks"]
    edges = [{"task_id": t["task_id"], "downstream": sorted(t["downstream_task_ids"]),
              "retries": None if t.get("retries") is None else int(t["retries"]),
              "execution_timeout": _seconds(t.get("execution_timeout"))} for t in tasks]
    text = "\n".join(f"  {e['task_id']:<24} -> {', '.join(e['downstream']) or '(end)'}   retries={e['retries']}"
                     f" timeout={e['execution_timeout']}" for e in sorted(edges, key=lambda e: e["task_id"]))
    return text, edges


def task_log(api: Api, run_id: str, task_id: str, try_number: int, max_lines: int = 40) -> str:
    """Error-relevant lines of one task attempt (the API returns structured log records)."""
    try:
        res = api.call("GET", f"/dags/{DAG_ID}/dagRuns/{urllib.request.quote(run_id, safe='')}/taskInstances/"
                              f"{task_id}/logs/{try_number}?full_content=true")
    except urllib.error.HTTPError as exc:
        return f"(log unavailable: HTTP {exc.code})"
    lines = []
    for rec in res.get("content", []):
        event = rec.get("event", "") if isinstance(rec, dict) else str(rec)
        level = rec.get("level", "") if isinstance(rec, dict) else ""
        lines.append(f"{level.upper():<7} {event}".strip())
    keep = [ln for ln in lines if any(k in ln for k in
            ("ERROR", "WARNING", "DatabaseUnavailable", "db_connect_failed", "stage_failed", "TASK FAILED",
             "TASK RETRY", "exit code", "Command exited", "command_failed"))]
    return "\n".join(f"  {ln[:420]}" for ln in keep[-max_lines:]) or "(no matching lines)"


def scenario_success(api: Api, t: Transcript) -> bool:
    text, edges = graph_text(api)
    t.add("DAG dependency graph (task -> downstream)", text)
    t.data["graph"] = edges
    conf = {"ingest_mode": "incremental", "impute_scope": "baseline", "selected_partition": "PM2.5"}
    run_id = trigger(api, conf)
    t.add(f"Triggered run {run_id} with conf {conf}")
    state = wait_for(api, run_id, lambda a, r: run_state(a, r) in ("success", "failed"))
    text, rows = task_table(api, run_id)
    t.add(f"Run finished: state={state}", text)
    t.data.update(run_id=run_id, conf=conf, final_state=state, tasks=rows)
    summary = json.loads((ROOT / "outputs" / "run_summary.json").read_text(encoding="utf-8"))
    t.add("outputs/run_summary.json for this run", json.dumps(summary, indent=2))
    t.data["run_summary"] = summary
    return state == "success" and summary["pipeline_run_id"] == run_id


def scenario_db_outage(api: Api, t: Transcript) -> bool:
    t.add("Stopping the warehouse database (docker compose stop postgres)", compose("stop", "postgres"))
    conf = {"ingest_mode": "incremental", "impute_scope": "baseline", "selected_partition": "PM2.5"}
    run_id = trigger(api, conf)
    t.add(f"Triggered run {run_id} with conf {conf}")
    t.data.update(run_id=run_id, conf=conf)

    def load_failed(a: Api, r: str) -> bool:
        _, rows = task_table(a, r)
        return any(x["task_id"] == "load_postgres" and x["state"] == "failed" for x in rows) or run_state(a, r) == "failed"

    state = wait_for(api, run_id, load_failed, timeout_s=900, every_s=15)
    time.sleep(10)
    state = run_state(api, run_id)
    text, rows = task_table(api, run_id)
    t.add(f"After retries were exhausted: run state={state}", text)
    t.data["failed_state"] = {"run_state": state, "tasks": rows}
    tries = next(r["try_number"] for r in rows if r["task_id"] == "load_postgres")
    t.add(f"load_postgres task log (attempt {tries}, error lines)", task_log(api, run_id, "load_postgres", tries))
    record_path = ROOT / "outputs" / "runs"
    records = [p for p in record_path.glob("*/run_record.json") if json.loads(p.read_text(encoding="utf-8"))["pipeline_run_id"] == run_id]
    if records:
        rec = json.loads(records[0].read_text(encoding="utf-8"))
        t.add("Run record while failed (outputs/runs/<run>/run_record.json: status + load_postgres)",
              json.dumps({"status": rec["status"], "load_postgres": rec["stages"].get("load_postgres")}, indent=2))
    published = json.loads((ROOT / "outputs" / "run_summary.json").read_text(encoding="utf-8"))
    t.add("outputs/run_summary.json still belongs to the previous successful run (nothing published)",
          f"  pipeline_run_id={published['pipeline_run_id']} (failed run: {run_id})")

    t.add("Restarting the database (docker compose start postgres)", compose("start", "postgres"))
    t.add(f"postgres health: {wait_healthy('postgres')}")
    cleared = api.call("POST", f"/dags/{DAG_ID}/clearTaskInstances", {
        "dry_run": False, "dag_run_id": run_id, "task_ids": ["load_postgres"], "include_downstream": True,
        "include_upstream": False, "only_failed": False, "reset_dag_runs": True})
    t.add("Cleared load_postgres and its downstream tasks for this run (POST clearTaskInstances)",
          "\n".join(f"  {ti['task_id']}" for ti in cleared.get("task_instances", [])))
    state = wait_for(api, run_id, lambda a, r: run_state(a, r) in ("success", "failed"))
    text, rows = task_table(api, run_id)
    t.add(f"Recovered: run state={state}", text)
    t.data["recovered_state"] = {"run_state": state, "tasks": rows}
    summary = json.loads((ROOT / "outputs" / "run_summary.json").read_text(encoding="utf-8"))
    t.add("outputs/run_summary.json after recovery (same run id)", json.dumps(
        {k: summary[k] for k in ("pipeline_run_id", "status", "reproducibility_outcome", "counts")}, indent=2))
    t.data["run_summary"] = summary
    return state == "success" and summary["pipeline_run_id"] == run_id


def main() -> int:
    scenario = sys.argv[1] if len(sys.argv) > 1 else "success"
    t = Transcript(scenario.replace("-", "_"))
    api = Api()
    ok = {"success": scenario_success, "db-outage": scenario_db_outage}[scenario](api, t)
    t.add(f"RESULT: {'PASS' if ok else 'FAIL'}")
    t.data["result"] = "PASS" if ok else "FAIL"
    t.save()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
