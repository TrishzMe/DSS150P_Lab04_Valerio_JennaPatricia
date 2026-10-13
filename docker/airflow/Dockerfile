# Airflow image for the Lab 04 DAG.
# The pipeline runs in its own virtual environment (/opt/pipeline-venv) built from the same pinned
# requirements.txt as the pipeline image, so Airflow's own dependency constraints can never change
# the pandas / numpy / scikit-learn versions the reproduction depends on.
FROM apache/airflow:3.3.2-python3.12

USER root
RUN mkdir -p /opt/pipeline-venv && chown airflow:root /opt/pipeline-venv
USER airflow

COPY requirements.txt /tmp/pipeline-requirements.txt
RUN python -m venv /opt/pipeline-venv \
    && PIP_USER=false /opt/pipeline-venv/bin/pip install --no-cache-dir --disable-pip-version-check \
       -r /tmp/pipeline-requirements.txt

ENV PIPELINE_PYTHON=/opt/pipeline-venv/bin/python \
    REPRO_PROJECT_ROOT=/opt/project \
    PYTHONHASHSEED=0
