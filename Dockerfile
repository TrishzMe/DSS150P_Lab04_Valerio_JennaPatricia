# Pipeline image: pinned Python + the fully pinned dependency lock + the pipeline code.
# Data, outputs and metadata are bind-mounted by docker-compose so results land in the repository.
FROM python:3.12.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY config ./config
COPY sql ./sql
COPY src ./src
COPY tests ./tests
COPY scripts ./scripts
COPY baseline ./baseline

ENTRYPOINT ["python", "-m", "src.cli"]
CMD ["run-all"]
