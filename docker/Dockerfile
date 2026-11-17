# Tenth: extraction and matching (stage 1) and narrative check (stage 2) in one service.
# Build from the repo root, so extraction/, matching/, and phase2_model/ are in the build context:
#   docker build -f docker/Dockerfile -t tenth .
# or: docker compose -f docker/compose.yaml up --build
# What may enter the image is listed in Dockerfile.dockerignore next to this file.
FROM python:3.12-slim

# Pin this tag to the version of `uv --version` on your machine once you've built successfully
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app/src:/app/extraction \
    PHASE2_DIR=/app/phase2 \
    EXTRACTION_DIR=/app/extraction \
    TIMEKEEPERS_PATH=/app/data/timekeepers.csv \
    HF_HOME=/opt/hf

# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so this layer is cached until pyproject.toml or uv.lock changes.
# --frozen installs exactly what uv.lock says and fails if the lock is missing or out of date.
COPY matching/pyproject.toml matching/uv.lock ./
RUN uv sync --frozen --no-dev

# Stage 1: matching
COPY matching/src ./src
COPY matching/results ./results
COPY matching/data ./data
COPY matching/frontend ./frontend
COPY matching/app.py .

# Stage 2: narrative check. The run-time code, then the trained models
# (phase2_model/training/train_classifier.py; without models/rewriter/ rewrites come from rules).
COPY phase2_model/pipeline ./phase2
COPY phase2_model/models ./phase2/models

# Stage 1: extraction. The run-time code and prompts only (no photos, voice memos, or saved model answers),
# and the made-up timekeeper list it credits entries to. AWS credentials are passed in at run time, never copied.
COPY extraction/src ./extraction
COPY extraction/data/timekeepers.csv ./data/timekeepers.csv

# Bake the embedding model into the image so the container never needs the network at runtime
RUN python -c "import json; from sentence_transformers import SentenceTransformer as S; S(json.load(open('results/02_config.json'))['embedding_model'])"
ENV HF_HUB_OFFLINE=1

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
