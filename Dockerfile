# syntax=docker/dockerfile:1
# Use official Python 3.10 slim image as base
FROM python:3.10-slim

# Prevent Python from writing bytecode and enable unbuffered output.
# CUDA_VISIBLE_DEVICES is pinned empty: a TEE has no GPU passthrough, and this
# keeps torch from probing for one on every start.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_VISIBLE_DEVICES="" \
    HF_HUB_ENABLE_HF_TRANSFER=1

# Install essential build tools and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside container
WORKDIR /app

# Copy requirements file first for layer caching
COPY requirements.txt .

# Install main application Python dependencies.
# torch is installed from the CPU-only wheel index first: the default PyPI wheel
# bundles CUDA/cuDNN/NCCL (~2GB extra) that a TEE never uses, since GPU passthrough
# isn't available there and this pipeline already falls back to CPU automatically.
# hf-transfer is installed unconditionally, not best-effort: HF_HUB_ENABLE_HF_TRANSFER=1
# above makes huggingface_hub fail hard if the package is absent.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir hf-transfer

# NOTE: en_core_web_lg is deliberately NOT downloaded here. Presidio is disabled
# on this branch (app/main.py: full_pii_scan returns regex hits only and
# _get_presidio_engine is commented out), so the ~600MB spaCy model would never
# be loaded. Re-add `RUN python -m spacy download en_core_web_lg --quiet` if
# Presidio is switched back on.

# Copy application source code
COPY app/ ./app/

# Pre-cache HuggingFace NER models for offline container execution.
# ai4bharat/IndicNER is a gated (but self-service, MIT-licensed) repo: accept its
# terms once at https://huggingface.co/ai4bharat/IndicNER, generate a read token at
# https://huggingface.co/settings/tokens, then build with:
#   DOCKER_BUILDKIT=1 docker build --secret id=hf_token,env=HF_TOKEN -t <tag> .
# The token is only mounted for this RUN step and is never written to image
# layers or history. Without it, this step still succeeds for the other two
# models — IndicNER alone is skipped here and again (gracefully) at runtime.
# The model list and, crucially, each model's use_fast flag live in
# main._HYBRID_NER_SPECS, and --prefetch reads them from there, so the build
# caches exactly the tokenizer variant the pipeline loads at runtime. Caching a
# tokenizer as slow when the pipeline asks for it fast leaves a hole in the cache
# that only shows up as a network call inside an offline TEE.
# The three models download concurrently; add --strict to fail the build when any
# of them is missing (default: warn, so a tokenless build still produces an image
# that runs with the two ungated models).
RUN --mount=type=secret,id=hf_token,env=HF_TOKEN \
    python app/model_setup.py --prefetch

# Environment Variables
ENV HF_TOKEN=""

# Create the mount points. These are absolute paths under /, NOT /app/app —
# the entrypoint below reads /app/data and writes /app/output, which is where
# docker-compose.yml and the README mount the host directories.
RUN mkdir -p /app/config /app/data /app/output /app/work

# Config is mounted at runtime. The job scans /app/config for its JSON config
# file rather than requiring a fixed name (see batch_pipeline._resolve_config_file),
# so it doesn't matter what the caller names it. The job exits after writing the
# staged dataset.
VOLUME ["/app/config", "/app/data", "/app/output", "/app/work"]
ENTRYPOINT ["python", "app/batch_pipeline.py", "--config", "/app/config"]
