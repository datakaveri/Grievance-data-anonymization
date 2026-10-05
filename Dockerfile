# syntax=docker/dockerfile:1
# ── Stage 1: Build stage ──────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_VISIBLE_DEVICES="" \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir hf-transfer && \
    pip install --no-cache-dir .

# ── Stage 2: Final runtime stage ──────────────────────────────────────────────
FROM python:3.10-slim AS final

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_VISIBLE_DEVICES="" \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTHONPATH=/app/src

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -g 10001 appuser && \
    useradd -u 10001 -g appuser -s /bin/bash -m appuser

WORKDIR /app

# Copy site-packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src/ ./src/
COPY config/ ./config/

RUN mkdir -p /app/config /app/data /app/output /app/work && \
    chown -R appuser:appuser /app

# Pre-cache HuggingFace NER models
RUN --mount=type=secret,id=hf_token,env=HF_TOKEN \
    python -m grievance_anonymization.model_setup --prefetch

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import grievance_anonymization; print('healthy')" || exit 1

VOLUME ["/app/config", "/app/data", "/app/output", "/app/work"]
ENTRYPOINT ["python", "-m", "grievance_anonymization.main"]
CMD ["--config", "/app/config"]
