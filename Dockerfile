# Use official Python 3.10 slim image as base
FROM python:3.10-slim

# Prevent Python from writing bytecode and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install essential build tools and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside container
WORKDIR /app

# Copy requirements file first for layer caching
COPY requirements.txt .

# Install main application Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download spaCy English model for Presidio NLP engine
RUN python -m spacy download en_core_web_lg --quiet

# Copy application source code
COPY app/ ./app/

# Pre-cache HuggingFace NER models for offline container execution
RUN python -c "from transformers import AutoTokenizer, AutoModelForTokenClassification; \
models=['cfilt/HiNER-original-muril-base-cased','ai4bharat/IndicNER','Babelscape/wikineural-multilingual-ner']; \
[AutoTokenizer.from_pretrained(m, use_fast=False) for m in models]; \
[AutoModelForTokenClassification.from_pretrained(m) for m in models]" || true

# Create input and output directories
RUN mkdir -p /app/data /app/output

# Environment Variables
ENV HF_TOKEN=""

# Config is mounted at runtime. The job exits after writing the staged dataset.
RUN mkdir -p /app/config
VOLUME ["/app/config", "/app/data", "/app/output", "/app/work"]
ENTRYPOINT ["python", "app/batch_pipeline.py", "--config", "/app/config/config.json"]
