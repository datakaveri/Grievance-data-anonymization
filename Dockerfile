# Use official Python 3.10 slim image as base
FROM python:3.10-slim

# Prevent Python from writing bytecode and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_VISIBLE_DEVICES="" \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_TOKEN=""

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

# Install main application Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir hf-transfer || true

# Copy application source code
COPY app/ ./app/

# Pre-cache HuggingFace NER models for offline container execution
RUN python app/model_setup.py || true

# Create input, output, and config directories
RUN mkdir -p /app/data /app/output /app/config /app/work

VOLUME ["/app/config", "/app/data", "/app/output", "/app/work"]

# Default command launches app/main.py on mounted /app/data folder
ENTRYPOINT ["python", "app/main.py"]
CMD ["--input", "app/data", "--output", "app/output/pii_ner_report.xlsx"]
