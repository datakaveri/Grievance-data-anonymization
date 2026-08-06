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

# Create input and output directories
RUN mkdir -p /app/data /app/output

# Copy application source code
COPY app/ ./app/

# Environment Variables
ENV HF_TOKEN=""

# Default command to run the PII & NER Pipeline
CMD ["python", "app/main.py", "--input", "/app/data", "--output", "/app/output/pii_ner_report.xlsx"]
