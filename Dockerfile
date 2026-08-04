# Use official Python 3.10 slim image as base
FROM python:3.10-slim

# Prevent Python from writing bytecode and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install essential system dependencies for OpenCV, PyTorch, and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    python3-venv \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside container
WORKDIR /app

# Copy requirements file first for layer caching
COPY requirements.txt .

# Install main application Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-build isolated virtual environment for Chandra 2 OCR engine
RUN python3 -m venv /app/chandra_venv && \
    /app/chandra_venv/bin/python -m pip install --no-cache-dir --upgrade pip && \
    /app/chandra_venv/bin/python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121 && \
    /app/chandra_venv/bin/python -m pip install --no-cache-dir "transformers>=4.40,<5.0" "chandra-ocr[hf]" "pillow>=10.0" "accelerate>=0.26"

# Create directories for data inputs and outputs
RUN mkdir -p /app/data/source_images /app/output

# Copy application source code
COPY app/ ./app/

# Environment Variables
ENV BASE_DIR=/app \
    IMAGE_FOLDER_PATH=/app/data/source_images \
    OUTPUT_DIR=/app/output \
    CHANDRA_VENV=/app/chandra_venv \
    CHANDRA_VENV_PY=/app/chandra_venv/bin/python \
    CHANDRA_SCRIPT=/app/chandra_infer_script.py

# Command to execute setup and main batch anonymization pipeline
CMD ["sh", "-c", "python app/chandra2_setup.py && python app/main.py"]
