# 🛡️ Grievance Data Anonymization Pipeline

A production-ready, unified batch processing pipeline for **OCR-based document text extraction, Named Entity Recognition (NER), multi-category PII detection, and automated redaction/anonymization**.

Built specifically for handling complex document grievances in English and Devanagari/Hindi scripts.

---

## 🌟 Key Features

1. **Chandra 2 Layout-Aware OCR Engine**: Batch document text extraction with GPU memory optimization and automatic image resizing.
2. **Multi-Model NER Benchmark**: Evaluates and compares 4 specialized NER models:
   - `cfilt/HiNER-original-muril-base-cased` (HiNER - IIT Bombay / MuRIL)
   - `ai4bharat/IndicNER` / `techysanoj/fine-tuned-IndicNER` (IndicNER)
   - `dslim/bert-base-NER` (BERT English)
   - `Babelscape/wikineural-multilingual-ner` (XLM-RoBERTa Multilingual)
3. **20-Category Expanded PII Detection**:
   - **Identifiers**: Aadhaar, PAN, Voter ID, Passport, PPP ID, Vehicle Number, User ID
   - **Contact & Web**: Email, Phone Number, IP Address
   - **Financial**: Credit Card Number, Bank Account Number
   - **Dates & Temporal**: Date of Birth (DOB), Generic Dates
   - **Entities & Context**: Name (PER), Location (LOC), Address, Organization (ORG), Personal Relations (S/o, D/o), Sign/Signature Keywords
4. **Custom Anonymization Strategies**: Redacts and masks sensitive values while retaining data utility (e.g., masking Aadhaar as `XXXX XXXX 1234`, PAN as `ABCDE****F`, Email as `ab****@domain.com`).
5. **Comprehensive Excel Report Generation**: Exports complete comparative metrics across 3 structured sheets:
   - **Model Comparison Summary**: Aggregated accuracy, detection counts, and metrics per model.
   - **OCR & NER Line Details**: Line-by-line detailed breakdown of extracted text, detected entities, and anonymized outputs.
   - **PII Aggregated by Image**: Document-level consolidated PII detection and redaction log.

---

## 📂 Project Structure

```text
Grievance-data-anonymization/
├── app/
│   ├── chandra2_setup.py   # Initializes Chandra OCR isolated venv and inference script
│   └── main.py             # Main execution workflow (OCR -> NER -> PII -> Excel Export)
├── data/
│   └── source_images/      # Place your input document images here (.jpg, .png, .tiff, etc.)
├── output/                 # Output folder for generated Excel reports and temp manifests
├── Dockerfile              # Container definition for reproducible deployment
├── docker-compose.yml      # Orchestration config for running container with volumes
├── .dockerignore           # Excluded files for Docker build context
├── .gitignore              # Git ignored files and cache patterns
├── requirements.txt        # Clean Python dependencies
└── README.md               # Project documentation
```

---

## ⚡ Quickstart Guide

### Option 1: Running with Docker (Recommended)

#### Prerequisites
- [Docker Engine](https://docs.docker.com/get-docker/) installed.
- (Optional but recommended for speed) [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for GPU acceleration.

#### 1. Build Docker Image
```bash
docker build -t grievance-anonymizer:latest .
```

#### 2. Run Container

**GPU Mode (Recommended):**
```bash
docker run --gpus all \
  -v $(pwd)/data/source_images:/app/data/source_images \
  -v $(pwd)/output:/app/output \
  -e HF_TOKEN="your_huggingface_token_here" \
  grievance-anonymizer:latest
```

**CPU Mode:**
```bash
docker run \
  -v $(pwd)/data/source_images:/app/data/source_images \
  -v $(pwd)/output:/app/output \
  grievance-anonymizer:latest
```

---

### Option 2: Running with Docker Compose

```bash
# 1. Add your images into data/source_images/
mkdir -p data/source_images output

# 2. (Optional) Set your HuggingFace token
export HF_TOKEN="your_huggingface_token_here"

# 3. Build and launch
docker compose up --build
```

---

### Option 3: Local Setup (Native Python)

#### Prerequisites
- Python 3.10+
- PyTorch (CUDA recommended if GPU is present)

#### Steps:
```bash
# 1. Clone repo & navigate into project directory
git clone https://github.com/datakaveri/Grievance-data-anonymization.git
cd Grievance-data-anonymization

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install core dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Run Chandra 2 Setup Script
python app/chandra2_setup.py

# 5. Add input document images to data/source_images/
mkdir -p data/source_images output

# 6. Run the Main Pipeline
python app/main.py
```

---

## ⚙️ Environment Variables

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `HF_TOKEN` | *None* | Optional HuggingFace Access Token (required for gated models like IndicNER). |
| `IMAGE_FOLDER_PATH` | `/app/data/source_images` | Path to directory containing source document images. |
| `OUTPUT_DIR` | `/app/output` | Path to directory where output Excel reports are saved. |
| `CHANDRA_VENV_PY` | `/app/chandra_venv/bin/python` | Executable path for Chandra OCR virtual environment. |
| `CHANDRA_SCRIPT` | `/app/chandra_infer_script.py` | Path to Chandra OCR inference worker script. |

---

## 📊 Output Files

Upon successful execution, the pipeline generates:
`output/chandra_ner_pii_batch_comparison.xlsx` containing:
1. **Model Comparison Summary**: Overall benchmark matrix comparing entity detection rates across models.
2. **OCR & NER Line Details**: Complete line-level extraction, classification, and masking results.
3. **PII Aggregated by Image**: Image-level consolidated view of all redacted personal information.

---

## 🔒 Security & Privacy Note

All processing (OCR extraction, NER inference, regex scanning, and redaction) is performed **100% locally** or within your container instance. No document text or extracted PII is sent to external API endpoints.
