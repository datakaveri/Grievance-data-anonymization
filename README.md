# 🛡️ Grievance Data PII Anonymization & High-Performance NER Evaluation Pipeline

A production-ready, unified processing pipeline for **multi-format grievance analysis, language identification, multi-category PII detection (Regex + Presidio), automated contextual redaction, and parallel multi-model Named Entity Recognition (NER) benchmarking**.

Built to process administrative, medical, and public grievance documents in **English, Indic scripts (Hindi/Devanagari, Bengali, Tamil, Telugu), and Mixed languages**.

---

## 🌟 Key Features

1. **High-Performance Queue-Driven Parallel NER Pipeline (`app/main.py`)**:
   - Multi-threaded fan-out / fan-in worker architecture processing models in parallel.
   - **Optimized CPU Execution**: Runs cleanly on CPU (`CUDA_VISIBLE_DEVICES=""`) with zero CUDA driver probing warnings or delays.
   - **Rust Multi-Threaded Downloads (`hf_transfer`)**: Accelerates HuggingFace model weight downloads by **5x to 10x**.
   - **Autograd Memory Reduction (`torch.inference_mode()`)**: Eliminates PyTorch tensor tracking overhead on CPU forward passes.
   - **Smart Single-Pass True-Casing & Line Pre-Filtering**: Cuts inference passes in half and filters out short non-text noise (< 3 characters).

2. **Multi-Format Document & Inline Text String Input Support**:
   - Reads `.txt`, `.docx`, `.doc`, `.html`, `.json`, and `.csv` files (single file or entire directory).
   - Supports direct **inline text string analysis** via command-line argument (`--text "Your text here"`).

3. **Automatic Script & Language Detection**:
   - Classifies document languages into **Hindi (Devanagari)**, **Bengali**, **Tamil**, **Telugu**, **English**, **Mixed**, or **Unknown**.

4. **Structured & Contextual PII Detection Engine**:
   - Labeled and unlabelled pattern scanning with false-positive filtering.
   - **20+ PII Categories**:
     - *Government & Identifiers*: Aadhaar Number, PAN Card, Voter ID, Passport Number, Driving License, Parivar Pehchan Patra (PPP / Family ID), User ID / Portal ID.
     - *Financial*: Bank Account Number (with context validation), Credit / Debit Card Number, IFSC Code.
     - *Contact & Network*: Phone Number, Email Address, IP Address.
     - *Administrative & Location*: Indian Cities / Districts (e.g., Karnal, Jhajjar, Rohtak, Gurugram, Delhi), Pin Codes.
     - *Personal Particulars*: Person Names with Honorifics (*Shri*, *Smt.*, *Dr.*, *Prof.*), Date of Birth (DOB), Age.

5. **Granular Word & Character Position Tracking**:
   - Computes **Line No**, **Word No**, **Word Position** (ordinal: `1st`, `2nd`, `3rd`), **Start Letter** & **End Letter** character offsets, and **Letter Span** (`12-24`).
   - Includes **Confidence Scores** (`1.00 (Exact Regex)` or model prediction probabilities).

6. **Privacy-Preserving Anonymization Strategies**:
   - **Partial Masking**: Aadhaar (`XXXX XXXX 1234`), Phone (`XXXXXX9876`), Bank Account (`********5678`), PAN (`ABCDE****F`).
   - **Domain-Preserving Masking**: Email (`jo****@domain.com`).
   - **Tokenization**: Credit Card (`XXXX-XXXX-XXXX-4321`).
   - **Initial-Only Masking**: Person Names (`R. K. S.`).
   - **One-Way Hashing**: SHA-256 hash for generic tokens.

7. **Multilingual NER Ensemble**:
   Uses three models for multilingual entity extraction:
   - **HiNER**: `cfilt/HiNER-original-muril-base-cased` (IIT Bombay / MuRIL)
   - **IndicNER**: `ai4bharat/IndicNER` (AI4Bharat Multilingual)
   - **XLM-RoBERTa**: `Babelscape/wikineural-multilingual-ner` (Multilingual)
   - **Hybrid**: Ensemble of HiNER + IndicNER + XLM-RoBERTa.

8. **Multi-Report Output System (Excel & Per-Model JSONs)**:
   - **4-Sheet Color-Coded Excel Workbook**:
     - *Sheet 1: Summary*: Overview of processed files, script language, total PII hits, and detection status.
     - *Sheet 2: PII Detection*: Detailed audit log of detected PII with line numbers, word positions, letter spans, detector source, display names, and XML tags.
     - *Sheet 3: NER Comparison*: Model-by-model comparison of predicted types (`PERSON`, `LOCATION`, `ORGANIZATION`), extracted entity text, and strict missed entity computation.
     - *Sheet 4: Anonymization*: Complete record of original values vs anonymized output, confidence scores, position metrics, redaction technique, and description.
   - **Per-Model JSON Reports**: Automatically generates a master JSON report (`pii_ner_report.json`) as well as separate model-specific JSON files (`pii_ner_report_HiNER.json`, `pii_ner_report_IndicNER.json`, `pii_ner_report_Hybrid.json`, etc.).

---

## 📂 Project Structure

```text
Grievance-data-anonymization/
├── app/
│   ├── main.py              # Main parallel PII/NER pipeline execution script
│   ├── batch_pipeline.py    # Config-driven dataset batch mode for downstream SKALD integration
│   └── model_setup.py       # Dependency setup & HuggingFace model downloader
├── data/                    # Input folder for document datasets (.txt, .docx, .doc, .html, .json, .csv)
├── output/                  # Output directory for generated Excel (.xlsx) and JSON (.json) reports
├── Dockerfile               # Production Docker container definition
├── docker-compose.yml       # Docker Compose configuration with volume mounts
├── requirements.txt         # Python dependency specifications
└── README.md                # Pipeline documentation
```

---

## ⚡ Quickstart Guide

### Option 1: Local Setup (Native Python)

#### Prerequisites
- Python 3.10+
- PyTorch (CPU or CUDA)

#### Steps:
```bash
# 1. Clone repository
git clone https://github.com/datakaveri/Grievance-data-anonymization.git
cd Grievance-data-anonymization

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Pre-download NER models (HiNER, IndicNER, XLM-RoBERTa)
python app/model_setup.py

# 4. Create data and output directories
mkdir -p app/data app/output

# 5. Run High-Performance Parallel Pipeline on input folder or file
python app/main.py --input app/data/sample_complaint.txt --output app/output/pii_ner_report.xlsx

# 6. OR Run pipeline directly on an inline text string
python app/main.py --text "Shri Ramesh Kumar, Aadhaar 2345 6789 0123, email: ramesh@example.com" --output app/output/inline_report.xlsx
```

---

### Option 2: Running with Docker

#### 1. Build Docker Image
```bash
docker build -t grievance-anonymizer:latest .
```

#### 2. Run Container
```bash
docker run --rm \
  -v $(pwd)/app/data:/app/data \
  -v $(pwd)/app/output:/app/output \
  grievance-anonymizer:latest
```

---

### Option 3: Running with Docker Compose

```bash
# 1. Create data and output directories
mkdir -p app/data app/output

# 2. Build and launch container
docker compose up --build
```

---

## ⚙️ Command Line Arguments (`app/main.py`)

| Argument | Short Flag | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input` | `-i` | `app/data/sample_complaint.txt` | Path to a single file (`.txt`/`.docx`/`.doc`/`.html`/`.json`/`.csv`) or folder containing documents. |
| `--text` | `-t` | `None` | Inline text string to analyze directly instead of reading files. |
| `--output` | `-o` | `pii_ner_report.xlsx` | Output `.xlsx` file path (JSON reports generated automatically). |
| `--ner-batch-size` | | `24` | Batch size for parallel line inference across model workers. |
| `--queue-size` | | `64` | Maximum queue size for bounded dispatcher backpressure. |
| `--hf_token` | | `""` | Optional HuggingFace Access Token for gated models. |

---

## 📊 Performance & Timing Benchmarks

| Execution Stage | Original Baseline | Optimized Parallel Pipeline | Net Speedup |
| :--- | :---: | :---: | :---: |
| **Model Downloads** *(Uncached)* | ~12.2 mins | ~1.26 mins | **9.6x Faster Download** |
| **Model Weight Loading** | 10.15s | 3.74s | **2.7x Faster Loading** |
| **NER Model Execution** | 23.53s | 14.70s | **1.6x Faster Inference** |
| **Total Completion Time (Cached)** | 34.62s | **15.81s** | **2.19x Faster Total Pipeline** |

---

## 🔒 Security & Privacy

All processing is executed **100% locally** on your machine or container. No document content, personal identifiers, or metadata are transmitted to external services.
