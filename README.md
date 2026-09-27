# Grievance Data PII Anonymization & High-Performance NER Evaluation Pipeline

A production-ready, unified processing pipeline for multi-format grievance analysis, language identification, multi-category PII detection (Regex + Presidio), automated contextual redaction, and parallel multi-model Named Entity Recognition (NER) benchmarking.

Built to process administrative, medical, and public grievance documents in English, Indic scripts (Hindi/Devanagari, Bengali, Tamil, Telugu), and mixed languages.

---

## Features

1. **High-Performance Queue-Driven Parallel NER Pipeline (`src/grievance_anonymization/main.py`)**:
   - Multi-threaded fan-out / fan-in worker architecture processing models in parallel.
   - **Optimized CPU Execution**: Runs cleanly on CPU (`CUDA_VISIBLE_DEVICES=""`) with zero CUDA driver probing warnings or delays.
   - **Rust Multi-Threaded Downloads (`hf_transfer`)**: Accelerates HuggingFace model weight downloads by 5x to 10x.
   - **Autograd Memory Reduction (`torch.inference_mode()`)**: Eliminates PyTorch tensor tracking overhead on CPU forward passes.
   - **Smart Single-Pass True-Casing & Line Pre-Filtering**: Cuts inference passes in half and filters out short non-text noise (< 3 characters).

2. **Multi-Format Document & Inline Text String Input Support**:
   - Reads `.txt`, `.docx`, `.doc`, `.html`, `.json`, and `.csv` files (single file or entire directory).
   - Supports direct **inline text string analysis** via command-line argument (`--text "Your text here"`).

3. **Automatic Script & Language Detection**:
   - Classifies document languages into Hindi (Devanagari), Bengali, Tamil, Telugu, English, Mixed, or Unknown.

4. **Structured & Contextual PII Detection Engine**:
   - Labeled and unlabelled pattern scanning with false-positive filtering.
   - **20+ PII Categories**:
     - *Government & Identifiers*: Aadhaar Number, PAN Card, Voter ID, Passport Number, Driving License, Parivar Pehchan Patra (PPP / Family ID), User ID / Portal ID.
     - *Financial*: Bank Account Number (with context validation), Credit / Debit Card Number, IFSC Code.
     - *Contact & Network*: Phone Number, Email Address, IP Address.
     - *Administrative & Location*: Indian Cities / Districts (e.g., Karnal, Jhajjar, Rohtak, Gurugram, Delhi), Pin Codes.
     - *Personal Particulars*: Person Names with Honorifics (*Shri*, *Smt.*, *Dr.*, *Prof.*), Date of Birth (DOB), Age.

5. **Granular Word & Character Position Tracking**:
   - Computes Line No, Word No, Word Position (ordinal: `1st`, `2nd`), Start Letter & End Letter character offsets, and Letter Span (`12-24`).
   - Includes Confidence Scores (`1.00 (Exact Regex)` or model prediction probabilities).

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

## Repository Layout

```text
Grievance-data-anonymization/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml               # Lint, test, build CI workflow
│   │   ├── release.yml          # Tagged release and Docker publish
│   │   └── codeql.yml           # Security scanning
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.yml
│   │   └── feature_request.yml
│   ├── PULL_REQUEST_TEMPLATE.md
│   ├── CODEOWNERS               # Code reviewers
│   └── dependabot.yml           # Dependency update configuration
├── src/
│   └── grievance_anonymization/
│       ├── __init__.py          # Package initialization (__version__)
│       ├── main.py              # Main parallel PII/NER pipeline execution script
│       ├── batch_pipeline.py    # Config-driven dataset batch mode for downstream integration
│       ├── model_setup.py       # Dependency setup & HuggingFace model downloader
│       └── run_100_cases_pipeline.py # Benchmark & process runner
├── tests/
│   ├── unit/                    # Unit tests
│   └── integration/             # End-to-end integration tests
├── docs/
│   └── architecture.md          # Architecture overview
├── config/
│   ├── config.json              # Active configuration
│   └── config.example.json      # Committed example configuration template
├── .dockerignore
├── .gitignore
├── .env.example                 # Environment variables specification
├── .pre-commit-config.yaml      # Code quality & secret scanner hooks
├── pyproject.toml               # Python package metadata & dependencies
├── requirements.txt             # Pinned requirements with comments
├── requirements.lock            # Exact pinned lockfile
├── Dockerfile                   # Production multi-stage Docker build
├── docker-compose.yml           # Docker Compose deployment setup
├── CHANGELOG.md                 # Release history
├── CONTRIBUTING.md              # Guidelines for contributing
├── SECURITY.md                  # Security and vulnerability reporting
├── LICENSE                      # Apache-2.0 open-source licence
└── README.md                    # Project documentation
```

---

## Quickstart Guide

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

# 3. Install package in editable mode
pip install -e .[dev]

# 4. Pre-download NER models (HiNER, IndicNER, XLM-RoBERTa)
python -m grievance_anonymization.model_setup

# 5. Run High-Performance Parallel Pipeline on input folder or file
python -m grievance_anonymization.main --input data/sample_complaint.txt --output output/pii_ner_report.xlsx

# 6. OR Run pipeline directly on an inline text string
python -m grievance_anonymization.main --text "Shri Ramesh Kumar, Aadhaar 2345 6789 0123, email: ramesh@example.com" --output output/inline_report.xlsx
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
  -v $(pwd)/config:/app/config \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  grievance-anonymizer:latest
```

---

### Option 3: Running with Docker Compose

```bash
# Build and launch container
docker compose up --build
```

---

## Command Line Arguments (`src/grievance_anonymization/main.py`)

| Argument | Short Flag | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input` | `-i` | `data/sample_complaint.txt` | Path to a single file (`.txt`/`.docx`/`.doc`/`.html`/`.json`/`.csv`) or folder containing documents. |
| `--text` | `-t` | `None` | Inline text string to analyze directly instead of reading files. |
| `--output` | `-o` | `pii_ner_report.xlsx` | Output `.xlsx` file path (JSON reports generated automatically). |
| `--ner-batch-size` | | `24` | Batch size for parallel line inference across model workers. |
| `--queue-size` | | `64` | Maximum queue size for bounded dispatcher backpressure. |
| `--hf_token` | | `""` | Optional HuggingFace Access Token for gated models. |

---

## Dataset Batch Job (`src/grievance_anonymization/batch_pipeline.py`)

The batch job anonymizes configured columns of a CSV/XLSX/JSON dataset. It reads its settings from the `free_text_anonymization` object in the dataset config:

| Config key | Default | Description |
| :--- | :---: | :--- |
| `enabled` | — | Must be `true` for the job to write a staged file. |
| `columns` | — | List of column names to anonymize. All must exist in the input. |
| `minimum_confidence` | `0.0` | Detections below this confidence are ignored. |
| `ner_batch_size` | `256` | Lines handed to each inference call. |
| `staged_input_path` | — | Where the anonymized CSV is written (must end in `.csv`). |
| `audit_output_path` | — | Optional JSON audit of every applied detection. |
| `on_failure` | `"fail"` | `"fail"` aborts the run on a cell error; `"continue"` records it. |

---

## Security & Privacy

All processing is executed **100% locally** on your machine or container. No document content, personal identifiers, or metadata are transmitted to external services.
