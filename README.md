# 🛡️ Grievance Data PII Anonymization & NER Evaluation Pipeline

A production-ready, unified processing pipeline for **multi-format grievance analysis, language identification, multi-category PII detection (Regex + Microsoft Presidio), automated contextual redaction, and multi-model Named Entity Recognition (NER) benchmarking**.

Built to process administrative, medical, and public grievance documents in **English, Indic scripts (Hindi/Devanagari, Bengali, Tamil, Telugu), and Mixed languages**.

---

## 🌟 Key Features

1. **Multi-Format Document & Inline Text String Input Support**:
   - Reads `.txt`, `.docx`, `.doc`, and `.html` files (single file or entire folder).
   - Supports direct **inline text string analysis** via command-line argument (`--text "Your text here"`).

2. **Automatic Script & Language Detection**:
   - Classifies document languages into **Hindi (Devanagari)**, **Bengali**, **Tamil**, **Telugu**, **English**, **Mixed**, or **Unknown**.

3. **Hybrid PII Detection Engine (Regex + Presidio)**:
   - **Structured Regex & Contextual Rules**: Labeled and unlabelled pattern scanning with false-positive filtering.
   - **Microsoft Presidio Analyzer**: Leverages spaCy (`en_core_web_lg`) for deep NLP-based entity detection.
   - **20+ PII Categories**:
     - *Government & Identifiers*: Aadhaar Number, PAN Card, Voter ID, Passport Number, Driving License, Parivar Pehchan Patra (PPP / Family ID), User ID / Portal ID.
     - *Financial*: Bank Account Number (with context validation), Credit / Debit Card Number, IFSC Code.
     - *Contact & Network*: Phone Number, Email Address, IP Address.
     - *Administrative & Location*: Haryana & Indian Cities / Districts (e.g., Karnal, Jhajjar, Rohtak, Gurugram, Delhi), Pin Codes.
     - *Personal Particulars*: Person Names with Honorifics (*Shri*, *Smt.*, *Dr.*, *Prof.*), Date of Birth (DOB), Age.

4. **Granular Word & Character Position Tracking**:
   - Computes **Line No**, **Word No**, **Word Position** (ordinal: `1st`, `2nd`, `3rd`), **Start Letter** & **End Letter** character offsets, and **Letter Span** (`12-24`).
   - Includes **Confidence Scores** (`1.00 (Exact Regex/Presidio)` or model prediction probabilities).

5. **Privacy-Preserving Anonymization Strategies**:
   - **Partial Masking**: Aadhaar (`XXXX XXXX 1234`), Phone (`XXXXXX9876`), Bank Account (`********5678`), PAN (`ABCDE****F`).
   - **Domain-Preserving Masking**: Email (`jo****@domain.com`).
   - **Tokenization**: Credit Card (`XXXX-XXXX-XXXX-4321`).
   - **Initial-Only Masking**: Person Names (`R. K. S.`).
   - **One-Way Hashing**: SHA-256 hash for generic tokens.

6. **Multilingual NER Ensemble**:
   Uses three models for multilingual entity extraction:
   - **HiNER**: `cfilt/HiNER-original-muril-base-cased` (IIT Bombay / MuRIL)
   - **IndicNER**: `ai4bharat/IndicNER` (AI4Bharat Multilingual)
   - **XLM-RoBERTa**: `Babelscape/wikineural-multilingual-ner` (Multilingual)
   - **Hybrid**: Ensemble of HiNER + IndicNER + XLM-RoBERTa.

7. **Multi-Report Output System (Excel & Per-Model JSONs)**:
   - **4-Sheet Color-Coded Excel Workbook**:
     - *Sheet 1: Summary*: Overview of processed files, script language, total PII hits, and detection status.
     - *Sheet 2: PII Detection*: Detailed audit log of detected PII with line numbers, word positions, letter spans, detector source (Regex vs Presidio), display names, and XML tags (`<Aadhaar>...</Aadhaar>`).
     - *Sheet 3: NER Comparison*: Model-by-model comparison of predicted types (`PERSON`, `LOCATION`, `ORGANIZATION`), extracted entity text, and strict missed entity computation.
     - *Sheet 4: Anonymization*: Complete record of original values vs anonymized output, confidence scores, position metrics, redaction technique, and description.
   - **Per-Model JSON Reports**: Automatically generates a master JSON report (`pii_ner_report.json`) as well as separate model-specific JSON files (`pii_ner_report_hiner.json`, `pii_ner_report_indicner.json`, `pii_ner_report_hybrid.json`, etc.).

---

## 📂 Project Structure

```text
Grievance-data-anonymization/
├── app/
│   ├── main.py             # Core pipeline execution script (Language -> PII -> NER -> Position Tracking -> Excel/JSON Export)
│   └── model_setup.py      # Dependency setup & Hugging Face model pre-downloader
├── data/                   # Input folder for document datasets (.txt, .docx, .doc, .html)
├── output/                 # Output directory for generated Excel (.xlsx) and JSON (.json) reports
├── Dockerfile              # Docker container definition with model pre-caching
├── docker-compose.yml      # Docker Compose configuration with volume mounts
├── requirements.txt        # Python dependency specifications
└── README.md               # Pipeline documentation
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

# 3. Automated Dependency & Model Setup (Downloads spaCy & 4 NER models to cache)
python app/model_setup.py

# OR Manual Setup
pip install -r requirements.txt
python -m spacy download en_core_web_lg

# 4. Create data and output folders
mkdir -p data output

# 5. Run pipeline on input folder/file
python app/main.py --input data --output output/pii_ner_report.xlsx

# 6. OR Run pipeline directly on an inline text string
python app/main.py --text "Shri Ramesh Kumar, Aadhaar 2345 6789 0123, email: ramesh@example.com" --output output/inline_report.xlsx
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
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  grievance-anonymizer:latest
```

---

### Option 3: Running with Docker Compose

```bash
# 1. Create input data and output folders
mkdir -p data output

# 2. Build and launch container
docker compose up --build
```

---

## ⚙️ Command Line Arguments (`app/main.py`)

| Argument | Short Flag | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input` | `-i` | `/kaggle/input/datasets/gogul0604/test-dataset` | Path to a single file (`.txt`/`.docx`/`.doc`/`.html`) or folder containing documents. |
| `--text` | `-t` | `None` | Inline text string to analyse directly instead of reading files. |
| `--output` | `-o` | `pii_ner_report.xlsx` | Output `.xlsx` file path (JSON reports generated automatically). |
| `--hf_token` | | `""` | Optional HuggingFace Access Token for gated models (such as IndicNER). |

### Example CLI Usage:

**Folder / File Processing:**
```bash
python app/main.py --input data/sample_complaint.txt --output output/sample_report.xlsx
```

**Direct Text Processing:**
```bash
python app/main.py --text "Application by Dr. S. K. Sharma, PAN ABCDE1234F, Phone +91 9876543210" --output output/inline_report.xlsx
```

---

## 📊 Generated Reports Schema

Upon execution, the pipeline outputs:

1. **Excel Workbook (`.xlsx`)**:
   - `Summary`: Processed file summary, script language, PII hit counts, and detection status.
   - `PII Detection`: Line-level PII audit with `Word No`, `Word Position`, `Start Letter`, `End Letter`, `Letter Span`, detector source (Regex vs Presidio), display names, raw values, and XML tags.
   - `NER Comparison`: Comparative side-by-side entity extraction across 5 models (`HiNER`, `IndicNER`, `BERT_Base_NER`, `XLM_RoBERTa`, `Hybrid`) and missed entity terms.
   - `Anonymization`: Complete audit record of original text vs anonymized text, position offsets, confidence scores, redaction technique, and description.

2. **JSON Reports (`.json`)**:
   - Master JSON report (`pii_ner_report.json`) and individual model JSON reports (`pii_ner_report_hiner.json`, `pii_ner_report_hybrid.json`, etc.) containing metadata, line numbers, word positions, character offsets, confidence scores, and entity predictions.

---

## 🔒 Security & Privacy

All processing is executed **100% locally** on your machine or container. No document content, personal identifiers, or metadata are transmitted to external services.

## Config-driven dataset batch mode

The `dataset-batch-anonymization` branch also supports a single-shot dataset job. It reads CSV, JSON, XLS, or XLSX input, anonymizes only configured text columns, and writes a schema-preserving staged dataset for the downstream SKALD process.

Example `config/config.json`:

```json
{
  "data_type": "my_dataset",
  "my_dataset": {
    "source_input_path": "data/raw_input.csv",
    "free_text_anonymization": {
      "enabled": true,
      "columns": ["Narrative", "Remarks"],
      "staged_input_path": "output/sanitized_input.csv",
      "minimum_confidence": 0.75,
      "on_failure": "fail",
      "audit_output_path": "output/anonymization_audit.json"
    }
  }
}
```

Run locally:

```bash
python app/main.py --config config/config.json
```

The batch command exits after writing `output/sanitized_input.csv`. The optional audit contains only row/column references, labels, confidence, and offsets; it never writes detected values.

Batch anonymization policies are hardcoded to match the legacy behavior, with stricter suppression for person names and phone numbers: Aadhaar/PAN/bank account/pincode use partial masking, email preserves its domain, cards retain only their final four digits in a tokenized form, dates become `XX-XX-YYYY`, ages are generalized to decade ranges, person/location/organization/phone values become `*`, and unknown values are one-way hashed.

Run with Docker:

```bash
mkdir -p data work output
docker compose up --build
```

Compose mounts `config/`, `data/`, `work/`, and `output/` into the container. With `on_failure: "fail"`, a missing column, unreadable file, or processing error causes a non-zero exit and no completed staged output is published. `continue` records metadata about cell failures and completes the batch. PII policies are hardcoded in the application, not supplied through config.
