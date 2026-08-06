# 🛡️ Grievance Data PII Anonymization & NER Evaluation Pipeline

A production-ready, unified processing pipeline for **text-based grievance analysis, language identification, multi-category PII detection, automated contextual redaction, and multi-model Named Entity Recognition (NER) benchmarking**.

Built to process administrative, medical, and public grievance text documents in **English, Indic scripts (Hindi/Devanagari, Bengali, Tamil, Telugu), and Mixed languages**.

---

## 🌟 Key Features

1. **Automatic Script & Language Detection**:
   - Classifies document languages into **Hindi (Devanagari)**, **Bengali**, **Tamil**, **Telugu**, **English**, **Mixed**, or **Unknown**.

2. **20+ Category Contextual & Pattern PII Engine**:
   - **Government & National IDs**: Aadhaar Number, PAN Card, Voter ID, Passport Number, Driving License, Parivar Pehchan Patra (PPP / Family ID), User ID / Portal ID.
   - **Financial Particulars**: Bank Account Number (with labeled context detection), Credit / Debit Card Number, IFSC Code.
   - **Contact Information**: Phone Number, Email Address, IP Address.
   - **Administrative & Location**: Haryana & Indian Cities / Districts (e.g., Karnal, Jhajjar, Rohtak, Gurugram, Delhi, etc.), Pin Codes.
   - **Personal Particulars**: Honorific Person Names (e.g., *Shri Ramesh Kumar*, *Dr. Anjali Verma*), Date of Birth (DOB).

3. **Privacy-Preserving Anonymization Strategies**:
   - **Partial Masking**: Aadhaar (`XXXX XXXX 1234`), Phone (`XXXXXX9876`), Bank Account (`********5678`), PAN (`ABCDE****F`).
   - **Domain-Preserving Masking**: Email (`jo****@domain.com`).
   - **Tokenization**: Credit Card (`XXXX-XXXX-XXXX-4321`).
   - **Initial-Only Masking**: Person Names (`R. K. S.`).
   - **One-Way Hashing**: SHA-256 hash for generic tokens.

4. **Multi-Model NER Benchmark**:
   Compares entity extraction across 5 model configurations:
   - **HiNER**: `cfilt/HiNER-original-muril-base-cased` (IIT Bombay / MuRIL)
   - **IndicNER**: `ai4bharat/IndicNER` (AI4Bharat Multilingual)
   - **BERT-Base-NER**: `dslim/bert-base-NER` (English)
   - **XLM-RoBERTa**: `Babelscape/wikineural-multilingual-ner` (Multilingual)
   - **Hybrid**: Combined HiNER + IndicNER ensemble.

5. **Automated 4-Sheet Excel Audit Report**:
   - **Sheet 1: Summary**: File-level overview, language breakdown, total PII hits, and detection status.
   - **Sheet 2: PII Detection**: Detailed audit log of every detected PII entity, source detector, display name, and XML-style tags (`<Aadhaar>...</Aadhaar>`).
   - **Sheet 3: NER Comparison**: Model-by-model comparison of predicted entity types (`PERSON`, `LOCATION`, `ORGANIZATION`), extracted entity text, and strict missed entity computation.
   - **Sheet 4: Anonymization**: Full record of original values vs anonymized output, masking method used, and description.

---

## 📂 Project Structure

```text
Grievance-data-anonymization/
├── app/
│   ├── main.py             # Main execution workflow (Language -> PII -> NER -> Excel Export)
│   └── kaggle_setup.py     # Automated environment & dependency setup script
├── data/                   # Input folder for text document datasets (.txt)
├── output/                 # Output folder for generated Excel reports
├── Dockerfile              # Container definition for reproducible deployment
├── docker-compose.yml      # Docker Compose configuration for volume mounting
├── requirements.txt        # Python dependency specifications
└── README.md               # Comprehensive documentation
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

# 3. Option A: Run automated setup script
python app/kaggle_setup.py

# OR Option B: Manual pip installation
pip install -r requirements.txt
python -m spacy download en_core_web_lg

# 4. Place your input .txt files in data/ directory
mkdir -p data output

# 5. Run the main processing pipeline
python app/main.py --input data --output output/pii_ner_report.xlsx
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

# 2. Build and launch
docker compose up --build
```

---

## ⚙️ Command Line Arguments (`app/main.py`)

| Argument | Short Flag | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input` | `-i` | `/kaggle/input/datasets/gogul0604/text-dataset` | Path to single `.txt` file or input directory containing `.txt` files. |
| `--output` | `-o` | `pii_ner_report.xlsx` | Output Excel file path for generating the 4-sheet report. |
| `--hf_token` | | `""` | Optional HuggingFace Access Token for gated models. |

### Example Custom Execution:
```bash
python app/main.py --input /path/to/my_texts --output output/custom_report.xlsx
```

---

## 📊 Excel Report Schema

Upon completion, the pipeline outputs a color-formatted Excel workbook containing:

1. **`Summary`**: File name, script language, total PII entity count, detected PII types, and hybrid NER person names.
2. **`PII Detection`**: Full text, PII category, display name, detected value, detector source, and tagged XML output.
3. **`NER Comparison`**: Side-by-side entity predictions (`HiNER`, `IndicNER`, `BERT_Base_NER`, `XLM_RoBERTa`, `Hybrid`) and missed entity terms.
4. **`Anonymization`**: Original text vs anonymized text, masking technique (`Partial Masking`, `Domain-Preserving`, `Tokenization`, `Initial-Only`), and technique explanation.

---

## 🔒 Privacy & Security

All processing is executed **100% locally**. No document content, personal identifiers, or metadata are transmitted to external services.
