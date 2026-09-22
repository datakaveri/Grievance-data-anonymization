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

## 🗃️ Dataset Batch Job (`app/batch_pipeline.py`)

The batch job anonymizes configured columns of a CSV/XLSX/JSON dataset. It reads its
settings from the `free_text_anonymization` object in the dataset config:

| Config key | Default | Description |
| :--- | :---: | :--- |
| `enabled` | — | Must be `true` for the job to write a staged file. |
| `columns` | — | List of column names to anonymize. All must exist in the input. |
| `minimum_confidence` | `0.0` | Detections below this confidence are ignored. |
| `ner_batch_size` | `256` | Lines handed to each inference call. Affects progress reporting granularity, not throughput. |
| `staged_input_path` | — | Where the anonymized CSV is written (must end in `.csv`). |
| `audit_output_path` | — | Optional JSON audit of every applied detection. |
| `on_failure` | `"fail"` | `"fail"` aborts the run on a cell error; `"continue"` records it. |

### How it scales

NER runs **once over the whole job**, not once per cell. The job collects every
distinct line across the configured columns, infers each one exactly once in real
mini-batches, then projects the spans back onto every cell containing that line.
On a real PHED export roughly four in five complaint bodies are duplicates, so the
deduplication alone removes most of the work; the batching removes most of the rest.

Models are loaded one at a time and released after their pass, so peak memory is
roughly one model (~1 GB) rather than three.

### Debug instrumentation

Set `debug.enabled` in the `free_text_anonymization` block (or `ANON_DEBUG=1`) to
write two extra reports. Both are off by default and cost nothing when off — the
normal run is byte-identical with them disabled.

| Debug key | Default | Description |
| :--- | :---: | :--- |
| `enabled` | `false` | Turns on step profiling and per-row attribution. |
| `include_values` | `false` | Writes the detected PII text into the attribution report. **Off by default on purpose:** it puts the unredacted values the run just masked into a plaintext file beside the anonymized output. |
| `max_rows` | `0` (all) | Cap the rows listed in the attribution report. |
| `profile_output_path` | `output/debug_profile.json` | Per-step timing and peak RSS. |
| `attribution_output_path` | `output/debug_detections.json` | Per-row detections, machine-readable. |
| `attribution_text_path` | `output/debug_detections.txt` | Per-row detections, human-readable. |

**Profile report** — each model's load, inference and release measured separately,
with peak RSS sampled during the step rather than read at its edges (a model
allocates and frees inside a step, so boundary readings miss the spike that
decides whether the job fits in memory):

```
  [profile] load:    HiNER (IIT Bombay / MuRIL)     2.74s  peak  905.6 MB  delta  +87.6 MB
  [profile] infer:   HiNER (IIT Bombay / MuRIL)     5.60s  peak 1317.9 MB  delta +398.0 MB
  [profile] release: HiNER (IIT Bombay / MuRIL)     0.27s  peak 1291.9 MB  delta -389.6 MB
```

The JSON adds `ms_per_line` per model, so the three are directly comparable.

**Attribution report** — what was detected in each row, and which detector found
it. A detection claimed by two models is listed under both:

```
Row 2
    [complaint_details] Aadhaar  <- Regex  (conf 1.00, partial_mask)
    [complainant_address] LOCATION  <- XLM-RoBERTa  (conf 0.84, suppress)
    [complainant_name] PERSON  <- XLM-RoBERTa, HiNER  (conf 0.93, suppress)
```

Attribution is recovered by overlap: the merge step pools every model's spans, so
a merged entity no longer records its origin, and a model whose raw span covers
the same characters is taken to have found it.

### Environment variables

| Variable | Default | Description |
| :--- | :---: | :--- |
| `HF_TOKEN` | `""` | Required for the gated `ai4bharat/IndicNER` repo. Without it that model is skipped (the run still completes with the other two, so check the logs if you expect all three). |
| `NER_CORPUS_BATCH_SIZE` | `256` | Lines per inference call — progress granularity only, overridden by `ner_batch_size` in config. |
| `NER_TENSOR_BATCH_SIZE` | `0` (auto) | Tensor batch size. Sequences pad to the batch's longest, so bigger is not better and the optimum tracks the core count — auto picks `max(8, 4 x threads)`. Measured on the PHED corpus at 2 threads: 43 ms/line at 8, 52 at 16, 63 unbatched, 64 at 32. Pin an explicit value to override, or `1` to disable batching. |
| `NER_TORCH_THREADS` | `0` (leave as-is, i.e. 4) | Torch CPU threads for the batch job. Tune per machine — on a hybrid-core laptop CPU, raising it measured *slower*, so benchmark before changing it. |
| `NER_QUANTIZE` | unset | Dynamic INT8 quantization. **Not recommended for production anonymization.** Measured on the PHED corpus it is ~1.5x faster (90 → 59 ms/line) but changes what is detected: 386 entities found against fp32's 481, only 290 in common. Missing a fifth of the names and locations is a privacy failure, not a tuning trade-off. |

---

## 📊 Performance & Timing Benchmarks

| Execution Stage | Original Baseline | Optimized Parallel Pipeline | Net Speedup |
| :--- | :---: | :---: | :---: |
| **Model Downloads** *(Uncached)* | ~12.2 mins | ~1.26 mins | **9.6x Faster Download** |
| **Model Weight Loading** | 10.15s | 3.74s | **2.7x Faster Loading** |
| **NER Model Execution** | 23.53s | 14.70s | **1.6x Faster Inference** |
| **Total Completion Time (Cached)** | 34.62s | **15.81s** | **2.19x Faster Total Pipeline** |

### Dataset batch job

Measured on a PHED grievance export (`complaint_details`, `complainant_address`,
`complainant_name`), 2 of 3 models loaded, CPU only, identical detection counts
before and after (11,399):

| Rows | Cells | Distinct lines | Per-cell NER (old) | Corpus NER (new) |
| ---: | ---: | ---: | ---: | ---: |
| 6,000 | 17,913 | 7,833 (2.3x) | 2032 s | **1035 s** |

The gain comes from deduplication, so it grows with the dataset: at 6,000 rows
duplicates collapse the work 2.3x, but across the full 181,430-row export
544,193 cells reduce to 172,804 distinct lines — 3.15x. Extrapolating the
measured 132 ms per distinct line, the full export runs in roughly 6 hours
against about 17 for the per-cell path, on the same hardware.

Inference itself is the floor: three (or two) transformer passes over ~173k
distinct lines on CPU. Measured at 2 threads, one model costs ~43 ms per distinct
line, so all three over the full export land near 10 hours on a 2-vCPU host.

Past that the lever is capacity, not tuning. Line-level inference is embarrassingly
parallel and the three models are independent, so vCPUs convert almost directly
into wall time; batching and thread tweaks are worth well under 2x by comparison.
The GPU path in `main.py` is commented out and would need reinstating to use one.

---

## 🔒 Security & Privacy

All processing is executed **100% locally** on your machine or container. No document content, personal identifiers, or metadata are transmitted to external services.
