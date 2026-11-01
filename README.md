# Grievance Data PII Anonymization

Removes personal data from free-text grievance records written in English, Hindi
(Devanagari) and romanised Hindi, before the data is handed to SKALD. Two kinds of
detector work together:

* **Rules (regex)** for structured identifiers: Aadhaar, PAN, phone numbers (+91),
  Haryana family IDs (PPP), FIR / case / land-record numbers, dates, e-mail, bank
  accounts, and a catch-all for ID-shaped codes.
* **NER models** for names, places and organisations: `ai4bharat/IndicNER` and
  `Davlan/xlm-roberta-base-ner-hrl`, merged per line, with filters that keep
  everyday words ("ATR", "complainant", "Sir") from being hidden as names.

Everything runs locally on CPU. No text is sent to any external service.

---

## How the code is organised

The code has three layers. Entry scripts only parse the command line. A **job**
handles one input shape: it reads the input into `(column, text)` pairs, hands
them to the **pipeline**, and writes each result back where it came from. The
pipeline (`anonymizer/pipeline.py`) is the one place where the flow is written
out, and each of its four steps is one module. Region-specific knowledge is kept
in separate rule files.

```text
app/
├── batch_pipeline.py        Entry: anonymize table columns (Docker ENTRYPOINT) -> jobs/table.py
├── json_remarks.py          Entry: anonymize "remarks" fields of nested JSON  -> jobs/nested_json.py
├── ner_report.py            Entry: NER comparison report (Excel + JSON), redacts nothing
├── model_setup.py           Entry: download the NER models into the HF cache (Docker build)
│
├── anonymizer/
│   ├── jobs/                JOBS — one per input shape
│   │   ├── table.py            table columns -> staged CSV for SKALD + audit JSON
│   │   └── nested_json.py      *remark* fields of nested JSON -> same JSON + report
│   ├── config.py            find / validate the dataset config (DetectionSettings, TableJobSettings)
│   │
│   ├── pipeline.py          PIPELINE — the flow, steps 1-4, once per distinct value
│   ├── cleanup.py           1. clean source text (HTML entities, broken legacy-font Hindi)
│   ├── rules/               2. WHAT the regex detectors look for
│   │   ├── generic.py          e-mail, card, IP, numeric dates, user/patient IDs, age, deny list
│   │   ├── india.py            Aadhaar, PAN, DL, vehicle, voter ID, passport, bank, IFSC, UPI,
│   │   │                       GSTIN, ABHA, phones, Hindi dates, reference numbers, catch-all IDs,
│   │   │                       PIN codes, Hindi word lists
│   │   ├── haryana.py          PPP family ID, Haryana record words, Haryana NER exemptions
│   │   └── __init__.py         the order the rules run in (it decides ties)
│   ├── scanner.py           2. HOW: run the rules over each line
│   ├── ner/                 3. model detection
│   │   ├── models.py           model registry, loading, runtime knobs
│   │   ├── inference.py        batching, long-line chunking, whole-corpus passes
│   │   ├── spans.py            whole-word snapping, cleanup, merging both models
│   │   ├── filters.py          clear titles, grammar words, everyday vocabulary, allow-lists
│   │   └── index.py            infer each distinct line once, map entities back onto cells
│   ├── redaction.py         4. pool rule + model detections, apply the per-type policy
│   ├── text.py                 shared helpers (word positions, Indic word characters, scripts)
│   └── debug/                  profiling.py (step timing, peak memory), reports.py (debug reports)
│
└── report/                  The NER comparison report (reuses steps 2 and 3)
    ├── readers.py           TXT / DOCX / HTML / JSON / CSV -> text
    ├── parallel_ner.py      one worker thread per model
    ├── writers.py           4-sheet Excel workbook + per-model JSON
    ├── records.py           report data model
    └── cli.py               command line

config/grievance/            Haryana grievance dataset configs (one per table) + order.txt
tests/                       unittest suite: every rule's documented examples, and the pipeline steps
```

Every regex rule carries a comment with what it matches, an example it catches,
what it deliberately leaves alone, and why it exists. `#NN` in a comment refers to
the issue number in the reviewer's sheet (`data/ner_issues/ner_issues.xlsx`) that
motivated it.

### What happens to a cell

1. **Cleanup** (`cleanup.py`) — `&#039;` becomes `'`; invalid Devanagari sign
   sequences from legacy-font conversion are repaired (`शिक्ष़्ाा` → `शिक्षा`).
2. **Rules** (`scanner.py`, `rules/`) — every rule runs on every line, in a fixed
   order. Devanagari digits are read as ASCII digits.
3. **Models** (`ner/`) — each *distinct* line of the dataset is run through both
   models once. Spans are grown to whole words (Hindi vowel signs included), cleaned,
   merged across models, and filtered for false positives.
4. **Redaction** (`redaction.py`) — rule and model detections are pooled; overlapping
   ones become one replacement. Its policy comes from a detection that covers the whole
   span (rules before models, specific rules before broad ones); if none does, the span
   becomes `*`, so a partial mask is never applied to text it was not made for.

| Detected type | Replacement |
| :--- | :--- |
| Person, location, organisation, phone | `*` |
| Aadhaar | `XXXX XXXX 0124` |
| PAN | `ABCDE****F` |
| Bank account | `******0000` |
| Card | `XXXX-XXXX-XXXX-1111` |
| E-mail | `ra****@gmail.com` |
| Date | `XX-XX-2025` (year kept) |
| PIN code | `132XXX` |
| Age | kept |
| Everything else (PPP / Voter ID / reference and ID numbers / IFSC / UPI …) | SHA-256 of a per-column random salt + value |

---

## Quick start

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=...                # needed once to download the gated ai4bharat/IndicNER
python app/model_setup.py          # caches both models

# Anonymize a table (see "Table job" for the config)
python app/batch_pipeline.py --config config/config.json

# Remarks fields of nested JSON records
python app/json_remarks.py data/ --out output/json_remarks \
    --config config/grievance/jan_samvaad_reminders.json

# Compare what each model finds (Excel + JSON report)
python app/ner_report.py --input path/to/documents/ --output output/report.xlsx
python app/ner_report.py --text "Shri Ramesh Kumar, Aadhaar 2345 6789 0123" --output output/inline.xlsx

# Tests (no models needed, a few seconds)
python -m unittest discover -s tests
```

### Docker

```bash
DOCKER_BUILDKIT=1 docker build --secret id=hf_token,env=HF_TOKEN -t grievance-anonymizer .
docker run --rm -v $(pwd)/config:/app/config:ro -v $(pwd)/data:/app/data \
    -v $(pwd)/output:/app/output grievance-anonymizer
# or: docker compose up --build
```

The container runs the table job on the config it finds in `/app/config`.

---

## Table job (`app/batch_pipeline.py` → `anonymizer/jobs/table.py`)

Anonymizes configured columns of a CSV / XLSX / JSON table. Settings live in the
`free_text_anonymization` object of the dataset config:

| Config key | Default | Description |
| :--- | :---: | :--- |
| `enabled` | — | Must be `true` for the job to write a staged file. |
| `columns` | — | Columns to anonymize. All must exist in the input. |
| `staged_input_path` | — | Where the anonymized CSV is written (must end in `.csv`). |
| `audit_output_path` | — | Optional JSON audit of every replacement. |
| `minimum_confidence` | `0.0` | Model detections below this are ignored (the grievance configs use `0.75`). |
| `on_failure` | `"fail"` | `"fail"` aborts on a cell error; `"continue"` records it. |
| `ner_batch_size` | `256` | Lines per inference call — progress granularity only. |
| `normalize_source_text` | `true` | Run the cleanup step. The staged CSV carries the cleaned text. |
| `reference_keywords` | built-in | Regexes for record labels that are also ordinary words ("Case No. 45", "मकान नं. 1919"): a "No." marker or a number/year shape (`45/2025`) must follow. The number, and further numbers of a list ("No. 140 & 141"), are hashed. Replaces the default list (India + Haryana, see `rules/`); `[]` turns it off. |
| `strong_reference_keywords` | built-in | Labels always followed by their number (FIR, मुकदमा, अभियोग, खसरा, ID…): the marker is optional ("FIR 140"). |
| `ner_allow_terms` | `[]` | Local terms the models tag that are never personal ("ATR", "Jan Samvaad"). Matched against the whole entity, ignoring case, spacing and common spelling variants. Build it from the debug `debug_ner_terms.json` report — but never add officials' names, which also rank high there. |
| `ner_deny_terms` | `[]` | Names a reviewer confirmed the models miss; always redacted as PERSON wherever they appear as whole words. Real names belong here, in the config next to the data, never in code. |

`source_input_path` (in the dataset object) points at the input; without it the job
uses the single data file in `data/`, as SKALD does.

> **Hardcoded exemptions (temporary).** `NER_ALLOW_TERMS` in
> `app/anonymizer/rules/haryana.py` exempts "Milk", "Mark" and "kindly undetake" in
> every dataset. The models tag "Milk" as a place ("the said Milk Booth") and the verb
> "Mark" as a name ("Complaint is Mark to CIA-3"); the general filters deliberately
> leave both alone because they never clear locations ("Model Town" is plain English)
> or capitalised names ("Lucky", "Sunny"). "kindly undetake" carries a typo no
> dictionary knows. The cost: a person or place called just "Mark" or "Milk" is not
> redacted. Remove an entry once a general rule covers it.

### How it scales

NER runs **once over the whole job**, not once per cell: every distinct line across
the configured columns is inferred exactly once in real mini-batches and projected
back onto every cell containing it. On a real grievance export roughly four in five
complaint bodies are duplicates, so deduplication removes most of the work. Models
load one at a time and are released after their pass, so peak memory is about one
model (~1 GB).

### Debug instrumentation

Set `debug.enabled` in the `free_text_anonymization` block (or `ANON_DEBUG=1`).
Off by default and free when off.

| Debug key | Default | Description |
| :--- | :---: | :--- |
| `enabled` | `false` | Step profiling, per-row attribution and the NER-term report. |
| `include_values` | `true` | Write the detected PII text into the reports. **This writes unredacted personal data to disk** — set it to `false`, or leave debugging off, anywhere that matters, such as in the TEE. |
| `max_rows` | `0` (all) | Cap the rows listed in the attribution report. |
| `profile_output_path` | `<audit>.debug_profile.json` | Per-step and per-model time and peak RSS. |
| `attribution_output_path` | `<audit>.debug_detections.json` | Per-row detections, machine-readable. |
| `attribution_text_path` | `<audit>.debug_detections.txt` | Per-row detections, human-readable. |
| `ner_terms_path` | `<audit>.debug_ner_terms.json` | Most-redacted model strings with their allow-list key (needs `include_values`). |

`<audit>` is `audit_output_path` without its extension. Example attribution lines:

```
Row 2
    [remarks] PERSON 'Amit Sharma'  <- XLM-RoBERTa, IndicNER  (conf 1.00, suppress)
    [remarks] Phone_Number '98765 43210'  <- Regex  (conf 1.00, suppress)
```

A model hit names the model(s) whose raw span overlaps the merged entity.

### Environment variables

| Variable | Default | Description |
| :--- | :---: | :--- |
| `HF_TOKEN` | `""` | Needed to download the gated `ai4bharat/IndicNER`. Without it the model is skipped and the job runs with XLM-RoBERTa alone — check the log. |
| `NER_CORPUS_BATCH_SIZE` | `256` | Lines per inference call (progress granularity); `ner_batch_size` overrides it. |
| `NER_TENSOR_BATCH_SIZE` | `0` (auto) | Sequences per forward pass. They pad to the batch's longest, so bigger is not better; auto picks `max(8, 4 × threads)`. Measured at 2 threads: 43 ms/line at 8, 52 at 16, 63 unbatched, 64 at 32. `1` disables batching. |
| `NER_TORCH_THREADS` | `0` (keep 4) | Torch CPU threads. Benchmark before raising it — on a hybrid-core laptop CPU more threads measured slower. |
| `NER_QUANTIZE` | unset | Dynamic INT8 quantization. **Not for production anonymization**: ~1.5× faster but it found 386 entities where fp32 found 481 (only 290 shared) — a fifth of names and places unmasked. |
| `ANON_DEBUG` | unset | `1` turns on debug instrumentation without editing the config. |

### Performance

Measured on a reference grievance export (6,000 rows, 17,913 cells, 7,833 distinct
lines), CPU only: per-cell NER took 2,032 s, corpus NER **1,035 s**, with identical
detections. The gain grows with the data: on the full 181,430-row export, 544,193
cells reduce to 172,804 distinct lines (3.15×). Inference is the floor — about 43 ms
per distinct line per model at 2 threads — and line-level inference parallelises
almost perfectly, so more vCPUs convert directly into wall time.

---

## NER comparison report (`app/ner_report.py`)

An evaluation tool: it runs the rules and each model separately and writes an Excel
workbook (Summary, PII Detection, NER Comparison, Anonymization sheets) plus
`<output>.json`, `<output>_IndicNER.json`, `<output>_XLM_RoBERTa.json` and
`<output>_Hybrid.json`. It redacts nothing.

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--input`, `-i` | — | A `.txt` / `.docx` / `.doc` / `.html` / `.json` / `.csv` file, or a folder of them. |
| `--text`, `-t` | — | Inline text instead of files (`\n` escapes honoured). One of `--input` / `--text` is required. |
| `--output`, `-o` | `pii_ner_report.xlsx` | Workbook path; the JSON files are written next to it. |
| `--ner-batch-size` | `24` | Lines per worker mini-batch. |
| `--queue-size` | `64` | Bound on each worker queue. |
| `--hf_token` | `$HF_TOKEN` | HuggingFace token. |
