"""Command line of the NER report: python app/ner_report.py --input <path> | --text "<string>"."""

from __future__ import annotations

import argparse
import os
from typing import List

from anonymizer.ner import models
from anonymizer.text import detect_language

from .parallel_ner import NER_LINE_BATCH_SIZE, NER_QUEUE_MAXSIZE, analyze_records
from .readers import collect_files, extract_text
from .records import FileRecord
from .writers import build_excel, build_json

DEFAULT_OUTPUT_PATH = "pii_ner_report.xlsx"


def parse_args() -> argparse.Namespace:
    """--input or --text (one is required), --output, batching knobs, optional HF token."""
    ap = argparse.ArgumentParser(
        description=(
            "Multi-format (TXT/DOCX/HTML/JSON/CSV) -> rule + per-model NER detection -> Excel & JSON reports\n\n"
            "Input modes:\n"
            "  File / folder:  --input path/to/file_or_folder\n"
            '  Inline string:  --text "Your text enclosed in quotes"\n'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", "-i", default=None,
                    help="A file (.txt/.docx/.doc/.html/.json/.csv) or a folder of such files.")
    ap.add_argument("--text", "-t", default=None, metavar="STRING",
                    help="Inline text to analyse instead of files (\\n and \\t escapes are honoured).")
    ap.add_argument("--output", "-o", default=DEFAULT_OUTPUT_PATH,
                    help="Output .xlsx path; the JSON reports are written next to it.")
    ap.add_argument("--hf_token", default=os.getenv("HF_TOKEN", ""),
                    help="HuggingFace token (needed to download the gated IndicNER model).")
    ap.add_argument("--ner-batch-size", type=int, default=NER_LINE_BATCH_SIZE,
                    help="Lines per worker mini-batch (16-32 recommended, default 24).")
    ap.add_argument("--queue-size", type=int, default=NER_QUEUE_MAXSIZE,
                    help="Bound on each worker queue, for backpressure (default 64).")
    args = ap.parse_args()
    if args.input is None and args.text is None:
        ap.error("give --input <file or folder> or --text <string>")
    return args


def _records_from_text(text: str) -> List[FileRecord]:
    """One FileRecord for an inline string; literal "\\n"/"\\t" become real breaks."""
    text = text.replace("\\n", "\n").replace("\\t", "\t").strip("'\"")
    if not text.strip():
        print("[WARN] --text input is empty; nothing to process.", flush=True)
        return []
    return [FileRecord(path="<inline>", filename="inline_input", file_type="string",
                       language=detect_language(text), raw_text=text)]


def _records_from_files(input_path: str) -> List[FileRecord]:
    """One FileRecord per readable, non-empty supported file under `input_path`."""
    records: List[FileRecord] = []
    for path in collect_files(input_path):
        try:
            raw, file_type = extract_text(path)
        except Exception as exc:
            print(f"  [WARN] Could not read {path}: {exc}", flush=True)
            continue
        if raw.strip():
            records.append(FileRecord(path=path, filename=os.path.basename(path), file_type=file_type,
                                      language=detect_language(raw), raw_text=raw))
    return records


def main() -> int:
    """Run the report; exit code 1 when no readable document was found."""
    args = parse_args()
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
    status = f"✓ Available on {models.DEVICE_NAME}" if models.TRANSFORMERS_AVAILABLE else "✗ Not installed"
    print(f"[Pipeline] NER Transformers: {status}", flush=True)

    if args.text is not None:
        if args.input is not None:
            print("[WARN] Both --text and --input were supplied; --text takes priority.", flush=True)
        records = _records_from_text(args.text)
        if not records:
            return 0
    else:
        records = _records_from_files(args.input)
        if not records:
            print("[ERROR] No readable documents found after extraction.", flush=True)
            return 1
        print(f"[Pipeline] Documents: {len(records)} | Parallel NER workers + overlapping PII scan", flush=True)

    analyze_records(records, batch_size=max(1, args.ner_batch_size), queue_size=max(8, args.queue_size))
    build_excel(records, args.output)
    build_json(records, args.output)
    return 0
