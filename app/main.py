#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PIPELINE.PY — Multi-Format PII Detection & NER Comparison Pipeline          ║
# ║  Supports: .txt, .doc, .docx, .html files (single or folder)                 ║
# ║  PII Detection: Regex + Presidio (Structured PII only)                       ║
# ║  NER: 4 HuggingFace models + Hybrid (HiNER + IndicNER + XLM-RoBERTa)        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import openpyxl
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS — POSITION & WORD CALCULATIONS
# ─────────────────────────────────────────────────────────────────────────────


def ordinal(n: int) -> str:
    """Converts a 1-based word index into an ordinal representation (e.g. 1 -> '1st word', 5 -> '5th word')."""
    if n <= 0:
        return "N/A"
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix} word"


def get_word_number(text: str, start_char_idx: int) -> int:
    """Calculates the 1-based word index in text for a given character start position."""
    return len(text[:start_char_idx].split()) + 1


def truecase_line(text: str) -> str:
    """
    Performs a 1-to-1 length-preserving title-casing transformation on words.
    Enables cased Transformer models to recognize entities in ALL-CAPS or all-lowercase text
    without altering character indices.
    """
    return re.sub(r"\b[A-Za-z]+\b", lambda m: m.group(0).capitalize(), text)


# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL IMPORTS — Presidio / spaCy
# ─────────────────────────────────────────────────────────────────────────────
try:
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    _PRESIDIO_AVAILABLE = True
except ImportError:
    _PRESIDIO_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL IMPORTS — Document reading libs
# ─────────────────────────────────────────────────────────────────────────────
try:
    import docx as _docx_lib

    _DOCX_AVAILABLE = True
except ImportError:
    _DOCX_AVAILABLE = False

try:
    from bs4 import BeautifulSoup

    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEFAULT_INPUT_PATH = "/home/gogul/Documents/Grievance-data-anonymization/app/data/sample_complaint.txt"
DEFAULT_OUTPUT_PATH = "pii_ner_report.xlsx"
DEFAULT_HF_TOKEN = os.getenv("HF_TOKEN", "").strip()



NER_MODELS: Dict[str, Tuple[str, str]] = {
    "HiNER": (
        "HiNER (IIT Bombay / MuRIL)",
        "cfilt/HiNER-original-muril-base-cased",
    ),
    "IndicNER": ("IndicNER (AI4Bharat / Public)", "ai4bharat/IndicNER"),
    "BERT_Base_NER": ("BERT-Base-NER (English)", "dslim/bert-base-NER"),
    "XLM_RoBERTa": (
        "XLM-RoBERTa (Multilingual)",
        "Babelscape/wikineural-multilingual-ner",
    ),
}
HYBRID_KEY = "Hybrid (HiNER + IndicNER + XLM-RoBERTa)"

PII_DISPLAY_NAMES: Dict[str, str] = {
    "Aadhaar": "Aadhaar Number",
    "PAN": "PAN Card",
    "Phone_Number": "Phone Number",
    "Email": "Email Address",
    "Medical_UHID": "Medical UHID",
    "Passport": "Passport Number",
    "Driving_License": "Driving License",
    "Voter_ID": "Voter ID",
    "Vehicle_Number": "Vehicle Number",
    "Bank_Account": "Bank Account",
    "Credit_Card": "Credit / Debit Card",
    "Date": "Date",
    "PPP_ID": "PPP/Family ID",
    "User_ID": "User ID / Portal ID",
    "IP_Address": "IP Address",
    "IFSC_Code": "IFSC Code",
    "Pincode": "ZIP / Pin Code",
    "Age": "Age",
}


@dataclass
class PiiHit:
    source: str
    label: str
    value: str
    display: str
    line_no: int = 0
    word_no: int = 0
    start_char: int = 0
    end_char: int = 0


@dataclass
class NerEntity:
    category: str
    text: str
    score: float
    word_no: int = 0
    start_char: int = 0
    end_char: int = 0


@dataclass
class LineNerResult:
    model: str
    line_no: int
    text: str
    entities: List[NerEntity] = field(default_factory=list)
    persons: List[str] = field(default_factory=list)
    locs: List[str] = field(default_factory=list)
    orgs: List[str] = field(default_factory=list)
    missed: List[str] = field(default_factory=list)

    @property
    def predicted_type(self) -> str:
        types = []
        if self.persons:
            types.append("PERSON")
        if self.locs:
            types.append("LOCATION")
        if self.orgs:
            types.append("ORGANIZATION")
        return " | ".join(types) if types else "NONE"

    @property
    def extracted_text(self) -> str:
        parts = []
        for cat_label, cat_code in [
            ("PER", "PERSON"),
            ("LOC", "LOCATION"),
            ("ORG", "ORGANIZATION"),
        ]:
            cat_ents = [e for e in self.entities if e.category == cat_code]
            if cat_ents:
                formatted = [
                    f"{e.text} (conf: {e.score:.2f}, {ordinal(e.word_no)}, letters {e.start_char}-{e.end_char})"
                    for e in cat_ents
                ]
                parts.append(f"[{cat_label}] " + " | ".join(formatted))
        return " || ".join(parts) if parts else "None"


@dataclass
class FileRecord:
    path: str
    filename: str
    file_type: str
    language: str
    raw_text: str
    pii_hits: List[PiiHit] = field(default_factory=list)
    line_ners: Dict[str, List[LineNerResult]] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-FORMAT TEXT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────


def read_txt(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read().replace("\r\n", "\n").replace("\r", "\n").strip()


def read_docx(path: str) -> str:
    if not _DOCX_AVAILABLE:
        print(f"  [WARN] python-docx not installed; skipping {path}", flush=True)
        return ""
    doc = _docx_lib.Document(path)
    lines = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            lines.append(text)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                cell_text = cell.text.strip()
                if cell_text:
                    lines.append(cell_text)
    return "\n".join(lines)


def read_html(path: str) -> str:
    if not _BS4_AVAILABLE:
        with open(path, encoding="utf-8", errors="replace") as fh:
            raw = fh.read()
        text = re.sub(r"<[^>]+>", " ", raw)
        text = re.sub(r"&nbsp;", " ", text)
        text = re.sub(r"&amp;", "&", text)
        text = re.sub(r"&lt;", "<", text)
        text = re.sub(r"&gt;", ">", text)
        return re.sub(r"\s+", " ", text).strip()
    with open(path, encoding="utf-8", errors="replace") as fh:
        soup = BeautifulSoup(fh.read(), "html.parser")
    for tag in soup(["script", "style", "head", "meta"]):
        tag.decompose()
    lines = [elem.strip() for elem in soup.find_all(text=True) if elem.strip()]
    return "\n".join(lines)


def extract_text(path: str) -> Tuple[str, str]:
    ext = Path(path).suffix.lower()
    if ext == ".txt":
        return read_txt(path), "txt"
    elif ext in (".docx", ".doc"):
        return read_docx(path), "docx"
    elif ext in (".html", ".htm"):
        return read_html(path), "html"
    else:
        try:
            return read_txt(path), "txt"
        except Exception:
            return "", "unknown"


def collect_files(input_path: str) -> List[str]:
    SUPPORTED = {".txt", ".docx", ".doc", ".html", ".htm"}
    p = Path(input_path)
    if p.is_file():
        if p.suffix.lower() in SUPPORTED:
            return [str(p)]
        sys.exit(
            f"[ERROR] Unsupported file type: {input_path}. Supported: {SUPPORTED}"
        )
    if p.is_dir():
        files = []
        for ext in SUPPORTED:
            files.extend(p.rglob(f"*{ext}"))
        files = sorted(files)
        if not files:
            sys.exit(f"[ERROR] No supported files found in: {input_path}")
        return [str(f) for f in files]
    sys.exit(f"[ERROR] Path not found: {input_path}")


# ─────────────────────────────────────────────────────────────────────────────
# LANGUAGE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
_INDIC_RANGES = {
    "Hindi (Devanagari)": re.compile(r"[\u0900-\u097F]"),
    "Bengali": re.compile(r"[\u0980-\u09FF]"),
    "Tamil": re.compile(r"[\u0B80-\u0BFF]"),
    "Telugu": re.compile(r"[\u0C00-\u0C7F]"),
}
_LATIN_RE = re.compile(r"[A-Za-z]")


def detect_language(text: str) -> str:
    if not text or not text.strip():
        return "Unknown"
    text_clean = text.strip()
    indic_counts = {
        k: len(v.findall(text_clean)) for k, v in _INDIC_RANGES.items()
    }
    latin_count = len(_LATIN_RE.findall(text_clean))
    top_script = (
        max(indic_counts, key=indic_counts.get) if indic_counts else None
    )
    if top_script and indic_counts[top_script] > 0:
        if (
            latin_count > 0
            and (
                indic_counts[top_script]
                / max(latin_count + indic_counts[top_script], 1)
            )
            < 0.6
        ):
            return f"{top_script} / Mixed"
        return top_script
    if latin_count > 0:
        return "English"
    return "Unknown"


# ─────────────────────────────────────────────────────────────────────────────
# PRESIDIO INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

_presidio_engine: Optional[object] = None


def _get_presidio_engine():
    global _presidio_engine
    if _presidio_engine is not None:
        return _presidio_engine
    if not _PRESIDIO_AVAILABLE:
        return None
    try:
        model_name = "en_core_web_lg"
        try:
            import spacy

            spacy.load(model_name)
        except Exception:
            model_name = "en_core_web_sm"

        configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": model_name}],
        }
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()
        _presidio_engine = AnalyzerEngine(nlp_engine=nlp_engine)
        return _presidio_engine
    except Exception as e:
        print(f"  [WARN] Could not initialize Presidio: {e}", flush=True)
        return None


_PRESIDIO_LABEL_MAP = {
    "EMAIL_ADDRESS": "Email",
    "PHONE_NUMBER": "Phone_Number",
    "CREDIT_CARD": "Credit_Card",
    "IBAN_CODE": "Bank_Account",
    "DATE_TIME": "Date",
    "IP_ADDRESS": "IP_Address",
    "MEDICAL_LICENSE": "Medical_UHID",
}


def _run_presidio_on_line(line: str, engine) -> List[PiiHit]:
    hits: List[PiiHit] = []
    if not line.strip():
        return hits
    try:
        results = engine.analyze(text=line, language="en")
        for r in results:
            internal_label = _PRESIDIO_LABEL_MAP.get(r.entity_type)
            if internal_label is None or r.score < 0.6:
                continue
            value = line[r.start : r.end].strip()

            if not value or re.match(r"^\d{1,4}$", value):
                continue

            if internal_label == "Date" and re.search(
                r"\b(?:year|years|aged?|old)\b", value, re.I
            ):
                continue

            word_no = get_word_number(line, r.start)
            start_char = r.start + 1
            end_char = r.end
            display = PII_DISPLAY_NAMES.get(internal_label, internal_label)

            hits.append(
                PiiHit(
                    source="Presidio",
                    label=internal_label,
                    value=value,
                    display=display,
                    word_no=word_no,
                    start_char=start_char,
                    end_char=end_char,
                )
            )
    except Exception:
        pass
    return hits


# ─────────────────────────────────────────────────────────────────────────────
# LINE-BY-LINE STRUCTURED PII SCANNER (Regex)
# ─────────────────────────────────────────────────────────────────────────────


def scan_text_line_by_line(text: str) -> List[PiiHit]:
    hits: List[PiiHit] = []
    lines = text.splitlines()

    for line_idx, line in enumerate(lines):
        line_str = line.strip()
        if not line_str:
            continue
        lno = line_idx + 1

        for m in re.finditer(
            r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", line_str
        ):
            hits.append(
                PiiHit(
                    "Regex",
                    "Email",
                    m.group(0).strip(),
                    PII_DISPLAY_NAMES["Email"],
                    lno,
                    get_word_number(line_str, m.start()),
                    m.start() + 1,
                    m.end(),
                )
            )

        for m in re.finditer(
            r"\b(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6011)[\s\-]?(?:\d{4}[\s\-]?){2}\d{4}\b",
            line_str,
        ):
            hits.append(
                PiiHit(
                    "Regex",
                    "Credit_Card",
                    m.group(0).strip(),
                    PII_DISPLAY_NAMES["Credit_Card"],
                    lno,
                    get_word_number(line_str, m.start()),
                    m.start() + 1,
                    m.end(),
                )
            )

        for m in re.finditer(
            r"\b[A-Z]{2}[\-\s]?\d{2,4}[\-\s]?\d{6,11}\b", line_str
        ):
            val = m.group(0).strip()
            if not re.match(
                r"^[A-Z]{5}\d{4}[A-Z]$", val
            ) and not re.match(r"^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$", val):
                hits.append(
                    PiiHit(
                        "Regex",
                        "Driving_License",
                        val,
                        PII_DISPLAY_NAMES["Driving_License"],
                        lno,
                        get_word_number(line_str, m.start()),
                        m.start() + 1,
                        m.end(),
                    )
                )

        for m in re.finditer(r"\b[A-Z]{5}\d{4}[A-Z]\b", line_str):
            hits.append(
                PiiHit(
                    "Regex",
                    "PAN",
                    m.group(0).strip(),
                    PII_DISPLAY_NAMES["PAN"],
                    lno,
                    get_word_number(line_str, m.start()),
                    m.start() + 1,
                    m.end(),
                )
            )

        for m in re.finditer(
            r"\b[A-Z]{2}[\s\-]?\d{2}[\s\-]?[A-Z]{1,2}[\s\-]?\d{4}\b", line_str
        ):
            hits.append(
                PiiHit(
                    "Regex",
                    "Vehicle_Number",
                    m.group(0).strip(),
                    PII_DISPLAY_NAMES["Vehicle_Number"],
                    lno,
                    get_word_number(line_str, m.start()),
                    m.start() + 1,
                    m.end(),
                )
            )

        for m in re.finditer(
            r"\b(?:[A-Z]{3}\d{7}|[A-Z]{2}\/\d{2}\/\d{3}\/\d{6})\b", line_str
        ):
            hits.append(
                PiiHit(
                    "Regex",
                    "Voter_ID",
                    m.group(0).strip(),
                    PII_DISPLAY_NAMES["Voter_ID"],
                    lno,
                    get_word_number(line_str, m.start()),
                    m.start() + 1,
                    m.end(),
                )
            )

        for m in re.finditer(r"\b[A-PR-WY][1-9]\d{7}\b", line_str):
            hits.append(
                PiiHit(
                    "Regex",
                    "Passport",
                    m.group(0).strip(),
                    PII_DISPLAY_NAMES["Passport"],
                    lno,
                    get_word_number(line_str, m.start()),
                    m.start() + 1,
                    m.end(),
                )
            )

        for m in re.finditer(
            r"(?:User\s*ID(?:\s*\([^)]+\))?|UserId|Username|User_ID|Portal\s*ID)[_\-:\s]+([a-zA-Z0-9_\-]+)",
            line_str,
            re.I,
        ):
            hits.append(
                PiiHit(
                    "Contextual Regex",
                    "User_ID",
                    m.group(1).strip(),
                    PII_DISPLAY_NAMES["User_ID"],
                    lno,
                    get_word_number(line_str, m.start(1)),
                    m.start(1) + 1,
                    m.end(1),
                )
            )

        for m in re.finditer(
            r"(?:PPP\s*ID|PPP|FAMILY\s*ID|Parivar\s*Pehchan\s*Patra)[_\-:\s\(\)]*([A-Z0-9]{6,10})",
            line_str,
            re.I,
        ):
            hits.append(
                PiiHit(
                    "Contextual Regex",
                    "PPP_ID",
                    m.group(1).strip(),
                    PII_DISPLAY_NAMES["PPP_ID"],
                    lno,
                    get_word_number(line_str, m.start(1)),
                    m.start(1) + 1,
                    m.end(1),
                )
            )

        for m in re.finditer(
            r"(?:Bank\s*Account(?:\s*Number|\s*No)?|SBI\s*Account|Account\s*No|Account\s*Number)[_\-:\s]*(\d{9,18})",
            line_str,
            re.I,
        ):
            hits.append(
                PiiHit(
                    "Contextual Regex",
                    "Bank_Account",
                    m.group(1).strip(),
                    PII_DISPLAY_NAMES["Bank_Account"],
                    lno,
                    get_word_number(line_str, m.start(1)),
                    m.start(1) + 1,
                    m.end(1),
                )
            )

        for m in re.finditer(
            r"(?<!\d[\s\-])\b[2-9]\d{3}[\s\-]\d{4}[\s\-]\d{4}\b(?!\d|[\s\-]\d)",
            line_str,
        ):
            hits.append(
                PiiHit(
                    "Regex",
                    "Aadhaar",
                    m.group(0).strip(),
                    PII_DISPLAY_NAMES["Aadhaar"],
                    lno,
                    get_word_number(line_str, m.start()),
                    m.start() + 1,
                    m.end(),
                )
            )

        for m in re.finditer(r"\b[A-Z]{4}0[A-Z0-9]{6}\b", line_str):
            hits.append(
                PiiHit(
                    "Regex",
                    "IFSC_Code",
                    m.group(0).strip(),
                    PII_DISPLAY_NAMES["IFSC_Code"],
                    lno,
                    get_word_number(line_str, m.start()),
                    m.start() + 1,
                    m.end(),
                )
            )

        for m in re.finditer(
            r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b", line_str
        ):
            hits.append(
                PiiHit(
                    "Regex",
                    "IP_Address",
                    m.group(0).strip(),
                    PII_DISPLAY_NAMES["IP_Address"],
                    lno,
                    get_word_number(line_str, m.start()),
                    m.start() + 1,
                    m.end(),
                )
            )

        for m in re.finditer(
            r"(?:Phone|Mobile|Contact|Cell|Tel|Primary\s*Contact)?[^\n\d]*\b(\+91[\s\-]?[6-9]\d{9}|[6-9]\d{9})\b",
            line_str,
            re.I,
        ):
            val = m.group(1).strip()
            if (
                "account" not in line_str.lower()
                and "pension" not in line_str.lower()
            ):
                hits.append(
                    PiiHit(
                        "Regex",
                        "Phone_Number",
                        val,
                        PII_DISPLAY_NAMES["Phone_Number"],
                        lno,
                        get_word_number(line_str, m.start(1)),
                        m.start(1) + 1,
                        m.end(1),
                    )
                )

        for m in re.finditer(
            r"\b(?:0?[1-9]|[12]\d|3[01])[\/\-.](?:0?[1-9]|1[0-2])[\/\-.](?:19|20)\d{2}\b",
            line_str,
        ):
            hits.append(
                PiiHit(
                    "Regex",
                    "Date",
                    m.group(0).strip(),
                    PII_DISPLAY_NAMES["Date"],
                    lno,
                    get_word_number(line_str, m.start()),
                    m.start() + 1,
                    m.end(),
                )
            )

        for m in re.finditer(
            r"\baged?\s+(?:about\s+)?(\d{1,3}\s*[Yy]ears?)\b", line_str
        ):
            val = m.group(1).strip()
            hits.append(
                PiiHit(
                    "Regex",
                    "Age",
                    val,
                    PII_DISPLAY_NAMES["Age"],
                    lno,
                    get_word_number(line_str, m.start(1)),
                    m.start(1) + 1,
                    m.end(1),
                )
            )

        for m in re.finditer(r"\b[1-9][0-9]{2}\s?[0-9]{3}\b", line_str):
            val = m.group(0).strip()
            if (
                not val.startswith("200")
                and not val.startswith("201")
                and "voter" not in line_str.lower()
            ):
                hits.append(
                    PiiHit(
                        "Regex",
                        "Pincode",
                        val,
                        PII_DISPLAY_NAMES["Pincode"],
                        lno,
                        get_word_number(line_str, m.start()),
                        m.start() + 1,
                        m.end(),
                    )
                )

    line_dedup: List[PiiHit] = []
    seen_in_line: set = set()
    for h in hits:
        key = (
            h.line_no,
            h.label.upper(),
            re.sub(r"[\s\-]", "", h.value).lower(),
        )
        if key not in seen_in_line and h.value.strip():
            seen_in_line.add(key)
            line_dedup.append(h)

    return line_dedup


def full_pii_scan(text: str) -> Tuple[List[PiiHit], List[PiiHit]]:
    regex_hits = scan_text_line_by_line(text)

    presidio_hits: List[PiiHit] = []
    engine = _get_presidio_engine()
    if engine:
        lines = text.splitlines()
        raw_presidio: List[PiiHit] = []
        for lno, line in enumerate(lines, 1):
            for h in _run_presidio_on_line(line.strip(), engine):
                h.line_no = lno
                raw_presidio.append(h)
        seen_p: set = set()
        for h in raw_presidio:
            key = (
                h.line_no,
                h.label.upper(),
                re.sub(r"[\s\-]", "", h.value).lower(),
            )
            if key not in seen_p and h.value.strip():
                seen_p.add(key)
                presidio_hits.append(h)

    return regex_hits, presidio_hits


def merge_pii_hits(
    regex_hits: List[PiiHit], presidio_hits: List[PiiHit]
) -> List[PiiHit]:
    merged = list(regex_hits)

    for p in presidio_hits:
        p_val_clean = re.sub(r"[\s\-]", "", p.value).lower()
        overlap_found = False

        for r in regex_hits:
            if r.line_no == p.line_no:
                r_val_clean = re.sub(r"[\s\-]", "", r.value).lower()
                if p_val_clean in r_val_clean or r_val_clean in p_val_clean:
                    overlap_found = True
                    break

        if not overlap_found:
            merged.append(p)

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# ANONYMIZATION LOGIC
# ─────────────────────────────────────────────────────────────────────────────


def anonymize_value(label: str, val: str) -> Tuple[str, str, str]:
    lbl = label.upper().replace(" ", "_")
    val = val.strip()

    if "AADHAAR" in lbl:
        d = re.sub(r"\D", "", val)
        anon = f"XXXX XXXX {d[-4:]}" if len(d) == 12 else "[Aadhaar Redacted]"
        return (
            "Partial Masking",
            anon,
            "First 8 digits masked, last 4 visible",
        )
    elif "PAN" in lbl:
        c = re.sub(r"\s+", "", val)
        anon = f"{c[:5]}****{c[-1]}" if len(c) == 10 else "XXXXX****X"
        return ("Partial Masking", anon, "Middle 4 digits masked")
    elif "PHONE" in lbl:
        d = re.sub(r"\D", "", val)
        anon = f"XXXXXX{d[-4:]}" if len(d) >= 10 else "XXXXXXXXXX"
        return (
            "Partial Masking",
            anon,
            "First 6 digits masked, last 4 visible",
        )
    elif "EMAIL" in lbl:
        if "@" in val:
            local, domain = val.split("@", 1)
            mk = (
                local[:2] + "*" * max(1, len(local) - 2)
                if len(local) > 2
                else local[0] + "*"
            )
            anon = f"{mk}@{domain}"
        else:
            anon = "*****@***.com"
        return (
            "Domain-Preserving Masking",
            anon,
            "Local-part masked, domain kept",
        )
    elif "BANK" in lbl or "ACCOUNT" in lbl:
        d = re.sub(r"\D", "", val)
        anon = "*" * (len(d) - 4) + d[-4:] if len(d) >= 4 else "XXXXXXXXXXXX"
        return ("Partial Masking", anon, "All but last 4 digits masked")
    elif "CARD" in lbl or "CREDIT" in lbl:
        d = re.sub(r"\D", "", val)
        anon = (
            f"XXXX-XXXX-XXXX-{d[-4:]}"
            if len(d) >= 16
            else "XXXX-XXXX-XXXX-XXXX"
        )
        return (
            "Tokenization",
            anon,
            "Full card number tokenized; last 4 kept",
        )
    elif "PERSON" in lbl or "NAME" in lbl or "PER" in lbl:
        parts = val.split()
        if parts:
            anon = parts[0][0] + ". " + " ".join(p[0] + "." for p in parts[1:])
        else:
            anon = "[NAME REDACTED]"
        return (
            "Initial-Only Masking",
            anon,
            "First initial retained; rest reduced to initials",
        )
    elif "DATE" in lbl or "DOB" in lbl:
        return (
            "Date Generalization",
            "[DATE REDACTED]",
            "Full date replaced with placeholder",
        )
    elif "AGE" in lbl:
        m = re.search(r"\d+", val)
        if m:
            age = int(m.group())
            bucket = f"{(age // 10) * 10}s"
            anon = f"[Age range: {bucket}]"
        else:
            anon = "[Age Redacted]"
        return (
            "Generalization",
            anon,
            "Exact age bucketed into decade range",
        )
    elif "PINCODE" in lbl:
        d = re.sub(r"\D", "", val)
        anon = d[:3] + "XXX" if len(d) >= 6 else "XXXXXX"
        return ("Partial Masking", anon, "Last 3 digits of pincode masked")
    elif "LOCATION" in lbl or "LOC" in lbl:
        return (
            "Generalization",
            "[LOCATION REDACTED]",
            "Location replaced with placeholder",
        )
    elif "ORGANIZATION" in lbl or "ORG" in lbl:
        return (
            "Generalization",
            "[ORGANIZATION REDACTED]",
            "Organization replaced with placeholder",
        )
    else:
        anon = hashlib.sha256(val.encode()).hexdigest()[:16].upper()
        return ("One-Way Hashing", f"SHA256:{anon}", "Value hashed with SHA-256")


# ─────────────────────────────────────────────────────────────────────────────
# NER ENGINE & SPAN MERGING LOGIC
# ─────────────────────────────────────────────────────────────────────────────

try:
    import torch as _torch
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
    )
    from transformers import (
        pipeline as _hf_pipeline,
    )

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

_NER_DEVICE = (
    0 if (_TRANSFORMERS_AVAILABLE and _torch.cuda.is_available()) else -1
)
_NER_DEVICE_STR = "GPU" if _NER_DEVICE == 0 else "CPU"

_NER_PIPELINE_CACHE: Dict[str, object] = {}


def _load_ner_pipeline(model_id: str, use_fast: bool = True) -> Optional[object]:
    if model_id in _NER_PIPELINE_CACHE:
        return _NER_PIPELINE_CACHE[model_id]
    if not _TRANSFORMERS_AVAILABLE:
        return None
    try:
        hf_token = os.getenv("HF_TOKEN", DEFAULT_HF_TOKEN).strip() or None
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=hf_token,
            use_fast=use_fast,
        )
        model = AutoModelForTokenClassification.from_pretrained(
            model_id,
            token=hf_token,
        )
        ner_pipe = _hf_pipeline(
            task="ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=_NER_DEVICE,
        )
        _NER_PIPELINE_CACHE[model_id] = ner_pipe
        return ner_pipe
    except Exception as exc:
        print(f"  [NER] Could not load {model_id}: {exc}", flush=True)
        return None


def _clean_entity_text(text: str, start: int, end: int) -> Tuple[str, int, int]:
    """
    Cleans leading and trailing punctuation, isolated single characters,
    and slash-prefixes (e.g., 'o.Name', 'S/o Name', 'Name s') from NER spans.
    """
    val = text[start:end]

    # 1. Strip leading punctuation, slashes, and dangling prefix letters like "o.", "s.", "d."
    match_prefix = re.match(
        r"^([\s:\-.,'\"/()]+|[A-Za-z]/[a-zA-Z]?\.?\s*|[a-zA-Z]\.\s*)", val
    )
    while match_prefix and match_prefix.end() > 0:
        cut = match_prefix.end()
        # Ensure we don't trim valid initials if the entire name is initials
        if cut >= len(val):
            break
        start += cut
        val = text[start:end]
        match_prefix = re.match(
            r"^([\s:\-.,'\"/()]+|[A-Za-z]/[a-zA-Z]?\.?\s*|[a-zA-Z]\.\s*)", val
        )

    # 2. Strip trailing punctuation and dangling sub-word fragments (e.g. trailing " s")
    match_suffix = re.search(r"(\s+[a-zA-Z]|[\s:\-.,'\"/()]+)$", val)
    while match_suffix and match_suffix.start() < len(val):
        cut = len(val) - match_suffix.start()
        if cut >= len(val):
            break
        end -= cut
        val = text[start:end]
        match_suffix = re.search(r"(\s+[a-zA-Z]|[\s:\-.,'\"/()]+)$", val)

    return val.strip(), start, end


def _snap_to_word_boundary(text: str, start: int, end: int) -> Tuple[int, int]:
    """
    Expands character indices to complete word boundaries while strictly
    halting at delimiter boundaries (/ , ; : \\).
    """
    STRICT_DELIMITERS = set(" /\\:;,()[]{}<>\"'\t\n\r")

    # Expand start backwards
    while start > 0:
        prev = text[start - 1]
        if prev in STRICT_DELIMITERS:
            break
        if prev.isalnum():
            start -= 1
        elif (
            prev == "."
            and start > 1
            and text[start - 2].isalpha()
            and (
                start == 2
                or text[start - 3] in STRICT_DELIMITERS
                or text[start - 3].isspace()
            )
        ):
            start -= 1
        else:
            break

    # Expand end forwards
    while end < len(text):
        nxt = text[end]
        if nxt in STRICT_DELIMITERS:
            break
        if nxt.isalnum():
            end += 1
        elif nxt == "." and end + 1 < len(text) and text[end + 1].isalnum():
            end += 1
        else:
            break

    return start, end


def merge_line_spans(spans: List[Dict], original_line: str) -> List[NerEntity]:
    if not spans:
        return []

    snapped_spans: List[Dict] = []
    for s in spans:
        st, en = _snap_to_word_boundary(original_line, s["start"], s["end"])
        clean_val, st, en = _clean_entity_text(original_line, st, en)

        # Discard false positives: empty, numeric, or document abbreviations like S.I.No
        if (
            not clean_val
            or len(clean_val) < 2
            or re.match(r"^\d+$", clean_val)
            or re.match(r"^[A-Z]\.([A-Z]\.)+[A-Za-z]+$", clean_val)
            # or _is_orphaned_initial(clean_val)
        ):
            continue

        snapped_spans.append(
            {
                "cat": s["cat"],
                "start": st,
                "end": en,
                "score": float(s["score"]),
                "text": clean_val,
                "word_no": get_word_number(original_line, st),
                "start_char": st + 1,
                "end_char": en,
            }
        )

    if not snapped_spans:
        return []

    # 1. Merge contiguous identical-category spans
    sorted_spans = sorted(
        snapped_spans, key=lambda x: (x["start"], -(x["end"] - x["start"]))
    )
    same_cat_merged: List[Dict] = []

    for s in sorted_spans:
        target = None
        for m in same_cat_merged:
            if m["cat"] == s["cat"] and (
                s["start"] <= m["end"] + 1 and m["start"] <= s["end"] + 1
            ):
                intervening = original_line[
                    min(m["start"], s["start"]) : max(m["end"], s["end"])
                ]
                if not any(d in intervening for d in ["/", "\\", ";", ":"]):
                    target = m
                    break

        if target is None:
            same_cat_merged.append(dict(s))
        else:
            target["start"] = min(target["start"], s["start"])
            target["end"] = max(target["end"], s["end"])
            target_text, t_st, t_en = _clean_entity_text(
                original_line, target["start"], target["end"]
            )
            target["text"] = target_text
            target["start"] = t_st
            target["end"] = t_en
            target["score"] = max(target["score"], s["score"])
            target["word_no"] = get_word_number(original_line, target["start"])
            target["start_char"] = target["start"] + 1
            target["end_char"] = target["end"]

    # 2. Non-Maximum Suppression (Longest span with highest confidence)
    same_cat_merged.sort(key=lambda x: (-(x["end"] - x["start"]), -x["score"]))
    final_merged: List[Dict] = []

    for candidate in same_cat_merged:
        if not any(
            candidate["start"] < chosen["end"]
            and chosen["start"] < candidate["end"]
            for chosen in final_merged
        ):
            final_merged.append(candidate)

    final_merged.sort(key=lambda x: x["start"])

    entities: List[NerEntity] = []
    seen = set()
    for m in final_merged:
        clean_val = m["text"]
        if (
            clean_val
            and len(clean_val) >= 2
            # and not _is_orphaned_initial(clean_val)
        ):
            key = (m["cat"], clean_val.lower(), m["start"])
            if key not in seen:
                seen.add(key)
                entities.append(
                    NerEntity(
                        category=m["cat"],
                        text=clean_val,
                        score=round(m["score"], 4),
                        word_no=m["word_no"],
                        start_char=m["start_char"],
                        end_char=m["end_char"],
                    )
                )

    return entities


def _extract_raw_spans(
    pipe,
    line_str: str,
    min_score: float = 0.20,
    model_name: str = "",
) -> List[Dict]:
    if pipe is None or not line_str.strip():
        return []

    try:
        results = pipe(line_str)
    except Exception:
        return []

    if not results:
        return []

    raw_spans = []
    for item in results:
        score = float(item.get("score", 0.0))
        group = str(item.get("entity_group", item.get("entity", ""))).upper()
        group = re.sub(r"^[BI]-", "", group)

        if score < min_score:
            continue

        cat = None
        if group in ("PER", "PERSON"):
            cat = "PERSON"
        elif group in ("LOC", "LOCATION", "GPE"):
            cat = "LOCATION"
        elif group in ("ORG", "ORGANIZATION"):
            cat = "ORGANIZATION"

        if not cat:
            continue

        start = item.get("start", 0)
        end = item.get("end", 0)
        if start < end:
            val = line_str[start:end]
            raw_spans.append(
                {
                    "cat": cat,
                    "start": start,
                    "end": end,
                    "score": score,
                    "text": val,
                    "model": model_name,
                }
            )

    return raw_spans


def _chunk_line_with_offsets(
    line: str,
    tokenizer,
    max_tokens: int = 380,
) -> List[Tuple[int, str]]:
    pieces = [
        (m.start(), m.group())
        for m in re.finditer(r"[^.\n।]*[.\n।]|[^.\n।]+$", line)
        if m.group().strip()
    ]

    chunks: List[Tuple[int, str]] = []
    cur_start: Optional[int] = None
    cur: str = ""

    for start, piece in pieces:
        candidate = cur + piece
        token_count = len(
            tokenizer(candidate, add_special_tokens=True)["input_ids"]
        )
        if cur and token_count > max_tokens:
            chunks.append((cur_start, cur))
            cur_start = start
            cur = piece
        else:
            if cur_start is None:
                cur_start = start
            cur = candidate

    if cur.strip():
        chunks.append((cur_start if cur_start is not None else 0, cur))

    return chunks or [(0, line)]


def _extract_line_spans_chunked(
    pipe,
    line_str: str,
    min_score: float = 0.20,
    model_name: str = "",
) -> List[Dict]:
    if pipe is None or not line_str.strip():
        return []

    tokenizer = pipe.tokenizer
    max_tokens = 380

    token_count = len(tokenizer(line_str, add_special_tokens=True)["input_ids"])
    if token_count <= max_tokens:
        return _extract_raw_spans(
            pipe, line_str, min_score=min_score, model_name=model_name
        )

    all_spans: List[Dict] = []
    for char_offset, chunk in _chunk_line_with_offsets(
        line_str, tokenizer, max_tokens
    ):
        chunk_spans = _extract_raw_spans(
            pipe, chunk, min_score=min_score, model_name=model_name
        )
        for s in chunk_spans:
            s["start"] += char_offset
            s["end"] += char_offset
            all_spans.append(s)

    return all_spans


def extract_dual_pass_ner_spans(
    pipe,
    line_str: str,
    min_score: float = 0.20,
    model_name: str = "",
) -> List[Dict]:
    """
    Performs pure model inference in two passes:
      1. Verbatim Pass: processes line_str as written.
      2. Truecased Pass: processes truecase_line(line_str).
    Because truecase_line preserves character length and index offsets exactly,
    spans detected in both passes map 100% cleanly to original text character offsets.
    """
    spans_verbatim = _extract_line_spans_chunked(
        pipe, line_str, min_score=min_score, model_name=model_name
    )

    tc_line = truecase_line(line_str)
    spans_tc = []
    if tc_line != line_str:
        spans_tc = _extract_line_spans_chunked(
            pipe, tc_line, min_score=min_score, model_name=model_name
        )

    return spans_verbatim + spans_tc


# ─────────────────────────────────────────────────────────────────────────────
# MAIN NER RUNNER
# ─────────────────────────────────────────────────────────────────────────────


def run_line_by_line_ner(records: List[FileRecord]):
    if not _TRANSFORMERS_AVAILABLE:
        print(
            "  [NER] transformers not installed — skipping model inference.",
            flush=True,
        )
        return

    print(
        f"  [NER] Loading HuggingFace models on {_NER_DEVICE_STR} …",
        flush=True,
    )
    hiner_pipe = _load_ner_pipeline(
        "cfilt/HiNER-original-muril-base-cased", use_fast=False
    )
    indicner_pipe = _load_ner_pipeline("ai4bharat/IndicNER", use_fast=False)
    bert_pipe = _load_ner_pipeline("dslim/bert-base-NER", use_fast=True)
    xlm_pipe = _load_ner_pipeline(
        "Babelscape/wikineural-multilingual-ner", use_fast=True
    )

    pipes = {
        NER_MODELS["HiNER"][0]: (hiner_pipe, 0.20),
        NER_MODELS["IndicNER"][0]: (indicner_pipe, 0.20),
        NER_MODELS["BERT_Base_NER"][0]: (bert_pipe, 0.35),
        NER_MODELS["XLM_RoBERTa"][0]: (xlm_pipe, 0.35),
    }

    for rec in records:
        print(
            f"  [NER] Running line-by-line NER on {rec.filename} …", flush=True
        )
        lines = rec.raw_text.splitlines()

        for model_lbl in list(pipes.keys()) + [HYBRID_KEY]:
            rec.line_ners[model_lbl] = []

        for line_idx, line in enumerate(lines):
            original_line = line.strip()
            lno = line_idx + 1

            if not original_line:
                continue

            model_raw_spans: Dict[str, List[Dict]] = {}
            model_line_ents: Dict[str, List[NerEntity]] = {}

            # Dual-pass pure neural model inference
            for model_lbl, (pipe, threshold) in pipes.items():
                spans = extract_dual_pass_ner_spans(
                    pipe, original_line, min_score=threshold, model_name=model_lbl
                )
                model_raw_spans[model_lbl] = spans
                model_line_ents[model_lbl] = merge_line_spans(
                    spans, original_line
                )

            # Ensemble Hybrid: HiNER + IndicNER + XLM-RoBERTa
            hybrid_spans = (
                model_raw_spans.get(NER_MODELS["HiNER"][0], [])
                + model_raw_spans.get(NER_MODELS["IndicNER"][0], [])
                + model_raw_spans.get(NER_MODELS["XLM_RoBERTa"][0], [])
            )
            model_line_ents[HYBRID_KEY] = merge_line_spans(hybrid_spans, original_line)

            all_line_ents = set(
                (e.category, e.text)
                for ents in model_line_ents.values()
                for e in ents
            )

            for model_lbl, ents in model_line_ents.items():
                persons = [e.text for e in ents if e.category == "PERSON"]
                locs = [e.text for e in ents if e.category == "LOCATION"]
                orgs = [e.text for e in ents if e.category == "ORGANIZATION"]

                current_ent_pairs = set((e.category, e.text) for e in ents)
                missed = [
                    f"{val} ({cat})"
                    for cat, val in all_line_ents
                    if (cat, val) not in current_ent_pairs
                ]

                lres = LineNerResult(
                    model=model_lbl,
                    line_no=lno,
                    text=original_line,
                    entities=ents,
                    persons=persons,
                    locs=locs,
                    orgs=orgs,
                    missed=missed,
                )
                rec.line_ners[model_lbl].append(lres)


# ─────────────────────────────────────────────────────────────────────────────
# JSON GENERATOR (MAIN + INDIVIDUAL MODEL JSONs)
# ─────────────────────────────────────────────────────────────────────────────


def build_json(records: List[FileRecord], output_path: str):
    base_name, _ = os.path.splitext(output_path)
    main_json_path = base_name + ".json"
    print(f"\n▶ Writing JSON reports → {base_name}_*.json …", flush=True)

    all_models = [
        NER_MODELS["HiNER"][0],
        NER_MODELS["IndicNER"][0],
        NER_MODELS["BERT_Base_NER"][0],
        NER_MODELS["XLM_RoBERTa"][0],
        HYBRID_KEY,
    ]

    model_slugs = {
        NER_MODELS["HiNER"][0]: "HiNER",
        NER_MODELS["IndicNER"][0]: "IndicNER",
        NER_MODELS["BERT_Base_NER"][0]: "BERT_Base_NER",
        NER_MODELS["XLM_RoBERTa"][0]: "XLM_RoBERTa",
        HYBRID_KEY: "Hybrid",
    }

    # 1. Combined / Main JSON Report
    main_export = []
    for rec in records:
        file_entry = {
            "file_info": {
                "filename": rec.filename,
                "file_type": rec.file_type.upper(),
                "language": rec.language,
                "path": rec.path,
            },
            "pii_detections": [
                {
                    "line_no": h.line_no,
                    "word_no": h.word_no,
                    "word_position": ordinal(h.word_no),
                    "start_letter": h.start_char,
                    "end_letter": h.end_char,
                    "letter_span": f"{h.start_char}-{h.end_char}",
                    "source": h.source,
                    "type": h.label,
                    "display_type": h.display,
                    "original_value": h.value,
                    "tag": f"<{h.label}>{h.value}</{h.label}>",
                }
                for h in rec.pii_hits
            ],
            "ner_detections": [],
        }

        hybrid_lines = rec.line_ners.get(HYBRID_KEY, [])
        for lres in hybrid_lines:
            for ent in lres.entities:
                file_entry["ner_detections"].append(
                    {
                        "line_no": lres.line_no,
                        "word_no": ent.word_no,
                        "word_position": ordinal(ent.word_no),
                        "start_letter": ent.start_char,
                        "end_letter": ent.end_char,
                        "letter_span": f"{ent.start_char}-{ent.end_char}",
                        "source": HYBRID_KEY,
                        "type": ent.category,
                        "original_value": ent.text,
                        "confidence_score": ent.score,
                        "tag": f"<{ent.category}>{ent.text}</{ent.category}>",
                    }
                )

        main_export.append(file_entry)

    with open(main_json_path, "w", encoding="utf-8") as f:
        json.dump(main_export, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Main combined JSON report generated → {main_json_path}", flush=True)

    # 2. Individual JSON File for EACH NER Model
    for model_lbl in all_models:
        slug = model_slugs.get(model_lbl, re.sub(r"\W+", "_", model_lbl))
        model_json_path = f"{base_name}_{slug}.json"

        model_export = []
        for rec in records:
            file_entry = {
                "file_info": {
                    "filename": rec.filename,
                    "file_type": rec.file_type.upper(),
                    "language": rec.language,
                    "path": rec.path,
                },
                "model_name": model_lbl,
                "pii_detections": [
                    {
                        "line_no": h.line_no,
                        "word_no": h.word_no,
                        "word_position": ordinal(h.word_no),
                        "start_letter": h.start_char,
                        "end_letter": h.end_char,
                        "letter_span": f"{h.start_char}-{h.end_char}",
                        "source": h.source,
                        "type": h.label,
                        "display_type": h.display,
                        "original_value": h.value,
                        "tag": f"<{h.label}>{h.value}</{h.label}>",
                    }
                    for h in rec.pii_hits
                ],
                "ner_detections": [],
            }

            model_lines = rec.line_ners.get(model_lbl, [])
            for lres in model_lines:
                for ent in lres.entities:
                    file_entry["ner_detections"].append(
                        {
                            "line_no": lres.line_no,
                            "word_no": ent.word_no,
                            "word_position": ordinal(ent.word_no),
                            "start_letter": ent.start_char,
                            "end_letter": ent.end_char,
                            "letter_span": f"{ent.start_char}-{ent.end_char}",
                            "source": model_lbl,
                            "type": ent.category,
                            "original_value": ent.text,
                            "confidence_score": ent.score,
                            "tag": f"<{ent.category}>{ent.text}</{ent.category}>",
                        }
                    )

            model_export.append(file_entry)

        with open(model_json_path, "w", encoding="utf-8") as f:
            json.dump(model_export, f, indent=2, ensure_ascii=False)
        print(f"  ✅ Separate JSON generated for {slug} → {model_json_path}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# EXCEL GENERATOR (4 SHEETS)
# ─────────────────────────────────────────────────────────────────────────────
_C = {
    "header_bg": "1F4E79",
    "header_fg": "FFFFFF",
    "pii_hit": "FFE0E0",
    "anon_cell": "E8F5E9",
    "lang_cell": "FFF9C4",
    "regex_cell": "E3F2FD",
    "presidio_bg": "FFF3E0",
    "green_flag": "C8E6C9",
    "age_cell": "F3E5F5",
}
_FILLS = {
    k: PatternFill(start_color=v, end_color=v, fill_type="solid")
    for k, v in _C.items()
}
_H_FONT = Font(color=_C["header_fg"], bold=True, size=10)
_WRAP = Alignment(wrap_text=True, vertical="top")
_CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)


def _style_header(ws):
    for cell in ws[1]:
        cell.fill = _FILLS["header_bg"]
        cell.font = _H_FONT
        cell.alignment = _CENTER
    ws.freeze_panes = "A2"
    ws.row_dimensions[1].height = 30


def _auto_width(ws, max_w: int = 55):
    for col_cells in ws.columns:
        col_letter = get_column_letter(col_cells[0].column)
        best = max((len(str(c.value or "")) for c in col_cells), default=10)
        ws.column_dimensions[col_letter].width = min(best + 4, max_w)


def build_excel(
    records: List[FileRecord], output_path: str, presidio_available: bool
):
    print(f"\n▶ Writing 4-Sheet Excel report → {output_path} …", flush=True)

    all_model_labels = [
        NER_MODELS["HiNER"][0],
        NER_MODELS["IndicNER"][0],
        NER_MODELS["BERT_Base_NER"][0],
        NER_MODELS["XLM_RoBERTa"][0],
        HYBRID_KEY,
    ]

    # Sheet 1: Summary
    s1_rows = []
    for rec in records:
        all_hits = rec.pii_hits
        regex_hits = [
            h for h in all_hits if h.source in ("Regex", "Contextual Regex")
        ]
        presidio_h = [h for h in all_hits if h.source == "Presidio"]
        tot_pii = len(all_hits)
        pii_types = ", ".join(sorted({h.label for h in all_hits})) or "None"

        hybrid_lines = rec.line_ners.get(HYBRID_KEY, [])
        hy_persons = list(
            dict.fromkeys(p for lres in hybrid_lines for p in lres.persons)
        )
        hy_locs = list(
            dict.fromkeys(l for lres in hybrid_lines for l in lres.locs)
        )
        hy_orgs = list(
            dict.fromkeys(o for lres in hybrid_lines for o in lres.orgs)
        )

        s1_rows.append(
            {
                "File": rec.filename,
                "File Type": rec.file_type.upper(),
                "Language": rec.language,
                "Total PII Entities": tot_pii,
                "Regex PII Hits": len(regex_hits),
                "Presidio PII Hits": len(presidio_h),
                "PII Types Found": pii_types,
                "Hybrid NER Persons": " | ".join(hy_persons) or "None",
                "Hybrid NER Locations": " | ".join(hy_locs) or "None",
                "Hybrid NER Organizations": " | ".join(hy_orgs) or "None",
                "Has PII": "YES" if tot_pii > 0 else "NO",
                "Presidio Engine": (
                    "Active" if presidio_available else "Not Installed"
                ),
            }
        )

    # Sheet 2: PII Detection
    s2_rows = []
    for rec in records:
        if not rec.pii_hits:
            s2_rows.append(
                {
                    "File": rec.filename,
                    "File Type": rec.file_type.upper(),
                    "Language": rec.language,
                    "Line No": "-",
                    "Word No": "-",
                    "Word Position": "-",
                    "Start Letter": "-",
                    "End Letter": "-",
                    "Letter Span": "-",
                    "Extracted Text": rec.raw_text,
                    "PII Type": "None",
                    "PII Display Name": "None",
                    "Detected Value": "None Detected",
                    "Detected By": "None",
                    "PII Tag": "—",
                }
            )
        else:
            lines = rec.raw_text.splitlines()
            for h in rec.pii_hits:
                line_text = (
                    lines[h.line_no - 1]
                    if 0 < h.line_no <= len(lines)
                    else rec.raw_text
                )
                s2_rows.append(
                    {
                        "File": rec.filename,
                        "File Type": rec.file_type.upper(),
                        "Language": rec.language,
                        "Line No": h.line_no,
                        "Word No": h.word_no,
                        "Word Position": ordinal(h.word_no),
                        "Start Letter": h.start_char,
                        "End Letter": h.end_char,
                        "Letter Span": f"{h.start_char}-{h.end_char}",
                        "Extracted Text": line_text,
                        "PII Type": h.label,
                        "PII Display Name": h.display,
                        "Detected Value": h.value,
                        "Detected By": h.source,
                        "PII Tag": f"<{h.label}>{h.value}</{h.label}>",
                    }
                )

    # Sheet 3: NER Comparison
    s3_rows = []
    for rec in records:
        line_count = len(rec.line_ners.get(HYBRID_KEY, []))
        if line_count == 0:
            row = {
                "File": rec.filename,
                "File Type": rec.file_type.upper(),
                "Language": rec.language,
                "Line No": 1,
                "Input_Text": rec.raw_text,
                "Has_Context": len(rec.raw_text.split()) > 2,
            }
            for model_lbl in all_model_labels:
                row[f"{model_lbl}_Predicted_Type"] = "NONE"
                row[f"{model_lbl}_Extracted_Text"] = "None"
                row[f"{model_lbl}_Missed_Entities"] = "None"
                row[f"{model_lbl}_Status"] = "✗ None Detected"
            s3_rows.append(row)
        else:
            for idx in range(line_count):
                sample_lres = rec.line_ners[HYBRID_KEY][idx]
                row = {
                    "File": rec.filename,
                    "File Type": rec.file_type.upper(),
                    "Language": rec.language,
                    "Line No": sample_lres.line_no,
                    "Input_Text": sample_lres.text,
                    "Has_Context": len(sample_lres.text.split()) > 2,
                }
                for model_lbl in all_model_labels:
                    lres = rec.line_ners[model_lbl][idx]
                    row[f"{model_lbl}_Predicted_Type"] = lres.predicted_type
                    row[f"{model_lbl}_Extracted_Text"] = lres.extracted_text
                    row[f"{model_lbl}_Missed_Entities"] = (
                        " | ".join(lres.missed) if lres.missed else "None"
                    )
                    row[f"{model_lbl}_Status"] = (
                        "✓ Detected"
                        if (lres.persons or lres.locs or lres.orgs)
                        else "✗ None Detected"
                    )
                s3_rows.append(row)

    # Sheet 4: Anonymization
    s4_rows = []
    for rec in records:
        lines = rec.raw_text.splitlines()

        for h in rec.pii_hits:
            line_text = (
                lines[h.line_no - 1]
                if 0 < h.line_no <= len(lines)
                else rec.raw_text
            )
            method, anon_val, desc = anonymize_value(h.label, h.value)
            s4_rows.append(
                {
                    "File": rec.filename,
                    "File Type": rec.file_type.upper(),
                    "Language": rec.language,
                    "Line No": h.line_no,
                    "Word No": h.word_no,
                    "Word Position": ordinal(h.word_no),
                    "Start Letter": h.start_char,
                    "End Letter": h.end_char,
                    "Letter Span": f"{h.start_char}-{h.end_char}",
                    "Extracted Text": line_text,
                    "Source Type": "PII",
                    "Detection Engine": h.source,
                    "Entity / PII Type": h.display,
                    "Original Value": h.value,
                    "Confidence Score": "1.00 (Exact Regex/Presidio)",
                    "PII / NER Tag": f"<{h.label}>{h.value}</{h.label}>",
                    "Anonymization Method": method,
                    "Anonymized Value": anon_val,
                    "Method Description": desc,
                }
            )

        hybrid_lines = rec.line_ners.get(HYBRID_KEY, [])
        anon_seen: set = set()

        for lres in hybrid_lines:
            line_text = lres.text

            for ent in lres.entities:
                key = (lres.line_no, ent.category, ent.text.lower())
                if key in anon_seen:
                    continue
                anon_seen.add(key)

                display_label = (
                    "Person Name"
                    if ent.category == "PERSON"
                    else (
                        "Location / City"
                        if ent.category == "LOCATION"
                        else "Organization"
                    )
                )

                method, anon_val, desc = anonymize_value(ent.category, ent.text)
                s4_rows.append(
                    {
                        "File": rec.filename,
                        "File Type": rec.file_type.upper(),
                        "Language": rec.language,
                        "Line No": lres.line_no,
                        "Word No": ent.word_no,
                        "Word Position": ordinal(ent.word_no),
                        "Start Letter": ent.start_char,
                        "End Letter": ent.end_char,
                        "Letter Span": f"{ent.start_char}-{ent.end_char}",
                        "Extracted Text": line_text,
                        "Source Type": "NER Model",
                        "Detection Engine": HYBRID_KEY,
                        "Entity / PII Type": display_label,
                        "Original Value": ent.text,
                        "Confidence Score": f"{ent.score:.4f}",
                        "PII / NER Tag": f"<{ent.category}>{ent.text}</{ent.category}>",
                        "Anonymization Method": method,
                        "Anonymized Value": anon_val,
                        "Method Description": desc,
                    }
                )

    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame(s1_rows).to_excel(
            writer, sheet_name="Summary", index=False
        )
        pd.DataFrame(s2_rows).to_excel(
            writer, sheet_name="PII Detection", index=False
        )
        pd.DataFrame(s3_rows).to_excel(
            writer, sheet_name="NER Comparison", index=False
        )
        pd.DataFrame(s4_rows).to_excel(
            writer, sheet_name="Anonymization", index=False
        )

    wb = openpyxl.load_workbook(output_path)

    for sheet_name in [
        "Summary",
        "PII Detection",
        "NER Comparison",
        "Anonymization",
    ]:
        ws = wb[sheet_name]
        _style_header(ws)
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = _WRAP
                hdr = str(ws.cell(1, cell.column).value or "")
                val = str(cell.value or "")

                if hdr == "Language":
                    cell.fill = _FILLS["lang_cell"]
                elif hdr == "Has PII" and val == "YES":
                    cell.fill = _FILLS["pii_hit"]
                elif hdr == "Has PII" and val == "NO":
                    cell.fill = _FILLS["green_flag"]
                elif hdr == "Detected By" and val == "Presidio":
                    cell.fill = _FILLS["presidio_bg"]
                elif hdr == "Source Type" and val == "NER Model":
                    cell.fill = _FILLS["regex_cell"]
                elif "Original Value" in hdr:
                    cell.fill = _FILLS["pii_hit"]
                elif "Anonymized Value" in hdr:
                    cell.fill = _FILLS["anon_cell"]
        _auto_width(ws)

    wb.save(output_path)
    print(
        f"  ✅ Excel report generated successfully → {output_path}", flush=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# INLINE STRING INPUT SUPPORT
# ─────────────────────────────────────────────────────────────────────────────


def process_text_string(
    text: str,
    output_path: str = DEFAULT_OUTPUT_PATH,
    label: str = "inline_input",
) -> List[FileRecord]:
    text = text.replace("\\n", "\n").replace("\\t", "\t")
    text = text.strip("'\"")

    if not text.strip():
        print(
            "[WARN] --text input is empty; nothing to process.", flush=True
        )
        return []

    lang = detect_language(text)
    rec = FileRecord(
        path="<inline>",
        filename=label,
        file_type="string",
        language=lang,
        raw_text=text,
    )

    presidio_ok = _PRESIDIO_AVAILABLE and (_get_presidio_engine() is not None)
    regex_hits, presidio_hits = full_pii_scan(text)
    rec.pii_hits = merge_pii_hits(regex_hits, presidio_hits)

    run_line_by_line_ner([rec])
    build_excel([rec], output_path, presidio_ok)
    build_json([rec], output_path)

    return [rec]


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Multi-Format (TXT/DOCX/HTML) → Line-by-Line PII/NER Detection → Excel & JSON Reports\n\n"
            "Input modes:\n"
            "  File / folder:  --input path/to/file_or_folder\n"
            '  Inline string:  --text "Your text enclosed in quotes"\n'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--input",
        "-i",
        default=None,
        help="Path to a single file (.txt/.docx/.doc/.html) or a folder containing such files.",
    )
    ap.add_argument(
        "--text",
        "-t",
        default=None,
        metavar="STRING",
        help="Inline text string to analyse instead of a file.",
    )
    ap.add_argument(
        "--output",
        "-o",
        default=DEFAULT_OUTPUT_PATH,
        help="Output .xlsx file path (default: pii_ner_report.xlsx).",
    )
    ap.add_argument(
        "--hf_token",
        default=DEFAULT_HF_TOKEN,
        help="HuggingFace token (for gated models such as IndicNER).",
    )
    if any("jupyter" in arg or "kernel" in arg for arg in sys.argv):
        return ap.parse_args(args=[])
    return ap.parse_args()


def main():
    args = parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    presidio_ok = _PRESIDIO_AVAILABLE and (_get_presidio_engine() is not None)
    print(
        f"[Pipeline] Presidio:         {'✓ Active' if presidio_ok else '✗ Not available'}",
        flush=True,
    )
    print(
        f"[Pipeline] NER Transformers: {'✓ Available on ' + _NER_DEVICE_STR if _TRANSFORMERS_AVAILABLE else '✗ Not installed'}",
        flush=True,
    )

    if args.text is not None:
        if args.input is not None:
            print(
                "[WARN] Both --text and --input were supplied; --text takes priority.",
                flush=True,
            )
        process_text_string(args.text, output_path=args.output)
        return

    input_path = args.input if args.input is not None else DEFAULT_INPUT_PATH
    supported_files = collect_files(input_path)

    records: List[FileRecord] = []
    for fp in supported_files:
        raw, ftype = extract_text(fp)
        if not raw.strip():
            continue
        lang = detect_language(raw)
        records.append(
            FileRecord(
                path=fp,
                filename=os.path.basename(fp),
                file_type=ftype,
                language=lang,
                raw_text=raw,
            )
        )

    for rec in records:
        regex_hits, presidio_hits = full_pii_scan(rec.raw_text)
        rec.pii_hits = merge_pii_hits(regex_hits, presidio_hits)

    run_line_by_line_ner(records)
    build_excel(records, args.output, presidio_ok)
    build_json(records, args.output)


if __name__ == "__main__":
    main()