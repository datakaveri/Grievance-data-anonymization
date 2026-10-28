#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PIPELINE.PY — Multi-Format PII Detection & NER Comparison Pipeline          ║
# ║  Supports: .txt, .doc, .docx, .html, .json, .csv (single file or folder)     ║
# ║  PII Detection: Regex only                                                   ║
# ║  NER: Queue-driven parallel IndicNER / XLM-RoBERTa Davlan (pull cached files)║
# ╚══════════════════════════════════════════════════════════════════════════════╝

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import multiprocessing as mp
import os
# Force PyTorch to hide CUDA devices at module load time to prevent CUDA driver probing errors
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Enable high-speed Rust-based multi-threaded HuggingFace downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import re
import sys
import threading
import time
import unicodedata
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

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


try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

DEFAULT_INPUT_PATH = "/home/azureuser/anon/data/sample_haryana.json"  # set via config or CLI --input
DEFAULT_OUTPUT_PATH = "pii_ner_report.xlsx"
DEFAULT_HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
# Loading a .bin checkpoint (HiNER, IndicNER) otherwise spawns a background
# thread that queries the Hub to open a safetensors-conversion PR — a network
# call on every run, even from a warm cache, and a 401 on gated repos.
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")


def truecase_line(text: str) -> str:
    """
    WE ARE NOT USING THIS NOW
    Performs a 1-to-1 length-preserving title-casing transformation on words.
    Enables cased Transformer models to recognize entities in ALL-CAPS or all-lowercase text
    without altering character indices. 
    """
    return re.sub(r"\b[A-Za-z]+\b", lambda m: m.group(0).capitalize(), text)

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

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT PATHS & CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_OUTPUT_PATH = "pii_ner_report.xlsx"

NER_MODELS: Dict[str, Tuple[str, str]] = {
    "IndicNER": ("IndicNER (AI4Bharat / Public)", "ai4bharat/IndicNER"),
    "XLM_RoBERTa": (
        "XLM-RoBERTa (Multilingual)",
        "Davlan/xlm-roberta-base-ner-hrl",
    ),
}
HYBRID_KEY = "Hybrid (IndicNER + XLM-RoBERTa)"

# Queue-driven NER (see NER_Parallel_Pipeline_Short_Architecture.docx):
# mini-batches of 16–32 lines cut tokenizer overhead; bounded queues apply backpressure.
NER_LINE_BATCH_SIZE = 24
NER_QUEUE_MAXSIZE = 64
NER_RESULT_GET_TIMEOUT_S = 1.0
NER_WORKER_JOIN_TIMEOUT_S = 300.0

PII_DISPLAY_NAMES: Dict[str, str] = {
    "Aadhaar": "Aadhaar Number",
    "PAN": "PAN Card",
    "Phone_Number": "Phone Number",
    "Email": "Email Address",
    "Medical_UHID": "Medical UHID",
    "Patient_ID": "Patient ID",
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
    "Reference_Number": "Reference Number",
    "ID_Number": "Identifier",
    "GSTIN": "GSTIN",
    "ABHA": "ABHA Health ID",
    "UPI_ID": "UPI ID",
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
    confidence: float = 1.0


@dataclass
class NerEntity:
    category: str
    text: str
    score: float
    word_no: int = 0
    start_char: int = 0
    end_char: int = 0
    # Models whose raw spans make up this entity (any category), for agreement checks.
    models: List[str] = field(default_factory=list)


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


@dataclass
class LineTask:
    """One dispatcher item: a single source line for all NER workers."""

    doc_id: int
    line_no: int
    line_text: str


@dataclass
class ModelLineResult:
    """Fan-in item from a model worker into Q_M."""

    doc_id: int
    line_no: int
    model_name: str
    spans: List[Dict]
    error: Optional[str] = None


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


def _flatten_json_value(obj, prefix: str = "") -> List[str]:
    """Turn nested JSON into 'key: value' lines so PII/NER see every field."""
    lines: List[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            lines.extend(_flatten_json_value(value, next_prefix))
        return lines
    if isinstance(obj, list):
        for idx, value in enumerate(obj):
            next_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            lines.extend(_flatten_json_value(value, next_prefix))
        return lines
    if obj is None:
        return lines
    text = str(obj).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return lines
    for part in text.split("\n"):
        part = part.strip()
        if not part:
            continue
        lines.append(f"{prefix}: {part}" if prefix else part)
    return lines


def json_payload_to_text(payload) -> str:
    if isinstance(payload, str):
        return payload.replace("\r\n", "\n").replace("\r", "\n").strip()
    if isinstance(payload, list):
        blocks: List[str] = []
        for idx, item in enumerate(payload, 1):
            inner = "\n".join(_flatten_json_value(item))
            header = f"--- Record {idx} ---"
            blocks.append(f"{header}\n{inner}" if inner else header)
        return "\n".join(blocks).strip()
    if isinstance(payload, dict):
        for key in ("records", "data", "complaints", "items"):
            nested = payload.get(key)
            if isinstance(nested, list):
                return json_payload_to_text(nested)
        return "\n".join(_flatten_json_value(payload)).strip()
    return str(payload).strip()


def read_json(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as fh:
        payload = json.load(fh)
    return json_payload_to_text(payload)


def read_csv(path: str) -> str:
    df = pd.read_csv(path, dtype=object, keep_default_na=False)
    if df.empty:
        return ""
    columns = [str(col) for col in df.columns]
    lines: List[str] = []
    for row_idx, row in df.iterrows():
        parts: List[str] = []
        for col in columns:
            value = (
                str(row[col])
                .replace("\r\n", " ")
                .replace("\r", " ")
                .replace("\n", " ")
                .strip()
            )
            if value:
                parts.append(f"{col}: {value}")
        if parts:
            lines.append(f"Record {int(row_idx) + 1} | " + " | ".join(parts))
    return "\n".join(lines)


def extract_text(path: str) -> Tuple[str, str]:
    ext = Path(path).suffix.lower()
    if ext == ".txt":
        return read_txt(path), "txt"
    elif ext in (".docx", ".doc"):
        return read_docx(path), "docx"
    elif ext in (".html", ".htm"):
        return read_html(path), "html"
    elif ext == ".json":
        return read_json(path), "json"
    elif ext == ".csv":
        return read_csv(path), "csv"
    else:
        try:
            return read_txt(path), "txt"
        except Exception:
            return "", "unknown"


def collect_files(input_path: str) -> List[str]:
    SUPPORTED = {".txt", ".docx", ".doc", ".html", ".htm", ".json", ".csv"}
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
# LINE-BY-LINE STRUCTURED PII SCANNER (Regex)
# ─────────────────────────────────────────────────────────────────────────────


# Boundaries for patterns that must also work flush against Devanagari: `\b`
# sees no boundary between a Hindi letter and a digit ("दिनांक12.12.2025").
_NB = r"(?<![A-Za-z0-9])"
_NA = r"(?![A-Za-z0-9])"
# Optional "No." style marker between a label and its value, English and Hindi.
_NUM_MARKER = r"(?i:No|Nos|Number|नंबर|नम्बर|नं[०0o]?|न[०0o]|संख्या|क्रमांक|सं[०0o]?|#)\.?"

_USER_ID_RE = re.compile(
    r"(?i:User\s*ID(?:\s*\([^)]+\))?|UserId|Username|User_ID|Portal\s*ID)"
    r"[_\-:=\s]+([a-zA-Z0-9_\-]+)"
)
# The value must hold a digit: under a blanket re.I the value class matched any
# word, so "PPP id belongs" hashed "belongs". With the digit required, any case
# is safe ("ppp id 6use5293").
_PPP_ID_RE = re.compile(
    r"(?i:PPP(?:\s*ID)?|Fa(?:i|mi)?mily\s*I\.?\s?D|Parivar\s*Pehchan\s*Patra"
    r"|परिवार\s*पहचान\s*पत्र|(?:फैमिली|फॅमिली|फेमिली|फ़ैमिली)\s*(?:आई\.?\s*डी|आईडी))"
    rf"\.?(?:\s*{_NUM_MARKER})?[\s:\-.()#]*"
    rf"((?=[A-Za-z0-9]*\d)[A-Za-z0-9]{{6,12}}){_NA}"
)
_PATIENT_ID_RE = re.compile(
    r"(?i:Patient\s*ID|UHID|Medical\s*UHID|Health\s*ID)[_\-:\s.]+"
    r"((?=[A-Za-z0-9_\-]*\d)[A-Za-z0-9_\-]+)"
)
_BANK_ACCOUNT_RE = re.compile(
    r"(?i:Bank\s*Account(?:\s*Number|\s*No)?|SBI\s*Account|Account\s*No|Account\s*Number"
    r"|A/c(?:\s*No)?|खाता(?:\s*(?:संख्या|सं|नं|नंबर)\.?)?)"
    rf"[_\-:\s.०]*(\d{{9,18}}){_NA}"
)
_AADHAAR_LABELLED_RE = re.compile(
    r"(?i:Aadhaa?r|Adhaa?r|UIDAI|आधार)(?:\s*(?i:card|कार्ड))?"
    rf"(?:\s*{_NUM_MARKER})?[\s:\-.]*"
    r"(\d{4}[\s\-]?\d{4}[\s\-]?\d{4})(?!\d)"
)
_AADHAAR_RE = re.compile(r"(?<!\d)(?<!\d[\s\-])[2-9]\d{3}([\s\-]?)\d{4}\1\d{4}(?!\d|[\s\-]\d)")
# Mobile: optional +91 / 0 prefix, digits grouped with spaces or hyphens
# ("80535-09009", "97293 91193", "98765 - 43210").
_PHONE_SEP = r"(?:\s?-\s?|\s)?"
# India's country code in any written form: +91, 0091, (+91), and for mobiles a
# bare 91. It is part of the match, so no "(+91)" or "0091" is left behind.
_COUNTRY_CODE = rf"(?:\(\s*)?(?:\+\s?|00)91(?:\s*\))?{_PHONE_SEP}"
# The 10 digits as Indian mobiles are written: unbroken, 5+5, 3+3+4 or 4+3+3.
# A separator allowed between any two digits also read "Section 91 1000 2000"
# as a number.
_MOBILE_DIGITS = (
    rf"[6-9](?:\d{{9}}|\d{{4}}{_PHONE_SEP}\d{{5}}"
    rf"|\d{{2}}{_PHONE_SEP}\d{{3}}{_PHONE_SEP}\d{{4}}|\d{{3}}{_PHONE_SEP}\d{{3}}{_PHONE_SEP}\d{{3}})"
)
_MOBILE_RE = re.compile(
    rf"(?<![\d+(])(?:{_COUNTRY_CODE}|91{_PHONE_SEP}|0)?{_MOBILE_DIGITS}(?!\d)"
)
# Landline: national number = STD code + subscriber number, 9-10 digits, written
# after a trunk 0 ("0172-2521617") or after the country code, with or without
# the 0 ("+91 172 2521617", "+91 0172 2521617"). A bare 91 is not accepted here.
_LANDLINE_RE = re.compile(
    rf"(?<![\d+(])(?:{_COUNTRY_CODE}0?|0)(?P<national>[1-9]\d{{1,3}}{_PHONE_SEP}\d{{6,8}})(?!\d)"
)

_MONTHS_EN = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
    r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)
_MONTHS_HI = (
    r"जनवरी|फरवरी|फ़रवरी|मार्च|अप्रैल|अप्रेल|मई|जून|जुलाई|अगस्त|सितंबर|सितम्बर"
    r"|अक्टूबर|अक्तूबर|नवंबर|नवम्बर|दिसंबर|दिसम्बर"
)
_DAY = r"(?:0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?"
_MONTH = rf"(?:(?i:{_MONTHS_EN})\.?(?![A-Za-z])|(?:{_MONTHS_HI})(?![\u0900-\u097F]))"
_DATE_RES = [
    # 12.12.2025, 09/04/2026, 25-3-2026
    re.compile(
        r"(?<![\d.\/\-])(?:0?[1-9]|[12]\d|3[01])[\/\-.](?:0?[1-9]|1[0-2])[\/\-.](?:19|20)\d{2}(?!\d)"
    ),
    # 25.03.26 — same separator twice, so "1.2-3" style ranges are not dates
    re.compile(
        r"(?<![\d.\/\-])(?:0?[1-9]|[12]\d|3[01])([\/\-.])(?:0?[1-9]|1[0-2])\1\d{2}(?![\d.\/\-]?\d)"
    ),
    # 25 Feb 2025, 1 अप्रैल 2025, 3rd of March, 2024
    re.compile(rf"(?<!\d){_DAY}[\s\-.,/]*(?:of\s+)?{_MONTH}[\s\-.,/]*(?:19|20)\d{{2}}(?!\d)"),
    # Feb 25, 2025
    re.compile(rf"{_NB}{_MONTH}\s*{_DAY},?\s*(?:19|20)\d{{2}}(?!\d)"),
]

_OTHER_ID_RES = [
    ("GSTIN", re.compile(rf"{_NB}\d{{2}}[A-Z]{{5}}\d{{4}}[A-Z][1-9A-Z]Z[0-9A-Z]{_NA}")),
    ("ABHA", re.compile(r"(?<![\d\-])\d{2}-\d{4}-\d{4}-\d{4}(?![\d\-])")),
    # handle@bank: an '@' with no dot after it, so it is not an email address
    ("UPI_ID", re.compile(r"(?<![\w.\-])[A-Za-z0-9.\-_]{2,64}@[A-Za-z]{2,32}(?![\w.\-@])")),
]

# Catch-all for identifiers no named rule covers (family/property/meter/service
# numbers, case IDs, ...): a standalone run of 6-24 capitals and digits holding a
# digit, plus either a letter or 7+ digits. Words never contain digits, and the
# numbers prose does contain (amounts, years, section numbers) are short or
# punctuated, so neither shape passes.
# Codes the general ID rule would otherwise hash that identify no one: financial
# years (FY2025) and highway numbers (NH709A, SH20).
_PUBLIC_CODE = re.compile(r"(?:FY|NH|SH|MDR)\d{1,4}[A-Z]?")
_GENERIC_ID_RE = re.compile(
    rf"{_NB}(?=[A-Z0-9]*\d)(?=\d*[A-Z]|\d{{7}})[A-Z0-9]{{6,24}}(?![a-z0-9])"
    # not a system attachment name ("CRFBD17746178970.pdf"): hashing it would
    # break the link to the file, and the named rules above still scan it
    r"(?!\.(?i:pdf|jpe?g|png|gif|tiff?|docx?|xlsx?|csv|txt|zip)\b)"
)
# Slash-structured reference codes: CMOFF/J/2026/003159, DEV/2026-2027/322.
_SLASH_REFERENCE_RE = re.compile(
    r"(?<![A-Za-z0-9/])(?=[A-Z0-9/\-]*\d)[A-Z]{2,}(?:/[A-Z0-9\-]+){2,}(?![A-Za-z0-9/])"
)

# Labels whose number identifies a person's case, land, property or connection.
# A dataset config can replace either list (free_text_anonymization.reference_keywords /
# strong_reference_keywords); court and letter numbers are public references and
# are left off by default.
#
# Strong labels are always followed by their number, so the "No." marker is
# optional after them ("FIR 140", "मुकदमा 61"). Weak labels are ordinary words
# too ("case", "house"), so they need the marker ("Case No. 78") or a number/year
# shape ("case 78/2025"), or "in case 2 people" would be hashed.
DEFAULT_STRONG_REFERENCE_KEYWORDS: Tuple[str, ...] = (
    r"F\.?\s?I\.?\s?R", r"एफ\.?\s?आई\.?\s?आर", r"मुकदमा", r"मुकद्दमा", r"अभियोग",
    r"khasra", r"खसरा", r"khewat", r"खेवट", r"khat(?:a|au|o)ni", r"खतौनी", r"इंतकाल",
    r"I\.?\s?D", r"आई\.?\s?डी", r"आईडी",
)
DEFAULT_REFERENCE_KEYWORDS: Tuple[str, ...] = (
    r"case", r"complaint", r"शिकायत", r"परिवाद", r"pariwad", r"DDR", r"रपट",
    r"killa", r"किला", r"mutation", r"house", r"H", r"मकान",
    r"plot", r"प्लॉट", r"DSS", r"property(?:\s*ID)?", r"प्रॉपर्टी(?:\s*आईडी)?",
    r"meter", r"मीटर", r"connection", r"कनेक्शन", r"consumer", r"उपभोक्ता",
    r"account", r"खाता", r"registration", r"पंजीकरण", r"पंजी", r"service", r"सर्विस",
    r"ration\s*card", r"राशन\s*कार्ड", r"licen[cs]e", r"लाइसेंस", r"challan", r"चालान",
    r"PPO", r"pension", r"पेंशन", r"पेंसन", r"roll", r"रोल", r"vehicle", r"वाहन",
    r"receipt", r"रसीद", r"application", r"आवेदन",
)
_NEVER_MATCH = re.compile(r"(?!x)x")


# A reference value: letters, digits, '/' and '-', holding at least one digit.
# Any case: after a label, a digit is enough to rule out ordinary words.
_REF_VALUE = r"(?=[A-Za-z0-9/\-]*\d)[A-Za-z0-9](?:[A-Za-z0-9/\-]*[A-Za-z0-9])?"
# Further numbers of a list after the first: "No. 140 & 141", "No 589, 526".
_REF_CONTINUATION = re.compile(
    rf"\s*(?:,|&|\band\b|\bto\b|व|तथा|एवं|से)\s*(?P<value>{_REF_VALUE}){_NA}"
)
_YEAR = re.compile(r"(?:19|20)\d{2}")


@lru_cache(maxsize=32)
def _reference_number_re(
    keywords: Tuple[str, ...], strong_keywords: Tuple[str, ...]
) -> "re.Pattern[str]":
    """Label + optional "No." marker + number, e.g. "FIR No. 0133/2026"."""
    groups = []
    if strong_keywords:
        groups.append("(?P<strong>" + "|".join(f"(?:{k})" for k in strong_keywords) + ")")
    if keywords:
        groups.append("(?P<weak>" + "|".join(f"(?:{k})" for k in keywords) + ")")
    if not groups:
        return _NEVER_MATCH
    return re.compile(
        rf"(?<![A-Za-z\u0900-\u097F])(?i:{'|'.join(groups)})(?![A-Za-z])\.?"
        rf"(?:\s*(?P<marker>{_NUM_MARKER}))?[\s:\-–#.]*"
        rf"(?P<value>{_REF_VALUE}){_NA}"
    )


def _reference_spans(line: str, pattern: "re.Pattern[str]") -> List[Tuple[int, int]]:
    """(start, end) of every reference value in `line`, list continuations included."""
    spans: List[Tuple[int, int]] = []
    for m in pattern.finditer(line):
        value = m.group("value")
        if not m.group("marker") and "/" not in value:
            # No marker: a weak label needs a number/year shape, and a strong
            # one at least two characters ("FIR 140", not "FIR 2 baar").
            if (m.groupdict().get("strong") is None) or len(value) < 2:
                continue
        spans.append(m.span("value"))
        pos = m.end("value")
        while (cont := _REF_CONTINUATION.match(line, pos)) is not None:
            if _YEAR.fullmatch(cont.group("value")):
                break  # "FIR No. 140, 2025": the year, not a second FIR
            spans.append(cont.span("value"))
            pos = cont.end("value")
    return spans


_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
# Money is not an identifier: digits right after Rs/₹/रुपये or before "/-".
_AMOUNT_BEFORE = re.compile(r"(?i:rs\.?|inr|₹|रु\.?|रू\.?|रुपये|रूपये|रुपए)\s*$")
_AMOUNT_AFTER = re.compile(r"^\s*(?:/-|(?i:rupees|lakh|lac|crore)\b|रुपये|रूपये|लाख|करोड़)")


_VERHOEFF_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6), (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8), (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2), (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4), (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_VERHOEFF_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2), (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0), (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5), (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)


def _verhoeff_valid(number: str) -> bool:
    """Aadhaar's check digit (Verhoeff), over the digits of `number`."""
    check = 0
    for i, ch in enumerate(reversed(re.sub(r"\D", "", number))):
        check = _VERHOEFF_D[check][_VERHOEFF_P[i % 8][int(ch)]]
    return check == 0


def _add_span_hit(
    hits: List[PiiHit], source: str, label: str, line_str: str, lno: int, start: int, end: int
):
    hits.append(
        PiiHit(
            source,
            label,
            line_str[start:end],
            PII_DISPLAY_NAMES.get(label, label.replace("_", " ")),
            lno,
            get_word_number(line_str, start),
            start + 1,
            end,
        )
    )


def _add_hit(hits: List[PiiHit], source: str, label: str, line_str: str, lno: int, m, group=0):
    hits.append(
        PiiHit(
            source,
            label,
            m.group(group).strip(),
            PII_DISPLAY_NAMES.get(label, label.replace("_", " ")),
            lno,
            get_word_number(line_str, m.start(group)),
            m.start(group) + 1,
            m.end(group),
        )
    )


@lru_cache(maxsize=32)
def _deny_term_re(terms: Tuple[str, ...]) -> Optional["re.Pattern[str]"]:
    """Any of `terms`, any case, with any run of spaces between its words."""
    parts = [r"\s+".join(re.escape(w) for w in t.split()) for t in terms if t.strip()]
    return re.compile("|".join(parts), re.I) if parts else None



def scan_text_line_by_line(
    text: str,
    reference_keywords: Optional[Sequence[str]] = None,
    strong_reference_keywords: Optional[Sequence[str]] = None,
    deny_terms: Optional[Sequence[str]] = None,
) -> List[PiiHit]:
    hits: List[PiiHit] = []
    lines = text.splitlines()
    deny_re = _deny_term_re(tuple(deny_terms or ()))
    reference_re = _reference_number_re(
        tuple(DEFAULT_REFERENCE_KEYWORDS if reference_keywords is None else reference_keywords),
        tuple(
            DEFAULT_STRONG_REFERENCE_KEYWORDS
            if strong_reference_keywords is None
            else strong_reference_keywords
        ),
    )

    for line_idx, line in enumerate(lines):
        # Devanagari digits ("९८७६५४३२१०") read as ASCII for every pattern below.
        # One character for one, so hit offsets still index the original text.
        line_str = line.strip().translate(_DEVANAGARI_DIGITS)
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
            r"\b[A-Z]{2}[\s\-]?\d{2}[\s\-]?[A-Z]{1,2}[\s\-]?\d{4}\b", line_str, re.I
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

        for m in _USER_ID_RE.finditer(line_str):
            # A username may be all letters, so a digit-free value is accepted
            # only when an explicit ':' / '=' marks it as the field's value.
            if re.search(r"\d", m.group(1)) or re.search(r"[:=]", m.group(0)):
                _add_hit(hits, "Contextual Regex", "User_ID", line_str, lno, m, 1)

        for m in _PPP_ID_RE.finditer(line_str):
            _add_hit(hits, "Contextual Regex", "PPP_ID", line_str, lno, m, 1)

        for m in _PATIENT_ID_RE.finditer(line_str):
            _add_hit(hits, "Contextual Regex", "Patient_ID", line_str, lno, m, 1)

        for m in _BANK_ACCOUNT_RE.finditer(line_str):
            _add_hit(hits, "Contextual Regex", "Bank_Account", line_str, lno, m, 1)

        for m in _AADHAAR_LABELLED_RE.finditer(line_str):
            _add_hit(hits, "Contextual Regex", "Aadhaar", line_str, lno, m, 1)

        for m in _AADHAAR_RE.finditer(line_str):
            # Unseparated, a 12-digit run is as likely an account or reference
            # number, so it must also pass Aadhaar's Verhoeff check digit.
            if m.group(1) or _verhoeff_valid(m.group(0)):
                _add_hit(hits, "Regex", "Aadhaar", line_str, lno, m)

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

        for m in _MOBILE_RE.finditer(line_str):
            _add_hit(hits, "Regex", "Phone_Number", line_str, lno, m)

        for m in _LANDLINE_RE.finditer(line_str):
            if len(re.sub(r"\D", "", m.group("national"))) in (9, 10):
                _add_hit(hits, "Regex", "Phone_Number", line_str, lno, m)

        for pattern in _DATE_RES:
            for m in pattern.finditer(line_str):
                _add_hit(hits, "Regex", "Date", line_str, lno, m)

        for label, pattern in _OTHER_ID_RES:
            for m in pattern.finditer(line_str):
                _add_hit(hits, "Regex", label, line_str, lno, m)

        # Names a reviewer confirmed the models miss: always PERSON, whole words only
        # (so "Jagdish" does not hit inside "Jagdishpur").
        if deny_re is not None:
            for m in deny_re.finditer(line_str):
                s0, e0 = m.span()
                if (s0 > 0 and _is_word_char(line_str[s0 - 1])) or (
                    e0 < len(line_str) and _is_word_char(line_str[e0])
                ):
                    continue
                _add_hit(hits, "Deny list", "PERSON", line_str, lno, m)

        for start, end in _reference_spans(line_str, reference_re):
            _add_span_hit(hits, "Contextual Regex", "Reference_Number", line_str, lno, start, end)

        # Last: any standalone ID-shaped token the rules above did not name. It
        # overlaps their hits on purpose; _sanitize keeps the first (specific) label.
        for pattern in (_GENERIC_ID_RE, _SLASH_REFERENCE_RE):
            for m in pattern.finditer(line_str):
                if _PUBLIC_CODE.fullmatch(m.group(0)):
                    continue
                if m.group(0).isdigit() and (
                    _AMOUNT_BEFORE.search(line_str[max(0, m.start() - 12) : m.start()])
                    or _AMOUNT_AFTER.match(line_str[m.end() :])
                ):
                    continue  # "Rs 3500000", "3500000/-"
                _add_hit(hits, "Regex", "ID_Number", line_str, lno, m)

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


def full_pii_scan(
    text: str,
    reference_keywords: Optional[Sequence[str]] = None,
    strong_reference_keywords: Optional[Sequence[str]] = None,
    deny_terms: Optional[Sequence[str]] = None,
) -> List[PiiHit]:
    """Returns regex-based PII hits only. Presidio removed."""
    return scan_text_line_by_line(
        text, reference_keywords, strong_reference_keywords, deny_terms
    )


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

# Model imports & PyTorch CPU thread configuration

try:
    import torch as _torch
    # Limit PyTorch CPU thread contention when running multiple parallel model workers
    if hasattr(_torch, "set_num_threads"):
        _torch.set_num_threads(min(4, os.cpu_count() or 4))

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

# GPU check (COMMENTED OUT - forced CPU execution mode)
# _NER_DEVICE = -1
# if _TRANSFORMERS_AVAILABLE and _torch.cuda.is_available():
#     try:
#         _test_t = _torch.zeros(1, device="cuda:0")
#         del _test_t
#         if _torch.cuda.is_available():
#             _torch.cuda.empty_cache()
#         _NER_DEVICE = 0
#     except Exception as _cuda_err:
#         print(f"  [NER] GPU check failed ({_cuda_err}). Falling back to CPU mode.", flush=True)
#         _NER_DEVICE = -1

# _NER_DEVICE_STR = "GPU" if _NER_DEVICE == 0 else "CPU"
_NER_DEVICE = -1
_NER_DEVICE_STR = "CPU"

# # Concurrent CUDA calls from multiple model threads are unsafe; CPU can overlap.
# _NER_INFER_LOCK = threading.Lock() if _NER_DEVICE == 0 else None
_NER_INFER_LOCK = None

_NER_PIPELINE_CACHE: Dict[str, object] = {}

# Dynamic INT8 quantization of the Linear layers. Faster, but on a reference corpus
# it recovered 386 entities where fp32 found 481, with only 290 shared — for a
# redaction pipeline that is a fifth of the names and locations going unmasked,
# so it stays off unless a deployment has measured the trade-off on its own data.
_NER_QUANTIZE = os.getenv("NER_QUANTIZE", "").strip().lower() in {"1", "true", "yes"}


def _maybe_quantize(model, model_id: str):
    if not _NER_QUANTIZE or _NER_DEVICE != -1:
        return model
    try:
        quantized = _torch.ao.quantization.quantize_dynamic(
            model, {_torch.nn.Linear}, dtype=_torch.qint8
        )
        print(f"  [NER] Quantized {model_id} to INT8 (dynamic).", flush=True)
        return quantized
    except Exception as exc:
        print(f"  [NER] INT8 quantization unavailable for {model_id}: {exc}", flush=True)
        return model


def _load_ner_pipeline(model_id: str, use_fast: bool = True) -> Optional[object]:
    if model_id in _NER_PIPELINE_CACHE:
        return _NER_PIPELINE_CACHE[model_id]
    if not _TRANSFORMERS_AVAILABLE:
        return None
    try:
        hf_token = os.getenv("HF_TOKEN") or None
        # Cache first: an online load still probes the Hub (e.g. for a
        # model.safetensors that IndicNER doesn't ship), and on a gated repo
        # without a token that probe 401s and aborts the load even though every
        # file is already cached. Only go to the network on a real cache miss.
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, use_fast=use_fast, local_files_only=True
            )
            model = AutoModelForTokenClassification.from_pretrained(
                model_id, local_files_only=True
            )
        except OSError:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                token=hf_token,
                use_fast=use_fast,
            )
            model = AutoModelForTokenClassification.from_pretrained(
                model_id,
                token=hf_token,
            )
        # transformers warns that Davlan/xlm-roberta-base-ner-hrl loads "with an
        # incorrect regex pattern" and suggests fix_mistral_regex=True. That is a
        # false alarm for this model: as loaded, its tokens matched the reference
        # SentencePiece tokenizer on 3,000 of 3,000 real grievance lines, while
        # the suggested flag matched only 569. Do not set it.
        model.eval()
        model = _maybe_quantize(model, model_id)
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
        if _NER_DEVICE == 0:
            print(
                f"  [NER] GPU load failed for {model_id}: {exc}. Retrying on CPU...",
                flush=True,
            )
            try:
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
                ner_pipe = _hf_pipeline(
                    task="ner",
                    model=model,
                    tokenizer=tokenizer,
                    aggregation_strategy="simple",
                    device=-1,
                )
                _NER_PIPELINE_CACHE[model_id] = ner_pipe
                return ner_pipe
            except Exception as exc2:
                print(f"  [NER] CPU load failed for {model_id}: {exc2}", flush=True)
                _NER_PIPELINE_CACHE[model_id] = None
                return None
        print(f"  [NER] Could not load {model_id}: {exc}", flush=True)
        _NER_PIPELINE_CACHE[model_id] = None
        return None


def _is_word_char(ch: str) -> bool:
    """Letters and digits, plus the combining marks that complete Indic letters.

    Devanagari vowel signs, virama, nukta and anusvara (Unicode Mn/Mc) are not
    str.isalnum(), so treating them as boundaries cut words mid-letter: a span
    ending at "उर" in "उर्फ" was redacted to "*्फ", leaving an orphaned sign that
    renders as a dotted circle. ZWJ/ZWNJ shape conjuncts and belong to the word too.
    """
    return ch.isalnum() or ch in "\u200c\u200d" or unicodedata.category(ch) in ("Mn", "Mc")


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
        if _is_word_char(prev):
            start -= 1
        elif (
            prev == "."
            and start > 1
            and text[start - 2].isalpha()
            and (
                start == 2
                or text[start - 3] in STRICT_DELIMITERS
                or text[start - 3].isspace()
                # chains of single letters: a span starting at "A" of "C.I.A"
                # used to stop there and leave "C.I.*"
                or text[start - 3] == "."
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
        if _is_word_char(nxt):
            end += 1
        elif nxt == "." and end + 1 < len(text) and text[end + 1].isalnum():
            end += 1
        else:
            break

    return start, end


def _clean_entity_text(text: str, start: int, end: int) -> Tuple[str, int, int]:
    """
    Cleans leading and trailing punctuation, isolated single characters,
    and slash-prefixes (e.g., 'o.Name', 'S/o Name', 'Name s') from NER spans.
    """
    val = text[start:end]

    # 1. Strip leading punctuation, slashes, and dangling prefix letters like "o.", "s.", "d."
    #    Only relation markers (S/o, D/o, W/o, R/o, H/o; dotted: s.o.) are stripped as prefixes:
    #    a looser "letter/letter" form ate "R/S" from "R/Sir", leaving "ir".
    #    A letter-dot followed by another letter-dot is an abbreviation ("C.I.A"),
    #    not a stray initial, and is kept.
    match_prefix = re.match(
        r"^([\s:\-.,'\"/()]+|[A-Za-z]/[oO](?:\.\s*|\s+)|[A-Za-z]\.[oO]\.\s*|[a-zA-Z]\.(?![A-Za-z]\.)\s*)", val
    )
    while match_prefix and match_prefix.end() > 0:
        cut = match_prefix.end()
        if cut >= len(val):
            break
        start += cut
        val = text[start:end]
        match_prefix = re.match(
            r"^([\s:\-.,'\"/()]+|[A-Za-z]/[oO](?:\.\s*|\s+)|[A-Za-z]\.[oO]\.\s*|[a-zA-Z]\.(?![A-Za-z]\.)\s*)", val
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
                "models": {s["model"]} if s.get("model") else set(),
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
            same_cat_merged.append({**s, "models": set(s["models"])})
        else:
            target["models"] |= s["models"]
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

    # 2. Resolve overlaps between categories: one span covering the union, with the
    #    category and score of the most confident member. Keeping only the longest
    #    span dropped the rest outright, so a 0.46 ORG "sh Jag" (IndicNER) erased
    #    a 0.90 PERSON "Jag" (XLM-R) and the name fell under the confidence floor;
    #    and the uncovered part of a dropped span ("kailash" when LOC "reddy nagar"
    #    won over PERSON "kailash reddy") went unredacted.
    same_cat_merged.sort(key=lambda x: (x["start"], -x["end"]))
    groups: List[List[Dict]] = []
    for s in same_cat_merged:
        if groups and s["start"] < max(g["end"] for g in groups[-1]):
            groups[-1].append(s)
        else:
            groups.append([s])

    final_merged: List[Dict] = []
    for group in groups:
        best = max(group, key=lambda x: (x["score"], x["end"] - x["start"]))
        models = set().union(*(g["models"] for g in group))
        if len(group) == 1:
            final_merged.append(best)
            continue
        text_val, st, en = _clean_entity_text(
            original_line, min(g["start"] for g in group), max(g["end"] for g in group)
        )
        final_merged.append(
            {
                **best,
                "start": st,
                "end": en,
                "text": text_val,
                "word_no": get_word_number(original_line, st),
                "start_char": st + 1,
                "end_char": en,
                "models": models,
            }
        )

    entities: List[NerEntity] = []
    seen = set()
    for m in final_merged:
        clean_val = m["text"]
        if clean_val and len(clean_val) >= 2:
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
                        models=sorted(m["models"]),
                    )
                )

    return entities


# ─────────────────────────────────────────────────────────────────────────────
# NER FALSE-POSITIVE FILTERS
# ─────────────────────────────────────────────────────────────────────────────
# Each rule removes only what cannot be personal data, and every one was checked
# against names that look like the junk it targets: "Shree"/"श्री"/"Sahab" are
# names too, so they are not titles here; "Lucky", "Sunny", "Singh" and "Ram" are
# common English words, so the dictionary rule never touches a capitalised or
# PERSON entity; "Model Town" is all English, so LOCATION is never filtered.

# Forms of address that are never a name on their own.
NEVER_NAME_TITLES = frozenset(
    {"sir", "madam", "mam", "maam", "mr", "mrs", "ms", "smt", "dr", "respected", "r",
     "श्रीमान", "श्रीमती", "महोदय", "महोदया", "मान्यवर"}
)
# Hindi function words spaCy's Hindi stopword list leaves out.
_HI_FUNCTION_WORDS = frozenset({"हेतु", "हेतू", "बारे", "बाबत", "संबंध", "सम्बन्ध", "द्वारा"})
# English zipf frequency (wordfreq) above which a word counts as everyday vocabulary.
_COMMON_WORD_ZIPF = 3.0
# Casing evidence for the lowercase-PERSON rule: a word must have been seen at
# least this often mid-sentence, and lowercase at least this share of the time.
_CASING_MIN_SEEN = 3
_CASING_LOWER_SHARE = 0.9

# HARDCODED, temporary: everyday words the general rules deliberately do not
# clear, exempted by name until a general rule covers them. Matched like
# `ner_allow_terms` (whole entity, case/spelling-insensitive), in every dataset.
#   "Milk" — tagged LOCATION in "the said Milk Booth" (issue #4). Rule 3 never
#            filters LOCATION, because localities are often plain English
#            ("Model Town", "Civil Lines").
#   "Mark" — tagged PERSON for the verb in "Complaint is Mark to CIA-3" (#17).
#            Rule 3 never filters a capitalised name, because Indian names are
#            often English words ("Lucky", "Sunny").
#   "kindly undetake" — tagged PERSON (issue #14); "undetake" is a typo, so
#            neither the vocabulary rule nor any dictionary recognises it.
# Cost: a person or place called just "Mark" or "Milk" is no longer redacted.
# Remove an entry once a general rule handles it.
HARDCODED_NER_ALLOW_TERMS: Tuple[str, ...] = ("Milk", "Mark", "kindly undetake")

try:
    from spacy.lang.en.stop_words import STOP_WORDS as _EN_STOP
    from spacy.lang.hi.stop_words import STOP_WORDS as _HI_STOP
except ImportError:  # the rule then only knows the Hindi extras above
    _EN_STOP, _HI_STOP = frozenset(), frozenset()

try:
    from wordfreq import zipf_frequency as _zipf
except ImportError:
    _zipf = None

_TOKEN_SPLIT = re.compile(r"[\s/.,()\-]+")
_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z']*")


def _entity_tokens(text: str) -> List[str]:
    return [t for t in _TOKEN_SPLIT.split(text) if t]


def ner_term_key(term: str) -> str:
    """Spelling-insensitive key for allow-list matching.

    Case, spacing and punctuation are ignored, and common romanisation and
    Devanagari spelling variants fold together, so one entry covers
    "Jan Samvaad" / "Jansamvaad" / "jan samwad" / "Jan Sanwaad", and
    "सी.एम.विंडो" / "सीएम विन्डो" / "सी.एम.विन्ङो".
    """
    key = unicodedata.normalize("NFC", term).lower()
    key = re.sub(r"[^\wऀ-ॿ]+|_", "", key)
    key = key.replace("w", "v")
    key = re.sub(r"[nm](?=v)", "m", key)          # sanwaad / samvaad
    key = re.sub(r"([aeiou])\1+", r"\1", key)      # aa -> a, ee -> e
    key = key.replace("़", "")                # nukta
    key = re.sub(r"[नमङञण]्(?=[क-ह])", "ं", key)  # न्ड -> ंड
    key = key.replace("ङ", "ड").replace("ँ", "ं")            # ङ typo, chandrabindu
    return key


def word_casing_stats(texts: Sequence[str]) -> Dict[str, List[int]]:
    """{lowercased word: [times written lowercase, times capitalised]} mid-sentence.

    Only Latin words in lines that are not wholly upper- or lowercase count, and
    never the first word of a sentence, where capitals carry no information.
    """
    stats: Dict[str, List[int]] = {}
    for text in texts:
        for line in text.splitlines():
            if line.isupper() or line.islower():
                continue
            for sentence in re.split(r"[.!?।\n]+", line):
                words = _LATIN_WORD.findall(sentence)
                for word in words[1:]:
                    if word.isupper():
                        continue  # acronyms say nothing about name-ness
                    entry = stats.setdefault(word.lower(), [0, 0])
                    entry[0 if word.islower() else 1] += 1
    return stats


_HARDCODED_ALLOW_KEYS = frozenset(ner_term_key(t) for t in HARDCODED_NER_ALLOW_TERMS)


def ner_false_positive_reason(
    entity: NerEntity,
    line: str,
    allow_keys: frozenset = frozenset(),
    casing: Optional[Dict[str, List[int]]] = None,
) -> Optional[str]:
    """Why `entity` is not personal data, or None to keep it redacted."""
    tokens = _entity_tokens(entity.text)
    if not tokens:
        return None
    lowered = [t.lower() for t in tokens]

    # Rule 4: a curated, dataset-specific term, matched as the whole entity only,
    # so allow-listing "Jan Samvaad" never exempts "Jan Samvaad Ramesh".
    key = ner_term_key(entity.text)
    if allow_keys and key in allow_keys:
        return "allow_list"
    if key in _HARDCODED_ALLOW_KEYS:
        return "hardcoded_allow"

    # Rule 1: nothing but forms of address ("SIR", "R/Sir", "श्रीमान").
    if all(t in NEVER_NAME_TITLES for t in lowered):
        return "title_only"

    single_model = len(entity.models) <= 1
    latin = all(_LATIN_WORD.fullmatch(t) for t in tokens)
    latin_lower = latin and all(t.islower() for t in tokens)
    devanagari = not latin and all(re.fullmatch(r"[ऀ-ॿ]+", t) for t in tokens)

    # Rule 2: function words, when written lowercase or in Hindi ("us", "हेतू")
    # and found by one model; a capitalised "May" or "Will" may be a name.
    if single_model and (latin_lower or devanagari) and all(
        t in _EN_STOP or t in _HI_STOP or t in _HI_FUNCTION_WORDS for t in lowered
    ):
        return "function_word"

    if not single_model or not latin:
        return None

    # Rule 3 needs this dataset's own casing evidence: a word written lowercase
    # mid-sentence nearly every time it appears is vocabulary, not a name. That is
    # what separates "afresh" or "department" from "Ram", "Singh" or "Lucky",
    # which an English word list scores as just as common. A word seen too rarely
    # to judge keeps the entity redacted.
    if not casing or line.islower() or line.isupper():
        return None

    def mostly_lowercase(word: str) -> bool:
        lower_n, upper_n = casing.get(word, (0, 0))
        seen = lower_n + upper_n
        return seen >= _CASING_MIN_SEEN and lower_n / seen >= _CASING_LOWER_SHARE

    # Rule 3a: an ORGANIZATION made only of everyday English words ("concerned
    # department", "Pls", "Head Office"). "Ram Traders" survives: "ram" is
    # common English but this data writes it capitalised.
    if entity.category == "ORGANIZATION" and _zipf is not None and all(
        _zipf(t, "en") >= _COMMON_WORD_ZIPF and mostly_lowercase(t) for t in lowered
    ):
        return "common_words_org"

    # Rule 3b: a PERSON written lowercase in normal-case text, made of words this
    # dataset almost always writes lowercase ("afresh", "challan", "complainant").
    if entity.category == "PERSON" and latin_lower and all(mostly_lowercase(t) for t in lowered):
        return "lowercase_common_word"

    return None


def _parse_ner_items(
    results,
    line_str: str,
    min_score: float = 0.20,
    model_name: str = "",
) -> List[Dict]:
    if not results:
        return []

    raw_spans = []
    for item in results:
        if not isinstance(item, dict):
            continue
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


def _run_hf_pipe(pipe, texts, batch_size: Optional[int] = None):
    """Run a HuggingFace NER pipe with torch.inference_mode() for optimized CPU execution.

    A HuggingFace pipeline handed a list still defaults to batch_size=1, i.e. one
    forward pass per item — passing the list alone buys nothing. `batch_size` opts
    into real tensor batching; leave it None to keep the one-at-a-time behaviour.
    """
    kwargs = {} if batch_size is None else {"batch_size": batch_size}

    if _TRANSFORMERS_AVAILABLE and hasattr(_torch, "inference_mode"):
        with _torch.inference_mode():
            if _NER_INFER_LOCK is not None:
                with _NER_INFER_LOCK:
                    return pipe(texts, **kwargs)
            return pipe(texts, **kwargs)

    if _NER_INFER_LOCK is not None:
        with _NER_INFER_LOCK:
            return pipe(texts, **kwargs)
    return pipe(texts, **kwargs)


def _extract_raw_spans(
    pipe,
    line_str: str,
    min_score: float = 0.20,
    model_name: str = "",
) -> List[Dict]:
    if pipe is None or not line_str.strip():
        return []

    try:
        results = _run_hf_pipe(pipe, line_str)
    except Exception:
        return []

    return _parse_ner_items(results, line_str, min_score=min_score, model_name=model_name)


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
    Single-pass verbatim inference only.

    The original dual-pass (verbatim + truecased) was removed because truecasing
    ALL-CAPS text (e.g. "WATER PIPE LINE LEAKAGE" -> "Water Pipe Line Leakage")
    causes IndicNER to spuriously tag infrastructure and complaint phrases as LOC/ORG.
    IndicNER's tokenizer has do_lower_case=True, so casing is invisible anyway and
    the truecased pass added false positives without improving true recall.
    """
    return _extract_line_spans_chunked(
        pipe, line_str, min_score=min_score, model_name=model_name
    )


# ─────────────────────────────────────────────────────────────────────────────
# QUEUE-DRIVEN PARALLEL NER (fan-out / fan-in)
# ─────────────────────────────────────────────────────────────────────────────


def _normalize_pipe_batch_output(raw, batch_len: int) -> List:
    """HuggingFace returns a flat entity list for one string, nested lists for a batch."""
    if batch_len <= 0:
        return []
    if batch_len == 1:
        if raw is None:
            return [[]]
        if isinstance(raw, list) and (not raw or isinstance(raw[0], dict)):
            return [raw]
        if isinstance(raw, list) and isinstance(raw[0], list):
            return raw
        return [[]]
    if not isinstance(raw, list):
        return [[] for _ in range(batch_len)]
    if len(raw) != batch_len:
        return [[] for _ in range(batch_len)]
    return raw


def _infer_line_batch(
    pipe,
    tasks: List[LineTask],
    min_score: float,
    model_name: str,
    batch_size: Optional[int] = None,
) -> List[ModelLineResult]:
    """Mini-batch short lines through the pipe; chunk long lines individually."""
    results: List[ModelLineResult] = []
    if not tasks:
        return results
    if pipe is None:
        return [
            ModelLineResult(t.doc_id, t.line_no, model_name, []) for t in tasks
        ]

    tokenizer = pipe.tokenizer
    max_tokens = 380
    short_tasks: List[LineTask] = []
    long_tasks: List[LineTask] = []
    for task in tasks:
        # Skip the tokenizer only when the line provably fits: a subword tokenizer
        # emits at most one token per character plus one word-boundary piece per
        # word, plus the special tokens. The old "<= 1200 chars OR <= 180 words"
        # guess let lines of 400-1200 tokens through as short; the pipe then
        # truncated them at 512 (dropping any PII past that point) or, for HiNER,
        # whose tokenizer sets no max length, crashed the whole batch into the
        # slow one-line-at-a-time fallback.
        if len(task.line_text) + len(task.line_text.split()) + 2 <= max_tokens:
            short_tasks.append(task)
        else:
            try:
                token_count = len(
                    tokenizer(task.line_text, add_special_tokens=True)["input_ids"]
                )
            except Exception:
                token_count = max_tokens + 1
            if token_count <= max_tokens:
                short_tasks.append(task)
            else:
                long_tasks.append(task)

    span_map: Dict[Tuple[int, int], List[Dict]] = {}
    if short_tasks:
        batch_requests = []
        for task in short_tasks:
            text = task.line_text
            # Truecasing removed: converting ALL-CAPS -> Title Case causes infrastructure
            # phrases ("WATER PIPE LINE LEAKAGE") to be spuriously tagged as LOC/ORG.
            batch_requests.append((task, False, text))

        texts = [req[2] for req in batch_requests]
        try:
            raw = _run_hf_pipe(pipe, texts, batch_size=batch_size)
            batched = _normalize_pipe_batch_output(raw, len(texts))
            for (task, is_tc, text_str), items in zip(batch_requests, batched):
                spans = _parse_ner_items(
                    items,
                    text_str,
                    min_score=min_score,
                    model_name=model_name,
                )
                key = (task.doc_id, task.line_no)
                span_map.setdefault(key, []).extend(spans)
        except Exception:
            for task in short_tasks:
                span_map[(task.doc_id, task.line_no)] = extract_dual_pass_ner_spans(
                    pipe,
                    task.line_text,
                    min_score=min_score,
                    model_name=model_name,
                )

    for task in long_tasks:
        span_map[(task.doc_id, task.line_no)] = extract_dual_pass_ner_spans(
            pipe,
            task.line_text,
            min_score=min_score,
            model_name=model_name,
        )

    for task in tasks:
        results.append(
            ModelLineResult(
                doc_id=task.doc_id,
                line_no=task.line_no,
                model_name=model_name,
                spans=span_map.get((task.doc_id, task.line_no), []),
            )
        )
    return results


def _ner_model_worker(
    model_name: str,
    pipe,
    min_score: float,
    in_queue: Queue,
    out_queue: Queue,
    batch_size: int,
):
    """Persistent worker: consumes Q_H / Q_I / Q_X and writes (doc, line, entities) to Q_M."""
    batch: List[LineTask] = []

    def flush():
        if not batch:
            return
        try:
            for item in _infer_line_batch(pipe, list(batch), min_score, model_name):
                out_queue.put(item)
        except Exception as exc:
            for task in batch:
                out_queue.put(
                    ModelLineResult(
                        doc_id=task.doc_id,
                        line_no=task.line_no,
                        model_name=model_name,
                        spans=[],
                        error=str(exc),
                    )
                )
        batch.clear()

    while True:
        try:
            item = in_queue.get(timeout=0.2)
        except Empty:
            flush()
            continue

        if item is None:
            flush()
            out_queue.put(None)
            return

        batch.append(item)
        if len(batch) >= batch_size:
            flush()


def _store_line_ner_results(
    rec: FileRecord,
    line_no: int,
    line_text: str,
    model_spans: Dict[str, List[Dict]],
    model_labels: List[str],
):
    model_line_ents: Dict[str, List[NerEntity]] = {}
    for model_lbl in model_labels:
        model_line_ents[model_lbl] = merge_line_spans(
            model_spans.get(model_lbl, []), line_text
        )

    hybrid_spans = (
        model_spans.get(NER_MODELS["IndicNER"][0], [])
        + model_spans.get(NER_MODELS["XLM_RoBERTa"][0], [])
    )
    model_line_ents[HYBRID_KEY] = merge_line_spans(hybrid_spans, line_text)

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
        rec.line_ners[model_lbl].append(
            LineNerResult(
                model=model_lbl,
                line_no=line_no,
                text=line_text,
                entities=ents,
                persons=persons,
                locs=locs,
                orgs=orgs,
                missed=missed,
            )
        )


# ─────────────────────────────────────────────────────────────────────────────
# CORPUS-LEVEL NER (batch jobs)
# ─────────────────────────────────────────────────────────────────────────────
# run_line_by_line_ner() below is built for interactive document runs: every call
# spins up one worker thread per model, four queues and a fan-in loop. That cost
# is amortised over a whole document, but a dataset job calling it once per table
# cell pays the full setup to infer a single line, over and over.
# run_corpus_line_ner() is the batch-shaped counterpart: the same models and the
# same merge semantics as HYBRID_KEY, run straight over every line in the job at
# once — no threads, no queues, and each distinct line inferred exactly once.

# Lines handed to _infer_line_batch per call. This is bookkeeping only — it sets
# progress granularity, not how much work a forward pass does.
NER_CORPUS_BATCH_SIZE = int(os.getenv("NER_CORPUS_BATCH_SIZE", "256"))

# Real tensor batching. Every sequence in a batch pads up to the batch's longest,
# so the gain does not grow with batch size — past the optimum, padding waste
# overtakes it. On a reference export (172k distinct lines, mean 48 chars) the best
# size tracked the thread count: at 2 threads 8 won every repeat (43 vs 63 ms/line
# one-at-a-time, with 16 at 52 and 32 at 64), while at 4 threads 16 won. 0 picks a
# size from that relationship; set an explicit value to pin it, or 1 to batch off.
NER_TENSOR_BATCH_SIZE = int(os.getenv("NER_TENSOR_BATCH_SIZE", "0"))


def _tensor_batch_size() -> int:
    if NER_TENSOR_BATCH_SIZE > 0:
        return NER_TENSOR_BATCH_SIZE
    return max(8, 4 * _torch.get_num_threads())

# 0 means "leave torch's thread count alone". Raising it is not reliably a win:
# on a hybrid-core laptop CPU (a few performance cores plus several efficiency
# cores) spreading one short sequence over every core measured *slower* than the
# module-level cap of 4, so this is an opt-in knob to tune per machine rather
# than a default that assumes more cores are better.
NER_TORCH_THREADS = int(os.getenv("NER_TORCH_THREADS", "0"))

# (display label, model id, use_fast, min score) — mirrors run_line_by_line_ner's
# `pipes` mapping, which is what HYBRID_KEY merges.
_HYBRID_NER_SPECS: List[Tuple[str, str, bool, float]] = [
    (NER_MODELS["IndicNER"][0], "ai4bharat/IndicNER", False, 0.45),
    (NER_MODELS["XLM_RoBERTa"][0], "Davlan/xlm-roberta-base-ner-hrl", True, 0.45),
]
# min_score raised from 0.20 -> 0.45 for HiNER and IndicNER to match XLM-RoBERTa's
# threshold. The 0.20 floor kept detections where the model was only 20% confident,
# producing many spurious LOC/ORG tags on common Hindi words ("hamare", "pichhle", "do").


def run_corpus_line_ner(
    lines: List[str],
    batch_size: int = NER_CORPUS_BATCH_SIZE,
    release_models: bool = False,
    spans_by_model: Optional[Dict[str, List[List[Dict]]]] = None,
    profiler=None,
) -> List[List[NerEntity]]:
    """Hybrid NER over a whole corpus of lines, returning one entity list per line.

    Output is positionally aligned with `lines` and carries the same merged
    entities that run_line_by_line_ner stores under HYBRID_KEY.

    Lines are length-sorted so that if tensor batching is enabled (see
    NER_TENSOR_BATCH_SIZE) each batch pads to roughly its own longest member
    rather than the corpus maximum. Models run one after another over the full
    corpus rather than concurrently: on CPU they would otherwise fight over the
    same torch thread pool, and holding one model resident at a time keeps peak
    RSS to roughly a third.

    release_models evicts each pipe from the cache once its pass is done — worth
    it for a one-shot batch job, where nothing reuses the weights afterwards.

    Pass a dict as `spans_by_model` to have it filled with {model label: per-line
    raw spans}; the merged return value pools the models together and cannot say
    which one found what, which is what attribution reporting needs.
    """
    if not lines:
        return []
    if not _TRANSFORMERS_AVAILABLE:
        print("  [NER] transformers not installed — skipping model inference.", flush=True)
        return [[] for _ in lines]

    batch_size = max(1, batch_size)
    # Longest first: the slowest batches land while the run is young, so the
    # throughput estimate printed below settles on a pessimistic figure early.
    order = sorted(range(len(lines)), key=lambda i: len(lines[i]), reverse=True)
    spans_per_line: List[List[Dict]] = [[] for _ in lines]

    previous_threads = _torch.get_num_threads()
    if NER_TORCH_THREADS > 0:
        _torch.set_num_threads(NER_TORCH_THREADS)
    try:
        return _corpus_ner_passes(
            lines, order, spans_per_line, batch_size, release_models,
            spans_by_model=spans_by_model, profiler=profiler,
        )
    finally:
        _torch.set_num_threads(previous_threads)


def _corpus_ner_passes(
    lines: List[str],
    order: List[int],
    spans_per_line: List[List[Dict]],
    batch_size: int,
    release_models: bool,
    spans_by_model: Optional[Dict[str, List[List[Dict]]]] = None,
    profiler=None,
) -> List[List[NerEntity]]:
    """One full-corpus pass per model, then merge each line's spans (see caller).

    When `spans_by_model` is supplied it is filled in alongside the merged view, so
    a caller can still say which model produced a given span after the merge has
    pooled them.
    """
    from profiling import NULL_PROFILER

    profiler = profiler or NULL_PROFILER
    tensor_batch = _tensor_batch_size()
    print(
        f"  [NER] {len(lines)} lines, tensor batch {tensor_batch}, "
        f"{_torch.get_num_threads()} torch threads",
        flush=True,
    )
    for label, model_id, use_fast, min_score in _HYBRID_NER_SPECS:
        with profiler.step(f"load: {label}", parent="ner", model=model_id):
            pipe = _load_ner_pipeline(model_id, use_fast=use_fast)
        if pipe is None:
            print(f"  [NER] {label}: unavailable, skipped.", flush=True)
            continue

        model_spans = None if spans_by_model is None else spans_by_model.setdefault(
            label, [[] for _ in lines]
        )
        infer_cm = profiler.step(
            f"infer: {label}", parent="ner", model=model_id, lines=len(lines)
        )
        infer_cm.__enter__()
        started = time.time()
        done = 0
        for offset in range(0, len(order), batch_size):
            chunk = order[offset : offset + batch_size]
            tasks = [LineTask(doc_id=0, line_no=i, line_text=lines[i]) for i in chunk]
            for result in _infer_line_batch(
                pipe, tasks, min_score, label, batch_size=tensor_batch,
            ):
                if result.error:
                    print(
                        f"  [NER] {label} error on line {result.line_no}: {result.error}",
                        flush=True,
                    )
                spans_per_line[result.line_no].extend(result.spans)
                if model_spans is not None:
                    model_spans[result.line_no].extend(result.spans)
            done += len(chunk)
            if offset and (offset // batch_size) % 25 == 0:
                rate = done / max(time.time() - started, 1e-6)
                remaining = (len(order) - done) / max(rate, 1e-6)
                print(
                    f"  [NER] {label}: {done}/{len(order)} lines "
                    f"({rate:.1f} lines/s, ~{remaining / 60:.1f} min left)",
                    flush=True,
                )

        elapsed = time.time() - started
        infer_cm.__exit__(None, None, None)
        print(f"  [NER] {label}: {len(order)} lines in {elapsed:.1f}s", flush=True)

        if release_models:
            with profiler.step(f"release: {label}", parent="ner", model=model_id):
                _NER_PIPELINE_CACHE.pop(model_id, None)
                del pipe
                gc.collect()

    with profiler.step("merge spans", parent="ner"):
        merged = [merge_line_spans(spans_per_line[i], lines[i]) for i in range(len(lines))]
    return merged


def run_line_by_line_ner(
    records: List[FileRecord],
    batch_size: int = NER_LINE_BATCH_SIZE,
    queue_size: int = NER_QUEUE_MAXSIZE,
):
    """Load each NER model once, then fan lines out to dedicated worker threads.

    Architecture: Files → Dispatcher → Q_I / Q_X → workers → Q_M → merger.
    Threads share process memory (one model instance per thread). On CPU the two models overlap.
    """
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
    indicner_pipe = _load_ner_pipeline("ai4bharat/IndicNER", use_fast=False)
    xlm_pipe = _load_ner_pipeline(
        "Davlan/xlm-roberta-base-ner-hrl", use_fast=True
    )

    pipes = {
        NER_MODELS["IndicNER"][0]: (indicner_pipe, 0.45),
        NER_MODELS["XLM_RoBERTa"][0]: (xlm_pipe, 0.45),
    }

    model_labels = list(pipes.keys())
    all_model_labels = [v[0] for v in NER_MODELS.values()]
    for rec in records:
        rec.line_ners = {lbl: [] for lbl in all_model_labels + [HYBRID_KEY]}

    tasks: List[LineTask] = []
    line_lookup: Dict[Tuple[int, int], str] = {}
    for doc_id, rec in enumerate(records):
        for line_idx, line in enumerate(rec.raw_text.splitlines()):
            original_line = line.strip()
            # Skip empty lines or trivial noise (< 3 chars) to reduce unnecessary model passes on CPU
            if not original_line or len(original_line) < 3:
                continue
            line_no = line_idx + 1
            task = LineTask(doc_id=doc_id, line_no=line_no, line_text=original_line)
            tasks.append(task)
            line_lookup[(doc_id, line_no)] = original_line

    if not tasks:
        return

    expected_per_line = len(model_labels)
    total_expected = len(tasks) * expected_per_line
    bound = max(8, queue_size)
    in_queues = {lbl: Queue(maxsize=bound) for lbl in model_labels}
    result_queue: Queue = Queue(maxsize=max(bound * expected_per_line, bound))

    workers = []
    for model_lbl, (pipe, threshold) in pipes.items():
        thread = threading.Thread(
            target=_ner_model_worker,
            name=f"ner-{model_lbl.split()[0]}",
            args=(
                model_lbl,
                pipe,
                threshold,
                in_queues[model_lbl],
                result_queue,
                max(1, batch_size),
            ),
            daemon=True,
        )
        thread.start()
        workers.append(thread)

    print(
        f"  [NER] Parallel workers: {len(workers)} models, "
        f"{len(tasks)} lines, batch={max(1, batch_size)}, "
        f"device={_NER_DEVICE_STR}",
        flush=True,
    )

    def dispatch():
        for task in tasks:
            for q in in_queues.values():
                q.put(task)
        for q in in_queues.values():
            q.put(None)

    dispatcher = threading.Thread(target=dispatch, name="ner-dispatcher", daemon=True)
    dispatcher.start()

    pending: Dict[Tuple[int, int], Dict[str, List[Dict]]] = {}
    workers_done = 0
    received = 0
    idle_rounds = 0
    max_idle = int(NER_WORKER_JOIN_TIMEOUT_S / NER_RESULT_GET_TIMEOUT_S) + 1

    while workers_done < len(workers) and idle_rounds < max_idle:
        try:
            item = result_queue.get(timeout=NER_RESULT_GET_TIMEOUT_S)
            idle_rounds = 0
        except Empty:
            idle_rounds += 1
            if not any(w.is_alive() for w in workers) and result_queue.empty():
                break
            continue

        if item is None:
            workers_done += 1
            continue

        received += 1
        if item.error:
            print(
                f"  [NER] Worker error {item.model_name} "
                f"doc={item.doc_id} line={item.line_no}: {item.error}",
                flush=True,
            )
        key = (item.doc_id, item.line_no)
        pending.setdefault(key, {})[item.model_name] = item.spans
        if len(pending[key]) >= expected_per_line:
            rec = records[item.doc_id]
            _store_line_ner_results(
                rec,
                item.line_no,
                line_lookup[key],
                pending.pop(key),
                all_model_labels,
            )

    dispatcher.join(timeout=5)
    for worker in workers:
        worker.join(timeout=2)

    # Timeout / partial results: merge whatever arrived so a failed worker
    # cannot leave a document waiting indefinitely.
    for key, model_spans in list(pending.items()):
        doc_id, line_no = key
        if doc_id >= len(records) or key not in line_lookup:
            continue
        missing = [lbl for lbl in model_labels if lbl not in model_spans]
        if missing:
            print(
                f"  [NER] Incomplete results for {records[doc_id].filename} "
                f"line {line_no}; missing {missing} (empty spans used).",
                flush=True,
            )
        _store_line_ner_results(
            records[doc_id],
            line_no,
            line_lookup[key],
            model_spans,
            model_labels,
        )
        pending.pop(key, None)

    for rec in records:
        for model_lbl in rec.line_ners:
            rec.line_ners[model_lbl].sort(key=lambda r: r.line_no)

    print(
        f"  [NER] Fan-in complete: {received}/{total_expected} model-line results.",
        flush=True,
    )


def scan_records_pii(records: List[FileRecord]):
    for rec in records:
        rec.pii_hits = full_pii_scan(rec.raw_text)


def analyze_records(
    records: List[FileRecord],
    batch_size: int = NER_LINE_BATCH_SIZE,
    queue_size: int = NER_QUEUE_MAXSIZE,
):
    """Regex PII scan runs in a dispatcher thread while NER workers consume lines."""
    pii_errors: List[BaseException] = []

    def _pii_worker():
        try:
            scan_records_pii(records)
        except BaseException as exc:
            pii_errors.append(exc)

    pii_thread = threading.Thread(target=_pii_worker, name="pii-dispatcher", daemon=True)
    pii_thread.start()
    run_line_by_line_ner(
        records,
        batch_size=batch_size,
        queue_size=queue_size,
    )
    pii_thread.join()
    if pii_errors:
        raise pii_errors[0]


# ─────────────────────────────────────────────────────────────────────────────
# JSON GENERATOR (MAIN + INDIVIDUAL MODEL JSONs)
# ─────────────────────────────────────────────────────────────────────────────


def build_json(records: List[FileRecord], output_path: str):
    base_name, _ = os.path.splitext(output_path)
    main_json_path = base_name + ".json"
    print(f"\n▶ Writing JSON reports → {base_name}_*.json …", flush=True)

    all_models = [
        NER_MODELS["IndicNER"][0],
        NER_MODELS["XLM_RoBERTa"][0],
        HYBRID_KEY,
    ]

    model_slugs = {
        NER_MODELS["IndicNER"][0]: "IndicNER",
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
    records: List[FileRecord], output_path: str, presidio_available: bool = False
):
    print(f"\n▶ Writing 4-Sheet Excel report → {output_path} …", flush=True)

    all_model_labels = [
        NER_MODELS["IndicNER"][0],
        NER_MODELS["XLM_RoBERTa"][0],
        HYBRID_KEY,
    ]

    # Sheet 1: Summary
    s1_rows = []
    for rec in records:
        all_hits = rec.pii_hits
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
                "PII Types Found": pii_types,
                "Hybrid NER Persons": " | ".join(hy_persons) or "None",
                "Hybrid NER Locations": " | ".join(hy_locs) or "None",
                "Hybrid NER Organizations": " | ".join(hy_orgs) or "None",
                "Has PII": "YES" if tot_pii > 0 else "NO",
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
    batch_size: int = NER_LINE_BATCH_SIZE,
    queue_size: int = NER_QUEUE_MAXSIZE,
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

    analyze_records(
        [rec],
        batch_size=max(1, batch_size),
        queue_size=max(8, queue_size),
    )
    build_excel([rec], output_path)
    build_json([rec], output_path)

    return [rec]


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Multi-Format (TXT/DOCX/HTML/JSON/CSV) → Parallel PII/NER Detection → Excel & JSON Reports\n\n"
            "Input modes:\n"
            "  File / folder:  --input path/to/file_or_folder\n"
            '  Inline string:  --text "Your text enclosed in quotes"\n'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--config",
        default=None,
        help="Run the config-driven dataset batch job using this JSON config file or a directory to scan for one.",
    )
    ap.add_argument(
        "--input",
        "-i",
        default=None,
        help="Path to a single file (.txt/.docx/.doc/.html/.json/.csv) or a folder of such files.",
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
        default=os.getenv("HF_TOKEN", ""),
        help="HuggingFace token (for gated models such as IndicNER).",
    )
    ap.add_argument(
        "--ner-batch-size",
        type=int,
        default=NER_LINE_BATCH_SIZE,
        help="Mini-batch size for NER worker inference (16–32 recommended, default 24).",
    )
    ap.add_argument(
        "--queue-size",
        type=int,
        default=NER_QUEUE_MAXSIZE,
        help="Bounded queue size for NER backpressure (default 64).",
    )
    if any("jupyter" in arg or "kernel" in arg for arg in sys.argv):
        return ap.parse_args(args=[])
    return ap.parse_args()


def main():
    args = parse_args()

    if args.config is not None:
        from batch_pipeline import run as run_batch
        from batch_pipeline import BatchError

        try:
            run_batch(args.config)
        except BatchError as exc:
            print(f"[Batch][ERROR] {exc}", flush=True)
            raise SystemExit(1)
        return

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

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
        process_text_string(
            args.text,
            output_path=args.output,
            batch_size=max(1, args.ner_batch_size),
            queue_size=max(8, args.queue_size),
        )
        return

    input_path = args.input if args.input is not None else DEFAULT_INPUT_PATH
    supported_files = collect_files(input_path)

    records: List[FileRecord] = []
    for fp in supported_files:
        try:
            raw, ftype = extract_text(fp)
        except Exception as exc:
            print(f"  [WARN] Could not read {fp}: {exc}", flush=True)
            continue
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

    if not records:
        sys.exit("[ERROR] No readable documents found after extraction.")

    print(
        f"[Pipeline] Documents: {len(records)} | Parallel NER workers + overlapping PII scan",
        flush=True,
    )
    analyze_records(
        records,
        batch_size=max(1, args.ner_batch_size),
        queue_size=max(8, args.queue_size),
    )
    build_excel(records, args.output)
    build_json(records, args.output)


if __name__ == "__main__":
    main()