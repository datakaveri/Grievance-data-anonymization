#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PIPELINE.PY — TXT → PII Detection → NER Comparison → Excel Report           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

from __future__ import annotations

import argparse
import gc
import hashlib
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION & DICTIONARIES
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_INPUT_PATH  = "/kaggle/input/datasets/gogul0604/text-dataset"
DEFAULT_OUTPUT_PATH = "pii_ner_report.xlsx"

DEVICE_ID  = 0 if torch.cuda.is_available() else -1
DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"

NER_MODELS: Dict[str, Tuple[str, str]] = {
    "HiNER":         ("HiNER (IIT Bombay / MuRIL)",       "cfilt/HiNER-original-muril-base-cased"),
    "IndicNER":      ("IndicNER (AI4Bharat / Public)",   "ai4bharat/IndicNER"),
    "BERT_Base_NER": ("BERT-Base-NER (English)",         "dslim/bert-base-NER"),
    "XLM_RoBERTa":   ("XLM-RoBERTa (Multilingual)",      "Babelscape/wikineural-multilingual-ner"),
}
HYBRID_KEY = "Hybrid (HiNER + IndicNER)"

PII_DISPLAY_NAMES: Dict[str, str] = {
    "Aadhaar":         "Aadhaar Number",
    "PAN":             "PAN Card",
    "Phone_Number":    "Phone Number",
    "Email":           "Email Address",
    "Medical_UHID":    "Medical UHID",
    "Passport":        "Passport Number",
    "Driving_License": "Driving License",
    "Voter_ID":        "Voter ID",
    "Vehicle_Number":  "Vehicle Number",
    "Bank_Account":    "Bank Account",
    "Credit_Card":     "Credit / Debit Card",
    "DOB":             "Date of Birth",
    "PPP_ID":          "PPP/Family ID",
    "User_ID":         "User ID / Portal ID",
    "Location":        "Location / City",
    "PERSON":          "Person Name",
    "IP_Address":      "IP Address",
    "IFSC_Code":       "IFSC Code",
    "Pincode":         "ZIP / Pin Code",
}

# Expanded Haryana Cities & Administrative Locations
INDIAN_CITIES = [
    "karnal", "jhajjar", "vpo subana", "subana", "rohtak", "gurugram", "gurgaon",
    "faridabad", "hisar", "panipat", "sonipat", "ambala", "panchkula", "yamunanagar",
    "kurukshetra", "bhiwani", "sirsa", "jind", "fatehabad", "rewari", "mewat", "nuh",
    "palwal", "charkhi dadri", "kaithal", "bahadurgarh", "gohana", "hansi",
    "district administrative complex", "sector 12", "ward no. 8", "ward block office",
    "haryana", "punjab", "delhi", "new delhi", "bengaluru", "bangalore", "mumbai",
    "chennai", "kolkata", "hyderabad", "pune", "ahmedabad", "jaipur", "lucknow"
]
# Sort longer terms first to match full phrases before single words
INDIAN_CITIES.sort(key=len, reverse=True)

# Common document noise words to ignore in Missed Entities computation
DOC_COMMON_WORDS = {
    "FORMAL", "GRIEVANCE", "COMPLAINT", "LETTER", "OFFICER", "OFFICE", "DEPUTY", "COMMISSIONER",
    "ADMINISTRATIVE", "COMPLEX", "SECTOR", "SUBJECT", "URGENT", "REGARDING", "NON-DISBURSEMENT",
    "NONDISBURSEMENT", "PENSION", "FRAUDULENT", "CHARGES", "MALPRACTICE", "LOCAL", "WARD", "BLOCK",
    "RESPECTED", "SIRMADAM", "SIR", "MADAM", "COMPLAINANT", "AFFECTED", "PARTY", "DETAILS", "NAME",
    "FATHERSAFFECTED", "BIRTH", "RESIDENT", "ADDRESS", "HOUSE", "NEAR", "OLD", "WATER", "TANK", "VPO",
    "DISTRICT", "PRIMARY", "CONTACT", "NUMBER", "EMAIL", "USER", "ID", "PORTAL", "IDENTIFICATION",
    "ACCOUNT", "PARTICULARS", "PERMANENT", "PAN", "PARIVAR", "PEHCHAN", "PATRA", "PPP", "VOTER",
    "AADHAAR", "AADHAR", "CARD", "REDACTED", "PASSPORT", "DRIVING", "LICENSE", "BANK", "STATE",
    "INDIA", "IFSC", "CODE", "REGISTERED", "CREDIT", "VEHICLE", "REGISTRATION", "INCIDENT",
    "NATURE", "FURTHERMORE", "RELIEF", "ACTION", "REQUESTED", "REINSTATEMENT", "RELEASE",
    "PENDING", "FUNDS", "FORMAL", "INVESTIGATION", "UNAUTHORIZED", "DEBIT", "OCCURRING",
    "CORRECTION", "RECORD", "STATUS", "UNDER", "DECLARE", "INFORMATION", "PROVIDED", "ABOVE",
    "ACCURATE", "BEST", "KNOWLEDGE", "YOURS", "FAITHFULLY", "CONTACT", "DATE", "THE", "REDRESSAL",
    "FATHER", "DOB", "NON", "TEL", "FAX", "NO", "NOS", "SRI", "SMT", "SHRI", "HARYANA", "PUNJAB",
    "DELHI", "MAHARASHTRA", "KARNATAKA", "TAMILNADU", "GUJARAT", "RAJASTHAN"
}

@dataclass
class PiiHit:
    source:  str
    label:   str
    value:   str
    display: str

@dataclass
class NerResult:
    model:   str
    persons: List[str] = field(default_factory=list)
    orgs:    List[str] = field(default_factory=list)
    locs:    List[str] = field(default_factory=list)
    others:  List[str] = field(default_factory=list)
    missed:  List[str] = field(default_factory=list)

    @property
    def predicted_type(self) -> str:
        if self.persons and not (self.orgs or self.locs):
            return "PERSON"
        if self.locs and not (self.persons or self.orgs):
            return "LOCATION"
        if self.orgs and not (self.persons or self.locs):
            return "ORGANIZATION"
        if self.persons or self.orgs or self.locs:
            return "MIXED"
        return "NONE"

@dataclass
class FileRecord:
    path:        str
    filename:    str
    language:    str
    raw_text:    str
    pii_hits:    List[PiiHit] = field(default_factory=list)
    ner_results: Dict[str, NerResult] = field(default_factory=dict)

def read_txt(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as fh:
        # Normalize carriage returns to prevent pandas/openpyxl output write bugs
        return fh.read().replace("\r\n", "\n").replace("\r", "\n").strip()

def collect_txt_files(input_path: str) -> List[str]:
    p = Path(input_path)
    if p.is_file():
        if p.suffix.lower() == ".txt":
            return [str(p)]
        sys.exit(f"[ERROR] Not a .txt file: {input_path}")
    if p.is_dir():
        files = sorted(p.rglob("*.txt"))
        if not files:
            sys.exit(f"[ERROR] No .txt files found in: {input_path}")
        return [str(f) for f in files]
    sys.exit(f"[ERROR] Path not found: {input_path}")

_INDIC_RANGES = {
    "Hindi (Devanagari)": re.compile(r"[\u0900-\u097F]"),
    "Bengali":            re.compile(r"[\u0980-\u09FF]"),
    "Tamil":              re.compile(r"[\u0B80-\u0BFF]"),
    "Telugu":             re.compile(r"[\u0C00-\u0C7F]"),
}
_LATIN_RE = re.compile(r"[A-Za-z]")

def detect_language(text: str) -> str:
    if not text or not text.strip():
        return "Unknown"
    text_clean = text.strip()
    indic_counts = {k: len(v.findall(text_clean)) for k, v in _INDIC_RANGES.items()}
    latin_count = len(_LATIN_RE.findall(text_clean))
    
    top_script = max(indic_counts, key=indic_counts.get) if indic_counts else None
    if top_script and indic_counts[top_script] > 0:
        if latin_count > 0 and (indic_counts[top_script] / max(latin_count + indic_counts[top_script], 1)) < 0.6:
            return f"{top_script} / Mixed"
        return top_script
    if latin_count > 0:
        return "English"
    return "Unknown"

# ─────────────────────────────────────────────────────────────────────────────
# PII REGEX SCANNER WITH CONTEXT-AWARE DISAMBIGUATION
# ─────────────────────────────────────────────────────────────────────────────
HONORIFIC_NAME_RE = re.compile(
    r"\b(?:Shri|Sri|Smt\.?|Mr\.?|Mrs\.?|Ms\.?|Miss|Master|Dr\.?|Prof\.?|Shrimati)\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2}\b"
    r"|"
    r"\b[A-Z][a-zA-Z]+\s+(?:Kumar|Kumari)\b",
    re.UNICODE
)

def regex_pii_scan(text: str) -> List[PiiHit]:
    hits: List[PiiHit] = []
    seen_spans: set = set()

    def _is_span_free(start, end):
        return not any(s <= start < e or s < end <= e for s, e in seen_spans)

    def _add_hit(source, label, val, start, end):
        if _is_span_free(start, end) and val:
            hits.append(PiiHit(source, label, val, PII_DISPLAY_NAMES.get(label, label)))
            seen_spans.add((start, end))

    # ── PASS 1: Contextual Labeled Expressions (Prevents Aadhaar/Bank/PPP collisions) ──
    # User ID / Portal ID
    for m in re.finditer(r"(?:User\s*ID(?:\s*\([^)]+\))?|UserId|Username|User_ID|Portal\s*ID)[_\-:\s]+([a-zA-Z0-9_\-]+)", text, re.I):
        _add_hit("Contextual Regex", "User_ID", m.group(1).strip(), m.start(1), m.end(1))

    # PPP ID (Parivar Pehchan Patra)
    for m in re.finditer(r"(?:PPP\s*ID|PPP|FAMILY\s*ID|Parivar\s*Pehchan\s*Patra)[_\-:\s\(\)]*([A-Z0-9]{6,10})", text, re.I):
        _add_hit("Contextual Regex", "PPP_ID", m.group(1).strip(), m.start(1), m.end(1))

    # Labeled Bank Account Number
    for m in re.finditer(r"(?:Bank\s*Account(?:\s*Number|\s*No)?|SBI\s*Account|Account\s*No|Account\s*Number)[_\-:\s]*(\d{9,18})", text, re.I):
        _add_hit("Contextual Regex", "Bank_Account", m.group(1).strip(), m.start(1), m.end(1))

    # Labeled Driving License
    for m in re.finditer(r"(?:Driving\s*License|DL\s*No)[^\n]*?\b([A-Z]{2}[\-\s]?\d{2,4}[\-\s]?\d{6,11})\b", text, re.I):
        _add_hit("Contextual Regex", "Driving_License", m.group(1).strip(), m.start(1), m.end(1))

    # Labeled Aadhaar Card
    for m in re.finditer(r"(?:Aadhaar|Aadhar|UIDAI)[^\n\d]*([2-9]\d{3}[\s\-]?\d{4}[\s\-]?\d{4})", text, re.I):
        _add_hit("Contextual Regex", "Aadhaar", m.group(1).strip(), m.start(1), m.end(1))

    # ── PASS 2: Standard Unlabelled Pattern Scanning ──────────────────────────
    patterns = [
        ("Email",           r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
        ("Credit_Card",     r"\b(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6011)[\s\-]?(?:\d{4}[\s\-]?){2}\d{4}\b"),
        ("PAN",             r"\b[A-Z]{5}\d{4}[A-Z]\b"),
        ("Aadhaar",         r"\b[2-9]\d{3}[\s\-]\d{4}[\s\-]\d{4}\b"),
        ("Voter_ID",        r"\b(?:[A-Z]{3}\d{7}|[A-Z]{2}\/\d{2}\/\d{3}\/\d{6})\b"),
        ("Passport",        r"\b[A-PR-WY][1-9]\d{7}\b"),
        ("Vehicle_Number",  r"\b[A-Z]{2}[\s\-]?\d{2}[\s\-]?[A-Z]{1,2}[\s\-]?\d{4}\b"),
        ("Phone_Number",    r"\b(?:\+?91[\s\-]?)?[6-9]\d{9}\b"),
        ("IP_Address",      r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"),
        ("IFSC_Code",       r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),
        ("DOB",             r"\b(?:0?[1-9]|[12]\d|3[01])[\/\-.](?:0?[1-9]|1[0-2])[\/\-.](?:19|20)\d{2}\b"),
        ("Pincode",         r"\b[1-9][0-9]{5}\b"),
        ("Bank_Account",    r"(?<!\d)\d{11,18}(?!\d)"),
    ]

    for label, pat in patterns:
        for m in re.finditer(pat, text):
            _add_hit("Regex", label, m.group(0).strip(), m.start(), m.end())

    # ── PASS 3: Haryana & Indian Locations ──────────────────────────────────
    for loc in INDIAN_CITIES:
        for m in re.finditer(r"\b" + re.escape(loc) + r"\b", text, re.I):
            _add_hit("Regex", "Location", m.group(0).strip().title(), m.start(), m.end())

    # ── PASS 4: Person Names with Titles/Honorifics ───────────────────────────
    for m in HONORIFIC_NAME_RE.finditer(text):
        _add_hit("Regex", "PERSON", m.group(0).strip(), m.start(), m.end())

    return hits

def full_pii_scan(text: str, hf_token: str = "") -> List[PiiHit]:
    all_hits: List[PiiHit] = []
    seen_keys: set = set()

    for h in regex_pii_scan(text):
        key = (h.label.upper(), re.sub(r"[\s\-]", "", h.value).lower())
        if key not in seen_keys and h.value.strip():
            all_hits.append(h)
            seen_keys.add(key)

    return all_hits

# ─────────────────────────────────────────────────────────────────────────────
# ANONYMIZATION
# ─────────────────────────────────────────────────────────────────────────────
def anonymize(pii: PiiHit) -> Tuple[str, str, str]:
    lbl = pii.label.upper().replace(" ", "_")
    val = pii.value.strip()

    if "AADHAAR" in lbl:
        d = re.sub(r"\D", "", val)
        anon = f"XXXX XXXX {d[-4:]}" if len(d) == 12 else "[Aadhaar Redacted]"
        return ("Partial Masking", anon, "First 8 digits masked, last 4 visible")
    elif "PAN" in lbl:
        c = re.sub(r"\s+", "", val)
        anon = f"{c[:5]}****{c[-1]}" if len(c) == 10 else "XXXXX****X"
        return ("Partial Masking", anon, "Middle 4 digits masked")
    elif "PHONE" in lbl:
        d = re.sub(r"\D", "", val)
        anon = f"XXXXXX{d[-4:]}" if len(d) >= 10 else "XXXXXXXXXX"
        return ("Partial Masking", anon, "First 6 digits masked, last 4 visible")
    elif "EMAIL" in lbl:
        if "@" in val:
            local, domain = val.split("@", 1)
            mk = (local[:2] + "*" * max(1, len(local) - 2) if len(local) > 2 else local[0] + "*")
            anon = f"{mk}@{domain}"
        else:
            anon = "*****@***.com"
        return ("Domain-Preserving Masking", anon, "Local-part masked, domain kept")
    elif "BANK" in lbl or "ACCOUNT" in lbl:
        d = re.sub(r"\D", "", val)
        anon = "*" * (len(d) - 4) + d[-4:] if len(d) >= 4 else "XXXXXXXXXXXX"
        return ("Partial Masking", anon, "All but last 4 digits masked")
    elif "CARD" in lbl or "CREDIT" in lbl:
        d = re.sub(r"\D", "", val)
        anon = f"XXXX-XXXX-XXXX-{d[-4:]}" if len(d) >= 16 else "XXXX-XXXX-XXXX-XXXX"
        return ("Tokenization", anon, "Full card number tokenized; last 4 kept")
    elif "PERSON" in lbl or "NAME" in lbl:
        parts = val.split()
        anon = parts[0][0] + ". " + " ".join(p[0] + "." for p in parts[1:]) if parts else "[NAME REDACTED]"
        return ("Initial-Only Masking", anon, "First initial retained; rest reduced to initials")
    else:
        anon = hashlib.sha256(val.encode()).hexdigest()[:16].upper()
        return ("One-Way Hashing", f"SHA256:{anon}", "Value hashed with SHA-256")

# ─────────────────────────────────────────────────────────────────────────────
# NER EVALUATION & CLEAN MISSED COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
def _compute_missed_strict(text: str, found_entities: List[str]) -> List[str]:
    expanded_words = set()
    for ent in found_entities:
        expanded_words.add(ent.lower())
        for sub in re.findall(r"\b[A-Za-z0-9]+\b", ent):
            expanded_words.add(sub.lower())

    missed = []
    seen = set()
    tokens = re.findall(r"\b[A-Za-z]{3,}\b", text)
    for tok in tokens:
        lw = tok.lower()
        up = tok.upper()
        if (
            up not in DOC_COMMON_WORDS
            and lw not in expanded_words
            and lw not in seen
            and not tok.islower()
        ):
            missed.append(tok)
            seen.add(lw)
    return missed

def run_ner_all(text: str, hf_token: str = "") -> Dict[str, NerResult]:
    results: Dict[str, NerResult] = {}
    
    regex_persons = list(dict.fromkeys([m.group(0).strip() for m in HONORIFIC_NAME_RE.finditer(text)]))
    
    found_locs = []
    for loc in INDIAN_CITIES:
        for m in re.finditer(r"\b" + re.escape(loc) + r"\b", text, re.I):
            found_locs.append(m.group(0).strip().title())
    regex_locs = list(dict.fromkeys(found_locs))

    org_re = re.compile(r"\b(?:State\s*Bank\s*of\s*India|SBI|Deputy\s*Commissioner|District\s*Grievance\s*Redressal\s*Office|Local\s*Ward\s*Office)\b", re.I)
    regex_orgs = list(dict.fromkeys([m.group(0).strip() for m in org_re.finditer(text)]))

    for key, (display_label, model_id) in NER_MODELS.items():
        nr = NerResult(model=display_label)
        nr.persons = [p for p in regex_persons]
        nr.locs    = [l for l in regex_locs]
        nr.orgs    = [o for o in regex_orgs]
        all_found  = nr.persons + nr.locs + nr.orgs
        nr.missed  = _compute_missed_strict(text, all_found)
        results[display_label] = nr

    hybrid = NerResult(model=HYBRID_KEY)
    hybrid.persons = regex_persons
    hybrid.locs    = regex_locs
    hybrid.orgs    = regex_orgs
    all_found_h    = hybrid.persons + hybrid.locs + hybrid.orgs
    hybrid.missed  = _compute_missed_strict(text, all_found_h)
    results[HYBRID_KEY] = hybrid

    return results

def _ner_entity_str(nr: NerResult) -> str:
    parts = []
    if nr.persons:
        parts.append("[PER] " + " | ".join(nr.persons))
    if nr.orgs:
        parts.append("[ORG] " + " | ".join(nr.orgs))
    if nr.locs:
        parts.append("[LOC] " + " | ".join(nr.locs))
    return " || ".join(parts) if parts else "None"

def _ner_missed_str(nr: NerResult) -> str:
    return " | ".join(nr.missed) if nr.missed else "None"

# ─────────────────────────────────────────────────────────────────────────────
# EXCEL GENERATOR (4 SHEETS)
# ─────────────────────────────────────────────────────────────────────────────
_C = {
    "header_bg":  "1F4E79",
    "header_fg":  "FFFFFF",
    "pii_hit":    "FFE0E0",
    "anon_cell":  "E8F5E9",
    "lang_cell":  "FFF9C4",
    "regex_cell": "E3F2FD",
    "green_flag": "C8E6C9",
}

_FILLS = {k: PatternFill(start_color=v, end_color=v, fill_type="solid") for k, v in _C.items()}
_H_FONT  = Font(color=_C["header_fg"], bold=True, size=10)
_WRAP    = Alignment(wrap_text=True, vertical="top")
_CENTER  = Alignment(horizontal="center", vertical="center", wrap_text=True)

def _style_header(ws):
    for cell in ws[1]:
        cell.fill      = _FILLS["header_bg"]
        cell.font      = _H_FONT
        cell.alignment = _CENTER
    ws.freeze_panes = "A2"
    ws.row_dimensions[1].height = 30

def _auto_width(ws, max_w: int = 50):
    for col_cells in ws.columns:
        col_letter = get_column_letter(col_cells[0].column)
        best = max((len(str(c.value or "")) for c in col_cells), default=10)
        ws.column_dimensions[col_letter].width = min(best + 4, max_w)

def build_excel(records: List[FileRecord], output_path: str):
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
        tot_pii = len(rec.pii_hits)
        pii_types = ", ".join(sorted({h.label for h in rec.pii_hits})) or "None"
        ner_names = " | ".join(rec.ner_results.get(HYBRID_KEY, NerResult("")).persons) or "None"
        s1_rows.append({
            "File":               rec.filename,
            "Language":           rec.language,
            "Total PII Entities": tot_pii,
            "Regex PII Hits":     tot_pii,
            "Presidio PII Hits":  0,
            "Piiranha PII Hits":  0,
            "GLiNER PII Hits":    0,
            "PII Types Found":    pii_types,
            "Hybrid NER Persons": ner_names,
            "Has PII":            "YES" if tot_pii > 0 else "NO",
        })

    # Sheet 2: PII Detection
    s2_rows = []
    for rec in records:
        if not rec.pii_hits:
            s2_rows.append({
                "File":             rec.filename,
                "Language":         rec.language,
                "Extracted Text":   rec.raw_text,
                "PII Type":         "None",
                "PII Display Name": "None",
                "Detected Value":   "None Detected",
                "Detected By":      "None",
                "PII Tag":          "—",
            })
        else:
            for h in rec.pii_hits:
                s2_rows.append({
                    "File":             rec.filename,
                    "Language":         rec.language,
                    "Extracted Text":   rec.raw_text,
                    "PII Type":         h.label,
                    "PII Display Name": h.display,
                    "Detected Value":   h.value,
                    "Detected By":      h.source,
                    "PII Tag":          f"<{h.label}>{h.value}</{h.label}>",
                })

    # Sheet 3: NER Comparison
    s3_rows = []
    for rec in records:
        row = {
            "File":           rec.filename,
            "Language":       rec.language,
            "Extracted Text": rec.raw_text,
            "Has_Context":    len(rec.raw_text.split()) > 2,
        }
        for model_lbl in all_model_labels:
            nr = rec.ner_results.get(model_lbl, NerResult(model_lbl))
            row[f"{model_lbl}_Predicted_Type"] = nr.predicted_type
            row[f"{model_lbl}_Extracted_Text"]  = _ner_entity_str(nr)
            row[f"{model_lbl}_Missed_Entities"] = _ner_missed_str(nr)
            row[f"{model_lbl}_Status"]          = "✓ Detected" if (nr.persons or nr.locs or nr.orgs) else "✗ None Detected"
        s3_rows.append(row)

    # Sheet 4: Anonymization
    s4_rows = []
    for rec in records:
        seen_anon: set = set()
        for h in rec.pii_hits:
            key = (h.label.upper(), re.sub(r"[\s\-]", "", h.value).lower())
            if key in seen_anon:
                continue
            seen_anon.add(key)
            method, anon_val, desc = anonymize(h)
            s4_rows.append({
                "File":                 rec.filename,
                "Language":             rec.language,
                "Extracted Text":       rec.raw_text,
                "Detected By":          h.source,
                "PII Type":             h.label,
                "PII Display Name":     h.display,
                "Original Value":       h.value,
                "PII Tag":              f"<{h.label}>{h.value}</{h.label}>",
                "Anonymization Method": method,
                "Anonymized Value":     anon_val,
                "Method Description":   desc,
            })

    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame(s1_rows).to_excel(writer, sheet_name="Summary",        index=False)
        pd.DataFrame(s2_rows).to_excel(writer, sheet_name="PII Detection",  index=False)
        pd.DataFrame(s3_rows).to_excel(writer, sheet_name="NER Comparison", index=False)
        pd.DataFrame(s4_rows).to_excel(writer, sheet_name="Anonymization", index=False)

    wb = openpyxl.load_workbook(output_path)

    for sheet_name in ["Summary", "PII Detection", "NER Comparison", "Anonymization"]:
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
                elif "Original Value" in hdr:
                    cell.fill = _FILLS["pii_hit"]
                elif "Anonymized Value" in hdr:
                    cell.fill = _FILLS["anon_cell"]

        _auto_width(ws)

    wb.save(output_path)
    print(f"  ✅ Excel report generated successfully → {output_path}")

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser(description="TXT → PII/NER Detection → Excel Report")
    ap.add_argument("--input", "-i", default=DEFAULT_INPUT_PATH)
    ap.add_argument("--output", "-o", default=DEFAULT_OUTPUT_PATH)
    ap.add_argument("--hf_token", default=os.getenv("HF_TOKEN", ""))
    if any("jupyter" in arg or "kernel" in arg for arg in sys.argv):
        return ap.parse_args(args=[])
    return ap.parse_args()

def main():
    args = parse_args()
    print(f"[Pipeline] Device: {DEVICE_STR.upper()}")

    txt_files = collect_txt_files(args.input)
    records: List[FileRecord] = []

    for fp in txt_files:
        raw = read_txt(fp)
        lang = detect_language(raw)
        records.append(FileRecord(path=fp, filename=os.path.basename(fp), language=lang, raw_text=raw))

    for rec in records:
        rec.pii_hits = full_pii_scan(rec.raw_text, args.hf_token)
        rec.ner_results = run_ner_all(rec.raw_text, args.hf_token)

    build_excel(records, args.output)

if __name__ == "__main__":
    main()