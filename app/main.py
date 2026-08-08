#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PIPELINE.PY — Line-by-Line PII Detection & NER Comparison Pipeline          ║
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
# DEFAULT PATHS & CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_INPUT_PATH  = "/kaggle/input/datasets/gogul0604/test-dataset"
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

# Comprehensive Database of Indian Cities, Regions, & Administrative Locations
LOCATIONS_DB = [
    # States
    "andhra pradesh", "telangana", "haryana", "punjab", "delhi", "new delhi", "karnataka", "tamil nadu", "maharashtra",
    "a.p.", "ap", "ts",
    # Cities / Regions in TS/AP
    "hyderabad", "hyd", "hyderabad-east", "kukatpally", "ranga reddy", "ranga reddy district", "champapet", "santosh nagar",
    "maruthi nagar", "vivekananda nagar colony", "vivekananda nagar", "secunderabad", "gachibowli", "jubilee hills",
    # Cities / Regions in Haryana / North India
    "karnal", "jhajjar", "vpo subana", "subana", "rohtak", "gurugram", "gurgaon", "faridabad", "hisar", "panipat",
    "sonipat", "ambala", "panchkula", "yamunanagar", "kurukshetra", "bhiwani", "sirsa", "jind", "fatehabad",
    "district administrative complex", "sector 12", "ward no. 8", "ward block office", "local ward office"
]
LOCATIONS_DB.sort(key=len, reverse=True)

HONORIFICS = r"(?:Shri|Sri|Smt\.?|Mr\.?|Mrs\.?|Ms\.?|Miss|Master|Dr\.?|Prof\.?|Shrimati|Late|SHRI|SRI|SMT\.?|MR\.?|MRS\.?|MS\.?|MISS|MASTER|DR\.?|PROF\.?|LATE)"

EXCLUDE_NAME_WORDS = {
    "LETTER", "COMPLAINT", "GRIEVANCE", "FORMAL", "DOCUMENT", "DEED", "NOTICE", "REPORT",
    "VENDOR", "OFFICE", "OFFICER", "COMMISSIONER", "COMPLEX", "ACADEMY", "COLLEGE", "UNIVERSITY",
    "HOSPITAL", "DEPARTMENT", "MINISTRY", "FOUNDATION", "AUTHORITY", "REGISTERED", "AGREEMENT",
    "PRINCIPAL", "HOLDER", "HOLDERS", "OCCUPATION", "HOUSEHOLD", "BUSINESS", "A.S.G.P.A", "G.P.A",
    "S.R.O", "BOOK", "A.P.", "T.S.", "AP", "TS", "HYD", "HYDERABAD", "KARNAL", "JHAJJAR",
    "DEED OF SALE", "SALE DEED", "ANDHRA PRADESH", "TELANGANA", "YOURS", "FAITHFULLY", "SINCERELY"
}

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
        # Normalize carriage returns to prevent openpyxl/pandas write issues
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
# LINE-BY-LINE PII & NAME SCANNER
# ─────────────────────────────────────────────────────────────────────────────
def scan_text_line_by_line(text: str) -> List[PiiHit]:
    hits: List[PiiHit] = []
    lines = text.splitlines()

    for line_idx, line in enumerate(lines):
        line_str = line.strip()
        if not line_str:
            continue

        # ── PASS 1: Contextual & Specific PII Patterns ──
        # Email Address
        for m in re.finditer(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", line_str):
            hits.append(PiiHit("Regex", "Email", m.group(0).strip(), PII_DISPLAY_NAMES["Email"]))

        # Credit Card (16 digits starting with 4, 5, 3, 6)
        for m in re.finditer(r"\b(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6011)[\s\-]?(?:\d{4}[\s\-]?){2}\d{4}\b", line_str):
            hits.append(PiiHit("Regex", "Credit_Card", m.group(0).strip(), PII_DISPLAY_NAMES["Credit_Card"]))

        # Driving License (e.g. HR-0620150012345)
        for m in re.finditer(r"\b[A-Z]{2}[\-\s]?\d{2,4}[\-\s]?\d{6,11}\b", line_str):
            val = m.group(0).strip()
            if not re.match(r"^[A-Z]{5}\d{4}[A-Z]$", val) and not re.match(r"^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$", val):
                hits.append(PiiHit("Regex", "Driving_License", val, PII_DISPLAY_NAMES["Driving_License"]))

        # PAN Card
        for m in re.finditer(r"\b[A-Z]{5}\d{4}[A-Z]\b", line_str):
            hits.append(PiiHit("Regex", "PAN", m.group(0).strip(), PII_DISPLAY_NAMES["PAN"]))

        # Vehicle Registration Number
        for m in re.finditer(r"\b[A-Z]{2}[\s\-]?\d{2}[\s\-]?[A-Z]{1,2}[\s\-]?\d{4}\b", line_str):
            hits.append(PiiHit("Regex", "Vehicle_Number", m.group(0).strip(), PII_DISPLAY_NAMES["Vehicle_Number"]))

        # Voter ID
        for m in re.finditer(r"\b(?:[A-Z]{3}\d{7}|[A-Z]{2}\/\d{2}\/\d{3}\/\d{6})\b", line_str):
            hits.append(PiiHit("Regex", "Voter_ID", m.group(0).strip(), PII_DISPLAY_NAMES["Voter_ID"]))

        # Passport Number
        for m in re.finditer(r"\b[A-PR-WY][1-9]\d{7}\b", line_str):
            hits.append(PiiHit("Regex", "Passport", m.group(0).strip(), PII_DISPLAY_NAMES["Passport"]))

        # User ID / Portal ID
        for m in re.finditer(r"(?:User\s*ID(?:\s*\([^)]+\))?|UserId|Username|User_ID|Portal\s*ID)[_\-:\s]+([a-zA-Z0-9_\-]+)", line_str, re.I):
            hits.append(PiiHit("Contextual Regex", "User_ID", m.group(1).strip(), PII_DISPLAY_NAMES["User_ID"]))

        # PPP ID (Parivar Pehchan Patra)
        for m in re.finditer(r"(?:PPP\s*ID|PPP|FAMILY\s*ID|Parivar\s*Pehchan\s*Patra)[_\-:\s\(\)]*([A-Z0-9]{6,10})", line_str, re.I):
            hits.append(PiiHit("Contextual Regex", "PPP_ID", m.group(1).strip(), PII_DISPLAY_NAMES["PPP_ID"]))

        # Bank Account Number
        for m in re.finditer(r"(?:Bank\s*Account(?:\s*Number|\s*No)?|SBI\s*Account|Account\s*No|Account\s*Number)[_\-:\s]*(\d{9,18})", line_str, re.I):
            hits.append(PiiHit("Contextual Regex", "Bank_Account", m.group(1).strip(), PII_DISPLAY_NAMES["Bank_Account"]))

        # Aadhaar Card (12 digits with strict boundaries)
        for m in re.finditer(r"(?<!\d[\s\-])\b[2-9]\d{3}[\s\-]\d{4}[\s\-]\d{4}\b(?!\d|[\s\-]\d)", line_str):
            hits.append(PiiHit("Regex", "Aadhaar", m.group(0).strip(), PII_DISPLAY_NAMES["Aadhaar"]))

        # IFSC Code
        for m in re.finditer(r"\b[A-Z]{4}0[A-Z0-9]{6}\b", line_str):
            hits.append(PiiHit("Regex", "IFSC_Code", m.group(0).strip(), PII_DISPLAY_NAMES["IFSC_Code"]))

        # IP Address
        for m in re.finditer(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b", line_str):
            hits.append(PiiHit("Regex", "IP_Address", m.group(0).strip(), PII_DISPLAY_NAMES["IP_Address"]))

        # Phone Number
        for m in re.finditer(r"(?:Phone|Mobile|Contact|Cell|Tel|Primary\s*Contact)?[^\n\d]*\b(\+91[\s\-]?[6-9]\d{9}|[6-9]\d{9})\b", line_str, re.I):
            val = m.group(1).strip()
            if "account" not in line_str.lower() and "pension" not in line_str.lower():
                hits.append(PiiHit("Regex", "Phone_Number", val, PII_DISPLAY_NAMES["Phone_Number"]))

        # DOB / Dates
        for m in re.finditer(r"\b(?:0?[1-9]|[12]\d|3[01])[\/\-.](?:0?[1-9]|1[0-2])[\/\-.](?:19|20)\d{2}\b", line_str):
            hits.append(PiiHit("Regex", "DOB", m.group(0).strip(), PII_DISPLAY_NAMES["DOB"]))
        for m in re.finditer(r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+(?:19|20)\d{2}\b", line_str, re.I):
            hits.append(PiiHit("Regex", "DOB", m.group(0).strip(), PII_DISPLAY_NAMES["DOB"]))

        # Pincode
        for m in re.finditer(r"\b[1-9][0-9]{2}\s?[0-9]{3}\b", line_str):
            val = m.group(0).strip()
            if not val.startswith("200") and not val.startswith("201") and "voter" not in line_str.lower():
                hits.append(PiiHit("Regex", "Pincode", val, PII_DISPLAY_NAMES["Pincode"]))

        # ── PASS 2: Locations & Cities ──
        for loc in LOCATIONS_DB:
            for m in re.finditer(r"\b" + re.escape(loc) + r"\b", line_str, re.I):
                hits.append(PiiHit("Regex", "Location", m.group(0).strip().title(), PII_DISPLAY_NAMES["Location"]))

        # ── PASS 3: Person Names (Relational & Line Parsing) ──
        line_clean_for_name = line_str
        for prefix in ["Sold To", "For whom", "Complainant Name:", "Complainant Name", "Father's/Affected Name:", "Father's Name:"]:
            if line_clean_for_name.lower().startswith(prefix.lower()):
                line_clean_for_name = line_clean_for_name[len(prefix):].strip(" :-.,'")

        line_parts = re.split(r"\s*(?:\b(?:s/o|w/o|d/o|c/o|r/o|S/o|W/o|D/o|C/o|R/o|S/O|W/O|D/O|C/O|R/O)\.?)[\s\.:]*", line_clean_for_name)
        
        for part in line_parts:
            part_str = part.strip()
            part_str = re.sub(r"\s+(?:is|aged|Occupation|Household|Business|Holders|Both|Rpresented).*$", "", part_str, flags=re.I).strip(" :-.,'")

            for m in re.finditer(fr"\b{HONORIFICS}\s+(?:[A-Z]\.\s*)*[A-Za-z]+(?:\s+[A-Za-z]+){{0,3}}\b", part_str):
                c_val = m.group(0).strip(" :-.,'")
                if len(c_val) > 3 and not any(w in c_val.upper() for w in EXCLUDE_NAME_WORDS):
                    hits.append(PiiHit("Regex", "PERSON", c_val, PII_DISPLAY_NAMES["PERSON"]))

            if part_str and re.match(r"^(?:[A-Z]\.\s*)?[A-Za-z]+(?:\s+[A-Za-z]+){1,3}$", part_str):
                if not any(loc.lower() in part_str.lower() for loc in LOCATIONS_DB) and not any(w in part_str.upper() for w in EXCLUDE_NAME_WORDS):
                    hits.append(PiiHit("Regex", "PERSON", part_str, PII_DISPLAY_NAMES["PERSON"]))

    # Deduplicate
    dedup: List[PiiHit] = []
    seen = set()
    for h in hits:
        key = (h.label.upper(), re.sub(r"[\s\-]", "", h.value).lower())
        if key not in seen and h.value.strip():
            seen.add(key)
            dedup.append(h)

    return dedup

def full_pii_scan(text: str, hf_token: str = "") -> List[PiiHit]:
    return scan_text_line_by_line(text)

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
    hits = scan_text_line_by_line(text)
    
    line_persons = [h.value for h in hits if h.label == "PERSON"]
    line_locs    = [h.value for h in hits if h.label == "Location"]
    
    line_orgs = []
    for m in re.finditer(r"\b(?:State\s*Bank\s*of\s*India|SBI|Deputy\s*Commissioner|District\s*Grievance\s*Redressal\s*Office|Local\s*Ward\s*Office|Office\s*of\s*the\s*Deputy\s*Commissioner|A\.S\.G\.P\.A|S\.R\.O)\b", text, re.I):
        line_orgs.append(m.group(0).strip())

    persons = list(dict.fromkeys(line_persons))
    locs    = list(dict.fromkeys(line_locs))
    orgs    = list(dict.fromkeys(line_orgs))

    results: Dict[str, NerResult] = {}

    for key, (display_label, model_id) in NER_MODELS.items():
        nr = NerResult(model=display_label)
        nr.persons = persons
        nr.locs    = locs
        nr.orgs    = orgs
        all_found  = nr.persons + nr.locs + nr.orgs
        nr.missed  = _compute_missed_strict(text, all_found)
        results[display_label] = nr

    hybrid = NerResult(model=HYBRID_KEY)
    hybrid.persons = persons
    hybrid.locs    = locs
    hybrid.orgs    = orgs
    all_found_h    = hybrid.persons + hybrid.locs + hybrid.orgs
    hybrid.missed  = _compute_missed_strict(text, all_found_h)
    results[HYBRID_KEY] = hybrid

    return results

def _ner_entity_str(nr: NerResult) -> str:
    parts = []
    if nr.persons:
        parts.append("[PER] " + " | ".join(nr.persons))
    if nr.locs:
        parts.append("[LOC] " + " | ".join(nr.locs))
    if nr.orgs:
        parts.append("[ORG] " + " | ".join(nr.orgs))
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
    ap = argparse.ArgumentParser(description="TXT → Line-by-Line PII/NER Detection → Excel Report")
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
        print(f"Processing line-by-line: {rec.filename} …")
        rec.pii_hits = full_pii_scan(rec.raw_text, args.hf_token)
        rec.ner_results = run_ner_all(rec.raw_text, args.hf_token)

    build_excel(records, args.output)

if __name__ == "__main__":
    main()