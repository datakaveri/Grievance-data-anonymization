"""Step 2 — rule-based detection: run every regex rule over every line of a text.

    scan_text(text, options)  -> [PiiHit]   line-relative hits (used by the NER report)
    scan_cell(text, options)  -> [dict]     cell-relative detections (used by pipeline.py)

The rules themselves, and the order they run in, live in anonymizer/rules/.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .rules import ScanOptions, build_rules
from .rules.india import DEVANAGARI_DIGITS
from .text import line_bases, word_number

# Human-readable name of each PII type, shown in the NER report.
DISPLAY_NAMES: Dict[str, str] = {
    "Aadhaar": "Aadhaar Number",
    "PAN": "PAN Card",
    "Phone_Number": "Phone Number",
    "Email": "Email Address",
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
    """One value found by a rule, positioned within its (stripped) line.

    source      detector: "Regex", "Contextual Regex" or "Deny list"
    label       PII type, e.g. "Aadhaar"
    value       the matched text
    display     human-readable PII type (DISPLAY_NAMES)
    line_no     1-based line number within the text
    word_no     1-based index of the first word of the value within the line
    start_char  1-based offset of the first character within the stripped line
    end_char    offset just past the last character (so value = line[start_char-1:end_char])
    confidence  always 1.0 for rules; kept so rule and model hits share one shape
    """

    source: str
    label: str
    value: str
    display: str
    line_no: int = 0
    word_no: int = 0
    start_char: int = 0
    end_char: int = 0
    confidence: float = 1.0


def scan_text(text: str, options: Optional[ScanOptions] = None) -> List[PiiHit]:
    """Find every rule-detectable value in `text`, line by line.

    Each line is stripped, and Devanagari digits are read as ASCII digits so the
    numeric rules see "९८७६५४३२१०" as "9876543210" (a one-for-one translation, so
    offsets still index the original line). Rules run in the order of
    rules.build_rules(); a value claimed by several rules appears once per rule.
    """
    rules = build_rules(options or ScanOptions())
    hits: List[PiiHit] = []
    for line_idx, raw_line in enumerate(text.splitlines()):
        line = raw_line.strip().translate(DEVANAGARI_DIGITS)
        if not line:
            continue
        for rule in rules:
            for start, end in rule.find(line):
                hits.append(
                    PiiHit(
                        source=rule.source,
                        label=rule.label,
                        value=line[start:end].strip(),
                        display=DISPLAY_NAMES.get(rule.label, rule.label.replace("_", " ")),
                        line_no=line_idx + 1,
                        word_no=word_number(line, start),
                        start_char=start + 1,
                        end_char=end,
                    )
                )
    return _drop_duplicate_hits(hits)


def _drop_duplicate_hits(hits: List[PiiHit]) -> List[PiiHit]:
    """Keep the first hit per (line, label, position); drop empty values.

    Two rules of the same label claiming the same characters (the date rules
    can) collapse into one. Hits are NOT deduplicated by value: an earlier
    version did, so a value repeated in one line was redacted only the first
    time — "Call 9876543210 or 9876543210 again" became
    "Call * or 9876543210 again" (117 distinct lines in the Haryana grievance
    data had a repeated date, reference number or ID).
    """
    kept: List[PiiHit] = []
    seen: set = set()
    for hit in hits:
        key = (hit.line_no, hit.label, hit.start_char, hit.end_char)
        if key not in seen and hit.value.strip():
            seen.add(key)
            kept.append(hit)
    return kept


def scan_cell(text: str, options: Optional[ScanOptions] = None) -> List[Dict[str, Any]]:
    """scan_text() for one table cell, with offsets relative to the whole cell.

    Returns detection dicts in the shape the redaction step merges with model
    detections: {start, end, label, source, confidence, value}, zero-based,
    end exclusive.
    """
    bases = line_bases(text)
    detections: List[Dict[str, Any]] = []
    for hit in scan_text(text, options):
        if hit.line_no < 1 or hit.line_no > len(bases):
            continue
        base = bases[hit.line_no - 1]
        detections.append(
            {
                "start": base + max(hit.start_char - 1, 0),
                "end": base + hit.end_char,
                "label": hit.label,
                "source": hit.source,
                "confidence": hit.confidence,
                "value": hit.value,
            }
        )
    return detections
