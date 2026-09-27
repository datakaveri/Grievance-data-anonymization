import sys
from pathlib import Path

src_path = str(Path(__file__).resolve().parent.parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import pytest
from grievance_anonymization.main import scan_text_line_by_line, anonymize_value, truecase_line


def test_email_pii_detection():
    text = "Please contact me at test.user@example.com for further updates."
    hits = scan_text_line_by_line(text)
    email_hits = [h for h in hits if h.label == "Email"]
    assert len(email_hits) == 1
    assert email_hits[0].value == "test.user@example.com"


def test_aadhaar_pii_detection():
    text = "Aadhaar number is 2345 6789 0123."
    hits = scan_text_line_by_line(text)
    aadhaar_hits = [h for h in hits if h.label == "Aadhaar"]
    assert len(aadhaar_hits) == 1
    assert aadhaar_hits[0].value == "2345 6789 0123"


def test_pan_pii_detection():
    text = "My PAN card details: ABCDE1234F."
    hits = scan_text_line_by_line(text)
    pan_hits = [h for h in hits if h.label == "PAN"]
    assert len(pan_hits) == 1
    assert pan_hits[0].value == "ABCDE1234F"


def test_anonymize_value_aadhaar():
    strategy, masked, desc = anonymize_value("Aadhaar", "2345 6789 0123")
    assert strategy == "Partial Masking"
    assert masked == "XXXX XXXX 0123"


def test_anonymize_value_email():
    strategy, masked, desc = anonymize_value("Email", "john.doe@domain.com")
    assert strategy == "Domain-Preserving Masking"
    assert masked == "jo******@domain.com"


def test_truecase_line():
    text = "WATER PIPE LINE LEAKAGE"
    tc = truecase_line(text)
    assert tc == "Water Pipe Line Leakage"
    assert len(tc) == len(text)
