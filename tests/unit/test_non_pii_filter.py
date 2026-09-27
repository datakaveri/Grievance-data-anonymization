import sys
from pathlib import Path

src_path = str(Path(__file__).resolve().parent.parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import pytest
from grievance_anonymization.main import is_non_pii_match, prepare_non_pii_rules, PiiHit, filter_non_pii_hits


def test_non_pii_match_words():
    rules = prepare_non_pii_rules()
    assert is_non_pii_match("PLS", "ORG", rules) is True
    assert is_non_pii_match("PLEASE", "ALL", rules) is True
    assert is_non_pii_match("महोदय", "ALL", rules) is True


def test_non_pii_filter_hits():
    hits = [
        PiiHit("Regex", "Email", "test@example.com", "Email Address"),
        PiiHit("Regex", "Phone_Number", "9876543210", "Phone Number"),
    ]
    filtered = filter_non_pii_hits(hits)
    assert len(filtered) == 2
