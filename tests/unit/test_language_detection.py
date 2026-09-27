import sys
from pathlib import Path

src_path = str(Path(__file__).resolve().parent.parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import pytest
from grievance_anonymization.main import detect_language


def test_detect_english():
    assert detect_language("This is a simple grievance report in English.") == "English"


def test_detect_hindi():
    assert "Hindi" in detect_language("जिला करनाल में पानी की समस्या है")


def test_detect_unknown():
    assert detect_language("") == "Unknown"
    assert detect_language("1234567890") == "Unknown"
