#!/usr/bin/env python3
"""NER comparison report over documents (replaces the old app/main.py command line).

    python app/ner_report.py --input <file or folder> [--output report.xlsx]
    python app/ner_report.py --text "Shri Ramesh Kumar, Aadhaar 2345 6789 0123"

Runs the rule scanner and each NER model separately and writes a 4-sheet Excel
workbook plus per-model JSON files. An evaluation tool; it redacts nothing.
The work is done by the `report` package.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from report.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
