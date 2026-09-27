#!/usr/bin/env python3
import sys
import os
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from grievance_anonymization.main import (
    process_text_string,
    scan_text_line_by_line,
    _load_ner_pipeline,
    NER_MODELS,
    _extract_raw_spans,
    truecase_line,
    merge_line_spans,
)

test_phrases = [
    "WATER PIPE LINE LEAKAGE",
    "water pipe line leakage",
    "Water Pipe Line Leakage",
    "No Water Supply",
    "NO WATER SUPPLY",
    "no water supply",
    "PLS resolve my issue",
    "Shri Rajesh Kumar living in District Karnal Tehsil Gharaunda",
    "जिला करनाल में पानी की समस्या है",
    "Tehsil Gharaunda me paani ki samasya hai",
]

print("=========================================================")
print("VERIFICATION RESULTS FOR USER TEST CASES")
print("=========================================================\n")

for phrase in test_phrases:
    print(f"\nTesting phrase: '{phrase}'")
    recs = process_text_string(phrase, output_path="test_out.xlsx")
    for r in recs:
        print(f"  Detected Language: {r.language}")
        print(f"  PII Hits (Regex): {[(h.label, h.value) for h in r.pii_hits]}")
        for m_name, l_results in r.line_ners.items():
            for lr in l_results:
                if lr.entities:
                    ents_str = [(e.category, e.text, e.score) for e in lr.entities]
                    print(f"  {m_name}: {ents_str}")
