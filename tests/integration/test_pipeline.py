import os
import sys
from pathlib import Path

# Add src/ to sys.path so grievance_anonymization is importable when run directly via python
current_file = Path(__file__).resolve()
repo_root = current_file.parents[2] if len(current_file.parents) > 2 else current_file.parent
src_path = str(repo_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import pytest
from grievance_anonymization.main import process_text_string


def test_inline_text_pipeline_execution(tmp_path):
    output_excel = str(tmp_path / "report.xlsx")
    text = ("WATER PIPE LINE LEAKAGE",
    "water pipe line leakage",
    "Water Pipe Line Leakage",
    "No Water Supply",
    "NO WATER SUPPLY",
    "no water supply",
    "PLS resolve my issue",
    "Shri Rajesh Kumar living in District Karnal Tehsil Gharaunda",
    "जिला करनाल में पानी की समस्या है",
    "PLS resolve my issue. WATER PIPE LINE LEAKAGE and No Water Supply in District Karnal. "
    "Contact Shri Rajesh Kumar at rajesh.kumar@example.com or phone 9876543210."
    )
    recs = process_text_string(text, output_path=output_excel)

    assert len(recs) == 1
    assert ("English" in recs[0].language or "Mixed" in recs[0].language or "Hindi" in recs[0].language)
    assert len(recs[0].pii_hits) >= 2
    assert os.path.exists(output_excel)
    assert os.path.exists(str(tmp_path / "report.json"))

    # Verify that Shri Rajesh Kumar is detected as PERSON
    all_ents = [
        e for m_nets in recs[0].line_ners.values()
        for l_res in m_nets
        for e in l_res.entities
    ]
    person_texts = [e.text for e in all_ents if e.category == "PERSON"]
    assert any("Rajesh Kumar" in p for p in person_texts), f"Expected Rajesh Kumar in PERSON entities, got: {person_texts}"

    # Verify that WATER PIPE LINE LEAKAGE is not detected as ORGANIZATION
    org_texts = [e.text for e in all_ents if e.category == "ORGANIZATION"]
    assert not any("WATER" in o.upper() and "PIPE" in o.upper() for o in org_texts), f"Unexpected ORG detected for water pipe line: {org_texts}"


if __name__ == "__main__":
    output_dir = repo_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    test_inline_text_pipeline_execution(output_dir)
    print("✅ test_inline_text_pipeline_execution passed successfully!")
    print(f"📊 Output Excel report saved to: {output_dir / 'report.xlsx'}")
    print(f"📊 Output JSON report saved to:  {output_dir / 'report.json'}")
