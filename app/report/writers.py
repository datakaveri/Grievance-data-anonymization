"""Write the NER report: a 4-sheet Excel workbook and JSON files (combined + per model)."""

from __future__ import annotations

import json
import os
from typing import Dict, List

import openpyxl
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from anonymizer.ner import models

from .records import FileRecord, ordinal

# ─────────────────────────────────────────────────────────────────────────────
# JSON
# ─────────────────────────────────────────────────────────────────────────────


def _file_info(rec: FileRecord) -> Dict:
    return {"filename": rec.filename, "file_type": rec.file_type.upper(), "language": rec.language, "path": rec.path}


def _pii_rows(rec: FileRecord) -> List[Dict]:
    """The record's rule hits as report rows (position, detector, type, value, tag)."""
    return [
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
    ]


def _ner_rows(rec: FileRecord, model_label: str) -> List[Dict]:
    """One model's (or the hybrid's) entities in the record, as report rows."""
    return [
        {
            "line_no": lres.line_no,
            "word_no": ent.word_no,
            "word_position": ordinal(ent.word_no),
            "start_letter": ent.start_char,
            "end_letter": ent.end_char,
            "letter_span": f"{ent.start_char}-{ent.end_char}",
            "source": model_label,
            "type": ent.category,
            "original_value": ent.text,
            "confidence_score": ent.score,
            "tag": f"<{ent.category}>{ent.text}</{ent.category}>",
        }
        for lres in rec.line_ners.get(model_label, [])
        for ent in lres.entities
    ]


def build_json(records: List[FileRecord], output_path: str) -> None:
    """Write <output>.json (rule hits + hybrid NER) and <output>_<model>.json per model.

    For output "report.xlsx": report.json, report_IndicNER.json,
    report_XLM_RoBERTa.json, report_Hybrid.json.
    """
    base_name, _ = os.path.splitext(output_path)
    print(f"\n▶ Writing JSON reports → {base_name}_*.json …", flush=True)

    combined = [
        {"file_info": _file_info(rec), "pii_detections": _pii_rows(rec), "ner_detections": _ner_rows(rec, models.HYBRID_LABEL)}
        for rec in records
    ]
    with open(base_name + ".json", "w", encoding="utf-8") as fh:
        json.dump(combined, fh, indent=2, ensure_ascii=False)
    print(f"  ✅ Main combined JSON report generated → {base_name}.json", flush=True)

    slugs = {spec.label: spec.key for spec in models.NER_MODELS}
    slugs[models.HYBRID_LABEL] = "Hybrid"
    for model_label, slug in slugs.items():
        per_model = [
            {
                "file_info": _file_info(rec),
                "model_name": model_label,
                "pii_detections": _pii_rows(rec),
                "ner_detections": _ner_rows(rec, model_label),
            }
            for rec in records
        ]
        path = f"{base_name}_{slug}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(per_model, fh, indent=2, ensure_ascii=False)
        print(f"  ✅ Separate JSON generated for {slug} → {path}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Excel
# ─────────────────────────────────────────────────────────────────────────────

_FILLS = {
    name: PatternFill(start_color=color, end_color=color, fill_type="solid")
    for name, color in {
        "header": "1F4E79",
        "pii_hit": "FFE0E0",
        "language": "FFF9C4",
        "model_row": "E3F2FD",
        "no_pii": "C8E6C9",
    }.items()
}
_HEADER_FONT = Font(color="FFFFFF", bold=True, size=10)
_WRAP = Alignment(wrap_text=True, vertical="top")
_CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
_SHEETS = ("Summary", "PII Detection", "NER Comparison", "Anonymization")


def _line_of(rec: FileRecord, line_no: int) -> str:
    """Text of line `line_no` (1-based) of the record, or the whole text if out of range."""
    lines = rec.raw_text.splitlines()
    return lines[line_no - 1] if 0 < line_no <= len(lines) else rec.raw_text


def _summary_rows(records: List[FileRecord]) -> List[Dict]:
    """Sheet 1: per file — language, PII counts/types, unique hybrid entities."""
    rows = []
    for rec in records:
        hybrid = rec.line_ners.get(models.HYBRID_LABEL, [])
        unique = lambda attr: list(dict.fromkeys(x for lres in hybrid for x in getattr(lres, attr)))  # noqa: E731
        rows.append(
            {
                "File": rec.filename,
                "File Type": rec.file_type.upper(),
                "Language": rec.language,
                "Total PII Entities": len(rec.pii_hits),
                "PII Types Found": ", ".join(sorted({h.label for h in rec.pii_hits})) or "None",
                "Hybrid NER Persons": " | ".join(unique("persons")) or "None",
                "Hybrid NER Locations": " | ".join(unique("locs")) or "None",
                "Hybrid NER Organizations": " | ".join(unique("orgs")) or "None",
                "Has PII": "YES" if rec.pii_hits else "NO",
            }
        )
    return rows


def _pii_sheet_rows(records: List[FileRecord]) -> List[Dict]:
    """Sheet 2: one row per rule hit (or one "None Detected" row per clean file)."""
    rows = []
    for rec in records:
        base = {"File": rec.filename, "File Type": rec.file_type.upper(), "Language": rec.language}
        if not rec.pii_hits:
            rows.append({**base, "Line No": "-", "Word No": "-", "Word Position": "-", "Start Letter": "-",
                         "End Letter": "-", "Letter Span": "-", "Extracted Text": rec.raw_text,
                         "PII Type": "None", "PII Display Name": "None", "Detected Value": "None Detected",
                         "Detected By": "None", "PII Tag": "—"})
            continue
        for h in rec.pii_hits:
            rows.append({**base, "Line No": h.line_no, "Word No": h.word_no, "Word Position": ordinal(h.word_no),
                         "Start Letter": h.start_char, "End Letter": h.end_char,
                         "Letter Span": f"{h.start_char}-{h.end_char}", "Extracted Text": _line_of(rec, h.line_no),
                         "PII Type": h.label, "PII Display Name": h.display, "Detected Value": h.value,
                         "Detected By": h.source, "PII Tag": f"<{h.label}>{h.value}</{h.label}>"})
    return rows


def _comparison_rows(records: List[FileRecord]) -> List[Dict]:
    """Sheet 3: one row per line — what each model and the hybrid found, and what each missed."""
    labels = [spec.label for spec in models.NER_MODELS] + [models.HYBRID_LABEL]
    rows = []
    for rec in records:
        base = {"File": rec.filename, "File Type": rec.file_type.upper(), "Language": rec.language}
        hybrid = rec.line_ners.get(models.HYBRID_LABEL, [])
        if not hybrid:
            row = {**base, "Line No": 1, "Input_Text": rec.raw_text, "Has_Context": len(rec.raw_text.split()) > 2}
            for label in labels:
                row.update({f"{label}_Predicted_Type": "NONE", f"{label}_Extracted_Text": "None",
                            f"{label}_Missed_Entities": "None", f"{label}_Status": "✗ None Detected"})
            rows.append(row)
            continue
        for idx, sample in enumerate(hybrid):
            row = {**base, "Line No": sample.line_no, "Input_Text": sample.text,
                   "Has_Context": len(sample.text.split()) > 2}
            for label in labels:
                lres = rec.line_ners[label][idx]
                found = lres.persons or lres.locs or lres.orgs
                row.update({
                    f"{label}_Predicted_Type": lres.predicted_type,
                    f"{label}_Extracted_Text": lres.extracted_text,
                    f"{label}_Missed_Entities": " | ".join(lres.missed) if lres.missed else "None",
                    f"{label}_Status": "✓ Detected" if found else "✗ None Detected",
                })
            rows.append(row)
    return rows


def _detection_rows(records: List[FileRecord]) -> List[Dict]:
    """Sheet 4: every rule hit and every hybrid entity with its position and confidence."""
    display = {"PERSON": "Person Name", "LOCATION": "Location / City", "ORGANIZATION": "Organization"}
    rows = []
    for rec in records:
        base = {"File": rec.filename, "File Type": rec.file_type.upper(), "Language": rec.language}
        for h in rec.pii_hits:
            rows.append({**base, "Line No": h.line_no, "Word No": h.word_no, "Word Position": ordinal(h.word_no),
                         "Start Letter": h.start_char, "End Letter": h.end_char,
                         "Letter Span": f"{h.start_char}-{h.end_char}", "Extracted Text": _line_of(rec, h.line_no),
                         "Source Type": "PII", "Detection Engine": h.source, "Entity / PII Type": h.display,
                         "Original Value": h.value, "Confidence Score": "1.00 (Exact Regex)",
                         "PII / NER Tag": f"<{h.label}>{h.value}</{h.label}>"})
        seen = set()
        for lres in rec.line_ners.get(models.HYBRID_LABEL, []):
            for ent in lres.entities:
                key = (lres.line_no, ent.category, ent.text.lower())
                if key in seen:
                    continue
                seen.add(key)
                rows.append({**base, "Line No": lres.line_no, "Word No": ent.word_no,
                             "Word Position": ordinal(ent.word_no), "Start Letter": ent.start_char,
                             "End Letter": ent.end_char, "Letter Span": f"{ent.start_char}-{ent.end_char}",
                             "Extracted Text": lres.text, "Source Type": "NER Model",
                             "Detection Engine": models.HYBRID_LABEL,
                             "Entity / PII Type": display.get(ent.category, "Organization"),
                             "Original Value": ent.text, "Confidence Score": f"{ent.score:.4f}",
                             "PII / NER Tag": f"<{ent.category}>{ent.text}</{ent.category}>"})
    return rows


def _style_sheet(ws) -> None:
    """Header colours, frozen header row, wrapped cells, value-dependent fills, column widths."""
    for cell in ws[1]:
        cell.fill = _FILLS["header"]
        cell.font = _HEADER_FONT
        cell.alignment = _CENTER
    ws.freeze_panes = "A2"
    ws.row_dimensions[1].height = 30
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = _WRAP
            header = str(ws.cell(1, cell.column).value or "")
            value = str(cell.value or "")
            if header == "Language":
                cell.fill = _FILLS["language"]
            elif header == "Has PII":
                cell.fill = _FILLS["pii_hit"] if value == "YES" else _FILLS["no_pii"] if value == "NO" else cell.fill
            elif header == "Source Type" and value == "NER Model":
                cell.fill = _FILLS["model_row"]
            elif "Original Value" in header:
                cell.fill = _FILLS["pii_hit"]
    for column_cells in ws.columns:
        widest = max((len(str(c.value or "")) for c in column_cells), default=10)
        ws.column_dimensions[get_column_letter(column_cells[0].column)].width = min(widest + 4, 55)


def build_excel(records: List[FileRecord], output_path: str) -> None:
    """Write the 4-sheet workbook: Summary, PII Detection, NER Comparison, Anonymization."""
    print(f"\n▶ Writing 4-Sheet Excel report → {output_path} …", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    sheets = dict(zip(_SHEETS, (_summary_rows(records), _pii_sheet_rows(records),
                                _comparison_rows(records), _detection_rows(records))))
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for name, rows in sheets.items():
            pd.DataFrame(rows).to_excel(writer, sheet_name=name, index=False)
    workbook = openpyxl.load_workbook(output_path)
    for name in _SHEETS:
        _style_sheet(workbook[name])
    workbook.save(output_path)
    print(f"  ✅ Excel report generated successfully → {output_path}", flush=True)
