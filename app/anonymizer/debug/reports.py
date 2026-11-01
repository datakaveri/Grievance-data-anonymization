"""Optional debug reports of the table job: what was detected, by whom, and at what cost.

Turned on with free_text_anonymization.debug.enabled (or ANON_DEBUG=1). Three
reports, all written next to the audit file and sharing its stem:

  <audit>.debug_detections.json / .txt   per row: each detection and its detector
  <audit>.debug_ner_terms.json            most-redacted model strings (allow-list help)
  <audit>.debug_profile.json              time and peak memory per step and per model

With debug.include_values (default on), the reports contain the detected
personal data itself. Never enable that where the data must not leave, e.g. in
the TEE.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from ..ner.filters import ner_term_key


def debug_settings(block: Dict[str, Any]) -> Dict[str, Any]:
    """Read the optional `debug` object of a free_text_anonymization block.

    ANON_DEBUG=1/true/yes turns debugging on without editing the config. Default
    report paths share the audit file's stem (or the staged CSV's), so configs
    writing different outputs also get different debug reports.
    """
    debug = block.get("debug")
    if not isinstance(debug, dict):
        debug = {}
    enabled = bool(debug.get("enabled", False)) or os.getenv("ANON_DEBUG", "").strip() in {"1", "true", "yes"}
    base = Path(block.get("audit_output_path") or block.get("staged_input_path") or "output/debug")
    prefix = str(base.with_suffix("")) + "."
    return {
        "enabled": enabled,
        "include_values": bool(debug.get("include_values", True)),
        "profile_output_path": debug.get("profile_output_path", prefix + "debug_profile.json"),
        "attribution_output_path": debug.get("attribution_output_path", prefix + "debug_detections.json"),
        "attribution_text_path": debug.get("attribution_text_path", prefix + "debug_detections.txt"),
        "ner_terms_path": debug.get("ner_terms_path", prefix + "debug_ner_terms.json"),
        "max_rows": int(debug.get("max_rows", 0)),
    }


def _model_short_name(label: str) -> str:
    """'IndicNER (AI4Bharat / Public)' -> 'IndicNER', for compact per-row lines."""
    return label.split(" (")[0].strip()


def write_ner_terms(audit: List[Dict[str, Any]], path: Path, top: int = 300) -> None:
    """Write the `top` most-redacted model-detected strings, to curate ner_allow_terms.

    Local terms that are not personal ("ATR", "Jan Samvaad") rise to the top
    because they recur across many rows — but so do officials' names, which must
    NOT be allow-listed: read the list, never copy it whole. Each entry carries
    its allow_key (filters.ner_term_key), i.e. how an allow-list entry would match.
    """
    counts: Dict[tuple, int] = {}
    for item in audit:
        if item.get("source") == "Hybrid NER" and "detected_text" in item:
            key = (item["label"], item["detected_text"].strip())
            counts[key] = counts.get(key, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])[:top]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(
            [{"text": text, "label": label, "redactions": n, "allow_key": ner_term_key(text)}
             for (label, text), n in ranked],
            fh, indent=2, ensure_ascii=False,
        )
    print(f"[Batch] Wrote NER term frequencies: {path}")


def write_attribution(
    audit: List[Dict[str, Any]],
    json_path: Path,
    text_path: Path,
    include_values: bool,
    max_rows: int,
) -> None:
    """Write, per row, every detection and which detector(s) produced it.

    The JSON is grouped by row for diffing two runs programmatically; the text
    file is for reading ("why did row 12 come out like that?"):

        Row 2
            [remarks] PERSON 'Amit Sharma'  <- XLM-RoBERTa, IndicNER  (conf 1.00, suppress)
            [remarks] Phone_Number '98765 43210'  <- Regex  (conf 1.00, suppress)

    Rule hits name their rule source; model hits name the model(s) whose raw
    spans overlap the entity. max_rows > 0 limits how many rows are listed.
    """
    by_row: Dict[int, List[Dict[str, Any]]] = {}
    for item in audit:
        by_row.setdefault(item["row"], []).append(item)
    rows = sorted(by_row)
    if max_rows > 0:
        rows = rows[:max_rows]

    payload = {
        "rows_reported": len(rows),
        "rows_with_detections": len(by_row),
        "detections": len(audit),
        "values_included": include_values,
        "rows": [
            {
                "row": row,
                "detections": [
                    {
                        "column": it["column"],
                        "label": it["label"],
                        "detected_by": [m["model"] for m in it["models"]] if it.get("models") else [it["source"]],
                        "confidence": round(float(it["confidence"]), 4),
                        "technique": it["technique"],
                        "offsets": [it["start_offset"], it["end_offset"]],
                        **({"text": it["detected_text"]} if "detected_text" in it else {}),
                    }
                    for it in by_row[row]
                ],
            }
            for row in rows
        ],
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    lines = [
        f"Detections per row - {len(audit)} across {len(by_row)} rows"
        + (f" (showing first {len(rows)})" if max_rows > 0 and len(rows) < len(by_row) else ""),
        "Detected PII text included (debug.include_values is on)."
        if include_values
        else "PII values omitted; set debug.include_values to add them.",
        "",
    ]
    for row in rows:
        lines.append(f"Row {row}")
        for it in by_row[row]:
            who = ", ".join(_model_short_name(m["model"]) for m in it["models"]) if it.get("models") else it["source"]
            shown = f" {it['detected_text']!r}" if "detected_text" in it else ""
            lines.append(
                f"    [{it['column']}] {it['label']}{shown}"
                f"  <- {who}  (conf {float(it['confidence']):.2f}, {it['technique']})"
            )
        lines.append("")
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Batch] Wrote detection attribution: {json_path}")
    print(f"[Batch] Wrote detection attribution: {text_path}")


def model_load_summary(profiler) -> Dict[str, Dict[str, Any]]:
    """Regroup profiled steps into per-model load / infer / release cost.

    Separates the fixed price of getting a model into memory from the per-line
    price of running it (ms_per_line), with each model's own peak RSS.
    """
    summary: Dict[str, Dict[str, Any]] = {}
    for step in profiler.steps:
        for phase in ("load", "infer", "release"):
            prefix = f"{phase}: "
            if not step.name.startswith(prefix):
                continue
            entry = summary.setdefault(step.name[len(prefix):], {})
            entry[f"{phase}_seconds"] = round(step.seconds, 3)
            entry[f"{phase}_peak_rss_mb"] = round(step.rss_peak / 1e6, 1)
            entry[f"{phase}_rss_delta_mb"] = round((step.rss_end - step.rss_start) / 1e6, 1)
            if phase == "infer" and step.detail.get("lines"):
                lines = step.detail["lines"]
                entry["lines"] = lines
                entry["ms_per_line"] = round(step.seconds / lines * 1000, 2)
    return summary
