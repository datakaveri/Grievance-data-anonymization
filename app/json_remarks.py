#!/usr/bin/env python3
"""Anonymize every remarks field of nested JSON records (e.g. one complaint per file).

batch_pipeline.py reads tables; a complaint exported as nested JSON keeps its
free text at varying depths ($.remarks, $.closing_remark,
$.activity_log_details[3].extra_remarks, ...). This finds every string field
whose key contains "remark" and runs the same steps on it: source-text
normalisation, regex PII rules, hybrid NER with the false-positive filters, and
the same redaction policy. Keys ending in "_by" hold IDs, not text, and are skipped.

Writes, per input file, <stem>_anonymized.json (same structure, remarks
redacted) and <stem>_ner_report.json (each field: path, original, detections,
redacted text). The report holds the original text, so treat it like the input.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    import batch_pipeline as bp
except ModuleNotFoundError:  # python -m app.json_remarks
    from . import batch_pipeline as bp

_REMARK_KEY = re.compile(r"remark", re.I)


def _remark_fields(obj: Any, path: str = "$") -> list[tuple[str, Any, Any]]:
    """(path, container, key) for every non-empty string under a *remark* key."""
    found = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            here = f"{path}.{key}"
            if (
                isinstance(value, str)
                and value.strip()
                and _REMARK_KEY.search(key)
                and not key.lower().endswith("_by")
            ):
                found.append((here, obj, key))
            else:
                found.extend(_remark_fields(value, here))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            found.extend(_remark_fields(value, f"{path}[{i}]"))
    return found


def _settings(config: str | None) -> dict[str, Any]:
    """free_text_anonymization block of a dataset config, or the defaults."""
    if not config:
        return {"minimum_confidence": 0.75}
    with open(config, encoding="utf-8") as fh:
        cfg = json.load(fh)
    return cfg[cfg["data_type"]].get("free_text_anonymization", {})


def run(inputs: list[Path], out_dir: Path, settings: dict[str, Any]) -> None:
    minimum_confidence = float(settings.get("minimum_confidence", 0.75))
    scan_options = {
        key: settings[key]
        for key in ("reference_keywords", "strong_reference_keywords")
        if key in settings
    }
    if settings.get("ner_deny_terms"):
        scan_options["deny_terms"] = settings["ner_deny_terms"]

    records = []
    for path in inputs:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        fields = _remark_fields(data)
        for _, container, key in fields:
            if settings.get("normalize_source_text", True):
                container[key] = bp._normalize_source_text(container[key])
        records.append((path, data, fields))

    # One NER pass over every remark of every file, as the batch job does per table.
    texts = sorted({container[key] for _, _, fields in records for _, container, key in fields})
    index = bp._build_ner_index(
        texts, bp.NER_CORPUS_BATCH_SIZE, allow_terms=settings.get("ner_allow_terms")
    ) if texts else {}

    out_dir.mkdir(parents=True, exist_ok=True)
    salts: dict[str, str] = {}
    for path, data, fields in records:
        report = []
        for field_path, container, key in fields:
            original = container[key]
            redacted, audit = bp._sanitize(
                original,
                minimum_confidence,
                key,
                salts,
                index.get(original, []),
                include_values=True,
                scan_options=scan_options,
            )
            container[key] = redacted
            report.append(
                {
                    "path": field_path,
                    "original": original,
                    "anonymized": redacted,
                    "detections": [
                        {
                            "label": a["label"],
                            "text": a["detected_text"],
                            "source": a["source"],
                            "confidence": round(float(a["confidence"]), 4),
                            "technique": a["technique"],
                        }
                        for a in audit
                    ],
                }
            )
        stem = path.stem
        with (out_dir / f"{stem}_anonymized.json").open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        with (out_dir / f"{stem}_ner_report.json").open("w", encoding="utf-8") as fh:
            json.dump(
                {"file": str(path), "remark_fields": len(fields), "fields": report},
                fh,
                indent=2,
                ensure_ascii=False,
            )
        hits = sum(len(r["detections"]) for r in report)
        print(f"[json_remarks] {path.name}: {len(fields)} remark fields, {hits} detections -> {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Anonymize the remarks fields of JSON records")
    parser.add_argument("inputs", nargs="+", help="JSON files, or directories of them")
    parser.add_argument("--out", default="output/json_remarks", help="Output directory")
    parser.add_argument(
        "--config",
        default=None,
        help="Dataset config whose free_text_anonymization settings to reuse "
        "(cutoff, allow/deny lists); defaults to a 0.75 cutoff and no lists",
    )
    args = parser.parse_args()
    inputs: list[Path] = []
    for item in args.inputs:
        p = Path(item)
        inputs.extend(sorted(p.glob("*.json")) if p.is_dir() else [p])
    if not inputs:
        print("[json_remarks] no JSON input found", file=sys.stderr)
        return 1
    run(inputs, Path(args.out), _settings(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
