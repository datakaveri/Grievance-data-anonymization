"""The nested-JSON job: anonymize the remarks fields of nested JSON records.

    run(inputs, out_dir, settings)        (CLI: app/json_remarks.py)

A complaint exported as nested JSON keeps its free text at varying depths
($.remarks, $.closing_remark, $.activity_log_details[3].extra_remarks, ...),
so instead of configured columns this job takes every string field whose key
contains "remark". Keys ending in "_by" hold IDs ("closing_remark_by"), not
text, and are skipped. The fields of all input files go through
pipeline.anonymize() together, so the models make one pass over all of them.

  read     load each file, find its remark fields
  run      (key, text) of every field -> pipeline.anonymize()
  write    per input file:
             <stem>_anonymized.json   the same structure, remarks redacted
             <stem>_ner_report.json   per field: path, original, redacted text, detections
           The report holds the original text, so treat it like the input.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple

from ..config import DetectionSettings, free_text_block, load_dataset_config
from ..pipeline import anonymize

_REMARK_KEY = re.compile(r"remark", re.I)

# Cutoff when no dataset config is given, matching the grievance configs.
DEFAULT_MINIMUM_CONFIDENCE = 0.75


def settings_from(config: Optional[str]) -> DetectionSettings:
    """Detection settings of a dataset config, or the defaults with a 0.75 cutoff.

    Reusing a dataset config brings its cutoff, allow-list and deny-list along;
    a config without minimum_confidence also gets 0.75.
    """
    if not config:
        return DetectionSettings(minimum_confidence=DEFAULT_MINIMUM_CONFIDENCE)
    block = free_text_block(load_dataset_config(Path(config)))
    settings = DetectionSettings.from_block(block)
    if "minimum_confidence" not in block:
        settings.minimum_confidence = DEFAULT_MINIMUM_CONFIDENCE
    return settings


def remark_fields(obj: Any, path: str = "$") -> List[Tuple[str, Any, Any]]:
    """(JSON path, container, key) for every non-empty string under a *remark* key.

    The container and key let the caller replace the value in place. Example
    result entry: ("$.activity_log_details[2].extra_remarks", <that dict>, "extra_remarks").
    """
    found: List[Tuple[str, Any, Any]] = []
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
                found.extend(remark_fields(value, here))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            found.extend(remark_fields(value, f"{path}[{i}]"))
    return found


def run(inputs: List[Path], out_dir: Path, settings: DetectionSettings) -> None:
    """Anonymize the remarks of every file in `inputs`, writing results to `out_dir`."""
    # ── read ──────────────────────────────────────────────────────────────
    records = []
    for path in inputs:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        records.append((path, data, remark_fields(data)))

    # ── run ───────────────────────────────────────────────────────────────
    # The field's key is the "column": it picks the hash salt.
    results = anonymize(
        ((key, container[key]) for _, _, fields in records for _, container, key in fields),
        settings,
        include_values=True,
    )

    # ── write ─────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    for path, data, fields in records:
        report = []
        for field_path, container, key in fields:
            result = results[(key, container[key])]
            if result.error is not None:
                raise RuntimeError(f"{path.name} {field_path}: {result.error}")
            report.append(
                {
                    "path": field_path,
                    "original": result.cleaned,
                    "anonymized": result.text,
                    "detections": [
                        {
                            "label": a["label"],
                            "text": a["detected_text"],
                            "source": a["source"],
                            "confidence": round(float(a["confidence"]), 4),
                            "technique": a["technique"],
                        }
                        for a in result.audit
                    ],
                }
            )
            container[key] = result.text
        with (out_dir / f"{path.stem}_anonymized.json").open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        with (out_dir / f"{path.stem}_ner_report.json").open("w", encoding="utf-8") as fh:
            json.dump({"file": str(path), "remark_fields": len(fields), "fields": report}, fh, indent=2, ensure_ascii=False)
        hits = sum(len(r["detections"]) for r in report)
        print(f"[json_remarks] {path.name}: {len(fields)} remark fields, {hits} detections -> {out_dir}")
