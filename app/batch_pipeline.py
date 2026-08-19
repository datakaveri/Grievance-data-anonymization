#!/usr/bin/env python3
"""Config-driven, column-scoped dataset anonymization batch job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from main import (
        FileRecord,
        PiiHit,
        detect_language,
        full_pii_scan,
        merge_pii_hits,
        run_line_by_line_ner,
        _TRANSFORMERS_AVAILABLE,
    )
except ModuleNotFoundError:  # Supports `python -m app.batch_pipeline` too.
    from .main import (
        FileRecord,
        PiiHit,
        detect_language,
        full_pii_scan,
        merge_pii_hits,
        run_line_by_line_ner,
        _TRANSFORMERS_AVAILABLE,
    )

REDACTED = "*"


class BatchError(RuntimeError):
    """An expected, user-actionable batch failure."""


def _path(value: str, root: Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else root / candidate


def _load_config(config_path: Path) -> dict[str, Any]:
    try:
        with config_path.open(encoding="utf-8") as fh:
            config = json.load(fh)
    except Exception as exc:
        raise BatchError(f"Could not read config {config_path}: {exc}") from exc
    if not isinstance(config, dict) or not isinstance(config.get("data_type"), str):
        raise BatchError("Config must contain a string field: data_type")
    dataset = config.get(config["data_type"])
    if not isinstance(dataset, dict):
        raise BatchError(f"Config has no object for data_type={config['data_type']!r}")
    return dataset


def _read_table(path: Path) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        raise BatchError(f"Source input does not exist: {path}")
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(path, dtype=object, keep_default_na=False), "csv"
        if suffix in {".xls", ".xlsx"}:
            return pd.read_excel(path, dtype=object), suffix[1:]
        if suffix == ".json":
            with path.open(encoding="utf-8") as fh:
                payload = json.load(fh)
            if isinstance(payload, list):
                return pd.DataFrame(payload), "json"
            if isinstance(payload, dict):
                # Accept either a records container or a column-oriented object.
                records = payload.get("records", payload.get("data"))
                if isinstance(records, list):
                    return pd.DataFrame(records), "json"
                return pd.DataFrame(payload), "json"
    except Exception as exc:
        raise BatchError(f"Could not read {path}: {exc}") from exc
    raise BatchError(f"Unsupported input format {path.suffix!r}; use CSV, JSON, XLS, or XLSX")


_SUPPORTED_SOURCE_EXTS = {".csv", ".json", ".xls", ".xlsx"}


def _resolve_source_input(dataset: dict[str, Any], work_root: Path) -> Path:
    """Locate the raw dataset to anonymize.

    `source_input_path` is an optional convenience for local/manual runs. The
    config shared with SKALD downstream has no such key — SKALD locates its
    input by scanning `data/` for exactly one file (its `list_non_empty_csvs`),
    so when the key is absent we mirror that same convention against the same
    mounted `data/` directory instead of requiring a path in config.
    """
    configured = dataset.get("source_input_path")
    if configured:
        return _path(configured, work_root)

    data_dir = work_root / "data"
    if not data_dir.is_dir():
        raise BatchError(f"Data directory not found: {data_dir}")
    candidates = sorted(
        p for p in data_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _SUPPORTED_SOURCE_EXTS and p.stat().st_size > 0
    )
    if not candidates:
        raise BatchError(f"No CSV, JSON, or Excel file found in {data_dir}")
    if len(candidates) > 1:
        raise BatchError(
            f"Expected exactly one input file in {data_dir}, found {len(candidates)}: "
            f"{[p.name for p in candidates]}"
        )
    return candidates[0]


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write beside the destination and replace it only after the whole batch succeeds.
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=path.suffix, delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        df.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _cell_hits(text: str) -> list[PiiHit]:
    regex_hits, presidio_hits = full_pii_scan(text)
    return merge_pii_hits(regex_hits, presidio_hits)


def _absolute_hits(text: str) -> list[tuple[int, int, PiiHit]]:
    """Convert line-relative hit offsets into cell-relative zero-based offsets."""
    hits = _cell_hits(text)
    lines = text.splitlines(keepends=True)
    starts: list[int] = []
    cursor = 0
    for line in lines:
        starts.append(cursor)
        cursor += len(line)
    result = []
    for hit in hits:
        if hit.line_no < 1 or hit.line_no > len(starts):
            continue
        raw_line = lines[hit.line_no - 1]
        # Detection runs on line.strip(), so restore the removed leading space.
        leading = len(raw_line) - len(raw_line.lstrip())
        start = starts[hit.line_no - 1] + leading + max(hit.start_char - 1, 0)
        end = starts[hit.line_no - 1] + leading + hit.end_char
        result.append((start, end, hit))
    return result


def _hardcoded_policy(label: str, value: str, column: str, salts: dict[str, str]) -> tuple[str, str]:
    """Return (replacement, technique) for the legacy hardcoded policy map."""
    normalized = label.upper().replace(" ", "_")
    original = value
    value = value.strip()
    if "AGE" in normalized:
        return original, "retained"
    if "AADHAAR" in normalized:
        digits = "".join(ch for ch in value if ch.isdigit())
        return (f"XXXX XXXX {digits[-4:]}" if len(digits) == 12 else REDACTED, "partial_mask")
    if "PAN" in normalized:
        compact = value.replace(" ", "")
        return (f"{compact[:5]}****{compact[-1]}" if len(compact) == 10 else REDACTED, "partial_mask")
    if "PHONE" in normalized:
        return REDACTED, "suppress"
    if "EMAIL" in normalized:
        if "@" not in value:
            return REDACTED, "suppress"
        local, domain = value.split("@", 1)
        masked = local[:2] + "*" * max(1, len(local) - 2) if len(local) > 2 else local[:1] + "*"
        return f"{masked}@{domain}", "domain_preserving_mask"
    if "BANK" in normalized or "ACCOUNT" in normalized:
        digits = "".join(ch for ch in value if ch.isdigit())
        return ("*" * max(0, len(digits) - 4) + digits[-4:] if len(digits) >= 4 else REDACTED, "partial_mask")
    if "CARD" in normalized or "CREDIT" in normalized:
        digits = "".join(ch for ch in value if ch.isdigit())
        return (f"XXXX-XXXX-XXXX-{digits[-4:]}" if len(digits) >= 16 else REDACTED, "tokenization")
    if "DATE" in normalized or "DOB" in normalized:
        year = next(iter(__import__("re").findall(r"(?:19|20)\d{2}", value)), "")
        return (f"XX-XX-{year}" if year else "[DATE REDACTED]", "date_generalization")
    if "PINCODE" in normalized:
        digits = "".join(ch for ch in value if ch.isdigit())
        return (digits[:3] + "XXX" if len(digits) >= 6 else REDACTED, "partial_mask")
    if any(term in normalized for term in ("PERSON", "NAME", "LOCATION", "LOC", "ORGANIZATION", "ORG")):
        return REDACTED, "suppress"
    # Everything else (Voter ID, Passport, Driving License, Vehicle Number, IFSC,
    # User ID, PPP ID, IP address, Patient ID / UHID, ...) is hashed the same way
    # SKALD's `hashing_with_salt` does it: SHA256(salt + value), one random salt
    # per column generated once for this run and reused for every row in it.
    salt = salts.setdefault(column, secrets.token_hex(32))
    return hashlib.sha256((salt + value).encode()).hexdigest(), "salted_hash"


def _absolute_ner_hits(text: str) -> list[dict[str, Any]]:
    record = FileRecord(path="<cell>", filename="<cell>", file_type="string", language=detect_language(text), raw_text=text)
    run_line_by_line_ner([record], include_bert=False)
    results: list[dict[str, Any]] = []
    lines = text.splitlines(keepends=True)
    starts: list[int] = []
    cursor = 0
    for line in lines:
        starts.append(cursor)
        cursor += len(line)
    for line_result in record.line_ners.get("Hybrid (HiNER + IndicNER + XLM-RoBERTa)", []):
        if line_result.line_no < 1 or line_result.line_no > len(starts):
            continue
        raw_line = lines[line_result.line_no - 1]
        leading = len(raw_line) - len(raw_line.lstrip())
        for entity in line_result.entities:
            results.append({
                "start": starts[line_result.line_no - 1] + leading + entity.start_char - 1,
                "end": starts[line_result.line_no - 1] + leading + entity.end_char,
                "label": entity.category,
                "source": "Hybrid NER",
                "confidence": entity.score,
                "value": entity.text,
            })
    return results


def _sanitize(text: str, minimum_confidence: float, column: str, salts: dict[str, str]) -> tuple[str, list[dict[str, Any]]]:
    if not isinstance(text, str) or not text:
        return text, []

    language = detect_language(text)
    detections = [
        {"start": start, "end": end, "label": hit.label, "source": hit.source, "confidence": hit.confidence, "value": hit.value}
        for start, end, hit in _absolute_hits(text)
        if hit.confidence >= minimum_confidence
    ]
    detections.extend(item for item in _absolute_ner_hits(text) if item["confidence"] >= minimum_confidence)
    hits = sorted(detections, key=lambda item: (item["start"], item["end"]))
    if not hits:
        return text, []

    # Apply right-to-left so offsets remain valid. Overlapping detections are one replacement.
    spans: list[tuple[int, int, list[dict[str, Any]]]] = []
    for hit in hits:
        start, end = hit["start"], hit["end"]
        if spans and start <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], end), spans[-1][2] + [hit])
        else:
            spans.append((start, end, [hit]))

    output = text
    audit: list[dict[str, Any]] = []
    for start, end, span_hits in reversed(spans):
        # Use the first detection's hardcoded policy for an overlapping span.
        replacement, technique = _hardcoded_policy(span_hits[0]["label"], text[start:end], column, salts)
        output = output[:start] + replacement + output[end:]
        for hit in span_hits:
            audit.append({
                "label": hit["label"],
                "source": hit["source"],
                "language": language,
                "technique": technique,
                "confidence": hit["confidence"],
                "start_offset": start,
                "end_offset": end,
            })
    return output, list(reversed(audit))


def run(config_path: str, root: str | None = None) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    # In the container, config/config.json and data/... are siblings under /app.
    work_root = Path(root).resolve() if root else config_file.parent.parent
    dataset = _load_config(config_file)
    source = _resolve_source_input(dataset, work_root)
    settings = dataset.get("free_text_anonymization", {})
    if not isinstance(settings, dict):
        raise BatchError("free_text_anonymization must be an object")

    df, _source_format = _read_table(source)
    if settings.get("enabled") is not True:
        print("[Batch] free_text_anonymization.enabled is false; no staged file written.")
        return {"status": "disabled", "rows": len(df), "columns": len(df.columns)}
    if not _TRANSFORMERS_AVAILABLE:
        raise BatchError(
            "NER dependencies are not installed. Run 'pip install -r requirements.txt' "
            "or 'python app/model_setup.py' before running the multilingual batch job."
        )

    columns = settings.get("columns")
    if not isinstance(columns, list) or not all(isinstance(col, str) for col in columns):
        raise BatchError("free_text_anonymization.columns must be a list of column names")
    missing = [col for col in columns if col not in df.columns]
    on_failure = settings.get("on_failure", "fail")
    if on_failure not in {"fail", "continue"}:
        raise BatchError("free_text_anonymization.on_failure must be 'fail' or 'continue'")
    if missing:
        raise BatchError(f"Configured columns are missing from input: {missing}")

    minimum_confidence = float(settings.get("minimum_confidence", 0.0))
    audit: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    # One random salt per column, generated once for this run (mirrors SKALD's hashing_with_salt).
    column_hash_salts: dict[str, str] = {}
    for row_index in range(len(df)):
        for column in columns:
            value = df.iat[row_index, df.columns.get_loc(column)]
            if not isinstance(value, str) or not value:
                continue
            try:
                sanitized, accepted = _sanitize(value, minimum_confidence, column, column_hash_salts)
                if accepted:
                    df.iat[row_index, df.columns.get_loc(column)] = sanitized
                for item in accepted:
                    audit.append({"row": row_index + 1, "column": column, **item})
            except Exception as exc:
                failures.append({"row": row_index + 1, "column": column, "error": str(exc)})
                if on_failure == "fail":
                    raise BatchError(f"Anonymization failed at row {row_index + 1}, column {column!r}: {exc}") from exc

    staged = _path(settings.get("staged_input_path", ""), work_root)
    if not staged.name:
        raise BatchError("free_text_anonymization.staged_input_path is required")
    if staged.suffix.lower() != ".csv":
        raise BatchError(
            "free_text_anonymization.staged_input_path must end with '.csv' "
            "because SKALD consumes the staged CSV"
        )
    _write_table(df, staged)

    audit_path_value = settings.get("audit_output_path")
    if audit_path_value:
        audit_path = _path(audit_path_value, work_root)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open("w", encoding="utf-8") as fh:
            json.dump({"rows": len(df), "columns": list(df.columns), "detections": audit, "failures": failures}, fh, indent=2)
    print(f"[Batch] Wrote staged dataset: {staged}")
    print(f"[Batch] Rows: {len(df)}; detections: {len(audit)}; failures: {len(failures)}")
    return {"status": "completed", "rows": len(df), "detections": len(audit), "failures": len(failures), "staged_input_path": str(staged)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Config-driven dataset free-text anonymization")
    parser.add_argument("--config", required=True, help="Path to the JSON configuration")
    parser.add_argument("--root", default=None, help="Base directory for relative paths")
    args = parser.parse_args()
    try:
        run(args.config, args.root)
    except BatchError as exc:
        print(f"[Batch][ERROR] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
