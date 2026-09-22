#!/usr/bin/env python3
"""Config-driven, column-scoped dataset anonymization batch job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from main import (
        NER_CORPUS_BATCH_SIZE,
        PiiHit,
        detect_language,
        full_pii_scan,
        merge_pii_hits,
        run_corpus_line_ner,
        _tensor_batch_size,
        _TRANSFORMERS_AVAILABLE,
    )
    from profiling import NULL_PROFILER, Profiler
except ModuleNotFoundError:  # Supports `python -m app.batch_pipeline` too.
    from .profiling import NULL_PROFILER, Profiler
    from .main import (
        NER_CORPUS_BATCH_SIZE,
        PiiHit,
        detect_language,
        full_pii_scan,
        merge_pii_hits,
        run_corpus_line_ner,
        _tensor_batch_size,
        _TRANSFORMERS_AVAILABLE,
    )

REDACTED = "*"


class BatchError(RuntimeError):
    """An expected, user-actionable batch failure."""


def _path(value: str, root: Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else root / candidate


_RESERVED_CONFIG_NAMES = {"pipeline_config.json"}


def _resolve_config_file(config_arg: Path) -> Path:
    """Locate the dataset config JSON.

    Accepts either a direct file path (local/manual runs, e.g. `config/config.json`)
    or a directory to scan. The real deployment doesn't control what the UI names
    this file, so scanning mirrors `_resolve_source_input`'s convention:
      - exactly one .json (other than reserved names) -> use it, whatever it's called
      - a file literally named config.json -> preferred, even alongside others
      - none -> error (nothing to run)
      - more than one, none named config.json -> error naming the candidates

    `pipeline_config.json` is excluded: it's skald-image's config (see
    fetch_data.py:220, which treats its presence as the image-job signal) and
    must never be mistaken for this app's config.
    """
    if config_arg.is_file():
        return config_arg
    if not config_arg.is_dir():
        raise BatchError(f"Config path not found: {config_arg}")

    candidates = sorted(
        p for p in config_arg.iterdir()
        if p.is_file() and p.suffix.lower() == ".json" and p.name not in _RESERVED_CONFIG_NAMES
    )
    if not candidates:
        raise BatchError(f"No config JSON found in {config_arg}")
    preferred = next((p for p in candidates if p.name == "config.json"), None)
    if preferred is not None:
        return preferred
    if len(candidates) > 1:
        raise BatchError(
            f"Expected exactly one config JSON in {config_arg}, found {len(candidates)}: "
            f"{[p.name for p in candidates]}"
        )
    return candidates[0]


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


def _line_bases(text: str) -> list[int]:
    """Cell-relative offset of each line's first non-whitespace character.

    Detection runs on line.strip(), so a hit's offsets are relative to the
    stripped line; adding the base restores the cell-relative position.
    """
    bases: list[int] = []
    cursor = 0
    for line in text.splitlines(keepends=True):
        bases.append(cursor + len(line) - len(line.lstrip()))
        cursor += len(line)
    return bases


def _absolute_hits(text: str) -> list[tuple[int, int, PiiHit]]:
    """Convert line-relative hit offsets into cell-relative zero-based offsets."""
    bases = _line_bases(text)
    result = []
    for hit in _cell_hits(text):
        if hit.line_no < 1 or hit.line_no > len(bases):
            continue
        base = bases[hit.line_no - 1]
        result.append((base + max(hit.start_char - 1, 0), base + hit.end_char, hit))
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


# run_line_by_line_ner's dispatcher drops blank and sub-3-char lines rather than
# paying a model pass for them; the corpus builder below mirrors that.
_MIN_NER_LINE_CHARS = 3


def _ner_lines(text: str) -> list[tuple[int, str]]:
    """(line index, stripped line) for the lines worth sending to the models."""
    result = []
    for line_idx, line in enumerate(text.splitlines()):
        stripped = line.strip()
        if len(stripped) >= _MIN_NER_LINE_CHARS:
            result.append((line_idx, stripped))
    return result


def _models_for_span(
    start: int,
    end: int,
    line_spans: list[tuple[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Which models produced a raw span overlapping [start, end) on this line.

    merge_line_spans pools every model's spans before merging, so a merged entity
    carries no record of where it came from. Overlap is the reverse mapping: a
    model that fired on the same characters is a model that found this entity.
    """
    found: dict[str, float] = {}
    for label, span in line_spans:
        if span["start"] < end and start < span["end"]:
            score = float(span.get("score", 0.0))
            if score > found.get(label, -1.0):
                found[label] = score
    return [
        {"model": label, "confidence": round(score, 4)}
        for label, score in sorted(found.items(), key=lambda kv: -kv[1])
    ]


def _build_ner_index(
    texts: list[str],
    batch_size: int,
    profiler=None,
    attribute: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Run hybrid NER once over every distinct line in the job.

    This used to run per cell: each call re-orchestrated the model worker threads
    and queues to infer a single short line, so the mini-batching never engaged and
    a phrase repeated across thousands of rows was embedded thousands of times.
    Here every distinct line in the corpus is inferred exactly once, in real
    mini-batches, and the resulting spans are projected back onto each text that
    contains that line.

    With `attribute` set, each hit also carries a "models" list naming the models
    that fired on it — the merged entities alone cannot say.

    Returns {text: [hit dicts with cell-relative offsets]}.
    """
    profiler = profiler or NULL_PROFILER
    with profiler.step("build line corpus", parent="ner"):
        line_ids: dict[str, int] = {}
        corpus: list[str] = []
        for text in texts:
            for _, stripped in _ner_lines(text):
                if stripped not in line_ids:
                    line_ids[stripped] = len(corpus)
                    corpus.append(stripped)

    print(
        f"[Batch] Hybrid NER over {len(corpus)} distinct lines "
        f"from {len(texts)} distinct values …",
        flush=True,
    )
    spans_by_model: dict[str, list[list[dict[str, Any]]]] | None = {} if attribute else None
    entities = run_corpus_line_ner(
        corpus,
        batch_size=batch_size,
        release_models=True,
        spans_by_model=spans_by_model,
        profiler=profiler,
    )

    # Flatten to per-line (label, span) pairs once, rather than per entity.
    per_line_spans: list[list[tuple[str, dict[str, Any]]]] = []
    if spans_by_model:
        per_line_spans = [[] for _ in corpus]
        for label, line_lists in spans_by_model.items():
            for line_id, spans in enumerate(line_lists):
                for span in spans:
                    per_line_spans[line_id].append((label, span))

    with profiler.step("project spans onto cells", parent="ner"):
        index: dict[str, list[dict[str, Any]]] = {}
        for text in texts:
            bases = _line_bases(text)
            hits: list[dict[str, Any]] = []
            for line_idx, stripped in _ner_lines(text):
                base = bases[line_idx]
                line_id = line_ids[stripped]
                for entity in entities[line_id]:
                    hit = {
                        "start": base + entity.start_char - 1,
                        "end": base + entity.end_char,
                        "label": entity.category,
                        "source": "Hybrid NER",
                        "confidence": entity.score,
                        "value": entity.text,
                    }
                    if per_line_spans:
                        hit["models"] = _models_for_span(
                            entity.start_char - 1, entity.end_char, per_line_spans[line_id]
                        )
                    hits.append(hit)
            index[text] = hits
    return index


def _sanitize(
    text: str,
    minimum_confidence: float,
    column: str,
    salts: dict[str, str],
    ner_hits: list[dict[str, Any]],
    include_values: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    if not isinstance(text, str) or not text:
        return text, []

    language = detect_language(text)
    detections = [
        {"start": start, "end": end, "label": hit.label, "source": hit.source, "confidence": hit.confidence, "value": hit.value}
        for start, end, hit in _absolute_hits(text)
        if hit.confidence >= minimum_confidence
    ]
    detections.extend(item for item in ner_hits if item["confidence"] >= minimum_confidence)
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
            entry = {
                "label": hit["label"],
                "source": hit["source"],
                "language": language,
                "technique": technique,
                "confidence": hit["confidence"],
                "start_offset": start,
                "end_offset": end,
            }
            if "models" in hit:
                entry["models"] = hit["models"]
            if include_values:
                # Debug only: this is the unredacted PII, see _debug_settings.
                entry["detected_text"] = text[start:end]
            audit.append(entry)
    return output, list(reversed(audit))


def _torch_threads() -> int:
    try:
        import torch
        return torch.get_num_threads()
    except Exception:
        return 0


def _model_load_summary(profiler) -> dict[str, dict[str, Any]]:
    """Regroup the flat step list into per-model load / infer / release costs.

    This is the breakdown worth looking at first: it separates the fixed price of
    getting a model into memory from the per-line price of running it, and gives
    each model's own peak RSS rather than one number for the whole job.
    """
    summary: dict[str, dict[str, Any]] = {}
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


def _debug_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Read the optional `debug` block, with ANON_DEBUG=1 as an override.

    include_values is off by default and deliberately so: it writes the
    unredacted PII the run just masked into a plaintext report beside the
    anonymized output, which defeats the point of the job unless someone has
    decided that is acceptable for this dataset.
    """
    debug = settings.get("debug")
    if not isinstance(debug, dict):
        debug = {}
    enabled = bool(debug.get("enabled", False)) or os.getenv("ANON_DEBUG", "").strip() in {"1", "true", "yes"}
    return {
        "enabled": enabled,
        "include_values": bool(debug.get("include_values", False)),
        "profile_output_path": debug.get("profile_output_path", "output/debug_profile.json"),
        "attribution_output_path": debug.get("attribution_output_path", "output/debug_detections.json"),
        "attribution_text_path": debug.get("attribution_text_path", "output/debug_detections.txt"),
        "max_rows": int(debug.get("max_rows", 0)),
    }


def _model_short_name(label: str) -> str:
    """'HiNER (IIT Bombay / MuRIL)' -> 'HiNER', for readable per-row lines."""
    return label.split(" (")[0].strip()


def _write_attribution(
    audit: list[dict[str, Any]],
    json_path: Path,
    text_path: Path,
    include_values: bool,
    max_rows: int,
) -> None:
    """Emit what was detected, per row, and which detector found it.

    The JSON form is grouped by row for programmatic diffing between runs; the
    text form is the one to read when asking "why did this row come out like
    that". Regex hits carry their own source name and no model list.
    """
    by_row: dict[int, list[dict[str, Any]]] = {}
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
                        "detected_by": (
                            [m["model"] for m in it["models"]] if it.get("models")
                            else [it["source"]]
                        ),
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
        "PII values are NOT included; set debug.include_values to add them."
        if not include_values
        else "WARNING: this file contains unredacted PII (debug.include_values is on).",
        "",
    ]
    for row in rows:
        lines.append(f"Row {row}")
        for it in by_row[row]:
            who = (
                ", ".join(_model_short_name(m["model"]) for m in it["models"])
                if it.get("models") else it["source"]
            )
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


def run(config_path: str, root: str | None = None) -> dict[str, Any]:
    config_arg = Path(config_path).resolve()
    config_file = _resolve_config_file(config_arg)
    config_dir = config_arg if config_arg.is_dir() else config_file.parent
    # In the container, config/ and data/... are siblings under /app.
    work_root = Path(root).resolve() if root else config_dir.parent
    dataset = _load_config(config_file)
    source = _resolve_source_input(dataset, work_root)
    settings = dataset.get("free_text_anonymization", {})
    if not isinstance(settings, dict):
        raise BatchError("free_text_anonymization must be an object")

    debug = _debug_settings(settings)
    profiler = Profiler(enabled=debug["enabled"])
    profiler.start()
    if debug["enabled"]:
        print("[Batch] Debug instrumentation on.", flush=True)
        if debug["include_values"]:
            print(
                "[Batch] WARNING: debug.include_values writes unredacted PII to the "
                "attribution report.",
                flush=True,
            )

    with profiler.step("read input table", rows_source=str(source)):
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
    batch_size = int(settings.get("ner_batch_size", NER_CORPUS_BATCH_SIZE))
    audit: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    # One random salt per column, generated once for this run (mirrors SKALD's hashing_with_salt).
    column_hash_salts: dict[str, str] = {}
    column_locs = {column: df.columns.get_loc(column) for column in columns}
    started = time.time()

    # Pass 1 — collect the distinct values we actually have to anonymize.
    # Grievance free text repeats heavily (four in five complaint bodies in a real
    # PHED export are duplicates), and the sanitized form of a value depends only
    # on (value, column), so each distinct pair is computed once however many rows
    # carry it.
    with profiler.step("collect distinct values") as collect:
        distinct: set[str] = set()
        cells = 0
        for column in columns:
            for value in df.iloc[:, column_locs[column]].tolist():
                if isinstance(value, str) and value:
                    cells += 1
                    distinct.add(value)
        texts = sorted(distinct)
        collect.detail.update(cells=cells, distinct=len(texts))
    print(
        f"[Batch] {len(df)} rows x {len(columns)} column(s); {len(texts)} distinct values.",
        flush=True,
    )

    # Pass 2 — one NER sweep over the whole corpus, then apply the policy.
    with profiler.step("ner") as ner_step:
        ner_index = (
            _build_ner_index(texts, batch_size, profiler=profiler, attribute=debug["enabled"])
            if texts else {}
        )
        ner_step.detail.update(distinct_values=len(texts))

    apply_cm = profiler.step("apply policy")
    apply_step = apply_cm.__enter__()
    sanitized_cache: dict[tuple[str, str], tuple[str, list[dict[str, Any]]]] = {}
    for column in columns:
        loc = column_locs[column]
        values = df.iloc[:, loc].tolist()
        for row_index, value in enumerate(values):
            if not isinstance(value, str) or not value:
                continue
            try:
                key = (column, value)
                if key not in sanitized_cache:
                    sanitized_cache[key] = _sanitize(
                        value,
                        minimum_confidence,
                        column,
                        column_hash_salts,
                        ner_index.get(value, []),
                        include_values=debug["enabled"] and debug["include_values"],
                    )
                sanitized, accepted = sanitized_cache[key]
                if accepted:
                    values[row_index] = sanitized
                for item in accepted:
                    audit.append({"row": row_index + 1, "column": column, **item})
            except Exception as exc:
                failures.append({"row": row_index + 1, "column": column, "error": str(exc)})
                if on_failure == "fail":
                    raise BatchError(f"Anonymization failed at row {row_index + 1}, column {column!r}: {exc}") from exc
        df.isetitem(loc, values)
    apply_step.detail.update(detections=len(audit), failures=len(failures))
    apply_cm.__exit__(None, None, None)
    # Pass 2 walks column-major for cheap bulk column reads; the audit is still
    # emitted row-major, in configured column order, as consumers expect.
    column_order = {column: rank for rank, column in enumerate(columns)}
    audit.sort(key=lambda item: (item["row"], column_order[item["column"]]))
    failures.sort(key=lambda item: (item["row"], column_order[item["column"]]))
    print(f"[Batch] Anonymization took {time.time() - started:.1f}s", flush=True)

    staged = _path(settings.get("staged_input_path", ""), work_root)
    if not staged.name:
        raise BatchError("free_text_anonymization.staged_input_path is required")
    if staged.suffix.lower() != ".csv":
        raise BatchError(
            "free_text_anonymization.staged_input_path must end with '.csv' "
            "because SKALD consumes the staged CSV"
        )
    with profiler.step("write staged csv"):
        _write_table(df, staged)

    audit_path_value = settings.get("audit_output_path")
    if audit_path_value:
        with profiler.step("write audit json"):
            audit_path = _path(audit_path_value, work_root)
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            with audit_path.open("w", encoding="utf-8") as fh:
                json.dump({"rows": len(df), "columns": list(df.columns), "detections": audit, "failures": failures}, fh, indent=2)
    print(f"[Batch] Wrote staged dataset: {staged}")
    print(f"[Batch] Rows: {len(df)}; detections: {len(audit)}; failures: {len(failures)}")

    if debug["enabled"]:
        with profiler.step("write debug reports"):
            _write_attribution(
                audit,
                _path(debug["attribution_output_path"], work_root),
                _path(debug["attribution_text_path"], work_root),
                include_values=debug["include_values"],
                max_rows=debug["max_rows"],
            )
        profiler.stop()
        profiler.write(
            _path(debug["profile_output_path"], work_root),
            extra={
                "rows": len(df),
                "columns": columns,
                "distinct_values": len(texts),
                "detections": len(audit),
                "torch_threads": _torch_threads(),
                "tensor_batch_size": _tensor_batch_size(),
                "models": _model_load_summary(profiler),
            },
        )
        print(profiler.table(), flush=True)
    profiler.stop()

    return {"status": "completed", "rows": len(df), "detections": len(audit), "failures": len(failures), "staged_input_path": str(staged)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Config-driven dataset free-text anonymization")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the JSON configuration file, or a directory to scan for one",
    )
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
