"""Turn each supported input format into plain text, one meaningful line per field."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd

try:  # optional: .docx support
    import docx as _docx

    _DOCX_AVAILABLE = True
except ImportError:
    _DOCX_AVAILABLE = False

try:  # optional: proper HTML parsing (a tag-stripping fallback is used without it)
    from bs4 import BeautifulSoup

    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False

SUPPORTED_EXTENSIONS = {".txt", ".docx", ".doc", ".html", ".htm", ".json", ".csv"}


def _normalize_newlines(text: str) -> str:
    """CRLF and lone CR become LF; surrounding whitespace is stripped."""
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def read_txt(path: str) -> str:
    """Plain text file (UTF-8; undecodable bytes are replaced)."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        return _normalize_newlines(fh.read())


def read_docx(path: str) -> str:
    """Word document: every non-empty paragraph, then every non-empty table cell, one per line."""
    if not _DOCX_AVAILABLE:
        print(f"  [WARN] python-docx not installed; skipping {path}", flush=True)
        return ""
    document = _docx.Document(path)
    lines = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            lines.extend(cell.text.strip() for cell in row.cells if cell.text.strip())
    return "\n".join(lines)


def read_html(path: str) -> str:
    """HTML page: visible text nodes, one per line (script/style/head/meta dropped).

    Without BeautifulSoup, tags are stripped with a regex, a few common entities
    decoded, and the text collapsed onto one line.
    """
    with open(path, encoding="utf-8", errors="replace") as fh:
        raw = fh.read()
    if not _BS4_AVAILABLE:
        text = re.sub(r"<[^>]+>", " ", raw)
        for entity, char in (("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">")):
            text = text.replace(entity, char)
        return re.sub(r"\s+", " ", text).strip()
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "head", "meta"]):
        tag.decompose()
    return "\n".join(t.strip() for t in soup.find_all(text=True) if t.strip())


def _flatten_json(obj, prefix: str = "") -> List[str]:
    """Nested JSON -> "path: value" lines, so detection sees every field.

    {"a": {"b": "x"}, "c": ["y"]} -> ["a.b: x", "c[0]: y"]. Multi-line strings
    give one line per non-empty text line; None and empty values are skipped.
    """
    if isinstance(obj, dict):
        return [line for key, value in obj.items()
                for line in _flatten_json(value, f"{prefix}.{key}" if prefix else str(key))]
    if isinstance(obj, list):
        return [line for i, value in enumerate(obj)
                for line in _flatten_json(value, f"{prefix}[{i}]" if prefix else f"[{i}]")]
    if obj is None:
        return []
    lines = []
    for part in _normalize_newlines(str(obj)).split("\n"):
        part = part.strip()
        if part:
            lines.append(f"{prefix}: {part}" if prefix else part)
    return lines


def json_payload_to_text(payload) -> str:
    """Text for a parsed JSON document.

    A list becomes "--- Record N ---" blocks; an object holding a list under
    "records", "data", "complaints" or "items" is treated as that list; any
    other object is flattened as one record.
    """
    if isinstance(payload, str):
        return _normalize_newlines(payload)
    if isinstance(payload, list):
        blocks: List[str] = []
        for i, item in enumerate(payload, 1):
            inner = "\n".join(_flatten_json(item))
            header = f"--- Record {i} ---"
            blocks.append(f"{header}\n{inner}" if inner else header)
        return "\n".join(blocks).strip()
    if isinstance(payload, dict):
        for key in ("records", "data", "complaints", "items"):
            if isinstance(payload.get(key), list):
                return json_payload_to_text(payload[key])
        return "\n".join(_flatten_json(payload)).strip()
    return str(payload).strip()


def read_json(path: str) -> str:
    """JSON file, flattened by json_payload_to_text."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        return json_payload_to_text(json.load(fh))


def read_csv(path: str) -> str:
    """CSV: one line per row, "Record N | col: value | col: value" (empty cells skipped)."""
    df = pd.read_csv(path, dtype=object, keep_default_na=False)
    if df.empty:
        return ""
    columns = [str(col) for col in df.columns]
    lines: List[str] = []
    for row_idx, row in df.iterrows():
        parts = []
        for col in columns:
            value = str(row[col]).replace("\r\n", " ").replace("\r", " ").replace("\n", " ").strip()
            if value:
                parts.append(f"{col}: {value}")
        if parts:
            lines.append(f"Record {int(row_idx) + 1} | " + " | ".join(parts))
    return "\n".join(lines)


def extract_text(path: str) -> Tuple[str, str]:
    """(text, file type) for any supported file; unknown extensions are read as text."""
    ext = Path(path).suffix.lower()
    if ext == ".txt":
        return read_txt(path), "txt"
    if ext in (".docx", ".doc"):
        return read_docx(path), "docx"
    if ext in (".html", ".htm"):
        return read_html(path), "html"
    if ext == ".json":
        return read_json(path), "json"
    if ext == ".csv":
        return read_csv(path), "csv"
    try:
        return read_txt(path), "txt"
    except Exception:
        return "", "unknown"


def collect_files(input_path: str) -> List[str]:
    """The supported files at `input_path`: the file itself, or all of them under a folder.

    Exits with a message when the path is missing, unsupported or holds no
    supported file.
    """
    p = Path(input_path)
    if p.is_file():
        if p.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [str(p)]
        raise SystemExit(f"[ERROR] Unsupported file type: {input_path}. Supported: {SUPPORTED_EXTENSIONS}")
    if p.is_dir():
        files = sorted(f for ext in SUPPORTED_EXTENSIONS for f in p.rglob(f"*{ext}"))
        if not files:
            raise SystemExit(f"[ERROR] No supported files found in: {input_path}")
        return [str(f) for f in files]
    raise SystemExit(f"[ERROR] Path not found: {input_path}")
