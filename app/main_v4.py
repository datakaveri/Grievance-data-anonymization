#!/usr/bin/env python3
"""
main_v4.py  —  Smart Ensemble NER (IndicNER + XLM-RoBERTa)

Built on main_v3.py.  All output formats, regex PII, Excel/JSON generation,
and CLI are inherited unchanged via module patching.  Only the NER inference
core changes.

HOW TO RUN — identical to v3:
    python main_v4.py --input <path> --output report.xlsx

WHAT CHANGED vs v3 (each change backed by testbed evidence):
─────────────────────────────────────────────────────────────────────────────
  A. Unicode bold normalisation BEFORE inference          (C13: was 100% leak)
     WhatsApp-style bold letters (𝐑𝐚𝐦𝐞𝐬𝐡) map to [UNK] in both tokenisers.
     Normalising to ASCII before the model run eliminates the failure entirely.

  B. Script-aware confidence thresholds                   (tailors trust per model)
     IndicNER noise floor: 14.7%  → needs a higher bar on Latin/romanised text
     XLM noise floor:       3.2%  → reliable enough to accept lower thresholds

     Thresholds applied at merge time (raw inference runs at 0.30 so both
     models produce enough candidates for the agreement check):

       Model         Script          Effective threshold
       ──────────────────────────────────────────────────
       IndicNER      Devanagari      0.45   (model is strong here)
       IndicNER      Latin/Hinglish  0.55   (grammar-blind, spurious PER/ORG)
       IndicNER      Mixed           0.50
       XLM           Devanagari      0.50   (misses invented Devanagari names)
       XLM           Latin/Hinglish  0.38   (grammar-aware, clean noise floor)
       XLM           Mixed           0.43

  C. Agreement boost in hybrid merge
     When BOTH models find the same span (≥50% character overlap, same type),
     the span's confidence is boosted by 15% (capped at 1.0).  Cross-model
     agreement is strong evidence.

  D. False-positive suppression (three patterns from confirmed test failures)
     D1. Hindi adj-noun bigrams → spurious PER (I01: 62.5% FP rate, IndicNER)
         Words like "ganda", "kharab", "bura" as standalone PER spans: dropped.
     D2. hamare/humara locality phrases → spurious LOC (H14: 29% FP rate)
         LOC spans preceded by hamare/humara/hamara in the same line: dropped.
     D3. Infrastructure nouns → spurious LOC (H07/I03: 30-37% FP rate)
         LOC spans whose text contains a known infrastructure root word: dropped.

  E. Postposition trim on PER spans                       (C80: was 100% mis-type)
     When a Hindi postposition (ne/ko/ka/ki/se/ke/par/mein) is glued to the
     end of a PER span, strip it and re-clean the span boundary.  This turns
     "Narayandasne → LOC" into "Narayandas → PER".

  NOT changed:
    - Regex PII detection (all patterns identical to v3)
    - aggregation_strategy="simple"  (max-agg worsens leaks, confirmed by I05)
    - Excel / JSON output structure
    - CLI arguments
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── locate v3 next to this file ──────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

import main_v3 as _v3
from main_v3 import (
    NerEntity,
    LineTask,
    ModelLineResult,
    LineNerResult,
    FileRecord,
    NER_MODELS,
    HYBRID_KEY,
    get_word_number,
    _load_ner_pipeline,
    _infer_line_batch,
    _parse_ner_items,
    merge_line_spans,
    _NER_PIPELINE_CACHE,
)

# ═════════════════════════════════════════════════════════════════════════════
# A.  UNICODE BOLD NORMALISATION
# ═════════════════════════════════════════════════════════════════════════════
# Mathematical bold/italic Unicode code-points (U+1D400–U+1D7FF) that popular
# "fancy text" keyboard apps produce.  Both mBERT (IndicNER) and XLM-RoBERTa
# tokenise every one of them as [UNK], making the whole name invisible.
# Length is preserved (1 code-point → 1 ASCII code-point) so all span
# character offsets remain correct after normalisation.

_BOLD_MAPS: List[Tuple[int, int, int]] = [
    # (start_cp, end_cp_exclusive, base_ascii_cp)
    (0x1D400, 0x1D41A, ord("A")),   # Bold capital A–Z
    (0x1D41A, 0x1D434, ord("a")),   # Bold small a–z
    (0x1D434, 0x1D44E, ord("A")),   # Italic capital A–Z
    (0x1D44E, 0x1D468, ord("a")),   # Italic small a–z
    (0x1D468, 0x1D482, ord("A")),   # Bold italic capital A–Z
    (0x1D482, 0x1D49C, ord("a")),   # Bold italic small a–z
    (0x1D49C, 0x1D4B6, ord("A")),   # Script capital A–Z (partial)
    (0x1D5D4, 0x1D5EE, ord("A")),   # Sans-serif bold capital A–Z
    (0x1D5EE, 0x1D608, ord("a")),   # Sans-serif bold small a–z
    (0x1D608, 0x1D622, ord("A")),   # Sans-serif bold italic capital A–Z
    (0x1D622, 0x1D63C, ord("a")),   # Sans-serif bold italic small a–z
    (0x1D63C, 0x1D656, ord("A")),   # Monospace capital A–Z
    (0x1D656, 0x1D670, ord("a")),   # Monospace small a–z
    (0x1D670, 0x1D68A, ord("A")),   # Mathematical monospace capital
    (0x1D68A, 0x1D6A4, ord("a")),   # Mathematical monospace small
    (0x1D7CE, 0x1D7D8, ord("0")),   # Mathematical bold digits 0–9
    (0x1D7D8, 0x1D7E2, ord("0")),   # Mathematical double-struck digits
    (0x1D7E2, 0x1D7EC, ord("0")),   # Mathematical sans-serif digits
    (0x1D7EC, 0x1D7F6, ord("0")),   # Mathematical sans-serif bold digits
    (0x1D7F6, 0x1D800, ord("0")),   # Mathematical monospace digits
]


def _normalize_unicode_bold(text: str) -> str:
    """Replace mathematical bold/italic Unicode letters with their ASCII equivalents."""
    if not text:
        return text
    # Fast path: if no char is in the decorated range, return as-is
    if not any(0x1D400 <= ord(c) <= 0x1D800 for c in text):
        return text
    out = []
    for ch in text:
        cp = ord(ch)
        replaced = False
        for start, end, base in _BOLD_MAPS:
            if start <= cp < end:
                out.append(chr(base + (cp - start)))
                replaced = True
                break
        if not replaced:
            out.append(ch)
    return "".join(out)


# ═════════════════════════════════════════════════════════════════════════════
# B.  SCRIPT DETECTION & CONFIDENCE THRESHOLDS
# ═════════════════════════════════════════════════════════════════════════════

_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def _classify_script(text: str) -> str:
    """Return 'devanagari', 'latin', or 'mixed' for a single line."""
    dev = len(_DEVANAGARI_RE.findall(text))
    lat = len(_LATIN_RE.findall(text))
    total = dev + lat
    if total == 0:
        return "latin"
    ratio = dev / total
    if ratio >= 0.55:
        return "devanagari"
    if ratio <= 0.15:
        return "latin"
    return "mixed"


# Effective minimum confidence per (model_key, script_type).
# model_key values: "indicner" or "xlm"
_SCRIPT_THRESHOLDS: Dict[Tuple[str, str], float] = {
    ("indicner", "devanagari"): 0.45,   # model is strong here — standard
    ("indicner", "latin"):      0.55,   # grammar-blind + spurious PER/ORG
    ("indicner", "mixed"):      0.50,
    ("xlm",      "devanagari"): 0.50,   # grammar-blind for novel Devanagari names
    ("xlm",      "latin"):      0.38,   # grammar-aware, clean 3.2% noise floor
    ("xlm",      "mixed"):      0.43,
}

def _get_threshold(model_key: str, script: str) -> float:
    return _SCRIPT_THRESHOLDS.get((model_key, script), 0.50)


# Raw inference thresholds: lowered so both models produce enough candidates
# for the agreement check.  Script-aware filtering happens at merge time.
_RAW_INDICNER_THRESHOLD = 0.30
_RAW_XLM_THRESHOLD      = 0.28


# ═════════════════════════════════════════════════════════════════════════════
# D.  FALSE-POSITIVE SUPPRESSION
# ═════════════════════════════════════════════════════════════════════════════

# D1. Hindi adjective/common words that IndicNER spuriously tags as PER (I01)
_HINDI_ADJ_BLOCKLIST = frozenset({
    "ganda", "gannda", "gandha", "kharab", "bura", "buri", "maila", "saaf",
    "purana", "purani", "naya", "nayi", "bada", "badi", "chota", "choti",
    "khali", "bhari", "jyada", "kam", "thoda", "bahut", "naali", "nali",
    "toot", "tooti", "band", "jam", "zyada", "bekar", "sahi", "galat",
    "accha", "acchi", "achha", "achhi", "bura", "buri", "pakka", "kaccha",
})

# D2. Possessive prefixes that introduce common-noun localities, not named places (H14)
_HAMARE_RE = re.compile(
    r"\b(hamare|humara|hamara|hamari|apne|apna|apni|humare)\s+",
    re.IGNORECASE,
)

# D3. Infrastructure roots whose typo'd forms get mistaken for LOC/ORG (H07, I03)
_INFRA_ROOTS = frozenset({
    "pipeline", "pipe", "borewell", "bore", "nali", "naali", "drain", "sewer",
    "pump", "motor", "tank", "handpump", "tubewell", "connection", "supply",
    "valve", "overhead", "sump", "manhole", "meter", "leakage", "fitting",
    "sewage", "gutter", "drainage", "waterline", "mainline", "water",
})


def _text_contains_infra_root(text: str) -> bool:
    """True if text (after normalisation) overlaps with a known infrastructure root."""
    lower = re.sub(r"[^a-z]", "", text.lower())
    if not lower:
        return False
    # Direct substring check
    for root in _INFRA_ROOTS:
        root_nospace = root.replace(" ", "")
        if root_nospace in lower or lower in root_nospace:
            return True
    # Loose fuzzy: check each individual token
    for token in re.split(r"\s+", text.lower()):
        tok = re.sub(r"[^a-z]", "", token)
        if len(tok) < 3:
            continue
        for root in _INFRA_ROOTS:
            r = root.replace(" ", "")
            if len(r) < 3:
                continue
            # edit distance ≤ 2 for similar-length strings
            if abs(len(tok) - len(r)) <= 2 and _edit_dist(tok, r) <= 2:
                return True
    return False


def _edit_dist(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for ca in a:
        curr = [prev[0] + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]


def _suppress_false_positives(
    spans: List[Dict],
    line_text: str,
    model_key: str,
) -> List[Dict]:
    """Remove known false-positive spans before merging."""
    kept = []
    # Build hamare match positions for D2
    hamare_ends = {m.end() for m in _HAMARE_RE.finditer(line_text)}

    for s in spans:
        cat   = s["cat"]
        text  = s.get("text", line_text[s["start"]:s["end"]])
        start = s["start"]

        # D1 — spurious PER from IndicNER on Hindi adj-noun bigrams
        if (
            cat == "PERSON"
            and model_key == "indicner"
            and text.strip().lower() in _HINDI_ADJ_BLOCKLIST
        ):
            continue

        # D2 — spurious LOC on hamare/humara locality phrases
        if cat == "LOCATION" and any(he == start for he in hamare_ends):
            continue

        # D3 — spurious LOC/ORG on infrastructure words
        if cat in ("LOCATION", "ORGANIZATION") and _text_contains_infra_root(text):
            continue

        kept.append(s)
    return kept


# ═════════════════════════════════════════════════════════════════════════════
# E.  POSTPOSITION TRIM ON PER SPANS
# ═════════════════════════════════════════════════════════════════════════════
# When a Hindi postposition is glued to the end of a name with no space
# (e.g. "Narayandasne"), the model tags the compound as LOC/ORG or extends
# the PER span to include the postposition.  Strip the suffix and re-type.

_POSTPOSITIONS = ("mein", "par", "se", "ne", "ko", "ka", "ki", "ke")


def _trim_postposition(s: Dict) -> Dict:
    """If a PER or LOC/ORG span ends with a Hindi postposition, strip it."""
    text = s.get("text", "")
    lower = text.lower()
    for pp in sorted(_POSTPOSITIONS, key=len, reverse=True):
        if lower.endswith(pp) and len(text) - len(pp) >= 3:
            trimmed = text[: -len(pp)].rstrip(" -")
            if len(trimmed) >= 2:
                s = dict(s)
                s["text"] = trimmed
                s["end"]  = s["start"] + len(trimmed)
                # A span that was LOC/ORG only because of the glued postposition
                # is likely a person — promote it.
                if s["cat"] in ("LOCATION", "ORGANIZATION"):
                    s["cat"] = "PERSON"
            break
    return s


# ═════════════════════════════════════════════════════════════════════════════
# C.  SMART HYBRID MERGE WITH AGREEMENT BOOST
# ═════════════════════════════════════════════════════════════════════════════

_AGREEMENT_BOOST = 1.15   # multiply score when both models agree on a span
_AGREEMENT_MIN_OVERLAP = 0.50  # fraction of char overlap to count as agreement

def _char_overlap_ratio(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    """Fraction of the smaller span covered by the overlap."""
    overlap = max(0, min(a_end, b_end) - max(a_start, b_start))
    smaller = min(a_end - a_start, b_end - b_start)
    if smaller <= 0:
        return 0.0
    return overlap / smaller


def _smart_hybrid_merge(
    indicner_spans: List[Dict],
    xlm_spans: List[Dict],
    line_text: str,
) -> List[Dict]:
    """
    Merge IndicNER and XLM spans with three improvements over v3's flat union:

    1. Script-aware threshold filtering per model.
    2. Agreement boost: a span confirmed by both models gets +15% confidence.
    3. Cross-model coverage: each model's unique (non-overlapping) spans are kept
       at their model-specific threshold, so the combined output is always ≥ union.
    """
    script = _classify_script(line_text)
    thr_ind = _get_threshold("indicner", script)
    thr_xlm = _get_threshold("xlm",      script)

    # Apply thresholds + false-positive suppression + postposition trim per model
    def _clean(spans: List[Dict], model_key: str, thr: float) -> List[Dict]:
        out = []
        for s in spans:
            if s.get("score", 0.0) < thr:
                continue
            s = _trim_postposition(s)
            out.append(s)
        out = _suppress_false_positives(out, line_text, model_key)
        return out

    ind = _clean(indicner_spans, "indicner", thr_ind)
    xlm = _clean(xlm_spans,      "xlm",      thr_xlm)

    # Mark which spans are confirmed by BOTH models
    boosted: List[Dict] = []
    xlm_matched = [False] * len(xlm)

    for s_i in ind:
        best_j = -1
        best_overlap = _AGREEMENT_MIN_OVERLAP
        for j, s_x in enumerate(xlm):
            if s_i["cat"] != s_x["cat"]:
                continue
            ratio = _char_overlap_ratio(
                s_i["start"], s_i["end"], s_x["start"], s_x["end"]
            )
            if ratio > best_overlap:
                best_overlap = ratio
                best_j = j
        if best_j >= 0:
            # Agreement: keep IndicNER span (usually better span boundaries for
            # Devanagari), boost confidence, mark XLM span as matched
            boosted_score = min(1.0, s_i["score"] * _AGREEMENT_BOOST)
            boosted.append({**s_i, "score": boosted_score, "agreed": True})
            xlm_matched[best_j] = True
        else:
            # IndicNER-only span: keep as-is
            boosted.append({**s_i, "agreed": False})

    for j, s_x in enumerate(xlm):
        if not xlm_matched[j]:
            # XLM-only span: keep as-is
            boosted.append({**s_x, "agreed": False})

    return boosted


# ═════════════════════════════════════════════════════════════════════════════
# PATCHED CORE FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

# ---------- A: unicode bold normalisation wrapper for _infer_line_batch ----------

def _infer_line_batch_v4(
    pipe,
    tasks: List[LineTask],
    min_score: float,
    model_name: str,
    batch_size: Optional[int] = None,
) -> List[ModelLineResult]:
    """Normalise unicode bold in every line before passing to the model."""
    norm_tasks = [
        LineTask(t.doc_id, t.line_no, _normalize_unicode_bold(t.line_text))
        if any(0x1D400 <= ord(c) <= 0x1D800 for c in t.line_text)
        else t
        for t in tasks
    ]
    return _infer_line_batch(pipe, norm_tasks, min_score, model_name, batch_size)


# ---------- smart hybrid merge replacing v3's flat concatenation ----------

_INDICNER_LABEL = NER_MODELS["IndicNER"][0]
_XLM_LABEL      = NER_MODELS["XLM_RoBERTa"][0]


def _store_line_ner_results_v4(
    rec: "FileRecord",
    line_no: int,
    line_text: str,
    model_spans: Dict[str, List[Dict]],
    model_labels: List[str],
):
    """v4 version: individual model results use script-aware thresholds;
    HYBRID_KEY uses the smart merge with agreement boost."""
    from main_v3 import merge_line_spans as _merge  # original v3 merge

    # Per-model entity lists (with script-aware threshold applied inside smart merge)
    script  = _classify_script(line_text)
    model_line_ents: Dict[str, List[NerEntity]] = {}

    for model_lbl in model_labels:
        raw = model_spans.get(model_lbl, [])
        # Determine model key and threshold
        if "IndicNER" in model_lbl or "indicner" in model_lbl.lower():
            thr = _get_threshold("indicner", script)
            mkey = "indicner"
        else:
            thr = _get_threshold("xlm", script)
            mkey = "xlm"
        filtered = _suppress_false_positives(
            [s for s in raw if s.get("score", 0.0) >= thr],
            line_text, mkey,
        )
        filtered = [_trim_postposition(s) for s in filtered]
        model_line_ents[model_lbl] = _merge(filtered, line_text)

    # Hybrid: smart merge with agreement boost
    ind_spans = model_spans.get(_INDICNER_LABEL, [])
    xlm_spans = model_spans.get(_XLM_LABEL, [])
    hybrid_raw = _smart_hybrid_merge(ind_spans, xlm_spans, line_text)
    model_line_ents[HYBRID_KEY] = _merge(hybrid_raw, line_text)

    # Build the missed-entity list (same logic as v3)
    all_line_ents = set(
        (e.category, e.text)
        for ents in model_line_ents.values()
        for e in ents
    )
    for model_lbl, ents in model_line_ents.items():
        persons = [e.text for e in ents if e.category == "PERSON"]
        locs    = [e.text for e in ents if e.category == "LOCATION"]
        orgs    = [e.text for e in ents if e.category == "ORGANIZATION"]
        current = set((e.category, e.text) for e in ents)
        missed  = [
            f"{val} ({cat})"
            for cat, val in all_line_ents
            if (cat, val) not in current
        ]
        rec.line_ners[model_lbl].append(
            LineNerResult(
                model=model_lbl,
                line_no=line_no,
                text=line_text,
                entities=ents,
                persons=persons,
                locs=locs,
                orgs=orgs,
                missed=missed,
            )
        )


# ---------- lower raw inference thresholds so the merge can filter ----------

_HYBRID_NER_SPECS_V4 = [
    # (display label, model id, use_fast, raw min_score)
    (_INDICNER_LABEL, "ai4bharat/IndicNER",           False, _RAW_INDICNER_THRESHOLD),
    (_XLM_LABEL,      "Davlan/xlm-roberta-base-ner-hrl", True, _RAW_XLM_THRESHOLD),
]


# ═════════════════════════════════════════════════════════════════════════════
# PATCH v3 MODULE IN-PLACE
# Python resolves names in module globals at call time, so replacing a name
# here means all existing v3 functions that call it will see the v4 version.
# ═════════════════════════════════════════════════════════════════════════════

_v3._infer_line_batch        = _infer_line_batch_v4
_v3._store_line_ner_results  = _store_line_ner_results_v4
_v3._HYBRID_NER_SPECS        = _HYBRID_NER_SPECS_V4


# ═════════════════════════════════════════════════════════════════════════════
# RE-EXPORT EVERYTHING FROM v3 (CLI entry point, output builders, etc.)
# ═════════════════════════════════════════════════════════════════════════════

from main_v3 import *   # noqa: F401, F403  — intentional wildcard re-export

# Override the handful of names that we redefined above so that importing
# `from main_v4 import *` also gives callers the v4 versions.
from main_v3 import (   # noqa: F811
    run_line_by_line_ner,
    run_corpus_line_ner,
    analyze_records,
    build_excel,
    build_json,
    main,
)


# ═════════════════════════════════════════════════════════════════════════════
# QUICK SELF-TEST  (python main_v4.py --selftest)
# ═════════════════════════════════════════════════════════════════════════════

def _selftest():
    print("main_v4 self-test …")
    errors = 0

    # A. Unicode bold normalisation
    bold = "𝐑𝐚𝐦𝐞𝐬𝐡 ne paani ki shikayat ki"
    normed = _normalize_unicode_bold(bold)
    assert normed.startswith("Ramesh"), f"Unicode normalisation failed: {normed!r}"
    assert len(normed) == len(bold), "Length must be preserved"
    print("  [PASS] A. Unicode bold normalisation")

    # B. Script classification
    assert _classify_script("Ramesh ne shikayat ki") == "latin"
    assert _classify_script("रमेश ने शिकायत की") == "devanagari"
    assert _classify_script("Ramesh रमेश mix") == "mixed"
    print("  [PASS] B. Script classification")

    # B. Threshold lookup
    assert _get_threshold("indicner", "latin")      == 0.55
    assert _get_threshold("xlm",      "latin")      == 0.38
    assert _get_threshold("indicner", "devanagari") == 0.45
    print("  [PASS] B. Threshold lookup")

    # D1. Hindi adj-noun FP suppression
    fake_per = [{"cat": "PERSON", "start": 7, "end": 12, "score": 0.6,
                 "text": "ganda", "model": "indicner"}]
    result = _suppress_false_positives(fake_per, "hamare ganda paani aa raha hai", "indicner")
    assert len(result) == 0, f"D1 suppression failed: {result}"
    print("  [PASS] D1. Hindi adj-noun PER suppression")

    # D2. hamare LOC suppression
    fake_loc = [{"cat": "LOCATION", "start": 7, "end": 11, "score": 0.7,
                 "text": "area", "model": "xlm"}]
    result = _suppress_false_positives(fake_loc, "hamare area mein paani nahi", "xlm")
    assert len(result) == 0, f"D2 suppression failed: {result}"
    print("  [PASS] D2. hamare LOC suppression")

    # D3. Infrastructure LOC suppression
    fake_infra = [{"cat": "LOCATION", "start": 0, "end": 8, "score": 0.65,
                   "text": "borewell", "model": "indicner"}]
    result = _suppress_false_positives(fake_infra, "borewell kharab hai", "indicner")
    assert len(result) == 0, f"D3 suppression failed: {result}"
    print("  [PASS] D3. Infrastructure LOC suppression")

    # E. Postposition trim
    s = {"cat": "LOCATION", "start": 0, "end": 12, "score": 0.7,
         "text": "Narayandasne", "model": "indicner"}
    trimmed = _trim_postposition(s)
    assert trimmed["text"] == "Narayandas", f"Postposition trim failed: {trimmed['text']!r}"
    assert trimmed["cat"]  == "PERSON",     f"Category not promoted: {trimmed['cat']}"
    print("  [PASS] E. Postposition trim + PER promotion")

    # C. Agreement boost
    ind = [{"cat": "PERSON", "start": 0, "end": 6, "score": 0.60,
            "text": "Ramesh", "model": "indicner"}]
    xlm = [{"cat": "PERSON", "start": 0, "end": 6, "score": 0.65,
            "text": "Ramesh", "model": "xlm"}]
    merged = _smart_hybrid_merge(ind, xlm, "Ramesh ne paani ki shikayat ki")
    agreed = [s for s in merged if s.get("agreed")]
    assert len(agreed) == 1, f"Agreement not detected: {merged}"
    assert agreed[0]["score"] > 0.60, "Agreement boost not applied"
    print("  [PASS] C. Agreement boost")

    print(f"\n{'OK — all self-tests passed.' if errors == 0 else f'FAILED: {errors} errors'}")
    return errors == 0


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        ok = _selftest()
        sys.exit(0 if ok else 1)
    # Normal pipeline entry point — identical CLI to v3
    main()
