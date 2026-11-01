"""Model detection for many cells at once: corpus NER projected back onto cells.

    build_ner_index(texts, ...) -> {cell text: [detection dicts]}

pipeline.py hands over every distinct cell value. Each cell is split into
lines; every DISTINCT line in the whole run is inferred exactly once
(ner/inference.py), merged (ner/spans.py), filtered (ner/filters.py), and the
surviving entities are mapped back onto every cell containing that line, with
offsets relative to the cell. On a real grievance export about four in five
complaint bodies are duplicates, so this removes most of the model work.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..debug.profiling import NULL_PROFILER
from ..text import line_bases
from .filters import filter_entities, word_casing_stats
from .inference import run_corpus_line_ner

# Lines shorter than this are not worth a model pass (blank lines, "ok", "--").
MIN_NER_LINE_CHARS = 3

# A line that is one whitespace-free token with a digit, a file extension, a URL
# or an e-mail shape is a machine string, not language: the models only guess
# at it, and tagged the system file name "CRFBD12345678901.pdf" as an
# ORGANIZATION (#15). Such lines skip the models; the regex rules still scan them.
#   skipped:  "CRFBD12345678901.pdf", "9876543210", "www.x.com/a?b=1", "ab@c.in"
#   kept:     "NULL", "Rameshkumar", "Upload ATR again", "FIR No. 140"
_MACHINE_TOKEN = re.compile(r"\S*\d\S*|\S+\.[A-Za-z0-9]{2,5}|\S+://\S+|\S+@\S+")


def ner_lines(text: str) -> List[Tuple[int, str]]:
    """(line index, stripped line) for each line of `text` worth sending to the models."""
    result = []
    for line_idx, line in enumerate(text.splitlines()):
        stripped = line.strip()
        if len(stripped) >= MIN_NER_LINE_CHARS and not _MACHINE_TOKEN.fullmatch(stripped):
            result.append((line_idx, stripped))
    return result


def _models_for_span(start: int, end: int, line_spans: List[Tuple[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Which models fired on characters [start, end) of a line, best score first.

    Merging pools all models' spans, so a merged entity no longer records where
    it came from; a model whose raw span overlaps the entity is taken to have
    found it. Used only for the debug attribution report.
    """
    found: Dict[str, float] = {}
    for label, span in line_spans:
        if span["start"] < end and start < span["end"]:
            score = float(span.get("score", 0.0))
            if score > found.get(label, -1.0):
                found[label] = score
    return [
        {"model": label, "confidence": round(score, 4)}
        for label, score in sorted(found.items(), key=lambda kv: -kv[1])
    ]


def build_ner_index(
    texts: Sequence[str],
    batch_size: int,
    profiler=None,
    attribute: bool = False,
    allow_terms: Optional[Sequence[str]] = None,
    casing: Optional[Dict[str, List[int]]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Run hybrid NER once over every distinct line of `texts`; map results onto each text.

    Returns {text: [{start, end, label, source, confidence, value(, models)}]}
    with zero-based, cell-relative offsets — the same shape scanner.scan_cell()
    returns, so the redaction step can merge both.

    attribute    also attach "models": which model(s) found each entity (debug).
    allow_terms  the dataset's ner_allow_terms, for the false-positive filter.
    casing       word-casing evidence for the vocabulary filter; computed from
                 `texts` when not given (pass it in to judge a small batch by a
                 larger corpus).
    """
    profiler = profiler or NULL_PROFILER
    with profiler.step("build line corpus", parent="ner"):
        line_ids: Dict[str, int] = {}
        corpus: List[str] = []
        for text in texts:
            for _, stripped in ner_lines(text):
                if stripped not in line_ids:
                    line_ids[stripped] = len(corpus)
                    corpus.append(stripped)

    print(f"[Batch] Hybrid NER over {len(corpus)} distinct lines from {len(texts)} distinct values …", flush=True)
    spans_by_model: Optional[Dict[str, List[List[Dict[str, Any]]]]] = {} if attribute else None
    entities = run_corpus_line_ner(
        corpus, batch_size=batch_size, release_models=True, spans_by_model=spans_by_model, profiler=profiler
    )

    with profiler.step("filter false positives", parent="ner"):
        if casing is None:
            casing = word_casing_stats(texts)
        dropped = filter_entities(entities, corpus, allow_terms, casing)
    if dropped:
        print(f"[Batch] NER entities cleared as not personal: {dropped}", flush=True)

    # Per-line (model label, raw span) pairs, flattened once for attribution.
    per_line_spans: List[List[Tuple[str, Dict[str, Any]]]] = []
    if spans_by_model:
        per_line_spans = [[] for _ in corpus]
        for label, line_lists in spans_by_model.items():
            for line_id, spans in enumerate(line_lists):
                per_line_spans[line_id].extend((label, span) for span in spans)

    with profiler.step("project spans onto cells", parent="ner"):
        index: Dict[str, List[Dict[str, Any]]] = {}
        for text in texts:
            bases = line_bases(text)
            hits: List[Dict[str, Any]] = []
            for line_idx, stripped in ner_lines(text):
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
                        hit["models"] = _models_for_span(entity.start_char - 1, entity.end_char, per_line_spans[line_id])
                    hits.append(hit)
            index[text] = hits
    return index
