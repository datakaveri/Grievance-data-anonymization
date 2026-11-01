"""The anonymization flow — written once, used by every job.

    anonymize(values, settings) -> {(column, text): Result}

A job (jobs/table.py, jobs/nested_json.py) turns its input into
(column, text) pairs, calls anonymize(), and writes each Result back where the
text came from. Everything between reading and writing happens here:

  1. cleanup    repair HTML entities and broken Hindi              (cleanup.py)
  2. rules      regex detection, per value                         (rules/, scanner.py)
  3. models     hybrid NER over every DISTINCT line of the corpus  (ner/)
  4. redaction  pool rule + model hits, apply the per-type policy  (redaction.py)

Values repeat a lot (the same complaint text in many rows), so each distinct
(column, text) is processed once and the Result is shared by all its rows.
`column` only chooses the hash salt: equal values in one column hash equally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .cleanup import normalize_source_text
from .config import DetectionSettings
from .debug.profiling import NULL_PROFILER, Profiler
from .ner import models
from .ner.index import build_ner_index
from .redaction import redact_text
from .scanner import scan_cell


@dataclass
class Result:
    """What one input value becomes.

    cleaned  the text after step 1 — what detection ran on
    text     the cleaned, redacted text — equal to `cleaned` when nothing was found
    audit    one entry per detection (see redaction.redact_text)
    error    set when processing this value raised; `text` is then only cleaned
    """

    cleaned: str
    text: str
    audit: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


def anonymize(
    values: Iterable[Tuple[str, str]],
    settings: DetectionSettings,
    include_values: bool = False,
    attribute: bool = False,
    profiler: Profiler = NULL_PROFILER,
) -> Dict[Tuple[str, str], Result]:
    """Run steps 1-4 over (column, text) pairs; return a Result per distinct pair.

    Non-string and empty values are skipped (the caller leaves them as they
    are). include_values puts the detected text into the audit — personal data,
    for debug runs and reports only. attribute keeps per-model details on the
    model hits (debug reports). A value that raises gets a Result with `error`
    set; the caller decides whether that aborts the job.
    """
    # ── 1. cleanup, once per distinct text ────────────────────────────────
    with profiler.step("normalize source text"):
        pairs: Dict[Tuple[str, str], None] = {}
        cells = 0
        for column, text in values:
            if isinstance(text, str) and text:
                cells += 1
                pairs[(column, text)] = None
        clean = normalize_source_text if settings.normalize_source_text else (lambda text: text)
        cleaned = {text: clean(text) for _, text in pairs}

    with profiler.step("collect distinct values") as collect:
        texts = sorted({text for text in cleaned.values() if text})
        collect.detail.update(cells=cells, distinct=len(texts))
    print(f"[Pipeline] {cells} cells; {len(texts)} distinct values.", flush=True)

    # ── 3. models: one pass over every distinct line of the corpus ───────
    # (before step 2 because it is the expensive, batched one; rules are cheap)
    with profiler.step("ner") as ner_step:
        ner_index = (
            build_ner_index(
                texts,
                models.NER_CORPUS_BATCH_SIZE if settings.ner_batch_size is None else settings.ner_batch_size,
                profiler=profiler,
                attribute=attribute,
                allow_terms=settings.ner_allow_terms,
            )
            if texts
            else {}
        )
        ner_step.detail.update(distinct_values=len(texts))

    # ── 2 + 4. rules, then redaction, per distinct (column, text) ────────
    results: Dict[Tuple[str, str], Result] = {}
    salts: Dict[str, str] = {}
    with profiler.step("apply policy") as apply_step:
        for column, raw in pairs:
            text = cleaned[raw]
            try:
                rule_hits = scan_cell(text, settings.scan_options)
                redacted, audit = redact_text(
                    text, rule_hits, ner_index.get(text, []), column, salts,
                    minimum_confidence=settings.minimum_confidence, include_values=include_values,
                )
                results[(column, raw)] = Result(text, redacted, audit)
            except Exception as exc:
                results[(column, raw)] = Result(text, text, error=str(exc))
        # Counted per distinct value; the job's audit counts them per row.
        apply_step.detail.update(
            distinct_values_detections=sum(len(r.audit) for r in results.values()),
            distinct_values_failed=sum(r.error is not None for r in results.values()),
        )
    return results
