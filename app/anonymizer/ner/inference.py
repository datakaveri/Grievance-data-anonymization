"""Run the NER models over lines of text and collect their raw spans.

    infer_line_batch(pipe, tasks, ...)   one model, a batch of lines -> raw spans per line
    run_corpus_line_ner(lines, ...)      every model over a whole corpus -> merged entities

A raw span is a dict {cat, start, end, score, text, model}: one entity as a
single model saw it, before merging (ner/spans.py) and filtering (ner/filters.py).
"""

from __future__ import annotations

import gc
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..debug.profiling import NULL_PROFILER
from . import models
from .spans import NerEntity, merge_line_spans

# Longest input, in tokens, sent to a model in one piece. Both models accept
# 512; 380 leaves room for the special tokens and for the tokenizer splitting a
# chunk differently once it is cut out of its line.
MAX_TOKENS = 380

# Model tag -> pipeline category. Anything else (MISC, dates...) is ignored.
_CATEGORY_BY_TAG = {
    "PER": "PERSON", "PERSON": "PERSON",
    "LOC": "LOCATION", "LOCATION": "LOCATION", "GPE": "LOCATION",
    "ORG": "ORGANIZATION", "ORGANIZATION": "ORGANIZATION",
}


@dataclass
class LineTask:
    """One line to run through a model. doc_id/line_no say where to put the result."""

    doc_id: int
    line_no: int
    line_text: str


@dataclass
class ModelLineResult:
    """One model's raw spans for one LineTask (error set when inference failed)."""

    doc_id: int
    line_no: int
    model_name: str
    spans: List[Dict]
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Single-model inference
# ─────────────────────────────────────────────────────────────────────────────


def _parse_pipe_output(items, line: str, min_score: float, model_name: str) -> List[Dict]:
    """Convert a pipeline's output for one line into raw span dicts.

    Keeps PERSON/LOCATION/ORGANIZATION entities scoring at least `min_score`;
    strips B-/I- prefixes from tag names ("B-PER" -> "PER").
    """
    spans: List[Dict] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        score = float(item.get("score", 0.0))
        tag = re.sub(r"^[BI]-", "", str(item.get("entity_group", item.get("entity", ""))).upper())
        category = _CATEGORY_BY_TAG.get(tag)
        if score < min_score or category is None:
            continue
        start, end = item.get("start", 0), item.get("end", 0)
        if start < end:
            spans.append(
                {"cat": category, "start": start, "end": end, "score": score,
                 "text": line[start:end], "model": model_name}
            )
    return spans


def _chunk_line(line: str, tokenizer, max_tokens: int = MAX_TOKENS) -> List[Tuple[int, str]]:
    """Split a too-long line into (offset, chunk) pieces of at most `max_tokens`.

    Cuts only after a sentence end — ".", "।" (Hindi full stop) or a newline —
    and packs as many whole sentences per chunk as fit. A single sentence longer
    than the limit becomes its own chunk (the model then truncates it).
    """
    pieces = [
        (m.start(), m.group())
        for m in re.finditer(r"[^.\n।]*[.\n।]|[^.\n।]+$", line)
        if m.group().strip()
    ]
    chunks: List[Tuple[int, str]] = []
    chunk_start: Optional[int] = None
    chunk = ""
    for start, piece in pieces:
        candidate = chunk + piece
        if chunk and len(tokenizer(candidate, add_special_tokens=True)["input_ids"]) > max_tokens:
            chunks.append((chunk_start, chunk))
            chunk_start, chunk = start, piece
        else:
            if chunk_start is None:
                chunk_start = start
            chunk = candidate
    if chunk.strip():
        chunks.append((chunk_start if chunk_start is not None else 0, chunk))
    return chunks or [(0, line)]


def _infer_one_line(pipe, line: str, min_score: float, model_name: str) -> List[Dict]:
    """Raw spans for a single line, chunking it first if it exceeds MAX_TOKENS.

    Chunk offsets are added back, so spans index the full line. A failing
    inference yields no spans for that piece rather than failing the job.
    """
    if pipe is None or not line.strip():
        return []

    def infer(text: str) -> List[Dict]:
        try:
            return _parse_pipe_output(models.run_pipe(pipe, text), text, min_score, model_name)
        except Exception:
            return []

    if len(pipe.tokenizer(line, add_special_tokens=True)["input_ids"]) <= MAX_TOKENS:
        return infer(line)
    spans: List[Dict] = []
    for offset, chunk in _chunk_line(line, pipe.tokenizer):
        for span in infer(chunk):
            span["start"] += offset
            span["end"] += offset
            spans.append(span)
    return spans


def _per_line_outputs(raw, count: int) -> List:
    """Normalise a pipeline's batch output to one entity list per input line.

    HuggingFace returns a flat entity list for a single string and a list of
    lists for several; anything unexpected becomes empty results.
    """
    if count <= 0:
        return []
    if count == 1:
        if raw is None:
            return [[]]
        if isinstance(raw, list) and (not raw or isinstance(raw[0], dict)):
            return [raw]
        if isinstance(raw, list) and isinstance(raw[0], list):
            return raw
        return [[]]
    if not isinstance(raw, list) or len(raw) != count:
        return [[] for _ in range(count)]
    return raw


def infer_line_batch(
    pipe,
    tasks: List[LineTask],
    min_score: float,
    model_name: str,
    batch_size: Optional[int] = None,
) -> List[ModelLineResult]:
    """Run one model over `tasks`: short lines in one batched call, long lines one by one.

    A line provably fits in MAX_TOKENS when chars + words + 2 <= MAX_TOKENS (a
    sub-word tokenizer emits at most one token per character plus one
    word-boundary piece per word, plus 2 special tokens), so the tokenizer is
    only consulted for lines that might not fit. Lines that do not fit are
    chunked (_infer_one_line) rather than truncated by the model, which would
    silently drop any name past token 512. If the batched call fails, its lines
    are retried one at a time. Returns one ModelLineResult per task, in order.
    """
    if not tasks:
        return []
    if pipe is None:
        return [ModelLineResult(t.doc_id, t.line_no, model_name, []) for t in tasks]

    short: List[LineTask] = []
    long: List[LineTask] = []
    for task in tasks:
        text = task.line_text
        if len(text) + len(text.split()) + 2 <= MAX_TOKENS:
            short.append(task)
            continue
        try:
            fits = len(pipe.tokenizer(text, add_special_tokens=True)["input_ids"]) <= MAX_TOKENS
        except Exception:
            fits = False
        (short if fits else long).append(task)

    spans_by_task: Dict[Tuple[int, int], List[Dict]] = {}
    if short:
        try:
            raw = models.run_pipe(pipe, [t.line_text for t in short], batch_size=batch_size)
            for task, items in zip(short, _per_line_outputs(raw, len(short))):
                spans_by_task.setdefault((task.doc_id, task.line_no), []).extend(
                    _parse_pipe_output(items, task.line_text, min_score, model_name)
                )
        except Exception:
            for task in short:
                spans_by_task[(task.doc_id, task.line_no)] = _infer_one_line(
                    pipe, task.line_text, min_score, model_name
                )
    for task in long:
        spans_by_task[(task.doc_id, task.line_no)] = _infer_one_line(
            pipe, task.line_text, min_score, model_name
        )

    return [
        ModelLineResult(t.doc_id, t.line_no, model_name, spans_by_task.get((t.doc_id, t.line_no), []))
        for t in tasks
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Whole-corpus inference (the pipeline, for every job)
# ─────────────────────────────────────────────────────────────────────────────


def run_corpus_line_ner(
    lines: List[str],
    batch_size: int = models.NER_CORPUS_BATCH_SIZE,
    release_models: bool = False,
    spans_by_model: Optional[Dict[str, List[List[Dict]]]] = None,
    profiler=None,
) -> List[List[NerEntity]]:
    """Run every model over every line; return merged entities, one list per line.

    Built for the pipeline, which hands over every DISTINCT line of the dataset
    at once: each line is inferred exactly once, in real mini-batches, with no
    threads or queues.

    * Models run one after another over the whole corpus, not concurrently: on
      CPU they would fight over the same torch thread pool, and holding one model
      at a time keeps peak memory to about one model (~1 GB).
    * Lines go longest-first, so with tensor batching each batch pads to roughly
      its own longest member, and the slowest batches land early, which makes
      the printed time estimate pessimistic rather than optimistic.
    * release_models drops each model after its pass (nothing reuses it in a
      one-shot job).
    * Pass a dict as spans_by_model to receive {model label: raw spans per line};
      the merged result pools the models and can no longer say which one found
      what, which attribution reporting needs.
    """
    if not lines:
        return []
    if not models.TRANSFORMERS_AVAILABLE:
        print("  [NER] transformers not installed — skipping model inference.", flush=True)
        return [[] for _ in lines]

    profiler = profiler or NULL_PROFILER
    batch_size = max(1, batch_size)
    order = sorted(range(len(lines)), key=lambda i: len(lines[i]), reverse=True)
    raw_spans: List[List[Dict]] = [[] for _ in lines]

    previous_threads = models.torch.get_num_threads()
    if models.NER_TORCH_THREADS > 0:
        models.torch.set_num_threads(models.NER_TORCH_THREADS)
    try:
        tensor_batch = models.tensor_batch_size()
        print(
            f"  [NER] {len(lines)} lines, tensor batch {tensor_batch}, "
            f"{models.torch.get_num_threads()} torch threads",
            flush=True,
        )
        for spec in models.NER_MODELS:
            _one_model_pass(
                spec, lines, order, raw_spans, batch_size, tensor_batch,
                release_models, spans_by_model, profiler,
            )
        with profiler.step("merge spans", parent="ner"):
            return [merge_line_spans(raw_spans[i], lines[i]) for i in range(len(lines))]
    finally:
        models.torch.set_num_threads(previous_threads)


def _one_model_pass(
    spec: models.NerModelSpec,
    lines: List[str],
    order: List[int],
    raw_spans: List[List[Dict]],
    batch_size: int,
    tensor_batch: int,
    release_models: bool,
    spans_by_model: Optional[Dict[str, List[List[Dict]]]],
    profiler,
) -> None:
    """Load one model, run it over all lines (in `order`), append its spans to raw_spans.

    A model that cannot load (e.g. the gated IndicNER without HF_TOKEN) is
    skipped with a message; the job continues with the other model.
    """
    with profiler.step(f"load: {spec.label}", parent="ner", model=spec.model_id):
        pipe = models.load_ner_pipeline(spec.model_id, use_fast=spec.use_fast)
    if pipe is None:
        print(f"  [NER] {spec.label}: unavailable, skipped.", flush=True)
        return

    model_spans = None
    if spans_by_model is not None:
        model_spans = spans_by_model.setdefault(spec.label, [[] for _ in lines])

    with profiler.step(f"infer: {spec.label}", parent="ner", model=spec.model_id, lines=len(lines)):
        started = time.time()
        done = 0
        for offset in range(0, len(order), batch_size):
            chunk = order[offset : offset + batch_size]
            tasks = [LineTask(doc_id=0, line_no=i, line_text=lines[i]) for i in chunk]
            for result in infer_line_batch(pipe, tasks, spec.min_score, spec.label, batch_size=tensor_batch):
                if result.error:
                    print(f"  [NER] {spec.label} error on line {result.line_no}: {result.error}", flush=True)
                raw_spans[result.line_no].extend(result.spans)
                if model_spans is not None:
                    model_spans[result.line_no].extend(result.spans)
            done += len(chunk)
            if offset and (offset // batch_size) % 25 == 0:
                rate = done / max(time.time() - started, 1e-6)
                remaining = (len(order) - done) / max(rate, 1e-6)
                print(
                    f"  [NER] {spec.label}: {done}/{len(order)} lines "
                    f"({rate:.1f} lines/s, ~{remaining / 60:.1f} min left)",
                    flush=True,
                )
        elapsed = time.time() - started
    print(f"  [NER] {spec.label}: {len(order)} lines in {elapsed:.1f}s", flush=True)

    if release_models:
        with profiler.step(f"release: {spec.label}", parent="ner", model=spec.model_id):
            models.release_ner_pipeline(spec.model_id)
            del pipe
            gc.collect()
