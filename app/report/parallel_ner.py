"""Run every NER model over every line of the documents, one worker thread per model.

    analyze_records(records)   rule scan + NER for a list of FileRecords, in place

Architecture (fan-out / fan-in):

    documents ──> dispatcher ──> one input queue per model ──> model workers
                                                                    │
    per-line merge <── result queue <───────────────────────────────┘

Each model gets its own thread, so on CPU the two models overlap. Bounded
queues apply backpressure; workers mini-batch lines (24 by default) to cut
tokenizer overhead. When every model has answered for a line, the line's
results are stored per model and as the hybrid merge of both. The anonymization
pipeline does not use this (it runs ner/inference.run_corpus_line_ner over distinct
lines instead); this suits interactive runs over a few documents.
"""

from __future__ import annotations

import threading
from queue import Empty, Queue
from typing import Dict, List, Tuple

from anonymizer.ner import models
from anonymizer.ner.inference import LineTask, ModelLineResult, infer_line_batch
from anonymizer.ner.spans import NerEntity, merge_line_spans
from anonymizer.scanner import scan_text

from .records import FileRecord, LineNerResult

NER_LINE_BATCH_SIZE = 24          # lines per worker mini-batch
NER_QUEUE_MAXSIZE = 64            # bound on each queue (backpressure)
_RESULT_GET_TIMEOUT_S = 1.0       # how long the merger waits per poll
_WORKER_JOIN_TIMEOUT_S = 300.0    # give up on silent workers after this long
_MIN_LINE_CHARS = 3               # shorter lines are skipped (blank, "ok", "--")


def _model_worker(spec: models.NerModelSpec, pipe, in_queue: Queue, out_queue: Queue, batch_size: int) -> None:
    """Worker thread for one model: batch incoming LineTasks, emit ModelLineResults.

    Flushes a batch when it is full or the input queue is idle for 0.2 s. A None
    task means "no more input": flush, forward None, exit. A failing batch
    produces empty results carrying the error instead of killing the thread.
    """
    batch: List[LineTask] = []

    def flush() -> None:
        if not batch:
            return
        try:
            for item in infer_line_batch(pipe, list(batch), spec.min_score, spec.label):
                out_queue.put(item)
        except Exception as exc:
            for task in batch:
                out_queue.put(ModelLineResult(task.doc_id, task.line_no, spec.label, [], error=str(exc)))
        batch.clear()

    while True:
        try:
            item = in_queue.get(timeout=0.2)
        except Empty:
            flush()
            continue
        if item is None:
            flush()
            out_queue.put(None)
            return
        batch.append(item)
        if len(batch) >= batch_size:
            flush()


def _store_line_results(rec: FileRecord, line_no: int, line_text: str, model_spans: Dict[str, List[Dict]]) -> None:
    """Merge one line's raw spans per model and for the hybrid; append LineNerResults to `rec`.

    `missed` for each model lists the (text, category) pairs some other model
    (or the hybrid) found in this line but this one did not.
    """
    per_model: Dict[str, List[NerEntity]] = {
        spec.label: merge_line_spans(model_spans.get(spec.label, []), line_text) for spec in models.NER_MODELS
    }
    hybrid_spans = [span for spec in models.NER_MODELS for span in model_spans.get(spec.label, [])]
    per_model[models.HYBRID_LABEL] = merge_line_spans(hybrid_spans, line_text)

    everything = {(e.category, e.text) for ents in per_model.values() for e in ents}
    for label, ents in per_model.items():
        own = {(e.category, e.text) for e in ents}
        rec.line_ners[label].append(
            LineNerResult(
                model=label,
                line_no=line_no,
                text=line_text,
                entities=ents,
                persons=[e.text for e in ents if e.category == "PERSON"],
                locs=[e.text for e in ents if e.category == "LOCATION"],
                orgs=[e.text for e in ents if e.category == "ORGANIZATION"],
                missed=[f"{val} ({cat})" for cat, val in everything if (cat, val) not in own],
            )
        )


def run_line_by_line_ner(
    records: List[FileRecord],
    batch_size: int = NER_LINE_BATCH_SIZE,
    queue_size: int = NER_QUEUE_MAXSIZE,
) -> None:
    """Fill rec.line_ners of every record: per model and hybrid, line by line.

    Lines shorter than 3 characters are skipped. If a worker dies or stalls for
    _WORKER_JOIN_TIMEOUT_S, whatever arrived is merged anyway (missing models
    count as finding nothing), so a failure never leaves a document hanging.
    """
    if not models.TRANSFORMERS_AVAILABLE:
        print("  [NER] transformers not installed — skipping model inference.", flush=True)
        return

    print(f"  [NER] Loading HuggingFace models on {models.DEVICE_NAME} …", flush=True)
    pipes = {spec.label: (spec, models.load_ner_pipeline(spec.model_id, use_fast=spec.use_fast))
             for spec in models.NER_MODELS}
    labels = list(pipes)
    for rec in records:
        rec.line_ners = {label: [] for label in labels + [models.HYBRID_LABEL]}

    tasks: List[LineTask] = []
    line_lookup: Dict[Tuple[int, int], str] = {}
    for doc_id, rec in enumerate(records):
        for line_idx, line in enumerate(rec.raw_text.splitlines()):
            text = line.strip()
            if len(text) < _MIN_LINE_CHARS:
                continue
            tasks.append(LineTask(doc_id=doc_id, line_no=line_idx + 1, line_text=text))
            line_lookup[(doc_id, line_idx + 1)] = text
    if not tasks:
        return

    bound = max(8, queue_size)
    in_queues = {label: Queue(maxsize=bound) for label in labels}
    result_queue: Queue = Queue(maxsize=max(bound * len(labels), bound))
    workers = []
    for label, (spec, pipe) in pipes.items():
        thread = threading.Thread(
            target=_model_worker,
            name=f"ner-{label.split()[0]}",
            args=(spec, pipe, in_queues[label], result_queue, max(1, batch_size)),
            daemon=True,
        )
        thread.start()
        workers.append(thread)
    print(
        f"  [NER] Parallel workers: {len(workers)} models, {len(tasks)} lines, "
        f"batch={max(1, batch_size)}, device={models.DEVICE_NAME}",
        flush=True,
    )

    def dispatch() -> None:
        for task in tasks:
            for q in in_queues.values():
                q.put(task)
        for q in in_queues.values():
            q.put(None)

    dispatcher = threading.Thread(target=dispatch, name="ner-dispatcher", daemon=True)
    dispatcher.start()

    # Fan-in: collect results until every worker has sent its end marker.
    pending: Dict[Tuple[int, int], Dict[str, List[Dict]]] = {}
    workers_done = received = idle_rounds = 0
    max_idle = int(_WORKER_JOIN_TIMEOUT_S / _RESULT_GET_TIMEOUT_S) + 1
    while workers_done < len(workers) and idle_rounds < max_idle:
        try:
            item = result_queue.get(timeout=_RESULT_GET_TIMEOUT_S)
            idle_rounds = 0
        except Empty:
            idle_rounds += 1
            if not any(w.is_alive() for w in workers) and result_queue.empty():
                break
            continue
        if item is None:
            workers_done += 1
            continue
        received += 1
        if item.error:
            print(f"  [NER] Worker error {item.model_name} doc={item.doc_id} line={item.line_no}: {item.error}", flush=True)
        key = (item.doc_id, item.line_no)
        pending.setdefault(key, {})[item.model_name] = item.spans
        if len(pending[key]) >= len(labels):
            _store_line_results(records[item.doc_id], item.line_no, line_lookup[key], pending.pop(key))

    dispatcher.join(timeout=5)
    for worker in workers:
        worker.join(timeout=2)

    # Lines some model never answered for: merge what arrived.
    for key, model_spans in list(pending.items()):
        doc_id, line_no = key
        if doc_id >= len(records) or key not in line_lookup:
            continue
        missing = [label for label in labels if label not in model_spans]
        if missing:
            print(
                f"  [NER] Incomplete results for {records[doc_id].filename} line {line_no}; "
                f"missing {missing} (empty spans used).",
                flush=True,
            )
        _store_line_results(records[doc_id], line_no, line_lookup[key], model_spans)
        pending.pop(key, None)

    for rec in records:
        for results in rec.line_ners.values():
            results.sort(key=lambda r: r.line_no)
    print(f"  [NER] Fan-in complete: {received}/{len(tasks) * len(labels)} model-line results.", flush=True)


def analyze_records(
    records: List[FileRecord],
    batch_size: int = NER_LINE_BATCH_SIZE,
    queue_size: int = NER_QUEUE_MAXSIZE,
) -> None:
    """Rule scan (in a background thread) and NER (worker threads) for every record."""
    errors: List[BaseException] = []

    def scan_all() -> None:
        try:
            for rec in records:
                rec.pii_hits = scan_text(rec.raw_text)
        except BaseException as exc:
            errors.append(exc)

    scanner_thread = threading.Thread(target=scan_all, name="pii-dispatcher", daemon=True)
    scanner_thread.start()
    run_line_by_line_ner(records, batch_size=batch_size, queue_size=queue_size)
    scanner_thread.join()
    if errors:
        raise errors[0]
