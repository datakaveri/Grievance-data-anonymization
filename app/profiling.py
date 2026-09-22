#!/usr/bin/env python3
"""Step timing and peak-memory tracking for the anonymization batch job.

Peak RSS is sampled from /proc rather than read at step boundaries: a model load
allocates and frees inside the step, so start/end readings miss the spike that
actually determines whether the job fits in the container's memory limit.

Everything here is inert unless a run turns debugging on, so the normal batch
path pays nothing.
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def _rss_bytes() -> int:
    """Current resident set size, or 0 where /proc is unavailable."""
    try:
        with open("/proc/self/statm", "rb") as fh:
            return int(fh.read().split()[1]) * _PAGE_SIZE
    except Exception:
        return 0


class Step:
    """One measured step: wall time, and RSS before, after and at its peak."""

    def __init__(self, name: str, parent: Optional[str] = None):
        self.name = name
        self.parent = parent
        self.seconds = 0.0
        self.rss_start = 0
        self.rss_end = 0
        self.rss_peak = 0
        self.detail: dict[str, Any] = {}

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.name,
            "parent": self.parent,
            "seconds": round(self.seconds, 3),
            "rss_start_mb": round(self.rss_start / 1e6, 1),
            "rss_end_mb": round(self.rss_end / 1e6, 1),
            "rss_peak_mb": round(self.rss_peak / 1e6, 1),
            "rss_delta_mb": round((self.rss_end - self.rss_start) / 1e6, 1),
            **self.detail,
        }


class Profiler:
    """Collects Step records. When disabled, every method is a cheap no-op."""

    def __init__(self, enabled: bool = False, sample_interval: float = 0.05):
        self.enabled = enabled
        self.sample_interval = sample_interval
        self.steps: list[Step] = []
        self._peak = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._sampler: Optional[threading.Thread] = None

    # ── RSS sampling ────────────────────────────────────────────────────
    def _sample_loop(self) -> None:
        while not self._stop.wait(self.sample_interval):
            rss = _rss_bytes()
            with self._lock:
                if rss > self._peak:
                    self._peak = rss

    def start(self) -> None:
        if not self.enabled or self._sampler is not None:
            return
        self._sampler = threading.Thread(
            target=self._sample_loop, name="rss-sampler", daemon=True
        )
        self._sampler.start()

    def stop(self) -> None:
        if self._sampler is None:
            return
        self._stop.set()
        self._sampler.join(timeout=2)
        self._sampler = None

    # ── measurement ─────────────────────────────────────────────────────
    @contextmanager
    def step(self, name: str, parent: Optional[str] = None, **detail):
        """Measure the enclosed block. Yields the Step so callers can add detail."""
        if not self.enabled:
            yield Step(name, parent)
            return

        record = Step(name, parent)
        record.detail.update(detail)
        record.rss_start = _rss_bytes()
        with self._lock:
            self._peak = record.rss_start
        started = time.perf_counter()
        try:
            yield record
        finally:
            record.seconds = time.perf_counter() - started
            record.rss_end = _rss_bytes()
            with self._lock:
                record.rss_peak = max(self._peak, record.rss_end)
            self.steps.append(record)
            print(
                f"  [profile] {name:<38} {record.seconds:8.2f}s  "
                f"peak {record.rss_peak / 1e6:7.1f} MB  "
                f"delta {(record.rss_end - record.rss_start) / 1e6:+7.1f} MB",
                flush=True,
            )

    # ── reporting ───────────────────────────────────────────────────────
    def write(self, path: Path, extra: Optional[dict[str, Any]] = None) -> None:
        if not self.enabled:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "steps": [s.as_dict() for s in self.steps],
            "total_seconds": round(sum(s.seconds for s in self.steps if s.parent is None), 3),
            "max_rss_mb": round(max((s.rss_peak for s in self.steps), default=0) / 1e6, 1),
        }
        if extra:
            payload.update(extra)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"[Batch] Wrote profile: {path}", flush=True)

    def table(self) -> str:
        """Human-readable summary, printed at the end of a debug run."""
        if not self.enabled or not self.steps:
            return ""
        width = max(len(s.name) + (2 if s.parent else 0) for s in self.steps)
        lines = [
            "",
            "  " + "Step".ljust(width) + "      Time        Peak RSS       Delta",
            "  " + "-" * (width + 38),
        ]
        for s in self.steps:
            label = ("  " if s.parent else "") + s.name
            lines.append(
                f"  {label.ljust(width)}  {s.seconds:8.2f}s  {s.rss_peak / 1e6:9.1f} MB  "
                f"{(s.rss_end - s.rss_start) / 1e6:+8.1f} MB"
            )
        return "\n".join(lines)


# A disabled profiler the modules can import and call unconditionally.
NULL_PROFILER = Profiler(enabled=False)
