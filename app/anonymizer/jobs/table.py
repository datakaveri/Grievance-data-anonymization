"""The table job: anonymize configured columns of a CSV / Excel / JSON table.

    run(config_path, root) -> summary dict        (CLI: app/batch_pipeline.py,
                                                   the Docker entrypoint)

  read     find + validate the dataset config, read the input table
  run      every cell of the configured columns -> pipeline.anonymize()
  write    put each Result back into its cell; write the staged CSV for SKALD,
           the audit JSON and, in debug runs, the debug reports
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from ..config import BatchError, TableJobSettings, free_text_block, load_dataset_config, resolve_config_file, resolve_path
from ..debug import reports as debug_reports
from ..debug.profiling import Profiler
from ..ner import models
from ..pipeline import Result, anonymize

SUPPORTED_SOURCE_EXTS = {".csv", ".json", ".xls", ".xlsx"}


def run(config_path: str, root: str | None = None) -> Dict[str, Any]:
    """Anonymize the configured columns of one dataset; return a short summary.

    `config_path` is a config file or a directory to scan for one. Relative
    paths in the config resolve against `root`, or by default against the
    config directory's parent (in the container config/ and data/ are siblings
    under /app). Returns {"status": "disabled"|"completed", ...}.
    """
    # ── read ──────────────────────────────────────────────────────────────
    config_arg = Path(config_path).resolve()
    config_file = resolve_config_file(config_arg)
    config_dir = config_arg if config_arg.is_dir() else config_file.parent
    work_root = Path(root).resolve() if root else config_dir.parent
    dataset = load_dataset_config(config_file)
    source = resolve_source_input(dataset, work_root)
    block = free_text_block(dataset)

    debug = debug_reports.debug_settings(block)
    profiler = Profiler(enabled=debug["enabled"])
    profiler.start()
    if debug["enabled"]:
        print("[Batch] Debug instrumentation on.", flush=True)
        if debug["include_values"]:
            print("[Batch] Attribution report includes detected PII text (debug.include_values).", flush=True)

    with profiler.step("read input table", rows_source=str(source)):
        df = read_table(source)
    if block.get("enabled") is not True:
        print("[Batch] free_text_anonymization.enabled is false; no staged file written.")
        return {"status": "disabled", "rows": len(df), "columns": len(df.columns)}
    if not models.TRANSFORMERS_AVAILABLE:
        raise BatchError(
            "NER dependencies are not installed. Run 'pip install -r requirements.txt' "
            "and 'python app/model_setup.py' before running the table job."
        )
    settings = TableJobSettings.from_block(block, list(df.columns))
    column_locs = {column: df.columns.get_loc(column) for column in settings.columns}
    print(f"[Batch] {len(df)} rows x {len(settings.columns)} column(s).", flush=True)

    # ── run ───────────────────────────────────────────────────────────────
    started = time.time()
    results = anonymize(
        ((column, value) for column, loc in column_locs.items() for value in df.iloc[:, loc].tolist()),
        settings.detection,
        include_values=debug["enabled"] and debug["include_values"],
        attribute=debug["enabled"],
        profiler=profiler,
    )
    audit, failures = _write_back(df, settings, column_locs, results)
    print(f"[Batch] Anonymization took {time.time() - started:.1f}s", flush=True)

    # ── write ─────────────────────────────────────────────────────────────
    staged = resolve_path(settings.staged_input_path, work_root)
    with profiler.step("write staged csv"):
        write_csv_atomically(df, staged)
    if settings.audit_output_path:
        with profiler.step("write audit json"):
            audit_path = resolve_path(settings.audit_output_path, work_root)
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            with audit_path.open("w", encoding="utf-8") as fh:
                json.dump(
                    {"rows": len(df), "columns": list(df.columns), "detections": audit, "failures": failures},
                    fh, indent=2, ensure_ascii=False,
                )
    print(f"[Batch] Wrote staged dataset: {staged}")
    print(f"[Batch] Rows: {len(df)}; detections: {len(audit)}; failures: {len(failures)}")

    if debug["enabled"]:
        distinct = len({text for _, text in results})
        _write_debug_reports(debug, audit, profiler, work_root, df, settings.columns, distinct)
    profiler.stop()
    return {
        "status": "completed",
        "rows": len(df),
        "detections": len(audit),
        "failures": len(failures),
        "staged_input_path": str(staged),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reading and writing the table
# ─────────────────────────────────────────────────────────────────────────────


def resolve_source_input(dataset: Dict[str, Any], work_root: Path) -> Path:
    """Locate the raw dataset to anonymize.

    `source_input_path` in the dataset config is a convenience for local runs.
    The config shared with SKALD downstream has no such key: SKALD finds its
    input by scanning data/ for exactly one non-empty file (its
    list_non_empty_csvs), so without the key the same convention is applied to
    the same mounted data/ directory.
    """
    configured = dataset.get("source_input_path")
    if configured:
        return resolve_path(configured, work_root)

    data_dir = work_root / "data"
    if not data_dir.is_dir():
        raise BatchError(f"Data directory not found: {data_dir}")
    candidates = sorted(
        p for p in data_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_SOURCE_EXTS and p.stat().st_size > 0
    )
    if not candidates:
        raise BatchError(f"No CSV, JSON, or Excel file found in {data_dir}")
    if len(candidates) > 1:
        raise BatchError(
            f"Expected exactly one input file in {data_dir}, found {len(candidates)}: "
            f"{[p.name for p in candidates]}"
        )
    return candidates[0]


def read_table(path: Path) -> pd.DataFrame:
    """Load a CSV, XLS/XLSX or JSON table with every cell as a string-or-None object.

    CSV keeps empty cells as "" (keep_default_na=False) so "NA" or "NULL" text is
    not turned into missing values. JSON may be a list of records, an object
    holding one under "records" or "data", or a column-oriented object.
    """
    if not path.exists():
        raise BatchError(f"Source input does not exist: {path}")
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(path, dtype=object, keep_default_na=False)
        if suffix in {".xls", ".xlsx"}:
            return pd.read_excel(path, dtype=object)
        if suffix == ".json":
            with path.open(encoding="utf-8") as fh:
                payload = json.load(fh)
            if isinstance(payload, list):
                return pd.DataFrame(payload)
            if isinstance(payload, dict):
                records = payload.get("records", payload.get("data"))
                return pd.DataFrame(records if isinstance(records, list) else payload)
    except Exception as exc:
        raise BatchError(f"Could not read {path}: {exc}") from exc
    raise BatchError(f"Unsupported input format {path.suffix!r}; use CSV, JSON, XLS, or XLSX")


def _write_back(
    df: pd.DataFrame,
    settings: TableJobSettings,
    column_locs: Dict[str, int],
    results: Dict[Tuple[str, str], Result],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Put each cell's Result into `df` in place; return (audit, failures) per row.

    Audit entries get their row (1-based) and column, row-major in configured
    column order, as consumers expect. A failed value is recorded once per row
    holding it; with on_failure="fail" the first one aborts the job.
    """
    audit: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for column in settings.columns:
        loc = column_locs[column]
        values = df.iloc[:, loc].tolist()
        for row_index, value in enumerate(values):
            if not isinstance(value, str) or not value:  # left as is (pipeline skips these too)
                continue
            result = results[(column, value)]
            values[row_index] = result.text
            if result.error is not None:
                failures.append({"row": row_index + 1, "column": column, "error": result.error})
                if settings.on_failure == "fail":
                    raise BatchError(
                        f"Anonymization failed at row {row_index + 1}, column {column!r}: {result.error}"
                    )
                continue
            audit.extend({"row": row_index + 1, "column": column, **entry} for entry in result.audit)
        df.isetitem(loc, values)

    column_order = {column: rank for rank, column in enumerate(settings.columns)}
    audit.sort(key=lambda item: (item["row"], column_order[item["column"]]))
    failures.sort(key=lambda item: (item["row"], column_order[item["column"]]))
    return audit, failures


def write_csv_atomically(df: pd.DataFrame, path: Path) -> None:
    """Write `df` to `path` as CSV, replacing any old file only once the write succeeded.

    The CSV is written to a temporary file in the same directory and moved over
    the destination, so a crash never leaves a half-written staged file that
    SKALD might pick up.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=path.suffix, delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        df.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_debug_reports(debug, audit, profiler, work_root: Path, df, columns, distinct_values: int) -> None:
    """Write the attribution, NER-term and profile reports of a debug run."""
    with profiler.step("write debug reports"):
        if debug["include_values"]:
            debug_reports.write_ner_terms(audit, resolve_path(debug["ner_terms_path"], work_root))
        debug_reports.write_attribution(
            audit,
            resolve_path(debug["attribution_output_path"], work_root),
            resolve_path(debug["attribution_text_path"], work_root),
            include_values=debug["include_values"],
            max_rows=debug["max_rows"],
        )
    profiler.stop()
    profiler.write(
        resolve_path(debug["profile_output_path"], work_root),
        extra={
            "rows": len(df),
            "columns": columns,
            "distinct_values": distinct_values,
            "detections": len(audit),
            "torch_threads": models.torch_threads(),
            "tensor_batch_size": models.tensor_batch_size(),
            "models": debug_reports.model_load_summary(profiler),
        },
    )
    print(profiler.table(), flush=True)
