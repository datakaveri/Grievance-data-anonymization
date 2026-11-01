"""Find, load and validate a dataset config.

A dataset config looks like:

    {
      "data_type": "jan_samvaad_reminders",
      "jan_samvaad_reminders": {
        "source_input_path": "data/grievance_unique/jan_samvaad_reminders.csv",
        "free_text_anonymization": { "enabled": true, "columns": ["remarks"], ... }
      }
    }

`data_type` names the key holding the dataset object; free_text_anonymization
holds the settings (documented key by key in README.md). They are read into:

  DetectionSettings   how to detect and redact — used by every job
  TableJobSettings    the table job's columns and output paths, plus a DetectionSettings
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .rules import ScanOptions

# skald-image's own config. It may sit in the same mounted directory and must
# never be mistaken for this app's config (fetch_data.py treats its presence as
# the image-job signal).
RESERVED_CONFIG_NAMES = {"pipeline_config.json"}

# Settings that must be lists of strings when present.
_STRING_LIST_KEYS = ("reference_keywords", "strong_reference_keywords", "ner_allow_terms", "ner_deny_terms")


class BatchError(RuntimeError):
    """An expected, user-actionable failure: bad config, missing input, etc."""


def resolve_path(value: str, root: Path) -> Path:
    """`value` as a Path; relative paths are taken relative to `root`."""
    candidate = Path(value)
    return candidate if candidate.is_absolute() else root / candidate


def resolve_config_file(config_arg: Path) -> Path:
    """Locate the dataset config JSON from a file path or a directory to scan.

    A file is used as is (local runs: `--config config/config.json`). In the
    container the deployment does not control what the UI names the file, so a
    directory is scanned:
      * a file literally named config.json wins, even alongside others;
      * otherwise exactly one .json (reserved names excluded) is used;
      * none, or several without a config.json, is an error naming the candidates.
    """
    if config_arg.is_file():
        return config_arg
    if not config_arg.is_dir():
        raise BatchError(f"Config path not found: {config_arg}")
    candidates = sorted(
        p for p in config_arg.iterdir()
        if p.is_file() and p.suffix.lower() == ".json" and p.name not in RESERVED_CONFIG_NAMES
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


def load_dataset_config(config_path: Path) -> Dict[str, Any]:
    """Read the config file and return the dataset object named by `data_type`."""
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


def free_text_block(dataset: Dict[str, Any]) -> Dict[str, Any]:
    """The dataset's free_text_anonymization object ({} when absent)."""
    settings = dataset.get("free_text_anonymization", {})
    if not isinstance(settings, dict):
        raise BatchError("free_text_anonymization must be an object")
    return settings


@dataclass
class DetectionSettings:
    """How to detect and redact — shared by every job (table and JSON).

    minimum_confidence     model detections below this are ignored (default 0)
    normalize_source_text  run cleanup.normalize_source_text first (default on)
    ner_batch_size         lines per inference call; None = NER_CORPUS_BATCH_SIZE
    ner_allow_terms        local terms the models tag that are never personal
    scan_options           reference-keyword lists and deny terms for the rules
    """

    minimum_confidence: float = 0.0
    normalize_source_text: bool = True
    ner_batch_size: Optional[int] = None
    ner_allow_terms: List[str] = field(default_factory=list)
    scan_options: ScanOptions = field(default_factory=ScanOptions)

    @classmethod
    def from_block(cls, block: Dict[str, Any]) -> "DetectionSettings":
        """Read the detection keys of a free_text_anonymization block; BatchError if malformed."""
        for key in _STRING_LIST_KEYS:
            value = block.get(key)
            if value is not None and (not isinstance(value, list) or not all(isinstance(k, str) for k in value)):
                raise BatchError(f"free_text_anonymization.{key} must be a list of strings")
        batch_size = block.get("ner_batch_size")
        return cls(
            minimum_confidence=float(block.get("minimum_confidence", 0.0)),
            normalize_source_text=bool(block.get("normalize_source_text", True)),
            ner_batch_size=None if batch_size is None else int(batch_size),
            ner_allow_terms=list(block.get("ner_allow_terms") or []),
            scan_options=ScanOptions.from_settings(block),
        )


@dataclass
class TableJobSettings:
    """Validated free_text_anonymization settings of the table job.

    columns                columns to anonymize (must exist in the input)
    staged_input_path      where the anonymized CSV goes (must end in .csv)
    audit_output_path      optional JSON audit of every replacement
    on_failure             "fail" aborts on a cell error, "continue" records it
    detection              cutoff, cleanup, allow/deny lists (DetectionSettings)
    """

    columns: List[str]
    staged_input_path: str
    audit_output_path: Optional[str] = None
    on_failure: str = "fail"
    detection: DetectionSettings = field(default_factory=DetectionSettings)

    @classmethod
    def from_block(cls, block: Dict[str, Any], available_columns: Sequence[str]) -> "TableJobSettings":
        """Validate a free_text_anonymization block against the input's columns.

        Raises BatchError with the problem spelled out. Everything is checked up
        front, including the output path, so a bad config fails before any
        model work rather than after it.
        """
        columns = block.get("columns")
        if not isinstance(columns, list) or not all(isinstance(col, str) for col in columns):
            raise BatchError("free_text_anonymization.columns must be a list of column names")
        on_failure = block.get("on_failure", "fail")
        if on_failure not in {"fail", "continue"}:
            raise BatchError("free_text_anonymization.on_failure must be 'fail' or 'continue'")
        missing = [col for col in columns if col not in available_columns]
        if missing:
            raise BatchError(f"Configured columns are missing from input: {missing}")
        detection = DetectionSettings.from_block(block)

        staged = block.get("staged_input_path", "")
        if not Path(staged).name:
            raise BatchError("free_text_anonymization.staged_input_path is required")
        if Path(staged).suffix.lower() != ".csv":
            raise BatchError(
                "free_text_anonymization.staged_input_path must end with '.csv' "
                "because SKALD consumes the staged CSV"
            )
        return cls(
            columns=columns,
            staged_input_path=staged,
            audit_output_path=block.get("audit_output_path"),
            on_failure=on_failure,
            detection=detection,
        )
