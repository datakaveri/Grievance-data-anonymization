#!/usr/bin/env python3
"""Anonymize every remarks field of nested JSON records (e.g. one complaint per file).

    python app/json_remarks.py <file.json | folder> ... [--out DIR] [--config CONFIG]

Without --config: 0.75 confidence cutoff, no allow/deny lists. With --config,
the free_text_anonymization settings of that dataset config are reused. The
work is done by anonymizer/jobs/nested_json.py.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from anonymizer.jobs.nested_json import run, settings_from  # noqa: E402


def main() -> int:
    """Collect the input files and run the remarks anonymization over them."""
    parser = argparse.ArgumentParser(description="Anonymize the remarks fields of JSON records")
    parser.add_argument("inputs", nargs="+", help="JSON files, or directories of them")
    parser.add_argument("--out", default="output/json_remarks", help="Output directory")
    parser.add_argument(
        "--config",
        default=None,
        help="Dataset config whose free_text_anonymization settings to reuse "
        "(cutoff, allow/deny lists); defaults to a 0.75 cutoff and no lists",
    )
    args = parser.parse_args()
    inputs = []
    for item in args.inputs:
        path = Path(item)
        inputs.extend(sorted(path.glob("*.json")) if path.is_dir() else [path])
    if not inputs:
        print("[json_remarks] no JSON input found", file=sys.stderr)
        return 1
    run(inputs, Path(args.out), settings_from(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
