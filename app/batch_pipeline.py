#!/usr/bin/env python3
"""Anonymize the free-text columns of a dataset table (the Docker ENTRYPOINT).

    python app/batch_pipeline.py --config <config.json or directory> [--root <dir>]

Reads the dataset named in the config, redacts personal data in the configured
columns, and writes the staged CSV for SKALD plus an audit JSON. The work is
done by anonymizer/jobs/table.py; see anonymizer/__init__.py for the whole flow.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from anonymizer.config import BatchError  # noqa: E402
from anonymizer.jobs.table import run  # noqa: E402


def main() -> int:
    """Command line: --config <file or directory> [--root <dir>]. Exit code 1 on BatchError."""
    parser = argparse.ArgumentParser(description="Config-driven dataset free-text anonymization")
    parser.add_argument("--config", required=True, help="Path to the JSON configuration file, or a directory to scan for one")
    parser.add_argument("--root", default=None, help="Base directory for relative paths")
    args = parser.parse_args()
    try:
        run(args.config, args.root)
    except BatchError as exc:
        print(f"[Batch][ERROR] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
