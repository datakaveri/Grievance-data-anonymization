#!/usr/bin/env python3
"""Download the NER models into the HuggingFace cache (run once, and at Docker build time).

    pip install -r requirements.txt
    python app/model_setup.py [--strict]

The models and each one's tokenizer flavour (fast or slow) come from
anonymizer/ner/models.py NER_MODELS, so exactly what the pipeline loads at run
time is cached. That matters: caching a tokenizer as slow when the pipeline
asks for it fast is a cache miss, and inside an offline TEE a cache miss is a
failure, not a download.

ai4bharat/IndicNER is a gated repo: accept its terms on huggingface.co and set
HF_TOKEN. Without it IndicNER is reported as not cached, and the pipeline later
runs with XLM-RoBERTa alone. --strict makes any missing model a non-zero exit
(fail the Docker build). `--prefetch` is accepted for compatibility and does
nothing extra: prefetching is all this script does.

Previous versions also pip-installed the dependencies (now requirements.txt
only) and downloaded models the pipeline no longer uses (HiNER, wikineural).
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from anonymizer.ner.models import NER_MODELS, NerModelSpec  # noqa: E402


def _fetch(spec: NerModelSpec, token):
    """Download one model's tokenizer and weights; return (model_id, error or None)."""
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    try:
        AutoTokenizer.from_pretrained(spec.model_id, use_fast=spec.use_fast, token=token)
        AutoModelForTokenClassification.from_pretrained(spec.model_id, token=token)
        return spec.model_id, None
    except Exception as exc:
        return spec.model_id, exc


def prefetch_models() -> int:
    """Download every model concurrently; return how many failed.

    Downloads are network-bound, so fetching in parallel costs about as long as
    the slowest model instead of the sum.
    """
    token = os.getenv("HF_TOKEN", "").strip() or None
    print(f"\n  Fetching {len(NER_MODELS)} models …", flush=True)
    with ThreadPoolExecutor(max_workers=len(NER_MODELS)) as pool:
        results = list(pool.map(lambda spec: _fetch(spec, token), NER_MODELS))
    failed = 0
    for model_id, exc in results:
        if exc is None:
            print(f"  ✓ cached {model_id}")
        else:
            failed += 1
            print(f"  ✗ {model_id}: {exc}")
    if failed:
        print(
            f"\n  {failed} model(s) not cached; they will be skipped at runtime. "
            "Set HF_TOKEN if a repo is gated — ai4bharat/IndicNER is."
        )
    return failed


def main() -> int:
    """Prefetch; exit 1 only when --strict and a model failed."""
    failed = prefetch_models()
    return 1 if failed and "--strict" in sys.argv else 0


if __name__ == "__main__":
    raise SystemExit(main())
