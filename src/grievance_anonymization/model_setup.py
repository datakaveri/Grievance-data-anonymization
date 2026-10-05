#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODEL_SETUP.PY — Dependency Installer for PII / NER Pipeline               ║
# ║                                                                              ║
# ║  Run once before running the pipeline:                                       ║
# ║      python -m grievance_anonymization.model_setup                           ║
# ║                                                                              ║
# ║  Models downloaded to HuggingFace cache (~/.cache/huggingface/hub/):         ║
# ║    • cfilt/HiNER-original-muril-base-cased   (~900 MB)                         ║
# ║    • ai4bharat/IndicNER                       (~900 MB)                         ║
# ║    • Babelscape/wikineural-multilingual-ner   (~1.1 GB)                         ║
# ║  Total disk needed: ~4.0 GB + pipeline dependencies                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import gc
import os
import subprocess
import sys

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "").strip()
os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
else:
    os.environ.pop("HF_TOKEN", None)
    os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)


def run(*args, check=True, **kwargs):
    """Thin wrapper around subprocess.run that always prints the command."""
    cmd = list(args)
    print(f"\n  $ {' '.join(str(a) for a in cmd)}")
    result = subprocess.run(cmd, **kwargs)
    if check and result.returncode != 0:
        print(f"  ✗ Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    return result


def pip_install(*packages, extra_args=None):
    cmd = [sys.executable, "-m", "pip", "install", "-q"]
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(packages)
    run(*cmd)


NER_MODEL_IDS = [
    (
        "cfilt/HiNER-original-muril-base-cased",
        "HiNER — IIT Bombay / MuRIL (Indian multilingual NER)",
        "~900 MB",
    ),
    (
        "ai4bharat/IndicNER",
        "IndicNER — AI4Bharat (South Asian NER)",
        "~900 MB",
    ),
    (
        "Babelscape/wikineural-multilingual-ner",
        "XLM-RoBERTa — WikiNEural Multilingual NER",
        "~1.1 GB",
    ),
]


def prefetch_models(max_workers: int = 3) -> int:
    """Download every runtime NER model into the HuggingFace cache, in parallel."""
    from concurrent.futures import ThreadPoolExecutor

    from transformers import AutoModelForTokenClassification, AutoTokenizer

    try:
        from grievance_anonymization.main import _HYBRID_NER_SPECS
    except ImportError:
        try:
            from .main import _HYBRID_NER_SPECS
        except ImportError:
            from .main import _HYBRID_NER_SPECS

    token = os.getenv("HF_TOKEN", "").strip() or None

    def fetch(spec):
        _label, model_id, use_fast, _min_score = spec
        try:
            AutoTokenizer.from_pretrained(model_id, use_fast=use_fast, token=token)
            AutoModelForTokenClassification.from_pretrained(model_id, token=token)
            return model_id, None
        except Exception as exc:
            return model_id, exc

    print(
        f"\n  Fetching {len(_HYBRID_NER_SPECS)} models with {max_workers} workers …",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(fetch, _HYBRID_NER_SPECS))

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


def main():
    if "--prefetch" in sys.argv:
        failed = prefetch_models()
        raise SystemExit(1 if failed and "--strict" in sys.argv else 0)

    print("\n" + "═" * 72)
    print("  PII / NER Pipeline — Dependency Setup")
    print("═" * 72)

    print("\n[1] Upgrading pip / setuptools / wheel …")
    pip_install("pip", "setuptools", "wheel", extra_args=["--upgrade"])

    print("\n[2] Installing PyTorch …")
    try:
        import torch as _t

        if _t.cuda.is_available():
            print(f"  ✓ PyTorch already present with CUDA ({_t.version.cuda}).")
        else:
            print("  ✓ PyTorch already present (CPU only).")
    except ImportError:
        has_cuda = (
            run(
                "nvidia-smi",
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        )
        if has_cuda:
            pip_install(
                "torch",
                "torchvision",
                extra_args=["--index-url", "https://download.pytorch.org/whl/cu121"],
            )
        else:
            pip_install(
                "torch",
                "torchvision",
                extra_args=["--index-url", "https://download.pytorch.org/whl/cpu"],
            )

    print("\n[3] Installing Transformers + HuggingFace stack …")
    pip_install(
        "transformers>=4.40.0",
        "huggingface_hub>=0.22.0",
        "accelerate>=0.26",
        "sentencepiece",
        "protobuf<6.0.0,>=3.20.2",
    )

    print("\n[4] Installing python-docx and BeautifulSoup4 …")
    pip_install("python-docx", "beautifulsoup4", "lxml")

    print("\n[5] Installing utilities: Pandas, OpenPyXL, tqdm …")
    pip_install(
        "pandas>=2.0.0,<3.0.0",
        "openpyxl>=3.1.0",
        "tqdm",
        "colorama",
    )

    print("\n[6] Pre-downloading NER models to HuggingFace cache …")
    print("  (Models saved to: ~/.cache/huggingface/hub/)")
    print("  NOTE: This requires ~4.0 GB disk and internet access.\n")

    try:
        from transformers import AutoModelForTokenClassification, AutoTokenizer
    except ImportError:
        print("  ✗ transformers not importable after install — check PyTorch.")
        sys.exit(1)

    for model_id, description, size in NER_MODEL_IDS:
        print(f"\n  Downloading: {description}")
        print(f"  Model ID:    {model_id}")
        print(f"  Size:        {size}")
        try:
            hf_kwargs = {}
            if HF_TOKEN:
                hf_kwargs["token"] = HF_TOKEN

            tok = AutoTokenizer.from_pretrained(
                model_id,
                use_fast=(
                    False
                    if "muril" in model_id.lower() or "indic" in model_id.lower()
                    else True
                ),
                **hf_kwargs,
            )
            mdl = AutoModelForTokenClassification.from_pretrained(model_id, **hf_kwargs)
            del tok, mdl
            gc.collect()
            print(f"  ✓ Downloaded and cached: {model_id}")
        except Exception as e:
            print(f"  ✗ Download failed for {model_id}: {e}")

    print("\n" + "═" * 72)
    print("  ✅ Setup complete.")
    print("  Cache location: ~/.cache/huggingface/hub/")
    print(
        "  Next step:      python -m grievance_anonymization.main --input <folder_or_file>"
    )
    print("═" * 72 + "\n")


if __name__ == "__main__":
    main()
