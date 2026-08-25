#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SETUP.PY — Dependency Installer for PII / NER Pipeline                     ║
# ║                                                                              ║
# ║  Run ONCE on Kaggle before running pipeline.py:                              ║
# ║      python setup.py                                                         ║
# ║                                                                              ║
# ║  Models downloaded to HuggingFace cache (~/.cache/huggingface/hub/):         ║
# ║    • cfilt/HiNER-original-muril-base-cased   (~900 MB)                         ║
# ║    • ai4bharat/IndicNER                       (~900 MB)                         ║
# ║    • dslim/bert-base-NER                      (~430 MB)                         ║
# ║    • Babelscape/wikineural-multilingual-ner   (~1.1 GB)                         ║
# ║    • spaCy en_core_web_lg                      (~788 MB)                         ║
# ║  Total disk needed: ~4.5 GB + pipeline dependencies                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import subprocess
import sys
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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
    # Removed global --upgrade to prevent breaking pre-installed environment dependencies
    cmd = [sys.executable, "-m", "pip", "install", "-q"]
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(packages)
    run(*cmd)


# ─────────────────────────────────────────────────────────────────────────────
# NER MODELS TO DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────
# Each entry: (model_id, description, size_note)
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
        "dslim/bert-base-NER",
        "BERT-Base-NER — English CoNLL-2003",
        "~430 MB",
    ),
    (
        "Babelscape/wikineural-multilingual-ner",
        "XLM-RoBERTa — WikiNEural Multilingual NER",
        "~1.1 GB",
    ),
]


def main():
    print("\n" + "═" * 72)
    print("  PII / NER Pipeline — Dependency Setup")
    print("═" * 72)

    # ── 1. Base Toolchain ─────────────────────────────────────────────────
    print("\n[1] Upgrading pip / setuptools / wheel …")
    pip_install("pip", "setuptools", "wheel", extra_args=["--upgrade"])

    # ── 2. PyTorch ────────────────────────────────────────────────────────
    print("\n[2] Installing PyTorch …")
    try:
        import torch as _t
        if _t.cuda.is_available():
            print(f"  ✓ PyTorch already present with CUDA ({_t.version.cuda}).")
        else:
            print("  ✓ PyTorch already present (CPU only).")
    except ImportError:
        has_cuda = run(
            "nvidia-smi", check=False,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode == 0
        if has_cuda:
            pip_install(
                "torch", "torchvision",
                extra_args=["--index-url", "https://download.pytorch.org/whl/cu121"],
            )
        else:
            pip_install(
                "torch", "torchvision",
                extra_args=["--index-url", "https://download.pytorch.org/whl/cpu"],
            )

    # ── 3. Transformers Stack ─────────────────────────────────────────────
    print("\n[3] Installing Transformers + HuggingFace stack …")
    pip_install(
        "transformers>=4.40.0",
        "huggingface_hub>=0.22.0",
        "accelerate>=0.26",
        "sentencepiece",        # required by MuRIL tokenizer (HiNER)
        "protobuf<6.0.0,>=3.20.2", # pinned to avoid breaking google-cloud / grpc dependencies
    )

    # ── 4. spaCy (for Presidio) ───────────────────────────────────────────
    print("\n[4] Installing spaCy and downloading en_core_web_lg …")
    pip_install("spacy>=3.7.0")
    run(sys.executable, "-m", "spacy", "download", "en_core_web_lg", "--quiet")

    # ── 5. Presidio ───────────────────────────────────────────────────────
    print("\n[5] Installing Presidio Analyzer + Anonymizer …")
    pip_install("presidio-analyzer", "presidio-anonymizer")

    # ── 6. Document reading libs ──────────────────────────────────────────
    print("\n[6] Installing python-docx and BeautifulSoup4 …")
    pip_install("python-docx", "beautifulsoup4", "lxml")

    # ── 7. Utilities ──────────────────────────────────────────────────────
    print("\n[7] Installing utilities: Pandas, OpenPyXL, tqdm …")
    pip_install(
        "pandas>=2.0.0,<3.0.0", # pinned to <3.0.0 to prevent google-colab & gradio conflicts
        "openpyxl>=3.1.0",
        "tqdm",
        "colorama",
    )

    # ── 8. Download all 4 NER models ─────────────────────────────────────
    # WHY we download here and not in pipeline.py:
    #   • Kaggle kernels have no internet access during inference by default.
    #   • Downloading in setup.py populates ~/.cache/huggingface/hub/ once.
    #   • pipeline.py then loads from cache (offline, fast).
    #   • Each model is ~430 MB – 1.1 GB; downloading once saves runtime.
    print("\n[8] Pre-downloading NER models to HuggingFace cache …")
    print("  (Models saved to: ~/.cache/huggingface/hub/)")
    print("  NOTE: This requires ~4.5 GB disk and internet access.\n")

    try:
        from transformers import AutoTokenizer, AutoModelForTokenClassification
    except ImportError:
        print("  ✗ transformers not importable after install — check PyTorch.")
        sys.exit(1)

    for model_id, description, size in NER_MODEL_IDS:
        print(f"\n  Downloading: {description}")
        print(f"  Model ID:    {model_id}")
        print(f"  Size:        {size}")
        try:
            # Handle token parameter cleanly to prevent invalid authentication requests
            hf_kwargs = {}
            if HF_TOKEN:
                hf_kwargs["token"] = HF_TOKEN

            # Download tokenizer
            tok = AutoTokenizer.from_pretrained(
                model_id,
                # HiNER/IndicNER use sentencepiece — set use_fast=False as fallback
                use_fast=False if "muril" in model_id.lower() or "indic" in model_id.lower() else True,
                **hf_kwargs
            )
            # Download model weights
            mdl = AutoModelForTokenClassification.from_pretrained(
                model_id,
                **hf_kwargs
            )
            del tok, mdl          # free RAM; weights stay in disk cache
            import gc; gc.collect()
            print(f"  ✓ Downloaded and cached: {model_id}")
        except Exception as e:
            print(f"  ✗ Download failed for {model_id}: {e}")
            print("    → Check your internet connection and HF_TOKEN if the model is gated.")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "═" * 72)
    print("  ✅ Setup complete.")
    print("  Cache location: ~/.cache/huggingface/hub/")
    print("  Next step:      python pipeline.py --input <folder_or_file>")
    print("═" * 72 + "\n")


if __name__ == "__main__":
    main()