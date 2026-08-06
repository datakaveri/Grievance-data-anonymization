#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SETUP.PY — Dependency Installer for PII / NER Pipeline                     ║
# ║                                                                              ║
# ║  Run ONCE before running pipeline.py:                                         ║
# ║      python setup.py                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import subprocess
import sys
import os

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")
os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

def run(*args, check=True, **kwargs):
    """Thin wrapper around subprocess.run that always prints the command."""
    cmd = list(args)
    print(f"\n  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, **kwargs)
    if check and result.returncode != 0:
        print(f"  ✗ Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    return result

def pip_install(*packages, extra_args=None):
    cmd = [sys.executable, "-m", "pip", "install", "-q", "--upgrade"]
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(packages)
    run(*cmd)

def main():
    print("\n" + "═" * 72)
    print("  PII / NER Pipeline — Dependency Setup")
    print("═" * 72)

    # 1. Base Toolchain
    print("\n[1] Upgrading pip / setuptools / wheel …")
    pip_install("pip", "setuptools", "wheel")

    # 2. PyTorch
    print("\n[2] Installing PyTorch …")
    try:
        import torch as _t
        if _t.cuda.is_available():
            print(f"  ✓ PyTorch already present with CUDA ({_t.version.cuda}).")
        else:
            print("  ✓ PyTorch already present (CPU only).")
    except ImportError:
        has_cuda = run("nvidia-smi", check=False,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL).returncode == 0
        if has_cuda:
            pip_install("torch", "torchvision",
                        extra_args=["--index-url", "https://download.pytorch.org/whl/cu121"])
        else:
            pip_install("torch", "torchvision",
                        extra_args=["--index-url", "https://download.pytorch.org/whl/cpu"])

    # 3. Transformers Stack
    print("\n[3] Installing Transformers + HuggingFace stack …")
    pip_install(
        "transformers>=4.49.0",
        "huggingface_hub>=0.22.0",
        "accelerate>=0.26",
        "sentencepiece",
        "protobuf",
    )

    # 4. spaCy
    print("\n[4] Installing spaCy and downloading en_core_web_lg …")
    pip_install("spacy>=3.7.0")
    run(sys.executable, "-m", "spacy", "download", "en_core_web_lg", "--quiet")

    # 5. Presidio
    print("\n[5] Installing Presidio Analyzer + Anonymizer …")
    pip_install("presidio-analyzer", "presidio-anonymizer")

    # 6. GLiNER
    print("\n[6] Installing GLiNER …")
    pip_install("gliner>=0.2.0")

    # 7. Utilities
    print("\n[7] Installing SymSpell, RapidFuzz, Pandas, OpenPyXL …")
    pip_install("symspellpy", "rapidfuzz>=3.0", "pandas>=2.0", "openpyxl>=3.1.0", "tqdm", "colorama")

    print("\n" + "═" * 72)
    print("  ✅ Setup complete — run python pipeline.py next.")
    print("═" * 72 + "\n")

if __name__ == "__main__":
    main()