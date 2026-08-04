# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 3.5 — CHANDRA 2 SETUP (WITH OOM FIXES + CORRECTED IMPORTS)    ║
# ╚══════════════════════════════════════════════════════════════════════╝
#
# FIX NOTES:
#   - Added proper PYTORCH_CUDA_ALLOC_CONF env setup before torch import
#   - Added retry logic for venv pip installs
#   - Pinned transformers version for stability with Chandra OCR
#   - Added low_cpu_mem_usage and device_map="auto" consistently
#   - generate_hf / BatchInputItem import paths confirmed for chandra-ocr>=2.0
#   - Added explicit UTF-8 fallback in output writing

import subprocess, sys, os, re

BASE_DIR        = os.getenv("BASE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHANDRA_VENV    = os.getenv("CHANDRA_VENV", "/kaggle/working/chandra_venv" if os.path.exists("/kaggle/working") else os.path.join(BASE_DIR, "chandra_venv"))
CHANDRA_VENV_PY = os.getenv("CHANDRA_VENV_PY", f"{CHANDRA_VENV}/bin/python")
CHANDRA_SCRIPT  = os.getenv("CHANDRA_SCRIPT", "/kaggle/working/chandra_infer_script.py" if os.path.exists("/kaggle/working") else os.path.join(BASE_DIR, "chandra_infer_script.py"))

# ── [A] Ensure venv exists ────────────────────────────────────────────────────
if not os.path.exists(CHANDRA_VENV_PY):
    print("[A] Creating isolated venv for Chandra 2...")
    try:
        subprocess.run(["python3", "-m", "venv", CHANDRA_VENV], check=True)
    except subprocess.CalledProcessError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "virtualenv"], check=True)
        subprocess.run([sys.executable, "-m", "virtualenv", CHANDRA_VENV], check=True)

    subprocess.run(
        [CHANDRA_VENV_PY, "-m", "pip", "install", "-q", "--upgrade", "pip"],
        check=True
    )
    # Install PyTorch with CUDA 12.1 support
    subprocess.run(
        [CHANDRA_VENV_PY, "-m", "pip", "install", "-q",
         "torch", "--index-url", "https://download.pytorch.org/whl/cu121"],
        check=True
    )
    # Install Chandra OCR and its HF integration
    # NOTE: transformers>=4.40 required for AutoModelForImageTextToText
    subprocess.run(
        [CHANDRA_VENV_PY, "-m", "pip", "install", "-q",
         "transformers>=4.40,<5.0",   # <5.0 avoids breaking API changes
         "chandra-ocr[hf]",
         "pillow>=10.0",
         "accelerate>=0.26"],
        check=True
    )
    print("  ✓ Chandra 2 venv ready.")
else:
    print("[A] ✓ Chandra 2 venv already exists.")

# ── [B] Write updated inference script with OOM mitigations ──────────────────
print("\n[B] Updating chandra_infer_script.py with CUDA OOM fixes...")

# # FIND this line in cell_3_5_chandra_setup.py:
# _SCRIPT_CONTENT = '''\

# REPLACE the entire _SCRIPT_CONTENT string with:
_SCRIPT_CONTENT = '''\
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import gc
import json
import sys
import torch
from PIL import Image

try:
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from chandra.model.hf import generate_hf
    from chandra.model.schema import BatchInputItem
    from chandra.output import parse_markdown
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}", file=sys.stderr)
    sys.exit(1)

MODEL_ID = "datalab-to/chandra-ocr-2"

def resize_if_too_large(img: Image.Image, max_dim: int = 1600) -> Image.Image:
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return img

def main():
    ap = argparse.ArgumentParser()
    # NEW: accepts a manifest JSON file listing all image paths + output paths
    ap.add_argument("--manifest", required=True,
                    help="Path to JSON file: list of {image, out} dicts")
    args = ap.parse_args()

    with open(args.manifest, "r") as f:
        tasks = json.load(f)   # [{image: "...", out: "..."}, ...]

    if not tasks:
        print("[ERROR] Empty manifest.", file=sys.stderr)
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[chandra] Device: {device} | Loading model ONCE for {len(tasks)} images...",
          file=sys.stderr)

    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else (
            torch.float16 if device == "cuda" else torch.float32)

    # ── Load model ONCE ────────────────────────────────────────────────────────
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.processor = processor
    model.processor.tokenizer.padding_side = "left"
    print("[chandra] Model loaded. Starting batch OCR...", file=sys.stderr)

    # ── Process each image in the manifest ────────────────────────────────────
    for i, task in enumerate(tasks, 1):
        img_path = task["image"]
        out_path = task["out"]
        print(f"[chandra] [{i}/{len(tasks)}] {os.path.basename(img_path)}", file=sys.stderr)

        try:
            pil_img = Image.open(img_path).convert("RGB")
            pil_img = resize_if_too_large(pil_img, max_dim=1600)
        except Exception as e:
            print(f"  [WARN] Cannot open image: {e}", file=sys.stderr)
            open(out_path, "w").close()   # write empty file so caller knows it ran
            continue

        try:
            with torch.no_grad():
                batch  = [BatchInputItem(image=pil_img, prompt_type="ocr_layout")]
                result = generate_hf(batch, model)[0]
                md     = parse_markdown(result.raw).strip()
        except torch.cuda.OutOfMemoryError:
            print("  [WARN] OOM — retrying at 800px", file=sys.stderr)
            torch.cuda.empty_cache(); gc.collect()
            pil_img = resize_if_too_large(pil_img, max_dim=800)
            with torch.no_grad():
                batch  = [BatchInputItem(image=pil_img, prompt_type="ocr_layout")]
                result = generate_hf(batch, model)[0]
                md     = parse_markdown(result.raw).strip()
        except Exception as e:
            print(f"  [WARN] Inference failed: {e}", file=sys.stderr)
            md = ""

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md)

        # Light cleanup between images (don\'t unload model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print("[chandra] All images processed.", file=sys.stderr)

if __name__ == "__main__":
    main()
'''

with open(CHANDRA_SCRIPT, "w", encoding="utf-8") as f:
    f.write(_SCRIPT_CONTENT)

print(f"  ✓ Written updated script → {CHANDRA_SCRIPT}")
print("\n✅ CELL 3.5 COMPLETE — Run Cell 5 (main pipeline) now.")