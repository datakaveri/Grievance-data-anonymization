# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 3.5 — MULTI-FORMAT SUPPORT & STRICT CHANDRA 2 SETUP (FIXED v3)   ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# FIXES vs v2:
#   [G1]  Inference script now accepts BOTH --manifest (batch) AND
#         --image/--out (single) modes — Cell 4 and Cell 5 both work.
#   [G2]  OCR prompt tightened: explicitly asks for plain text, no markdown,
#         no HTML, no <div> tags — eliminates data-bbox pollution in output.
#   [G3]  build_inputs: added image path support via qwen_vl_utils messages
#         that use {"type":"image","image":"file://..."} to avoid PIL resize
#         memory issues on large scans.
#   [G4]  Tesseract pre-flight: script checks tesseract binary exists before
#         calling pytesseract so the fallback doesn't error silently.
#   [G5]  Added explicit error-file sentinel when every strategy fails so
#         Cell 5 can detect and skip rather than hang.
#   [G6]  HF_TOKEN length validation (must be >10 chars) before login.
#   [G7]  max_new_tokens raised 3000→4096 for very long reports.
#   [G8]  EasyOCR: added 'en' language + detail=0 + paragraph=False for
#         cleaner multi-column medical document output.
#   [G9]  Subprocess venv creation: --copies flag added for cross-platform
#         symlink issues on some Kaggle kernels.
#   [G10] py_compile check prints line number on failure for faster debug.

import subprocess
import sys
import os

CHANDRA_VENV    = "/kaggle/working/chandra_venv"
CHANDRA_VENV_PY = f"{CHANDRA_VENV}/bin/python"
CHANDRA_SCRIPT  = "/kaggle/working/chandra_infer_script.py"

# ── [0] Environment Setup ─────────────────────────────────────────────────────
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")
os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

HF_TOKEN = os.getenv("HF_TOKEN", "")
if HF_TOKEN and len(HF_TOKEN) > 10:
    print(f"[0] ✓ HF_TOKEN loaded (length={len(HF_TOKEN)}).")
else:
    print("[0] ✗ HF_TOKEN not set or too short — gated model downloads will fail.")

# ── [A] Create Isolated Venv ──────────────────────────────────────────────────
if not os.path.exists(CHANDRA_VENV_PY):
    print("\n[A] Creating isolated venv for Chandra 2...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "virtualenv"], check=True)
        subprocess.run(
            [sys.executable, "-m", "virtualenv", "--copies", CHANDRA_VENV],
            check=True,
        )
    except Exception:
        subprocess.run(
            ["python3", "-m", "venv", "--without-pip", "--copies", CHANDRA_VENV],
            check=True,
        )
        subprocess.run(
            [CHANDRA_VENV_PY, "-m", "ensurepip", "--default-pip"],
            check=False,
        )

    subprocess.run(
        [CHANDRA_VENV_PY, "-m", "pip", "install", "-q",
         "--upgrade", "pip", "setuptools", "wheel"],
        check=True,
    )
    subprocess.run([CHANDRA_VENV_PY, "-m", "pip", "uninstall", "-y", "docx"], check=False)

    import torch as _torch_probe
    _has_cuda = _torch_probe.cuda.is_available()
    print(f"  [A] CUDA available: {_has_cuda}")

    if _has_cuda:
        subprocess.run(
            [CHANDRA_VENV_PY, "-m", "pip", "install", "-q",
             "torch", "torchvision",
             "--index-url", "https://download.pytorch.org/whl/cu121"],
            check=True,
        )
    else:
        subprocess.run(
            [CHANDRA_VENV_PY, "-m", "pip", "install", "-q",
             "torch", "torchvision",
             "--index-url", "https://download.pytorch.org/whl/cpu"],
            check=True,
        )

    subprocess.run(
        [CHANDRA_VENV_PY, "-m", "pip", "install", "-q",
         "wrapt",
         "transformers>=4.49.0",
         "chandra-ocr[hf]",
         "pillow>=10.0",
         "accelerate>=0.26",
         "qwen-vl-utils",
         "huggingface_hub>=0.22.0",
         "openpyxl>=3.1.0",
         "beautifulsoup4>=4.12.0",
         "python-docx>=1.1.0",
         "pypdf>=4.0.0",
         "pytesseract",
         "easyocr"],
        check=True,
    )
    print("  ✓ Chandra 2 venv and multi-format libraries ready.")
else:
    print("[A] ✓ Chandra 2 venv already exists.")

# [F1] Write token env file
_ENV_FILE = "/kaggle/working/chandra_env.env"
with open(_ENV_FILE, "w", encoding="utf-8") as _ef:
    _ef.write(f"HF_TOKEN={HF_TOKEN}\n")
    _ef.write(f"HUGGING_FACE_HUB_TOKEN={HF_TOKEN}\n")
print(f"  ✓ Token env file written: {_ENV_FILE}")

# ── [B] Write Inference Script ────────────────────────────────────────────────
print("\n[B] Writing chandra_infer_script.py ...")

_SCRIPT_CONTENT = r'''
import os
import sys
import argparse
import gc
import json
import torch
from PIL import Image

# ── [G1] Env bootstrap (token) ─────────────────────────────────────────────
_ENV_FILE = "/kaggle/working/chandra_env.env"
if os.path.isfile(_ENV_FILE):
    with open(_ENV_FILE, "r", encoding="utf-8") as _ef:
        for _line in _ef:
            _line = _line.strip()
            if "=" in _line and not _line.startswith("#"):
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

_HF_TOKEN = os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
if _HF_TOKEN and len(_HF_TOKEN) > 10:
    try:
        from huggingface_hub import login as _hf_login
        _hf_login(token=_HF_TOKEN, add_to_git_credential=False)
        print(f"[HF] Logged in (token length={len(_HF_TOKEN)})", flush=True)
    except Exception as _e:
        print(f"[HF] Login warning: {_e}", flush=True)

MODEL_ID = "datalab-to/chandra-ocr-2"

# ── [G2] PLAIN-TEXT OCR PROMPT — no markdown, no HTML, no tags ─────────────
OCR_PROMPT = (
    "You are an expert OCR system for medical and laboratory documents. "
    "Extract ALL visible text from this image EXACTLY as it appears. "
    "Rules you MUST follow:\n"
    "1. Output ONLY raw plain text — NO markdown, NO HTML, NO <div> tags, "
    "   NO data-bbox attributes, NO JSON, NO XML.\n"
    "2. Preserve the original line structure: one line of document text = "
    "   one line of output.\n"
    "3. Keep all patient data, test names, values, units, doctor names, "
    "   reference ranges, hospital names, dates, and identifiers.\n"
    "4. Do NOT add any commentary, explanation, or preamble.\n"
    "5. Do NOT use bullet points, headers (#), bold (**), or tables (|).\n"
    "Output only the extracted plain text, nothing else."
)


# ── Model loader (three strategies) ────────────────────────────────────────
def load_model_safely():
    for strategy, loader_fn in [
        ("Qwen2_5_VL",             _load_qwen),
        ("AutoModelForImgTextToText", _load_auto_img_text),
        ("AutoModelForVision2Seq", _load_vision2seq),
    ]:
        model, processor, mtype = loader_fn()
        if model is not None:
            print(f"[Model] ✓ Loaded via {strategy}", flush=True)
            return model, processor, mtype
    print("[FATAL] All model loaders failed.", file=sys.stderr, flush=True)
    sys.exit(1)


def _dtype():
    return torch.float16 if torch.cuda.is_available() else torch.float32

def _device_map():
    return "auto" if torch.cuda.is_available() else "cpu"

def _common_kwargs():
    kw = dict(
        torch_dtype=_dtype(),
        device_map=_device_map(),
        trust_remote_code=True,
    )
    if _HF_TOKEN:
        kw["token"] = _HF_TOKEN
    return kw

def _load_processor(model_id):
    from transformers import AutoProcessor
    kw = dict(trust_remote_code=True)
    if _HF_TOKEN:
        kw["token"] = _HF_TOKEN
    return AutoProcessor.from_pretrained(model_id, **kw)

def _load_qwen():
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        print("[Model] Trying Qwen2_5_VLForConditionalGeneration...", flush=True)
        m = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, **_common_kwargs())
        p = _load_processor(MODEL_ID)
        return m, p, "qwen"
    except Exception as e:
        print(f"[WARN] Qwen2_5_VL failed: {e}", file=sys.stderr, flush=True)
        return None, None, None

def _load_auto_img_text():
    try:
        from transformers import AutoModelForImageTextToText
        print("[Model] Trying AutoModelForImageTextToText...", flush=True)
        m = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **_common_kwargs())
        p = _load_processor(MODEL_ID)
        return m, p, "auto"
    except Exception as e:
        print(f"[WARN] AutoModelForImageTextToText failed: {e}", file=sys.stderr, flush=True)
        return None, None, None

def _load_vision2seq():
    try:
        from transformers import AutoModelForVision2Seq
        print("[Model] Trying AutoModelForVision2Seq...", flush=True)
        m = AutoModelForVision2Seq.from_pretrained(MODEL_ID, **_common_kwargs())
        p = _load_processor(MODEL_ID)
        return m, p, "auto"
    except Exception as e:
        print(f"[WARN] AutoModelForVision2Seq failed: {e}", file=sys.stderr, flush=True)
        return None, None, None


# ── Image resize ──────────────────────────────────────────────────────────────
def resize_if_needed(img: Image.Image, max_dim: int = 2000) -> Image.Image:
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return img


# ── Input builder ─────────────────────────────────────────────────────────────
def build_inputs(processor, pil_img, img_path: str):
    """
    [G3] Prefer file:// path for qwen_vl_utils (avoids double-resize in memory).
    Falls back to PIL-based strategies.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{img_path}"},
                {"type": "text",  "text": OCR_PROMPT},
            ],
        }
    ]

    # Strategy 1: qwen_vl_utils with file path
    try:
        from qwen_vl_utils import process_vision_info
        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs
    except Exception:
        pass

    # Strategy 2: PIL image list via chat template
    messages_pil = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text",  "text": OCR_PROMPT},
            ],
        }
    ]
    try:
        text_prompt = processor.apply_chat_template(
            messages_pil, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text_prompt],
            images=[pil_img],
            padding=True,
            return_tensors="pt",
        )
        return inputs
    except Exception:
        pass

    # Strategy 3: plain text + PIL (no chat template)
    try:
        inputs = processor(
            text=[OCR_PROMPT],
            images=[pil_img],
            return_tensors="pt",
        )
        return inputs
    except Exception as e:
        raise RuntimeError(f"All input-building strategies failed: {e}")


# ── Single image OCR (with OOM retry) ────────────────────────────────────────
def ocr_single(model, processor, img_path: str, max_dim: int = 2000) -> str:
    try:
        pil_img = Image.open(img_path).convert("RGB")
        pil_img = resize_if_needed(pil_img, max_dim=max_dim)
    except Exception as e:
        print(f"    [WARN] Cannot open {img_path}: {e}", file=sys.stderr)
        return ""

    try:
        inputs = build_inputs(processor, pil_img, img_path)
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=4096)  # [G7]
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        raw = output_text[0].strip() if output_text else ""

        # [G2] Strip any residual HTML/markdown the model sneaks in
        raw = _strip_html_and_markdown(raw)
        return raw

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        if max_dim > 800:
            print(f"    [WARN] OOM on {img_path}, retrying at {max_dim // 2}px...", file=sys.stderr)
            return ocr_single(model, processor, img_path, max_dim=max_dim // 2)
        return ""
    except Exception as e:
        print(f"    [WARN] OCR error on {img_path}: {e}", file=sys.stderr)
        return ""


# ── [G2] Post-process: strip HTML tags, data-bbox, and markdown syntax ────────
import re as _re

_HTML_TAG_RE    = _re.compile(r"<[^>]+>", _re.DOTALL)
_DATA_ATTR_RE   = _re.compile(r'\s*data-\w+="[^"]*"', _re.IGNORECASE)
_MD_HEADER_RE   = _re.compile(r"^#{1,6}\s*", _re.MULTILINE)
_MD_BOLD_RE     = _re.compile(r"\*\*(.+?)\*\*")
_MD_TABLE_SEP   = _re.compile(r"^\s*\|?[-:\s|]+\|?\s*$", _re.MULTILINE)
_MD_TABLE_ROW   = _re.compile(r"^\|(.+)\|$", _re.MULTILINE)

def _strip_html_and_markdown(text: str) -> str:
    """Remove HTML tags, data-bbox attributes, and markdown syntax from OCR output."""
    if not text:
        return ""
    # Remove data-bbox and similar attributes
    text = _DATA_ATTR_RE.sub("", text)
    # Remove HTML tags (captures <div ...>, </div>, <span ...> etc.)
    text = _HTML_TAG_RE.sub("", text)
    # Convert markdown table rows to plain text
    def _table_row_to_text(m):
        cells = [c.strip() for c in m.group(1).split("|") if c.strip()]
        return "  ".join(cells)
    text = _MD_TABLE_SEP.sub("", text)
    text = _MD_TABLE_ROW.sub(_table_row_to_text, text)
    # Strip markdown headers
    text = _MD_HEADER_RE.sub("", text)
    # Strip bold markers but keep content
    text = _MD_BOLD_RE.sub(r"\1", text)
    # Collapse excessive blank lines
    text = _re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Fallback: Tesseract ────────────────────────────────────────────────────────
def tesseract_ocr(img_path: str) -> str:
    # [G4] Check binary exists first
    import shutil
    if shutil.which("tesseract") is None:
        print("    [WARN] tesseract binary not found on PATH", file=sys.stderr)
        return ""
    try:
        import pytesseract
        img = Image.open(img_path).convert("RGB")
        text = pytesseract.image_to_string(img, lang="eng", config="--psm 6 --oem 3")
        return text.strip()
    except Exception as e:
        print(f"    [WARN] Tesseract failed: {e}", file=sys.stderr)
        return ""


# ── Fallback: EasyOCR ─────────────────────────────────────────────────────────
def easyocr_fallback(img_path: str) -> str:
    try:
        import easyocr
        # [G8] paragraph=False gives better per-line output for structured docs
        reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
        results = reader.readtext(img_path, detail=0, paragraph=False)
        return "\n".join(results).strip()
    except Exception as e:
        print(f"    [WARN] EasyOCR failed: {e}", file=sys.stderr)
        return ""


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    # [G1] Support BOTH batch (--manifest) and single (--image/--out) modes
    ap.add_argument("--manifest", default=None,
                    help="JSON manifest file for batch processing (Cell 5 mode)")
    ap.add_argument("--image",    default=None,
                    help="Single image path (Cell 4 mode)")
    ap.add_argument("--out",      default=None,
                    help="Output text path for single-image mode")
    args = ap.parse_args()

    # Build task list
    if args.manifest:
        if not os.path.isfile(args.manifest):
            print(f"[FATAL] Manifest not found: {args.manifest}", file=sys.stderr)
            sys.exit(1)
        with open(args.manifest, "r", encoding="utf-8") as f:
            tasks = json.load(f)
    elif args.image and args.out:
        tasks = [{"image": args.image, "out": args.out}]
    else:
        print("[FATAL] Provide either --manifest or both --image and --out.",
              file=sys.stderr)
        sys.exit(1)

    if not tasks:
        print("[FATAL] Empty task list.", file=sys.stderr)
        sys.exit(1)

    model, processor, model_type = load_model_safely()

    for task in tasks:
        img_path = task["image"]
        out_path = task["out"]
        fname    = os.path.basename(img_path)

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

        if not os.path.isfile(img_path):
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(f"[IMAGE NOT FOUND: {fname}]")
            continue

        print(f"  → OCR: {fname}", flush=True)
        text = ocr_single(model, processor, img_path, max_dim=2000)

        # Fallback chain
        if not text or not text.strip():
            print(f"    [INFO] Chandra empty, trying Tesseract...", flush=True)
            text = tesseract_ocr(img_path)

        if not text or not text.strip():
            print(f"    [INFO] Tesseract empty, trying EasyOCR...", flush=True)
            text = easyocr_fallback(img_path)

        if not text or not text.strip():
            text = f"[OCR EMPTY: {fname}]"

        # [G9] UTF-8 with BOM for Excel compatibility
        with open(out_path, "w", encoding="utf-8-sig") as f:
            f.write(text)
        print(f"    ✓ Written {len(text)} chars to {out_path}", flush=True)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del model
    del processor
    gc.collect()


if __name__ == "__main__":
    main()
'''

with open(CHANDRA_SCRIPT, "w", encoding="utf-8") as f:
    f.write(_SCRIPT_CONTENT)
print(f"  ✓ Written: {CHANDRA_SCRIPT}")

# [G10] Syntax check with line-number on failure
import subprocess as _sp
chk = _sp.run(
    [sys.executable, "-m", "py_compile", CHANDRA_SCRIPT],
    capture_output=True, text=True,
)
if chk.returncode == 0:
    print("  ✓ Syntax OK")
else:
    print(f"  ✗ Syntax error:\n{chk.stderr}")
    raise SyntaxError("chandra_infer_script.py failed syntax check.")

print("\n✅ CELL 3.5 COMPLETE — Run Cell 4 or Cell 5 now.")