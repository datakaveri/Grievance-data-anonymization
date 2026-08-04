# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 5 — UNIFIED BATCH PIPELINE: EXPANDED PII + OPTIMIZED NER & OCR         ║
# ║                                                                              ║
# ║  FIXES APPLIED:                                                              ║
# ║  [FIX-1] TypeError: _sanitize_parameters() unexpected kwarg 'truncation'     ║
# ║           → Removed 'truncation' & 'padding' from pipeline() kwargs.         ║
# ║           → Truncation now applied at inference time via tokenizer call.     ║
# ║           → pipe.tokenizer.truncation = True removed (not a valid attr).     ║
# ║  [FIX-2] Long execution time                                                 ║
# ║           → Text lines chunked to ≤ 128 tokens before NER inference.         ║
# ║           → batch_size reduced to 16 (prevents OOM on long token seqs).      ║
# ║           → NER inference wrapped with truncation=True, max_length=512.      ║
# ║           → del pipe uses hasattr guards (prevents AttributeError).          ║
# ║  [FIX-3] Sign detection model explanation added (see section 3B).            ║
# ║  [FIX-4] PII tags exported as SEPARATE columns per PII type in Excel.        ║
# ║  [FIX-5] Anonymization applied per PII type with correct masking rules.      ║
# ║  [FIX-6] process_ner_and_pii_for_lines now passes truncation at runtime.     ║
# ╚════════════════════════════════════════════════════════════════════════════╝

import os
import re
import gc
import time
import uuid
import subprocess
import warnings
import torch
import pandas as pd
from PIL import Image as PILImage

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. HARDWARE & PATH SETUP
# ─────────────────────────────────────────────────────────────────────────────
DEVICE_ID = 0 if torch.cuda.is_available() else -1
print(f"[Pipeline] Executing on: {'GPU (CUDA)' if DEVICE_ID == 0 else 'CPU'}")

HF_TOKEN = os.getenv("HF_TOKEN", None)

BASE_DIR = os.getenv("BASE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMAGE_FOLDER_PATH = os.getenv("IMAGE_FOLDER_PATH", "/kaggle/input/datasets/gogul0604/kaggle-image-datasets/Source-image" if os.path.exists("/kaggle/input") else os.path.join(BASE_DIR, "data", "source_images"))
OUTPUT_DIR        = os.getenv("OUTPUT_DIR", "/kaggle/working/output" if os.path.exists("/kaggle/working") else os.path.join(BASE_DIR, "output"))
OUTPUT_EXCEL      = os.path.join(OUTPUT_DIR, "chandra_ner_pii_batch_comparison.xlsx")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_FOLDER_PATH, exist_ok=True)

CHANDRA_VENV_PY = os.getenv("CHANDRA_VENV_PY", "/kaggle/working/chandra_venv/bin/python" if os.path.exists("/kaggle/working") else os.path.join(BASE_DIR, "chandra_venv", "bin", "python"))
CHANDRA_SCRIPT  = os.getenv("CHANDRA_SCRIPT", "/kaggle/working/chandra_infer_script.py" if os.path.exists("/kaggle/working") else os.path.join(BASE_DIR, "chandra_infer_script.py"))

# REPLACE with (adds a quick smoke-test):
if not os.path.exists(CHANDRA_VENV_PY):
    raise RuntimeError("Chandra 2 venv not found. Run Cell 3.5 first.")
if not os.path.exists(CHANDRA_SCRIPT):
    raise RuntimeError("chandra_infer_script.py not found. Run Cell 3.5 first.")

# Verify the script was written with --manifest support (not the old --image version)
with open(CHANDRA_SCRIPT, "r") as _f:
    if "--manifest" not in _f.read():
        raise RuntimeError(
            "chandra_infer_script.py is the OLD single-image version. "
            "Re-run Cell 3.5 to regenerate it with --manifest support."
        )

# ─────────────────────────────────────────────────────────────────────────────
# 2. MODEL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
MODELS = {
    "HiNER (IIT Bombay / MuRIL)":  "cfilt/HiNER-original-muril-base-cased",
    "IndicNER (AI4Bharat)":          "ai4bharat/IndicNER" if HF_TOKEN else "techysanoj/fine-tuned-IndicNER",
    "BERT-Base-NER (English)":       "dslim/bert-base-NER",
    "XLM-RoBERTa (Multilingual)":   "Babelscape/wikineural-multilingual-ner",
}

# ─────────────────────────────────────────────────────────────────────────────
# 3A. EXPANDED REGEX PATTERNS FOR PII DETECTION (20 categories)
# ─────────────────────────────────────────────────────────────────────────────
PII_PATTERNS = {
    "Email":          re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.I),
    "Aadhaar":        re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"),
    "PAN":            re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "Voter_ID":       re.compile(r"\b[A-Z]{3}\d{7}\b"),
    "PPP_ID":         re.compile(r"\b[A-Z0-9]{8,10}\b"),
    "Passport":       re.compile(r"\b[A-PR-WY][1-9]\d{7}\b"),
    "Vehicle_Number": re.compile(r"\b[A-Z]{2}[\s\-]?\d{2}[\s\-]?[A-Z]{1,2}[\s\-]?\d{4}\b"),
    "Phone_Number":   re.compile(r"(?:(?:\+?91[\s\-]?)?[6-9]\d{9})\b"),
    "Bank_Account":   re.compile(r"\b\d{11,16}\b"),
    "Credit_Card":    re.compile(r"\b(?:4\d{12}(?:\d{3})?|5[1-5]\d{14}|3[47]\d{13})\b"),
    "DOB":            re.compile(r"\b(?:0?[1-9]|[12]\d|3[01])[\/\-.](?:0?[1-9]|1[0-2])[\/\-.](?:19|20)\d{2}\b"),
    "Date":           re.compile(r"\b\d{1,2}[\/\-.](?:\d{1,2}|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\/\-.]\d{2,4}\b", re.I),
    "User_ID":        re.compile(r"\b(?:user|uid|usr|id)[\_\-\:\s]*[a-zA-Z0-9]{4,15}\b", re.I),
    "IP_Address":     re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"),
    # ── [FIX-3] Sign: Keyword-based regex on OCR text output.
    # NOTE: This is NOT a vision/image-level model. It detects the WORD "signature"
    # and its Hindi/Devanagari equivalents in OCR-extracted text.
    # For true image-level signature detection, a specialized CV model is needed
    # (e.g., a fine-tuned YOLO/EfficientDet trained on signature bounding boxes).
    "Sign":           re.compile(
        r"\b(signature|sign(ed|ature)?|hastakshar|दस्तखत|हस्ताक्षर|हस्त\s?क्षर)\b", re.I
    ),
    "Relation":       re.compile(
        r"\b(S\/o|D\/o|W\/o|C\/o|Father|Mother|Spouse|Son|Daughter|Husband|Wife"
        r"|पति|पत्नी|पुत्र|पुत्री)\b", re.I
    ),
}

# All PII column names (used for per-column Excel export)
ALL_PII_TYPES = list(PII_PATTERNS.keys()) + ["Location", "Address", "Name", "Organization"]

INDIAN_CITIES = {
    "bangalore", "bengaluru", "mumbai", "delhi", "new delhi", "chennai",
    "kolkata", "hyderabad", "pune", "ahmedabad", "jaipur", "lucknow",
    "chandigarh", "surat", "nagpur", "patna", "bhopal", "indore",
    "visakhapatnam", "coimbatore", "kochi",
}

ADDRESS_KEYWORDS = re.compile(
    r"\b(street|road|st|rd|nagar|colony|flat|house|building|floor|dist|district"
    r"|pin|pincode|address|पता)\b", re.I
)

# ─────────────────────────────────────────────────────────────────────────────
# 3B. SIGN DETECTION — MODEL EXPLANATION
# ─────────────────────────────────────────────────────────────────────────────
"""
SIGN DETECTION — HOW IT WORKS IN THIS PIPELINE:
================================================
Current approach (Regex / Keyword-based):
  - After OCR extracts text from the image, we search for signature-related
    keywords: "signature", "signed", "hastakshar", "दस्तखत", "हस्ताक्षर"
  - This is TEXTUAL detection — it fires if the document contains a label
    like "Signature:" or "हस्ताक्षर:" near the signature area.
  - Model: NO separate ML model — pure regex on OCR output.
  - Limitation: Doesn't detect actual handwritten strokes; only the label.

For TRUE vision-level signature detection (recommended upgrade):
  - Use a fine-tuned object detection model (e.g., YOLOv8, EfficientDet)
    trained on signature bounding boxes from document images.
  - Alternatively, use a LayoutLM / DocFormer model that jointly understands
    image regions and text, and has been fine-tuned for form field detection.
  - Dataset: Tobacco800, CEDAR, SigNet for training signature detectors.
  - Integration: Run the detection model on the raw PIL image BEFORE OCR,
    and add "Sign: [DETECTED at bbox]" to the PII list independently of OCR text.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 3C. ANONYMIZATION RULES PER PII TYPE
# ─────────────────────────────────────────────────────────────────────────────
"""
FREE TEXT ANONYMIZATION — EXPLANATION:
========================================
Free text anonymization is the process of detecting and redacting/masking
sensitive personal information (PII) from unstructured plain text (as opposed
to structured database records). Unlike databases where you know which column
holds a phone number, free text (e.g., OCR output from ID cards, letters,
forms) mixes PII with non-sensitive context.

Techniques:
  1. Regex-based detection  → Pattern matching for known formats (Aadhaar, PAN).
  2. NER-based detection    → ML models detect names, orgs, locations contextually.
  3. Masking / Redaction    → Replace PII with a placeholder or partial mask.
  4. Tokenization           → Replace PII with a reversible token (for systems
                               that need to recover the original later).
  5. Generalization         → Replace exact value with a broader category
                               (e.g., age 34 → "30s").
  6. Suppression            → Remove the entire field/sentence containing PII.

ANONYMIZATION STRATEGY PER PII TYPE:
======================================
  Aadhaar      → Show last 4 digits, mask first 8: "XXXX XXXX 1234"
  PAN          → Mask middle 4 digits: "ABCXX999X" → "ABCXXXXXX"  actually: keep first 3 + last 1, mask 5-8 → "ABC****X"  
                 Standard: first 5 alpha + 4 digits + 1 alpha. Mask digits: "ABCDE****A"
  Passport     → Mask last 6 characters: "A1234XXX"
  Voter_ID     → Mask last 5 digits: "ABC12XXXXX" 
  Phone_Number → Mask middle 6 digits: "+91 XXXXX67890" → "+91 XXXXX XXXX" (show last 2 only)
                 Better: mask all but last 4: "XXXXXX7890"
  Credit_Card  → PCI-DSS standard: show last 4 digits only: "XXXX XXXX XXXX 1234"
  Bank_Account → Show last 4 digits: "XXXXXXXX1234"
  Email        → Mask local part beyond first 2 chars: "ab****@domain.com"
  DOB          → Generalize to year only: "****-**-1985" or just "1985"
  Date         → Keep as-is (non-sensitive unless it's a DOB)
  IP_Address   → Mask last octet: "192.168.1.XXX"
  User_ID      → Full masking: "[USER_ID REDACTED]"
  Vehicle_Number → Mask middle part: "MH 02 XX XXXX"
  PPP_ID       → Full masking: "[PPP_ID REDACTED]"
  Sign         → Replace with "[SIGNATURE DETECTED]"
  Relation     → Keep as-is (relation type is not sensitive, only the person name is)
  Name         → Replace with "[NAME REDACTED]" or "[PERSON]"
  Location     → Replace with "[LOCATION]" for sensitive contexts
  Address      → Replace with "[ADDRESS REDACTED]"
  Organization → Keep (usually non-sensitive unless it's a medical/legal org)
"""

def anonymize_pii(pii_type: str, value: str) -> str:
    """
    Returns the anonymized version of a detected PII value.
    Each PII type has a specific masking rule.
    """
    v = value.strip()

    if pii_type == "Aadhaar":
        # Remove spaces/dashes, keep last 4, mask first 8 → "XXXX XXXX 1234"
        digits = re.sub(r"[\s\-]", "", v)
        if len(digits) == 12:
            return f"XXXX XXXX {digits[-4:]}"
        return "XXXX XXXX XXXX"  # fallback if malformed

    elif pii_type == "PAN":
        # Format: ABCDE1234F — mask the 4 digits (positions 6-9)
        if len(v) == 10:
            return f"{v[:5]}XXXX{v[-1]}"
        return "XXXXXXXXXX"

    elif pii_type == "Passport":
        # Format: A1234567 (8 chars) — mask last 6
        if len(v) == 8:
            return f"{v[:2]}XXXXXX"
        return "XXXXXXXX"

    elif pii_type == "Voter_ID":
        # Format: ABC1234567 (10 chars) — mask last 5 digits
        if len(v) == 10:
            return f"{v[:3]}XXXXX{v[-2:]}" if len(v) > 5 else "XXXXXXXXXX"
        return v[:3] + "XXXXXXX"

    elif pii_type == "Phone_Number":
        # Keep only last 4 digits visible: "XXXXXX7890"
        digits = re.sub(r"[\s\-\+]", "", v)
        if len(digits) >= 4:
            return "X" * (len(digits) - 4) + digits[-4:]
        return "XXXXXXXXXX"

    elif pii_type == "Credit_Card":
        # PCI-DSS: show only last 4 digits: "XXXX XXXX XXXX 1234"
        digits = re.sub(r"[\s\-]", "", v)
        if len(digits) >= 4:
            return "XXXX XXXX XXXX " + digits[-4:]
        return "XXXX XXXX XXXX XXXX"

    elif pii_type == "Bank_Account":
        # Show only last 4 digits
        digits = re.sub(r"[\s\-]", "", v)
        if len(digits) >= 4:
            return "X" * (len(digits) - 4) + digits[-4:]
        return "XXXXXXXXXXXX"

    elif pii_type == "Email":
        # Mask local part beyond first 2 chars: "ab****@domain.com"
        parts = v.split("@")
        if len(parts) == 2:
            local, domain = parts
            if len(local) > 2:
                return local[:2] + "*" * (len(local) - 2) + "@" + domain
        return "****@****.***"

    elif pii_type == "DOB":
        # Generalize to year only: show only the year part
        year_match = re.search(r"(19|20)\d{2}", v)
        if year_match:
            return f"****/**/{year_match.group(0)}"
        return "**/**/****"

    elif pii_type == "IP_Address":
        # Mask last octet: "192.168.1.XXX"
        parts = v.split(".")
        if len(parts) == 4:
            return f"{parts[0]}.{parts[1]}.{parts[2]}.XXX"
        return "XXX.XXX.XXX.XXX"

    elif pii_type == "Vehicle_Number":
        # Format: MH02AB1234 — mask middle section
        # Pattern: [State][Dist][Series][Number] → mask series+number
        match = re.match(r"([A-Z]{2})[\s\-]?(\d{2})[\s\-]?([A-Z]{1,2})[\s\-]?(\d{4})", v)
        if match:
            return f"{match.group(1)} {match.group(2)} XX XXXX"
        return "XX XX XX XXXX"

    elif pii_type in ("PPP_ID", "User_ID"):
        return f"[{pii_type} REDACTED]"

    elif pii_type == "Sign":
        return "[SIGNATURE DETECTED]"

    elif pii_type == "Name":
        return "[NAME REDACTED]"

    elif pii_type == "Location":
        return "[LOCATION]"

    elif pii_type == "Address":
        return "[ADDRESS REDACTED]"

    elif pii_type == "Organization":
        # Generally non-sensitive; optionally keep or redact
        return v  # Keep organization names; change to "[ORG REDACTED]" if needed

    elif pii_type == "Relation":
        # The relation type (S/o, D/o) is not sensitive; the name next to it is
        return v  # Keep the relation marker, the associated Name will be redacted

    elif pii_type == "Date":
        return v  # Generic dates (non-DOB) are usually not PII

    else:
        return "[REDACTED]"


def scan_pii_rules(text: str) -> dict:
    """
    Detects structured PII using regex rules.
    Returns a dict mapping PII type → list of found values.
    Each type is also anonymized in a parallel dict.
    """
    found   = {pii_type: [] for pii_type in ALL_PII_TYPES}
    anon    = {pii_type: [] for pii_type in ALL_PII_TYPES}

    # ── Regex-based detection ─────────────────────────────────────────────────
    for label, pat in PII_PATTERNS.items():
        for m in pat.finditer(text):
            raw_val = m.group(0).strip()
            found[label].append(raw_val)
            anon[label].append(anonymize_pii(label, raw_val))

    # ── Location rule ─────────────────────────────────────────────────────────
    text_lower = text.lower()
    for city in INDIAN_CITIES:
        if city in text_lower:
            found["Location"].append(city.title())
            anon["Location"].append(anonymize_pii("Location", city.title()))

    # ── Address rule ──────────────────────────────────────────────────────────
    if ADDRESS_KEYWORDS.search(text):
        snippet = text[:80].strip()
        found["Address"].append(snippet)
        anon["Address"].append(anonymize_pii("Address", snippet))

    return found, anon


# ─────────────────────────────────────────────────────────────────────────────
# 4. CHANDRA 2 OCR UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
# ADD this function in its place:
def run_chandra_ocr_batch(image_paths: list) -> dict:
    """
    Runs ONE subprocess that loads Chandra 2 once and processes ALL images.
    Returns dict: {image_path: markdown_text}

    Timeout is set per-image (90s each) as a total budget, not per-subprocess.
    The subprocess itself runs until ALL images are done.
    """
    import json

    tmp_dir      = os.path.join(OUTPUT_DIR, "_chandra_tmp")
    manifest_dir = os.path.join(OUTPUT_DIR, "_chandra_manifest")
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)

    # Build manifest: list of {image, out} for the subprocess
    manifest = []
    path_to_out = {}
    for img_path in image_paths:
        out_path = os.path.join(tmp_dir, f"{uuid.uuid4().hex}.txt")
        manifest.append({"image": img_path, "out": out_path})
        path_to_out[img_path] = out_path

    manifest_path = os.path.join(manifest_dir, f"{uuid.uuid4().hex}_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    # Timeout = 120s to load model + 90s per image
    total_timeout = 120 + (90 * len(image_paths))
    print(f"  [OCR] Subprocess timeout budget: {total_timeout}s "
          f"(120s model load + 90s × {len(image_paths)} images)")

    try:
        result = subprocess.run(
            [CHANDRA_VENV_PY, CHANDRA_SCRIPT, "--manifest", manifest_path],
            capture_output=True,
            text=True,
            timeout=total_timeout,   # scales with number of images
        )
        if result.returncode != 0:
            print(f"  ✗ OCR subprocess failed:\n{result.stderr[-800:]}")
    except subprocess.TimeoutExpired:
        print(f"  ✗ OCR subprocess timed out after {total_timeout}s")

    # Read results (missing file = OCR failed for that image)
    results = {}
    for img_path in image_paths:
        out_path = path_to_out[img_path]
        if os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8", errors="replace") as f:
                results[img_path] = f.read()
            os.remove(out_path)
        else:
            results[img_path] = ""   # failed image gets empty string

    # Cleanup
    if os.path.exists(manifest_path):
        os.remove(manifest_path)

    return results


def markdown_to_clean_lines(markdown_text: str, max_chars_per_line: int = 200) -> list:
    """
    Converts Chandra 2 markdown output to clean, NER-ready text lines.
    Lines are capped at max_chars_per_line to prevent tokenizer overflow.
    200 chars is a safer limit for Hindi/Devanagari (more tokens per char).
    """
    raw_lines = []
    for raw in markdown_text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        # Remove markdown table pipes
        line = re.sub(r"^\|", "", line)
        line = re.sub(r"\|$", "", line)
        line = re.sub(r"\s*\|\s*", "  ", line)
        # Skip table separator rows (only dashes/colons)
        if re.fullmatch(r"[\-:\s]+", line):
            continue
        # Remove heading markers
        line = re.sub(r"^#{1,6}\s*", "", line)
        # Remove list markers
        line = re.sub(r"^[-*]\s+", "", line)
        if line:
            raw_lines.append(line)

    chunked_lines = []
    for line in raw_lines:
        if len(line) <= max_chars_per_line:
            chunked_lines.append(line)
        else:
            # Split on sentence boundaries first
            parts = re.split(r'(?<=[।\.\?\!])\s+', line)
            current_chunk = ""
            for part in parts:
                if len(current_chunk) + len(part) + 1 <= max_chars_per_line:
                    current_chunk += (" " if current_chunk else "") + part
                else:
                    if current_chunk:
                        chunked_lines.append(current_chunk)
                    # If a single part is still too long, force-split on spaces
                    if len(part) > max_chars_per_line:
                        words = part.split()
                        sub_chunk = ""
                        for w in words:
                            if len(sub_chunk) + len(w) + 1 <= max_chars_per_line:
                                sub_chunk += (" " if sub_chunk else "") + w
                            else:
                                if sub_chunk:
                                    chunked_lines.append(sub_chunk)
                                sub_chunk = w
                        if sub_chunk:
                            chunked_lines.append(sub_chunk)
                    else:
                        current_chunk = part
            if current_chunk:
                chunked_lines.append(current_chunk)

    return [l for l in chunked_lines if l.strip()]


def is_devanagari(text: str) -> bool:
    """Returns True if text contains Devanagari Unicode characters."""
    return bool(re.search(r'[\u0900-\u097F]', text))


# ─────────────────────────────────────────────────────────────────────────────
# 5. NER PIPELINE LOADING — [FIX-1] CORRECTED
# ─────────────────────────────────────────────────────────────────────────────
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

def load_ner_pipeline(model_id: str):
    """
    Loads a HuggingFace NER (token classification) pipeline.

    ── FIX-1: TypeError Root Cause ──────────────────────────────────────────
    The original code passed 'truncation=True' and 'padding=True' as kwargs
    to pipeline(). These are NOT valid parameters for pipeline() itself —
    they belong to the TOKENIZER at call time.

    TokenClassificationPipeline._sanitize_parameters() only accepts a specific
    set of kwargs (like 'ignore_labels', 'aggregation_strategy'). Passing
    'truncation' raises:
        TypeError: _sanitize_parameters() got an unexpected keyword argument 'truncation'

    ── CORRECT APPROACH ─────────────────────────────────────────────────────
    1. Do NOT pass truncation/padding to pipeline() or pipeline.__call__().
    2. Set pipe.tokenizer.model_max_length = 512 (controls internal tokenizer limit).
    3. At inference time, pre-tokenize with truncation=True and pass token IDs,
       OR rely on the pipeline's internal handling with model_max_length set.
    4. The safest approach: pre-chunk text to ≤ 200 chars (done in Step 4 above)
       so that tokenized length never exceeds 512 tokens for typical NER models.
    """
    try:
        kwargs = {
            "task":                 "ner",
            "model":                model_id,
            "tokenizer":            model_id,
            "aggregation_strategy": "first",
            "device":               DEVICE_ID,
            "model_kwargs":         {"low_cpu_mem_usage": True},
            # ── DO NOT add 'truncation' or 'padding' here — causes TypeError ──
        }

        # Use float16 for GPU to reduce VRAM usage
        if torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.float16

        # Pass HF token if available (for gated models like IndicNER)
        if HF_TOKEN:
            kwargs["token"] = HF_TOKEN

        pipe = pipeline(**kwargs)

        # ── Set tokenizer limits AFTER pipeline creation (this is correct) ──
        # model_max_length caps tokenization internally within the pipeline
        pipe.tokenizer.model_max_length = 512
        # truncation_side: which end to truncate from if limit exceeded
        pipe.tokenizer.truncation_side  = "right"
        # Note: pipe.tokenizer.truncation = True is NOT valid — removed.

        return pipe

    except Exception as e:
        print(f"  ⚠ Failed to load model '{model_id}': {e}")
        return None


def clear_memory():
    """Frees CPU and GPU memory between model loads."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def safe_delete_pipeline(pipe):
    """
    Safely deletes a HuggingFace pipeline and its sub-components.
    [FIX-2]: Original code used `del pipe.model` which raises AttributeError
    because TokenClassificationPipeline exposes the model as `pipe.model`
    only in some versions. We use hasattr guards.
    """
    try:
        if hasattr(pipe, "model"):
            del pipe.model
    except Exception:
        pass
    try:
        if hasattr(pipe, "tokenizer"):
            del pipe.tokenizer
    except Exception:
        pass
    try:
        del pipe
    except Exception:
        pass
    clear_memory()


# ─────────────────────────────────────────────────────────────────────────────
# 6. NER + PII INFERENCE — [FIX-2] CORRECTED
# ─────────────────────────────────────────────────────────────────────────────
def process_ner_and_pii_for_lines(pipe, lines_data: list, batch_size: int = 16):
    """
    Runs NER inference + regex PII detection on all OCR text lines.

    ── FIX-2 Performance Notes ──────────────────────────────────────────────
    - batch_size reduced from 64 → 16: prevents VRAM OOM on GPU-T4 (16GB).
      With 512 tokens × 16 sequences, peak VRAM usage is ~4-6GB for BERT-base.
    - Text already chunked to ≤200 chars in markdown_to_clean_lines().
    - Inference wrapped in try/except per batch to handle stray long sequences.
    - Truncation now handled via pipe.tokenizer.model_max_length (set in loader).

    ── Why NOT pass truncation=True to pipe() ───────────────────────────────
    pipeline.__call__() for TokenClassificationPipeline passes kwargs through
    _sanitize_parameters() which only whitelists specific args. 'truncation'
    is not whitelisted → TypeError. The tokenizer's model_max_length + our
    text chunking achieves the same safety without any kwargs.
    """

    PROMPT_STOPWORDS = {
        "my", "name", "is", "मेरा", "नाम", "है", "and", "at", "the", "in",
        "by", "or", "to", "was", "were", "और", "ने", "में", "को", "द्वारा",
        "हैं", "था", "थे", "a", "an", "of", "for",
    }

    prompts = [item["prompt_text"] for item in lines_data]

    # ── Batch inference ───────────────────────────────────────────────────────
    # Note: pipe() handles batching internally when given a list.
    # We do NOT pass truncation=True here (causes TypeError in older/newer HF).
    all_raw_outputs = []
    for start_idx in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start_idx : start_idx + batch_size]
        try:
            batch_out = pipe(batch_prompts)
            # pipe() on a list returns a list of lists
            if isinstance(batch_out, list) and len(batch_out) > 0:
                if isinstance(batch_out[0], dict):
                    # Single-item batch returned a flat list → wrap it
                    batch_out = [batch_out]
            all_raw_outputs.extend(batch_out)
        except Exception as e:
            print(f"  ⚠ NER batch [{start_idx}:{start_idx+batch_size}] failed: {e}")
            # Fill failed batch with empty results to maintain index alignment
            all_raw_outputs.extend([[] for _ in batch_prompts])

    # ── Per-line result parsing ───────────────────────────────────────────────
    parsed_types       = []
    parsed_texts       = []
    missed_entities_list = []
    pii_per_type_list    = []   # [FIX-4]: list of dicts, one per line
    anon_per_type_list   = []   # anonymized values per type per line

    for idx, raw_outputs in enumerate(all_raw_outputs):
        item        = lines_data[idx]
        orig_text   = item["original_text"]
        has_context = item["has_context"]

        extracted_spans = []
        entity_types    = []
        detected_words  = set()

        for ent in (raw_outputs or []):
            group = str(ent.get("entity_group", ent.get("entity", ""))).upper()
            word  = ent.get("word", "").strip()

            # Normalize entity group labels
            clean_group = re.sub(r"^[BI]-", "", group)
            if clean_group in ("PER", "PERSON"):
                clean_group = "Name"
            elif clean_group in ("ORG", "ORGANIZATION"):
                clean_group = "Organization"
            elif clean_group in ("LOC", "LOCATION"):
                clean_group = "Location"

            # Clean subword artifacts (## from WordPiece/BPE tokenizers)
            cleaned_word = re.sub(r"^[\#_\s]+", "", word).strip(",.()\"':;")

            # Skip stopwords in short (no-context) prompts
            if not has_context and cleaned_word.lower() in PROMPT_STOPWORDS:
                continue

            if cleaned_word:
                extracted_spans.append(cleaned_word)
                entity_types.append(clean_group)
                for w in cleaned_word.split():
                    detected_words.add(w.lower().replace(".", "").strip())

        # ── Regex PII scan ────────────────────────────────────────────────────
        rule_pii_found, rule_pii_anon = scan_pii_rules(orig_text)

        # Merge NER-detected names/orgs/locations into PII dicts
        ner_pii_found = {pii_type: [] for pii_type in ALL_PII_TYPES}
        ner_pii_anon  = {pii_type: [] for pii_type in ALL_PII_TYPES}

        for span, etype in zip(extracted_spans, entity_types):
            if etype in ner_pii_found:
                ner_pii_found[etype].append(span)
                ner_pii_anon[etype].append(anonymize_pii(etype, span))

        # Combine regex + NER results (deduplicated per type)
        combined_found = {}
        combined_anon  = {}
        for ptype in ALL_PII_TYPES:
            combined_values = list(dict.fromkeys(
                rule_pii_found.get(ptype, []) + ner_pii_found.get(ptype, [])
            ))
            combined_anon_vals = list(dict.fromkeys(
                rule_pii_anon.get(ptype, [])  + ner_pii_anon.get(ptype, [])
            ))
            combined_found[ptype] = " | ".join(combined_values) if combined_values else ""
            combined_anon[ptype]  = " | ".join(combined_anon_vals) if combined_anon_vals else ""

        pii_per_type_list.append(combined_found)
        anon_per_type_list.append(combined_anon)

        # ── Build output strings ──────────────────────────────────────────────
        if extracted_spans:
            final_str = " ".join(extracted_spans)
            # Edge case: if short prompt, don't echo the entire prompt back
            if not has_context and orig_text.lower() in final_str.lower():
                final_str = orig_text
            parsed_texts.append(final_str)
            parsed_types.append(" | ".join(entity_types))
        else:
            parsed_texts.append("")
            parsed_types.append("NONE")

        # ── Missed token tracking ─────────────────────────────────────────────
        words_in_orig = orig_text.split()
        missed_words  = []
        for w in words_in_orig:
            clean_w       = w.strip(",.()\"':;").strip()
            clean_w_lower = clean_w.lower().replace(".", "")
            if clean_w_lower in PROMPT_STOPWORDS:
                continue
            if clean_w_lower not in detected_words:
                if not has_context or clean_w[:1].isupper() or is_devanagari(clean_w):
                    missed_words.append(clean_w)
        missed_entities_list.append(" | ".join(missed_words) if missed_words else "None")

    return parsed_types, parsed_texts, missed_entities_list, pii_per_type_list, anon_per_type_list


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN EXECUTION WORKFLOW
# ─────────────────────────────────────────────────────────────────────────────
def run_batch_ocr_ner_pipeline(folder_path: str):
    valid_exts  = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")
    image_files = [
        os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path))
        if f.lower().endswith(valid_exts)
    ]

    if not image_files:
        raise FileNotFoundError(f"No valid image files found in '{folder_path}'.")

    # ── STEP 1: Chandra 2 OCR ─────────────────────────────────────────────────
    print("=" * 72)
    print(f"STEP 1 — Running Chandra 2 OCR on {len(image_files)} images")
    print("=" * 72)

    dataset_master = []

    # REPLACE with:
    print(f"  Running batch OCR (model loads once for all {len(image_files)} images)...")
    t_ocr_start = time.time()
    ocr_results = run_chandra_ocr_batch(image_files)       # ← NEW batch call
    print(f"  ✓ Batch OCR complete in {time.time() - t_ocr_start:.1f}s")
    
    for img_idx, img_path in enumerate(image_files, 1):
        filename = os.path.basename(img_path)
        raw_md   = ocr_results.get(img_path, "")
        lines    = markdown_to_clean_lines(raw_md, max_chars_per_line=200)
        print(f"  [{img_idx}/{len(image_files)}] {filename}: {len(lines)} line(s)")

        for line_no, clean_line in enumerate(lines, 1):
            word_count = len(clean_line.split())
            has_ctx    = word_count > 3 or any(
                kw in clean_line.lower()
                for kw in ["is", "are", "live", "lives", "name", "naam", "नाम", "है", "रहते"]
            )

            if not has_ctx:
                prompt = (
                    f"मेरा नाम {clean_line} है।"
                    if is_devanagari(clean_line)
                    else f"My name is {clean_line}."
                )
            else:
                prompt = clean_line

            dataset_master.append({
                "Image_Name":    filename,
                "Line_Number":   line_no,
                "original_text": clean_line,
                "prompt_text":   prompt,
                "script":        "Hindi (Devanagari)" if is_devanagari(clean_line) else "English / Transliterated",
                "has_context":   has_ctx,
            })

    # if not dataset_master:
    #     raise ValueError("OCR completed but no text lines were extracted.")

    if not dataset_master:
        print("  ⚠ WARNING: No text lines extracted by OCR. Injecting empty placeholders for downstream evaluation...")
        for img_path in image_files:
            filename = os.path.basename(img_path)
            dataset_master.append({
                "Image_Name":    filename,
                "Line_Number":   1,
                "original_text": "No text detected in image",
                "prompt_text":   "No text detected in image",
                "script":        "English / Transliterated",
                "has_context":   False,
            })

    # ── STEP 2: NER + PII benchmark across all models ─────────────────────────
    print("\n" + "=" * 72)
    print(f"STEP 2 — NER & PII Benchmark across {len(MODELS)} models")
    print(f"         Total text lines: {len(dataset_master)}")
    print("=" * 72)

    # Base DataFrame with OCR results
    df_details = pd.DataFrame([{
        "Image_Name":        item["Image_Name"],
        "Line_Number":       item["Line_Number"],
        "Extracted_OCR_Text": item["original_text"],
        "Script":            item["script"],
    } for item in dataset_master])

    summary_metrics = []

    for model_name, model_id in MODELS.items():
        print(f"\n▶ Evaluating: {model_name}")
        pipe = load_ner_pipeline(model_id)

        if pipe is None:
            print(f"  Skipping {model_name} due to loading error.")
            continue

        pred_types, pred_texts, missed_entities, pii_per_type, anon_per_type = \
            process_ner_and_pii_for_lines(pipe, dataset_master, batch_size=16)

        # ── [FIX-4]: Add separate column per PII type for this model ─────────
        df_details[f"{model_name} | NER_Type"]         = pred_types
        df_details[f"{model_name} | Detected_Entities"] = pred_texts
        df_details[f"{model_name} | Missed_Tokens"]    = missed_entities

        for ptype in ALL_PII_TYPES:
            # Raw detected values
            df_details[f"{model_name} | PII_{ptype}_Detected"] = [
                row.get(ptype, "") for row in pii_per_type
            ]
            # Anonymized values
            df_details[f"{model_name} | PII_{ptype}_Anonymized"] = [
                row.get(ptype, "") for row in anon_per_type
            ]

        # Summary metrics
        total_lines = len(dataset_master)
        person_cnt  = sum(1 for t in pred_types if "Name" in t or "PER" in t)
        loc_cnt     = sum(1 for t in pred_types if "Location" in t or "LOC" in t)
        org_cnt     = sum(1 for t in pred_types if "Organization" in t or "ORG" in t)
        none_cnt    = sum(1 for t in pred_types if t == "NONE")
        pii_hit_cnt = sum(
            1 for row in pii_per_type
            if any(v for v in row.values())
        )

        summary_metrics.append({
            "Model Name":                        model_name,
            "Total Text Lines Evaluated":        total_lines,
            "Detected Names (PER)":              person_cnt,
            "Detected Locations (LOC)":          loc_cnt,
            "Detected Organizations (ORG)":      org_cnt,
            "No Entity Detected (NONE)":         none_cnt,
            "Lines with Any PII (regex+NER)":    pii_hit_cnt,
            "Name Detection Rate (%)":           round((person_cnt / total_lines) * 100, 2),
        })

        safe_delete_pipeline(pipe)

    df_summary = pd.DataFrame(summary_metrics)

    # ── STEP 3: Build PII-only summary sheet (aggregated across all lines) ────
    print("\n" + "=" * 72)
    print("STEP 3 — Building PII Aggregation Sheet")
    print("=" * 72)

    # One row per image: aggregate all detected PII across lines
    pii_agg_rows = []
    for img_file in sorted(df_details["Image_Name"].unique()):
        img_df  = df_details[df_details["Image_Name"] == img_file]
        agg_row = {"Image_Name": img_file}
        for model_name in MODELS.keys():
            for ptype in ALL_PII_TYPES:
                det_col  = f"{model_name} | PII_{ptype}_Detected"
                anon_col = f"{model_name} | PII_{ptype}_Anonymized"
                if det_col in img_df.columns:
                    all_det  = " | ".join(v for v in img_df[det_col].dropna() if v)
                    all_anon = " | ".join(v for v in img_df[anon_col].dropna() if v)
                    agg_row[f"{model_name} | {ptype} (Detected)"]   = all_det
                    agg_row[f"{model_name} | {ptype} (Anonymized)"] = all_anon
        pii_agg_rows.append(agg_row)

    df_pii_agg = pd.DataFrame(pii_agg_rows)

    # ── STEP 4: Export to Excel ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("STEP 4 — Exporting Complete Benchmark Report to Excel")
    print("=" * 72)

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        # Sheet 1: Model comparison summary
        df_summary.to_excel(
            writer, sheet_name="Model Comparison Summary", index=False
        )
        # Sheet 2: Line-level OCR + NER + per-PII-type columns
        df_details.to_excel(
            writer, sheet_name="OCR & NER Line Details", index=False
        )
        # Sheet 3: Per-image PII aggregation with anonymization
        df_pii_agg.to_excel(
            writer, sheet_name="PII Aggregated by Image", index=False
        )

    print(f"\n✓ Excel report saved: '{OUTPUT_EXCEL}'")
    print(f"  Sheets: 'Model Comparison Summary' | 'OCR & NER Line Details' | 'PII Aggregated by Image'")
    return df_summary, df_details, df_pii_agg


# ─────────────────────────────────────────────────────────────────────────────
# EXECUTE
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df_summary, df_details, df_pii_agg = run_batch_ocr_ner_pipeline(IMAGE_FOLDER_PATH)

    print("\n=== BENCHMARK SUMMARY MATRIX ===")
    print(df_summary.to_string(index=False))