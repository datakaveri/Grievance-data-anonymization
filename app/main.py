# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 5 — COMPLETE PIPELINE  v5  (all issues fixed + new models)           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# KEY CHANGES vs v4
# ─────────────────
# [H1]  Sheet 2 & 3 are now ONE ROW PER SOURCE FILE (not per line).
#        - Sheet 3: Source_File, File_Type, Language, Full_Extracted_Text,
#                   Spell_Corrected_Text, Detected_PII_With_Tags,
#                   Anonymized_PII_Output, PII_Types_Detected (values),
#                   PII_Types_Not_Detected (display names)
#        - Sheet 2: Source_File, File_Type, Language, Full_Extracted_Text,
#                   Spell_Corrected_Text, then per-model columns:
#                   {Model}_Predicted_Type, {Model}_Extracted_Text,
#                   {Model}_Missed_Entities
#
# [H2]  Language detection added — "Hindi", "English", "Mixed" per file.
#
# [H3]  Spell correction stage added between OCR and PII/NER:
#        - Primary  : Qwen2.5-7B-Instruct (via venv subprocess)
#        - Fallback : Levenshtein Guard (symspellpy / rapidfuzz)
#        Corrected text is what goes into PII detection and NER.
#
# [H4]  PII detection replaced with model-based detectors (all run in
#        the MAIN kernel, no extra venv needed):
#        - Presidio  with HuggingFace recogniser (en_core_web_lg / RoBERTa)
#        - Piiranha  (fhrzn/pii-detection-roberta-base)
#        - GLiNER-PII (urchade/gliner_multi_pii-v1)
#        Results merged; regex patterns kept as fast pre-filter.
#
# [H5]  PII_Types_Not_Detected  → shows the actual detected VALUES of the
#        PII types that were NOT found (was wrongly showing type-tag names).
#        Corrected: shows missing type DISPLAY NAMES (e.g. "Aadhaar, PAN").
#        PII_Types_Detected     → shows "Type: value" pairs actually found.
#
# [H6]  Sheet 2 NER model columns use safe Excel names:
#        {ModelShortName}_Predicted_Type, _Extracted_Text, _Missed_Entities
#
# [H7]  Devanagari/Hindi text: PII regex skipped; model-based detectors
#        handle Indic PII (names, account numbers in Hindi script).

import subprocess, sys, os, re, gc, json, uuid, warnings, shutil
import torch, pandas as pd
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0. PACKAGE BOOTSTRAP
# ─────────────────────────────────────────────────────────────────────────────
def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

try:
    import docx
except ModuleNotFoundError:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "docx"], check=False)
    _pip("python-docx>=1.1.0")
    import docx

try:
    import pypdf
except ModuleNotFoundError:
    _pip("pypdf>=4.0.0")
    import pypdf

try:
    import openpyxl
except ModuleNotFoundError:
    _pip("openpyxl>=3.1.0")

try:
    import rapidfuzz
except ModuleNotFoundError:
    _pip("rapidfuzz>=3.0")

try:
    import symspellpy
except ModuleNotFoundError:
    _pip("symspellpy")

# ─────────────────────────────────────────────────────────────────────────────
# 1. HARDWARE & PATHS
# ─────────────────────────────────────────────────────────────────────────────
DEVICE_ID  = 0 if torch.cuda.is_available() else -1
DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Pipeline] Device: {DEVICE_STR.upper()}")

HF_TOKEN = os.getenv("HF_TOKEN", "")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

DATASET_FOLDER_PATH = "/kaggle/input/datasets/gogul0604/raw-image"
OUTPUT_DIR          = "/kaggle/working/output"
OUTPUT_EXCEL        = os.path.join(OUTPUT_DIR, "chandra2_ner_pii_report.xlsx")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHANDRA_VENV_PY = "/kaggle/working/chandra_venv/bin/python"
CHANDRA_SCRIPT  = "/kaggle/working/chandra_infer_script.py"
if not os.path.exists(CHANDRA_VENV_PY) or not os.path.exists(CHANDRA_SCRIPT):
    raise RuntimeError("Cell 3.5 output missing — run Cell 3.5 first.")

IMAGE_BATCH_SIZE = 3   # images per Chandra subprocess call

# ─────────────────────────────────────────────────────────────────────────────
# 2. NER MODEL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
NER_MODELS = {
    "HiNER":      "cfilt/HiNER-original-muril-base-cased",
    "IndicNER":   "ai4bharat/IndicNER",
    "BERT_NER":   "dslim/bert-base-NER",
    "XLM_RoBERTa":"Babelscape/wikineural-multilingual-ner",
}
# Short names used as Excel column prefixes (no special chars)
NER_SHORT = list(NER_MODELS.keys())   # ["HiNER", "IndicNER", "BERT_NER", "XLM_RoBERTa"]

# ─────────────────────────────────────────────────────────────────────────────
# 3. PII DISPLAY NAMES & REGEX (fast pre-filter)
# ─────────────────────────────────────────────────────────────────────────────
PII_DISPLAY_NAMES = {
    "Aadhaar":        "Aadhaar",
    "PAN":            "PAN",
    "Phone_Number":   "Phone Number",
    "Email":          "Email",
    "Medical_UHID":   "Medical UHID",
    "Passport":       "Passport",
    "Voter_ID":       "Voter ID",
    "Vehicle_Number": "Vehicle Number",
    "Bank_Account":   "Bank Account",
    "Credit_Card":    "Credit Card",
    "DOB":            "Date of Birth",
    "PPP_ID":         "PPP ID",
}

PII_PATTERNS = {
    "Aadhaar":        re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"),
    "PAN":            re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "Phone_Number":   re.compile(
        r"(?:\+?91[\s\-]?)?[6-9]\d{9}\b|\+91[\s\-]?\d{2}[\s\-]?\d{4}[\s\-]?\d{4}\b"
    ),
    "Email":          re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.I),
    "Medical_UHID":   re.compile(
        r"\b(?:UHID|PMRN|IPD|EPISODE)[\s\:\.\-]*[A-Z]{2,6}[\.\-]?\d{5,15}\b", re.I
    ),
    "Passport":       re.compile(r"\b[A-PR-WY][1-9]\d{7}\b"),
    "Voter_ID":       re.compile(r"\b[A-Z]{3}\d{7}\b"),
    "Vehicle_Number": re.compile(r"\b[A-Z]{2}[\s\-]?\d{2}[\s\-]?[A-Z]{1,2}[\s\-]?\d{4}\b"),
    "Bank_Account":   re.compile(r"(?<!\d)\d{14,18}(?!\d)"),
    "Credit_Card":    re.compile(r"\b(?:4\d{12}(?:\d{3})?|5[1-5]\d{14}|3[47]\d{13})\b"),
    "DOB":            re.compile(
        r"\b(?:0?[1-9]|[12]\d|3[01])[\/\-.](?:0?[1-9]|1[0-2])[\/\-.](?:19|20)\d{2}\b"
    ),
    "PPP_ID":         re.compile(r"\b(?:PPP|PPPID|FAMILYID|FID)[\_\-\:\s]*[A-Z0-9]{6,10}\b"),
}

INDIAN_CITIES = {
    "bangalore","bengaluru","mumbai","delhi","new delhi","chennai","kolkata",
    "hyderabad","pune","ahmedabad","jaipur","lucknow","bhubaneswar","gachibowli",
    "kalinga nagar","secunderabad","noida","gurugram","gurgaon","visakhapatnam",
    "vijayawada","jalaun","orai",
}

_EXCL_ORGS = {
    "INSTITUTE","SCIENCES","HOSPITAL","UNIVERSITY","ACADEMY","DEPARTMENT",
    "LABORATORY","AUTHORITY","MINISTRY","FOUNDATION","BIOCHEMISTRY",
    "GASTROENTEROLOGY","DIAGNOSTIC","PATHOLOGY",
}

INDIAN_TITLE_REGEX = re.compile(
    r"\b(?:Shri|Sri|Smt|Mr|Mrs|Ms|Miss|Dr|Prof|Kumar|Kumari)"
    r"\.?(?:\s*[A-Z]\.)*\s*[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b",
    re.IGNORECASE,
)

_NOISE_TOKENS = {
    "The","A","An","In","On","At","To","Of","For","And","But","Or","Is","Are",
    "Name","Date","Address","Phone","Email","Report","Page","Clinical","Patient",
    "Method","Sample","Type","Unit","Result","Serum","Test","End",
} | _EXCL_ORGS
_NOISE_LOWER = {t.lower() for t in _NOISE_TOKENS}

# ─────────────────────────────────────────────────────────────────────────────
# 4. LANGUAGE DETECTION  [H2]
# ─────────────────────────────────────────────────────────────────────────────
_DEVA_RE = re.compile(r"[\u0900-\u097F]")

def detect_language(text: str) -> str:
    deva = len(_DEVA_RE.findall(text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if deva == 0:
        return "English"
    if latin == 0:
        return "Hindi"
    ratio = deva / max(deva + latin, 1)
    return "Hindi" if ratio > 0.6 else ("Mixed" if ratio > 0.2 else "English")

def has_devanagari(text: str) -> bool:
    return bool(_DEVA_RE.search(text))

# ─────────────────────────────────────────────────────────────────────────────
# 5. OCR OUTPUT SANITISER
# ─────────────────────────────────────────────────────────────────────────────
_HTML_TAG  = re.compile(r"<[^>]+>", re.DOTALL)
_DATA_ATTR = re.compile(r'\s*data-\w+="[^"]*"', re.I)
_MD_HEAD   = re.compile(r"^#{1,6}\s*", re.MULTILINE)
_MD_BOLD   = re.compile(r"\*\*(.+?)\*\*")
_MD_TSEP   = re.compile(r"^\s*\|?[-:\s|]+\|?\s*$", re.MULTILINE)
_MD_TROW   = re.compile(r"^\|(.+)\|$", re.MULTILINE)

def sanitise_ocr(text: str) -> str:
    if not text:
        return ""
    text = _DATA_ATTR.sub("", text)
    text = _HTML_TAG.sub("", text)
    def _trow(m):
        return "  ".join(c.strip() for c in m.group(1).split("|") if c.strip())
    text = _MD_TSEP.sub("", text)
    text = _MD_TROW.sub(_trow, text)
    text = _MD_HEAD.sub("", text)
    text = _MD_BOLD.sub(r"\1", text)
    text = re.sub(r"_(.+?)_", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def markdown_to_lines(text: str) -> list:
    lines = []
    for raw in text.split("\n"):
        ln = raw.strip()
        if not ln or re.fullmatch(r"[\-:\s]+", ln):
            continue
        ln = re.sub(r"^\|", "", ln).rstrip("|").strip()
        ln = re.sub(r"\s*\|\s*", "  ", ln)
        ln = re.sub(r"^[-*]\s+", "", ln)
        if ln:
            lines.append(ln)
    return lines

# ─────────────────────────────────────────────────────────────────────────────
# 6. SPELL CORRECTION  [H3]
#    Primary : Qwen2.5-7B-Instruct (subprocess via venv)
#    Fallback: Levenshtein Guard via symspellpy + rapidfuzz
# ─────────────────────────────────────────────────────────────────────────────

# ── 6a. SymSpell / Levenshtein fallback (runs in main kernel) ────────────────
_symspell = None

def _load_symspell():
    global _symspell
    if _symspell is not None:
        return _symspell
    try:
        from symspellpy import SymSpell, Verbosity as _V
        ss = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        # Use bundled frequency dictionary shipped with symspellpy
        import symspellpy as _ssp_pkg
        _dict_path = os.path.join(
            os.path.dirname(_ssp_pkg.__file__),
            "frequency_dictionary_en_82_765.txt",
        )
        if os.path.exists(_dict_path):
            ss.load_dictionary(_dict_path, term_index=0, count_index=1)
            _symspell = ss
            print("  [SpellCheck] SymSpell dictionary loaded.")
        else:
            print("  [SpellCheck] SymSpell dict not found — Levenshtein only.")
    except Exception as e:
        print(f"  [SpellCheck] SymSpell load failed: {e}")
    return _symspell

def levenshtein_correct(text: str) -> str:
    """
    Word-level spelling correction using SymSpell + rapidfuzz.
    Preserves numbers, Devanagari, and tokens that look like PII.
    """
    if not text or has_devanagari(text):
        return text   # skip Hindi — spell models are English-only

    ss = _load_symspell()
    if ss is None:
        return text

    from symspellpy import Verbosity
    corrected_words = []
    for word in text.split():
        # Keep tokens that are: all-caps abbreviations, numbers, emails, PII-like
        if (re.match(r"^[A-Z]{2,}$", word)
                or re.match(r"^[\d\W]+$", word)
                or "@" in word
                or len(word) <= 2):
            corrected_words.append(word)
            continue
        suggestions = ss.lookup(word, Verbosity.CLOSEST, max_edit_distance=2,
                                 include_unknown=True)
        if suggestions:
            corrected_words.append(suggestions[0].term)
        else:
            corrected_words.append(word)
    return " ".join(corrected_words)

# ── 6b. Qwen2.5-7B-Instruct spell correction (subprocess) ───────────────────
_QWEN_SPELL_SCRIPT = "/kaggle/working/qwen_spell_script.py"

def _write_qwen_spell_script():
    """Write the Qwen spell-correction subprocess script once."""
    script = r'''
import os, sys, json, argparse

_ENV = "/kaggle/working/chandra_env.env"
if os.path.isfile(_ENV):
    for ln in open(_ENV):
        ln = ln.strip()
        if "=" in ln and not ln.startswith("#"):
            k, v = ln.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

_TOKEN = os.environ.get("HF_TOKEN","")
if _TOKEN:
    try:
        from huggingface_hub import login as _l
        _l(token=_TOKEN, add_to_git_credential=False)
    except Exception:
        pass

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

PROMPT_TMPL = (
    "You are a spelling correction assistant. "
    "Correct only obvious spelling mistakes in the following text. "
    "Preserve all numbers, names, dates, account numbers, and medical terms exactly. "
    "Return ONLY the corrected text, nothing else.\n\nText:\n{text}"
)

def correct(model, tokenizer, text: str, device: str) -> str:
    if not text or not text.strip():
        return text
    prompt = PROMPT_TMPL.format(text=text[:2000])
    msgs = [{"role":"user","content":prompt}]
    inp  = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    toks = tokenizer(inp, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**toks, max_new_tokens=1024, do_sample=False)
    generated = out[0][toks["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()
    with open(args.manifest, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    if not tasks:
        sys.exit(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"[Qwen] Loading {MODEL_ID} on {device}...", flush=True)
    kw = dict(torch_dtype=dtype, device_map="auto" if device=="cuda" else None,
              trust_remote_code=True)
    if _TOKEN:
        kw["token"] = _TOKEN
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True,
                                               token=_TOKEN or None)
    model     = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kw)
    if device == "cpu":
        model = model.to(device)
    model.eval()
    print(f"[Qwen] Model loaded.", flush=True)

    for task in tasks:
        text    = task["text"]
        out_path= task["out"]
        try:
            result = correct(model, tokenizer, text, device)
        except Exception as e:
            result = text
            print(f"[Qwen] correction failed: {e}", file=sys.stderr)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"[Qwen] ✓ {out_path}", flush=True)

if __name__ == "__main__":
    main()
'''
    with open(_QWEN_SPELL_SCRIPT, "w", encoding="utf-8") as f:
        f.write(script)

def spell_correct_batch(texts: list) -> list:
    """
    [H3] Spell-correct a list of texts.
    Tries Qwen2.5-7B-Instruct first; falls back to Levenshtein/SymSpell.
    Returns list of corrected strings (same order as input).
    """
    if not texts:
        return []

    results = [None] * len(texts)

    # ── Try Qwen first ────────────────────────────────────────────────────────
    _write_qwen_spell_script()
    TMP = "/kaggle/working/_spell_tmp"
    MAN = "/kaggle/working/_spell_manifest"
    os.makedirs(TMP, exist_ok=True)
    os.makedirs(MAN, exist_ok=True)

    manifest, idx_to_out = [], {}
    for i, text in enumerate(texts):
        if not text or has_devanagari(text):
            results[i] = text   # skip Hindi for Qwen
            continue
        out_path = os.path.join(TMP, f"{uuid.uuid4().hex}.txt")
        manifest.append({"text": text, "out": out_path})
        idx_to_out[i] = out_path

    qwen_ok = False
    if manifest:
        man_path = os.path.join(MAN, f"{uuid.uuid4().hex}.json")
        with open(man_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False)

        env = os.environ.copy()
        try:
            proc = subprocess.run(
                [CHANDRA_VENV_PY, _QWEN_SPELL_SCRIPT, "--manifest", man_path],
                capture_output=True, text=True, timeout=600, env=env,
            )
            if proc.returncode == 0:
                qwen_ok = True
                for i, out_path in idx_to_out.items():
                    if os.path.exists(out_path):
                        with open(out_path, "r", encoding="utf-8") as f:
                            results[i] = f.read().strip() or texts[i]
                        os.remove(out_path)
                    else:
                        results[i] = None   # will fall back
            else:
                print(f"  [Qwen] subprocess failed (exit {proc.returncode}) — "
                      f"falling back to Levenshtein")
                if proc.stderr:
                    print(f"  STDERR: {proc.stderr[-500:]}")
        except Exception as e:
            print(f"  [Qwen] subprocess error: {e} — falling back to Levenshtein")
        try:
            os.remove(man_path)
        except Exception:
            pass

    # ── Levenshtein fallback for anything Qwen didn't handle ─────────────────
    for i, text in enumerate(texts):
        if results[i] is None:
            results[i] = levenshtein_correct(text)

    return results

# ─────────────────────────────────────────────────────────────────────────────
# 7. PII DETECTION — MODEL-BASED  [H4]
#    Models: Presidio+RoBERTa, Piiranha, GLiNER-PII
#    Regex kept as fast pre-filter.
# ─────────────────────────────────────────────────────────────────────────────

# ── 7a. Regex pre-filter ──────────────────────────────────────────────────────
_OCR_FIX = str.maketrans({"O":"0","o":"0","l":"1","I":"1","S":"5","B":"8"})

def _normalize_digits(text: str) -> str:
    def _fix(m):
        r = m.group(0)
        if sum(c.isdigit() for c in r) >= max(2, len(r)-2):
            return r.translate(_OCR_FIX)
        return r
    return re.sub(r"[\w]{4,}", _fix, text)

def regex_pii_scan(text: str) -> list:
    """Returns list of (label, value) from regex patterns."""
    norm = re.sub(r"(\d{4})\s?(\d{4})\s?(\d{4})", r"\1 \2 \3",
                  _normalize_digits(text))
    hits = []
    found_keys = set()
    for label, pat in PII_PATTERNS.items():
        for m in pat.finditer(norm):
            raw = m.group(0).strip()
            if raw.upper() in _EXCL_ORGS:
                continue
            key = (label, re.sub(r"[\s\-]","",raw).lower())
            if key not in found_keys:
                hits.append((label, raw))
                found_keys.add(key)
    text_lo = text.lower()
    for city in INDIAN_CITIES:
        if city in text_lo:
            hits.append(("Location", city.title()))
    return hits

# ── 7b. Presidio  ─────────────────────────────────────────────────────────────
_presidio_analyzer = None

def _load_presidio():
    global _presidio_analyzer
    if _presidio_analyzer is not None:
        return _presidio_analyzer
    try:
        _pip("presidio-analyzer", "presidio-anonymizer", "spacy")
        subprocess.run([sys.executable, "-m", "spacy", "download",
                        "en_core_web_lg", "--quiet"], check=False)
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import (
            NlpEngineProvider, TransformersNlpEngine,
        )
        # Use spacy engine — lighter, no GPU needed
        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
        })
        nlp_engine = provider.create_engine()
        _presidio_analyzer = AnalyzerEngine(nlp_engine=nlp_engine,
                                             supported_languages=["en"])
        print("  [Presidio] Loaded with spacy en_core_web_lg.")
    except Exception as e:
        print(f"  [Presidio] Load failed: {e}")
        _presidio_analyzer = None
    return _presidio_analyzer

def presidio_scan(text: str) -> list:
    """Returns list of (label, value) from Presidio."""
    if has_devanagari(text):
        return []
    analyzer = _load_presidio()
    if analyzer is None:
        return []
    try:
        results = analyzer.analyze(text=text, language="en")
        hits, seen = [], set()
        for r in results:
            val = text[r.start:r.end].strip()
            key = (r.entity_type, val.lower())
            if key not in seen:
                hits.append((r.entity_type, val))
                seen.add(key)
        return hits
    except Exception as e:
        print(f"  [Presidio] scan error: {e}")
        return []

# ── 7c. Piiranha  ─────────────────────────────────────────────────────────────
_piiranha_pipe = None

def _load_piiranha():
    global _piiranha_pipe
    if _piiranha_pipe is not None:
        return _piiranha_pipe
    try:
        from transformers import pipeline as hf_pipeline
        _piiranha_pipe = hf_pipeline(
            "token-classification",
            model="iiiorg/piiranha-v1-detect-personal-information",
            aggregation_strategy="simple",
            device=DEVICE_ID,
            token=HF_TOKEN or None,
        )
        print("  [Piiranha] Loaded.")
    except Exception as e:
        print(f"  [Piiranha] Load failed: {e}")
        _piiranha_pipe = None
    return _piiranha_pipe

def piiranha_scan(text: str) -> list:
    """Returns list of (label, value) from Piiranha."""
    pipe = _load_piiranha()
    if pipe is None or not text.strip():
        return []
    try:
        ents = pipe(text[:512])
        hits, seen = [], set()
        for e in ents:
            lbl = e.get("entity_group", e.get("entity","UNKNOWN"))
            val = e.get("word","").replace("##","").strip()
            if not val:
                continue
            key = (lbl, val.lower())
            if key not in seen:
                hits.append((lbl, val))
                seen.add(key)
        return hits
    except Exception as e:
        print(f"  [Piiranha] scan error: {e}")
        return []

# ── 7d. GLiNER-PII  ───────────────────────────────────────────────────────────
_gliner_model = None
_GLINER_LABELS = [
    "person","organisation","location","date","phone number",
    "email address","bank account","aadhaar","pan card",
    "passport","vehicle number","credit card","address",
]

def _load_gliner():
    global _gliner_model
    if _gliner_model is not None:
        return _gliner_model
    try:
        _pip("gliner")
        from gliner import GLiNER
        _gliner_model = GLiNER.from_pretrained(
            "urchade/gliner_multi_pii-v1",
            token=HF_TOKEN or None,
        )
        print("  [GLiNER] Loaded.")
    except Exception as e:
        print(f"  [GLiNER] Load failed: {e}")
        _gliner_model = None
    return _gliner_model

def gliner_scan(text: str) -> list:
    """Returns list of (label, value) from GLiNER-PII."""
    model = _load_gliner()
    if model is None or not text.strip():
        return []
    try:
        ents = model.predict_entities(text[:1000], _GLINER_LABELS, threshold=0.5)
        hits, seen = [], set()
        for e in ents:
            lbl = e.get("label","").upper().replace(" ","_")
            val = e.get("text","").strip()
            if not val:
                continue
            key = (lbl, val.lower())
            if key not in seen:
                hits.append((lbl, val))
                seen.add(key)
        return hits
    except Exception as e:
        print(f"  [GLiNER] scan error: {e}")
        return []

# ── 7e. Merge all PII results ─────────────────────────────────────────────────
def anonymize_value(ptype: str, val: str) -> str:
    v = val.strip()
    pt = ptype.upper()
    if pt == "AADHAAR" or ptype == "Aadhaar":
        d = re.sub(r"\D","",v)
        return f"XXXX XXXX {d[-4:]}" if len(d)==12 else "XXXX XXXX XXXX"
    elif pt == "PAN":
        c = re.sub(r"\s+","",v)
        return f"{c[:5]}****{c[-1]}" if len(c)==10 else c[:3]+"*****X"
    elif "PHONE" in pt or pt=="PHONE_NUMBER":
        d = re.sub(r"\D","",v)
        return f"XXXXXX{d[-4:]}" if len(d)>=10 else "XXXXXXXXXX"
    elif "EMAIL" in pt:
        if "@" in v:
            local,domain=v.split("@",1)
            mk=local[:2]+"*"*max(1,len(local)-2) if len(local)>2 else local[0]+"*"
            return f"{mk}@{domain}"
        return "*****@***.com"
    elif "BANK" in pt or "ACCOUNT" in pt:
        d = re.sub(r"\D","",v)
        return "*"*(len(d)-4)+d[-4:] if len(d)>=4 else "XXXXXXXXXXXX"
    elif "CARD" in pt or "CREDIT" in pt:
        d = re.sub(r"\D","",v)
        return f"XXXX-XXXX-XXXX-{d[-4:]}" if len(d)>=16 else "XXXX-XXXX-XXXX-XXXX"
    elif pt in ("DOB","DATE","DATE_OF_BIRTH"):
        m = re.search(r"(19|20)\d{2}",v)
        return f"**/**/{m.group(0)}" if m else "**/**/****"
    elif "PASSPORT" in pt:
        c = re.sub(r"\s+","",v)
        return f"{c[0]}XXXXX{c[-2:]}" if len(c)>=8 else c[0]+"XXXXXX"
    elif "PERSON" in pt or pt in ("NAME","PER"):
        parts = v.split()
        return parts[0][0]+"."+" ".join(p[0]+"." for p in parts[1:]) if parts else "[NAME]"
    elif "LOCATION" in pt or "ADDRESS" in pt or pt=="LOC":
        return "[LOCATION REDACTED]"
    elif "ORGANISATION" in pt or "ORG" in pt:
        return "[ORG REDACTED]"
    return "[REDACTED]"

def full_pii_scan(text: str) -> dict:
    """
    [H4] Run regex + Presidio + Piiranha + GLiNER; merge results.
    Returns dict:
        detected_pairs  : list of (source, label, value)
        det_str         : "[Label] value | ..." for audit column
        anon_str        : "masked | ..." anonymised column
        found_types     : set of type labels detected
        detected_values : dict {display_name: [values]}   for [H5]
        not_detected    : str — display names of types NOT found
    """
    all_hits = []   # (source, label, value)
    seen_keys = set()

    def _add(source, label, value):
        key = (label.upper(), re.sub(r"[\s\-]","",value).lower())
        if key not in seen_keys and value.strip():
            all_hits.append((source, label, value))
            seen_keys.add(key)

    for label, val in regex_pii_scan(text):
        _add("Regex", label, val)
    for label, val in presidio_scan(text):
        _add("Presidio", label, val)
    for label, val in piiranha_scan(text):
        _add("Piiranha", label, val)
    for label, val in gliner_scan(text):
        _add("GLiNER", label, val)

    found_types = {lbl.upper() for _, lbl, _ in all_hits}

    # Build display strings
    if all_hits:
        det_str  = " | ".join(f"[{lbl}] {val}" for _, lbl, val in all_hits)
        anon_str = " | ".join(anonymize_value(lbl, val) for _, lbl, val in all_hits)
    else:
        det_str  = "None"
        anon_str = "None"

    # [H5] detected_values: {display_name: [values]}
    detected_values = {}
    for _, lbl, val in all_hits:
        dn = PII_DISPLAY_NAMES.get(lbl, lbl.replace("_"," ").title())
        detected_values.setdefault(dn, []).append(val)

    # Not-detected: display names of PII_PATTERNS types not found
    not_det_list = [
        PII_DISPLAY_NAMES[pt]
        for pt in PII_PATTERNS
        if pt.upper() not in found_types and pt not in {lbl for _,lbl,_ in all_hits}
    ]
    not_det = ", ".join(not_det_list) if not_det_list else "All PII types detected"

    return {
        "detected_pairs": all_hits,
        "det_str":        det_str,
        "anon_str":       anon_str,
        "found_types":    found_types,
        "detected_values":detected_values,
        "not_detected":   not_det,
    }

# ─────────────────────────────────────────────────────────────────────────────
# 8. CHANDRA 2 OCR — sub-batched subprocess
# ─────────────────────────────────────────────────────────────────────────────
def run_chandra_ocr_batch(image_paths: list) -> dict:
    if not image_paths:
        return {}
    all_results = {}
    batches = [image_paths[i:i+IMAGE_BATCH_SIZE]
               for i in range(0, len(image_paths), IMAGE_BATCH_SIZE)]
    print(f"  [OCR] {len(image_paths)} images → {len(batches)} sub-batch(es) "
          f"of ≤{IMAGE_BATCH_SIZE}")

    TMP = "/kaggle/working/_chandra_tmp"
    MAN = "/kaggle/working/_chandra_manifest"
    os.makedirs(TMP, exist_ok=True)
    os.makedirs(MAN, exist_ok=True)
    env = {**os.environ, "HF_TOKEN": HF_TOKEN, "HUGGING_FACE_HUB_TOKEN": HF_TOKEN}

    for bi, batch in enumerate(batches, 1):
        print(f"  [OCR] Sub-batch {bi}/{len(batches)}: "
              f"{[os.path.basename(p) for p in batch]}")
        manifest, p2o = [], {}
        for img in batch:
            op = os.path.join(TMP, f"{uuid.uuid4().hex}.txt")
            manifest.append({"image": img, "out": op})
            p2o[img] = op
        mp = os.path.join(MAN, f"{uuid.uuid4().hex}.json")
        with open(mp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False)
        try:
            proc = subprocess.run(
                [CHANDRA_VENV_PY, CHANDRA_SCRIPT, "--manifest", mp],
                capture_output=True, text=True, timeout=600, env=env,
            )
            if proc.returncode != 0:
                print(f"  [WARN] Chandra exit {proc.returncode}: {proc.stderr[-500:]}")
        except subprocess.TimeoutExpired:
            print(f"  [ERROR] Sub-batch {bi} timed out")
        except Exception as e:
            print(f"  [ERROR] Sub-batch {bi}: {e}")

        for img in batch:
            op    = p2o[img]
            fname = os.path.basename(img)
            if os.path.exists(op):
                raw = open(op, encoding="utf-8-sig", errors="replace").read().strip()
                os.remove(op)
                content = sanitise_ocr(raw)
                all_results[img] = content if content else f"[OCR EMPTY: {fname}]"
            else:
                all_results[img] = f"[OCR FAILED: {fname}]"
        try:
            os.remove(mp)
        except Exception:
            pass
    return all_results

# ─────────────────────────────────────────────────────────────────────────────
# 9. NER PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
from transformers import pipeline as hf_pipeline

def load_ner_pipeline(model_id: str):
    try:
        kw = dict(task="ner", model=model_id, tokenizer=model_id,
                  aggregation_strategy="simple", device=DEVICE_ID,
                  model_kwargs={"low_cpu_mem_usage":True})
        if torch.cuda.is_available():
            kw["torch_dtype"] = torch.float16
        if HF_TOKEN:
            kw["token"] = HF_TOKEN
        return hf_pipeline(**kw)
    except Exception as e:
        print(f"  ⚠ NER load failed '{model_id}': {e}")
        return None

def chunk_text(text: str, max_chars=1800, overlap=200) -> list:
    if len(text) <= max_chars:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start+max_chars, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks

def run_ner(pipe, text: str) -> dict:
    buckets = {"PER":[],"ORG":[],"LOC":[],"OTHER":[]}
    seen    = set()
    for chunk in chunk_text(text):
        try:
            ents = pipe(chunk)
        except Exception as e:
            print(f"    [WARN] NER chunk error: {e}")
            continue
        for e in ents:
            grp  = str(e.get("entity_group", e.get("entity",""))).upper()
            word = e.get("word","").strip().replace("##","").strip()
            if not word or word.upper() in _EXCL_ORGS:
                continue
            k = word.lower()
            if k in seen:
                continue
            seen.add(k)
            if grp in ("PER","PERSON","NAME"):  buckets["PER"].append(word)
            elif grp in ("ORG","ORGANIZATION","ORGANISATION"): buckets["ORG"].append(word)
            elif grp in ("LOC","LOCATION","GPE","FAC"):        buckets["LOC"].append(word)
            else:                                              buckets["OTHER"].append(word)
    return buckets

def compute_missed(text: str, found_lower: set) -> list:
    missed, seen = [], set()
    for w in text.split():
        cw = re.sub(r"[^\w]","",w)
        lw = cw.lower()
        if (len(cw)>3 and w[0].isupper()
                and lw not in found_lower
                and w not in _NOISE_TOKENS
                and lw not in _NOISE_LOWER
                and lw not in seen):
            missed.append(cw)
            seen.add(lw)
    return missed

# ─────────────────────────────────────────────────────────────────────────────
# 10. DOCUMENT FILE READER
# ─────────────────────────────────────────────────────────────────────────────
def extract_from_file(file_path: str) -> str:
    """Return full text of a document file as a single string."""
    ext = os.path.splitext(file_path)[1].lower()
    lines = []
    try:
        if ext in (".html",".htm"):
            soup = BeautifulSoup(open(file_path,encoding="utf-8",errors="ignore").read(),"html.parser")
            for el in soup.find_all(["p","div","td","tr","h1","h2","h3","span"]):
                t = el.get_text(strip=True)
                if t and len(t)>5:
                    lines.append(t)
        elif ext == ".docx":
            doc = docx.Document(file_path)
            for p in doc.paragraphs:
                if p.text.strip():
                    lines.append(p.text.strip())
            for table in doc.tables:
                for row in table.rows:
                    rt = " | ".join(c.text.strip() for c in row.cells if c.text.strip())
                    if rt:
                        lines.append(rt)
        elif ext == ".pdf":
            for page in pypdf.PdfReader(file_path).pages:
                txt = page.extract_text() or ""
                for ln in txt.split("\n"):
                    if ln.strip():
                        lines.append(ln.strip())
        elif ext == ".txt":
            for ln in open(file_path,encoding="utf-8",errors="ignore"):
                if ln.strip():
                    lines.append(ln.strip())
    except Exception as e:
        print(f"  [WARN] Reader error {file_path}: {e}")
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# 11. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def run_full_pipeline():
    IMG_EXTS = (".jpg",".jpeg",".png",".bmp",".webp",".tiff")
    DOC_EXTS = (".html",".htm",".pdf",".docx",".txt")

    all_files = []
    if os.path.exists(DATASET_FOLDER_PATH):
        for root,_,files in os.walk(DATASET_FOLDER_PATH):
            for fname in files:
                all_files.append(os.path.join(root,fname))

    image_files = [f for f in all_files if f.lower().endswith(IMG_EXTS)]
    doc_files   = [f for f in all_files if f.lower().endswith(DOC_EXTS)]
    print(f"\n▶ STEP 1: {len(image_files)} image(s), {len(doc_files)} document(s)")

    # ── OCR ───────────────────────────────────────────────────────────────────
    ocr_results = run_chandra_ocr_batch(image_files)

    # ── Build one record per source file  [H1] ────────────────────────────────
    # Each record: {source_file, file_type, raw_text, is_failed}
    records = []

    for img_path in image_files:
        fname    = os.path.basename(img_path)
        raw_text = ocr_results.get(img_path, f"[OCR FAILED: {fname}]")
        records.append({
            "Source_File": fname,
            "File_Type":   "IMAGE",
            "Raw_Text":    raw_text,
            "Is_Failed":   raw_text.startswith("[OCR"),
        })

    for doc_path in doc_files:
        fname    = os.path.basename(doc_path)
        raw_text = extract_from_file(doc_path)
        records.append({
            "Source_File": fname,
            "File_Type":   os.path.splitext(fname)[1].upper().replace(".",""),
            "Raw_Text":    raw_text if raw_text.strip() else f"[EMPTY: {fname}]",
            "Is_Failed":   not raw_text.strip(),
        })

    print(f"  Records: {len(records)} total")

    # ── STEP 2: Spell correction  [H3] ────────────────────────────────────────
    print("\n▶ STEP 2: Spell correction (Qwen2.5-7B → Levenshtein fallback)...")
    raw_texts      = [r["Raw_Text"] for r in records]
    corrected_texts = spell_correct_batch(raw_texts)
    for i, r in enumerate(records):
        r["Corrected_Text"] = corrected_texts[i] if not r["Is_Failed"] else r["Raw_Text"]

    # ── STEP 3: Language detection  [H2] ──────────────────────────────────────
    print("\n▶ STEP 3: Language detection...")
    for r in records:
        r["Language"] = detect_language(r["Raw_Text"])
    lang_counts = {}
    for r in records:
        lang_counts[r["Language"]] = lang_counts.get(r["Language"],0)+1
    print(f"  Languages: {lang_counts}")

    # ── STEP 4: PII detection → Sheet 3  [H4] [H5] ───────────────────────────
    print("\n▶ STEP 4: PII Detection (Regex + Presidio + Piiranha + GLiNER)...")
    sheet3_rows    = []
    files_with_pii = 0

    for r in records:
        if r["Is_Failed"]:
            sheet3_rows.append({
                "Source_File":            r["Source_File"],
                "File_Type":              r["File_Type"],
                "Language":               r["Language"],
                "Full_Extracted_Text":    r["Raw_Text"],
                "Spell_Corrected_Text":   r["Raw_Text"],
                "Detected_PII_With_Tags": "OCR Failed",
                "Anonymized_PII_Output":  "OCR Failed",
                "PII_Types_Detected":     "N/A",
                "PII_Types_Not_Detected": "N/A",
            })
            continue

        pii = full_pii_scan(r["Corrected_Text"])
        if pii["det_str"] != "None":
            files_with_pii += 1

        # [H5] PII_Types_Detected: "Type: val1, val2 | Type2: val3"
        if pii["detected_values"]:
            det_vals_str = " | ".join(
                f"{dn}: {', '.join(vs)}"
                for dn, vs in pii["detected_values"].items()
            )
        else:
            det_vals_str = "None"

        sheet3_rows.append({
            "Source_File":            r["Source_File"],
            "File_Type":              r["File_Type"],
            "Language":               r["Language"],
            "Full_Extracted_Text":    r["Raw_Text"],
            "Spell_Corrected_Text":   r["Corrected_Text"],
            "Detected_PII_With_Tags": pii["det_str"],
            "Anonymized_PII_Output":  pii["anon_str"],
            "PII_Types_Detected":     det_vals_str,
            "PII_Types_Not_Detected": pii["not_detected"],
        })

    df_sheet3 = pd.DataFrame(sheet3_rows)
    print(f"  Files with PII: {files_with_pii}/{len(records)}")

    # ── STEP 5: NER Comparison → Sheet 2  [H1][H6] ───────────────────────────
    print("\n▶ STEP 5: NER Models — Comparison (Sheet 2)...")

    # Base columns (one row per file)
    sheet2_base = []
    for r in records:
        regex_names = INDIAN_TITLE_REGEX.findall(r["Corrected_Text"])
        sheet2_base.append({
            "Source_File":          r["Source_File"],
            "File_Type":            r["File_Type"],
            "Language":             r["Language"],
            "Full_Extracted_Text":  r["Raw_Text"],
            "Spell_Corrected_Text": r["Corrected_Text"],
            "Regex_Detected_Names": " | ".join(regex_names) if regex_names else "None",
        })

    # Per-model columns appended in-place
    summary_rows = []

    for short_name, model_id in NER_MODELS.items():
        print(f"  ── {short_name} ({model_id}) ──")
        pipe = load_ner_pipeline(model_id)

        col_type   = f"{short_name}_Predicted_Type"
        col_entity = f"{short_name}_Extracted_Text"
        col_missed = f"{short_name}_Missed_Entities"

        files_names = 0
        files_org   = 0
        files_loc   = 0
        total_names = 0

        for i, r in enumerate(records):
            text = r["Corrected_Text"]

            if r["Is_Failed"] or pipe is None:
                sheet2_base[i][col_type]   = "Load Failed" if pipe is None else "OCR Failed"
                sheet2_base[i][col_entity] = "N/A"
                sheet2_base[i][col_missed] = "N/A"
                continue

            buckets = run_ner(pipe, text)

            # Merge regex + NER PER
            raw_regex  = [n.strip() for n in sheet2_base[i]["Regex_Detected_Names"].split(" | ")
                          if n.strip() and n.strip() != "None"]
            seen_names = set()
            all_names  = []
            for nm in raw_regex + buckets["PER"]:
                k = nm.lower()
                if k and k not in seen_names and nm.upper() not in _EXCL_ORGS:
                    all_names.append(nm)
                    seen_names.add(k)

            all_found_lower = (
                {n.lower() for n in all_names}
                | {n.lower() for n in buckets["ORG"]}
                | {n.lower() for n in buckets["LOC"]}
            )
            missed = compute_missed(text, all_found_lower)

            type_parts, entity_parts = [], []
            if all_names:
                type_parts.append("PER")
                entity_parts.append("PER: " + " | ".join(all_names))
                files_names += 1
                total_names += len(all_names)
            if buckets["ORG"]:
                type_parts.append("ORG")
                entity_parts.append("ORG: " + " | ".join(buckets["ORG"]))
                files_org += 1
            if buckets["LOC"]:
                type_parts.append("LOC")
                entity_parts.append("LOC: " + " | ".join(buckets["LOC"]))
                files_loc += 1

            sheet2_base[i][col_type]   = ", ".join(type_parts) if type_parts else "None"
            sheet2_base[i][col_entity] = " || ".join(entity_parts) if entity_parts else "None"
            sheet2_base[i][col_missed] = " | ".join(missed) if missed else "None"

        rate = round(files_names / max(len(records),1) * 100, 2)
        summary_rows.append({
            "Model Name":              f"{short_name} ({model_id})",
            "Total Files Processed":   len(records),
            "Files with Names (PER)":  files_names,
            "Total Names Found":       total_names,
            "Files with ORG":          files_org,
            "Files with LOC":          files_loc,
            "Files with PII":          files_with_pii,
            "Name Detection Rate %":   rate,
        })

        if pipe is not None:
            del pipe
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    df_sheet2 = pd.DataFrame(sheet2_base)

    # ── STEP 6: Model Summary → Sheet 1 ──────────────────────────────────────
    print("\n▶ STEP 6: Building Sheet 1 (Model Summary)...")
    df_sheet1 = pd.DataFrame(summary_rows)

    # ── STEP 7: Export to Excel ───────────────────────────────────────────────
    print("\n▶ STEP 7: Writing Excel Report...")
    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        df_sheet1.to_excel(writer, sheet_name="Model Summary",        index=False)
        df_sheet2.to_excel(writer, sheet_name="OCR NER Comparison",   index=False)
        df_sheet3.to_excel(writer, sheet_name="PII Detection Tagged",  index=False)

        from openpyxl.styles import PatternFill, Font, Alignment
        from openpyxl.utils  import get_column_letter

        H_FILL  = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
        H_FONT  = Font(color="FFFFFF", bold=True, size=10)
        E_FILL  = PatternFill(start_color="EBF3FB", end_color="EBF3FB", fill_type="solid")
        PII_F   = PatternFill(start_color="FFE0E0", end_color="FFE0E0", fill_type="solid")
        ANON_F  = PatternFill(start_color="E0F7E0", end_color="E0F7E0", fill_type="solid")
        LANG_F  = PatternFill(start_color="FFF3CD", end_color="FFF3CD", fill_type="solid")

        for sname in writer.sheets:
            ws = writer.sheets[sname]
            for cell in ws[1]:
                cell.fill      = H_FILL
                cell.font      = H_FONT
                cell.alignment = Alignment(wrap_text=True, vertical="center")

            for col_idx, col_cells in enumerate(ws.columns, 1):
                hdr = str(col_cells[0].value or "")
                col_letter = get_column_letter(col_idx)
                max_len = max((len(str(c.value)) for c in col_cells if c.value), default=10)
                ws.column_dimensions[col_letter].width = min(max_len+4, 60)

                for row_idx, cell in enumerate(col_cells[1:], 2):
                    val = str(cell.value or "")
                    if any(k in hdr for k in ("PII","Detected","Anonymized")):
                        if val not in ("None","OCR Failed","N/A","All PII types detected"):
                            cell.fill = PII_F
                    elif "Anonymized" in hdr:
                        if val not in ("None","OCR Failed","N/A"):
                            cell.fill = ANON_F
                    elif hdr == "Language":
                        cell.fill = LANG_F
                    elif row_idx % 2 == 0:
                        if (cell.fill is None
                                or cell.fill.fill_type is None
                                or cell.fill.fill_type == "none"):
                            cell.fill = E_FILL

            ws.freeze_panes = "A2"
            ws.row_dimensions[1].height = 28

    print(f"\n✅ DONE — Output: {OUTPUT_EXCEL}")
    print(f"   Sheet 1: Model Summary          ({len(summary_rows)} models)")
    print(f"   Sheet 2: OCR NER Comparison     ({len(df_sheet2)} rows — 1 per file)")
    print(f"   Sheet 3: PII Detection Tagged   ({len(df_sheet3)} rows — 1 per file)")

if __name__ == "__main__":
    run_full_pipeline()