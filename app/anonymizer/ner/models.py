"""Which NER models run, how they are loaded, and the runtime knobs around them.

Two HuggingFace token-classification models make up the "hybrid" NER:

  IndicNER     ai4bharat/IndicNER                 strong on Devanagari Hindi
  XLM-RoBERTa  Davlan/xlm-roberta-base-ner-hrl    strong on English / romanised text

Both tag PERSON, LOCATION and ORGANIZATION. Their spans are pooled and merged
per line (ner/spans.py). The registry NER_MODELS is the single source of truth:
the pipeline (every job), the NER report and the Docker build's model prefetch
(app/model_setup.py) all read it.

Everything runs on CPU: a TEE has no GPU passthrough, and CUDA is hidden before
torch is imported so it never probes for one.
"""

from __future__ import annotations

import importlib.util
import os
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# ── Environment, set before torch / transformers are imported ───────────────
# Hide CUDA so torch never probes for a GPU driver (the pipeline is CPU-only).
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Loading a .bin checkpoint (IndicNER ships one) otherwise starts a background
# thread that asks the Hub to open a safetensors-conversion PR: a network call
# on every run, even from a warm cache, and a 401 on the gated repo.
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
# Rust-based parallel downloads, only when the hf_transfer package is installed
# (huggingface_hub fails hard if the flag is set without it). The Docker image
# installs it; at run time models load from the local cache anyway.
if importlib.util.find_spec("hf_transfer") is not None:
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
# transformers / torch emit many deprecation warnings that drown the job log.
warnings.filterwarnings("ignore")

try:  # optional: read HF_TOKEN from a local .env file
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    import torch

    # Cap CPU threads: more threads on one short sequence measured slower, and
    # the NER report runs two model threads side by side.
    torch.set_num_threads(min(4, os.cpu_count() or 4))

    from transformers import AutoModelForTokenClassification, AutoTokenizer
    from transformers import pipeline as hf_pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    TRANSFORMERS_AVAILABLE = False

# HuggingFace pipeline device id: -1 = CPU.
DEVICE = -1
DEVICE_NAME = "CPU"


# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class NerModelSpec:
    """One NER model.

    key        short name used in report file names ("IndicNER" -> report_IndicNER.json)
    label      display name, written to audits and reports as the detector name
    model_id   HuggingFace repo id
    use_fast   load the fast (Rust) tokenizer; IndicNER needs the slow one
    min_score  spans scoring below this are discarded straight out of the model.
               0.45 for both: at the earlier 0.20 floor the models tagged common
               Hindi words ("hamare", "pichhle", "do") as places/organisations.
    """

    key: str
    label: str
    model_id: str
    use_fast: bool
    min_score: float


NER_MODELS: Tuple[NerModelSpec, ...] = (
    NerModelSpec("IndicNER", "IndicNER (AI4Bharat / Public)", "ai4bharat/IndicNER", False, 0.45),
    NerModelSpec(
        "XLM_RoBERTa", "XLM-RoBERTa (Multilingual)", "Davlan/xlm-roberta-base-ner-hrl", True, 0.45
    ),
)

# Name of the merged result of all models (used as a detector name in the report).
HYBRID_LABEL = "Hybrid (IndicNER + XLM-RoBERTa)"


# ─────────────────────────────────────────────────────────────────────────────
# Runtime knobs (environment variables)
# ─────────────────────────────────────────────────────────────────────────────

# Lines handed to one inference call by the pipeline. Bookkeeping only: it sets
# progress-report granularity, not how much work a forward pass does.
NER_CORPUS_BATCH_SIZE = int(os.getenv("NER_CORPUS_BATCH_SIZE", "256"))

# Real tensor batching. Every sequence in a batch pads to the batch's longest,
# so the gain does not grow with batch size: past the optimum, padding waste
# overtakes it. On a reference export (172k distinct lines, mean 48 chars) the
# best size tracked the thread count — at 2 threads 8 won every repeat (43 vs 63
# ms/line one-at-a-time; 16 gave 52, 32 gave 64), at 4 threads 16 won. 0 = pick
# from that relationship; an explicit value pins it; 1 turns batching off.
NER_TENSOR_BATCH_SIZE = int(os.getenv("NER_TENSOR_BATCH_SIZE", "0"))

# Torch CPU threads for the pipeline; 0 = leave the cap of 4 set above. Raising
# it is not reliably a win (on a hybrid-core laptop CPU it measured slower), so
# it is an opt-in knob to benchmark per machine.
NER_TORCH_THREADS = int(os.getenv("NER_TORCH_THREADS", "0"))

# Dynamic INT8 quantization of the Linear layers. ~1.5x faster, but on a
# reference corpus it found 386 entities where fp32 found 481, only 290 shared —
# a fifth of the names and places going unmasked. Off unless a deployment has
# measured the trade-off on its own data.
NER_QUANTIZE = os.getenv("NER_QUANTIZE", "").strip().lower() in {"1", "true", "yes"}


def tensor_batch_size() -> int:
    """Sequences per forward pass: NER_TENSOR_BATCH_SIZE, or max(8, 4 x torch threads)."""
    if NER_TENSOR_BATCH_SIZE > 0:
        return NER_TENSOR_BATCH_SIZE
    return max(8, 4 * torch.get_num_threads())


def torch_threads() -> int:
    """Current torch CPU thread count, or 0 when torch is not installed."""
    return torch.get_num_threads() if torch is not None else 0


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

# model_id -> loaded pipeline (None when loading failed, so it is not retried).
_PIPELINE_CACHE: Dict[str, object] = {}


def _maybe_quantize(model, model_id: str):
    """Return the INT8-quantized model when NER_QUANTIZE is set, else `model` itself."""
    if not NER_QUANTIZE:
        return model
    try:
        quantized = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        print(f"  [NER] Quantized {model_id} to INT8 (dynamic).", flush=True)
        return quantized
    except Exception as exc:
        print(f"  [NER] INT8 quantization unavailable for {model_id}: {exc}", flush=True)
        return model


def load_ner_pipeline(model_id: str, use_fast: bool = True) -> Optional[object]:
    """Load (once) a HuggingFace "ner" pipeline for `model_id`; None if it cannot load.

    The local cache is tried first: an online load still probes the Hub (for a
    model.safetensors IndicNER does not ship), and on the gated IndicNER repo
    without a token that probe returns 401 and aborts the load even though every
    file is cached. Only a real cache miss goes to the network (with HF_TOKEN).

    aggregation_strategy="simple" makes the pipeline join word pieces into whole
    entity spans with character offsets.

    transformers warns that Davlan/xlm-roberta-base-ner-hrl loads "with an
    incorrect regex pattern" and suggests fix_mistral_regex=True. That is a false
    alarm for this model: as loaded, its tokens matched the reference
    SentencePiece tokenizer on 3,000 of 3,000 real grievance lines, while the
    suggested flag matched only 569. Do not set it.
    """
    if model_id in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[model_id]
    if not TRANSFORMERS_AVAILABLE:
        return None
    try:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=use_fast, local_files_only=True)
            model = AutoModelForTokenClassification.from_pretrained(model_id, local_files_only=True)
        except OSError:
            token = os.getenv("HF_TOKEN") or None
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, use_fast=use_fast)
            model = AutoModelForTokenClassification.from_pretrained(model_id, token=token)
        model.eval()
        model = _maybe_quantize(model, model_id)
        pipe = hf_pipeline(
            task="ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=DEVICE,
        )
    except Exception as exc:
        print(f"  [NER] Could not load {model_id}: {exc}", flush=True)
        pipe = None
    _PIPELINE_CACHE[model_id] = pipe
    return pipe


def release_ner_pipeline(model_id: str) -> None:
    """Forget a loaded pipeline so its weights can be garbage-collected."""
    _PIPELINE_CACHE.pop(model_id, None)


def run_pipe(pipe, texts, batch_size: Optional[int] = None):
    """Run a NER pipeline on one string or a list of strings, without autograd.

    A pipeline handed a list still defaults to batch_size=1 — one forward pass
    per item — so passing a list alone buys nothing; `batch_size` opts into real
    tensor batching. torch.inference_mode() skips gradient bookkeeping.
    """
    kwargs = {} if batch_size is None else {"batch_size": batch_size}
    with torch.inference_mode():
        return pipe(texts, **kwargs)
