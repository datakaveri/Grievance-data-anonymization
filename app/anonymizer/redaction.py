"""Step 4 — redaction: replace every detected value in a cell according to its type.

    redact_text(text, rule_hits, model_hits, column, salts, ...) -> (redacted text, audit entries)
    replacement_for(label, value, column, salts) -> (replacement, technique)

This module only applies policy; it detects nothing. The detections come from
scanner.scan_cell (step 2) and ner/index.py (step 3), wired up in pipeline.py.
"""

from __future__ import annotations

import hashlib
import re
import secrets
from typing import Any, Callable, Dict, List, Optional, Tuple

from .text import detect_language

# What a suppressed value becomes.
REDACTED = "*"


def replacement_for(label: str, value: str, column: str, salts: Dict[str, str]) -> Tuple[str, str]:
    """Return (replacement text, technique name) for one detected value.

    The label decides the policy (POLICY_BY_LABEL, exact label names):

      Age                    kept as is ("retained") — an age alone identifies no one
      Aadhaar                "XXXX XXXX 0124" (last 4 kept)            partial_mask
      PAN                    "ABCDE****F"                               partial_mask
      Phone_Number           "*"                                        suppress
      Email                  "ra****@gmail.com" (domain kept)           domain_preserving_mask
      Bank_Account           "******0000" (last 4 kept)                 partial_mask
      Credit_Card            "XXXX-XXXX-XXXX-1111"                      tokenization
      Date                   "XX-XX-2025" (year kept)                   date_generalization
      Pincode                "132XXX" (district digits kept)            partial_mask
      PERSON / LOCATION /
      ORGANIZATION           "*"                                        suppress
      any other label        SHA-256(salt + value), hex                 salted_hash
                             (Voter ID, passport, DL, vehicle, IFSC, PPP ID,
                             reference and ID numbers, IP, UPI, GSTIN, ABHA...)

    Labels are matched exactly, not by substring: an earlier version tested
    substrings ("AGE", "PAN", "CARD", "ORG"...), so a new label such as
    "Company_ID" (contains PAN) or "Ration_Card" would silently have taken a
    partial mask meant for another type. A new label now falls to the hash,
    which reveals nothing, until it is given a policy here on purpose.

    The salted hash matches SKALD's hashing_with_salt: one random salt per
    column, generated once per run (stored in `salts`) and reused for every row,
    so equal values in a column hash equally within a run but not across runs.
    A value too short for its mask (e.g. an "Aadhaar" that is not 12 digits) is
    suppressed instead.
    """
    policy = POLICY_BY_LABEL.get(label)
    if policy is not None:
        return policy(value)
    salt = salts.setdefault(column, secrets.token_hex(32))
    return hashlib.sha256((salt + value.strip()).encode()).hexdigest(), "salted_hash"


def _digits(value: str) -> str:
    return "".join(ch for ch in value if ch.isdigit())


def _mask_aadhaar(value: str) -> Tuple[str, str]:
    digits = _digits(value)
    return (f"XXXX XXXX {digits[-4:]}" if len(digits) == 12 else REDACTED, "partial_mask")


def _mask_pan(value: str) -> Tuple[str, str]:
    compact = value.strip().replace(" ", "")
    return (f"{compact[:5]}****{compact[-1]}" if len(compact) == 10 else REDACTED, "partial_mask")


def _mask_email(value: str) -> Tuple[str, str]:
    value = value.strip()
    if "@" not in value:
        return REDACTED, "suppress"
    local, domain = value.split("@", 1)
    masked = local[:2] + "*" * max(1, len(local) - 2) if len(local) > 2 else local[:1] + "*"
    return f"{masked}@{domain}", "domain_preserving_mask"


def _mask_account(value: str) -> Tuple[str, str]:
    digits = _digits(value)
    return ("*" * max(0, len(digits) - 4) + digits[-4:] if len(digits) >= 4 else REDACTED, "partial_mask")


def _tokenize_card(value: str) -> Tuple[str, str]:
    digits = _digits(value)
    return (f"XXXX-XXXX-XXXX-{digits[-4:]}" if len(digits) >= 16 else REDACTED, "tokenization")


def _generalize_date(value: str) -> Tuple[str, str]:
    year = next(iter(re.findall(r"(?:19|20)\d{2}", value)), "")
    return (f"XX-XX-{year}" if year else "[DATE REDACTED]", "date_generalization")


def _mask_pincode(value: str) -> Tuple[str, str]:
    digits = _digits(value)
    return (digits[:3] + "XXX" if len(digits) >= 6 else REDACTED, "partial_mask")


def _suppress(value: str) -> Tuple[str, str]:
    return REDACTED, "suppress"


# Policy per detection label (rule labels from anonymizer/rules/, model labels
# from ner/inference.py). Labels not listed are hashed.
POLICY_BY_LABEL: Dict[str, Callable[[str], Tuple[str, str]]] = {
    "Age": lambda value: (value, "retained"),
    "Aadhaar": _mask_aadhaar,
    "PAN": _mask_pan,
    "Phone_Number": _suppress,
    "Email": _mask_email,
    "Bank_Account": _mask_account,
    "Credit_Card": _tokenize_card,
    "Date": _generalize_date,
    "Pincode": _mask_pincode,
    "PERSON": _suppress,
    "LOCATION": _suppress,
    "ORGANIZATION": _suppress,
}
# Labels whose policy keeps the text: never applied to a span that holds another type.
RETAINED_LABELS = frozenset({"Age"})


def redact_text(
    text: str,
    rule_hits: List[Dict[str, Any]],
    model_hits: List[Dict[str, Any]],
    column: str,
    salts: Dict[str, str],
    minimum_confidence: float = 0.0,
    include_values: bool = False,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Redact one cell. Returns (redacted text, audit entries in text order).

    `rule_hits` (scanner.scan_cell) and `model_hits` (ner/index.py) are
    detection dicts {start, end, label, source, confidence, value} with
    zero-based cell offsets. `column` picks the hash salt in `salts`.

    a. Both lists are pooled; anything below `minimum_confidence` is dropped
       (rule hits are always 1.0).
    b. Detections are sorted by position. Overlapping or touching ones become ONE
       replacement covering their union; _policy_hit() picks whose policy it uses.
    c. Replacements are applied right to left so earlier offsets stay valid.

    Each audit entry records label, source, language, technique, confidence and
    the replaced span; `models` when present; and, with include_values (debug
    only), the unredacted text — which is personal data.
    """
    if not isinstance(text, str) or not text:
        return text, []

    language = detect_language(text)
    # Priority = list position: rule hits in scanner order (specific rules before
    # broad ones, see rules/__init__.py), then model hits, most confident first.
    detections = [d for d in rule_hits if d["confidence"] >= minimum_confidence]
    detections.extend(
        sorted((hit for hit in model_hits if hit["confidence"] >= minimum_confidence), key=lambda h: -h["confidence"])
    )
    ranked = sorted(enumerate(detections), key=lambda item: (item[1]["start"], item[1]["end"]))
    if not ranked:
        return text, []

    # Group overlapping/touching detections into one span each.
    spans: List[Tuple[int, int, List[Tuple[int, Dict[str, Any]]]]] = []
    for rank, hit in ranked:
        start, end = hit["start"], hit["end"]
        if spans and start <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], end), spans[-1][2] + [(rank, hit)])
        else:
            spans.append((start, end, [(rank, hit)]))

    output = text
    audit: List[Dict[str, Any]] = []
    for start, end, span_hits in reversed(spans):
        winner = _policy_hit(span_hits, start, end)
        if winner is None:
            replacement, technique = REDACTED, "suppress"
        else:
            replacement, technique = replacement_for(winner["label"], text[start:end], column, salts)
        output = output[:start] + replacement + output[end:]
        for _, hit in span_hits:
            entry = {
                "label": hit["label"],
                "source": hit["source"],
                "language": language,
                "technique": technique,
                "confidence": hit["confidence"],
                "start_offset": start,
                "end_offset": end,
            }
            if "models" in hit:
                entry["models"] = hit["models"]
            if include_values:
                entry["detected_text"] = text[start:end]
            audit.append(entry)
    return output, list(reversed(audit))


def _policy_hit(
    span_hits: List[Tuple[int, Dict[str, Any]]], start: int, end: int
) -> Optional[Dict[str, Any]]:
    """The detection whose policy replaces the merged span [start, end), or None.

    Only a detection covering the WHOLE span may lend its policy, because a
    policy is applied to the whole span's text: a partial mask or a kept age
    would otherwise reveal what the other detections found. Among those, the
    highest priority (lowest rank) wins — so a reference number keeps its hash
    when the model also tags "CMOFF" inside "CMOFF/J/2026/001234", and an
    Aadhaar keeps its partial mask when the catch-all ID rule claims it too.
    "Retained" labels (Age) never win when another type is in the span.
    None means no detection covers the span (partial overlaps): suppress it all.
    """
    labels = {hit["label"] for _, hit in span_hits}
    covering = [
        (rank, hit)
        for rank, hit in span_hits
        if hit["start"] <= start and hit["end"] >= end
        and not (hit["label"] in RETAINED_LABELS and len(labels) > 1)
    ]
    return min(covering, key=lambda item: item[0])[1] if covering else None
