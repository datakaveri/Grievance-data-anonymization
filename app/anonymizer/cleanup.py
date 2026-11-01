"""Step 1 — clean source text before anything reads it.

Repairs export artefacts that are not personal data but would otherwise reach
the detectors and the output as junk. Switchable per dataset with
free_text_anonymization.normalize_source_text (default on). The staged output
carries the cleaned text.
"""

from __future__ import annotations

import html
import re
from typing import Any

# An HTML character reference: "&#039;", "&#x27;", "&amp;", "&quot;".
# The Haryana grievance export stores free text HTML-escaped, so
# "Haryana Government&#039;&#039;s Guidelines" (#35) and "demand &amp; Send"
# (#29) reached the models and the output with the codes in them.
_HTML_ENTITY = re.compile(r"&(?:#\d+|#[xX][0-9a-fA-F]+|[A-Za-z]+);")

# Devanagari repairs for Hindi that went through a broken legacy-font (Kruti
# Dev style) converter: "शिक्ष़्ाा" instead of "शिक्षा" (#33), "श्रेण्ी" instead of
# "श्रेणी". Each pattern only removes a sign Unicode does not allow in that
# position, so valid text is never changed:
#   nukta (U+093C) on a letter that never takes one — valid only on
#   क ख ग ज ड ढ फ य न र ळ (क़ ख़ ग़ ज़ ड़ ढ़ फ़ य़ ऩ ऱ ऴ)
_STRAY_NUKTA = re.compile(r"(?<![कखगजडढफयनरळ])़")
#   virama (U+094D) directly before a vowel sign, nukta or another virama
_VIRAMA_BEFORE_SIGN = re.compile(r"्(?=[़ा-्])")
#   the same vowel sign twice in a row ("ाा")
_REPEATED_SIGN = re.compile(r"([ा-ौ])\1+")


def normalize_source_text(value: Any) -> Any:
    """Decode HTML entities and repair invalid Devanagari sign sequences.

    Non-strings and empty strings are returned unchanged, so this can be mapped
    over a whole table column. Examples:
        "Government&#039;s"  -> "Government's"
        "demand &amp; Send"  -> "demand & Send"
        "शिक्ष़्ाा विभाग"      -> "शिक्षा विभाग"
    On the 37,598 configured cells of the Haryana grievance tables this changed
    499 cells: 489 HTML-entity cells and 10 Hindi repairs.
    """
    if not isinstance(value, str) or not value:
        return value
    if "&" in value and _HTML_ENTITY.search(value):
        value = html.unescape(value)
    if "़" in value or "्" in value or _REPEATED_SIGN.search(value):
        value = _STRAY_NUKTA.sub("", value)
        value = _VIRAMA_BEFORE_SIGN.sub("", value)
        value = _REPEATED_SIGN.sub(r"\1", value)
    return value
