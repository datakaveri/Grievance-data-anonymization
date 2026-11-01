"""Small text helpers shared across the pipeline (no models, no regex rules)."""

from __future__ import annotations

import re
import unicodedata
from typing import List


def word_number(text: str, start: int) -> int:
    """1-based index of the word that starts at character `start` of `text`.

    Used for the "word position" columns of the reports: in "Call Ramesh now",
    the word at offset 5 is word 2.
    """
    return len(text[:start].split()) + 1


def is_word_char(ch: str) -> bool:
    """True for a character that belongs inside a word, in any script.

    Letters and digits, plus the combining marks that complete Indic letters.
    Devanagari vowel signs, virama, nukta and anusvara (Unicode categories Mn/Mc)
    are not str.isalnum(), so treating them as word boundaries cut words in the
    middle: a span ending at "उर" in "उर्फ" was redacted to "*्फ", leaving an
    orphaned sign that renders as a dotted circle (issues #21, #26, #53).
    ZWNJ/ZWJ (U+200C/U+200D) shape conjuncts and belong to the word as well.
    """
    return ch.isalnum() or ch in "‌‍" or unicodedata.category(ch) in ("Mn", "Mc")


def line_bases(text: str) -> List[int]:
    """Cell-relative offset of the first non-blank character of each line.

    Detection works on stripped lines (line.strip()), so a span found in line i
    is relative to that stripped line; adding line_bases(text)[i] turns it back
    into an offset in the whole cell. Example: for "a\\n  bc", the bases are
    [0, 4] — "bc" starts at offset 4 of the cell.
    """
    bases: List[int] = []
    cursor = 0
    for line in text.splitlines(keepends=True):
        bases.append(cursor + len(line) - len(line.lstrip()))
        cursor += len(line)
    return bases


# ─────────────────────────────────────────────────────────────────────────────
# Script / language detection
# ─────────────────────────────────────────────────────────────────────────────
# Unicode blocks of the Indian scripts the reports name.
_INDIC_SCRIPTS = {
    "Hindi (Devanagari)": re.compile(r"[ऀ-ॿ]"),
    "Bengali": re.compile(r"[ঀ-৿]"),
    "Tamil": re.compile(r"[஀-௿]"),
    "Telugu": re.compile(r"[ఀ-౿]"),
}
_LATIN_LETTER = re.compile(r"[A-Za-z]")


def detect_language(text: str) -> str:
    """Name the dominant script of `text` by counting letters per Unicode block.

    Returns e.g. "Hindi (Devanagari)", "Hindi (Devanagari) / Mixed" (when the
    Indic script is under 60% of all letters), "English", or "Unknown". This is
    a script count, not language identification: romanised Hindi ("shikayat
    ki") reports as "English". It only labels audit entries and reports; no
    detection decision depends on it.
    """
    if not text or not text.strip():
        return "Unknown"
    text = text.strip()
    indic_counts = {name: len(pattern.findall(text)) for name, pattern in _INDIC_SCRIPTS.items()}
    latin_count = len(_LATIN_LETTER.findall(text))
    top_script = max(indic_counts, key=indic_counts.get)
    top_count = indic_counts[top_script]
    if top_count > 0:
        if latin_count > 0 and top_count / max(latin_count + top_count, 1) < 0.6:
            return f"{top_script} / Mixed"
        return top_script
    if latin_count > 0:
        return "English"
    return "Unknown"
