"""Decide which model-detected entities are NOT personal data and must stay visible.

The models tag many everyday words as names ("ATR", "complainant", "Sir",
"हेतू"). Hiding them costs readability, but a filter that clears a real name
causes a leak, so every rule below was tested against names that look like the
junk it targets, and narrowed until none leaked:

  * "Shree", "श्री", "Sahab" are names too        -> not on the title list
  * "Lucky", "Sunny", "Singh", "Ram" are English words -> the vocabulary rule
    never clears a capitalised name, and checks the dataset's own casing
  * "Model Town", "Civil Lines" are plain English  -> LOCATION is never cleared
    by the vocabulary rule

On 3,000 random lines of the Haryana grievance data the filters cleared 123
detections, all non-personal (ATR, Jan Samvaad, श्रीमान, R/Sir, Pls, challan,
hereby...), and no real name.

    false_positive_reason(entity, line, allow_keys, casing) -> reason or None
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, FrozenSet, List, Optional, Sequence

from ..rules import haryana
from ..rules.india import HINDI_FUNCTION_WORDS, NEVER_NAME_TITLES
from .spans import NerEntity

try:
    from spacy.lang.en.stop_words import STOP_WORDS as _ENGLISH_STOPWORDS
    from spacy.lang.hi.stop_words import STOP_WORDS as _HINDI_STOPWORDS
except ImportError:  # the function-word rule then only knows HINDI_FUNCTION_WORDS
    _ENGLISH_STOPWORDS, _HINDI_STOPWORDS = frozenset(), frozenset()

try:
    from wordfreq import zipf_frequency
except ImportError:  # the organisation vocabulary rule is then skipped
    zipf_frequency = None

# English zipf frequency (wordfreq) from which a word counts as everyday
# vocabulary: 3.0 = about once per million words ("complainant" 3.1, "office" 5.4).
COMMON_WORD_ZIPF = 3.0
# Casing evidence for the vocabulary rules: a word must have been seen at least
# this often mid-sentence, and written lowercase at least this share of the time.
CASING_MIN_SEEN = 3
CASING_LOWER_SHARE = 0.9

_TOKEN_SPLIT = re.compile(r"[\s/.,()\-]+")
_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z']*")
_DEVANAGARI_WORD = re.compile(r"[ऀ-ॿ]+")


def _tokens(text: str) -> List[str]:
    """Split entity text into words on spaces and . , / ( ) -"""
    return [t for t in _TOKEN_SPLIT.split(text) if t]


def ner_term_key(term: str) -> str:
    """Spelling-insensitive key used to match entities against allow-lists.

    Ignores case, spaces and punctuation, and folds common romanisation and
    Devanagari spelling variants together, so one allow-list entry covers them all:
        "Jan Samvaad" / "Jansamvaad" / "jan samwad" / "Jan Sanwaad" -> "jansamvad"
        "सी.एम.विंडो" / "सीएम विन्डो" / "सी.एम.विन्ङो"                   -> same key
    Folds applied, in order: drop non-word characters; w -> v; n/m before v -> m;
    doubled vowels -> one; drop nukta; nasal consonant + virama -> anusvara
    (न्ड -> ंड); ङ -> ड (a typo seen in the data); chandrabindu -> anusvara.
    """
    key = unicodedata.normalize("NFC", term).lower()
    key = re.sub(r"[^\wऀ-ॿ]+|_", "", key)
    key = key.replace("w", "v")
    key = re.sub(r"[nm](?=v)", "m", key)
    key = re.sub(r"([aeiou])\1+", r"\1", key)
    key = key.replace("़", "")
    key = re.sub(r"[नमङञण]्(?=[क-ह])", "ं", key)
    key = key.replace("ङ", "ड").replace("ँ", "ं")
    return key


# The hardcoded Haryana exemptions, as match keys.
_HARDCODED_ALLOW_KEYS = frozenset(ner_term_key(t) for t in haryana.NER_ALLOW_TERMS)


def word_casing_stats(texts: Sequence[str]) -> Dict[str, List[int]]:
    """Count how a dataset writes each Latin word mid-sentence.

    Returns {lowercased word: [times written lowercase, times capitalised]}.
    Skips lines written wholly in upper or lower case (no casing signal), the
    first word of each sentence (always capitalised), and ALL-CAPS words
    (acronyms say nothing about name-ness). In the Haryana grievance data
    "singh" is [21, 176] (a name) and "afresh" is [9, 0] (vocabulary).
    """
    stats: Dict[str, List[int]] = {}
    for text in texts:
        for line in text.splitlines():
            if line.isupper() or line.islower():
                continue
            for sentence in re.split(r"[.!?।\n]+", line):
                for word in _LATIN_WORD.findall(sentence)[1:]:
                    if word.isupper():
                        continue
                    entry = stats.setdefault(word.lower(), [0, 0])
                    entry[0 if word.islower() else 1] += 1
    return stats


def false_positive_reason(
    entity: NerEntity,
    line: str,
    allow_keys: FrozenSet[str] = frozenset(),
    casing: Optional[Dict[str, List[int]]] = None,
) -> Optional[str]:
    """Why `entity` is not personal data, or None to keep it redacted.

    Rules, checked in order (the returned reason names the rule):

    allow_list / hardcoded_allow
        The whole entity matches a dataset allow-list term (ner_allow_terms) or a
        hardcoded Haryana exemption (rules/haryana.py), via ner_term_key. Whole
        entity only: allow-listing "Jan Samvaad" never exempts "Jan Samvaad Ramesh".
    title_only
        Every word is a form of address that is never a name: "SIR", "R/Sir",
        "Sir/Madam", "श्रीमान" (rules/india.py NEVER_NAME_TITLES).
    function_word
        Found by one model, written lowercase or in Hindi, and every word is a
        grammar word (spaCy English/Hindi stopwords + HINDI_FUNCTION_WORDS):
        "us", "हेतू", "hereby". A capitalised "May" or "Will" may be a name, so
        it is kept.
    common_words_org
        An ORGANIZATION found by one model whose every word is everyday English
        (zipf >= 3.0) AND written lowercase in this dataset: "concerned
        department", "Pls", "higher authority". "Ram Traders" is kept: "ram" is
        common English, but this data writes it capitalised.
    lowercase_common_word
        A PERSON found by one model, written in lowercase inside normal-case
        text, made of words this dataset almost always writes lowercase:
        "afresh", "challan", "complainant". A word seen too rarely to judge
        keeps the entity redacted.

    LOCATION is never cleared by the vocabulary rules, and neither is anything
    both models agree on.
    """
    tokens = _tokens(entity.text)
    if not tokens:
        return None
    lowered = [t.lower() for t in tokens]

    key = ner_term_key(entity.text)
    if allow_keys and key in allow_keys:
        return "allow_list"
    if key in _HARDCODED_ALLOW_KEYS:
        return "hardcoded_allow"

    if all(t in NEVER_NAME_TITLES for t in lowered):
        return "title_only"

    single_model = len(entity.models) <= 1
    latin = all(_LATIN_WORD.fullmatch(t) for t in tokens)
    latin_lower = latin and all(t.islower() for t in tokens)
    devanagari = not latin and all(_DEVANAGARI_WORD.fullmatch(t) for t in tokens)

    if single_model and (latin_lower or devanagari) and all(
        t in _ENGLISH_STOPWORDS or t in _HINDI_STOPWORDS or t in HINDI_FUNCTION_WORDS for t in lowered
    ):
        return "function_word"

    # The vocabulary rules need casing evidence from normal-case text.
    if not single_model or not latin or not casing or line.islower() or line.isupper():
        return None

    def mostly_lowercase(word: str) -> bool:
        lower_n, upper_n = casing.get(word, (0, 0))
        seen = lower_n + upper_n
        return seen >= CASING_MIN_SEEN and lower_n / seen >= CASING_LOWER_SHARE

    if entity.category == "ORGANIZATION" and zipf_frequency is not None and all(
        zipf_frequency(t, "en") >= COMMON_WORD_ZIPF and mostly_lowercase(t) for t in lowered
    ):
        return "common_words_org"

    if entity.category == "PERSON" and latin_lower and all(mostly_lowercase(t) for t in lowered):
        return "lowercase_common_word"

    return None


def filter_entities(
    entities_per_line: List[List[NerEntity]],
    lines: Sequence[str],
    allow_terms: Optional[Sequence[str]],
    casing: Dict[str, List[int]],
) -> Dict[str, int]:
    """Drop false positives from entities_per_line IN PLACE; return {reason: count}."""
    allow_keys = frozenset(ner_term_key(t) for t in allow_terms or [])
    dropped: Dict[str, int] = {}
    for line_id, line in enumerate(lines):
        kept = []
        for entity in entities_per_line[line_id]:
            reason = false_positive_reason(entity, line, allow_keys, casing)
            if reason is None:
                kept.append(entity)
            else:
                dropped[reason] = dropped.get(reason, 0) + 1
        entities_per_line[line_id] = kept
    return dropped
