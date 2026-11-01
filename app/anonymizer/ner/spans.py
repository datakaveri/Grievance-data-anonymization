"""Turn raw model spans into clean, whole-word entities, one merged list per line.

A model reports character spans that can start or end inside a word (it works
on sub-word pieces), include stray punctuation, or disagree with the other
model about where an entity ends or what type it is. merge_line_spans() fixes
all of that in four steps:

  a. snap each span outward to whole words          (_snap_to_word_boundary)
  b. trim punctuation and relation prefixes          (_clean_entity_text)
  c. join touching/overlapping spans of one type     ("Ram" + "Kumar" -> "Ram Kumar")
  d. resolve overlaps between different types        (keep the most confident type)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from ..text import is_word_char, word_number


@dataclass
class NerEntity:
    """One merged entity in one line.

    category    PERSON, LOCATION or ORGANIZATION
    text        the entity text after cleanup
    score       best model confidence among the spans merged into it (0-1)
    word_no     1-based index of its first word in the line
    start_char  1-based offset of its first character in the line
    end_char    offset just past its last character
    models      labels of the models whose spans make up this entity (any type);
                the false-positive filters treat "found by one model" more strictly
    """

    category: str
    text: str
    score: float
    word_no: int = 0
    start_char: int = 0
    end_char: int = 0
    models: List[str] = field(default_factory=list)


# Characters a span never grows across: whitespace, slash, backslash, colon,
# semicolon, comma, brackets and quotes. "DCP/Gohana" stays two words.
_STRICT_DELIMITERS = set(" /\\:;,()[]{}<>\"'\t\n\r")


def _snap_to_word_boundary(text: str, start: int, end: int) -> Tuple[int, int]:
    """Grow [start, end) outward until both ends sit on word boundaries.

    Word characters include Devanagari vowel signs and virama (is_word_char), so
    a span ending at "उर" of "उर्फ" grows to the whole word instead of leaving
    "*्फ" behind (#53). Two kinds of dot are crossed:
      * backwards, a dot preceded by a single letter that starts a word or
        continues a chain of initials — so a span starting at "A" of "C.I.A" or
        at "P" of "D.S.P" grows to the whole abbreviation (#38);
      * forwards, a dot followed by a letter or digit ("S.I.No", "10.5").
    """
    while start > 0:
        prev = text[start - 1]
        if prev in _STRICT_DELIMITERS:
            break
        if is_word_char(prev):
            start -= 1
        elif (
            prev == "."
            and start > 1
            and text[start - 2].isalpha()
            and (
                start == 2
                or text[start - 3] in _STRICT_DELIMITERS
                or text[start - 3].isspace()
                or text[start - 3] == "."  # chain of initials: C.I.A
            )
        ):
            start -= 1
        else:
            break

    while end < len(text):
        nxt = text[end]
        if nxt in _STRICT_DELIMITERS:
            break
        if is_word_char(nxt):
            end += 1
        elif nxt == "." and end + 1 < len(text) and text[end + 1].isalnum():
            end += 1
        else:
            break
    return start, end


# What to strip from the START of a span, repeatedly, alternatives in order:
#   [\s:\-.,'"/()]+          leading punctuation and spaces
#   X/o followed by "." or space    relation markers S/o, D/o, W/o, R/o, H/o
#                                    ("S/o Ram Lal" -> "Ram Lal"). Only "/o":
#                                    a looser letter/letter form cut "R/Sir" to "ir".
#   X.o.                     dotted relation marker "s.o.Suresh" -> "Suresh"
#   X. not followed by Y.    a stray initial ("o.Name" -> "Name"); a letter-dot
#                            followed by another letter-dot is an abbreviation
#                            ("C.I.A", #38) and is kept.
_LEADING_JUNK = re.compile(
    r"^([\s:\-.,'\"/()]+|[A-Za-z]/[oO](?:\.\s*|\s+)|[A-Za-z]\.[oO]\.\s*|[a-zA-Z]\.(?![A-Za-z]\.)\s*)"
)
# What to strip from the END: trailing punctuation/spaces, or a dangling single
# Latin letter after a space ("Ram Lal s" -> "Ram Lal").
_TRAILING_JUNK = re.compile(r"(\s+[a-zA-Z]|[\s:\-.,'\"/()]+)$")


def _clean_entity_text(text: str, start: int, end: int) -> Tuple[str, int, int]:
    """Trim junk from both ends of text[start:end]; return (clean text, start, end).

    Never trims the span to nothing: a strip that would remove everything left
    is skipped.
    """
    value = text[start:end]
    match = _LEADING_JUNK.match(value)
    while match and match.end() > 0:
        cut = match.end()
        if cut >= len(value):
            break
        start += cut
        value = text[start:end]
        match = _LEADING_JUNK.match(value)

    match = _TRAILING_JUNK.search(value)
    while match and match.start() < len(value):
        cut = len(value) - match.start()
        if cut >= len(value):
            break
        end -= cut
        value = text[start:end]
        match = _TRAILING_JUNK.search(value)
    return value.strip(), start, end


# Abbreviations like "S.I.No" are document furniture, not entities.
_DOTTED_ABBREVIATION = re.compile(r"^[A-Z]\.([A-Z]\.)+[A-Za-z]+$")


def merge_line_spans(spans: List[Dict], line: str) -> List[NerEntity]:
    """Merge every model's raw spans for one line into a clean entity list.

    `spans` are dicts {cat, start, end, score, model} from ner/inference.py,
    possibly from several models. Returns entities sorted by position.

    Stage c (same type): spans of one category that overlap or touch (at most one
    character apart) become one entity covering both, unless a "/", "\\", ";" or
    ":" lies between them — so XLM-R's "ramesh verma" and IndicNER's "ramesh"
    become one "ramesh verma", while "DCP/Gohana" stays two entities.

    Stage d (different types): overlapping entities of different categories become
    one entity covering their union, typed as the MOST CONFIDENT member. Keeping
    only the longest (the earlier rule) let a 0.46 ORGANIZATION "sh Mah" erase a
    0.90 PERSON "Mah" so the name fell under the confidence cutoff, and left the
    uncovered part of a dropped span visible ("ramesh" when a LOCATION
    "verma nagar" won over the PERSON "ramesh verma").
    """
    if not spans:
        return []

    # Steps 1-2: snap and clean every span; drop what is left empty or meaningless.
    snapped: List[Dict] = []
    for s in spans:
        start, end = _snap_to_word_boundary(line, s["start"], s["end"])
        clean, start, end = _clean_entity_text(line, start, end)
        if not clean or len(clean) < 2 or re.match(r"^\d+$", clean) or _DOTTED_ABBREVIATION.match(clean):
            continue
        snapped.append(
            {
                "cat": s["cat"],
                "start": start,
                "end": end,
                "score": float(s["score"]),
                "text": clean,
                "models": {s["model"]} if s.get("model") else set(),
            }
        )
    if not snapped:
        return []

    # Stage c: join same-category spans that overlap or touch.
    same_type: List[Dict] = []
    for s in sorted(snapped, key=lambda x: (x["start"], -(x["end"] - x["start"]))):
        target = None
        for m in same_type:
            touching = s["start"] <= m["end"] + 1 and m["start"] <= s["end"] + 1
            if m["cat"] == s["cat"] and touching:
                between = line[min(m["start"], s["start"]) : max(m["end"], s["end"])]
                if not any(d in between for d in ("/", "\\", ";", ":")):
                    target = m
                    break
        if target is None:
            same_type.append({**s, "models": set(s["models"])})
            continue
        target["models"] |= s["models"]
        target["text"], target["start"], target["end"] = _clean_entity_text(
            line, min(target["start"], s["start"]), max(target["end"], s["end"])
        )
        target["score"] = max(target["score"], s["score"])

    # Stage d: group overlapping spans of any category, keep the most confident type.
    same_type.sort(key=lambda x: (x["start"], -x["end"]))
    groups: List[List[Dict]] = []
    for s in same_type:
        if groups and s["start"] < max(g["end"] for g in groups[-1]):
            groups[-1].append(s)
        else:
            groups.append([s])

    merged: List[Dict] = []
    for group in groups:
        best = max(group, key=lambda x: (x["score"], x["end"] - x["start"]))
        if len(group) == 1:
            merged.append(best)
            continue
        text, start, end = _clean_entity_text(
            line, min(g["start"] for g in group), max(g["end"] for g in group)
        )
        merged.append(
            {**best, "start": start, "end": end, "text": text, "models": set().union(*(g["models"] for g in group))}
        )

    entities: List[NerEntity] = []
    seen = set()
    for m in merged:
        if not m["text"] or len(m["text"]) < 2:
            continue
        key = (m["cat"], m["text"].lower(), m["start"])
        if key in seen:
            continue
        seen.add(key)
        entities.append(
            NerEntity(
                category=m["cat"],
                text=m["text"],
                score=round(m["score"], 4),
                word_no=word_number(line, m["start"]),
                start_char=m["start"] + 1,
                end_char=m["end"],
                models=sorted(m["models"]),
            )
        )
    return entities
