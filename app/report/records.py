"""Data model of the NER report: one FileRecord per input document."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from anonymizer.ner.spans import NerEntity
from anonymizer.scanner import PiiHit


def ordinal(n: int) -> str:
    """Word position as text for the report: 1 -> "1st word", 12 -> "12th word", 0 -> "N/A"."""
    if n <= 0:
        return "N/A"
    suffix = "th" if 11 <= (n % 100) <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix} word"


@dataclass
class LineNerResult:
    """What one model (or the hybrid merge) found in one line.

    persons/locs/orgs list entity texts by category; `missed` lists entities
    that some other model found in this line but this one did not.
    """

    model: str
    line_no: int
    text: str
    entities: List[NerEntity] = field(default_factory=list)
    persons: List[str] = field(default_factory=list)
    locs: List[str] = field(default_factory=list)
    orgs: List[str] = field(default_factory=list)
    missed: List[str] = field(default_factory=list)

    @property
    def predicted_type(self) -> str:
        """Categories present, e.g. "PERSON | LOCATION", or "NONE"."""
        types = [name for name, found in (("PERSON", self.persons), ("LOCATION", self.locs),
                                          ("ORGANIZATION", self.orgs)) if found]
        return " | ".join(types) if types else "NONE"

    @property
    def extracted_text(self) -> str:
        """Entities with score and position, e.g.
        "[PER] Ramesh Kumar (conf: 1.00, 2nd word, letters 6-17) || [LOC] Karnal (...)"."""
        parts = []
        for short, category in (("PER", "PERSON"), ("LOC", "LOCATION"), ("ORG", "ORGANIZATION")):
            found = [e for e in self.entities if e.category == category]
            if found:
                formatted = [
                    f"{e.text} (conf: {e.score:.2f}, {ordinal(e.word_no)}, letters {e.start_char}-{e.end_char})"
                    for e in found
                ]
                parts.append(f"[{short}] " + " | ".join(formatted))
        return " || ".join(parts) if parts else "None"


@dataclass
class FileRecord:
    """One input document and everything found in it.

    line_ners maps a model label (or HYBRID_LABEL) to its per-line results.
    """

    path: str
    filename: str
    file_type: str
    language: str
    raw_text: str
    pii_hits: List[PiiHit] = field(default_factory=list)
    line_ners: Dict[str, List[LineNerResult]] = field(default_factory=dict)
