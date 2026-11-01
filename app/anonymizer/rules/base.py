"""Building blocks shared by every rule file (generic, India, Haryana).

A *rule* finds one kind of personal data in one line of text and reports the
character span of each value it finds. The scanner (anonymizer/scanner.py) runs
every rule over every line, in the order fixed by anonymizer/rules/__init__.py,
and turns the spans into PiiHit records.

Two kinds of rule exist:

* RegexRule — one compiled pattern, optionally with a capture group that holds
  the value and an `accept` check for things a regex cannot express (a checksum,
  a look at the surrounding line). Almost every rule is one of these.
* Configurable rules (the deny list, reference numbers) — their pattern depends
  on the dataset config, so they are built per ScanOptions. They expose the same
  `label` / `source` / `find()` interface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional, Protocol, Sequence, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Boundaries that work next to Devanagari
# ─────────────────────────────────────────────────────────────────────────────
# Python's `\b` treats Hindi letters as word characters, so it finds NO boundary
# between a Hindi letter and a digit: `\b12.12.2025` fails on "दिनांक12.12.2025"
# (issue #53). These lookarounds only forbid an ASCII letter/digit on that side,
# so a value glued to Hindi text still matches.
NO_ALNUM_BEFORE = r"(?<![A-Za-z0-9])"
NO_ALNUM_AFTER = r"(?![A-Za-z0-9])"


class Rule(Protocol):
    """What the scanner needs from a rule."""

    label: str   # PII type written to the audit, e.g. "Aadhaar", "Phone_Number"
    source: str  # detector name in the audit: "Regex", "Contextual Regex", "Deny list"

    def find(self, line: str) -> Iterator[Tuple[int, int]]:
        """Yield (start, end) character offsets of every value in `line`."""
        ...


@dataclass(frozen=True)
class RegexRule:
    """A rule made of one regex.

    label   PII type of every value this rule finds.
    pattern Compiled regex, run with finditer over the line.
    group   Which group is the value: 0 = the whole match. Contextual rules
            match a label ("Aadhaar No.") plus the value and keep only the value.
    source  "Regex" for shape-only rules, "Contextual Regex" when a label word
            must precede the value.
    accept  Optional extra test (match, line) -> bool for what the regex cannot
            say on its own, e.g. Aadhaar's check digit.
    """

    label: str
    pattern: "re.Pattern[str]"
    group: int | str = 0
    source: str = "Regex"
    accept: Optional[Callable[["re.Match[str]", str], bool]] = None

    def find(self, line: str) -> Iterator[Tuple[int, int]]:
        for m in self.pattern.finditer(line):
            if self.accept is None or self.accept(m, line):
                yield m.start(self.group), m.end(self.group)


@dataclass(frozen=True)
class ScanOptions:
    """Dataset-specific settings that change what the rules look for.

    reference_keywords         Labels that need a "No." marker before their number
                               ("Case No. 45"). None = the built-in list.
    strong_reference_keywords  Labels always followed by their number ("FIR 140").
                               None = the built-in list.
    deny_terms                 Names that must always be redacted as PERSON.

    Built from the `free_text_anonymization` block of a dataset config with
    from_settings(); the defaults equal a config that sets none of these keys.
    """

    reference_keywords: Optional[Tuple[str, ...]] = None
    strong_reference_keywords: Optional[Tuple[str, ...]] = None
    deny_terms: Tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_settings(cls, settings: dict) -> "ScanOptions":
        """Read the three keys from a free_text_anonymization block (already validated)."""

        def as_tuple(value: Optional[Sequence[str]]) -> Optional[Tuple[str, ...]]:
            return None if value is None else tuple(value)

        return cls(
            reference_keywords=as_tuple(settings.get("reference_keywords")),
            strong_reference_keywords=as_tuple(settings.get("strong_reference_keywords")),
            deny_terms=tuple(settings.get("ner_deny_terms") or ()),
        )
