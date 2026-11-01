"""Detection rules that hold for text from anywhere: no country or language knowledge.

Each rule below documents what it matches, an example it catches, what it
deliberately leaves alone, and why it exists. India-specific rules live in
rules/india.py and Haryana-specific ones in rules/haryana.py; the order the
scanner runs them in is fixed in rules/__init__.py.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterator, Optional, Tuple

from ..text import is_word_char
from .base import RegexRule

# ─────────────────────────────────────────────────────────────────────────────
# Contact details
# ─────────────────────────────────────────────────────────────────────────────

# E-mail address: local part, "@", domain with at least one dot and a 2+ letter TLD.
#   catches:  "ramesh.k@gmail.com", "info@hsvp.gov.in"
#   ignores:  "ramesh@okaxis" (no dot after @: that is a UPI handle, see india.UPI_ID)
EMAIL = RegexRule(
    "Email",
    re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
)

# IPv4 address: four dot-separated groups of 1-3 digits.
#   catches:  "10.12.20.1"
#   note:     does not check each group is <= 255; "999.1.1.1" also matches.
IP_ADDRESS = RegexRule(
    "IP_Address",
    re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"),
)

# ─────────────────────────────────────────────────────────────────────────────
# Payment cards
# ─────────────────────────────────────────────────────────────────────────────

# Card number: a Visa (4xxx), Mastercard (51-55xx), Amex (34/37xx) or Discover
# (6011) prefix, then 12 more digits in groups of four, spaces/dashes optional.
#   catches:  "4111 1111 1111 1111", "5500-0000-0000-0004"
#   note:     always 16 digits, so 15-digit Amex numbers are not matched here
#             (the catch-all ID rule in india.py still hashes them).
CREDIT_CARD = RegexRule(
    "Credit_Card",
    re.compile(r"\b(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6011)[\s\-]?(?:\d{4}[\s\-]?){2}\d{4}\b"),
)

# ─────────────────────────────────────────────────────────────────────────────
# Numeric dates (day first). Month-name and Hindi dates are in india.py.
# ─────────────────────────────────────────────────────────────────────────────
# Both use (?<![\d./-]) rather than \b so a date glued to a Hindi word still
# matches: "दिनांक12.12.2025" (issue #53) has no \b before the 1.

# dd.mm.yyyy with / - or . separators (may be mixed), 4-digit year 1900-2099.
#   catches:  "12.12.2025", "09/04/2026", "25-3-2026", "दिनांक12.12.2025"
#   ignores:  "10.12.20.1" style IP addresses (the year must be 19xx/20xx and
#             nothing digit-like may follow)
NUMERIC_DATE = RegexRule(
    "Date",
    re.compile(
        r"(?<![\d.\/\-])(?:0?[1-9]|[12]\d|3[01])[\/\-.](?:0?[1-9]|1[0-2])[\/\-.](?:19|20)\d{2}(?!\d)"
    ),
)

# dd.mm.yy — 2-digit year, and the SAME separator twice (backreference \1).
#   catches:  "25.03.26" ("SP Sir on DT 25.03.26", issue #17), "1/4/25"
#   ignores:  "1.2-3" style ranges (separators differ), "10.12.20.1" (a digit
#             follows), section numbers like "3.12" (no third part)
SHORT_YEAR_DATE = RegexRule(
    "Date",
    re.compile(
        r"(?<![\d.\/\-])(?:0?[1-9]|[12]\d|3[01])([\/\-.])(?:0?[1-9]|1[0-2])\1\d{2}(?![\d.\/\-]?\d)"
    ),
)

# ─────────────────────────────────────────────────────────────────────────────
# Labelled account identifiers
# ─────────────────────────────────────────────────────────────────────────────

# User / portal ID after its English label; the value (group 1) is what is hidden.
#   catches:  "User ID: rameshk", "Portal ID - HR12345", "Username=amit_92"
#   ignores:  "User ID of the complainant" — a digit-free value counts only when
#             an explicit ':' or '=' marks it as the field's value; otherwise any
#             word after "User ID" would be hashed.
USER_ID = RegexRule(
    "User_ID",
    re.compile(
        r"(?i:User\s*ID(?:\s*\([^)]+\))?|UserId|Username|User_ID|Portal\s*ID)"
        r"[_\-:=\s]+([a-zA-Z0-9_\-]+)"
    ),
    group=1,
    source="Contextual Regex",
    accept=lambda m, line: bool(re.search(r"\d", m.group(1)) or re.search(r"[:=]", m.group(0))),
)

# Hospital / patient record number after its label.
#   catches:  "UHID: MRD-20391", "Patient ID 4471"
#   ignores:  "Health ID card issued" — the value must contain a digit.
PATIENT_ID = RegexRule(
    "Patient_ID",
    re.compile(
        r"(?i:Patient\s*ID|UHID|Medical\s*UHID|Health\s*ID)[_\-:\s.]+"
        r"((?=[A-Za-z0-9_\-]*\d)[A-Za-z0-9_\-]+)"
    ),
    group=1,
    source="Contextual Regex",
)

# ─────────────────────────────────────────────────────────────────────────────
# Personal particulars
# ─────────────────────────────────────────────────────────────────────────────

# Age after "age"/"aged". Detected so it shows in the audit, but the redaction
# policy RETAINS it (an age alone does not identify anyone).
#   catches:  "aged 45 years", "age about 60 Years"
AGE = RegexRule(
    "Age",
    re.compile(r"\baged?\s+(?:about\s+)?(\d{1,3}\s*[Yy]ears?)\b"),
    group=1,
)

# ─────────────────────────────────────────────────────────────────────────────
# Deny list — names from the dataset config that must always be hidden
# ─────────────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=32)
def _deny_term_pattern(terms: Tuple[str, ...]) -> Optional["re.Pattern[str]"]:
    """One regex for all `terms`: any case, any run of spaces between words.

    "SURAJ PAL" therefore also matches "suraj  pal". None when the list is empty.
    """
    parts = [r"\s+".join(re.escape(word) for word in term.split()) for term in terms if term.strip()]
    return re.compile("|".join(parts), re.I) if parts else None


class DenyListRule:
    """Always tag the configured names (`ner_deny_terms`) as PERSON.

    For names a reviewer confirmed the models miss (issues #26, #43, #44, #51,
    #52: some scored below the confidence cutoff, some were not found at all).
    The names live in the dataset config next to the data, never in code, so
    real people's names stay out of the code; examples here are made up.

    Whole words only, judged with is_word_char so Hindi vowel signs count as part
    of a word: "Naresh" is not matched inside "Nareshpur".
    """

    label = "PERSON"
    source = "Deny list"

    def __init__(self, terms: Tuple[str, ...]):
        self.pattern = _deny_term_pattern(tuple(terms))

    def find(self, line: str) -> Iterator[Tuple[int, int]]:
        if self.pattern is None:
            return
        for m in self.pattern.finditer(line):
            start, end = m.span()
            glued_before = start > 0 and is_word_char(line[start - 1])
            glued_after = end < len(line) and is_word_char(line[end])
            if not (glued_before or glued_after):
                yield start, end
