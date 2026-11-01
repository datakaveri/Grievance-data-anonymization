"""Haryana-specific fixes, all in one place.

Everything here came from reviewing the anonymized output of the Haryana CM
Window / Jan Samvaad grievance export (config/grievance/, reviewer sheet
data/ner_issues/ner_issues.xlsx; "#NN" = issue number in that sheet). Rules that
help any Indian dataset were moved to rules/india.py; what remains here only
makes sense for Haryana data:

  1. PPP_ID                     — the Parivar Pehchan Patra (family ID) rule.
  2. REFERENCE_KEYWORDS         — record words used in Haryana land/plot records.
  3. NER_ALLOW_TERMS            — words the models mistake for names in this export,
                                  exempted by name (hardcoded, temporary).

Related Haryana settings that live in the dataset configs, not in code
(config/grievance/*.json):
  * ner_allow_terms — local terms that are never personal: ATR, Jan Samvaad, CM
    Window, Head Office, ZP, हरियाणा सरकार, Haryana Government, हरियाणा पुलिस.
  * ner_deny_terms  — names a reviewer confirmed the models miss. Real names, so
    they stay in config next to the data and out of the code.

Fixes prompted by this export but general enough to live elsewhere:
  * HTML-escaped text ("&#039;", #35) and legacy-font Hindi ("शिक्ष़्ाा", #33):
    anonymizer/cleanup.py.
  * Grievance numbers such as "CMOFF/J/2026/001234" (#28): rules/india.py
    SLASH_REFERENCE.
"""

from __future__ import annotations

import re
from typing import Tuple

from .base import NO_ALNUM_AFTER, RegexRule
from .india import NUM_MARKER

# ─────────────────────────────────────────────────────────────────────────────
# 1. Parivar Pehchan Patra (PPP) — Haryana's family ID
# ─────────────────────────────────────────────────────────────────────────────
# Every Haryana family has an 8-character PPP / Family ID ("6ABC1234",
# "GHIJ3456"). This rule finds it after its label, in English, Hindi and the
# misspellings seen in the data:
#   "PPP ID", "Family ID", "FAIMILY ID" (#36), "Parivar Pehchan Patra",
#   "परिवार पहचान पत्र" (#27), "फैमिली/फॅमिली/फेमिली आई.डी./आईडी" (#27, #37, #49).
# The value is 6-12 letters/digits and MUST contain a digit: the original rule
# was fully case-insensitive, so its value class matched any word and
# "This ppp id belongs to ..." hashed the word "belongs" (#32; "PPP detail" in
# #34 the same). With a digit required, any case is safe ("ppp id 6abc1234").
# Unlabelled PPP IDs are still hashed, as ID_Number, by india.CATCH_ALL_ID.
#   catches:  "परिवार पहचान पत्र 6ABC1234", "फॅमिली आई.डी. 2DEF2345",
#             "फॅमिली आई डी नंबर GHIJ3456", "FAIMILY ID 5KLM4567",
#             "फैमिली आई डी.न0 5NP56789", "Family ID (7QRS5678)"
#   ignores:  "PPP id belongs to", "PPP detail verification"
PPP_ID = RegexRule(
    "PPP_ID",
    re.compile(
        r"(?i:PPP(?:\s*ID)?|Fa(?:i|mi)?mily\s*I\.?\s?D|Parivar\s*Pehchan\s*Patra"
        r"|परिवार\s*पहचान\s*पत्र|(?:फैमिली|फॅमिली|फेमिली|फ़ैमिली)\s*(?:आई\.?\s*डी|आईडी))"
        rf"\.?(?:\s*{NUM_MARKER})?[\s:\-.()#]*"
        rf"((?=[A-Za-z0-9]*\d)[A-Za-z0-9]{{6,12}}){NO_ALNUM_AFTER}"
    ),
    group=1,
    source="Contextual Regex",
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Haryana record words for the reference-number rule
# ─────────────────────────────────────────────────────────────────────────────
# Added to india.REFERENCE_KEYWORDS (weak labels: a "No." marker must follow).
#   "killa" / "किला" — killa number, the Haryana/Punjab land-parcel unit
#                      ("किला नं. 14"). As a plain Hindi word किला means "fort",
#                      which is harmless here because a marker must follow.
#   "DSS"            — HSVP plot category: "DSS No. 134, Sector 25, Panipat"
#                      (#39 — plot numbers identify a property, so they are hidden).
REFERENCE_KEYWORDS: Tuple[str, ...] = (r"killa", r"किला", r"DSS")
STRONG_REFERENCE_KEYWORDS: Tuple[str, ...] = ()

# ─────────────────────────────────────────────────────────────────────────────
# 3. Hardcoded NER exemptions (temporary)
# ─────────────────────────────────────────────────────────────────────────────
# Everyday words the models tag as names in this export that the general
# false-positive rules (ner/filters.py) deliberately do NOT clear, exempted by
# name until a general rule covers them. Matched like ner_allow_terms: the whole
# entity, ignoring case/spacing/spelling variants, in every dataset.
#   "Milk"            — LOCATION in "the said Milk Booth" (#4). The vocabulary
#                       rule never clears a LOCATION, because localities are often
#                       plain English ("Model Town", "Civil Lines").
#   "Mark"            — PERSON for the verb in "Complaint is Mark to CIA-3" (#17).
#                       The vocabulary rule never clears a capitalised PERSON,
#                       because Indian names are often English words ("Lucky").
#   "kindly undetake" — PERSON (#14); "undetake" is a typo no dictionary knows.
# Cost: a person or place called just "Mark" or "Milk" is no longer redacted.
# Remove an entry once a general rule handles it.
NER_ALLOW_TERMS: Tuple[str, ...] = ("Milk", "Mark", "kindly undetake")
