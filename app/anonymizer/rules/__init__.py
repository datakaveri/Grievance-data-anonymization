"""The detection rules, by scope, and the order the scanner runs them in.

  rules/base.py     RegexRule, ScanOptions, boundary helpers
  rules/generic.py  e-mail, cards, IP, numeric dates, user/patient IDs, age, deny list
  rules/india.py    Aadhaar, PAN, DL, vehicle, voter ID, passport, bank, IFSC, UPI,
                    GSTIN, ABHA, phones (+91), month-name/Hindi dates, reference
                    numbers, the catch-all ID rule, PIN codes; Hindi word lists
  rules/haryana.py  Parivar Pehchan Patra family ID, Haryana record words,
                    hardcoded NER exemptions from the Haryana grievance export

build_rules(options) returns the ordered list the scanner applies to each line.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from . import generic, haryana, india
from .base import Rule, ScanOptions

# Default label lists for the reference-number rule: India-wide plus Haryana.
# A dataset config replaces them with reference_keywords / strong_reference_keywords.
DEFAULT_REFERENCE_KEYWORDS = india.REFERENCE_KEYWORDS + haryana.REFERENCE_KEYWORDS
DEFAULT_STRONG_REFERENCE_KEYWORDS = india.STRONG_REFERENCE_KEYWORDS + haryana.STRONG_REFERENCE_KEYWORDS

# Fixed rules in scanning order. ORDER MATTERS in two ways:
#  * When several detections cover the same span, the redaction step uses the
#    policy of the earliest rule in this list (rules beat model hits; see
#    redaction._policy_hit). Specific rules therefore run before broad ones: an
#    Aadhaar is masked "XXXX XXXX 0124", not hashed as ID_Number, and an
#    account number keeps its last 4 digits instead of becoming a phone "*".
#  * The NER report lists hits in this order.
# The two configurable rules (deny list, reference numbers) are inserted by
# build_rules() between _BEFORE_CONFIGURABLE and _AFTER_CONFIGURABLE.
_BEFORE_CONFIGURABLE: List[Rule] = [
    generic.EMAIL,
    generic.CREDIT_CARD,
    india.DRIVING_LICENSE,
    india.PAN,
    india.VEHICLE_NUMBER,
    india.VOTER_ID,
    india.PASSPORT,
    generic.USER_ID,
    haryana.PPP_ID,
    generic.PATIENT_ID,
    india.BANK_ACCOUNT,
    india.AADHAAR_LABELLED,
    india.AADHAAR,
    india.IFSC_CODE,
    generic.IP_ADDRESS,
    india.MOBILE,
    india.LANDLINE,
    generic.NUMERIC_DATE,
    generic.SHORT_YEAR_DATE,
    india.MONTH_NAME_DATE,
    india.MONTH_FIRST_DATE,
    india.GSTIN,
    india.ABHA,
    india.UPI_ID,
]
# Broad rules last, so every specific rule above has already claimed its span.
_AFTER_CONFIGURABLE: List[Rule] = [
    india.CATCH_ALL_ID,
    india.SLASH_REFERENCE,
    generic.AGE,
    india.PINCODE,
]


@lru_cache(maxsize=32)
def build_rules(options: ScanOptions = ScanOptions()) -> List[Rule]:
    """All rules for one dataset, in scanning order.

    The deny-list rule is added only when the config names terms; the
    reference-number rule always runs, with the config's label lists or the
    defaults above. Cached per options, so a job builds its rules once.
    """
    configurable: List[Rule] = []
    if options.deny_terms:
        configurable.append(generic.DenyListRule(options.deny_terms))
    configurable.append(
        india.ReferenceNumberRule(
            DEFAULT_REFERENCE_KEYWORDS if options.reference_keywords is None else options.reference_keywords,
            DEFAULT_STRONG_REFERENCE_KEYWORDS
            if options.strong_reference_keywords is None
            else options.strong_reference_keywords,
        )
    )
    return _BEFORE_CONFIGURABLE + configurable + _AFTER_CONFIGURABLE


__all__ = [
    "ScanOptions",
    "Rule",
    "build_rules",
    "DEFAULT_REFERENCE_KEYWORDS",
    "DEFAULT_STRONG_REFERENCE_KEYWORDS",
]
