"""Detection rules and word lists that need Indian context.

Indian government identifiers (Aadhaar, PAN, EPIC, DL, GSTIN, ABHA...), Indian
phone numbering (+91, STD codes), Hindi month names and number markers,
reference numbers of Indian police / land / utility records, and the Hindi and
Indian-English words the NER false-positive filters need.

Things that only make sense for Haryana data (the Parivar Pehchan Patra family
ID, Haryana-only record words, exemptions observed in the Haryana grievance
export) are in rules/haryana.py.

Every rule documents what it matches, an example it catches, what it leaves
alone and why. "#NN" refers to the issue number in the reviewer's sheet
(data/ner_issues/ner_issues.xlsx) that motivated a rule or a fix.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterator, List, Tuple

from .base import NO_ALNUM_AFTER, NO_ALNUM_BEFORE, RegexRule

# ─────────────────────────────────────────────────────────────────────────────
# Shared pieces
# ─────────────────────────────────────────────────────────────────────────────

# Devanagari digits read as ASCII digits before any rule runs, so every numeric
# pattern also catches "मोबाइल ९८७६५४३२१०" and "दिनांक १२.०३.२०२५". It is a
# one-character-for-one translation, so offsets into the line stay valid.
DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

# A "number" marker between a label and its value, English and Hindi, any case:
#   "No.", "Nos", "Number", "#", "नंबर"/"नम्बर", "नं."/"नं0", "न०", "संख्या",
#   "क्रमांक", "सं."/"सं०".
# The ०/0/o variants cover "नं0" written with a Devanagari zero, a Latin zero or
# the letter o (Devanagari digits are already translated by the time rules run,
# but the alternatives are harmless). "नं[०0o]?" is listed before "नं" would
# match alone, so "मकान नं0 53" yields 53, not "0" (found on real data).
NUM_MARKER = r"(?i:No|Nos|Number|नंबर|नम्बर|नं[०0o]?|न[०0o]|संख्या|क्रमांक|सं[०0o]?|#)\.?"

# ─────────────────────────────────────────────────────────────────────────────
# Government identity documents
# ─────────────────────────────────────────────────────────────────────────────

# PAN: 5 capitals, 4 digits, 1 capital.
#   catches:  "ABCDE1234F"
PAN = RegexRule("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"))

# Two-letter state / union territory codes that start driving licences and
# vehicle registrations, current and legacy (OR → OD, UA → UK, AP → TS → TG,
# DN + DD merged as DD), so older documents still match.
# Without this list "any 2 capitals + digits" also matched an abbreviation or a
# label in front of a number: "MD 8800012345" (a phone number) and
# "Property ID 600012345" were read as driving licences, and the label was
# hashed along with the number.
STATE_CODE = (
    r"(?:AN|AP|AR|AS|BR|CG|CH|DD|DL|DN|GA|GJ|HP|HR|JH|JK|KA|KL|LA|LD|MH|ML|MN|MP|MZ"
    r"|NL|OD|OR|PB|PY|RJ|SK|TG|TN|TR|TS|UA|UK|UP|WB)"
)

# Driving licence: state code, RTO code (2-4 digits), then 6-11 digits
# (year + serial), optional space/dash between parts.
#   catches:  "HR-0620110012345", "DL04 20150123456"
#   ignores:  "MD 8800012345", "ID 600012345" (not state codes)
DRIVING_LICENSE = RegexRule(
    "Driving_License",
    re.compile(rf"\b{STATE_CODE}[\-\s]?\d{{2,4}}[\-\s]?\d{{6,11}}\b"),
)

# Vehicle registration: state code, RTO number, 1-2 series letters, 4 digits,
# spaces/dashes optional, any case.
#   catches:  "HR26AB1234", "hr 26 ab 1234", "HR-26-AB-1234"
#   ignores:  Delhi-style series with a letter in the RTO part ("DL-3C-AB-1234"),
#             a non-state prefix ("XY 12 AB 1234").
VEHICLE_NUMBER = RegexRule(
    "Vehicle_Number",
    re.compile(rf"\b{STATE_CODE}[\s\-]?\d{{2}}[\s\-]?[A-Z]{{1,2}}[\s\-]?\d{{4}}\b", re.I),
)

# Voter ID (EPIC): 3 capitals + 7 digits, or the old state format SS/NN/NNN/NNNNNN.
#   catches:  "ABC1234567", "HR/05/123/456789"
VOTER_ID = RegexRule(
    "Voter_ID",
    re.compile(r"\b(?:[A-Z]{3}\d{7}|[A-Z]{2}\/\d{2}\/\d{3}\/\d{6})\b"),
)

# Passport: one series letter (Q, X and Z are not issued) followed by 7 digits,
# the first non-zero — 8 characters in all.
#   catches:  "J8369854", "K1234567"
#   note:     an earlier pattern required 8 digits (9 characters), so real
#             passport numbers were only hashed by the catch-all ID rule, under
#             the label ID_Number. The redacted text is the same either way.
PASSPORT = RegexRule("Passport", re.compile(r"\b[A-PR-WY][1-9]\d{6}\b"))

# ─────────────────────────────────────────────────────────────────────────────
# Aadhaar
# ─────────────────────────────────────────────────────────────────────────────

# Verhoeff check-digit tables. The last digit of every Aadhaar number is a
# Verhoeff checksum of the other eleven, so an unseparated 12-digit run can be
# told apart from an account or reference number of the same length.
_VERHOEFF_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6), (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8), (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2), (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4), (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_VERHOEFF_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2), (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0), (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5), (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)


def verhoeff_valid(number: str) -> bool:
    """True when the digits of `number` pass the Verhoeff check (Aadhaar's checksum).

    Example: verhoeff_valid("234567890124") is True (a synthetic number; issue #34 had a real one).
    """
    check = 0
    for i, ch in enumerate(reversed(re.sub(r"\D", "", number))):
        check = _VERHOEFF_D[check][_VERHOEFF_P[i % 8][int(ch)]]
    return check == 0


# Aadhaar after its label (English or Hindi), any spacing, checksum not required:
# the label already says what the number is, and a typo in it is still personal.
#   catches:  "Aadhaar No 234567890124" (#34), "आधार कार्ड नं. 2345 6789 0123",
#             "UIDAI: 2345-6789-0123"
AADHAAR_LABELLED = RegexRule(
    "Aadhaar",
    re.compile(
        r"(?i:Aadhaa?r|Adhaa?r|UIDAI|आधार)(?:\s*(?i:card|कार्ड))?"
        rf"(?:\s*{NUM_MARKER})?[\s:\-.]*"
        r"(\d{4}[\s\-]?\d{4}[\s\-]?\d{4})(?!\d)"
    ),
    group=1,
    source="Contextual Regex",
)

# Aadhaar without a label: 12 digits starting 2-9, as 4+4+4 with the same
# separator twice (\1) or unbroken. Not part of a longer digit run on either side.
#   catches:  "2345 6789 0124", "2345-6789-0124", "234567890124" (checksum OK)
#   ignores:  an unbroken 12-digit run that fails the checksum — far more often an
#             account or reference number (the catch-all ID rule hashes it anyway).
AADHAAR = RegexRule(
    "Aadhaar",
    re.compile(r"(?<!\d)(?<!\d[\s\-])[2-9]\d{3}([\s\-]?)\d{4}\1\d{4}(?!\d|[\s\-]\d)"),
    accept=lambda m, line: bool(m.group(1)) or verhoeff_valid(m.group(0)),
)

# ─────────────────────────────────────────────────────────────────────────────
# Banking & payments
# ─────────────────────────────────────────────────────────────────────────────

# Bank account number (9-18 digits) after an English or Hindi account label.
#   catches:  "Account No. 6000012340" (the dot was missing from the original
#             rule, so this leaked), "A/C No. 200012300", "खाता संख्या 123456789"
#   ignores:  bare 9-18 digit runs with no label (too many other numbers look alike).
BANK_ACCOUNT = RegexRule(
    "Bank_Account",
    re.compile(
        r"(?i:Bank\s*Account(?:\s*Number|\s*No)?|SBI\s*Account|Account\s*No|Account\s*Number"
        r"|A/c(?:\s*No)?|खाता(?:\s*(?:संख्या|सं|नं|नंबर)\.?)?)"
        rf"[_\-:\s.०]*(\d{{9,18}}){NO_ALNUM_AFTER}"
    ),
    group=1,
    source="Contextual Regex",
)

# IFSC bank branch code: 4 bank letters, a literal 0, 6 branch characters.
#   catches:  "SBIN0001234"
IFSC_CODE = RegexRule("IFSC_Code", re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"))

# UPI handle: name "@" bank, with NO dot after the @ (a dot would make it an e-mail).
#   catches:  "ramesh@okaxis", "9876543210@ybl"
#   ignores:  "a.b@gmail.com" (e-mail rule handles it)
UPI_ID = RegexRule(
    "UPI_ID",
    re.compile(r"(?<![\w.\-])[A-Za-z0-9.\-_]{2,64}@[A-Za-z]{2,32}(?![\w.\-@])"),
)

# GSTIN (GST registration): 2-digit state code, the holder's PAN, entity digit, "Z", check.
#   catches:  "06AABCU9603R1ZM"
GSTIN = RegexRule(
    "GSTIN",
    re.compile(rf"{NO_ALNUM_BEFORE}\d{{2}}[A-Z]{{5}}\d{{4}}[A-Z][1-9A-Z]Z[0-9A-Z]{NO_ALNUM_AFTER}"),
)

# ABHA (Ayushman Bharat health account) number: 14 digits written 2-4-4-4.
#   catches:  "12-3456-7890-1234"
ABHA = RegexRule("ABHA", re.compile(r"(?<![\d\-])\d{2}-\d{4}-\d{4}-\d{4}(?![\d\-])"))

# ─────────────────────────────────────────────────────────────────────────────
# Phone numbers
# ─────────────────────────────────────────────────────────────────────────────

# One optional separator between digit groups: a space, a dash, or a dash with
# spaces around it ("98765 - 43210").
_PHONE_SEP = r"(?:\s?-\s?|\s)?"

# India's country code in every written form — "+91", "+ 91", "0091", "(+91)",
# "( +91 )" — followed by a separator. It is part of the match, so no "(+91)" or
# "0091" is left visible next to the redacted number.
_COUNTRY_CODE = rf"(?:\(\s*)?(?:\+\s?|00)91(?:\s*\))?{_PHONE_SEP}"

# The 10 mobile digits (first digit 6-9) in the groupings people actually write:
# unbroken, 5+5, 3+3+4 or 4+3+3. An earlier version allowed a separator between
# ANY two digits, which read "Section 91 1000 2000" as a phone number.
_MOBILE_DIGITS = (
    rf"[6-9](?:\d{{9}}|\d{{4}}{_PHONE_SEP}\d{{5}}"
    rf"|\d{{2}}{_PHONE_SEP}\d{{3}}{_PHONE_SEP}\d{{4}}|\d{{3}}{_PHONE_SEP}\d{{3}}{_PHONE_SEP}\d{{3}})"
)

# Mobile number with an optional country code, bare "91", or trunk "0".
#   catches:  "9876543210", "80000-12345" (#21), "97000 12345" (#43),
#             "94000-12345" (#51), "+91 98765 43210", "(+91) 9876543210",
#             "0091 9876543210", "91-9876543210", "987 654 3210"
#   ignores:  "Section 91 1000 2000" (not a mobile grouping), digits that are
#             part of a longer number (no digit, "+" or "(" right before).
MOBILE = RegexRule(
    "Phone_Number",
    re.compile(rf"(?<![\d+(])(?:{_COUNTRY_CODE}|91{_PHONE_SEP}|0)?{_MOBILE_DIGITS}(?!\d)"),
)

# Landline: STD code + subscriber number (the "national" part, 9-10 digits),
# after a trunk 0 or after the country code with or without the 0.
#   catches:  "0172-2521617", "01744 200123" (#43 — previously only the last six
#             digits were masked as a pincode, leaving the STD code visible),
#             "+91 172 2521617", "+91 0172 2521617", "(+91) 1744 242838"
#   ignores:  a bare "91 172 ..." (too easily an ordinary number list).
LANDLINE = RegexRule(
    "Phone_Number",
    re.compile(
        rf"(?<![\d+(])(?:{_COUNTRY_CODE}0?|0)(?P<national>[1-9]\d{{1,3}}{_PHONE_SEP}\d{{6,8}})(?!\d)"
    ),
    accept=lambda m, line: len(re.sub(r"\D", "", m.group("national"))) in (9, 10),
)

# ─────────────────────────────────────────────────────────────────────────────
# Dates with month names (English and Hindi). Numeric dates are in generic.py.
# ─────────────────────────────────────────────────────────────────────────────

_MONTHS_EN = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
    r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)
# Hindi month names, including the common alternative spellings.
_MONTHS_HI = (
    r"जनवरी|फरवरी|फ़रवरी|मार्च|अप्रैल|अप्रेल|मई|जून|जुलाई|अगस्त|सितंबर|सितम्बर"
    r"|अक्टूबर|अक्तूबर|नवंबर|नवम्बर|दिसंबर|दिसम्बर"
)
_DAY = r"(?:0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?"
# A month name that is not the start of a longer word: "mar" in "market" and
# "may" in "maybe" are refused by the letter lookahead.
_MONTH = rf"(?:(?i:{_MONTHS_EN})\.?(?![A-Za-z])|(?:{_MONTHS_HI})(?![\u0900-\u097F]))"

# Day, month name, 4-digit year.
#   catches:  "25 Feb 2025" (#42), "1 अप्रैल 2025" (#46), "03 नवंबर 2025" (#50),
#             "31 may 2016", "3rd of March, 2024"
#   ignores:  "the market may 2016 see" (no day before the month)
MONTH_NAME_DATE = RegexRule(
    "Date",
    re.compile(rf"(?<!\d){_DAY}[\s\-.,/]*(?:of\s+)?{_MONTH}[\s\-.,/]*(?:19|20)\d{{2}}(?!\d)"),
)

# Month name, day, 4-digit year (US order).
#   catches:  "Feb 25, 2025"
MONTH_FIRST_DATE = RegexRule(
    "Date",
    re.compile(rf"{NO_ALNUM_BEFORE}{_MONTH}\s*{_DAY},?\s*(?:19|20)\d{{2}}(?!\d)"),
)

# ─────────────────────────────────────────────────────────────────────────────
# Reference numbers: the number after a record label ("FIR No. 140")
# ─────────────────────────────────────────────────────────────────────────────
# A case, land, house or connection number identifies a person's record, so the
# NUMBER is hashed and the label kept: "FIR No. <hash>". Two lists of labels:
#
#   strong — always followed by their number, so the "No." marker is optional:
#            "FIR 140", "मुकदमा 61" (#22, #25).
#   weak   — also ordinary words ("case", "house"), so they need the marker
#            ("Case No. 45/2025", "मकान नं. 1919") or a number/year shape
#            ("case 45/2025"); otherwise "in case 2 people" would be hashed.
#
# Court cases (CWP No., Civil Writ Petition) and letter numbers are public
# references and are deliberately not listed. A dataset config can replace
# either list (reference_keywords / strong_reference_keywords). The defaults
# are these lists plus the Haryana additions in rules/haryana.py.

STRONG_REFERENCE_KEYWORDS: Tuple[str, ...] = (
    # Police: FIR in English/Hindi, case (मुकदमा/मुकद्दमा), prosecution (अभियोग)
    r"F\.?\s?I\.?\s?R", r"एफ\.?\s?आई\.?\s?आर", r"मुकदमा", r"मुकद्दमा", r"अभियोग",
    # Land records used across north India: field number (khasra), holding
    # (khewat), cultivator record (khatauni), mutation entry (इंतकाल)
    r"khasra", r"खसरा", r"khewat", r"खेवट", r"khat(?:a|au|o)ni", r"खतौनी", r"इंतकाल",
    # "ID" / "आईडी": "Pension ID 7000123" (#36), "id 6abc1234"
    r"I\.?\s?D", r"आई\.?\s?डी", r"आईडी",
)

REFERENCE_KEYWORDS: Tuple[str, ...] = (
    # Complaints and police diary entries: "Pariwad No 589" (#24), "DDR No. 12"
    r"case", r"complaint", r"शिकायत", r"परिवाद", r"pariwad", r"DDR", r"रपट",
    # Land and property: "mutation No.", "House No.", "H. No.", "मकान नं. 1919"
    # (#47), "Plot No. 963", "Property ID No. 1ABCD2E3" (#45)
    r"mutation", r"house", r"H", r"मकान",
    r"plot", r"प्लॉट", r"property(?:\s*ID)?", r"प्रॉपर्टी(?:\s*आईडी)?",
    # Utility connections: meter / connection / consumer numbers (#40, #41)
    r"meter", r"मीटर", r"connection", r"कनेक्शन", r"consumer", r"उपभोक्ता",
    r"account", r"खाता",
    # Registrations and service records: "पंजी.सं." (#47), service numbers (#50)
    r"registration", r"पंजीकरण", r"पंजी", r"service", r"सर्विस",
    # Welfare documents and permits
    r"ration\s*card", r"राशन\s*कार्ड", r"licen[cs]e", r"लाइसेंस", r"challan", r"चालान",
    r"PPO", r"pension", r"पेंशन", r"पेंसन", r"roll", r"रोल", r"vehicle", r"वाहन",
    r"receipt", r"रसीद", r"application", r"आवेदन",
)

# A reference value: letters, digits, "/" and "-", holding at least one digit,
# starting and ending with a letter or digit. Any case: after a label, the digit
# alone rules out ordinary words. Examples: "140", "0456/2026", "151-NT".
_REF_VALUE = r"(?=[A-Za-z0-9/\-]*\d)[A-Za-z0-9](?:[A-Za-z0-9/\-]*[A-Za-z0-9])?"

# Further numbers of a list after the first: "No. 140 & 141", "No 589, 526",
# "न0 47 से 49". Matched from the end of the previous value.
_REF_CONTINUATION = re.compile(
    rf"\s*(?:,|&|\band\b|\bto\b|व|तथा|एवं|से)\s*(?P<value>{_REF_VALUE}){NO_ALNUM_AFTER}"
)
# A list continuation that is a year ends the list: "FIR No. 140, 2025" is one FIR.
_YEAR = re.compile(r"(?:19|20)\d{2}")
_NEVER_MATCHES = re.compile(r"(?!x)x")


@lru_cache(maxsize=32)
def _reference_pattern(weak: Tuple[str, ...], strong: Tuple[str, ...]) -> "re.Pattern[str]":
    """Label + optional marker + value, with named groups strong/weak/marker/value.

    The label may not be glued to a preceding letter (Latin or Devanagari), nor
    run on into a following Latin letter ("FIR" but not "FIRM").
    """
    groups: List[str] = []
    if strong:
        groups.append("(?P<strong>" + "|".join(f"(?:{k})" for k in strong) + ")")
    if weak:
        groups.append("(?P<weak>" + "|".join(f"(?:{k})" for k in weak) + ")")
    if not groups:
        return _NEVER_MATCHES
    return re.compile(
        rf"(?<![A-Za-z\u0900-\u097F])(?i:{'|'.join(groups)})(?![A-Za-z])\.?"
        rf"(?:\s*(?P<marker>{NUM_MARKER}))?[\s:\-–#.]*"
        rf"(?P<value>{_REF_VALUE}){NO_ALNUM_AFTER}"
    )


class ReferenceNumberRule:
    """Hash the number that follows a record label (FIR, case, khasra, house...).

    catches:  "FIR NO. 140" (#22), "मुकदमा न०: 61" (#25), "FIR No. 0456/2026"
              (#48), "अभियोग संख्या 726" (#53), "FIR 140" (strong label, no
              marker), "Pariwad No 589 , 526" (both numbers), "मुकदमा नंबर 151-NT"
    ignores:  "in case 2 people" (weak label, no marker), "FIR 2 baar" (a single
              character after a strong label without marker), "Table No. 2",
              "letter no. 221" (labels not listed), the year in "FIR No. 140, 2025"
    """

    label = "Reference_Number"
    source = "Contextual Regex"

    def __init__(self, weak: Tuple[str, ...], strong: Tuple[str, ...]):
        self.pattern = _reference_pattern(tuple(weak), tuple(strong))

    def find(self, line: str) -> Iterator[Tuple[int, int]]:
        for m in self.pattern.finditer(line):
            value = m.group("value")
            if not m.group("marker") and "/" not in value:
                # No marker: a weak label needs a number/year shape ("case 45/2025"),
                # and a strong one at least two characters ("FIR 140", not "FIR 2").
                if m.groupdict().get("strong") is None or len(value) < 2:
                    continue
            yield m.span("value")
            pos = m.end("value")
            while (cont := _REF_CONTINUATION.match(line, pos)) is not None:
                if _YEAR.fullmatch(cont.group("value")):
                    break
                yield cont.span("value")
                pos = cont.end("value")


# ─────────────────────────────────────────────────────────────────────────────
# Catch-all for identifiers no named rule covers
# ─────────────────────────────────────────────────────────────────────────────
# Family, property, meter, service and registration numbers come in endless
# formats, and most appear without a label anyone could list in advance. They
# share one shape that ordinary text never has: a standalone run of capitals
# and digits that contains a digit and either a letter or 7+ digits. Words never
# contain digits; the numbers prose does contain (amounts "3,00,000", years,
# section numbers "103(1)") are short or punctuated. Hashed as ID_Number.

# Codes of that shape that identify no one: financial years and road numbers
# (National / State Highway, Major District Road).
#   "FY2025", "NH709A", "SH20", "MDR134"
_PUBLIC_CODE = re.compile(r"(?:FY|NH|SH|MDR)\d{1,4}[A-Z]?")
# Money is not an identifier: digits right after Rs / INR / ₹ / रु / रुपये, or
# right before "/-", "rupees", "lakh", "crore", "रुपये", "लाख", "करोड़".
_MONEY_BEFORE = re.compile(r"(?i:rs\.?|inr|₹|रु\.?|रू\.?|रुपये|रूपये|रुपए)\s*$")
_MONEY_AFTER = re.compile(r"^\s*(?:/-|(?i:rupees|lakh|lac|crore)\b|रुपये|रूपये|लाख|करोड़)")


def _not_public_code_or_money(m: "re.Match[str]", line: str) -> bool:
    """Accept-check shared by the two catch-all rules below."""
    value = m.group(0)
    if _PUBLIC_CODE.fullmatch(value):
        return False
    if value.isdigit() and (
        _MONEY_BEFORE.search(line[max(0, m.start() - 12) : m.start()])
        or _MONEY_AFTER.match(line[m.end() :])
    ):
        return False  # "Rs 3500000", "3500000/-"
    return True


# 6-24 capitals/digits with a digit, plus a letter or 7+ digits, standalone.
#   catches:  "6ABC1234", "GHIJ3456" (family IDs, #27/#37/#49), "RL1D0123"
#             (electricity account, #40), "1ABCD2E3" (property ID, #45),
#             "12345678Y" (army service number glued to Hindi, #50),
#             "001200012345" (registration number, #47), "7000123" (pension ID, #36)
#   ignores:  "FY2025", "NH709A" (public codes), "Rs 3500000" (money), lowercase
#             "1200mtrs"/"covid19" (units and words), "CRFBD12345678901.pdf"
#             (a system attachment name — hashing it would break the link to the
#             file; issue #15)
CATCH_ALL_ID = RegexRule(
    "ID_Number",
    re.compile(
        rf"{NO_ALNUM_BEFORE}(?=[A-Z0-9]*\d)(?=\d*[A-Z]|\d{{7}})[A-Z0-9]{{6,24}}(?![a-z0-9])"
        r"(?!\.(?i:pdf|jpe?g|png|gif|tiff?|docx?|xlsx?|csv|txt|zip)\b)"
    ),
    accept=_not_public_code_or_money,
)

# Slash-structured reference codes: 2+ capitals, then 2+ "/"-separated parts,
# containing a digit somewhere.
#   catches:  "CMOFF/J/2026/001234" (another grievance's number, #28 — before
#             this only "CMOFF" was hidden and "/J/2026/001234" stayed),
#             "DEV/2026-2027/101"
#   ignores:  "SDO/JE/XEN" (no digit), "5/12/2009-2H" (starts with a digit)
SLASH_REFERENCE = RegexRule(
    "ID_Number",
    re.compile(r"(?<![A-Za-z0-9/])(?=[A-Z0-9/\-]*\d)[A-Z]{2,}(?:/[A-Z0-9\-]+){2,}(?![A-Za-z0-9/])"),
    accept=_not_public_code_or_money,
)

# ─────────────────────────────────────────────────────────────────────────────
# Postal code
# ─────────────────────────────────────────────────────────────────────────────

# PIN code: 6 digits (first non-zero), optionally written 3+3.
#   catches:  "132001", "132 001"
#   ignores:  values starting 200/201 and any line mentioning "voter" — kept from
#             the original rule set; the intent appears to be skipping year-like
#             numbers (2001xx, 2019xx) and voter-list serials. Note it also
#             matches any other 6-digit number (amounts such as "Rs. 240000/-",
#             file numbers), which the policy then partially masks as "240XXX".
PINCODE = RegexRule(
    "Pincode",
    re.compile(r"\b[1-9][0-9]{2}\s?[0-9]{3}\b"),
    accept=lambda m, line: not m.group(0).startswith(("200", "201")) and "voter" not in line.lower(),
)

# ─────────────────────────────────────────────────────────────────────────────
# Word lists for the NER false-positive filters (anonymizer/ner/filters.py)
# ─────────────────────────────────────────────────────────────────────────────

# Forms of address that are never a name on their own. The models tag "SIR" in
# "R/SIR" (#16) and "श्रीमान" (#6) as PERSON. "r" covers the "R" of "R/Sir".
# Deliberately NOT listed, because they are also given names or mean "the
# officer": Shri, Shree, Sri, श्री, Sahab/साहब, Ji. "Shree ne ye kiya" and
# "श्री ने शिकायत की" must stay redacted.
NEVER_NAME_TITLES = frozenset(
    {"sir", "madam", "mam", "maam", "mr", "mrs", "ms", "smt", "dr", "respected", "r",
     "श्रीमान", "श्रीमती", "महोदय", "महोदया", "मान्यवर"}
)

# Hindi function words missing from spaCy's Hindi stopword list. The models tag
# "हेतू" ("for", #23) as PERSON. Common misspellings are included (हेतू/हेतु).
HINDI_FUNCTION_WORDS = frozenset({"हेतु", "हेतू", "बारे", "बाबत", "संबंध", "सम्बन्ध", "द्वारा"})
