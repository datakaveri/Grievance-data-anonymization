"""The examples documented next to each regex rule, as executable checks.

Run from the repository root:  python -m unittest discover -s tests
No models are loaded; these tests take a few seconds.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from anonymizer.rules import ScanOptions  # noqa: E402
from anonymizer.scanner import scan_text  # noqa: E402


def found(text, label, options=None):
    """Values of `label` the scanner finds in `text`."""
    return [h.value for h in scan_text(text, options) if h.label == label]


class RuleTest(unittest.TestCase):
    """assertCatches / assertIgnores over (text, label[, value]) examples."""

    def assertCatches(self, label, cases, options=None):
        for text, value in cases:
            with self.subTest(text=text):
                self.assertIn(value, found(text, label, options))

    def assertIgnores(self, label, texts, options=None):
        for text in texts:
            with self.subTest(text=text):
                self.assertEqual(found(text, label, options), [])


class GenericRules(RuleTest):
    def test_email(self):
        self.assertCatches("Email", [("mail ramesh.k@gmail.com now", "ramesh.k@gmail.com"),
                                     ("info@hsvp.gov.in", "info@hsvp.gov.in")])
        self.assertIgnores("Email", ["pay ramesh@okaxis"])

    def test_ip_and_card(self):
        self.assertCatches("IP_Address", [("IP 10.12.20.1 here", "10.12.20.1")])
        self.assertCatches("Credit_Card", [("card 4111 1111 1111 1111", "4111 1111 1111 1111"),
                                           ("5500-0000-0000-0004", "5500-0000-0000-0004")])

    def test_numeric_dates(self):
        self.assertCatches("Date", [
            ("on 12.12.2025", "12.12.2025"), ("09/04/2026 12:01 PM", "09/04/2026"),
            ("दिनांक12.12.2025 धारा", "12.12.2025"), ("SP Sir on DT 25.03.26 Complaint", "25.03.26"),
            ("25-3-2026", "25-3-2026"), ("on 1/4/25", "1/4/25"),
        ])
        self.assertIgnores("Date", ["IP 10.12.20.1 here", "version 1.2.3"])

    def test_labelled_account_ids(self):
        self.assertCatches("User_ID", [("User ID: rameshk", "rameshk"), ("Portal ID - HR12345", "HR12345"),
                                       ("Username=amit_92", "amit_92")])
        self.assertIgnores("User_ID", ["the User ID of the complainant"])
        self.assertCatches("Patient_ID", [("UHID: MRD-20391", "MRD-20391"), ("Patient ID 4471", "4471")])
        self.assertIgnores("Patient_ID", ["Health ID card issued"])

    def test_age(self):
        self.assertCatches("Age", [("aged 45 years", "45 years"), ("age about 60 Years", "60 Years")])

    def test_deny_list(self):
        options = ScanOptions(deny_terms=("Naresh", "SURAJ PAL"))
        self.assertCatches("PERSON", [("sh Naresh CA", "Naresh"), ("si suraj  pal 675", "suraj  pal")], options)
        self.assertIgnores("PERSON", ["Nareshpur village"], options)
        self.assertIgnores("PERSON", ["sh Naresh CA"])  # no deny list configured


class IndiaRules(RuleTest):
    def test_identity_documents(self):
        self.assertCatches("PAN", [("PAN ABCDE1234F", "ABCDE1234F")])
        self.assertCatches("Driving_License", [("DL HR-0620110012345", "HR-0620110012345"),
                                               ("DL04 20150123456", "DL04 20150123456")])
        self.assertIgnores("Driving_License", ["PAN ABCDE1234F", "HR26AB1234", "IC PP MD 8800012345",
                                               "Property ID 600012345"])
        self.assertCatches("Phone_Number", [("IC PP MD 8800012345", "8800012345")])
        self.assertCatches("Vehicle_Number", [("HR26AB1234", "HR26AB1234"), ("hr 26 ab 1234", "hr 26 ab 1234"),
                                              ("HR-26-AB-1234", "HR-26-AB-1234"), ("CH04A1234", "CH04A1234")])
        self.assertIgnores("Vehicle_Number", ["DL-3C-AB-1234", "XY 12 AB 1234"])
        self.assertCatches("Voter_ID", [("EPIC ABC1234567", "ABC1234567"), ("HR/05/123/456789", "HR/05/123/456789")])
        self.assertCatches("Passport", [("passport J8369854", "J8369854"), ("K1234567", "K1234567")])

    def test_aadhaar(self):
        self.assertCatches("Aadhaar", [
            ("Aadhaar No 234567890124 not received", "234567890124"),  # labelled, #34
            ("2345 6789 0124", "2345 6789 0124"),                       # separated
            ("234567890124", "234567890124"),                           # passes Verhoeff
            ("आधार कार्ड नं. 2345 6789 0123", "2345 6789 0123"),
            ("UIDAI: 2345-6789-0123", "2345-6789-0123"),
            ("2345-6789-0124", "2345-6789-0124"),
        ])
        self.assertIgnores("Aadhaar", ["ref 234567890120"])  # fails the checksum, no label

    def test_bank_and_payments(self):
        self.assertCatches("Bank_Account", [
            ("vide account no. 6000012340. After", "6000012340"),
            ("A/C No. 200012300 उपभोगता", "200012300"),
            ("खाता संख्या 123456789", "123456789"),
        ])
        self.assertCatches("IFSC_Code", [("IFSC SBIN0001234", "SBIN0001234")])
        self.assertCatches("UPI_ID", [("UPI ramesh@okaxis", "ramesh@okaxis"), ("9876543210@ybl", "9876543210@ybl")])
        self.assertIgnores("UPI_ID", ["mail a.b@gmail.com"])
        self.assertCatches("GSTIN", [("GSTIN 06AABCU9603R1ZM", "06AABCU9603R1ZM")])
        self.assertCatches("ABHA", [("ABHA 12-3456-7890-1234", "12-3456-7890-1234")])

    def test_mobile_numbers(self):
        self.assertCatches("Phone_Number", [
            ("मो. नंबर 80000-12345 पुत्र", "80000-12345"),
            ("मो0-97000 12345", "97000 12345"),
            ("MOB. 94000-12345", "94000-12345"),
            ("+91 98765 43210", "+91 98765 43210"),
            ("(+91) 9876543210", "(+91) 9876543210"),
            ("0091 9876543210", "0091 9876543210"),
            ("91-9876543210", "91-9876543210"),
            ("987 654 3210", "987 654 3210"),
            ("98765 - 43210", "98765 - 43210"),
            ("मोबाइल ९८७६५४३२१० पर", "9876543210"),  # Devanagari digits
        ])
        self.assertIgnores("Phone_Number", ["Section 91 1000 2000 300"])

    def test_landlines(self):
        self.assertCatches("Phone_Number", [
            ("0172-2521617", "0172-2521617"), ("मो0-01744 200123", "01744 200123"),
            ("+91 172 2521617", "+91 172 2521617"), ("+91 0172 2521617", "+91 0172 2521617"),
            ("(+91) 1744 242838", "(+91) 1744 242838"),
        ])

    def test_month_name_dates(self):
        self.assertCatches("Date", [
            ("सील 25 Feb 2025 को", "25 Feb 2025"), ("1 अप्रैल 2025 से", "1 अप्रैल 2025"),
            ("(03 नवंबर 2025 को", "03 नवंबर 2025"), ("Dated 31 may 2016 and", "31 may 2016"),
            ("on Feb 25, 2025", "Feb 25, 2025"), ("3rd of March, 2024", "3rd of March, 2024"),
        ])
        self.assertIgnores("Date", ["the market may 2016 see"])

    def test_reference_numbers(self):
        self.assertCatches("Reference_Number", [
            ("FIR NO. 140 बाबत", "140"), ("मुकदमा न०: 61 दिनांक", "61"), ("Police FIR No. 0456/2026", "0456/2026"),
            ("अभियोग संख्या 726 दिनांक", "726"), ("FIR 140 dated", "140"), ("मुकदमा 61 थाना", "61"),
            ("मकान नं. 1919,", "1919"), ("निवासी मकान नं0 53/बी गली", "53"), ("मुकदमा नंबर 151-NT ब", "151-NT"),
            ("Pension ID 7000123", "7000123"), ("id 6abc1234 hai", "6abc1234"),
            ("DDR No. 12 dated", "12"), ("H. No. 455, Sector 7", "455"), ("Plot No. 963 Sec-19", "963"),
            ("khasra No. 6772 situated", "6772"), ("case 45/2025 pending", "45/2025"),
        ])
        self.assertEqual(found("Pariwad No 589 , 526.", "Reference_Number"), ["589", "526"])
        self.assertEqual(found("FIR No. 140 & 141 dated", "Reference_Number"), ["140", "141"])
        self.assertEqual(found("FIR No. 140, 2025 me", "Reference_Number"), ["140"])
        self.assertIgnores("Reference_Number", [
            "in case 2 people came", "FIR 2 baar", "Table No. 2 of the Guidelines", "letter No. 249 dated",
            "CWP No. 30725 of 2019", "ID card banwana hai",
        ])

    def test_reference_keywords_from_config(self):
        options = ScanOptions(reference_keywords=("letter",), strong_reference_keywords=())
        self.assertEqual(found("letter No. 249 dated", "Reference_Number", options), ["249"])
        self.assertEqual(found("FIR 140 dated", "Reference_Number", options), [])

    def test_catch_all_ids(self):
        self.assertCatches("ID_Number", [
            ("परिवार पहचान पत्र 6ABC1234 में", "6ABC1234"), ("मीटर (RL1D0123) का", "RL1D0123"),
            ("Property ID No. 1ABCD2E3 falls", "1ABCD2E3"), ("नं0 12345678Yहवलदार", "12345678Y"),
            ("एच.आर.001200012345) के", "001200012345"), ("pension ID 7000123", "7000123"),
            ("DEV/2026-2027/101 dated", "DEV/2026-2027/101"),
            ("Already Exists as CMOFF/J/2026/001234 dated", "CMOFF/J/2026/001234"),
        ])
        self.assertIgnores("ID_Number", [
            "FY2025 budget", "NH709A road", "Rs 3500000 sanctioned", "3500000/- ki rashi",
            "length of 1200mtrs road", "covid19 duty", "CRFBD12345678901.pdf", "SDO/JE/XEN",
        ])

    def test_pincode(self):
        self.assertCatches("Pincode", [("Kurukshetra 136118", "136118"), ("132 001", "132 001")])
        self.assertIgnores("Pincode", ["Voter list 136118"])


class HaryanaRules(RuleTest):
    def test_ppp_family_id(self):
        self.assertCatches("PPP_ID", [
            ("परिवार पहचान पत्र 6ABC1234 में", "6ABC1234"), ("फॅमिली आई.डी. 2DEF2345 में", "2DEF2345"),
            ("फॅमिली आई डी नंबर GHIJ3456 में", "GHIJ3456"), ("FAIMILY ID 5KLM4567 में", "5KLM4567"),
            ("फैमिली आई डी.न0 5NP56789 मे", "5NP56789"), ("1. Family ID (7QRS5678) में", "7QRS5678"),
            ("ppp id 6abc1234 me", "6abc1234"),
        ])
        self.assertIgnores("PPP_ID", ["This ppp id belongs to Yamunanagar District", "sent for PPP detail verification"])

    def test_haryana_record_words(self):
        self.assertEqual(found("Restoration of DSS No. 134, Sector 25", "Reference_Number"), ["134"])
        self.assertEqual(found("किला नं. 14 में", "Reference_Number"), ["14"])


class ScannerBehaviour(unittest.TestCase):
    def test_repeated_value_is_found_every_time(self):
        hits = [h for h in scan_text("Call 9876543210 or 9876543210 again") if h.label == "Phone_Number"]
        self.assertEqual([(h.start_char, h.end_char) for h in hits], [(6, 15), (20, 29)])

    def test_positions_are_one_based_in_stripped_line(self):
        hit = next(h for h in scan_text("  Call 9876543210") if h.label == "Phone_Number")
        self.assertEqual((hit.line_no, hit.word_no, hit.start_char, hit.end_char), (1, 2, 6, 15))


if __name__ == "__main__":
    unittest.main()
