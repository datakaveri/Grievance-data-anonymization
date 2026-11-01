"""Checks for cleanup, span merging, the NER false-positive filters, redaction and config.

Run from the repository root:  python -m unittest discover -s tests
No models are loaded: model output is simulated with hand-written spans.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from anonymizer.cleanup import normalize_source_text  # noqa: E402
from anonymizer.config import BatchError, TableJobSettings  # noqa: E402
from anonymizer.ner.filters import false_positive_reason, ner_term_key, word_casing_stats  # noqa: E402
from anonymizer.ner.spans import NerEntity, merge_line_spans  # noqa: E402
from anonymizer.redaction import redact_text, replacement_for  # noqa: E402
from anonymizer.scanner import scan_cell  # noqa: E402
from anonymizer.text import detect_language, is_word_char, line_bases  # noqa: E402
from anonymizer import pipeline  # noqa: E402
from anonymizer.config import DetectionSettings  # noqa: E402
from unittest import mock  # noqa: E402

XLM, INDIC = "XLM-RoBERTa (Multilingual)", "IndicNER (AI4Bharat / Public)"


def span(line, cat, text, score=0.9, model=XLM):
    """A raw model span covering `text` in `line`."""
    start = line.index(text)
    return {"cat": cat, "start": start, "end": start + len(text), "score": score, "model": model}


def merged(line, *spans):
    """(category, text) of merge_line_spans over the given spans."""
    return [(e.category, e.text) for e in merge_line_spans(list(spans), line)]


class Cleanup(unittest.TestCase):
    def test_html_entities(self):
        self.assertEqual(normalize_source_text("Government&#039;&#039;s Guidelines &amp; rules"),
                         "Government''s Guidelines & rules")

    def test_devanagari_repairs(self):
        self.assertEqual(normalize_source_text("शिक्ष़्ाा विभाग"), "शिक्षा विभाग")
        self.assertEqual(normalize_source_text("श्रेण्ी मे"), "श्रेणी मे")

    def test_valid_text_unchanged(self):
        for text in ("गाँव पड़ोसी ज़मीन फ़ाइल क्षेत्र", "सामान्य हिंदी पाठ", "plain text", "", None, 5):
            self.assertEqual(normalize_source_text(text), text)


class TextHelpers(unittest.TestCase):
    def test_vowel_signs_are_word_characters(self):
        self.assertTrue(all(is_word_char(ch) for ch in "उर्फ"))
        self.assertFalse(is_word_char(" "))

    def test_line_bases_and_language(self):
        self.assertEqual(line_bases("a\n  bc"), [0, 4])
        self.assertEqual(detect_language("रमेश कुमार"), "Hindi (Devanagari)")
        self.assertEqual(detect_language("shikayat ki"), "English")


class SpanMerging(unittest.TestCase):
    def test_partial_hindi_word_grows_to_whole_word(self):
        line = "1. सुरेन्द्र उर्फ सोनू पुत्र रामपाल"
        self.assertEqual(merged(line, span(line, "PERSON", "सुरेन्द्र उर")), [("PERSON", "सुरेन्द्र उर्फ")])

    def test_dotted_abbreviation_kept_whole(self):
        line = "की तफ्तीश C.I.A नारनौल से"
        self.assertEqual(merged(line, span(line, "ORGANIZATION", "A नारनौल")), [("ORGANIZATION", "C.I.A नारनौल")])

    def test_relation_prefixes(self):
        for line, text, expected in [
            ("Kumar S/o Ram Lal", "S/o Ram Lal", "Ram Lal"),
            ("Ramesh s.o.Suresh Kumar", "s.o.Suresh Kumar", "Suresh Kumar"),
            ("R/Sir, see the attached", "R/Sir", "R/Sir"),  # not cut to "ir"
        ]:
            with self.subTest(line=line):
                self.assertEqual(merged(line, span(line, "PERSON", text)), [("PERSON", expected)])

    def test_same_type_spans_join(self):
        line = "complaint by ramesh verma nagar resident"
        self.assertEqual(
            merged(line, span(line, "PERSON", "ramesh verma"), span(line, "PERSON", "ramesh", model=INDIC)),
            [("PERSON", "ramesh verma")],
        )

    def test_different_types_keep_most_confident_and_cover_union(self):
        line = "sh Mahesh CA"
        result = merge_line_spans(
            [span(line, "ORGANIZATION", "sh Mah", 0.457, INDIC), span(line, "PERSON", "Mah", 0.90)], line
        )
        self.assertEqual([(e.category, e.text, e.score) for e in result], [("PERSON", "sh Mahesh", 0.9)])
        self.assertEqual(result[0].models, sorted([INDIC, XLM]))


class FalsePositiveFilters(unittest.TestCase):
    def entity(self, text, category="PERSON", models=(XLM,)):
        return NerEntity(category=category, text=text, score=0.9, models=list(models))

    def test_allow_list_key_folds_spelling_variants(self):
        keys = {ner_term_key(t) for t in ("Jan Samvaad", "Jansamvaad", "jan samwad", "Jan Sanwaad")}
        self.assertEqual(len(keys), 1)
        self.assertEqual(ner_term_key("सी.एम.विंडो"), ner_term_key("सी.एम.विन्ङो"))

    def test_allow_list_matches_whole_entity_only(self):
        allow = frozenset({ner_term_key("Jan Samvaad")})
        self.assertEqual(false_positive_reason(self.entity("Jan Samvaad"), "x", allow), "allow_list")
        self.assertIsNone(false_positive_reason(self.entity("Jan Samvaad Ramesh"), "x", allow))

    def test_hardcoded_haryana_exemptions(self):
        self.assertEqual(false_positive_reason(self.entity("Milk", "LOCATION"), "x"), "hardcoded_allow")

    def test_titles_but_not_names_that_look_like_titles(self):
        self.assertEqual(false_positive_reason(self.entity("R/SIR"), "x"), "title_only")
        self.assertEqual(false_positive_reason(self.entity("श्रीमान"), "x"), "title_only")
        for name in ("Shree", "श्री", "साहब"):
            self.assertIsNone(false_positive_reason(self.entity(name), "Shree ne ye kiya"))

    def test_function_words(self):
        self.assertEqual(false_positive_reason(self.entity("हेतू"), "शिकायत हेतू"), "function_word")
        self.assertIsNone(false_positive_reason(self.entity("May"), "May Kumari resident"))
        self.assertIsNone(false_positive_reason(self.entity("हेतू", models=(XLM, INDIC)), "शिकायत हेतू"))

    def test_vocabulary_rule_uses_dataset_casing(self):
        corpus = ["Please submit the complainant report. Sent by complainant today."] * 3 + \
                 ["Met Ram Singh today. Called Ram Singh again."] * 3
        casing = word_casing_stats(corpus)
        line = "Upload ATR signed by complainant."
        self.assertEqual(false_positive_reason(self.entity("complainant"), line, casing=casing), "lowercase_common_word")
        self.assertIsNone(false_positive_reason(self.entity("Ram Singh"), line, casing=casing))
        self.assertIsNone(false_positive_reason(self.entity("complainant", "LOCATION"), line, casing=casing))


def redact(text, cutoff, column, salts, model_hits):
    """Steps 2 + 4 for one cell, as pipeline.anonymize runs them."""
    return redact_text(text, scan_cell(text), model_hits, column, salts, minimum_confidence=cutoff)


class Redaction(unittest.TestCase):
    def test_policies(self):
        salts = {"c": "salt"}
        self.assertEqual(replacement_for("Aadhaar", "2345 6789 0124", "c", salts), ("XXXX XXXX 0124", "partial_mask"))
        self.assertEqual(replacement_for("PAN", "ABCDE1234F", "c", salts), ("ABCDE****F", "partial_mask"))
        self.assertEqual(replacement_for("Phone_Number", "98765 43210", "c", salts), ("*", "suppress"))
        self.assertEqual(replacement_for("Email", "ramesh@gmail.com", "c", salts)[0], "ra****@gmail.com")
        self.assertEqual(replacement_for("Date", "12.12.2025", "c", salts), ("XX-XX-2025", "date_generalization"))
        self.assertEqual(replacement_for("Pincode", "132001", "c", salts), ("132XXX", "partial_mask"))
        self.assertEqual(replacement_for("Age", "45 years", "c", salts), ("45 years", "retained"))
        hashed, technique = replacement_for("PPP_ID", "6ABC1234", "c", salts)
        self.assertEqual((len(hashed), technique), (64, "salted_hash"))
        self.assertEqual(hashed, replacement_for("PPP_ID", "6ABC1234", "c", salts)[0])  # same salt, same hash
        # Exact labels: "Company_ID" contains "PAN" but must not get the PAN mask.
        self.assertEqual(replacement_for("Company_ID", "ABCDE12345", "c", salts)[1], "salted_hash")

    def test_every_occurrence_is_redacted(self):
        out, audit = redact("Call 9876543210 or 9876543210 again", 0.75, "c", {}, [])
        self.assertEqual(out, "Call * or * again")
        # Each number is claimed by the phone rule and the catch-all ID rule; the
        # audit lists both detectors per replaced span.
        self.assertEqual(sorted({(a["start_offset"], a["end_offset"]) for a in audit}), [(5, 15), (19, 29)])

    def test_rules_and_model_hits_merge(self):
        text = "Aadhaar 2345 6789 0124 of Ramesh Kumar"
        ner = [{"start": 26, "end": 38, "label": "PERSON", "source": "Hybrid NER", "confidence": 0.99, "value": "Ramesh Kumar"},
               {"start": 26, "end": 32, "label": "PERSON", "source": "Hybrid NER", "confidence": 0.5, "value": "Ramesh"}]
        out, _ = redact(text, 0.75, "c", {}, ner)
        self.assertEqual(out, "Aadhaar XXXX XXXX 0124 of *")

    def test_rule_policy_beats_model_tag_inside_it(self):
        text = "Already Exists as CMOFF/J/2026/001234 dated"
        start = text.index("CMOFF")
        ner = [{"start": start, "end": start + 5, "label": "ORGANIZATION", "source": "Hybrid NER",
                "confidence": 0.95, "value": "CMOFF"}]
        out, audit = redact(text, 0.75, "c", {"c": "salt"}, ner)
        self.assertRegex(out, r"^Already Exists as [0-9a-f]{64} dated$")
        self.assertEqual({a["technique"] for a in audit}, {"salted_hash"})

    def test_policy_never_applied_to_text_it_does_not_cover(self):
        # The e-mail starts first; its mask keeps everything after "@", so applied
        # to the merged span it printed "ra****@gmail.com Ramesh". Nothing covers both: "*".
        text = "mail ramesh@gmail.com Ramesh resident"
        start = text.index("com Ramesh")
        ner = [{"start": start, "end": start + 10, "label": "PERSON", "source": "Hybrid NER",
                "confidence": 0.9, "value": "com Ramesh"}]
        self.assertEqual(redact(text, 0.75, "c", {}, ner)[0], "mail * resident")
        # An age is kept only when nothing else is in its span.
        text = "aged 45 years"
        ner = [{"start": 5, "end": 13, "label": "PERSON", "source": "Hybrid NER", "confidence": 0.9,
                "value": "45 years"}]
        self.assertEqual(redact(text, 0.75, "c", {}, ner)[0], "aged *")


class Pipeline(unittest.TestCase):
    """pipeline.anonymize with the models replaced by a stub."""

    def run_pipeline(self, values, model_hits=None):
        stub = lambda texts, *args, **kwargs: {t: (model_hits or {}).get(t, []) for t in texts}  # noqa: E731
        with mock.patch.object(pipeline, "build_ner_index", side_effect=stub) as ner:
            results = pipeline.anonymize(values, DetectionSettings(minimum_confidence=0.75))
        return results, ner

    def test_distinct_values_cleanup_and_skips(self):
        values = [("a", "Call 9876543210 &amp; go"), ("a", "Call 9876543210 &amp; go"), ("a", ""), ("a", None),
                  ("b", "Call 9876543210 &amp; go")]
        results, ner = self.run_pipeline(values)
        self.assertEqual(set(results), {("a", "Call 9876543210 &amp; go"), ("b", "Call 9876543210 &amp; go")})
        result = results[("a", "Call 9876543210 &amp; go")]
        self.assertEqual((result.cleaned, result.text), ("Call 9876543210 & go", "Call * & go"))
        self.assertEqual(ner.call_args.args[0], ["Call 9876543210 & go"])  # one model pass, cleaned text

    def test_model_hits_and_salt_per_column(self):
        text = "PPP 6ABC1234 of Ramesh"
        hits = {text: [{"start": 16, "end": 22, "label": "PERSON", "source": "Hybrid NER", "confidence": 0.9,
                        "value": "Ramesh"}]}
        results, _ = self.run_pipeline([("a", text), ("b", text)], hits)
        a, b = results[("a", text)].text, results[("b", text)].text
        self.assertTrue(a.endswith(" of *") and b.endswith(" of *"))
        self.assertNotEqual(a, b)  # same value, different column salt

    def test_failure_is_reported_not_raised(self):
        with mock.patch.object(pipeline, "scan_cell", side_effect=ValueError("boom")):
            results, _ = self.run_pipeline([("a", "x&amp;y")])
        self.assertEqual((results[("a", "x&amp;y")].error, results[("a", "x&amp;y")].text), ("boom", "x&y"))


class Config(unittest.TestCase):
    def test_valid_block(self):
        settings = TableJobSettings.from_block(
            {"columns": ["remarks"], "staged_input_path": "out/x.csv", "ner_deny_terms": ["Naresh"]}, ["remarks"]
        )
        self.assertEqual(settings.detection.scan_options.deny_terms, ("Naresh",))
        self.assertTrue(settings.detection.normalize_source_text)

    def test_errors(self):
        cases = [
            ({"columns": "remarks", "staged_input_path": "x.csv"}, "list of column names"),
            ({"columns": ["missing"], "staged_input_path": "x.csv"}, "missing from input"),
            ({"columns": ["remarks"], "staged_input_path": "x.json"}, "must end with '.csv'"),
            ({"columns": ["remarks"], "staged_input_path": "x.csv", "ner_allow_terms": "ATR"}, "list of strings"),
            ({"columns": ["remarks"], "staged_input_path": "x.csv", "on_failure": "skip"}, "on_failure"),
        ]
        for block, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(BatchError, message):
                    TableJobSettings.from_block(block, ["remarks"])


if __name__ == "__main__":
    unittest.main()
