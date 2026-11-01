"""NER comparison report (entry point: app/ner_report.py).

Reads documents (TXT, DOCX, HTML, JSON, CSV — a file, a folder, or an inline
string), runs the rule scanner and each NER model separately, and writes an
Excel workbook plus JSON files comparing what each model found. It is an
evaluation tool for judging the models; it does not redact anything. The
anonymization itself is the `anonymizer` package.

  report/records.py       FileRecord / LineNerResult data model
  report/readers.py       turn each input format into plain text lines
  report/parallel_ner.py  one worker thread per model, results merged per line
  report/writers.py       the 4-sheet Excel workbook and the per-model JSON files
  report/cli.py           command line
"""
