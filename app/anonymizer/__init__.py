"""PII anonymization of free-text grievance data (English, Hindi, Hinglish).

Three layers, read top-down:

  ENTRY SCRIPTS (app/*.py) — command line only
       batch_pipeline.py   -> jobs/table.py          (the Docker entrypoint)
       json_remarks.py     -> jobs/nested_json.py
       model_setup.py      download / cache the NER models
       ner_report.py       -> the separate `report` package (diagnostic tool)

  JOBS (jobs/) — one per input shape: read input -> (column, text) pairs ->
  pipeline.anonymize() -> write each result back where it came from
       jobs/table.py         columns of a table -> staged CSV for SKALD + audit JSON
       jobs/nested_json.py   *remark* fields of nested JSON -> same JSON + report
       config.py             find / validate the dataset config:
                               DetectionSettings  (how to detect; every job)
                               TableJobSettings   (columns, output paths)

  PIPELINE (pipeline.py) — the flow, written once; each step is one module:
    1. Cleanup
       cleanup.py        HTML entities, broken legacy-font Hindi
    2. Rule-based detection (regex)
       rules/            WHAT to look for, by scope:
         generic.py        e-mail, card, IP, numeric dates, user/patient IDs, age, deny list
         india.py          Aadhaar, PAN, DL, phones (+91), Hindi dates, reference numbers,
                           catch-all IDs, PIN codes, Hindi word lists
         haryana.py        PPP family ID, Haryana record words, Haryana NER exemptions
         __init__.py       the order rules run in (it decides ties)
       scanner.py        HOW: runs the rules over each line -> PiiHit / cell detections
    3. Model-based detection (names, places, organisations)
       ner/models.py     model registry, loading, runtime knobs
       ner/inference.py  batching, long-line chunking, whole-corpus passes
       ner/spans.py      snap to whole words, clean, merge both models' spans
       ner/filters.py    clear titles, grammar words, everyday vocabulary, allow-lists
       ner/index.py      infer each distinct line once, map entities back onto cells
    4. Redaction
       redaction.py      pool rule + model detections, apply the per-type policy

  shared:  text.py       word positions, Indic-aware word characters, script detection
  debug/:  profiling.py, reports.py — timing and attribution reports of debug runs

The NER comparison report (app/ner_report.py) lives in the separate `report`
package and reuses steps 2 and 3 on documents, line by line.
"""
