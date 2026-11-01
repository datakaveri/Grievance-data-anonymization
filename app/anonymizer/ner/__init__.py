"""Step 3 — model-based detection of names, places and organisations.

  ner/models.py     which models (registry), loading, runtime knobs
  ner/inference.py  run models over lines: batching, chunking, corpus passes
  ner/spans.py      raw model spans -> clean whole-word entities, merged per line
  ner/filters.py    clear entities that are not personal data (titles, grammar
                    words, everyday vocabulary, allow-lists)
  ner/index.py      batch helper: infer each distinct line once, map onto cells

Importing this package does not load torch; ner/models.py does, on first import.
"""
