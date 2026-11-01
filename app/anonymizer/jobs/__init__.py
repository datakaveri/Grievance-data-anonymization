"""Jobs: one per input shape. Each reads its input into (column, text) pairs,
runs pipeline.anonymize() over them, and writes the results back.

  table.py         columns of a CSV / Excel / JSON table -> staged CSV + audit
  nested_json.py   *remark* fields of nested JSON records -> same JSON + report
"""
