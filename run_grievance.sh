#!/usr/bin/env bash
# Runs every grievance table through batch_pipeline.py, one at a time.
# Per table: output/grievance/<stem>/{run.log,status.json,DONE} plus the pipeline's own files.
cd ~/anon || exit 1
source venv/bin/activate
export HF_HUB_OFFLINE=1
PROG=output/grievance/_progress.log
mkdir -p output/grievance
echo "$(date -u +%FT%TZ) START $(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l)" >> "$PROG"
while read -r stem; do
  [ -z "$stem" ] && continue
  out="output/grievance/$stem"
  [ -f "$out/DONE" ] && { echo "$(date -u +%FT%TZ) SKIP $stem (already done)" >> "$PROG"; continue; }
  mkdir -p "$out"
  echo "$(date -u +%FT%TZ) BEGIN $stem" >> "$PROG"
  t0=$(date +%s)
  /usr/bin/time -v -o "$out/time.txt" python -u app/batch_pipeline.py --config "config/grievance/$stem.json" --root ~/anon > "$out/run.log" 2>&1
  rc=$?
  t1=$(date +%s)
  peak=$(awk -F': ' '/Maximum resident/{print $2}' "$out/time.txt" 2>/dev/null)
  printf '{"table":"%s","exit_code":%d,"seconds":%d,"peak_rss_kb":%s,"finished_utc":"%s"}\n' "$stem" "$rc" $((t1-t0)) "${peak:-null}" "$(date -u +%FT%TZ)" > "$out/status.json"
  cp "config/grievance/$stem.json" "$out/config_used.json"
  touch "$out/DONE"
  echo "$(date -u +%FT%TZ) END $stem rc=$rc secs=$((t1-t0)) peak_rss_kb=${peak:-?}" >> "$PROG"
done < config/grievance/order.txt
echo "$(date -u +%FT%TZ) ALL_DONE" >> "$PROG"
