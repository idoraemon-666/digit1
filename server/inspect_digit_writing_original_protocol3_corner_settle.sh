#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash server/inspect_digit_writing_original_protocol3_corner_settle.sh RUN_ROOT" >&2
  exit 2
fi

RUN_ROOT="$(realpath "$1")"
case "$RUN_ROOT" in
  */runs/digit_writing_original_protocol3_corner_settle_gate/scale2p50/ref50/*) ;;
  *) echo "ABORT: run root is outside the fast corner-settle namespace" >&2; exit 1 ;;
esac

echo "TIME=$(date '+%F %T')"
echo '--- PROCESSES ---'
pgrep -af '[p]rotocol3_corner_settle (run-case|continue-case)' || true
echo '--- CASE PROGRESS ---'
find "$RUN_ROOT" -name test_losses.txt -type f -print0 2>/dev/null | sort -z | \
while IFS= read -r -d '' FILE; do
  LINES="$(wc -l < "$FILE")"
  ESTIMATED_UPDATES=$(( (LINES - 1) * 100 ))
  CASE_PATH="${FILE#"$RUN_ROOT"/}"
  CASE_PATH="${CASE_PATH%/test_losses.txt}"
  printf '%-30s lines=%-4s estimated_completed_updates=%s\n' \
    "$CASE_PATH" "$LINES" "$ESTIMATED_UPDATES"
done
echo '--- PROCESS EXITS ---'
cat "$RUN_ROOT/process_exit_codes.txt" 2>/dev/null || true
echo '--- SUMMARY ---'
if [[ -f "$RUN_ROOT/corner_settle_summary.json" ]]; then
  echo CORNER_SETTLE_STATUS=REVIEW_READY
else
  echo CORNER_SETTLE_STATUS=RUNNING_OR_INCOMPLETE
fi
