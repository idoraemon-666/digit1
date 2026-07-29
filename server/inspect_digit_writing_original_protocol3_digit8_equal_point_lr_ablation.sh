#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash server/inspect_digit_writing_original_protocol3_digit8_equal_point_lr_ablation.sh RUN_ROOT" >&2
  exit 2
fi

RUN_ROOT="$(realpath -m "$1")"
case "$RUN_ROOT" in
  */runs/digit_writing_original_protocol3/scale2p50/ref100/digit8_equal_point_lr_from6000_to7000) ;;
  *) echo 'ABORT: run root is outside the digit8 LR-ablation namespace' >&2; exit 1 ;;
esac

echo "TIME=$(date '+%F %T')"
echo '--- PROCESSES ---'
PROCESSES="$(pgrep -af '[p]rotocol3_digit8_lr_ablation run-arm' | grep -F -- "$RUN_ROOT" || true)"
printf '%s\n' "$PROCESSES"
echo '--- ARM PROGRESS ---'
ARMS=(digit8_medium_lr1e3 digit8_medium_lr3e4 digit8_medium_lr1e4)
for ARM in "${ARMS[@]}"; do
  ARM_ROOT="$RUN_ROOT/$ARM"
  STATUS=WAITING_OR_STOPPED
  if [[ -f "$ARM_ROOT/lr_ablation_arm_summary.json" ]]; then
    STATUS=FINISHED
  elif grep -Fq -- "--arm-label $ARM" <<<"$PROCESSES"; then
    STATUS=RUNNING
  elif grep -Eq 'Traceback|FAILED|ERROR|Killed|NaN|Inf' "$RUN_ROOT/${ARM}.console.log" 2>/dev/null; then
    STATUS=FAILED
  fi
  if [[ -f "$ARM_ROOT/test_losses.txt" ]]; then
    LINES="$(wc -l < "$ARM_ROOT/test_losses.txt")"
    ESTIMATED_UPDATES=$(( (LINES - 1) * 100 ))
    printf '%-24s status=%-18s estimated_completed_updates=%s\n' \
      "$ARM" "$STATUS" "$ESTIMATED_UPDATES"
  else
    printf '%-24s status=%s\n' "$ARM" "$STATUS"
  fi
done
echo '--- PROCESS EXITS ---'
cat "$RUN_ROOT/process_exit_codes.txt" 2>/dev/null || true
echo '--- SUMMARY ---'
ARCHIVE="${RUN_ROOT}.tar.gz"
ARCHIVE_HASH="${ARCHIVE}.sha256"
if [[ -f "$RUN_ROOT/RUN_COMPLETE.txt" && -f "$ARCHIVE" && -f "$ARCHIVE_HASH" ]]; then
  if (cd "$(dirname "$ARCHIVE")" && sha256sum -c "$(basename "$ARCHIVE_HASH")"); then
    echo DIGIT8_LR_ABLATION_STATUS=DELIVERY_READY
    echo DIGIT8_LR_ABLATION_DELIVERY_READY=1
  else
    echo DIGIT8_LR_ABLATION_STATUS=ARCHIVE_HASH_FAILED
    echo DIGIT8_LR_ABLATION_DELIVERY_READY=0
  fi
elif [[ -f "$RUN_ROOT/digit8_equal_point_lr_ablation_summary.json" ]]; then
  echo DIGIT8_LR_ABLATION_STATUS=SUMMARY_READY_ARCHIVE_PENDING_OR_FAILED
  echo DIGIT8_LR_ABLATION_DELIVERY_READY=0
elif [[ -n "$PROCESSES" ]]; then
  echo DIGIT8_LR_ABLATION_STATUS=RUNNING
  echo DIGIT8_LR_ABLATION_DELIVERY_READY=0
else
  echo DIGIT8_LR_ABLATION_STATUS=FAILED_OR_ORCHESTRATOR_STOPPED
  echo DIGIT8_LR_ABLATION_DELIVERY_READY=0
fi
echo 'AUTOMATIC_FURTHER_CONTINUATION_STARTED=0'
echo 'AUTOMATIC_REFERENCE_FALLBACK_STARTED=0'
echo 'FORMAL_FULL10_STARTED=0'
