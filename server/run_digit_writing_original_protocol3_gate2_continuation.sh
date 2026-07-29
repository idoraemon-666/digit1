#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 8 ]]; then
  echo "Usage: MANUAL_RESUME_AUTHORIZED=1 bash server/run_digit_writing_original_protocol3_gate2_continuation.sh REPO CONFIG SOURCE_SUMMARY RESUME_CHECKPOINT CASE TARGET_UPDATES EXPECTED_HEAD EXPECTED_SOURCE_HEAD" >&2
  exit 2
fi
if [[ "${MANUAL_RESUME_AUTHORIZED:-}" != "1" ]]; then
  echo "ABORT: explicit MANUAL_RESUME_AUTHORIZED=1 is required" >&2
  exit 1
fi

REPO="$(realpath "$1")"
CONFIG="$(realpath "$2")"
SOURCE_SUMMARY="$(realpath "$3")"
RESUME_CHECKPOINT="$(realpath "$4")"
CASE_LABEL="$5"
TARGET_UPDATES="$6"
EXPECTED_HEAD="$7"
EXPECTED_SOURCE_HEAD="$8"
PYTHON=/root/autodl-tmp/conda/envs/motor-compat-cpu/bin/python
SUBMODULE=ac0c4f589eae37bbde63968912925de99232e306
SOURCE_ROOT="$(dirname "$SOURCE_SUMMARY")"
SOURCE_ARCHIVE="${SOURCE_ROOT}.tar.gz"
SOURCE_ARCHIVE_HASH="${SOURCE_ARCHIVE}.sha256"

[[ "$CASE_LABEL" == "o1_digit1" || "$CASE_LABEL" == "o2_digit5" || "$CASE_LABEL" == "o3_digit8" ]]
[[ "$TARGET_UPDATES" =~ ^[0-9]+$ ]]
[[ "$EXPECTED_HEAD" =~ ^[0-9a-f]{40}$ ]]
[[ "$EXPECTED_SOURCE_HEAD" =~ ^[0-9a-f]{40}$ ]]
test -d "$REPO/.git"
test -x "$PYTHON"
test "$(basename "$RESUME_CHECKPOINT")" = "final_continuation_checkpoint.pt"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
test "$(git -C "$REPO" rev-parse HEAD:mRNNTorch)" = "$SUBMODULE"
test "$(git -C "$REPO/mRNNTorch" rev-parse HEAD)" = "$SUBMODULE"
test -z "$(git -C "$REPO" status --short)"
test -z "$(git -C "$REPO/mRNNTorch" status --short)"
test "$(cat /root/autodl-tmp/conda/envs/motor-compat-cpu/.compatibility_environment_complete)" = "device=cpu"
case "$CONFIG" in
  "$REPO"/configurations/digit_writing_original_protocol3_gate2_scale*_ref*.json) ;;
  *) echo "ABORT: config is not a checked-in Protocol3 Gate 2 config" >&2; exit 1 ;;
esac
case "$SOURCE_ROOT" in
  "$REPO"/runs/digit_writing_original_protocol3/*) ;;
  *) echo "ABORT: source summary is outside the Protocol3 run root" >&2; exit 1 ;;
esac
case "$RESUME_CHECKPOINT" in
  "$SOURCE_ROOT"/*) ;;
  *) echo "ABORT: checkpoint is outside its source result root" >&2; exit 1 ;;
esac
test -f "$SOURCE_ARCHIVE"
test -f "$SOURCE_ARCHIVE_HASH"
test -f "$SOURCE_ROOT/SHA256SUMS"
(
  cd "$SOURCE_ROOT"
  sha256sum -c SHA256SUMS >/dev/null
)
echo 'SOURCE_TREE_SHA256_OK=1'
(
  cd "$(dirname "$SOURCE_ARCHIVE")"
  sha256sum -c "$(basename "$SOURCE_ARCHIVE_HASH")"
)

REVIEW="$(
  (
  cd "$REPO"
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 "$PYTHON" \
      -m digit_writing.protocol3_gate2_continuation inspect \
      --repository-root "$REPO" \
      --protocol-config "$CONFIG" \
      --source-summary "$SOURCE_SUMMARY" \
      --resume-checkpoint "$RESUME_CHECKPOINT" \
      --case-label "$CASE_LABEL" \
      --expected-source-head "$EXPECTED_SOURCE_HEAD"
  )
)"
printf '%s\n' "$REVIEW"
SOURCE_UPDATES="$(printf '%s\n' "$REVIEW" | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["source_completed_updates"])')"
CONFIG_VALUES="$("$PYTHON" - "$CONFIG" <<'PY'
import json
import sys
config = json.load(open(sys.argv[1], encoding="utf-8"))
print(config["output"]["directory"])
print(config["selected_reference_steps"])
PY
)"
BASE_GATE2_DIRECTORY="$(printf '%s\n' "$CONFIG_VALUES" | sed -n '1p')"
SELECTED_REFERENCE="$(printf '%s\n' "$CONFIG_VALUES" | sed -n '2p')"
test "$TARGET_UPDATES" -gt "$SOURCE_UPDATES"
test $((TARGET_UPDATES % 100)) -eq 0
BASE_GATE2_ROOT="$(realpath -m "$REPO/$BASE_GATE2_DIRECTORY")"
RUN_ROOT="${BASE_GATE2_ROOT}_continuations/${CASE_LABEL}/from${SOURCE_UPDATES}_to${TARGET_UPDATES}"
ARCHIVE="${RUN_ROOT}.tar.gz"
ARCHIVE_HASH="${ARCHIVE}.sha256"
TEST_LOG="${RUN_ROOT}.tests.log"
CONSOLE_LOG="${RUN_ROOT}.console.log"
case "$RUN_ROOT" in
  "$REPO"/runs/digit_writing_original_protocol3/*/ref"$SELECTED_REFERENCE"/gate2_continuations/*) ;;
  *) echo "ABORT: continuation output identity is invalid" >&2; exit 1 ;;
esac
for path in "$RUN_ROOT" "$ARCHIVE" "$ARCHIVE_HASH" "$TEST_LOG" "$CONSOLE_LOG"; do
  if [[ -e "$path" ]]; then
    echo "ABORT: output already exists: $path" >&2
    exit 1
  fi
done
mkdir -p "$(dirname "$RUN_ROOT")"

set +e
(
  cd "$REPO"
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 "$PYTHON" -m unittest -v \
      tests.test_digit_geometry \
      tests.test_phase_normalized_loss \
      tests.test_digit_environment \
      tests.test_geometry_audit \
      tests.test_phase_d_protocol \
      tests.test_protocol3_geometry \
      tests.test_protocol3_schedule \
      tests.test_protocol3_checkpoint \
      tests.test_protocol3_early_audit \
      tests.test_protocol3_configuration \
      tests.test_protocol3_gate2_continuation
) 2>&1 | tee "$TEST_LOG"
TEST_CODES=("${PIPESTATUS[@]}")
set -e
test "${TEST_CODES[0]}" -eq 0
test "${TEST_CODES[1]}" -eq 0
grep -q '^OK$' "$TEST_LOG"
if grep -q 'skipped=' "$TEST_LOG"; then
  echo 'ABORT: the continuation server test set contains skipped tests.' >&2
  exit 1
fi

set +e
(
  cd "$REPO"
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 "$PYTHON" \
      -m digit_writing.protocol3_gate2_continuation run \
      --repository-root "$REPO" \
      --protocol-config "$CONFIG" \
      --source-summary "$SOURCE_SUMMARY" \
      --resume-checkpoint "$RESUME_CHECKPOINT" \
      --case-label "$CASE_LABEL" \
      --target-completed-updates "$TARGET_UPDATES" \
      --output-directory "$RUN_ROOT" \
      --expected-source-head "$EXPECTED_SOURCE_HEAD" \
      --manual-resume-authorized
) 2>&1 | tee "$CONSOLE_LOG"
RUN_CODES=("${PIPESTATUS[@]}")
set -e
RUN_EXIT="${RUN_CODES[0]}"
TEE_EXIT="${RUN_CODES[1]}"
test "$TEE_EXIT" -eq 0
if [[ ! -f "$RUN_ROOT/gate2_continuation_summary.json" ]]; then
  echo "ABORT: continuation did not produce a summary (run exit $RUN_EXIT)" >&2
  exit 1
fi
test "$RUN_EXIT" -eq 0
mv "$TEST_LOG" "$RUN_ROOT/tests.log"
mv "$CONSOLE_LOG" "$RUN_ROOT/console.log"
printf 'RUN_EXIT=%s\nTEE_EXIT=%s\n' "$RUN_EXIT" "$TEE_EXIT" > "$RUN_ROOT/exit_codes.txt"
"$PYTHON" -m pip freeze > "$RUN_ROOT/environment.freeze.txt"
printf '%s\n' "$EXPECTED_HEAD" > "$RUN_ROOT/parent_head.txt"
printf '%s\n' "$EXPECTED_SOURCE_HEAD" > "$RUN_ROOT/source_head.txt"
printf '%s\n' "$SUBMODULE" > "$RUN_ROOT/submodule_head.txt"
sha256sum "$SOURCE_ARCHIVE" > "$RUN_ROOT/source_archive.sha256"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/end_utc.txt"

(
  cd "$RUN_ROOT"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
  sha256sum -c SHA256SUMS
)
tar -C "$(dirname "$RUN_ROOT")" -czf "$ARCHIVE" "$(basename "$RUN_ROOT")"
(
  cd "$(dirname "$ARCHIVE")"
  sha256sum "$(basename "$ARCHIVE")" > "$(basename "$ARCHIVE_HASH")"
)
test -z "$(git -C "$REPO" status --short)"
test -z "$(git -C "$REPO/mRNNTorch" status --short)"

CLASSIFICATION="$("$PYTHON" - "$RUN_ROOT/gate2_continuation_summary.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["classification"])
PY
)"
echo "GATE2_CONTINUATION_CLASSIFICATION=$CLASSIFICATION"
echo "RUN_ROOT=$RUN_ROOT"
echo "ARCHIVE=$ARCHIVE"
echo "ARCHIVE_HASH=$ARCHIVE_HASH"
echo 'AUTOMATIC_CONTINUATION_STARTED=0'
echo 'AUTOMATIC_MEDIUM_FALLBACK_STARTED=0'
echo 'FORMAL_FULL10_STARTED=0'
if [[ "$CLASSIFICATION" == "engineering_or_safety_failure" ]]; then
  exit 1
fi
if [[ "$CLASSIFICATION" == "behavior_failure" ]]; then
  exit 22
fi
test "$CLASSIFICATION" = "provisional_pass_pending_qualitative_review"
