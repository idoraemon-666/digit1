#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "Usage: MANUAL_RESUME_AUTHORIZED=1 bash server/run_digit_writing_original_protocol3_corner_settle_continue_parallel.sh REPO CONFIG SOURCE_ROOT TARGET_UPDATES EXPECTED_HEAD EXPECTED_SOURCE_HEAD SOURCE_ARCHIVE" >&2
  exit 2
fi
if [[ "${MANUAL_RESUME_AUTHORIZED:-}" != "1" ]]; then
  echo "ABORT: explicit MANUAL_RESUME_AUTHORIZED=1 is required" >&2
  exit 1
fi

REPO="$(realpath "$1")"
CONFIG="$(realpath "$2")"
SOURCE_ROOT="$(realpath "$3")"
TARGET_UPDATES="$4"
EXPECTED_HEAD="$5"
EXPECTED_SOURCE_HEAD="$6"
SOURCE_ARCHIVE="$(realpath "$7")"
PYTHON=/root/autodl-tmp/conda/envs/motor-compat-cpu/bin/python
SUBMODULE=ac0c4f589eae37bbde63968912925de99232e306
SOURCE_ARCHIVE_HASH="${SOURCE_ARCHIVE}.sha256"

[[ "$TARGET_UPDATES" =~ ^[0-9]+$ ]]
[[ "$EXPECTED_HEAD" =~ ^[0-9a-f]{40}$ ]]
[[ "$EXPECTED_SOURCE_HEAD" =~ ^[0-9a-f]{40}$ ]]
test -d "$REPO/.git"
test -x "$PYTHON"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
test "$(git -C "$REPO" rev-parse HEAD:mRNNTorch)" = "$SUBMODULE"
test "$(git -C "$REPO/mRNNTorch" rev-parse HEAD)" = "$SUBMODULE"
test -z "$(git -C "$REPO" status --short)"
test -z "$(git -C "$REPO/mRNNTorch" status --short)"
test "$(cat /root/autodl-tmp/conda/envs/motor-compat-cpu/.compatibility_environment_complete)" = "device=cpu"
case "$CONFIG" in
  "$REPO"/configurations/digit_writing_original_protocol3_corner_settle_scale2p50_ref50.json) ;;
  *) echo "ABORT: config is not the checked-in fast corner-settle config" >&2; exit 1 ;;
esac
case "$SOURCE_ROOT" in
  "$REPO"/runs/digit_writing_original_protocol3_corner_settle_gate/scale2p50/ref50/*) ;;
  *) echo "ABORT: source is outside the fast corner-settle result root" >&2; exit 1 ;;
esac
test -f "$SOURCE_ROOT/corner_settle_summary.json"
test -f "$SOURCE_ROOT/SHA256SUMS"
test -f "$SOURCE_ARCHIVE"
test -f "$SOURCE_ARCHIVE_HASH"
(
  cd "$SOURCE_ROOT"
  sha256sum -c SHA256SUMS >/dev/null
)
(
  cd "$(dirname "$SOURCE_ARCHIVE")"
  sha256sum -c "$(basename "$SOURCE_ARCHIVE_HASH")"
)

readarray -t SOURCE_VALUES < <("$PYTHON" - "$SOURCE_ROOT/corner_settle_summary.json" "$EXPECTED_SOURCE_HEAD" <<'PY'
import json
import sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
assert summary["git_identity"]["repository_head"] == sys.argv[2]
assert summary["engineering_passed"] is True
assert summary["classification"] == "pending_manual_review"
assert summary["automatic_continuation_started"] is False
assert summary["automatic_medium_fallback_started"] is False
print(summary["completed_updates"])
PY
)
SOURCE_UPDATES="${SOURCE_VALUES[0]}"
test "$TARGET_UPDATES" -eq $((SOURCE_UPDATES + 6000))
test $((SOURCE_UPDATES % 6000)) -eq 0

BASE_PARENT="$(realpath -m "$REPO/runs/digit_writing_original_protocol3_corner_settle_gate/scale2p50/ref50")"
RUN_ROOT="$BASE_PARENT/continuations/from${SOURCE_UPDATES}_to${TARGET_UPDATES}"
ARCHIVE="${RUN_ROOT}.tar.gz"
ARCHIVE_HASH="${ARCHIVE}.sha256"
TEST_LOG="${RUN_ROOT}.tests.log"
case "$RUN_ROOT" in
  "$REPO"/runs/digit_writing_original_protocol3_corner_settle_gate/scale2p50/ref50/continuations/from*_to*) ;;
  *) echo "ABORT: continuation output identity is invalid" >&2; exit 1 ;;
esac
for path in "$RUN_ROOT" "$ARCHIVE" "$ARCHIVE_HASH" "$TEST_LOG"; do
  if [[ -e "$path" ]]; then
    echo "ABORT: output already exists: $path" >&2
    exit 1
  fi
done

set +e
(
  cd "$REPO"
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
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
      tests.test_protocol3_gate2_continuation \
      tests.test_protocol3_corner_settle
) 2>&1 | tee "$TEST_LOG"
TEST_CODES=("${PIPESTATUS[@]}")
set -e
test "${TEST_CODES[0]}" -eq 0
test "${TEST_CODES[1]}" -eq 0
grep -q '^OK$' "$TEST_LOG"
if grep -q 'skipped=' "$TEST_LOG"; then
  echo 'ABORT: the corner-settle continuation server tests contain skips.' >&2
  exit 1
fi

mkdir -p "$RUN_ROOT"
cp "$SOURCE_ROOT/resolved_config.json" "$RUN_ROOT/resolved_config.json"
cp "$SOURCE_ROOT/source_gate2_config.json" "$RUN_ROOT/source_gate2_config.json"
cp "$SOURCE_ROOT/workspace_audit.json" "$RUN_ROOT/workspace_audit.json"
cp "$SOURCE_ROOT/corner_manifest.csv" "$RUN_ROOT/corner_manifest.csv"
mv "$TEST_LOG" "$RUN_ROOT/tests.log"
grep '^Ran [0-9][0-9]* tests in ' "$RUN_ROOT/tests.log" > "$RUN_ROOT/test_summary.txt"
printf 'SKIPPED_TESTS=0\n' >> "$RUN_ROOT/test_summary.txt"
printf '%s\n' "$EXPECTED_HEAD" > "$RUN_ROOT/parent_head.txt"
printf '%s\n' "$EXPECTED_SOURCE_HEAD" > "$RUN_ROOT/source_head.txt"
printf '%s\n' "$SUBMODULE" > "$RUN_ROOT/submodule_head.txt"
sha256sum "$SOURCE_ARCHIVE" > "$RUN_ROOT/source_archive.sha256"
"$PYTHON" -m pip freeze > "$RUN_ROOT/environment.freeze.txt"
cat /sys/fs/cgroup/cpu.max > "$RUN_ROOT/cpu.max.txt" 2>/dev/null || true
lscpu > "$RUN_ROOT/lscpu.txt"

CASES=(digit4_baseline digit4_settle100ms digit7_baseline digit7_settle100ms)
PIDS=()
for CASE_LABEL in "${CASES[@]}"; do
  (
    cd "$REPO"
    CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 "$PYTHON" \
        -m digit_writing.protocol3_corner_settle continue-case \
        --repository-root "$REPO" \
        --protocol-config "$CONFIG" \
        --source-run-root "$SOURCE_ROOT" \
        --output-run-root "$RUN_ROOT" \
        --case-label "$CASE_LABEL" \
        --target-completed-updates "$TARGET_UPDATES" \
        --expected-source-head "$EXPECTED_SOURCE_HEAD" \
        --manual-resume-authorized
  ) > "$RUN_ROOT/${CASE_LABEL}.console.log" 2>&1 &
  PIDS+=("$!")
done

FAILED=0
: > "$RUN_ROOT/process_exit_codes.txt"
for INDEX in "${!PIDS[@]}"; do
  EXIT_CODE=0
  wait "${PIDS[$INDEX]}" || EXIT_CODE=$?
  printf '%s=%s\n' "${CASES[$INDEX]}" "$EXIT_CODE" >> "$RUN_ROOT/process_exit_codes.txt"
  if [[ "$EXIT_CODE" -ne 0 ]]; then FAILED=1; fi
done
if [[ "$FAILED" -ne 0 ]]; then
  echo "ABORT: one or more continuation cases failed; outputs are preserved at $RUN_ROOT" >&2
  exit 1
fi

(
  cd "$REPO"
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 "$PYTHON" \
      -m digit_writing.protocol3_corner_settle summarize \
      --repository-root "$REPO" \
      --protocol-config "$CONFIG" \
      --run-root "$RUN_ROOT"
) > "$RUN_ROOT/summarize.log" 2>&1

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

echo "COMPLETED_UPDATES=$TARGET_UPDATES"
echo "RUN_ROOT=$RUN_ROOT"
echo "ARCHIVE=$ARCHIVE"
echo "ARCHIVE_HASH=$ARCHIVE_HASH"
echo 'AUTOMATIC_CONTINUATION_STARTED=0'
echo 'AUTOMATIC_MEDIUM_FALLBACK_STARTED=0'
echo 'FORMAL_FULL10_STARTED=0'
