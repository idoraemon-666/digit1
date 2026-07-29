#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: bash server/run_digit_writing_original_protocol3_corner_settle_fast_parallel.sh REPO CONFIG GATE1_SUMMARY EXPECTED_HEAD" >&2
  exit 2
fi

REPO="$(realpath "$1")"
CONFIG="$(realpath "$2")"
GATE1_SUMMARY="$(realpath "$3")"
EXPECTED_HEAD="$4"
PYTHON=/root/autodl-tmp/conda/envs/motor-compat-cpu/bin/python
SUBMODULE=ac0c4f589eae37bbde63968912925de99232e306

[[ "$EXPECTED_HEAD" =~ ^[0-9a-f]{40}$ ]]
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

readarray -t CONFIG_VALUES < <("$PYTHON" - "$CONFIG" <<'PY'
import json
import sys
config = json.load(open(sys.argv[1], encoding="utf-8"))
assert config["selected_reference_steps"] == 50
assert config["corner_settle"]["settle_intervals"] == 10
assert config["corner_settle"]["physical_pause_ms"] == 100
assert config["training"]["max_updates"] == 6000
assert config["corner_settle"]["automatic_continuation"] is False
assert config["corner_settle"]["automatic_medium_fallback"] is False
print(config["output"]["directory"])
PY
)
RUN_ROOT="$(realpath -m "$REPO/${CONFIG_VALUES[0]}")"
ARCHIVE="${RUN_ROOT}.tar.gz"
ARCHIVE_HASH="${ARCHIVE}.sha256"
case "$RUN_ROOT" in
  "$REPO"/runs/digit_writing_original_protocol3_corner_settle_gate/scale2p50/ref50/review6000) ;;
  *) echo "ABORT: corner-settle output identity is invalid" >&2; exit 1 ;;
esac
for path in "$RUN_ROOT" "$ARCHIVE" "$ARCHIVE_HASH"; do
  if [[ -e "$path" ]]; then
    echo "ABORT: output already exists: $path" >&2
    exit 1
  fi
done

"$PYTHON" - "$GATE1_SUMMARY" "$EXPECTED_HEAD" <<'PY'
import json
import sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
assert summary["passed"] is True
assert summary["scale_token"] == "2p50"
assert summary["git_identity"]["repository_head"] == sys.argv[2]
PY

CPU_MAX="$(cat /sys/fs/cgroup/cpu.max 2>/dev/null || true)"
if [[ -n "$CPU_MAX" ]]; then
  read -r CPU_QUOTA CPU_PERIOD <<<"$CPU_MAX"
  if [[ "$CPU_QUOTA" != "max" ]] && (( CPU_QUOTA / CPU_PERIOD < 4 )); then
    echo "ABORT: at least four CPU-equivalent cores are required" >&2
    exit 1
  fi
fi

# The review output hierarchy may not exist on a freshly provisioned card.
# Create only its parent here so the pre-training test log can be captured;
# `prepare` remains responsible for atomically creating RUN_ROOT itself.
mkdir -p "$(dirname "$RUN_ROOT")"
TEST_LOG="${RUN_ROOT}.tests.log"
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
  echo 'ABORT: the corner-settle server test set contains skipped tests.' >&2
  exit 1
fi

(
  cd "$REPO"
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 "$PYTHON" \
      -m digit_writing.protocol3_corner_settle prepare \
      --repository-root "$REPO" \
      --protocol-config "$CONFIG" \
      --output-directory "$RUN_ROOT"
)
mv "$TEST_LOG" "$RUN_ROOT/tests.log"
grep '^Ran [0-9][0-9]* tests in ' "$RUN_ROOT/tests.log" > "$RUN_ROOT/test_summary.txt"
printf 'SKIPPED_TESTS=0\n' >> "$RUN_ROOT/test_summary.txt"
printf '%s\n' "$EXPECTED_HEAD" > "$RUN_ROOT/parent_head.txt"
printf '%s\n' "$SUBMODULE" > "$RUN_ROOT/submodule_head.txt"
"$PYTHON" -m pip freeze > "$RUN_ROOT/environment.freeze.txt"
printf '%s\n' "$CPU_MAX" > "$RUN_ROOT/cpu.max.txt"
lscpu > "$RUN_ROOT/lscpu.txt"

CASES=(digit4_baseline digit4_settle100ms digit7_baseline digit7_settle100ms)
PIDS=()
for CASE_LABEL in "${CASES[@]}"; do
  (
    cd "$REPO"
    CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 "$PYTHON" \
        -m digit_writing.protocol3_corner_settle run-case \
        --repository-root "$REPO" \
        --protocol-config "$CONFIG" \
        --run-root "$RUN_ROOT" \
        --case-label "$CASE_LABEL"
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
  echo "ABORT: one or more corner-settle cases failed; outputs are preserved at $RUN_ROOT" >&2
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

readarray -t SUMMARY_VALUES < <("$PYTHON" - "$RUN_ROOT/corner_settle_summary.json" <<'PY'
import json
import sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
print(summary["classification"])
print(summary["completed_updates"])
print(int(summary["engineering_passed"]))
PY
)
echo "CORNER_SETTLE_CLASSIFICATION=${SUMMARY_VALUES[0]}"
echo "COMPLETED_UPDATES=${SUMMARY_VALUES[1]}"
echo "ENGINEERING_PASSED=${SUMMARY_VALUES[2]}"
echo "RUN_ROOT=$RUN_ROOT"
echo "ARCHIVE=$ARCHIVE"
echo "ARCHIVE_HASH=$ARCHIVE_HASH"
echo 'AUTOMATIC_CONTINUATION_STARTED=0'
echo 'AUTOMATIC_MEDIUM_FALLBACK_STARTED=0'
echo 'FORMAL_FULL10_STARTED=0'
test "${SUMMARY_VALUES[2]}" -eq 1
