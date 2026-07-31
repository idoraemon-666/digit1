#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "Usage: PROTOCOL3_DIGIT0_DIGIT8_MATCHED_LR_CONTINUATION_AUTHORIZED=1 bash server/run_digit_writing_original_protocol3_digit0_digit8_matched_lr_continuation_parallel.sh REPO CONFIG SOURCE_RUN_ROOT SOURCE_EVIDENCE_ROOT EXPECTED_HEAD EXPECTED_SOURCE_HEAD" >&2
  exit 2
fi
if [[ "${PROTOCOL3_DIGIT0_DIGIT8_MATCHED_LR_CONTINUATION_AUTHORIZED:-0}" != "1" ]]; then
  echo 'ABORT: explicit Protocol3 digit0/digit8 matched continuation approval is required' >&2
  exit 1
fi

REPO="$(realpath "$1")"
CONFIG="$(realpath "$2")"
SOURCE_RUN_ROOT="$(realpath "$3")"
SOURCE_EVIDENCE_ROOT="$(realpath "$4")"
EXPECTED_HEAD="$5"
EXPECTED_SOURCE_HEAD="$6"
PYTHON=/root/autodl-tmp/conda/envs/motor-compat-cpu/bin/python
SUBMODULE=ac0c4f589eae37bbde63968912925de99232e306
FROZEN_SOURCE_HEAD=ee5a1900a33f8a8b7921fd9fc5b09aeed6600925

[[ "$EXPECTED_HEAD" =~ ^[0-9a-f]{40}$ ]]
test "$EXPECTED_SOURCE_HEAD" = "$FROZEN_SOURCE_HEAD"
test -d "$REPO/.git"
test -x "$PYTHON"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
test "$(git -C "$REPO" rev-parse HEAD:mRNNTorch)" = "$SUBMODULE"
test "$(git -C "$REPO/mRNNTorch" rev-parse HEAD)" = "$SUBMODULE"
test -z "$(git -C "$REPO" status --short)"
test -z "$(git -C "$REPO/mRNNTorch" status --short)"
test "$(cat /root/autodl-tmp/conda/envs/motor-compat-cpu/.compatibility_environment_complete)" = "device=cpu"

case "$CONFIG" in
  "$REPO"/configurations/digit_writing_original_protocol3_digit0_digit8_matched_lr_continuation_6000_to8000.json) ;;
  *) echo 'ABORT: config is not the checked-in digit0/digit8 matched continuation config' >&2; exit 1 ;;
esac
case "$SOURCE_RUN_ROOT" in
  "$REPO"/runs/digit_writing_original_protocol3/scale2p50/ref100/corner_ease_v3/single_digit_overfit) ;;
  *) echo 'ABORT: source run root identity is invalid' >&2; exit 1 ;;
esac
case "$SOURCE_EVIDENCE_ROOT" in
  "$REPO"/runs/digit_writing_original_protocol3/scale2p50/ref100/corner_ease_v3/single_digit_overfit_server_evidence) ;;
  *) echo 'ABORT: source evidence root identity is invalid' >&2; exit 1 ;;
esac

test "$(cat "$SOURCE_EVIDENCE_ROOT/repository_head.txt")" = "$FROZEN_SOURCE_HEAD"
test "$(cat "$SOURCE_EVIDENCE_ROOT/submodule_head.txt")" = "$SUBMODULE"
(
  cd "$SOURCE_RUN_ROOT"
  sha256sum -c SHA256SUMS
)
(
  cd "$SOURCE_EVIDENCE_ROOT"
  sha256sum -c SHA256SUMS
)

CONFIG_OUTPUT="$("$PYTHON" - "$CONFIG" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
assert config["protocol"] == "digit_writing_original_protocol3"
assert config["run_kind"] == "protocol3_digit0_digit8_matched_lr_continuation"
assert config["variant"] == "digit0_digit8_four_arm_from6000_to8000"
assert config["source"]["repository_head"] == "ee5a1900a33f8a8b7921fd9fc5b09aeed6600925"
assert config["source"]["completed_updates"] == 6000
assert [case["digit"] for case in config["source"]["cases"]] == [0, 8]
assert [case["source_status"] for case in config["source"]["cases"]] == [
    "PASS_UNSTABLE",
    "FAIL",
]
assert [(arm["label"], arm["learning_rate"]) for arm in config["arms"]] == [
    ("lr1e3", 0.001),
    ("lr3e4", 0.0003),
]
assert config["training"] == {
    "target_completed_updates": 8000,
    "additional_updates": 2000,
    "validation_interval": 100,
    "batch_size": 8,
    "fixed_target_no_early_stop": True,
    "synchronized_parallel_arms": 4,
}
assert config["decision"] == {
    "automatic_winner_selection": False,
    "automatic_further_continuation": False,
    "automatic_second_seed": False,
    "digit8_included": True,
    "formal_full10_start": False,
    "qualitative_overlay_review_required": True,
    "automatic_geometry_change": False,
    "automatic_loss_change": False,
    "synchronized_parallel_training": True,
}
print(config["output"]["directory"])
PY
)"

RUN_ROOT="$(realpath -m "$REPO/$CONFIG_OUTPUT")"
EVIDENCE_ROOT="${RUN_ROOT}_server_evidence"
ARCHIVE="${RUN_ROOT}.tar.gz"
ARCHIVE_HASH="${ARCHIVE}.sha256"
case "$RUN_ROOT" in
  "$REPO"/runs/digit_writing_original_protocol3/scale2p50/ref100/corner_ease_v3/digit0_digit8_matched_lr_continuation_from6000_to8000) ;;
  *) echo 'ABORT: matched continuation output identity is invalid' >&2; exit 1 ;;
esac
for path in "$RUN_ROOT" "$EVIDENCE_ROOT" "$ARCHIVE" "$ARCHIVE_HASH"; do
  if [[ -e "$path" ]]; then
    echo "ABORT: output already exists: $path" >&2
    exit 1
  fi
done
if pgrep -af '[p]rotocol3_corner_ease_lr_continuation run-arm' >/dev/null; then
  echo 'ABORT: another Protocol3 continuation worker is already running' >&2
  exit 1
fi

AVAILABLE_CPUS="$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)"
if (( AVAILABLE_CPUS < 4 )); then
  echo 'ABORT: four-way synchronized training requires nproc >= 4' >&2
  exit 1
fi
CPU_MAX="$(cat /sys/fs/cgroup/cpu.max 2>/dev/null || true)"
if [[ -n "$CPU_MAX" ]]; then
  read -r CPU_QUOTA CPU_PERIOD <<<"$CPU_MAX"
  if [[ "$CPU_QUOTA" != "max" ]] && (( CPU_QUOTA / CPU_PERIOD < 4 )); then
    echo 'ABORT: at least four CPU-equivalent cores are required' >&2
    exit 1
  fi
fi

mkdir -p "$EVIDENCE_ROOT"
TEST_LOG="$EVIDENCE_ROOT/tests.log"
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
      tests.test_protocol3_corner_settle \
      tests.test_protocol3_digit8_lr_ablation \
      tests.test_protocol3_corner_ease \
      tests.test_protocol3_corner_ease_lr_continuation \
      tests.test_protocol3_closed_loop_diagnostic \
      tests.test_protocol3_digit0_digit8_matched_lr_continuation
) 2>&1 | tee "$TEST_LOG"
TEST_CODES=("${PIPESTATUS[@]}")
set -e
test "${TEST_CODES[0]}" -eq 0
test "${TEST_CODES[1]}" -eq 0
grep -q '^OK$' "$TEST_LOG"
if grep -q 'skipped=' "$TEST_LOG"; then
  echo 'ABORT: the matched continuation server test set contains skipped tests' >&2
  exit 1
fi

if ! (
  cd "$REPO"
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 "$PYTHON" \
      -m digit_writing.protocol3_corner_ease_lr_continuation prepare \
      --repository-root "$REPO" \
      --continuation-config "$CONFIG" \
      --source-run-root "$SOURCE_RUN_ROOT" \
      --source-evidence-root "$SOURCE_EVIDENCE_ROOT" \
      --run-root "$RUN_ROOT"
) > "$EVIDENCE_ROOT/prepare.log" 2>&1; then
  echo 'ABORT: matched continuation prepare failed' >&2
  tail -n 80 "$EVIDENCE_ROOT/prepare.log" >&2
  exit 1
fi

grep '^Ran [0-9][0-9]* tests in ' "$TEST_LOG" > "$EVIDENCE_ROOT/test_summary.txt"
printf 'SKIPPED_TESTS=0\n' >> "$EVIDENCE_ROOT/test_summary.txt"
printf '%s\n' "$EXPECTED_HEAD" > "$EVIDENCE_ROOT/repository_head.txt"
printf '%s\n' "$FROZEN_SOURCE_HEAD" > "$EVIDENCE_ROOT/source_repository_head.txt"
printf '%s\n' "$SUBMODULE" > "$EVIDENCE_ROOT/submodule_head.txt"
"$PYTHON" -m pip freeze > "$EVIDENCE_ROOT/environment.freeze.txt"
printf '%s\n' "$CPU_MAX" > "$EVIDENCE_ROOT/cpu.max.txt"
printf '%s\n' "$AVAILABLE_CPUS" > "$EVIDENCE_ROOT/nproc.txt"
lscpu > "$EVIDENCE_ROOT/lscpu.txt"

CASE_LABELS=(digit0_corner_ease digit8_corner_ease)
DIGITS=(0 8)
ARMS=(lr1e3 lr3e4)
PIDS=()
RUN_LABELS=()
for INDEX in "${!CASE_LABELS[@]}"; do
  CASE_LABEL="${CASE_LABELS[$INDEX]}"
  DIGIT="${DIGITS[$INDEX]}"
  for ARM in "${ARMS[@]}"; do
    ARM_DIR="$RUN_ROOT/digit${DIGIT}/${ARM}"
    mkdir -p "$ARM_DIR"
    (
      cd "$REPO"
      CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
        PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 "$PYTHON" \
          -m digit_writing.protocol3_corner_ease_lr_continuation run-arm \
          --repository-root "$REPO" \
          --continuation-config "$CONFIG" \
          --source-run-root "$SOURCE_RUN_ROOT" \
          --run-root "$RUN_ROOT" \
          --case-label "$CASE_LABEL" \
          --arm-label "$ARM" \
          --manual-continuation-authorized
    ) > "$ARM_DIR/run.log" 2>&1 &
    PIDS+=("$!")
    RUN_LABELS+=("digit${DIGIT}_${ARM}")
  done
done

FAILED=0
: > "$EVIDENCE_ROOT/process_exit_codes.txt"
for INDEX in "${!PIDS[@]}"; do
  EXIT_CODE=0
  wait "${PIDS[$INDEX]}" || EXIT_CODE=$?
  printf '%s=%s\n' "${RUN_LABELS[$INDEX]}" "$EXIT_CODE" >> "$EVIDENCE_ROOT/process_exit_codes.txt"
  if [[ "$EXIT_CODE" -ne 0 ]]; then FAILED=1; fi
done
if [[ "$FAILED" -ne 0 ]]; then
  echo "ABORT: one or more matched continuation arms failed; outputs are preserved at $RUN_ROOT" >&2
  cat "$EVIDENCE_ROOT/process_exit_codes.txt" >&2
  for DIGIT in "${DIGITS[@]}"; do
    for ARM in "${ARMS[@]}"; do
      ARM_DIR="$RUN_ROOT/digit${DIGIT}/${ARM}"
      if [[ ! -f "$ARM_DIR/run_summary.json" ]]; then
        echo "===== FAILED ARM digit${DIGIT}/${ARM} =====" >&2
        tail -n 80 "$ARM_DIR/run.log" >&2
      fi
    done
  done
  exit 1
fi

if ! (
  cd "$REPO"
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 "$PYTHON" \
      -m digit_writing.protocol3_corner_ease_lr_continuation summarize \
      --repository-root "$REPO" \
      --continuation-config "$CONFIG" \
      --run-root "$RUN_ROOT"
) > "$EVIDENCE_ROOT/summarize.log" 2>&1; then
  echo 'ABORT: matched continuation summarize failed' >&2
  tail -n 80 "$EVIDENCE_ROOT/summarize.log" >&2
  exit 1
fi

"$PYTHON" - "$RUN_ROOT/digit0_digit8_matched_lr_continuation_summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
assert summary["digits"] == [0, 8]
assert summary["completed_arms"] == 4
assert summary["completed_updates"] == 8000
assert summary["engineering_passed"] is True
assert summary["synchronized_parallel_training"] is True
assert summary["automatic_winner_selection"] is False
assert summary["automatic_further_continuation"] is False
assert summary["automatic_second_seed"] is False
assert summary["digit8_included"] is True
assert summary["automatic_geometry_change"] is False
assert summary["automatic_loss_change"] is False
assert summary["formal_full10_started"] is False
PY

date -u +%Y-%m-%dT%H:%M:%SZ > "$EVIDENCE_ROOT/end_utc.txt"
(
  cd "$RUN_ROOT"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
  sha256sum -c SHA256SUMS
)
(
  cd "$EVIDENCE_ROOT"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
  sha256sum -c SHA256SUMS
)
tar -C "$(dirname "$RUN_ROOT")" -czf "$ARCHIVE" \
  "$(basename "$RUN_ROOT")" "$(basename "$EVIDENCE_ROOT")"
(
  cd "$(dirname "$ARCHIVE")"
  sha256sum "$(basename "$ARCHIVE")" > "$(basename "$ARCHIVE_HASH")"
)
test -z "$(git -C "$REPO" status --short)"
test -z "$(git -C "$REPO/mRNNTorch" status --short)"

echo "RUN_ROOT=$RUN_ROOT"
echo "EVIDENCE_ROOT=$EVIDENCE_ROOT"
echo "ARCHIVE=$ARCHIVE"
echo "ARCHIVE_HASH=$ARCHIVE_HASH"
echo 'COMPLETED_ARMS=4'
echo 'COMPLETED_UPDATES=8000'
echo 'SYNCHRONIZED_PARALLEL_TRAINING=1'
echo 'DIGIT0_STARTED=1'
echo 'DIGIT8_STARTED=1'
echo 'AUTOMATIC_WINNER_SELECTION_STARTED=0'
echo 'AUTOMATIC_FURTHER_CONTINUATION_STARTED=0'
echo 'AUTOMATIC_SECOND_SEED_STARTED=0'
echo 'AUTOMATIC_GEOMETRY_CHANGE_STARTED=0'
echo 'AUTOMATIC_LOSS_CHANGE_STARTED=0'
echo 'FORMAL_FULL10_STARTED=0'
