#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: PROTOCOL3_DIGIT0_DIGIT8_INTERVAL_LR_SCRATCH_AUTHORIZED=1 bash server/run_digit_writing_original_protocol3_digit0_digit8_movement_interval_lr_scratch_parallel.sh REPO CONFIG EXPECTED_HEAD" >&2
  exit 2
fi
if [[ "${PROTOCOL3_DIGIT0_DIGIT8_INTERVAL_LR_SCRATCH_AUTHORIZED:-0}" != "1" ]]; then
  echo 'ABORT: explicit Protocol3 twelve-arm scratch approval is required' >&2
  exit 1
fi

REPO="$(realpath "$1")"
CONFIG="$(realpath "$2")"
EXPECTED_HEAD="$3"
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
  "$REPO"/configurations/digit_writing_original_protocol3_digit0_digit8_movement_interval_lr_scratch6000.json) ;;
  *) echo 'ABORT: config is not the checked-in twelve-arm scratch config' >&2; exit 1 ;;
esac

CONFIG_OUTPUT="$("$PYTHON" - "$CONFIG" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
assert config["protocol"] == "digit_writing_original_protocol3"
assert config["run_kind"] == "protocol3_digit0_digit8_movement_interval_lr_scratch"
assert config["variant"] == "digit0_digit8_twelve_arm_interval_lr_scratch6000"
assert config["digits"] == [0, 8]
assert config["source_movement_intervals"] == {"0": 170, "8": 200}
assert config["movement_interval_arms"] == {
    "0": [170, 200, 220],
    "8": [200, 220, 240],
}
assert [(row["label"], row["learning_rate"]) for row in config["learning_rate_arms"]] == [
    ("lr1e3", 0.001),
    ("lr3e4", 0.0003),
]
assert len(config["arms"]) == 12
assert config["optimizer"] == {
    "name": "Adam",
    "grad_clip_norm": 1.0,
    "fixed_learning_rate_no_schedule": True,
}
assert config["training"] == {
    "batch_size": 8,
    "max_updates": 6000,
    "validation_interval": 100,
    "from_scratch": True,
    "early_stopping": False,
    "synchronized_parallel_arms": 12,
}
assert config["decision"] == {
    "automatic_interval_selection": False,
    "automatic_learning_rate_selection": False,
    "automatic_continuation": False,
    "automatic_second_seed": False,
    "automatic_loss_change": False,
    "automatic_geometry_change": False,
    "formal_full10_start": False,
    "qualitative_overlay_review_required": True,
}
print(config["output"]["directory"])
PY
)"

RUN_ROOT="$(realpath -m "$REPO/$CONFIG_OUTPUT")"
EVIDENCE_ROOT="${RUN_ROOT}_server_evidence"
ARCHIVE="${RUN_ROOT}.tar.gz"
ARCHIVE_HASH="${ARCHIVE}.sha256"
case "$RUN_ROOT" in
  "$REPO"/runs/digit_writing_original_protocol3/scale2p50/ref100/corner_ease_v3/digit0_digit8_movement_interval_lr_scratch6000) ;;
  *) echo 'ABORT: twelve-arm scratch output identity is invalid' >&2; exit 1 ;;
esac
for path in "$RUN_ROOT" "$EVIDENCE_ROOT" "$ARCHIVE" "$ARCHIVE_HASH"; do
  if [[ -e "$path" ]]; then
    echo "ABORT: output already exists: $path" >&2
    exit 1
  fi
done
if pgrep -af '[p]rotocol3_movement_interval_lr_scratch run-arm' >/dev/null; then
  echo 'ABORT: another Protocol3 movement-interval scratch worker is already running' >&2
  exit 1
fi

AVAILABLE_CPUS="$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)"
if (( AVAILABLE_CPUS < 12 )); then
  echo 'ABORT: twelve-way synchronized training requires nproc >= 12' >&2
  exit 1
fi
CPU_MAX="$(cat /sys/fs/cgroup/cpu.max 2>/dev/null || true)"
if [[ -n "$CPU_MAX" ]]; then
  read -r CPU_QUOTA CPU_PERIOD <<<"$CPU_MAX"
  if [[ "$CPU_QUOTA" != "max" ]] && (( CPU_QUOTA / CPU_PERIOD < 12 )); then
    echo 'ABORT: at least twelve CPU-equivalent cores are required' >&2
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
      tests.test_protocol3_digit0_digit8_matched_lr_continuation \
      tests.test_protocol3_movement_interval_lr_scratch
) 2>&1 | tee "$TEST_LOG"
TEST_CODES=("${PIPESTATUS[@]}")
set -e
test "${TEST_CODES[0]}" -eq 0
test "${TEST_CODES[1]}" -eq 0
grep -q '^OK$' "$TEST_LOG"
if grep -q 'skipped=' "$TEST_LOG"; then
  echo 'ABORT: the twelve-arm scratch server test set contains skipped tests' >&2
  exit 1
fi

if ! (
  cd "$REPO"
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 "$PYTHON" \
      -m digit_writing.protocol3_movement_interval_lr_scratch prepare \
      --repository-root "$REPO" \
      --experiment-config "$CONFIG" \
      --run-root "$RUN_ROOT" \
      --evidence-root "$EVIDENCE_ROOT"
) > "$EVIDENCE_ROOT/prepare.log" 2>&1; then
  echo 'ABORT: twelve-arm scratch prepare failed' >&2
  tail -n 100 "$EVIDENCE_ROOT/prepare.log" >&2
  exit 1
fi

grep '^Ran [0-9][0-9]* tests in ' "$TEST_LOG" > "$EVIDENCE_ROOT/test_summary.txt"
printf 'SKIPPED_TESTS=0\n' >> "$EVIDENCE_ROOT/test_summary.txt"
printf '%s\n' "$EXPECTED_HEAD" > "$EVIDENCE_ROOT/repository_head.txt"
printf '%s\n' "$SUBMODULE" > "$EVIDENCE_ROOT/submodule_head.txt"
"$PYTHON" -m pip freeze > "$EVIDENCE_ROOT/environment.freeze.txt"
printf '%s\n' "$CPU_MAX" > "$EVIDENCE_ROOT/cpu.max.txt"
printf '%s\n' "$AVAILABLE_CPUS" > "$EVIDENCE_ROOT/nproc.txt"
lscpu > "$EVIDENCE_ROOT/lscpu.txt"

ARM_LABELS=(
  digit0_time170_lr1e3 digit0_time200_lr1e3 digit0_time220_lr1e3
  digit8_time200_lr1e3 digit8_time220_lr1e3 digit8_time240_lr1e3
  digit0_time170_lr3e4 digit0_time200_lr3e4 digit0_time220_lr3e4
  digit8_time200_lr3e4 digit8_time220_lr3e4 digit8_time240_lr3e4
)
ARM_DIRECTORIES=(
  digit0/time170/lr1e3 digit0/time200/lr1e3 digit0/time220/lr1e3
  digit8/time200/lr1e3 digit8/time220/lr1e3 digit8/time240/lr1e3
  digit0/time170/lr3e4 digit0/time200/lr3e4 digit0/time220/lr3e4
  digit8/time200/lr3e4 digit8/time220/lr3e4 digit8/time240/lr3e4
)
PIDS=()
for INDEX in "${!ARM_LABELS[@]}"; do
  ARM_LABEL="${ARM_LABELS[$INDEX]}"
  ARM_DIR="$RUN_ROOT/${ARM_DIRECTORIES[$INDEX]}"
  mkdir -p "$ARM_DIR"
  (
    cd "$REPO"
    CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 "$PYTHON" \
        -m digit_writing.protocol3_movement_interval_lr_scratch run-arm \
        --repository-root "$REPO" \
        --experiment-config "$CONFIG" \
        --run-root "$RUN_ROOT" \
        --evidence-root "$EVIDENCE_ROOT" \
        --arm-label "$ARM_LABEL"
  ) > "$ARM_DIR/run.log" 2>&1 &
  PIDS+=("$!")
done

FAILED=0
: > "$EVIDENCE_ROOT/process_exit_codes.txt"
for INDEX in "${!PIDS[@]}"; do
  EXIT_CODE=0
  wait "${PIDS[$INDEX]}" || EXIT_CODE=$?
  printf '%s=%s\n' "${ARM_LABELS[$INDEX]}" "$EXIT_CODE" >> "$EVIDENCE_ROOT/process_exit_codes.txt"
  if [[ "$EXIT_CODE" -ne 0 ]]; then FAILED=1; fi
done
if [[ "$FAILED" -ne 0 ]]; then
  echo "ABORT: one or more twelve-arm scratch workers failed; outputs are preserved at $RUN_ROOT" >&2
  cat "$EVIDENCE_ROOT/process_exit_codes.txt" >&2
  for INDEX in "${!ARM_LABELS[@]}"; do
    ARM_DIR="$RUN_ROOT/${ARM_DIRECTORIES[$INDEX]}"
    if [[ ! -f "$ARM_DIR/run_summary.json" ]]; then
      echo "===== FAILED ARM ${ARM_LABELS[$INDEX]} =====" >&2
      tail -n 100 "$ARM_DIR/run.log" >&2
    fi
  done
  exit 1
fi

if ! (
  cd "$REPO"
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 "$PYTHON" \
      -m digit_writing.protocol3_movement_interval_lr_scratch summarize \
      --repository-root "$REPO" \
      --experiment-config "$CONFIG" \
      --run-root "$RUN_ROOT"
) > "$EVIDENCE_ROOT/summarize.log" 2>&1; then
  echo 'ABORT: twelve-arm scratch summarize failed' >&2
  tail -n 100 "$EVIDENCE_ROOT/summarize.log" >&2
  exit 1
fi

"$PYTHON" - "$RUN_ROOT/twelve_arm_summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
assert summary["digits"] == [0, 8]
assert summary["source_movement_intervals"] == {"0": 170, "8": 200}
assert summary["movement_interval_arms"] == {
    "0": [170, 200, 220],
    "8": [200, 220, 240],
}
assert summary["completed_arms"] == 12
assert summary["completed_updates_per_arm"] == 6000
assert summary["from_scratch"] is True
assert summary["fixed_learning_rate_no_schedule_verified"] is True
assert summary["engineering_passed"] is True
assert summary["automatic_interval_selection"] is False
assert summary["automatic_learning_rate_selection"] is False
assert summary["automatic_continuation"] is False
assert summary["automatic_second_seed"] is False
assert summary["automatic_loss_change"] is False
assert summary["automatic_geometry_change"] is False
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
echo 'COMPLETED_ARMS=12'
echo 'COMPLETED_UPDATES_PER_ARM=6000'
echo 'FROM_SCRATCH=1'
echo 'FIXED_LR_NO_SCHEDULE=1'
echo 'DIGIT0_SOURCE_INTERVALS=170'
echo 'DIGIT8_SOURCE_INTERVALS=200'
echo 'DIGIT0_MOVEMENT_INTERVAL_ARMS=170,200,220'
echo 'DIGIT8_MOVEMENT_INTERVAL_ARMS=200,220,240'
echo 'AUTOMATIC_INTERVAL_SELECTION_STARTED=0'
echo 'AUTOMATIC_LEARNING_RATE_SELECTION_STARTED=0'
echo 'AUTOMATIC_CONTINUATION_STARTED=0'
echo 'AUTOMATIC_SECOND_SEED_STARTED=0'
echo 'AUTOMATIC_LOSS_CHANGE_STARTED=0'
echo 'AUTOMATIC_GEOMETRY_CHANGE_STARTED=0'
echo 'FORMAL_FULL10_STARTED=0'
