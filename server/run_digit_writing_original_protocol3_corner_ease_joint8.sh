#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: bash server/run_digit_writing_original_protocol3_corner_ease_joint8.sh REPO CONFIG EXPECTED_HEAD" >&2
  exit 2
fi
if [[ "${PROTOCOL3_CORNER_EASE_JOINT8_AUTHORIZED:-}" != "1" ]]; then
  echo "ABORT: set PROTOCOL3_CORNER_EASE_JOINT8_AUTHORIZED=1 after explicit authorization" >&2
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
  "$REPO"/configurations/digit_writing_original_protocol3_corner_ease_joint8_excluding_digit0_digit8.json) ;;
  *) echo "ABORT: config is not the checked-in joint8 config" >&2; exit 1 ;;
esac

readarray -t CONFIG_VALUES < <("$PYTHON" - "$CONFIG" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
assert config["variant"] == "corner_ease_v3_joint8_excluding_digit0_digit8_seed42"
assert config["train_digits"] == [1, 2, 3, 4, 5, 6, 7, 9]
assert config["excluded_digits"] == [0, 8]
assert config["direction_index"] == 0
assert config["delay_steps"] == 50
assert config["training"] == {
    "batch_size": 8,
    "updates_per_digit": 8000,
    "max_updates": 64000,
    "validation_interval": 800,
    "samples_per_digit": 64000,
    "total_training_samples": 512000,
}
assert config["optimizer"]["schedule"] == [
    {
        "per_digit_exposure_start": 1,
        "per_digit_exposure_end": 6000,
        "global_update_start": 1,
        "global_update_end": 48000,
        "learning_rate": 0.001,
    },
    {
        "per_digit_exposure_start": 6001,
        "per_digit_exposure_end": 8000,
        "global_update_start": 48001,
        "global_update_end": 64000,
        "learning_rate": 0.0003,
    },
]
assert config["optimizer"]["preserve_adam_state_at_switch"] is True
assert config["corner_ease"]["automatic_extension"] is False
assert config["corner_ease"]["automatic_second_seed"] is False
assert config["corner_ease"]["automatic_full10"] is False
print(config["output"]["directory"])
PY
)
RUN_ROOT="$(realpath -m "$REPO/${CONFIG_VALUES[0]}")"
EVIDENCE_ROOT="${RUN_ROOT}_server_evidence"
ARCHIVE="${RUN_ROOT}.tar.gz"
ARCHIVE_HASH="${ARCHIVE}.sha256"
case "$RUN_ROOT" in
  "$REPO"/runs/digit_writing_original_protocol3/scale2p50/ref100/corner_ease_v3/joint8_excluding_digit0_digit8/seed42) ;;
  *) echo "ABORT: joint8 output identity is invalid" >&2; exit 1 ;;
esac
for path in "$RUN_ROOT" "$EVIDENCE_ROOT" "$ARCHIVE" "$ARCHIVE_HASH"; do
  if [[ -e "$path" ]]; then
    echo "ABORT: output already exists: $path" >&2
    exit 1
  fi
done
if pgrep -af '[p]rotocol3_corner_ease_joint8 run' >/dev/null; then
  echo 'ABORT: another Protocol3 joint8 training process is already running' >&2
  exit 1
fi

PREPARE_LOG="$(mktemp /tmp/protocol3_joint8_prepare.XXXXXX.log)"
set +e
(
  cd "$REPO"
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 "$PYTHON" \
      -m digit_writing.protocol3_corner_ease_joint8 prepare \
      --repository-root "$REPO" \
      --protocol-config "$CONFIG" \
      --evidence-directory "$EVIDENCE_ROOT"
) > "$PREPARE_LOG" 2>&1
PREPARE_CODE=$?
set -e
if [[ "$PREPARE_CODE" -ne 0 ]]; then
  echo "ABORT: joint8 prepare failed; log preserved at $PREPARE_LOG" >&2
  exit "$PREPARE_CODE"
fi
mv "$PREPARE_LOG" "$EVIDENCE_ROOT/prepare.log"

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
      tests.test_protocol3_movement_interval_lr_scratch \
      tests.test_protocol3_corner_ease_joint8
) 2>&1 | tee "$TEST_LOG"
TEST_CODES=("${PIPESTATUS[@]}")
set -e
test "${TEST_CODES[0]}" -eq 0
test "${TEST_CODES[1]}" -eq 0
grep -q '^OK$' "$TEST_LOG"
if grep -q 'skipped=' "$TEST_LOG"; then
  echo 'ABORT: the joint8 server test set contains skipped tests.' >&2
  exit 1
fi

grep '^Ran [0-9][0-9]* tests in ' "$TEST_LOG" > "$EVIDENCE_ROOT/test_summary.txt"
printf 'SKIPPED_TESTS=0\n' >> "$EVIDENCE_ROOT/test_summary.txt"
printf '%s\n' "$EXPECTED_HEAD" > "$EVIDENCE_ROOT/repository_head.txt"
printf '%s\n' "$SUBMODULE" > "$EVIDENCE_ROOT/submodule_head.txt"
"$PYTHON" -m pip freeze > "$EVIDENCE_ROOT/environment.freeze.txt"
env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc > "$EVIDENCE_ROOT/nproc.txt"
cat /sys/fs/cgroup/cpu.max > "$EVIDENCE_ROOT/cpu.max.txt" 2>/dev/null || true
lscpu > "$EVIDENCE_ROOT/lscpu.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$EVIDENCE_ROOT/start_utc.txt"

mkdir -p "$RUN_ROOT"
RUN_LOG="$RUN_ROOT/run.log"
set +e
(
  cd "$REPO"
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 "$PYTHON" \
      -m digit_writing.protocol3_corner_ease_joint8 run \
      --repository-root "$REPO" \
      --protocol-config "$CONFIG" \
      --run-directory "$RUN_ROOT" \
      --evidence-directory "$EVIDENCE_ROOT"
) > "$RUN_LOG" 2>&1
RUN_CODE=$?
set -e
printf 'joint8=%s\n' "$RUN_CODE" > "$EVIDENCE_ROOT/process_exit_codes.txt"
if [[ "$RUN_CODE" -ne 0 ]]; then
  echo "ABORT: joint8 training failed; outputs and run.log are preserved at $RUN_ROOT" >&2
  exit "$RUN_CODE"
fi

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
echo 'COMPLETED_OPTIMIZER_STEPS=64000'
echo 'UPDATES_PER_DIGIT=8000'
echo 'TRAIN_DIGITS=1,2,3,4,5,6,7,9'
echo 'EXCLUDED_DIGITS=0,8'
echo 'AUTOMATIC_EXTENSION_STARTED=0'
echo 'AUTOMATIC_SECOND_SEED_STARTED=0'
echo 'FORMAL_FULL10_STARTED=0'
