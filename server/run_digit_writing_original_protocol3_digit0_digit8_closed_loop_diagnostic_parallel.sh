#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 9 ]]; then
  echo "Usage: PROTOCOL3_CLOSED_LOOP_DIAGNOSTIC_AUTHORIZED=1 bash server/run_digit_writing_original_protocol3_digit0_digit8_closed_loop_diagnostic_parallel.sh REPO CONFIG SOURCE_RUN_ROOT SOURCE_EVIDENCE_ROOT CONTINUATION_RUN_ROOT CONTINUATION_EVIDENCE_ROOT EXPECTED_HEAD EXPECTED_SOURCE_HEAD EXPECTED_CONTINUATION_HEAD" >&2
  exit 2
fi
if [[ "${PROTOCOL3_CLOSED_LOOP_DIAGNOSTIC_AUTHORIZED:-0}" != "1" ]]; then
  echo 'ABORT: explicit Protocol3 closed-loop diagnostic approval is required' >&2
  exit 1
fi

REPO="$(realpath "$1")"
CONFIG="$(realpath "$2")"
SOURCE_RUN_ROOT="$(realpath "$3")"
SOURCE_EVIDENCE_ROOT="$(realpath "$4")"
CONTINUATION_RUN_ROOT="$(realpath "$5")"
CONTINUATION_EVIDENCE_ROOT="$(realpath "$6")"
EXPECTED_HEAD="$7"
EXPECTED_SOURCE_HEAD="$8"
EXPECTED_CONTINUATION_HEAD="$9"
PYTHON=/root/autodl-tmp/conda/envs/motor-compat-cpu/bin/python
SUBMODULE=ac0c4f589eae37bbde63968912925de99232e306
FROZEN_SOURCE_HEAD=ee5a1900a33f8a8b7921fd9fc5b09aeed6600925
FROZEN_CONTINUATION_HEAD=76349d51222eed62bdb26d78d251b7d57760f2e8

[[ "$EXPECTED_HEAD" =~ ^[0-9a-f]{40}$ ]]
test "$EXPECTED_SOURCE_HEAD" = "$FROZEN_SOURCE_HEAD"
test "$EXPECTED_CONTINUATION_HEAD" = "$FROZEN_CONTINUATION_HEAD"
test -d "$REPO/.git"
test -x "$PYTHON"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
test "$(git -C "$REPO" rev-parse HEAD:mRNNTorch)" = "$SUBMODULE"
test "$(git -C "$REPO/mRNNTorch" rev-parse HEAD)" = "$SUBMODULE"
test -z "$(git -C "$REPO" status --short)"
test -z "$(git -C "$REPO/mRNNTorch" status --short)"
test "$(cat /root/autodl-tmp/conda/envs/motor-compat-cpu/.compatibility_environment_complete)" = "device=cpu"
case "$CONFIG" in
  "$REPO"/configurations/digit_writing_original_protocol3_digit0_digit8_closed_loop_diagnostic.json) ;;
  *) echo 'ABORT: config is not the checked-in closed-loop diagnostic config' >&2; exit 1 ;;
esac
case "$SOURCE_RUN_ROOT" in
  "$REPO"/runs/digit_writing_original_protocol3/scale2p50/ref100/corner_ease_v3/single_digit_overfit) ;;
  *) echo 'ABORT: source result root identity is invalid' >&2; exit 1 ;;
esac
case "$SOURCE_EVIDENCE_ROOT" in
  "$REPO"/runs/digit_writing_original_protocol3/scale2p50/ref100/corner_ease_v3/single_digit_overfit_server_evidence) ;;
  *) echo 'ABORT: source evidence root identity is invalid' >&2; exit 1 ;;
esac
case "$CONTINUATION_RUN_ROOT" in
  "$REPO"/runs/digit_writing_original_protocol3/scale2p50/ref100/corner_ease_v3/lr_continuation_from6000_to8000) ;;
  *) echo 'ABORT: continuation result root identity is invalid' >&2; exit 1 ;;
esac
case "$CONTINUATION_EVIDENCE_ROOT" in
  "$REPO"/runs/digit_writing_original_protocol3/scale2p50/ref100/corner_ease_v3/lr_continuation_from6000_to8000_server_evidence) ;;
  *) echo 'ABORT: continuation evidence root identity is invalid' >&2; exit 1 ;;
esac
test "$(cat "$SOURCE_EVIDENCE_ROOT/repository_head.txt")" = "$FROZEN_SOURCE_HEAD"
test "$(cat "$CONTINUATION_EVIDENCE_ROOT/repository_head.txt")" = "$FROZEN_CONTINUATION_HEAD"
test "$(cat "$CONTINUATION_EVIDENCE_ROOT/source_repository_head.txt")" = "$FROZEN_SOURCE_HEAD"
test "$(cat "$SOURCE_EVIDENCE_ROOT/submodule_head.txt")" = "$SUBMODULE"
test "$(cat "$CONTINUATION_EVIDENCE_ROOT/submodule_head.txt")" = "$SUBMODULE"
for DIRECTORY in \
  "$SOURCE_RUN_ROOT" \
  "$SOURCE_EVIDENCE_ROOT" \
  "$CONTINUATION_RUN_ROOT" \
  "$CONTINUATION_EVIDENCE_ROOT"; do
  (
    cd "$DIRECTORY"
    sha256sum -c SHA256SUMS
  )
done

readarray -t CONFIG_VALUES < <("$PYTHON" - "$CONFIG" <<'PY'
import json
import sys
config = json.load(open(sys.argv[1], encoding="utf-8"))
assert config["protocol"] == "digit_writing_original_protocol3"
assert config["run_kind"] == "protocol3_digit0_digit8_closed_loop_diagnostic"
assert config["variant"] == "digit0_digit8_joint_zero_step_v1"
assert config["source"]["repository_head"] == "ee5a1900a33f8a8b7921fd9fc5b09aeed6600925"
assert config["continuation"]["repository_head"] == "76349d51222eed62bdb26d78d251b7d57760f2e8"
assert config["mrnntorch_head"] == "ac0c4f589eae37bbde63968912925de99232e306"
assert [row["label"] for row in config["checkpoints"]] == [
    "d0_source_best4800",
    "d0_source_final6000",
    "d0_lr1e3_final8000",
    "d0_lr3e4_final8000",
    "d8_source_best5800",
    "d8_source_final6000",
    "d9_positive_control_best4800",
]
assert [row["digit"] for row in config["checkpoints"]] == [0, 0, 0, 0, 8, 8, 9]
assert config["audit"]["optimizer_steps"] == 0
assert config["audit"]["network_noise"] is False
assert config["audit"]["deterministic_observations"] is True
assert config["decision"]["automatic_root_cause_selection"] is False
assert config["decision"]["automatic_stage_b_start"] is False
assert config["decision"]["automatic_training"] is False
assert config["decision"]["formal_full10_start"] is False
print(config["output"]["directory"])
PY
)
RUN_ROOT="$(realpath -m "$REPO/${CONFIG_VALUES[0]}")"
EVIDENCE_ROOT="${RUN_ROOT}_server_evidence"
ARCHIVE="${RUN_ROOT}.tar.gz"
ARCHIVE_HASH="${ARCHIVE}.sha256"
case "$RUN_ROOT" in
  "$REPO"/runs/digit_writing_original_protocol3/scale2p50/ref100/corner_ease_v3/closed_loop_digit0_digit8_diagnostic) ;;
  *) echo 'ABORT: closed-loop output identity is invalid' >&2; exit 1 ;;
esac
for path in "$RUN_ROOT" "$EVIDENCE_ROOT" "$ARCHIVE" "$ARCHIVE_HASH"; do
  if [[ -e "$path" ]]; then
    echo "ABORT: output already exists: $path" >&2
    exit 1
  fi
done

AVAILABLE_CPUS="$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)"
if (( AVAILABLE_CPUS < 7 )); then
  echo 'ABORT: seven-way diagnostic requires nproc >= 7' >&2
  exit 1
fi
CPU_MAX="$(cat /sys/fs/cgroup/cpu.max 2>/dev/null || true)"
if [[ -n "$CPU_MAX" ]]; then
  read -r CPU_QUOTA CPU_PERIOD <<<"$CPU_MAX"
  if [[ "$CPU_QUOTA" != "max" ]] && (( CPU_QUOTA / CPU_PERIOD < 7 )); then
    echo 'ABORT: at least seven CPU-equivalent cores are required' >&2
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
      tests.test_protocol3_closed_loop_diagnostic
) 2>&1 | tee "$TEST_LOG"
TEST_CODES=("${PIPESTATUS[@]}")
set -e
test "${TEST_CODES[0]}" -eq 0
test "${TEST_CODES[1]}" -eq 0
grep -q '^OK$' "$TEST_LOG"
if grep -q 'skipped=' "$TEST_LOG"; then
  echo 'ABORT: the closed-loop server test set contains skipped tests' >&2
  exit 1
fi

(
  cd "$REPO"
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 "$PYTHON" \
      -m digit_writing.protocol3_closed_loop_diagnostic prepare \
      --repository-root "$REPO" \
      --diagnostic-config "$CONFIG" \
      --source-run-root "$SOURCE_RUN_ROOT" \
      --source-evidence-root "$SOURCE_EVIDENCE_ROOT" \
      --continuation-run-root "$CONTINUATION_RUN_ROOT" \
      --continuation-evidence-root "$CONTINUATION_EVIDENCE_ROOT" \
      --run-root "$RUN_ROOT"
) > "$EVIDENCE_ROOT/prepare.log" 2>&1

grep '^Ran [0-9][0-9]* tests in ' "$TEST_LOG" > "$EVIDENCE_ROOT/test_summary.txt"
printf 'SKIPPED_TESTS=0\n' >> "$EVIDENCE_ROOT/test_summary.txt"
printf '%s\n' "$EXPECTED_HEAD" > "$EVIDENCE_ROOT/repository_head.txt"
printf '%s\n' "$FROZEN_SOURCE_HEAD" > "$EVIDENCE_ROOT/source_repository_head.txt"
printf '%s\n' "$FROZEN_CONTINUATION_HEAD" > "$EVIDENCE_ROOT/continuation_repository_head.txt"
printf '%s\n' "$SUBMODULE" > "$EVIDENCE_ROOT/submodule_head.txt"
"$PYTHON" -m pip freeze > "$EVIDENCE_ROOT/environment.freeze.txt"
printf '%s\n' "$CPU_MAX" > "$EVIDENCE_ROOT/cpu.max.txt"
printf '%s\n' "$AVAILABLE_CPUS" > "$EVIDENCE_ROOT/nproc.txt"
lscpu > "$EVIDENCE_ROOT/lscpu.txt"

LABELS=(
  d0_source_best4800
  d0_source_final6000
  d0_lr1e3_final8000
  d0_lr3e4_final8000
  d8_source_best5800
  d8_source_final6000
  d9_positive_control_best4800
)
PIDS=()
for LABEL in "${LABELS[@]}"; do
  WORKER_ROOT="$RUN_ROOT/per_checkpoint/$LABEL"
  mkdir -p "$WORKER_ROOT"
  (
    cd "$REPO"
    CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 "$PYTHON" \
        -m digit_writing.protocol3_closed_loop_diagnostic run-checkpoint \
        --repository-root "$REPO" \
        --diagnostic-config "$CONFIG" \
        --source-run-root "$SOURCE_RUN_ROOT" \
        --continuation-run-root "$CONTINUATION_RUN_ROOT" \
        --run-root "$RUN_ROOT" \
        --checkpoint-label "$LABEL" \
        --manual-diagnostic-authorized
  ) > "$WORKER_ROOT/run.log" 2>&1 &
  PIDS+=("$!")
done

FAILED=0
: > "$EVIDENCE_ROOT/process_exit_codes.txt"
for INDEX in "${!PIDS[@]}"; do
  EXIT_CODE=0
  wait "${PIDS[$INDEX]}" || EXIT_CODE=$?
  printf '%s=%s\n' "${LABELS[$INDEX]}" "$EXIT_CODE" >> "$EVIDENCE_ROOT/process_exit_codes.txt"
  if [[ "$EXIT_CODE" -ne 0 ]]; then FAILED=1; fi
done
if [[ "$FAILED" -ne 0 ]]; then
  echo "ABORT: one or more closed-loop workers failed; outputs are preserved at $RUN_ROOT" >&2
  exit 1
fi

(
  cd "$REPO"
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 "$PYTHON" \
      -m digit_writing.protocol3_closed_loop_diagnostic summarize \
      --repository-root "$REPO" \
      --diagnostic-config "$CONFIG" \
      --run-root "$RUN_ROOT"
) > "$EVIDENCE_ROOT/summarize.log" 2>&1

"$PYTHON" - "$RUN_ROOT/closed_loop_diagnostic_summary.json" <<'PY'
import json
import sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
assert summary["completed_checkpoints"] == 7
assert summary["optimizer_steps_during_audit"] == 0
assert summary["all_model_states_unchanged"] is True
assert summary["all_parameter_gradients_absent"] is True
assert summary["engineering_passed"] is True
assert summary["automatic_root_cause_selection"] is False
assert summary["automatic_stage_b_start"] is False
assert summary["automatic_training"] is False
assert summary["automatic_geometry_change"] is False
assert summary["automatic_loss_change"] is False
assert summary["automatic_second_seed"] is False
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
echo 'COMPLETED_CHECKPOINTS=7'
echo 'OPTIMIZER_STEPS=0'
echo 'AUTOMATIC_ROOT_CAUSE_SELECTION_STARTED=0'
echo 'AUTOMATIC_STAGE_B_STARTED=0'
echo 'AUTOMATIC_TRAINING_STARTED=0'
echo 'AUTOMATIC_GEOMETRY_CHANGE_STARTED=0'
echo 'AUTOMATIC_LOSS_CHANGE_STARTED=0'
echo 'AUTOMATIC_SECOND_SEED_STARTED=0'
echo 'FORMAL_FULL10_STARTED=0'
