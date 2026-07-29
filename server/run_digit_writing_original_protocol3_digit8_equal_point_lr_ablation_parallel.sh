#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 9 ]]; then
  echo "Usage: bash server/run_digit_writing_original_protocol3_digit8_equal_point_lr_ablation_parallel.sh REPO CONFIG SOURCE_SUMMARY SOURCE_CHECKPOINT SOURCE_OVERLAY_CSV SOURCE_OVERLAY_PNG GATE1_SUMMARY EXPECTED_HEAD EXPECTED_SOURCE_HEAD" >&2
  exit 2
fi
if [[ "${PROTOCOL3_DIGIT8_EQUAL_POINT_LR_ABLATION_AUTHORIZED:-0}" != "1" ]]; then
  echo 'ABORT: explicit Protocol3 digit8 LR-ablation approval is required' >&2
  exit 1
fi

REPO="$(realpath "$1")"
CONFIG="$(realpath "$2")"
SOURCE_SUMMARY="$(realpath "$3")"
SOURCE_CHECKPOINT="$(realpath "$4")"
SOURCE_OVERLAY_CSV="$(realpath "$5")"
SOURCE_OVERLAY_PNG="$(realpath "$6")"
GATE1_SUMMARY="$(realpath "$7")"
EXPECTED_HEAD="$8"
EXPECTED_SOURCE_HEAD="$9"
PYTHON=/root/autodl-tmp/conda/envs/motor-compat-cpu/bin/python
SUBMODULE=ac0c4f589eae37bbde63968912925de99232e306
SOURCE_CHECKPOINT_SHA=45373d9db856afa08652ed700081d23a375f5d772499e2ae56efc89423088c64
SOURCE_SUMMARY_SHA=991385a6862af4ddff99101c8704fb07da31c239e1b5c6711e7b9d41fcaf0112

[[ "$EXPECTED_HEAD" =~ ^[0-9a-f]{40}$ ]]
test "$EXPECTED_SOURCE_HEAD" = 792c4d3d20ae6ec0c5deef4f06bd41a0859730ba
test -d "$REPO/.git"
test -x "$PYTHON"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
test "$(git -C "$REPO" rev-parse HEAD:mRNNTorch)" = "$SUBMODULE"
test "$(git -C "$REPO/mRNNTorch" rev-parse HEAD)" = "$SUBMODULE"
test -z "$(git -C "$REPO" status --short)"
test -z "$(git -C "$REPO/mRNNTorch" status --short)"
test "$(cat /root/autodl-tmp/conda/envs/motor-compat-cpu/.compatibility_environment_complete)" = "device=cpu"
test "$(sha256sum "$SOURCE_CHECKPOINT" | cut -d' ' -f1)" = "$SOURCE_CHECKPOINT_SHA"
test "$(sha256sum "$SOURCE_SUMMARY" | cut -d' ' -f1)" = "$SOURCE_SUMMARY_SHA"
case "$CONFIG" in
  "$REPO"/configurations/digit_writing_original_protocol3_digit8_equal_point_lr_ablation.json) ;;
  *) echo 'ABORT: config is not the checked-in digit8 LR-ablation config' >&2; exit 1 ;;
esac

readarray -t CONFIG_VALUES < <("$PYTHON" - "$CONFIG" <<'PY'
import json
import sys
config = json.load(open(sys.argv[1], encoding="utf-8"))
assert config["run_kind"] == "protocol3_digit8_equal_point_lr_ablation"
assert config["source"]["completed_updates"] == 6000
assert config["training"]["target_completed_updates"] == 7000
assert config["training"]["fixed_target_no_early_stop"] is True
assert config["objective"]["movement_points_equal_weight"] is True
assert config["objective"]["start_point_extra_weight"] == 0.0
assert config["objective"]["endpoint_extra_weight"] == 0.0
assert config["objective"]["closure_extra_weight"] == 0.0
assert [(arm["label"], arm["learning_rate"]) for arm in config["arms"]] == [
    ("digit8_medium_lr1e3", 0.001),
    ("digit8_medium_lr3e4", 0.0003),
    ("digit8_medium_lr1e4", 0.0001),
]
assert config["decision"]["automatic_further_continuation"] is False
assert config["decision"]["automatic_reference_fallback"] is False
assert config["decision"]["formal_full10_start"] is False
print(config["output"]["directory"])
PY
)
RUN_ROOT="$(realpath -m "$REPO/${CONFIG_VALUES[0]}")"
ARCHIVE="${RUN_ROOT}.tar.gz"
ARCHIVE_HASH="${ARCHIVE}.sha256"
TEST_LOG="${RUN_ROOT}.tests.log"
SOURCE_INSPECTION_LOG="${RUN_ROOT}.source-inspection.json"
case "$RUN_ROOT" in
  "$REPO"/runs/digit_writing_original_protocol3/scale2p50/ref100/digit8_equal_point_lr_from6000_to7000) ;;
  *) echo 'ABORT: digit8 LR-ablation output identity is invalid' >&2; exit 1 ;;
esac
for path in "$RUN_ROOT" "$ARCHIVE" "$ARCHIVE_HASH" "$TEST_LOG" "$SOURCE_INSPECTION_LOG"; do
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
assert summary["timing_passed"] is True
assert summary["git_identity"]["repository_head"] == sys.argv[2]
PY

AVAILABLE_CPUS="$(nproc)"
if (( AVAILABLE_CPUS < 3 )); then
  echo 'ABORT: three-way parallel execution requires nproc >= 3' >&2
  exit 1
fi
CPU_MAX="$(cat /sys/fs/cgroup/cpu.max 2>/dev/null || true)"
if [[ -n "$CPU_MAX" ]]; then
  read -r CPU_QUOTA CPU_PERIOD <<<"$CPU_MAX"
  if [[ "$CPU_QUOTA" != "max" ]] && (( CPU_QUOTA / CPU_PERIOD < 3 )); then
    echo 'ABORT: at least three CPU-equivalent cores are required' >&2
    exit 1
  fi
fi

mkdir -p "$(dirname "$RUN_ROOT")"
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
      tests.test_protocol3_digit8_lr_ablation
) 2>&1 | tee "$TEST_LOG"
TEST_CODES=("${PIPESTATUS[@]}")
set -e
test "${TEST_CODES[0]}" -eq 0
test "${TEST_CODES[1]}" -eq 0
grep -q '^OK$' "$TEST_LOG"
if grep -q 'skipped=' "$TEST_LOG"; then
  echo 'ABORT: the digit8 LR-ablation server test set contains skipped tests' >&2
  exit 1
fi

COMMON_ARGS=(
  --repository-root "$REPO"
  --ablation-config "$CONFIG"
  --source-summary "$SOURCE_SUMMARY"
  --source-checkpoint "$SOURCE_CHECKPOINT"
  --source-overlay-csv "$SOURCE_OVERLAY_CSV"
  --source-overlay-png "$SOURCE_OVERLAY_PNG"
)
(
  cd "$REPO"
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 "$PYTHON" \
      -m digit_writing.protocol3_digit8_lr_ablation inspect \
      "${COMMON_ARGS[@]}"
) > "$SOURCE_INSPECTION_LOG"
(
  cd "$REPO"
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 "$PYTHON" \
      -m digit_writing.protocol3_digit8_lr_ablation prepare \
      "${COMMON_ARGS[@]}" --run-root "$RUN_ROOT"
)
mv "$TEST_LOG" "$RUN_ROOT/tests.log"
mv "$SOURCE_INSPECTION_LOG" "$RUN_ROOT/source_inspection.json"
grep '^Ran [0-9][0-9]* tests in ' "$RUN_ROOT/tests.log" > "$RUN_ROOT/test_summary.txt"
printf 'SKIPPED_TESTS=0\n' >> "$RUN_ROOT/test_summary.txt"
printf '%s\n' "$EXPECTED_HEAD" > "$RUN_ROOT/parent_head.txt"
printf '%s\n' "$EXPECTED_SOURCE_HEAD" > "$RUN_ROOT/source_head.txt"
printf '%s\n' "$SUBMODULE" > "$RUN_ROOT/submodule_head.txt"
"$PYTHON" -m pip freeze > "$RUN_ROOT/environment.freeze.txt"
printf '%s\n' "$CPU_MAX" > "$RUN_ROOT/cpu.max.txt"
printf '%s\n' "$AVAILABLE_CPUS" > "$RUN_ROOT/nproc.txt"
lscpu > "$RUN_ROOT/lscpu.txt"

ARMS=(digit8_medium_lr1e3 digit8_medium_lr3e4 digit8_medium_lr1e4)
PIDS=()
for ARM_LABEL in "${ARMS[@]}"; do
  (
    cd "$REPO"
    CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 "$PYTHON" \
        -m digit_writing.protocol3_digit8_lr_ablation run-arm \
        "${COMMON_ARGS[@]}" --run-root "$RUN_ROOT" \
        --arm-label "$ARM_LABEL" --manual-ablation-authorized
  ) > "$RUN_ROOT/${ARM_LABEL}.console.log" 2>&1 &
  PIDS+=("$!")
done

FAILED=0
: > "$RUN_ROOT/process_exit_codes.txt"
for INDEX in "${!PIDS[@]}"; do
  EXIT_CODE=0
  wait "${PIDS[$INDEX]}" || EXIT_CODE=$?
  printf '%s=%s\n' "${ARMS[$INDEX]}" "$EXIT_CODE" >> "$RUN_ROOT/process_exit_codes.txt"
  if [[ "$EXIT_CODE" -ne 0 ]]; then FAILED=1; fi
done
if [[ "$FAILED" -ne 0 ]]; then
  echo "ABORT: one or more digit8 LR arms failed; outputs are preserved at $RUN_ROOT" >&2
  exit 1
fi

(
  cd "$REPO"
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 "$PYTHON" \
      -m digit_writing.protocol3_digit8_lr_ablation summarize \
      --repository-root "$REPO" --ablation-config "$CONFIG" \
      --run-root "$RUN_ROOT"
) > "$RUN_ROOT/summarize.log" 2>&1

"$PYTHON" - "$RUN_ROOT/digit8_equal_point_lr_ablation_summary.json" <<'PY'
import json
import sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
assert summary["engineering_passed"] is True
assert [arm["completed_updates"] for arm in summary["arms"]] == [7000, 7000, 7000]
assert [arm["effective_optimizer_param_group_learning_rates"] for arm in summary["arms"]] == [
    [0.001], [0.0003], [0.0001]
]
PY
grep -Fxq 'digit8_medium_lr1e3=0' "$RUN_ROOT/process_exit_codes.txt"
grep -Fxq 'digit8_medium_lr3e4=0' "$RUN_ROOT/process_exit_codes.txt"
grep -Fxq 'digit8_medium_lr1e4=0' "$RUN_ROOT/process_exit_codes.txt"
test -z "$(git -C "$REPO" status --short)"
test -z "$(git -C "$REPO/mRNNTorch" status --short)"
printf 'DIGIT8_EQUAL_POINT_LR_ABLATION_COMPLETE=1\n' > "$RUN_ROOT/RUN_COMPLETE.txt"
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

readarray -t SUMMARY_VALUES < <("$PYTHON" - "$RUN_ROOT/digit8_equal_point_lr_ablation_summary.json" <<'PY'
import json
import sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
assert summary["completed_updates"] == 7000
assert len(summary["arms"]) == 3
assert summary["movement_points_equal_weight_in_training"] is True
assert summary["extra_start_endpoint_or_closure_weight"] is False
assert summary["automatic_further_continuation"] is False
assert summary["automatic_reference_fallback"] is False
assert summary["formal_full10_started"] is False
print(summary["classification"])
print(int(summary["engineering_passed"]))
PY
)
echo "DIGIT8_LR_ABLATION_CLASSIFICATION=${SUMMARY_VALUES[0]}"
echo "ENGINEERING_PASSED=${SUMMARY_VALUES[1]}"
echo "RUN_ROOT=$RUN_ROOT"
echo "ARCHIVE=$ARCHIVE"
echo "ARCHIVE_HASH=$ARCHIVE_HASH"
echo 'DIGIT8_LR_ABLATION_ARMS=3'
echo 'AUTOMATIC_FURTHER_CONTINUATION_STARTED=0'
echo 'AUTOMATIC_REFERENCE_FALLBACK_STARTED=0'
echo 'FORMAL_FULL10_STARTED=0'
test "${SUMMARY_VALUES[1]}" -eq 1
