#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: bash server/run_digit_writing_original_protocol3_ten_digit_corner_ease_v3_parallel.sh REPO CONFIG EXPECTED_HEAD" >&2
  exit 2
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
  "$REPO"/configurations/digit_writing_original_protocol3_ten_digit_corner_ease_overfit_v3.json) ;;
  *) echo "ABORT: config is not the checked-in corner-ease config" >&2; exit 1 ;;
esac

readarray -t CONFIG_VALUES < <("$PYTHON" - "$CONFIG" <<'PY'
import json
import sys
config = json.load(open(sys.argv[1], encoding="utf-8"))
assert config["variant"] == "ten_digit_corner_ease_v3_overfit6000"
assert config["timing_mode"] == "fixed_segment_timing_corner_ease_v3"
assert config["selected_reference_steps"] == 100
assert config["training"] == {
    "batch_size": 8,
    "max_updates": 6000,
    "validation_interval": 100,
}
assert [case["digit"] for case in config["cases"]] == list(range(10))
assert all(case["direction_index"] == 0 for case in config["cases"])
assert all(case["delay_steps"] == 50 for case in config["cases"])
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
  "$REPO"/runs/digit_writing_original_protocol3/scale2p50/ref100/corner_ease_v3/single_digit_overfit) ;;
  *) echo "ABORT: corner-ease output identity is invalid" >&2; exit 1 ;;
esac
for path in "$RUN_ROOT" "$EVIDENCE_ROOT" "$ARCHIVE" "$ARCHIVE_HASH"; do
  if [[ -e "$path" ]]; then
    echo "ABORT: output already exists: $path" >&2
    exit 1
  fi
done

AVAILABLE_CPUS="$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)"
if (( AVAILABLE_CPUS < 10 )); then
  echo "ABORT: ten-way parallel execution requires nproc >= 10" >&2
  exit 1
fi
CPU_MAX="$(cat /sys/fs/cgroup/cpu.max 2>/dev/null || true)"
if [[ -n "$CPU_MAX" ]]; then
  read -r CPU_QUOTA CPU_PERIOD <<<"$CPU_MAX"
  if [[ "$CPU_QUOTA" != "max" ]] && (( CPU_QUOTA / CPU_PERIOD < 10 )); then
    echo "ABORT: at least ten CPU-equivalent cores are required" >&2
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
      tests.test_protocol3_corner_ease
) 2>&1 | tee "$TEST_LOG"
TEST_CODES=("${PIPESTATUS[@]}")
set -e
test "${TEST_CODES[0]}" -eq 0
test "${TEST_CODES[1]}" -eq 0
grep -q '^OK$' "$TEST_LOG"
if grep -q 'skipped=' "$TEST_LOG"; then
  echo 'ABORT: the corner-ease server test set contains skipped tests.' >&2
  exit 1
fi

(
  cd "$REPO"
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 "$PYTHON" \
      -m digit_writing.protocol3_corner_ease prepare \
      --repository-root "$REPO" \
      --protocol-config "$CONFIG" \
      --output-directory "$RUN_ROOT" \
      --evidence-directory "$EVIDENCE_ROOT"
) > "$EVIDENCE_ROOT/prepare.log" 2>&1

grep '^Ran [0-9][0-9]* tests in ' "$TEST_LOG" > "$EVIDENCE_ROOT/test_summary.txt"
printf 'SKIPPED_TESTS=0\n' >> "$EVIDENCE_ROOT/test_summary.txt"
printf '%s\n' "$EXPECTED_HEAD" > "$EVIDENCE_ROOT/repository_head.txt"
printf '%s\n' "$SUBMODULE" > "$EVIDENCE_ROOT/submodule_head.txt"
"$PYTHON" -m pip freeze > "$EVIDENCE_ROOT/environment.freeze.txt"
printf '%s\n' "$CPU_MAX" > "$EVIDENCE_ROOT/cpu.max.txt"
printf '%s\n' "$AVAILABLE_CPUS" > "$EVIDENCE_ROOT/nproc.txt"
lscpu > "$EVIDENCE_ROOT/lscpu.txt"

CASES=(
  digit0_corner_ease digit1_corner_ease digit2_corner_ease
  digit3_corner_ease digit4_corner_ease digit5_corner_ease
  digit6_corner_ease digit7_corner_ease digit8_corner_ease
  digit9_corner_ease
)
PIDS=()
for INDEX in "${!CASES[@]}"; do
  CASE_LABEL="${CASES[$INDEX]}"
  CASE_DIR="$RUN_ROOT/digit${INDEX}/seed42"
  mkdir -p "$CASE_DIR"
  (
    cd "$REPO"
    CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 "$PYTHON" \
        -m digit_writing.protocol3_corner_ease run-case \
        --repository-root "$REPO" \
        --protocol-config "$CONFIG" \
        --run-root "$RUN_ROOT" \
        --evidence-directory "$EVIDENCE_ROOT" \
        --case-label "$CASE_LABEL"
  ) > "$CASE_DIR/run.log" 2>&1 &
  PIDS+=("$!")
done

FAILED=0
: > "$EVIDENCE_ROOT/process_exit_codes.txt"
for INDEX in "${!PIDS[@]}"; do
  EXIT_CODE=0
  wait "${PIDS[$INDEX]}" || EXIT_CODE=$?
  printf '%s=%s\n' "${CASES[$INDEX]}" "$EXIT_CODE" >> "$EVIDENCE_ROOT/process_exit_codes.txt"
  if [[ "$EXIT_CODE" -ne 0 ]]; then FAILED=1; fi
done
if [[ "$FAILED" -ne 0 ]]; then
  echo "ABORT: one or more corner-ease cases failed; outputs are preserved at $RUN_ROOT" >&2
  exit 1
fi

(
  cd "$REPO"
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 "$PYTHON" \
      -m digit_writing.protocol3_corner_ease summarize \
      --repository-root "$REPO" \
      --protocol-config "$CONFIG" \
      --run-root "$RUN_ROOT"
) > "$EVIDENCE_ROOT/summarize.log" 2>&1

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
echo 'COMPLETED_CASES=10'
echo 'AUTOMATIC_EXTENSION_STARTED=0'
echo 'AUTOMATIC_SECOND_SEED_STARTED=0'
echo 'FORMAL_FULL10_STARTED=0'
