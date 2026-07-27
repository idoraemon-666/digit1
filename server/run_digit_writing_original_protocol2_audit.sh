#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash server/run_digit_writing_original_protocol2_audit.sh REPO RUN_ROOT" >&2
  exit 2
fi

REPO="$(realpath "$1")"
RUN_ROOT="$(realpath -m "$2")"
PYTHON=/root/autodl-tmp/conda/envs/motor-compat-cpu/bin/python
BASELINE=105cd0c6b153f0e80a2593e43b7ffec7cc1f5e33
SUBMODULE=ac0c4f589eae37bbde63968912925de99232e306
GEOMETRY_CONFIG="$REPO/configurations/digit_writing_original_protocol2_geometry.json"
BASE_CONFIG="$REPO/configurations/digit_writing_original_protocol2_full10_dev42.json"
ARCHIVE="${RUN_ROOT}.tar.gz"
ARCHIVE_HASH="${ARCHIVE}.sha256"
START_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

test -d "$REPO/.git"
test -x "$PYTHON"
test -f "$GEOMETRY_CONFIG"
test -f "$BASE_CONFIG"
test "$(cat /root/autodl-tmp/conda/envs/motor-compat-cpu/.compatibility_environment_complete)" = "device=cpu"

for path in "$RUN_ROOT" "$ARCHIVE" "$ARCHIVE_HASH"; do
  if [[ -e "$path" ]]; then
    echo "ABORT: output already exists: $path" >&2
    exit 1
  fi
done

PARENT_HEAD="$(git -C "$REPO" rev-parse HEAD)"
SUBMODULE_HEAD="$(git -C "$REPO/mRNNTorch" rev-parse HEAD)"
test "$SUBMODULE_HEAD" = "$SUBMODULE"
git -C "$REPO" merge-base --is-ancestor "$BASELINE" "$PARENT_HEAD"
test -z "$(git -C "$REPO" status --short)"
test -z "$(git -C "$REPO/mRNNTorch" status --short)"

mkdir -p "$RUN_ROOT"
printf '%s\n' "$PARENT_HEAD" > "$RUN_ROOT/parent_head.txt"
printf '%s\n' "$SUBMODULE_HEAD" > "$RUN_ROOT/submodule_head.txt"
printf '%s\n' "$BASELINE" > "$RUN_ROOT/original_baseline.txt"
cp "$GEOMETRY_CONFIG" "$RUN_ROOT/geometry_configuration.json"
cp "$BASE_CONFIG" "$RUN_ROOT/base_configuration.json"
sha256sum "$RUN_ROOT/geometry_configuration.json" "$RUN_ROOT/base_configuration.json" \
  > "$RUN_ROOT/configuration.sha256"
(
  cd "$REPO"
  git ls-files | while IFS= read -r path; do
    if [[ -f "$path" ]]; then
      sha256sum "$path"
    fi
  done
) > "$RUN_ROOT/repository_tracked_sha256.txt"
{
  printf 'START_UTC=%s\n' "$START_UTC"
  printf 'PARENT_HEAD=%s\n' "$PARENT_HEAD"
  printf 'SUBMODULE_HEAD=%s\n' "$SUBMODULE_HEAD"
  printf 'DEVICE=cpu\n'
  "$PYTHON" --version
  "$PYTHON" -m pip check
  "$PYTHON" -c "import motornet,numpy,torch; print(torch.__version__,motornet.__version__,numpy.__version__,torch.cuda.is_available())"
} 2>&1 | tee "$RUN_ROOT/preflight.txt"
"$PYTHON" -m pip freeze > "$RUN_ROOT/environment.freeze.txt"

set +e
(
  cd "$REPO"
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 "$PYTHON" -m unittest -v \
      tests.test_digit_geometry \
      tests.test_phase_normalized_loss \
      tests.test_digit_environment \
      tests.test_geometry_audit \
      tests.test_phase_d_protocol
) 2>&1 | tee "$RUN_ROOT/tests.log"
TEST_CODES=("${PIPESTATUS[@]}")
set -e
TEST_EXIT="${TEST_CODES[0]}"
TEST_TEE_EXIT="${TEST_CODES[1]}"
printf 'TEST_EXIT=%s\nTEST_TEE_EXIT=%s\n' "$TEST_EXIT" "$TEST_TEE_EXIT" > "$RUN_ROOT/test_exit_codes.txt"
test "$TEST_EXIT" -eq 0
test "$TEST_TEE_EXIT" -eq 0
grep -q '^Ran 32 tests in ' "$RUN_ROOT/tests.log"
grep -q '^OK$' "$RUN_ROOT/tests.log"
if grep -q 'skipped=' "$RUN_ROOT/tests.log"; then
  echo 'ABORT: the required server test set contains skipped tests.' >&2
  exit 1
fi
printf 'TEST_COUNT=32\nSKIPPED_TESTS=0\n' > "$RUN_ROOT/test_summary.txt"

set +e
(
  cd "$REPO"
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 "$PYTHON" \
      -m digit_writing.digit_geometry_final --self-test \
      --output-dir "$RUN_ROOT/final_digit_geometry"
) 2>&1 | tee "$RUN_ROOT/geometry.log"
GEOMETRY_CODES=("${PIPESTATUS[@]}")
set -e
GEOMETRY_EXIT="${GEOMETRY_CODES[0]}"
GEOMETRY_TEE_EXIT="${GEOMETRY_CODES[1]}"
printf 'GEOMETRY_EXIT=%s\nGEOMETRY_TEE_EXIT=%s\n' \
  "$GEOMETRY_EXIT" "$GEOMETRY_TEE_EXIT" > "$RUN_ROOT/geometry_exit_codes.txt"
test "$GEOMETRY_EXIT" -eq 0
test "$GEOMETRY_TEE_EXIT" -eq 0

set +e
(
  cd "$REPO"
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 "$PYTHON" \
      -m digit_writing.final_protocol_audit \
      --geometry-config "$GEOMETRY_CONFIG" \
      --base-config "$BASE_CONFIG" \
      --output "$RUN_ROOT/final_protocol_audit.json"
) 2>&1 | tee "$RUN_ROOT/audit.log"
AUDIT_CODES=("${PIPESTATUS[@]}")
set -e
AUDIT_EXIT="${AUDIT_CODES[0]}"
AUDIT_TEE_EXIT="${AUDIT_CODES[1]}"
printf 'AUDIT_EXIT=%s\nAUDIT_TEE_EXIT=%s\n' \
  "$AUDIT_EXIT" "$AUDIT_TEE_EXIT" > "$RUN_ROOT/audit_exit_codes.txt"
test "$AUDIT_EXIT" -eq 0
test "$AUDIT_TEE_EXIT" -eq 0

"$PYTHON" -c "import json; data=json.load(open('$RUN_ROOT/final_protocol_audit.json')); assert data['passed'] is True; assert data['formal_training_started'] is False"
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
echo "PARENT_HEAD=$PARENT_HEAD"
echo "RUN_ROOT=$RUN_ROOT"
echo "ARCHIVE=$ARCHIVE"
echo "ARCHIVE_HASH=$ARCHIVE_HASH"
echo 'FORMAL_TRAINING_STARTED=0'
echo 'PROTOCOL2_AUDIT_COMPLETE=1'
