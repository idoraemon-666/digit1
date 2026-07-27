#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash server/run_digit_geometry_protocol_audit.sh REPO RUN_ROOT" >&2
  exit 2
fi

REPO="$(realpath "$1")"
RUN_ROOT="$(realpath -m "$2")"
PYTHON=/root/autodl-tmp/conda/envs/motor-compat-cpu/bin/python
BASELINE=105cd0c6b153f0e80a2593e43b7ffec7cc1f5e33
SUBMODULE=ac0c4f589eae37bbde63968912925de99232e306
CONFIG="$REPO/configurations/digit_original_protocol_geometry.json"
ARCHIVE="${RUN_ROOT}.tar.gz"
ARCHIVE_HASH="${ARCHIVE}.sha256"
START_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

test -d "$REPO/.git"
test -x "$PYTHON"
test -f "$CONFIG"
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
UNIT_TESTS_INCLUDED=0
if [[ -n "${PRECOMMIT_TEST_ROOT:-}" ]]; then
  test -f "$PRECOMMIT_TEST_ROOT/tests.log"
  test -f "$PRECOMMIT_TEST_ROOT/test_exit_codes.txt"
  cp "$PRECOMMIT_TEST_ROOT/tests.log" "$RUN_ROOT/tests.log"
  cp "$PRECOMMIT_TEST_ROOT/test_exit_codes.txt" "$RUN_ROOT/test_exit_codes.txt"
  UNIT_TESTS_INCLUDED=1
fi
SOURCE_FILES=(
  envs.py
  PROJECT_PROTOCOL.md
  digit_writing/geometry.py
  digit_writing/geometry_audit.py
  digit_writing/protocol_audit.py
  configurations/digit_original_protocol_geometry.json
  tests/test_digit_geometry.py
  tests/test_geometry_audit.py
  tests/test_digit_environment.py
  tests/test_protocol_audit.py
  server/run_digit_geometry_protocol_audit.sh
)

{
  printf 'START_UTC=%s\n' "$START_UTC"
  printf 'REPO=%s\n' "$REPO"
  printf 'RUN_ROOT=%s\n' "$RUN_ROOT"
  printf 'PARENT_HEAD=%s\n' "$PARENT_HEAD"
  printf 'SUBMODULE_HEAD=%s\n' "$SUBMODULE_HEAD"
  printf 'BASELINE=%s\n' "$BASELINE"
  printf 'DEVICE=cpu\n'
  uname -a
  lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core|Socket'
  free -h
  df -h /root/autodl-tmp
  "$PYTHON" --version
  "$PYTHON" -m pip --version
  "$PYTHON" -m pip check
  "$PYTHON" -c "import motornet,numpy,torch; print('motornet',motornet.__version__); print('numpy',numpy.__version__); print('torch',torch.__version__); print('cuda_available',torch.cuda.is_available())"
  (
    cd "$REPO"
    sha256sum "${SOURCE_FILES[@]}"
  )
} 2>&1 | tee "$RUN_ROOT/preflight.txt"

cp "$CONFIG" "$RUN_ROOT/configuration.json"
"$PYTHON" -m pip freeze > "$RUN_ROOT/environment.freeze.txt"

set +e
(
  cd "$REPO"
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONHASHSEED=0 "$PYTHON" -m digit_writing.protocol_audit \
      --repository "$REPO" \
      --config "$CONFIG" \
      --output "$RUN_ROOT/audit"
) 2>&1 | tee "$RUN_ROOT/audit.log"
AUDIT_PIPE_CODES=("${PIPESTATUS[@]}")
set -e

AUDIT_EXIT="${AUDIT_PIPE_CODES[0]}"
AUDIT_TEE_EXIT="${AUDIT_PIPE_CODES[1]}"
printf 'AUDIT_EXIT=%s\nAUDIT_TEE_EXIT=%s\n' "$AUDIT_EXIT" "$AUDIT_TEE_EXIT" \
  > "$RUN_ROOT/audit_exit_codes.txt"
test "$AUDIT_TEE_EXIT" -eq 0
[[ "$AUDIT_EXIT" -eq 0 || "$AUDIT_EXIT" -eq 1 ]]
test -f "$RUN_ROOT/audit/DIGIT_GEOMETRY_PROTOCOL_AUDIT.md"
test -f "$RUN_ROOT/audit/digit_geometry_protocol_audit.json"
test -f "$RUN_ROOT/audit/digit_vs_original_geometry_metrics.csv"
test -f "$RUN_ROOT/audit/digit_condition_workspace_audit.csv"
test -f "$RUN_ROOT/audit/digit_geometry_protocol_audit.png"
test -f "$RUN_ROOT/audit/OUTPUT_SHA256.json"

BLOCKING_ISSUES="$($PYTHON -c "import json; data=json.load(open('$RUN_ROOT/audit/digit_geometry_protocol_audit.json')); print(len(data['conclusions']['D_blocking_must_fix']))")"
PASSED_WITHOUT_BLOCKERS="$($PYTHON -c "import json; data=json.load(open('$RUN_ROOT/audit/digit_geometry_protocol_audit.json')); print(int(data['passed_without_blockers']))")"

END_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
PARENT_HEAD="$PARENT_HEAD" \
SUBMODULE_HEAD="$SUBMODULE_HEAD" \
BASELINE="$BASELINE" \
START_UTC="$START_UTC" \
END_UTC="$END_UTC" \
AUDIT_EXIT="$AUDIT_EXIT" \
BLOCKING_ISSUES="$BLOCKING_ISSUES" \
PASSED_WITHOUT_BLOCKERS="$PASSED_WITHOUT_BLOCKERS" \
UNIT_TESTS_INCLUDED="$UNIT_TESTS_INCLUDED" \
RUN_ROOT="$RUN_ROOT" \
"$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

manifest = {
    "parent_head": os.environ["PARENT_HEAD"],
    "submodule_head": os.environ["SUBMODULE_HEAD"],
    "original_baseline": os.environ["BASELINE"],
    "device": "cpu",
    "random_seed": None,
    "start_utc": os.environ["START_UTC"],
    "end_utc": os.environ["END_UTC"],
    "audit_exit": int(os.environ["AUDIT_EXIT"]),
    "unit_tests_passed_before_commit": bool(int(os.environ["UNIT_TESTS_INCLUDED"])),
    "blocking_issue_count": int(os.environ["BLOCKING_ISSUES"]),
    "passed_without_blockers": bool(int(os.environ["PASSED_WITHOUT_BLOCKERS"])),
    "development_seed_run": False,
    "formal_training_started": False,
    "execution_completed": True,
}
path = Path(os.environ["RUN_ROOT"]) / "run_manifest.json"
path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

(
  cd "$RUN_ROOT"
  find . -type f ! -name SHA256SUMS -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum \
    > SHA256SUMS
  sha256sum -c SHA256SUMS
)

tar -C "$(dirname "$RUN_ROOT")" -czf "$ARCHIVE" "$(basename "$RUN_ROOT")"
(
  cd "$(dirname "$ARCHIVE")"
  sha256sum "$(basename "$ARCHIVE")" > "$(basename "$ARCHIVE_HASH")"
  sha256sum -c "$(basename "$ARCHIVE_HASH")"
)

test -z "$(git -C "$REPO" status --short)"
test -z "$(git -C "$REPO/mRNNTorch" status --short)"

echo "PARENT_HEAD=$PARENT_HEAD"
echo "RUN_ROOT=$RUN_ROOT"
echo "ARCHIVE=$ARCHIVE"
echo "ARCHIVE_HASH=$ARCHIVE_HASH"
echo "AUDIT_EXIT=$AUDIT_EXIT"
echo "BLOCKING_ISSUES=$BLOCKING_ISSUES"
echo "PASSED_WITHOUT_BLOCKERS=$PASSED_WITHOUT_BLOCKERS"
echo 'DEVELOPMENT_SEED_RUN=0'
echo 'FORMAL_TRAINING_STARTED=0'
echo 'DIGIT_GEOMETRY_PROTOCOL_AUDIT_EXECUTION_COMPLETE=1'
