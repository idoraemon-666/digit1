#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: bash server/run_digit_writing_original_protocol3_gate1.sh REPO RUN_ROOT EXPECTED_HEAD SCALE_TOKEN" >&2
  exit 2
fi

REPO="$(realpath "$1")"
RUN_ROOT="$(realpath -m "$2")"
EXPECTED_HEAD="$3"
SCALE_TOKEN="$4"
PYTHON=/root/autodl-tmp/conda/envs/motor-compat-cpu/bin/python
SUBMODULE=ac0c4f589eae37bbde63968912925de99232e306
ARCHIVE="${RUN_ROOT}.tar.gz"
ARCHIVE_HASH="${ARCHIVE}.sha256"

[[ "$SCALE_TOKEN" == "2p50" || "$SCALE_TOKEN" == "2p25" ]]
test -d "$REPO/.git"
test -x "$PYTHON"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
test "$(git -C "$REPO" rev-parse HEAD:mRNNTorch)" = "$SUBMODULE"
test "$(git -C "$REPO/mRNNTorch" rev-parse HEAD)" = "$SUBMODULE"
test -z "$(git -C "$REPO" status --short)"
test -z "$(git -C "$REPO/mRNNTorch" status --short)"
test "$(cat /root/autodl-tmp/conda/envs/motor-compat-cpu/.compatibility_environment_complete)" = "device=cpu"

for path in "$RUN_ROOT" "$ARCHIVE" "$ARCHIVE_HASH"; do
  if [[ -e "$path" ]]; then
    echo "ABORT: output already exists: $path" >&2
    exit 1
  fi
done
mkdir -p "$RUN_ROOT"
printf '%s\n' "$EXPECTED_HEAD" > "$RUN_ROOT/parent_head.txt"
printf '%s\n' "$SUBMODULE" > "$RUN_ROOT/submodule_head.txt"
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
      tests.test_phase_d_protocol \
      tests.test_protocol3_geometry \
      tests.test_protocol3_schedule \
      tests.test_protocol3_checkpoint \
      tests.test_protocol3_early_audit \
      tests.test_protocol3_configuration
) 2>&1 | tee "$RUN_ROOT/tests.log"
TEST_CODES=("${PIPESTATUS[@]}")
set -e
test "${TEST_CODES[0]}" -eq 0
test "${TEST_CODES[1]}" -eq 0
grep -q '^OK$' "$RUN_ROOT/tests.log"
if grep -q 'skipped=' "$RUN_ROOT/tests.log"; then
  echo 'ABORT: the required Gate 1 server test set contains skipped tests.' >&2
  exit 1
fi
grep '^Ran [0-9][0-9]* tests in ' "$RUN_ROOT/tests.log" > "$RUN_ROOT/test_summary.txt"
printf 'SKIPPED_TESTS=0\n' >> "$RUN_ROOT/test_summary.txt"

(
  cd "$REPO"
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 "$PYTHON" \
      -m digit_writing.protocol3_gate1 \
      --scale-token "$SCALE_TOKEN" \
      --output "$RUN_ROOT/gate1"
) 2>&1 | tee "$RUN_ROOT/gate1.log"

"$PYTHON" - "$RUN_ROOT/gate1/preflight_summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
assert summary["training_updates"] == 0
assert summary["optimizer_used"] is False
assert summary["passed"] or summary["requires_scale_fallback"]
PY

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

"$PYTHON" - "$RUN_ROOT/gate1/preflight_summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
print(f"GATE1_PASS={int(summary['passed'])}")
print(f"GATE1_REQUIRES_SCALE_FALLBACK={int(summary['requires_scale_fallback'])}")
PY
echo "RUN_ROOT=$RUN_ROOT"
echo "ARCHIVE=$ARCHIVE"
echo "ARCHIVE_HASH=$ARCHIVE_HASH"
echo 'FORMAL_TRAINING_STARTED=0'
