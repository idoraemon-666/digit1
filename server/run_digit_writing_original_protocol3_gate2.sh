#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "Usage: bash server/run_digit_writing_original_protocol3_gate2.sh REPO CONFIG GATE1_SUMMARY EXPECTED_HEAD [FAST_GATE2_SUMMARY]" >&2
  exit 2
fi

REPO="$(realpath "$1")"
CONFIG="$(realpath "$2")"
GATE1_SUMMARY="$(realpath "$3")"
EXPECTED_HEAD="$4"
FAST_SUMMARY="${5:-}"
PYTHON=/root/autodl-tmp/conda/envs/motor-compat-cpu/bin/python
SUBMODULE=ac0c4f589eae37bbde63968912925de99232e306

test -d "$REPO/.git"
test -x "$PYTHON"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
test "$(git -C "$REPO" rev-parse HEAD:mRNNTorch)" = "$SUBMODULE"
test "$(git -C "$REPO/mRNNTorch" rev-parse HEAD)" = "$SUBMODULE"
test -z "$(git -C "$REPO" status --short)"
test -z "$(git -C "$REPO/mRNNTorch" status --short)"

readarray -t CONFIG_VALUES < <("$PYTHON" - "$CONFIG" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
assert "scale2p50" in config["geometry_config"] or "scale2p25" in config["geometry_config"]
print(config["output"]["directory"])
print(config["selected_reference_steps"])
print("2p50" if config["geometry_config"].find("scale2p50") >= 0 else "2p25")
PY
)
RUN_ROOT="$(realpath -m "$REPO/${CONFIG_VALUES[0]}")"
REFERENCE="${CONFIG_VALUES[1]}"
SCALE_TOKEN="${CONFIG_VALUES[2]}"
ARCHIVE="${RUN_ROOT}.tar.gz"
ARCHIVE_HASH="${ARCHIVE}.sha256"

case "$RUN_ROOT" in
  "$REPO"/runs/digit_writing_original_protocol3/*) ;;
  *) echo "ABORT: Gate 2 output is outside the Protocol3 run root" >&2; exit 1 ;;
esac

"$PYTHON" - "$GATE1_SUMMARY" "$SCALE_TOKEN" "$EXPECTED_HEAD" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
assert summary["passed"] is True
assert summary["scale_token"] == sys.argv[2]
assert summary["git_identity"]["repository_head"] == sys.argv[3]
PY

if [[ "$REFERENCE" == "50" ]]; then
  test -z "$FAST_SUMMARY"
elif [[ "$REFERENCE" == "100" ]]; then
  test -n "$FAST_SUMMARY"
  FAST_SUMMARY="$(realpath "$FAST_SUMMARY")"
  "$PYTHON" - "$FAST_SUMMARY" "$SCALE_TOKEN" "$EXPECTED_HEAD" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
assert summary["selected_reference_steps"] == 50
assert summary["scale_multiplier"] == (2.5 if sys.argv[2] == "2p50" else 2.25)
assert summary["git_identity"]["repository_head"] == sys.argv[3]
assert summary["classification"] == "behavior_failure"
assert summary["medium_fallback_allowed"] is True
PY
else
  echo "ABORT: Gate 2 reference must be 50 or 100" >&2
  exit 1
fi

for path in "$RUN_ROOT" "$ARCHIVE" "$ARCHIVE_HASH"; do
  if [[ -e "$path" ]]; then
    echo "ABORT: output already exists: $path" >&2
    exit 1
  fi
done
mkdir -p "$(dirname "$RUN_ROOT")"

set +e
(
  cd "$REPO"
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 "$PYTHON" \
      config.py --protocol_config "$CONFIG"
) 2>&1 | tee "${RUN_ROOT}.console.log"
TRAIN_CODES=("${PIPESTATUS[@]}")
set -e
TRAIN_EXIT="${TRAIN_CODES[0]}"
TEE_EXIT="${TRAIN_CODES[1]}"
test "$TEE_EXIT" -eq 0

if [[ ! -f "$RUN_ROOT/gate2_summary.json" ]]; then
  echo "ABORT: Gate 2 did not produce a summary (training exit $TRAIN_EXIT)" >&2
  exit 1
fi
mv "${RUN_ROOT}.console.log" "$RUN_ROOT/console.log"
printf 'TRAIN_EXIT=%s\nTEE_EXIT=%s\n' "$TRAIN_EXIT" "$TEE_EXIT" > "$RUN_ROOT/exit_codes.txt"
"$PYTHON" -m pip freeze > "$RUN_ROOT/environment.freeze.txt"
printf '%s\n' "$EXPECTED_HEAD" > "$RUN_ROOT/parent_head.txt"
printf '%s\n' "$SUBMODULE" > "$RUN_ROOT/submodule_head.txt"

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

CLASSIFICATION="$("$PYTHON" - "$RUN_ROOT/gate2_summary.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["classification"])
PY
)"
echo "GATE2_CLASSIFICATION=$CLASSIFICATION"
echo "RUN_ROOT=$RUN_ROOT"
echo "ARCHIVE=$ARCHIVE"
echo "ARCHIVE_HASH=$ARCHIVE_HASH"
echo 'FORMAL_FULL10_STARTED=0'
if [[ "$CLASSIFICATION" == "engineering_or_safety_failure" ]]; then
  exit 1
fi
if [[ "$CLASSIFICATION" == "behavior_failure" ]]; then
  [[ "$REFERENCE" == "50" ]] && exit 20
  exit 21
fi
test "$CLASSIFICATION" = "provisional_pass_pending_qualitative_review"
