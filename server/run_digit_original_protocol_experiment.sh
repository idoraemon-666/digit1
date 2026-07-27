#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 CONFIG_PATH RUN_LABEL" >&2
  exit 2
fi
if [[ "${DIGIT_PROTOCOL_AUTHORIZED_RUN:-}" != "$2" ]]; then
  echo "STOP: Phase E user authorization must name this exact experiment run." >&2
  exit 2
fi

WORK_ROOT="/root/autodl-tmp"
REPO="$WORK_ROOT/digit_writing_original_protocol_4b8b1db"
CPU_ENV="$WORK_ROOT/conda/envs/motor-compat-cpu"
PYTHON="$CPU_ENV/bin/python"
CONFIG_RELATIVE="$1"
RUN_LABEL="$2"
CONFIG="$REPO/$CONFIG_RELATIVE"

test -d "$REPO/.git"
test -x "$PYTHON"
test -f "$CONFIG"
test "$(cat "$CPU_ENV/.compatibility_environment_complete")" = "device=cpu"
test -z "$(git -C "$REPO" status --porcelain)"
test -z "$(git -C "$REPO/mRNNTorch" status --porcelain)"

"$PYTHON" -m pip check
"$PYTHON" - <<'PY'
import torch
assert torch.__version__ == "2.6.0+cpu"
assert not torch.cuda.is_available()
print("CPU_BASELINE_VERIFIED=1")
PY

HEAD="$(git -C "$REPO" rev-parse HEAD)"
SHORT_HEAD="${HEAD:0:7}"
SEED="$($PYTHON -c "import json; print(json.load(open('$CONFIG'))['seed'])")"
OUTPUT_RELATIVE="$($PYTHON -c "import json; c=json.load(open('$CONFIG')); print(c.get('output', {}).get('directory', c.get('output_directory')))" )"
OUTPUT_DIR="$REPO/$OUTPUT_RELATIVE"
RUN_ROOT="$WORK_ROOT/digit-writing-${RUN_LABEL}-${SHORT_HEAD}-seed${SEED}"

if [[ -e "$RUN_ROOT" || -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite an existing run directory." >&2
  echo "RUN_ROOT=$RUN_ROOT" >&2
  echo "OUTPUT_DIR=$OUTPUT_DIR" >&2
  exit 1
fi
mkdir -p "$RUN_ROOT"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/start_utc.txt"
cp "$CONFIG" "$RUN_ROOT/configuration.json"
sha256sum "$RUN_ROOT/configuration.json" > "$RUN_ROOT/configuration.json.sha256"
SOURCE_RELATIVE="$($PYTHON -c "import json; print(json.load(open('$CONFIG')).get('source_checkpoint', ''))")"
if [[ -n "$SOURCE_RELATIVE" ]]; then
  SOURCE_CHECKPOINT="$REPO/$SOURCE_RELATIVE"
  test -f "$SOURCE_CHECKPOINT"
  sha256sum "$SOURCE_CHECKPOINT" > "$RUN_ROOT/source_checkpoint.sha256"
fi
git -C "$REPO" rev-parse HEAD > "$RUN_ROOT/parent_head.txt"
git -C "$REPO/mRNNTorch" rev-parse HEAD > "$RUN_ROOT/submodule_head.txt"
printf '%s\n' "105cd0c6b153f0e80a2593e43b7ffec7cc1f5e33" > "$RUN_ROOT/original_baseline.txt"
printf '%s\n' "$SEED" > "$RUN_ROOT/random_seed.txt"
printf 'device=cpu\n' > "$RUN_ROOT/device.txt"
"$PYTHON" -m pip freeze > "$RUN_ROOT/environment.freeze.txt"

cd "$REPO"
set +e
PYTHONDONTWRITEBYTECODE=1 \
  "$PYTHON" config.py --protocol_config "$CONFIG_RELATIVE" 2>&1 | tee "$RUN_ROOT/run.log"
PIPE_STATUS=("${PIPESTATUS[@]}")
RUN_EXIT="${PIPE_STATUS[0]}"
TEE_EXIT="${PIPE_STATUS[1]}"
set -e
printf 'RUN_EXIT=%s\nTEE_EXIT=%s\n' "$RUN_EXIT" "$TEE_EXIT" > "$RUN_ROOT/exit_codes.txt"
if [[ "$RUN_EXIT" -ne 0 || "$TEE_EXIT" -ne 0 ]]; then
  echo "PROTOCOL_EXPERIMENT_FAILED=1" >&2
  exit 1
fi

test -z "$(git -C "$REPO" status --porcelain)"
test -z "$(git -C "$REPO/mRNNTorch" status --porcelain)"
test -d "$OUTPUT_DIR"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/end_utc.txt"
cp -a "$OUTPUT_DIR" "$RUN_ROOT/output"
(
  cd "$RUN_ROOT"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
)
cd "$WORK_ROOT"
tar -czf "$(basename "$RUN_ROOT").tar.gz" "$(basename "$RUN_ROOT")"
sha256sum "$(basename "$RUN_ROOT").tar.gz" > "$(basename "$RUN_ROOT").tar.gz.sha256"

printf 'PARENT_HEAD=%s\n' "$HEAD"
printf 'RUN_ROOT=%s\n' "$RUN_ROOT"
printf 'ARCHIVE=%s.tar.gz\n' "$RUN_ROOT"
printf 'RUN_EXIT=0\nTEE_EXIT=0\n'
printf 'FORMAL_RUN_COMPLETE=1\n'
