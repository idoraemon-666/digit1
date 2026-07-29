#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "Usage: bash server/inspect_digit_writing_original_protocol3_gate2_continuation.sh REPO CONFIG SOURCE_SUMMARY RESUME_CHECKPOINT CASE EXPECTED_HEAD EXPECTED_SOURCE_HEAD" >&2
  exit 2
fi

REPO="$(realpath "$1")"
CONFIG="$(realpath "$2")"
SOURCE_SUMMARY="$(realpath "$3")"
RESUME_CHECKPOINT="$(realpath "$4")"
CASE_LABEL="$5"
EXPECTED_HEAD="$6"
EXPECTED_SOURCE_HEAD="$7"
PYTHON=/root/autodl-tmp/conda/envs/motor-compat-cpu/bin/python
SUBMODULE=ac0c4f589eae37bbde63968912925de99232e306
SOURCE_ROOT="$(dirname "$SOURCE_SUMMARY")"
SOURCE_ARCHIVE="${SOURCE_ROOT}.tar.gz"
SOURCE_ARCHIVE_HASH="${SOURCE_ARCHIVE}.sha256"

[[ "$CASE_LABEL" == "o1_digit1" || "$CASE_LABEL" == "o2_digit5" || "$CASE_LABEL" == "o3_digit8" ]]
[[ "$EXPECTED_HEAD" =~ ^[0-9a-f]{40}$ ]]
[[ "$EXPECTED_SOURCE_HEAD" =~ ^[0-9a-f]{40}$ ]]
test -d "$REPO/.git"
test -x "$PYTHON"
test "$(basename "$RESUME_CHECKPOINT")" = "final_continuation_checkpoint.pt"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
test "$(git -C "$REPO" rev-parse HEAD:mRNNTorch)" = "$SUBMODULE"
test "$(git -C "$REPO/mRNNTorch" rev-parse HEAD)" = "$SUBMODULE"
test -z "$(git -C "$REPO" status --short)"
test -z "$(git -C "$REPO/mRNNTorch" status --short)"
case "$CONFIG" in
  "$REPO"/configurations/digit_writing_original_protocol3_gate2_scale*_ref*.json) ;;
  *) echo "ABORT: config is not a checked-in Protocol3 Gate 2 config" >&2; exit 1 ;;
esac
case "$SOURCE_ROOT" in
  "$REPO"/runs/digit_writing_original_protocol3/*) ;;
  *) echo "ABORT: source summary is outside the Protocol3 run root" >&2; exit 1 ;;
esac
case "$RESUME_CHECKPOINT" in
  "$SOURCE_ROOT"/*) ;;
  *) echo "ABORT: checkpoint is outside its source result root" >&2; exit 1 ;;
esac
test -f "$SOURCE_ARCHIVE"
test -f "$SOURCE_ARCHIVE_HASH"
test -f "$SOURCE_ROOT/SHA256SUMS"
(
  cd "$SOURCE_ROOT"
  sha256sum -c SHA256SUMS >/dev/null
)
echo 'SOURCE_TREE_SHA256_OK=1'
(
  cd "$(dirname "$SOURCE_ARCHIVE")"
  sha256sum -c "$(basename "$SOURCE_ARCHIVE_HASH")"
)

(
  cd "$REPO"
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 "$PYTHON" \
      -m digit_writing.protocol3_gate2_continuation inspect \
      --repository-root "$REPO" \
      --protocol-config "$CONFIG" \
      --source-summary "$SOURCE_SUMMARY" \
      --resume-checkpoint "$RESUME_CHECKPOINT" \
      --case-label "$CASE_LABEL" \
      --expected-source-head "$EXPECTED_SOURCE_HEAD"
)
test -z "$(git -C "$REPO" status --short)"
test -z "$(git -C "$REPO/mRNNTorch" status --short)"
echo 'CONTINUATION_STARTED=0'
echo 'FORMAL_FULL10_STARTED=0'
