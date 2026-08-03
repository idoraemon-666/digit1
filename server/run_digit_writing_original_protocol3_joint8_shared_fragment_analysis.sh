#!/usr/bin/env bash

set -uo pipefail

if [[ $# -ne 5 ]]; then
  echo "Usage: bash server/run_digit_writing_original_protocol3_joint8_shared_fragment_analysis.sh REPO CONFIG CHECKPOINT EXPECTED_HEAD OUTPUT_ROOT" >&2
  exit 2
fi
if [[ "${PROTOCOL3_JOINT8_SHARED_FRAGMENT_ANALYSIS_AUTHORIZED:-}" != "1" ]]; then
  echo "ABORT: set PROTOCOL3_JOINT8_SHARED_FRAGMENT_ANALYSIS_AUTHORIZED=1 after explicit authorization" >&2
  exit 1
fi

REPO="$(realpath "$1")"
CONFIG="$(realpath "$2")"
CHECKPOINT="$(realpath "$3")"
EXPECTED_HEAD="$4"
OUTPUT_ROOT="$(realpath -m "$5")"
PYTHON=/root/autodl-tmp/conda/envs/motor-compat-cpu/bin/python
EXPECTED_CHECKPOINT_SHA256=df0e634021bf24eae1018c2d54325258b20457d67e2db0656921257d3342d345
EXPECTED_PROTOCOL3_REMOTE=https://github.com/idoraemon-666/digit1.git
EXPECTED_CONFIG="$REPO/configurations/digit_writing_original_protocol3_joint8_shared_fragment_analysis_v1.json"
EXPECTED_CHECKPOINT="$REPO/runs/digit_writing_original_protocol3/scale2p50/ref100/corner_ease_v3/joint8_excluding_digit0_digit8/seed42/best_checkpoint.pt"
EXPECTED_OUTPUT="$REPO/runs/digit_writing_original_protocol3/scale2p50/ref100/corner_ease_v3/joint8_excluding_digit0_digit8/seed42_shared_fragment_analysis_v1"
ARCHIVE="${OUTPUT_ROOT}.tar.gz"
ARCHIVE_HASH="${ARCHIVE}.sha256"

fail() {
  echo "ABORT: $1" >&2
  exit "${2:-1}"
}

[[ "$EXPECTED_HEAD" =~ ^[0-9a-f]{40}$ ]] || fail "EXPECTED_HEAD must be a 40-character lowercase commit"
[[ -d "$REPO/.git" ]] || fail "repository .git directory is missing"
[[ -x "$PYTHON" ]] || fail "frozen MotorNet Python is not executable"
[[ "$CONFIG" == "$EXPECTED_CONFIG" ]] || fail "config is not the checked-in shared-fragment config"
[[ "$CHECKPOINT" == "$EXPECTED_CHECKPOINT" ]] || fail "checkpoint is not the frozen joint8 common-best path"
[[ "$OUTPUT_ROOT" == "$EXPECTED_OUTPUT" ]] || fail "output root differs from the frozen analysis directory"
[[ ! -e "$OUTPUT_ROOT" ]] || fail "analysis output already exists: $OUTPUT_ROOT"
[[ ! -e "$ARCHIVE" ]] || fail "analysis archive already exists: $ARCHIVE"
[[ ! -e "$ARCHIVE_HASH" ]] || fail "analysis archive hash already exists: $ARCHIVE_HASH"
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$EXPECTED_HEAD" ]] || fail "repository HEAD differs"
RECORDED_SUBMODULE="$(git -C "$REPO" rev-parse HEAD:mRNNTorch)"
WORKTREE_SUBMODULE="$(git -C "$REPO/mRNNTorch" rev-parse HEAD)"
[[ "$RECORDED_SUBMODULE" == "$WORKTREE_SUBMODULE" ]] || fail "mRNNTorch worktree differs from recorded gitlink"
[[ -z "$(git -C "$REPO" status --short --untracked-files=all)" ]] || fail "repository worktree is not clean"
[[ -z "$(git -C "$REPO/mRNNTorch" status --short --untracked-files=all)" ]] || fail "mRNNTorch worktree is not clean"
[[ "$(git -C "$REPO" remote get-url protocol3)" == "$EXPECTED_PROTOCOL3_REMOTE" ]] || fail "protocol3 remote differs from the only authorized remote"
[[ "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" == "$EXPECTED_CHECKPOINT_SHA256" ]] || fail "checkpoint SHA256 differs"
[[ "$(cat /root/autodl-tmp/conda/envs/motor-compat-cpu/.compatibility_environment_complete)" == "device=cpu" ]] || fail "MotorNet compatibility environment is incomplete"

mkdir -p "$OUTPUT_ROOT/stage_exit_codes" || fail "cannot create analysis output root"
git -C "$REPO" status --short --branch --untracked-files=all > "$OUTPUT_ROOT/git_status_before.txt"
git -C "$REPO" branch --all --verbose --no-abbrev > "$OUTPUT_ROOT/git_branches_before.txt"
git -C "$REPO" worktree list --porcelain > "$OUTPUT_ROOT/git_worktrees_before.txt"
git -C "$REPO" remote -v > "$OUTPUT_ROOT/git_remotes_before.txt"
printf '%s\n' "$EXPECTED_HEAD" > "$OUTPUT_ROOT/repository_head.txt"
printf '%s\n' "$RECORDED_SUBMODULE" > "$OUTPUT_ROOT/submodule_head.txt"
sha256sum "$CHECKPOINT" > "$OUTPUT_ROOT/checkpoint.sha256"
"$PYTHON" -m pip freeze > "$OUTPUT_ROOT/environment.freeze.txt"

TEST_LOG="$OUTPUT_ROOT/tests.log"
(
  cd "$REPO" || exit 97
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 "$PYTHON" -m unittest -v \
      tests.test_digit_geometry \
      tests.test_digit_environment \
      tests.test_geometry_audit \
      tests.test_phase_d_protocol \
      tests.test_protocol3_geometry \
      tests.test_protocol3_schedule \
      tests.test_protocol3_checkpoint \
      tests.test_protocol3_configuration \
      tests.test_protocol3_corner_ease \
      tests.test_protocol3_corner_ease_joint8 \
      tests.test_protocol3_joint8_shared_fragment_analysis
) > "$TEST_LOG" 2>&1
TEST_CODE=$?
printf '%s\n' "$TEST_CODE" > "$OUTPUT_ROOT/stage_exit_codes/tests.exit_code"
if [[ "$TEST_CODE" -ne 0 ]]; then
  tail -n 120 "$TEST_LOG" >&2
  fail "server test gate failed; output preserved at $OUTPUT_ROOT" "$TEST_CODE"
fi
grep -q '^OK$' "$TEST_LOG" || fail "server tests did not end in OK"
if grep -q 'skipped=' "$TEST_LOG"; then
  fail "server analysis test gate contains skipped tests"
fi

run_stage() {
  local stage="$1"
  local log="$OUTPUT_ROOT/${stage}.log"
  (
    cd "$REPO" || exit 97
    CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 "$PYTHON" \
        -m digit_writing.protocol3_joint8_shared_fragment_analysis "$stage" \
        --repository-root "$REPO" \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --output-root "$OUTPUT_ROOT"
  ) > "$log" 2>&1
  local code=$?
  printf '%s\n' "$code" > "$OUTPUT_ROOT/stage_exit_codes/${stage}.exit_code"
  if [[ "$code" -ne 0 ]]; then
    tail -n 120 "$log" >&2
    fail "analysis stage $stage failed; all logs and artifacts are preserved" "$code"
  fi
}

run_stage validate
run_stage prepare
run_stage collect
run_stage analyze
run_stage report

[[ "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" == "$EXPECTED_CHECKPOINT_SHA256" ]] || fail "checkpoint changed during analysis"
[[ -z "$(git -C "$REPO" status --short --untracked-files=all)" ]] || fail "repository worktree changed during analysis"
[[ -z "$(git -C "$REPO/mRNNTorch" status --short --untracked-files=all)" ]] || fail "mRNNTorch worktree changed during analysis"

cat > "$OUTPUT_ROOT/download_manifest.txt" <<EOF
analysis_report=$OUTPUT_ROOT/05_report/analysis_report.md
figures=$OUTPUT_ROOT/04_figures
metrics=$OUTPUT_ROOT/03_metrics
activity_manifest=$OUTPUT_ROOT/02_activity/activity_collection_manifest.json
archive=$ARCHIVE
archive_sha256=$ARCHIVE_HASH
EOF

(
  cd "$OUTPUT_ROOT" || exit 97
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
  sha256sum -c SHA256SUMS
) || fail "final SHA256SUMS verification failed"

tar -C "$(dirname "$OUTPUT_ROOT")" -czf "$ARCHIVE" "$(basename "$OUTPUT_ROOT")" || fail "archive creation failed"
(
  cd "$(dirname "$ARCHIVE")" || exit 97
  sha256sum "$(basename "$ARCHIVE")" > "$(basename "$ARCHIVE_HASH")"
  sha256sum -c "$(basename "$ARCHIVE_HASH")"
) || fail "archive SHA256 verification failed"

echo "ANALYSIS_COMPLETED=1"
echo "TRAINING_OPTIMIZER_STEPS=0"
echo "CHECKPOINT_SHA256=$EXPECTED_CHECKPOINT_SHA256"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "REPORT=$OUTPUT_ROOT/05_report/analysis_report.md"
echo "FIGURES=$OUTPUT_ROOT/04_figures"
echo "ARCHIVE=$ARCHIVE"
echo "ARCHIVE_HASH=$ARCHIVE_HASH"
