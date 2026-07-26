#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]] || [[ "$1" != "cpu" && "$1" != "cuda" ]]; then
  echo "Usage: bash server/create_compat_environment.sh {cpu|cuda}" >&2
  exit 2
fi

DEVICE="$1"
DATA_ROOT=/root/autodl-tmp
COMMON_REQUIREMENTS=server/requirements-linux-common.txt
if [[ "$DEVICE" == "cpu" ]]; then
  ENV_PATH="$DATA_ROOT/conda/envs/motor-compat-cpu"
  REQUIREMENTS=server/requirements-linux-cpu.txt
  TORCH_INDEX=https://download.pytorch.org/whl/cpu
else
  ENV_PATH="$DATA_ROOT/conda/envs/motor-compat-cu118"
  REQUIREMENTS=server/requirements-linux-cu118.txt
  TORCH_INDEX=https://download.pytorch.org/whl/cu118
fi

COMPLETE_MARKER="$ENV_PATH/.compatibility_environment_complete"
PYTHON="$ENV_PATH/bin/python"
if [[ -f "$COMPLETE_MARKER" ]]; then
  echo "ABORT: completed environment already exists: $ENV_PATH" >&2
  exit 1
elif [[ -e "$ENV_PATH" ]]; then
  if [[ ! -x "$PYTHON" ]]; then
    echo "ABORT: incomplete environment has no executable Python: $ENV_PATH" >&2
    exit 1
  fi
  echo "RESUMING_INCOMPLETE_ENVIRONMENT=$ENV_PATH"
else
  conda create -y -p "$ENV_PATH" python=3.10.20 pip=26.1.2 setuptools=83.0.0 wheel=0.47.0
fi

TORCH_REQUIREMENT="$(grep -m 1 '^torch==' "$REQUIREMENTS" || true)"
if [[ -z "$TORCH_REQUIREMENT" ]]; then
  echo "ABORT: no pinned torch requirement in $REQUIREMENTS" >&2
  exit 1
fi

"$PYTHON" -m pip install --no-deps -r "$COMMON_REQUIREMENTS"
"$PYTHON" -m pip install --index-url "$TORCH_INDEX" "$TORCH_REQUIREMENT"
"$PYTHON" -m pip check
"$PYTHON" -m pip freeze
"$PYTHON" -c "import motornet,numpy,torch; print('torch',torch.__version__); print('numpy',numpy.__version__); print('cuda_available',torch.cuda.is_available())"
printf 'device=%s\n' "$DEVICE" > "$COMPLETE_MARKER"

echo "ENVIRONMENT_READY=$ENV_PATH"
