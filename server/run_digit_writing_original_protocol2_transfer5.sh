#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_digit_writing_original_protocol2_experiment.sh" \
  configurations/digit_writing_original_protocol2_transfer5.json \
  transfer5-dev42
