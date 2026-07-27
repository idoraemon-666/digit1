#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_digit_original_protocol_experiment.sh" \
  configurations/digit_original_protocol_heldout5_dev42.json \
  heldout5-dev42
