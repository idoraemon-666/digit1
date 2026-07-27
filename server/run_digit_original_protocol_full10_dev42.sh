#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_digit_original_protocol_experiment.sh" \
  configurations/digit_original_protocol_full10_dev42.json \
  full10-dev42
