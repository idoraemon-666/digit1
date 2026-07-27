#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_digit_original_protocol_experiment.sh" \
  configurations/digit_original_protocol_transfer5.json \
  transfer5-dev42
