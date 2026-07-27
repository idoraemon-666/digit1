#!/usr/bin/env bash
set -euo pipefail
exec bash server/run_digit_writing_original_protocol2_experiment.sh \
  configurations/digit_writing_original_protocol2_heldout5_dev42.json \
  heldout5-dev42
