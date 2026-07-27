#!/usr/bin/env bash
set -euo pipefail
exec bash server/run_digit_writing_original_protocol2_experiment.sh \
  configurations/digit_writing_original_protocol2_full10_dev42.json \
  full10-dev42
