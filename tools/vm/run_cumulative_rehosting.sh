#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 PROJECT_ROOT RUN_ROOT HELPER MODEL LIBLLAMA OUTPUT_ROOT" >&2
  exit 2
fi

readonly project_root=$1
readonly run_root=$2
readonly helper=$3
readonly model=$4
readonly libllama=$5
readonly output_root=$6

test -d "$project_root/src"
test -d "$run_root/bundles"
test -x "$helper"
test -f "$model"
test -f "$libllama"
test ! -e "$output_root"

PYTHONPATH="$project_root/src" python3 -m laris_onpremise.rehosting.command \
  --run-root "$run_root" \
  --helper "$helper" \
  --model "$model" \
  --libllama "$libllama" \
  --output "$output_root" \
  --timeout-seconds 120 \
  --predict 64
