#!/bin/bash
# Runs inside WSL: float vs int8 paddle detection parity.
set -eo pipefail
BASE="/mnt/c/Users/hibye/Documents/ACTUAL PROJECTS/Thesis"
export OMP_NUM_THREADS=6
python3 "$BASE/scripts/check_tflite_parity.py" --n 40
