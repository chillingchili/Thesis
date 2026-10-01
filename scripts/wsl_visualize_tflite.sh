#!/bin/bash
# Runs inside WSL: float vs w8a32 detection montage.
set -eo pipefail
BASE="/mnt/c/Users/hibye/Documents/ACTUAL PROJECTS/Thesis"
export OMP_NUM_THREADS=6
python3 "$BASE/scripts/visualize_tflite_inference.py" --n 8 --seed 11
