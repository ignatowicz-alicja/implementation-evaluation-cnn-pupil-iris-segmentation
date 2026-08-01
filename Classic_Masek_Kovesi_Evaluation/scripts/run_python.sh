#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 -m pip install -r "$REPO_ROOT/requirements.txt"
python3 "$REPO_ROOT/src/python/run_casia_full_eval_python.py" \
  --dataset bmp \
  --code-dir "$REPO_ROOT/third_party/Iris-Recognition-master/python" \
  --bmp-dir "$REPO_ROOT/data/CASIA1_BMP" \
  --gt-dir "$REPO_ROOT/data/ground_truth/pupil_masks" \
  --output-dir "$REPO_ROOT/results/python"
