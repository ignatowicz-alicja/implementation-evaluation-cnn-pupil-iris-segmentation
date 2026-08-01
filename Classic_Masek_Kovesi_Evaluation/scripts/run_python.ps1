$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

python -m pip install -r (Join-Path $RepoRoot "requirements.txt")
python (Join-Path $RepoRoot "src/python/run_casia_full_eval_python.py") `
  --dataset bmp `
  --code-dir (Join-Path $RepoRoot "third_party/Iris-Recognition-master/python") `
  --bmp-dir (Join-Path $RepoRoot "data/CASIA1_BMP") `
  --gt-dir (Join-Path $RepoRoot "data/ground_truth/pupil_masks") `
  --output-dir (Join-Path $RepoRoot "results/python")
