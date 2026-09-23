# Implementation and Evaluation of CNN for Pupil and Iris Segmentation on Devices with Varying Computational Resources

Supplementary repository for the article **Implementation and Evaluation of CNN for Pupil and Iris Segmentation on Devices with Varying Computational Resources**.

The repository contains reproducible experiment wrappers for CASIA-IrisV1, IIT Delhi and Cataract-1K. Original image datasets, trained weights and generated results are intentionally not published here.

## What is original and what is project code

The original external implementations are kept in separate directories and must not be edited by project wrappers:

- `Classic_Masek_Kovesi_Evaluation/third_party/Iris-Recognition-master/` — external classical Masek/Kovesi implementation;
- `pupil-segmentation-unet-evaluation/original_author_code/` — unchanged third-party U-Net source snapshot.

The files at repository root, `scripts/`, `Classic_Masek_Kovesi_Evaluation/src/` and `pupil-segmentation-unet-evaluation/our_scripts/` are project-authored launchers, preparation, measurement, evaluation and export code. Source provenance for U-Net is recorded in `pupil-segmentation-unet-evaluation/provenance/`.

## Structure

```text
repository/
├── run_experiment.py                 # one main launcher: database / method / device
├── experiment_profiles.py            # per-dataset settings, no local paths
├── experiment_runner.py               # GPU detection and dispatch
├── scripts/                           # 12 explicit database/method/device launchers
│   ├── run_casia_classical_cpu.py
│   ├── run_casia_classical_gpu.py
│   ├── run_casia_unet_cpu.py
│   ├── run_casia_unet_gpu.py
│   ├── run_iitd_*.py
│   └── run_cataract1k_*.py
├── Classic_Masek_Kovesi_Evaluation/
│   ├── src/python/                    # project evaluation wrapper
│   └── third_party/Iris-Recognition-master/
└── pupil-segmentation-unet-evaluation/
    ├── our_scripts/                   # project U-Net wrapper
    └── original_author_code/          # unchanged third-party code
```

There is a separate launcher for every combination of database, method and requested device. They use one shared dispatcher so hardware detection and result metadata are implemented once and consistently.

## Datasets

| Identifier | Dataset | Default dataset-specific setting |
|---|---|---|
| `casia` | CASIA-IrisV1 | subject is read from the filename prefix |
| `iitd` | IIT Delhi Iris Database | subject is read from the parent directory |
| `cataract1k` | Cataract-1K | parent directory is treated as a video/case |

Obtain the original data from its respective publisher and provide local image and mask paths at runtime. The project does not redistribute Cataract-1K videos/annotations, or the original CASIA and IITD images.

## Installation

Create and activate a virtual environment, then install dependencies for the method that will be run:

```bash
python -m venv .venv
source .venv/bin/activate              # Linux / WSL
python -m pip install --upgrade pip
python -m pip install -r Classic_Masek_Kovesi_Evaluation/requirements.txt
python -m pip install -r pupil-segmentation-unet-evaluation/requirements.txt
```

On Windows PowerShell activate it with `& .\.venv\Scripts\Activate.ps1`.

For the classical implementation, also place the correctly licensed external `Iris-Recognition-master` source under `Classic_Masek_Kovesi_Evaluation/third_party/Iris-Recognition-master/` as described in its README.

## Main launcher

The main launcher accepts all choices as options:

```bash
python run_experiment.py \
  --dataset iitd --method unet --device gpu -- \
  --images /path/to/IITD/images \
  --masks /path/to/IITD/masks_pupil \
  --output results/iitd/unet
```

Or run it without the three selection options; it will ask for the database, `classical`/`unet`, and `cpu`/`gpu` in the terminal:

```bash
python run_experiment.py -- --images /path/to/images --masks /path/to/masks
```

Options placed after the second `--` are passed to the selected pipeline. For example, a short U-Net test can be requested with `-- --images ... --masks ... -- --epochs 1 --batch-size 1`.

## Explicit per-profile scripts

Each profile can also be run directly, which is useful for a fixed experiment command in a lab notebook or batch job:

```bash
python scripts/run_casia_classical_cpu.py \
  --images /path/to/CASIA/images \
  --masks /path/to/CASIA/pupil_masks

python scripts/run_cataract1k_unet_gpu.py \
  --images /path/to/Cataract1K/frames \
  --masks /path/to/Cataract1K/pupil_masks \
  -- --epochs 20 --batch-size 2
```

All scripts accept `--dry-run` to display the final underlying command without processing data. The optional `--max-images N` applies to the classical wrapper; use U-Net options after `--` for U-Net-specific parameters.

## CPU and GPU behavior

The requested device and the actually used device are saved in `run_configuration.json` inside the result directory.

- U-Net with `--device gpu` queries TensorFlow for CUDA-capable devices. When none is available, it prints `Nie ma dostępnego GPU; uruchamiam U-Net na CPU` and forces CPU execution.
- The external classical `Iris-Recognition-master` implementation has no GPU backend. Its `*_gpu.py` scripts are retained for a complete, uniform matrix of experiments, but explicitly state that they fall back to CPU.
- `--device cpu` always disables CUDA for the launched U-Net process.

## Metrics and outputs

The wrappers save per-image processing time, segmentation metrics such as IoU and Dice, diagnostic images and summaries. The classical wrapper additionally runs the implemented biometric identification evaluation where its inputs are available. U-Net output includes training/evaluation artifacts as documented in `pupil-segmentation-unet-evaluation/README.md`.

## Data protection and reproducibility

Do not commit private datasets, paths exported by experiment results, model weights, or generated results. The root `.gitignore` excludes common local data/result locations. Before publishing any new artifact, verify that it contains no image data or absolute local paths.

## Citation

```bibtex
@article{ignatowicz2026segmentation,
  title  = {Implementation and Evaluation of CNN for Pupil and Iris Segmentation on Devices with Varying Computational Resources},
  author = {Ignatowicz, Alicja Anna and Marciniak, Tomasz},
  year   = {2026}
}
```
