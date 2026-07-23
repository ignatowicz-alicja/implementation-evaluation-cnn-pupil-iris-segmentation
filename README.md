# Implementation and Evaluation of CNN for Pupil and Iris Segmentation on Devices with Varying Computational Resources

Supplementary repository for the article:

**Implementation and Evaluation of CNN for Pupil and Iris Segmentation on Devices with Varying Computational Resources**

## Overview

This repository contains segmentation masks and reproducible experiment pipelines for classical and U-Net-based iris and pupil segmentation.

The experiments were prepared for biometric iris images from the **CASIA-IrisV1** and **IIT Delhi Iris Database** datasets.

The segmentation masks provided in this repository were prepared and manually verified for both datasets.

The original third-party implementations are preserved without modification. Dataset integration, experiment execution, evaluation, timing measurements, visualisation, and result export are performed using separate wrapper scripts.

## Datasets

### CASIA-IrisV1

CASIA-IrisV1 is a near-infrared biometric iris dataset containing 756 grayscale images with a resolution of 320 × 280 pixels.

The dataset is used for classical iris segmentation, mask evaluation, and biometric recognition experiments.

The pupil and iris masks included in this repository were prepared and manually verified.

Official dataset sources:

- [CASIA Iris Databases](https://english.ia.cas.cn/db/201610/t20161026_169399.html)
- [CASIA Database Access Portal](https://www.idealtest.org/findDownloadDbByMode.do?mode=Iris)

### IIT Delhi Iris Database

The IIT Delhi Iris Database contains near-infrared grayscale iris images with a resolution of 320 × 240 pixels.

The dataset is used for classical segmentation experiments and U-Net training and evaluation.

The pupil and iris masks included in this repository were prepared and manually verified.

Official dataset source:

- [IIT Delhi Iris Database](https://www4.comp.polyu.edu.hk/~csajaykr/IITD/Database_Iris.htm)

### Cataract-1K

Cataract-1K is a medical dataset containing cataract surgery videos and pixel-level semantic segmentation annotations.

In this study, the semantic segmentation subset was used for U-Net-based pupil segmentation. The pupil class was extracted from the original multi-class annotations and converted into binary masks, where the pupil represents the foreground and all remaining pixels represent the background.

The experiments used 2,256 annotated frames extracted from 30 cataract surgery videos. The original frame resolution is 1024 × 768 pixels.

Official sources:

- [Cataract-1K GitHub repository](https://github.com/Negin-Ghamsarian/Cataract-1K)
- [Cataract-1K publication](https://doi.org/10.1038/s41597-024-03193-4)

The original Cataract-1K videos and annotations are not included in this repository. Users must obtain the dataset from the official source and comply with its licence and terms of use.


## Repository structure

```text
repository/
├── datasets/
│   ├── CASIA-IrisV1/
│   │   └── masks/
│   └── IITD/
│       └── masks/
│
├── classical_pipeline/
│   ├── third_party/
│   ├── wrappers/
│   ├── configs/
│   └── run_pipeline.py
│
├── unet_pipeline/
│   ├── third_party/
│   ├── wrappers/
│   ├── configs/
│   └── run_pipeline.py
│
├── evaluation/
├── reference_results/
└── requirements.txt
```

The `third_party` directories contain the original implementations used in the study. Their source code and original directory names are preserved without modification.

The `wrappers` directories contain scripts developed for dataset preparation, experiment execution, evaluation, timing measurements, visualisation, and result export.

## Dataset preparation

The original CASIA-IrisV1 and IITD images are not included in this repository.

Download both datasets from their official providers and store them locally, for example:

```text
data/
├── CASIA-IrisV1/
└── IITD/
```

The provided segmentation masks should retain the same file names and directory structure as the corresponding source images.

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Activate it on Linux:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Classical pipeline

Run the classical pipeline for CASIA-IrisV1:

```bash
python classical_pipeline/run_pipeline.py \
    --dataset casia \
    --images-dir "/path/to/CASIA-IrisV1" \
    --masks-dir "datasets/CASIA-IrisV1/masks" \
    --output-dir "results/CASIA-IrisV1/classical"
```

Run the classical pipeline for IITD:

```bash
python classical_pipeline/run_pipeline.py \
    --dataset iitd \
    --images-dir "/path/to/IITD" \
    --masks-dir "datasets/IITD/masks" \
    --output-dir "results/IITD/classical"
```

## U-Net pipeline

Run U-Net training and evaluation for CASIA-IrisV1:

```bash
python unet_pipeline/run_pipeline.py \
    --dataset casia \
    --images-dir "/path/to/CASIA-IrisV1" \
    --masks-dir "datasets/CASIA-IrisV1/masks" \
    --output-dir "results/CASIA-IrisV1/unet" \
    --mode train-evaluate
```

Run U-Net training and evaluation for IITD:

```bash
python unet_pipeline/run_pipeline.py \
    --dataset iitd \
    --images-dir "/path/to/IITD" \
    --masks-dir "datasets/IITD/masks" \
    --output-dir "results/IITD/unet" \
    --mode train-evaluate
```

## Evaluation

The pipelines support the calculation of:

- Intersection over Union;
- Dice coefficient;
- pixel accuracy;
- precision;
- recall;
- specificity;
- processing time per image;
- Top-1 biometric identification accuracy.

## Data availability

The original CASIA-IrisV1 and IITD images are not redistributed in this repository.

Users must obtain the datasets independently and comply with the licences and terms specified by their official providers.

## Citation

```bibtex
@article{ignatowicz2026segmentation,
  title  = {Implementation and Evaluation of CNN for Pupil and Iris Segmentation on Devices with Varying Computational Resources},
  author = {Ignatowicz, Alicja Anna and Marciniak, Tomasz},
  year   = {2026}
}

