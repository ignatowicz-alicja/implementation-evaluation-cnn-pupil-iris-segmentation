\
    # U-Net Pupil Segmentation and Biometric Evaluation

    Reproducible training and evaluation pipeline for pupil segmentation based on an **unchanged third-party U-Net implementation**. The repository separates the original author code from the additional dataset, evaluation, visualization, timing, and biometric-identification scripts.

    ## Repository structure

    ```text
    .
    ├── original_author_code/       # unmodified third-party repository snapshot
    ├── our_scripts/
    │   ├── run_pipeline.py         # command-line entry point
    │   └── wrapper/                # dataset, metrics, plots, timing and biometrics
    ├── tools/
    │   ├── verify_original.py      # verifies SHA-256 of original files
    │   └── self_test.py            # lightweight wrapper test
    ├── provenance/                 # source notice and checksums
    ├── requirements.txt
    ├── requirements-core.txt
    └── LICENSE_OUR_SCRIPTS.txt
    ```

    `original_author_code/` is not modified. The added pipeline imports and calls `create_model()` from the original `UNet_iris_segmentation.py`.

    ## Data

    Datasets and manually prepared masks are **not included**. A recommended structure is:

    ```text
    dataset/
    ├── images/
    │   ├── 001/01_L.bmp
    │   └── 002/01_L.bmp
    └── masks_pupil/
        ├── 001/01_L.png
        └── 002/01_L.png
    ```

    Supported image formats: BMP, PNG, JPG/JPEG and TIFF. Masks may use the same stem as the image or suffixes such as `_pupil` and `_mask`.

    ## Installation

    Python 3.11 is recommended.

    ### Windows PowerShell

    ```powershell
    py -3.11 -m venv .venv
    & ".\.venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
    & ".\.venv\Scripts\python.exe" -m pip install -r ".\requirements.txt"
    & ".\.venv\Scripts\python.exe" ".\tools\verify_original.py"
    & ".\.venv\Scripts\python.exe" ".\tools\self_test.py"
    ```

    These are direct terminal commands; no PowerShell script execution policy change is required.

    ### Linux / WSL2

    ```bash
    python3.11 -m venv .venv
    .venv/bin/python -m pip install --upgrade pip setuptools wheel
    .venv/bin/python -m pip install -r requirements.txt
    .venv/bin/python tools/verify_original.py
    .venv/bin/python tools/self_test.py
    ```

    ## Full IITD-style experiment

    Replace the three example paths with local dataset locations.

    ### Windows PowerShell

    ```powershell
    & ".\.venv\Scripts\python.exe" -u ".\our_scripts\run_pipeline.py" full `
      --images "D:\datasets\IITD\images" `
      --masks "D:\datasets\IITD\masks_pupil" `
      --output "D:\results\IITD_UNET_PUPIL" `
      --subject-mode parent `
      --allow-missing-masks `
      --split-mode subject_disjoint `
      --train-ratio 0.70 `
      --val-ratio 0.15 `
      --epochs 20 `
      --batch-size 2 `
      --learning-rate 0.0001 `
      --patience 8 `
      --mask-polarity auto `
      --eval-split test `
      --threshold 0.5 `
      --save-all `
      --save-examples 20 `
      --pixels-per-image 3000 `
      --identification-scope test `
      --mask-source both `
      --gallery-ratio 0.70 `
      --outer-scale 3.0 `
      --seed 2026
    ```

    ### Linux / WSL2

    ```bash
    .venv/bin/python -u our_scripts/run_pipeline.py full \
      --images /path/to/dataset/images \
      --masks /path/to/dataset/masks_pupil \
      --output /path/to/results/IITD_UNET_PUPIL \
      --subject-mode parent \
      --allow-missing-masks \
      --split-mode subject_disjoint \
      --train-ratio 0.70 \
      --val-ratio 0.15 \
      --epochs 20 \
      --batch-size 2 \
      --learning-rate 0.0001 \
      --patience 8 \
      --mask-polarity auto \
      --eval-split test \
      --threshold 0.5 \
      --save-all \
      --save-examples 20 \
      --pixels-per-image 3000 \
      --identification-scope test \
      --mask-source both \
      --gallery-ratio 0.70 \
      --outer-scale 3.0 \
      --seed 2026
    ```

    ## Running individual stages

    ```powershell
    $PYTHON = ".\.venv\Scripts\python.exe"

    # Dataset manifest
    & $PYTHON ".\our_scripts\run_pipeline.py" prepare `
      --images "D:\dataset\images" `
      --masks "D:\dataset\masks_pupil" `
      --output "D:\results\experiment" `
      --subject-mode parent `
      --allow-missing-masks

    # Training
    & $PYTHON ".\our_scripts\run_pipeline.py" train `
      --manifest "D:\results\experiment\dataset_manifest.csv" `
      --output "D:\results\experiment\01_training" `
      --epochs 20 `
      --batch-size 2

    # Segmentation evaluation
    & $PYTHON ".\our_scripts\run_pipeline.py" evaluate `
      --manifest "D:\results\experiment\dataset_manifest.csv" `
      --weights "D:\results\experiment\01_training\best_author_unet.weights.h5" `
      --output "D:\results\experiment\02_segmentation_evaluation" `
      --split test `
      --save-all

    # Biometric baseline
    & $PYTHON ".\our_scripts\run_pipeline.py" identify `
      --manifest "D:\results\experiment\dataset_manifest.csv" `
      --weights "D:\results\experiment\01_training\best_author_unet.weights.h5" `
      --evaluation-dir "D:\results\experiment\02_segmentation_evaluation" `
      --output "D:\results\experiment\03_biometric_identification" `
      --mask-source both
    ```

    ## Adapting to other datasets

    Subject identifiers can be inferred with:

    - `--subject-mode parent`: direct parent directory, suitable for IITD-style layouts;
    - `--subject-mode first_folder`: first relative directory;
    - `--subject-mode stem_prefix`: filename prefix before `_`, `-` or whitespace;
    - `--subject-regex "(...)"`: custom regular expression with a capturing group.

    Two split strategies are available:

    - `subject_disjoint`: subjects do not overlap between train, validation and test;
    - `per_subject`: every subject is split internally into train, validation and test images.

    ## Generated results

    The pipeline records:

    - training history and per-epoch training time;
    - preprocessing, U-Net inference, postprocessing, saving and end-to-end time per image;
    - TP, TN, FP and FN;
    - Dice/F1, IoU/Jaccard and the compatibility alias `ioc_iou`;
    - precision, recall, specificity, NPV, balanced accuracy and MCC;
    - boundary F1, boundary IoU, HD95 and ASSD;
    - raw and normalized segmentation confusion matrices;
    - ROC, precision-recall, calibration and threshold-sweep plots;
    - predicted masks, probability maps and TP/FP/FN overlays;
    - best and worst segmentation examples;
    - subject-level statistics;
    - an optional biometric baseline with Top-1/5/10, CMC, MRR, EER, genuine/impostor distributions and identity confusion matrices.

    The biometric module is an added baseline and is **not** part of the original U-Net repository.

    ## Reproducibility and privacy

    The public source tree contains no local user paths, usernames, machine names, datasets, model weights or experiment outputs. Generated manifests may contain absolute dataset paths, so experiment output directories are ignored by `.gitignore` and should be reviewed before publication.

    ## Licensing and attribution

    - `our_scripts/` and `tools/` are covered by `LICENSE_OUR_SCRIPTS.txt`.
    - `original_author_code/` remains third-party material and is explicitly excluded from that license.
    - See `provenance/ORIGINAL_CODE_NOTICE.md` before redistribution.
