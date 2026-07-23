from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

repository_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repository_root / "our_scripts"))

from wrapper.biometrics import annular_feature
from wrapper.dataset import prepare_manifest, read_manifest
from wrapper.metrics import binary_metrics

with tempfile.TemporaryDirectory() as temporary_directory:
    root = Path(temporary_directory)
    images = root / "images"
    masks = root / "masks_pupil"

    for subject in ("001", "002", "003"):
        (images / subject).mkdir(parents=True)
        (masks / subject).mkdir(parents=True)
        for index in range(4):
            image = np.zeros((240, 320), np.uint8)
            cv2.circle(image, (160, 120), 70, 140, -1)
            cv2.circle(image, (160, 120), 25, 20, -1)
            mask = np.zeros_like(image)
            cv2.circle(mask, (160, 120), 25, 255, -1)
            cv2.imwrite(str(images / subject / f"{index + 1:02d}_L.bmp"), image)
            cv2.imwrite(str(masks / subject / f"{index + 1:02d}_L.png"), mask)

    manifest = root / "manifest.csv"
    prepare_manifest(images, masks, manifest, split_mode="subject_disjoint", strict=True)
    records = read_manifest(manifest)
    assert len(records) == 12

    mask = cv2.imread(records[0].mask_path, cv2.IMREAD_GRAYSCALE) > 127
    metrics = binary_metrics(mask, mask)
    assert abs(metrics["dice_f1"] - 1.0) < 1e-9

    gray = cv2.imread(records[0].image_path, cv2.IMREAD_GRAYSCALE)
    feature = annular_feature(gray, mask.astype(np.uint8))
    assert feature is not None and feature.ndim == 1

print("[OK] Wrapper self-test completed successfully.")
