# -*- coding: utf-8 -*-
"""
Pełna ewaluacja klasycznej metody Iris-Recognition / Masek-Kovesi dla CASIA-IrisV1.

Co robi skrypt:
1) uruchamia segmentację źrenicy i tęczówki z dokładnym pomiarem czasu etapów,
2) tworzy maskę binarną źrenicy z okręgu cirpupil,
3) porównuje maskę źrenicy z binarną maską ground truth,
4) liczy IoU, Dice, pixel accuracy, precision, recall, specificity,
5) zapisuje obrazy błędnych segmentacji i nakładki diagnostyczne,
6) wykonuje normalizację, kodowanie i identyfikację top-1 dystansem Hamminga,
7) zapisuje CSV, TXT, JSON i wykresy dla baz JPG oraz BMP.

Pochodzenie: autorski skrypt ewaluacyjny przygotowany do eksperymentów artykułu i pracy dyplomowej.
Uruchomienie przykładowe:
    python run_casia_full_eval_python.py

Wymagane: numpy, scipy, opencv-python, matplotlib.
Skrypt nie zawiera oryginalnych funkcji segmentacji. Folder fnc/ musi pochodzić z użytej implementacji Iris-Recognition-master/python/fnc.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

# =============================================================================
# PRZENOŚNE ŚCIEŻKI DOMYŚLNE
# =============================================================================

# Repozytorium ma strukturę:
#   src/python/                         - ten skrypt (kod ewaluacyjny projektu)
#   third_party/Iris-Recognition-master/python/fnc/
#                                      - zewnętrzna implementacja metody
#   data/CASIA1_BMP/                   - obrazy CASIA-IrisV1 (nie są dystrybuowane)
#   data/CASIA1_JPG/                   - opcjonalna konwersja JPG (nie jest dystrybuowana)
#   data/ground_truth/pupil_masks/     - binarne maski źrenicy
#   results/python/                    - wyniki

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYTHON_CODE_DIR = str(REPO_ROOT / "third_party" / "Iris-Recognition-master" / "python")
DEFAULT_GT_DIR = str(REPO_ROOT / "data" / "ground_truth" / "pupil_masks")
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "results" / "python")

DEFAULT_DATASETS = [
    {
        "name": "CASIA1_JPG",
        "images_dir": str(REPO_ROOT / "data" / "CASIA1_JPG"),
        "extensions": [".jpg", ".jpeg", ".png", ".bmp"],
    },
    {
        "name": "CASIA1_BMP",
        "images_dir": str(REPO_ROOT / "data" / "CASIA1_BMP"),
        "extensions": [".bmp", ".jpg", ".jpeg", ".png"],
    },
]

# Parametry oryginalnego kodu CASIA1
EYELASHES_THRESHOLD = 80
RADIAL_RES = 20
ANGULAR_RES = 240
MIN_WAVELENGTH = 18
MULT = 1
SIGMA_ONF = 0.5

# Próg do zapisywania obrazów jako błędnie zsegmentowanych.
# Zmieniaj np. na 0.60, jeśli chcesz ostrzejszą selekcję.
BAD_IOU_THRESHOLD = 0.50
BAD_DICE_THRESHOLD = 0.65

# Maksymalna liczba najgorszych przykładów kopiowana dodatkowo do folderu worst_examples.
N_WORST_EXAMPLES = 80


# =============================================================================
# IMPORT ORYGINALNEGO KODU
# =============================================================================

def add_original_code_to_path(code_dir: str) -> None:
    """Dodaje folder z oryginalnym python/fnc do sys.path."""
    candidates = []
    this_dir = Path(__file__).resolve().parent
    candidates.append(this_dir)
    candidates.append(this_dir / "python")
    candidates.append(Path(code_dir))

    for cand in candidates:
        if (cand / "fnc").is_dir():
            sys.path.insert(0, str(cand))
            return

    raise FileNotFoundError(
        "Nie znaleziono folderu fnc. Uruchom skrypt z folderu python projektu "
        "Iris-Recognition-master albo podaj --code-dir ze ścieżką do folderu python."
    )


# =============================================================================
# STRUKTURY DANYCH
# =============================================================================

@dataclass
class FeatureRecord:
    dataset: str
    image_path: str
    rel_path: str
    identity: str
    template: np.ndarray
    mask: np.ndarray
    feature_total_s: float
    segmentation_total_s: float


# =============================================================================
# FUNKCJE POMOCNICZE
# =============================================================================

def now_s() -> float:
    return time.perf_counter()


def safe_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def natural_key(path: Path) -> List[object]:
    txt = str(path).lower()
    return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", txt)]


def find_images(root: Path, extensions: Iterable[str]) -> List[Path]:
    extensions = {e.lower() for e in extensions}
    if not root.exists():
        return []
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in extensions]
    return sorted(files, key=natural_key)


def identity_from_path(path: Path) -> str:
    """
    Dla CASIA V1 najpewniejszy identyfikator to pierwsze 3 cyfry z nazwy pliku, np. 001_1_1.bmp -> 001.
    Jeśli nazwa nie pasuje, używany jest folder nadrzędny.
    """
    m = re.match(r"(\d{1,4})[_\-]", path.stem)
    if m:
        return m.group(1).zfill(3)
    m2 = re.search(r"(\d{1,4})", path.parent.name)
    if m2:
        return m2.group(1).zfill(3)
    return path.parent.name


def clean_stem(stem: str) -> str:
    s = stem.lower()
    suffixes = [
        "_pupil_mask", "_mask_pupil", "_pupil_gt", "_gt_pupil",
        "_pseudo_gt", "_pseudo_ground_truth", "_ground_truth",
        "_pupil", "_mask", "_gt", "_sobel_hough",
    ]
    changed = True
    while changed:
        changed = False
        for suf in suffixes:
            if s.endswith(suf):
                s = s[: -len(suf)]
                changed = True
    return s


def build_gt_index(gt_root: Path) -> Dict[str, Path]:
    """
    Indeksuje maski GT po nazwie bazowej, np.:
    001_1_1.png, 001_1_1_pupil.png, 001_1_1_mask.png -> klucz 001_1_1.
    Dzięki temu ten sam GT pasuje do JPG i BMP.
    """
    gt_index: Dict[str, Path] = {}
    if not gt_root.exists():
        return gt_index

    mask_exts = {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}
    for p in gt_root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in mask_exts:
            continue
        rel_no_ext = str(p.relative_to(gt_root).with_suffix("" )).replace("\\", "/").lower()
        stem_clean = clean_stem(p.stem)
        keys = {
            stem_clean,
            p.stem.lower(),
            clean_stem(rel_no_ext.split("/")[-1]),
            clean_stem(rel_no_ext),
        }
        for key in keys:
            if key and key not in gt_index:
                gt_index[key] = p
    return gt_index


def locate_gt_mask(img_path: Path, dataset_root: Path, gt_index: Dict[str, Path]) -> Optional[Path]:
    try:
        rel = img_path.relative_to(dataset_root)
        rel_key = str(rel.with_suffix("")).replace("\\", "/").lower()
    except Exception:
        rel_key = img_path.stem.lower()

    candidates = [
        clean_stem(img_path.stem),
        img_path.stem.lower(),
        clean_stem(rel_key),
        rel_key,
        clean_stem(rel_key.split("/")[-1]),
    ]
    for k in candidates:
        if k in gt_index:
            return gt_index[k]
    return None


def load_gt_mask(mask_path: Path, target_shape: Tuple[int, int]) -> Tuple[np.ndarray, str]:
    """Wczytuje GT jako maskę bool. Automatycznie odwraca, jeśli wygląda na białe tło i czarną źrenicę."""
    gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise ValueError(f"Nie można wczytać maski GT: {mask_path}")
    if gt.shape != target_shape:
        gt = cv2.resize(gt, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
        resize_note = "resized_to_image_shape"
    else:
        resize_note = "original_shape"

    # Najpierw zakładamy: obiekt = jasne piksele.
    binary = gt > 0
    fg_ratio = float(binary.mean())

    # Dla maski źrenicy foreground zwykle jest mały. Jeśli ponad połowa obrazu jest biała,
    # prawdopodobnie maska ma białe tło i czarną źrenicę, więc odwracamy.
    if fg_ratio > 0.50:
        binary = ~binary
        resize_note += ";inverted_auto"
    return binary.astype(bool), resize_note


def circle_mask(shape: Tuple[int, int], circle_yxr: List[int]) -> np.ndarray:
    y0, x0, r = [int(round(v)) for v in circle_yxr]
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return ((yy - y0) ** 2 + (xx - x0) ** 2) <= (r ** 2)


def annulus_mask(shape: Tuple[int, int], ciriris_yxr: List[int], cirpupil_yxr: List[int]) -> np.ndarray:
    iris = circle_mask(shape, ciriris_yxr)
    pupil = circle_mask(shape, cirpupil_yxr)
    return np.logical_and(iris, ~pupil)


def binary_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())

    eps = 1e-12
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    pixel_accuracy = (tp + tn) / (tp + fp + fn + tn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    fpr = fp / (fp + tn + eps)
    fnr = fn / (fn + tp + eps)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "iou": iou,
        "dice": dice,
        "pixel_accuracy": pixel_accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "fpr": fpr,
        "fnr": fnr,
    }


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    mkdir(path.parent)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def contour_from_mask(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if mask.size == 0:
        return mask
    up = np.roll(mask, 1, axis=0)
    down = np.roll(mask, -1, axis=0)
    left = np.roll(mask, 1, axis=1)
    right = np.roll(mask, -1, axis=1)
    interior = mask & up & down & left & right
    contour = mask & ~interior
    contour[0, :] = mask[0, :]
    contour[-1, :] = mask[-1, :]
    contour[:, 0] = mask[:, 0]
    contour[:, -1] = mask[:, -1]
    return contour


def draw_circle_on_bgr(bgr: np.ndarray, circle_yxr: Optional[List[int]], color: Tuple[int, int, int], thickness: int = 1) -> None:
    if circle_yxr is None:
        return
    y, x, r = [int(round(v)) for v in circle_yxr]
    cv2.circle(bgr, (x, y), r, color, thickness)
    cv2.drawMarker(bgr, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=8, thickness=1)


def make_diagnostic_image(
    image_gray: np.ndarray,
    pred_pupil: Optional[np.ndarray],
    gt_pupil: Optional[np.ndarray],
    cirpupil: Optional[List[int]],
    ciriris: Optional[List[int]],
    text_lines: List[str],
) -> np.ndarray:
    """
    Tworzy panel diagnostyczny:
    1) obraz z okręgami,
    2) predykcja źrenicy,
    3) GT,
    4) mapa błędów: TP zielony, FP czerwony, FN niebieski.
    """
    base = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    panel1 = base.copy()
    draw_circle_on_bgr(panel1, ciriris, (255, 255, 0), 1)   # cyjan: tęczówka
    draw_circle_on_bgr(panel1, cirpupil, (0, 0, 255), 1)    # czerwony: źrenica z metody

    h, w = image_gray.shape[:2]
    panel2 = base.copy()
    panel3 = base.copy()
    panel4 = np.zeros_like(base) + 40

    if pred_pupil is not None:
        overlay = panel2.copy()
        overlay[pred_pupil] = (0, 0, 255)
        panel2 = cv2.addWeighted(overlay, 0.35, panel2, 0.65, 0)
        c = contour_from_mask(pred_pupil)
        panel2[c] = (0, 0, 255)

    if gt_pupil is not None:
        overlay = panel3.copy()
        overlay[gt_pupil] = (0, 255, 0)
        panel3 = cv2.addWeighted(overlay, 0.35, panel3, 0.65, 0)
        c = contour_from_mask(gt_pupil)
        panel3[c] = (0, 255, 0)

    if pred_pupil is not None and gt_pupil is not None:
        tp = np.logical_and(pred_pupil, gt_pupil)
        fp = np.logical_and(pred_pupil, ~gt_pupil)
        fn = np.logical_and(~pred_pupil, gt_pupil)
        panel4[:] = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR) // 3
        panel4[tp] = (0, 180, 0)
        panel4[fp] = (0, 0, 255)
        panel4[fn] = (255, 0, 0)

    panels = [panel1, panel2, panel3, panel4]
    titles = ["circles: iris cyan, pupil red", "pred pupil", "GT pupil", "TP green / FP red / FN blue"]
    for p, title in zip(panels, titles):
        cv2.rectangle(p, (0, 0), (w, 22), (0, 0, 0), -1)
        cv2.putText(p, title, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    out = np.hstack(panels)
    # pasek tekstowy pod spodem
    info_h = 22 + 18 * min(len(text_lines), 8)
    info = np.zeros((info_h, out.shape[1], 3), dtype=np.uint8)
    y = 18
    for line in text_lines[:8]:
        cv2.putText(info, str(line), (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        y += 18
    return np.vstack([out, info])


def maybe_make_plots(dataset_out: Path, image_rows: List[dict], summary: dict) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    metrics = [("iou", "IoU pupil"), ("dice", "Dice pupil"), ("segmentation_total_s", "Segmentation total [s]"), ("feature_total_s", "Feature extraction total [s]")]
    for key, title in metrics:
        vals = []
        for r in image_rows:
            v = safe_float(r.get(key))
            if v is not None:
                vals.append(v)
        if not vals:
            continue
        plt.figure(figsize=(8, 5))
        plt.hist(vals, bins=30)
        plt.title(title)
        plt.xlabel(key)
        plt.ylabel("liczba obrazów")
        plt.tight_layout()
        plt.savefig(dataset_out / f"hist_{key}.png", dpi=160)
        plt.close()

    # Średnie czasy etapów segmentacji
    time_keys = [
        "read_s", "inner_pupil_s", "outer_iris_s", "top_eyelid_s", "bottom_eyelid_s",
        "eyelashes_s", "segmentation_total_s", "pupil_mask_s", "gt_load_s", "metrics_s",
        "normalization_s", "encoding_s", "feature_total_s",
    ]
    labels = []
    means = []
    for key in time_keys:
        vals = [safe_float(r.get(key)) for r in image_rows]
        vals = [v for v in vals if v is not None]
        if vals:
            labels.append(key.replace("_s", ""))
            means.append(float(np.mean(vals)))
    if means:
        plt.figure(figsize=(12, 5))
        plt.bar(range(len(means)), means)
        plt.xticks(range(len(means)), labels, rotation=55, ha="right")
        plt.ylabel("średni czas [s]")
        plt.title("Średni czas etapów")
        plt.tight_layout()
        plt.savefig(dataset_out / "mean_stage_times.png", dpi=160)
        plt.close()


def aggregate_numeric(rows: List[dict], key: str) -> Dict[str, Optional[float]]:
    vals = [safe_float(r.get(key)) for r in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"mean": None, "std": None, "min": None, "max": None, "sum": None}
    arr = np.asarray(vals, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "sum": float(np.sum(arr)),
    }


# =============================================================================
# SZCZEGÓŁOWA SEGMENTACJA, NORMALIZACJA I KODOWANIE
# =============================================================================

def detailed_extract_feature(image_path: Path, eyelashes_threshold: int = 80):
    """
    Wersja rozbita na etapy czasowe. Korzysta z oryginalnych funkcji fnc.
    Zwraca template/mask oraz wszystkie dane potrzebne do oceny segmentacji.
    """
    from fnc.boundary import searchInnerBound, searchOuterBound
    from fnc.segment import findTopEyelid, findBottomEyelid
    from fnc.normalize import normalize
    from fnc.encode import encode

    t0_feature = now_s()
    times: Dict[str, float] = {}

    t = now_s()
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    times["read_s"] = now_s() - t
    if image is None:
        raise ValueError(f"Nie można wczytać obrazu: {image_path}")

    t_seg0 = now_s()

    t = now_s()
    rowp, colp, rp = searchInnerBound(image)
    times["inner_pupil_s"] = now_s() - t

    t = now_s()
    row, col, r = searchOuterBound(image, rowp, colp, rp)
    times["outer_iris_s"] = now_s() - t

    rowp = int(np.round(rowp))
    colp = int(np.round(colp))
    rp = int(np.round(rp))
    row = int(np.round(row))
    col = int(np.round(col))
    r = int(np.round(r))
    cirpupil = [rowp, colp, rp]
    ciriris = [row, col, r]

    imsz = image.shape
    irl = int(np.round(row - r))
    iru = int(np.round(row + r))
    icl = int(np.round(col - r))
    icu = int(np.round(col + r))
    irl = max(irl, 0)
    icl = max(icl, 0)
    iru = min(iru, imsz[0] - 1)
    icu = min(icu, imsz[1] - 1)
    imageiris = image[irl: iru + 1, icl: icu + 1]

    t = now_s()
    try:
        mask_top = findTopEyelid(imsz, imageiris, irl, icl, rowp, rp)
    except Exception:
        mask_top = np.zeros(imsz, dtype=float)
        times["top_eyelid_error"] = 1
    times["top_eyelid_s"] = now_s() - t

    t = now_s()
    try:
        mask_bot = findBottomEyelid(imsz, imageiris, rowp, rp, irl, icl)
    except Exception:
        mask_bot = np.zeros(imsz, dtype=float)
        times["bottom_eyelid_error"] = 1
    times["bottom_eyelid_s"] = now_s() - t

    t = now_s()
    imwithnoise = image.astype(float)
    imwithnoise = imwithnoise + mask_top + mask_bot
    ref = image < eyelashes_threshold
    imwithnoise[np.where(ref == 1)] = np.nan
    times["eyelashes_s"] = now_s() - t

    times["segmentation_total_s"] = now_s() - t_seg0

    t = now_s()
    polar_array, noise_array = normalize(
        imwithnoise,
        ciriris[1], ciriris[0], ciriris[2],
        cirpupil[1], cirpupil[0], cirpupil[2],
        RADIAL_RES, ANGULAR_RES,
    )
    times["normalization_s"] = now_s() - t

    t = now_s()
    template, mask = encode(polar_array, noise_array, MIN_WAVELENGTH, MULT, SIGMA_ONF)
    times["encoding_s"] = now_s() - t

    times["feature_total_s"] = now_s() - t0_feature

    return {
        "image": image,
        "cirpupil": cirpupil,
        "ciriris": ciriris,
        "imwithnoise": imwithnoise,
        "polar_array": polar_array,
        "noise_array": noise_array,
        "template": template,
        "mask": mask,
        "times": times,
    }


# =============================================================================
# EWALUACJA JEDNEJ BAZY
# =============================================================================

def evaluate_dataset(
    dataset_name: str,
    images_dir: Path,
    extensions: List[str],
    gt_dir: Path,
    output_dir: Path,
    bad_iou_threshold: float,
    bad_dice_threshold: float,
    max_images: int = 0,
) -> dict:
    from fnc.matching import calHammingDist

    dataset_out = output_dir / dataset_name
    overlays_bad_dir = dataset_out / "bad_segmentations_overlays"
    overlays_missing_gt_dir = dataset_out / "missing_gt_overlays"
    overlays_failed_dir = dataset_out / "failed_processing"
    masks_pred_dir = dataset_out / "predicted_pupil_masks"
    worst_dir = dataset_out / "worst_examples"
    for d in [dataset_out, overlays_bad_dir, overlays_missing_gt_dir, overlays_failed_dir, masks_pred_dir, worst_dir]:
        mkdir(d)

    gt_index = build_gt_index(gt_dir)
    images = find_images(images_dir, extensions)
    if max_images and max_images > 0:
        images = images[:max_images]

    print("\n" + "=" * 90)
    print(f"[DATASET] {dataset_name}")
    print(f"[OBRAZY]  {images_dir}")
    print(f"[GT]      {gt_dir}")
    print(f"[WYNIKI]  {dataset_out}")
    print(f"[LICZBA OBRAZÓW] {len(images)}")
    print(f"[LICZBA MASEK GT W INDEKSIE] {len(gt_index)}")
    print("=" * 90)

    image_rows: List[dict] = []
    features: List[FeatureRecord] = []
    bad_candidates: List[Tuple[float, Path]] = []

    global_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    for idx, img_path in enumerate(images, 1):
        try:
            rel_path = str(img_path.relative_to(images_dir))
        except Exception:
            rel_path = img_path.name
        identity = identity_from_path(img_path)
        base_row = {
            "dataset": dataset_name,
            "index": idx,
            "image_path": str(img_path),
            "rel_path": rel_path,
            "filename": img_path.name,
            "stem": img_path.stem,
            "identity": identity,
            "status": "started",
            "error": "",
            "gt_path": "",
            "gt_note": "",
            "bad_segmentation": "",
        }

        if idx == 1 or idx % 25 == 0 or idx == len(images):
            print(f"[{dataset_name}] {idx}/{len(images)}: {rel_path}")

        try:
            result = detailed_extract_feature(img_path, EYELASHES_THRESHOLD)
            image = result["image"]
            cirpupil = result["cirpupil"]
            ciriris = result["ciriris"]
            template = result["template"]
            mask = result["mask"]
            times = result["times"]

            base_row.update(times)
            base_row.update({
                "status": "ok",
                "pupil_y": cirpupil[0],
                "pupil_x": cirpupil[1],
                "pupil_r": cirpupil[2],
                "iris_y": ciriris[0],
                "iris_x": ciriris[1],
                "iris_r": ciriris[2],
            })

            # Maski predykcji
            t = now_s()
            pred_pupil = circle_mask(image.shape, cirpupil)
            pred_iris_annulus = annulus_mask(image.shape, ciriris, cirpupil)
            base_row["pupil_mask_s"] = now_s() - t
            base_row["pred_pupil_area_px"] = int(pred_pupil.sum())
            base_row["pred_iris_annulus_area_px"] = int(pred_iris_annulus.sum())

            # Zapis maski predykcji źrenicy
            pred_mask_out = masks_pred_dir / (img_path.stem + "_pred_pupil.png")
            cv2.imwrite(str(pred_mask_out), (pred_pupil.astype(np.uint8) * 255))
            base_row["pred_pupil_mask_path"] = str(pred_mask_out)

            # GT + metryki
            gt_path = locate_gt_mask(img_path, images_dir, gt_index)
            gt_pupil = None
            if gt_path is not None:
                base_row["gt_path"] = str(gt_path)
                t = now_s()
                gt_pupil, gt_note = load_gt_mask(gt_path, image.shape)
                base_row["gt_load_s"] = now_s() - t
                base_row["gt_note"] = gt_note
                base_row["gt_area_px"] = int(gt_pupil.sum())

                t = now_s()
                met = binary_metrics(pred_pupil, gt_pupil)
                base_row.update(met)
                base_row["metrics_s"] = now_s() - t

                for k in global_counts:
                    global_counts[k] += int(met[k])

                bad = (met["iou"] < bad_iou_threshold) or (met["dice"] < bad_dice_threshold)
                base_row["bad_segmentation"] = int(bad)
                bad_candidates.append((float(met["iou"]), img_path))

                if bad:
                    diag = make_diagnostic_image(
                        image,
                        pred_pupil,
                        gt_pupil,
                        cirpupil,
                        ciriris,
                        [
                            f"file: {rel_path}",
                            f"IoU={met['iou']:.4f}, Dice={met['dice']:.4f}, PA={met['pixel_accuracy']:.4f}",
                            f"pupil pred y/x/r = {cirpupil[0]}/{cirpupil[1]}/{cirpupil[2]}",
                            f"iris y/x/r = {ciriris[0]}/{ciriris[1]}/{ciriris[2]}",
                            f"GT: {gt_path.name}",
                        ],
                    )
                    out_name = f"{idx:04d}_{img_path.stem}_IoU_{met['iou']:.3f}_Dice_{met['dice']:.3f}.png"
                    cv2.imwrite(str(overlays_bad_dir / out_name), diag)
            else:
                base_row["status"] = "ok_missing_gt"
                base_row["bad_segmentation"] = "missing_gt"
                diag = make_diagnostic_image(
                    image,
                    pred_pupil,
                    None,
                    cirpupil,
                    ciriris,
                    [
                        f"file: {rel_path}",
                        "Brak maski GT dla tego obrazu.",
                        f"pupil pred y/x/r = {cirpupil[0]}/{cirpupil[1]}/{cirpupil[2]}",
                        f"iris y/x/r = {ciriris[0]}/{ciriris[1]}/{ciriris[2]}",
                    ],
                )
                cv2.imwrite(str(overlays_missing_gt_dir / f"{idx:04d}_{img_path.stem}_missing_gt.png"), diag)

            features.append(
                FeatureRecord(
                    dataset=dataset_name,
                    image_path=str(img_path),
                    rel_path=rel_path,
                    identity=identity,
                    template=template,
                    mask=mask,
                    feature_total_s=float(base_row.get("feature_total_s", 0.0)),
                    segmentation_total_s=float(base_row.get("segmentation_total_s", 0.0)),
                )
            )

        except Exception as e:
            base_row["status"] = "failed"
            base_row["error"] = repr(e)
            base_row["traceback"] = traceback.format_exc(limit=5)
            try:
                failed_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if failed_img is not None:
                    diag = make_diagnostic_image(
                        failed_img,
                        None,
                        None,
                        None,
                        None,
                        [
                            f"file: {rel_path}",
                            "BŁĄD PRZETWARZANIA / SEGMENTACJI",
                            repr(e)[:200],
                        ],
                    )
                    cv2.imwrite(str(overlays_failed_dir / f"{idx:04d}_{img_path.stem}_FAILED.png"), diag)
            except Exception:
                pass

        image_rows.append(base_row)

    # Dodatkowy folder z najgorszymi przykładami wg IoU
    rows_with_iou = [r for r in image_rows if safe_float(r.get("iou")) is not None]
    worst_rows = sorted(rows_with_iou, key=lambda r: float(r["iou"]))[:N_WORST_EXAMPLES]
    for rank, r in enumerate(worst_rows, 1):
        stem = Path(r["image_path"]).stem
        # Spróbuj znaleźć już zapisany overlay bad; jeśli nie, stwórz z pred i GT ponownie.
        for cand in overlays_bad_dir.glob(f"*_{stem}_IoU_*.png"):
            shutil.copy2(cand, worst_dir / f"rank_{rank:03d}_{cand.name}")
            break

    # CSV obrazów
    fieldnames = sorted({k for row in image_rows for k in row.keys()})
    write_csv(dataset_out / "per_image_segmentation_and_feature_times.csv", image_rows, fieldnames)

    # Identyfikacja top-1
    identification_rows = evaluate_identification_top1(features, dataset_out)

    # Podsumowanie
    summary = build_summary(dataset_name, images, image_rows, identification_rows, global_counts, bad_iou_threshold, bad_dice_threshold)
    with open(dataset_out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    write_summary_txt(dataset_out / "summary.txt", summary)

    maybe_make_plots(dataset_out, image_rows, summary)

    print(f"\n[{dataset_name}] ZAKOŃCZONO. Wyniki: {dataset_out}")
    print(f"[{dataset_name}] IoU mean: {summary.get('pupil_iou', {}).get('mean')} | Dice mean: {summary.get('pupil_dice', {}).get('mean')}")
    print(f"[{dataset_name}] Top-1 identification accuracy: {summary.get('identification_top1_accuracy')}")

    return summary


def evaluate_identification_top1(features: List[FeatureRecord], dataset_out: Path) -> List[dict]:
    from fnc.matching import calHammingDist

    rows: List[dict] = []
    if not features:
        write_csv(dataset_out / "identification_top1_results.csv", rows, [])
        return rows

    # Pierwszy poprawnie przetworzony obraz każdej osoby jako gallery, reszta jako probe.
    by_id: Dict[str, List[FeatureRecord]] = {}
    for f in features:
        by_id.setdefault(f.identity, []).append(f)
    for ident in by_id:
        by_id[ident] = sorted(by_id[ident], key=lambda x: natural_key(Path(x.rel_path)))

    gallery: Dict[str, FeatureRecord] = {ident: recs[0] for ident, recs in by_id.items() if recs}
    probes: List[FeatureRecord] = []
    for ident, recs in by_id.items():
        probes.extend(recs[1:])

    print(f"[IDENTYFIKACJA] gallery identities: {len(gallery)}, probes: {len(probes)}")

    for i, probe in enumerate(probes, 1):
        t_total = now_s()
        distances = []
        comparison_times = []
        for gid, grec in gallery.items():
            t = now_s()
            hd = calHammingDist(probe.template, probe.mask, grec.template, grec.mask)
            comparison_times.append(now_s() - t)
            distances.append((gid, hd, grec.rel_path))
        distances_valid = [(gid, hd, gpath) for gid, hd, gpath in distances if hd is not None and not np.isnan(hd)]
        if distances_valid:
            best_gid, best_hd, best_gallery_path = min(distances_valid, key=lambda x: x[1])
            correct = int(best_gid == probe.identity)
        else:
            best_gid, best_hd, best_gallery_path, correct = "", np.nan, "", 0

        rows.append({
            "probe_index": i,
            "probe_path": probe.image_path,
            "probe_rel_path": probe.rel_path,
            "true_identity": probe.identity,
            "predicted_identity": best_gid,
            "correct": correct,
            "best_hamming_distance": best_hd,
            "best_gallery_rel_path": best_gallery_path,
            "num_gallery_comparisons": len(gallery),
            "matching_total_s": now_s() - t_total,
            "matching_mean_single_comparison_s": float(np.mean(comparison_times)) if comparison_times else None,
            "matching_min_single_comparison_s": float(np.min(comparison_times)) if comparison_times else None,
            "matching_max_single_comparison_s": float(np.max(comparison_times)) if comparison_times else None,
        })

        if i == 1 or i % 50 == 0 or i == len(probes):
            print(f"[IDENTYFIKACJA] probe {i}/{len(probes)}")

    if rows:
        fieldnames = sorted({k for row in rows for k in row.keys()})
        write_csv(dataset_out / "identification_top1_results.csv", rows, fieldnames)
    else:
        with open(dataset_out / "identification_top1_results.csv", "w", encoding="utf-8") as f:
            f.write("no_probe_images\n")
    return rows


def build_summary(dataset_name: str, images: List[Path], image_rows: List[dict], identification_rows: List[dict], global_counts: Dict[str, int], bad_iou_threshold: float, bad_dice_threshold: float) -> dict:
    ok_rows = [r for r in image_rows if str(r.get("status", "")).startswith("ok")]
    failed_rows = [r for r in image_rows if r.get("status") == "failed"]
    gt_rows = [r for r in image_rows if safe_float(r.get("iou")) is not None]
    bad_rows = [r for r in gt_rows if str(r.get("bad_segmentation")) == "1"]

    summary = {
        "dataset": dataset_name,
        "number_of_images_found": len(images),
        "number_of_images_processed_ok": len(ok_rows),
        "number_of_processing_failures": len(failed_rows),
        "number_of_images_with_gt_metrics": len(gt_rows),
        "number_of_bad_segmentations_saved": len(bad_rows),
        "bad_iou_threshold": bad_iou_threshold,
        "bad_dice_threshold": bad_dice_threshold,
        "global_counts": global_counts,
        "stage_times_seconds": {},
    }

    for key in [
        "read_s", "inner_pupil_s", "outer_iris_s", "top_eyelid_s", "bottom_eyelid_s",
        "eyelashes_s", "segmentation_total_s", "pupil_mask_s", "gt_load_s", "metrics_s",
        "normalization_s", "encoding_s", "feature_total_s",
    ]:
        summary["stage_times_seconds"][key] = aggregate_numeric(image_rows, key)

    for key, out_key in [
        ("iou", "pupil_iou"),
        ("dice", "pupil_dice"),
        ("pixel_accuracy", "pupil_pixel_accuracy"),
        ("precision", "pupil_precision"),
        ("recall", "pupil_recall"),
        ("specificity", "pupil_specificity"),
        ("fpr", "pupil_fpr"),
        ("fnr", "pupil_fnr"),
    ]:
        summary[out_key] = aggregate_numeric(image_rows, key)

    tp = global_counts.get("tp", 0)
    fp = global_counts.get("fp", 0)
    fn = global_counts.get("fn", 0)
    tn = global_counts.get("tn", 0)
    eps = 1e-12
    summary["global_pupil_iou"] = tp / (tp + fp + fn + eps) if (tp + fp + fn) > 0 else None
    summary["global_pupil_dice"] = 2 * tp / (2 * tp + fp + fn + eps) if (2 * tp + fp + fn) > 0 else None
    summary["global_pupil_pixel_accuracy"] = (tp + tn) / (tp + fp + fn + tn + eps) if (tp + fp + fn + tn) > 0 else None

    if identification_rows:
        correct = sum(int(r.get("correct", 0)) for r in identification_rows)
        total = len(identification_rows)
        summary["identification_num_probes"] = total
        summary["identification_correct_top1"] = correct
        summary["identification_top1_accuracy"] = correct / total if total else None
        summary["matching_time_seconds"] = {
            "matching_total_s": aggregate_numeric(identification_rows, "matching_total_s"),
            "matching_mean_single_comparison_s": aggregate_numeric(identification_rows, "matching_mean_single_comparison_s"),
        }
    else:
        summary["identification_num_probes"] = 0
        summary["identification_correct_top1"] = 0
        summary["identification_top1_accuracy"] = None
        summary["matching_time_seconds"] = {}

    return summary


def write_summary_txt(path: Path, summary: dict) -> None:
    mkdir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write("PODSUMOWANIE EKSPERYMENTU CASIA V1 — MASEK/KOVESI + PUPIL GT\n")
        f.write("=" * 90 + "\n")
        f.write(f"Dataset: {summary.get('dataset')}\n")
        f.write(f"Liczba obrazów: {summary.get('number_of_images_found')}\n")
        f.write(f"Przetworzone OK: {summary.get('number_of_images_processed_ok')}\n")
        f.write(f"Błędy przetwarzania: {summary.get('number_of_processing_failures')}\n")
        f.write(f"Obrazy z metrykami GT: {summary.get('number_of_images_with_gt_metrics')}\n")
        f.write(f"Zapisane błędne segmentacje: {summary.get('number_of_bad_segmentations_saved')}\n")
        f.write("\n[SEGMENTACJA ŹRENICY VS GT]\n")
        for key in ["pupil_iou", "pupil_dice", "pupil_pixel_accuracy", "pupil_precision", "pupil_recall", "pupil_specificity", "pupil_fpr", "pupil_fnr"]:
            stats = summary.get(key, {})
            f.write(f"{key}: mean={stats.get('mean')} std={stats.get('std')} min={stats.get('min')} max={stats.get('max')}\n")
        f.write(f"global_pupil_iou={summary.get('global_pupil_iou')}\n")
        f.write(f"global_pupil_dice={summary.get('global_pupil_dice')}\n")
        f.write(f"global_pupil_pixel_accuracy={summary.get('global_pupil_pixel_accuracy')}\n")
        f.write("\n[CZASY ETAPÓW — SEKUNDY]\n")
        for key, stats in summary.get("stage_times_seconds", {}).items():
            f.write(f"{key}: mean={stats.get('mean')} std={stats.get('std')} min={stats.get('min')} max={stats.get('max')} sum={stats.get('sum')}\n")
        f.write("\n[IDENTYFIKACJA TOP-1]\n")
        f.write(f"Probe: {summary.get('identification_num_probes')}\n")
        f.write(f"Correct: {summary.get('identification_correct_top1')}\n")
        f.write(f"Top-1 accuracy: {summary.get('identification_top1_accuracy')}\n")
        f.write("\n[MATCHING]\n")
        for key, stats in summary.get("matching_time_seconds", {}).items():
            f.write(f"{key}: mean={stats.get('mean')} std={stats.get('std')} min={stats.get('min')} max={stats.get('max')} sum={stats.get('sum')}\n")


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pełna ewaluacja CASIA-IrisV1 — Masek/Kovesi Python.")
    parser.add_argument("--code-dir", default=DEFAULT_PYTHON_CODE_DIR, help="Folder python z oryginalnego Iris-Recognition-master, zawierający fnc/.")
    parser.add_argument("--gt-dir", default=DEFAULT_GT_DIR, help="Folder z maskami GT źrenicy.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Folder wyjściowy wyników.")
    parser.add_argument("--dataset", choices=["jpg", "bmp", "both"], default="both", help="Którą bazę uruchomić.")
    parser.add_argument("--jpg-dir", default=DEFAULT_DATASETS[0]["images_dir"], help="Folder bazy CASIA JPG.")
    parser.add_argument("--bmp-dir", default=DEFAULT_DATASETS[1]["images_dir"], help="Folder bazy CASIA BMP.")
    parser.add_argument("--bad-iou-threshold", type=float, default=BAD_IOU_THRESHOLD, help="Próg IoU poniżej którego zapisujemy błąd.")
    parser.add_argument("--bad-dice-threshold", type=float, default=BAD_DICE_THRESHOLD, help="Próg Dice poniżej którego zapisujemy błąd.")
    parser.add_argument("--max-images", type=int, default=0, help="Opcjonalnie: ogranicz liczbę obrazów na bazę, np. 10 do testu. 0 = cała baza.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    add_original_code_to_path(args.code_dir)

    selected = []
    if args.dataset in ("jpg", "both"):
        ds = dict(DEFAULT_DATASETS[0])
        ds["images_dir"] = args.jpg_dir
        selected.append(ds)
    if args.dataset in ("bmp", "both"):
        ds = dict(DEFAULT_DATASETS[1])
        ds["images_dir"] = args.bmp_dir
        selected.append(ds)

    output_dir = Path(args.output_dir)
    mkdir(output_dir)

    all_summaries = {}
    t_all = now_s()
    for ds in selected:
        summary = evaluate_dataset(
            dataset_name=ds["name"],
            images_dir=Path(ds["images_dir"]),
            extensions=ds["extensions"],
            gt_dir=Path(args.gt_dir),
            output_dir=output_dir,
            bad_iou_threshold=args.bad_iou_threshold,
            bad_dice_threshold=args.bad_dice_threshold,
            max_images=args.max_images,
        )
        all_summaries[ds["name"]] = summary

    all_summaries["total_runtime_s"] = now_s() - t_all
    with open(output_dir / "summary_all_datasets.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 90)
    print("ZAKOŃCZONO CAŁY EKSPERYMENT PYTHON")
    print(f"Folder wyników: {output_dir}")
    print(f"Całkowity czas: {all_summaries['total_runtime_s']:.3f} s")
    print("=" * 90)


if __name__ == "__main__":
    main()
