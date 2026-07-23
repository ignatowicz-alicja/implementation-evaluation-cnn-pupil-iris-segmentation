from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    t = y_true.astype(bool).ravel()
    p = y_pred.astype(bool).ravel()
    tp = int(np.count_nonzero(t & p))
    tn = int(np.count_nonzero(~t & ~p))
    fp = int(np.count_nonzero(~t & p))
    fn = int(np.count_nonzero(t & ~p))
    return tn, fp, fn, tp


def boundary_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    m = mask.astype(bool)
    if not m.any():
        return m
    eroded = binary_erosion(m, iterations=iterations, border_value=0)
    return m ^ eroded


def boundary_scores(y_true: np.ndarray, y_pred: np.ndarray, tolerance: int = 2) -> dict[str, float]:
    bt = boundary_mask(y_true)
    bp = boundary_mask(y_pred)
    if not bt.any() and not bp.any():
        return {"boundary_f1": 1.0, "boundary_iou": 1.0, "hd95_px": 0.0, "assd_px": 0.0}
    if not bt.any() or not bp.any():
        return {"boundary_f1": 0.0, "boundary_iou": 0.0, "hd95_px": float("nan"), "assd_px": float("nan")}

    kernel = np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8)
    dt = cv2.dilate(bt.astype(np.uint8), kernel).astype(bool)
    dp = cv2.dilate(bp.astype(np.uint8), kernel).astype(bool)
    matched_p = np.count_nonzero(bp & dt)
    matched_t = np.count_nonzero(bt & dp)
    precision = safe_div(matched_p, np.count_nonzero(bp))
    recall = safe_div(matched_t, np.count_nonzero(bt))
    f1 = safe_div(2 * precision * recall, precision + recall)
    union = np.count_nonzero(bt | bp)
    iou = safe_div(np.count_nonzero(bt & bp), union)

    dist_to_t = distance_transform_edt(~bt)
    dist_to_p = distance_transform_edt(~bp)
    d_pt = dist_to_t[bp]
    d_tp = dist_to_p[bt]
    distances = np.concatenate([d_pt, d_tp])
    hd95 = float(np.percentile(distances, 95)) if distances.size else 0.0
    assd = float((d_pt.mean() + d_tp.mean()) / 2.0) if d_pt.size and d_tp.size else float("nan")
    return {"boundary_f1": f1, "boundary_iou": iou, "hd95_px": hd95, "assd_px": assd}


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, with_boundary: bool = True) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    npv = safe_div(tn, tn + fn)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    dice = safe_div(2 * tp, 2 * tp + fp + fn)
    iou = safe_div(tp, tp + fp + fn)
    balanced = (recall + specificity) / 2.0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = safe_div(tp * tn - fp * fn, denom)
    result: dict[str, Any] = {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "pixel_accuracy": accuracy,
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "npv": npv,
        "dice_f1": dice,
        "iou_jaccard": iou,
        "ioc_iou": iou,
        "balanced_accuracy": balanced,
        "mcc": mcc,
        "fpr": safe_div(fp, fp + tn),
        "fnr": safe_div(fn, fn + tp),
        "fdr": safe_div(fp, fp + tp),
        "for": safe_div(fn, fn + tn),
        "foreground_fraction_gt": float(np.mean(y_true > 0)),
        "foreground_fraction_pred": float(np.mean(y_pred > 0)),
    }
    if with_boundary:
        result.update(boundary_scores(y_true, y_pred))
    return result


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    total_tn = sum(int(r["tn"]) for r in rows)
    total_fp = sum(int(r["fp"]) for r in rows)
    total_fn = sum(int(r["fn"]) for r in rows)
    total_tp = sum(int(r["tp"]) for r in rows)
    micro = binary_metrics_from_counts(total_tn, total_fp, total_fn, total_tp)
    numeric_keys = [k for k, v in rows[0].items() if isinstance(v, (int, float, np.integer, np.floating)) and k not in {"tn", "fp", "fn", "tp"}]
    macro: dict[str, float] = {}
    std: dict[str, float] = {}
    med: dict[str, float] = {}
    for key in numeric_keys:
        values = np.asarray([float(r[key]) for r in rows], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size:
            macro[key] = float(np.mean(finite))
            std[key] = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
            med[key] = float(np.median(finite))
    return {
        "count": len(rows),
        "confusion_total": {"tn": total_tn, "fp": total_fp, "fn": total_fn, "tp": total_tp},
        "micro": micro,
        "macro_mean": macro,
        "macro_std": std,
        "median": med,
    }


def binary_metrics_from_counts(tn: int, fp: int, fn: int, tp: int) -> dict[str, float]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    dice = safe_div(2 * tp, 2 * tp + fp + fn)
    iou = safe_div(tp, tp + fp + fn)
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "pixel_accuracy": safe_div(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "npv": safe_div(tn, tn + fn),
        "dice_f1": dice,
        "iou_jaccard": iou,
        "ioc_iou": iou,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "mcc": safe_div(tp * tn - fp * fn, denom),
        "fpr": safe_div(fp, fp + tn),
        "fnr": safe_div(fn, fn + tp),
    }
