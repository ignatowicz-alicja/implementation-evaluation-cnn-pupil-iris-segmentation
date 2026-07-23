from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, auc, precision_recall_curve, roc_curve


def save_history_plots(history: dict[str, list[float]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = [
        ("loss", "val_loss", "Loss"),
        ("binary_accuracy", "val_binary_accuracy", "Pixel accuracy"),
        ("dice_coef", "val_dice_coef", "Dice"),
        ("iou_coef", "val_iou_coef", "IoU"),
        ("precision", "val_precision", "Precision"),
        ("recall", "val_recall", "Recall"),
    ]
    for train_key, val_key, title in groups:
        if train_key not in history:
            continue
        plt.figure(figsize=(8, 5))
        plt.plot(history[train_key], label="train")
        if val_key in history:
            plt.plot(history[val_key], label="validation")
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"history_{train_key}.png", dpi=180)
        plt.close()


def save_confusion(cm: np.ndarray, labels: list[str], path: Path, normalize: bool = False, title: str = "Confusion matrix") -> None:
    values = cm.astype(float)
    if normalize:
        row_sum = values.sum(axis=1, keepdims=True)
        values = np.divide(values, row_sum, out=np.zeros_like(values), where=row_sum != 0)
    size = min(18, max(6, len(labels) * 0.18))
    fig, ax = plt.subplots(figsize=(size, size))
    display = ConfusionMatrixDisplay(values, display_labels=labels)
    display.plot(ax=ax, include_values=len(labels) <= 30, xticks_rotation=90, colorbar=True, values_format=".2f" if normalize else ".0f")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_metric_distributions(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = ["dice_f1", "iou_jaccard", "precision", "recall_sensitivity", "specificity", "pixel_accuracy", "mcc", "boundary_f1", "hd95_px", "time_inference_s", "time_total_s"]
    for key in keys:
        vals = np.asarray([float(r.get(key, np.nan)) for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        if not vals.size:
            continue
        plt.figure(figsize=(8, 5))
        plt.hist(vals, bins=min(40, max(10, int(math.sqrt(vals.size)))))
        plt.axvline(float(np.mean(vals)), linestyle="--", label=f"mean={np.mean(vals):.4f}")
        plt.axvline(float(np.median(vals)), linestyle=":", label=f"median={np.median(vals):.4f}")
        plt.xlabel(key)
        plt.ylabel("Number of images")
        plt.title(f"Distribution: {key}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"hist_{key}.png", dpi=180)
        plt.close()

    if rows:
        x = np.asarray([float(r.get("foreground_fraction_gt", np.nan)) for r in rows])
        y = np.asarray([float(r.get("dice_f1", np.nan)) for r in rows])
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.any():
            plt.figure(figsize=(7, 5))
            plt.scatter(x[ok], y[ok], s=12, alpha=0.6)
            plt.xlabel("GT pupil area fraction")
            plt.ylabel("Dice")
            plt.title("Dice vs pupil size")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / "scatter_dice_vs_pupil_size.png", dpi=180)
            plt.close()


def save_roc_pr(y_true: np.ndarray, y_score: np.ndarray, out_dir: Path, prefix: str = "pixel") -> dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = float(auc(fpr, tpr))
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = float(auc(recall[::-1], precision[::-1]))

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.5f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_roc.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"AUC={pr_auc:.5f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_precision_recall.png", dpi=180)
    plt.close()
    return {"roc_auc": roc_auc, "pr_auc": pr_auc}


def save_threshold_sweep(rows: list[dict], out_dir: Path) -> None:
    if not rows:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = [r["threshold"] for r in rows]
    plt.figure(figsize=(8, 5))
    for key in ("dice_f1", "iou_jaccard", "precision", "recall_sensitivity", "specificity"):
        plt.plot(thresholds, [r[key] for r in rows], label=key)
    plt.xlabel("Threshold")
    plt.ylabel("Metric")
    plt.title("Global threshold sweep")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "threshold_sweep.png", dpi=180)
    plt.close()
