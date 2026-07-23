from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sklearn.calibration import calibration_curve

from .dataset import Record, read_image, read_manifest, read_mask
from .metrics import aggregate_rows, binary_metrics, binary_metrics_from_counts, confusion_counts
from .model_author import create_author_model
from .plots import save_confusion, save_history_plots, save_metric_distributions, save_roc_pr, save_threshold_sweep


def environment_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version,
        "processor": platform.processor(),
        "machine": platform.machine(),
    }
    for name in ("numpy", "cv2", "scipy", "sklearn", "matplotlib", "tensorflow", "keras"):
        try:
            module = __import__(name)
            report[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            report[name] = f"unavailable: {exc}"
    try:
        import tensorflow as tf
        report["tensorflow_devices"] = [str(x) for x in tf.config.list_physical_devices()]
        report["tensorflow_gpu_devices"] = [str(x) for x in tf.config.list_physical_devices("GPU")]
    except Exception:
        pass
    try:
        text = subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.STDOUT, timeout=8)
        report["nvidia_smi"] = text
    except Exception as exc:
        report["nvidia_smi"] = f"unavailable: {exc}"
    return report


def tf_metrics():
    import keras.backend as K
    from keras.metrics import BinaryAccuracy, Precision, Recall

    def dice_coef(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2.0 * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

    def iou_coef(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
        return (intersection + K.epsilon()) / (union + K.epsilon())

    dice_coef.__name__ = "dice_coef"
    iou_coef.__name__ = "iou_coef"
    return [BinaryAccuracy(name="binary_accuracy"), Precision(name="precision"), Recall(name="recall"), dice_coef, iou_coef]


class SegmentationSequence:
    def __new__(cls, *args, **kwargs):
        from keras.utils import Sequence

        class _Sequence(Sequence):
            def __init__(self, records, batch_size, shuffle, polarity, seed):
                self.records = list(records)
                self.batch_size = int(batch_size)
                self.shuffle = bool(shuffle)
                self.polarity = polarity
                self.rng = np.random.default_rng(seed)
                self.indices = np.arange(len(self.records))
                self.on_epoch_end()

            def __len__(self):
                return int(np.ceil(len(self.records) / self.batch_size))

            def __getitem__(self, index):
                ids = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
                x = np.zeros((len(ids), 240, 320, 1), np.float32)
                y = np.zeros((len(ids), 240, 320, 1), np.float32)
                for j, idx in enumerate(ids):
                    _, image = read_image(self.records[idx].image_path)
                    mask = read_mask(self.records[idx].mask_path, polarity=self.polarity)
                    x[j] = image
                    y[j, ..., 0] = mask
                return x, y

            def on_epoch_end(self):
                if self.shuffle:
                    self.rng.shuffle(self.indices)

        return _Sequence(*args, **kwargs)


class EpochTimer:
    def __new__(cls, csv_path: Path):
        from keras.callbacks import Callback

        class _Timer(Callback):
            def __init__(self, path):
                super().__init__()
                self.path = Path(path)
                self.rows = []
                self.start = 0.0

            def on_epoch_begin(self, epoch, logs=None):
                self.start = time.perf_counter()

            def on_epoch_end(self, epoch, logs=None):
                row = {"epoch": epoch + 1, "time_s": time.perf_counter() - self.start}
                row.update({k: float(v) for k, v in (logs or {}).items()})
                self.rows.append(row)
                self.path.parent.mkdir(parents=True, exist_ok=True)
                keys = sorted({k for item in self.rows for k in item})
                with self.path.open("w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(self.rows)

        return _Timer(csv_path)


def train_model(author_dir: Path, manifest: Path, output_dir: Path, epochs: int, batch_size: int, learning_rate: float, patience: int, polarity: str, seed: int) -> Path:
    import keras
    from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "environment.json").write_text(json.dumps(environment_report(), indent=2, ensure_ascii=False), encoding="utf-8")
    train_records = read_manifest(manifest, "train")
    val_records = read_manifest(manifest, "val")
    if not train_records or not val_records:
        raise RuntimeError("Manifest musi zawierać rekordy train i val.")

    keras.utils.set_random_seed(seed)
    model = create_author_model(author_dir)
    summary_lines = []
    model.summary(print_fn=summary_lines.append)
    (output_dir / "model_architecture_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="binary_crossentropy", metrics=tf_metrics())

    train_seq = SegmentationSequence(train_records, batch_size, True, polarity, seed)
    val_seq = SegmentationSequence(val_records, batch_size, False, polarity, seed)
    weights_path = output_dir / "best_author_unet.weights.h5"
    callbacks = [
        ModelCheckpoint(str(weights_path), monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(2, patience // 2), min_lr=1e-7, verbose=1),
        CSVLogger(str(output_dir / "training_history.csv"), append=False),
        EpochTimer(output_dir / "epoch_times.csv"),
    ]
    start = time.perf_counter()
    history = model.fit(train_seq, validation_data=val_seq, epochs=epochs, callbacks=callbacks, verbose=1)
    total = time.perf_counter() - start
    model.save_weights(str(output_dir / "last_author_unet.weights.h5"))
    save_history_plots(history.history, output_dir / "plots")
    summary = {
        "total_training_time_s": total,
        "epochs_requested": epochs,
        "epochs_completed": len(history.history.get("loss", [])),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_images": len(train_records),
        "val_images": len(val_records),
        "architecture_source": str(Path(author_dir) / "UNet_iris_segmentation.py"),
        "author_code_modified": False,
        "best_weights": str(weights_path),
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return weights_path


def overlay_image(gray: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    tp = (gt > 0) & (pred > 0)
    fp = (gt == 0) & (pred > 0)
    fn = (gt > 0) & (pred == 0)
    overlay = rgb.astype(np.float32)
    alpha = 0.58
    for mask, color in ((tp, (0, 255, 0)), (fp, (0, 0, 255)), (fn, (255, 0, 0))):
        overlay[mask] = (1 - alpha) * overlay[mask] + alpha * np.asarray(color)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_model(author_dir: Path, manifest: Path, weights: Path, output_dir: Path, split: str, threshold: float, polarity: str, save_all: bool, save_examples: int, pixels_per_image: int, seed: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = read_manifest(manifest, split)
    if not records:
        raise RuntimeError(f"Brak rekordów split={split}")
    model_load_start = time.perf_counter()
    model = create_author_model(author_dir)
    model.load_weights(str(weights))
    model_load_s = time.perf_counter() - model_load_start

    # Warm-up is deliberately excluded from reported image times.
    dummy = np.zeros((1, 240, 320, 1), dtype=np.float32)
    for _ in range(3):
        _ = model(dummy, training=False).numpy()

    rows: list[dict] = []
    sampled_true: list[np.ndarray] = []
    sampled_score: list[np.ndarray] = []
    threshold_counts = {float(t): [0, 0, 0, 0] for t in np.linspace(0.05, 0.95, 19)}
    rng = np.random.default_rng(seed)
    pred_dir = output_dir / "predicted_masks"
    prob_dir = output_dir / "probability_maps"
    overlay_dir = output_dir / "overlays_all"
    if save_all:
        pred_dir.mkdir(parents=True, exist_ok=True)
        prob_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)

    for index, record in enumerate(records, start=1):
        total_start = time.perf_counter()
        pre_start = time.perf_counter()
        gray, image = read_image(record.image_path)
        gt = read_mask(record.mask_path, polarity=polarity)
        pre_s = time.perf_counter() - pre_start

        infer_start = time.perf_counter()
        probability = model(image[None, ...], training=False).numpy()[0, ..., 0]
        infer_s = time.perf_counter() - infer_start

        post_start = time.perf_counter()
        pred = (probability >= threshold).astype(np.uint8)
        metrics = binary_metrics(gt, pred, with_boundary=True)
        post_s = time.perf_counter() - post_start

        save_start = time.perf_counter()
        rel = Path(record.relative_image)
        out_rel = rel.with_suffix(".png")
        if save_all:
            for base in (pred_dir, prob_dir, overlay_dir):
                (base / out_rel.parent).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(pred_dir / out_rel), pred * 255)
            cv2.imwrite(str(prob_dir / out_rel), np.clip(probability * 255, 0, 255).astype(np.uint8))
            cv2.imwrite(str(overlay_dir / out_rel), overlay_image(gray, gt, pred))
        save_s = time.perf_counter() - save_start
        total_s = time.perf_counter() - total_start

        row = {
            "record_id": record.record_id,
            "subject": record.subject,
            "relative_image": record.relative_image,
            "image_path": record.image_path,
            "mask_path": record.mask_path,
            "threshold": threshold,
            **metrics,
            "time_preprocess_s": pre_s,
            "time_inference_s": infer_s,
            "time_postprocess_metrics_s": post_s,
            "time_save_s": save_s,
            "time_total_s": total_s,
        }
        rows.append(row)

        flat_n = gt.size
        count = min(pixels_per_image, flat_n)
        idx = rng.choice(flat_n, size=count, replace=False)
        sampled_true.append(gt.ravel()[idx])
        sampled_score.append(probability.ravel()[idx])
        for t, counts in threshold_counts.items():
            tn, fp, fn, tp = confusion_counts(gt, probability >= t)
            counts[0] += tn; counts[1] += fp; counts[2] += fn; counts[3] += tp
        print(f"[EVAL] {index}/{len(records)} Dice={metrics['dice_f1']:.4f} IoU={metrics['iou_jaccard']:.4f} infer={infer_s:.4f}s")

    _write_csv(output_dir / "per_image_metrics_and_times.csv", rows)
    aggregate = aggregate_rows(rows)
    aggregate["model_load_time_s"] = model_load_s
    aggregate["split"] = split
    aggregate["threshold"] = threshold
    aggregate["weights"] = str(Path(weights).resolve())
    aggregate["timing_note"] = "Inference measured after 3 warm-up calls; .numpy() synchronizes device execution. Save time is separate."

    y_true = np.concatenate(sampled_true).astype(np.uint8)
    y_score = np.concatenate(sampled_score).astype(np.float32)
    aggregate["sampled_pixel_curves"] = save_roc_pr(y_true, y_score, output_dir / "plots")

    cm = np.array([[aggregate["confusion_total"]["tn"], aggregate["confusion_total"]["fp"]], [aggregate["confusion_total"]["fn"], aggregate["confusion_total"]["tp"]]])
    np.savetxt(output_dir / "segmentation_confusion_raw.csv", cm, delimiter=",", fmt="%d", header="pred_background,pred_pupil", comments="")
    save_confusion(cm, ["background", "pupil"], output_dir / "plots" / "segmentation_confusion_raw.png", False, "Segmentation confusion matrix – raw pixels")
    save_confusion(cm, ["background", "pupil"], output_dir / "plots" / "segmentation_confusion_normalized.png", True, "Segmentation confusion matrix – row normalized")
    save_metric_distributions(rows, output_dir / "plots")

    sweep_rows = []
    for t, (tn, fp, fn, tp) in threshold_counts.items():
        sweep_rows.append({"threshold": t, "tn": tn, "fp": fp, "fn": fn, "tp": tp, **binary_metrics_from_counts(tn, fp, fn, tp)})
    _write_csv(output_dir / "threshold_sweep.csv", sweep_rows)
    save_threshold_sweep(sweep_rows, output_dir / "plots")
    best = max(sweep_rows, key=lambda x: x["dice_f1"])
    aggregate["best_global_threshold_by_dice"] = best
    best_cm = np.array([[best["tn"], best["fp"]], [best["fn"], best["tp"]]])
    np.savetxt(output_dir / "segmentation_confusion_best_threshold_raw.csv", best_cm, delimiter=",", fmt="%d", header="pred_background,pred_pupil", comments="")
    save_confusion(best_cm, ["background", "pupil"], output_dir / "plots" / "segmentation_confusion_best_threshold_raw.png", False, f"Segmentation confusion – best Dice threshold {best['threshold']:.2f}")
    save_confusion(best_cm, ["background", "pupil"], output_dir / "plots" / "segmentation_confusion_best_threshold_normalized.png", True, f"Segmentation confusion normalized – best Dice threshold {best['threshold']:.2f}")

    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=15, strategy="quantile")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 5))
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed pupil frequency")
    plt.title("Pixel probability calibration")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "pixel_calibration.png", dpi=180)
    plt.close()

    # Per-subject macro statistics.
    subject_rows = []
    for subject in sorted({r["subject"] for r in rows}):
        group = [r for r in rows if r["subject"] == subject]
        item = {"subject": subject, **aggregate_rows(group)["macro_mean"], "images": len(group)}
        subject_rows.append(item)
    _write_csv(output_dir / "per_subject_metrics.csv", subject_rows)

    ranked = sorted(rows, key=lambda r: r["dice_f1"])
    examples_dir = output_dir / "examples_best_worst"
    examples_dir.mkdir(parents=True, exist_ok=True)
    chosen = [("worst", r) for r in ranked[:save_examples]] + [("best", r) for r in ranked[-save_examples:]]
    for label, row in chosen:
        gray, image = read_image(row["image_path"])
        gt = read_mask(row["mask_path"], polarity=polarity)
        probability = model(image[None, ...], training=False).numpy()[0, ..., 0]
        pred = (probability >= threshold).astype(np.uint8)
        name = f"{label}_dice_{row['dice_f1']:.4f}_{row['subject']}_{Path(row['relative_image']).stem}.png"
        cv2.imwrite(str(examples_dir / name), overlay_image(gray, gt, pred))

    (output_dir / "evaluation_summary.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")
    return aggregate
