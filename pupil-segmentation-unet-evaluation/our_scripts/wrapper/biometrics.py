from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

from .dataset import read_image, read_manifest, read_mask
from .model_author import create_author_model
from .plots import save_confusion


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
        writer.writeheader(); writer.writerows(rows)


def pupil_circle(mask: np.ndarray) -> tuple[float, float, float] | None:
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 8:
        return None
    (x, y), radius = cv2.minEnclosingCircle(contour)
    return float(x), float(y), float(radius)


def annular_feature(gray: np.ndarray, pupil: np.ndarray, outer_scale: float = 3.0) -> np.ndarray | None:
    circle = pupil_circle(pupil)
    if circle is None:
        return None
    x, y, r = circle
    max_possible = min(x, y, gray.shape[1] - 1 - x, gray.shape[0] - 1 - y)
    outer = min(max_possible, max(r + 8.0, r * outer_scale))
    if outer <= r + 4:
        return None
    angular = 256
    radial = 96
    polar = cv2.warpPolar(gray, (radial, angular), (x, y), outer, cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS)
    inner_col = int(np.clip(round(radial * r / outer), 0, radial - 2))
    annulus = polar[:, inner_col:]
    annulus = cv2.resize(annulus, (128, 64), interpolation=cv2.INTER_AREA)
    annulus = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(annulus)
    f = annulus.astype(np.float32) / 255.0
    mean = cv2.GaussianBlur(f, (0, 0), 3.0)
    sq_mean = cv2.GaussianBlur(f * f, (0, 0), 3.0)
    std = np.sqrt(np.maximum(sq_mean - mean * mean, 1e-4))
    normalized = (f - mean) / std
    gx = cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(normalized, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    features = []
    for plane in (normalized, magnitude):
        for yy in range(0, 64, 8):
            for xx in range(0, 128, 8):
                block = plane[yy:yy+8, xx:xx+8]
                features.extend([float(block.mean()), float(block.std())])
    vector = np.asarray(features, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def cosine_matrix(probes: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    return probes @ gallery.T


def interpolate_tar_at_far(fpr: np.ndarray, tpr: np.ndarray, target: float) -> float:
    valid = np.where(fpr <= target)[0]
    return float(np.max(tpr[valid])) if valid.size else 0.0


def verification_metrics(genuine: list[float], impostor: list[float], out_dir: Path) -> dict:
    y = np.asarray([1] * len(genuine) + [0] * len(impostor), dtype=np.uint8)
    scores = np.asarray(genuine + impostor, dtype=float)
    fpr, tpr, thresholds = roc_curve(y, scores)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    roc_auc = float(auc(fpr, tpr))

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.5f}")
    plt.xscale("log")
    plt.xlim(max(1e-5, np.min(fpr[fpr > 0]) if np.any(fpr > 0) else 1e-5), 1)
    plt.xlabel("FAR / FPR")
    plt.ylabel("TAR / TPR")
    plt.title("Biometric verification ROC")
    plt.grid(True, alpha=0.3)
    plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "verification_roc_log_far.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.hist(genuine, bins=40, alpha=0.6, label="genuine")
    plt.hist(impostor, bins=40, alpha=0.6, label="impostor")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Count")
    plt.title("Genuine and impostor score distributions")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(out_dir / "genuine_impostor_histogram.png", dpi=180)
    plt.close()

    return {
        "verification_roc_auc": roc_auc,
        "eer": eer,
        "eer_threshold": float(thresholds[idx]),
        "tar_at_far_1e-1": interpolate_tar_at_far(fpr, tpr, 1e-1),
        "tar_at_far_1e-2": interpolate_tar_at_far(fpr, tpr, 1e-2),
        "tar_at_far_1e-3": interpolate_tar_at_far(fpr, tpr, 1e-3),
        "genuine_scores": len(genuine),
        "impostor_scores": len(impostor),
    }


def run_identification(author_dir: Path, manifest: Path, weights: Path, evaluation_dir: Path, output_dir: Path, scope: str, mask_source: str, threshold: float, polarity: str, gallery_ratio: float, seed: int, outer_scale: float) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = read_manifest(manifest, None if scope == "all" else scope)
    if not records:
        raise RuntimeError(f"Brak rekordów dla scope={scope}")
    model = None
    if mask_source == "predicted":
        model = create_author_model(author_dir)
        model.load_weights(str(weights))
        _ = model(np.zeros((1, 240, 320, 1), np.float32), training=False).numpy()

    items = []
    failed = []
    for i, record in enumerate(records, 1):
        start = time.perf_counter()
        gray, image = read_image(record.image_path)
        if mask_source == "ground_truth":
            mask = read_mask(record.mask_path, polarity=polarity)
            segmentation_time = 0.0
        else:
            seg_start = time.perf_counter()
            prob = model(image[None, ...], training=False).numpy()[0, ..., 0]
            mask = (prob >= threshold).astype(np.uint8)
            segmentation_time = time.perf_counter() - seg_start
        feat_start = time.perf_counter()
        feature = annular_feature(gray, mask, outer_scale=outer_scale)
        feature_time = time.perf_counter() - feat_start
        if feature is None:
            failed.append({"record_id": record.record_id, "subject": record.subject, "relative_image": record.relative_image, "reason": "invalid pupil mask"})
            continue
        items.append({"record": record, "feature": feature, "segmentation_time_s": segmentation_time, "feature_time_s": feature_time, "total_prepare_time_s": time.perf_counter() - start})
        print(f"[BIO-{mask_source}] feature {i}/{len(records)}")

    grouped = {}
    for item in items:
        grouped.setdefault(item["record"].subject, []).append(item)
    gallery, probes = [], []
    rng = np.random.default_rng(seed)
    for subject, group in sorted(grouped.items()):
        order = np.arange(len(group)); rng.shuffle(order)
        n_gallery = max(1, int(round(len(group) * gallery_ratio)))
        if n_gallery >= len(group):
            n_gallery = len(group) - 1
        if n_gallery < 1:
            continue
        gallery.extend(group[j] for j in order[:n_gallery])
        probes.extend(group[j] for j in order[n_gallery:])
    if not gallery or not probes:
        raise RuntimeError("Za mało obrazów na osobę do podziału gallery/probe.")

    gallery_x = np.stack([x["feature"] for x in gallery])
    probe_x = np.stack([x["feature"] for x in probes])
    gallery_y = np.asarray([x["record"].subject for x in gallery])
    probe_y = np.asarray([x["record"].subject for x in probes])
    subjects = sorted(set(gallery_y) & set(probe_y))

    match_start = time.perf_counter()
    similarities = cosine_matrix(probe_x, gallery_x)
    match_total = time.perf_counter() - match_start
    subject_scores = np.full((len(probes), len(subjects)), -np.inf, dtype=float)
    for j, subject in enumerate(subjects):
        cols = np.where(gallery_y == subject)[0]
        subject_scores[:, j] = similarities[:, cols].max(axis=1)
    ranking = np.argsort(-subject_scores, axis=1)
    predictions = np.asarray([subjects[idx] for idx in ranking[:, 0]])

    rows = []
    ranks = []
    genuine, impostor = [], []
    for i, item in enumerate(probes):
        true = probe_y[i]
        true_idx = subjects.index(true)
        rank = int(np.where(ranking[i] == true_idx)[0][0] + 1)
        ranks.append(rank)
        genuine.append(float(subject_scores[i, true_idx]))
        impostor.extend(float(x) for j, x in enumerate(subject_scores[i]) if j != true_idx)
        rows.append({
            "record_id": item["record"].record_id,
            "subject_true": true,
            "subject_pred": predictions[i],
            "rank_true": rank,
            "top1_correct": int(predictions[i] == true),
            "best_similarity": float(subject_scores[i, ranking[i, 0]]),
            "genuine_similarity": float(subject_scores[i, true_idx]),
            "segmentation_time_s": item["segmentation_time_s"],
            "feature_time_s": item["feature_time_s"],
            "matching_time_s": match_total / len(probes),
            "end_to_end_query_time_s": item["segmentation_time_s"] + item["feature_time_s"] + match_total / len(probes),
            "relative_image": item["record"].relative_image,
        })
    _write_csv(output_dir / "probe_results.csv", rows)
    _write_csv(output_dir / "failed_features.csv", failed)

    cm = confusion_matrix(probe_y, predictions, labels=subjects)
    with (output_dir / "identity_confusion_raw.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f); writer.writerow(["true/pred", *subjects])
        for label, row_cm in zip(subjects, cm): writer.writerow([label, *row_cm.tolist()])
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(float), row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0)
    with (output_dir / "identity_confusion_normalized.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f); writer.writerow(["true/pred", *subjects])
        for label, row_cm in zip(subjects, cm_norm): writer.writerow([label, *[f"{v:.8f}" for v in row_cm]])
    save_confusion(cm, subjects, output_dir / "identity_confusion_raw.png", False, f"Identity confusion – {mask_source}")
    save_confusion(cm, subjects, output_dir / "identity_confusion_normalized.png", True, f"Identity confusion normalized – {mask_source}")

    max_rank = min(20, len(subjects))
    cmc = [{"rank": k, "accuracy": float(np.mean(np.asarray(ranks) <= k))} for k in range(1, max_rank + 1)]
    _write_csv(output_dir / "cmc.csv", cmc)
    plt.figure(figsize=(7, 5))
    plt.plot([r["rank"] for r in cmc], [r["accuracy"] for r in cmc], marker="o")
    plt.xlabel("Rank")
    plt.ylabel("Identification rate")
    plt.title(f"CMC – {mask_source}")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(output_dir / "cmc_curve.png", dpi=180)
    plt.close()

    verification = verification_metrics(genuine, impostor, output_dir)
    summary = {
        "important_note": "This is an added biometric baseline, not a recognition module from the original U-Net author. It estimates an iris annulus from the pupil mask and uses normalized texture features with cosine nearest-neighbour matching.",
        "mask_source": mask_source,
        "scope": scope,
        "subjects": len(subjects),
        "gallery_images": len(gallery),
        "probe_images": len(probes),
        "failed_features": len(failed),
        "top1_accuracy": float(np.mean(np.asarray(ranks) <= 1)),
        "top5_accuracy": float(np.mean(np.asarray(ranks) <= min(5, len(subjects)))),
        "top10_accuracy": float(np.mean(np.asarray(ranks) <= min(10, len(subjects)))),
        "mean_reciprocal_rank": float(np.mean(1.0 / np.asarray(ranks))),
        "mean_rank": float(np.mean(ranks)),
        "gallery_ratio": gallery_ratio,
        "matching_total_s": match_total,
        "matching_mean_per_probe_s": match_total / len(probes),
        "mean_segmentation_time_s": float(np.mean([r["segmentation_time_s"] for r in rows])),
        "mean_feature_time_s": float(np.mean([r["feature_time_s"] for r in rows])),
        "mean_end_to_end_query_time_s": float(np.mean([r["end_to_end_query_time_s"] for r in rows])),
        **verification,
    }
    (output_dir / "identification_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
