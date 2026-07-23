from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
MASK_SUFFIXES = ("_pupil", "_mask", "-pupil", "-mask", " pupil", " mask")


@dataclass
class Record:
    record_id: str
    subject: str
    image_path: str
    mask_path: str
    relative_image: str
    split: str = ""


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def normalize_stem(stem: str) -> str:
    value = stem.lower().strip()
    for suffix in MASK_SUFFIXES:
        if value.endswith(suffix):
            value = value[: -len(suffix)]
    return re.sub(r"[^a-z0-9]+", "", value)


def resolve_mask_root(mask_root: Path) -> Path:
    root = Path(mask_root)
    candidates = [
        root / "masks_pupil",
        root / "mask_pupil",
        root / "pupil_masks",
        root / "masks",
        root,
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(p.suffix.lower() in IMAGE_EXTENSIONS for p in candidate.rglob("*")):
            return candidate
    raise FileNotFoundError(f"Nie znaleziono masek w: {root}")


def list_images(root: Path, extensions: Iterable[str] | None = None) -> list[Path]:
    allowed = {e.lower() if str(e).startswith(".") else f".{str(e).lower()}" for e in (extensions or IMAGE_EXTENSIONS)}
    return sorted(p for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower() in allowed)


def infer_subject(image: Path, images_root: Path, subject_mode: str, subject_regex: str | None) -> str:
    rel = image.relative_to(images_root)
    if subject_regex:
        match = re.search(subject_regex, rel.as_posix())
        if not match:
            raise ValueError(f"Regex podmiotu nie pasuje do: {rel}")
        return match.group(1) if match.groups() else match.group(0)
    if subject_mode == "parent":
        return rel.parent.name or "root"
    if subject_mode == "first_folder":
        return rel.parts[0] if len(rel.parts) > 1 else "root"
    if subject_mode == "stem_prefix":
        return re.split(r"[_\-\s]", image.stem)[0]
    raise ValueError(f"Nieznany subject_mode: {subject_mode}")


def _path_subject_tokens(path: Path, mask_root: Path) -> set[str]:
    """Zwraca bezpieczne kandydaty identyfikatora osoby z folderów i nazwy pliku."""
    tokens: set[str] = set()
    try:
        rel = path.relative_to(mask_root)
        parent_parts = rel.parts[:-1]
    except ValueError:
        parent_parts = path.parts[:-1]
    for part in parent_parts:
        norm = normalize_stem(part)
        if norm:
            tokens.add(norm)
    # Obsługa płaskich nazw, np. 087_02_L.png albo 087-02_L-pupil.png.
    stem_parts = [normalize_stem(x) for x in re.split(r"[_\-\s]+", path.stem)]
    tokens.update(x for x in stem_parts if x)
    return tokens


def build_mask_indices(mask_root: Path) -> tuple[
    dict[str, list[Path]],
    dict[str, list[Path]],
    dict[str, list[Path]],
]:
    by_stem: dict[str, list[Path]] = {}
    by_subject_stem: dict[str, list[Path]] = {}
    by_compound: dict[str, list[Path]] = {}
    for path in list_images(mask_root):
        key = normalize_stem(path.stem)
        by_stem.setdefault(key, []).append(path)
        for subject_token in _path_subject_tokens(path, mask_root):
            by_subject_stem.setdefault(f"{subject_token}::{key}", []).append(path)
        # Indeks do nazw płaskich zawierających jednocześnie ID i nazwę obrazu.
        full_key = normalize_stem(path.stem)
        by_compound.setdefault(full_key, []).append(path)
    return by_stem, by_subject_stem, by_compound


def _unique(paths: Iterable[Path]) -> Path | None:
    unique = sorted({Path(p) for p in paths})
    return unique[0] if len(unique) == 1 else None


def find_mask(
    image: Path,
    images_root: Path,
    mask_root: Path,
    subject: str,
    by_stem: dict[str, list[Path]],
    by_subject_stem: dict[str, list[Path]],
    by_compound: dict[str, list[Path]],
) -> Path | None:
    rel = image.relative_to(images_root)

    # 1. Najbezpieczniejsze dopasowanie: taka sama struktura katalogów.
    suffix_variants = ("", "_pupil", "_mask", "-pupil", "-mask")
    for suffix in suffix_variants:
        for ext in sorted(IMAGE_EXTENSIONS):
            candidate = mask_root / rel.parent / f"{image.stem}{suffix}{ext}"
            if candidate.is_file():
                return candidate

    key = normalize_stem(image.stem)
    subject_key = normalize_stem(subject)

    # 2. ID osoby pochodzi z dowolnego folderu nadrzędnego maski.
    match = _unique(by_subject_stem.get(f"{subject_key}::{key}", []))
    if match is not None:
        return match

    # 3. Płaskie nazwy typu 087_02_L.png / 087_02_L_pupil.png.
    compound_candidates: list[Path] = []
    wanted = normalize_stem(f"{subject}_{image.stem}")
    for compound_key, paths in by_compound.items():
        if compound_key == wanted or compound_key.endswith(wanted) or wanted.endswith(compound_key):
            compound_candidates.extend(paths)
    match = _unique(compound_candidates)
    if match is not None:
        return match

    # 4. Sama nazwa obrazu jest używana tylko wtedy, gdy jest globalnie jednoznaczna.
    return _unique(by_stem.get(key, []))

def assign_subject_disjoint(records: list[Record], train_ratio: float, val_ratio: float, seed: int) -> None:
    subjects = sorted({r.subject for r in records})
    rnd = random.Random(seed)
    rnd.shuffle(subjects)
    n = len(subjects)
    if n < 3:
        assign_per_subject(records, train_ratio, val_ratio, seed)
        return
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    train_subjects = set(subjects[:n_train])
    val_subjects = set(subjects[n_train : n_train + n_val])
    for record in records:
        record.split = "train" if record.subject in train_subjects else "val" if record.subject in val_subjects else "test"


def assign_per_subject(records: list[Record], train_ratio: float, val_ratio: float, seed: int) -> None:
    grouped: dict[str, list[Record]] = {}
    for record in records:
        grouped.setdefault(record.subject, []).append(record)
    for subject, group in grouped.items():
        rnd = random.Random(f"{seed}:{subject}")
        rnd.shuffle(group)
        n = len(group)
        n_train = max(1, int(round(n * train_ratio))) if n else 0
        n_val = max(1, int(round(n * val_ratio))) if n >= 3 else 0
        if n_train + n_val >= n and n >= 2:
            n_train = max(1, n - 1 - n_val)
        for i, record in enumerate(group):
            record.split = "train" if i < n_train else "val" if i < n_train + n_val else "test"
        if n >= 2 and not any(r.split == "test" for r in group):
            group[-1].split = "test"


def prepare_manifest(
    images_root: Path,
    masks_root: Path,
    output_csv: Path,
    subject_mode: str = "parent",
    subject_regex: str | None = None,
    split_mode: str = "subject_disjoint",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 2026,
    strict: bool = True,
) -> list[Record]:
    images_root = Path(images_root).resolve()
    masks_root = resolve_mask_root(Path(masks_root).resolve())
    images = list_images(images_root)
    if not images:
        raise FileNotFoundError(f"Brak obrazów w: {images_root}")
    by_stem, by_subject_stem, by_compound = build_mask_indices(masks_root)

    records: list[Record] = []
    missing: list[str] = []
    for image in images:
        subject = infer_subject(image, images_root, subject_mode, subject_regex)
        mask = find_mask(image, images_root, masks_root, subject, by_stem, by_subject_stem, by_compound)
        if mask is None:
            missing.append(str(image))
            continue
        rel = image.relative_to(images_root).as_posix()
        records.append(
            Record(
                record_id=sha1_text(rel),
                subject=subject,
                image_path=str(image),
                mask_path=str(mask.resolve()),
                relative_image=rel,
            )
        )

    if strict and missing:
        preview = "\n".join(missing[:20])
        raise RuntimeError(f"Nie znaleziono masek dla {len(missing)} obrazów. Pierwsze:\n{preview}")
    if not records:
        raise RuntimeError("Nie utworzono żadnych par obraz–maska.")

    if split_mode == "subject_disjoint":
        assign_subject_disjoint(records, train_ratio, val_ratio, seed)
    elif split_mode == "per_subject":
        assign_per_subject(records, train_ratio, val_ratio, seed)
    else:
        raise ValueError("split_mode musi być subject_disjoint albo per_subject")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(r) for r in records)

    report = {
        "images_root": str(images_root),
        "requested_masks_root": str(Path(masks_root)),
        "resolved_masks_root": str(masks_root),
        "paired": len(records),
        "missing_masks": len(missing),
        "subjects": len({r.subject for r in records}),
        "split_counts": {s: sum(r.split == s for r in records) for s in ("train", "val", "test")},
        "split_subjects": {s: len({r.subject for r in records if r.split == s}) for s in ("train", "val", "test")},
        "split_mode": split_mode,
        "seed": seed,
    }
    output_csv.with_suffix(".summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if missing:
        output_csv.with_suffix(".missing_masks.txt").write_text("\n".join(missing), encoding="utf-8")
    return records


def read_manifest(path: Path, split: str | None = None) -> list[Record]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        rows = [Record(**row) for row in csv.DictReader(f)]
    return [r for r in rows if split is None or r.split == split]


def read_image(path: str, target_hw: tuple[int, int] = (240, 320)) -> tuple[np.ndarray, np.ndarray]:
    raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise RuntimeError(f"Nie można odczytać obrazu: {path}")
    h, w = target_hw
    resized = cv2.resize(raw, (w, h), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return resized, normalized[..., None]


def read_mask(path: str, target_hw: tuple[int, int] = (240, 320), polarity: str = "auto") -> np.ndarray:
    raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise RuntimeError(f"Nie można odczytać maski: {path}")
    h, w = target_hw
    mask = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST) > 127
    if polarity == "black":
        mask = ~mask
    elif polarity == "auto" and mask.mean() > 0.5:
        mask = ~mask
    elif polarity not in {"auto", "white", "black"}:
        raise ValueError("polarity: auto, white albo black")
    return mask.astype(np.uint8)
