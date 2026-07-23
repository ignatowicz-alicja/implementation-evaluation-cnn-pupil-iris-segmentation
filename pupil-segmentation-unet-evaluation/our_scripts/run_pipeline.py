from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPTS_ROOT.parent
AUTHOR_DIR = REPOSITORY_ROOT / "original_author_code"
sys.path.insert(0, str(SCRIPTS_ROOT))

from wrapper.dataset import prepare_manifest


def add_common_data(parser):
    parser.add_argument("--images", required=True, help="Katalog obrazów; przeszukiwany rekurencyjnie.")
    parser.add_argument("--masks", required=True, help="Katalog masek lub katalog zawierający masks_pupil.")
    parser.add_argument("--output", required=True, help="Katalog wyników eksperymentu.")
    parser.add_argument("--subject-mode", choices=["parent", "first_folder", "stem_prefix"], default="parent")
    parser.add_argument("--subject-regex", default=None)
    parser.add_argument("--split-mode", choices=["subject_disjoint", "per_subject"], default="subject_disjoint")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--allow-missing-masks", action="store_true")


def prepare(args):
    output = Path(args.output)
    manifest = output / "dataset_manifest.csv"
    prepare_manifest(
        Path(args.images), Path(args.masks), manifest,
        subject_mode=args.subject_mode, subject_regex=args.subject_regex,
        split_mode=args.split_mode, train_ratio=args.train_ratio,
        val_ratio=args.val_ratio, seed=args.seed, strict=not args.allow_missing_masks,
    )
    print(f"[OK] Manifest: {manifest}")
    return manifest


def command_train(args):
    from wrapper.train_eval import train_model
    weights = train_model(AUTHOR_DIR, Path(args.manifest), Path(args.output), args.epochs, args.batch_size, args.learning_rate, args.patience, args.mask_polarity, args.seed)
    print(f"[OK] Wagi: {weights}")


def command_evaluate(args):
    from wrapper.train_eval import evaluate_model
    summary = evaluate_model(AUTHOR_DIR, Path(args.manifest), Path(args.weights), Path(args.output), args.split, args.threshold, args.mask_polarity, args.save_all, args.save_examples, args.pixels_per_image, args.seed)
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True))


def command_identify(args):
    from wrapper.biometrics import run_identification
    sources = ["ground_truth", "predicted"] if args.mask_source == "both" else [args.mask_source]
    for source in sources:
        out = Path(args.output) / source
        result = run_identification(AUTHOR_DIR, Path(args.manifest), Path(args.weights), Path(args.evaluation_dir), out, args.scope, source, args.threshold, args.mask_polarity, args.gallery_ratio, args.seed, args.outer_scale)
        print(json.dumps(result, indent=2, ensure_ascii=False))


def command_full(args):
    from wrapper.train_eval import train_model, evaluate_model
    from wrapper.biometrics import run_identification

    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "dataset_manifest.csv"
    prepare_manifest(
        Path(args.images), Path(args.masks), manifest,
        subject_mode=args.subject_mode, subject_regex=args.subject_regex,
        split_mode=args.split_mode, train_ratio=args.train_ratio,
        val_ratio=args.val_ratio, seed=args.seed, strict=not args.allow_missing_masks,
    )
    weights = train_model(AUTHOR_DIR, manifest, root / "01_training", args.epochs, args.batch_size, args.learning_rate, args.patience, args.mask_polarity, args.seed)
    evaluate_model(AUTHOR_DIR, manifest, weights, root / "02_segmentation_evaluation", args.eval_split, args.threshold, args.mask_polarity, args.save_all, args.save_examples, args.pixels_per_image, args.seed)
    sources = ["ground_truth", "predicted"] if args.mask_source == "both" else [args.mask_source]
    for source in sources:
        run_identification(AUTHOR_DIR, manifest, weights, root / "02_segmentation_evaluation", root / "03_biometric_identification" / source, args.identification_scope, source, args.threshold, args.mask_polarity, args.gallery_ratio, args.seed, args.outer_scale)
    print(f"[DONE] Wszystkie wyniki: {root.resolve()}")


def build_parser():
    parser = argparse.ArgumentParser(description="Pipeline around the unchanged third-party U-Net code: dataset preparation, pupil training, segmentation evaluation, timing, and optional biometric identification.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare")
    add_common_data(p); p.set_defaults(func=prepare)

    p = sub.add_parser("train")
    p.add_argument("--manifest", required=True); p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=20); p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=1e-4); p.add_argument("--patience", type=int, default=8)
    p.add_argument("--mask-polarity", choices=["auto", "white", "black"], default="auto"); p.add_argument("--seed", type=int, default=2026)
    p.set_defaults(func=command_train)

    p = sub.add_parser("evaluate")
    p.add_argument("--manifest", required=True); p.add_argument("--weights", required=True); p.add_argument("--output", required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="test"); p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--mask-polarity", choices=["auto", "white", "black"], default="auto")
    p.add_argument("--save-all", action="store_true"); p.add_argument("--save-examples", type=int, default=20)
    p.add_argument("--pixels-per-image", type=int, default=3000); p.add_argument("--seed", type=int, default=2026)
    p.set_defaults(func=command_evaluate)

    p = sub.add_parser("identify")
    p.add_argument("--manifest", required=True); p.add_argument("--weights", required=True); p.add_argument("--evaluation-dir", required=True); p.add_argument("--output", required=True)
    p.add_argument("--scope", choices=["all", "train", "val", "test"], default="test")
    p.add_argument("--mask-source", choices=["ground_truth", "predicted", "both"], default="both")
    p.add_argument("--threshold", type=float, default=0.5); p.add_argument("--mask-polarity", choices=["auto", "white", "black"], default="auto")
    p.add_argument("--gallery-ratio", type=float, default=0.70); p.add_argument("--outer-scale", type=float, default=3.0); p.add_argument("--seed", type=int, default=2026)
    p.set_defaults(func=command_identify)

    p = sub.add_parser("full")
    add_common_data(p)
    p.add_argument("--epochs", type=int, default=20); p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=1e-4); p.add_argument("--patience", type=int, default=8)
    p.add_argument("--mask-polarity", choices=["auto", "white", "black"], default="auto")
    p.add_argument("--eval-split", choices=["train", "val", "test"], default="test")
    p.add_argument("--threshold", type=float, default=0.5); p.add_argument("--save-all", action="store_true")
    p.add_argument("--save-examples", type=int, default=20); p.add_argument("--pixels-per-image", type=int, default=3000)
    p.add_argument("--identification-scope", choices=["all", "train", "val", "test"], default="test")
    p.add_argument("--mask-source", choices=["ground_truth", "predicted", "both"], default="both")
    p.add_argument("--gallery-ratio", type=float, default=0.70); p.add_argument("--outer-scale", type=float, default=3.0)
    p.set_defaults(func=command_full)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
