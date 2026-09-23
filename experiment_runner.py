#!/usr/bin/env python3
"""Wspólna implementacja osobnych launcherów eksperymentów."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from experiment_profiles import profile_for


ROOT = Path(__file__).resolve().parent
UNET_PIPELINE = ROOT / "pupil-segmentation-unet-evaluation" / "our_scripts" / "run_pipeline.py"
CLASSICAL_PIPELINE = ROOT / "Classic_Masek_Kovesi_Evaluation" / "src" / "python" / "run_casia_full_eval_python.py"


def gpu_is_available() -> tuple[bool, str]:
    """Sprawdza GPU przez TensorFlow, czyli bibliotekę używaną przez U-Net."""
    try:
        import tensorflow as tf
        devices = tf.config.list_physical_devices("GPU")
    except Exception as exc:
        return False, f"TensorFlow nie wykrył GPU ({exc.__class__.__name__}: {exc})"
    if not devices:
        return False, "TensorFlow nie wykrył urządzenia GPU/CUDA"
    return True, ", ".join(str(device) for device in devices)


def actual_device(method: str, requested: str) -> tuple[str, str]:
    if requested == "cpu":
        return "cpu", "Wybrano CPU."
    if method == "classical":
        return "cpu", "Iris-Recognition-master nie ma backendu GPU; uruchamiam na CPU."
    available, detail = gpu_is_available()
    if available:
        return "gpu", f"Uruchamiam U-Net na GPU: {detail}"
    return "cpu", f"Nie ma dostępnego GPU; uruchamiam U-Net na CPU. Szczegóły: {detail}"


def build_parser(dataset: str, method: str, requested_device: str) -> argparse.ArgumentParser:
    profile = profile_for(dataset)
    parser = argparse.ArgumentParser(description=f"{profile.label}: {method}, żądane urządzenie: {requested_device}.")
    parser.add_argument("--images", required=True, type=Path, help="Katalog obrazów wejściowych.")
    parser.add_argument("--masks", required=True, type=Path, help="Katalog masek ground truth.")
    parser.add_argument("--output", type=Path, default=None, help="Katalog wyników; domyślnie results/<baza>/<metoda>/<urządzenie>.")
    parser.add_argument("--max-images", type=int, default=0, help="Ograniczenie liczby obrazów; 0 oznacza całą bazę.")
    parser.add_argument("--dry-run", action="store_true", help="Pokaż polecenie bez uruchamiania eksperymentu.")
    parser.add_argument("pipeline_args", nargs=argparse.REMAINDER, help="Argumenty po -- przekazywane do właściwego pipeline.")
    return parser


def run_profile(dataset: str, method: str, requested_device: str, argv: list[str] | None = None) -> int:
    profile = profile_for(dataset)
    args = build_parser(dataset, method, requested_device).parse_args(argv)
    if not args.images.is_dir():
        raise SystemExit(f"[BŁĄD] Nie ma katalogu obrazów: {args.images}")
    if not args.masks.is_dir():
        raise SystemExit(f"[BŁĄD] Nie ma katalogu masek: {args.masks}")

    device, message = actual_device(method, requested_device)
    output = (args.output or ROOT / "results" / dataset / method / device).resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset": dataset, "dataset_label": profile.label, "method": method,
        "requested_device": requested_device, "actual_device": device,
        "device_message": message, "images": str(args.images.resolve()),
        "masks": str(args.masks.resolve()), "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "run_configuration.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if method == "unet":
        command = [sys.executable, str(UNET_PIPELINE), "full", "--images", str(args.images), "--masks", str(args.masks), "--output", str(output), "--subject-mode", profile.unet_subject_mode, "--split-mode", profile.unet_split_mode]
    else:
        command = [sys.executable, str(CLASSICAL_PIPELINE), "--images-dir", str(args.images), "--gt-dir", str(args.masks), "--output-dir", str(output), "--dataset-label", profile.label, "--identity-mode", profile.classical_identity_mode]
        if args.max_images:
            command.extend(["--max-images", str(args.max_images)])
    command.extend(args.pipeline_args[1:] if args.pipeline_args[:1] == ["--"] else args.pipeline_args)

    environment = os.environ.copy()
    if method == "unet" and device == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = "-1"
    print(f"[PROFIL] {profile.label} | {method} | żądano: {requested_device} | użyto: {device}")
    print(f"[SPRZĘT] {message}")
    print("[POLECENIE] " + " ".join(command))
    if args.dry_run:
        return 0
    return subprocess.run(command, cwd=ROOT, env=environment, check=False).returncode
