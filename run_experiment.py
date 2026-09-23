#!/usr/bin/env python3
"""Główny, interaktywny i nieinteraktywny wybór eksperymentu."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
CHOICES = {"dataset": ("casia", "iitd", "cataract1k"), "method": ("classical", "unet"), "device": ("cpu", "gpu")}


def ask(name: str) -> str:
    options = "/".join(CHOICES[name])
    while True:
        value = input(f"Wybierz {name} ({options}): ").strip().lower()
        if value in CHOICES[name]:
            return value
        print(f"Nieprawidłowa wartość. Dostępne: {options}.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Wybór bazy, metody i urządzenia dla eksperymentu segmentacji.")
    parser.add_argument("--dataset", choices=CHOICES["dataset"])
    parser.add_argument("--method", choices=CHOICES["method"])
    parser.add_argument("--device", choices=CHOICES["device"])
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Argumenty przekazywane do skryptu, np. -- --images ... --masks ...")
    args = parser.parse_args()
    dataset = args.dataset or ask("dataset")
    method = args.method or ask("method")
    device = args.device or ask("device")
    target = SCRIPTS / f"run_{dataset}_{method}_{device}.py"
    if not target.is_file():
        raise SystemExit(f"[BŁĄD] Brak skryptu wariantu: {target}")
    forwarded = args.args[1:] if args.args[:1] == ["--"] else args.args
    return subprocess.run([sys.executable, str(target), *forwarded], cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
