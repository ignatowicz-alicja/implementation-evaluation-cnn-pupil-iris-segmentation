"""Adapter wspólny dla osobnych punktów wejścia profili."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_runner import run_profile


def main(dataset: str, method: str, device: str) -> None:
    raise SystemExit(run_profile(dataset, method, device))
