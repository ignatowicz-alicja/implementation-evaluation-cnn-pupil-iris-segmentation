from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _install_legacy_keras_shims() -> None:
    """Provide import aliases expected by the unchanged 2021 author code."""
    import keras.layers as layers

    module_name = "keras.layers.normalization"
    if module_name not in sys.modules:
        normalization = types.ModuleType(module_name)
        normalization.BatchNormalization = layers.BatchNormalization
        sys.modules[module_name] = normalization


def load_author_module(author_dir: Path):
    author_dir = Path(author_dir).resolve()
    source = author_dir / "UNet_iris_segmentation.py"
    if not source.is_file():
        raise FileNotFoundError(f"Nie znaleziono oryginalnego pliku autora: {source}")

    _install_legacy_keras_shims()
    if str(author_dir) not in sys.path:
        sys.path.insert(0, str(author_dir))

    module_name = "unchanged_author_unet"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Nie można załadować modułu autora: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def create_author_model(author_dir: Path):
    """Build the network by calling create_model() from the unchanged author file."""
    module = load_author_module(author_dir)
    return module.create_model()
