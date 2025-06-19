from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO

from .logging_utils import log_step
from .steps.annotation import _load_tagger
from .steps.upscaling import _load_model

MODELS_DIR = Path("models")

_executor = ThreadPoolExecutor(max_workers=3)
_futures: dict[str, Future[Any]] = {}


def detect_yolo_model() -> Path | None:
    """Return a YOLOv8 weight file from ``models/`` if present."""
    MODELS_DIR.mkdir(exist_ok=True)
    patterns = ["*.pt", "*.pth"]
    for pattern in patterns:
        models = sorted(MODELS_DIR.glob(pattern))
        if models:
            log_step(f"Found YOLO model: {models[0]}")
            return models[0]
    log_step("No YOLO model found")
    return None


def preload_yolo(model_path: Path | None) -> Future[Any]:
    """Start loading a YOLO model in the background."""
    if model_path is None:
        return _executor.submit(lambda: None)

    def _load() -> YOLO:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log_step("Loading YOLO model")
        return YOLO(str(model_path)).to(device)

    fut = _executor.submit(_load)
    _futures["yolo"] = fut
    return fut


def preload_tagger(device: torch.device) -> Future[Any]:
    """Start loading the WD14 ONNX tagger in the background."""

    fut = _executor.submit(_load_tagger, device)
    _futures["tagger"] = fut
    return fut


def preload_realesrgan(device: torch.device, scale: int) -> Future[Any]:
    """Start loading RealESRGAN weights in the background."""

    fut = _executor.submit(_load_model, device, scale)
    _futures["realesrgan"] = fut
    return fut


def get(name: str) -> Any:
    """Return a loaded model by name, waiting for completion if necessary."""

    fut = _futures.get(name)
    if fut is not None:
        return fut.result()
    return None


def clear() -> None:
    """Clear any stored futures."""

    _futures.clear()
