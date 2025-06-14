"""Automatic upscaling and quality checking."""

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import torch
import cv2

from ..logging_utils import log_step


try:  # Optional dependency
    from realesrgan import RealESRGAN
except Exception:  # pragma: no cover - library may not be installed
    RealESRGAN = None  # type: ignore[misc]


def _load_model(device: torch.device, scale: int) -> Optional[object]:
    """Load RealESRGAN anime model if available."""

    if RealESRGAN is None:
        log_step("RealESRGAN not available – using PIL resize")
        return None
    try:
        model = RealESRGAN(device, scale=scale)
        model.load_weights(f"RealESRGAN_x{scale}plus_anime_6B.pth")
        model.eval()
        return model
    except Exception as exc:  # pragma: no cover - runtime download may fail
        log_step(f"RealESRGAN load failed: {exc}; falling back to PIL resize")
        return None


def _is_acceptable(img: Image.Image, blur_thresh: float, dark_thresh: float) -> bool:
    """Return ``True`` if image passes basic quality checks."""

    gray = np.array(img.convert("L"))
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    if variance < blur_thresh:
        return False
    brightness = gray.mean()
    if brightness < dark_thresh:
        return False
    return True


def run(
    filtered_dir: Path,
    workdir: Path,
    *,
    scale: int = 2,
    blur_threshold: float = 100.0,
    dark_threshold: float = 40.0,
) -> Path:
    """Upscale images with RealESRGAN and drop low-quality frames."""

    workdir.mkdir(parents=True, exist_ok=True)
    log_step("Upscaling started")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(device, scale)

    for img_path in sorted(filtered_dir.glob("*.png")):
        with Image.open(img_path).convert("RGB") as img:
            if not _is_acceptable(img, blur_threshold, dark_threshold):
                continue

            if model is not None:
                with torch.no_grad():  # pragma: no cover - heavy model inference
                    upscaled = model.predict(np.array(img))
                up_img = Image.fromarray(upscaled)
            else:
                width, height = img.size
                up_img = img.resize((width * scale, height * scale), Image.LANCZOS)

            out_path = workdir / img_path.name
            up_img.save(out_path)

    log_step("Upscaling completed")
    return workdir
