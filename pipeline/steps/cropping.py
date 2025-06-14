"""Face cropping step using ``animeface`` or a YOLOv8 model."""

from pathlib import Path

import shutil
from PIL import Image
import animeface
from ultralytics import YOLO
import torch

from ..logging_utils import log_step


def _crop_box(img: Image.Image, x: int, y: int, w: int, h: int, margin: float) -> Image.Image:
    """Return a cropped region defined by ``x``, ``y``, ``w`` and ``h`` with optional margin."""

    m_w = int(w * margin / 2)
    m_h = int(h * margin / 2)
    left = max(0, x - m_w)
    top = max(0, y - m_h)
    right = min(img.width, x + w + m_w)
    bottom = min(img.height, y + h + m_h)
    return img.crop((left, top, right, bottom))


def _crop_animeface(img: Image.Image, margin: float) -> list[Image.Image]:
    faces = animeface.detect(img)
    crops = []
    for face in faces:
        box = face.face.pos
        crops.append(_crop_box(img, box.x, box.y, box.width, box.height, margin))
    return crops


def _crop_yolo(img: Image.Image, model: YOLO, margin: float, conf: float) -> list[Image.Image]:
    results = model(img)
    crops = []
    for box in results[0].boxes:
        c = float(box.conf[0])
        if c < conf:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        crops.append(_crop_box(img, x1, y1, x2 - x1, y2 - y1, margin))
    return crops


def run(
    upscaled_dir: Path,
    workdir: Path,
    *,
    margin: float = 0.3,
    yolo_model: Path | None = None,
    conf_threshold: float = 0.5,
) -> Path:
    """Crop faces from images.

    Parameters
    ----------
    upscaled_dir:
        Directory with upscaled images.
    workdir:
        Output directory for cropped results.
    margin:
        Extra border size around the detected face, expressed as a fraction
        of the bounding box dimensions.
    yolo_model:
        Optional path to a YOLOv8 model. If provided, YOLO detection is used.
    conf_threshold:
        Minimum confidence for YOLO detections.
    """

    workdir.mkdir(parents=True, exist_ok=True)

    if yolo_model is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = YOLO(str(yolo_model)).to(device)
        log_step("Cropping started with YOLOv8")
    else:
        model = None
        log_step("Cropping started with animeface")

    for img_path in sorted(upscaled_dir.glob("*.png")):
        with Image.open(img_path).convert("RGB") as img:
            if model is not None:
                crops = _crop_yolo(img, model, margin, conf_threshold)
            else:
                crops = _crop_animeface(img, margin)

            if not crops:
                # No detection -- keep the whole image to avoid losing data
                shutil.copy(img_path, workdir / img_path.name)
                continue

            for idx, cropped in enumerate(crops):
                out_name = (
                    f"{img_path.stem}_{idx:02d}.png" if len(crops) > 1 else img_path.name
                )
                cropped.save(workdir / out_name)

    log_step("Cropping completed")
    return workdir
