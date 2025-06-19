"""Face cropping step using ``animeface``, ``mediapipe`` or a YOLOv8 model."""

from pathlib import Path

import shutil
from PIL import Image
import animeface
from ultralytics import YOLO
import torch

try:  # Optional dependency for better face detection
    import mediapipe as mp  # type: ignore
except Exception:  # pragma: no cover - library may be missing
    mp = None  # type: ignore

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


def _crop_mediapipe(img: Image.Image, detector: "mp.solutions.face_detection.FaceDetection", margin: float) -> list[Image.Image]:
    """Crop faces using ``mediapipe`` if available."""

    import numpy as np

    results = detector.process(np.array(img))
    crops: list[Image.Image] = []
    if results.detections:
        for det in results.detections:
            box = det.location_data.relative_bounding_box
            x = int(box.xmin * img.width)
            y = int(box.ymin * img.height)
            w = int(box.width * img.width)
            h = int(box.height * img.height)
            crops.append(_crop_box(img, x, y, w, h, margin))
    return crops


def _crop_yolo(imgs: list[Image.Image], model: YOLO, margin: float, conf: float) -> list[list[Image.Image]]:
    """Return crops for a batch of images using YOLOv8."""

    results = model(imgs)
    batch_crops: list[list[Image.Image]] = []
    for img, res in zip(imgs, results):
        img_crops = []
        for box in res.boxes:
            c = float(box.conf[0])
            if c < conf:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            img_crops.append(_crop_box(img, x1, y1, x2 - x1, y2 - y1, margin))
        batch_crops.append(img_crops)
    return batch_crops


def run(
    upscaled_dir: Path,
    workdir: Path,
    *,
    margin: float = 0.3,
    yolo_model: Path | None = None,
    conf_threshold: float = 0.5,
    batch_size: int = 4,
    use_mediapipe: bool | None = None,
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

    detector = None
    if yolo_model is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = YOLO(str(yolo_model)).to(device)
        method = "yolo"
        log_step("Cropping started with YOLOv8")
    elif (use_mediapipe is None and mp is not None) or (use_mediapipe is True and mp is not None):
        detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=conf_threshold)
        model = None
        method = "mediapipe"
        log_step("Cropping started with mediapipe")
    else:
        model = None
        method = "animeface"
        log_step("Cropping started with animeface")

    img_paths = sorted(upscaled_dir.glob("*.png"))

    if method == "yolo":
        for i in range(0, len(img_paths), batch_size):
            batch_paths = img_paths[i : i + batch_size]
            imgs = [Image.open(p).convert("RGB") for p in batch_paths]
            batch_crops = _crop_yolo(imgs, model, margin, conf_threshold)
            for p, img, crops in zip(batch_paths, imgs, batch_crops):
                if not crops:
                    shutil.copy(p, workdir / p.name)
                else:
                    for idx, cropped in enumerate(crops):
                        out_name = f"{p.stem}_{idx:02d}.png" if len(crops) > 1 else p.name
                        cropped.save(workdir / out_name)
                img.close()
    else:
        for p in img_paths:
            with Image.open(p).convert("RGB") as img:
                if method == "mediapipe" and detector is not None:
                    crops = _crop_mediapipe(img, detector, margin)
                else:
                    crops = _crop_animeface(img, margin)

                if not crops:
                    shutil.copy(p, workdir / p.name)
                    continue
                for idx, cropped in enumerate(crops):
                    out_name = f"{p.stem}_{idx:02d}.png" if len(crops) > 1 else p.name
                    cropped.save(workdir / out_name)

    if detector is not None:
        detector.close()

    log_step("Cropping completed")
    return workdir
