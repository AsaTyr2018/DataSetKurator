"""Face cropping step using the ``animeface`` detector."""

from pathlib import Path

import shutil
from PIL import Image
import animeface

from ..logging_utils import log_step


def _crop(img: Image.Image, face: animeface.Face, margin: float) -> Image.Image:
    """Return a cropped face region with optional margin."""

    box = face.face.pos
    x, y, w, h = box.x, box.y, box.width, box.height
    m_w = int(w * margin / 2)
    m_h = int(h * margin / 2)
    left = max(0, x - m_w)
    top = max(0, y - m_h)
    right = min(img.width, x + w + m_w)
    bottom = min(img.height, y + h + m_h)
    return img.crop((left, top, right, bottom))


def run(upscaled_dir: Path, workdir: Path, margin: float = 0.3) -> Path:
    """Crop faces from images using ``animeface`` detection.

    Parameters
    ----------
    upscaled_dir:
        Directory with upscaled images.
    workdir:
        Output directory for cropped results.
    margin:
        Extra border size around the detected face, expressed as a fraction
        of the bounding box dimensions.
    """

    workdir.mkdir(parents=True, exist_ok=True)
    log_step("Cropping started")

    for img_path in sorted(upscaled_dir.glob("*.png")):
        with Image.open(img_path).convert("RGB") as img:
            faces = animeface.detect(img)
            if not faces:
                # No detection -- keep the whole image to avoid losing data
                shutil.copy(img_path, workdir / img_path.name)
                continue

            for idx, face in enumerate(faces):
                cropped = _crop(img, face, margin)
                out_name = (
                    f"{img_path.stem}_{idx:02d}.png" if len(faces) > 1 else img_path.name
                )
                cropped.save(workdir / out_name)

    log_step("Cropping completed")
    return workdir
