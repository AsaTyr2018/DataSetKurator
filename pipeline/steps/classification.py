"""Character classification based on hair and eye color detection."""

from pathlib import Path
from typing import List
import shutil

import torch

from ..logging_utils import log_step
from .annotation import _load_tagger, _tag_image

HAIR_COLORS = [
    "blonde hair",
    "black hair",
    "brown hair",
    "red hair",
    "blue hair",
    "green hair",
    "purple hair",
    "pink hair",
    "orange hair",
    "silver hair",
    "white hair",
    "gray hair",
    "aqua hair",
]

EYE_COLORS = [
    "blue eyes",
    "brown eyes",
    "red eyes",
    "green eyes",
    "purple eyes",
    "yellow eyes",
    "pink eyes",
    "aqua eyes",
    "orange eyes",
    "gray eyes",
]


def _detect_color(tag_str: str, colors: List[str], suffix: str) -> str:
    for c in colors:
        if c in tag_str:
            return c.replace(suffix, "").strip()
    return "unknown"


def _detect_attributes(tag_str: str) -> tuple[str, str]:
    hair = _detect_color(tag_str, HAIR_COLORS, " hair")
    eyes = _detect_color(tag_str, EYE_COLORS, " eyes")
    return hair, eyes


def run(images_dir: Path, workdir: Path) -> Path:
    """Group images into folders based on detected hair and eye color."""

    workdir.mkdir(parents=True, exist_ok=True)
    log_step("Classification started")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        session, img_size, tags = _load_tagger(device)
    except Exception as exc:  # pragma: no cover - download may fail
        log_step(f"Tagger unavailable: {exc}; putting all images in 'unclassified'")
        for img in sorted(images_dir.glob("*.png")):
            char_dir = workdir / "unclassified"
            char_dir.mkdir(exist_ok=True)
            shutil.copy(img, char_dir / img.name)
        log_step("Classification completed with fallback")
        return workdir

    for img_path in sorted(images_dir.glob("*.png")):
        tag_str = _tag_image(session, img_size, img_path, tags)
        hair, eyes = _detect_attributes(tag_str)
        if hair == "unknown" or eyes == "unknown":
            char_dir = workdir / "unclassified"
        else:
            char_dir = workdir / f"{hair}_{eyes}"
        char_dir.mkdir(exist_ok=True)
        shutil.copy(img_path, char_dir / img_path.name)

    log_step("Classification completed")
    return workdir
