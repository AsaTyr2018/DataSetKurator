"""Frame deduplication step using perceptual hashing."""

from pathlib import Path
import shutil
from typing import List

from PIL import Image
import imagehash

from ..logging_utils import log_step


def run(frames_dir: Path, workdir: Path, threshold: int = 8) -> Path:
    """Remove near-duplicate frames using perceptual hash.

    Parameters
    ----------
    frames_dir:
        Directory containing extracted frames.
    workdir:
        Destination directory for deduplicated frames.
    threshold:
        Maximum Hamming distance between perceptual hashes to consider frames
        duplicates. Lower values remove more images.
    """

    workdir.mkdir(parents=True, exist_ok=True)
    log_step("Deduplication started")

    hashes: List[imagehash.ImageHash] = []
    for frame in sorted(frames_dir.glob("*.png")):
        with Image.open(frame) as img:
            phash = imagehash.phash(img)

        if all(phash - h > threshold for h in hashes):
            hashes.append(phash)
            shutil.copy(frame, workdir / frame.name)

    log_step("Deduplication completed")
    return workdir
