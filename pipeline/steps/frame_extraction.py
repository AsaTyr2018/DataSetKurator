"""Frame extraction step using ffmpeg."""

from pathlib import Path
import shutil
import subprocess
from typing import Iterable

from ..logging_utils import log_step


SUPPORTED_EXTS: Iterable[str] = {".mp4", ".mkv", ".avi", ".mov", ".webm"}


def _check_ffmpeg() -> None:
    """Ensure ffmpeg is available."""
    if not shutil.which("ffmpeg"):
        raise EnvironmentError("ffmpeg is not installed or not in PATH")


def run(video: Path, workdir: Path, fps: int = 1) -> Path:
    """Extract frames from the video using ffmpeg.

    Parameters
    ----------
    video: Path
        Path to the input video file.
    workdir: Path
        Directory where extracted frames will be stored.
    fps: int, optional
        Number of frames per second to extract. Defaults to ``1``.
    """
    if video.suffix.lower() not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported video format: {video.suffix}")

    _check_ffmpeg()

    workdir.mkdir(parents=True, exist_ok=True)
    output_pattern = workdir / "frame_%04d.png"
    log_step("Frame Extraction started")
    try:
        subprocess.run(
            ["ffmpeg", "-i", str(video), "-vf", f"fps={fps}", str(output_pattern)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        log_step(f"Frame extraction failed: {e}")
        raise
    log_step("Frame Extraction completed")
    return workdir
