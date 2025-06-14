from pathlib import Path
import subprocess
from ..logging_utils import log_step


def run(video: Path, workdir: Path) -> Path:
    """Extract frames from the video using ffmpeg."""
    workdir.mkdir(parents=True, exist_ok=True)
    output_pattern = workdir / 'frame_%04d.png'
    log_step('Frame Extraction started')
    try:
        subprocess.run([
            'ffmpeg', '-i', str(video), '-vf', 'fps=1', str(output_pattern)
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        log_step(f'Frame extraction failed: {e}')
        raise
    log_step('Frame Extraction completed')
    return workdir
