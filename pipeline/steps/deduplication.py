from pathlib import Path
import shutil
from ..logging_utils import log_step


def run(frames_dir: Path, workdir: Path) -> Path:
    """Simplified deduplication that copies files."""
    workdir.mkdir(parents=True, exist_ok=True)
    log_step('Deduplication started')
    for frame in sorted(frames_dir.glob('*.png')):
        dest = workdir / frame.name
        shutil.copy(frame, dest)
    log_step('Deduplication completed')
    return workdir
