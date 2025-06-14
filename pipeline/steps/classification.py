from pathlib import Path
import shutil
from ..logging_utils import log_step


def run(frames_dir: Path, workdir: Path) -> Path:
    """Placeholder classification step."""
    workdir.mkdir(parents=True, exist_ok=True)
    log_step('Classification started')
    target = workdir / 'character'
    target.mkdir(exist_ok=True)
    for frame in sorted(frames_dir.glob('*.png')):
        shutil.copy(frame, target / frame.name)
    log_step('Classification completed')
    return target
