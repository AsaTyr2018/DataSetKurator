from pathlib import Path
import shutil
from ..logging_utils import log_step


def run(upscaled_dir: Path, workdir: Path) -> Path:
    """Placeholder cropping step."""
    workdir.mkdir(parents=True, exist_ok=True)
    log_step('Cropping started')
    for img in sorted(upscaled_dir.glob('*.png')):
        shutil.copy(img, workdir / img.name)
    log_step('Cropping completed')
    return workdir
