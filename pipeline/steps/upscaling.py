from pathlib import Path
import shutil
from ..logging_utils import log_step


def run(filtered_dir: Path, workdir: Path) -> Path:
    """Placeholder upscaling step."""
    workdir.mkdir(parents=True, exist_ok=True)
    log_step('Upscaling started')
    for img in sorted(filtered_dir.glob('*.png')):
        shutil.copy(img, workdir / img.name)
    log_step('Upscaling completed')
    return workdir
