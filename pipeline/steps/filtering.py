from pathlib import Path
import shutil
from ..logging_utils import log_step


def run(classified_dir: Path, workdir: Path) -> Path:
    """Placeholder filtering step."""
    workdir.mkdir(parents=True, exist_ok=True)
    log_step('Filtering started')
    for img in sorted(classified_dir.glob('*.png')):
        shutil.copy(img, workdir / img.name)
    log_step('Filtering completed')
    return workdir
