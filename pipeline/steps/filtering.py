from pathlib import Path
import shutil
from ..logging_utils import log_step, log_progress


def run(classified_dir: Path, workdir: Path) -> Path:
    """Placeholder filtering step."""
    workdir.mkdir(parents=True, exist_ok=True)
    log_step('Filtering started')
    images = sorted(classified_dir.rglob('*.png'))
    total = len(images)
    for idx, img in enumerate(images, 1):
        # flatten the directory structure for downstream steps
        shutil.copy(img, workdir / img.name)
        log_progress('Filtering', idx, total)
    log_step('Filtering completed')
    return workdir
