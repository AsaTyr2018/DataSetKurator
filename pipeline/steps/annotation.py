from pathlib import Path
from ..logging_utils import log_step


def run(cropped_dir: Path, captions_dir: Path) -> None:
    """Placeholder annotation step that creates dummy captions."""
    captions_dir.mkdir(parents=True, exist_ok=True)
    log_step('Annotation started')
    for img in sorted(cropped_dir.glob('*.png')):
        caption_file = captions_dir / f"{img.stem}.txt"
        caption_file.write_text('1girl, solo, anime_style')
    log_step('Annotation completed')
