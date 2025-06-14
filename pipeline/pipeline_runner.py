from pathlib import Path
import shutil
import zipfile
from typing import Callable

from .logging_utils import log_step
from .steps import frame_extraction, deduplication, classification, filtering, upscaling, cropping, annotation

MODELS_DIR = Path("models")


class Pipeline:
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        work_dir: Path,
        *,
        yolo_model: Path | None = None,
    ) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.work_dir = work_dir
        if yolo_model is not None:
            self.yolo_model = yolo_model
        else:
            self.yolo_model = self._detect_yolo_model()

    def _detect_yolo_model(self) -> Path | None:
        """Return a YOLOv8 weight file from ``models/`` if present."""
        MODELS_DIR.mkdir(exist_ok=True)
        patterns = ["*.pt", "*.pth"]
        for pattern in patterns:
            models = sorted(MODELS_DIR.glob(pattern))
            if models:
                log_step(f"Found YOLO model: {models[0]}")
                return models[0]
        log_step("No YOLO model found")
        return None

    def cleanup(self):
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir)

    def run(
        self,
        video_path: Path,
        *,
        trigger_word: str = "name",
        progress_cb: Callable[[int, str], None] | None = None,
    ):
        """Execute the full pipeline on ``video_path``.

        Parameters
        ----------
        video_path:
            Input video file to process.
        trigger_word:
            Tag to prepend to every caption. Defaults to ``"name"``.
        """
        try:
            if progress_cb:
                progress_cb(0, 'Starting')
            if self.output_dir.exists():
                shutil.rmtree(self.output_dir)
            zip_path = self.output_dir.with_suffix('.zip')
            if zip_path.exists():
                zip_path.unlink()
            if self.work_dir.exists():
                shutil.rmtree(self.work_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if progress_cb:
                progress_cb(1, 'Frame Extraction')
            work_frames = self.work_dir / 'frames'
            frames = frame_extraction.run(video_path, work_frames, fps=1)

            work_dedup = self.work_dir / 'dedup'
            if progress_cb:
                progress_cb(2, 'Deduplication')
            deduped = deduplication.run(frames, work_dedup)
            shutil.rmtree(work_frames)

            work_filter = self.work_dir / 'filtering'
            if progress_cb:
                progress_cb(3, 'Filtering')
            filtered = filtering.run(deduped, work_filter)
            shutil.rmtree(work_dedup)

            work_upscale = self.work_dir / 'upscaling'
            if progress_cb:
                progress_cb(4, 'Upscaling')
            upscaled = upscaling.run(filtered, work_upscale)
            shutil.rmtree(work_filter)

            work_crop = self.work_dir / 'cropping'
            if progress_cb:
                progress_cb(5, 'Cropping')
            cropped = cropping.run(
                upscaled,
                work_crop,
                yolo_model=self.yolo_model,
            )
            shutil.rmtree(work_upscale)

            captions_dir = self.output_dir / 'captions'
            if progress_cb:
                progress_cb(6, 'Annotation')
            annotation.run(cropped, captions_dir, trigger_word=trigger_word)

            work_class = self.work_dir / 'classification'
            if progress_cb:
                progress_cb(7, 'Classification')
            classified = classification.run(cropped, work_class)
            images_dir = self.output_dir / 'images'
            shutil.copytree(classified, images_dir)
            shutil.rmtree(work_class)
            shutil.rmtree(work_crop)

            # Zip output
            if progress_cb:
                progress_cb(8, 'Packaging')
            zip_path = self.output_dir.with_suffix('.zip')
            with zipfile.ZipFile(zip_path, 'w') as zf:
                for path in self.output_dir.rglob('*'):
                    zf.write(path, path.relative_to(self.output_dir))
            shutil.rmtree(self.output_dir)
            log_step(f'Pipeline completed successfully: {zip_path}')
            return zip_path
        except Exception as e:
            log_step(f'Pipeline failed: {e}')
            raise
        finally:
            self.cleanup()
