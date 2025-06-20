from pathlib import Path
import os
import shutil
import zipfile
from typing import Callable
import torch

from .logging_utils import log_step
from .steps import (
    frame_extraction,
    deduplication,
    classification,
    filtering,
    upscaling,
    cropping,
    annotation,
)
from .preloader import (
    detect_yolo_model,
    preload_yolo,
    preload_tagger,
    preload_realesrgan,
    get as get_model,
)


class Pipeline:
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        work_dir: Path,
        *,
        yolo_model: Path | None = None,
        preload: bool | None = None,
    ) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.work_dir = work_dir
        if yolo_model is not None:
            self.yolo_model = yolo_model
        else:
            self.yolo_model = detect_yolo_model()

        env_preload = os.getenv("DSK_PRELOAD", "1")
        self.preload = preload if preload is not None else env_preload != "0"

    def cleanup(self):
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir)

    def run(
        self,
        video_path: Path,
        *,
        trigger_word: str = "name",
        progress_cb: Callable[[int, str], None] | None = None,
        fps: int = 1,
        dedup_threshold: int = 8,
        scale: int = 4,
        blur_threshold: float = 100.0,
        dark_threshold: float = 40.0,
        margin: float = 0.3,
        conf_threshold: float = 0.5,
        batch_size: int = 4,
        skip_deduplication: bool = False,
        skip_filtering: bool = False,
        skip_upscaling: bool = False,
        skip_cropping: bool = False,
        skip_annotation: bool = False,
        skip_classification: bool = False,
    ):
        """Execute the full pipeline on ``video_path``.

        Parameters
        ----------
        video_path:
            Input video file to process.
        trigger_word:
            Tag to prepend to every caption. Defaults to ``"name"``.
        fps:
            Frames per second for extraction.
        dedup_threshold:
            Hamming distance for deduplication.
        scale:
            Upscaling factor.
        blur_threshold:
            Minimum Laplacian variance to keep a frame.
        dark_threshold:
            Minimum brightness level.
        margin:
            Extra border around detected faces.
        conf_threshold:
            YOLO confidence threshold.
        batch_size:
            How many images to process per YOLO batch.
        """
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self.preload:
                preload_realesrgan(device, 4)
                preload_yolo(self.yolo_model)
                preload_tagger(device)

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

            # Frame Extraction is mandatory
            if progress_cb:
                progress_cb(1, 'Frame Extraction')
            work_frames = self.work_dir / 'frames'
            frames = frame_extraction.run(video_path, work_frames, fps=fps)
            current = frames

            # Deduplication
            work_dedup = self.work_dir / 'dedup'
            if skip_deduplication:
                if progress_cb:
                    progress_cb(2, 'Deduplication (skipped)')
                deduped = current
            else:
                if progress_cb:
                    progress_cb(2, 'Deduplication')
                deduped = deduplication.run(current, work_dedup, threshold=dedup_threshold)
                shutil.rmtree(current)
                current = deduped

            # Filtering
            work_filter = self.work_dir / 'filtering'
            if skip_filtering:
                if progress_cb:
                    progress_cb(3, 'Filtering (skipped)')
                filtered = current
            else:
                if progress_cb:
                    progress_cb(3, 'Filtering')
                filtered = filtering.run(current, work_filter)
                shutil.rmtree(current)
                current = filtered

            # Upscaling
            work_upscale = self.work_dir / 'upscaling'
            if skip_upscaling:
                if progress_cb:
                    progress_cb(4, 'Upscaling (skipped)')
                upscaled = current
            else:
                if progress_cb:
                    progress_cb(4, 'Upscaling')
                upscaled = upscaling.run(
                    current,
                    work_upscale,
                    scale=scale,
                    blur_threshold=blur_threshold,
                    dark_threshold=dark_threshold,
                    model=get_model("realesrgan") if self.preload else None,
                    device=device,
                )
                shutil.rmtree(current)
                current = upscaled

            # Cropping
            work_crop = self.work_dir / 'cropping'
            if skip_cropping:
                if progress_cb:
                    progress_cb(5, 'Cropping (skipped)')
                cropped = current
            else:
                if progress_cb:
                    progress_cb(5, 'Cropping')
                cropped = cropping.run(
                    current,
                    work_crop,
                    margin=margin,
                    yolo_model=self.yolo_model,
                    yolo=get_model("yolo") if self.preload else None,
                    conf_threshold=conf_threshold,
                    batch_size=batch_size,
                )
                shutil.rmtree(current)
                current = cropped

            captions_dir = self.output_dir / 'captions'
            if skip_annotation:
                if progress_cb:
                    progress_cb(6, 'Annotation (skipped)')
            else:
                if progress_cb:
                    progress_cb(6, 'Annotation')
                annotation.run(
                    current,
                    captions_dir,
                    trigger_word=trigger_word,
                    preloaded=get_model("tagger") if self.preload else None,
                )

            work_class = self.work_dir / 'classification'
            if skip_classification:
                if progress_cb:
                    progress_cb(7, 'Classification (skipped)')
                classified = current
            else:
                if progress_cb:
                    progress_cb(7, 'Classification')
                classified = classification.run(
                    current,
                    work_class,
                    preloaded=get_model("tagger") if self.preload else None,
                )
                shutil.rmtree(current)
                current = classified

            images_dir = self.output_dir / 'images'
            shutil.copytree(current, images_dir)
            if current.exists() and current != images_dir:
                shutil.rmtree(current)
            if work_crop.exists() and not skip_cropping:
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
