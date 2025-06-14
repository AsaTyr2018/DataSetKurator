from pathlib import Path
import shutil
import zipfile

from .logging_utils import log_step
from .steps import frame_extraction, deduplication, classification, filtering, upscaling, cropping, annotation


class Pipeline:
    def __init__(self, input_dir: Path, output_dir: Path, work_dir: Path):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.work_dir = work_dir

    def cleanup(self):
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir)

    def run(self, video_path: Path):
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            work_frames = self.work_dir / 'frames'
            frames = frame_extraction.run(video_path, work_frames, fps=1)

            work_dedup = self.work_dir / 'dedup'
            deduped = deduplication.run(frames, work_dedup)
            shutil.rmtree(work_frames)

            work_class = self.work_dir / 'classification'
            classified = classification.run(deduped, work_class)
            shutil.rmtree(work_dedup)

            work_filter = self.work_dir / 'filtering'
            filtered = filtering.run(classified, work_filter)
            shutil.rmtree(work_class)

            work_upscale = self.work_dir / 'upscaling'
            upscaled = upscaling.run(filtered, work_upscale)
            shutil.rmtree(work_filter)

            work_crop = self.work_dir / 'cropping'
            cropped = cropping.run(upscaled, work_crop)
            shutil.rmtree(work_upscale)

            captions_dir = self.output_dir / 'captions'
            images_dir = self.output_dir / 'images'
            images_dir.mkdir(exist_ok=True)
            for img in sorted(cropped.glob('*.png')):
                shutil.copy(img, images_dir / img.name)
            annotation.run(cropped, captions_dir)
            shutil.rmtree(work_crop)

            # Zip output
            zip_path = self.output_dir.with_suffix('.zip')
            with zipfile.ZipFile(zip_path, 'w') as zf:
                for path in self.output_dir.rglob('*'):
                    zf.write(path, path.relative_to(self.output_dir))
            log_step(f'Pipeline completed successfully: {zip_path}')
            return zip_path
        except Exception as e:
            log_step(f'Pipeline failed: {e}')
            raise
        finally:
            self.cleanup()
