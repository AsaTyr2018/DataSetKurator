"""Automatic upscaling and quality checking."""

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import torch
import cv2

from ..logging_utils import log_step, log_progress


try:  # Optional dependency
    try:
        from torchvision.transforms.functional_tensor import rgb_to_grayscale  # type: ignore
    except Exception:  # pragma: no cover - new torchvision versions
        from torchvision.transforms.functional import rgb_to_grayscale
        import types, sys

        shim = types.ModuleType("torchvision.transforms.functional_tensor")
        shim.rgb_to_grayscale = rgb_to_grayscale
        sys.modules["torchvision.transforms.functional_tensor"] = shim

    try:
        from realesrgan import RealESRGAN  # old API
    except Exception:  # pragma: no cover - fallback to new API
        from realesrgan.utils import RealESRGANer as RealESRGAN
except Exception:  # pragma: no cover - library may not be installed
    RealESRGAN = None  # type: ignore[misc]


def _load_model(device: torch.device, scale: int) -> Optional[object]:
    """Load RealESRGAN anime model if available."""

    if RealESRGAN is None:
        log_step("RealESRGAN not available – using PIL resize")
        return None
    try:
        if RealESRGAN.__name__ == "RealESRGANer":  # modernes API
            from realesrgan.archs.srvgg_arch import SRVGGNetCompact

            url = (
                "https://github.com/xinntao/Real-ESRGAN/releases/download/"
                "v0.2.5.0/realesr-animevideov3.pth"
            )
            arch = SRVGGNetCompact(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_conv=16,
                upscale=scale,
                act_type="prelu",
            )
            # ``RealESRGANer`` lädt die Gewichte automatisch
            model = RealESRGAN(
                scale=scale,
                model_path=url,
                model=arch,
                device=device,
                half=False,
            )
        else:  # ältere API
            url = (
                "https://github.com/xinntao/Real-ESRGAN/releases/download/"
                "v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
            )
            model = RealESRGAN(device, scale=scale)
            model.load_weights(url)
        return model
    except Exception as exc:  # pragma: no cover - runtime download may fail
        log_step(f"RealESRGAN load failed: {exc}; falling back to PIL resize")
        return None


def _is_acceptable(img: Image.Image, blur_thresh: float, dark_thresh: float) -> bool:
    """Return ``True`` if image passes basic quality checks."""

    gray = np.array(img.convert("L"))
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    if variance < blur_thresh:
        return False
    brightness = gray.mean()
    if brightness < dark_thresh:
        return False
    return True


def run(
    filtered_dir: Path,
    workdir: Path,
    *,
    scale: int = 4,
    blur_threshold: float = 100.0,
    dark_threshold: float = 40.0,
) -> Path:
    """Upscale images with RealESRGAN and drop low-quality frames."""

    workdir.mkdir(parents=True, exist_ok=True)
    log_step("Upscaling started")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(device, scale)

    images = sorted(filtered_dir.glob("*.png"))
    total = len(images)
    for idx, img_path in enumerate(images, 1):
        with Image.open(img_path).convert("RGB") as img:
            if not _is_acceptable(img, blur_threshold, dark_threshold):
                continue

            if model is not None:
                with torch.no_grad():  # pragma: no cover - heavy model inference
                    upscaled, _ = model.enhance(np.array(img))
                up_img = Image.fromarray(upscaled)
            else:
                width, height = img.size
                up_img = img.resize((width * scale, height * scale), Image.LANCZOS)

            out_path = workdir / img_path.name
            up_img.save(out_path)
        log_progress("Upscaling", idx, total)

    log_step("Upscaling completed")
    return workdir
