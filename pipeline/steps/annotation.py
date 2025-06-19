"""Automatic tagging using the WD14 tagger."""

from pathlib import Path
from typing import List, Any
import csv

from PIL import Image
import torch
from huggingface_hub import hf_hub_download
import numpy as np
import cv2
from onnxruntime import InferenceSession

from ..logging_utils import log_step, log_progress


_REPO = "SmilingWolf/wd-swinv2-tagger-v3"
_TAGS_FILE = "selected_tags.csv"


def _load_tagger(device: torch.device) -> tuple[InferenceSession, int, List[str]]:
    """Load the ONNX tagger model and tag list from the Hugging Face Hub."""

    log_step("Downloading tagger weights")
    model_path = hf_hub_download(_REPO, "model.onnx")
    tags_path = hf_hub_download(_REPO, _TAGS_FILE)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device.type == "cpu":
        providers = ["CPUExecutionProvider"]

    session = InferenceSession(model_path, providers=providers)

    with open(tags_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip header row
        tags = [row[1] for row in reader]

    input_height = session.get_inputs()[0].shape[2]
    return session, input_height, tags


def _preprocess_image(img: Image.Image, image_size: int) -> np.ndarray:
    """Prepare an image for the ONNX tagger."""

    img = img.convert("RGBA")
    new = Image.new("RGBA", img.size, "WHITE")
    new.paste(img, mask=img)
    img = new.convert("RGB")
    img = np.asarray(img)[:, :, ::-1]

    # padding and resizing
    old_h, old_w = img.shape[:2]
    desired = max(old_h, old_w, image_size)
    delta_w = desired - old_w
    delta_h = desired - old_h
    top, bottom = delta_h // 2, delta_h - delta_h // 2
    left, right = delta_w // 2, delta_w - delta_w // 2
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    if img.shape[0] != image_size:
        interp = cv2.INTER_AREA if img.shape[0] > image_size else cv2.INTER_CUBIC
        img = cv2.resize(img, (image_size, image_size), interpolation=interp)

    img = img.astype(np.float32)
    img = np.expand_dims(img, 0)
    return img


def _tag_image(
    session: InferenceSession,
    image_size: int,
    img_path: Path,
    tags: List[str],
    *,
    threshold: float = 0.3,
    max_tags: int | None = None,
    min_tags: int | None = None,
) -> str:
    """Return a comma-separated tag string for an image.

    Parameters
    ----------
    session:
        Inference session for the WD14 model.
    image_size:
        Target image size expected by the model.
    img_path:
        Image to tag.
    tags:
        List of tag names corresponding to the model outputs.
    threshold:
        Minimum score required for a tag to be included. Defaults to ``0.3``.
    max_tags:
        Optional upper bound for the number of tags to return. If ``None`` the
        number of tags is unrestricted.
    min_tags:
        Optional minimum number of tags to return. If there are fewer than this
        amount above ``threshold`` the highest scoring tags are added until the
        minimum is reached.
    """

    with Image.open(img_path) as img:
        img_tensor = _preprocess_image(img, image_size)

    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name
    scores = session.run([label_name], {input_name: img_tensor})[0][0]

    tag_scores = [
        (tags[i], float(scores[i + 4])) for i in range(len(tags))
    ]
    tag_scores.sort(key=lambda x: x[1], reverse=True)

    selected = [tag for tag, s in tag_scores if s > threshold]

    if max_tags is not None:
        selected = selected[:max_tags]

    if min_tags is not None and len(selected) < min_tags:
        additional = [tag for tag, _ in tag_scores if tag not in selected]
        selected.extend(additional[: min_tags - len(selected)])

    return ", ".join(selected)


def run(
    cropped_dir: Path,
    captions_dir: Path,
    *,
    trigger_word: str = "name",
    preloaded: tuple[InferenceSession, int, List[str]] | None = None,
) -> None:
    """Run image annotation with automatic tagging and fallback.

    Parameters
    ----------
    cropped_dir:
        Directory containing the cropped images to tag.
    captions_dir:
        Output directory for generated caption files.
    trigger_word:
        The first tag to prepend to every caption. Defaults to ``"name"``.
    """

    captions_dir.mkdir(parents=True, exist_ok=True)
    log_step("Annotation started")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        if preloaded is not None:
            session, img_size, tags = preloaded
        else:
            session, img_size, tags = _load_tagger(device)
    except Exception as exc:  # pragma: no cover - download may fail
        log_step(f"Tagger unavailable: {exc}; using fallback captions")
        for img in sorted(cropped_dir.glob("*.png")):
            caption_file = captions_dir / f"{img.stem}.txt"
            caption_file.write_text(f"{trigger_word}, anime_style")
        log_step("Annotation completed with fallback")
        return

    images = sorted(cropped_dir.glob("*.png"))
    total = len(images)
    for idx, img in enumerate(images, 1):
        caption = _tag_image(
            session,
            img_size,
            img,
            tags,
            threshold=0.3,
            max_tags=30,
            min_tags=10,
        )
        if caption:
            caption = f"{trigger_word}, {caption}"
        else:
            caption = trigger_word
        caption_file = captions_dir / f"{img.stem}.txt"
        caption_file.write_text(caption)
        log_progress("Annotation", idx, total)

    log_step("Annotation completed")
