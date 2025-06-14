"""Automatic tagging using the WD14 tagger."""

from pathlib import Path
from typing import List, Any
import csv

from PIL import Image
import torch
import open_clip
from huggingface_hub import hf_hub_download

from ..logging_utils import log_step


_REPO = "SmilingWolf/wd-swinv2-tagger-v3"
_TAGS_FILE = "selected_tags.csv"


def _load_tagger(device: torch.device) -> tuple[torch.nn.Module, Any, List[str]]:
    """Load the WD14 tagger model and tag list from the Hugging Face Hub."""

    log_step("Downloading tagger weights")
    model, _, preprocess = open_clip.create_model_and_transforms(
        f"hf-hub:{_REPO}",
        device=device,
    )
    model.eval()

    tags_path = hf_hub_download(_REPO, _TAGS_FILE)
    with open(tags_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        tags = [row[0] for row in reader]

    return model, preprocess, tags


def _tag_image(
    model: torch.nn.Module,
    preprocess: Any,
    img_path: Path,
    device: torch.device,
    tags: List[str],
    threshold: float = 0.35,
) -> str:
    """Return a comma-separated tag string for an image."""

    with Image.open(img_path).convert("RGB") as img:
        img_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img_tensor)
        scores = logits.sigmoid().cpu().numpy()[0]

    selected = [tags[i] for i, s in enumerate(scores) if s > threshold]
    return ", ".join(selected)


def run(cropped_dir: Path, captions_dir: Path) -> None:
    """Run image annotation with automatic tagging and fallback."""

    captions_dir.mkdir(parents=True, exist_ok=True)
    log_step("Annotation started")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model, preprocess, tags = _load_tagger(device)
    except Exception as exc:  # pragma: no cover - download may fail
        log_step(f"Tagger unavailable: {exc}; using fallback captions")
        for img in sorted(cropped_dir.glob("*.png")):
            caption_file = captions_dir / f"{img.stem}.txt"
            caption_file.write_text("anime_style")
        log_step("Annotation completed with fallback")
        return

    for img in sorted(cropped_dir.glob("*.png")):
        caption = _tag_image(model, preprocess, img, device, tags)
        caption_file = captions_dir / f"{img.stem}.txt"
        caption_file.write_text(caption)

    log_step("Annotation completed")
