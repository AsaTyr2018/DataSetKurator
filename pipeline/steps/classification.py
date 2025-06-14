"""Character classification using CLIP embeddings and DBSCAN clustering."""

from pathlib import Path
from typing import List, Any
import shutil

import numpy as np
from PIL import Image
import torch
import open_clip
from sklearn.cluster import DBSCAN

from ..logging_utils import log_step


def _load_model(device: torch.device) -> tuple[torch.nn.Module, Any]:
    """Load an anime-focused CLIP model with fallback to the OpenAI weights."""
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16",
            pretrained="hf-hub:dudcjs2779/anime-style-tag-clip",
            device=device,
        )
    except Exception as exc:  # pragma: no cover - external download may fail
        log_step(f"Custom weights unavailable: {exc}; falling back to 'openai'")
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16",
            pretrained="openai",
            device=device,
        )
    model.eval()
    return model, preprocess


def _embed_images(
    model: torch.nn.Module, preprocess: Any, images: List[Path], device: torch.device
) -> np.ndarray:
    """Compute normalized CLIP embeddings for all images."""
    features = []
    for img_path in images:
        with Image.open(img_path).convert("RGB") as img:
            img_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(img_tensor)
        emb = emb.cpu().numpy()[0]
        emb /= np.linalg.norm(emb)
        features.append(emb)
    return np.stack(features)


def run(frames_dir: Path, workdir: Path, eps: float = 0.3, min_samples: int = 2) -> Path:
    """Group frames by character using DBSCAN clustering on CLIP embeddings."""
    workdir.mkdir(parents=True, exist_ok=True)
    log_step("Classification started")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = _load_model(device)

    images = sorted(frames_dir.glob("*.png"))
    if not images:
        log_step("No frames found for classification")
        return workdir

    embeddings = _embed_images(model, preprocess, images, device)

    clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = clusterer.fit_predict(embeddings)

    for img_path, label in zip(images, labels):
        label_dir = workdir / (f"character_{label}" if label >= 0 else "unclassified")
        label_dir.mkdir(exist_ok=True)
        shutil.copy(img_path, label_dir / img_path.name)

    log_step("Classification completed")
    return workdir
