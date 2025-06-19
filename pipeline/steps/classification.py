"""Character classification based on hair, eye and style detection."""

from pathlib import Path
from typing import List
from onnxruntime import InferenceSession
import shutil

import torch
import numpy as np
from PIL import Image
import open_clip
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from ..logging_utils import log_step, log_progress
from .annotation import _load_tagger, _tag_image

HAIR_COLORS = [
    "blonde hair",
    "black hair",
    "brown hair",
    "red hair",
    "blue hair",
    "green hair",
    "purple hair",
    "pink hair",
    "orange hair",
    "silver hair",
    "white hair",
    "gray hair",
    "aqua hair",
    "grey hair",
    "violet hair",
    "lavender hair",
    "teal hair",
    "auburn hair",
]

HAIR_LENGTHS = [
    "long hair",
    "short hair",
    "twintails",
    "ponytail",
]

ACCESSORIES = [
    "glasses",
]

EYE_COLORS = [
    "blue eyes",
    "brown eyes",
    "red eyes",
    "green eyes",
    "purple eyes",
    "yellow eyes",
    "pink eyes",
    "aqua eyes",
    "orange eyes",
    "gray eyes",
    "grey eyes",
    "silver eyes",
    "gold eyes",
    "teal eyes",
    "violet eyes",
]


def _detect_color(tag_str: str, colors: List[str], suffix: str) -> str:
    for c in colors:
        if c in tag_str:
            return c.replace(suffix, "").strip()
    return "unknown"


def _detect_feature(tag_str: str, features: List[str]) -> str:
    for f in features:
        if f in tag_str:
            return f.replace(" ", "_")
    return "none"


def _detect_attributes(tag_str: str) -> tuple[str, str, str, str]:
    hair = _detect_color(tag_str, HAIR_COLORS, " hair")
    eyes = _detect_color(tag_str, EYE_COLORS, " eyes")
    length = _detect_feature(tag_str, HAIR_LENGTHS)
    accessory = _detect_feature(tag_str, ACCESSORIES)
    return hair, eyes, length, accessory


def _cluster_unknowns(unclassified_dir: Path, *, n_clusters: int | None = None) -> None:
    """Cluster images in ``unclassified_dir`` using CLIP embeddings and KMeans.

    If ``n_clusters`` is ``None`` an optimal value is estimated via the
    silhouette score in the range 2..10 (or the number of images).
    """

    images = sorted(unclassified_dir.glob("*.png"))
    if not images:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model = model.to(device)
    model.eval()

    feats = []
    for img_path in images:
        with Image.open(img_path) as img:
            img_t = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model.encode_image(img_t)
            feats.append(emb.cpu().numpy()[0])

    feats = np.stack(feats)
    reduced = umap.UMAP(n_components=5, random_state=42).fit_transform(feats)

    if n_clusters is None:
        max_k = min(len(feats), 10)
        if max_k <= 1:
            n_clusters = 1
        else:
            best_k = 2
            best_score = -1.0
            for k in range(2, max_k + 1):
                labels_tmp = KMeans(n_clusters=k, random_state=42).fit_predict(reduced)
                score = silhouette_score(reduced, labels_tmp)
                if score > best_score:
                    best_k = k
                    best_score = score
            n_clusters = best_k
    elif len(feats) < n_clusters:
        n_clusters = max(1, len(feats))

    labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(reduced)

    for img_path, label in zip(images, labels):
        cluster_dir = unclassified_dir / f"cluster_{label:02d}"
        cluster_dir.mkdir(exist_ok=True)
        shutil.move(img_path, cluster_dir / img_path.name)


def run(
    images_dir: Path,
    workdir: Path,
    *,
    preloaded: tuple[InferenceSession, int, List[str]] | None = None,
) -> Path:
    """Group images into folders based on detected hair, eye and style tags."""

    workdir.mkdir(parents=True, exist_ok=True)
    log_step("Classification started")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        if preloaded is not None:
            session, img_size, tags = preloaded
        else:
            session, img_size, tags = _load_tagger(device)
    except Exception as exc:  # pragma: no cover - download may fail
        log_step(f"Tagger unavailable: {exc}; putting all images in 'unclassified'")
        for img in sorted(images_dir.glob("*.png")):
            char_dir = workdir / "unclassified"
            char_dir.mkdir(exist_ok=True)
            shutil.copy(img, char_dir / img.name)
        log_step("Classification completed with fallback")
        return workdir

    images = sorted(images_dir.glob("*.png"))
    total = len(images)
    for idx, img_path in enumerate(images, 1):
        tag_str = _tag_image(session, img_size, img_path, tags, threshold=0.20)
        hair, eyes, length, accessory = _detect_attributes(tag_str)
        if hair == "unknown" or eyes == "unknown":
            tag_str = _tag_image(session, img_size, img_path, tags, threshold=0.15)
            hair, eyes, length, accessory = _detect_attributes(tag_str)
        if hair == "unknown" or eyes == "unknown":
            char_dir = workdir / "unclassified"
        else:
            parts = [hair, eyes]
            if length != "none":
                parts.append(length)
            if accessory != "none":
                parts.append(accessory)
            char_dir = workdir / "_".join(parts)
        char_dir.mkdir(exist_ok=True)
        shutil.copy(img_path, char_dir / img_path.name)
        log_progress("Classification", idx, total)

    unclassified = workdir / "unclassified"
    if unclassified.exists():
        log_step("Clustering unclassified images")
        _cluster_unknowns(unclassified)

    log_step("Classification completed")
    return workdir
