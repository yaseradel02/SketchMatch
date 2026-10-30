
import argparse
import json
import time
from pathlib import Path

import numpy as np

from search_core import (
    load_feature_extractor,
    embed,
    search
)


# ============================================================
# 1. LOAD GALLERY DATA
# ============================================================

def load_gallery(data_path):
    """
    Read gallery images and labels from Person 1's subset.npz.

    Expected keys:
        gallery_images
        gallery_labels
    """

    with np.load(data_path, allow_pickle=False) as data:

        images = data["gallery_images"]
        labels = data["gallery_labels"]

    # Support flattened Quick Draw images, if needed
    if images.ndim == 2 and images.shape[1] == 784:
        images = images.reshape(-1, 28, 28)

    # Validate image dimensions
    if images.ndim != 3 or images.shape[1:] != (28, 28):
        raise ValueError(
            "Gallery images must have shape (N,28,28) "
            "or (N,784)."
        )

    if len(images) == 0:
        raise ValueError("Gallery is empty.")

    if len(images) != len(labels):
        raise ValueError(
            "Number of images and labels must match."
        )

    if images.min() < 0 or images.max() > 255:
        raise ValueError(
            "Image pixel values must be between 0 and 255."
        )

    images = images.astype(np.uint8)
    labels = labels.astype(str)

    return images, labels


# ============================================================
# 2. BUILD THE SEARCH INDEX
# ============================================================

def build_index(
    data_path="data/subset.npz",
    output_dir="index",
    batch_size=64,
    invert=False
):
    """
    Convert gallery doodles into normalized embeddings
    and save everything needed for similarity search.
    """

    start_time = time.time()

    data_path = Path(data_path)
    output_dir = Path(output_dir)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # STEP A: LOAD GALLERY DOODLES
    # --------------------------------------------------------

    gallery_images, gallery_labels = load_gallery(
        data_path
    )

    print("Gallery images:", gallery_images.shape)
    print("Gallery labels:", gallery_labels.shape)
    print("Unique categories:", np.unique(gallery_labels))

    # --------------------------------------------------------
    # STEP B: LOAD THE PRETRAINED MODEL
    # --------------------------------------------------------

    print("\nLoading pretrained MobileNetV2...")

    model = load_feature_extractor()

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # STEP C: EXTRACT EMBEDDINGS
    # --------------------------------------------------------

    print("\nGenerating gallery embeddings...")

    embeddings = embed(
        model,
        gallery_images,
        batch_size=batch_size,
        invert=invert
    )

    print("Embedding shape:", embeddings.shape)

    # --------------------------------------------------------
    # STEP D: CHECK EMBEDDING QUALITY
    # --------------------------------------------------------

    if len(embeddings) != len(gallery_images):
        raise ValueError(
            "Embedding count does not match gallery size."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Embeddings contain invalid numerical values."
        )

    norms = np.linalg.norm(
        embeddings,
        axis=1
    )

    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError(
            "Embeddings are not properly L2-normalized."
        )

    print(
        "Minimum vector length:",
        round(float(norms.min()), 6)
    )

    print(
        "Maximum vector length:",
        round(float(norms.max()), 6)
    )

    # --------------------------------------------------------
    # STEP E: SAVE SEARCH FILES
    # --------------------------------------------------------

    np.save(
        output_dir / "embeddings.npy",
        embeddings
    )

    np.save(
        output_dir / "gallery_images.npy",
        gallery_images
    )

    # Labels are saved for evaluation only.
    # The Streamlit app must never display them.
    np.save(
        output_dir / "gallery_labels.npy",
        gallery_labels
    )

    # --------------------------------------------------------
    # STEP F: SAVE CONFIGURATION
    # --------------------------------------------------------

    config = {
        "model_name": "MobileNetV2",
        "weights": "imagenet",
        "input_size": [128, 128],
        "embedding_dim": int(embeddings.shape[1]),
        "gallery_size": int(len(gallery_images)),
        "invert": bool(invert),
        "top_k": 25,
        "threshold": None
    }

    with open(
        output_dir / "config.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            config,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # STEP G: TEST THE SAVED INDEX
    # --------------------------------------------------------

    print("\nTesting the saved search index...")

    saved_embeddings = np.load(
        output_dir / "embeddings.npy"
    )

    # Use the first gallery doodle as an identical query
    query_vector = saved_embeddings[0]

    indices, scores = search(
        query_vector,
        saved_embeddings,
        k=5
    )

    print("Top 5 indices:", indices)
    print("Top 5 scores:", np.round(scores, 4))

    # The identical image must score approximately 1.
    if not np.isclose(scores[0], 1.0, atol=1e-4):
        raise ValueError(
            "Self-similarity check failed."
        )

    elapsed = time.time() - start_time

    print("\nSEARCH INDEX BUILT SUCCESSFULLY!")
    print("Output folder:", output_dir.resolve())
    print("Total images:", len(gallery_images))
    print("Vector dimension:", embeddings.shape[1])
    print("Inverted colors:", invert)
    print("Build duration (seconds):", round(elapsed, 2))

    return embeddings, config


# ============================================================
# 3. COMMAND LINE ENTRY POINT
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Build DoodleDex gallery search index."
    )

    parser.add_argument(
        "--data",
        default="data/subset.npz"
    )

    parser.add_argument(
        "--output",
        default="index"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64
    )

    parser.add_argument(
        "--invert",
        action="store_true"
    )

    args = parser.parse_args()

    build_index(
        data_path=args.data,
        output_dir=args.output,
        batch_size=args.batch_size,
        invert=args.invert
    )
