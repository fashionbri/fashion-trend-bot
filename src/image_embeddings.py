# src/image_embeddings.py
from pathlib import Path
from typing import List

import pandas as pd
from PIL import Image
from sentence_transformers import SentenceTransformer

from .config import LATEST

# CLIP-like vision model (still via sentence-transformers)
MODEL_NAME = "clip-ViT-B-32"


def _load_image(path: Path) -> Image.Image | None:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def build_image_embeddings(
    manifest_path: Path = LATEST / "download_manifest.csv",
    out_path: Path | None = None,
    image_path_column: str = "path",
) -> Path:
    """
    Read download_manifest.csv, embed each image using a vision model,
    and save to data/latest/image_features.csv.

    You may need to change `image_path_column` to match your CSV:
      common options: "local_path", "filepath", "image_path"
    """

    if out_path is None:
        out_path = LATEST / "image_features.csv"

    if not manifest_path.exists():
        print(f"[image_embeddings] No manifest at {manifest_path}, skipping.")
        return out_path

    df = pd.read_csv(manifest_path)

    if "image_id" not in df.columns:
        # Fall back to using index
        df["image_id"] = df.index.astype(str)

    if image_path_column not in df.columns:
        print(f"[image_embeddings] Column {image_path_column!r} not in download_manifest, available: {df.columns.tolist()}")
        return out_path

    model = SentenceTransformer(MODEL_NAME)

    image_ids: List[str] = []
    emb_list = []

    for _, row in df.iterrows():
        img_path = Path(row[image_path_column])
        if not img_path.is_absolute():
            # assume relative to repo root
            img_path = Path(".") / img_path
        img = _load_image(img_path)
        if img is None:
            continue

        emb = model.encode(img, convert_to_numpy=True, normalize_embeddings=True)
        image_ids.append(str(row["image_id"]))
        emb_list.append(emb)

    if not emb_list:
        print("[image_embeddings] No embeddings generated, skipping.")
        return out_path

    import numpy as np

    embs = np.vstack(emb_list)
    dim = embs.shape[1]
    emb_cols = [f"emb_{i}" for i in range(dim)]
    out_df = pd.DataFrame(embs, columns=emb_cols)
    out_df.insert(0, "image_id", image_ids)

    LATEST.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"[image_embeddings] wrote {out_path} with {len(out_df)} rows, dim={dim}")
    return out_path


if __name__ == "__main__":
    build_image_embeddings()
