# src/text_features.py
from pathlib import Path
from typing import List

import pandas as pd
from sentence_transformers import SentenceTransformer

from .config import LATEST

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _combine_text_fields(row: pd.Series, fields: List[str]) -> str:
    parts = []
    for f in fields:
        val = str(row.get(f, "") or "").strip()
        if val:
            parts.append(val)
    return " ".join(parts)


def build_text_features(
    metadata_path: Path = Path("data/metadata.csv"),
    out_path: Path | None = None,
) -> Path:
    """
    Read metadata.csv, combine text fields into one string per image,
    embed with a sentence-transformer, and save to data/latest/text_features.csv.

    Expected columns in metadata.csv (tweak to match your file):
      - image_id
      - title
      - alt_text
      - caption
      - description
    """

    if out_path is None:
        out_path = LATEST / "text_features.csv"

    if not metadata_path.exists():
        print(f"[text_features] No metadata file at {metadata_path}, skipping.")
        return out_path

    df = pd.read_csv(metadata_path)
    if "image_id" not in df.columns:
        print("[text_features] metadata.csv has no image_id column, skipping.")
        return out_path

    # tweak this list if your columns are named differently
    text_fields = [c for c in ["title", "alt_text", "caption", "description", "query"] if c in df.columns]
    if not text_fields:
        print("[text_features] No text fields found (title/alt_text/caption/description/query), skipping.")
        return out_path

    df["text"] = df.apply(lambda r: _combine_text_fields(r, text_fields), axis=1)

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(df["text"].tolist(), batch_size=64, convert_to_numpy=True, normalize_embeddings=True)

    # Build output frame: image_id + emb_0..emb_(dim-1)
    dim = embeddings.shape[1]
    emb_cols = {i: f"emb_{i}" for i in range(dim)}
    out_df = pd.DataFrame(embeddings, columns=[emb_cols[i] for i in range(dim)])
    out_df.insert(0, "image_id", df["image_id"].values)

    LATEST.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"[text_features] wrote {out_path} with {len(out_df)} rows, dim={dim}")
    return out_path


if __name__ == "__main__":
    build_text_features()
