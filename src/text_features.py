# src/text_features.py
from pathlib import Path
from typing import List

import pandas as pd
from sentence_transformers import SentenceTransformer

from .config import LATEST

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _combine_text_fields(row: pd.Series, fields: List[str]) -> str:
    """
    Join available text fields into one string for embedding.
    """
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
    Read data/metadata.csv, combine text fields per image, embed using a
    sentence-transformer model, and save to data/latest/text_features.csv.

    It is robust to missing columns: it will use any of these that exist:
      - title, alt_text, caption, description, query, headline, body
    """
    if out_path is None:
        out_path = LATEST / "text_features.csv"

    if not metadata_path.exists():
        print(f"[text_features] No metadata file at {metadata_path}, skipping.")
        return out_path

    df = pd.read_csv(metadata_path)

    if "image_id" not in df.columns:
        # Best effort: try to construct an image_id if missing
        if "id" in df.columns:
            df["image_id"] = df["id"].astype(str)
        elif "path" in df.columns:
            df["image_id"] = df["path"].astype(str)
        else:
            df["image_id"] = df.index.astype(str)

    # pick any text-like fields that exist
    candidate_fields = [
        "title",
        "alt_text",
        "caption",
        "description",
        "query",
        "headline",
        "body",
    ]
    text_fields = [c for c in candidate_fields if c in df.columns]

    if not text_fields:
        print("[text_features] No text-like fields found in metadata.csv, skipping.")
        return out_path

    df["text"] = df.apply(lambda r: _combine_text_fields(r, text_fields), axis=1)

    # filter out completely empty text rows
    df = df[df["text"].str.strip() != ""]
    if df.empty:
        print("[text_features] All rows have empty text, skipping.")
        return out_path

    print(f"[text_features] Using fields {text_fields} for text embeddings.")
    print(f"[text_features] Encoding {len(df)} rows with {MODEL_NAME}…")

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        df["text"].tolist(),
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    dim = embeddings.shape[1]
    emb_cols = [f"emb_{i}" for i in range(dim)]
    out_df = pd.DataFrame(embeddings, columns=emb_cols)
    out_df.insert(0, "image_id", df["image_id"].astype(str).values)

    LATEST.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"[text_features] wrote {out_path} with {len(out_df)} rows, dim={dim}")
    return out_path


if __name__ == "__main__":
    build_text_features()
