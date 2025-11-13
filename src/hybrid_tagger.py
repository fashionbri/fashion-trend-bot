# src/hybrid_tagger.py
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

from .config import LATEST


def _load_embeddings(csv_path: Path) -> pd.DataFrame:
    """
    Load an embeddings CSV with columns: image_id, emb_0, emb_1, ...
    """
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if "image_id" not in df.columns or not emb_cols:
        return pd.DataFrame()
    return df[["image_id"] + emb_cols]


def _collect_terms_from_module(mod, group_name: str) -> List[Tuple[str, str]]:
    """
    Extract string terms from a taxonomy module.

    Strategy:
      1) If module has a list/tuple named TERMS, use that.
      2) Otherwise, collect all list/tuple attributes that are all-strings.
    Returns list of (group, term) pairs.
    """
    terms: List[Tuple[str, str]] = []

    # 1) Preferred: TERMS attribute
    if hasattr(mod, "TERMS"):
        try:
            raw = getattr(mod, "TERMS")
            if isinstance(raw, (list, tuple)):
                for t in raw:
                    if isinstance(t, str) and t.strip():
                        terms.append((group_name, t.strip()))
                if terms:
                    return terms
        except Exception:
            pass

    # 2) Fallback: any all-string lists/tuples
    for name, val in vars(mod).items():
        if name.startswith("_"):
            continue
        if isinstance(val, (list, tuple)) and val:
            if all(isinstance(x, str) for x in val):
                for t in val:
                    if t.strip():
                        # group can be "group_name.attr" for more flavor
                        terms.append((f"{group_name}.{name}", t.strip()))

    return terms


def _import_tax_module(mod_name: str):
    """
    Try to import either modules.<name> or <name> directly.
    """
    try:
        # try modules.<name>
        mod = __import__(f"modules.{mod_name}", fromlist=["*"])
        return mod
    except ImportError:
        try:
            mod = __import__(mod_name)
            return mod
        except ImportError:
            return None


def _load_taxonomy() -> pd.DataFrame:
    """
    Load taxonomy terms from your taxonomy modules (garments, fabrics, etc.).

    It supports either:
      - a package 'modules' with accessories.py, garments.py, etc.
      - or flat files accessories.py, garments.py at repo root.

    It will dynamically collect all string lists it finds.
    """
    module_names = [
        "garments",
        "fabrics",
        "accessories",
        "silhouettes",
        "styling",
        "mood",
        "motifs",
        "textures",
        "geography",
        "lighting",
    ]

    rows: List[Dict[str, str]] = []

    for name in module_names:
        mod = _import_tax_module(name)
        if mod is None:
            continue
        terms = _collect_terms_from_module(mod, group_name=name)
        for group, term in terms:
            rows.append({"group": group, "term": term})

    if not rows:
        print("[hybrid_tagger] No taxonomy terms found. Check your modules/*.py.")
        return pd.DataFrame()

    df = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    print(f"[hybrid_tagger] Loaded {len(df)} taxonomy terms.")
    return df


def run_hybrid_tagger(out_path: Path | None = None) -> Path:
    """
    Combine text + image embeddings and match them against taxonomy terms
    using semantic similarity.

    Output: data/latest/hybrid_tags.csv with columns:
      image_id, group, term, score
    """
    if out_path is None:
        out_path = LATEST / "hybrid_tags.csv"

    text_df = _load_embeddings(LATEST / "text_features.csv")
    img_df = _load_embeddings(LATEST / "image_features.csv")

    if text_df.empty and img_df.empty:
        print("[hybrid_tagger] No text_features.csv or image_features.csv, skipping.")
        return out_path

    # merge on image_id; if one side is missing, fall back to the one we have
    if not text_df.empty and not img_df.empty:
        merged = pd.merge(text_df, img_df, on="image_id", suffixes=("_text", "_img"))
        text_cols = [c for c in merged.columns if c.startswith("emb_") and c.endswith("_text")]
        img_cols = [c for c in merged.columns if c.startswith("emb_") and c.endswith("_img")]

        if not text_cols or not img_cols:
            print("[hybrid_tagger] Could not find matched text/img columns, skipping.")
            return out_path

        text_arr = merged[text_cols].to_numpy(dtype="float32")
        img_arr = merged[img_cols].to_numpy(dtype="float32")
        emb_arr = (text_arr + img_arr) / 2.0
        image_ids = merged["image_id"].astype(str).tolist()
    else:
        src = text_df if not text_df.empty else img_df
        emb_cols = [c for c in src.columns if c.startswith("emb_")]
        emb_arr = src[emb_cols].to_numpy(dtype="float32")
        image_ids = src["image_id"].astype(str).tolist()

    tax_df = _load_taxonomy()
    if tax_df.empty:
        print("[hybrid_tagger] Taxonomy is empty, skipping.")
        return out_path

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Encode taxonomy terms
    term_embs = model.encode(
        tax_df["term"].tolist(),
        batch_size=64,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    # Turn image embeddings into tensor + normalize
    img_emb_tensor = model.to_tensor(emb_arr)
    img_emb_tensor = util.normalize_embeddings(img_emb_tensor)

    sims = util.matmul(img_emb_tensor, term_embs.T)  # [num_images x num_terms]

    rows = []
    threshold = 0.35  # tweak higher if you want only very strong tags

    sims_np = sims.cpu().numpy()
    for i, image_id in enumerate(image_ids):
        sim_row = sims_np[i]
        idxs = np.where(sim_row >= threshold)[0]
        for idx in idxs:
            rows.append(
                {
                    "image_id": image_id,
                    "group": tax_df.iloc[idx]["group"],
                    "term": tax_df.iloc[idx]["term"],
                    "score": float(sim_row[idx]),
                }
            )

    if not rows:
        print("[hybrid_tagger] No matches above threshold.")
        return out_path

    out_df = pd.DataFrame(rows).sort_values(
        ["image_id", "group", "score"], ascending=[True, True, False]
    )
    LATEST.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"[hybrid_tagger] wrote {out_path} with {len(out_df)} rows")
    return out_path


if __name__ == "__main__":
    run_hybrid_tagger()
