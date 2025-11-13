# src/save_colors.py

from pathlib import Path
from datetime import datetime

import pandas as pd
from PIL import Image

from .config import LATEST
from .image_features import dominant_with_pantone


def analyze_and_save(image_path: Path, image_id: str, k: int = 5) -> Path:
    """
    Run dominant color analysis (with optional Pantone info) on a single image,
    normalize column names, and update both per-run and master color CSVs
    under data/latest/.

    Expected downstream columns after normalization:
      - hex      → main color hex
      - family   → color family name (if available)
      - share    → relative share / weight of that swatch (if available)
    """

    # Open image
    img = Image.open(image_path)

    # dominant_with_pantone should return a list[dict] with keys like:
    #   color_hex, color_family, count / rank_share, pantone_code, etc.
    rows = dominant_with_pantone(img, k=k)
    if not rows:
        # Nothing to save; just bail quietly
        return LATEST / f"colors_{datetime.utcnow():%Y%m%d}.csv"

    df = pd.DataFrame(rows)

    # Ensure we track which image each row came from
    df.insert(0, "image_id", image_id)

    # ---------- Normalize column names ----------
    # So everything downstream (including your summary scripts) can rely on:
    #   hex, family, share
    rename_map = {}

    # color_hex / color → hex
    if "color_hex" in df.columns:
        rename_map["color_hex"] = "hex"
    if "color" in df.columns:
        # If both exist, "color_hex" wins; this just covers the other case
        rename_map["color"] = "hex"

    # color_family → family
    if "color_family" in df.columns:
        rename_map["color_family"] = "family"

    # share-like columns → share
    if "share" not in df.columns:
        if "count" in df.columns:
            rename_map["count"] = "share"
        elif "rank_share" in df.columns:
            rename_map["rank_share"] = "share"

    if rename_map:
        df = df.rename(columns=rename_map)

    # Make sure the latest folder exists
    LATEST.mkdir(parents=True, exist_ok=True)

    # ---------- 1) Per-run, date-stamped file ----------
    # e.g. data/latest/colors_20251112.csv
    out = LATEST / f"colors_{datetime.utcnow():%Y%m%d}.csv"
    df.to_csv(out, mode="a", index=False, header=not out.exists())

    # ---------- 2) Master colors file (used by daily/weekly summaries) ----------
    # This is what your summary code expects as "colors_opencv.csv"
    colors_master = LATEST / "colors_opencv.csv"
    df.to_csv(
        colors_master,
        mode="a",
        index=False,
        header=not colors_master.exists(),
    )

    # ---------- 3) Master families file (optional, if family exists) ----------
    if "family" in df.columns:
        fam_cols = ["image_id"]
        if "hex" in df.columns:
            fam_cols.append("hex")
        fam_cols.append("family")
        if "share" in df.columns:
            fam_cols.append("share")

        df[fam_cols].to_csv(
            LATEST / "colors_families.csv",
            mode="a",
            index=False,
            header=not (LATEST / "colors_families.csv").exists(),
        )

    return out
