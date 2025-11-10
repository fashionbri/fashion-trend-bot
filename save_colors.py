# src/save_colors.py
from pathlib import Path
import pandas as pd
from PIL import Image
from datetime import datetime
from .config import LATEST
from .image_features import dominant_with_pantone

def analyze_and_save(image_path: Path, image_id: str, k: int = 5) -> Path:
    img = Image.open(image_path)
    rows = dominant_with_pantone(img, k=k)
    df = pd.DataFrame(rows)
    df.insert(0, "image_id", image_id)
    out = LATEST / f"colors_{datetime.utcnow().strftime('%Y%m%d')}.csv"
    header = not out.exists()
    df.to_csv(out, mode="a", index=False, header=header)
    return out
