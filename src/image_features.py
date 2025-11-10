# src/image_features.py
from PIL import Image
import numpy as np
from collections import Counter
from typing import List, Dict
from .color_to_pantone import PantoneLookup

def dominant_colors(pil_img: Image.Image, k: int = 5) -> List[str]:
    arr = np.array(pil_img.convert("RGB"))
    pixels = arr.reshape(-1, 3)
    if pixels.shape[0] > 8000:
        idx = np.random.choice(pixels.shape[0], 8000, replace=False)
        pixels = pixels[idx]
    cnt = Counter(map(tuple, pixels))
    top = [c for c, _ in cnt.most_common(k)]
    return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in top]

def dominant_with_pantone(pil_img: Image.Image, k: int = 5) -> List[Dict]:
    hexes = dominant_colors(pil_img, k=k)
    lookup = PantoneLookup()  # loads FULL 3200+ list
    return [{ "hex": h, **lookup.nearest(h) } for h in hexes]
