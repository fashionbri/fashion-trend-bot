from PIL import Image
import numpy as np
from collections import Counter

def dominant_colors(pil_img: Image.Image, k: int = 5):
    """Return k hex colors from an image (very simple sampler)."""
    a = np.array(pil_img.convert("RGB"))
    h, w, _ = a.shape
    sample = a.reshape(-1,3)
    if sample.shape[0] > 8000:
        idx = np.random.choice(sample.shape[0], 8000, replace=False)
        sample = sample[idx]
    cnt = Counter(map(tuple, sample))
    top = [c for c,_ in cnt.most_common(k)]
    return ["#%02x%02x%02x" % c for c in top]
