import os, time, hashlib, re, shutil
from datetime import datetime, timezone
from pathlib import Path
import requests
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2
import pandas as pd

# ========= Settings you can edit =========
QUERIES = [
    "vogue runway emerald dress",
    "leopard print runway 1990s",
    "Gucci Fall 2013 runway",
    "Chanel haute couture 2020",
    "Balmain burgundy runway 2025",
]
IMAGES_PER_QUERY = 60          # results to attempt per query
MIN_BYTES = 20_000             # skip tiny icons
OUT_DIR = Path("data/latest")  # where CSVs & images land
# ========================================

# ---- API key (SerpAPI only) ----
SERP_KEY = os.environ.get("SERP_API_KEY")
assert SERP_KEY, "Set SERP_API_KEY in repo Settings → Secrets and variables → Actions."

OUT_DIR.mkdir(parents=True, exist_ok=True)
IM_DIR = OUT_DIR / "images"
IM_DIR.mkdir(parents=True, exist_ok=True)

# -------- helpers --------
def _ext_from_ct(ctype: str) -> str:
    ctype = (ctype or "").lower()
    if "png" in ctype:  return ".png"
    if "webp" in ctype: return ".webp"
    if "jpeg" in ctype: return ".jpg"
    if "gif" in ctype:  return ".gif"
    return ".jpg"

def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_")[:60]

def search_google_serpapi(query: str, count: int):
    """Return a list of original image URLs using SerpAPI (Google Images)."""
    url = "https://serpapi.com/search.json"
    params = {"engine": "google_images", "q": query, "ijn": 0, "api_key": SERP_KEY}
    try:
        r = requests.get(url, params=params, timeout=25)
        r.raise_for_status()
        data = r.json()
        urls = [img.get("original") for img in data.get("images_results", []) if img.get("original")]
        return urls[:count]
    except Exception:
        return []

def download_image(url: str, prefix: str) -> str | None:
    try:
        r = requests.get(url, stream=True, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "")
        data = r.content
        if len(data) < MIN_BYTES:
            return None
        h = hashlib.sha256(data).hexdigest()[:16]
        ext = _ext_from_ct(ctype)
        fpath = IM_DIR / f"{prefix}_{h}{ext}"
        with open(fpath, "wb") as f:
            f.write(data)
        return str(fpath)
    except Exception:
        return None

def dominant_colors(path: Path, k: int = 5):
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    Z = arr.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 8, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    counts = np.bincount(labels.flatten(), minlength=k)
    fracs = counts / counts.sum()
    # list of (hex, fraction) sorted by fraction desc
    out = [(f"#{r:02x}{g:02x}{b:02x}", float(fr)) for (r, g, b), fr in zip(centers, fracs)]
    return sorted(out, key=lambda x: x[1], reverse=True)

# -------- 1) search + download --------
download_manifest = []
for q in QUERIES:
    urls = list(set(search_google_serpapi(q, IMAGES_PER_QUERY)))
    for u in tqdm(urls, desc=f"Downloading {q[:28]}"):
        p = download_image(u, prefix=_slug(q))
        if p:
            download_manifest.append({"query": q, "path": p, "url": u})
    time.sleep(1)  # polite pause

pd.DataFrame(download_manifest).to_csv(OUT_DIR / "download_manifest.csv", index=False)

# -------- 2) color extraction (top-5) --------
rows = []
for p in tqdm(list(IM_DIR.glob("*")), desc="Extracting colors"):
    fp = Path(p)
    if fp.is_file() and fp.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
        try:
            for rank, (hexcol, share) in enumerate(dominant_colors(fp, k=5), start=1):
                rows.append({"image_path": str(fp), "rank": rank, "color_hex": hexcol, "share": share})
        except Exception:
            continue

colors_df = pd.DataFrame(rows)
colors_df.to_csv(OUT_DIR / "colors_opencv.csv", index=False)

# -------- 3) quick daily summary --------
summary = (
    colors_df.query("rank == 1")
    .groupby("color_hex")["image_path"]
    .count()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)
summary.to_csv(OUT_DIR / "top_colors_today.csv", index=False)

# -------- 4) archive snapshot --------
stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M_UTC")
hist = Path(f"data/history/{stamp}")
hist.mkdir(parents=True, exist_ok=True)
for name in ["download_manifest.csv", "colors_opencv.csv", "top_colors_today.csv"]:
    src = OUT_DIR / name
    dst = hist / name
    shutil.copyfile(src, dst)

print("Done.")
