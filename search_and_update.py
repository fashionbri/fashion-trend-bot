import os, time, hashlib, re, shutil, math
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm
from PIL import Image, ImageOps
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
IMAGES_PER_QUERY = 60          # target per query
MIN_BYTES = 20_000
OUT_DIR = Path("data/latest")
MAX_WORKERS = 8                # parallel downloads
PAGE_SIZE = 100                # SerpAPI page size for images
RECENCY_MODE = os.getenv("RECENCY_MODE", "w")  # d = day, w = week, m = month
K_COLORS = 5
RANDOM_SEED = 42               # reproducible KMeans
MAX_SIDE = 768                 # resize longer side for speed
SAMPLE_PIXELS = 120_000        # subsample pixels for kmeans
TIMEOUT = 25                   # network timeout (seconds)
# ========================================

SERP_KEY = os.environ.get("SERP_API_KEY")
assert SERP_KEY, "Set SERP_API_KEY in repo Settings → Secrets and variables → Actions."

OUT_DIR.mkdir(parents=True, exist_ok=True)
IM_DIR = OUT_DIR / "images"
IM_DIR.mkdir(parents=True, exist_ok=True)

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0 (TrendBot)"})


def _ext_from_ct(ctype: str) -> str:
    ctype = (ctype or "").lower()
    if "png" in ctype:  return ".png"
    if "webp" in ctype: return ".webp"
    if "jpeg" in ctype or "jpg" in ctype: return ".jpg"
    if "gif" in ctype:  return ".gif"
    return ".jpg"


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_")[:60]


def serpapi_search_urls(query: str, target_count: int, recency: str = "w") -> list[str]:
    """Paginate SerpAPI Google Images results until target_count or pages exhausted.
       recency: 'd' (past day), 'w' (past week), 'm' (past month)"""
    urls = []
    pages = math.ceil(target_count / PAGE_SIZE)
    for i in range(pages):
        params = {
            "engine": "google_images",
            "q": query,
            "ijn": i,
            "api_key": SERP_KEY,
            "tbs": f"qdr:{recency}"   # ⬅️ NEW: time-bounded results
        }
        ...

    """Paginate SerpAPI Google Images results until target_count or pages exhausted."""
    urls = []
    pages = math.ceil(target_count / PAGE_SIZE)
    for i in range(pages):
        params = {"engine": "google_images", "q": query, "ijn": i, "api_key": SERP_KEY}
        for attempt in range(3):
            try:
                r = session.get(
                    "https://serpapi.com/search.json",
                    params=params,
                    timeout=TIMEOUT
                )
                r.raise_for_status()
                data = r.json()
                batch = [img.get("original") for img in data.get("images_results", []) if img.get("original")]
                urls.extend(batch)
                break
            except Exception:
                time.sleep(1.5 * (attempt + 1))
        time.sleep(0.4)  # gentle pacing
        if len(urls) >= target_count:
            break
    # dedupe while preserving order
    seen, ordered = set(), []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            ordered.append(u)
    return ordered[:target_count]


def download_image(url: str, prefix: str) -> dict | None:
    for attempt in range(2):  # tiny retry for flaky hosts
        try:
            r = session.get(url, stream=True, timeout=TIMEOUT)
            r.raise_for_status()
            data = r.content
            if len(data) < MIN_BYTES:
                return None
            h = hashlib.sha256(data).hexdigest()[:16]
            ext = _ext_from_ct(r.headers.get("Content-Type", ""))
            fpath = IM_DIR / f"{prefix}_{h}{ext}"
            with open(fpath, "wb") as f:
                f.write(data)
            return {"path": str(fpath), "url": url}
        except Exception:
            time.sleep(0.6 * (attempt + 1))
    return None


def _prepare_for_kmeans(img: Image.Image) -> np.ndarray:
    # Fix orientation, convert to RGB, resize to speed
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    scale = min(1.0, MAX_SIDE / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    arr = np.array(img)  # RGB uint8
    # Convert to CIELAB (perceptual)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    Z = lab.reshape((-1, 3)).astype(np.float32)
    # Subsample pixels for speed while keeping distribution
    if Z.shape[0] > SAMPLE_PIXELS:
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(Z.shape[0], size=SAMPLE_PIXELS, replace=False)
        Z = Z[idx]
    return Z


def lab_to_hex(center_lab: np.ndarray) -> str:
    # center_lab: [L,a,b] float32 -> single pixel LAB -> RGB -> hex
    lab = np.uint8(np.clip(center_lab, 0, 255))[None, None, :]
    rgb = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB).reshape(3)
    r, g, b = [int(x) for x in rgb]
    return f"#{r:02x}{g:02x}{b:02x}"


def dominant_colors(path: Path, k: int = K_COLORS):
    with Image.open(path) as img:
        Z = _prepare_for_kmeans(img)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    cv2.setRNGSeed(RANDOM_SEED)  # reproducible
    _compactness, labels, centers = cv2.kmeans(
        Z, k, None, criteria, 8, cv2.KMEANS_PP_CENTERS
    )
    centers = centers.astype(np.float32)
    counts = np.bincount(labels.flatten(), minlength=k).astype(np.float64)
    fracs = (counts / counts.sum()).tolist()
    hexes = [lab_to_hex(c) for c in centers]
    out = sorted(zip(hexes, fracs), key=lambda x: x[1], reverse=True)
    return out


# ---- 1) search + download (parallel) ----
download_manifest = []
for q in QUERIES:
    print(f"🔍 Searching for '{q}' (recency='{RECENCY_MODE}')")
    urls = serpapi_search_urls(q, IMAGES_PER_QUERY, recency=RECENCY_MODE)
    prefix = _slug(q)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(download_image, u, prefix) for u in urls]
        for f in tqdm(as_completed(futures), total=len(futures), desc=f"Downloading {q[:28]}"):
            res = f.result()
            if res:
                download_manifest.append({"query": q, "path": res["path"], "url": res["url"]})
    time.sleep(0.5)  # page-level pause

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(download_image, u, prefix) for u in urls]
        for f in tqdm(as_completed(futures), total=len(futures), desc=f"Downloading {q[:28]}"):
            res = f.result()
            if res:
                download_manifest.append({"query": q, "path": res["path"], "url": res["url"]})
    time.sleep(0.5)  # page-level pause

pd.DataFrame(download_manifest).to_csv(OUT_DIR / "download_manifest.csv", index=False)

# ---- 2) color extraction ----
rows = []
images = [p for p in IM_DIR.glob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
for p in tqdm(images, desc="Extracting colors"):
    try:
        for rank, (hexcol, share) in enumerate(dominant_colors(p, k=K_COLORS), start=1):
            rows.append({"image_path": str(p), "rank": rank, "color_hex": hexcol, "share": share})
    except Exception:
        continue

colors_df = pd.DataFrame(rows)
colors_df.to_csv(OUT_DIR / "colors_opencv.csv", index=False)

# ---- 2b) simple color families (HSV buckets) ----
def hex_to_hsv(hexcode: str):
    r = int(hexcode[1:3], 16)
    g = int(hexcode[3:5], 16)
    b = int(hexcode[5:7], 16)
    hsv = cv2.cvtColor(np.uint8([[[r, g, b]]]), cv2.COLOR_RGB2HSV)[0, 0]
    h, s, v = float(hsv[0]) * 2.0, float(hsv[1]) / 255.0, float(hsv[2]) / 255.0  # hue scaled to 0–360
    return h, s, v


def color_family(hexcode: str) -> str:
    h, s, v = hex_to_hsv(hexcode)
    if v < 0.18 or (s < 0.08 and v < 0.9):
        return "neutrals"
    if s < 0.18 and v >= 0.9:
        return "pastels"
    if s > 0.7 and v > 0.7:
        return "neon"
    # hue bands (degrees)
    if 15 <= h < 45:   return "earth"          # yellow-orange
    if 45 <= h < 75:   return "earth"          # yellow-green
    if 75 <= h < 165:  return "green/blue"
    if 165 <= h < 255: return "blue/purple"
    if 255 <= h < 285: return "jewel"
    if h >= 285 or h < 15: return "primary/jewel"  # reds
    return "neutrals"


if not colors_df.empty:
    fam = colors_df.assign(color_family=colors_df["color_hex"].map(color_family))
    fam.to_csv(OUT_DIR / "colors_families.csv", index=False)
else:
    fam = pd.DataFrame(columns=["image_path", "rank", "color_hex", "share", "color_family"])
    fam.to_csv(OUT_DIR / "colors_families.csv", index=False)

# ---- 3) summaries (top-1 swatch per image) ----
if not colors_df.empty:
    summary = (
        colors_df.query("rank == 1")
        .groupby("color_hex")["image_path"].count()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
else:
    summary = pd.DataFrame(columns=["color_hex", "count"])
summary.to_csv(OUT_DIR / "top_colors_today.csv", index=False)

if not fam.empty:
    fam_summary = (
        fam.query("rank == 1")
        .groupby("color_family")["image_path"].count()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
else:
    fam_summary = pd.DataFrame(columns=["color_family", "count"])
fam_summary.to_csv(OUT_DIR / "top_color_families_today.csv", index=False)

# ---- 4) archive snapshot ----
# --- optional analysis modules (safe to skip if not installed) ---
try:
    from modules import textures, fabrics, silhouettes, garments, accessories, motifs, mood, geography, styling, lighting
    HAVE_MODULES = True
except Exception:
    HAVE_MODULES = False

if HAVE_MODULES:
    IMAGE_LIST = [str(p) for p in IM_DIR.glob("*") if p.is_file() and p.suffix.lower() in {".jpg",".jpeg",".png",".webp"}]

    # lightweight modules (always run)
    textures.run(IMAGE_LIST, OUT_DIR)
    fabrics.run(IMAGE_LIST, OUT_DIR)
    motifs.run(IMAGE_LIST, OUT_DIR)
    mood.run(IMAGE_LIST, OUT_DIR)
    geography.run(download_manifest, OUT_DIR)   # uses queries/urls

    # heavy modules (run when env flag set)
    RUN_HEAVY = os.getenv("RUN_HEAVY", "0") == "1"
    if RUN_HEAVY:
        garments.run(IMAGE_LIST, OUT_DIR)
        accessories.run(IMAGE_LIST, OUT_DIR)
        silhouettes.run(IMAGE_LIST, OUT_DIR)
        styling.run(IMAGE_LIST, OUT_DIR)
        lighting.run(IMAGE_LIST, OUT_DIR)
else:
    print("Modules folder not found — running colors only.")

stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M_UTC")
hist = Path(f"data/history/{stamp}")
hist.mkdir(parents=True, exist_ok=True)
for name in [
    "download_manifest.csv",
    "colors_opencv.csv",
    "colors_families.csv",
    "top_colors_today.csv",
    "top_color_families_today.csv",
]:
    src = OUT_DIR / name
    if src.exists():
        shutil.copyfile(src, hist / name)

if colors_df.empty:
    print("Done. (No colors extracted — check queries/connectivity.)")
else:
    print("Done.")
