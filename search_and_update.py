from typing import Optional
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
# ========= Query Configuration =========
import random
...
CURATED_QUERIES = [
    # global runway signals
    "Paris Fashion Week runway 2025",
    "Milan Fashion Week runway 2025",
    "New York Fashion Week runway 2025",
    "London Fashion Week runway 2025",
    "Tokyo Fashion Week runway 2025",
    "Seoul Fashion Week runway 2025",

    # early 2026 collections (forward look)
    "Paris Fashion Week runway 2026",
    "Milan Fashion Week runway 2026",
    "New York Fashion Week runway 2026",
    "London Fashion Week runway 2026",

    # street style
...
    "fashion editorial 2026",
    "fashion presentation 2026",

    # runway details
    "runway close up details 2025",
    "runway backstage fashion 2025",
    "runway backstage fashion 2026",
    "fashion week accessories 2025",
    "fashion week accessories 2026",
    "global fashion week street style 2025",
    "global fashion week street style 2026"
]

# 2️⃣ Auto-generated global combos — to expand diversity
CITIES = [
    "Paris", "Milan", "New York", "London",
    "Tokyo", "Seoul", "Copenhagen", "Stockholm"
]
CONTEXTS = [
    "runway", "street style", "backstage", "presentation",
    "fashion editorial", "lookbook", "campaign"
]
YEARS = ["2024", "2025", "2026"]

AUTO_QUERIES = [
    f"{city} Fashion Week {ctx} {year}"
    for city in CITIES
    for ctx in ["street style", "runway", "backstage"]
    for year in YEARS
] + [
    f"{ctx} fashion {year}"
    for ctx in CONTEXTS
    for year in YEARS
]

# 3️⃣ Combine both curated + generated lists
ALL_QUERIES = list(set(CURATED_QUERIES + AUTO_QUERIES))

# 4️⃣ Shuffle randomly each run (fresh sample order)
random.seed()           # system time
random.shuffle(ALL_QUERIES)

# 5️⃣ How many queries per run?
QUERIES_PER_RUN = int(os.getenv("QUERIES", "25"))
IMAGES_PER_QUERY = int(os.getenv("IMAGES_PER_QUERY", "60"))
MAX_IMAGES = QUERIES_PER_RUN * IMAGES_PER_QUERY

# ========= Paths =========
ROOT = Path(".")
OUT_DIR = ROOT / "data/latest"
HIST_DIR = ROOT / "data/history"

OUT_DIR.mkdir(parents=True, exist_ok=True)
HIST_DIR.mkdir(parents=True, exist_ok=True)

# ========= SerpAPI / KMeans / Imaging Settings =========
PAGE_SIZE = 100              # SerpAPI page size for images
RECENCY_MODE = os.getenv("RECENCY_MODE", "w")  # d = day, w = week, m = month
K_COLORS = 5
RANDOM_SEED = 42             # reproducible KMeans
MAX_SIDE = 768               # resize longer side for speed
SAMPLE_PIXELS = 120_000      # subsample pixels for kmeans
TIMEOUT = 25                 # network timeout (seconds)
# ========================================

SERP_KEY = os.environ.get("SERP_API_KEY")
assert SERP_KEY, "Set SERP_API_KEY in repo Settings → Secrets and variables → Actions."

OUT_DIR.mkdir(parents=True, exist_ok=True)
IM_DIR = OUT_DIR / "images"
IM_DIR.mkdir(parents=True, exist_ok=True)

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0 (TrendBot)"})

# ====== RECENCY / AGE FILTERS ======
# Any image whose URL/date suggests a year below this is treated as "too old"
YEAR_CUTOFF = 2022  # adjust to 2023+ if you want ultra-fresh only

def looks_too_old(result: dict) -> bool:
    """
    Return True if this SerpAPI result clearly looks too old
    based on URL or explicit date fields.
    """
    # 1) Check URL for old years
    url = (
        result.get("original")
        or result.get("image")
        or result.get("link")
        or result.get("thumbnail")
        or ""
    )

    m_url = re.search(r"(19\d{2}|20\d{2})", url)
    if m_url:
        year = int(m_url.group(0))
        if year < YEAR_CUTOFF:
            return True

    # 2) Check date-like fields if present (e.g. "Apr 12, 2019", "2013", etc.)
    date_str = result.get("date") or result.get("date_published") or ""
    if date_str:
        m_date = re.search(r"(19\d{2}|20\d{2})", date_str)
        if m_date:
            year2 = int(m_date.group(0))
            if year2 < YEAR_CUTOFF:
                return True
        # Strings like "3 days ago", "2 weeks ago" → assumed recent → keep

    return False


def _ext_from_ct(ctype: str) -> str:
    ctype = (ctype or "").lower()
    if "png" in ctype:  return ".png"
    if "webp" in ctype: return ".webp"
    if "jpeg" in ctype or "jpg" in ctype: return ".jpg"
    if "gif" in ctype:  return ".gif"
    return ".jpg"


def _hash_url(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def _download_one(args) -> Optional[dict]:
    url, idx, total = args
    try:
        r = session.get(url, timeout=TIMEOUT, stream=True)
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "")
        ext = _ext_from_ct(ctype)
        fname = f"img_{_hash_url(url)}{ext}"
        outpath = IM_DIR / fname
        with open(outpath, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return {
            "image_path": str(outpath),
            "url": url,
            "index": idx,
            "content_type": ctype,
            "status": "ok",
        }
    except Exception as e:
        return {
            "image_path": "",
            "url": url,
            "index": idx,
            "content_type": "",
            "status": f"error: {e}",
        }


def serpapi_search_urls(query: str, target_count: int, recency: str = "w") -> list[str]:
    """
    Paginate SerpAPI Google Images results until target_count or pages exhausted.

    recency:
      'd' → past day
      'w' → past week
      'm' → past month

    We ALSO:
      - filter out obviously old images using looks_too_old(...)
      - cap results per domain to avoid over-weighting one site
    """
    urls: list[str] = []
    pages = math.ceil(target_count / PAGE_SIZE)

    for i in range(pages):
        params = {
            "engine": "google_images",
            "q": query,
            "ijn": i,
            "api_key": SERP_KEY,
            # Force Google Images recency (day/week/month)
            "tbs": f"qdr:{recency}",
        }

        for attempt in range(3):
            try:
                r = session.get(
                    "https://serpapi.com/search.json",
                    params=params,
                    timeout=TIMEOUT,
                )
                r.raise_for_status()
                data = r.json()
                results = data.get("images_results") or []
                if not results:
                    # no more results for this query
                    break

                for res in results:
                    # 🔒 HARD FILTER: drop anything that looks too old
                    if looks_too_old(res):
                        continue

                    url = (
                        res.get("original")
                        or res.get("image")
                        or res.get("link")
                        or res.get("thumbnail")
                    )
                    if not url:
                        continue

                    urls.append(url)
                    if len(urls) >= target_count:
                        break

                break  # break out of retry loop if this page succeeded

            except Exception:
                # backoff & retry
                time.sleep(0.8 * (attempt + 1))

        if len(urls) >= target_count:
            break  # enough URLs, stop paginating

    # ---- Per-domain cap + de-dupe ----
    seen_per_domain: dict[str, int] = {}
    safe_urls: list[str] = []
    seen_global: set[str] = set()

    for u in urls:
        if not u or u in seen_global:
            continue
        seen_global.add(u)
        try:
            domain = re.findall(r"https?://([^/]+)/", u)[0].lower()
        except Exception:
            domain = "unknown"
        cnt = seen_per_domain.get(domain, 0)
        if cnt < 5:  # cap e.g. 5 images per domain
            safe_urls.append(u)
            seen_per_domain[domain] = cnt + 1

    return safe_urls


def download_image_manifest(urls: list[str]) -> pd.DataFrame:
    """Download all URLs to data/latest/images + build manifest CSV."""
    tasks = [(u, i, len(urls)) for i, u in enumerate(urls)]
    rows = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for res in tqdm(ex.map(_download_one, tasks), total=len(tasks), desc="Downloading"):
            if res:
                rows.append(res)
    df = pd.DataFrame(rows)
    out = OUT_DIR / "download_manifest.csv"
    df.to_csv(out, index=False)
    return df


def kmeans_colors_for_image(img_path: Path, k: int = K_COLORS) -> list[dict]:
    arr_bgr = cv2.imread(str(img_path))
    if arr_bgr is None:
        return []
    arr = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = arr.shape
    if h * w > SAMPLE_PIXELS:
        idx = np.random.choice(h * w, SAMPLE_PIXELS, replace=False)
        sample = arr.reshape(-1, 3)[idx]
    else:
        sample = arr.reshape(-1, 3)

    Z = np.float32(sample)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.1)
    attempts = 5
    flags = cv2.KMEANS_PP_CENTERS
    compactness, labels, centers = cv2.kmeans(
        Z, k, None, criteria, attempts, flags
    )
    centers = np.clip(centers, 0, 255).astype(np.uint8)

    # share per cluster
    counts = np.bincount(labels.flatten(), minlength=k).astype(float)
    shares = counts / counts.sum()

    rows = []
    for i, (center, share) in enumerate(zip(centers, shares)):
        r, g, b = center.tolist()
        hexcode = f"#{r:02X}{g:02X}{b:02X}"
        rows.append(
            {
                "image_path": str(img_path),
                "rank": i,
                "color_hex": hexcode,
                "share": float(share),
            }
        )
    return rows


def main():
    # 1) pick queries
    sampled_queries = ALL_QUERIES[:QUERIES_PER_RUN]
    print(f"[config] QUERIES_PER_RUN={QUERIES_PER_RUN}, IMAGES_PER_QUERY={IMAGES_PER_QUERY}")
    print(f"[config] Using RECENCY_MODE={RECENCY_MODE}, YEAR_CUTOFF={YEAR_CUTOFF}")
    print("[search] Sampled queries:")
    for q in sampled_queries:
        print("   •", q)

    # 2) fetch image URLs via SerpAPI
    all_urls = []
    for q in sampled_queries:
        urls = serpapi_search_urls(q, IMAGES_PER_QUERY, recency=RECENCY_MODE)
        print(f"[search] {q} → {len(urls)} urls")
        all_urls.extend(urls)

    print(f"[search] Total URLs (before dedupe): {len(all_urls)}")

    # 3) de-duplicate URLs globally
    all_urls = list(dict.fromkeys(all_urls))
    print(f"[search] Total URLs (after dedupe): {len(all_urls)}")

    # 4) download images + manifest
    manifest = download_image_manifest(all_urls)

    # 5) color extraction for each downloaded image
    rows = []
    for p in tqdm(manifest["image_path"].dropna(), desc="Colors"):
        ipath = Path(p)
        rows.extend(kmeans_colors_for_image(ipath, k=K_COLORS))

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
        if v < 0.25: return "deep/neutrals"
        if s < 0.15: return "neutrals"
        if 15 <= h < 45: return "earthy"
        if 45 <= h < 75: return "sunlit"
        if 75 <= h < 150: return "green"
        if 150 <= h < 255: return "blue/purple"
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
            colors_df.sort_values(["image_path", "share"], ascending=[True, False])
            .groupby("image_path", as_index=False)
            .first()
        )
        summary.to_csv(OUT_DIR / "top_colors_today.csv", index=False)

    # Copy a snapshot into history with date-stamped names
    today = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    hist = HIST_DIR / today
    hist.mkdir(parents=True, exist_ok=True)

    for name in [
        "download_manifest.csv",
        "colors_opencv.csv",
        "colors_families.csv",
        "top_colors_today.csv",
    ]:
        src = OUT_DIR / name
        if src.exists():
            shutil.copyfile(src, hist / name)

    if colors_df.empty:
        print("Done. (No colors extracted — check queries/connectivity.)")
    else:
        print("Done.")

    # --- Run text metadata extractor ---
    from scripts.extract_meta import extract_folder
    extract_folder("data/articles", "data/metadata.csv")
    print("[extractor] wrote data/metadata.csv")


if __name__ == "__main__":
    main()
