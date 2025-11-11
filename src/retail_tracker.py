# src/retail_tracker.py
from datetime import datetime
from typing import List
import time, random, json, re
from pathlib import Path
from collections import Counter

import pandas as pd
from pytrends.request import TrendReq
from pytrends.exceptions import ResponseError, TooManyRequestsError
from .config import LATEST

# =======================
# Tunables
# =======================
SEED_TERMS_PATH = "data/seed_terms_1000.txt"     # <-- 1000-term file lives here
TERMS_CACHE_PATH = LATEST / "google_trends_terms_cache.json"

DAILY_TERMS_TARGET = 120      # how many terms to hit per run (rotation sample)
BATCH_SIZE = 3                # pytrends supports up to 5; keep 3-4 to avoid 429s
BASE_SLEEP = 4.0
MAX_ATTEMPTS_PER_BATCH = 8
MAX_NEW_TERMS_FROM_RELATED = 400   # growth per run from related_queries()

# One shared TrendReq (urllib3 compat: do NOT pass retries/backoff_factor)
pytrend = TrendReq(hl="en-US", tz=0, timeout=(10, 30))

# =======================
# Helpers
# =======================
def load_terms(path: str = SEED_TERMS_PATH) -> List[str]:
    p = Path(path)
    if not p.exists():
        print(f"[terms] Seed file missing at {path}. Using minimal fallback list.")
        return [
            "sheer dress", "metallic skirt", "oversized blazer", "ballet flats",
            "kitten heels", "crochet top", "cargo pants", "leather jacket"
        ]
    return [t.strip() for t in p.read_text(encoding="utf-8").splitlines() if t.strip()]

def _chunks(lst: List[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def _build_with_retry(pytrend: TrendReq, terms: List[str]) -> bool:
    """
    Build payload with robust retry/backoff to survive 429s on CI.
    """
    for attempt in range(1, MAX_ATTEMPTS_PER_BATCH + 1):
        try:
            pytrend.build_payload(
                kw_list=terms,
                timeframe="today 5-y",
                geo="",     # global (set "US" if you prefer)
                gprop=""    # web search
            )
            return True
        except (TooManyRequestsError, ResponseError) as e:
            sleep_s = (BASE_SLEEP * (2 ** (attempt - 1))) + random.uniform(0, 1.25)
            print(f"[pytrends] build_payload failed (attempt {attempt}/{MAX_ATTEMPTS_PER_BATCH}): {e}. Sleeping {sleep_s:.1f}s")
            time.sleep(min(sleep_s, 60))
        except Exception as e:
            sleep_s = BASE_SLEEP + random.uniform(0, 1.0)
            print(f"[pytrends] unexpected error: {e}. Sleeping {sleep_s:.1f}s")
            time.sleep(sleep_s)
    return False

# -------- term growth helpers --------
_slug = re.compile(r"\s+")
def _norm(t: str) -> str:
    return _slug.sub(" ", t.strip().lower())

def _load_terms_cache() -> dict:
    if Path(TERMS_CACHE_PATH).exists():
        try:
            return json.loads(Path(TERMS_CACHE_PATH).read_text(encoding="utf-8"))
        except Exception:
            pass
    # default structure
    return {"seeds": [], "expanded": []}

def _save_terms_cache(cache: dict):
    TERMS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    Path(TERMS_CACHE_PATH).write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

def _score_related_bucket(bucket: dict) -> list[str]:
    """
    bucket:
      {'top': [{'query': 'x', 'value': 100}, ...],
       'rising': [{'query': 'y', 'value': 1200}, ...]}
    Return ranked terms.
    """
    c = Counter()
    for k in ("top", "rising"):
        rows = bucket.get(k) or []
        for r in rows:
            q = _norm(str(r.get("query", "")))
            if not q or len(q) < 3:
                continue
            val = r.get("value", 0) or 0
            # rising can be huge; compress contribution
            c[q] += 1 + min(int(val), 100)
    # prune obvious generic noise
    banned = {"fashion", "style", "outfit", "trend", "runway", "2024", "2025", "2026"}
    for b in banned:
        if b in c:
            del c[b]
    return [q for q, _ in c.most_common()]

def _expand_terms_via_pytrends(pytrend: TrendReq, seeds: list[str]) -> list[str]:
    found = []
    for seed in seeds:
        ok = _build_with_retry(pytrend, [seed])
        if not ok:
            continue
        try:
            rq = pytrend.related_queries()
            for _seed_key, bucket in (rq or {}).items():
                if not bucket:
                    continue
                ranked = _score_related_bucket(bucket)
                found.extend(ranked)
            # be polite
            time.sleep(0.6 + random.uniform(0, 0.6))
        except Exception as e:
            print(f"[pytrends] related_queries error for '{seed}': {e}")
            time.sleep(0.8 + random.uniform(0, 1.0))
    # normalize & dedupe
    normed, seen = [], set()
    for t in found:
        n = _norm(t)
        if n and n not in seen:
            seen.add(n)
            normed.append(n)
    return normed

def _rotating_sample(terms: list[str], k: int) -> list[str]:
    """
    Deterministic daily sample so different slices run each day.
    """
    if not terms:
        return []
    seed = int(datetime.utcnow().strftime("%Y%m%d"))
    rng = random.Random(seed)
    terms = terms[:]  # copy
    rng.shuffle(terms)
    return terms[:k]

# =======================
# Main
# =======================
def google_trends():
    """
    Pull interest_over_time for dynamically-grown term universe in small batches and merge.
    Writes: data/latest/google_trends_YYYYMMDD.csv
    Always writes a CSV (may be placeholder) so downstream steps won’t crash.
    Also caches the 'today' CSV and reuses it if it already exists and is non-trivial.
    """
    # --- cache check (today) ---
    today_out = LATEST / f"google_trends_{datetime.utcnow():%Y%m%d}.csv"
    if today_out.exists() and today_out.stat().st_size > 50:
        print("[pytrends] using cached today file:", today_out)
        return today_out

    # --- load seeds from file; merge with cache; expand; rotate ---
    file_seeds = [_norm(t) for t in load_terms()]
    cache = _load_terms_cache()

    # Initialize seeds in cache if empty; always ensure file seeds are included
    seeds = list(dict.fromkeys((cache.get("seeds") or []) + file_seeds))
    expanded_existing = [_norm(t) for t in cache.get("expanded", [])]

    # Grow from related queries (limit seed count here for speed)
    new_candidates = _expand_terms_via_pytrends(pytrend, seeds[:50])
    expanded_set = set(expanded_existing)
    fresh = [t for t in new_candidates if t not in expanded_set and t not in seeds]
    if fresh:
        expanded_existing.extend(fresh[:MAX_NEW_TERMS_FROM_RELATED])

    # Persist cache
    cache["seeds"] = seeds
    cache["expanded"] = expanded_existing
    _save_terms_cache(cache)

    # Build today's working list and rotate
    universe = list(dict.fromkeys(seeds + expanded_existing))
    working_terms = _rotating_sample(universe, DAILY_TERMS_TARGET)
    if not working_terms:
        working_terms = universe[:DAILY_TERMS_TARGET]

    print(f"[pytrends] seeds={len(seeds)} expanded={len(expanded_existing)} today={len(working_terms)}")

    # --- fetch series in batches ---
    frames = []
    for batch in _chunks(working_terms, BATCH_SIZE):
        ok = _build_with_retry(pytrend, batch)
        if not ok:
            print(f"[pytrends] Giving up on batch: {batch}")
            continue

        # request data with retry if 429 happens here
        for attempt in range(1, MAX_ATTEMPTS_PER_BATCH + 1):
            try:
                df = pytrend.interest_over_time()
                if isinstance(df, pd.DataFrame) and not df.empty:
                    if "isPartial" in df.columns:
                        df = df.drop(columns=["isPartial"])
                    frames.append(df)
                break
            except (TooManyRequestsError, ResponseError) as e:
                sleep_s = (BASE_SLEEP * (2 ** (attempt - 1))) + random.uniform(0, 1.25)
                print(f"[pytrends] interest_over_time failed (attempt {attempt}/{MAX_ATTEMPTS_PER_BATCH}): {e}. Sleeping {sleep_s:.1f}s")
                time.sleep(min(sleep_s, 60))
            except Exception as e:
                sleep_s = BASE_SLEEP + random.uniform(0, 1.0)
                print(f"[pytrends] unexpected error in interest_over_time: {e}. Sleeping {sleep_s:.1f}s")
                time.sleep(sleep_s)

        # small pause between batches to be polite
        time.sleep(1.5 + random.uniform(0, 0.75))

    # --- write result (always write something so later steps can proceed) ---
    out = today_out
    if not frames:
        print("[pytrends] No frames collected; writing placeholder CSV.")
        pd.DataFrame(columns=["date"]).to_csv(out, index=False)
        return out

    out_df = frames[0].reset_index().rename(columns={"date": "date"})
    for f in frames[1:]:
        m = f.reset_index().rename(columns={"date": "date"})
        out_df = pd.merge(out_df, m, on="date", how="outer")

    out_df = out_df.sort_values("date")
    out_df.to_csv(out, index=False)
    return out

if __name__ == "__main__":
    print(google_trends())
