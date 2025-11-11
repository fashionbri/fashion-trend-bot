# src/retail_tracker.py
from datetime import datetime
from typing import List
import time, random
import pandas as pd
from pytrends.request import TrendReq
from pytrends.exceptions import ResponseError, TooManyRequestsError
from .config import LATEST

# ===== Config =====
TERMS = [
    "sheer dress", "metallic skirt", "oversized blazer", "ballet flats", "kitten heels",
    "crochet top", "cargo pants", "leather jacket"
]

BATCH_SIZE = 2
BASE_SLEEP = 4.0
MAX_ATTEMPTS_PER_BATCH = 8

# One shared TrendReq with HTTP-level retries/backoff.
pytrend = TrendReq(hl="en-US", tz=0, timeout=(10, 30), retries=8, backoff_factor=4)

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
            # exponential backoff with jitter
            sleep_s = (BASE_SLEEP * (2 ** (attempt - 1))) + random.uniform(0, 1.25)
            print(f"[pytrends] build_payload failed (attempt {attempt}/{MAX_ATTEMPTS_PER_BATCH}): {e}. Sleeping {sleep_s:.1f}s")
            time.sleep(min(sleep_s, 60))
        except Exception as e:
            # unexpected error — short sleep & retry
            sleep_s = BASE_SLEEP + random.uniform(0, 1.0)
            print(f"[pytrends] unexpected error: {e}. Sleeping {sleep_s:.1f}s")
            time.sleep(sleep_s)
    return False

def google_trends():
    """
    Pull interest_over_time for TERMS in small batches and merge.
    Writes: data/latest/google_trends_YYYYMMDD.csv
    Always writes a CSV (may be placeholder) so downstream steps won’t crash.
    Also caches the 'today' CSV and reuses it if it already exists and is non-trivial.
    """
    # --- cache check (today) ---
    today_out = LATEST / f"google_trends_{datetime.utcnow():%Y%m%d}.csv"
    if today_out.exists() and today_out.stat().st_size > 50:
        print("[pytrends] using cached today file:", today_out)
        return today_out

    frames = []

    for batch in _chunks(TERMS, BATCH_SIZE):
        ok = _build_with_retry(pytrend, batch)
        if not ok:
            print(f"[pytrends] Giving up on batch: {batch}")
            continue

        # request data with retry if 429 happens here
        for attempt in range(1, MAX_ATTEMPTS_PER_BATCH + 1):
            try:
                df = pytrend.interest_over_time()
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

    # Always write something so later steps can proceed
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
