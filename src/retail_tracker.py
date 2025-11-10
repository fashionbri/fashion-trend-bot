# src/retail_tracker.py
from datetime import datetime
from pathlib import Path
from typing import List
import time
import pandas as pd
from pytrends.request import TrendReq
from pytrends.exceptions import ResponseError
from .config import LATEST

TERMS = [
    "sheer dress","metallic skirt","oversized blazer","ballet flats","kitten heels",
    "crochet top","cargo pants","leather jacket"
]

def _chunks(lst: List[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def _safe_build(pytrend: TrendReq, terms: List[str], retries: int = 3, wait: float = 2.0):
    """
    Build payload with retries to avoid transient 400/429 errors on CI.
    """
    last_err = None
    for attempt in range(1, retries+1):
        try:
            pytrend.build_payload(
                kw_list=terms,
                timeframe="today 5-y",
                geo="",            # global; set "US" if you prefer
                gprop="",          # web search
            )
            return True
        except ResponseError as e:
            last_err = e
            time.sleep(wait * attempt)  # simple backoff
    if last_err:
        raise last_err
    return False

def google_trends():
    """
    Pull interest_over_time for TERMS in small batches and merge.
    Writes: data/latest/google_trends_YYYYMMDD.csv
    """
    # Locale + tz help stabilize CI runs
pytrend = TrendReq(hl="en-US", tz=0, requests_args={"timeout": 30})
    frames = []

    for batch in _chunks(TERMS, 5):   # <= 5 terms per call is safer
        _safe_build(pytrend, batch, retries=4, wait=2.0)
        df = pytrend.interest_over_time()
        if "isPartial" in df.columns:
            df = df.drop(columns=["isPartial"])
        frames.append(df)

        # polite pause to avoid throttling
        time.sleep(1.5)

    if not frames:
        # No data; write an empty CSV so downstream doesn’t crash
        out = LATEST / f"google_trends_{datetime.utcnow().strftime('%Y%m%d')}.csv"
        pd.DataFrame(columns=["date"]).to_csv(out, index=False)
        return out

    # Merge on date (outer to be safe), deduplicate columns
    out_df = frames[0].reset_index().rename(columns={"date": "date"})
    for f in frames[1:]:
        m = f.reset_index().rename(columns={"date": "date"})
        out_df = pd.merge(out_df, m, on="date", how="outer")

    # sort by date for consistency
    out_df = out_df.sort_values("date")

    out = LATEST / f"google_trends_{datetime.utcnow().strftime('%Y%m%d')}.csv"
    out_df.to_csv(out, index=False)
    return out

if __name__ == "__main__":
    print(google_trends())
