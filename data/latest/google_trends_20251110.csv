"""
Google Trends Module
Used by the Fashion Forecast Bot to track retail search interest over time.
Includes caching, retry logic, and exponential backoff to handle rate limits.
"""

from datetime import datetime
import time
import random
from pathlib import Path
import pandas as pd
from pytrends.request import TrendReq
from .config import LATEST

# -------- SETTINGS --------
BATCH_SIZE = 2            # small batch size prevents Google 429 errors
BASE_SLEEP = 4.0          # seconds between retries (base delay)
MAX_ATTEMPTS_PER_BATCH = 8


def google_trends() -> Path:
    """
    Pull Google Trends data for predefined fashion/retail keywords.
    Saves to data/latest/google_trends_YYYYMMDD.csv
    Returns Path to the output file.
    """

    # ---- SAME-DAY CACHE ----
    today_out = LATEST / f"google_trends_{datetime.utcnow():%Y%m%d}.csv"
    if today_out.exists() and today_out.stat().st_size > 50:
        print(f"[google_trends] using cached today file: {today_out}")
        return today_out

    # ---- Initialize pytrends client ----
    pytrend = TrendReq(
        hl="en-US",
        tz=0,
        timeout=(10, 30),   # (connect, read)
        retries=8,
        backoff_factor=4
    )

    # ---- Keywords to track ----
    keywords = [
        "sheer", "neon", "quiet luxury", "ballet flats", "maxi dress",
        "slouchy blazer", "pleated skirt", "metallic shoes", "crochet dress",
        "lace tights", "shearling coat", "utility jacket", "mesh top",
        "cargo skirt", "denim maxi skirt", "silver bag", "off shoulder top",
        "leather trench", "statement earring", "micro mini skirt"
    ]

    print(f"[google_trends] loaded {len(keywords)} keywords")
    frames = []

    # ---- Fetch in small batches ----
    for i in range(0, len(keywords), BATCH_SIZE):
        batch = keywords[i:i + BATCH_SIZE]
        attempts = 0

        while attempts < MAX_ATTEMPTS_PER_BATCH:
            try:
                pytrend.build_payload(
                    kw_list=batch,
                    timeframe="today 5-y",
                    geo="US",      # use 'US' to avoid global throttling
                    gprop=""
                )

                df = pytrend.interest_over_time()

                if not df.empty:
                    df = df.reset_index().rename(columns={"date": "date"})
                    if "isPartial" in df.columns:
                        df = df.drop(columns=["isPartial"])
                    frames.append(df)

                # short polite pause between successful batches
                time.sleep(BASE_SLEEP + random.uniform(0.5, 1.5))
                break  # success → next batch

            except Exception as e:
                attempts += 1
                wait = BASE_SLEEP * (2 ** (attempts - 1)) + random.uniform(0.5, 2.0)
                print(f"[google_trends] batch {batch} failed ({attempts}/{MAX_ATTEMPTS_PER_BATCH}): {e} → sleep {wait:.1f}s")
                time.sleep(min(wait, 60.0))

    # ---- Save output ----
    if frames:
        out = pd.concat(frames, axis=0).sort_values("date")
        out = out.groupby("date", as_index=False).max(numeric_only=True)
        out.to_csv(today_out, index=False)
        print(f"[google_trends] saved → {today_out}")
    else:
        pd.DataFrame({"date": []}).to_csv(today_out, index=False)
        print(f"[google_trends] no data collected; wrote placeholder: {today_out}")

    return today_out
