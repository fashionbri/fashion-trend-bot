from datetime import datetime
from pytrends.request import TrendReq
import pandas as pd
from .config import LATEST

TERMS = [
    "sheer dress","metallic skirt","oversized blazer","ballet flats","kitten heels",
    "crochet top","cargo pants","leather jacket"
]

def google_trends():
    pytrend = TrendReq()
    pytrend.build_payload(TERMS, timeframe="today 5-y")
    df = pytrend.interest_over_time().drop(columns=["isPartial"])
    out = LATEST / f"google_trends_{datetime.utcnow().strftime('%Y%m%d')}.csv"
    df.to_csv(out)
    return out

if __name__ == "__main__":
    print(google_trends())
