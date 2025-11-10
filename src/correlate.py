import pandas as pd
from pathlib import Path
from datetime import datetime
from .config import HISTORY, LATEST
from .text_features import extract_keywords

def _latest_serp():
    files = sorted(HISTORY.glob("runway_serp_*.csv"))
    return files[-1] if files else None

def run():
    serp = _latest_serp()
    if not serp:
        print("no serp file found")
        return None
    df = pd.read_csv(serp)
    df["kw"] = (df["title"].fillna("") + " " + df["snippet"].fillna("")).map(extract_keywords)
    df = df.explode("kw").dropna(subset=["kw"])

    # aggregate by (year, kw) and create a mid-year date
    grp = df.groupby(["year","kw"]).size().reset_index(name="runway_count")
    grp["date"] = pd.to_datetime(grp["year"].astype(str) + "-06-01")
    grp = grp[["date","kw","runway_count"]]

    # bring in latest google trends
    trends_files = sorted(LATEST.glob("google_trends_*.csv"))
    retail = pd.read_csv(trends_files[-1], parse_dates=["date"]) if trends_files else None
    if retail is not None:
        tlong = retail.melt(id_vars=["date"], var_name="term", value_name="trend_score")
        tlong["kw"] = tlong["term"].str.replace(r"(dress|skirt|jacket|top|pants)$", "", regex=True).str.strip()
        merged = grp.merge(tlong, on=["date","kw"], how="left")
    else:
        merged = grp.assign(trend_score=pd.NA)

    out = LATEST / f"runway_retail_corr_{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.csv"
    merged.to_csv(out, index=False)
    return out

if __name__ == "__main__":
    print(run())
