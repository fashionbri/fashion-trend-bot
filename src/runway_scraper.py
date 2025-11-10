import csv, re, time
from datetime import datetime
import requests
from tqdm import tqdm
from .config import HISTORY, SERP_API_KEY

SEARCH_ENGINE = "google"
BASE = "https://serpapi.com/search.json"

CITIES  = ["Paris","Milan","New York","London","Seoul","Tokyo"]
YEARS   = ["2024","2025"]
SEASONS = ["SS","FW"]

def _query(city, year, season):
    return f'"{city} Fashion Week" {year} {season} runway site:vogue.com OR site:wwd.com OR site:theimpression.com'

def _fetch(query, num=10):
    params = {"engine": SEARCH_ENGINE, "q": query, "api_key": SERP_API_KEY, "num": num}
    r = requests.get(BASE, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("organic_results", [])

def run():
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    outfile = HISTORY / f"runway_serp_{stamp}.csv"
    rows = []
    for city in tqdm(CITIES, desc="cities"):
        for year in YEARS:
            for season in SEASONS:
                q = _query(city, year, season)
                results = _fetch(q, num=20)
                for r in results:
                    url = r.get("link","")
                    rows.append({
                        "city": city, "year": year, "season": season,
                        "title": r.get("title",""),
                        "url": url,
                        "snippet": r.get("snippet",""),
                        "source": re.sub(r"^https?://(www\.)?", "", url).split("/")[0],
                        "scraped_at_utc": stamp
                    })
                time.sleep(0.4)

    if not rows:
        print("no results")
        return None
    with outfile.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return outfile

if __name__ == "__main__":
    run()
