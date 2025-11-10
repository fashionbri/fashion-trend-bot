"""
End-to-end pipeline runner for the Fashion Trend Bot.

Steps:
1) Scrape runway SERP → data/history/runway_serp_*.csv
2) Pull retail signals (Google Trends) → data/latest/google_trends_*.csv
3) Correlate runway ↔ retail → data/latest/runway_retail_corr_*.csv
4) Forecast per keyword → data/latest/forecast_kw_*.csv  and models/*
5) (Optional) Evaluate with time-aware CV → data/latest/eval_*.csv
"""

from pathlib import Path

# --- Import pipeline modules ---
# Make sure these files exist in src/
from src.runway_scraper import run as scrape_runway
from src.retail_tracker import google_trends
from src.correlate import run as correlate
from src.forecast import run as forecast

# Evaluation is optional—comment out if you haven't added src/evaluate.py yet.
try:
    from src.evaluate import run_evaluation_pipeline
    HAVE_EVAL = True
except Exception:
    HAVE_EVAL = False

# If you used the config.py I gave you, these paths will already be created.
from src.config import PROCESSED

import pandas as pd

def run_eval():
    """Run time-aware evaluation if evaluation module is available."""
    if not HAVE_EVAL:
        print("5) Evaluation: module not found, skipping.")
        return
    latest_corr = sorted(PROCESSED.glob("runway_retail_corr_*.csv"))
    if not latest_corr:
        print("5) Evaluation: no correlation file found, skipping.")
        return
    df = pd.read_csv(latest_corr[-1], parse_dates=["date"])
    keep = [c for c in ["date", "kw", "runway_count", "retail_count"] if c in df.columns]
    if not keep:
        print("5) Evaluation: needed columns missing, skipping.")
        return
    df = df[keep].dropna(subset=["date", "kw", "runway_count"])

    holdout, cv = run_evaluation_pipeline(df)
    out1 = PROCESSED / "eval_holdout_latest.csv"
    out2 = PROCESSED / "eval_cv_summary_latest.csv"
    holdout.to_csv(out1, index=False)
    cv.to_csv(out2, index=False)
    print(f"   → {out1}")
    print(f"   → {out2}")

def main():
    print("1) Scraping runway SERP…")
    f1 = scrape_runway()
    print(f"   → {f1}")

    print("2) Pulling Google Trends…")
    f2 = google_trends()
    print(f"   → {f2}")

    print("3) Correlating runway ↔ retail…")
    f3 = correlate()
    print(f"   → {f3}")

    print("4) Forecasting…")
    f4 = forecast()
    print(f"   → {f4}")

    print("5) Evaluating (optional)…")
    run_eval()

if __name__ == "__main__":
    main()
