"""
End-to-end pipeline runner for the Fashion Trend Bot.

Steps:
1) Scrape runway SERP → data/history/runway_serp_*.csv
2) Pull retail signals (Google Trends) → data/latest/google_trends_*.csv
3) Correlate runway ↔ retail → data/latest/runway_retail_corr_*.csv
4) Forecast per keyword → data/latest/forecast_kw_*.csv  and models/*
5) (Optional) Evaluate with time-aware CV → data/latest/eval_*.csv
"""

# Ensure the repository root is on sys.path so "src" can be imported
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
import pandas as pd

# --- Import pipeline modules ---
from src.runway_scraper import run as scrape_runway
from src.retail_tracker import google_trends
from src.correlate import run as correlate
from src.forecast import run as forecast

# Evaluation is optional—will be skipped if module isn't present
try:
    from src.evaluate import run_evaluation_pipeline
    HAVE_EVAL = True
except Exception:
    HAVE_EVAL = False

# Folders from config (your processed folder is data/latest)
from src.config import LATEST

def run_eval():
    """Run time-aware evaluation if evaluation module is available."""
    if not HAVE_EVAL:
        print("5) Evaluation: module not found, skipping.")
        return

    latest_corr = sorted(LATEST.glob("runway_retail_corr_*.csv"))
    if not latest_corr:
        print("5) Evaluation: no correlation file found, skipping.")
        return

    df = pd.read_csv(latest_corr[-1], parse_dates=["date"])
    needed = ["date", "kw", "runway_count"]
    if not all(c in df.columns for c in needed):
        print("5) Evaluation: needed columns missing, skipping.")
        return

    keep = [c for c in ["date", "kw", "runway_count", "retail_count"] if c in df.columns]
    df = df[keep].dropna(subset=["date", "kw", "runway_count"])

    holdout, cv = run_evaluation_pipeline(df)

    out1 = LATEST / "eval_holdout_latest.csv"
    out2 = LATEST / "eval_cv_summary_latest.csv"
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
from src.visualize import visualize_correlations, visualize_forecasts

print("6) Generating visuals…")
visualize_correlations()
visualize_forecasts()
# ==== NEW: Weekly roundup generator ====
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path

LATEST = Path("data/latest")
HISTORY = Path("data/history")
HISTORY.mkdir(parents=True, exist_ok=True)

def weekly_roundup():
    cutoff = datetime.utcnow() - timedelta(days=7)
    rows = []

    # Collect all google_trends CSVs in latest or history
    trend_files = list(LATEST.glob("google_trends_*.csv")) + list(HISTORY.glob("google_trends_*.csv"))
    for f in trend_files:
        try:
            date_str = f.stem.split("_")[-1].split("-")[0]
            dt = datetime.strptime(date_str, "%Y%m%d")
            if dt >= cutoff:
                df = pd.read_csv(f)
                if "date" in df.columns:
                    df["__date"] = pd.to_datetime(df["date"], errors="coerce")
                rows.append(df)
        except Exception:
            continue

    if not rows:
        print("[weekly] No Google Trends files from last 7 days found.")
        return

    df_all = pd.concat(rows, ignore_index=True)
    df_all = df_all.select_dtypes(include=["number"]).fillna(0)

    # Compute weekly average interest per term
    weekly_mean = df_all.mean(numeric_only=True).sort_values(ascending=False).head(15)
    top_terms = weekly_mean.index.tolist()

    # Top colors (aggregate past week)
    color_files = list(LATEST.glob("top_colors_today.csv")) + list(HISTORY.glob("top_colors_*.csv"))
    color_rows = []
    for f in color_files:
        try:
            date_str = f.stem.split("_")[-1].split("-")[0]
            dt = datetime.strptime(date_str, "%Y%m%d")
            if dt >= cutoff:
                color_rows.append(pd.read_csv(f))
        except Exception:
            continue
    colors = pd.concat(color_rows, ignore_index=True) if color_rows else pd.DataFrame()
    top_colors = []
    if not colors.empty:
        if "hex" in colors.columns:
            c = colors["hex"].value_counts().head(8)
            top_colors = c.index.tolist()

    # --- write outputs ---
    out_csv = LATEST / "weekly_signals.csv"
    pd.DataFrame({"top_trend_terms": top_terms, "top_colors": top_colors[:len(top_terms)]}).to_csv(out_csv, index=False)

    # Markdown summary
    today = datetime.utcnow().strftime("%Y-%m-%d")
    md_lines = [
        f"# Weekly Fashion Roundup — Week Ending {today}",
        "",
        "**Top Google Trend Terms:** " + ", ".join(top_terms),
    ]
    if top_colors:
        md_lines.append("**Most Frequent Colors Extracted:** " + ", ".join(top_colors))
    md = "\n\n".join(md_lines)

    (LATEST / "weekly_roundup.md").write_text(md, encoding="utf-8")
    (HISTORY / f"weekly_roundup_{datetime.utcnow():%Y%m%d}.md").write_text(md, encoding="utf-8")
    print("[weekly] roundup complete")

# call it after your main run
if __name__ == "__main__":
    weekly_roundup()

