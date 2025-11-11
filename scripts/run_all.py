

# =========================
# COMPACT DAILY + WEEKLY OUTPUTS
# (Append this entire block at the end of scripts/run_all.py)
# =========================
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Resolve paths (do not rely on other modules here)
_LATEST = Path("data/latest")
_HISTORY = Path("data/history")
_HISTORY.mkdir(parents=True, exist_ok=True)

def _safe_read_csv(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

def write_daily_summary():
    """
    Produce small, human-usable daily outputs:
      - data/latest/daily_summary.md
      - data/latest/top_signals.csv
      - data/latest/top_images_manifest.csv
      - data/history/summary_YYYYMMDD.md (archive)
    """
    top_colors = _safe_read_csv(_LATEST / "top_colors_today.csv")
    color_fams = _safe_read_csv(_LATEST / "top_color_families_today.csv")
    manifest   = _safe_read_csv(_LATEST / "download_manifest.csv")

    # latest google trends file (if present)
    gt = pd.DataFrame()
    gt_candidates = sorted(_LATEST.glob("google_trends_*.csv"))
    if gt_candidates:
        gt = _safe_read_csv(gt_candidates[-1])

    # ---- Build compact signals table ----
    rows = []

    # Colors (top 8)
    if not top_colors.empty:
        for _, r in top_colors.head(8).iterrows():
            rows.append({
                "signal": "color",
                "name": str(r.get("hex") or r.get("color") or "").strip(),
                "score": float(r.get("share", 0) or r.get("rank_share", 0)),
                "extra": "hex"
            })

    # Color families (top 6)
    if not color_fams.empty:
        for _, r in color_fams.head(6).iterrows():
            rows.append({
                "signal": "color_family",
                "name": str(r.get("family") or "").strip(),
                "score": float(r.get("share", 0)),
                "extra": ""
            })

    # Google Trends (top 10 terms on latest row)
    if not gt.empty:
        try:
            latest_row = gt.iloc[-1]
            sr = latest_row.drop(labels=[c for c in gt.columns if str(c).lower() in ("date", "ispartial")], errors="ignore")
            sr = pd.to_numeric(sr, errors="coerce")
            sr = sr.sort_values(ascending=False).head(10)
            for term, val in sr.items():
                rows.append({"signal": "trend_term", "name": str(term), "score": float(val), "extra": ""})
        except Exception:
            pass

    top_signals = pd.DataFrame(rows)
    top_signals_out = _LATEST / "top_signals.csv"
    top_signals.to_csv(top_signals_out, index=False)

    # ---- Pick the best 20 images with their dominant color (if present) ----
    if not manifest.empty:
        cols = [c for c in manifest.columns if c in ("image_url", "source_url", "rank", "hex", "color")]
        if not cols:
            cols = list(manifest.columns[:6])
        best_imgs = manifest.sort_values(manifest.columns[0]).head(20)[cols]
        best_imgs.to_csv(_LATEST / "top_images_manifest.csv", index=False)

    # ---- Write a tiny human summary (markdown) ----
    today = datetime.utcnow().strftime("%Y-%m-%d")
    lines = [f"# Daily Fashion Signals — {today}\n"]

    if not top_signals.empty:
        try:
            cols = (top_signals.query("signal == 'color'")
                               .sort_values("score", ascending=False)
                               .head(5)["name"].tolist())
        except Exception:
            cols = []
        try:
            fams = (top_signals.query("signal == 'color_family'")
                               .sort_values("score", ascending=False)
                               .head(3)["name"].tolist())
        except Exception:
            fams = []
        try:
            terms = (top_signals.query("signal == 'trend_term'")
                                .sort_values("score", ascending=False)
                                .head(7)["name"].tolist())
        except Exception:
            terms = []

        if cols:
            lines.append("**Top colors today**: " + ", ".join(cols))
        if fams:
            lines.append("**Leading color families**: " + ", ".join(fams))
        if terms:
            lines.append("**Google searches spiking**: " + ", ".join(terms))

    # If we wrote a top images manifest above, mention count
    tim = _LATEST / "top_images_manifest.csv"
    if tim.exists():
        try:
            n = len(pd.read_csv(tim))
            lines.append(f"\n**Hero images picked**: {n} (see `top_images_manifest.csv`)")
        except Exception:
            pass

    md = "\n\n".join(lines) + "\n"
    (_LATEST / "daily_summary.md").write_text(md, encoding="utf-8")
    (_HISTORY / f"summary_{datetime.utcnow():%Y%m%d}.md").write_text(md, encoding="utf-8")
    print("[summary] daily_summary.md + top_signals.csv written")

def write_weekly_roundup():
    """
    Aggregate the last 7 days into:
      - data/latest/weekly_roundup.md
      - data/latest/weekly_signals.csv
      - data/history/weekly_roundup_YYYYMMDD.md
    Works even if some days or files are missing.
    """
    cutoff = datetime.utcnow() - timedelta(days=7)

    # Collect GT files from latest/history
    trend_files = list(_LATEST.glob("google_trends_*.csv")) + list(_HISTORY.glob("google_trends_*.csv"))
    trend_frames = []
    for f in trend_files:
        try:
            # filenames: google_trends_YYYYMMDD[ -HHMMSS].csv
            stem = f.stem
            chunk = stem.replace("google_trends_", "")
            date_part = chunk.split("-")[0]
            dt = datetime.strptime(date_part, "%Y%m%d")
            if dt >= cutoff:
                df = pd.read_csv(f)
                trend_frames.append(df)
        except Exception:
            continue

    top_terms = []
    if trend_frames:
        df_all = pd.concat(trend_frames, ignore_index=True)
        # keep numeric columns only (trend series)
        num = df_all.select_dtypes(include=["number"]).fillna(0)
        if not num.empty:
            weekly_mean = num.mean(numeric_only=True).sort_values(ascending=False).head(15)
            top_terms = [str(k) for k in weekly_mean.index.tolist()]

    # Weekly colors: try to union daily top_colors_today over the week
    color_files = list(_LATEST.glob("top_colors_today.csv")) + list(_HISTORY.glob("top_colors_*.csv"))
    color_frames = []
    for f in color_files:
        # If your history naming scheme doesn't include dates for colors, we still try latest
        try:
            # when file name has no date, include anyway (best-effort)
            color_frames.append(pd.read_csv(f))
        except Exception:
            pass
    top_colors = []
    if color_frames:
        cf = pd.concat(color_frames, ignore_index=True)
        if "hex" in cf.columns:
            top_colors = cf["hex"].value_counts().head(10).index.tolist()

    # Write compact CSV
    weekly_csv = _LATEST / "weekly_signals.csv"
    # pad shorter list so CSV aligns
    maxlen = max(len(top_terms), len(top_colors))
    def _pad(lst, n): 
        lst = lst[:]
        while len(lst) < n:
            lst.append("")
        return lst
    pd.DataFrame({
        "top_trend_terms": _pad(top_terms, maxlen),
        "top_colors": _pad(top_colors, maxlen)
    }).to_csv(weekly_csv, index=False)

    # Markdown roundup
    today = datetime.utcnow().strftime("%Y-%m-%d")
    md_lines = [
        f"# Weekly Fashion Roundup — Week Ending {today}",
        "",
        ("**Top Google Trend Terms:** " + ", ".join(top_terms)) if top_terms else "**Top Google Trend Terms:** (no data)",
        ("**Most Frequent Colors Extracted:** " + ", ".join(top_colors)) if top_colors else "**Most Frequent Colors Extracted:** (no data)",
        "",
        "_Files: `weekly_signals.csv` for charts_",
    ]
    md_text = "\n".join(md_lines) + "\n"
    (_LATEST / "weekly_roundup.md").write_text(md_text, encoding="utf-8")
    (_HISTORY / f"weekly_roundup_{datetime.utcnow():%Y%m%d}.md").write_text(md_text, encoding="utf-8")
    print("[summary] weekly_roundup.md + weekly_signals.csv written")

# ---- CALLS ----
# Call these after your pipeline has produced its usual CSVs.
# If your file already has a "main" section, just add these two calls
# at the very end of the run (once per run).
try:
    write_daily_summary()
    write_weekly_roundup()
except Exception as _e:
    print("[summary] skipped due to error:", _e)
