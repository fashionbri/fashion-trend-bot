# =========================
# COMPACT DAILY + WEEKLY OUTPUTS
# =========================
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

_LATEST = Path("data/latest")
_HISTORY = Path("data/history")
_HISTORY.mkdir(parents=True, exist_ok=True)

def _safe_read_csv(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

def write_daily_summary():
    top_colors = _safe_read_csv(_LATEST / "top_colors_today.csv")
    color_fams = _safe_read_csv(_LATEST / "top_color_families_today.csv")
    manifest   = _safe_read_csv(_LATEST / "download_manifest.csv")
    gt = pd.DataFrame()
    gt_files = sorted(_LATEST.glob("google_trends_*.csv"))
    if gt_files:
        gt = _safe_read_csv(gt_files[-1])

    rows = []

    # Colors
    if not top_colors.empty:
        for _, r in top_colors.head(8).iterrows():
            rows.append({
                "signal": "color",
                "name": str(r.get("hex") or r.get("color") or "").strip(),
                "score": float(r.get("share", 0) or r.get("rank_share", 0)),
                "extra": "hex"
            })

    # Color families
    if not color_fams.empty:
        for _, r in color_fams.head(6).iterrows():
            rows.append({
                "signal": "color_family",
                "name": str(r.get("family") or "").strip(),
                "score": float(r.get("share", 0)),
                "extra": ""
            })

    # Google Trends
    if not gt.empty:
        try:
            latest_row = gt.iloc[-1]
            sr = latest_row.drop(labels=[c for c in gt.columns if str(c).lower() in ("date", "ispartial")], errors="ignore")
            sr = pd.to_numeric(sr, errors="coerce").sort_values(ascending=False).head(10)
            for term, val in sr.items():
                rows.append({"signal": "trend_term", "name": str(term), "score": float(val), "extra": ""})
        except Exception:
            pass

    top_signals = pd.DataFrame(rows)
    top_signals.to_csv(_LATEST / "top_signals.csv", index=False)

    # Images
    if not manifest.empty:
        cols = [c for c in manifest.columns if c in ("image_url", "source_url", "rank", "hex", "color")]
        best_imgs = manifest.sort_values(manifest.columns[0]).head(20)[cols]
        best_imgs.to_csv(_LATEST / "top_images_manifest.csv", index=False)

    today = datetime.utcnow().strftime("%Y-%m-%d")
    lines = [f"# Daily Fashion Signals — {today}\n"]

    if not top_signals.empty:
        cols = top_signals.query("signal == 'color'").head(5)["name"].tolist()
        fams = top_signals.query("signal == 'color_family'").head(3)["name"].tolist()
        terms = top_signals.query("signal == 'trend_term'").head(7)["name"].tolist()
        if cols:
            lines.append("**Top colors today**: " + ", ".join(cols))
        if fams:
            lines.append("**Leading color families**: " + ", ".join(fams))
        if terms:
            lines.append("**Google searches spiking**: " + ", ".join(terms))

    if (_LATEST / "top_images_manifest.csv").exists():
        try:
            n = len(pd.read_csv(_LATEST / "top_images_manifest.csv"))
            lines.append(f"\n**Hero images picked**: {n} (see `top_images_manifest.csv`)")
        except Exception:
            pass

    md = "\n\n".join(lines) + "\n"
    (_LATEST / "daily_summary.md").write_text(md, encoding="utf-8")
    (_HISTORY / f"summary_{datetime.utcnow():%Y%m%d}.md").write_text(md, encoding="utf-8")
    print("[summary] daily_summary.md + top_signals.csv written")

def write_weekly_roundup():
    cutoff = datetime.utcnow() - timedelta(days=7)
    trend_files = list(_LATEST.glob("google_trends_*.csv")) + list(_HISTORY.glob("google_trends_*.csv"))
    trend_frames = []

    for f in trend_files:
        try:
            date_part = f.stem.replace("google_trends_", "").split("-")[0]
            dt = datetime.strptime(date_part, "%Y%m%d")
            if dt >= cutoff:
                df = pd.read_csv(f)
                trend_frames.append(df)
        except Exception:
            continue

    top_terms = []
    if trend_frames:
        df_all = pd.concat(trend_frames, ignore_index=True)
        num = df_all.select_dtypes(include=["number"]).fillna(0)
        if not num.empty:
            weekly_mean = num.mean(numeric_only=True).sort_values(ascending=False).head(15)
            top_terms = [str(k) for k in weekly_mean.index.tolist()]

    color_files = list(_LATEST.glob("top_colors_today.csv")) + list(_HISTORY.glob("top_colors_*.csv"))
    color_frames = []
    for f in color_files:
        try:
            color_frames.append(pd.read_csv(f))
        except Exception:
            pass
    top_colors = []
    if color_frames:
        cf = pd.concat(color_frames, ignore_index=True)
        if "hex" in cf.columns:
            top_colors = cf["hex"].value_counts().head(10).index.tolist()

    weekly_csv = _LATEST / "weekly_signals.csv"
    maxlen = max(len(top_terms), len(top_colors))
    def _pad(lst, n):
        return lst + [""] * (n - len(lst))
    pd.DataFrame({
        "top_trend_terms": _pad(top_terms, maxlen),
        "top_colors": _pad(top_colors, maxlen)
    }).to_csv(weekly_csv, index=False)

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

# ---- EXECUTION ----
try:
    write_daily_summary()

    # Only make weekly roundup on Sundays (UTC)
    if datetime.utcnow().weekday() == 6:
        write_weekly_roundup()
        print("[summary] Weekly roundup generated (Sunday run)")
    else:
        print("[summary] Skipping weekly roundup (not Sunday)")
except Exception as _e:
    print("[summary] skipped due to error:", _e)
