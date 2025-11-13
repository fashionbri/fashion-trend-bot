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


def _ensure_top_color_tables() -> None:
    """
    If top_colors_today.csv / top_color_families_today.csv are missing or empty,
    build them from colors_opencv.csv and colors_families.csv.
    """
    top_colors_path = _LATEST / "top_colors_today.csv"
    top_fams_path = _LATEST / "top_color_families_today.csv"

    top_colors = _safe_read_csv(top_colors_path)
    top_fams = _safe_read_csv(top_fams_path)

    colors_all = _safe_read_csv(_LATEST / "colors_opencv.csv")
    fams_all = _safe_read_csv(_LATEST / "colors_families.csv")

    # Build top_colors_today.csv if needed
    if (top_colors.empty or "hex" not in top_colors.columns) and not colors_all.empty:
        if "hex" in colors_all.columns:
            if "share" in colors_all.columns:
                agg = (
                    colors_all.groupby("hex")["share"]
                    .sum()
                    .reset_index()
                    .sort_values("share", ascending=False)
                )
            else:
                agg = (
                    colors_all["hex"]
                    .value_counts()
                    .reset_index()
                    .rename(columns={"index": "hex", "hex": "share"})
                )
            agg.to_csv(top_colors_path, index=False)
            top_colors = agg

    # Build top_color_families_today.csv if needed
    if (top_fams.empty or "family" not in top_fams.columns) and not fams_all.empty:
        if "family" in fams_all.columns:
            if "share" in fams_all.columns:
                fam_agg = (
                    fams_all.groupby("family")["share"]
                    .sum()
                    .reset_index()
                    .sort_values("share", ascending=False)
                )
            else:
                fam_agg = (
                    fams_all["family"]
                    .value_counts()
                    .reset_index()
                    .rename(columns={"index": "family", "family": "share"})
                )
            fam_agg.to_csv(top_fams_path, index=False)
            top_fams = fam_agg

    # just write files + return
    return


def write_daily_summary():
    # make sure the top_* tables exist
    _ensure_top_color_tables()

    top_colors = _safe_read_csv(_LATEST / "top_colors_today.csv")
    color_fams = _safe_read_csv(_LATEST / "top_color_families_today.csv")
    manifest = _safe_read_csv(_LATEST / "download_manifest.csv")
    gt = pd.DataFrame()
    gt_files = sorted(_LATEST.glob("google_trends_*.csv"))
    if gt_files:
        gt = _safe_read_csv(gt_files[-1])

    rows = []

    # Colors
    if not top_colors.empty:
        # normalize names just in case
        if "hex" not in top_colors.columns:
            if "color_hex" in top_colors.columns:
                top_colors = top_colors.rename(columns={"color_hex": "hex"})
            elif "color" in top_colors.columns:
                top_colors = top_colors.rename(columns={"color": "hex"})
        if "share" not in top_colors.columns and "count" in top_colors.columns:
            top_colors = top_colors.rename(columns={"count": "share"})

        for _, r in top_colors.head(8).iterrows():
            rows.append({
                "signal": "color",
                "name": str(r.get("hex") or "").strip(),
                "score": float(r.get("share", 0) or r.get("rank_share", 0) or 0),
                "extra": "hex"
            })

    # Color families
    if not color_fams.empty:
        if "family" not in color_fams.columns and "color_family" in color_fams.columns:
            color_fams = color_fams.rename(columns={"color_family": "family"})
        if "share" not in color_fams.columns and "count" in color_fams.columns:
            color_fams = color_fams.rename(columns={"count": "share"})

        for _, r in color_fams.head(6).iterrows():
            rows.append({
                "signal": "color_family",
                "name": str(r.get("family") or "").strip(),
                "score": float(r.get("share", 0) or 0),
                "extra": ""
            })

    # Google Trends
    if not gt.empty:
        try:
            latest_row = gt.iloc[-1]
            sr = latest_row.drop(
                labels=[c for c in gt.columns if str(c).lower() in ("date", "ispartial")],
                errors="ignore"
            )
            sr = pd.to_numeric(sr, errors="coerce").sort_values(ascending=False).head(10)
            for term, val in sr.items():
                rows.append({
                    "signal": "trend_term",
                    "name": str(term),
                    "score": float(val),
                    "extra": ""
                })
        except Exception:
            pass

    top_signals = pd.DataFrame(rows)
    top_signals.to_csv(_LATEST / "top_signals.csv", index=False)

    # Images
    if not manifest.empty:
        cols = [c for c in manifest.columns if c in ("image_url", "source_url", "rank", "hex", "color")]
        if cols:
            best_imgs = manifest.sort_values(manifest.columns[0]).head(20)[cols]
            best_imgs.to_csv(_LATEST / "top_images_manifest.csv", index=False)

    today = datetime.utcnow().strftime("%Y-%m-%d")
    lines = [f"# Daily Fashion Signals — {today}\n"]

    if not top_signals.empty:
        cols = top_signals.query("signal == 'color'").head(5)["name"].tolist()
        fams = top_signals.query("signal == 'color_family'").head(3)["name"].tolist()
        terms = top_signals.query("signal == 'trend_term'").head(7)["na]()
