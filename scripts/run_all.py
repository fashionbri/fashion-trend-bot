# scripts/run_all.py
# ======================================================
# PIPELINE ORCHESTRATOR FOR THE FASHION TREND BOT
# ======================================================

from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import subprocess
import sys
import traceback

ROOT = Path(".")
LATEST = ROOT / "data/latest"
HISTORY = ROOT / "data/history"
HISTORY.mkdir(parents=True, exist_ok=True)

# ----------------------
# Safe CSV loader
# ----------------------
def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


# ======================================================
# 1) Ensure normalized top color tables exist
# ======================================================
def _ensure_top_color_tables():
    """
    Convert ANY existing color output format into unified:
        top_colors_today.csv      → hex, share
        top_color_families_today.csv → family, share
    """
    # try to find any colors file for today
    today = datetime.utcnow().strftime("%Y%m%d")
    possible = list(LATEST.glob(f"colors_{today}.csv"))

    if not possible:
        return  # nothing to normalize

    df = pd.concat([_safe_read_csv(p) for p in possible], ignore_index=True)

    # Normalize column names
    df = df.rename(
        columns={
            "color_hex": "hex",
            "color": "hex",
            "color_family": "family",
            "count": "share",
            "rank_share": "share",
        }
    )

    # COLOR TABLE
    top_colors = df[["hex", "share"]].dropna().copy()
    top_colors = top_colors.groupby("hex", as_index=False)["share"].sum()
    top_colors = top_colors.sort_values("share", ascending=False)
    top_colors.to_csv(LATEST / "top_colors_today.csv", index=False)

    # FAMILY TABLE
    if "family" in df.columns:
        fam = df[["family", "share"]].dropna().copy()
        fam = fam.groupby("family", as_index=False)["share"].sum()
        fam = fam.sort_values("share", ascending=False)
        fam.to_csv(LATEST / "top_color_families_today.csv", index=False)


# ======================================================
# 2) DAILY SUMMARY WITH EDITORIAL SENTENCE
# ======================================================
def write_daily_summary():
    _ensure_top_color_tables()

    top_colors = _safe_read_csv(LATEST / "top_colors_today.csv")
    color_fams = _safe_read_csv(LATEST / "top_color_families_today.csv")
    manifest = _safe_read_csv(LATEST / "download_manifest.csv")
    hybrid = _safe_read_csv(LATEST / "hybrid_tags.csv")

    # Load Google Trends if exists
    gt = pd.DataFrame()
    gt_files = sorted(LATEST.glob("google_trends_*.csv"))
    if gt_files:
        gt = _safe_read_csv(gt_files[-1])

    rows = []

    # -------------------------
    # Colors
    # -------------------------
    if not top_colors.empty:
        # ensure normalized
        if "hex" not in top_colors.columns:
            if "color_hex" in top_colors.columns:
                top_colors = top_colors.rename(columns={"color_hex": "hex"})
            elif "color" in top_colors.columns:
                top_colors = top_colors.rename(columns={"color": "hex"})

        if "share" not in top_colors.columns and "count" in top_colors.columns:
            top_colors = top_colors.rename(columns={"count": "share"})

        for _, r in top_colors.head(8).iterrows():
            rows.append(
                {
                    "signal": "color",
                    "name": str(r.get("hex") or "").strip(),
                    "score": float(r.get("share", 0)),
                    "extra": "hex",
                }
            )

    # -------------------------
    # Color families
    # -------------------------
    if not color_fams.empty:
        if "family" not in color_fams.columns and "color_family" in color_fams.columns:
            color_fams = color_fams.rename(columns={"color_family": "family"})
        if "share" not in color_fams.columns and "count" in color_fams.columns:
            color_fams = color_fams.rename(columns={"count": "share"})

        for _, r in color_fams.head(6).iterrows():
            rows.append(
                {
                    "signal": "color_family",
                    "name": str(r.get("family") or "").strip(),
                    "score": float(r.get("share", 0)),
                    "extra": "",
                }
            )

    # -------------------------
    # Google Trends
    # -------------------------
    if not gt.empty:
        try:
            latest_row = gt.iloc[-1]
            sr = latest_row.drop(
                labels=[c for c in gt.columns if str(c).lower() in ("date", "ispartial")],
                errors="ignore",
            )
            sr = pd.to_numeric(sr, errors="coerce").sort_values(ascending=False).head(10)
            for term, val in sr.items():
                rows.append(
                    {
                        "signal": "trend_term",
                        "name": str(term),
                        "score": float(val),
                        "extra": "",
                    }
                )
        except Exception:
            pass

    top_signals = pd.DataFrame(rows)
    top_signals.to_csv(LATEST / "top_signals.csv", index=False)

    # ======================================================
    # Build Markdown summary
    # ======================================================
    today = datetime.utcnow().strftime("%Y-%m-%d")
    lines = [f"# Daily Fashion Signals — {today}\n"]

    if not top_signals.empty:
        cols = top_signals.query("signal == 'color'").head(5)["name"].tolist()
        fams = top_signals.query("signal == 'color_family'").head(3)["name"].tolist()
        terms = top_signals.query("signal == 'trend_term'").head(7)["name"].tolist()

        if cols:
            lines.append("**Top colors today:** " + ", ".join(cols))
        if fams:
            lines.append("**Leading color families:** " + ", ".join(fams))
        if terms:
            lines.append("**Google searches spiking:** " + ", ".join(terms))

    # -------------------------
    # NEW Editorial Line from Hybrid Tags
    # -------------------------
    if not hybrid.empty:

        def _top_term(sub: str):
            subdf = hybrid[hybrid["group"].str.contains(sub, case=False, na=False)]
            if subdf.empty:
                return None
            return subdf["term"].value_counts().index[0]

        top_garment = _top_term("garment") or _top_term("silhouette")
        top_fabric = _top_term("fabric") or _top_term("texture")
        top_mood = _top_term("mood")
        top_motif = _top_term("motif")
        top_geo = _top_term("geography")

        bits = []

        if top_garment and top_fabric:
            bits.append(f"{top_garment} in {top_fabric}")
        elif top_garment:
            bits.append(top_garment)
        elif top_fabric:
            bits.append(top_fabric)

        if top_mood:
            bits.append(f"with a {top_mood} feel")
        if top_motif:
            bits.append(f"marked by {top_motif}")

        if bits:
            sentence = "Today’s read: " + ", ".join(bits)
            if top_geo:
                sentence += f", most visible in {top_geo}."
            else:
                sentence += "."
            lines.append("\n" + sentence)

    # -------------------------
    # Images (hero manifest)
    # -------------------------
    if not manifest.empty:
        cols = [c for c in manifest.columns if c in ("image_url", "source_url", "rank", "hex", "color")]
        if cols:
            best_imgs = manifest.sort_values(manifest.columns[0]).head(20)[cols]
            best_imgs.to_csv(LATEST / "top_images_manifest.csv", index=False)

            try:
                n = len(best_imgs)
                lines.append(f"\n**Hero images picked:** {n} (see `top_images_manifest.csv`)")
            except:
                pass

    md = "\n\n".join(lines) + "\n"
    (LATEST / "daily_summary.md").write_text(md, encoding="utf-8")
    (HISTORY / f"summary_{datetime.utcnow():%Y%m%d}.md").write_text(md, encoding="utf-8")

    print("[summary] daily_summary.md + top_signals.csv written")


# ======================================================
# 3) WEEKLY ROUNDUP
# ======================================================
def write_weekly_roundup():
    cutoff = datetime.utcnow() - timedelta(days=7)

    # Gather Trends
    trend_files = list(LATEST.glob("google_trends_*.csv")) + list(HISTORY.glob("google_trends_*.csv"))
    trend_frames = []
    for f in trend_files:
        try:
            date_part = f.stem.replace("google_trends_", "").split("-")[0]
            dt = datetime.strptime(date_part, "%Y%m%d")
            if dt >= cutoff:
                trend_frames.append(pd.read_csv(f))
        except Exception:
            pass

    top_terms = []
    if trend_frames:
        df_all = pd.concat(trend_frames, ignore_index=True)
        num = df_all.select_dtypes(include=["number"]).fillna(0)
        if not num.empty:
            top_terms = (
                num.mean().sort_values(ascending=False).head(15).index.tolist()
            )

    # Gather Colors
    color_files = list(LATEST.glob("top_colors_today.csv")) + list(HISTORY.glob("top_colors_*.csv"))
    color_frames = []
    for f in color_files:
        try:
            color_frames.append(pd.read_csv(f))
        except:
            pass

    top_colors = []
    if color_frames:
        cf = pd.concat(color_frames, ignore_index=True)
        if "hex" in cf.columns:
            top_colors = cf["hex"].value_counts().head(10).index.tolist()

    # Export CSV
    weekly_csv = LATEST / "weekly_signals.csv"
    maxlen = max(len(top_terms), len(top_colors))

    def _pad(lst, n):
        return lst + [""] * (n - len(lst))

    pd.DataFrame(
        {
            "top_trend_terms": _pad(top_terms, maxlen),
            "top_colors": _pad(top_colors, maxlen),
        }
    ).to_csv(weekly_csv, index=False)

    # Markdown summary
    today = datetime.utcnow().strftime("%Y-%m-%d")
    md_lines = [
        f"# Weekly Fashion Roundup — Week Ending {today}",
        "",
        "**Top Google Trend Terms:** " + (", ".join(top_terms) if top_terms else "(no data)"),
        "**Most Frequent Colors Extracted:** " + (", ".join(top_colors) if top_colors else "(no data)"),
        "",
        "_Files: `weekly_signals.csv` for charts_",
    ]
    md_text = "\n".join(md_lines) + "\n"

    (LATEST / "weekly_roundup.md").write_text(md_text, encoding="utf-8")
    (HISTORY / f"weekly_roundup_{datetime.utcnow():%Y%m%d}.md").write_text(md_text, encoding="utf-8")

    print("[summary] weekly_roundup.md + weekly_signals.csv written")


# ======================================================
# 4) MAIN PIPELINE RUNNER
# ======================================================
if __name__ == "__main__":
    try:
        # --------------------------------------------------
        # Stage 1: Search scraper
        # --------------------------------------------------
        try:
            subprocess.run([sys.executable, "scripts/search_and_update.py"], check=True)
        except Exception:
            print("[WARN] search_and_update.py failed")
            traceback.print_exc()

        # --------------------------------------------------
        # Stage 2: Text Features
        # --------------------------------------------------
        try:
            subprocess.run([sys.executable, "scripts/run_extract.py"], check=True)
        except Exception:
            print("[WARN] run_extract.py failed (text extraction)")
            traceback.print_exc()

        # --------------------------------------------------
        # Stage 3: Image Embeddings
        # --------------------------------------------------
        try:
            import src.image_embeddings as IE
            IE.build_image_embeddings()
        except Exception:
            print("[WARN] image_embeddings failed")
            traceback.print_exc()

        # --------------------------------------------------
        # Stage 4: Hybrid Tagging
        # --------------------------------------------------
        try:
            import src.hybrid_tagger as HT
            HT.run_hybrid()
        except Exception:
            print("[WARN] hybrid tagging failed")
            traceback.print_exc()

        # --------------------------------------------------
        # Stage 5: Daily Summary
        # --------------------------------------------------
        write_daily_summary()

        # Weekly only runs Sunday
        if datetime.utcnow().weekday() == 6:
            write_weekly_roundup()

    except Exception as e:
        print("[FATAL ERROR in run_all.py]:", e)
        traceback.print_exc()
