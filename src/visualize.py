"""
Visualization module for Fashion Trend Bot.
Generates interactive charts using Plotly.
"""

import pandas as pd
import plotly.express as px
from pathlib import Path

DATA = Path("data/latest")
OUT = Path("data/latest")

def visualize_correlations():
    """Plot runway vs retail correlation results."""
    files = sorted(DATA.glob("runway_retail_corr_*.csv"))
    if not files:
        print("No correlation file found.")
        return
    df = pd.read_csv(files[-1])
    if "kw" not in df.columns:
        print("No 'kw' column in correlation file.")
        return

    fig = px.scatter(
        df,
        x="runway_count",
        y="retail_count",
        color="kw",
        title="Runway vs Retail Correlations",
        hover_data=["kw"],
        template="plotly_white"
    )
    fig.update_layout(height=600, width=800)
    out_html = OUT / "correlation_visual.html"
    fig.write_html(out_html)
    print(f"✅ Saved interactive chart → {out_html}")

def visualize_forecasts():
    """Plot keyword forecast results."""
    files = sorted(DATA.glob("forecast_kw_*.csv"))
    if not files:
        print("No forecast file found.")
        return
    df = pd.read_csv(files[-1])
    if not {"ds", "yhat", "kw"}.issubset(df.columns):
        print("Forecast CSV missing expected columns.")
        return

    fig = px.line(
        df,
        x="ds",
        y="yhat",
        color="kw",
        title="6-Month Trend Forecasts",
        template="plotly_white"
    )
    fig.update_layout(height=600, width=800)
    out_html = OUT / "forecast_visual.html"
    fig.write_html(out_html)
    print(f"✅ Saved interactive chart → {out_html}")
