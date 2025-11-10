# src/forecast.py
import pandas as pd
from prophet import Prophet
from datetime import datetime
from .config import LATEST, MODELS

MIN_ROWS = 2  # require at least 2 points to fit a model

def _latest_corr():
    files = sorted(LATEST.glob("runway_retail_corr_*.csv"))
    return files[-1] if files else None

def _prep_series(df, kw):
    s = df[df["kw"] == kw].groupby("date")["runway_count"].sum().reset_index()
    s = s.dropna(subset=["date", "runway_count"])
    # Prophet expects columns ds (datetime) and y (float)
    s = s.rename(columns={"date": "ds", "runway_count": "y"})
    s["ds"] = pd.to_datetime(s["ds"])
    s = s.sort_values("ds")
    return s

def _forecast_kw(kw, df, months=6):
    s = _prep_series(df, kw)
    if len(s) < MIN_ROWS:
        # not enough data → skip gracefully
        return None
    try:
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.fit(s)
        future = m.make_future_dataframe(periods=months, freq="M")
        fc = m.predict(future)
        # save model (json path as str to avoid pathlib issues)
        m.save((MODELS / f"prophet_{kw.replace(' ','_')}.json").as_posix())
        out = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(months).copy()
        out["kw"] = kw
        return out
    except Exception as e:
        # if prophet fails for any specific kw, skip it
        print(f"[forecast] skipping '{kw}' due to error: {e}")
        return None

def run():
    corr = _latest_corr()
    if not corr:
        print("[forecast] no correlation file found")
        return None
    df = pd.read_csv(corr, parse_dates=["date"])
    if df.empty or "kw" not in df.columns or "runway_count" not in df.columns:
        print("[forecast] correlation file missing required columns or empty")
        return None

    outputs = []
    # limit to a reasonable number of kws (adjust if you want)
    for kw in sorted(df["kw"].dropna().unique()):
        fr = _forecast_kw(kw, df, months=6)
        if fr is not None and not fr.empty:
            outputs.append(fr)

    if not outputs:
        print("[forecast] no valid forecasts produced (insufficient data)")
        return None

    out_df = pd.concat(outputs, ignore_index=True)
    outfile = LATEST / f"forecast_kw_{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.csv"
    out_df.to_csv(outfile, index=False)
    return outfile

if __name__ == "__main__":
    print(run())
