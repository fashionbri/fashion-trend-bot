import pandas as pd
from prophet import Prophet
from datetime import datetime
from .config import LATEST, MODELS

def _latest_corr():
    files = sorted(LATEST.glob("runway_retail_corr_*.csv"))
    return files[-1] if files else None

def _forecast_kw(kw, df, months=12):
    s = df[df["kw"]==kw].groupby("date")["runway_count"].sum().reset_index()
    if s.empty: return None
    s = s.rename(columns={"date":"ds","runway_count":"y"})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(s)
    future = m.make_future_dataframe(periods=months, freq="M")
    fc = m.predict(future)
    m.save((MODELS / f"prophet_{kw.replace(' ','_')}.json").as_posix())
    return fc[["ds","yhat","yhat_lower","yhat_upper"]].tail(months).assign(kw=kw)

def run():
    corr = _latest_corr()
    if not corr:
        print("no correlation file found")
        return None
    df = pd.read_csv(corr, parse_dates=["date"])
    out_rows = []
    for kw in sorted(df["kw"].dropna().unique())[:20]:
        fr = _forecast_kw(kw, df, months=6)
        if fr is not None:
            out_rows.append(fr)
    if not out_rows:
        print("no forecasts")
        return None
    out = pd.concat(out_rows)
    outfile = LATEST / f"forecast_kw_{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.csv"
    out.to_csv(outfile, index=False)
    return outfile

if __name__ == "__main__":
    print(run())
