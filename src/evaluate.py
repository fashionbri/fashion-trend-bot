"""
Time-aware evaluation for runway→retail forecasting.
- Splits by time (no leakage).
- Rolling / expanding CV.
- Metrics: hit rate, ROI proxy, CLV proxy, drawdowns, return distribution.

Input expectation:
A tidy DataFrame with columns:
  date (datetime64), kw (str), runway_count (float or int), retail_count (float or int)

We forecast runway_count (or retail_count), then evaluate directional accuracy
and return-based proxies. You can swap the model to Prophet/ARIMA easily.

Author: you
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable
from sklearn.linear_model import LinearRegression

# ---------------------------
# Utilities
# ---------------------------

def _ensure_datetime(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    if not np.issubdtype(df[col].dtype, np.datetime64):
        df = df.copy()
        df[col] = pd.to_datetime(df[col])
    return df.sort_values(col)

def pct_change_safe(a: pd.Series, periods: int = 1) -> pd.Series:
    return a.pct_change(periods=periods).replace([np.inf, -np.inf], np.nan)

def rolling_drawdown(returns: pd.Series) -> Tuple[float, float]:
    """Max drawdown and average drawdown from a return series (simple cum PnL)."""
    equity = (1 + returns.fillna(0.0)).cumprod()
    peaks = equity.cummax()
    dd = (equity / peaks) - 1.0
    return dd.min(), dd.mean()

# ---------------------------
# Splitters
# ---------------------------

@dataclass
class TemporalSplit:
    """Single holdout split by time."""
    train_end: pd.Timestamp  # inclusive
    test_start: pd.Timestamp # inclusive

def temporal_holdout_split(df: pd.DataFrame, train_frac: float = 0.7) -> TemporalSplit:
    df = _ensure_datetime(df)
    dates = df["date"].sort_values().unique()
    cut_idx = int(len(dates) * train_frac)
    cut_idx = max(1, min(cut_idx, len(dates)-1))
    train_end = pd.Timestamp(dates[cut_idx-1])
    test_start = pd.Timestamp(dates[cut_idx])
    return TemporalSplit(train_end, test_start)

def rolling_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    min_train_points: int = 12,
) -> List[TemporalSplit]:
    """Chronological rolling splits (expanding train, fixed forward test of 1 block)."""
    df = _ensure_datetime(df)
    dates = list(df["date"].sort_values().unique())
    if len(dates) < min_train_points + n_splits:
        n_splits = max(1, len(dates) - min_train_points)
    splits: List[TemporalSplit] = []
    for i in range(n_splits):
        train_end_idx = min_train_points + i - 1
        if train_end_idx >= len(dates) - 1:
            break
        splits.append(TemporalSplit(
            train_end=pd.Timestamp(dates[train_end_idx]),
            test_start=pd.Timestamp(dates[train_end_idx + 1]),
        ))
    return splits

# ---------------------------
# Simple baseline model (replaceable)
# ---------------------------

class BaselineLinear:
    """
    A tiny, transparent baseline:
    - Features: lagged target, simple moving averages, month dummies
    - Target: runway_count (or any y column you pass)
    Swap with Prophet/ARIMA/your model by keeping the same fit/predict API.
    """
    def __init__(self, lags: int = 3, sma_windows: List[int] = [3,6]):
        self.lags = lags
        self.sma_windows = sma_windows
        self.model = LinearRegression()
        self.feature_cols: List[str] = []
        self.fitted = False

    def _featurize(self, df: pd.DataFrame, y_col: str) -> pd.DataFrame:
        df = df.copy().sort_values("date")
        # Lags of target
        for L in range(1, self.lags + 1):
            df[f"{y_col}_lag{L}"] = df[y_col].shift(L)
        # SMAs
        for w in self.sma_windows:
            df[f"{y_col}_sma{w}"] = df[y_col].rolling(w).mean()
        # Month seasonality
        df["month"] = df["date"].dt.month.astype("category")
        dummies = pd.get_dummies(df["month"], prefix="m", drop_first=True)
        df = pd.concat([df.drop(columns=["month"]), dummies], axis=1)
        return df

    def fit(self, df: pd.DataFrame, y_col: str) -> "BaselineLinear":
        df = self._featurize(df, y_col).dropna()
        self.feature_cols = [c for c in df.columns if c not in ["date", y_col, "kw"]]
        X = df[self.feature_cols].values
        y = df[y_col].values
        if len(df) < 3:
            # not enough data; mark as not fitted
            self.fitted = False
            return self
        self.model.fit(X, y)
        self.fitted = True
        return self

    def predict(self, df: pd.DataFrame, y_col: str) -> pd.Series:
        if not self.fitted:
            return pd.Series(index=df.index, dtype=float)
        dfF = self._featurize(df, y_col)
        X = dfF[self.feature_cols].fillna(method="ffill").fillna(method="bfill").values
        yhat = self.model.predict(X)
        return pd.Series(yhat, index=dfF.index)

# ---------------------------
# Metrics
# ---------------------------

def metric_hit_rate(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Directional accuracy: sign of Δy next step vs sign of predicted Δ."""
    dy = y_true.diff()
    dyp = pd.Series(y_pred, index=y_true.index).diff()
    mask = dy.notna() & dyp.notna()
    if mask.sum() == 0:
        return np.nan
    return (np.sign(dy[mask]) == np.sign(dyp[mask])).mean()

def metric_roi_proxy(y_true: pd.Series, y_pred: pd.Series, k: float = 1.0) -> float:
    """
    Simple ROI proxy:
    - Go long when predicted next change > 0, short when < 0.
    - Return ≈ k * actual pct change next step.
    """
    ret = pct_change_safe(y_true).shift(-1)   # next-step realized return
    signal = np.sign(pd.Series(y_pred, index=y_true.index).diff())
    pnl = k * signal * ret
    return pnl.mean(skipna=True)

def metric_clv_proxy(y_true: pd.Series, y_pred: pd.Series, lookahead: int = 1) -> float:
    """
    CLV (closing line value) proxy:
    How much better was our predicted 'line' vs realized move?
    Here: (predicted next Δ) - (realized next Δ), averaged.
    """
    next_real = y_true.diff().shift(-lookahead)
    next_pred = pd.Series(y_pred, index=y_true.index).diff().shift(-lookahead)
    diff = next_pred - next_real
    return diff.mean(skipna=True)

def metric_drawdowns(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    ret = pct_change_safe(y_true).shift(-1)
    signal = np.sign(pd.Series(y_pred, index=y_true.index).diff())
    pnl = signal * ret
    max_dd, avg_dd = rolling_drawdown(pnl)
    return {"max_drawdown": float(max_dd), "avg_drawdown": float(avg_dd)}

def metric_return_distribution(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    ret = pct_change_safe(y_true).shift(-1)
    signal = np.sign(pd.Series(y_pred, index=y_true.index).diff())
    pnl = (signal * ret).dropna()
    if pnl.empty:
        return {"mean": np.nan, "std": np.nan, "skew": np.nan, "p05": np.nan, "p95": np.nan}
    return {
        "mean": float(pnl.mean()),
        "std": float(pnl.std(ddof=1)),
        "skew": float(((pnl - pnl.mean())**3).mean() / (pnl.std(ddof=0)**3 + 1e-12)),
        "p05": float(np.percentile(pnl, 5)),
        "p95": float(np.percentile(pnl, 95)),
    }

# ---------------------------
# Evaluation Pipelines
# ---------------------------

@dataclass
class EvalResult:
    split: TemporalSplit
    hit_rate: float
    roi_proxy: float
    clv_proxy: float
    max_drawdown: float
    avg_drawdown: float
    dist: Dict[str, float]
    n_test_points: int

def fit_predict_on_split(
    df_kw: pd.DataFrame,
    split: TemporalSplit,
    y_col: str = "runway_count",
    model_factory: Callable[[], BaselineLinear] = lambda: BaselineLinear()
) -> EvalResult:
    df_kw = _ensure_datetime(df_kw)
    train = df_kw[df_kw["date"] <= split.train_end]
    test = df_kw[df_kw["date"] >= split.test_start]
    model = model_factory().fit(train[["date", y_col]], y_col=y_col)

    # For evaluation, we predict on the combined set to generate contiguous signals.
    combined = pd.concat([train, test], ignore_in_
