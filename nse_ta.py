"""
nse_ta.py - Supertrend / RSI / multi-timeframe helpers for the NSE dashboard.

Conventions (match TradingView defaults):
  * ATR = Wilder RMA of True Range (seeded with SMA of the first `period` TRs)
  * Supertrend(period=4, mult=1.0) on hl2 = (High + Low) / 2
  * RSI = Wilder RMA, length 14
Frames use lower-case columns: open, high, low, close, volume and a DatetimeIndex.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TF_RULES = {"Daily": None, "Weekly": "W-FRI", "Monthly": "ME"}


# --------------------------------------------------------------- normalisation
def normalize_ohlc(raw: pd.DataFrame) -> pd.DataFrame:
    """nselib price_volume_data -> clean OHLCV frame (EQ preferred, one row per date)."""
    df = raw.copy()
    colmap = {"OpenPrice": "open", "HighPrice": "high", "LowPrice": "low",
              "ClosePrice": "close", "TotalTradedQuantity": "volume", "Series": "series",
              "Date": "date"}
    df = df.rename(columns=colmap)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False).str.strip(),
                              errors="coerce")
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="mixed", errors="coerce")
    df = df.dropna(subset=["date", "open", "high", "low", "close"])
    if "series" in df:  # prefer EQ when a symbol trades in several series on the same day
        df["_p"] = (df["series"].astype(str).str.strip() != "EQ").astype(int)
        df = df.sort_values(["date", "_p"]).drop_duplicates("date", keep="first")
    df = df.drop_duplicates("date").sort_values("date").set_index("date")
    return df[["open", "high", "low", "close", "volume"]]


def resample_ohlc(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Daily -> Weekly (W-FRI) / Monthly. Index = last actual trading day in each period."""
    rule = TF_RULES[tf]
    if rule is None:
        return df.copy()
    try:
        g = df.groupby(pd.Grouper(freq=rule))
    except Exception:  # older pandas used 'M'
        g = df.groupby(pd.Grouper(freq="M" if rule == "ME" else rule))
    out = g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                close=("close", "last"), volume=("volume", "sum"))
    last_day = df.index.to_series().groupby(pd.Grouper(freq=rule)).last()
    out["last_day"] = last_day
    out = out.dropna(subset=["open", "close"])
    out.index = pd.DatetimeIndex(out["last_day"])
    return out.drop(columns="last_day")


# ----------------------------------------------------------------- indicators
def rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder's RMA seeded with the SMA of the first n valid values."""
    v = s.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    valid = np.where(~np.isnan(v))[0]
    if len(valid) < n:
        return pd.Series(out, index=s.index)
    start = valid[0] + n - 1
    out[start] = np.nanmean(v[valid[0]:start + 1])
    for i in range(start + 1, len(v)):
        out[i] = (out[i - 1] * (n - 1) + v[i]) / n if not np.isnan(v[i]) else out[i - 1]
    return pd.Series(out, index=s.index)


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()],
                   axis=1).max(axis=1)
    tr.iloc[0] = df["high"].iloc[0] - df["low"].iloc[0]
    return rma(tr, period)


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    d = close.diff()
    up, dn = d.clip(lower=0), (-d).clip(lower=0)
    rs = rma(up, length) / rma(dn, length)
    out = 100 - 100 / (1 + rs)
    return out.where(rma(dn, length) != 0, 100.0)


def supertrend(df: pd.DataFrame, period: int = 4, mult: float = 1.0) -> pd.DataFrame:
    """Returns df copy + columns: atr, st, st_dir (+1 up / -1 down), flip (+1 buy / -1 sell / 0)."""
    out = df.copy()
    a = atr(out, period).to_numpy()
    hl2 = ((out["high"] + out["low"]) / 2).to_numpy()
    close = out["close"].to_numpy()
    n = len(out)
    ub, lb = hl2 + mult * a, hl2 - mult * a
    fub, flb = ub.copy(), lb.copy()
    st = np.full(n, np.nan)
    direction = np.zeros(n, dtype=int)
    first = int(np.argmax(~np.isnan(a))) if (~np.isnan(a)).any() else n
    for i in range(first, n):
        if i == first:
            direction[i] = 1 if close[i] >= hl2[i] else -1
        else:
            fub[i] = ub[i] if (ub[i] < fub[i - 1] or close[i - 1] > fub[i - 1]) else fub[i - 1]
            flb[i] = lb[i] if (lb[i] > flb[i - 1] or close[i - 1] < flb[i - 1]) else flb[i - 1]
            if close[i] > fub[i - 1]:
                direction[i] = 1
            elif close[i] < flb[i - 1]:
                direction[i] = -1
            else:
                direction[i] = direction[i - 1]
        st[i] = flb[i] if direction[i] == 1 else fub[i]
    out["atr"], out["st"], out["st_dir"] = a, st, direction
    prev = pd.Series(direction, index=out.index).shift(1).fillna(0).astype(int)
    flip = np.where((direction != prev.to_numpy()) & (prev.to_numpy() != 0), direction, 0)
    out["flip"] = flip
    return out


def add_indicators(df: pd.DataFrame, st_period=4, st_mult=1.0, rsi_len=14) -> pd.DataFrame:
    out = supertrend(df, st_period, st_mult)
    out["rsi"] = rsi(out["close"], rsi_len)
    return out


# ------------------------------------------------------------------ MTF logic
def build_mtf(daily: pd.DataFrame, st_period=4, st_mult=1.0, rsi_len=14) -> dict:
    """{'Daily': df, 'Weekly': df, 'Monthly': df} each with indicators."""
    return {tf: add_indicators(resample_ohlc(daily, tf), st_period, st_mult, rsi_len) for tf in TF_RULES}


def tf_summary(df: pd.DataFrame) -> dict:
    """Latest state for one timeframe."""
    v = df.dropna(subset=["st"])
    if v.empty:
        return {"trend": "n/a", "dir": 0}
    last = v.iloc[-1]
    flips = v[v["flip"] != 0]
    if len(flips):
        fd = flips.index[-1]
        bars = int(len(v.loc[fd:]) - 1)
    else:
        fd, bars = v.index[0], int(len(v) - 1)
    d = int(last["st_dir"])
    return {
        "dir": d, "trend": "Bullish" if d == 1 else "Bearish",
        "st": float(last["st"]), "close": float(last["close"]),
        "dist_pct": float((last["close"] / last["st"] - 1) * 100),
        "flip_date": fd, "bars_since_flip": bars,
        "rsi": float(last["rsi"]) if pd.notna(last["rsi"]) else float("nan"),
        "partial": False,
    }


def rsi_zone(v: float) -> str:
    if pd.isna(v):
        return "-"
    return "Overbought" if v >= 70 else "Oversold" if v <= 30 else "Bull zone" if v >= 50 else "Bear zone"


def confluence(summaries: dict) -> dict:
    dirs = [s.get("dir", 0) for s in summaries.values()]
    bull, bear = dirs.count(1), dirs.count(-1)
    if bull == 3:
        label = "Strong bullish (D+W+M aligned)"
    elif bear == 3:
        label = "Strong bearish (D+W+M aligned)"
    elif summaries["Monthly"].get("dir") == 1 and summaries["Weekly"].get("dir") == 1:
        label = "Bullish higher timeframes, daily pullback" if dirs[0] == -1 else "Bullish"
    elif summaries["Monthly"].get("dir") == -1 and summaries["Weekly"].get("dir") == -1:
        label = "Bearish higher timeframes, daily bounce" if dirs[0] == 1 else "Bearish"
    else:
        label = "Mixed / transition"
    return {"bull": bull, "bear": bear, "score": bull - bear, "label": label}


def missing_bdays(idx: pd.DatetimeIndex) -> list:
    """Business days absent from a daily index (holidays) - for plotly rangebreaks."""
    full = pd.bdate_range(idx.min(), idx.max())
    return [d.strftime("%Y-%m-%d") for d in full.difference(idx)]
