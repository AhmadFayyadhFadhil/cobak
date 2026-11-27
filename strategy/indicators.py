"""Indikator teknikal murni (anti look-ahead): ATR(14), Pivot N-bar terkonfirmasi,
dan rolling correlation log-return XAUUSD vs DXY.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np
import pandas as pd

PivotKind = Literal["HIGH", "LOW"]


@dataclass(frozen=True)
class PivotPoint:
    """Titik swing yang baru sah dipakai setelah window kanan tertutup penuh."""

    index: int
    confirmed_index: int
    kind: PivotKind
    price: float
    time_utc: pd.Timestamp


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Hitung Average True Range menggunakan rata-rata sederhana pada True Range."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    prev_close = df["close"].astype(float).shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()


def detect_confirmed_pivots(df: pd.DataFrame, window: int = 5) -> List[PivotPoint]:
    """Deteksi Pivot High/Low tanpa look-ahead bias.

    Pivot pada indeks `k` hanya valid apabila seluruh `window` bar setelahnya
    (`k + 1 .. k + window`) sudah selesai tertutup di `df`.
    """
    n = len(df)
    if n < (2 * window + 1):
        return []

    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    times = df["time_utc"].tolist()
    pivots: List[PivotPoint] = []

    # Vektorisasi rolling max/min agar 100x lebih cepat untuk backtest 10-20 tahun
    s_high = pd.Series(highs)
    s_low = pd.Series(lows)
    left_max = s_high.shift(1).rolling(window).max().to_numpy()
    right_max = s_high.iloc[::-1].shift(1).rolling(window).max().iloc[::-1].to_numpy()
    left_min = s_low.shift(1).rolling(window).min().to_numpy()
    right_min = s_low.iloc[::-1].shift(1).rolling(window).min().iloc[::-1].to_numpy()

    # Batas atas `k` memastikan bar konfirmasi `k + window` sudah ada di slice `df`
    for k in range(window, n - window):
        if highs[k] > left_max[k] and highs[k] >= right_max[k]:
            pivots.append(
                PivotPoint(
                    index=k,
                    confirmed_index=k + window,
                    kind="HIGH",
                    price=float(highs[k]),
                    time_utc=times[k],
                )
            )
        if lows[k] < left_min[k] and lows[k] <= right_min[k]:
            pivots.append(
                PivotPoint(
                    index=k,
                    confirmed_index=k + window,
                    kind="LOW",
                    price=float(lows[k]),
                    time_utc=times[k],
                )
            )

    return pivots


def compute_rolling_correlation(
    xau_h4: pd.DataFrame,
    dxy_h4: pd.DataFrame,
    window: int = 30,
) -> Optional[float]:
    """Hitung korelasi Pearson terbaru antara log-return XAUUSD dan DXY pada H4."""
    if xau_h4.empty or dxy_h4.empty:
        return None

    left = xau_h4[["time_utc", "close"]].drop_duplicates("time_utc").rename(columns={"close": "xau"})
    right = dxy_h4[["time_utc", "close"]].drop_duplicates("time_utc").rename(columns={"close": "dxy"})
    merged = left.merge(right, on="time_utc", how="inner").sort_values("time_utc")

    if len(merged) < window + 1:
        return None

    xau_ret = np.log(np.clip(merged["xau"].to_numpy(dtype=float), 1e-9, None))
    dxy_ret = np.log(np.clip(merged["dxy"].to_numpy(dtype=float), 1e-9, None))
    rx = pd.Series(np.diff(xau_ret))
    ry = pd.Series(np.diff(dxy_ret))

    tail_x = rx.iloc[-window:]
    tail_y = ry.iloc[-window:]
    if float(tail_x.std()) == 0.0 or float(tail_y.std()) == 0.0:
        return None

    corr = float(tail_x.corr(tail_y))
    return None if np.isnan(corr) else corr
