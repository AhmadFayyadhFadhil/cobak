"""Layer 2: Validasi Break of Structure (BOS) searah bias Layer 1 dan deteksi zona
Supply/Demand fresh menggunakan Pivot High/Low objektif (tanpa look-ahead).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence

import numpy as np
import pandas as pd

from strategy.indicators import detect_confirmed_pivots

Direction = Literal["BUY", "SELL"]
TimeframeLabel = Literal["D1", "H4", "H1"]


@dataclass(frozen=True)
class SupplyDemandZone:
    """Zona Supply atau Demand pada H4/H1 beserta status kesegarannya."""

    zone_id: str
    timeframe: TimeframeLabel
    direction: Direction
    low: float
    high: float
    formed_at_utc: str
    is_fresh: bool


def detect_bos_and_fresh_zones(
    df: pd.DataFrame,
    timeframe: TimeframeLabel,
    bias: Direction,
    pivot_window: int = 5,
) -> List[SupplyDemandZone]:
    """Deteksi BOS yang searah bias lalu ekstrak zona SnD yang belum termitigasi."""
    if df.empty or len(df) < (2 * pivot_window + 3):
        return []

    pivots = detect_confirmed_pivots(df, window=pivot_window)
    closes = df["close"].to_numpy(dtype=float)
    opens = df["open"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    times = df["time_utc"].tolist()

    zones: List[SupplyDemandZone] = []
    target_kind = "HIGH" if bias == "BUY" else "LOW"
    relevant_pivots = [p for p in pivots if p.kind == target_kind]

    for pv in relevant_pivots:
        # BOS baru sah dievaluasi mulai bar setelah pivot dikonfirmasi (`pv.confirmed_index + 1`)
        bos_idx: Optional[int] = None
        for i in range(pv.confirmed_index + 1, len(df)):
            if bias == "BUY" and closes[i] > pv.price:
                bos_idx = i
                break
            if bias == "SELL" and closes[i] < pv.price:
                bos_idx = i
                break

        if bos_idx is None:
            continue

        # Cari candle base terakhir berlawanan warna sebelum impulsive BOS di antara [pv.index, bos_idx)
        base_idx = pv.index
        for j in range(bos_idx - 1, max(-1, pv.index - 1), -1):
            if bias == "BUY" and closes[j] < opens[j]:
                base_idx = j
                break
            if bias == "SELL" and closes[j] > opens[j]:
                base_idx = j
                break

        z_low = float(lows[base_idx])
        z_high = float(highs[base_idx])

        # Zona dianggap tidak fresh jika sudah disentuh sebelum bar terakhir atau ditembus close
        is_fresh = True
        if bos_idx + 1 < len(df) - 1:
            mid_closes = closes[bos_idx + 1 : -1]
            if bias == "BUY":
                mid_lows = lows[bos_idx + 1 : -1]
                if bool(np.any(mid_closes < z_low) or np.any(mid_lows <= z_high)):
                    is_fresh = False
            else:
                mid_highs = highs[bos_idx + 1 : -1]
                if bool(np.any(mid_closes > z_high) or np.any(mid_highs >= z_low)):
                    is_fresh = False

        # Cek invalidasi di bar paling akhir (jika close langsung menembus ekstrem zona)
        if is_fresh:
            last_close = float(closes[-1])
            if (bias == "BUY" and last_close < z_low) or (
                bias == "SELL" and last_close > z_high
            ):
                is_fresh = False

        if is_fresh:
            ts_str = pd.Timestamp(times[base_idx]).isoformat()
            zones.append(
                SupplyDemandZone(
                    zone_id=f"{timeframe}_{bias}_{int(pd.Timestamp(times[base_idx]).timestamp())}",
                    timeframe=timeframe,
                    direction=bias,
                    low=z_low,
                    high=z_high,
                    formed_at_utc=ts_str,
                    is_fresh=True,
                )
            )

    return zones


def is_price_touching_zone(
    zone: SupplyDemandZone,
    price_low: float,
    price_high: float,
) -> bool:
    """True jika rentang candle menyentuh atau berada di dalam zona SnD."""
    return price_high >= zone.low and price_low <= zone.high


def select_priority_touched_zone(
    zones: Sequence[SupplyDemandZone],
    current_price: float,
    bias: Direction,
    bar_low: Optional[float] = None,
    bar_high: Optional[float] = None,
) -> Optional[SupplyDemandZone]:
    """Pilih tepat 1 zona aktif saat beberapa zona tersentuh: H4 > H1, lalu terbaru."""
    lo = current_price if bar_low is None else min(current_price, bar_low)
    hi = current_price if bar_high is None else max(current_price, bar_high)

    touched = [
        z
        for z in zones
        if z.is_fresh and z.direction == bias and is_price_touching_zone(z, lo, hi)
    ]
    if not touched:
        return None

    def _sort_key(z: SupplyDemandZone) -> tuple[int, float, float]:
        tf_rank = {"D1": 0, "H4": 1, "H1": 2}.get(z.timeframe, 3)
        formed_ts = pd.Timestamp(z.formed_at_utc).timestamp()
        dist = abs(((z.low + z.high) / 2.0) - current_price)
        return (tf_rank, -formed_ts, dist)

    touched.sort(key=_sort_key)
    return touched[0]
