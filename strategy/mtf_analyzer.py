"""Analisa Multi-Timeframe (D1, H4, H1, M15, M5) dan penyusun MTF Matrix untuk dashboard."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Sequence

import pandas as pd

from config import CONFIG, BotConfig
from strategy.layer1_macro import BiasType, infer_h4_structure_bias
from strategy.layer2_structure import SupplyDemandZone, is_price_touching_zone


@dataclass(frozen=True)
class MTFTimeframeSnapshot:
    """Ringkasan kondisi teknikal satu timeframe pada matriks MTF."""

    timeframe: str
    trend_bias: BiasType
    fresh_zones_count: int
    active_zone_touched: bool
    summary: str


@dataclass(frozen=True)
class MTFMatrixReport:
    """Laporan lengkap penyelarasan seluruh timeframe (D1 -> H4 -> H1 -> M15 -> M5)."""

    snapshots: Dict[str, MTFTimeframeSnapshot]
    aligned_count: int
    alignment_score_pct: float
    dominant_bias: BiasType
    recommended_entry_tf: str

    def to_json(self) -> str:
        payload = {
            "snapshots": {k: asdict(v) for k, v in self.snapshots.items()},
            "aligned_count": self.aligned_count,
            "alignment_score_pct": self.alignment_score_pct,
            "dominant_bias": self.dominant_bias,
            "recommended_entry_tf": self.recommended_entry_tf,
        }
        return json.dumps(payload, ensure_ascii=False)


def build_mtf_matrix(
    d1_df: pd.DataFrame,
    h4_df: pd.DataFrame,
    h1_df: pd.DataFrame,
    m15_df: pd.DataFrame,
    m5_df: pd.DataFrame,
    macro_bias: Optional[str],
    zones_h4: Sequence[SupplyDemandZone],
    zones_h1: Sequence[SupplyDemandZone],
    cfg: BotConfig = CONFIG,
) -> MTFMatrixReport:
    """Evaluasi struktur tren dan zona di seluruh timeframe (D1, H4, H1, M15, M5)."""
    cur_price = (
        float(m5_df["close"].iloc[-1])
        if not m5_df.empty
        else (float(h1_df["close"].iloc[-1]) if not h1_df.empty else 0.0)
    )
    cur_low = float(m5_df["low"].iloc[-1]) if not m5_df.empty else cur_price
    cur_high = float(m5_df["high"].iloc[-1]) if not m5_df.empty else cur_price

    frames = {
        "D1": d1_df,
        "H4": h4_df,
        "H1": h1_df,
        "M15": m15_df,
        "M5": m5_df,
    }
    snapshots: Dict[str, MTFTimeframeSnapshot] = {}

    h4_touched = any(is_price_touching_zone(z, cur_low, cur_high) for z in zones_h4)
    h1_touched = any(is_price_touching_zone(z, cur_low, cur_high) for z in zones_h1)

    for tf_name, df in frames.items():
        bias = infer_h4_structure_bias(df, pivot_window=min(cfg.pivot_window, max(2, len(df) // 6)))
        if tf_name == "H4":
            z_count, touched = len(zones_h4), h4_touched
        elif tf_name == "H1":
            z_count, touched = len(zones_h1), h1_touched
        else:
            z_count, touched = 0, False

        status_txt = f"Bias={bias}"
        if z_count > 0:
            status_txt += f" | {z_count} zona ({'TOUCHED' if touched else 'WAIT'})"
        snapshots[tf_name] = MTFTimeframeSnapshot(
            timeframe=tf_name,
            trend_bias=bias,
            fresh_zones_count=z_count,
            active_zone_touched=touched,
            summary=status_txt,
        )

    target_bias: BiasType
    if macro_bias in ("BUY", "SELL"):
        target_bias = macro_bias  # type: ignore[assignment]
    else:
        buy_votes = sum(1 for s in snapshots.values() if s.trend_bias == "BUY")
        sell_votes = sum(1 for s in snapshots.values() if s.trend_bias == "SELL")
        target_bias = "BUY" if buy_votes > sell_votes else ("SELL" if sell_votes > buy_votes else "NEUTRAL")

    aligned = (
        sum(1 for s in snapshots.values() if s.trend_bias == target_bias)
        if target_bias != "NEUTRAL"
        else 0
    )
    score = (aligned / 5.0) * 100.0
    rec_tf = "M15" if cfg.entry_timeframe == "M15" else ("M15/M5" if cfg.entry_timeframe == "M15_OR_M5" else "M5")

    return MTFMatrixReport(
        snapshots=snapshots,
        aligned_count=aligned,
        alignment_score_pct=score,
        dominant_bias=target_bias,
        recommended_entry_tf=rec_tf,
    )
