"""Provider data DXY: mendukung simbol broker langsung maupun DXY sintetis ICE
dengan fallback 5-pair (tanpa USDSEK) dan inner-join timestamp alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from config import CONFIG, BotConfig, get_symbol_aliases_map
from core.mt5_client import MT5Client


@dataclass(frozen=True)
class DXYSeriesResult:
    """Hasil penyediaan data DXY beserta metadata sumbernya."""

    df: pd.DataFrame
    source_mode: str  # "BROKER_SYMBOL" | "SYNTHETIC_ICE_6" | "SYNTHETIC_FALLBACK_5" | "UNAVAILABLE"
    used_symbols: Tuple[str, ...]
    missing_symbols: Tuple[str, ...]


def compute_synthetic_dxy_from_frames(
    pair_frames: Mapping[str, pd.DataFrame],
    cfg: BotConfig = CONFIG,
) -> DXYSeriesResult:
    """Bangun OHLC DXY sintetis dari dictionary DataFrame pair mayor.

    Semua pair diselaraskan dengan inner-join pada kolom `time_utc` agar bar hilang
    tidak menggeser korelasi. Jika USDSEK tidak tersedia, bobot 5 pair tersisa
    di-renormalisasi proporsional sehingga korelasi log-return dan struktur tren
    tetap hampir identik.
    """
    weights_map: Dict[str, float] = dict(cfg.ice_dxy_weights)
    available = [pair for pair in weights_map if pair in pair_frames and not pair_frames[pair].empty]
    missing = tuple(pair for pair in weights_map if pair not in available)

    # Minimal butuh 5 pair inti (EURUSD, USDJPY, GBPUSD, USDCAD, USDCHF)
    core_required = ("EURUSD", "USDJPY", "GBPUSD", "USDCAD", "USDCHF")
    if not all(pair in available for pair in core_required):
        return DXYSeriesResult(
            df=pd.DataFrame(),
            source_mode="UNAVAILABLE",
            used_symbols=tuple(available),
            missing_symbols=missing,
        )

    # Re-normalisasi skala bobot bila USDSEK absen (bobot USDSEK = 0.042 -> sum abs tetap terjaga)
    total_abs_all = sum(abs(w) for w in weights_map.values())
    total_abs_avail = sum(abs(weights_map[p]) for p in available)
    scale = total_abs_all / total_abs_avail if total_abs_avail > 0 else 1.0
    effective_weights = {p: weights_map[p] * scale for p in available}

    # Inner-join alignment berdasarkan time_utc
    aligned: Optional[pd.DataFrame] = None
    for pair in available:
        sub = pair_frames[pair][["time_utc", "open", "high", "low", "close"]].copy()
        sub = sub.drop_duplicates(subset=["time_utc"]).rename(
            columns={
                "open": f"{pair}_open",
                "high": f"{pair}_high",
                "low": f"{pair}_low",
                "close": f"{pair}_close",
            }
        )
        aligned = sub if aligned is None else aligned.merge(sub, on="time_utc", how="inner")

    if aligned is None or aligned.empty:
        return DXYSeriesResult(
            df=pd.DataFrame(),
            source_mode="UNAVAILABLE",
            used_symbols=tuple(available),
            missing_symbols=missing,
        )

    aligned = aligned.sort_values("time_utc").reset_index(drop=True)

    def _calc_index(col_suffix: str) -> np.ndarray:
        log_val = np.full(len(aligned), np.log(cfg.ice_dxy_constant), dtype=float)
        for pair, w in effective_weights.items():
            prices = np.clip(aligned[f"{pair}_{col_suffix}"].to_numpy(dtype=float), 1e-9, None)
            log_val += w * np.log(prices)
        return np.exp(log_val)

    open_arr = _calc_index("open")
    close_arr = _calc_index("close")
    raw_high = _calc_index("high")
    raw_low = _calc_index("low")

    # Untuk bobot negatif (EURUSD/GBPUSD), high/low basket bisa terbalik sehingga di-envelope aman
    high_arr = np.maximum.reduce([open_arr, close_arr, raw_high, raw_low])
    low_arr = np.minimum.reduce([open_arr, close_arr, raw_high, raw_low])

    out = pd.DataFrame(
        {
            "time_utc": aligned["time_utc"],
            "open": open_arr,
            "high": high_arr,
            "low": low_arr,
            "close": close_arr,
        }
    )
    mode = "SYNTHETIC_ICE_6" if not missing else "SYNTHETIC_FALLBACK_5"
    return DXYSeriesResult(
        df=out,
        source_mode=mode,
        used_symbols=tuple(available),
        missing_symbols=missing,
    )


class DXYProvider:
    """Mengambil data DXY H4 dari MT5 (simbol broker tervalidasi atau sintetis ICE)."""

    def __init__(self, client: MT5Client, cfg: BotConfig = CONFIG) -> None:
        self.client = client
        self.cfg = cfg

    @staticmethod
    def is_valid_broker_dxy(
        mt5_module: object, symbol: str, last_close: Optional[float] = None
    ) -> bool:
        """Pastikan simbol bukan saham/ETF kebetulan bernama USDX (mis. SGI ETF @ $24)."""
        info = mt5_module.symbol_info(symbol)  # type: ignore[attr-defined]
        if info is None:
            return False
        desc = str(getattr(info, "description", "")).lower()
        path = str(getattr(info, "path", "")).lower()
        non_index_markers = ("etf", "nasdaq", "nyse", "stock", "equity", "shares")
        if any(m in desc or m in path for m in non_index_markers):
            if "dollar index" not in desc and "usd index" not in desc:
                return False

        price_to_check = last_close
        if price_to_check is None and hasattr(mt5_module, "symbol_info_tick"):
            tick = mt5_module.symbol_info_tick(symbol)  # type: ignore[attr-defined]
            if tick is not None:
                price_to_check = float(getattr(tick, "bid", 0.0) or 0.0)

        if price_to_check is not None and price_to_check > 0:
            if not (65.0 <= price_to_check <= 140.0):
                return False
        return True

    def fetch_dxy_h4(self, count: int, offset_hours: int) -> DXYSeriesResult:
        """Ambil DataFrame H4 DXY dengan inner-join timestamp alignment."""
        tf_h4 = getattr(self.client.mt5, "TIMEFRAME_H4", 16388)
        for cand in self.cfg.dxy_symbol_candidates:
            resolved = self.client.resolve_symbol((cand,))
            if resolved is None:
                continue
            df = self.client.fetch_rates_df(resolved, tf_h4, count, offset_hours)
            if df.empty:
                continue
            last_close = float(df["close"].iloc[-1])
            if self.is_valid_broker_dxy(self.client.mt5, resolved, last_close=last_close):
                return DXYSeriesResult(
                    df=df[["time_utc", "open", "high", "low", "close"]].copy(),
                    source_mode="BROKER_SYMBOL",
                    used_symbols=(resolved,),
                    missing_symbols=(),
                )

        aliases_map = get_symbol_aliases_map(self.cfg)
        pair_frames: Dict[str, pd.DataFrame] = {}
        resolved_pairs: Dict[str, str] = {}
        for canonical_pair, aliases in aliases_map.items():
            resolved = self.client.resolve_symbol(aliases)
            if resolved is None:
                continue
            resolved_pairs[canonical_pair] = resolved
            df_pair = self.client.fetch_rates_df(resolved, tf_h4, count, offset_hours)
            if not df_pair.empty:
                pair_frames[canonical_pair] = df_pair

        res = compute_synthetic_dxy_from_frames(pair_frames, self.cfg)
        # Jika cache lokal MT5 untuk salah satu pair ICE belum ter-sync penuh, paksa sync range
        if len(res.df) < min(count, 40) and hasattr(self.client.mt5, "copy_rates_range"):
            import time as _time

            end_srv = int(_time.time()) + offset_hours * 3600 + 86400
            start_srv = end_srv - max(count, 60) * 14400 * 2
            for canonical_pair, resolved in resolved_pairs.items():
                self.client.mt5.copy_rates_range(resolved, tf_h4, start_srv, end_srv)
                df_pair = self.client.fetch_rates_df(resolved, tf_h4, count, offset_hours)
                if not df_pair.empty:
                    pair_frames[canonical_pair] = df_pair
            res = compute_synthetic_dxy_from_frames(pair_frames, self.cfg)

        return res
