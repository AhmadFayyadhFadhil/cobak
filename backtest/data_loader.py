"""Loader data historis eksternal (Dukascopy CSV, HistData CSV, MT5 Export, Parquet)
untuk mendukung backtest 10-20 tahun XAUUSD dan DXY tanpa batas histori MT5 broker.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd

from config import CONFIG, BotConfig
from core.dxy_provider import compute_synthetic_dxy_from_frames


def load_ohlcv_file(
    path: Path,
    default_spread_points: float = 25.0,
) -> pd.DataFrame:
    """Muat file CSV/Parquet OHLCV M1 dari Dukascopy, HistData, atau export MT5 dalam UTC."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File data historis tidak ditemukan: {path}")

    if path.suffix.lower() == ".parquet":
        raw = pd.read_parquet(path)
    else:
        # Deteksi delimiter koma vs titik-koma (HistData sering memakai ';')
        sample = path.read_text(encoding="utf-8", errors="ignore").splitlines()[:2]
        sep = ";" if sample and ";" in sample[0] else ","
        raw = pd.read_csv(path, sep=sep)

    cols_lower = {c.lower().strip(): c for c in raw.columns}

    # Normalisasi kolom waktu ke UTC
    if "time_utc" in cols_lower:
        time_series = pd.to_datetime(raw[cols_lower["time_utc"]], utc=True)
    elif "gmt time" in cols_lower:
        time_series = pd.to_datetime(
            raw[cols_lower["gmt time"]], format="mixed", dayfirst=True, utc=True
        )
    elif "date" in cols_lower and "time" in cols_lower:
        combined = raw[cols_lower["date"]].astype(str) + " " + raw[cols_lower["time"]].astype(str)
        time_series = pd.to_datetime(combined, format="mixed", utc=True)
    elif "datetime" in cols_lower:
        time_series = pd.to_datetime(raw[cols_lower["datetime"]], format="mixed", utc=True)
    elif "time" in cols_lower:
        col = raw[cols_lower["time"]]
        if np.issubdtype(col.dtype, np.number):
            time_series = pd.to_datetime(col, unit="s", utc=True)
        else:
            time_series = pd.to_datetime(col, format="mixed", utc=True)
    else:
        raise ValueError(f"Kolom waktu tidak dikenali di {path}: {list(raw.columns)}")

    def _pick(name: str) -> pd.Series:
        if name not in cols_lower:
            raise ValueError(f"Kolom '{name}' tidak ada di {path}")
        return raw[cols_lower[name]].astype(float)

    vol_col = cols_lower.get("tick_volume") or cols_lower.get("volume")
    vol_series = raw[vol_col].astype(float) if vol_col else pd.Series(100.0, index=raw.index)

    spread_col = cols_lower.get("spread")
    spread_series = (
        raw[spread_col].astype(float)
        if spread_col
        else pd.Series(float(default_spread_points), index=raw.index)
    )

    df = pd.DataFrame(
        {
            "time_utc": time_series,
            "open": _pick("open"),
            "high": _pick("high"),
            "low": _pick("low"),
            "close": _pick("close"),
            "tick_volume": vol_series,
            "spread": spread_series,
        }
    )
    return df.dropna(subset=["time_utc", "open", "high", "low", "close"]).sort_values("time_utc").drop_duplicates("time_utc").reset_index(drop=True)


def resample_m1_to_all_timeframes(m1_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Resample data M1 menjadi M1, M5, M15, H1, H4, dan D1 yang konsisten."""
    if m1_df.empty:
        return {tf: m1_df.copy() for tf in ("M1", "M5", "M15", "H1", "H4", "D1")}

    indexed = m1_df.set_index("time_utc").sort_index()
    agg_rules = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "tick_volume": "sum",
        "spread": "median",
    }
    tf_map = {
        "M5": "5min",
        "M15": "15min",
        "H1": "1h",
        "H4": "4h",
        "D1": "1D",
    }
    out: Dict[str, pd.DataFrame] = {"M1": m1_df.reset_index(drop=True)}
    for label, rule in tf_map.items():
        res = indexed.resample(rule, label="left", closed="left").agg(agg_rules).dropna(subset=["open", "close"]).reset_index()
        out[label] = res
    return out


def build_20y_dataset(
    xau_m1_path: Path,
    dxy_h4_or_m1_path: Optional[Path] = None,
    ice_pair_paths: Optional[Mapping[str, Path]] = None,
    cfg: BotConfig = CONFIG,
) -> Dict[str, pd.DataFrame]:
    """Muat dataset multi-dekade XAUUSD dan DXY (langsung atau sintetis ICE)."""
    xau_m1 = load_ohlcv_file(xau_m1_path)
    xau_mtf = resample_m1_to_all_timeframes(xau_m1)

    if dxy_h4_or_m1_path is not None:
        dxy_raw = load_ohlcv_file(dxy_h4_or_m1_path)
        dxy_h4 = resample_m1_to_all_timeframes(dxy_raw)["H4"]
    elif ice_pair_paths:
        pair_h4: Dict[str, pd.DataFrame] = {}
        for pair, p_path in ice_pair_paths.items():
            pair_h4[pair] = resample_m1_to_all_timeframes(load_ohlcv_file(p_path))["H4"]
        dxy_h4 = compute_synthetic_dxy_from_frames(pair_h4, cfg).df
    else:
        raise ValueError("Sediakan `dxy_h4_or_m1_path` atau `ice_pair_paths` untuk data DXY.")

    xau_mtf["DXY_H4"] = dxy_h4
    return xau_mtf
