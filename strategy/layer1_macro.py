"""Layer 1: Filter Sesi UTC (07:00-16:00), News Filter (Live fail-closed JSON /
Backtest CSV), dan Dynamic Correlation DXY vs XAUUSD H4.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

import pandas as pd

from config import CONFIG, BotConfig
from strategy.indicators import compute_rolling_correlation, detect_confirmed_pivots

BiasType = Literal["BUY", "SELL", "NEUTRAL"]


@dataclass(frozen=True)
class Layer1Result:
    """Hasil evaluasi Layer 1 beserta detail bias dan korelasi."""

    passed: bool
    bias: Optional[Literal["BUY", "SELL"]]
    correlation: Optional[float]
    dxy_trend: BiasType
    xau_trend: BiasType
    reason: str


def is_within_session_utc(
    now_utc: datetime, start_hhmm: str, end_hhmm: str
) -> bool:
    """Cek apakah waktu UTC berada di sesi London & New York."""
    utc_dt = now_utc.astimezone(timezone.utc)
    sh, sm = map(int, start_hhmm.split(":"))
    eh, em = map(int, end_hhmm.split(":"))
    cur_min = utc_dt.hour * 60 + utc_dt.minute
    start_min = sh * 60 + sm
    end_min = eh * 60 + em
    return start_min <= cur_min < end_min


def _is_high_impact_usd(currency: str, impact: str, title: str) -> bool:
    cur = currency.strip().upper()
    imp = impact.strip().upper()
    ttl = title.strip().upper()
    if cur != "USD":
        return False
    if imp == "HIGH":
        return True
    return any(k in ttl for k in ("CPI", "NFP", "NON-FARM", "FOMC"))


def check_news_filter_live(now_utc: datetime, cfg: BotConfig = CONFIG) -> tuple[bool, str]:
    """Periksa jadwal berita High Impact USD secara fail-closed dari JSON lokal."""
    path = cfg.news_events_json
    if not path.exists():
        return False, "Skipped: File news_events.json tidak ditemukan (fail-closed)."

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        updated_raw = payload.get("updated_at_utc")
        if not updated_raw:
            return False, "Skipped: Metadata updated_at_utc di news JSON kosong (fail-closed)."
        updated_dt = datetime.fromisoformat(str(updated_raw).replace("Z", "+00:00")).astimezone(
            timezone.utc
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"Skipped: Gagal membaca news_events.json ({exc}) (fail-closed)."

    age_hours = (now_utc.astimezone(timezone.utc) - updated_dt).total_seconds() / 3600.0
    if age_hours > cfg.news_max_age_hours:
        return False, f"Skipped: Jadwal news kedaluwarsa ({age_hours:.1f}j > {cfg.news_max_age_hours}j)."

    before = timedelta(minutes=cfg.news_before_minutes)
    after = timedelta(minutes=cfg.news_after_minutes)
    for ev in payload.get("events", []):
        try:
            ev_dt = datetime.fromisoformat(str(ev["time_utc"]).replace("Z", "+00:00")).astimezone(
                timezone.utc
            )
        except Exception:
            continue
        if not _is_high_impact_usd(
            str(ev.get("currency", "")), str(ev.get("impact", "")), str(ev.get("title", ""))
        ):
            continue
        if (ev_dt - before) <= now_utc.astimezone(timezone.utc) <= (ev_dt + after):
            return False, f"Skipped: News High Impact USD aktif ({ev.get('title', 'USD News')})."

    return True, "OK"


def check_news_filter_history(
    now_utc: datetime, news_history_df: pd.DataFrame, cfg: BotConfig = CONFIG
) -> tuple[bool, str]:
    """Periksa jadwal berita historis CSV untuk fase backtest."""
    if news_history_df is None or news_history_df.empty:
        return True, "OK"
    utc_dt = now_utc.astimezone(timezone.utc)
    before = timedelta(minutes=cfg.news_before_minutes)
    after = timedelta(minutes=cfg.news_after_minutes)

    mask = (news_history_df["time_utc"] - before <= utc_dt) & (
        utc_dt <= news_history_df["time_utc"] + after
    )
    if mask.any():
        title = str(news_history_df.loc[mask, "title"].iloc[0])
        return False, f"Skipped: News historis aktif ({title})."
    return True, "OK"


def infer_h4_structure_bias(df: pd.DataFrame, pivot_window: int = 5) -> BiasType:
    """Tentukan bias tren H4 berdasarkan pivot terkonfirmasi dan posisi close terakhir."""
    if df.empty or len(df) < max(15, 2 * pivot_window + 2):
        return "NEUTRAL"

    pivots = detect_confirmed_pivots(df, window=pivot_window)
    highs = [p for p in pivots if p.kind == "HIGH"]
    lows = [p for p in pivots if p.kind == "LOW"]
    last_close = float(df["close"].iloc[-1])

    if highs and lows:
        last_h = highs[-1].price
        last_l = lows[-1].price
        if last_close > last_h:
            return "BUY"
        if last_close < last_l:
            return "SELL"
        if len(highs) >= 2 and len(lows) >= 2:
            if highs[-1].price > highs[-2].price and lows[-1].price > lows[-2].price:
                return "BUY"
            if highs[-1].price < highs[-2].price and lows[-1].price < lows[-2].price:
                return "SELL"

    # Fallback objektif jika belum terbentuk 2 pasang pivot (misal awal dataset): EMA(20) slope
    ema20 = df["close"].astype(float).ewm(span=20, adjust=False).mean()
    if last_close > float(ema20.iloc[-1]) and float(ema20.iloc[-1]) > float(ema20.iloc[-5]):
        return "BUY"
    if last_close < float(ema20.iloc[-1]) and float(ema20.iloc[-1]) < float(ema20.iloc[-5]):
        return "SELL"
    return "NEUTRAL"


def evaluate_layer1(
    now_utc: datetime,
    xau_h4: pd.DataFrame,
    dxy_h4: pd.DataFrame,
    cfg: BotConfig = CONFIG,
    news_history_df: Optional[pd.DataFrame] = None,
) -> Layer1Result:
    """Evaluasi berurutan Layer 1: Sesi -> News -> Korelasi Dinamis DXY vs XAUUSD."""
    if not is_within_session_utc(now_utc, cfg.session_start_utc, cfg.session_end_utc):
        return Layer1Result(
            passed=False,
            bias=None,
            correlation=None,
            dxy_trend="NEUTRAL",
            xau_trend="NEUTRAL",
            reason="Skipped: Di luar sesi aktif 07:00-16:00 UTC.",
        )

    if news_history_df is not None:
        news_ok, news_reason = check_news_filter_history(now_utc, news_history_df, cfg)
    else:
        news_ok, news_reason = check_news_filter_live(now_utc, cfg)
    if not news_ok:
        return Layer1Result(
            passed=False,
            bias=None,
            correlation=None,
            dxy_trend="NEUTRAL",
            xau_trend="NEUTRAL",
            reason=news_reason,
        )

    corr = compute_rolling_correlation(xau_h4, dxy_h4, window=cfg.corr_window_h4)
    xau_trend = infer_h4_structure_bias(xau_h4, pivot_window=cfg.pivot_window)
    dxy_trend = infer_h4_structure_bias(dxy_h4, pivot_window=cfg.pivot_window)

    if corr is None:
        return Layer1Result(
            passed=False,
            bias=None,
            correlation=None,
            dxy_trend=dxy_trend,
            xau_trend=xau_trend,
            reason="Skipped: Korelasi DXY vs XAUUSD tidak tersedia/NaN.",
        )

    dxy_inverse_bias: BiasType = (
        "BUY" if dxy_trend == "SELL" else ("SELL" if dxy_trend == "BUY" else "NEUTRAL")
    )

    # Kasus 1: Korelasi negatif kuat (< -0.5) -> ikuti tren DXY sebagai bias arah
    if corr < cfg.corr_negative_threshold:
        if dxy_inverse_bias == "NEUTRAL":
            return Layer1Result(
                passed=False,
                bias=None,
                correlation=corr,
                dxy_trend=dxy_trend,
                xau_trend=xau_trend,
                reason=f"Skipped: Korelasi {corr:.2f} < -0.5 tetapi tren DXY H4 netral.",
            )
        return Layer1Result(
            passed=True,
            bias=dxy_inverse_bias,
            correlation=corr,
            dxy_trend=dxy_trend,
            xau_trend=xau_trend,
            reason=f"OK (DXY corr={corr:.2f}, DXY={dxy_trend} -> Bias {dxy_inverse_bias})",
        )

    # Kasus 2: Korelasi anomali/tidak sinkron (> -0.3) -> abaikan DXY, pakai struktur XAUUSD murni
    if corr > cfg.corr_anomaly_threshold:
        if xau_trend == "NEUTRAL":
            return Layer1Result(
                passed=False,
                bias=None,
                correlation=corr,
                dxy_trend=dxy_trend,
                xau_trend=xau_trend,
                reason=f"Skipped: Anomali DXY (corr={corr:.2f}), struktur XAUUSD H4 netral.",
            )
        return Layer1Result(
            passed=True,
            bias=xau_trend,
            correlation=corr,
            dxy_trend=dxy_trend,
            xau_trend=xau_trend,
            reason=f"OK (Anomali DXY corr={corr:.2f} -> Struktur XAUUSD {xau_trend})",
        )

    # Kasus 3: Zona abu-abu (-0.5 <= corr <= -0.3) -> skip kecuali DXY inverse & XAUUSD searah
    if dxy_inverse_bias != "NEUTRAL" and dxy_inverse_bias == xau_trend:
        return Layer1Result(
            passed=True,
            bias=xau_trend,
            correlation=corr,
            dxy_trend=dxy_trend,
            xau_trend=xau_trend,
            reason=f"OK (Grey zone corr={corr:.2f}, DXY & XAUUSD searah {xau_trend})",
        )
    return Layer1Result(
        passed=False,
        bias=None,
        correlation=corr,
        dxy_trend=dxy_trend,
        xau_trend=xau_trend,
        reason=(
            f"Skipped: Zona abu-abu korelasi ({corr:.2f}) dan DXY ({dxy_trend}) "
            f"tidak sejalan dengan XAUUSD ({xau_trend})."
        ),
    )
