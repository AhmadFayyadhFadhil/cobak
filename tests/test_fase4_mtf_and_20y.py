"""Unit & integration tests untuk Upgrade Multi-Timeframe (D1..M5) & Backtest 20 Tahun
(sniper-tdd: test dibuat lebih dulu sebelum implementasi).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest.data_loader import load_ohlcv_file, resample_m1_to_all_timeframes
from backtest.engine import BacktestEngine
from config import BotConfig
from core.state_manager import StateManager
from dashboard.app import load_mtf_matrix
from strategy.layer2_structure import SupplyDemandZone
from strategy.layer4_risk_exit import evaluate_active_position_actions
from strategy.mtf_analyzer import build_mtf_matrix


def test_mtf_matrix_covers_all_five_timeframes_and_saves_to_dashboard(tmp_path: Path) -> None:
    """Analisa MTF mencakup D1, H4, H1, M15, dan M5 serta terbaca oleh dashboard."""
    t_end = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)

    def _bull_df(periods: int, freq: str) -> pd.DataFrame:
        times = pd.date_range(end=t_end, periods=periods, freq=freq, tz="UTC")
        closes = np.linspace(2600.0, 2660.0, periods)
        return pd.DataFrame(
            {
                "time_utc": times,
                "open": closes - 0.5,
                "high": closes + 1.0,
                "low": closes - 1.0,
                "close": closes,
                "tick_volume": 100.0,
            }
        )

    d1_df = _bull_df(40, "1D")
    h4_df = _bull_df(60, "4h")
    h1_df = _bull_df(80, "1h")
    m15_df = _bull_df(80, "15min")
    m5_df = _bull_df(80, "5min")

    zones_h4 = [
        SupplyDemandZone("Z_H4", "H4", "BUY", 2658.0, 2662.0, t_end.isoformat(), True)
    ]
    report = build_mtf_matrix(
        d1_df=d1_df,
        h4_df=h4_df,
        h1_df=h1_df,
        m15_df=m15_df,
        m5_df=m5_df,
        macro_bias="BUY",
        zones_h4=zones_h4,
        zones_h1=[],
        cfg=BotConfig(pivot_window=3, entry_timeframe="M15_OR_M5"),
    )
    assert set(report.snapshots.keys()) == {"D1", "H4", "H1", "M15", "M5"}
    assert report.snapshots["D1"].trend_bias == "BUY"
    assert report.alignment_score_pct >= 60.0

    # Persist ke SQLite lalu baca via helper read-only dashboard
    db_path = tmp_path / "state.db"
    sm = StateManager(db_path)
    sm.set_meta("mtf_matrix_json", report.to_json())
    loaded = load_mtf_matrix(db_path)
    assert loaded is not None
    assert loaded["dominant_bias"] == "BUY"
    assert set(loaded["snapshots"].keys()) == {"D1", "H4", "H1", "M15", "M5"}


def test_data_loader_dukascopy_csv_resamples_all_timeframes(tmp_path: Path) -> None:
    """Loader eksternal CSV format Dukascopy menghasilkan M1, M5, M15, H1, H4, dan D1 dalam UTC."""
    csv_path = tmp_path / "xau_dukascopy_m1.csv"
    times = pd.date_range("2006-06-01 07:00:00", periods=1500, freq="1min", tz="UTC")
    prices = 620.0 + np.cumsum(np.random.default_rng(9).normal(0.0, 0.08, size=len(times)))
    df_raw = pd.DataFrame(
        {
            "Gmt time": [t.strftime("%d.%m.%Y %H:%M:%S.000") for t in times],
            "Open": prices - 0.05,
            "High": prices + 0.15,
            "Low": prices - 0.15,
            "Close": prices,
            "Volume": 12.5,
        }
    )
    df_raw.to_csv(csv_path, index=False)

    m1_df = load_ohlcv_file(csv_path, default_spread_points=30.0)
    assert len(m1_df) == 1500
    assert str(m1_df["time_utc"].dt.tz) == "UTC"
    assert "spread" in m1_df.columns

    mtf = resample_m1_to_all_timeframes(m1_df)
    assert set(mtf.keys()) == {"M1", "M5", "M15", "H1", "H4", "D1"}
    assert len(mtf["M5"]) == 300
    assert len(mtf["M15"]) == 100
    assert len(mtf["H1"]) == 25


def test_20y_regime_breakdown_news_coverage_and_atr_relative_trail() -> None:
    """Backtest lintas era ($600 vs $2,600) mendukung step trail berbasis ATR & laporan per 5 tahun."""
    # Uji min_trail_step_atr_ratio saat emas harga $600 (ATR M5 = $0.50 -> step 0.15*0.50 = $0.075)
    act = evaluate_active_position_actions(
        direction="BUY",
        entry_price=600.0,
        initial_sl=598.5,
        current_sl=600.1,
        current_bid=604.0,
        current_ask=604.2,
        current_volume=0.10,
        volume_min=0.01,
        atr_m5=0.50,
        partial_done=True,
        allow_partial=True,
        cfg=BotConfig(min_trail_step_usd=0.25, min_trail_step_atr_ratio=0.15),
    )
    # 604.0 - 1.5*0.50 = 603.25 > 600.1 (diff 3.15 >= 0.075)
    assert act.new_sl == pytest.approx(603.25, abs=1e-4)

    # Uji regime_breakdown 5-tahunan & news_coverage_note pada BacktestEngine
    # Bangun 2 blok tanggal: 2008 & 2024 pada H4/H1/M5
    t_2008 = pd.date_range("2008-05-01", periods=300, freq="5min", tz="UTC")
    t_2024 = pd.date_range("2024-05-01", periods=300, freq="5min", tz="UTC")
    times = t_2008.append(t_2024)
    c = np.concatenate([np.full(300, 880.0), np.full(300, 2350.0)])
    m5_df = pd.DataFrame(
        {
            "time_utc": times,
            "open": c - 0.2,
            "high": c + 0.5,
            "low": c - 0.5,
            "close": c,
            "tick_volume": 50.0,
            "spread": 25.0,
        }
    )
    h1_df = m5_df.set_index("time_utc").resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "tick_volume": "sum", "spread": "median"}
    ).dropna().reset_index()
    h4_df = m5_df.set_index("time_utc").resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "tick_volume": "sum", "spread": "median"}
    ).dropna().reset_index()
    dxy_h4 = pd.DataFrame(
        {
            "time_utc": h4_df["time_utc"],
            "open": 100.0,
            "high": 100.2,
            "low": 99.8,
            "close": 100.0,
        }
    )

    news_only_2024 = pd.DataFrame(
        {
            "time_utc": pd.to_datetime(["2024-05-01T12:30:00Z"], utc=True),
            "currency": ["USD"],
            "impact": ["HIGH"],
            "title": ["FOMC Statement"],
        }
    )
    res = BacktestEngine(BotConfig(corr_window_h4=20)).run(
        m5_df=m5_df,
        h1_df=h1_df,
        h4_df=h4_df,
        dxy_h4=dxy_h4,
        news_history_df=news_only_2024,
    )
    assert isinstance(res.regime_breakdown, dict)
    assert "PERINGATAN" in res.news_coverage_note or "2024" in res.news_coverage_note


def test_save_backtest_report_json(tmp_path: Path) -> None:
    """save_backtest_report menghasilkan JSON valid untuk dibaca dashboard Streamlit."""
    from backtest.engine import BacktestResult
    from run_backtest import save_backtest_report

    out_json = tmp_path / "bt.json"
    res = BacktestResult([], 1000.0, 1000.0, 0, 0.0, 0.0, 0.0, 0.0, {"total_m5_bars": 100})
    save_backtest_report(res, "MT5-Demo", "2026-03-01 s/d 2026-11-27", out_path=out_json)
    assert out_json.exists()
    assert '"total_m5_bars": 100' in out_json.read_text(encoding="utf-8")


def test_data_loader_histdata_headerless_est_conversion(tmp_path: Path) -> None:
    """File HistData tanpa header (; separated, EST UTC-5) otomatis dikonversi ke UTC."""
    csv_path = tmp_path / "DAT_ASCII_XAUUSD_M1_2020.csv"
    csv_path.write_text(
        "20200102 020000;1517.25;1517.80;1517.10;1517.60;0\n"
        "20200102 020100;1517.60;1518.00;1517.50;1517.90;0\n",
        encoding="utf-8",
    )
    df = load_ohlcv_file(csv_path, default_spread_points=25.0)
    assert len(df) == 2
    # 02:00 EST (UTC-5) == 07:00 UTC
    assert df["time_utc"].iloc[0] == pd.Timestamp("2020-01-02 07:00:00+00:00")


def test_data_loader_mt5_tab_export_zero_spread_and_h1_backtest_mode(tmp_path: Path) -> None:
    """Export tab-separated MT5 (<DATE> <TIME> ... <SPREAD>=0) & entry_timeframe=H1 berjalan bersih."""
    import zipfile

    tsv_content = (
        "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n"
        + "\n".join(
            f"2020.01.{(i // 24) + 2:02d}\t{i % 24:02d}:00:00\t{1520 + i * 0.2:.2f}\t{1522 + i * 0.2:.2f}\t{1519 + i * 0.2:.2f}\t{1521 + i * 0.2:.2f}\t1000\t0\t0"
            for i in range(100)
        )
    )
    zip_path = tmp_path / "mt5_h1.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("XAUUSD_H1.csv", tsv_content)

    h1_df = load_ohlcv_file(zip_path, default_spread_points=25.0, source_utc_offset_hours=2)
    assert len(h1_df) == 100
    assert (h1_df["spread"] == 25.0).all()

    from backtest.data_loader import resample_h1_to_higher_timeframes

    mtf = resample_h1_to_higher_timeframes(h1_df)
    res = BacktestEngine(BotConfig(entry_timeframe="H1")).run(
        m5_df=pd.DataFrame(),
        h1_df=mtf["H1"],
        h4_df=mtf["H4"],
        dxy_h4=None,
        d1_df=mtf["D1"],
        initial_balance=2000.0,
    )
    assert res.funnel["total_m5_bars"] == len(mtf["H1"]) - 25
