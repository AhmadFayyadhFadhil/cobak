"""Unit & integration tests Fase 3 (sniper-tdd: test dibuat lebih dulu).

Menguji:
- Backtester intrabar M1 dengan asumsi pesimis (SL kena duluan saat ambigu di bar M1 yang sama)
- Penggunaan spread historis per bar & kalender news_history.csv
- Perhitungan Realized R, Profit Factor, Expectancy, Max Drawdown, dan OOS Pass Criteria
- Walk-Forward bergulir (IS -> OOS, dibatasi 3-4 parameter inti)
- Runner main.py (single iteration tick & new bar dalam dry-run mode dengan FakeMT5)
- Dashboard Streamlit helpers (read-only SQLite -> 6 kartu pipeline & activity tape tanpa impor MT5)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestEngine, load_news_history_csv, simulate_trade_intrabar_m1
from backtest.walk_forward import evaluate_oos_pass_criteria, run_walk_forward
from config import BotConfig
from core.state_manager import StateManager
from dashboard.app import build_six_cards_snapshot, load_activity_tape
from main import SniperEngineRunner
from tests.test_fase1_foundation import FakeMT5


def test_intrabar_m1_pessimistic_sl_first_on_ambiguity() -> None:
    """Jika dalam 1 bar M1 yang sama SL dan TP/1:2 tersentuh bersamaan, SL harus dianggap kena duluan."""
    t0 = datetime(2026, 11, 24, 9, 0, tzinfo=timezone.utc)
    # BUY limit di 2650.0, SL 2645.0, TP 1:2 = 2660.0, TP 1:5 = 2675.0
    m1_bars = pd.DataFrame(
        {
            "time_utc": pd.date_range(t0, periods=3, freq="1min", tz="UTC"),
            "open": [2652.0, 2650.5, 2651.0],
            "high": [2653.0, 2662.0, 2652.0],  # Bar 1 menyentuh limit (2650), TP 1:2 (2660), DAN SL (2645) sekaligus!
            "low": [2650.8, 2644.0, 2650.0],
            "close": [2651.0, 2655.0, 2651.0],
            "spread": [20, 20, 20],
        }
    )
    outcome = simulate_trade_intrabar_m1(
        direction="BUY",
        limit_price=2650.0,
        sl_price=2645.0,
        tp_1_2_price=2660.0,
        tp_final_price=2675.0,
        atr_m5=2.0,
        m1_slice=m1_bars,
        expiry_utc=t0 + timedelta(hours=2),
        session_end_hhmm="16:00",
        point=0.01,
        cfg=BotConfig(min_lot_partial_policy="NO_PARTIAL"),
        allow_partial=True,
    )
    assert outcome.filled is True
    assert outcome.exit_reason == "STOP_LOSS"
    assert outcome.realized_r == pytest.approx(-1.0)


def test_intrabar_m1_partial_1_2_then_final_tp_realized_3_5r() -> None:
    """Jika bar M1 menyentuh 1:2 lebih dulu lalu bar berikutnya menyentuh 1:5, Realized R = 3.5R."""
    t0 = datetime(2026, 11, 24, 9, 0, tzinfo=timezone.utc)
    m1_bars = pd.DataFrame(
        {
            "time_utc": pd.date_range(t0, periods=4, freq="1min", tz="UTC"),
            "open": [2652.0, 2650.5, 2655.0, 2662.0],
            "high": [2652.5, 2651.0, 2661.0, 2676.0],  # Bar 1 fill 2650; Bar 2 hit 1:2 (2660); Bar 3 hit 1:5 (2675)
            "low": [2651.2, 2649.8, 2654.0, 2661.0],
            "close": [2651.5, 2650.6, 2660.5, 2675.5],
            "spread": [20, 20, 20, 20],
        }
    )
    outcome = simulate_trade_intrabar_m1(
        direction="BUY",
        limit_price=2650.0,
        sl_price=2645.0,
        tp_1_2_price=2660.0,
        tp_final_price=2675.0,
        atr_m5=2.0,
        m1_slice=m1_bars,
        expiry_utc=t0 + timedelta(hours=2),
        session_end_hhmm="16:00",
        point=0.01,
        cfg=BotConfig(commission_per_lot_account=0.0),
        allow_partial=True,
    )
    assert outcome.filled is True
    assert outcome.hit_partial is True
    assert outcome.exit_reason == "TAKE_PROFIT"
    assert outcome.realized_r == pytest.approx(3.5, abs=1e-4)


def test_backtest_engine_and_oos_pass_criteria(tmp_path: Path) -> None:
    """Backtest engine berjalan deterministik dengan historical spread & news CSV serta evaluasi OOS."""
    news_df = load_news_history_csv(Path("data/news_history.csv"))
    assert not news_df.empty

    # Buat data sintetis multi-hari (M1, M5, H1, H4, DXY H4)
    start = datetime(2026, 10, 1, 0, 0, tzinfo=timezone.utc)
    m1_times = pd.date_range(start, periods=15 * 24 * 60, freq="1min", tz="UTC")
    rng = np.random.default_rng(123)
    steps = rng.normal(0.0, 0.25, size=len(m1_times))
    close_m1 = 2650.0 + np.cumsum(steps)
    m1_df = pd.DataFrame(
        {
            "time_utc": m1_times,
            "open": close_m1 - 0.1,
            "high": close_m1 + 0.4,
            "low": close_m1 - 0.4,
            "close": close_m1,
            "tick_volume": 50.0,
            "spread": 25,
        }
    )

    def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
        r = (
            df.set_index("time_utc")
            .resample(rule)
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "tick_volume": "sum",
                    "spread": "median",
                }
            )
            .dropna()
            .reset_index()
        )
        return r

    m5_df = _resample(m1_df, "5min")
    h1_df = _resample(m1_df, "1h")
    h4_df = _resample(m1_df, "4h")

    # DXY berkorelasi negatif
    dxy_close = 105.0 - (h4_df["close"] - 2650.0) * 0.05
    dxy_h4 = pd.DataFrame(
        {
            "time_utc": h4_df["time_utc"],
            "open": dxy_close + 0.05,
            "high": dxy_close + 0.1,
            "low": dxy_close - 0.1,
            "close": dxy_close,
        }
    )

    engine = BacktestEngine(
        BotConfig(
            corr_window_h4=20,
            entry_timeframe="M5",
            zone_armed_timeout_hours=5.0,
            pending_order_expiry_hours=3.0,
        ),
        point=0.01,
    )
    result = engine.run(
        m5_df=m5_df,
        h1_df=h1_df,
        h4_df=h4_df,
        dxy_h4=dxy_h4,
        m1_df=m1_df,
        news_history_df=news_df,
        initial_balance=2000.0,
    )
    assert result.funnel["total_m5_bars"] == len(m5_df) - 25
    assert result.max_drawdown_pct >= 0.0

    # Uji evaluasi 4 Kriteria Lulus OOS
    passed = evaluate_oos_pass_criteria(
        profit_factor=1.45,
        expectancy_r=0.35,
        max_drawdown_pct=11.2,
        trade_count=60,
    )
    assert passed.passed is True

    failed = evaluate_oos_pass_criteria(
        profit_factor=1.10,
        expectancy_r=0.10,
        max_drawdown_pct=18.0,
        trade_count=30,
    )
    assert failed.passed is False
    assert len(failed.failed_reasons) == 4


def test_walk_forward_limits_params_and_splits_rolling_windows() -> None:
    """Walk-forward membatasi optimasi ke 3-4 parameter inti dan menghasilkan agregasi OOS."""
    start = datetime(2026, 9, 1, 0, 0, tzinfo=timezone.utc)
    m5_times = pd.date_range(start, periods=30 * 24 * 12, freq="5min", tz="UTC")
    rng = np.random.default_rng(77)
    c = 2650.0 + np.cumsum(rng.normal(0.0, 0.4, size=len(m5_times)))
    m5_df = pd.DataFrame(
        {
            "time_utc": m5_times,
            "open": c - 0.1,
            "high": c + 0.5,
            "low": c - 0.5,
            "close": c,
            "tick_volume": 100.0,
            "spread": 25,
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
            "open": 105.0 - (h4_df["close"] - 2650.0) * 0.05,
            "high": 105.1 - (h4_df["close"] - 2650.0) * 0.05,
            "low": 104.9 - (h4_df["close"] - 2650.0) * 0.05,
            "close": 105.0 - (h4_df["close"] - 2650.0) * 0.05,
        }
    )

    wf = run_walk_forward(
        m5_df=m5_df,
        h1_df=h1_df,
        h4_df=h4_df,
        dxy_h4=dxy_h4,
        m1_df=None,
        news_history_df=pd.DataFrame(),
        n_splits=2,
        param_grid=[
            {"pivot_window": 4, "sl_atr_multiplier": 0.5, "zone_armed_timeout_hours": 5.0},
            {"pivot_window": 5, "sl_atr_multiplier": 0.6, "zone_armed_timeout_hours": 4.0},
        ],
        base_cfg=BotConfig(
            corr_window_h4=20,
            entry_timeframe="M5",
            zone_armed_timeout_hours=5.0,
            pending_order_expiry_hours=3.0,
        ),
    )
    assert len(wf.windows) == 2
    assert wf.oos_criteria is not None


def test_runner_single_cycle_dry_run_and_dashboard_6_cards(tmp_path: Path) -> None:
    """Runner main.py berjalan 1 siklus tanpa error di dry-run dan Streamlit helper merender 6 kartu."""
    db_path = tmp_path / "state.db"
    news_path = tmp_path / "news.json"
    now_utc = datetime(2026, 11, 24, 10, 0, tzinfo=timezone.utc)
    news_path.write_text(f'{{"updated_at_utc": "{now_utc.isoformat()}", "events": []}}', encoding="utf-8")

    fake_mt5 = FakeMT5(digits=2)
    fake_mt5.tick_time = int(now_utc.timestamp())
    fake_mt5.account_info = lambda: SimpleNamespace(balance=2000.0, equity=2000.0)  # type: ignore[attr-defined]
    fake_mt5.TIMEFRAME_M5 = 5
    fake_mt5.TIMEFRAME_H1 = 16385
    fake_mt5.TIMEFRAME_H4 = 16388

    def _make_rates(n: int, step_sec: int, base: float = 2650.0):
        start_ts = int(now_utc.timestamp()) - n * step_sec
        return [
            {
                "time": start_ts + i * step_sec,
                "open": base,
                "high": base + 1.0,
                "low": base - 1.0,
                "close": base + 0.2,
                "tick_volume": 100,
                "spread": 25,
            }
            for i in range(n)
        ]

    def _copy_rates(sym, tf, start_pos, count):
        if tf == fake_mt5.TIMEFRAME_M5:
            return _make_rates(count, 300)
        if tf == fake_mt5.TIMEFRAME_H1:
            return _make_rates(count, 3600)
        return _make_rates(count, 14400, base=105.0 if sym != "XAUUSD" else 2650.0)

    fake_mt5.copy_rates_from_pos = _copy_rates  # type: ignore[attr-defined]
    # Tambahkan EURUSD..USDCHF agar DXY sintetis tersedia
    orig_sym_info = fake_mt5.symbol_info

    def _sym_info(s: str):
        if s in ("EURUSD", "USDJPY", "GBPUSD", "USDCAD", "USDSEK", "USDCHF"):
            return SimpleNamespace(
                digits=5, point=0.00001, trade_tick_size=0.00001, trade_tick_value=1.0,
                volume_min=0.01, volume_max=100.0, volume_step=0.01, trade_contract_size=100000.0, visible=True
            )
        return orig_sym_info(s)

    fake_mt5.symbol_info = _sym_info  # type: ignore[method-assign]

    cfg = BotConfig(dry_run_mode=True, news_events_json=news_path, corr_window_h4=20)
    sm = StateManager(db_path)
    runner = SniperEngineRunner(cfg=cfg, state_manager=sm, mt5_module=fake_mt5)
    runner.run_once(now_utc=now_utc)

    cards = build_six_cards_snapshot(db_path, now_utc=now_utc)
    assert set(cards.keys()) == {"CHIEF", "SCAN", "VET", "SIZE", "FILLS", "RISK"}
    tape = load_activity_tape(db_path)
    assert isinstance(tape, pd.DataFrame)
