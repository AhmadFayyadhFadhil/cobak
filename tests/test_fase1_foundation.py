"""Unit tests untuk Fase 1: MT5Client (2/3 digit, fresh-tick UTC offset, idempotent retry),
DXYProvider (ICE 6-pair & fallback 5-pair), StateManager (SQLite & UTC resets), dan AntiGanasGuard.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from config import BotConfig
from core.dxy_provider import compute_synthetic_dxy_from_frames
from core.logger import format_funnel_report
from core.mt5_client import (
    TRADE_RETCODE_CONNECTION,
    TRADE_RETCODE_DONE,
    TRADE_RETCODE_PLACED,
    TRADE_RETCODE_REQUOTE,
    MT5Client,
    make_unique_comment,
)
from core.state_manager import StateManager, StrategyState
from guardrails.anti_ganas import AntiGanasGuard


@dataclass
class FakeMT5:
    """Mock modular antarmuka MetaTrader5 untuk unit test deterministik."""

    ORDER_TYPE_BUY: int = 0
    ORDER_TYPE_SELL: int = 1
    ORDER_TYPE_BUY_LIMIT: int = 2
    ORDER_TYPE_SELL_LIMIT: int = 3
    POSITION_TYPE_BUY: int = 0
    POSITION_TYPE_SELL: int = 1
    TRADE_ACTION_DEAL: int = 1
    TRADE_ACTION_PENDING: int = 5
    TRADE_ACTION_SLTP: int = 6
    TRADE_ACTION_REMOVE: int = 8
    ORDER_TIME_GTC: int = 0
    ORDER_TIME_SPECIFIED: int = 2
    ORDER_FILLING_IOC: int = 1
    ORDER_FILLING_RETURN: int = 2

    def __init__(self, digits: int = 2) -> None:
        self.digits = digits
        self.point = 0.01 if digits == 2 else 0.001
        self.tick_bid = 2650.00
        self.tick_ask = 2650.25
        self.tick_time = 0
        self.positions: list[SimpleNamespace] = []
        self.orders: list[SimpleNamespace] = []
        self.history_orders: list[SimpleNamespace] = []
        self.send_outcomes: list[SimpleNamespace] = []
        self.send_calls: list[dict] = []

    def initialize(self) -> bool:
        return True

    def shutdown(self) -> None:
        return None

    def symbol_info(self, symbol: str):
        if symbol not in ("XAUUSD", "XAUUSDm"):
            return None
        return SimpleNamespace(
            digits=self.digits,
            point=self.point,
            trade_tick_size=self.point,
            trade_tick_value=1.0 if self.digits == 2 else 0.1,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
            trade_contract_size=100.0,
            visible=True,
        )

    def symbol_select(self, symbol: str, enable: bool) -> bool:
        return True

    def symbol_info_tick(self, symbol: str):
        return SimpleNamespace(bid=self.tick_bid, ask=self.tick_ask, time=self.tick_time)

    def positions_get(self):
        return list(self.positions)

    def orders_get(self):
        return list(self.orders)

    def history_orders_get(self, start_ts: int, end_ts: int):
        return list(self.history_orders)

    def order_send(self, request: dict):
        self.send_calls.append(request)
        if self.send_outcomes:
            return self.send_outcomes.pop(0)
        return SimpleNamespace(retcode=TRADE_RETCODE_DONE, order=999, comment="OK")

    def last_error(self):
        return (0, "OK")


def test_symbol_digits_2_vs_3_spread_usd(tmp_path: Path) -> None:
    """Spread dalam USD konsisten di broker 2 digit maupun 3 digit."""
    sm = StateManager(tmp_path / "state.db")
    for digits in (2, 3):
        fake = FakeMT5(digits=digits)
        fake.tick_bid = 2650.100
        fake.tick_ask = 2650.400
        client = MT5Client(BotConfig(), sm, mt5_module=fake, sleep_fn=lambda _: None)
        spec = client.get_symbol_spec("XAUUSD")
        assert spec.digits == digits
        assert pytest.approx(client.get_spread_usd("XAUUSD"), abs=1e-6) == 0.30


def test_fresh_tick_utc_offset_and_weekend_fallback(tmp_path: Path) -> None:
    """Kalibrasi offset hanya saat tick fresh; di weekend pakai offset valid terakhir."""
    sm = StateManager(tmp_path / "state.db")
    fake = FakeMT5()
    client = MT5Client(BotConfig(), sm, mt5_module=fake, sleep_fn=lambda _: None)

    # Rabu 10:00 UTC, server broker GMT+3 -> tick.time = 13:00
    wed_utc = datetime(2026, 11, 25, 10, 0, tzinfo=timezone.utc)
    fake.tick_time = int(wed_utc.timestamp()) + 3 * 3600 - 15
    assert client.calibrate_utc_offset_hours("XAUUSD", now_utc=wed_utc) == 3

    # Minggu 12:00 UTC (market tutup, tick masih dari Jumat malam) -> harus tetap +3!
    sun_utc = datetime(2026, 11, 29, 12, 0, tzinfo=timezone.utc)
    fake.tick_time = int(datetime(2026, 11, 27, 21, 55, tzinfo=timezone.utc).timestamp()) + 3 * 3600
    assert client.calibrate_utc_offset_hours("XAUUSD", now_utc=sun_utc) == 3


def test_idempotent_retry_stops_if_order_already_in_server(tmp_path: Path) -> None:
    """Jika order pertama timeout namun sudah masuk di server, retry kedua dibatalkan."""
    sm = StateManager(tmp_path / "state.db")
    fake = FakeMT5()
    now_utc = datetime(2026, 11, 25, 10, 0, tzinfo=timezone.utc)
    comment = make_unique_comment("SNP_", 1732528800)
    assert len(comment) <= 31

    # Simulasi: attempt 1 gagal koneksi, padahal order sudah nongol di server tanpa comment
    def fake_send_side_effect(req: dict):
        fake.send_calls.append(req)
        fake.orders.append(
            SimpleNamespace(
                ticket=777111,
                symbol="XAUUSD",
                type=fake.ORDER_TYPE_BUY_LIMIT,
                volume_initial=0.02,
                price_open=2645.50,
                time_setup=int(now_utc.timestamp()),
                magic=BotConfig().magic_number,
                comment="",  # Broker menghapus comment -> fallback multi-field harus mendeteksi!
            )
        )
        return SimpleNamespace(retcode=TRADE_RETCODE_CONNECTION, comment="Timeout")

    fake.order_send = fake_send_side_effect  # type: ignore[method-assign]
    client = MT5Client(BotConfig(), sm, mt5_module=fake, sleep_fn=lambda _: None)

    res = client.open_order(
        symbol="XAUUSD",
        order_type=fake.ORDER_TYPE_BUY_LIMIT,
        volume=0.02,
        price=2645.50,
        sl=2640.00,
        tp=2673.00,
        comment=comment,
        offset_hours=0,
        now_utc=now_utc,
    )
    assert res.success is True
    assert res.already_existed is True
    assert res.ticket == 777111
    assert len(fake.send_calls) == 1  # Tidak mengirim order ke-2!


def test_retry_on_requote_recovers(tmp_path: Path) -> None:
    """Requote diulang maksimal 5x lalu sukses jika percobaan berikutnya berhasil."""
    sm = StateManager(tmp_path / "state.db")
    fake = FakeMT5()
    fake.send_outcomes = [
        SimpleNamespace(retcode=TRADE_RETCODE_REQUOTE, comment="Requote 1"),
        SimpleNamespace(retcode=TRADE_RETCODE_REQUOTE, comment="Requote 2"),
        SimpleNamespace(retcode=TRADE_RETCODE_PLACED, order=555, comment="Placed"),
    ]
    slept: list[float] = []
    client = MT5Client(BotConfig(), sm, mt5_module=fake, sleep_fn=slept.append)

    res = client.open_order(
        symbol="XAUUSD",
        order_type=fake.ORDER_TYPE_BUY_LIMIT,
        volume=0.02,
        price=2645.0,
        sl=2640.0,
        tp=2670.0,
        comment="SNP_123",
    )
    assert res.success is True
    assert res.ticket == 555
    assert res.attempts == 3
    assert slept == [0.5, 0.5]


def test_synthetic_dxy_6_pair_and_fallback_5_pair_alignment() -> None:
    """DXY sintetis menyelaraskan timestamp dan fallback 5 pair tetap berkorelasi sangat tinggi."""
    times = pd.date_range("2026-01-01", periods=60, freq="4h", tz="UTC")
    rng = np.random.default_rng(42)

    def make_pair_df(base: float, drop_idx: int | None = None) -> pd.DataFrame:
        ret = rng.normal(0.0, 0.003, size=len(times))
        close = base * np.exp(np.cumsum(ret))
        df = pd.DataFrame(
            {
                "time_utc": times,
                "open": close * 0.999,
                "high": close * 1.001,
                "low": close * 0.998,
                "close": close,
            }
        )
        if drop_idx is not None:
            df = df.drop(index=drop_idx).reset_index(drop=True)
        return df

    frames_6 = {
        "EURUSD": make_pair_df(1.08),
        "USDJPY": make_pair_df(150.0, drop_idx=10),  # Sengaja hilangkan 1 bar untuk uji inner-join
        "GBPUSD": make_pair_df(1.26),
        "USDCAD": make_pair_df(1.38),
        "USDSEK": make_pair_df(10.5),
        "USDCHF": make_pair_df(0.88),
    }
    res6 = compute_synthetic_dxy_from_frames(frames_6)
    assert res6.source_mode == "SYNTHETIC_ICE_6"
    assert len(res6.df) == 59  # Bar hilang pada USDJPY diselaraskan via inner join

    frames_5 = {k: v for k, v in frames_6.items() if k != "USDSEK"}
    res5 = compute_synthetic_dxy_from_frames(frames_5)
    assert res5.source_mode == "SYNTHETIC_FALLBACK_5"
    assert res5.missing_symbols == ("USDSEK",)

    ret6 = np.log(res6.df["close"]).diff().dropna()
    ret5 = np.log(res5.df["close"]).diff().dropna()
    corr = float(ret6.corr(ret5))
    assert corr > 0.98


def test_state_machine_in_position_immune_to_session_reset(tmp_path: Path) -> None:
    """IN_POSITION dan COOLDOWN kebal terhadap reset sesi/zona pra-posisi."""
    sm = StateManager(tmp_path / "state.db")
    sm.save_strategy_state(StrategyState(state="ZONE_ARMED", zone_id="Z1"))
    assert sm.reset_pre_position_to_idle().state == "IDLE"

    sm.save_strategy_state(
        StrategyState(state="IN_POSITION", position_ticket=888, position_entry=2650.0)
    )
    after = sm.reset_pre_position_to_idle()
    assert after.state == "IN_POSITION"
    assert after.position_ticket == 888


def test_guardrails_circuit_breaker_utc_resets_and_funnel(tmp_path: Path) -> None:
    """Uji isNewBar, spread, cooldown, daily 2x SL, weekly 6%, monthly 10% & reset UTC."""
    sm = StateManager(tmp_path / "state.db")
    guard = AntiGanasGuard(sm, BotConfig())

    # 1. isNewBar
    assert guard.check_is_new_bar(1000).allowed is True
    guard.mark_bar_evaluated(1000)
    assert guard.check_is_new_bar(1000).allowed is False
    assert guard.check_is_new_bar(1300).allowed is True

    # 2. Spread guard
    mon = datetime(2026, 11, 23, 10, 0, tzinfo=timezone.utc)  # Senin UTC
    dec_spread = guard.check_pre_entry_gates(mon, 1000.0, 1000.0, 0.50, [], [])
    assert dec_spread.allowed is False
    assert dec_spread.gate == "MAX_SPREAD"

    # 3. Cooldown 60 menit
    dec_cd = guard.check_pre_entry_gates(
        mon,
        1000.0,
        1000.0,
        0.20,
        [],
        [],
        last_closed_trade_utc=mon - timedelta(minutes=30),
    )
    assert dec_cd.allowed is False
    assert dec_cd.gate == "COOLDOWN"

    # 4. Daily 2x SL -> halt sampai 00:00 UTC hari berikutnya
    sm.sync_risk_periods(mon, 1000.0, 1000.0)
    sm.record_closed_trade_pnl(mon, -15.0, 985.0, 985.0, is_stop_loss=True)
    sm.record_closed_trade_pnl(mon, -15.0, 970.0, 970.0, is_stop_loss=True)
    dec_daily = guard.check_pre_entry_gates(mon + timedelta(hours=2), 970.0, 970.0, 0.20, [], [])
    assert dec_daily.allowed is False
    assert dec_daily.gate == "DAILY_CIRCUIT_BREAKER"

    # Hari Selasa 00:05 UTC -> daily SL reset otomatis!
    tue = datetime(2026, 11, 24, 0, 5, tzinfo=timezone.utc)
    dec_tue = guard.check_pre_entry_gates(tue, 970.0, 970.0, 0.20, [], [])
    assert dec_tue.allowed is True
    assert dec_tue.effective_risk_pct == 1.5

    # 5. Weekly loss >= 6% (tersebar di Selasa & Rabu agar tidak kena daily 3%) -> risiko dipotong jadi 0.75%
    sm.record_closed_trade_pnl(tue, -20.0, 950.0, 950.0, is_stop_loss=False)
    wed = datetime(2026, 11, 25, 10, 0, tzinfo=timezone.utc)
    sm.record_closed_trade_pnl(wed, -15.0, 935.0, 935.0, is_stop_loss=False)  # total week loss = -65 / 1000 = 6.5%
    dec_wk = guard.check_pre_entry_gates(wed + timedelta(hours=1), 935.0, 935.0, 0.20, [], [])
    assert dec_wk.allowed is True
    assert dec_wk.effective_risk_pct == 0.75

    next_mon = datetime(2026, 11, 30, 0, 5, tzinfo=timezone.utc)
    dec_next_mon = guard.check_pre_entry_gates(next_mon, 935.0, 935.0, 0.20, [], [])
    assert dec_next_mon.effective_risk_pct == 1.5

    # 6. Monthly DD >= 10% -> HALT permanen bulan itu (PEAK_BALANCE=1000 -> equity 895 = 10.5% DD)
    dec_mo = guard.check_pre_entry_gates(next_mon + timedelta(hours=1), 895.0, 895.0, 0.20, [], [])
    assert dec_mo.allowed is False
    assert dec_mo.gate == "MONTHLY_CIRCUIT_BREAKER"

    # 7. Funnel counters & format
    sm.increment_funnel(mon, "total_m5_bars", 10)
    sm.increment_funnel(mon, "passed_session_news", 5)
    report = format_funnel_report("2026-W48", sm.get_funnel_stats(mon))
    assert "Total bar M5 dievaluasi : 10" in report
    assert "Lolos sesi + news       : 5" in report
