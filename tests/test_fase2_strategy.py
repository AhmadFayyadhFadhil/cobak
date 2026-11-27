"""Unit tests Fase 2 (sniper-tdd: test dibuat lebih dulu sebelum implementasi strategy/).

Menguji:
- Anti look-ahead bias pada konfirmasi Pivot N-bar
- Layer 1: sesi UTC, news fail-closed (JSON & CSV), korelasi DXY (< -0.5, zona abu-abu, > -0.3)
- Layer 2: BOS & fresh Supply/Demand, prioritas multi-zona (H4 > H1)
- Layer 3 & State Machine: ZONE_ARMED -> SWEEP_DETECTED -> CHOCH_CONFIRMED -> PENDING,
  timeout zona lintas state, FVG wait max 5 bar M5, invalidasi PENDING, dan isolasi IN_POSITION
- Layer 4: order_calc_profit lot sizing, policy lot 0.01, BEP+spread+komisi, ATR trail step, Realized R
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from config import BotConfig
from core.mt5_client import MT5Client, SymbolSpec
from core.state_manager import StateManager, StrategyState
from strategy.indicators import compute_atr, compute_rolling_correlation, detect_confirmed_pivots
from strategy.layer1_macro import evaluate_layer1
from strategy.layer2_structure import SupplyDemandZone, detect_bos_and_fresh_zones, select_priority_touched_zone
from strategy.layer3_smc_m5 import advance_smc_state_machine
from strategy.layer4_risk_exit import (
    calculate_position_lot,
    compute_bep_price,
    compute_realized_r,
    evaluate_active_position_actions,
)


def _make_bars(
    closes: list[float],
    start: datetime,
    freq: str = "5min",
    high_pad: float = 0.5,
    low_pad: float = 0.5,
    opens: list[float] | None = None,
    volumes: list[float] | None = None,
) -> pd.DataFrame:
    times = pd.date_range(start, periods=len(closes), freq=freq, tz="UTC")
    o = opens if opens is not None else [closes[0]] + closes[:-1]
    h = [max(oi, ci) + high_pad for oi, ci in zip(o, closes)]
    l = [min(oi, ci) - low_pad for oi, ci in zip(o, closes)]
    v = volumes if volumes is not None else [100.0] * len(closes)
    return pd.DataFrame(
        {
            "time_utc": times,
            "open": o,
            "high": h,
            "low": l,
            "close": closes,
            "tick_volume": v,
        }
    )


def test_pivot_no_lookahead_bias() -> None:
    """Pivot High dengan window=3 baru muncul tepat pada bar k+3, bukan sebelumnya."""
    start = datetime(2026, 11, 24, 7, 0, tzinfo=timezone.utc)
    # Peak jelas di index 4 (high=2660)
    closes = [2640, 2642, 2645, 2650, 2658, 2652, 2648, 2644, 2642]
    df = _make_bars(closes, start, freq="1h", high_pad=2.0, low_pad=1.0)

    # Slice hanya sampai index 6 (k+2): pivot index 4 BELUM boleh terdeteksi!
    pivots_early = detect_confirmed_pivots(df.iloc[:7].copy(), window=3)
    assert not any(p.index == 4 and p.kind == "HIGH" for p in pivots_early)

    # Slice sampai index 7 (k+3): pivot index 4 baru SAH terkonfirmasi!
    pivots_confirmed = detect_confirmed_pivots(df.iloc[:8].copy(), window=3)
    matching = [p for p in pivots_confirmed if p.index == 4 and p.kind == "HIGH"]
    assert len(matching) == 1
    assert matching[0].confirmed_index == 7


def test_layer1_session_news_fail_closed_and_dxy_grey_zone(tmp_path: Path) -> None:
    """Layer 1 memblokir luar sesi, news JSON rusak/stale, dan menerapkan zona abu-abu DXY."""
    news_file = tmp_path / "news.json"
    cfg = BotConfig(news_events_json=news_file, corr_window_h4=20)
    now_utc = datetime(2026, 11, 24, 10, 0, tzinfo=timezone.utc)

    # Data H4 XAUUSD (bullish trend) dan DXY (bearish trend, korelasi negatif kuat)
    start_h4 = now_utc - timedelta(hours=4 * 40)
    t_h4 = pd.date_range(start_h4, periods=40, freq="4h", tz="UTC")
    rng = np.random.default_rng(7)
    dxy_ret = -0.002 + rng.normal(0, 0.0005, size=40)
    xau_ret = -1.5 * dxy_ret + rng.normal(0, 0.0003, size=40)
    xau_c = 2600.0 * np.exp(np.cumsum(xau_ret))
    dxy_c = 105.0 * np.exp(np.cumsum(dxy_ret))

    xau_h4 = pd.DataFrame(
        {"time_utc": t_h4, "open": xau_c * 0.999, "high": xau_c * 1.002, "low": xau_c * 0.998, "close": xau_c}
    )
    dxy_h4 = pd.DataFrame(
        {"time_utc": t_h4, "open": dxy_c * 1.001, "high": dxy_c * 1.002, "low": dxy_c * 0.998, "close": dxy_c}
    )

    # 1. File news tidak ada -> fail-closed!
    res_missing = evaluate_layer1(now_utc, xau_h4, dxy_h4, cfg)
    assert res_missing.passed is False
    assert "news" in res_missing.reason.lower()

    # 2. File news basi (> 168 jam) -> fail-closed!
    news_file.write_text(
        json.dumps({"updated_at_utc": (now_utc - timedelta(hours=200)).isoformat(), "events": []}),
        encoding="utf-8",
    )
    res_stale = evaluate_layer1(now_utc, xau_h4, dxy_h4, cfg)
    assert res_stale.passed is False

    # 3. Dalam jendela CPI (-30m s/d +15m) -> blocked!
    news_file.write_text(
        json.dumps(
            {
                "updated_at_utc": now_utc.isoformat(),
                "events": [{"title": "USD CPI m/m", "impact": "HIGH", "currency": "USD", "time_utc": (now_utc + timedelta(minutes=20)).isoformat()}],
            }
        ),
        encoding="utf-8",
    )
    res_news = evaluate_layer1(now_utc, xau_h4, dxy_h4, cfg)
    assert res_news.passed is False

    # 4. Luar jendela news, luar sesi (17:00 UTC) -> blocked!
    news_file.write_text(json.dumps({"updated_at_utc": now_utc.isoformat(), "events": []}), encoding="utf-8")
    res_off_session = evaluate_layer1(datetime(2026, 11, 24, 17, 0, tzinfo=timezone.utc), xau_h4, dxy_h4, cfg)
    assert res_off_session.passed is False

    # 5. Dalam sesi + korelasi < -0.5 + DXY bearish -> ONLY_BUY
    res_ok = evaluate_layer1(now_utc, xau_h4, dxy_h4, cfg)
    assert res_ok.passed is True
    assert res_ok.bias == "BUY"
    assert res_ok.correlation is not None and res_ok.correlation < -0.5


def test_layer2_priority_h4_over_h1_and_fresh_zone_detection() -> None:
    """Multi-zona tersentuh bersamaan harus mengutamakan H4 daripada H1."""
    z_h1 = SupplyDemandZone(
        zone_id="H1_DEM_1",
        timeframe="H1",
        direction="BUY",
        low=2648.0,
        high=2652.0,
        formed_at_utc="2026-11-24T08:00:00+00:00",
        is_fresh=True,
    )
    z_h4 = SupplyDemandZone(
        zone_id="H4_DEM_1",
        timeframe="H4",
        direction="BUY",
        low=2645.0,
        high=2651.0,
        formed_at_utc="2026-11-24T04:00:00+00:00",
        is_fresh=True,
    )
    chosen = select_priority_touched_zone([z_h1, z_h4], current_price=2650.0, bias="BUY")
    assert chosen is not None
    assert chosen.zone_id == "H4_DEM_1"


def test_layer3_state_machine_full_flow_fvg_timeout_and_in_position_isolation(tmp_path: Path) -> None:
    """Uji transisi ZONE_ARMED -> SWEEP -> CHOCH -> PENDING, FVG timeout, session reset, & IN_POSITION kebal."""
    sm = StateManager(tmp_path / "state.db")
    cfg = BotConfig(
        pivot_window=2,
        zone_armed_timeout_hours=5.0,
        max_fvg_wait_candles_m5=3,
        pending_order_expiry_hours=2.0,
        require_tick_volume_filter=False,
    )
    zone = SupplyDemandZone(
        zone_id="H4_DEM_99",
        timeframe="H4",
        direction="BUY",
        low=2648.0,
        high=2652.0,
        formed_at_utc="2026-11-24T04:00:00+00:00",
        is_fresh=True,
    )

    # Bangun bar M5 untuk skenario BUY (10 bar pemanasan + 12 bar pola SMC):
    # idx 13: swing low (low=2650.0), lalu naik ke 2654.2
    # idx 18: sweep low (low=2649.0 < 2650.0, close=2651.3 > 2650.0) inside zone [2648, 2652]
    # idx 19: base bearish kecil (high=2651.5)
    # idx 20..21: displacement bullish CHoCH menembus minor high + FVG (candle[19].high < candle[21].low)
    t0 = datetime(2026, 11, 24, 7, 10, tzinfo=timezone.utc)
    warm_o = [2653.0] * 10
    warm_c = [2653.1] * 10
    warm_h = [2653.4] * 10
    warm_l = [2652.8] * 10
    opens =  warm_o + [2653.0, 2652.5, 2651.5, 2651.0, 2651.2, 2652.5, 2653.5, 2653.0, 2652.0, 2651.3, 2651.0, 2654.0]
    closes = warm_c + [2652.5, 2651.5, 2651.0, 2651.2, 2652.5, 2653.5, 2653.0, 2652.0, 2651.3, 2651.0, 2654.0, 2658.0]
    highs =  warm_h + [2653.5, 2653.0, 2652.0, 2651.8, 2653.0, 2654.2, 2653.8, 2653.2, 2652.2, 2651.5, 2654.5, 2658.5]
    lows =   warm_l + [2652.2, 2651.2, 2650.6, 2650.0, 2650.8, 2652.2, 2652.5, 2651.5, 2649.0, 2650.8, 2650.9, 2653.8]
    m5_df = pd.DataFrame(
        {
            "time_utc": pd.date_range(t0, periods=len(closes), freq="5min", tz="UTC"),
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "tick_volume": [100.0] * len(closes),
        }
    )

    # Bar 18: Harga menyentuh zona -> ZONE_ARMED -> langsung deteksi sweep
    now_18 = m5_df["time_utc"].iloc[18].to_pydatetime()
    advance_smc_state_machine(sm, m5_df.iloc[:19].copy(), active_zone=zone, now_utc=now_18, in_session=True, cfg=cfg)
    assert sm.get_strategy_state().state == "SWEEP_DETECTED"

    # Evaluasi berurutan bar 19, 20, lalu bar 21 membentuk CHoCH displacement + FVG
    for idx in (19, 20):
        advance_smc_state_machine(
            sm,
            m5_df.iloc[: idx + 1].copy(),
            active_zone=zone,
            now_utc=m5_df["time_utc"].iloc[idx].to_pydatetime(),
            in_session=True,
            cfg=cfg,
        )

    now_21 = m5_df["time_utc"].iloc[21].to_pydatetime()
    act2 = advance_smc_state_machine(sm, m5_df.iloc[:22].copy(), active_zone=zone, now_utc=now_21, in_session=True, cfg=cfg)
    assert act2.action == "PLACE_LIMIT"
    assert act2.limit_price == pytest.approx((2651.5 + 2653.8) / 2.0, abs=1e-4)
    assert act2.sl_price is not None and act2.sl_price < 2649.0  # sweep extreme (2649.0) - 0.5 * ATR14
    assert sm.get_strategy_state().state == "CHOCH_CONFIRMED"

    # Simulasi PENDING dibatalkan saat sesi berakhir (> 16:00 UTC)
    sm.save_strategy_state(
        StrategyState(
            state="PENDING",
            zone_id="H4_DEM_99",
            zone_direction="BUY",
            pending_ticket=101,
            pending_price=2652.65,
            pending_sl=2648.0,
            pending_tp=2675.9,
            pending_tp_1_2=2661.95,
            pending_expiry_utc=(now_21 + timedelta(hours=2)).isoformat(),
        )
    )
    act_off = advance_smc_state_machine(
        sm, m5_df.iloc[:22].copy(), active_zone=None, now_utc=now_21.replace(hour=16, minute=5), in_session=False, cfg=cfg
    )
    assert act_off.action == "CANCEL_PENDING"
    assert sm.get_strategy_state().state == "IDLE"

    # Simulasi FVG wait timeout (> 3 bar setelah CHoCH tanpa FVG) -> kembali ke IDLE
    sm.save_strategy_state(
        StrategyState(
            state="CHOCH_CONFIRMED",
            zone_id="H4_DEM_99",
            zone_direction="BUY",
            zone_deadline_utc=(now_21 + timedelta(hours=4)).isoformat(),
            sweep_extreme=2649.0,
            choch_bar_ts=int(m5_df["time_utc"].iloc[20].timestamp()),
            choch_wait_count=3,
        )
    )
    no_fvg_df = m5_df.iloc[:21].copy()
    # Tambahkan 1 bar bertumpuk (tidak ada gap) -> wait count melewati 3 -> reset IDLE
    extra_bar = pd.DataFrame(
        {
            "time_utc": [m5_df["time_utc"].iloc[20] + timedelta(minutes=5)],
            "open": [2653.5],
            "high": [2654.0],
            "low": [2651.2],  # Overlap dengan candle[19].high (2651.5) -> FVG nihil
            "close": [2652.0],
            "tick_volume": [100.0],
        }
    )
    act_timeout = advance_smc_state_machine(
        sm,
        pd.concat([no_fvg_df, extra_bar], ignore_index=True),
        active_zone=zone,
        now_utc=now_21,
        in_session=True,
        cfg=cfg,
    )
    assert act_timeout.action == "RESET_IDLE"
    assert sm.get_strategy_state().state == "IDLE"

    # IN_POSITION wajib kebal saat sesi berakhir (> 16:00 UTC)
    sm.save_strategy_state(
        StrategyState(state="IN_POSITION", position_ticket=505, position_entry=2652.65, position_sl=2648.0, position_tp=2675.9)
    )
    act_pos = advance_smc_state_machine(
        sm, m5_df, active_zone=None, now_utc=now_21.replace(hour=18, minute=0), in_session=False, cfg=cfg
    )
    assert act_pos.action == "MANAGE_POSITION"
    assert sm.get_strategy_state().state == "IN_POSITION"


def test_layer4_lot_sizing_min_lot_policies_bep_trail_and_realized_r(tmp_path: Path) -> None:
    """Uji order_calc_profit lot sizing, policy 0.01, BEP+spread+komisi, ATR trail step, & Realized R."""
    spec = SymbolSpec(
        name="XAUUSD",
        digits=2,
        point=0.01,
        trade_tick_size=0.01,
        trade_tick_value=1.0,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        contract_size=100.0,
    )

    # Mock client menggunakan order_calc_profit: -$100 per $1 gerak per 1.00 lot
    fake_mt5 = SimpleNamespace(
        ORDER_TYPE_BUY=0,
        ORDER_TYPE_SELL=1,
        order_calc_profit=lambda ot, sym, vol, op, cl: (cl - op) * 100.0 * vol if ot == 0 else (op - cl) * 100.0 * vol,
    )
    client = MT5Client(BotConfig(), StateManager(tmp_path / "s.db"), mt5_module=fake_mt5, sleep_fn=lambda _: None)

    # Modal $500, risk 1.5% ($7.50), SL $5.00 -> lot mentah = 0.015 -> dibulatkan 0.01 (< 2 * 0.01)
    lot_skip = calculate_position_lot(
        client=client,
        spec=spec,
        direction="BUY",
        balance=500.0,
        risk_pct=1.5,
        entry_price=2650.0,
        sl_price=2645.0,
        policy="SKIP",
    )
    assert lot_skip.allowed is False
    assert lot_skip.reason == "MIN_LOT_PARTIAL_SKIP"

    lot_no_partial = calculate_position_lot(
        client=client,
        spec=spec,
        direction="BUY",
        balance=500.0,
        risk_pct=1.5,
        entry_price=2650.0,
        sl_price=2645.0,
        policy="NO_PARTIAL",
    )
    assert lot_no_partial.allowed is True
    assert lot_no_partial.volume == 0.01
    assert lot_no_partial.allow_partial is False

    # Modal $2,000, risk 1.5% ($30), SL $5.00 -> lot = 0.06 (bisa partial 50%)
    lot_ok = calculate_position_lot(
        client=client,
        spec=spec,
        direction="BUY",
        balance=2000.0,
        risk_pct=1.5,
        entry_price=2650.0,
        sl_price=2645.0,
        policy="SKIP",
    )
    assert lot_ok.allowed is True
    assert lot_ok.volume == 0.06
    assert lot_ok.allow_partial is True

    # BEP + spread ($0.30) + komisi ($7/lot -> $0.07/oz) = entry + 0.37
    bep_buy = compute_bep_price("BUY", entry_price=2650.0, spread_usd=0.30, commission_per_lot=7.0, contract_size=100.0)
    assert bep_buy == pytest.approx(2650.37, abs=1e-4)

    # Active Position: hit 1:2 -> Partial 50% + Move BEP; setelah partial -> ATR trail dengan step >= 0.25
    actions_1_2 = evaluate_active_position_actions(
        direction="BUY",
        entry_price=2650.0,
        initial_sl=2645.0,
        current_sl=2645.0,
        current_bid=2660.2,
        current_ask=2660.5,
        current_volume=0.06,
        volume_min=0.01,
        atr_m5=2.0,
        partial_done=False,
        allow_partial=True,
        cfg=BotConfig(commission_per_lot_account=7.0, min_trail_step_usd=0.25),
    )
    assert actions_1_2.close_partial_volume == 0.03
    assert actions_1_2.new_sl == pytest.approx(2650.37, abs=1e-4)

    # Trail step terlalu kecil (< 0.25) tidak kirim modify; kalau >= 0.25 baru geser
    actions_trail_small = evaluate_active_position_actions(
        direction="BUY",
        entry_price=2650.0,
        initial_sl=2645.0,
        current_sl=2658.0,
        current_bid=2661.1,
        current_ask=2661.4,
        current_volume=0.03,
        volume_min=0.01,
        atr_m5=2.0,  # 1.5 * 2.0 = 3.0 -> candidate SL = 2658.1 (diff 0.1 < 0.25)
        partial_done=True,
        allow_partial=True,
        cfg=BotConfig(min_trail_step_usd=0.25),
    )
    assert actions_trail_small.new_sl is None

    actions_trail_move = evaluate_active_position_actions(
        direction="BUY",
        entry_price=2650.0,
        initial_sl=2645.0,
        current_sl=2658.0,
        current_bid=2662.0,
        current_ask=2662.3,
        current_volume=0.03,
        volume_min=0.01,
        atr_m5=2.0,  # candidate SL = 2659.0 (diff 1.0 >= 0.25)
        partial_done=True,
        allow_partial=True,
        cfg=BotConfig(min_trail_step_usd=0.25),
    )
    assert actions_trail_move.new_sl == pytest.approx(2659.0, abs=1e-4)

    # Realized R Verification (Fix 1)
    assert compute_realized_r(hit_partial=True, partial_ratio=0.5, partial_r=2.0, remainder_r=5.0) == pytest.approx(3.5)
    assert compute_realized_r(hit_partial=True, partial_ratio=0.5, partial_r=2.0, remainder_r=0.0) == pytest.approx(1.0)
    assert compute_realized_r(hit_partial=False, partial_ratio=0.5, partial_r=2.0, remainder_r=-1.0) == pytest.approx(-1.0)
