"""Layer 3: Konfirmasi Entry Presisi SMC pada M5 & State Machine persisten SQLite.

Alur State:
  IDLE -> ZONE_ARMED -> SWEEP_DETECTED -> CHOCH_CONFIRMED -> PENDING -> IN_POSITION -> COOLDOWN -> IDLE

Aturan kritis:
- Timer ZONE_ARMED berjalan terus melintasi SWEEP_DETECTED & CHOCH_CONFIRMED.
- CHOCH_CONFIRMED tanpa FVG menunggu maksimal `max_fvg_wait_candles_m5` lalu reset IDLE.
- Reset sesi/zona hanya berlaku sebelum IN_POSITION; IN_POSITION kebal reset sesi.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

import pandas as pd

from config import CONFIG, BotConfig
from core.state_manager import PRE_POSITION_STATES, StateManager, StrategyState
from strategy.indicators import compute_atr
from strategy.layer2_structure import SupplyDemandZone

ActionType = Literal[
    "WAIT",
    "RESET_IDLE",
    "CANCEL_PENDING",
    "PLACE_LIMIT",
    "MANAGE_POSITION",
]


@dataclass(frozen=True)
class SMCAction:
    """Instruksi hasil evaluasi State Machine M5 untuk dieksekusi oleh runner/backtester."""

    action: ActionType
    direction: Optional[Literal["BUY", "SELL"]] = None
    limit_price: Optional[float] = None
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    tp_1_2_price: Optional[float] = None
    fvg_low: Optional[float] = None
    fvg_high: Optional[float] = None
    reason: str = ""


def _detect_liquidity_sweep(
    m5_df: pd.DataFrame, direction: str, lookback: int = 12
) -> Optional[float]:
    """Cek apakah candle M5 terakhir melakukan sweep Swing Low/High lalu close kembali."""
    if len(m5_df) < lookback + 2:
        return None
    prior = m5_df.iloc[-(lookback + 1) : -1]
    last = m5_df.iloc[-1]

    if direction == "BUY":
        swing_low = float(prior["low"].min())
        # Tembus sedikit swing low namun close kembali di atas swing low (meninggalkan wick bawah)
        if float(last["low"]) < swing_low and float(last["close"]) > swing_low:
            return float(last["low"])
    else:
        swing_high = float(prior["high"].max())
        if float(last["high"]) > swing_high and float(last["close"]) < swing_high:
            return float(last["high"])
    return None


def _detect_choch_displacement(
    m5_df: pd.DataFrame,
    direction: str,
    cfg: BotConfig,
) -> Optional[int]:
    """Deteksi CHoCH minor dengan candle displacement (body >= 1.5x avg & opsional volume)."""
    avg_n = cfg.displacement_avg_period
    if len(m5_df) < max(avg_n + 1, 8):
        return None

    last = m5_df.iloc[-1]
    hist = m5_df.iloc[-(avg_n + 1) : -1]
    avg_body = float((hist["close"] - hist["open"]).abs().mean())
    body = abs(float(last["close"]) - float(last["open"]))
    if avg_body <= 0 or body < (cfg.displacement_body_multiplier * avg_body):
        return None

    if cfg.require_tick_volume_filter and "tick_volume" in m5_df.columns:
        avg_vol = float(hist["tick_volume"].mean())
        if float(last["tick_volume"]) <= avg_vol:
            return None

    minor_window = m5_df.iloc[-8:-1]
    if direction == "BUY":
        minor_high = float(minor_window["high"].max())
        if float(last["close"]) > float(last["open"]) and float(last["close"]) > minor_high:
            return int(pd.Timestamp(last["time_utc"]).timestamp())
    else:
        minor_low = float(minor_window["low"].min())
        if float(last["close"]) < float(last["open"]) and float(last["close"]) < minor_low:
            return int(pd.Timestamp(last["time_utc"]).timestamp())
    return None


def _detect_fvg(
    m5_df: pd.DataFrame, direction: str, choch_bar_ts: int
) -> Optional[tuple[float, float]]:
    """Cari Fair Value Gap 3-candle pada atau setelah candle CHoCH."""
    if len(m5_df) < 3:
        return None
    ts_series = m5_df["time_utc"].map(lambda x: int(pd.Timestamp(x).timestamp()))
    # Periksa 3-candle triplet terbaru yang mengandung atau setelah choch_bar_ts
    for idx in range(len(m5_df) - 1, 1, -1):
        if int(ts_series.iloc[idx]) < choch_bar_ts:
            break
        c1 = m5_df.iloc[idx - 2]
        c3 = m5_df.iloc[idx]
        if direction == "BUY":
            if float(c1["high"]) < float(c3["low"]):
                return float(c1["high"]), float(c3["low"])
        else:
            if float(c1["low"]) > float(c3["high"]):
                return float(c3["high"]), float(c1["low"])
    return None


def _is_zone_broken_by_close(
    last_close: float, direction: Optional[str], z_low: Optional[float], z_high: Optional[float]
) -> bool:
    if direction is None or z_low is None or z_high is None:
        return False
    if direction == "BUY":
        return last_close < z_low
    return last_close > z_high


def advance_smc_state_machine(
    state_manager: StateManager,
    m5_df: pd.DataFrame,
    active_zone: Optional[SupplyDemandZone],
    now_utc: datetime,
    in_session: bool,
    cfg: BotConfig = CONFIG,
) -> SMCAction:
    """Evaluasi dan majukan State Machine Layer 3 satu langkah (per M5 candle close)."""
    now_utc = now_utc.astimezone(timezone.utc)
    st = state_manager.get_strategy_state()

    # 1. IN_POSITION wajib kebal terhadap reset sesi maupun invalidasi zona
    if st.state == "IN_POSITION":
        return SMCAction(
            action="MANAGE_POSITION",
            direction="SELL" if st.zone_direction == "SELL" else "BUY",
            reason="IN_POSITION aktif (kebal reset sesi/zona).",
        )

    # 2. COOLDOWN: tunggu hingga cooldown_until_utc lewat
    if st.state == "COOLDOWN":
        if st.cooldown_until_utc:
            cd_until = datetime.fromisoformat(st.cooldown_until_utc).astimezone(timezone.utc)
            if now_utc < cd_until:
                return SMCAction(action="WAIT", reason="Cooldown pasca-close masih aktif.")
        st = StrategyState(state="IDLE")
        state_manager.save_strategy_state(st)

    last_close = float(m5_df["close"].iloc[-1]) if not m5_df.empty else 0.0
    last_high = float(m5_df["high"].iloc[-1]) if not m5_df.empty else 0.0
    last_low = float(m5_df["low"].iloc[-1]) if not m5_df.empty else 0.0

    # 3. Reset pra-posisi jika sesi berakhir (> 16:00 UTC)
    if st.state in PRE_POSITION_STATES and not in_session:
        was_pending = st.state == "PENDING"
        state_manager.save_strategy_state(StrategyState(state="IDLE"))
        if was_pending:
            return SMCAction(
                action="CANCEL_PENDING",
                reason="Sesi aktif berakhir (>16:00 UTC): pending limit dibatalkan.",
            )
        return SMCAction(
            action="RESET_IDLE",
            reason="Sesi aktif berakhir (>16:00 UTC): state pra-posisi di-reset ke IDLE.",
        )

    # 4. Kelola state PENDING (expiry, hit 1:2 duluan, atau ekstrem sweep tertembus)
    if st.state == "PENDING":
        expired = False
        if st.pending_expiry_utc:
            exp_dt = datetime.fromisoformat(st.pending_expiry_utc).astimezone(timezone.utc)
            expired = now_utc >= exp_dt
        hit_1_2 = False
        sl_broken = False
        if st.zone_direction == "BUY":
            if st.pending_tp_1_2 is not None and last_high >= st.pending_tp_1_2:
                hit_1_2 = True
            if st.pending_sl is not None and last_low <= st.pending_sl:
                sl_broken = True
        else:
            if st.pending_tp_1_2 is not None and last_low <= st.pending_tp_1_2:
                hit_1_2 = True
            if st.pending_sl is not None and last_high >= st.pending_sl:
                sl_broken = True

        if expired or hit_1_2 or sl_broken:
            state_manager.save_strategy_state(StrategyState(state="IDLE"))
            why = "expiry habis" if expired else ("hit 1:2 duluan" if hit_1_2 else "sweep/SL tertembus")
            return SMCAction(action="CANCEL_PENDING", reason=f"Pending limit dibatalkan ({why}).")
        return SMCAction(action="WAIT", reason="Menunggu limit order terjemput.")

    # 5. Cek timer ZONE_ARMED & invalidasi zona lintas ZONE_ARMED / SWEEP_DETECTED / CHOCH_CONFIRMED
    if st.state in ("ZONE_ARMED", "SWEEP_DETECTED", "CHOCH_CONFIRMED"):
        if st.zone_deadline_utc:
            deadline = datetime.fromisoformat(st.zone_deadline_utc).astimezone(timezone.utc)
            if now_utc >= deadline:
                state_manager.save_strategy_state(StrategyState(state="IDLE"))
                return SMCAction(action="RESET_IDLE", reason="Timer ZONE_ARMED habis.")
        if not m5_df.empty and _is_zone_broken_by_close(
            last_close, st.zone_direction, st.zone_low, st.zone_high
        ):
            state_manager.save_strategy_state(StrategyState(state="IDLE"))
            return SMCAction(action="RESET_IDLE", reason="Zona SnD ditembus oleh close candle.")

    # 6. IDLE -> ZONE_ARMED jika in_session dan ada zona prioritas tersentuh
    if st.state == "IDLE":
        if not in_session or active_zone is None:
            return SMCAction(action="WAIT", reason="Menunggu harga menyentuh zona SnD fresh.")
        deadline = now_utc + timedelta(hours=cfg.zone_armed_timeout_hours)
        st = StrategyState(
            state="ZONE_ARMED",
            zone_id=active_zone.zone_id,
            zone_timeframe=active_zone.timeframe,
            zone_direction=active_zone.direction,
            zone_low=active_zone.low,
            zone_high=active_zone.high,
            zone_armed_at_utc=now_utc.isoformat(),
            zone_deadline_utc=deadline.isoformat(),
        )
        state_manager.save_strategy_state(st)
        state_manager.increment_funnel(now_utc, "zones_armed", 1)

    direction = st.zone_direction
    if direction not in ("BUY", "SELL") or m5_df.empty:
        return SMCAction(action="WAIT", reason="Menunggu data M5 valid.")

    # 7. ZONE_ARMED -> SWEEP_DETECTED
    if st.state == "ZONE_ARMED":
        sweep_ext = _detect_liquidity_sweep(m5_df, direction)
        if sweep_ext is None:
            return SMCAction(action="WAIT", reason="ZONE_ARMED: menunggu Liquidity Sweep M5.")
        st.state = "SWEEP_DETECTED"
        st.sweep_extreme = sweep_ext
        state_manager.save_strategy_state(st)
        state_manager.increment_funnel(now_utc, "sweeps_detected", 1)

    # 8. SWEEP_DETECTED -> CHOCH_CONFIRMED
    if st.state == "SWEEP_DETECTED":
        choch_ts = _detect_choch_displacement(m5_df, direction, cfg)
        if choch_ts is None:
            return SMCAction(
                action="WAIT",
                reason="SWEEP_DETECTED: menunggu CHoCH/MSS dengan displacement.",
            )
        st.state = "CHOCH_CONFIRMED"
        st.choch_bar_ts = choch_ts
        st.choch_wait_count = 0
        state_manager.save_strategy_state(st)
        state_manager.increment_funnel(now_utc, "choch_confirmed", 1)

    # 9. CHOCH_CONFIRMED -> Cari FVG dalam batas `max_fvg_wait_candles_m5` bar
    if st.state == "CHOCH_CONFIRMED":
        fvg = _detect_fvg(m5_df, direction, st.choch_bar_ts or 0)
        if fvg is None:
            st.choch_wait_count += 1
            if st.choch_wait_count > cfg.max_fvg_wait_candles_m5:
                state_manager.save_strategy_state(StrategyState(state="IDLE"))
                return SMCAction(
                    action="RESET_IDLE",
                    reason=(
                        f"CHoCH tidak diikuti FVG dalam {cfg.max_fvg_wait_candles_m5} candle M5."
                    ),
                )
            state_manager.save_strategy_state(st)
            return SMCAction(
                action="WAIT",
                reason=f"CHOCH_CONFIRMED: menunggu FVG ({st.choch_wait_count}/{cfg.max_fvg_wait_candles_m5}).",
            )

        fvg_low, fvg_high = fvg
        limit_price = (fvg_low + fvg_high) / 2.0
        atr_val = float(compute_atr(m5_df, cfg.atr_period).iloc[-1])
        sweep_ext = float(st.sweep_extreme if st.sweep_extreme is not None else limit_price)
        buffer = cfg.sl_atr_multiplier * atr_val

        if direction == "BUY":
            sl_price = sweep_ext - buffer
            risk_dist = limit_price - sl_price
            tp_1_2 = limit_price + cfg.partial_tp_rrr * risk_dist
            tp_final = limit_price + cfg.final_tp_rrr * risk_dist
        else:
            sl_price = sweep_ext + buffer
            risk_dist = sl_price - limit_price
            tp_1_2 = limit_price - cfg.partial_tp_rrr * risk_dist
            tp_final = limit_price - cfg.final_tp_rrr * risk_dist

        if risk_dist <= 0:
            state_manager.save_strategy_state(StrategyState(state="IDLE"))
            return SMCAction(action="RESET_IDLE", reason="Jarak SL tidak valid (<= 0).")

        st.fvg_low = fvg_low
        st.fvg_high = fvg_high
        state_manager.save_strategy_state(st)
        state_manager.increment_funnel(now_utc, "fvg_limits_placed", 1)

        return SMCAction(
            action="PLACE_LIMIT",
            direction=direction,
            limit_price=limit_price,
            sl_price=sl_price,
            tp_price=tp_final,
            tp_1_2_price=tp_1_2,
            fvg_low=fvg_low,
            fvg_high=fvg_high,
            reason="Setup Sniper lengkap: Limit 50% FVG siap dipasang.",
        )

    return SMCAction(action="WAIT", reason="Tidak ada transisi.")
