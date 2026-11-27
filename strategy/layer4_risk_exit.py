"""Layer 4: Dynamic Lot Sizing via mt5.order_calc_profit(), kebijakan lot minimum 0.01,
BEP + spread + komisi, Partial TP 50% @ RRR 1:2, ATR Trailing Stop, dan Realized R.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

from config import CONFIG, BotConfig, MinLotPolicy
from core.mt5_client import MT5Client, SymbolSpec


@dataclass(frozen=True)
class LotCalcResult:
    """Hasil kalkulasi lot dinamis beserta izin partial close."""

    allowed: bool
    volume: float
    allow_partial: bool
    risk_amount_account: float
    reason: str


@dataclass(frozen=True)
class PositionAction:
    """Aksi manajemen posisi aktif (partial close dan/atau geser SL)."""

    close_partial_volume: Optional[float]
    new_sl: Optional[float]
    reason: str


def _floor_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    steps = math.floor((value + 1e-12) / step)
    return round(steps * step, 8)


def calculate_position_lot(
    client: MT5Client,
    spec: SymbolSpec,
    direction: Literal["BUY", "SELL"],
    balance: float,
    risk_pct: float,
    entry_price: float,
    sl_price: float,
    policy: MinLotPolicy = CONFIG.min_lot_partial_policy,
) -> LotCalcResult:
    """Hitung lot berdasarkan risiko % saldo memakai mt5.order_calc_profit()."""
    risk_amount = max(0.0, balance * (risk_pct / 100.0))
    if risk_amount <= 0:
        return LotCalcResult(False, 0.0, False, 0.0, "INVALID_RISK_AMOUNT")

    order_type = (
        getattr(client.mt5, "ORDER_TYPE_BUY", 0)
        if direction == "BUY"
        else getattr(client.mt5, "ORDER_TYPE_SELL", 1)
    )
    pnl_per_1_lot = client.calculate_order_profit(
        order_type, spec.name, 1.0, entry_price, sl_price
    )
    if pnl_per_1_lot is None or pnl_per_1_lot >= 0:
        return LotCalcResult(False, 0.0, False, risk_amount, "INVALID_SL_PROFIT_CALC")

    loss_per_1_lot = abs(pnl_per_1_lot)
    raw_lot = risk_amount / loss_per_1_lot
    lot = _floor_to_step(raw_lot, spec.volume_step)
    lot = min(lot, spec.volume_max)

    if lot < spec.volume_min:
        return LotCalcResult(False, 0.0, False, risk_amount, "LOT_BELOW_VOLUME_MIN")

    # Cek apakah lot cukup untuk dibagi 50% pada Partial TP 1:2
    min_for_partial = _floor_to_step(2.0 * spec.volume_min, spec.volume_step)
    if lot < min_for_partial:
        if policy == "SKIP":
            return LotCalcResult(
                allowed=False,
                volume=0.0,
                allow_partial=False,
                risk_amount_account=risk_amount,
                reason="MIN_LOT_PARTIAL_SKIP",
            )
        return LotCalcResult(
            allowed=True,
            volume=lot,
            allow_partial=False,
            risk_amount_account=risk_amount,
            reason="OK_NO_PARTIAL_MIN_LOT",
        )

    return LotCalcResult(
        allowed=True,
        volume=lot,
        allow_partial=True,
        risk_amount_account=risk_amount,
        reason="OK",
    )


def compute_bep_price(
    direction: Literal["BUY", "SELL"],
    entry_price: float,
    spread_usd: float,
    commission_per_lot: float = 0.0,
    contract_size: float = 100.0,
) -> float:
    """Hitung harga BEP yang sudah menutup spread dan komisi round-turn broker."""
    commission_price_offset = (
        commission_per_lot / contract_size if contract_size > 0 else 0.0
    )
    total_buffer = max(0.0, spread_usd) + max(0.0, commission_price_offset)
    if direction == "BUY":
        return entry_price + total_buffer
    return entry_price - total_buffer


def evaluate_active_position_actions(
    direction: Literal["BUY", "SELL"],
    entry_price: float,
    initial_sl: float,
    current_sl: float,
    current_bid: float,
    current_ask: float,
    current_volume: float,
    volume_min: float,
    atr_m5: float,
    partial_done: bool,
    allow_partial: bool,
    cfg: BotConfig = CONFIG,
    volume_step: float = 0.01,
    contract_size: float = 100.0,
) -> PositionAction:
    """Evaluasi Partial TP 1:2, BEP + spread/komisi, dan ATR Trailing Stop."""
    spread_usd = max(0.0, current_ask - current_bid)
    # Harga exit riil: BUY exit di Bid, SELL exit di Ask
    exit_price = current_bid if direction == "BUY" else current_ask
    risk_dist = abs(entry_price - initial_sl)
    if risk_dist <= 0:
        return PositionAction(None, None, "INVALID_RISK_DIST")

    profit_dist = (
        (exit_price - entry_price) if direction == "BUY" else (entry_price - exit_price)
    )
    current_r = profit_dist / risk_dist

    # Fase 1: Capai RRR 1:2 -> tutup 50% (bila diizinkan) dan geser SL ke BEP + spread + komisi
    if not partial_done and current_r >= cfg.partial_tp_rrr:
        partial_vol: Optional[float] = None
        if allow_partial:
            candidate_vol = _floor_to_step(current_volume * cfg.partial_close_ratio, volume_step)
            if candidate_vol >= volume_min and (current_volume - candidate_vol) >= volume_min:
                partial_vol = candidate_vol

        bep = compute_bep_price(
            direction=direction,
            entry_price=entry_price,
            spread_usd=spread_usd,
            commission_per_lot=cfg.commission_per_lot_account,
            contract_size=contract_size,
        )
        bep_is_better = (bep > current_sl) if direction == "BUY" else (bep < current_sl)
        return PositionAction(
            close_partial_volume=partial_vol,
            new_sl=bep if bep_is_better else None,
            reason="HIT_1_2_PARTIAL_AND_BEP",
        )

    # Fase 2: Setelah melewati 1:2 (partial_done), jalankan ATR Trailing Stop bila multiplier > 0
    if partial_done and atr_m5 > 0 and cfg.trailing_atr_multiplier > 0:
        trail_dist = cfg.trailing_atr_multiplier * atr_m5
        min_step = (
            max(0.05, cfg.min_trail_step_atr_ratio * atr_m5)
            if cfg.min_trail_step_atr_ratio > 0
            else cfg.min_trail_step_usd
        )
        if direction == "BUY":
            candidate_sl = exit_price - trail_dist
            if candidate_sl > current_sl and (candidate_sl - current_sl) >= min_step:
                return PositionAction(None, candidate_sl, "ATR_TRAIL_UPDATE")
        else:
            candidate_sl = exit_price + trail_dist
            if candidate_sl < current_sl and (current_sl - candidate_sl) >= min_step:
                return PositionAction(None, candidate_sl, "ATR_TRAIL_UPDATE")

    return PositionAction(None, None, "HOLD")


def compute_realized_r(
    hit_partial: bool,
    partial_ratio: float,
    partial_r: float,
    remainder_r: float,
) -> float:
    """Hitung Realized R sesungguhnya (misal 0.5*2R + 0.5*5R = 3.5R, bukan 5R)."""
    if not hit_partial:
        return remainder_r
    return (partial_ratio * partial_r) + ((1.0 - partial_ratio) * remainder_r)
