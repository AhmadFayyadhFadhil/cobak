"""Gerbang pengaman Anti-Ganas:
1. Evaluasi per Candle Close (isNewBar M5)
2. Maksimal 1 Posisi / Pending Order aktif (Magic Number)
3. Max Spread Guard (berbasis USD distance, kebal 2 vs 3 digit)
4. Cooldown 60 menit pasca-close
5. Circuit Breaker 3 tingkat berbasis UTC (Harian, Mingguan, Bulanan)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Sequence

from config import CONFIG, BotConfig
from core.state_manager import RiskPeriodState, StateManager


@dataclass(frozen=True)
class GuardrailDecision:
    """Hasil evaluasi seluruh gerbang pengaman Anti-Ganas."""

    allowed: bool
    gate: str
    reason: str
    effective_risk_pct: float


def compute_monthly_drawdown_pct(
    period: RiskPeriodState, equity: float, mode: str
) -> float:
    """Hitung persentase drawdown bulanan sesuai MONTHLY_DD_MODE."""
    if mode == "START_BALANCE":
        ref = period.monthly_start_balance
    elif mode == "PEAK_BALANCE":
        ref = period.monthly_peak_balance
    elif mode == "PEAK_EQUITY":
        ref = period.monthly_peak_equity
    else:
        raise ValueError(f"MONTHLY_DD_MODE tidak dikenal: {mode}")

    if ref <= 0:
        return 0.0
    dd = ((ref - equity) / ref) * 100.0
    return max(0.0, dd)


class AntiGanasGuard:
    """Validator 5 gerbang pengaman sebelum mengizinkan evaluasi/pembukaan entry."""

    def __init__(self, state_manager: StateManager, cfg: BotConfig = CONFIG) -> None:
        self.state = state_manager
        self.cfg = cfg

    def check_is_new_bar(self, latest_closed_bar_ts_utc: int) -> GuardrailDecision:
        """Gate 1: Pastikan sinyal hanya dievaluasi sekali per candle M5 baru close."""
        last_ts_raw = self.state.get_meta("last_evaluated_bar_ts")
        if last_ts_raw is not None and int(latest_closed_bar_ts_utc) <= int(last_ts_raw):
            return GuardrailDecision(
                allowed=False,
                gate="NEW_BAR",
                reason="Skipped: Menunggu candle M5 baru close (anti tick-spam).",
                effective_risk_pct=self.cfg.risk_per_trade_pct,
            )
        return GuardrailDecision(
            allowed=True,
            gate="NEW_BAR",
            reason="OK",
            effective_risk_pct=self.cfg.risk_per_trade_pct,
        )

    def mark_bar_evaluated(self, latest_closed_bar_ts_utc: int) -> None:
        """Tandai timestamp bar M5 tertutup yang baru selesai dievaluasi."""
        self.state.set_meta("last_evaluated_bar_ts", str(int(latest_closed_bar_ts_utc)))

    def check_pre_entry_gates(
        self,
        now_utc: datetime,
        balance: float,
        equity: float,
        spread_usd: float,
        open_positions: Sequence[Any],
        pending_orders: Sequence[Any],
        last_closed_trade_utc: Optional[datetime] = None,
    ) -> GuardrailDecision:
        """Evaluasi Gate 2-5 (Single Position, Cooldown, Circuit Breaker, Max Spread)."""
        now_utc = now_utc.astimezone(timezone.utc)
        period = self.state.sync_risk_periods(now_utc, balance, equity)

        # Gate 5A: Monthly Drawdown Circuit Breaker (>= 10% -> HALT total)
        monthly_dd_pct = compute_monthly_drawdown_pct(
            period, equity, self.cfg.monthly_dd_mode
        )
        if period.monthly_halted or monthly_dd_pct >= self.cfg.monthly_max_dd_pct:
            self.state.set_monthly_halted(True)
            return GuardrailDecision(
                allowed=False,
                gate="MONTHLY_CIRCUIT_BREAKER",
                reason=(
                    f"Skipped: Monthly Killswitch aktif "
                    f"(DD={monthly_dd_pct:.2f}% >= {self.cfg.monthly_max_dd_pct:.1f}% "
                    f"mode={self.cfg.monthly_dd_mode})."
                ),
                effective_risk_pct=0.0,
            )

        # Gate 5B: Weekly Loss -> Potong risiko setengah (1.5% -> 0.75%)
        weekly_loss_pct = (
            (-period.weekly_realized_pnl / period.weekly_start_balance) * 100.0
            if period.weekly_start_balance > 0 and period.weekly_realized_pnl < 0
            else 0.0
        )
        effective_risk = (
            self.cfg.weekly_reduced_risk_pct
            if weekly_loss_pct >= self.cfg.weekly_max_loss_pct
            else self.cfg.risk_per_trade_pct
        )

        # Gate 5C: Daily Killswitch (2x SL atau rugi harian >= 3%)
        daily_loss_pct = (
            (-period.daily_realized_pnl / period.daily_start_balance) * 100.0
            if period.daily_start_balance > 0 and period.daily_realized_pnl < 0
            else 0.0
        )
        if period.daily_sl_count >= self.cfg.daily_max_sl_count:
            return GuardrailDecision(
                allowed=False,
                gate="DAILY_CIRCUIT_BREAKER",
                reason=(
                    f"Skipped: Daily Killswitch aktif "
                    f"({period.daily_sl_count}x SL hari ini UTC)."
                ),
                effective_risk_pct=effective_risk,
            )
        if daily_loss_pct >= self.cfg.daily_max_loss_pct:
            return GuardrailDecision(
                allowed=False,
                gate="DAILY_CIRCUIT_BREAKER",
                reason=(
                    f"Skipped: Daily Killswitch aktif "
                    f"(rugi harian {daily_loss_pct:.2f}% >= {self.cfg.daily_max_loss_pct:.1f}%)."
                ),
                effective_risk_pct=effective_risk,
            )

        # Gate 2: Maksimal 1 Posisi / Pending Order aktif per Magic Number
        if len(open_positions) > 0 or len(pending_orders) > 0:
            return GuardrailDecision(
                allowed=False,
                gate="SINGLE_POSITION",
                reason=(
                    f"Skipped: Masih ada {len(open_positions)} posisi & "
                    f"{len(pending_orders)} pending order aktif."
                ),
                effective_risk_pct=effective_risk,
            )

        # Gate 4: Cooldown minimal 60 menit setelah posisi terakhir tertutup
        strat_state = self.state.get_strategy_state()
        cooldown_candidates: list[datetime] = []
        if strat_state.cooldown_until_utc:
            cooldown_candidates.append(
                datetime.fromisoformat(strat_state.cooldown_until_utc).astimezone(
                    timezone.utc
                )
            )
        if last_closed_trade_utc is not None:
            cooldown_candidates.append(
                last_closed_trade_utc.astimezone(timezone.utc)
                + timedelta(minutes=self.cfg.cooldown_minutes)
            )

        if cooldown_candidates:
            cooldown_until = max(cooldown_candidates)
            if now_utc < cooldown_until:
                rem_min = int((cooldown_until - now_utc).total_seconds() // 60) + 1
                return GuardrailDecision(
                    allowed=False,
                    gate="COOLDOWN",
                    reason=f"Skipped: Cooldown aktif ({rem_min} menit tersisa).",
                    effective_risk_pct=effective_risk,
                )

        # Gate 3: Max Spread Guard (USD distance agar aman 2-digit vs 3-digit)
        if spread_usd > self.cfg.max_spread_usd:
            return GuardrailDecision(
                allowed=False,
                gate="MAX_SPREAD",
                reason=(
                    f"Skipped: Spread terlalu tinggi "
                    f"(${spread_usd:.3f} > ${self.cfg.max_spread_usd:.3f})."
                ),
                effective_risk_pct=effective_risk,
            )

        return GuardrailDecision(
            allowed=True,
            gate="ALL_PASSED",
            reason="OK",
            effective_risk_pct=effective_risk,
        )
