"""Backtester event-driven berbagi fungsi murni strategy/ dengan simulasi intrabar M1
(asumsi pesimis SL kena duluan jika ambigu), spread historis per bar, optimasi O(N)
untuk rentang 10-20 tahun, dukungan MTF (D1/H4/H1/M15/M5), dan breakdown per 5 tahun.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from config import CONFIG, BotConfig
from core.state_manager import (
    FUNNEL_COLUMNS,
    PRE_POSITION_STATES,
    RiskPeriodState,
    StrategyState,
)
from guardrails.anti_ganas import AntiGanasGuard
from strategy.indicators import compute_atr, compute_rolling_correlation
from strategy.layer1_macro import (
    BiasType,
    Layer1Result,
    check_news_filter_history,
    infer_h4_structure_bias,
    is_within_session_utc,
)
from strategy.layer2_structure import (
    SupplyDemandZone,
    detect_bos_and_fresh_zones,
    select_priority_touched_zone,
)
from strategy.layer3_smc_m5 import advance_smc_state_machine
from strategy.layer4_risk_exit import (
    compute_realized_r,
    evaluate_active_position_actions,
)


@dataclass(frozen=True)
class TradeOutcome:
    """Hasil simulasi satu setup limit order sampai exit atau dibatalkan."""

    filled: bool
    direction: Literal["BUY", "SELL"]
    entry_time_utc: Optional[datetime]
    exit_time_utc: Optional[datetime]
    entry_price: float
    sl_price: float
    tp_price: float
    hit_partial: bool
    exit_reason: Literal["TAKE_PROFIT", "STOP_LOSS", "BEP_OR_TRAIL", "CANCELLED", "OPEN_END"]
    realized_r: float


@dataclass
class BacktestResult:
    """Ringkasan hasil backtest termasuk Realized R, breakdown 5-tahunan, dan funnel."""

    trades: List[TradeOutcome]
    initial_balance: float
    final_balance: float
    total_trades: int
    win_rate: float
    profit_factor: float
    expectancy_r: float
    max_drawdown_pct: float
    funnel: Dict[str, int] = field(default_factory=dict)
    regime_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    news_coverage_note: str = ""


class _InMemoryBacktestState:
    """Implementasi in-memory kompatibel StateManager agar loop 20 tahun tetap cepat."""

    def __init__(self) -> None:
        self._strategy_state = StrategyState(state="IDLE")
        self._meta: Dict[str, str] = {}
        self._risk: Optional[RiskPeriodState] = None
        self._funnel: Dict[str, int] = {col: 0 for col in FUNNEL_COLUMNS}

    def get_strategy_state(self) -> StrategyState:
        return self._strategy_state

    def save_strategy_state(self, st: StrategyState) -> None:
        self._strategy_state = st

    def reset_pre_position_to_idle(self) -> StrategyState:
        if self._strategy_state.state in PRE_POSITION_STATES:
            self._strategy_state = StrategyState(state="IDLE")
        return self._strategy_state

    def set_meta(self, key: str, value: str) -> None:
        self._meta[key] = value

    def get_meta(self, key: str, default: Optional[str] = None) -> Optional[str]:
        return self._meta.get(key, default)

    def sync_risk_periods(
        self, now_utc: datetime, balance: float, equity: float
    ) -> RiskPeriodState:
        d_key = now_utc.strftime("%Y-%m-%d")
        iso = now_utc.isocalendar()
        w_key = f"{iso.year}-W{iso.week:02d}"
        m_key = now_utc.strftime("%Y-%m")

        if self._risk is None:
            self._risk = RiskPeriodState(
                daily_key=d_key,
                daily_sl_count=0,
                daily_realized_pnl=0.0,
                daily_start_balance=balance,
                weekly_key=w_key,
                weekly_realized_pnl=0.0,
                weekly_start_balance=balance,
                monthly_key=m_key,
                monthly_start_balance=balance,
                monthly_peak_balance=balance,
                monthly_peak_equity=max(balance, equity),
                monthly_halted=False,
            )
            return self._risk

        r = self._risk
        daily_sl = 0 if r.daily_key != d_key else r.daily_sl_count
        daily_pnl = 0.0 if r.daily_key != d_key else r.daily_realized_pnl
        daily_start = balance if r.daily_key != d_key else r.daily_start_balance
        weekly_pnl = 0.0 if r.weekly_key != w_key else r.weekly_realized_pnl
        weekly_start = balance if r.weekly_key != w_key else r.weekly_start_balance
        monthly_start = balance if r.monthly_key != m_key else r.monthly_start_balance
        monthly_peak_bal = balance if r.monthly_key != m_key else max(r.monthly_peak_balance, balance)
        monthly_peak_eq = max(balance, equity) if r.monthly_key != m_key else max(r.monthly_peak_equity, equity)
        monthly_halted = False if r.monthly_key != m_key else r.monthly_halted

        self._risk = RiskPeriodState(
            daily_key=d_key,
            daily_sl_count=daily_sl,
            daily_realized_pnl=daily_pnl,
            daily_start_balance=daily_start,
            weekly_key=w_key,
            weekly_realized_pnl=weekly_pnl,
            weekly_start_balance=weekly_start,
            monthly_key=m_key,
            monthly_start_balance=monthly_start,
            monthly_peak_balance=monthly_peak_bal,
            monthly_peak_equity=monthly_peak_eq,
            monthly_halted=monthly_halted,
        )
        return self._risk

    def record_closed_trade_pnl(
        self,
        now_utc: datetime,
        realized_pnl: float,
        new_balance: float,
        new_equity: float,
        is_stop_loss: bool,
    ) -> RiskPeriodState:
        r = self.sync_risk_periods(now_utc, new_balance, new_equity)
        self._risk = RiskPeriodState(
            daily_key=r.daily_key,
            daily_sl_count=r.daily_sl_count + (1 if is_stop_loss else 0),
            daily_realized_pnl=r.daily_realized_pnl + realized_pnl,
            daily_start_balance=r.daily_start_balance,
            weekly_key=r.weekly_key,
            weekly_realized_pnl=r.weekly_realized_pnl + realized_pnl,
            weekly_start_balance=r.weekly_start_balance,
            monthly_key=r.monthly_key,
            monthly_start_balance=r.monthly_start_balance,
            monthly_peak_balance=max(r.monthly_peak_balance, new_balance),
            monthly_peak_equity=max(r.monthly_peak_equity, new_equity),
            monthly_halted=r.monthly_halted,
        )
        return self._risk

    def set_monthly_halted(self, halted: bool = True) -> None:
        if self._risk is not None:
            r = self._risk
            self._risk = RiskPeriodState(
                daily_key=r.daily_key,
                daily_sl_count=r.daily_sl_count,
                daily_realized_pnl=r.daily_realized_pnl,
                daily_start_balance=r.daily_start_balance,
                weekly_key=r.weekly_key,
                weekly_realized_pnl=r.weekly_realized_pnl,
                weekly_start_balance=r.weekly_start_balance,
                monthly_key=r.monthly_key,
                monthly_start_balance=r.monthly_start_balance,
                monthly_peak_balance=r.monthly_peak_balance,
                monthly_peak_equity=r.monthly_peak_equity,
                monthly_halted=halted,
            )

    def increment_funnel(self, now_utc: datetime, column: str, amount: int = 1) -> None:
        self._funnel[column] = self._funnel.get(column, 0) + amount

    def get_funnel_stats(self, now_utc: datetime) -> Dict[str, int]:
        return dict(self._funnel)


def load_news_history_csv(path: Path) -> pd.DataFrame:
    """Muat kalender historis CPI/NFP/FOMC CSV dalam UTC."""
    if not path.exists():
        return pd.DataFrame(columns=["time_utc", "currency", "impact", "title"])
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    return df


def _evaluate_news_coverage(
    data_start: datetime, data_end: datetime, news_df: pd.DataFrame
) -> str:
    """Laporkan secara jujur cakupan tahun kalender berita historis terhadap rentang backtest."""
    if news_df.empty:
        return (
            f"PERINGATAN: news_history.csv kosong untuk rentang backtest "
            f"{data_start:%Y-%m-%d} s/d {data_end:%Y-%m-%d}; hasil belum terfilter berita."
        )
    min_n = pd.Timestamp(news_df["time_utc"].min()).to_pydatetime().astimezone(timezone.utc)
    max_n = pd.Timestamp(news_df["time_utc"].max()).to_pydatetime().astimezone(timezone.utc)
    if min_n.year > data_start.year or max_n.year < data_end.year:
        return (
            f"PERINGATAN: news_history.csv hanya mencakup {min_n:%Y-%m-%d} s/d {max_n:%Y-%m-%d}, "
            f"sedangkan backtest berjalan {data_start:%Y-%m-%d} s/d {data_end:%Y-%m-%d}."
        )
    return f"OK: Kalender berita mencakup {min_n:%Y-%m-%d} s/d {max_n:%Y-%m-%d}."


def _compute_5y_regime_breakdown(trades: List[TradeOutcome]) -> Dict[str, Dict[str, Any]]:
    """Kelompokkan metrik performa per blok 5 tahun untuk melihat ketahanan lintas rezim."""
    buckets: Dict[str, List[TradeOutcome]] = {}
    for t in trades:
        ref_dt = t.entry_time_utc or t.exit_time_utc
        if ref_dt is None:
            continue
        start_y = (ref_dt.year // 5) * 5
        if ref_dt.year >= 2006:
            # Blok alami 2006-2010, 2011-2015, 2016-2020, 2021-2025, 2026-2030
            start_y = 2006 + ((ref_dt.year - 2006) // 5) * 5
        label = f"{start_y}-{start_y + 4}"
        buckets.setdefault(label, []).append(t)

    out: Dict[str, Dict[str, Any]] = {}
    for label, grp in sorted(buckets.items()):
        r_vals = [x.realized_r for x in grp]
        wins = [r for r in r_vals if r > 0]
        losses = [abs(r) for r in r_vals if r < 0]
        pf = (sum(wins) / sum(losses)) if losses else (99.0 if wins else 0.0)
        out[label] = {
            "trades": len(grp),
            "win_rate": (len(wins) / len(grp)) * 100.0,
            "profit_factor": pf,
            "expectancy_r": float(np.mean(r_vals)),
        }
    return out


def simulate_trade_intrabar_m1(
    direction: Literal["BUY", "SELL"],
    limit_price: float,
    sl_price: float,
    tp_1_2_price: float,
    tp_final_price: float,
    atr_m5: float,
    m1_slice: pd.DataFrame,
    expiry_utc: datetime,
    session_end_hhmm: str,
    point: float,
    cfg: BotConfig = CONFIG,
    allow_partial: bool = True,
) -> TradeOutcome:
    """Simulasikan satu setup menggunakan array bar M1 (asumsi pesimis SL-first)."""
    filled = False
    entry_time: Optional[datetime] = None
    hit_partial = False
    bep_moved = False
    current_sl = sl_price
    risk_dist = abs(limit_price - sl_price)
    if m1_slice.empty or risk_dist <= 0:
        return TradeOutcome(False, direction, None, None, limit_price, sl_price, tp_final_price, False, "CANCELLED", 0.0)

    times = pd.DatetimeIndex(m1_slice["time_utc"]).tz_convert("UTC").to_pydatetime()
    lows = m1_slice["low"].to_numpy(dtype=float)
    highs = m1_slice["high"].to_numpy(dtype=float)
    spreads = (
        m1_slice["spread"].fillna(25.0).to_numpy(dtype=float)
        if "spread" in m1_slice.columns
        else np.full(len(lows), 25.0)
    )

    for idx in range(len(lows)):
        t_utc = times[idx]
        low = float(lows[idx])
        high = float(highs[idx])
        spread_usd = max(0.0, float(spreads[idx]) * point)

        bid_low, bid_high = low, high
        ask_low, ask_high = low + spread_usd, high + spread_usd

        if not filled:
            if t_utc >= expiry_utc or not is_within_session_utc(t_utc, cfg.session_start_utc, session_end_hhmm):
                return TradeOutcome(False, direction, None, t_utc, limit_price, sl_price, tp_final_price, False, "CANCELLED", 0.0)

            if direction == "BUY":
                if ask_low <= limit_price <= ask_high:
                    filled = True
                    entry_time = t_utc
                elif bid_high >= tp_1_2_price or bid_low <= sl_price:
                    return TradeOutcome(False, direction, None, t_utc, limit_price, sl_price, tp_final_price, False, "CANCELLED", 0.0)
            else:
                if bid_low <= limit_price <= bid_high:
                    filled = True
                    entry_time = t_utc
                elif ask_low <= tp_1_2_price or ask_high >= sl_price:
                    return TradeOutcome(False, direction, None, t_utc, limit_price, sl_price, tp_final_price, False, "CANCELLED", 0.0)

        if filled:
            # ASUMSI PESIMIS: Cek SL terlebih dahulu di setiap bar M1 (termasuk bar fill)
            sl_hit = (bid_low <= current_sl) if direction == "BUY" else (ask_high >= current_sl)
            if sl_hit:
                if not hit_partial:
                    exit_r = ((current_sl - limit_price) / risk_dist) if direction == "BUY" else ((limit_price - current_sl) / risk_dist)
                    reason: Literal["STOP_LOSS", "BEP_OR_TRAIL"] = "BEP_OR_TRAIL" if bep_moved else "STOP_LOSS"
                    return TradeOutcome(True, direction, entry_time, t_utc, limit_price, current_sl, tp_final_price, False, reason, exit_r)
                rem_r = ((current_sl - limit_price) / risk_dist) if direction == "BUY" else ((limit_price - current_sl) / risk_dist)
                real_r = compute_realized_r(True, cfg.partial_close_ratio, cfg.partial_tp_rrr, rem_r)
                return TradeOutcome(True, direction, entry_time, t_utc, limit_price, current_sl, tp_final_price, True, "BEP_OR_TRAIL", real_r)

            probe_bid = bid_high if direction == "BUY" else (ask_low - spread_usd)
            probe_ask = probe_bid + spread_usd
            act = evaluate_active_position_actions(
                direction=direction,
                entry_price=limit_price,
                initial_sl=sl_price,
                current_sl=current_sl,
                current_bid=probe_bid,
                current_ask=probe_ask,
                current_volume=0.04 if allow_partial else 0.01,
                volume_min=0.01,
                atr_m5=atr_m5,
                partial_done=hit_partial or bep_moved,
                allow_partial=allow_partial,
                cfg=cfg,
            )
            if act.reason == "HIT_1_2_PARTIAL_AND_BEP":
                if allow_partial:
                    hit_partial = True
                bep_moved = True
                if act.new_sl is not None:
                    current_sl = act.new_sl
            elif act.new_sl is not None:
                current_sl = act.new_sl

            tp_hit = (bid_high >= tp_final_price) if direction == "BUY" else (ask_low <= tp_final_price)
            if tp_hit:
                real_r = compute_realized_r(hit_partial, cfg.partial_close_ratio, cfg.partial_tp_rrr, cfg.final_tp_rrr)
                return TradeOutcome(True, direction, entry_time, t_utc, limit_price, current_sl, tp_final_price, hit_partial, "TAKE_PROFIT", real_r)

    if not filled:
        return TradeOutcome(False, direction, None, None, limit_price, sl_price, tp_final_price, False, "CANCELLED", 0.0)
    return TradeOutcome(True, direction, entry_time, None, limit_price, current_sl, tp_final_price, hit_partial, "OPEN_END", 0.0)


class BacktestEngine:
    """Mesin backtest O(N) untuk rentang 1-20 tahun memakai fungsi produksi strategy/ & guardrails/."""

    def __init__(self, cfg: BotConfig = CONFIG, point: float = 0.01) -> None:
        self.cfg = cfg
        self.point = point

    def run(
        self,
        m5_df: pd.DataFrame,
        h1_df: pd.DataFrame,
        h4_df: pd.DataFrame,
        dxy_h4: pd.DataFrame,
        m1_df: Optional[pd.DataFrame] = None,
        news_history_df: Optional[pd.DataFrame] = None,
        initial_balance: float = 2000.0,
        d1_df: Optional[pd.DataFrame] = None,
        m15_df: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        """Jalankan backtest berurutan tanpa look-ahead bias (kompleksitas O(N))."""
        if m5_df.empty:
            return BacktestResult([], initial_balance, initial_balance, 0, 0.0, 0.0, 0.0, 0.0)

        m5 = m5_df.sort_values("time_utc").reset_index(drop=True)
        h1 = h1_df.sort_values("time_utc").reset_index(drop=True)
        h4 = h4_df.sort_values("time_utc").reset_index(drop=True)
        dxy = dxy_h4.sort_values("time_utc").reset_index(drop=True)
        m1 = (m1_df if m1_df is not None and not m1_df.empty else m5).sort_values("time_utc").reset_index(drop=True)
        d1 = d1_df.sort_values("time_utc").reset_index(drop=True) if d1_df is not None and not d1_df.empty else None
        m15 = m15_df.sort_values("time_utc").reset_index(drop=True) if m15_df is not None and not m15_df.empty else None

        m5_ns = pd.DatetimeIndex(m5["time_utc"]).as_unit("ns").asi8
        h1_ns = pd.DatetimeIndex(h1["time_utc"]).as_unit("ns").asi8
        h4_ns = pd.DatetimeIndex(h4["time_utc"]).as_unit("ns").asi8
        dxy_ns = pd.DatetimeIndex(dxy["time_utc"]).as_unit("ns").asi8
        m1_ns = pd.DatetimeIndex(m1["time_utc"]).as_unit("ns").asi8
        d1_ns = pd.DatetimeIndex(d1["time_utc"]).as_unit("ns").asi8 if d1 is not None else None
        m15_ns = pd.DatetimeIndex(m15["time_utc"]).as_unit("ns").asi8 if m15 is not None else None

        m5_times = pd.DatetimeIndex(m5["time_utc"]).tz_convert("UTC").to_pydatetime()
        m5_lows = m5["low"].to_numpy(dtype=float)
        m5_highs = m5["high"].to_numpy(dtype=float)
        m5_closes = m5["close"].to_numpy(dtype=float)
        m5_atr = compute_atr(m5, self.cfg.atr_period).to_numpy(dtype=float)
        m5_spreads = (
            m5["spread"].fillna(25.0).to_numpy(dtype=float)
            if "spread" in m5.columns
            else np.full(len(m5), 25.0)
        )
        news_df = news_history_df if news_history_df is not None else pd.DataFrame()
        news_note = _evaluate_news_coverage(m5_times[0], m5_times[-1], news_df)

        sm = _InMemoryBacktestState()
        balance = initial_balance
        peak_balance = initial_balance
        max_dd_pct = 0.0
        trades: List[TradeOutcome] = []
        busy_until_utc: Optional[datetime] = None
        last_closed_utc: Optional[datetime] = None

        cached_h4_key: Optional[Tuple[int, int]] = None
        cached_macro: Tuple[Optional[float], BiasType, BiasType] = (None, "NEUTRAL", "NEUTRAL")
        cached_d1: Tuple[int, BiasType] = (-1, "NEUTRAL")
        cached_zones_h4: Tuple[int, str, List[SupplyDemandZone]] = (-1, "", [])
        cached_zones_h1: Tuple[int, str, List[SupplyDemandZone]] = (-1, "", [])

        warmup = max(25, self.cfg.displacement_avg_period + 5)
        for i in range(warmup, len(m5)):
            now_utc = m5_times[i]
            sm.increment_funnel(now_utc, "total_m5_bars", 1)

            if busy_until_utc is not None and now_utc < busy_until_utc:
                continue

            if not is_within_session_utc(now_utc, self.cfg.session_start_utc, self.cfg.session_end_utc):
                sm.reset_pre_position_to_idle()
                continue

            news_ok, _ = check_news_filter_history(now_utc, news_df, self.cfg)
            if not news_ok:
                continue

            cur_ns = int(m5_ns[i])
            h4_idx = int(np.searchsorted(h4_ns, cur_ns, side="right"))
            dxy_idx = int(np.searchsorted(dxy_ns, cur_ns, side="right"))
            h1_idx = int(np.searchsorted(h1_ns, cur_ns, side="right"))

            # Rolling window H4/H1 menjaga kompleksitas O(N) pada data 20 tahun
            h4_start = max(0, h4_idx - 250)
            dxy_start = max(0, dxy_idx - 250)
            h1_start = max(0, h1_idx - 400)

            if cached_h4_key != (h4_idx, dxy_idx):
                xau_slice = h4.iloc[h4_start:h4_idx]
                dxy_slice = dxy.iloc[dxy_start:dxy_idx]
                corr = compute_rolling_correlation(xau_slice, dxy_slice, window=self.cfg.corr_window_h4)
                xau_tr = infer_h4_structure_bias(xau_slice, pivot_window=self.cfg.pivot_window)
                dxy_tr = infer_h4_structure_bias(dxy_slice, pivot_window=self.cfg.pivot_window)
                cached_macro = (corr, dxy_tr, xau_tr)
                cached_h4_key = (h4_idx, dxy_idx)

            corr, dxy_trend, xau_trend = cached_macro
            l1 = _resolve_macro_bias(corr, dxy_trend, xau_trend, self.cfg)
            if not l1.passed or l1.bias is None:
                continue

            # Opsional filter Daily (D1) alignment bila diaktifkan di config
            if self.cfg.require_d1_alignment and d1 is not None and d1_ns is not None:
                d1_idx = int(np.searchsorted(d1_ns, cur_ns, side="right"))
                if cached_d1[0] != d1_idx:
                    d1_bias = infer_h4_structure_bias(
                        d1.iloc[max(0, d1_idx - 120) : d1_idx], pivot_window=self.cfg.pivot_window
                    )
                    cached_d1 = (d1_idx, d1_bias)
                if cached_d1[1] != l1.bias:
                    continue

            sm.increment_funnel(now_utc, "passed_session_news", 1)

            atr_val = float(m5_atr[i])
            eff_cfg = (
                replace(self.cfg, max_spread_usd=max(0.05, self.cfg.max_spread_atr_ratio * atr_val))
                if self.cfg.max_spread_atr_ratio > 0 and atr_val > 0
                else self.cfg
            )
            guard = AntiGanasGuard(sm, eff_cfg)  # type: ignore[arg-type]

            spread_usd = float(m5_spreads[i]) * self.point
            dec = guard.check_pre_entry_gates(
                now_utc=now_utc,
                balance=balance,
                equity=balance,
                spread_usd=spread_usd,
                open_positions=[],
                pending_orders=[],
                last_closed_trade_utc=last_closed_utc,
            )
            if not dec.allowed:
                continue
            sm.increment_funnel(now_utc, "passed_spread", 1)
            sm.increment_funnel(now_utc, "passed_dxy_corr", 1)

            if cached_zones_h4[0] != h4_idx or cached_zones_h4[1] != l1.bias:
                cached_zones_h4 = (
                    h4_idx,
                    l1.bias,
                    detect_bos_and_fresh_zones(h4.iloc[h4_start:h4_idx], "H4", l1.bias, self.cfg.pivot_window),
                )
            if cached_zones_h1[0] != h1_idx or cached_zones_h1[1] != l1.bias:
                cached_zones_h1 = (
                    h1_idx,
                    l1.bias,
                    detect_bos_and_fresh_zones(h1.iloc[h1_start:h1_idx], "H1", l1.bias, self.cfg.pivot_window),
                )

            active_zone = select_priority_touched_zone(
                cached_zones_h4[2] + cached_zones_h1[2],
                current_price=float(m5_closes[i]),
                bias=l1.bias,
                bar_low=float(m5_lows[i]),
                bar_high=float(m5_highs[i]),
            )

            # Pilih timeframe pemicu entry konsisten (M15_OR_M5: zona H4 -> M15, zona H1 -> M5)
            cur_st = sm.get_strategy_state()
            zone_tf = cur_st.zone_timeframe or (active_zone.timeframe if active_zone else "H1")
            use_m15 = (
                self.cfg.entry_timeframe == "M15"
                or (self.cfg.entry_timeframe == "M15_OR_M5" and zone_tf == "H4")
            )
            if use_m15 and m15 is not None and m15_ns is not None:
                m15_idx = int(np.searchsorted(m15_ns, cur_ns, side="right"))
                entry_slice = m15.iloc[max(0, m15_idx - 120) : m15_idx]
            else:
                entry_slice = m5.iloc[max(0, i - 120) : i + 1]

            act = advance_smc_state_machine(
                sm, entry_slice, active_zone=active_zone, now_utc=now_utc, in_session=True, cfg=eff_cfg  # type: ignore[arg-type]
            )

            if act.action == "PLACE_LIMIT" and act.direction and act.limit_price and act.sl_price and act.tp_price and act.tp_1_2_price:
                risk_dist = abs(act.limit_price - act.sl_price)
                risk_amt = balance * (dec.effective_risk_pct / 100.0)
                raw_lot = (risk_amt / (risk_dist * 100.0)) if risk_dist > 0 else 0.0
                lot = np.floor(raw_lot * 100.0) / 100.0
                allow_partial = lot >= 0.02
                if lot < 0.01 or (not allow_partial and self.cfg.min_lot_partial_policy == "SKIP"):
                    sm.increment_funnel(now_utc, "rejected_min_lot", 1)
                    sm.save_strategy_state(StrategyState(state="IDLE"))
                    continue

                m1_start = int(np.searchsorted(m1_ns, cur_ns, side="right"))
                horizon_ns = cur_ns + int(48 * 3600 * 1e9)
                m1_end = int(np.searchsorted(m1_ns, horizon_ns, side="right"))
                outcome = simulate_trade_intrabar_m1(
                    direction=act.direction,
                    limit_price=act.limit_price,
                    sl_price=act.sl_price,
                    tp_1_2_price=act.tp_1_2_price,
                    tp_final_price=act.tp_price,
                    atr_m5=atr_val,
                    m1_slice=m1.iloc[m1_start:m1_end],
                    expiry_utc=now_utc + timedelta(hours=self.cfg.pending_order_expiry_hours),
                    session_end_hhmm=self.cfg.session_end_utc,
                    point=self.point,
                    cfg=eff_cfg,
                    allow_partial=allow_partial,
                )
                if outcome.filled:
                    trades.append(outcome)
                    pnl = risk_amt * outcome.realized_r
                    balance += pnl
                    peak_balance = max(peak_balance, balance)
                    dd = ((peak_balance - balance) / peak_balance) * 100.0 if peak_balance > 0 else 0.0
                    max_dd_pct = max(max_dd_pct, dd)
                    exit_dt = outcome.exit_time_utc or now_utc
                    sm.record_closed_trade_pnl(
                        exit_dt, pnl, balance, balance, is_stop_loss=(outcome.exit_reason == "STOP_LOSS")
                    )
                    busy_until_utc = exit_dt + timedelta(minutes=self.cfg.cooldown_minutes)
                    last_closed_utc = exit_dt
                elif outcome.exit_time_utc is not None:
                    busy_until_utc = outcome.exit_time_utc
                sm.save_strategy_state(StrategyState(state="IDLE"))

        funnel = sm.get_funnel_stats(m5_times[-1])
        breakdown = _compute_5y_regime_breakdown(trades)
        if not trades:
            return BacktestResult(
                [], initial_balance, balance, 0, 0.0, 0.0, 0.0, max_dd_pct, funnel, breakdown, news_note
            )

        r_vals = [t.realized_r for t in trades]
        wins = [r for r in r_vals if r > 0]
        losses = [abs(r) for r in r_vals if r < 0]
        win_rate = (len(wins) / len(r_vals)) * 100.0
        gross_profit = sum(wins)
        gross_loss = sum(losses)
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)
        expectancy_r = float(np.mean(r_vals))

        return BacktestResult(
            trades=trades,
            initial_balance=initial_balance,
            final_balance=balance,
            total_trades=len(trades),
            win_rate=win_rate,
            profit_factor=profit_factor,
            expectancy_r=expectancy_r,
            max_drawdown_pct=max_dd_pct,
            funnel=funnel,
            regime_breakdown=breakdown,
            news_coverage_note=news_note,
        )


def _resolve_macro_bias(
    corr: Optional[float],
    dxy_trend: BiasType,
    xau_trend: BiasType,
    cfg: BotConfig,
) -> Layer1Result:
    """Evaluasi bias DXY vs XAUUSD dari nilai korelasi H4 yang sudah di-cache."""
    if corr is None:
        return Layer1Result(False, None, None, dxy_trend, xau_trend, "Skipped: Korelasi NaN")
    dxy_inv: BiasType = "BUY" if dxy_trend == "SELL" else ("SELL" if dxy_trend == "BUY" else "NEUTRAL")
    if corr < cfg.corr_negative_threshold:
        if dxy_inv == "NEUTRAL":
            return Layer1Result(False, None, corr, dxy_trend, xau_trend, "Skipped: DXY netral")
        return Layer1Result(True, dxy_inv, corr, dxy_trend, xau_trend, "OK")
    if corr > cfg.corr_anomaly_threshold:
        if xau_trend == "NEUTRAL":
            return Layer1Result(False, None, corr, dxy_trend, xau_trend, "Skipped: XAUUSD netral")
        return Layer1Result(True, xau_trend, corr, dxy_trend, xau_trend, "OK")
    if dxy_inv != "NEUTRAL" and dxy_inv == xau_trend:
        return Layer1Result(True, xau_trend, corr, dxy_trend, xau_trend, "OK")
    return Layer1Result(False, None, corr, dxy_trend, xau_trend, "Skipped: Grey zone tidak searah")
