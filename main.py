"""Runner tunggal (single-process) untuk bot live/demo MT5 XAUUSD Sniper.

Menjalankan dua ritme berbeda:
1. Loop cepat (`poll_interval_seconds`) untuk Position Manager saat `IN_POSITION`
   (Partial TP 1:2, BEP + spread/komisi, ATR Trailing Stop dengan step hysteresis).
2. Evaluasi sinyal strategi 4 Layer + MTF Matrix hanya saat candle M5 baru close (`isNewBar`).
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from config import CONFIG, BotConfig
from core.dxy_provider import DXYProvider
from core.logger import log_execution, log_skip, setup_logger
from core.mt5_client import MT5Client, make_unique_comment
from core.state_manager import StateManager, StrategyState
from guardrails.anti_ganas import AntiGanasGuard
from strategy.indicators import compute_atr
from strategy.layer1_macro import evaluate_layer1, infer_h4_structure_bias, is_within_session_utc
from strategy.layer2_structure import detect_bos_and_fresh_zones, select_priority_touched_zone
from strategy.layer3_smc_m5 import advance_smc_state_machine
from strategy.layer4_risk_exit import calculate_position_lot, evaluate_active_position_actions
from strategy.mtf_analyzer import build_mtf_matrix


class SniperEngineRunner:
    """Orkestrator single-process MT5 untuk XAUUSD Sniper."""

    def __init__(
        self,
        cfg: BotConfig = CONFIG,
        state_manager: Optional[StateManager] = None,
        mt5_module: Any = None,
    ) -> None:
        self.cfg = cfg
        self.logger = setup_logger("sniper_bot")
        self.state = state_manager or StateManager()
        self.client = MT5Client(cfg=cfg, state_manager=self.state, mt5_module=mt5_module)
        self.dxy_provider = DXYProvider(self.client, cfg=cfg)
        self.guard = AntiGanasGuard(self.state, cfg=cfg)
        self._symbol: Optional[str] = None

    def start(self) -> bool:
        if not self.client.connect():
            self.logger.error("Gagal menginisialisasi terminal MT5 Desktop.")
            return False
        self._symbol = self.client.resolve_symbol(self.cfg.symbol_aliases)
        if self._symbol is None:
            self.logger.error("Simbol XAUUSD tidak ditemukan di broker MT5.")
            return False
        spec = self.client.get_symbol_spec(self._symbol)
        self.logger.info(
            "MT5 terhubung: symbol=%s digits=%d point=%s entry_tf=%s dry_run=%s",
            spec.name,
            spec.digits,
            spec.point,
            self.cfg.entry_timeframe,
            self.cfg.dry_run_mode,
        )
        return True

    def stop(self) -> None:
        self.client.shutdown()

    def _manage_active_position(
        self,
        symbol: str,
        open_positions: list[Any],
        offset_hours: int,
        now_utc: datetime,
    ) -> None:
        """Kelola posisi terbuka (kebal reset sesi): Partial 1:2, BEP, & ATR Trailing."""
        st = self.state.get_strategy_state()
        if not open_positions:
            if st.state == "IN_POSITION":
                cd_until = now_utc + timedelta(minutes=self.cfg.cooldown_minutes)
                self.state.save_strategy_state(
                    StrategyState(state="COOLDOWN", cooldown_until_utc=cd_until.isoformat())
                )
            return

        pos = open_positions[0]
        spec = self.client.get_symbol_spec(symbol)
        bid, ask, _ = self.client.get_current_bid_ask(symbol)
        tf_m5 = getattr(self.client.mt5, "TIMEFRAME_M5", 5)
        m5_df = self.client.fetch_rates_df(symbol, tf_m5, 40, offset_hours)
        atr_val = float(compute_atr(m5_df, self.cfg.atr_period).iloc[-1]) if not m5_df.empty else 0.0

        buy_type = getattr(self.client.mt5, "POSITION_TYPE_BUY", 0)
        direction = "BUY" if int(getattr(pos, "type", 0)) == buy_type else "SELL"
        entry = float(getattr(pos, "price_open", st.position_entry or 0.0))
        cur_sl = float(getattr(pos, "sl", st.position_sl or 0.0))
        cur_tp = float(getattr(pos, "tp", st.position_tp or 0.0))
        cur_vol = float(getattr(pos, "volume", 0.0))
        ticket = int(getattr(pos, "ticket", 0))

        act = evaluate_active_position_actions(
            direction=direction,
            entry_price=entry,
            initial_sl=st.position_sl or cur_sl,
            current_sl=cur_sl,
            current_bid=bid,
            current_ask=ask,
            current_volume=cur_vol,
            volume_min=spec.volume_min,
            atr_m5=atr_val,
            partial_done=st.partial_done,
            allow_partial=(cur_vol >= 2 * spec.volume_min),
            cfg=self.cfg,
            volume_step=spec.volume_step,
            contract_size=spec.contract_size,
        )

        if act.close_partial_volume is not None and not self.cfg.dry_run_mode:
            res = self.client.close_order(symbol, ticket, int(getattr(pos, "type", 0)), act.close_partial_volume, "SNP_PARTIAL")
            if res.success:
                st.partial_done = True
                self.state.save_strategy_state(st)

        if act.new_sl is not None and not self.cfg.dry_run_mode:
            res = self.client.modify_order(symbol, ticket, act.new_sl, cur_tp)
            if res.success and act.reason == "HIT_1_2_PARTIAL_AND_BEP":
                st.partial_done = True
                self.state.save_strategy_state(st)

    def run_once(self, now_utc: Optional[datetime] = None) -> None:
        """Jalankan satu iterasi engine (dipakai oleh loop utama maupun unit test)."""
        now_utc = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
        if self._symbol is None:
            self._symbol = self.client.resolve_symbol(self.cfg.symbol_aliases)
        if self._symbol is None:
            return

        offset_hours = self.client.calibrate_utc_offset_hours(self._symbol, now_utc=now_utc)
        open_positions = self.client.get_open_positions(self.cfg.magic_number)
        pending_orders = self.client.get_pending_orders(self.cfg.magic_number)

        self._manage_active_position(self._symbol, open_positions, offset_hours, now_utc)
        if open_positions:
            return

        tf_m5 = getattr(self.client.mt5, "TIMEFRAME_M5", 5)
        tf_h1 = getattr(self.client.mt5, "TIMEFRAME_H1", 16385)
        m5_raw = self.client.fetch_rates_df(self._symbol, tf_m5, 160, offset_hours)
        h1_raw = self.client.fetch_rates_df(self._symbol, tf_h1, 200, offset_hours)
        if len(m5_raw) < 20 or len(h1_raw) < 20:
            return
        m5_closed = m5_raw.iloc[:-1].reset_index(drop=True)
        xau_h1 = h1_raw.iloc[:-1].reset_index(drop=True)

        # Jika entry_timeframe == "H1", evaluasi isNewBar pada candle H1 baru close
        latest_bar_ts = (
            int(xau_h1["time_server"].iloc[-1])
            if self.cfg.entry_timeframe == "H1"
            else int(m5_closed["time_server"].iloc[-1])
        )

        bar_dec = self.guard.check_is_new_bar(latest_bar_ts)
        if not bar_dec.allowed:
            return
        self.guard.mark_bar_evaluated(latest_bar_ts)
        self.state.increment_funnel(now_utc, "total_m5_bars", 1)

        balance, equity = self.client.get_account_snapshot()
        spread_usd = self.client.get_spread_usd(self._symbol)

        tf_d1 = getattr(self.client.mt5, "TIMEFRAME_D1", 16408)
        tf_h4 = getattr(self.client.mt5, "TIMEFRAME_H4", 16388)
        tf_m15 = getattr(self.client.mt5, "TIMEFRAME_M15", 15)

        d1_raw = self.client.fetch_rates_df(self._symbol, tf_d1, 90, offset_hours)
        xau_d1 = d1_raw.iloc[:-1] if len(d1_raw) > 1 else d1_raw
        xau_h4 = self.client.fetch_rates_df(self._symbol, tf_h4, 160, offset_hours).iloc[:-1]
        m15_raw = self.client.fetch_rates_df(self._symbol, tf_m15, 160, offset_hours)
        xau_m15 = m15_raw.iloc[:-1] if len(m15_raw) > 1 else m15_raw

        dxy_res = self.dxy_provider.fetch_dxy_h4(160, offset_hours)
        dxy_h4 = dxy_res.df.iloc[:-1] if len(dxy_res.df) > 1 else dxy_res.df

        in_session = is_within_session_utc(now_utc, self.cfg.session_start_utc, self.cfg.session_end_utc)
        l1 = evaluate_layer1(now_utc, xau_h4, dxy_h4, self.cfg)

        zones_h4 = (
            detect_bos_and_fresh_zones(xau_h4, "H4", l1.bias, self.cfg.pivot_window)
            if l1.bias is not None
            else []
        )
        zones_h1 = (
            detect_bos_and_fresh_zones(xau_h1, "H1", l1.bias, self.cfg.pivot_window)
            if l1.bias is not None
            else []
        )
        mtf_report = build_mtf_matrix(
            xau_d1, xau_h4, xau_h1, xau_m15, m5_closed, l1.bias, zones_h4, zones_h1, self.cfg
        )
        self.state.set_meta("mtf_matrix_json", mtf_report.to_json())

        if not l1.passed or l1.bias is None:
            log_skip(self.logger, now_utc.isoformat(), l1.reason, "LAYER1")
            self.state.record_skip(now_utc, "LAYER1", l1.reason)
            if not in_session:
                act_clean = advance_smc_state_machine(self.state, m5_closed, None, now_utc, False, self.cfg)
                if act_clean.action == "CANCEL_PENDING" and pending_orders and not self.cfg.dry_run_mode:
                    self.client.cancel_pending_order(int(pending_orders[0].ticket))
            return

        if self.cfg.require_d1_alignment:
            d1_bias = infer_h4_structure_bias(xau_d1, pivot_window=self.cfg.pivot_window)
            if d1_bias != l1.bias:
                reason_d1 = f"Skipped: D1 bias ({d1_bias}) tidak searah Layer 1 ({l1.bias})."
                log_skip(self.logger, now_utc.isoformat(), reason_d1, "D1_ALIGNMENT")
                self.state.record_skip(now_utc, "D1_ALIGNMENT", reason_d1)
                return

        self.state.increment_funnel(now_utc, "passed_session_news", 1)

        pre = self.guard.check_pre_entry_gates(
            now_utc=now_utc,
            balance=balance,
            equity=equity,
            spread_usd=spread_usd,
            open_positions=open_positions,
            pending_orders=pending_orders,
        )
        if not pre.allowed:
            log_skip(self.logger, now_utc.isoformat(), pre.reason, pre.gate)
            self.state.record_skip(now_utc, pre.gate, pre.reason)
            return
        self.state.increment_funnel(now_utc, "passed_spread", 1)
        self.state.increment_funnel(now_utc, "passed_dxy_corr", 1)

        if self.cfg.entry_timeframe == "H1":
            zones_d1 = (
                detect_bos_and_fresh_zones(xau_d1, "D1", l1.bias, self.cfg.pivot_window)
                if l1.bias is not None and not xau_d1.empty
                else []
            )
            last_bar = xau_h1.iloc[-1]
            active_zone = select_priority_touched_zone(
                zones_d1 + zones_h4,
                current_price=float(last_bar["close"]),
                bias=l1.bias,
                bar_low=float(last_bar["low"]),
                bar_high=float(last_bar["high"]),
            )
            entry_df = xau_h1
        else:
            last_bar = m5_closed.iloc[-1]
            active_zone = select_priority_touched_zone(
                zones_h4 + zones_h1,
                current_price=float(last_bar["close"]),
                bias=l1.bias,
                bar_low=float(last_bar["low"]),
                bar_high=float(last_bar["high"]),
            )

            cur_st = self.state.get_strategy_state()
            zone_tf = cur_st.zone_timeframe or (active_zone.timeframe if active_zone else "H1")
            use_m15 = (
                self.cfg.entry_timeframe == "M15"
                or (self.cfg.entry_timeframe == "M15_OR_M5" and zone_tf == "H4")
            )
            entry_df = xau_m15 if (use_m15 and not xau_m15.empty) else m5_closed

        act = advance_smc_state_machine(
            self.state, entry_df, active_zone=active_zone, now_utc=now_utc, in_session=in_session, cfg=self.cfg
        )

        if act.action == "CANCEL_PENDING" and pending_orders and not self.cfg.dry_run_mode:
            self.client.cancel_pending_order(int(pending_orders[0].ticket))
            return

        if act.action != "PLACE_LIMIT" or act.direction is None or act.limit_price is None or act.sl_price is None or act.tp_price is None:
            return

        spec = self.client.get_symbol_spec(self._symbol)
        lot_res = calculate_position_lot(
            client=self.client,
            spec=spec,
            direction=act.direction,
            balance=balance,
            risk_pct=pre.effective_risk_pct,
            entry_price=act.limit_price,
            sl_price=act.sl_price,
            policy=self.cfg.min_lot_partial_policy,
        )
        if not lot_res.allowed:
            self.state.increment_funnel(now_utc, "rejected_min_lot", 1)
            log_skip(self.logger, now_utc.isoformat(), lot_res.reason, "LOT_SIZE")
            self.state.record_skip(now_utc, "LOT_SIZE", lot_res.reason)
            self.state.save_strategy_state(StrategyState(state="IDLE"))
            return

        comment = make_unique_comment(self.cfg.order_comment_prefix, latest_bar_ts)
        if self.cfg.dry_run_mode:
            self.state.increment_funnel(now_utc, "dry_run_signals", 1)
            self.state.record_dry_run_signal(
                now_utc, self._symbol, act.direction, act.limit_price, act.sl_price, act.tp_price, lot_res.volume, act.reason
            )
            log_execution(
                self.logger, "PLACE_LIMIT", self._symbol, act.direction, lot_res.volume,
                act.limit_price, act.sl_price, act.tp_price, comment, dry_run=True
            )
            self.state.save_strategy_state(StrategyState(state="IDLE"))
            return

        order_type = (
            getattr(self.client.mt5, "ORDER_TYPE_BUY_LIMIT", 2)
            if act.direction == "BUY"
            else getattr(self.client.mt5, "ORDER_TYPE_SELL_LIMIT", 3)
        )
        exp_utc = now_utc + timedelta(hours=self.cfg.pending_order_expiry_hours)
        res = self.client.open_order(
            symbol=self._symbol,
            order_type=order_type,
            volume=lot_res.volume,
            price=act.limit_price,
            sl=act.sl_price,
            tp=act.tp_price,
            comment=comment,
            offset_hours=offset_hours,
            expiration_utc=exp_utc,
            now_utc=now_utc,
        )
        if res.success:
            self.state.increment_funnel(now_utc, "orders_sent", 1)
            st = self.state.get_strategy_state()
            st.state = "PENDING"
            st.pending_ticket = res.ticket
            st.pending_price = act.limit_price
            st.pending_sl = act.sl_price
            st.pending_tp = act.tp_price
            st.pending_tp_1_2 = act.tp_1_2_price
            st.pending_comment = comment
            st.pending_expiry_utc = exp_utc.isoformat()
            self.state.save_strategy_state(st)
            log_execution(
                self.logger, "PLACE_LIMIT", self._symbol, act.direction, lot_res.volume,
                act.limit_price, act.sl_price, act.tp_price, comment, dry_run=False
            )

    def run_forever(self) -> None:
        if not self.start():
            return
        try:
            while True:
                self.run_once()
                time.sleep(self.cfg.poll_interval_seconds)
        except KeyboardInterrupt:
            self.logger.info("Bot dihentikan oleh pengguna.")
        finally:
            self.stop()


if __name__ == "__main__":
    SniperEngineRunner().run_forever()
