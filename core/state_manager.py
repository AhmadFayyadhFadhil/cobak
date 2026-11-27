"""Manajemen state persisten berbasis SQLite untuk State Machine, UTC offset,
reset Circuit Breaker UTC, dan Signal Funnel Tracker.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from config import DB_PATH

VALID_STATES = (
    "IDLE",
    "ZONE_ARMED",
    "SWEEP_DETECTED",
    "CHOCH_CONFIRMED",
    "PENDING",
    "IN_POSITION",
    "COOLDOWN",
)

PRE_POSITION_STATES = ("ZONE_ARMED", "SWEEP_DETECTED", "CHOCH_CONFIRMED", "PENDING")

FUNNEL_COLUMNS = (
    "total_m5_bars",
    "passed_session_news",
    "passed_spread",
    "passed_dxy_corr",
    "zones_armed",
    "sweeps_detected",
    "choch_confirmed",
    "fvg_limits_placed",
    "rejected_min_lot",
    "orders_sent",
    "dry_run_signals",
)


@dataclass
class StrategyState:
    """Representasi state aktif mesin strategi SMC Layer 2 -> Layer 4."""

    state: str = "IDLE"
    zone_id: Optional[str] = None
    zone_timeframe: Optional[str] = None
    zone_direction: Optional[str] = None
    zone_low: Optional[float] = None
    zone_high: Optional[float] = None
    zone_armed_at_utc: Optional[str] = None
    zone_deadline_utc: Optional[str] = None
    sweep_extreme: Optional[float] = None
    choch_bar_ts: Optional[int] = None
    choch_wait_count: int = 0
    fvg_low: Optional[float] = None
    fvg_high: Optional[float] = None
    pending_ticket: Optional[int] = None
    pending_price: Optional[float] = None
    pending_sl: Optional[float] = None
    pending_tp: Optional[float] = None
    pending_tp_1_2: Optional[float] = None
    pending_comment: Optional[str] = None
    pending_expiry_utc: Optional[str] = None
    position_ticket: Optional[int] = None
    position_entry: Optional[float] = None
    position_sl: Optional[float] = None
    position_tp: Optional[float] = None
    position_initial_volume: Optional[float] = None
    partial_done: bool = False
    cooldown_until_utc: Optional[str] = None


@dataclass
class RiskPeriodState:
    """Status metrik risiko harian, mingguan, dan bulanan (seluruh reset berbasis UTC)."""

    daily_key: str
    daily_sl_count: int
    daily_realized_pnl: float
    daily_start_balance: float
    weekly_key: str
    weekly_realized_pnl: float
    weekly_start_balance: float
    monthly_key: str
    monthly_start_balance: float
    monthly_peak_balance: float
    monthly_peak_equity: float
    monthly_halted: bool


def _utc_day_key(now_utc: datetime) -> str:
    return now_utc.astimezone(timezone.utc).strftime("%Y-%m-%d")


def _utc_week_key(now_utc: datetime) -> str:
    # ISO week dimulai Senin 00:00 UTC
    iso = now_utc.astimezone(timezone.utc).isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def _utc_month_key(now_utc: datetime) -> str:
    return now_utc.astimezone(timezone.utc).strftime("%Y-%m")


class StateManager:
    """Menangani baca/tulis state bot ke SQLite secara atomik."""

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;

                CREATE TABLE IF NOT EXISTS strategy_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    state TEXT NOT NULL,
                    zone_id TEXT,
                    zone_timeframe TEXT,
                    zone_direction TEXT,
                    zone_low REAL,
                    zone_high REAL,
                    zone_armed_at_utc TEXT,
                    zone_deadline_utc TEXT,
                    sweep_extreme REAL,
                    choch_bar_ts INTEGER,
                    choch_wait_count INTEGER NOT NULL DEFAULT 0,
                    fvg_low REAL,
                    fvg_high REAL,
                    pending_ticket INTEGER,
                    pending_price REAL,
                    pending_sl REAL,
                    pending_tp REAL,
                    pending_tp_1_2 REAL,
                    pending_comment TEXT,
                    pending_expiry_utc TEXT,
                    position_ticket INTEGER,
                    position_entry REAL,
                    position_sl REAL,
                    position_tp REAL,
                    position_initial_volume REAL,
                    partial_done INTEGER NOT NULL DEFAULT 0,
                    cooldown_until_utc TEXT,
                    updated_at_utc TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS runtime_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS risk_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    daily_key TEXT NOT NULL,
                    daily_sl_count INTEGER NOT NULL DEFAULT 0,
                    daily_realized_pnl REAL NOT NULL DEFAULT 0.0,
                    daily_start_balance REAL NOT NULL DEFAULT 0.0,
                    weekly_key TEXT NOT NULL,
                    weekly_realized_pnl REAL NOT NULL DEFAULT 0.0,
                    weekly_start_balance REAL NOT NULL DEFAULT 0.0,
                    monthly_key TEXT NOT NULL,
                    monthly_start_balance REAL NOT NULL DEFAULT 0.0,
                    monthly_peak_balance REAL NOT NULL DEFAULT 0.0,
                    monthly_peak_equity REAL NOT NULL DEFAULT 0.0,
                    monthly_halted INTEGER NOT NULL DEFAULT 0,
                    updated_at_utc TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS funnel_stats (
                    period_key TEXT PRIMARY KEY,
                    total_m5_bars INTEGER NOT NULL DEFAULT 0,
                    passed_session_news INTEGER NOT NULL DEFAULT 0,
                    passed_spread INTEGER NOT NULL DEFAULT 0,
                    passed_dxy_corr INTEGER NOT NULL DEFAULT 0,
                    zones_armed INTEGER NOT NULL DEFAULT 0,
                    sweeps_detected INTEGER NOT NULL DEFAULT 0,
                    choch_confirmed INTEGER NOT NULL DEFAULT 0,
                    fvg_limits_placed INTEGER NOT NULL DEFAULT 0,
                    rejected_min_lot INTEGER NOT NULL DEFAULT 0,
                    orders_sent INTEGER NOT NULL DEFAULT 0,
                    dry_run_signals INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS skip_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_utc TEXT NOT NULL,
                    gate TEXT NOT NULL,
                    reason TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS dry_run_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_utc TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    sl_price REAL NOT NULL,
                    tp_price REAL NOT NULL,
                    lot_size REAL NOT NULL,
                    reason TEXT NOT NULL
                );
                """
            )
            now_iso = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                INSERT OR IGNORE INTO strategy_state (id, state, updated_at_utc)
                VALUES (1, 'IDLE', ?)
                """,
                (now_iso,),
            )

    # --- Strategy State Machine Persistence ---

    def get_strategy_state(self) -> StrategyState:
        """Ambil snapshot State Machine saat ini dari SQLite."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM strategy_state WHERE id = 1").fetchone()
        return StrategyState(
            state=row["state"],
            zone_id=row["zone_id"],
            zone_timeframe=row["zone_timeframe"],
            zone_direction=row["zone_direction"],
            zone_low=row["zone_low"],
            zone_high=row["zone_high"],
            zone_armed_at_utc=row["zone_armed_at_utc"],
            zone_deadline_utc=row["zone_deadline_utc"],
            sweep_extreme=row["sweep_extreme"],
            choch_bar_ts=row["choch_bar_ts"],
            choch_wait_count=row["choch_wait_count"],
            fvg_low=row["fvg_low"],
            fvg_high=row["fvg_high"],
            pending_ticket=row["pending_ticket"],
            pending_price=row["pending_price"],
            pending_sl=row["pending_sl"],
            pending_tp=row["pending_tp"],
            pending_tp_1_2=row["pending_tp_1_2"],
            pending_comment=row["pending_comment"],
            pending_expiry_utc=row["pending_expiry_utc"],
            position_ticket=row["position_ticket"],
            position_entry=row["position_entry"],
            position_sl=row["position_sl"],
            position_tp=row["position_tp"],
            position_initial_volume=row["position_initial_volume"],
            partial_done=bool(row["partial_done"]),
            cooldown_until_utc=row["cooldown_until_utc"],
        )

    def save_strategy_state(self, st: StrategyState) -> None:
        """Simpan perubahan State Machine ke SQLite."""
        if st.state not in VALID_STATES:
            raise ValueError(f"State tidak valid: {st.state}")
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE strategy_state SET
                    state = ?,
                    zone_id = ?,
                    zone_timeframe = ?,
                    zone_direction = ?,
                    zone_low = ?,
                    zone_high = ?,
                    zone_armed_at_utc = ?,
                    zone_deadline_utc = ?,
                    sweep_extreme = ?,
                    choch_bar_ts = ?,
                    choch_wait_count = ?,
                    fvg_low = ?,
                    fvg_high = ?,
                    pending_ticket = ?,
                    pending_price = ?,
                    pending_sl = ?,
                    pending_tp = ?,
                    pending_tp_1_2 = ?,
                    pending_comment = ?,
                    pending_expiry_utc = ?,
                    position_ticket = ?,
                    position_entry = ?,
                    position_sl = ?,
                    position_tp = ?,
                    position_initial_volume = ?,
                    partial_done = ?,
                    cooldown_until_utc = ?,
                    updated_at_utc = ?
                WHERE id = 1
                """,
                (
                    st.state,
                    st.zone_id,
                    st.zone_timeframe,
                    st.zone_direction,
                    st.zone_low,
                    st.zone_high,
                    st.zone_armed_at_utc,
                    st.zone_deadline_utc,
                    st.sweep_extreme,
                    st.choch_bar_ts,
                    st.choch_wait_count,
                    st.fvg_low,
                    st.fvg_high,
                    st.pending_ticket,
                    st.pending_price,
                    st.pending_sl,
                    st.pending_tp,
                    st.pending_tp_1_2,
                    st.pending_comment,
                    st.pending_expiry_utc,
                    st.position_ticket,
                    st.position_entry,
                    st.position_sl,
                    st.position_tp,
                    st.position_initial_volume,
                    1 if st.partial_done else 0,
                    st.cooldown_until_utc,
                    now_iso,
                ),
            )

    def reset_pre_position_to_idle(self) -> StrategyState:
        """Reset hanya jika berada di state pra-posisi; IN_POSITION dan COOLDOWN kebal."""
        current = self.get_strategy_state()
        if current.state in PRE_POSITION_STATES:
            reset_state = StrategyState(state="IDLE")
            self.save_strategy_state(reset_state)
            return reset_state
        return current

    # --- Runtime Metadata (Fresh UTC Offset & Last Bar) ---

    def set_meta(self, key: str, value: str) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runtime_meta (key, value, updated_at_utc)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at_utc = excluded.updated_at_utc
                """,
                (key, value, now_iso),
            )

    def get_meta(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM runtime_meta WHERE key = ?",
                (key,),
            ).fetchone()
        return row["value"] if row else default

    # --- Risk & Circuit Breaker State (Reset Strictly UTC) ---

    def sync_risk_periods(
        self, now_utc: datetime, balance: float, equity: float
    ) -> RiskPeriodState:
        """Sinkronisasi reset harian (00:00 UTC), mingguan (Senin 00:00 UTC), & bulanan (tgl 1 UTC)."""
        d_key = _utc_day_key(now_utc)
        w_key = _utc_week_key(now_utc)
        m_key = _utc_month_key(now_utc)
        now_iso = now_utc.astimezone(timezone.utc).isoformat()

        with self._connect() as conn:
            row = conn.execute("SELECT * FROM risk_state WHERE id = 1").fetchone()
            if row is None:
                state = RiskPeriodState(
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
                conn.execute(
                    """
                    INSERT INTO risk_state (
                        id, daily_key, daily_sl_count, daily_realized_pnl, daily_start_balance,
                        weekly_key, weekly_realized_pnl, weekly_start_balance,
                        monthly_key, monthly_start_balance, monthly_peak_balance,
                        monthly_peak_equity, monthly_halted, updated_at_utc
                    ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        state.daily_key,
                        state.daily_sl_count,
                        state.daily_realized_pnl,
                        state.daily_start_balance,
                        state.weekly_key,
                        state.weekly_realized_pnl,
                        state.weekly_start_balance,
                        state.monthly_key,
                        state.monthly_start_balance,
                        state.monthly_peak_balance,
                        state.monthly_peak_equity,
                        0,
                        now_iso,
                    ),
                )
                return state

            daily_sl = row["daily_sl_count"]
            daily_pnl = row["daily_realized_pnl"]
            daily_start = row["daily_start_balance"]
            weekly_pnl = row["weekly_realized_pnl"]
            weekly_start = row["weekly_start_balance"]
            monthly_start = row["monthly_start_balance"]
            monthly_peak_bal = max(row["monthly_peak_balance"], balance)
            monthly_peak_eq = max(row["monthly_peak_equity"], equity)
            monthly_halted = bool(row["monthly_halted"])

            if row["daily_key"] != d_key:
                daily_sl = 0
                daily_pnl = 0.0
                daily_start = balance

            if row["weekly_key"] != w_key:
                weekly_pnl = 0.0
                weekly_start = balance

            if row["monthly_key"] != m_key:
                monthly_start = balance
                monthly_peak_bal = balance
                monthly_peak_eq = max(balance, equity)
                monthly_halted = False

            conn.execute(
                """
                UPDATE risk_state SET
                    daily_key = ?,
                    daily_sl_count = ?,
                    daily_realized_pnl = ?,
                    daily_start_balance = ?,
                    weekly_key = ?,
                    weekly_realized_pnl = ?,
                    weekly_start_balance = ?,
                    monthly_key = ?,
                    monthly_start_balance = ?,
                    monthly_peak_balance = ?,
                    monthly_peak_equity = ?,
                    monthly_halted = ?,
                    updated_at_utc = ?
                WHERE id = 1
                """,
                (
                    d_key,
                    daily_sl,
                    daily_pnl,
                    daily_start,
                    w_key,
                    weekly_pnl,
                    weekly_start,
                    m_key,
                    monthly_start,
                    monthly_peak_bal,
                    monthly_peak_eq,
                    1 if monthly_halted else 0,
                    now_iso,
                ),
            )

        return RiskPeriodState(
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

    def record_closed_trade_pnl(
        self,
        now_utc: datetime,
        realized_pnl: float,
        new_balance: float,
        new_equity: float,
        is_stop_loss: bool,
    ) -> RiskPeriodState:
        """Catat hasil PnL posisi yang tertutup dan perbarui hitungan SL harian."""
        period = self.sync_risk_periods(now_utc, new_balance, new_equity)
        daily_sl = period.daily_sl_count + (1 if is_stop_loss else 0)
        daily_pnl = period.daily_realized_pnl + realized_pnl
        weekly_pnl = period.weekly_realized_pnl + realized_pnl
        peak_bal = max(period.monthly_peak_balance, new_balance)
        peak_eq = max(period.monthly_peak_equity, new_equity)
        now_iso = now_utc.astimezone(timezone.utc).isoformat()

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE risk_state SET
                    daily_sl_count = ?,
                    daily_realized_pnl = ?,
                    weekly_realized_pnl = ?,
                    monthly_peak_balance = ?,
                    monthly_peak_equity = ?,
                    updated_at_utc = ?
                WHERE id = 1
                """,
                (daily_sl, daily_pnl, weekly_pnl, peak_bal, peak_eq, now_iso),
            )
        return self.sync_risk_periods(now_utc, new_balance, new_equity)

    def set_monthly_halted(self, halted: bool = True) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE risk_state SET monthly_halted = ?, updated_at_utc = ? WHERE id = 1",
                (1 if halted else 0, now_iso),
            )

    # --- Funnel Tracker & Diagnostic Dry-Run Log ---

    def increment_funnel(self, now_utc: datetime, column: str, amount: int = 1) -> None:
        if column not in FUNNEL_COLUMNS:
            raise ValueError(f"Kolom funnel tidak dikenal: {column}")
        period_key = _utc_week_key(now_utc)
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO funnel_stats (period_key) VALUES (?)",
                (period_key,),
            )
            conn.execute(
                f"UPDATE funnel_stats SET {column} = {column} + ? WHERE period_key = ?",
                (amount, period_key),
            )

    def get_funnel_stats(self, now_utc: datetime) -> Dict[str, int]:
        period_key = _utc_week_key(now_utc)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM funnel_stats WHERE period_key = ?",
                (period_key,),
            ).fetchone()
        if row is None:
            return {col: 0 for col in FUNNEL_COLUMNS}
        return {col: int(row[col]) for col in FUNNEL_COLUMNS}

    def record_skip(self, now_utc: datetime, gate: str, reason: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO skip_log (timestamp_utc, gate, reason) VALUES (?, ?, ?)",
                (now_utc.astimezone(timezone.utc).isoformat(), gate, reason),
            )

    def record_dry_run_signal(
        self,
        now_utc: datetime,
        symbol: str,
        direction: str,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        lot_size: float,
        reason: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO dry_run_signals (
                    timestamp_utc, symbol, direction, entry_price, sl_price, tp_price, lot_size, reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now_utc.astimezone(timezone.utc).isoformat(),
                    symbol,
                    direction,
                    entry_price,
                    sl_price,
                    tp_price,
                    lot_size,
                    reason,
                ),
            )
