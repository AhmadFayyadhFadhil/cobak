"""Dashboard Review Read-Only (Streamlit) — 6 Kartu Pipeline + Multi-Timeframe Matrix:
CHIEF | SCAN | VET | SIZE | FILLS | RISK + MTF Matrix (D1..M5) + Activity Tape + Signal Funnel.

PENTING: Modul ini dilarang mengimpor atau memanggil `MetaTrader5` secara langsung
(thread-safety). Seluruh data dibaca read-only dari SQLite & CSV hasil backtest.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pandas as pd

from config import CONFIG, DB_PATH
from core.state_manager import StateManager


def build_six_cards_snapshot(
    db_path: Path = DB_PATH,
    now_utc: Optional[datetime] = None,
) -> Dict[str, Dict[str, str]]:
    """Susun ringkasan 6 kartu status pipeline langsung dari SQLite secara read-only."""
    now_utc = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if not Path(db_path).exists():
        return {
            "CHIEF": {"status": "NO_DB", "detail": "Database belum dibuat"},
            "SCAN": {"status": "0 / 0", "detail": "Menunggu engine"},
            "VET": {"status": "0", "detail": "Menunggu engine"},
            "SIZE": {"status": f"{CONFIG.risk_per_trade_pct:.2f}%", "detail": CONFIG.min_lot_partial_policy},
            "FILLS": {"status": "0", "detail": "Belum ada eksekusi"},
            "RISK": {"status": "OK", "detail": "Daily SL 0/2"},
        }

    sm = StateManager(db_path)
    st = sm.get_strategy_state()
    funnel = sm.get_funnel_stats(now_utc)
    mtf = load_mtf_matrix(db_path)
    mtf_txt = (
        f" | MTF: {mtf['aligned_count']}/5 ({mtf['dominant_bias']})"
        if mtf is not None
        else ""
    )

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        risk_row = conn.execute("SELECT * FROM risk_state WHERE id = 1").fetchone()

    daily_sl = int(risk_row["daily_sl_count"]) if risk_row else 0
    monthly_halted = bool(risk_row["monthly_halted"]) if risk_row else False

    return {
        "CHIEF": {
            "status": "HALTED" if monthly_halted else st.state,
            "detail": f"Mode: {'DRY-RUN' if CONFIG.dry_run_mode else 'LIVE/DEMO'} | TF: {CONFIG.entry_timeframe}",
        },
        "SCAN": {
            "status": f"{funnel['passed_dxy_corr']} / {funnel['total_m5_bars']}",
            "detail": f"Zona Armed: {funnel['zones_armed']}{mtf_txt}",
        },
        "VET": {
            "status": f"{funnel['sweeps_detected']} Sweep | {funnel['choch_confirmed']} CHoCH",
            "detail": f"Lolos spread: {funnel['passed_spread']}",
        },
        "SIZE": {
            "status": f"{CONFIG.risk_per_trade_pct:.2f}%",
            "detail": f"Policy 0.01: {CONFIG.min_lot_partial_policy}",
        },
        "FILLS": {
            "status": f"{funnel['orders_sent']} Order | {funnel['dry_run_signals']} Dry-Run",
            "detail": f"Ditolak min lot: {funnel['rejected_min_lot']}",
        },
        "RISK": {
            "status": "HALTED" if monthly_halted else ("STOP_TODAY" if daily_sl >= CONFIG.daily_max_sl_count else "OK"),
            "detail": f"Daily SL: {daily_sl}/{CONFIG.daily_max_sl_count} | DD Mode: {CONFIG.monthly_dd_mode}",
        },
    }


def load_mtf_matrix(db_path: Path = DB_PATH) -> Optional[Dict[str, Any]]:
    """Baca snapshot Multi-Timeframe Matrix (D1..M5) terakhir dari SQLite."""
    if not Path(db_path).exists():
        return None
    raw = StateManager(db_path).get_meta("mtf_matrix_json")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def load_activity_tape(db_path: Path = DB_PATH, limit: int = 50) -> pd.DataFrame:
    """Muat riwayat penolakan gerbang & sinyal dry-run terbaru untuk Activity Tape."""
    if not Path(db_path).exists():
        return pd.DataFrame(columns=["timestamp_utc", "event_type", "detail"])

    with sqlite3.connect(db_path) as conn:
        skips = pd.read_sql_query(
            "SELECT timestamp_utc, 'SKIP [' || gate || ']' AS event_type, reason AS detail "
            "FROM skip_log ORDER BY id DESC LIMIT ?",
            conn,
            params=(limit,),
        )
        dry = pd.read_sql_query(
            "SELECT timestamp_utc, 'DRY-RUN ' || direction AS event_type, "
            "'lot=' || lot_size || ' entry=' || entry_price || ' sl=' || sl_price AS detail "
            "FROM dry_run_signals ORDER BY id DESC LIMIT ?",
            conn,
            params=(limit,),
        )

    combined = pd.concat([dry, skips], ignore_index=True)
    if combined.empty:
        return combined
    return combined.sort_values("timestamp_utc", ascending=False).head(limit).reset_index(drop=True)


def render_streamlit_dashboard() -> None:
    """Render UI Streamlit read-only saat dijalankan lewat `streamlit run dashboard/app.py`."""
    import streamlit as st

    st.set_page_config(page_title="XAUUSD Sniper Terminal", layout="wide")
    st.title("XAUUSD Sniper Quant Terminal (Read-Only)")
    st.caption(f"Updated UTC: {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC")

    cards = build_six_cards_snapshot(DB_PATH)
    cols = st.columns(6)
    for col, (name, data) in zip(cols, cards.items()):
        with col:
            st.metric(label=name, value=data["status"])
            st.caption(data["detail"])

    mtf = load_mtf_matrix(DB_PATH)
    if mtf is not None:
        st.divider()
        st.subheader(
            f"Multi-Timeframe Matrix ({mtf['aligned_count']}/5 Aligned | Bias: {mtf['dominant_bias']} | Entry TF: {mtf['recommended_entry_tf']})"
        )
        tf_rows = list(mtf.get("snapshots", {}).values())
        st.dataframe(pd.DataFrame(tf_rows), use_container_width=True, hide_index=True)

    st.divider()
    left, right = st.columns([1.2, 1.0])
    with left:
        st.subheader("Activity Tape (Terbaru)")
        tape = load_activity_tape(DB_PATH)
        st.dataframe(tape, use_container_width=True, hide_index=True)

    with right:
        st.subheader("Signal Funnel Minggu Ini")
        if DB_PATH.exists():
            funnel = StateManager(DB_PATH).get_funnel_stats(datetime.now(timezone.utc))
            st.dataframe(
                pd.DataFrame(
                    [{"Tahap": k, "Jumlah Bar M5": v} for k, v in funnel.items()]
                ),
                use_container_width=True,
                hide_index=True,
            )


if __name__ == "__main__":
    render_streamlit_dashboard()
