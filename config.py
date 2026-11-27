"""Konfigurasi terpusat XAUUSD "Sniper" Quantitative Trading Engine.

Semua parameter risiko, sesi UTC, batas spread, Magic Number, kebijakan lot minimum,
dan State Machine ditempatkan di sini agar mudah diaudit dan disesuaikan.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Tuple

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
DB_PATH = DATA_DIR / "sniper_state.db"

MinLotPolicy = Literal["SKIP", "NO_PARTIAL"]
MonthlyDDMode = Literal["START_BALANCE", "PEAK_BALANCE", "PEAK_EQUITY"]
EntryTimeframeMode = Literal["M5", "M15", "M15_OR_M5", "H1"]


@dataclass(frozen=True)
class BotConfig:
    """Konfigurasi utama bot trading XAUUSD Sniper."""

    # --- 1. Eksekusi & Simbol Broker ---
    symbol: str = "XAUUSD"
    symbol_aliases: Tuple[str, ...] = ("XAUUSD", "XAUUSDm", "XAUUSD.a", "GOLD")
    dxy_symbol_candidates: Tuple[str, ...] = ("USDX", "DXY", "DX-Y.NYB", "USDX.a", "DXY.a")
    magic_number: int = 20261127
    order_comment_prefix: str = "SNP_"  # MT5 limit <= 31 karakter; dijaga <= 31 saat runtime
    dry_run_mode: bool = True  # True: mode diagnostik (log saja tanpa kirim order ke MT5)

    # --- 2. DXY Sintetis ICE Formula ---
    # 50.14348112 * EURUSD^-0.576 * USDJPY^0.136 * GBPUSD^-0.119
    #             * USDCAD^0.091 * USDSEK^0.042 * USDCHF^0.036
    ice_dxy_constant: float = 50.14348112
    ice_dxy_weights: Tuple[Tuple[str, float], ...] = (
        ("EURUSD", -0.576),
        ("USDJPY", 0.136),
        ("GBPUSD", -0.119),
        ("USDCAD", 0.091),
        ("USDSEK", 0.042),
        ("USDCHF", 0.036),
    )

    # --- 3. Risiko, RRR & Lot Sizing ---
    risk_per_trade_pct: float = 1.5
    weekly_reduced_risk_pct: float = 0.75
    partial_tp_rrr: float = 2.0
    partial_close_ratio: float = 0.5
    final_tp_rrr: float = 5.0  # Target akhir 1:5 hingga 1:6
    sl_atr_multiplier: float = 0.5
    trailing_atr_multiplier: float = 1.5
    atr_period: int = 14
    # Pergeseran SL minimum sebelum mengirim modify_order (dalam harga USD, misal $0.25)
    min_trail_step_usd: float = 0.25
    # Jika > 0 (misal 0.15), step trail otomatis mengikuti ATR M5 untuk backtest lintas era 20 tahun
    min_trail_step_atr_ratio: float = 0.0
    # Komisi round-turn per 1.00 lot dalam mata uang akun (misal $7.0 untuk akun Raw/ECN)
    commission_per_lot_account: float = 0.0
    # Akun Standard < $1,500: "NO_PARTIAL" mencegah sinyal 0.01 lot terblokir total.
    # Ubah ke "SKIP" jika modal >= $1,500-$2,000 atau memakai akun Cent.
    min_lot_partial_policy: MinLotPolicy = "NO_PARTIAL"

    # --- 4. Layer 1: Sesi UTC, News Filter & Korelasi DXY ---
    session_start_utc: str = "07:00"
    session_end_utc: str = "16:00"
    news_before_minutes: int = 30
    news_after_minutes: int = 15
    news_events_json: Path = DATA_DIR / "news_events.json"
    news_history_csv: Path = DATA_DIR / "news_history.csv"
    news_max_age_hours: int = 168  # Fail-closed jika jadwal live lebih tua dari 7 hari
    corr_window_h4: int = 30  # Rentang 20-50 candle H4
    corr_negative_threshold: float = -0.5
    corr_anomaly_threshold: float = -0.3

    # --- 5. Layer 2 & 3: Struktur, SMC, Multi-Timeframe & State Machine ---
    entry_timeframe: EntryTimeframeMode = "H1"  # "H1" | "M5" | "M15" | "M15_OR_M5"
    require_d1_alignment: bool = False  # True: wajib searah struktur Daily (D1)
    allow_xau_only_when_dxy_unavailable: bool = True  # Fallback ke struktur XAUUSD H4 bila DXY historis tidak ada
    pivot_window: int = 5  # Konfirmasi pivot baru valid pada bar i + pivot_window
    displacement_body_multiplier: float = 1.5
    displacement_avg_period: int = 10
    require_tick_volume_filter: bool = False  # Opsional karena tick volume antar broker bervariasi
    zone_armed_timeout_hours: float = 36.0  # 36 jam (~36 bar H1) untuk H1 mode
    max_fvg_wait_candles_m5: int = 5  # 3-6 candle setelah CHoCH
    pending_order_expiry_hours: float = 12.0  # Maksimal batas expiry pending order
    fvg_proximal_offset_ratio: float = 0.5  # 0.5 = 50% FVG; 0.2 = dekat tepi FVG menghadap harga

    # --- 6. Guardrails Anti-Ganas & Eksekusi ---
    # Gunakan batas spread dalam jarak harga USD agar kebal perbedaan broker 2-digit vs 3-digit.
    # Kalibrasi nilai ini menggunakan tools/measure_spread.py (persentil ke-90 sesi 07-16 UTC).
    max_spread_usd: float = 0.35
    # Jika > 0 (misal 0.25), batas spread backtest 20 tahun diskalakan terhadap ATR(14) M5
    max_spread_atr_ratio: float = 0.0
    cooldown_minutes: int = 60
    daily_max_sl_count: int = 2
    daily_max_loss_pct: float = 3.0
    weekly_max_loss_pct: float = 6.0
    monthly_max_dd_pct: float = 10.0
    monthly_dd_mode: MonthlyDDMode = "PEAK_BALANCE"

    # --- 7. Retry & Stabilitas MT5 ---
    max_order_retries: int = 5
    retry_delay_ms: int = 500
    fresh_tick_max_age_seconds: int = 120
    poll_interval_seconds: float = 1.0


CONFIG = BotConfig()


def get_symbol_aliases_map(cfg: BotConfig = CONFIG) -> Dict[str, Tuple[str, ...]]:
    """Daftar alias standar untuk instrumen DXY sintetis bila broker memakai suffix."""
    return {
        pair: (pair, f"{pair}m", f"{pair}.a", f"{pair}.r")
        for pair, _ in cfg.ice_dxy_weights
    }
