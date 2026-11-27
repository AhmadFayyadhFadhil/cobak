"""Utilitas audit ketersediaan & alignment data historis 3-5 tahun (M1, M5, H1, H4)
untuk XAUUSD dan instrumen DXY/ICE di terminal MT5 sebelum menjalankan Fase 3.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import MetaTrader5 as mt5  # type: ignore
import pandas as pd

from config import CONFIG, get_symbol_aliases_map
from core.dxy_provider import DXYProvider
from core.mt5_client import MT5Client

TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
}


def inspect_symbol_history(
    client: MT5Client, symbol: str, years: int, offset_hours: int
) -> None:
    """Cek rentang tanggal dan jumlah bar yang tersedia di MT5 untuk tiap timeframe."""
    now_utc = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(days=365 * years)
    start_srv = client.utc_to_server_ts(start_utc, offset_hours)
    end_srv = client.utc_to_server_ts(now_utc, offset_hours)

    print(f"\n--- Audit Historis: {symbol} (Target {years} tahun) ---")
    for tf_name, tf_code in TIMEFRAMES.items():
        rates = mt5.copy_rates_range(symbol, tf_code, start_srv, end_srv)
        note = ""
        if rates is None or len(rates) == 0:
            for fallback_cnt in (100000, 50000, 20000, 5000):
                rates = mt5.copy_rates_from_pos(symbol, tf_code, 0, fallback_cnt)
                if rates is not None and len(rates) > 0:
                    note = f" (fallback from_pos={fallback_cnt:,})"
                    break
        if rates is None or len(rates) == 0:
            print(f"  {tf_name}: KOSONG / tidak tersedia di MT5!")
            continue
        df = pd.DataFrame(rates)
        first_dt = client.server_ts_to_utc(int(df["time"].iloc[0]), offset_hours)
        last_dt = client.server_ts_to_utc(int(df["time"].iloc[-1]), offset_hours)
        span_days = (last_dt - first_dt).days
        print(
            f"  {tf_name}: {len(df):>7,} bar | {first_dt:%Y-%m-%d} s/d {last_dt:%Y-%m-%d} "
            f"({span_days / 365.25:.2f} tahun) | spread_median={df['spread'].median():.1f} pts{note}"
        )


def main(years: int) -> None:
    client = MT5Client(CONFIG)
    if not client.connect():
        raise RuntimeError("Gagal terhubung ke terminal MT5 Desktop.")

    try:
        xau = client.resolve_symbol(CONFIG.symbol_aliases)
        if xau is None:
            raise RuntimeError("Simbol XAUUSD tidak ditemukan.")
        spec = client.get_symbol_spec(xau)
        offset_hours = client.calibrate_utc_offset_hours(xau)
        print(f"XAUUSD aktif: {xau} (digits={spec.digits}, UTC offset={offset_hours:+d}h)")
        inspect_symbol_history(client, xau, years, offset_hours)

        dxy = client.resolve_symbol(CONFIG.dxy_symbol_candidates)
        if dxy is not None and DXYProvider.is_valid_broker_dxy(client.mt5, dxy):
            inspect_symbol_history(client, dxy, years, offset_hours)
        else:
            if dxy is not None:
                print(f"\nSimbol {dxy} diabaikan (bukan indeks DXY valid); memakai 6 pair ICE sintetis:")
            else:
                print("\nSimbol DXY langsung tidak ditemukan; mengecek 6 pair ICE sintetis:")
            for canonical, aliases in get_symbol_aliases_map(CONFIG).items():
                resolved = client.resolve_symbol(aliases)
                if resolved is None:
                    print(f"  {canonical}: TIDAK TERSEDIA di broker!")
                else:
                    inspect_symbol_history(client, resolved, years, offset_hours)
    finally:
        client.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cek ketersediaan data historis MT5 3-5 tahun")
    parser.add_argument("--years", type=int, default=4, help="Target tahun data historis")
    args = parser.parse_args()
    main(args.years)
