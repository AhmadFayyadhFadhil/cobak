"""Utilitas pengukur spread XAUUSD broker selama sesi London & New York (07:00-16:00 UTC)
untuk menentukan `max_spread_usd` berdasarkan persentil ke-90 (tidak menebak-nebak).
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config import CONFIG, DATA_DIR
from core.mt5_client import MT5Client


def measure_spread(duration_minutes: float, sample_interval_sec: float, output_csv: Path) -> None:
    """Rekam tick spread XAUUSD dari MT5 dan tampilkan ringkasan statistik."""
    client = MT5Client(CONFIG)
    if not client.connect():
        raise RuntimeError("Gagal terhubung ke terminal MT5 Desktop.")

    try:
        symbol = client.resolve_symbol(CONFIG.symbol_aliases)
        if symbol is None:
            raise RuntimeError("Simbol XAUUSD tidak ditemukan di MT5.")
        spec = client.get_symbol_spec(symbol)
        offset_hours = client.calibrate_utc_offset_hours(symbol)
        print(
            f"Memulai pengukuran spread {symbol} (digits={spec.digits}, point={spec.point}, "
            f"UTC offset={offset_hours:+d}h) selama {duration_minutes} menit..."
        )

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        end_ts = time.time() + duration_minutes * 60.0
        spreads_usd: list[float] = []

        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_utc", "bid", "ask", "spread_usd", "spread_points"])

            while time.time() < end_ts:
                now_utc = datetime.now(timezone.utc)
                bid, ask, _ = client.get_current_bid_ask(symbol)
                spread_usd = max(0.0, ask - bid)
                spread_points = int(round(spread_usd / spec.point)) if spec.point > 0 else 0
                spreads_usd.append(spread_usd)
                writer.writerow(
                    [now_utc.isoformat(), bid, ask, f"{spread_usd:.4f}", spread_points]
                )
                time.sleep(sample_interval_sec)

        if spreads_usd:
            arr = np.array(spreads_usd, dtype=float)
            p90 = float(np.percentile(arr, 90))
            print("\n=== Hasil Kalibrasi Spread Broker ===")
            print(f"Sampel       : {len(arr)}")
            print(f"Rata-rata    : ${arr.mean():.3f}")
            print(f"Median (p50) : ${np.percentile(arr, 50):.3f}")
            print(f"Persentil 90 : ${p90:.3f}  <-- Rekomendasi `max_spread_usd` di config.py")
            print(f"Persentil 95 : ${np.percentile(arr, 95):.3f}")
            print(f"Maksimum     : ${arr.max():.3f}")
            print(f"CSV disimpan : {output_csv}")
    finally:
        client.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ukur spread live XAUUSD untuk kalibrasi config.py")
    parser.add_argument("--minutes", type=float, default=15.0, help="Durasi sampling dalam menit")
    parser.add_argument("--interval", type=float, default=2.0, help="Jeda sampling dalam detik")
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_DIR / "spread_samples.csv",
        help="Path output CSV sampel spread",
    )
    args = parser.parse_args()
    measure_spread(args.minutes, args.interval, args.out)
