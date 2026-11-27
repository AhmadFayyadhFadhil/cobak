r"""CLI Runner Backtest & Walk-Forward untuk data MT5 maupun file CSV/Parquet 10-20 tahun.

Contoh:
  .\.venv\Scripts\python.exe run_backtest.py --source mt5 --balance 1000
  .\.venv\Scripts\python.exe run_backtest.py --source csv --xau data/xau_20y_m1.csv --dxy data/dxy_20y_m1.csv --atr-relative
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from backtest.data_loader import build_20y_dataset
from backtest.engine import BacktestEngine, BacktestResult, load_news_history_csv
from backtest.walk_forward import evaluate_oos_pass_criteria
from config import CONFIG, DATA_DIR, BotConfig
from core.dxy_provider import DXYProvider
from core.logger import format_funnel_report
from core.mt5_client import MT5Client


def _fetch_mt5_dataset(cfg: BotConfig) -> tuple[Dict[str, pd.DataFrame], float, str]:
    """Tarik histori maksimum XAUUSD (M1/M5/H1/H4/D1) dan DXY H4 dari terminal MT5."""
    client = MT5Client(cfg)
    if not client.connect():
        raise RuntimeError("Gagal terhubung ke MetaTrader 5. Pastikan terminal MT5 terbuka & login.")

    try:
        xau = client.resolve_symbol(cfg.symbol_aliases)
        if xau is None:
            raise RuntimeError("Simbol XAUUSD tidak ditemukan di MT5.")
        spec = client.get_symbol_spec(xau)
        offset_hours = client.calibrate_utc_offset_hours(xau)

        mt5 = client.mt5
        m1 = client.fetch_rates_df(xau, mt5.TIMEFRAME_M1, 50000, offset_hours)
        m5 = client.fetch_rates_df(xau, mt5.TIMEFRAME_M5, 50000, offset_hours)
        h1 = client.fetch_rates_df(xau, mt5.TIMEFRAME_H1, 25000, offset_hours)
        h4 = client.fetch_rates_df(xau, mt5.TIMEFRAME_H4, 10000, offset_hours)
        d1 = client.fetch_rates_df(xau, mt5.TIMEFRAME_D1, 3000, offset_hours)

        # Pastikan DXY H4 menjangkau cukup jauh ke belakang sesuai rentang M5
        dxy_res = DXYProvider(client, cfg).fetch_dxy_h4(2000, offset_hours)
        if dxy_res.df.empty:
            raise RuntimeError("Data DXY H4 tidak tersedia di MT5.")

        dataset = {
            "M1": m1,
            "M5": m5,
            "H1": h1,
            "H4": h4,
            "D1": d1,
            "DXY_H4": dxy_res.df,
        }
        return dataset, spec.point, f"MT5 ({xau}, digits={spec.digits}, DXY={dxy_res.source_mode})"
    finally:
        client.shutdown()


def save_backtest_report(
    result: BacktestResult,
    source_label: str,
    date_range: str,
    out_path: Path = DATA_DIR / "backtest_latest.json",
) -> None:
    """Simpan ringkasan backtest ke JSON agar bisa dibaca oleh Dashboard Streamlit."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "source": source_label,
        "date_range": date_range,
        "initial_balance": result.initial_balance,
        "final_balance": round(result.final_balance, 2),
        "total_trades": result.total_trades,
        "win_rate": round(result.win_rate, 2),
        "profit_factor": round(result.profit_factor, 2),
        "expectancy_r": round(result.expectancy_r, 3),
        "max_drawdown_pct": round(result.max_drawdown_pct, 2),
        "funnel": result.funnel,
        "regime_breakdown": result.regime_breakdown,
        "news_coverage_note": result.news_coverage_note,
        "trades": [
            {
                "direction": t.direction,
                "entry_time_utc": t.entry_time_utc.isoformat() if t.entry_time_utc else "",
                "exit_time_utc": t.exit_time_utc.isoformat() if t.exit_time_utc else "",
                "entry_price": round(t.entry_price, 3),
                "sl_price": round(t.sl_price, 3),
                "tp_price": round(t.tp_price, 3),
                "hit_partial": t.hit_partial,
                "exit_reason": t.exit_reason,
                "realized_r": round(t.realized_r, 2),
            }
            for t in result.trades
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(
    source: str,
    balance: float,
    xau_csv: Optional[Path],
    dxy_csv: Optional[Path],
    atr_relative: bool,
    save_json: bool = True,
) -> BacktestResult:
    cfg = CONFIG
    if atr_relative:
        cfg = replace(cfg, max_spread_atr_ratio=0.25, min_trail_step_atr_ratio=0.15)

    if source == "mt5":
        dataset, point, source_label = _fetch_mt5_dataset(cfg)
    else:
        if xau_csv is None or dxy_csv is None:
            raise ValueError("Untuk --source csv wajib menyertakan --xau dan --dxy.")
        dataset = build_20y_dataset(xau_csv, dxy_h4_or_m1_path=dxy_csv, cfg=cfg)
        point = 0.01
        source_label = f"CSV ({xau_csv.name})"

    m5 = dataset["M5"]
    if m5.empty:
        raise RuntimeError("Data M5 kosong, backtest tidak dapat dijalankan.")

    start_dt = pd.Timestamp(m5["time_utc"].iloc[0]).strftime("%Y-%m-%d")
    end_dt = pd.Timestamp(m5["time_utc"].iloc[-1]).strftime("%Y-%m-%d")
    date_range = f"{start_dt} s/d {end_dt}"

    news_df = load_news_history_csv(cfg.news_history_csv)
    result = BacktestEngine(cfg, point=point).run(
        m5_df=m5,
        h1_df=dataset["H1"],
        h4_df=dataset["H4"],
        dxy_h4=dataset["DXY_H4"],
        m1_df=dataset.get("M1"),
        news_history_df=news_df,
        initial_balance=balance,
        d1_df=dataset.get("D1"),
        m15_df=dataset.get("M15"),
    )

    if save_json:
        save_backtest_report(result, source_label, date_range)

    criteria = evaluate_oos_pass_criteria(
        result.profit_factor, result.expectancy_r, result.max_drawdown_pct, result.total_trades
    )

    print(f"\n=== HASIL BACKTEST XAUUSD SNIPER ({source_label}) ===")
    print(f"Periode Data   : {date_range} ({len(m5):,} bar M5)")
    print(f"Saldo Awal     : ${result.initial_balance:,.2f}")
    print(f"Saldo Akhir    : ${result.final_balance:,.2f}")
    print(f"Total Trade    : {result.total_trades}")
    print(f"Win Rate       : {result.win_rate:.1f}%")
    print(f"Profit Factor  : {result.profit_factor:.2f}")
    print(f"Expectancy     : {result.expectancy_r:.2f}R per trade")
    print(f"Max Drawdown   : {result.max_drawdown_pct:.2f}%")
    print(f"Cek News       : {result.news_coverage_note}")
    print(f"Status Kriteria: {'LULUS' if criteria.passed else 'BELUM LULUS -> ' + ', '.join(criteria.failed_reasons)}")
    if result.regime_breakdown:
        print("\n--- Breakdown per 5 Tahun ---")
        for blk, stats in result.regime_breakdown.items():
            print(
                f"  {blk}: {stats['trades']} trade | WR={stats['win_rate']:.1f}% | "
                f"PF={stats['profit_factor']:.2f} | Exp={stats['expectancy_r']:.2f}R"
            )
    print("\n" + format_funnel_report(date_range, result.funnel))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jalankan backtest XAUUSD Sniper dari MT5 atau CSV")
    parser.add_argument("--source", choices=("mt5", "csv"), default="mt5")
    parser.add_argument("--balance", type=float, default=1000.0)
    parser.add_argument("--xau", type=Path, default=None)
    parser.add_argument("--dxy", type=Path, default=None)
    parser.add_argument("--atr-relative", action="store_true", help="Gunakan spread & trail berbasis ATR (untuk 10-20 tahun)")
    args = parser.parse_args()
    main(args.source, args.balance, args.xau, args.dxy, args.atr_relative)
