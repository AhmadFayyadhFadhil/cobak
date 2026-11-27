"""Sistem logging terstruktur (console + rotating file) dan laporan Signal Funnel."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Mapping, Optional

from config import LOG_DIR


def setup_logger(
    name: str = "sniper_bot",
    log_dir: Path = LOG_DIR,
    level: int = logging.INFO,
) -> logging.Logger:
    """Inisialisasi logger console dan file log berotasi."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        log_dir / f"{name}.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def log_skip(logger: logging.Logger, bar_time_utc: str, reason: str, gate: str) -> None:
    """Catat alasan penolakan sinyal secara seragam agar mudah diaudit."""
    logger.info("Skipped [%s] @ %s: %s", gate, bar_time_utc, reason)


def log_execution(
    logger: logging.Logger,
    action: str,
    symbol: str,
    direction: str,
    volume: float,
    price: float,
    sl: float,
    tp: float,
    comment: str,
    dry_run: bool = False,
) -> None:
    """Catat detail eksekusi order atau simulasi mode diagnostik (dry-run)."""
    prefix = "DRY-RUN" if dry_run else "EXECUTED"
    logger.info(
        "%s [%s] %s %s lot=%.2f price=%.3f sl=%.3f tp=%.3f comment=%s",
        prefix,
        action,
        symbol,
        direction,
        volume,
        price,
        sl,
        tp,
        comment,
    )


def format_funnel_report(
    period_label: str,
    counts: Mapping[str, int],
    extra_notes: Optional[str] = None,
) -> str:
    """Format rekap Signal Funnel M5 untuk mendeteksi gerbang yang menyumbat."""
    lines = [
        f"=== Signal Funnel Report ({period_label}) ===",
        f"  Total bar M5 dievaluasi : {counts.get('total_m5_bars', 0)}",
        f"  Lolos sesi + news       : {counts.get('passed_session_news', 0)}",
        f"  Lolos spread            : {counts.get('passed_spread', 0)}",
        f"  Lolos korelasi DXY      : {counts.get('passed_dxy_corr', 0)}",
        f"  Zona ARMED              : {counts.get('zones_armed', 0)}",
        f"  Sweep terdeteksi        : {counts.get('sweeps_detected', 0)}",
        f"  CHoCH terkonfirmasi     : {counts.get('choch_confirmed', 0)}",
        f"  FVG + limit dipasang    : {counts.get('fvg_limits_placed', 0)}",
        f"  Ditolak lot < minimum   : {counts.get('rejected_min_lot', 0)}",
        f"  Order dikirim           : {counts.get('orders_sent', 0)}",
        f"  Sinyal Dry-Run          : {counts.get('dry_run_signals', 0)}",
    ]
    if extra_notes:
        lines.append(f"  Catatan                 : {extra_notes}")
    return "\n".join(lines)
