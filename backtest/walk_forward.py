"""Walk-Forward Analysis bergulir (IS -> OOS) dan evaluator 4 Kriteria Lulus OOS
yang dikunci sebelum eksekusi untuk mencegah overfitting.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from backtest.engine import BacktestEngine, TradeOutcome
from config import CONFIG, BotConfig

ALLOWED_WF_PARAMS = frozenset(
    {"pivot_window", "sl_atr_multiplier", "zone_armed_timeout_hours", "final_tp_rrr"}
)


@dataclass(frozen=True)
class OOSPassCriteriaResult:
    """Hasil uji kelulusan Out-of-Sample terhadap 4 syarat pra-komitmen."""

    passed: bool
    profit_factor: float
    expectancy_r: float
    max_drawdown_pct: float
    trade_count: int
    failed_reasons: List[str]


@dataclass(frozen=True)
class WalkForwardWindowResult:
    """Hasil satu jendela bergulir In-Sample -> Out-of-Sample."""

    window_index: int
    best_params: Dict[str, Any]
    is_expectancy_r: float
    oos_trades: List[TradeOutcome]
    oos_max_dd_pct: float


@dataclass(frozen=True)
class WalkForwardSummary:
    """Agregasi seluruh hasil Out-of-Sample lintas jendela Walk-Forward."""

    windows: List[WalkForwardWindowResult]
    oos_criteria: OOSPassCriteriaResult


def evaluate_oos_pass_criteria(
    profit_factor: float,
    expectancy_r: float,
    max_drawdown_pct: float,
    trade_count: int,
) -> OOSPassCriteriaResult:
    """Evaluasi 4 Kriteria Lulus OOS: PF >= 1.3, Exp >= 0.2R, MaxDD < 15%, Trades >= 50."""
    failures: List[str] = []
    if profit_factor < 1.3:
        failures.append(f"Profit Factor OOS {profit_factor:.2f} < 1.30")
    if expectancy_r < 0.2:
        failures.append(f"Expectancy OOS {expectancy_r:.2f}R < 0.20R")
    if max_drawdown_pct >= 15.0:
        failures.append(f"Max Drawdown OOS {max_drawdown_pct:.2f}% >= 15.0%")
    if trade_count < 50:
        failures.append(f"Jumlah Trade OOS {trade_count} < 50")

    return OOSPassCriteriaResult(
        passed=(len(failures) == 0),
        profit_factor=profit_factor,
        expectancy_r=expectancy_r,
        max_drawdown_pct=max_drawdown_pct,
        trade_count=trade_count,
        failed_reasons=failures,
    )


def run_walk_forward(
    m5_df: pd.DataFrame,
    h1_df: pd.DataFrame,
    h4_df: pd.DataFrame,
    dxy_h4: pd.DataFrame,
    m1_df: Optional[pd.DataFrame],
    news_history_df: pd.DataFrame,
    n_splits: int,
    param_grid: Sequence[Dict[str, Any]],
    base_cfg: BotConfig = CONFIG,
) -> WalkForwardSummary:
    """Jalankan Walk-Forward rolling (IS -> OOS) dengan maksimal 4 parameter inti."""
    for p in param_grid:
        unknown = set(p.keys()) - ALLOWED_WF_PARAMS
        if unknown:
            raise ValueError(
                f"Parameter tidak diizinkan di Walk-Forward (risiko overfit): {sorted(unknown)}"
            )

    n = len(m5_df)
    if n_splits < 1 or n < 100:
        raise ValueError("Data atau n_splits tidak mencukupi untuk Walk-Forward.")

    chunk = n // (n_splits + 1)
    windows: List[WalkForwardWindowResult] = []
    all_oos_trades: List[TradeOutcome] = []
    worst_dd = 0.0

    for w in range(n_splits):
        is_start = w * chunk
        is_end = (w + 1) * chunk
        oos_end = min(n, (w + 2) * chunk)

        m5_is = m5_df.iloc[is_start:is_end].copy()
        m5_oos = m5_df.iloc[is_end:oos_end].copy()

        best_score = -1e9
        best_params = dict(param_grid[0])
        best_is_exp = 0.0

        for params in param_grid:
            cfg = replace(base_cfg, **params)
            res_is = BacktestEngine(cfg).run(m5_is, h1_df, h4_df, dxy_h4, m1_df, news_history_df)
            score = res_is.expectancy_r * max(1, res_is.total_trades)
            if score > best_score:
                best_score = score
                best_params = dict(params)
                best_is_exp = res_is.expectancy_r

        best_cfg = replace(base_cfg, **best_params)
        res_oos = BacktestEngine(best_cfg).run(m5_oos, h1_df, h4_df, dxy_h4, m1_df, news_history_df)
        windows.append(
            WalkForwardWindowResult(
                window_index=w + 1,
                best_params=best_params,
                is_expectancy_r=best_is_exp,
                oos_trades=res_oos.trades,
                oos_max_dd_pct=res_oos.max_drawdown_pct,
            )
        )
        all_oos_trades.extend(res_oos.trades)
        worst_dd = max(worst_dd, res_oos.max_drawdown_pct)

    r_vals = [t.realized_r for t in all_oos_trades]
    wins = sum(r for r in r_vals if r > 0)
    losses = sum(abs(r) for r in r_vals if r < 0)
    pf = (wins / losses) if losses > 0 else (99.0 if wins > 0 else 0.0)
    exp_r = float(np.mean(r_vals)) if r_vals else 0.0

    criteria = evaluate_oos_pass_criteria(pf, exp_r, worst_dd, len(all_oos_trades))
    return WalkForwardSummary(windows=windows, oos_criteria=criteria)
