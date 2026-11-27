# Handoff: XAUUSD "Sniper" Quant Engine (Fase 1 & Fase 2)

Dokumen handoff ini bebas kredensial/data login.

## Status Terakhir
- **Fase 1 (Foundation & Guardrails):** Selesai dan terverifikasi (`7 passed`).
- **Fase 2 (Pure-Function Strategy Layer 1–4 & State Machine):** Selesai dengan pendekatan `sniper-tdd` (`12 passed in 3.81s`).

## Komponen Terpasang
- `config.py`: Parameter terpusat (`dry_run_mode=True`, `min_lot_partial_policy="NO_PARTIAL"`, `monthly_dd_mode="PEAK_BALANCE"`).
- `core/`: `mt5_client.py` (2/3 digit, fresh-tick UTC offset, idempotent retry + multi-field fallback), `dxy_provider.py` (ICE 6-pair & fallback 5-pair inner-join), `state_manager.py` (SQLite State Machine, reset UTC, funnel counters), `logger.py`.
- `guardrails/anti_ganas.py`: `isNewBar`, Single Position, Max Spread USD, Cooldown 60m, Circuit Breaker 3 tingkat berbasis UTC.
- `strategy/`:
  - `indicators.py`: `compute_atr`, `detect_confirmed_pivots` (valid di `k + window`), `compute_rolling_correlation`.
  - `layer1_macro.py`: Sesi `07:00–16:00 UTC`, News fail-closed (`JSON` live / `CSV` backtest), korelasi DXY (`< -0.5`, grey zone `-0.5..-0.3`, `> -0.3`).
  - `layer2_structure.py`: BOS objektif, zona SnD fresh, prioritas multi-zona (`H4 > H1`).
  - `layer3_smc_m5.py`: State Machine `IDLE -> ZONE_ARMED -> SWEEP_DETECTED -> CHOCH_CONFIRMED -> PENDING -> IN_POSITION -> COOLDOWN`, timeout zona lintas state, FVG wait maks 5 bar M5, `PENDING` batal saat sesi tutup, dan `IN_POSITION` kebal reset sesi.
  - `layer4_risk_exit.py`: Lot via `mt5.order_calc_profit()`, policy `0.01`, BEP + spread + komisi, Partial 50% @ `1:2`, ATR Trailing hysteresis, dan Realized R (`3.5R` / `+1.0R`).
- `tools/`: `measure_spread.py`, `check_history_data.py`.

## Langkah Berikutnya (Fase 3)
1. Siapkan `data/news_history.csv` (CPI, NFP, FOMC 3–5 tahun) dan cek data M1/M5/H1/H4 via `tools/check_history_data.py`.
2. Bangun `backtest/engine.py` (resolusi intrabar M1 asumsi pesimis SL-first, spread historis bar) dan `backtest/walk_forward.py` (3–4 parameter inti + 4 Kriteria Lulus OOS).
3. Bangun `dashboard/app.py` (Streamlit *read-only* dengan tata letak 6 kartu `CHIEF / SCAN / VET / SIZE / FILLS / RISK`, `Activity tape`, dan zoom M5).
4. Bangun `main.py` runner live/demo dan lakukan forward test Demo MT5 1–2 bulan sebelum Fase 4.
