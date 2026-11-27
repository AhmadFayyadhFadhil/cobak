# cobak

XAUUSD "Sniper" Quantitative Trading Engine (Python + MetaTrader 5).

## Fitur Utama (Fase 1 – Fase 3)
- **Konfigurasi Terpusat (`config.py`):** Risiko, RRR, sesi UTC (`07:00-16:00`), kebijakan lot minimum, dan parameter State Machine.
- **Koneksi MT5 Single-Process (`core/mt5_client.py`):** Deteksi otomatis broker 2 vs 3 digit, kalibrasi UTC offset *fresh-tick only*, dan *idempotent auto-retry* (5x @ 500ms) dengan pencocokan cadangan multi-field.
- **DXY Provider (`core/dxy_provider.py`):** Simbol broker langsung atau DXY sintetis rumus ICE (dengan fallback 5-pair dan *inner-join timestamp alignment*).
- **SQLite State & Funnel (`core/state_manager.py`, `core/logger.py`):** Persistensi State Machine, reset Circuit Breaker berbasis UTC, dan *Signal Funnel Tracker*.
- **Pengaman Anti-Ganas (`guardrails/anti_ganas.py`):** `isNewBar`, Single Position Guard, Max Spread USD, Cooldown 60 menit, dan Circuit Breaker 3 tingkat.
- **Strategi 4 Layer & MTF Matrix (`strategy/`):** Pivot anti *look-ahead*, filter sesi/news/korelasi DXY, analisa seluruh timeframe (`D1 → H4 → H1 → M15 → M5`) via `mtf_analyzer.py`, BOS + SnD *fresh* (`H4 > H1`), State Machine SMC (`M5` / `M15` / `M15_OR_M5`), serta `order_calc_profit()` lot sizing, Partial `1:2`, BEP + komisi, & ATR Trailing Stop.
- **Backtest 10–20 Tahun & Walk-Forward (`backtest/`):** Loader CSV/Parquet eksternal (`data_loader.py` untuk Dukascopy/HistData/MT5), simulasi intrabar M1 pesimis (*SL-first*), optimasi $O(N)$, opsi parameter ATR-relatif, breakdown per 5 tahun, `data/news_history.csv`, dan evaluator 4 Kriteria Lulus OOS.
- **Dashboard & Runner (`dashboard/app.py`, `main.py`):** Streamlit *read-only* 6 kartu (`CHIEF / SCAN / VET / SIZE / FILLS / RISK`) + `Activity Tape` dan runner single-process (`dry_run_mode=True` default).
- **Utilitas Diagnostik (`tools/`):** `measure_spread.py` (persentil ke-90) dan `check_history_data.py` (audit data historis 3-5 tahun).

## Menjalankan Unit Test
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

## Referensi Arsitektur & Dokumentasi Skill
- Pemetaan skill & adaptasi (termasuk adaptasi selektif pola `TauricResearch/TradingAgents` untuk layer AI reasoning di Fase 4): [`docs/Skill AI – Project XAUUSD Sniper.xlsx`](docs/Skill%20AI%20%E2%80%93%20Project%20XAUUSD%20Sniper.xlsx)

