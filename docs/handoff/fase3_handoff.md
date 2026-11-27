# Handoff: XAUUSD "Sniper" Quant Engine (Fase 1–3 Lengkap)

Dokumen handoff ini bebas kredensial/data login.

## Status Terakhir
- **Fase 1 (Foundation & Guardrails):** Selesai (`7 passed`).
- **Fase 2 (Pure-Function Strategy Layer 1–4 & State Machine):** Selesai (`5 passed`).
- **Fase 3 (Intrabar M1 Backtester, Walk-Forward, Streamlit 6-Card UI & Runner):** Selesai (`5 passed`).
- **Total Suite:** Seluruh 17 unit & integration tests lulus.

## Komponen Fase 3
- `data/news_history.csv`: Kalender historis High Impact USD (`CPI`, `NFP`, `FOMC`) untuk backtest.
- `backtest/engine.py`:
  - Simulasi intrabar M1 dengan asumsi pesimis (`STOP_LOSS` dicek duluan bila ambigu di bar M1 yang sama).
  - Memakai spread historis per bar MT5 (`spread * point`) dan fungsi produksi `strategy/` + `guardrails/` yang identik dengan live.
  - Menggunakan `_InMemoryBacktestState` + caching H4/H1 untuk eksekusi cepat tanpa disk lock SQLite.
- `backtest/walk_forward.py`:
  - Walk-Forward rolling (`IS -> OOS`) yang mengunci batas maksimal 4 parameter inti (`pivot_window`, `sl_atr_multiplier`, `zone_armed_timeout_hours`, `final_tp_rrr`).
  - Evaluator 4 Kriteria Lulus OOS (`PF >= 1.3`, `Expectancy >= 0.2R`, `Max DD < 15%`, `Trades >= 50`).
- `dashboard/app.py`:
  - Streamlit *read-only* (tidak pernah mengimpor/memanggil `MetaTrader5`).
  - Tata letak 6 kartu (`CHIEF | SCAN | VET | SIZE | FILLS | RISK`), `Activity Tape`, dan tabel `Signal Funnel`.
- `main.py`:
  - Runner single-process MT5 (`SniperEngineRunner`) dengan ritme cepat untuk Position Manager (`IN_POSITION`) dan ritme `isNewBar` M5 untuk evaluasi sinyal (`dry_run_mode = True` secara default).

## Prosedur Menjalankan di MT5 Demo (Pra-Live)
1. Buka terminal **MetaTrader 5** di Windows, login akun **Demo**, dan aktifkan **Algo Trading**.
2. Audit ketersediaan data historis 3–5 tahun dan alignment DXY:
   ```powershell
   .\.venv\Scripts\python.exe tools/check_history_data.py --years 4
   ```
3. Ukur spread broker di sesi `07:00–16:00 UTC` untuk mengkalibrasi `max_spread_usd` di `config.py`:
   ```powershell
   .\.venv\Scripts\python.exe tools/measure_spread.py --minutes 60
   ```
4. Jalankan runner dalam mode diagnostik (`dry_run_mode=True`) dan pantau lewat dashboard Streamlit:
   ```powershell
   .\.venv\Scripts\python.exe main.py
   .\.venv\Scripts\streamlit.exe run dashboard/app.py
   ```
5. Lakukan **Forward Test Demo 1–2 bulan** dan pastikan **4 Kriteria Lulus OOS** terpenuhi di data historis 3–5 tahun sebelum mempertimbangkan Fase 4 (Full Web UI Next.js + FastAPI) atau akun real.
