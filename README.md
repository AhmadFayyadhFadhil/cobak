# cobak

XAUUSD "Sniper" Quantitative Trading Engine (Python + MetaTrader 5).

## Fitur Utama (Fase 1)
- **Konfigurasi Terpusat (`config.py`):** Risiko, RRR, sesi UTC (`07:00-16:00`), kebijakan lot minimum, dan parameter State Machine.
- **Koneksi MT5 Single-Process (`core/mt5_client.py`):** Deteksi otomatis broker 2 vs 3 digit, kalibrasi UTC offset *fresh-tick only*, dan *idempotent auto-retry* (5x @ 500ms) dengan pencocokan cadangan multi-field.
- **DXY Provider (`core/dxy_provider.py`):** Simbol broker langsung atau DXY sintetis rumus ICE (dengan fallback 5-pair dan *inner-join timestamp alignment*).
- **SQLite State & Funnel (`core/state_manager.py`, `core/logger.py`):** Persistensi State Machine, reset Circuit Breaker berbasis UTC, dan *Signal Funnel Tracker*.
- **Pengaman Anti-Ganas (`guardrails/anti_ganas.py`):** `isNewBar`, Single Position Guard, Max Spread USD, Cooldown 60 menit, dan Circuit Breaker 3 tingkat.
- **Utilitas Diagnostik (`tools/`):** `measure_spread.py` (persentil ke-90) dan `check_history_data.py` (audit data historis 3-5 tahun).

## Menjalankan Unit Test
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```
