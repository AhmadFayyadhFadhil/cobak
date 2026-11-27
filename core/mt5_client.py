"""Wrapper single-process MetaTrader 5: deteksi 2/3 digit XAUUSD, kalibrasi UTC offset
berbasis fresh-tick, dan eksekusi idempotent dengan auto-retry 5x @ 500ms.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable, Optional, Sequence

import pandas as pd

from config import CONFIG, BotConfig
from core.state_manager import StateManager

# Kode retcode MT5 standar (disediakan juga sebagai konstanta agar bisa di-test tanpa MT5)
TRADE_RETCODE_DONE = 10009
TRADE_RETCODE_PLACED = 10008
TRADE_RETCODE_REQUOTE = 10004
TRADE_RETCODE_PRICE_OFF = 10021
TRADE_RETCODE_PRICE_CHANGED = 10020
TRADE_RETCODE_CONNECTION = 10031
TRADE_RETCODE_TIMEOUT = 10012

RETRYABLE_RETCODES = {
    TRADE_RETCODE_REQUOTE,
    TRADE_RETCODE_PRICE_OFF,
    TRADE_RETCODE_PRICE_CHANGED,
    TRADE_RETCODE_CONNECTION,
    TRADE_RETCODE_TIMEOUT,
}


@dataclass(frozen=True)
class SymbolSpec:
    """Spesifikasi simbol MT5 yang menormalkan broker 2-digit vs 3-digit."""

    name: str
    digits: int
    point: float
    trade_tick_size: float
    trade_tick_value: float
    volume_min: float
    volume_max: float
    volume_step: float
    contract_size: float


@dataclass(frozen=True)
class OrderResult:
    """Hasil eksekusi order MT5 yang terstruktur."""

    success: bool
    retcode: int
    ticket: Optional[int]
    message: str
    attempts: int
    already_existed: bool = False


def make_unique_comment(prefix: str, bar_ts: int) -> str:
    """Buat comment order unik yang selalu mematuhi batas <= 31 karakter MT5."""
    comment = f"{prefix}{int(bar_ts)}"
    return comment[:31]


def is_market_likely_open_utc(now_utc: datetime) -> bool:
    """Heuristik konservatif jam buka market FX/XAUUSD untuk validasi fresh tick."""
    wd = now_utc.weekday()  # 0=Senin .. 6=Minggu
    hhmm = now_utc.hour * 60 + now_utc.minute
    if wd == 5:  # Sabtu
        return False
    if wd == 6 and hhmm < 22 * 60:  # Minggu sebelum 22:00 UTC
        return False
    if wd == 4 and hhmm >= 22 * 60:  # Jumat setelah 22:00 UTC
        return False
    return True


class MT5Client:
    """Klien single-process MetaTrader 5 yang mendukung injeksi modul untuk unit test."""

    def __init__(
        self,
        cfg: BotConfig = CONFIG,
        state_manager: Optional[StateManager] = None,
        mt5_module: Any = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cfg = cfg
        self.state_manager = state_manager
        self._sleep = sleep_fn
        if mt5_module is not None:
            self.mt5 = mt5_module
        else:
            import MetaTrader5 as mt5  # type: ignore

            self.mt5 = mt5
        self._connected = False

    def connect(self) -> bool:
        """Inisialisasi koneksi ke terminal MT5 Desktop."""
        if not self.mt5.initialize():
            return False
        self._connected = True
        return True

    def shutdown(self) -> None:
        """Tutup koneksi MT5 secara bersih."""
        if self._connected:
            self.mt5.shutdown()
            self._connected = False

    def resolve_symbol(self, candidates: Sequence[str]) -> Optional[str]:
        """Cari nama simbol yang aktif di broker (menangani suffix m/.a/dll)."""
        for sym in candidates:
            info = self.mt5.symbol_info(sym)
            if info is not None:
                if not getattr(info, "visible", True):
                    self.mt5.symbol_select(sym, True)
                return sym
        return None

    def get_symbol_spec(self, symbol: str) -> SymbolSpec:
        """Ambil spesifikasi presisi (2 vs 3 digit), volume step, dan tick size."""
        info = self.mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"Simbol MT5 tidak ditemukan: {symbol}")
        return SymbolSpec(
            name=symbol,
            digits=int(info.digits),
            point=float(info.point),
            trade_tick_size=float(info.trade_tick_size),
            trade_tick_value=float(info.trade_tick_value),
            volume_min=float(info.volume_min),
            volume_max=float(info.volume_max),
            volume_step=float(info.volume_step),
            contract_size=float(getattr(info, "trade_contract_size", 100.0)),
        )

    def get_account_snapshot(self) -> tuple[float, float]:
        """Kembalikan (balance, equity) akun MT5 saat ini."""
        acc = self.mt5.account_info()
        if acc is None:
            raise RuntimeError("Gagal membaca account_info dari MT5.")
        return float(acc.balance), float(acc.equity)

    def get_current_bid_ask(self, symbol: str) -> tuple[float, float, int]:
        """Kembalikan (bid, ask, tick_time_server) dari MT5."""
        tick = self.mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"Gagal membaca tick untuk {symbol}.")
        return float(tick.bid), float(tick.ask), int(tick.time)

    def get_spread_usd(self, symbol: str) -> float:
        """Hitung spread dalam jarak harga USD (kebal perbedaan 2-digit vs 3-digit)."""
        bid, ask, _ = self.get_current_bid_ask(symbol)
        return max(0.0, ask - bid)

    def calibrate_utc_offset_hours(
        self, symbol: str, now_utc: Optional[datetime] = None
    ) -> int:
        """Kalibrasi offset jam server MT5 terhadap UTC hanya saat tick masih fresh.

        Jika tick basi (misal weekend/market tutup), gunakan offset valid terakhir dari SQLite.
        """
        now_utc = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
        stored_offset_str = (
            self.state_manager.get_meta("server_utc_offset_hours")
            if self.state_manager is not None
            else None
        )
        fallback_offset = int(stored_offset_str) if stored_offset_str is not None else 0

        if not is_market_likely_open_utc(now_utc):
            return fallback_offset

        tick = self.mt5.symbol_info_tick(symbol)
        # Cold-start setelah symbol_select: beri jeda singkat bila tick pertama belum masuk
        for _ in range(3):
            if tick is not None and getattr(tick, "time", 0) > 0:
                break
            self._sleep(0.25)
            tick = self.mt5.symbol_info_tick(symbol)
        if tick is None or getattr(tick, "time", 0) <= 0:
            return fallback_offset

        raw_diff_sec = float(tick.time) - now_utc.timestamp()
        candidate_offset_hours = int(round(raw_diff_sec / 3600.0))
        if not (-12 <= candidate_offset_hours <= 14):
            return fallback_offset

        tick_utc_est = float(tick.time) - (candidate_offset_hours * 3600.0)
        tick_age = abs(now_utc.timestamp() - tick_utc_est)
        if tick_age > self.cfg.fresh_tick_max_age_seconds:
            return fallback_offset

        if self.state_manager is not None:
            self.state_manager.set_meta(
                "server_utc_offset_hours", str(candidate_offset_hours)
            )
            self.state_manager.set_meta("server_utc_offset_ts", now_utc.isoformat())
        return candidate_offset_hours

    def server_ts_to_utc(self, server_ts: int, offset_hours: int) -> datetime:
        """Konversi timestamp bar/tick server MT5 menjadi datetime UTC."""
        return datetime.fromtimestamp(int(server_ts) - offset_hours * 3600, tz=timezone.utc)

    def utc_to_server_ts(self, dt_utc: datetime, offset_hours: int) -> int:
        """Konversi datetime UTC menjadi timestamp waktu server MT5."""
        return int(dt_utc.astimezone(timezone.utc).timestamp()) + offset_hours * 3600

    def fetch_rates_df(
        self,
        symbol: str,
        timeframe: int,
        count: int,
        offset_hours: int,
    ) -> pd.DataFrame:
        """Ambil OHLCV + spread MT5 dan konversi indeks waktu menjadi UTC."""
        rates = self.mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None or len(rates) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(rates)
        df["time_server"] = df["time"].astype(int)
        df["time_utc"] = pd.to_datetime(
            df["time_server"] - offset_hours * 3600, unit="s", utc=True
        )
        return df

    def calculate_order_profit(
        self,
        order_type: int,
        symbol: str,
        volume: float,
        price_open: float,
        price_close: float,
    ) -> Optional[float]:
        """Hitung profit/rugi dalam mata uang akun via mt5.order_calc_profit()."""
        res = self.mt5.order_calc_profit(
            order_type, symbol, float(volume), float(price_open), float(price_close)
        )
        return float(res) if res is not None else None

    def get_open_positions(self, magic: Optional[int] = None) -> list[Any]:
        positions = self.mt5.positions_get()
        if not positions:
            return []
        if magic is None:
            return list(positions)
        return [p for p in positions if int(getattr(p, "magic", -1)) == int(magic)]

    def get_pending_orders(self, magic: Optional[int] = None) -> list[Any]:
        orders = self.mt5.orders_get()
        if not orders:
            return []
        if magic is None:
            return list(orders)
        return [o for o in orders if int(getattr(o, "magic", -1)) == int(magic)]

    def _find_existing_for_idempotency(
        self,
        symbol: str,
        magic: int,
        order_type: int,
        price: float,
        volume: float,
        comment: str,
        price_tolerance: float,
        now_utc: datetime,
        offset_hours: int,
    ) -> Optional[int]:
        """Cek posisi/order aktif atau riwayat baru sebelum retry open_order."""
        # 1. Cek posisi terbuka & pending order (primer: comment, cadangan: multi-field)
        for item in list(self.get_open_positions(magic)) + list(
            self.get_pending_orders(magic)
        ):
            item_comment = str(getattr(item, "comment", ""))
            if comment and item_comment == comment:
                return int(getattr(item, "ticket", 0))
            if self._matches_fallback(
                item, symbol, order_type, price, volume, price_tolerance, now_utc, offset_hours
            ):
                return int(getattr(item, "ticket", 0))

        # 2. Cek history_orders_get 10 menit terakhir jika terisi lalu cepat tertutup
        if hasattr(self.mt5, "history_orders_get"):
            start_srv = self.utc_to_server_ts(now_utc - timedelta(minutes=10), offset_hours)
            end_srv = self.utc_to_server_ts(now_utc + timedelta(minutes=2), offset_hours)
            hist = self.mt5.history_orders_get(start_srv, end_srv) or []
            for h in hist:
                if int(getattr(h, "magic", -1)) != int(magic):
                    continue
                if comment and str(getattr(h, "comment", "")) == comment:
                    return int(getattr(h, "ticket", 0))
                if self._matches_fallback(
                    h, symbol, order_type, price, volume, price_tolerance, now_utc, offset_hours
                ):
                    return int(getattr(h, "ticket", 0))
        return None

    def _matches_fallback(
        self,
        item: Any,
        symbol: str,
        order_type: int,
        price: float,
        volume: float,
        price_tolerance: float,
        now_utc: datetime,
        offset_hours: int,
    ) -> bool:
        item_symbol = str(getattr(item, "symbol", ""))
        item_type = int(getattr(item, "type", -1))
        item_vol = float(
            getattr(item, "volume_initial", getattr(item, "volume", getattr(item, "volume_current", 0.0)))
        )
        item_price = float(getattr(item, "price_open", 0.0))
        item_time_srv = int(
            getattr(item, "time_setup", getattr(item, "time", 0))
        )
        if item_symbol != symbol or item_type != order_type:
            return False
        if abs(item_vol - volume) > 1e-6:
            return False
        if abs(item_price - price) > price_tolerance:
            return False
        if item_time_srv > 0:
            item_utc = self.server_ts_to_utc(item_time_srv, offset_hours)
            if abs((now_utc - item_utc).total_seconds()) > 600:
                return False
        return True

    def _execute_with_retry(
        self,
        request: dict[str, Any],
        pre_retry_check: Optional[Callable[[], Optional[int]]] = None,
    ) -> OrderResult:
        """Jalankan mt5.order_send dengan auto-retry maksimal 5x @ 500ms."""
        last_retcode = -1
        last_msg = "Unknown error"
        max_attempts = max(1, self.cfg.max_order_retries)

        for attempt in range(1, max_attempts + 1):
            if attempt > 1 and pre_retry_check is not None:
                existing_ticket = pre_retry_check()
                if existing_ticket is not None:
                    return OrderResult(
                        success=True,
                        retcode=TRADE_RETCODE_DONE,
                        ticket=existing_ticket,
                        message="Order sudah tercatat di server saat pre-retry check.",
                        attempts=attempt - 1,
                        already_existed=True,
                    )

            result = self.mt5.order_send(request)
            if result is not None:
                retcode = int(getattr(result, "retcode", -1))
                last_retcode = retcode
                last_msg = str(getattr(result, "comment", ""))
                if retcode in (TRADE_RETCODE_DONE, TRADE_RETCODE_PLACED):
                    ticket = int(
                        getattr(result, "order", getattr(result, "deal", 0)) or 0
                    )
                    return OrderResult(
                        success=True,
                        retcode=retcode,
                        ticket=ticket,
                        message=last_msg or "OK",
                        attempts=attempt,
                    )
                if retcode not in RETRYABLE_RETCODES:
                    return OrderResult(
                        success=False,
                        retcode=retcode,
                        ticket=None,
                        message=last_msg or f"Non-retryable retcode {retcode}",
                        attempts=attempt,
                    )
            else:
                last_msg = f"order_send returned None: {self.mt5.last_error()}"

            if attempt < max_attempts:
                self._sleep(self.cfg.retry_delay_ms / 1000.0)

        return OrderResult(
            success=False,
            retcode=last_retcode,
            ticket=None,
            message=last_msg,
            attempts=max_attempts,
        )

    def open_order(
        self,
        symbol: str,
        order_type: int,
        volume: float,
        price: float,
        sl: float,
        tp: float,
        comment: str,
        offset_hours: int = 0,
        expiration_utc: Optional[datetime] = None,
        now_utc: Optional[datetime] = None,
    ) -> OrderResult:
        """Kirim pending limit / market order secara idempotent."""
        spec = self.get_symbol_spec(symbol)
        now_utc = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
        norm_comment = comment[:31]
        is_pending = order_type in (
            getattr(self.mt5, "ORDER_TYPE_BUY_LIMIT", 2),
            getattr(self.mt5, "ORDER_TYPE_SELL_LIMIT", 3),
        )
        action = (
            getattr(self.mt5, "TRADE_ACTION_PENDING", 5)
            if is_pending
            else getattr(self.mt5, "TRADE_ACTION_DEAL", 1)
        )
        request: dict[str, Any] = {
            "action": action,
            "symbol": symbol,
            "volume": round(float(volume), 2),
            "type": int(order_type),
            "price": round(float(price), spec.digits),
            "sl": round(float(sl), spec.digits),
            "tp": round(float(tp), spec.digits),
            "deviation": 30,
            "magic": self.cfg.magic_number,
            "comment": norm_comment,
            "type_time": getattr(self.mt5, "ORDER_TIME_SPECIFIED", 2)
            if expiration_utc is not None and is_pending
            else getattr(self.mt5, "ORDER_TIME_GTC", 0),
            "type_filling": getattr(self.mt5, "ORDER_FILLING_RETURN", 2),
        }
        if expiration_utc is not None and is_pending:
            request["expiration"] = self.utc_to_server_ts(expiration_utc, offset_hours)

        price_tol = max(spec.point * 50, 0.25)

        def _pre_check() -> Optional[int]:
            return self._find_existing_for_idempotency(
                symbol=symbol,
                magic=self.cfg.magic_number,
                order_type=order_type,
                price=request["price"],
                volume=request["volume"],
                comment=norm_comment,
                price_tolerance=price_tol,
                now_utc=now_utc,
                offset_hours=offset_hours,
            )

        return self._execute_with_retry(request, pre_retry_check=_pre_check)

    def modify_order(
        self, symbol: str, position_ticket: int, new_sl: float, new_tp: float
    ) -> OrderResult:
        """Modifikasi SL/TP posisi aktif dengan auto-retry."""
        spec = self.get_symbol_spec(symbol)
        request = {
            "action": getattr(self.mt5, "TRADE_ACTION_SLTP", 6),
            "symbol": symbol,
            "position": int(position_ticket),
            "sl": round(float(new_sl), spec.digits),
            "tp": round(float(new_tp), spec.digits),
            "magic": self.cfg.magic_number,
        }
        return self._execute_with_retry(request)

    def close_order(
        self,
        symbol: str,
        position_ticket: int,
        position_type: int,
        volume: float,
        comment: str = "SNP_CLOSE",
    ) -> OrderResult:
        """Tutup seluruh atau sebagian (partial) volume posisi aktif."""
        spec = self.get_symbol_spec(symbol)
        bid, ask, _ = self.get_current_bid_ask(symbol)
        buy_type = getattr(self.mt5, "POSITION_TYPE_BUY", 0)
        close_order_type = (
            getattr(self.mt5, "ORDER_TYPE_SELL", 1)
            if position_type == buy_type
            else getattr(self.mt5, "ORDER_TYPE_BUY", 0)
        )
        close_price = bid if position_type == buy_type else ask
        request = {
            "action": getattr(self.mt5, "TRADE_ACTION_DEAL", 1),
            "symbol": symbol,
            "position": int(position_ticket),
            "volume": round(float(volume), 2),
            "type": close_order_type,
            "price": round(float(close_price), spec.digits),
            "deviation": 30,
            "magic": self.cfg.magic_number,
            "comment": comment[:31],
            "type_filling": getattr(self.mt5, "ORDER_FILLING_IOC", 1),
        }
        return self._execute_with_retry(request)

    def cancel_pending_order(self, ticket: int) -> OrderResult:
        """Batalkan pending limit order di MT5."""
        request = {
            "action": getattr(self.mt5, "TRADE_ACTION_REMOVE", 8),
            "order": int(ticket),
            "magic": self.cfg.magic_number,
        }
        return self._execute_with_retry(request)
