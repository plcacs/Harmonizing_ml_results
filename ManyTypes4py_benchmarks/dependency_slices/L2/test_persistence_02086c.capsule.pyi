from typing import Any

# === Internal dependency: freqtrade.constants ===
DATETIME_PRINT_FORMAT: str
CUSTOM_TAG_MAX_LENGTH: int

# === Internal dependency: freqtrade.enums.TradingMode ===
FUTURES: Any
MARGIN: Any
SPOT: Any

# === Internal dependency: freqtrade.exceptions ===
class DependencyException(FreqtradeException): ...

# === Internal dependency: freqtrade.exchange.exchange_utils ===
# re-export: from ccxt import TICK_SIZE

# === Internal dependency: freqtrade.persistence ===
# re-export: from freqtrade.persistence.models import init_db
# re-export: from freqtrade.persistence.trade_model import LocalTrade
# re-export: from freqtrade.persistence.trade_model import Order
# re-export: from freqtrade.persistence.trade_model import Trade

# === Internal dependency: freqtrade.util ===
# re-export: from freqtrade.util.datetime_helpers import dt_now

# === Third-party dependency: pytest ===
# Used symbols: approx, mark, raises

# === Third-party dependency: sqlalchemy ===
# Used symbols: select

# === Internal dependency: tests.conftest ===
def log_has(line, logs) -> Any: ...
def log_has_re(line, logs) -> Any: ...
def create_mock_trades(fee, is_short: bool | None = ..., use_db: bool = ...) -> Any: ...
def create_mock_trades_with_leverage(fee, use_db: bool = ...) -> Any: ...
def create_mock_trades_usdt(fee, is_short: bool | None = ..., use_db: bool = ...) -> Any: ...