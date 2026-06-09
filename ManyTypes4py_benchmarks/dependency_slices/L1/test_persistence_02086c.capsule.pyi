from typing import Any

# === Internal dependency: freqtrade.constants ===
DATETIME_PRINT_FORMAT = '%Y-%m-%d %H:%M:%S'
CUSTOM_TAG_MAX_LENGTH = 255

# === Internal dependency: freqtrade.enums.TradingMode ===
FUTURES: Any
MARGIN: Any
SPOT: Any

# === Internal dependency: freqtrade.exceptions ===
class DependencyException(FreqtradeException): ...

# === Internal dependency: freqtrade.exchange.exchange_utils ===
from ccxt import TICK_SIZE

# === Internal dependency: freqtrade.persistence ===
from freqtrade.persistence.models import init_db
from freqtrade.persistence.trade_model import LocalTrade
from freqtrade.persistence.trade_model import Order
from freqtrade.persistence.trade_model import Trade

# === Internal dependency: freqtrade.util ===
from freqtrade.util.datetime_helpers import dt_now

# === Third-party dependency: pytest ===
# Used symbols: approx, mark, raises

# === Third-party dependency: sqlalchemy ===
# Used symbols: select

# === Internal dependency: tests.conftest ===
def log_has(line, logs): ...
def log_has_re(line, logs): ...
def create_mock_trades(fee, is_short=..., use_db=...): ...
def create_mock_trades_with_leverage(fee, use_db=...): ...
def create_mock_trades_usdt(fee, is_short=..., use_db=...): ...