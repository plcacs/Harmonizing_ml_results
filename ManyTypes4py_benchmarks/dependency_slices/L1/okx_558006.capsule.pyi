from typing import Any

# === Third-party dependency: ccxt ===
# Used symbols: BaseError, DDoSProtection, ExchangeError, InvalidOrder, OperationFailed, OrderNotFound

# === Internal dependency: freqtrade.enums.CandleType ===
FUTURES: Any
SPOT: Any

# === Internal dependency: freqtrade.enums.MarginMode ===
ISOLATED: Any

# === Internal dependency: freqtrade.enums.PriceType ===
INDEX: Any
LAST: Any
MARK: Any

# === Internal dependency: freqtrade.enums.TradingMode ===
FUTURES: Any
SPOT: Any

# === Internal dependency: freqtrade.exceptions ===
class OperationalException(FreqtradeException): ...
class RetryableOrderError(InvalidOrderException): ...
class TemporaryError(ExchangeError): ...
class DDosProtection(TemporaryError): ...

# === Internal dependency: freqtrade.exchange ===
from freqtrade.exchange.exchange_utils import date_minus_candles

# === Internal dependency: freqtrade.exchange.common ===
def retrier(_func): ...
def retrier(_func, *, retries=...): ...
def retrier(*, retries=...): ...
def retrier(_func=..., *, retries=...): ...
API_RETRY_COUNT = 4

# === Internal dependency: freqtrade.exchange.exchange_types ===
class FtHas(TypedDict): ...
CcxtOrder = dict[str, Any]

# === Internal dependency: freqtrade.misc ===
def safe_value_fallback2(dict1, dict2, key1, key2, default_value=...): ...

# === Internal dependency: freqtrade.util ===
from freqtrade.util.datetime_helpers import dt_now
from freqtrade.util.datetime_helpers import dt_ts