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
# re-export: from freqtrade.exchange.exchange_utils import date_minus_candles

# === Internal dependency: freqtrade.exchange.common ===
def retrier(_func: F) -> F: ...
def retrier(_func: F, *, retries = ...) -> F: ...
def retrier(*, retries = ...) -> Callable[[F], F]: ...
def retrier(_func: F | None = ..., *, retries = ...) -> Any: ...
API_RETRY_COUNT: int

# === Internal dependency: freqtrade.exchange.exchange_types ===
class FtHas(TypedDict): ...
CcxtOrder: Any

# === Internal dependency: freqtrade.misc ===
def safe_value_fallback2(dict1: DictMap, dict2: DictMap, key1: str, key2: str, default_value = ...) -> Any: ...

# === Internal dependency: freqtrade.util ===
# re-export: from freqtrade.util.datetime_helpers import dt_now
# re-export: from freqtrade.util.datetime_helpers import dt_ts