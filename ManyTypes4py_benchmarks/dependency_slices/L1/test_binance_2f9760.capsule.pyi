from typing import Any

# === Third-party dependency: ccxt ===
# Used symbols: InsufficientFunds, InvalidOrder

# === Internal dependency: freqtrade.enums.CandleType ===
SPOT: Any

# === Internal dependency: freqtrade.enums.MarginMode ===
ISOLATED: Any

# === Internal dependency: freqtrade.enums.TradingMode ===
FUTURES: Any
SPOT: Any

# === Internal dependency: freqtrade.exceptions ===
class OperationalException(FreqtradeException): ...
class DependencyException(FreqtradeException): ...
class InvalidOrderException(ExchangeError): ...

# === Internal dependency: freqtrade.exchange.exchange_utils_timeframe ===
def timeframe_to_seconds(timeframe): ...

# === Internal dependency: freqtrade.persistence ===
from freqtrade.persistence.trade_model import Trade

# === Internal dependency: freqtrade.util.datetime_helpers ===
def dt_utc(year, month, day, hour=..., minute=..., second=..., microsecond=...): ...
def dt_ts(dt=...): ...
def dt_from_ts(timestamp): ...

# === Third-party dependency: pandas ===
# Used symbols: DataFrame, date_range

# === Third-party dependency: pytest ===
# Used symbols: approx, mark, raises

# === Internal dependency: tests.conftest ===
def get_patched_exchange(mocker, config, api_mock=..., exchange=..., mock_markets=..., mock_supported_modes=...): ...
EXMS = 'freqtrade.exchange.exchange.Exchange'

# === Internal dependency: tests.exchange.test_exchange ===
def ccxt_exceptionhandlers(mocker, default_conf, api_mock, exchange_name, fun, mock_ccxt_fun, retries=..., **kwargs): ...