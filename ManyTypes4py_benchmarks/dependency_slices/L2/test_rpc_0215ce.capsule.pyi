from typing import Any

# === Internal dependency: freqtrade.edge ===
# re-export: from .edge_positioning import PairInfo

# === Internal dependency: freqtrade.enums ===
# re-export: from freqtrade.enums.signaltype import SignalDirection

# === Internal dependency: freqtrade.enums.State ===
RUNNING: Any
STOPPED: Any

# === Internal dependency: freqtrade.enums.TradingMode ===
SPOT: Any

# === Internal dependency: freqtrade.exceptions ===
class ExchangeError(DependencyException): ...
class InvalidOrderException(ExchangeError): ...
class TemporaryError(ExchangeError): ...

# === Internal dependency: freqtrade.persistence ===
# re-export: from freqtrade.persistence.trade_model import Order
# re-export: from freqtrade.persistence.trade_model import Trade

# === Internal dependency: freqtrade.persistence.key_value_store ===
def set_startup_time() -> Any: ...

# === Internal dependency: freqtrade.rpc ===
# re-export: from .rpc import RPCException

# === Internal dependency: freqtrade.rpc.fiat_convert ===
class CryptoToFiatConverter(LoggingMixin):
    def __init__(self, config: Config) -> None: ...

# === Third-party dependency: numpy ===
# Used symbols: isnan

# === Third-party dependency: pytest ===
# Used symbols: approx, mark, raises

# === Third-party dependency: sqlalchemy ===
# Used symbols: select

# === Internal dependency: tests.conftest ===
def get_patched_freqtradebot(mocker, config) -> FreqtradeBot: ...
def patch_get_signal(freqtrade: FreqtradeBot, enter_long = ..., exit_long = ..., enter_short = ..., exit_short = ..., enter_tag: str | None = ..., exit_tag: str | None = ...) -> None: ...
def create_mock_trades(fee, is_short: bool | None = ..., use_db: bool = ...) -> Any: ...
def create_mock_trades_usdt(fee, is_short: bool | None = ..., use_db: bool = ...) -> Any: ...
EXMS: str