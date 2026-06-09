from typing import Any

# === Internal dependency: freqtrade ===
__version__: str

# === Internal dependency: freqtrade.constants ===
CANCEL_REASON: Any

# === Internal dependency: freqtrade.edge ===
# re-export: from .edge_positioning import PairInfo

# === Internal dependency: freqtrade.enums ===
# re-export: from freqtrade.enums.marketstatetype import MarketDirection
# re-export: from freqtrade.enums.signaltype import SignalDirection

# === Internal dependency: freqtrade.enums.ExitType ===
FORCE_EXIT: Any
STOP_LOSS: Any

# === Internal dependency: freqtrade.enums.RPCMessageType ===
ENTRY: Any
ENTRY_CANCEL: Any
ENTRY_FILL: Any
EXIT: Any
EXIT_CANCEL: Any
EXIT_FILL: Any
PROTECTION_TRIGGER: Any
PROTECTION_TRIGGER_GLOBAL: Any
STARTUP: Any
STATUS: Any
STRATEGY_MSG: Any
WARNING: Any

# === Internal dependency: freqtrade.enums.RunMode ===
DRY_RUN: Any

# === Internal dependency: freqtrade.enums.State ===
RELOAD_CONFIG: Any
RUNNING: Any
STOPPED: Any

# === Internal dependency: freqtrade.exceptions ===
class OperationalException(FreqtradeException): ...

# === Internal dependency: freqtrade.freqtradebot ===
class FreqtradeBot(LoggingMixin):
    def __init__(self, config: Config) -> None: ...
    def enter_positions(self) -> int: ...

# === Internal dependency: freqtrade.loggers ===
def setup_logging(config: Config) -> None: ...

# === Internal dependency: freqtrade.persistence ===
# re-export: from freqtrade.persistence.pairlock_middleware import PairLocks
# re-export: from freqtrade.persistence.trade_model import Trade

# === Internal dependency: freqtrade.persistence.models ===
# re-export: from freqtrade.persistence.trade_model import Order

# === Internal dependency: freqtrade.rpc.rpc ===
class RPCException(Exception): ...

# === Internal dependency: freqtrade.rpc.telegram ===
def authorized_only(command_handler: Callable[..., Coroutine[Any, Any, None]]) -> Any: ...
class Telegram(RPCHandler):
    def __init__(self, rpc: RPC, config: Config) -> None: ...
    async def _force_exit(self, update: Update, context: CallbackContext) -> None: ...

# === Internal dependency: freqtrade.util.datetime_helpers ===
def dt_now() -> datetime: ...

# === Third-party dependency: pandas ===
# Used symbols: DataFrame

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises

# === Third-party dependency: sqlalchemy ===
# Used symbols: select

# === Third-party dependency: telegram ===
# Used symbols: Chat, Message, ReplyKeyboardMarkup, Update

# === Third-party dependency: telegram.error ===
class TelegramError(Exception): ...
class NetworkError(TelegramError): ...
class BadRequest(NetworkError): ...

# === Internal dependency: tests.conftest ===
def log_has(line, logs) -> Any: ...
def log_has_re(line, logs) -> Any: ...
def patch_exchange(mocker, api_mock = ..., exchange = ..., mock_markets = ..., mock_supported_modes = ...) -> None: ...
def patch_whitelist(mocker, conf) -> None: ...
def get_patched_freqtradebot(mocker, config) -> FreqtradeBot: ...
def patch_get_signal(freqtrade: FreqtradeBot, enter_long = ..., exit_long = ..., enter_short = ..., exit_short = ..., enter_tag: str | None = ..., exit_tag: str | None = ...) -> None: ...
def create_mock_trades(fee, is_short: bool | None = ..., use_db: bool = ...) -> Any: ...
def create_mock_trades_usdt(fee, is_short: bool | None = ..., use_db: bool = ...) -> Any: ...
CURRENT_TEST_STRATEGY: str
EXMS: str

# === Third-party dependency: time_machine ===
class travel: ...