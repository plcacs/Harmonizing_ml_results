from typing import Any

# === Internal dependency: freqtrade ===
__version__ = '2025.2-dev'

# === Internal dependency: freqtrade.constants ===
CANCEL_REASON = {'TIMEOUT': 'cancelled due to timeout', 'PARTIALLY_FILLED_KEEP_OPEN': 'partially filled - keeping order open', 'PARTIALLY_FILLED': 'partially filled', 'FULLY_CANCELLED': 'fully cancelled', 'ALL_CANCELLED': 'cancelled (all unfilled and partially filled open orders cancelled)', 'CANCELLED_ON_EXCHANGE': 'cancelled on exchange', 'FORCE_EXIT': 'forcesold', 'REPLACE': 'cancelled to be replaced by new limit order', ...}

# === Internal dependency: freqtrade.edge ===
from .edge_positioning import PairInfo

# === Internal dependency: freqtrade.enums ===
from freqtrade.enums.marketstatetype import MarketDirection
from freqtrade.enums.signaltype import SignalDirection

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
    def __init__(self, config): ...
    def enter_positions(self): ...

# === Internal dependency: freqtrade.loggers ===
def setup_logging(config): ...

# === Internal dependency: freqtrade.persistence ===
from freqtrade.persistence.pairlock_middleware import PairLocks
from freqtrade.persistence.trade_model import Trade

# === Internal dependency: freqtrade.persistence.models ===
from freqtrade.persistence.trade_model import Order

# === Internal dependency: freqtrade.rpc.rpc ===
class RPCException(Exception): ...

# === Internal dependency: freqtrade.rpc.telegram ===
def authorized_only(command_handler): ...
class Telegram(RPCHandler):
    def __init__(self, rpc, config): ...
    async def _force_exit(self, update, context): ...

# === Internal dependency: freqtrade.util.datetime_helpers ===
def dt_now(): ...

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
def log_has(line, logs): ...
def log_has_re(line, logs): ...
def patch_exchange(mocker, api_mock=..., exchange=..., mock_markets=..., mock_supported_modes=...): ...
def patch_whitelist(mocker, conf): ...
def get_patched_freqtradebot(mocker, config): ...
def patch_get_signal(freqtrade, enter_long=..., exit_long=..., enter_short=..., exit_short=..., enter_tag=..., exit_tag=...): ...
def create_mock_trades(fee, is_short=..., use_db=...): ...
def create_mock_trades_usdt(fee, is_short=..., use_db=...): ...
CURRENT_TEST_STRATEGY = 'StrategyTestV3'
EXMS = 'freqtrade.exchange.exchange.Exchange'

# === Third-party dependency: time_machine ===
class travel: ...