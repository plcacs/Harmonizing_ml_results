from typing import Any

# === Internal dependency: freqtrade.constants ===
UNLIMITED_STAKE_AMOUNT = 'unlimited'
DEFAULT_TRADES_COLUMNS = ['timestamp', 'id', 'type', 'side', 'price', 'amount', 'cost']

# === Internal dependency: freqtrade.data.converter ===
from freqtrade.data.converter.converter import ohlcv_to_dataframe
from freqtrade.data.converter.trade_converter import trades_list_to_df

# === Internal dependency: freqtrade.edge ===
from .edge_positioning import PairInfo

# === Internal dependency: freqtrade.enums ===
from freqtrade.enums.signaltype import SignalDirection

# === Internal dependency: freqtrade.enums.CandleType ===
SPOT: Any

# === Internal dependency: freqtrade.enums.MarginMode ===
CROSS: Any
ISOLATED: Any

# === Internal dependency: freqtrade.enums.RunMode ===
DRY_RUN: Any

# === Internal dependency: freqtrade.enums.TradingMode ===
FUTURES: Any
MARGIN: Any

# === Internal dependency: freqtrade.exchange ===
from freqtrade.exchange.exchange_utils_timeframe import timeframe_to_minutes
from freqtrade.exchange.exchange_utils_timeframe import timeframe_to_seconds

# === Internal dependency: freqtrade.freqtradebot ===
class FreqtradeBot(LoggingMixin): ...

# === Internal dependency: freqtrade.persistence ===
from freqtrade.persistence.models import init_db
from freqtrade.persistence.trade_model import LocalTrade
from freqtrade.persistence.trade_model import Order
from freqtrade.persistence.trade_model import Trade

# === Internal dependency: freqtrade.resolvers ===
from freqtrade.resolvers.exchange_resolver import ExchangeResolver

# === Internal dependency: freqtrade.util ===
from freqtrade.util.datetime_helpers import dt_now
from freqtrade.util.datetime_helpers import dt_ts

# === Internal dependency: freqtrade.worker ===
class Worker: ...

# === Third-party dependency: numpy ===
# Used symbols: concatenate, cumsum, int64, random, seterr

# === Third-party dependency: pandas ===
# Used symbols: DataFrame, date_range, to_datetime

# === Third-party dependency: pytest ===
# Used symbols: fixture

# === Internal dependency: tests.conftest_trades ===
def mock_trade_1(fee, is_short): ...
def mock_trade_2(fee, is_short): ...
def mock_trade_3(fee, is_short): ...
def mock_trade_4(fee, is_short): ...
def mock_trade_5(fee, is_short): ...
def mock_trade_6(fee, is_short): ...
def short_trade(fee): ...
def leverage_trade(fee): ...

# === Internal dependency: tests.conftest_trades_usdt ===
def mock_trade_usdt_1(fee, is_short): ...
def mock_trade_usdt_2(fee, is_short): ...
def mock_trade_usdt_3(fee, is_short): ...
def mock_trade_usdt_4(fee, is_short): ...
def mock_trade_usdt_5(fee, is_short): ...
def mock_trade_usdt_6(fee, is_short): ...
def mock_trade_usdt_7(fee, is_short): ...

# === Third-party dependency: xdist.scheduler.loadscope ===
class LoadScopeScheduling:
    def nodes(self) -> list[WorkerController]: ...
    def collection_is_completed(self) -> bool: ...
    def tests_finished(self) -> bool: ...
    def has_pending(self) -> bool: ...