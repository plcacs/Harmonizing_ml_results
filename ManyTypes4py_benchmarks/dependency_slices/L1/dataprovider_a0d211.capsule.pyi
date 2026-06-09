from typing import Any

# === Internal dependency: freqtrade.configuration.TimeRange ===
parse_timerange: Any

# === Internal dependency: freqtrade.constants ===
FULL_DATAFRAME_THRESHOLD = 100
PairWithTimeframe = tuple[str, str, CandleType]
ListPairsWithTimeframes = list[PairWithTimeframe]

# === Internal dependency: freqtrade.data.history ===
from .datahandlers import get_datahandler
from .history_utils import load_pair_history

# === Internal dependency: freqtrade.enums.CandleType ===
SPOT: Any
from_string: Any

# === Internal dependency: freqtrade.enums.RPCMessageType ===
ANALYZED_DF: Any
NEW_CANDLE: Any

# === Internal dependency: freqtrade.enums.RunMode ===
DRY_RUN: Any
LIVE: Any
OTHER: Any

# === Internal dependency: freqtrade.enums.TradingMode ===
SPOT: Any

# === Internal dependency: freqtrade.exceptions ===
class OperationalException(FreqtradeException): ...
class ExchangeError(DependencyException): ...

# === Internal dependency: freqtrade.exchange ===
from freqtrade.exchange.exchange_utils_timeframe import timeframe_to_prev_date
from freqtrade.exchange.exchange_utils_timeframe import timeframe_to_seconds

# === Internal dependency: freqtrade.exchange.exchange_types ===
class OrderBook(TypedDict): ...

# === Internal dependency: freqtrade.misc ===
def append_candles_to_dataframe(left, right): ...

# === Internal dependency: freqtrade.rpc.rpc_types ===
class RPCAnalyzedDFMsg(RPCSendMsgBase): ...

# === Internal dependency: freqtrade.util ===
from freqtrade.util.periodic_cache import PeriodicCache

# === Third-party dependency: pandas ===
# Used symbols: DataFrame, Timedelta, Timestamp, to_timedelta