import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any, List, Tuple
from pandas import DataFrame, Timedelta, Timestamp, to_timedelta
from freqtrade.configuration import TimeRange
from freqtrade.constants import FULL_DATAFRAME_THRESHOLD, Config, ListPairsWithTimeframes, PairWithTimeframe
from freqtrade.data.history import get_datahandler, load_pair_history
from freqtrade.enums import CandleType, RPCMessageType, RunMode, TradingMode
from freqtrade.exceptions import ExchangeError, OperationalException
from freqtrade.exchange import Exchange, timeframe_to_prev_date, timeframe_to_seconds
from freqtrade.exchange.exchange_types import OrderBook
from freqtrade.misc import append_candles_to_dataframe
from freqtrade.rpc import RPCManager
from freqtrade.rpc.rpc_types import RPCAnalyzedDFMsg
from freqtrade.util import PeriodicCache

logger: logging.Logger = logging.getLogger(__name__)
NO_EXCHANGE_EXCEPTION: str = 'Exchange is not available to DataProvider.'
MAX_DATAFRAME_CANDLES: int = 1000

class DataProvider:

    def __init__(self, config: Config, exchange: Exchange, pairlists: ListPairsWithTimeframes = None, rpc: RPCManager = None) -> None:
        self._config: Config = config
        self._exchange: Exchange = exchange
        self._pairlists: ListPairsWithTimeframes = pairlists
        self.__rpc: RPCManager = rpc
        self.__cached_pairs: dict = {}
        self.__slice_index: int = None
        self.__slice_date: datetime = None
        self.__cached_pairs_backtesting: dict = {}
        self.__producer_pairs_df: dict = {}
        self.__producer_pairs: dict = {}
        self._msg_queue: deque = deque()
        self._default_candle_type: CandleType = self._config.get('candle_type_def', CandleType.SPOT)
        self._default_timeframe: str = self._config.get('timeframe', '1h')
        self.__msg_cache: PeriodicCache = PeriodicCache(maxsize=1000, ttl=timeframe_to_seconds(self._default_timeframe))
        self.producers: List[str] = self._config.get('external_message_consumer', {}).get('producers', [])
        self.external_data_enabled: bool = len(self.producers) > 0

    def _set_dataframe_max_index(self, limit_index: int) -> None:
        ...

    def _set_dataframe_max_date(self, limit_date: datetime) -> None:
        ...

    def _set_cached_df(self, pair: str, timeframe: str, dataframe: DataFrame, candle_type: CandleType) -> None:
        ...

    def _set_producer_pairs(self, pairlist: List[str], producer_name: str = 'default') -> None:
        ...

    def get_producer_pairs(self, producer_name: str = 'default') -> List[str]:
        ...

    def _emit_df(self, pair_key: PairWithTimeframe, dataframe: DataFrame, new_candle: bool) -> None:
        ...

    def _replace_external_df(self, pair: str, dataframe: DataFrame, last_analyzed: datetime, timeframe: str, candle_type: CandleType, producer_name: str = 'default') -> None:
        ...

    def _add_external_df(self, pair: str, dataframe: DataFrame, last_analyzed: datetime, timeframe: str, candle_type: CandleType, producer_name: str = 'default') -> Tuple[bool, int]:
        ...

    def get_producer_df(self, pair: str, timeframe: str = None, candle_type: CandleType = None, producer_name: str = 'default') -> Tuple[DataFrame, datetime]:
        ...

    def add_pairlisthandler(self, pairlists: ListPairsWithTimeframes) -> None:
        ...

    def historic_ohlcv(self, pair: str, timeframe: str, candle_type: str = '') -> DataFrame:
        ...

    def get_required_startup(self, timeframe: str) -> int:
        ...

    def get_pair_dataframe(self, pair: str, timeframe: str = None, candle_type: str = '') -> DataFrame:
        ...

    def get_analyzed_dataframe(self, pair: str, timeframe: str) -> Tuple[DataFrame, datetime]:
        ...

    @property
    def runmode(self) -> RunMode:
        ...

    def current_whitelist(self) -> List[str]:
        ...

    def clear_cache(self) -> None:
        ...

    def refresh(self, pairlist: List[str], helping_pairs: List[str] = None) -> None:
        ...

    def refresh_latest_trades(self, pairlist: List[str]) -> None:
        ...

    @property
    def available_pairs(self) -> List[Tuple[str, str]]:
        ...

    def ohlcv(self, pair: str, timeframe: str = None, copy: bool = True, candle_type: str = '') -> DataFrame:
        ...

    def trades(self, pair: str, timeframe: str = None, copy: bool = True, candle_type: str = '') -> DataFrame:
        ...

    def market(self, pair: str) -> dict:
        ...

    def ticker(self, pair: str) -> dict:
        ...

    def orderbook(self, pair: str, maximum: int) -> dict:
        ...

    def send_msg(self, message: str, always_send: bool = False) -> None:
        ...
