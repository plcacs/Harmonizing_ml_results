```pyi
from collections import deque
from datetime import datetime, timezone
from typing import Any, Optional, Tuple, Dict, List

from pandas import DataFrame, Timedelta, Timestamp
from freqtrade.configuration import TimeRange
from freqtrade.constants import Config, ListPairsWithTimeframes, PairWithTimeframe
from freqtrade.enums import CandleType, RunMode
from freqtrade.exchange import Exchange
from freqtrade.exchange.exchange_types import OrderBook
from freqtrade.rpc import RPCManager
from freqtrade.util import PeriodicCache

NO_EXCHANGE_EXCEPTION: str
MAX_DATAFRAME_CANDLES: int

class DataProvider:
    _config: Config
    _exchange: Optional[Exchange]
    _pairlists: Any
    _DataProvider__rpc: Optional[RPCManager]
    _DataProvider__cached_pairs: Dict[Tuple[str, str, CandleType], Tuple[DataFrame, datetime]]
    _DataProvider__slice_index: Optional[int]
    _DataProvider__slice_date: Optional[datetime]
    _DataProvider__cached_pairs_backtesting: Dict[Tuple[str, str, CandleType], DataFrame]
    _DataProvider__producer_pairs_df: Dict[str, Dict[Tuple[str, str, CandleType], Tuple[DataFrame, datetime]]]
    _DataProvider__producer_pairs: Dict[str, List[str]]
    _msg_queue: deque[str]
    _default_candle_type: CandleType
    _default_timeframe: str
    _DataProvider__msg_cache: PeriodicCache
    producers: List[Any]
    external_data_enabled: bool

    def __init__(
        self,
        config: Config,
        exchange: Optional[Exchange] = None,
        pairlists: Any = None,
        rpc: Optional[RPCManager] = None,
    ) -> None: ...
    def _set_dataframe_max_index(self, limit_index: int) -> None: ...
    def _set_dataframe_max_date(self, limit_date: datetime) -> None: ...
    def _set_cached_df(
        self, pair: str, timeframe: str, dataframe: DataFrame, candle_type: CandleType
    ) -> None: ...
    def _set_producer_pairs(self, pairlist: List[str], producer_name: str = ...) -> None: ...
    def get_producer_pairs(self, producer_name: str = ...) -> List[str]: ...
    def _emit_df(self, pair_key: PairWithTimeframe, dataframe: DataFrame, new_candle: bool) -> None: ...
    def _replace_external_df(
        self,
        pair: str,
        dataframe: DataFrame,
        last_analyzed: Optional[datetime],
        timeframe: str,
        candle_type: CandleType,
        producer_name: str = ...,
    ) -> None: ...
    def _add_external_df(
        self,
        pair: str,
        dataframe: DataFrame,
        last_analyzed: Optional[datetime],
        timeframe: str,
        candle_type: CandleType,
        producer_name: str = ...,
    ) -> Tuple[bool, int]: ...
    def get_producer_df(
        self,
        pair: str,
        timeframe: Optional[str] = None,
        candle_type: Optional[CandleType] = None,
        producer_name: str = ...,
    ) -> Tuple[DataFrame, datetime]: ...
    def add_pairlisthandler(self, pairlists: Any) -> None: ...
    def historic_ohlcv(self, pair: str, timeframe: str, candle_type: str = ...) -> DataFrame: ...
    def get_required_startup(self, timeframe: str) -> int: ...
    def get_pair_dataframe(
        self, pair: str, timeframe: Optional[str] = None, candle_type: str = ...
    ) -> DataFrame: ...
    def get_analyzed_dataframe(self, pair: str, timeframe: str) -> Tuple[DataFrame, datetime]: ...
    @property
    def runmode(self) -> RunMode: ...
    def current_whitelist(self) -> List[str]: ...
    def clear_cache(self) -> None: ...
    def refresh(self, pairlist: List[str], helping_pairs: Optional[List[str]] = None) -> None: ...
    def refresh_latest_trades(self, pairlist: List[str]) -> None: ...
    @property
    def available_pairs(self) -> List[PairWithTimeframe]: ...
    def ohlcv(
        self, pair: str, timeframe: Optional[str] = None, copy: bool = ..., candle_type: str = ...
    ) -> DataFrame: ...
    def trades(
        self, pair: str, timeframe: Optional[str] = None, copy: bool = ..., candle_type: str = ...
    ) -> DataFrame: ...
    def market(self, pair: str) -> Optional[Dict[str, Any]]: ...
    def ticker(self, pair: str) -> Dict[str, Any]: ...
    def orderbook(self, pair: str, maximum: int) -> OrderBook: ...
    def send_msg(self, message: str, *, always_send: bool = ...) -> None: ...
```