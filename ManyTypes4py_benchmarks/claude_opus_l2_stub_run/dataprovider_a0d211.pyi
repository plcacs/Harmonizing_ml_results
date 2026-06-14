import logging
from collections import deque
from datetime import datetime
from typing import Any

from pandas import DataFrame

from freqtrade.configuration import TimeRange
from freqtrade.constants import Config, ListPairsWithTimeframes, PairWithTimeframe
from freqtrade.enums import CandleType, RunMode
from freqtrade.exchange import Exchange
from freqtrade.exchange.exchange_types import OrderBook
from freqtrade.rpc import RPCManager
from freqtrade.util import PeriodicCache

logger: logging.Logger
NO_EXCHANGE_EXCEPTION: str
MAX_DATAFRAME_CANDLES: int

class DataProvider:
    _config: Config
    _exchange: Exchange | None
    _pairlists: Any
    __rpc: RPCManager | None
    __cached_pairs: dict[PairWithTimeframe, tuple[DataFrame, datetime]]
    __slice_index: int | None
    __slice_date: datetime | None
    __cached_pairs_backtesting: dict[tuple[str, str, CandleType], DataFrame]
    __producer_pairs_df: dict[str, dict[PairWithTimeframe, tuple[DataFrame, datetime]]]
    __producer_pairs: dict[str, list[str]]
    _msg_queue: deque[str]
    _default_candle_type: CandleType
    _default_timeframe: str
    __msg_cache: PeriodicCache
    producers: list[dict[str, Any]]
    external_data_enabled: bool

    def __init__(
        self,
        config: Config,
        exchange: Exchange | None,
        pairlists: Any = ...,
        rpc: RPCManager | None = ...,
    ) -> None: ...
    def _set_dataframe_max_index(self, limit_index: int) -> None: ...
    def _set_dataframe_max_date(self, limit_date: datetime) -> None: ...
    def _set_cached_df(
        self,
        pair: str,
        timeframe: str,
        dataframe: DataFrame,
        candle_type: CandleType,
    ) -> None: ...
    def _set_producer_pairs(
        self, pairlist: list[str], producer_name: str = "default"
    ) -> None: ...
    def get_producer_pairs(self, producer_name: str = "default") -> list[str]: ...
    def _emit_df(
        self,
        pair_key: PairWithTimeframe,
        dataframe: DataFrame,
        new_candle: bool,
    ) -> None: ...
    def _replace_external_df(
        self,
        pair: str,
        dataframe: DataFrame,
        last_analyzed: datetime | None,
        timeframe: str,
        candle_type: CandleType,
        producer_name: str = "default",
    ) -> None: ...
    def _add_external_df(
        self,
        pair: str,
        dataframe: DataFrame,
        last_analyzed: datetime | None,
        timeframe: str,
        candle_type: CandleType,
        producer_name: str = "default",
    ) -> tuple[bool, int]: ...
    def get_producer_df(
        self,
        pair: str,
        timeframe: str | None = None,
        candle_type: CandleType | None = None,
        producer_name: str = "default",
    ) -> tuple[DataFrame, datetime]: ...
    def add_pairlisthandler(self, pairlists: Any) -> None: ...
    def historic_ohlcv(
        self, pair: str, timeframe: str | None, candle_type: str = ""
    ) -> DataFrame: ...
    def get_required_startup(self, timeframe: str) -> int: ...
    def get_pair_dataframe(
        self, pair: str, timeframe: str | None = None, candle_type: str = ""
    ) -> DataFrame: ...
    def get_analyzed_dataframe(
        self, pair: str, timeframe: str
    ) -> tuple[DataFrame, datetime]: ...
    @property
    def runmode(self) -> RunMode: ...
    def current_whitelist(self) -> list[str]: ...
    def clear_cache(self) -> None: ...
    def refresh(
        self,
        pairlist: ListPairsWithTimeframes,
        helping_pairs: ListPairsWithTimeframes | None = None,
    ) -> None: ...
    def refresh_latest_trades(self, pairlist: ListPairsWithTimeframes) -> None: ...
    @property
    def available_pairs(self) -> list[PairWithTimeframe]: ...
    def ohlcv(
        self,
        pair: str,
        timeframe: str | None = None,
        copy: bool = True,
        candle_type: str = "",
    ) -> DataFrame: ...
    def trades(
        self,
        pair: str,
        timeframe: str | None = None,
        copy: bool = True,
        candle_type: str = "",
    ) -> DataFrame: ...
    def market(self, pair: str) -> dict[str, Any] | None: ...
    def ticker(self, pair: str) -> dict[str, Any]: ...
    def orderbook(self, pair: str, maximum: int) -> OrderBook: ...
    def send_msg(self, message: str, *, always_send: bool = False) -> None: ...