from collections import deque
from datetime import datetime
from typing import Any, Deque, Optional

from pandas import DataFrame

from freqtrade.constants import Config, ListPairsWithTimeframes, PairWithTimeframe
from freqtrade.enums import CandleType, RunMode
from freqtrade.exchange import Exchange
from freqtrade.exchange.exchange_types import OrderBook
from freqtrade.exceptions import OperationalException
from freqtrade.rpc import RPCManager

NO_EXCHANGE_EXCEPTION: str
MAX_DATAFRAME_CANDLES: int

class DataProvider:
    producers: list[Any]
    external_data_enabled: bool

    def __init__(
        self,
        config: Config,
        exchange: Optional[Exchange],
        pairlists: Optional[Any] = ...,
        rpc: Optional[RPCManager] = ...,
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
    def _set_producer_pairs(self, pairlist: list[str], producer_name: str = ...) -> None: ...
    def get_producer_pairs(self, producer_name: str = ...) -> list[str]: ...
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
    ) -> tuple[bool, int]: ...
    def get_producer_df(
        self,
        pair: str,
        timeframe: Optional[str] = ...,
        candle_type: Optional[CandleType] = ...,
        producer_name: str = ...,
    ) -> tuple[DataFrame, datetime]: ...
    def add_pairlisthandler(self, pairlists: Any) -> None: ...
    def historic_ohlcv(self, pair: str, timeframe: str, candle_type: str = ...) -> DataFrame: ...
    def get_required_startup(self, timeframe: str) -> int: ...
    def get_pair_dataframe(
        self,
        pair: str,
        timeframe: Optional[str] = ...,
        candle_type: str = ...,
    ) -> DataFrame: ...
    def get_analyzed_dataframe(self, pair: str, timeframe: str) -> tuple[DataFrame, datetime]: ...
    @property
    def runmode(self) -> RunMode: ...
    def current_whitelist(self) -> list[str]: ...
    def clear_cache(self) -> None: ...
    def refresh(
        self,
        pairlist: ListPairsWithTimeframes,
        helping_pairs: Optional[ListPairsWithTimeframes] = ...,
    ) -> None: ...
    def refresh_latest_trades(self, pairlist: ListPairsWithTimeframes) -> None: ...
    @property
    def available_pairs(self) -> list[PairWithTimeframe]: ...
    def ohlcv(
        self,
        pair: str,
        timeframe: Optional[str] = ...,
        copy: bool = ...,
        candle_type: str = ...,
    ) -> DataFrame: ...
    def trades(
        self,
        pair: str,
        timeframe: Optional[str] = ...,
        copy: bool = ...,
        candle_type: str = ...,
    ) -> DataFrame: ...
    def market(self, pair: str) -> Optional[dict[str, Any]]: ...
    def ticker(self, pair: str) -> dict[str, Any]: ...
    def orderbook(self, pair: str, maximum: int) -> OrderBook: ...
    def send_msg(self, message: str, *, always_send: bool = ...) -> None: ...