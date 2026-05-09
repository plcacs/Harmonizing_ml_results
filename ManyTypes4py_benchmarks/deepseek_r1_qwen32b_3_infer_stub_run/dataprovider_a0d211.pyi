"""
Stub file for dataprovider_a0d211 module
"""

from datetime import datetime
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    TYPE_CHECKING,
)
from collections import deque
from pandas import DataFrame
from freqtrade.enums import CandleType, RunMode, TradingMode
from freqtrade.exchange import Exchange, OrderBook
from freqtrade.rpc.rpc_types import RPCAnalyzedDFMsg
from freqtrade.pairlist.pairlistmanager import PairListManager
from freqtrade.rpc.rpcmanager import RPCManager

if TYPE_CHECKING:
    from freqtrade.exchange import Exchange

logger: logging.Logger = ...

class DataProvider:
    """
    Data provider class stub
    """

    def __init__(self, config: dict, exchange: Exchange, pairlists: Optional[PairListManager] = None, rpc: Optional[RPCManager] = None) -> None:
        ...

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

    def _emit_df(self, pair_key: Tuple[str, str, CandleType], dataframe: DataFrame, new_candle: bool) -> None:
        ...

    def _replace_external_df(self, pair: str, dataframe: DataFrame, last_analyzed: Optional[datetime], timeframe: str, candle_type: CandleType, producer_name: str = 'default') -> None:
        ...

    def _add_external_df(self, pair: str, dataframe: DataFrame, last_analyzed: Optional[datetime], timeframe: str, candle_type: CandleType, producer_name: str = 'default') -> Tuple[bool, int]:
        ...

    def get_producer_df(self, pair: str, timeframe: Optional[str] = None, candle_type: Optional[CandleType] = None, producer_name: str = 'default') -> Tuple[DataFrame, datetime]:
        ...

    def add_pairlisthandler(self, pairlists: PairListManager) -> None:
        ...

    def historic_ohlcv(self, pair: str, timeframe: str, candle_type: str = '') -> DataFrame:
        ...

    def get_required_startup(self, timeframe: str) -> int:
        ...

    def get_pair_dataframe(self, pair: str, timeframe: Optional[str] = None, candle_type: str = '') -> DataFrame:
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

    def refresh(self, pairlist: List[str], helping_pairs: Optional[List[str]] = None) -> None:
        ...

    def refresh_latest_trades(self, pairlist: List[str]) -> None:
        ...

    @property
    def available_pairs(self) -> List[Tuple[str, str]]:
        ...

    def ohlcv(self, pair: str, timeframe: Optional[str] = None, copy: bool = True, candle_type: str = '') -> DataFrame:
        ...

    def trades(self, pair: str, timeframe: Optional[str] = None, copy: bool = True, candle_type: str = '') -> DataFrame:
        ...

    def market(self, pair: str) -> Optional[Dict[str, Any]]:
        ...

    def ticker(self, pair: str) -> Dict[str, Any]:
        ...

    def orderbook(self, pair: str, maximum: int) -> OrderBook:
        ...

    def send_msg(self, message: str, *, always_send: bool = False) -> None:
        ...