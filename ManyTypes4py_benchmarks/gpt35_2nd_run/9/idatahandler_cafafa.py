import logging
import re
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from pandas import DataFrame, to_datetime
from freqtrade import misc
from freqtrade.configuration import TimeRange
from freqtrade.constants import DEFAULT_TRADES_COLUMNS, ListPairsWithTimeframes
from freqtrade.data.converter import clean_ohlcv_dataframe, trades_convert_types, trades_df_remove_duplicates, trim_dataframe
from freqtrade.enums import CandleType, TradingMode
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import timeframe_to_seconds
logger = logging.getLogger(__name__)

class IDataHandler(ABC):
    _OHLCV_REGEX: str = '^([a-zA-Z_\\d-]+)\\-(\\d+[a-zA-Z]{1,2})\\-?([a-zA-Z_]*)?(?=\\.)'
    _TRADES_REGEX: str = '^([a-zA-Z_\\d-]+)\\-(trades)?(?=\\.)'

    def __init__(self, datadir: Path) -> None:
        self._datadir: Path = datadir

    @classmethod
    @abstractmethod
    def _get_file_extension(cls) -> str:
        ...

    @classmethod
    def ohlcv_get_available_data(cls, datadir: Path, trading_mode: TradingMode) -> ListPairsWithTimeframes:
        ...

    @classmethod
    def ohlcv_get_pairs(cls, datadir: Path, timeframe: str, candle_type: CandleType) -> List[str]:
        ...

    @abstractmethod
    def ohlcv_store(self, pair: str, timeframe: str, data: DataFrame, candle_type: CandleType) -> None:
        ...

    def ohlcv_data_min_max(self, pair: str, timeframe: str, candle_type: CandleType) -> Tuple[datetime, datetime, int]:
        ...

    @abstractmethod
    def _ohlcv_load(self, pair: str, timeframe: str, timerange: TimeRange, candle_type: CandleType) -> DataFrame:
        ...

    def ohlcv_purge(self, pair: str, timeframe: str, candle_type: CandleType) -> bool:
        ...

    @abstractmethod
    def ohlcv_append(self, pair: str, timeframe: str, data: DataFrame, candle_type: CandleType) -> None:
        ...

    @classmethod
    def trades_get_available_data(cls, datadir: Path, trading_mode: TradingMode) -> List[str]:
        ...

    def trades_data_min_max(self, pair: str, trading_mode: TradingMode) -> Tuple[datetime, datetime, int]:
        ...

    @classmethod
    def trades_get_pairs(cls, datadir: Path) -> List[str]:
        ...

    @abstractmethod
    def _trades_store(self, pair: str, data: DataFrame, trading_mode: TradingMode) -> None:
        ...

    @abstractmethod
    def trades_append(self, pair: str, data: DataFrame) -> None:
        ...

    @abstractmethod
    def _trades_load(self, pair: str, trading_mode: TradingMode, timerange: TimeRange = None) -> DataFrame:
        ...

    def trades_store(self, pair: str, data: DataFrame, trading_mode: TradingMode) -> None:
        ...

    def trades_purge(self, pair: str, trading_mode: TradingMode) -> bool:
        ...

    def trades_load(self, pair: str, trading_mode: TradingMode, timerange: TimeRange = None) -> DataFrame:
        ...

    @classmethod
    def create_dir_if_needed(cls, datadir: Path) -> None:
        ...

    @classmethod
    def _pair_data_filename(cls, datadir: Path, pair: str, timeframe: str, candle_type: CandleType, no_timeframe_modify: bool = False) -> Path:
        ...

    @classmethod
    def _pair_trades_filename(cls, datadir: Path, pair: str, trading_mode: TradingMode) -> Path:
        ...

    @staticmethod
    def timeframe_to_file(timeframe: str) -> str:
        ...

    @staticmethod
    def rebuild_timeframe_from_filename(timeframe: str) -> str:
        ...

    @staticmethod
    def rebuild_pair_from_filename(pair: str) -> str:
        ...

    def ohlcv_load(self, pair: str, timeframe: str, candle_type: CandleType, *, timerange: TimeRange = None, fill_missing: bool = True, drop_incomplete: bool = False, startup_candles: int = 0, warn_no_data: bool = True) -> DataFrame:
        ...

    def _check_empty_df(self, pairdf: DataFrame, pair: str, timeframe: str, candle_type: CandleType, warn_no_data: bool, warn_price: bool = False) -> bool:
        ...

    def _validate_pairdata(self, pair: str, pairdata: DataFrame, timeframe: str, candle_type: CandleType, timerange: TimeRange) -> None:
        ...

    def rename_futures_data(self, pair: str, new_pair: str, timeframe: str, candle_type: CandleType) -> None:
        ...

    def fix_funding_fee_timeframe(self, ff_timeframe: str) -> None:
        ...

def get_datahandlerclass(datatype: str) -> IDataHandler:
    ...

def get_datahandler(datadir: Path, data_format: str = None, data_handler: IDataHandler = None) -> IDataHandler:
    ...
