import logging
from pandas import DataFrame, read_parquet, to_datetime
from freqtrade.configuration import TimeRange
from freqtrade.constants import DEFAULT_DATAFRAME_COLUMNS, DEFAULT_TRADES_COLUMNS
from freqtrade.enums import CandleType, TradingMode
from .idatahandler import IDataHandler
from typing import Any

logger: logging.Logger = logging.getLogger(__name__)

class ParquetDataHandler(IDataHandler):
    _columns: list[str] = DEFAULT_DATAFRAME_COLUMNS

    def ohlcv_store(self, pair: Any, timeframe: Any, data: DataFrame, candle_type: CandleType) -> None:
        ...

    def _ohlcv_load(self, pair: Any, timeframe: Any, timerange: TimeRange, candle_type: CandleType) -> DataFrame:
        ...

    def ohlcv_append(self, pair: Any, timeframe: Any, data: DataFrame, candle_type: CandleType) -> None:
        ...

    def _trades_store(self, pair: Any, data: DataFrame, trading_mode: TradingMode) -> None:
        ...

    def trades_append(self, pair: Any, data: DataFrame) -> None:
        ...

    def _trades_load(self, pair: Any, trading_mode: TradingMode, timerange: TimeRange = None) -> DataFrame:
        ...

    @classmethod
    def _get_file_extension(cls) -> str:
        ...
