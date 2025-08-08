from pandas import DataFrame, read_feather, to_datetime
from freqtrade.enums import CandleType, TradingMode
from .idatahandler import IDataHandler

class FeatherDataHandler(IDataHandler):
    _columns: List[str] = DEFAULT_DATAFRAME_COLUMNS

    def ohlcv_store(self, pair: str, timeframe: str, data: DataFrame, candle_type: CandleType) -> None:
        ...

    def _ohlcv_load(self, pair: str, timeframe: str, timerange: TimeRange, candle_type: CandleType) -> DataFrame:
        ...

    def ohlcv_append(self, pair: str, timeframe: str, data: DataFrame, candle_type: CandleType) -> None:
        ...

    def _trades_store(self, pair: str, data: DataFrame, trading_mode: TradingMode) -> None:
        ...

    def trades_append(self, pair: str, data: DataFrame) -> None:
        ...

    def _trades_load(self, pair: str, trading_mode: TradingMode, timerange: TimeRange = None) -> DataFrame:
        ...

    @classmethod
    def _get_file_extension(cls) -> str:
        ...
