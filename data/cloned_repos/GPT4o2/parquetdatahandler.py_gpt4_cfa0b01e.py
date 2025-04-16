```python
import logging
from pandas import DataFrame, read_parquet, to_datetime
from freqtrade.configuration import TimeRange
from freqtrade.constants import DEFAULT_DATAFRAME_COLUMNS, DEFAULT_TRADES_COLUMNS
from freqtrade.enums import CandleType, TradingMode
from .idatahandler import IDataHandler

logger = logging.getLogger(__name__)

class ParquetDataHandler(IDataHandler):
    _columns = DEFAULT_DATAFRAME_COLUMNS

    def ohlcv_store(
        self, pair: str, timeframe: str, data: DataFrame, candle_type: CandleType
    ) -> None:
        filename: str = self._pair_data_filename(self._datadir, pair, timeframe, candle_type)
        self.create_dir_if_needed(filename)
        data.reset_index(drop=True).loc[:, self._columns].to_parquet(filename)

    def _ohlcv_load(
        self, pair: str, timeframe: str, timerange: TimeRange | None, candle_type: CandleType
    ) -> DataFrame:
        filename: str = self._pair_data_filename(self._datadir, pair, timeframe, candle_type=candle_type)
        if not filename.exists():
            filename = self._pair_data_filename(
                self._datadir, pair, timeframe, candle_type=candle_type, no_timeframe_modify=True
            )
            if not filename.exists():
                return DataFrame(columns=self._columns)
        try:
            pairdata: DataFrame = read_parquet(filename)
            pairdata.columns = self._columns
            pairdata = pairdata.astype(
                dtype={
                    "open": "float",
                    "high": "float",
                    "low": "float",
                    "close": "float",
                    "volume": "float",
                }
            )
            pairdata["date"] = to_datetime(pairdata["date"], unit="ms", utc=True)
            return pairdata
        except Exception as e:
            logger.exception(
                f"Error loading data from {filename}. Exception: {e}. Returning empty dataframe."
            )
            return DataFrame(columns=self._columns)

    def ohlcv_append(
        self, pair: str, timeframe: str, data: DataFrame, candle_type: CandleType
    ) -> None:
        raise NotImplementedError()

    def _trades_store(self, pair: str, data: DataFrame, trading_mode: TradingMode) -> None:
        filename: str = self._pair_trades_filename(self._datadir, pair, trading_mode)
        self.create_dir_if_needed(filename)
        data.reset_index(drop=True).to_parquet(filename)

    def trades_append(self, pair: str, data: DataFrame) -> None:
        raise NotImplementedError()

    def _trades_load(
        self, pair: str, trading_mode: TradingMode, timerange: TimeRange | None = None
    ) -> DataFrame:
        filename: str = self._pair_trades_filename(self._datadir, pair, trading_mode)
        if not filename.exists():
            return DataFrame(columns=DEFAULT_TRADES_COLUMNS)

        tradesdata: DataFrame = read_parquet(filename)
        return tradesdata

    @classmethod
    def _get_file_extension(cls) -> str:
        return "parquet"
```