```python
import logging
from typing import Optional

import numpy as np
from pandas import DataFrame, read_json, to_datetime

from freqtrade import misc
from freqtrade.configuration import TimeRange
from freqtrade.constants import DEFAULT_DATAFRAME_COLUMNS, DEFAULT_TRADES_COLUMNS
from freqtrade.data.converter import trades_dict_to_list, trades_list_to_df
from freqtrade.enums import CandleType, TradingMode

from .idatahandler import IDataHandler


logger = logging.getLogger(__name__)


class JsonDataHandler(IDataHandler):
    _use_zip: bool = False
    _columns: list[str] = DEFAULT_DATAFRAME_COLUMNS

    def ohlcv_store(
        self, pair: str, timeframe: str, data: DataFrame, candle_type: CandleType
    ) -> None:
        filename: str = self._pair_data_filename(self._datadir, pair, timeframe, candle_type)
        self.create_dir_if_needed(filename)
        _data: DataFrame = data.copy()
        _data["date"] = _data["date"].astype(np.int64) // 1000 // 1000
        _data.reset_index(drop=True).loc[:, self._columns].to_json(
            filename, orient="values", compression="gzip" if self._use_zip else None
        )

    def _ohlcv_load(
        self, pair: str, timeframe: str, timerange: Optional[TimeRange], candle_type: CandleType
    ) -> DataFrame:
        filename: str = self._pair_data_filename(self._datadir, pair, timeframe, candle_type=candle_type)
        if not filename.exists():
            filename = self._pair_data_filename(
                self._datadir, pair, timeframe, candle_type=candle_type, no_timeframe_modify=True
            )
            if not filename.exists():
                return DataFrame(columns=self._columns)
        try:
            pairdata: DataFrame = read_json(filename, orient="values")
            pairdata.columns = self._columns
        except ValueError:
            logger.error(f"Could not load data for {pair}.")
            return DataFrame(columns=self._columns)
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

    def ohlcv_append(
        self, pair: str, timeframe: str, data: DataFrame, candle_type: CandleType
    ) -> None:
        raise NotImplementedError()

    def _trades_store(self, pair: str, data: DataFrame, trading_mode: TradingMode) -> None:
        filename: str = self._pair_trades_filename(self._datadir, pair, trading_mode)
        trades: list = data.values.tolist()
        misc.file_dump_json(filename, trades, is_zip=self._use_zip)

    def trades_append(self, pair: str, data: DataFrame) -> None:
        raise NotImplementedError()

    def _trades_load(
        self, pair: str, trading_mode: TradingMode, timerange: Optional[TimeRange] = None
    ) -> DataFrame:
        filename: str = self._pair_trades_filename(self._datadir, pair, trading_mode)
        tradesdata: list = misc.file_load_json(filename)

        if not tradesdata:
            return DataFrame(columns=DEFAULT_TRADES_COLUMNS)

        if isinstance(tradesdata[0], dict):
            logger.info("Old trades format detected - converting")
            tradesdata = trades_dict_to_list(tradesdata)
            pass
        return trades_list_to_df(tradesdata, convert=False)

    @classmethod
    def _get_file_extension(cls) -> str:
        return "json.gz" if cls._use_zip else "json"


class JsonGzDataHandler(JsonDataHandler):
    _use_zip: bool = True
```