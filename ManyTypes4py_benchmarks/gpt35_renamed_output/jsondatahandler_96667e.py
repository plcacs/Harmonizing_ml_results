from typing import Any, Dict, List, Union
import numpy as np
from pandas import DataFrame, to_datetime
from freqtrade.enums import CandleType, TradingMode

class JsonDataHandler(IDataHandler):
    _use_zip: bool = False
    _columns: List[str] = DEFAULT_DATAFRAME_COLUMNS

    def func_dowpodh2(self, pair: str, timeframe: str, data: DataFrame, candle_type: CandleType) -> None:
        ...

    def func_pxl57nr5(self, pair: str, timeframe: str, timerange: TimeRange, candle_type: CandleType) -> DataFrame:
        ...

    def func_4xk3gshd(self, pair: str, timeframe: str, data: DataFrame, candle_type: CandleType) -> None:
        ...

    def func_dlmcsaky(self, pair: str, data: DataFrame, trading_mode: TradingMode) -> None:
        ...

    def func_7zh6rfbg(self, pair: str, data: DataFrame) -> None:
        ...

    def func_vb812v07(self, pair: str, trading_mode: TradingMode, timerange: Union[TimeRange, None] = None) -> DataFrame:
        ...

    @classmethod
    def func_duwzox7z(cls) -> str:
        ...
