import logging
from typing import List, Dict
import numpy as np
import pandas as pd
from pandas import DataFrame, to_datetime
from freqtrade.constants import DEFAULT_DATAFRAME_COLUMNS, Config
from freqtrade.enums import CandleType, TradingMode

logger = logging.getLogger(__name__)

def ohlcv_to_dataframe(ohlcv: List[Dict[str, float]], timeframe: str, pair: str, *, fill_missing: bool = True, drop_incomplete: bool = True) -> DataFrame:
    ...

def clean_ohlcv_dataframe(data: DataFrame, timeframe: str, pair: str, *, fill_missing: bool, drop_incomplete: bool) -> DataFrame:
    ...

def ohlcv_fill_up_missing_data(dataframe: DataFrame, timeframe: str, pair: str) -> DataFrame:
    ...

def trim_dataframe(df: DataFrame, timerange, *, df_date_col: str = 'date', startup_candles: int = 0) -> DataFrame:
    ...

def trim_dataframes(preprocessed: Dict[str, DataFrame], timerange, startup_candles: int) -> Dict[str, DataFrame]:
    ...

def order_book_to_dataframe(bids: List[Dict[str, float]], asks: List[Dict[str, float]]) -> DataFrame:
    ...

def convert_ohlcv_format(config: Config, convert_from: str, convert_to: str, erase: bool) -> None:
    ...

def reduce_dataframe_footprint(df: DataFrame) -> DataFrame:
    ...
