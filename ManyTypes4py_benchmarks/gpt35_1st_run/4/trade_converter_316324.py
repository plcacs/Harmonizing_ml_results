from typing import List
import pandas as pd
from pandas import DataFrame, to_datetime
from freqtrade.constants import DEFAULT_DATAFRAME_COLUMNS, DEFAULT_TRADES_COLUMNS, TRADES_DTYPES, Config, TradeList
from freqtrade.enums import CandleType, TradingMode
from freqtrade.exceptions import OperationalException

def trades_df_remove_duplicates(trades: DataFrame) -> DataFrame:
    ...

def trades_dict_to_list(trades: List[dict]) -> List[List]:
    ...

def trades_convert_types(trades: DataFrame) -> DataFrame:
    ...

def trades_list_to_df(trades: List[List], convert: bool = True) -> DataFrame:
    ...

def trades_to_ohlcv(trades: DataFrame, timeframe: str) -> DataFrame:
    ...

def convert_trades_to_ohlcv(pairs: List[str], timeframes: List[str], datadir: str, timerange: str, erase: bool, data_format_ohlcv: str, data_format_trades: str, candle_type: str) -> None:
    ...

def convert_trades_format(config: dict, convert_from: str, convert_to: str, erase: bool) -> None:
    ...
