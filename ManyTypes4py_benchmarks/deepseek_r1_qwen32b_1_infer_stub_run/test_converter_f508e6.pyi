import logging
from typing import Any, Optional, List, Tuple, Dict, Union
import numpy as np
import pandas as pd
import pytest
from pytest import LogCaptureFixture
from freqtrade.enums import CandleType
from freqtrade.data.history.datahandlers import IDataHandler

def test_dataframe_correct_columns(dataframe_1m: pd.DataFrame) -> None:
    ...

def test_ohlcv_to_dataframe(ohlcv_history_list: List[List[float]], timeframe: str, pair: str, fill_missing: bool) -> pd.DataFrame:
    ...

def test_trades_to_ohlcv(trades_history_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    ...

def test_ohlcv_fill_up_missing_data(data: pd.DataFrame, timeframe: str, pair: str) -> pd.DataFrame:
    ...

def test_ohlcv_fill_up_missing_data2(caplog: LogCaptureFixture) -> None:
    ...

def test_ohlcv_to_dataframe_multi(timeframe: str) -> None:
    ...

def test_ohlcv_to_dataframe_1M() -> None:
    ...

def test_ohlcv_drop_incomplete(caplog: LogCaptureFixture) -> None:
    ...

def test_trim_dataframe(data: pd.DataFrame, tr: TimeRange, startup_candles: int) -> pd.DataFrame:
    ...

def test_trades_df_remove_duplicates(trades_history_df: pd.DataFrame) -> pd.DataFrame:
    ...

def test_trades_dict_to_list(fetch_trades_result: List[Dict[str, Any]]) -> List[List[Any]]:
    ...

def test_convert_trades_format(default_conf: Dict[str, Any], testdatadir: str, tmp_path: str) -> None:
    ...

def test_convert_ohlcv_format(default_conf: Dict[str, Any], testdatadir: str, tmp_path: str, file_base: List[str], candletype: CandleType) -> None:
    ...

def test_reduce_dataframe_footprint(data: pd.DataFrame) -> pd.DataFrame:
    ...

def test_convert_trades_to_ohlcv(testdatadir: str, tmp_path: str, caplog: LogCaptureFixture) -> None:
    ...