import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pytest import LogCaptureFixture
from freqtrade.configuration.timerange import TimeRange
from freqtrade.data.history.datahandlers import IDataHandler
from freqtrade.enums import CandleType

def test_dataframe_correct_columns(dataframe_1m: pd.DataFrame) -> None:
    ...

def test_ohlcv_to_dataframe(ohlcv_history_list: List[List[Any]], caplog: LogCaptureFixture) -> None:
    ...

def test_trades_to_ohlcv(trades_history_df: pd.DataFrame, caplog: LogCaptureFixture) -> None:
    ...

def test_trades_to_ohlcv_multi(
    timeframe: str,
    rows: int,
    days: int,
    candles: int,
    start: str,
    end: str,
    weekday: Optional[str]
) -> None:
    ...

def test_ohlcv_fill_up_missing_data(testdatadir: Path, caplog: LogCaptureFixture) -> None:
    ...

def test_ohlcv_fill_up_missing_data2(caplog: LogCaptureFixture) -> None:
    ...

def test_ohlcv_to_dataframe_multi(timeframe: str) -> None:
    ...

def test_ohlcv_to_dataframe_1M() -> None:
    ...

def test_ohlcv_drop_incomplete(caplog: LogCaptureFixture) -> None:
    ...

def test_trim_dataframe(testdatadir: Path) -> None:
    ...

def test_trades_df_remove_duplicates(trades_history_df: pd.DataFrame) -> None:
    ...

def test_trades_dict_to_list(fetch_trades_result: List[Dict[str, Any]]) -> None:
    ...

def test_convert_trades_format(default_conf: Dict[str, Any], testdatadir: Path, tmp_path: Path) -> None:
    ...

def test_convert_ohlcv_format(
    default_conf: Dict[str, Any],
    testdatadir: Path,
    tmp_path: Path,
    file_base: List[str],
    candletype: CandleType
) -> None:
    ...

def test_reduce_dataframe_footprint() -> None:
    ...

def test_convert_trades_to_ohlcv(
    testdatadir: Path,
    tmp_path: Path,
    caplog: LogCaptureFixture
) -> None:
    ...