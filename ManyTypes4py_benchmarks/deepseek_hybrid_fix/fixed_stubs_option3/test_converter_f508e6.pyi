from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from pathlib import Path
from _pytest.logging import LogCaptureFixture
import numpy as np
import pandas as pd
from freqtrade.configuration.timerange import TimeRange
from freqtrade.data.converter import (
    convert_ohlcv_format,
    convert_trades_format,
    convert_trades_to_ohlcv,
    ohlcv_fill_up_missing_data,
    ohlcv_to_dataframe,
    reduce_dataframe_footprint,
    trades_df_remove_duplicates,
    trades_dict_to_list,
    trades_to_ohlcv,
    trim_dataframe,
)
from freqtrade.data.history import get_timerange, load_data, load_pair_history, validate_backtest_data
from freqtrade.data.history.datahandlers import IDataHandler
from freqtrade.enums import CandleType
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_seconds
from tests.conftest import generate_test_data, generate_trades_history, log_has, log_has_re
from tests.data.test_history import _clean_test_file

def test_dataframe_correct_columns(dataframe_1m: pd.DataFrame) -> None: ...
def test_ohlcv_to_dataframe(ohlcv_history_list: List[List[Union[int, float, str]]], caplog: LogCaptureFixture) -> None: ...
def test_trades_to_ohlcv(trades_history_df: pd.DataFrame, caplog: LogCaptureFixture) -> None: ...
def test_trades_to_ohlcv_multi(
    timeframe: str,
    rows: int,
    days: int,
    candles: int,
    start: str,
    end: str,
    weekday: Optional[str],
) -> None: ...
def test_ohlcv_fill_up_missing_data(testdatadir: Path, caplog: LogCaptureFixture) -> None: ...
def test_ohlcv_fill_up_missing_data2(caplog: LogCaptureFixture) -> None: ...
def test_ohlcv_to_dataframe_multi(timeframe: str) -> None: ...
def test_ohlcv_to_dataframe_1M() -> None: ...
def test_ohlcv_drop_incomplete(caplog: LogCaptureFixture) -> None: ...
def test_trim_dataframe(testdatadir: Path) -> None: ...
def test_trades_df_remove_duplicates(trades_history_df: pd.DataFrame) -> None: ...
def test_trades_dict_to_list(fetch_trades_result: List[Dict[str, Any]]) -> None: ...
def test_convert_trades_format(
    default_conf: Dict[str, Any],
    testdatadir: Path,
    tmp_path: Path,
) -> None: ...
def test_convert_ohlcv_format(
    default_conf: Dict[str, Any],
    testdatadir: Path,
    tmp_path: Path,
    file_base: List[str],
    candletype: CandleType,
) -> None: ...
def test_reduce_dataframe_footprint() -> None: ...
def test_convert_trades_to_ohlcv(
    testdatadir: Path,
    tmp_path: Path,
    caplog: LogCaptureFixture,
) -> None: ...