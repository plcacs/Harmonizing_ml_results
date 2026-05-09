import json
import logging
import uuid
from datetime import timedelta
from pathlib import Path
from shutil import copyfile
from unittest.mock import MagicMock, PropertyMock
from typing import Any, Dict, List, Optional, Union

import pytest
from pandas import DataFrame
from pytest import LogCaptureFixture
from unittest.mock import Mocker
from time_machine import TimeMachine

from freqtrade.configuration import TimeRange
from freqtrade.constants import DATETIME_PRINT_FORMAT
from freqtrade.data.history.datahandlers.jsondatahandler import JsonDataHandler, JsonGzDataHandler
from freqtrade.enums import CandleType, TradingMode
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.util import dt_ts, dt_utc

def _clean_test_file(file: Path) -> None:
    ...

def test_load_data_30min_timeframe(caplog: LogCaptureFixture, testdatadir: Path) -> None:
    ...

def test_load_data_7min_timeframe(caplog: LogCaptureFixture, testdatadir: Path) -> None:
    ...

def test_load_data_1min_timeframe(ohlcv_history: List[List], mocker: Mocker, caplog: LogCaptureFixture, testdatadir: Path) -> None:
    ...

def test_load_data_mark(ohlcv_history: List[List], mocker: Mocker, caplog: LogCaptureFixture, testdatadir: Path) -> None:
    ...

def test_load_data_startup_candles(mocker: Mocker, testdatadir: Path) -> None:
    ...

def test_load_data_with_new_pair_1min(ohlcv_history: List[List], mocker: Mocker, caplog: LogCaptureFixture, default_conf: Dict, tmp_path: Path, candle_type: str) -> None:
    ...

def test_testdata_path(testdatadir: Path) -> None:
    ...

def test_json_pair_data_filename(pair: str, timeframe: str, expected_result: str, candle_type: CandleType) -> None:
    ...

def test_json_pair_trades_filename(pair: str, trading_mode: str, expected_result: str) -> None:
    ...

def test_load_cached_data_for_updating(testdatadir: Path) -> None:
    ...

def test_download_pair_history(ohlcv_history: List[List], mocker: Mocker, default_conf: Dict, tmp_path: Path, candle_type: str, subdir: str, file_tail: str) -> None:
    ...

def test_download_pair_history2(mocker: Mocker, default_conf: Dict, testdatadir: Path, ohlcv_history: List[List]) -> None:
    ...

def test_download_backtesting_data_exception(mocker: Mocker, caplog: LogCaptureFixture, default_conf: Dict, tmp_path: Path) -> None:
    ...

def test_load_partial_missing(testdatadir: Path, caplog: LogCaptureFixture) -> None:
    ...

def test_init(default_conf: Dict) -> None:
    ...

def test_init_with_refresh(default_conf: Dict, mocker: Mocker) -> None:
    ...

def test_file_dump_json_tofile(testdatadir: Path) -> None:
    ...

def test_get_timerange(default_conf: Dict, mocker: Mocker, testdatadir: Path) -> None:
    ...

def test_validate_backtest_data_warn(default_conf: Dict, mocker: Mocker, caplog: LogCaptureFixture, testdatadir: Path) -> None:
    ...

def test_validate_backtest_data(default_conf: Dict, mocker: Mocker, caplog: LogCaptureFixture, testdatadir: Path) -> None:
    ...

def test_refresh_backtest_ohlcv_data(mocker: Mocker, default_conf: Dict, markets: List, caplog: LogCaptureFixture, testdatadir: Path, trademode: str, callcount: int) -> None:
    ...

def test_download_data_no_markets(mocker: Mocker, default_conf: Dict, caplog: LogCaptureFixture, testdatadir: Path) -> None:
    ...

def test_refresh_backtest_trades_data(mocker: Mocker, default_conf: Dict, markets: List, caplog: LogCaptureFixture, testdatadir: Path) -> None:
    ...

def test_download_trades_history(trades_history: List[List], mocker: Mocker, default_conf: Dict, testdatadir: Path, caplog: LogCaptureFixture, tmp_path: Path, time_machine: TimeMachine) -> None:
    ...