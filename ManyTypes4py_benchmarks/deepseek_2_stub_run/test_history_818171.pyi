```python
import _pytest.capture
import _pytest.fixtures
import _pytest.logging
import _pytest.monkeypatch
import _pytest.pytester
import _pytest.tmpdir
from _pytest.config import Config
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest
from freqtrade.configuration import TimeRange
from freqtrade.data.history.datahandlers.jsondatahandler import (
    JsonDataHandler,
    JsonGzDataHandler,
)
from freqtrade.enums import CandleType, TradingMode
from freqtrade.exchange import Exchange
from pandas import DataFrame

def _clean_test_file(file: Path) -> None: ...

def test_load_data_30min_timeframe(
    caplog: _pytest.logging.LogCaptureFixture,
    testdatadir: Path,
) -> None: ...

def test_load_data_7min_timeframe(
    caplog: _pytest.logging.LogCaptureFixture,
    testdatadir: Path,
) -> None: ...

def test_load_data_1min_timeframe(
    ohlcv_history: Any,
    mocker: _pytest.monkeypatch.MonkeyPatch,
    caplog: _pytest.logging.LogCaptureFixture,
    testdatadir: Path,
) -> None: ...

def test_load_data_mark(
    ohlcv_history: Any,
    mocker: _pytest.monkeypatch.MonkeyPatch,
    caplog: _pytest.logging.LogCaptureFixture,
    testdatadir: Path,
) -> None: ...

def test_load_data_startup_candles(
    mocker: _pytest.monkeypatch.MonkeyPatch,
    testdatadir: Path,
) -> None: ...

@pytest.mark.parametrize("candle_type", ["mark", ""])
def test_load_data_with_new_pair_1min(
    ohlcv_history: Any,
    mocker: _pytest.monkeypatch.MonkeyPatch,
    caplog: _pytest.logging.LogCaptureFixture,
    default_conf: dict[str, Any],
    tmp_path: Path,
    candle_type: str,
) -> None: ...

def test_testdata_path(testdatadir: Path) -> None: ...

@pytest.mark.parametrize(
    "pair,timeframe,expected_result,candle_type",
    [
        ("ETH/BTC", "5m", "freqtrade/hello/world/ETH_BTC-5m.json", ""),
        ("ETH/USDT", "1M", "freqtrade/hello/world/ETH_USDT-1Mo.json", ""),
        ("Fabric Token/ETH", "5m", "freqtrade/hello/world/Fabric_Token_ETH-5m.json", ""),
        ("ETHH20", "5m", "freqtrade/hello/world/ETHH20-5m.json", ""),
        (".XBTBON2H", "5m", "freqtrade/hello/world/_XBTBON2H-5m.json", ""),
        ("ETHUSD.d", "5m", "freqtrade/hello/world/ETHUSD_d-5m.json", ""),
        ("ACC_OLD/BTC", "5m", "freqtrade/hello/world/ACC_OLD_BTC-5m.json", ""),
        ("ETH/BTC", "5m", "freqtrade/hello/world/futures/ETH_BTC-5m-mark.json", "mark"),
        ("ACC_OLD/BTC", "5m", "freqtrade/hello/world/futures/ACC_OLD_BTC-5m-index.json", "index"),
    ],
)
def test_json_pair_data_filename(
    pair: str,
    timeframe: str,
    expected_result: str,
    candle_type: str,
) -> None: ...

@pytest.mark.parametrize(
    "pair,trading_mode,expected_result",
    [
        ("ETH/BTC", "", "freqtrade/hello/world/ETH_BTC-trades.json"),
        ("ETH/USDT:USDT", "futures", "freqtrade/hello/world/futures/ETH_USDT_USDT-trades.json"),
        ("Fabric Token/ETH", "", "freqtrade/hello/world/Fabric_Token_ETH-trades.json"),
        ("ETHH20", "", "freqtrade/hello/world/ETHH20-trades.json"),
        (".XBTBON2H", "", "freqtrade/hello/world/_XBTBON2H-trades.json"),
        ("ETHUSD.d", "", "freqtrade/hello/world/ETHUSD_d-trades.json"),
        ("ACC_OLD_BTC", "", "freqtrade/hello/world/ACC_OLD_BTC-trades.json"),
    ],
)
def test_json_pair_trades_filename(
    pair: str,
    trading_mode: str,
    expected_result: str,
) -> None: ...

def test_load_cached_data_for_updating(testdatadir: Path) -> None: ...

@pytest.mark.parametrize("candle_type,subdir,file_tail", [("mark", "futures/", "-mark"), ("spot", "", "")])
def test_download_pair_history(
    ohlcv_history: Any,
    mocker: _pytest.monkeypatch.MonkeyPatch,
    default_conf: dict[str, Any],
    tmp_path: Path,
    candle_type: str,
    subdir: str,
    file_tail: str,
) -> None: ...

def test_download_pair_history2(
    mocker: _pytest.monkeypatch.MonkeyPatch,
    default_conf: dict[str, Any],
    testdatadir: Path,
    ohlcv_history: Any,
) -> None: ...

def test_download_backtesting_data_exception(
    mocker: _pytest.monkeypatch.MonkeyPatch,
    caplog: _pytest.logging.LogCaptureFixture,
    default_conf: dict[str, Any],
    tmp_path: Path,
) -> None: ...

def test_load_partial_missing(
    testdatadir: Path,
    caplog: _pytest.logging.LogCaptureFixture,
) -> None: ...

def test_init(default_conf: dict[str, Any]) -> None: ...

def test_init_with_refresh(
    default_conf: dict[str, Any],
    mocker: _pytest.monkeypatch.MonkeyPatch,
) -> None: ...

def test_file_dump_json_tofile(testdatadir: Path) -> None: ...

def test_get_timerange(
    default_conf: dict[str, Any],
    mocker: _pytest.monkeypatch.MonkeyPatch,
    testdatadir: Path,
) -> None: ...

def test_validate_backtest_data_warn(
    default_conf: dict[str, Any],
    mocker: _pytest.monkeypatch.MonkeyPatch,
    caplog: _pytest.logging.LogCaptureFixture,
    testdatadir: Path,
) -> None: ...

def test_validate_backtest_data(
    default_conf: dict[str, Any],
    mocker: _pytest.monkeypatch.MonkeyPatch,
    caplog: _pytest.logging.LogCaptureFixture,
    testdatadir: Path,
) -> None: ...

@pytest.mark.parametrize("trademode,callcount", [("spot", 4), ("margin", 4), ("futures", 8)])
def test_refresh_backtest_ohlcv_data(
    mocker: _pytest.monkeypatch.MonkeyPatch,
    default_conf: dict[str, Any],
    markets: Any,
    caplog: _pytest.logging.LogCaptureFixture,
    testdatadir: Path,
    trademode: str,
    callcount: int,
) -> None: ...

def test_download_data_no_markets(
    mocker: _pytest.monkeypatch.MonkeyPatch,
    default_conf: dict[str, Any],
    caplog: _pytest.logging.LogCaptureFixture,
    testdatadir: Path,
) -> None: ...

def test_refresh_backtest_trades_data(
    mocker: _pytest.monkeypatch.MonkeyPatch,
    default_conf: dict[str, Any],
    markets: Any,
    caplog: _pytest.logging.LogCaptureFixture,
    testdatadir: Path,
) -> None: ...

def test_download_trades_history(
    trades_history: Any,
    mocker: _pytest.monkeypatch.MonkeyPatch,
    default_conf: dict[str, Any],
    testdatadir: Path,
    caplog: _pytest.logging.LogCaptureFixture,
    tmp_path: Path,
    time_machine: Any,
) -> None: ...
```