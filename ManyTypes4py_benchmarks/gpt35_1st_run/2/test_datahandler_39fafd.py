from datetime import datetime, timezone
from pathlib import Path
from typing import Set, Tuple
import pytest
from pandas import DataFrame
from freqtrade.configuration import TimeRange
from freqtrade.constants import AVAILABLE_DATAHANDLERS
from freqtrade.data.history.datahandlers.featherdatahandler import FeatherDataHandler
from freqtrade.data.history.datahandlers.idatahandler import IDataHandler, get_datahandler, get_datahandlerclass
from freqtrade.data.history.datahandlers.jsondatahandler import JsonDataHandler, JsonGzDataHandler
from freqtrade.data.history.datahandlers.parquetdatahandler import ParquetDataHandler
from freqtrade.enums import CandleType, TradingMode
from freqtrade.exceptions import OperationalException

def test_datahandler_ohlcv_get_pairs(testdatadir: Path) -> None:
    ...

def test_datahandler_ohlcv_regex(filename: str, pair: str, timeframe: str, candletype: str) -> None:
    ...

def test_rebuild_pair_from_filename(pair: str, expected: str) -> None:
    ...

def test_datahandler_ohlcv_get_available_data(testdatadir: Path) -> None:
    ...

def test_jsondatahandler_ohlcv_purge(mocker, testdatadir: Path) -> None:
    ...

def test_jsondatahandler_ohlcv_load(testdatadir: Path, caplog) -> None:
    ...

def test_datahandler_ohlcv_data_min_max(testdatadir: Path) -> None:
    ...

def test_datahandler__check_empty_df(testdatadir: Path, caplog) -> None:
    ...

def test_datahandler_trades_not_supported(datahandler: str, testdatadir: Path) -> None:
    ...

def test_jsondatahandler_trades_load(testdatadir: Path, caplog) -> None:
    ...

def test_datahandler_ohlcv_append(datahandler: str, testdatadir: Path) -> None:
    ...

def test_datahandler_trades_append(datahandler: str, testdatadir: Path) -> None:
    ...

def test_datahandler_trades_get_pairs(testdatadir: Path, datahandler: str, expected: Set[str]) -> None:
    ...

def test_hdf5datahandler_deprecated(testdatadir: Path) -> None:
    ...

def test_generic_datahandler_ohlcv_load_and_resave(datahandler: str, mocker, testdatadir: Path, tmp_path: Path, pair: str, timeframe: str, candle_type: str, candle_append: str, startdt: str, enddt: str, caplog) -> None:
    ...

def test_datahandler_trades_load(testdatadir: Path, datahandler: str) -> None:
    ...

def test_datahandler_trades_store(testdatadir: Path, tmp_path: Path, datahandler: str) -> None:
    ...

def test_datahandler_trades_purge(mocker, testdatadir: Path, datahandler: str) -> None:
    ...

def test_datahandler_trades_get_available_data(testdatadir: Path) -> None:
    ...

def test_datahandler_trades_data_min_max(testdatadir: Path) -> None:
    ...

def test_gethandlerclass() -> None:
    ...

def test_get_datahandler(testdatadir: Path) -> None:
    ...
