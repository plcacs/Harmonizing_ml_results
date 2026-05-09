import re
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from pandas import DataFrame, Timestamp
from pandas.testing import assert_frame_equal
from freqtrade.configuration import TimeRange
from freqtrade.constants import AVAILABLE_DATAHANDLERS
from freqtrade.data.history.datahandlers.featherdatahandler import FeatherDataHandler
from freqtrade.data.history.datahandlers.idatahandler import IDataHandler, get_datahandler, get_datahandlerclass
from freqtrade.data.history.datahandlers.jsondatahandler import JsonDataHandler, JsonGzDataHandler
from freqtrade.data.history.datahandlers.parquetdatahandler import ParquetDataHandler
from freqtrade.enums import CandleType, TradingMode
from freqtrade.exceptions import OperationalException
from tests.conftest import log_has, log_has_re

def test_datahandler_ohlcv_get_pairs(testdatadir: Path) -> None:
    # ... rest of the function ...

@pytest.mark.parametrize('filename, pair, timeframe, candletype', [('XMR_BTC-5m.json', 'XMR_BTC', '5m', ''), ...])
def test_datahandler_ohlcv_regex(filename: str, pair: str, timeframe: str, candletype: str) -> None:
    # ... rest of the function ...

def test_rebuild_pair_from_filename(pair: str) -> str:
    # ... rest of the function ...

def test_datahandler_ohlcv_get_available_data(testdatadir: Path) -> None:
    # ... rest of the function ...

def test_jsondatahandler_ohlcv_purge(mocker: pytest.Mock, testdatadir: Path) -> None:
    # ... rest of the function ...

def test_datahandler_ohlcv_data_min_max(testdatadir: Path) -> None:
    # ... rest of the function ...

def test_datahandler__check_empty_df(testdatadir: Path, caplog: pytest.LogCaptureFixture) -> None:
    # ... rest of the function ...

def test_jsondatahandler_trades_load(testdatadir: Path, caplog: pytest.LogCaptureFixture) -> None:
    # ... rest of the function ...

@pytest.mark.parametrize('datahandler', AVAILABLE_DATAHANDLERS)
def test_datahandler_ohlcv_append(datahandler: str, testdatadir: Path) -> None:
    # ... rest of the function ...

@pytest.mark.parametrize('datahandler', AVAILABLE_DATAHANDLERS)
def test_datahandler_trades_append(datahandler: str, testdatadir: Path) -> None:
    # ... rest of the function ...

@pytest.mark.parametrize('datahandler, expected', [('jsongz', {'XRP/ETH', 'XRP/OLD'}), ...])
def test_datahandler_trades_get_pairs(testdatadir: Path, datahandler: str, expected: set) -> None:
    # ... rest of the function ...

def test_hdf5datahandler_deprecated(testdatadir: Path) -> None:
    # ... rest of the function ...

@pytest.mark.parametrize('pair, timeframe, candle_type, candle_append, startdt, enddt', ...)
def test_generic_datahandler_ohlcv_load_and_resave(datahandler: str, mocker: pytest.Mock, testdatadir: Path, tmp_path: Path, pair: str, timeframe: str, candle_type: str, candle_append: str, startdt: str, enddt: str, caplog: pytest.LogCaptureFixture) -> None:
    # ... rest of the function ...

@pytest.mark.parametrize('datahandler', ['jsongz', 'feather', 'parquet'])
def test_datahandler_trades_load(testdatadir: Path, datahandler: str) -> None:
    # ... rest of the function ...

@pytest.mark.parametrize('datahandler', ['jsongz', 'feather', 'parquet'])
def test_datahandler_trades_store(testdatadir: Path, tmp_path: Path, datahandler: str) -> None:
    # ... rest of the function ...

@pytest.mark.parametrize('datahandler', ['jsongz', 'feather', 'parquet'])
def test_datahandler_trades_purge(mocker: pytest.Mock, testdatadir: Path, datahandler: str) -> None:
    # ... rest of the function ...

def test_gethandlerclass() -> IDataHandler:
    # ... rest of the function ...

def test_get_datahandler(testdatadir: Path) -> IDataHandler:
    # ... rest of the function ...
