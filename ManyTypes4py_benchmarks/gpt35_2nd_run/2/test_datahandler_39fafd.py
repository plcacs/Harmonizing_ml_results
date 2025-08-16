from datetime import datetime, timezone
from pathlib import Path
from typing import Set, Tuple
import pytest
from pandas import DataFrame, Timestamp
from freqtrade.configuration import TimeRange
from freqtrade.constants import AVAILABLE_DATAHANDLERS
from freqtrade.data.history.datahandlers.featherdatahandler import FeatherDataHandler
from freqtrade.data.history.datahandlers.idatahandler import IDataHandler, get_datahandler, get_datahandlerclass
from freqtrade.data.history.datahandlers.jsondatahandler import JsonDataHandler, JsonGzDataHandler
from freqtrade.data.history.datahandlers.parquetdatahandler import ParquetDataHandler
from freqtrade.enums import CandleType, TradingMode
from freqtrade.exceptions import OperationalException

def test_datahandler_ohlcv_get_pairs(testdatadir: Path) -> None:
    pairs: Set[str] = FeatherDataHandler.ohlcv_get_pairs(testdatadir, '5m', candle_type=CandleType.SPOT)
    assert set(pairs) == {'UNITTEST/BTC', 'XLM/BTC', 'ETH/BTC', 'TRX/BTC', 'LTC/BTC', 'XMR/BTC', 'ZEC/BTC', 'ADA/BTC', 'ETC/BTC', 'NXT/BTC', 'DASH/BTC', 'XRP/ETH'}
    pairs = JsonGzDataHandler.ohlcv_get_pairs(testdatadir, '8m', candle_type=CandleType.SPOT)
    assert set(pairs) == {'UNITTEST/BTC'}
    pairs = FeatherDataHandler.ohlcv_get_pairs(testdatadir, '1h', candle_type=CandleType.MARK)
    assert set(pairs) == {'UNITTEST/USDT:USDT', 'XRP/USDT:USDT'}
    pairs = JsonGzDataHandler.ohlcv_get_pairs(testdatadir, '1h', candle_type=CandleType.FUTURES)
    assert set(pairs) == {'XRP/USDT:USDT'}

def test_datahandler_ohlcv_regex(filename: str, pair: str, timeframe: str, candletype: str) -> None:
    regex = JsonDataHandler._OHLCV_REGEX
    match = re.search(regex, filename)
    assert len(match.groups()) > 1
    assert match[1] == pair
    assert match[2] == timeframe
    assert match[3] == candletype

def test_rebuild_pair_from_filename(pair: str, expected: str) -> None:
    assert IDataHandler.rebuild_pair_from_filename(pair) == expected

def test_datahandler_ohlcv_get_available_data(testdatadir: Path) -> None:
    paircombs: Set[Tuple[str, str, CandleType]] = FeatherDataHandler.ohlcv_get_available_data(testdatadir, TradingMode.SPOT)
    assert set(paircombs) == {('UNITTEST/BTC', '5m', CandleType.SPOT), ('ETH/BTC', '5m', CandleType.SPOT), ('XLM/BTC', '5m', CandleType.SPOT), ('TRX/BTC', '5m', CandleType.SPOT), ('LTC/BTC', '5m', CandleType.SPOT), ('XMR/BTC', '5m', CandleType.SPOT), ('ZEC/BTC', '5m', CandleType.SPOT), ('UNITTEST/BTC', '1m', CandleType.SPOT), ('ADA/BTC', '5m', CandleType.SPOT), ('ETC/BTC', '5m', CandleType.SPOT), ('NXT/BTC', '5m', CandleType.SPOT), ('DASH/BTC', '5m', CandleType.SPOT), ('XRP/ETH', '1m', CandleType.SPOT), ('XRP/ETH', '5m', CandleType.SPOT), ('UNITTEST/BTC', '30m', CandleType.SPOT), ('UNITTEST/BTC', '8m', CandleType.SPOT)}
    paircombs = FeatherDataHandler.ohlcv_get_available_data(testdatadir, TradingMode.FUTURES)
    assert set(paircombs) == {('UNITTEST/USDT:USDT', '1h', 'mark'), ('XRP/USDT:USDT', '5m', 'futures'), ('XRP/USDT:USDT', '1h', 'futures'), ('XRP/USDT:USDT', '1h', 'mark'), ('XRP/USDT:USDT', '8h', 'mark'), ('XRP/USDT:USDT', '8h', 'funding_rate')}
    paircombs = JsonGzDataHandler.ohlcv_get_available_data(testdatadir, TradingMode.SPOT)
    assert set(paircombs) == {('UNITTEST/BTC', '8m', CandleType.SPOT)}

def test_jsondatahandler_ohlcv_purge(mocker, testdatadir: Path) -> None:
    ...

def test_jsondatahandler_ohlcv_load(testdatadir: Path, caplog: MagicMock) -> None:
    ...

def test_datahandler_ohlcv_data_min_max(testdatadir: Path) -> None:
    ...

def test_datahandler__check_empty_df(testdatadir: Path, caplog: MagicMock) -> None:
    ...

def test_datahandler_trades_not_supported(datahandler: str, testdatadir: Path) -> None:
    ...

def test_jsondatahandler_trades_load(testdatadir: Path, caplog: MagicMock) -> None:
    ...

def test_datahandler_ohlcv_append(datahandler: str, testdatadir: Path) -> None:
    ...

def test_datahandler_trades_append(datahandler: str, testdatadir: Path) -> None:
    ...

def test_datahandler_trades_get_pairs(testdatadir: Path, datahandler: str, expected: Set[str]) -> None:
    ...

def test_hdf5datahandler_deprecated(testdatadir: Path) -> None:
    ...

def test_generic_datahandler_ohlcv_load_and_resave(datahandler: str, mocker, testdatadir: Path, tmp_path: Path, pair: str, timeframe: str, candle_type: str, candle_append: str, startdt: str, enddt: str, caplog: MagicMock) -> None:
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
