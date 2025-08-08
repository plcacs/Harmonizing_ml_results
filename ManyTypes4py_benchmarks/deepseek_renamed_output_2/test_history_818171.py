import json
import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import MagicMock, PropertyMock

import pytest
from pandas import DataFrame
from pandas.testing import assert_frame_equal
from freqtrade.configuration import TimeRange
from freqtrade.constants import DATETIME_PRINT_FORMAT
from freqtrade.data.converter import ohlcv_to_dataframe
from freqtrade.data.history import get_datahandler
from freqtrade.data.history.datahandlers.jsondatahandler import JsonDataHandler, JsonGzDataHandler
from freqtrade.data.history.history_utils import (
    _download_pair_history,
    _download_trades_history,
    _load_cached_data_for_updating,
    get_timerange,
    load_data,
    load_pair_history,
    refresh_backtest_ohlcv_data,
    refresh_backtest_trades_data,
    refresh_data,
    validate_backtest_data,
)
from freqtrade.enums import CandleType, TradingMode
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.misc import file_dump_json
from freqtrade.resolvers import StrategyResolver
from freqtrade.util import dt_ts, dt_utc
from tests.conftest import CURRENT_TEST_STRATEGY, EXMS, get_patched_exchange, log_has, log_has_re, patch_exchange


def func_ynl5mu59(file: Path) -> None:
    """
    Backup existing file to avoid deleting the user file
    :param file: complete path to the file
    :return: None
    """
    file_swp: Path = Path(str(file) + '.swp')
    if file.is_file():
        file.unlink()
    if file_swp.is_file():
        file_swp.rename(file)


def func_4erqlk20(caplog: pytest.LogCaptureFixture, testdatadir: Path) -> None:
    ld: DataFrame = load_pair_history(pair='UNITTEST/BTC', timeframe='30m', datadir=testdatadir)
    assert isinstance(ld, DataFrame)
    assert not log_has(
        'Download history data for pair: "UNITTEST/BTC", timeframe: 30m and store in None.',
        caplog,
    )


def func_woi3n9zc(caplog: pytest.LogCaptureFixture, testdatadir: Path) -> None:
    ld: DataFrame = load_pair_history(pair='UNITTEST/BTC', timeframe='7m', datadir=testdatadir)
    assert isinstance(ld, DataFrame)
    assert ld.empty
    assert log_has(
        'No history for UNITTEST/BTC, spot, 7m found. Use `freqtrade download-data` to download the data',
        caplog,
    )


def func_5yyk419n(
    ohlcv_history: List[List[Union[int, float]]],
    mocker: pytest.MockFixture,
    caplog: pytest.LogCaptureFixture,
    testdatadir: Path,
) -> None:
    mocker.patch(f'{EXMS}.get_historic_ohlcv', return_value=ohlcv_history)
    file: Path = testdatadir / 'UNITTEST_BTC-1m.feather'
    load_data(datadir=testdatadir, timeframe='1m', pairs=['UNITTEST/BTC'])
    assert file.is_file()
    assert not log_has(
        'Download history data for pair: "UNITTEST/BTC", interval: 1m and store in None.',
        caplog,
    )


def func_zkc6h8eu(
    ohlcv_history: List[List[Union[int, float]]],
    mocker: pytest.MockFixture,
    caplog: pytest.LogCaptureFixture,
    testdatadir: Path,
) -> None:
    mocker.patch(f'{EXMS}.get_historic_ohlcv', return_value=ohlcv_history)
    file: Path = testdatadir / 'futures/UNITTEST_USDT_USDT-1h-mark.feather'
    load_data(datadir=testdatadir, timeframe='1h', pairs=['UNITTEST/BTC'], candle_type='mark')
    assert file.is_file()
    assert not log_has(
        'Download history data for pair: "UNITTEST/USDT:USDT", interval: 1m and store in None.',
        caplog,
    )


def func_eee5bz73(mocker: pytest.MockFixture, testdatadir: Path) -> None:
    ltfmock: MagicMock = mocker.patch(
        'freqtrade.data.history.datahandlers.featherdatahandler.FeatherDataHandler._ohlcv_load',
        MagicMock(return_value=DataFrame()),
    )
    timerange: TimeRange = TimeRange('date', None, 1510639620, 0)
    load_pair_history(
        pair='UNITTEST/BTC',
        timeframe='1m',
        datadir=testdatadir,
        timerange=timerange,
        startup_candles=20,
    )
    assert ltfmock.call_count == 1
    assert ltfmock.call_args_list[0][1]['timerange'] != timerange
    assert ltfmock.call_args_list[0][1]['timerange'].startts == timerange.startts - 20 * 60


@pytest.mark.parametrize('candle_type', ['mark', ''])
def func_egsqkbl4(
    ohlcv_history: List[List[Union[int, float]]],
    mocker: pytest.MockFixture,
    caplog: pytest.LogCaptureFixture,
    default_conf: Dict[str, Any],
    tmp_path: Path,
    candle_type: str,
) -> None:
    """
    Test load_pair_history() with 1 min timeframe
    """
    exchange: MagicMock = get_patched_exchange(mocker, default_conf)
    mocker.patch.object(exchange, 'get_historic_ohlcv', return_value=ohlcv_history)
    file: Path = tmp_path / 'MEME_BTC-1m.feather'
    load_pair_history(datadir=tmp_path, timeframe='1m', pair='MEME/BTC', candle_type=candle_type)
    assert not file.is_file()
    assert log_has(
        f'No history for MEME/BTC, {candle_type}, 1m found. Use `freqtrade download-data` to download the data',
        caplog,
    )
    refresh_data(
        datadir=tmp_path,
        timeframe='1m',
        pairs=['MEME/BTC'],
        exchange=exchange,
        candle_type=CandleType.SPOT,
    )
    load_pair_history(datadir=tmp_path, timeframe='1m', pair='MEME/BTC', candle_type=candle_type)
    assert file.is_file()
    assert log_has_re(
        'Download history data for "MEME/BTC", 1m, spot and store in .*',
        caplog,
    )


def func_i304547y(testdatadir: Path) -> None:
    assert str(Path('tests') / 'testdata') in str(testdatadir)


@pytest.mark.parametrize('pair,timeframe,expected_result,candle_type', [
    ('ETH/BTC', '5m', 'freqtrade/hello/world/ETH_BTC-5m.json', ''),
    ('ETH/USDT', '1M', 'freqtrade/hello/world/ETH_USDT-1Mo.json', ''),
    ('Fabric Token/ETH', '5m', 'freqtrade/hello/world/Fabric_Token_ETH-5m.json', ''),
    ('ETHH20', '5m', 'freqtrade/hello/world/ETHH20-5m.json', ''),
    ('.XBTBON2H', '5m', 'freqtrade/hello/world/_XBTBON2H-5m.json', ''),
    ('ETHUSD.d', '5m', 'freqtrade/hello/world/ETHUSD_d-5m.json', ''),
    ('ACC_OLD/BTC', '5m', 'freqtrade/hello/world/ACC_OLD_BTC-5m.json', ''),
    ('ETH/BTC', '5m', 'freqtrade/hello/world/futures/ETH_BTC-5m-mark.json', 'mark'),
    ('ACC_OLD/BTC', '5m', 'freqtrade/hello/world/futures/ACC_OLD_BTC-5m-index.json', 'index'),
])
def func_7zomgmuy(
    pair: str,
    timeframe: str,
    expected_result: str,
    candle_type: str,
) -> None:
    fn: Path = JsonDataHandler._pair_data_filename(
        Path('freqtrade/hello/world'),
        pair,
        timeframe,
        CandleType.from_string(candle_type),
    )
    assert isinstance(fn, Path)
    assert fn == Path(expected_result)
    fn = JsonGzDataHandler._pair_data_filename(
        Path('freqtrade/hello/world'),
        pair,
        timeframe,
        candle_type=CandleType.from_string(candle_type),
    )
    assert isinstance(fn, Path)
    assert fn == Path(expected_result + '.gz')


@pytest.mark.parametrize('pair,trading_mode,expected_result', [
    ('ETH/BTC', '', 'freqtrade/hello/world/ETH_BTC-trades.json'),
    ('ETH/USDT:USDT', 'futures', 'freqtrade/hello/world/futures/ETH_USDT_USDT-trades.json'),
    ('Fabric Token/ETH', '', 'freqtrade/hello/world/Fabric_Token_ETH-trades.json'),
    ('ETHH20', '', 'freqtrade/hello/world/ETHH20-trades.json'),
    ('.XBTBON2H', '', 'freqtrade/hello/world/_XBTBON2H-trades.json'),
    ('ETHUSD.d', '', 'freqtrade/hello/world/ETHUSD_d-trades.json'),
    ('ACC_OLD_BTC', '', 'freqtrade/hello/world/ACC_OLD_BTC-trades.json'),
])
def func_2juoovk0(pair: str, trading_mode: str, expected_result: str) -> None:
    fn: Path = JsonDataHandler._pair_trades_filename(Path('freqtrade/hello/world'), pair, trading_mode)
    assert isinstance(fn, Path)
    assert fn == Path(expected_result)
    fn = JsonGzDataHandler._pair_trades_filename(Path('freqtrade/hello/world'), pair, trading_mode)
    assert isinstance(fn, Path)
    assert fn == Path(expected_result + '.gz')


def func_2qny32s1(testdatadir: Path) -> None:
    data_handler: JsonDataHandler = get_datahandler(testdatadir, 'json')
    test_data: Optional[List[List[Union[int, float]]] = None
    test_filename: Path = testdatadir.joinpath('UNITTEST_BTC-1m.json')
    with test_filename.open('rt') as file:
        test_data = json.load(file)
    test_data_df: DataFrame = ohlcv_to_dataframe(
        test_data,
        '1m',
        'UNITTEST/BTC',
        fill_missing=False,
        drop_incomplete=False,
    )
    now_ts: float = test_data[-1][0] / 1000 + 60 * 60
    timerange: TimeRange = TimeRange('date', None, test_data[0][0] / 1000 - 1, 0)
    data: DataFrame
    start_ts: Optional[int]
    end_ts: Optional[int]
    data, start_ts, end_ts = _load_cached_data_for_updating(
        'UNITTEST/BTC',
        '1m',
        timerange,
        data_handler,
        CandleType.SPOT,
    )
    assert not data.empty
    assert start_ts == test_data[-1][0] - 60 * 1000
    assert end_ts is None
    timerange = TimeRange('date', None, test_data[0][0] / 1000 - 1, 0)
    data, start_ts, end_ts = _load_cached_data_for_updating(
        'UNITTEST/BTC',
        '1m',
        timerange,
        data_handler,
        CandleType.SPOT,
        True,
    )
    assert_frame_equal(data, test_data_df.iloc[:-1])
    assert start_ts == test_data[0][0] - 1000
    assert end_ts == test_data[0][0]
    timerange = TimeRange('date', None, test_data[0][0] / 1000 + 1, 0)
    data, start_ts, end_ts = _load_cached_data_for_updating(
        'UNITTEST/BTC',
        '1m',
        timerange,
        data_handler,
        CandleType.SPOT,
    )
    assert_frame_equal(data, test_data_df.iloc[:-1])
    assert test_data[-2][0] <= start_ts < test_data[-1][0]
    assert end_ts is None
    timerange = TimeRange('date', None, test_data[-1][0] / 1000 + 100, 0)
    data, start_ts, end_ts = _load_cached_data_for_updating(
        'UNITTEST/BTC',
        '1m',
        timerange,
        data_handler,
        CandleType.SPOT,
    )
    assert_frame_equal(data, test_data_df.iloc[:-1])
    assert test_data[-2][0] <= start_ts < test_data[-1][0]
    assert end_ts is None
    timerange = TimeRange('date', None, now_ts - 10000, 0)
    data, start_ts, end_ts = _load_cached_data_for_updating(
        'NONEXIST/BTC',
        '1m',
        timerange,
        data_handler,
        CandleType.SPOT,
    )
    assert data.empty
    assert start_ts == (now_ts - 10000) * 1000
    assert end_ts is None
    timerange = TimeRange('date', 'date', now_ts - 1000000, now_ts - 100000)
    data, start_ts, end_ts = _load_cached_data_for_updating(
        'NONEXIST/BTC',
        '1m',
        timerange,
        data_handler,
        CandleType.SPOT,
    )
    assert data.empty
    assert start_ts == (now_ts - 1000000) * 1000
    assert end_ts == (now_ts - 100000) * 1000
    data, start_ts, end_ts = _load_cached_data_for_updating(
        'NONEXIST/BTC',
        '1m',
        None,
        data_handler,
        CandleType.SPOT,
    )
    assert data.empty
    assert start_ts is None
    assert end_ts is None


@pytest.mark.parametrize('candle_type,subdir,file_tail', [
    ('mark', 'futures/', '-mark'),
    ('spot', '', ''),
])
def func_8obqgcxu(
    ohlcv_history: List[List[Union[int, float]]],
    mocker: pytest.MockFixture,
    default_conf: Dict[str, Any],
    tmp_path: Path,
    candle_type: str,
    subdir: str,
    file_tail: str,
) -> None:
    exchange: MagicMock = get_patched_exchange(mocker, default_conf)
    mocker.patch.object(exchange, 'get_historic_ohlcv', return_value=ohlcv_history)
    file1_1: Path = tmp_path / f'{subdir}MEME_BTC-1m{file_tail}.feather'
    file1_5: Path = tmp_path / f'{subdir}MEME_BTC-5m{file_tail}.feather'
    file2_1: Path = tmp_path / f'{subdir}CFI_BTC-1m{file_tail}.feather'
    file2_5: Path = tmp_path / f'{subdir}CFI_BTC-5m{file_tail}.feather'
    assert not file1_1.is_file()
    assert not file2_1.is_file()
    assert _download_pair_history(
        datadir=tmp_path,
        exchange=exchange,
        pair='MEME/BTC',
        timeframe='1m',
        candle_type=candle_type,
    )
    assert _download_pair_history(
        datadir=tmp_path,
        exchange=exchange,
        pair='CFI/BTC',
        timeframe='1m',
        candle_type=candle_type,
    )
    assert not exchange._pairs_last_refresh_time
    assert file1_1.is_file()
    assert file2_1.is_file()
    func_ynl5mu59(file1_1)
    func_ynl5mu59(file2_1)
    assert not file1_5.is_file()
    assert not file2_5.is_file()
    assert _download_pair_history(
        datadir=tmp_path,
        exchange=exchange,
        pair='MEME/BTC',
        timeframe='5m',
        candle_type=candle_type,
    )
    assert _download_pair_history(
        datadir=tmp_path,
        exchange=exchange,
        pair='CFI/BTC',
        timeframe='5m',
        candle_type=candle_type,
    )
    assert not exchange._pairs_last_refresh_time
    assert file1_5.is_file()
    assert file2_5.is_file()


def func_axfngv96(
    mocker: pytest.MockFixture,
    default_conf: Dict[str, Any],
    testdatadir: Path,
    ohlcv_history: List[List[Union[int, float]]],
) -> None:
    json_dump_mock: MagicMock = mocker.patch(
        'freqtrade.data.history.datahandlers.featherdatahandler.FeatherDataHandler.ohlcv_store',
        return_value=None,
    )
    exchange: MagicMock = get_patched_exchange(mocker, default_conf)
    mocker.patch.object(exchange, 'get_historic_ohlcv', return_value=ohlcv_history)
    _download_pair_history(
        datadir=testdatadir,
        exchange=exchange