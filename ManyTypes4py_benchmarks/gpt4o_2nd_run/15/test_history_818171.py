import json
import logging
import uuid
from datetime import timedelta
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple
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
from tests.conftest import (
    CURRENT_TEST_STRATEGY,
    EXMS,
    get_patched_exchange,
    log_has,
    log_has_re,
    patch_exchange,
)


def _clean_test_file(file: Path) -> None:
    """
    Backup existing file to avoid deleting the user file
    :param file: complete path to the file
    :return: None
    """
    file_swp = Path(str(file) + '.swp')
    if file.is_file():
        file.unlink()
    if file_swp.is_file():
        file_swp.rename(file)


def test_load_data_30min_timeframe(caplog: Any, testdatadir: Path) -> None:
    ld = load_pair_history(pair='UNITTEST/BTC', timeframe='30m', datadir=testdatadir)
    assert isinstance(ld, DataFrame)
    assert not log_has('Download history data for pair: "UNITTEST/BTC", timeframe: 30m and store in None.', caplog)


def test_load_data_7min_timeframe(caplog: Any, testdatadir: Path) -> None:
    ld = load_pair_history(pair='UNITTEST/BTC', timeframe='7m', datadir=testdatadir)
    assert isinstance(ld, DataFrame)
    assert ld.empty
    assert log_has('No history for UNITTEST/BTC, spot, 7m found. Use `freqtrade download-data` to download the data', caplog)


def test_load_data_1min_timeframe(ohlcv_history: List[List[Any]], mocker: Any, caplog: Any, testdatadir: Path) -> None:
    mocker.patch(f'{EXMS}.get_historic_ohlcv', return_value=ohlcv_history)
    file = testdatadir / 'UNITTEST_BTC-1m.feather'
    load_data(datadir=testdatadir, timeframe='1m', pairs=['UNITTEST/BTC'])
    assert file.is_file()
    assert not log_has('Download history data for pair: "UNITTEST/BTC", interval: 1m and store in None.', caplog)


def test_load_data_mark(ohlcv_history: List[List[Any]], mocker: Any, caplog: Any, testdatadir: Path) -> None:
    mocker.patch(f'{EXMS}.get_historic_ohlcv', return_value=ohlcv_history)
    file = testdatadir / 'futures/UNITTEST_USDT_USDT-1h-mark.feather'
    load_data(datadir=testdatadir, timeframe='1h', pairs=['UNITTEST/BTC'], candle_type='mark')
    assert file.is_file()
    assert not log_has('Download history data for pair: "UNITTEST/USDT:USDT", interval: 1m and store in None.', caplog)


def test_load_data_startup_candles(mocker: Any, testdatadir: Path) -> None:
    ltfmock = mocker.patch('freqtrade.data.history.datahandlers.featherdatahandler.FeatherDataHandler._ohlcv_load', MagicMock(return_value=DataFrame()))
    timerange = TimeRange('date', None, 1510639620, 0)
    load_pair_history(pair='UNITTEST/BTC', timeframe='1m', datadir=testdatadir, timerange=timerange, startup_candles=20)
    assert ltfmock.call_count == 1
    assert ltfmock.call_args_list[0][1]['timerange'] != timerange
    assert ltfmock.call_args_list[0][1]['timerange'].startts == timerange.startts - 20 * 60


@pytest.mark.parametrize('candle_type', ['mark', ''])
def test_load_data_with_new_pair_1min(ohlcv_history: List[List[Any]], mocker: Any, caplog: Any, default_conf: Dict[str, Any], tmp_path: Path, candle_type: str) -> None:
    """
    Test load_pair_history() with 1 min timeframe
    """
    exchange = get_patched_exchange(mocker, default_conf)
    mocker.patch.object(exchange, 'get_historic_ohlcv', return_value=ohlcv_history)
    file = tmp_path / 'MEME_BTC-1m.feather'
    load_pair_history(datadir=tmp_path, timeframe='1m', pair='MEME/BTC', candle_type=candle_type)
    assert not file.is_file()
    assert log_has(f'No history for MEME/BTC, {candle_type}, 1m found. Use `freqtrade download-data` to download the data', caplog)
    refresh_data(datadir=tmp_path, timeframe='1m', pairs=['MEME/BTC'], exchange=exchange, candle_type=CandleType.SPOT)
    load_pair_history(datadir=tmp_path, timeframe='1m', pair='MEME/BTC', candle_type=candle_type)
    assert file.is_file()
    assert log_has_re('Download history data for "MEME/BTC", 1m, spot and store in .*', caplog)


def test_testdata_path(testdatadir: Path) -> None:
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
    ('ACC_OLD/BTC', '5m', 'freqtrade/hello/world/futures/ACC_OLD_BTC-5m-index.json', 'index')
])
def test_json_pair_data_filename(pair: str, timeframe: str, expected_result: str, candle_type: str) -> None:
    fn = JsonDataHandler._pair_data_filename(Path('freqtrade/hello/world'), pair, timeframe, CandleType.from_string(candle_type))
    assert isinstance(fn, Path)
    assert fn == Path(expected_result)
    fn = JsonGzDataHandler._pair_data_filename(Path('freqtrade/hello/world'), pair, timeframe, candle_type=CandleType.from_string(candle_type))
    assert isinstance(fn, Path)
    assert fn == Path(expected_result + '.gz')


@pytest.mark.parametrize('pair,trading_mode,expected_result', [
    ('ETH/BTC', '', 'freqtrade/hello/world/ETH_BTC-trades.json'),
    ('ETH/USDT:USDT', 'futures', 'freqtrade/hello/world/futures/ETH_USDT_USDT-trades.json'),
    ('Fabric Token/ETH', '', 'freqtrade/hello/world/Fabric_Token_ETH-trades.json'),
    ('ETHH20', '', 'freqtrade/hello/world/ETHH20-trades.json'),
    ('.XBTBON2H', '', 'freqtrade/hello/world/_XBTBON2H-trades.json'),
    ('ETHUSD.d', '', 'freqtrade/hello/world/ETHUSD_d-trades.json'),
    ('ACC_OLD_BTC', '', 'freqtrade/hello/world/ACC_OLD_BTC-trades.json')
])
def test_json_pair_trades_filename(pair: str, trading_mode: str, expected_result: str) -> None:
    fn = JsonDataHandler._pair_trades_filename(Path('freqtrade/hello/world'), pair, trading_mode)
    assert isinstance(fn, Path)
    assert fn == Path(expected_result)
    fn = JsonGzDataHandler._pair_trades_filename(Path('freqtrade/hello/world'), pair, trading_mode)
    assert isinstance(fn, Path)
    assert fn == Path(expected_result + '.gz')


def test_load_cached_data_for_updating(testdatadir: Path) -> None:
    data_handler = get_datahandler(testdatadir, 'json')
    test_data: Optional[List[List[Any]]] = None
    test_filename = testdatadir.joinpath('UNITTEST_BTC-1m.json')
    with test_filename.open('rt') as file:
        test_data = json.load(file)
    test_data_df = ohlcv_to_dataframe(test_data, '1m', 'UNITTEST/BTC', fill_missing=False, drop_incomplete=False)
    now_ts = test_data[-1][0] / 1000 + 60 * 60
    timerange = TimeRange('date', None, test_data[0][0] / 1000 - 1, 0)
    data, start_ts, end_ts = _load_cached_data_for_updating('UNITTEST/BTC', '1m', timerange, data_handler, CandleType.SPOT)
    assert not data.empty
    assert start_ts == test_data[-1][0] - 60 * 1000
    assert end_ts is None
    timerange = TimeRange('date', None, test_data[0][0] / 1000 - 1, 0)
    data, start_ts, end_ts = _load_cached_data_for_updating('UNITTEST/BTC', '1m', timerange, data_handler, CandleType.SPOT, True)
    assert_frame_equal(data, test_data_df.iloc[:-1])
    assert start_ts == test_data[0][0] - 1000
    assert end_ts == test_data[0][0]
    timerange = TimeRange('date', None, test_data[0][0] / 1000 + 1, 0)
    data, start_ts, end_ts = _load_cached_data_for_updating('UNITTEST/BTC', '1m', timerange, data_handler, CandleType.SPOT)
    assert_frame_equal(data, test_data_df.iloc[:-1])
    assert test_data[-2][0] <= start_ts < test_data[-1][0]
    assert end_ts is None
    timerange = TimeRange('date', None, test_data[-1][0] / 1000 + 100, 0)
    data, start_ts, end_ts = _load_cached_data_for_updating('UNITTEST/BTC', '1m', timerange, data_handler, CandleType.SPOT)
    assert_frame_equal(data, test_data_df.iloc[:-1])
    assert test_data[-2][0] <= start_ts < test_data[-1][0]
    assert end_ts is None
    timerange = TimeRange('date', None, now_ts - 10000, 0)
    data, start_ts, end_ts = _load_cached_data_for_updating('NONEXIST/BTC', '1m', timerange, data_handler, CandleType.SPOT)
    assert data.empty
    assert start_ts == (now_ts - 10000) * 1000
    assert end_ts is None
    timerange = TimeRange('date', 'date', now_ts - 1000000, now_ts - 100000)
    data, start_ts, end_ts = _load_cached_data_for_updating('NONEXIST/BTC', '1m', timerange, data_handler, CandleType.SPOT)
    assert data.empty
    assert start_ts == (now_ts - 1000000) * 1000
    assert end_ts == (now_ts - 100000) * 1000
    data, start_ts, end_ts = _load_cached_data_for_updating('NONEXIST/BTC', '1m', None, data_handler, CandleType.SPOT)
    assert data.empty
    assert start_ts is None
    assert end_ts is None


@pytest.mark.parametrize('candle_type,subdir,file_tail', [('mark', 'futures/', '-mark'), ('spot', '', '')])
def test_download_pair_history(ohlcv_history: List[List[Any]], mocker: Any, default_conf: Dict[str, Any], tmp_path: Path, candle_type: str, subdir: str, file_tail: str) -> None:
    exchange = get_patched_exchange(mocker, default_conf)
    mocker.patch.object(exchange, 'get_historic_ohlcv', return_value=ohlcv_history)
    file1_1 = tmp_path / f'{subdir}MEME_BTC-1m{file_tail}.feather'
    file1_5 = tmp_path / f'{subdir}MEME_BTC-5m{file_tail}.feather'
    file2_1 = tmp_path / f'{subdir}CFI_BTC-1m{file_tail}.feather'
    file2_5 = tmp_path / f'{subdir}CFI_BTC-5m{file_tail}.feather'
    assert not file1_1.is_file()
    assert not file2_1.is_file()
    assert _download_pair_history(datadir=tmp_path, exchange=exchange, pair='MEME/BTC', timeframe='1m', candle_type=candle_type)
    assert _download_pair_history(datadir=tmp_path, exchange=exchange, pair='CFI/BTC', timeframe='1m', candle_type=candle_type)
    assert not exchange._pairs_last_refresh_time
    assert file1_1.is_file()
    assert file2_1.is_file()
    _clean_test_file(file1_1)
    _clean_test_file(file2_1)
    assert not file1_5.is_file()
    assert not file2_5.is_file()
    assert _download_pair_history(datadir=tmp_path, exchange=exchange, pair='MEME/BTC', timeframe='5m', candle_type=candle_type)
    assert _download_pair_history(datadir=tmp_path, exchange=exchange, pair='CFI/BTC', timeframe='5m', candle_type=candle_type)
    assert not exchange._pairs_last_refresh_time
    assert file1_5.is_file()
    assert file2_5.is_file()


def test_download_pair_history2(mocker: Any, default_conf: Dict[str, Any], testdatadir: Path, ohlcv_history: List[List[Any]]) -> None:
    json_dump_mock = mocker.patch('freqtrade.data.history.datahandlers.featherdatahandler.FeatherDataHandler.ohlcv_store', return_value=None)
    exchange = get_patched_exchange(mocker, default_conf)
    mocker.patch.object(exchange, 'get_historic_ohlcv', return_value=ohlcv_history)
    _download_pair_history(datadir=testdatadir, exchange=exchange, pair='UNITTEST/BTC', timeframe='1m', candle_type='spot')
    _download_pair_history(datadir=testdatadir, exchange=exchange, pair='UNITTEST/BTC', timeframe='3m', candle_type='spot')
    _download_pair_history(datadir=testdatadir, exchange=exchange, pair='UNITTEST/USDT', timeframe='1h', candle_type='mark')
    assert json_dump_mock.call_count == 3


def test_download_backtesting_data_exception(mocker: Any, caplog: Any, default_conf: Dict[str, Any], tmp_path: Path) -> None:
    mocker.patch(f'{EXMS}.get_historic_ohlcv', side_effect=Exception('File Error'))
    exchange = get_patched_exchange(mocker, default_conf)
    assert not _download_pair_history(datadir=tmp_path, exchange=exchange, pair='MEME/BTC', timeframe='1m', candle_type='spot')
    assert log_has('Failed to download history data for pair: "MEME/BTC", timeframe: 1m.', caplog)


def test_load_partial_missing(testdatadir: Path, caplog: Any) -> None:
    start = dt_utc(2018, 1, 1)
    end = dt_utc(2018, 1, 11)
    data = load_data(testdatadir, '5m', ['UNITTEST/BTC'], startup_candles=20, timerange=TimeRange('date', 'date', start.timestamp(), end.timestamp()))
    assert log_has('Using indicator startup period: 20 ...', caplog)
    td = (end - start).total_seconds() // 60 // 5 + 1
    assert td != len(data['UNITTEST/BTC'])
    start_real = data['UNITTEST/BTC'].iloc[0, 0]
    assert log_has(f'UNITTEST/BTC, spot, 5m, data starts at {start_real.strftime(DATETIME_PRINT_FORMAT)}', caplog)
    caplog.clear()
    start = dt_utc(2018, 1, 10)
    end = dt_utc(2018, 2, 20)
    data = load_data(datadir=testdatadir, timeframe='5m', pairs=['UNITTEST/BTC'], timerange=TimeRange('date', 'date', start.timestamp(), end.timestamp()))
    td = (end - start).total_seconds() // 60 // 5 + 1
    assert td != len(data['UNITTEST/BTC'])
    end_real = data['UNITTEST/BTC'].iloc[-1, 0].to_pydatetime()
    assert log_has(f'UNITTEST/BTC, spot, 5m, data ends at {end_real.strftime(DATETIME_PRINT_FORMAT)}', caplog)


def test_init(default_conf: Dict[str, Any]) -> None:
    assert {} == load_data(datadir=Path(), pairs=[], timeframe=default_conf['timeframe'])


def test_init_with_refresh(default_conf: Dict[str, Any], mocker: Any) -> None:
    exchange = get_patched_exchange(mocker, default_conf)
    refresh_data(datadir=Path(), pairs=[], timeframe=default_conf['timeframe'], exchange=exchange, candle_type=CandleType.SPOT)
    assert {} == load_data(datadir=Path(), pairs=[], timeframe=default_conf['timeframe'])


def test_file_dump_json_tofile(testdatadir: Path) -> None:
    file = testdatadir / f'test_{uuid.uuid4()}.json'
    data = {'bar': 'foo'}
    assert not file.is_file()
    file_dump_json(file, data)
    assert file.is_file()
    with file.open() as data_file:
        json_from_file = json.load(data_file)
    assert 'bar' in json_from_file
    assert json_from_file['bar'] == 'foo'
    _clean_test_file(file)


def test_get_timerange(default_conf: Dict[str, Any], mocker: Any, testdatadir: Path) -> None:
    patch_exchange(mocker)
    default_conf.update({'strategy': CURRENT_TEST_STRATEGY})
    strategy = StrategyResolver.load_strategy(default_conf)
    data = strategy.advise_all_indicators(load_data(datadir=testdatadir, timeframe='1m', pairs=['UNITTEST/BTC']))
    min_date, max_date = get_timerange(data)
    assert min_date.isoformat() == '2017-11-04T23:02:00+00:00'
    assert max_date.isoformat() == '2017-11-14T22:59:00+00:00'


def test_validate_backtest_data_warn(default_conf: Dict[str, Any], mocker: Any, caplog: Any, testdatadir: Path) -> None:
    patch_exchange(mocker)
    default_conf.update({'strategy': CURRENT_TEST_STRATEGY})
    strategy = StrategyResolver.load_strategy(default_conf)
    data = strategy.advise_all_indicators(load_data(datadir=testdatadir, timeframe='1m', pairs=['UNITTEST/BTC'], fill_up_missing=False))
    min_date, max_date = get_timerange(data)
    caplog.clear()
    assert validate_backtest_data(data['UNITTEST/BTC'], 'UNITTEST/BTC', min_date, max_date, timeframe_to_minutes('1m'))
    assert len(caplog.record_tuples) == 1
    assert log_has("UNITTEST/BTC has missing frames: expected 14397, got 13681, that's 716 missing values", caplog)


def test_validate_backtest_data(default_conf: Dict[str, Any], mocker: Any, caplog: Any, testdatadir: Path) -> None:
    patch_exchange(mocker)
    default_conf.update({'strategy': CURRENT_TEST_STRATEGY})
    strategy = StrategyResolver.load_strategy(default_conf)
    timerange = TimeRange()
    data = strategy.advise_all_indicators(load_data(datadir=testdatadir, timeframe='5m', pairs=['UNITTEST/BTC'], timerange=timerange))
    min_date, max_date = get_timerange(data)
    caplog.clear()
    assert not validate_backtest_data(data['UNITTEST/BTC'], 'UNITTEST/BTC', min_date, max_date, timeframe_to_minutes('5m'))
    assert len(caplog.record_tuples) == 0


@pytest.mark.parametrize('trademode,callcount', [('spot', 4), ('margin', 4), ('futures', 8)])
def test_refresh_backtest_ohlcv_data(mocker: Any, default_conf: Dict[str, Any], markets: Dict[str, Any], caplog: Any, testdatadir: Path, trademode: str, callcount: int) -> None:
    caplog.set_level(logging.DEBUG)
    dl_mock = mocker.patch('freqtrade.data.history.history_utils._download_pair_history')
    mocker.patch(f'{EXMS}.markets', PropertyMock(return_value=markets))
    mocker.patch.object(Path, 'exists', MagicMock(return_value=True))
    mocker.patch.object(Path, 'unlink', MagicMock())
    default_conf['trading_mode'] = trademode
    ex = get_patched_exchange(mocker, default_conf, exchange='bybit')
    timerange = TimeRange.parse_timerange('20190101-20190102')
    refresh_backtest_ohlcv_data(exchange=ex, pairs=['ETH/BTC', 'XRP/BTC'], timeframes=['1m', '5m'], datadir=testdatadir, timerange=timerange, erase=True, trading_mode=trademode)
    assert dl_mock.call_count == callcount
    assert dl_mock.call_args[1]['timerange'].starttype == 'date'
    assert log_has_re('Downloading pair ETH/BTC, .* interval 1m\\.', caplog)
    if trademode == 'futures':
        assert log_has_re('Downloading pair ETH/BTC, funding_rate, interval 8h\\.', caplog)
        assert log_has_re('Downloading pair ETH/BTC, mark, interval 4h\\.', caplog)


def test_download_data_no_markets(mocker: Any, default_conf: Dict[str, Any], caplog: Any, testdatadir: Path) -> None:
    dl_mock = mocker.patch('freqtrade.data.history.history_utils._download_pair_history', MagicMock())
    ex = get_patched_exchange(mocker, default_conf)
    mocker.patch(f'{EXMS}.markets', PropertyMock(return_value={}))
    timerange = TimeRange.parse_timerange('20190101-20190102')
    unav_pairs = refresh_backtest_ohlcv_data(exchange=ex, pairs=['BTT/BTC', 'LTC/USDT'], timeframes=['1m', '5m'], datadir=testdatadir, timerange=timerange, erase=False, trading_mode='spot')
    assert dl_mock.call_count == 0
    assert 'BTT/BTC: Pair not available on exchange.' in unav_pairs
    assert 'LTC/USDT: Pair not available on exchange.' in unav_pairs
    assert log_has('Skipping pair BTT/BTC...', caplog)


def test_refresh_backtest_trades_data(mocker: Any, default_conf: Dict[str, Any], markets: Dict[str, Any], caplog: Any, testdatadir: Path) -> None:
    dl_mock = mocker.patch('freqtrade.data.history.history_utils._download_trades_history', MagicMock())
    mocker.patch(f'{EXMS}.markets', PropertyMock(return_value=markets))
    mocker.patch.object(Path, 'exists', MagicMock(return_value=True))
    mocker.patch.object(Path, 'unlink', MagicMock())
    ex = get_patched_exchange(mocker, default_conf)
    timerange = TimeRange.parse_timerange('20190101-20190102')
    unavailable_pairs = refresh_backtest_trades_data(exchange=ex, pairs=['ETH/BTC', 'XRP/BTC', 'XRP/ETH'], datadir=testdatadir, timerange=timerange, erase=True, trading_mode=TradingMode.SPOT)
    assert dl_mock.call_count == 2
    assert dl_mock.call_args[1]['timerange'].starttype == 'date'
    assert log_has('Downloading trades for pair ETH/BTC.', caplog)
    assert [p for p in unavailable_pairs if 'XRP/ETH' in p]
    assert log_has('Skipping pair XRP/ETH...', caplog)


def test_download_trades_history(trades_history: List[List[Any]], mocker: Any, default_conf: Dict[str, Any], testdatadir: Path, caplog: Any, tmp_path: Path, time_machine: Any) -> None:
    start_dt = dt_utc(2023, 1, 1)
    time_machine.move_to(start_dt, tick=False)
    ght_mock = MagicMock(side_effect=lambda pair, *args, **kwargs: (pair, trades_history))
    mocker.patch(f'{EXMS}.get_historic_trades', ght_mock)
    exchange = get_patched_exchange(mocker, default_conf)
    file1 = tmp_path / 'ETH_BTC-trades.json.gz'
    data_handler = get_datahandler(tmp_path, data_format='jsongz')
    assert not file1.is_file()
    assert _download_trades_history(data_handler=data_handler, exchange=exchange, pair='ETH/BTC', trading_mode=TradingMode.SPOT)
    assert log_has('Current Amount of trades: 0', caplog)
    assert log_has('New Amount of trades: 6', caplog)
    assert ght_mock.call_count == 1
    assert ght_mock.call_args_list[0][1]['since'] == dt_ts(start_dt - timedelta(days=30))
    assert file1.is_file()
    caplog.clear()
    ght_mock.reset_mock()
    since_time = int(trades_history[-3][0] // 1000)
    since_time2 = int(trades_history[-1][0] // 1000)
    timerange = TimeRange('date', None, since_time, 0)
    assert _download_trades_history(data_handler=data_handler, exchange=exchange, pair='ETH/BTC', timerange=timerange, trading_mode=TradingMode.SPOT)
    assert ght_mock.call_count == 1
    assert int(ght_mock.call_args_list[0][1]['since'] // 1000) == since_time2 - 5
    assert ght_mock.call_args_list[0][1]['from_id'] is not None
    file1.unlink()
    mocker.patch(f'{EXMS}.get_historic_trades', MagicMock(side_effect=ValueError('he ho!')))
    caplog.clear()
    with pytest.raises(ValueError, match='he ho!'):
        _download_trades_history(data_handler=data_handler, exchange=exchange, pair='ETH/BTC', trading_mode=TradingMode.SPOT)
    file2 = tmp_path / 'XRP_ETH-trades.json.gz'
    copyfile(testdatadir / file2.name, file2)
    ght_mock.reset_mock()
    mocker.patch(f'{EXMS}.get_historic_trades', ght_mock)
    since_time = int(trades_history[0][0] // 1000) - 500
    timerange = TimeRange('date', None, since_time, 0)
    with pytest.raises(ValueError, match='Start .* earlier than available data'):
        _download_trades_history(data_handler=data_handler, exchange=exchange, pair='XRP/ETH', timerange=timerange, trading_mode=TradingMode.SPOT)
    assert ght_mock.call_count == 0
    _clean_test_file(file2)
