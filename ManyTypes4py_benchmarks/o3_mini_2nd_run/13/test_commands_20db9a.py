#!/usr/bin/env python3
from datetime import datetime, timedelta
import json
import re
import shutil
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
from typing import Any, List, Tuple

import pytest
from pytest_mock import MockerFixture
from _pytest.capture import CaptureFixture
from _pytest.logging import LogCaptureFixture

from freqtrade.commands import (
    start_backtesting_show,
    start_convert_data,
    start_convert_db,
    start_convert_trades,
    start_create_userdir,
    start_download_data,
    start_hyperopt_list,
    start_hyperopt_show,
    start_install_ui,
    start_list_data,
    start_list_exchanges,
    start_list_freqAI_models,
    start_list_hyperopt_loss_functions,
    start_list_markets,
    start_list_timeframes,
    start_new_strategy,
    start_show_config,
    start_show_trades,
    start_strategy_update,
    start_test_pairlist,
    start_trading,
    start_webserver,
)
from freqtrade.commands.deploy_ui import clean_ui_subdir, download_and_install_ui, get_ui_download_url, read_ui_version
from freqtrade.configuration import setup_utils_configuration
from freqtrade.enums import RunMode
from freqtrade.exceptions import OperationalException
from freqtrade.persistence.models import init_db
from freqtrade.persistence.pairlock_middleware import PairLocks
from freqtrade.util import dt_utc
from tests.conftest import (
    CURRENT_TEST_STRATEGY,
    EXMS,
    create_mock_trades,
    get_args,
    log_has,
    log_has_re,
    patch_exchange,
    patched_configuration_load_config_file,
)
from tests.conftest_hyperopt import hyperopt_test_result
from tests.conftest_trades import MOCK_TRADE_COUNT


def test_setup_utils_configuration() -> None:
    args: List[str] = ['list-exchanges', '--config', 'tests/testdata/testconfigs/main_test_config.json']
    config = setup_utils_configuration(get_args(args), RunMode.OTHER)
    assert 'exchange' in config
    assert config['dry_run'] is True
    args = ['list-exchanges', '--config', 'tests/testdata/testconfigs/testconfig.json']
    config = setup_utils_configuration(get_args(args), RunMode.OTHER, set_dry=False)
    assert 'exchange' in config
    assert config['dry_run'] is False


def test_start_trading_fail(mocker: MockerFixture, caplog: LogCaptureFixture) -> None:
    mocker.patch('freqtrade.worker.Worker.run', return_value=None, side_effect=OperationalException)
    mocker.patch('freqtrade.worker.Worker.__init__', return_value=None)
    exitmock = mocker.patch('freqtrade.worker.Worker.exit')
    args: List[str] = ['trade', '-c', 'tests/testdata/testconfigs/main_test_config.json']
    with pytest.raises(OperationalException):
        start_trading(get_args(args))
    assert exitmock.call_count == 1
    exitmock.reset_mock()
    caplog.clear()
    mocker.patch('freqtrade.worker.Worker.__init__', side_effect=OperationalException)
    with pytest.raises(OperationalException):
        start_trading(get_args(args))
    assert exitmock.call_count == 0


def test_start_webserver(mocker: MockerFixture, caplog: LogCaptureFixture) -> None:
    api_server_mock = mocker.patch('freqtrade.rpc.api_server.ApiServer')
    args: List[str] = ['webserver', '-c', 'tests/testdata/testconfigs/main_test_config.json']
    start_webserver(get_args(args))
    assert api_server_mock.call_count == 1


def test_list_exchanges(capsys: CaptureFixture) -> None:
    args: List[str] = ['list-exchanges']
    start_list_exchanges(get_args(args))
    captured = capsys.readouterr()
    assert re.search('.*Exchanges available for Freqtrade.*', captured.out)
    assert re.search('.*binance.*', captured.out)
    assert re.search('.*bybit.*', captured.out)
    args = ['list-exchanges', '--one-column']
    start_list_exchanges(get_args(args))
    captured = capsys.readouterr()
    assert re.search('^binance$', captured.out, re.MULTILINE)
    assert re.search('^bybit$', captured.out, re.MULTILINE)
    args = ['list-exchanges', '--all']
    start_list_exchanges(get_args(args))
    captured = capsys.readouterr()
    assert re.search('All exchanges supported by the ccxt library.*', captured.out)
    assert re.search('.*binance.*', captured.out)
    assert re.search('.*bingx.*', captured.out)
    assert re.search('.*bitmex.*', captured.out)
    args = ['list-exchanges', '--one-column', '--all']
    start_list_exchanges(get_args(args))
    captured = capsys.readouterr()
    assert re.search('^binance$', captured.out, re.MULTILINE)
    assert re.search('^bingx$', captured.out, re.MULTILINE)
    assert re.search('^bitmex$', captured.out, re.MULTILINE)


def test_list_timeframes(mocker: MockerFixture, capsys: CaptureFixture[str]) -> None:
    api_mock = mocker.MagicMock()
    api_mock.timeframes = {'1m': 'oneMin', '5m': 'fiveMin', '30m': 'thirtyMin', '1h': 'hour', '1d': 'day'}
    patch_exchange(mocker, api_mock=api_mock, exchange='bybit')
    args: List[str] = ['list-timeframes']
    pargs = get_args(args)
    pargs['config'] = None
    with pytest.raises(OperationalException, match='This command requires a configured exchange.*'):
        start_list_timeframes(pargs)
    args = ['list-timeframes', '--config', 'tests/testdata/testconfigs/main_test_config.json']
    start_list_timeframes(get_args(args))
    captured = capsys.readouterr()
    assert re.match('Timeframes available for the exchange `Bybit`: 1m, 5m, 30m, 1h, 1d', captured.out)
    args = ['list-timeframes', '--exchange', 'bybit']
    start_list_timeframes(get_args(args))
    captured = capsys.readouterr()
    assert re.match('Timeframes available for the exchange `Bybit`: 1m, 5m, 30m, 1h, 1d', captured.out)
    api_mock.timeframes = {'1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m', '1h': '1h', '6h': '6h', '12h': '12h', '1d': '1d', '3d': '3d'}
    patch_exchange(mocker, api_mock=api_mock, exchange='binance')
    args = ['list-timeframes', '--exchange', 'binance']
    start_list_timeframes(get_args(args))
    captured = capsys.readouterr()
    assert re.match('Timeframes available for the exchange `Binance`: 1m, 5m, 15m, 30m, 1h, 6h, 12h, 1d, 3d', captured.out)
    args = ['list-timeframes', '--config', 'tests/testdata/testconfigs/main_test_config.json', '--one-column']
    start_list_timeframes(get_args(args))
    captured = capsys.readouterr()
    assert re.search('^1m$', captured.out, re.MULTILINE)
    assert re.search('^5m$', captured.out, re.MULTILINE)
    assert re.search('^1h$', captured.out, re.MULTILINE)
    assert re.search('^1d$', captured.out, re.MULTILINE)
    args = ['list-timeframes', '--exchange', 'binance', '--one-column']
    start_list_timeframes(get_args(args))
    captured = capsys.readouterr()
    assert re.search('^1m$', captured.out, re.MULTILINE)
    assert re.search('^5m$', captured.out, re.MULTILINE)
    assert re.search('^1h$', captured.out, re.MULTILINE)
    assert re.search('^1d$', captured.out, re.MULTILINE)


def test_list_markets(mocker: MockerFixture, markets_static: Any, capsys: CaptureFixture[str]) -> None:
    api_mock = mocker.MagicMock()
    patch_exchange(mocker, api_mock=api_mock, exchange='binance', mock_markets=markets_static)
    args: List[str] = ['list-markets']
    pargs = get_args(args)
    pargs['config'] = None
    with pytest.raises(OperationalException, match='This command requires a configured exchange.*'):
        start_list_markets(pargs, False)
    args = ['list-markets', '--config', 'tests/testdata/testconfigs/main_test_config.json', '--print-list']
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert 'Exchange Binance has 12 active markets: ADA/USDT:USDT, BLK/BTC, ETH/BTC, ETH/USDT, ETH/USDT:USDT, LTC/BTC, LTC/ETH, LTC/USD, NEO/BTC, TKN/BTC, XLTCUSDT, XRP/BTC.\n' in captured.out
    patch_exchange(mocker, api_mock=api_mock, exchange='binance', mock_markets=markets_static)
    args = ['list-markets', '--exchange', 'binance']
    pargs = get_args(args)
    pargs['config'] = None
    start_list_markets(pargs, False)
    captured = capsys.readouterr()
    assert re.search('.*Exchange Binance has 12 active markets.*', captured.out)
    patch_exchange(mocker, api_mock=api_mock, exchange='binance', mock_markets=markets_static)
    args = ['list-markets', '--all', '--config', 'tests/testdata/testconfigs/main_test_config.json', '--print-list']
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert 'Exchange Binance has 14 markets: ADA/USDT:USDT, BLK/BTC, BTT/BTC, ETH/BTC, ETH/USDT, ETH/USDT:USDT, LTC/BTC, LTC/ETH, LTC/USD, LTC/USDT, NEO/BTC, TKN/BTC, XLTCUSDT, XRP/BTC.\n' in captured.out
    args = ['list-pairs', '--config', 'tests/testdata/testconfigs/main_test_config.json', '--print-list']
    start_list_markets(get_args(args), True)
    captured = capsys.readouterr()
    assert 'Exchange Binance has 9 active pairs: BLK/BTC, ETH/BTC, ETH/USDT, LTC/BTC, LTC/ETH, LTC/USD, NEO/BTC, TKN/BTC, XRP/BTC.\n' in captured.out
    args = ['list-pairs', '--all', '--config', 'tests/testdata/testconfigs/main_test_config.json', '--print-list']
    start_list_markets(get_args(args), True)
    captured = capsys.readouterr()
    assert 'Exchange Binance has 11 pairs: BLK/BTC, BTT/BTC, ETH/BTC, ETH/USDT, LTC/BTC, LTC/ETH, LTC/USD, LTC/USDT, NEO/BTC, TKN/BTC, XRP/BTC.\n' in captured.out
    args = ['list-markets', '--config', 'tests/testdata/testconfigs/main_test_config.json', '--base', 'ETH', 'LTC', '--print-list']
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert 'Exchange Binance has 7 active markets with ETH, LTC as base currencies: ETH/BTC, ETH/USDT, ETH/USDT:USDT, LTC/BTC, LTC/ETH, LTC/USD, XLTCUSDT.\n' in captured.out
    args = ['list-markets', '--config', 'tests/testdata/testconfigs/main_test_config.json', '--base', 'LTC', '--print-list']
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert 'Exchange Binance has 4 active markets with LTC as base currency: LTC/BTC, LTC/ETH, LTC/USD, XLTCUSDT.\n' in captured.out
    args = ['list-markets', '--config', 'tests/testdata/testconfigs/main_test_config.json', '--quote', 'USDT', 'USD', '--print-list']
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert 'Exchange Binance has 5 active markets with USDT, USD as quote currencies: ADA/USDT:USDT, ETH/USDT, ETH/USDT:USDT, LTC/USD, XLTCUSDT.\n' in captured.out
    args = ['list-markets', '--config', 'tests/testdata/testconfigs/main_test_config.json', '--quote', 'USDT', '--print-list']
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert 'Exchange Binance has 4 active markets with USDT as quote currency: ADA/USDT:USDT, ETH/USDT, ETH/USDT:USDT, XLTCUSDT.\n' in captured.out
    args = ['list-markets', '--config', 'tests/testdata/testconfigs/main_test_config.json', '--base', 'LTC', '--quote', 'USDT', '--print-list']
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert 'Exchange Binance has 1 active market with LTC as base currency and with USDT as quote currency: XLTCUSDT.\n' in captured.out
    args = ['list-pairs', '--config', 'tests/testdata/testconfigs/main_test_config.json', '--base', 'LTC', '--quote', 'USD', '--print-list']
    start_list_markets(get_args(args), True)
    captured = capsys.readouterr()
    assert 'Exchange Binance has 1 active pair with LTC as base currency and with USD as quote currency: LTC/USD.\n' in captured.out
    args = ['list-markets', '--config', 'tests/testdata/testconfigs/main_test_config.json', '--base', 'LTC', '--quote', 'USDT', 'NONEXISTENT', '--print-list']
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert 'Exchange Binance has 1 active market with LTC as base currency and with USDT, NONEXISTENT as quote currencies: XLTCUSDT.\n' in captured.out
    args = ['list-markets', '--config', 'tests/testdata/testconfigs/main_test_config.json', '--base', 'LTC', '--quote', 'NONEXISTENT', '--print-list']
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert 'Exchange Binance has 0 active markets with LTC as base currency and with NONEXISTENT as quote currency.\n' in captured.out
    args = ['list-markets', '--config', 'tests/testdata/testconfigs/main_test_config.json']
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert 'Exchange Binance has 12 active markets' in captured.out
    args = ['list-markets', '--config', 'tests/testdata/testconfigs/main_test_config.json', '--base', 'LTC', '--quote', 'NONEXISTENT']
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert 'Exchange Binance has 0 active markets with LTC as base currency and with NONEXISTENT as quote currency.\n' in captured.out
    args = ['list-markets', '--config', 'tests/testdata/testconfigs/main_test_config.json', '--print-json']
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert '["ADA/USDT:USDT","BLK/BTC","ETH/BTC","ETH/USDT","ETH/USDT:USDT","LTC/BTC","LTC/ETH","LTC/USD","NEO/BTC","TKN/BTC","XLTCUSDT","XRP/BTC"]' in captured.out
    args = ['list-markets', '--config', 'tests/testdata/testconfigs/main_test_config.json', '--print-csv']
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert 'Id,Symbol,Base,Quote,Active,Spot,Margin,Future,Leverage' in captured.out
    assert 'blkbtc,BLK/BTC,BLK,BTC,True,Spot' in captured.out
    assert 'USD-LTC,LTC/USD,LTC,USD,True,Spot' in captured.out
    args = ['list-markets', '--config', 'tests/testdata/testconfigs/main_test_config.json', '--one-column']
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert re.search('^BLK/BTC$', captured.out, re.MULTILINE)
    assert re.search('^LTC/USD$', captured.out, re.MULTILINE)
    mocker.patch(f'{EXMS}.markets', new_callable=mocker.PropertyMock, side_effect=ValueError)
    args = ['list-markets', '--config', 'tests/testdata/testconfigs/main_test_config.json', '--one-column']
    with pytest.raises(OperationalException, match='Cannot get markets.*'):
        start_list_markets(get_args(args), False)


def test_create_datadir_failed(caplog: LogCaptureFixture) -> None:
    args: List[str] = ['create-userdir']
    with pytest.raises(SystemExit):
        start_create_userdir(get_args(args))
    assert log_has('`create-userdir` requires --userdir to be set.', caplog)


def test_create_datadir(mocker: MockerFixture) -> None:
    cud = mocker.patch('freqtrade.configuration.directory_operations.create_userdata_dir')
    csf = mocker.patch('freqtrade.configuration.directory_operations.copy_sample_files')
    args: List[str] = ['create-userdir', '--userdir', '/temp/freqtrade/test']
    start_create_userdir(get_args(args))
    assert cud.call_count == 1
    assert csf.call_count == 1


def test_start_new_strategy(caplog: LogCaptureFixture, user_dir: Path) -> None:
    strategy_dir: Path = user_dir / 'strategies'
    strategy_dir.mkdir(parents=True, exist_ok=True)
    assert strategy_dir.is_dir()
    args: List[str] = ['new-strategy', '--strategy', 'CoolNewStrategy']
    start_new_strategy(get_args(args))
    assert strategy_dir.exists()
    assert (strategy_dir / 'CoolNewStrategy.py').exists()
    assert log_has_re('Writing strategy to .*', caplog)
    with pytest.raises(OperationalException, match='.* already exists. Please choose another Strategy Name\\.'):
        start_new_strategy(get_args(args))
    args = ['new-strategy', '--strategy', 'CoolNewStrategy', '--strategy-path', str(user_dir)]
    start_new_strategy(get_args(args))
    assert (user_dir / 'CoolNewStrategy.py').exists()
    args = ['new-strategy', '--strategy', 'CoolNewStrategy', '--strategy-path', str(user_dir / 'nonexistent')]
    start_new_strategy(get_args(args))
    assert (user_dir / 'CoolNewStrategy.py').exists()
    assert log_has_re('Creating strategy directory .*', caplog)
    assert (user_dir / 'nonexistent').is_dir()
    assert (user_dir / 'nonexistent' / 'CoolNewStrategy.py').exists()
    shutil.rmtree(str(user_dir))


def test_start_new_strategy_no_arg() -> None:
    args: List[str] = ['new-strategy']
    with pytest.raises(OperationalException, match='`new-strategy` requires --strategy to be set.'):
        start_new_strategy(get_args(args))


def test_start_install_ui(mocker: MockerFixture) -> None:
    clean_mock = mocker.patch('freqtrade.commands.deploy_ui.clean_ui_subdir')
    get_url_mock = mocker.patch('freqtrade.commands.deploy_ui.get_ui_download_url', return_value=('https://example.com/whatever', '0.0.1'))
    download_mock = mocker.patch('freqtrade.commands.deploy_ui.download_and_install_ui')
    mocker.patch('freqtrade.commands.deploy_ui.read_ui_version', return_value=None)
    args: List[str] = ['install-ui']
    start_install_ui(get_args(args))
    assert clean_mock.call_count == 1
    assert get_url_mock.call_count == 1
    assert download_mock.call_count == 1
    clean_mock.reset_mock()
    get_url_mock.reset_mock()
    download_mock.reset_mock()
    args = ['install-ui', '--erase']
    start_install_ui(get_args(args))
    assert clean_mock.call_count == 1
    assert get_url_mock.call_count == 1
    assert download_mock.call_count == 0


def test_clean_ui_subdir(mocker: MockerFixture, tmp_path: Path, caplog: LogCaptureFixture) -> None:
    mocker.patch('freqtrade.commands.deploy_ui.Path.is_dir', side_effect=[True, True])
    mocker.patch('freqtrade.commands.deploy_ui.Path.is_file', side_effect=[False, True])
    rd_mock = mocker.patch('freqtrade.commands.deploy_ui.Path.rmdir')
    ul_mock = mocker.patch('freqtrade.commands.deploy_ui.Path.unlink')
    mocker.patch('freqtrade.commands.deploy_ui.Path.glob', return_value=[Path('test1'), Path('test2'), Path('.gitkeep')])
    folder: Path = tmp_path / 'uitests'
    clean_ui_subdir(folder)
    assert log_has('Removing UI directory content.', caplog)
    assert rd_mock.call_count == 1
    assert ul_mock.call_count == 1


def test_download_and_install_ui(mocker: MockerFixture, tmp_path: Path) -> None:
    requests_mock = mocker.MagicMock()
    file_like_object = BytesIO()
    with ZipFile(file_like_object, mode='w') as zipfile:
        for file in ('test1.txt', 'hello/', 'test2.txt'):
            zipfile.writestr(file, file)
    file_like_object.seek(0)
    requests_mock.content = file_like_object.read()
    mocker.patch('freqtrade.commands.deploy_ui.requests.get', return_value=requests_mock)
    mocker.patch('freqtrade.commands.deploy_ui.Path.is_dir', side_effect=[True, False])
    wb_mock = mocker.patch('freqtrade.commands.deploy_ui.Path.write_bytes')
    folder: Path = tmp_path / 'uitests_dl'
    folder.mkdir(exist_ok=True)
    assert read_ui_version(folder) is None
    download_and_install_ui(folder, 'http://whatever.xxx/download/file.zip', '22')
    assert wb_mock.call_count == 2
    assert read_ui_version(folder) == '22'


def test_get_ui_download_url(mocker: MockerFixture) -> None:
    response = mocker.MagicMock()
    response.json = mocker.MagicMock(side_effect=[[{'assets_url': 'http://whatever.json', 'name': '0.0.1'}], [{'browser_download_url': 'http://download.zip'}]])
    get_mock = mocker.patch('freqtrade.commands.deploy_ui.requests.get', return_value=response)
    x, last_version = get_ui_download_url()
    assert get_mock.call_count == 2
    assert last_version == '0.0.1'
    assert x == 'http://download.zip'


def test_get_ui_download_url_direct(mocker: MockerFixture) -> None:
    response = mocker.MagicMock()
    response.json = mocker.MagicMock(return_value=[{'assets_url': 'http://whatever.json', 'name': '0.0.2', 'assets': [{'browser_download_url': 'http://download22.zip'}]}, {'assets_url': 'http://whatever.json', 'name': '0.0.1', 'assets': [{'browser_download_url': 'http://download1.zip'}]}])
    get_mock = mocker.patch('freqtrade.commands.deploy_ui.requests.get', return_value=response)
    x, last_version = get_ui_download_url()
    assert get_mock.call_count == 1
    assert last_version == '0.0.2'
    assert x == 'http://download22.zip'
    get_mock.reset_mock()
    response.json.reset_mock()
    x, last_version = get_ui_download_url('0.0.1')
    assert last_version == '0.0.1'
    assert x == 'http://download1.zip'
    with pytest.raises(ValueError, match='UI-Version not found.'):
        get_ui_download_url('0.0.3')


def test_download_data_keyboardInterrupt(mocker: MockerFixture, markets: Any) -> None:
    dl_mock = mocker.patch('freqtrade.data.history.download_data_main', return_value=None, side_effect=KeyboardInterrupt)
    patch_exchange(mocker)
    mocker.patch(f'{EXMS}.markets', new_callable=mocker.PropertyMock, return_value=markets)
    args: List[str] = ['download-data', '--exchange', 'binance', '--pairs', 'ETH/BTC', 'XRP/BTC']
    pargs = get_args(args)
    pargs['config'] = None
    with pytest.raises(SystemExit):
        start_download_data(pargs)
    assert dl_mock.call_count == 1


@pytest.mark.parametrize('time', ['00:00', '00:03', '00:30', '23:56'])
@pytest.mark.parametrize('tzoffset', ['00:00', '+01:00', '-01:00', '+05:00', '-05:00'])
def test_download_data_timerange(mocker: MockerFixture, markets: Any, time_machine: Any, time: str, tzoffset: str) -> None:
    time_machine.move_to(f'2024-11-01 {time}:00 {tzoffset}')
    dl_mock = mocker.patch('freqtrade.data.history.history_utils.refresh_backtest_ohlcv_data', return_value=['ETH/BTC', 'XRP/BTC'])
    patch_exchange(mocker)
    mocker.patch(f'{EXMS}.markets', new_callable=mocker.PropertyMock, return_value=markets)
    args: List[str] = ['download-data', '--exchange', 'binance', '--pairs', 'ETH/BTC', 'XRP/BTC', '--days', '20', '--timerange', '20200101-']
    with pytest.raises(OperationalException, match='--days and --timerange are mutually.*'):
        pargs = get_args(args)
        pargs['config'] = None
        start_download_data(pargs)
    assert dl_mock.call_count == 0
    args = ['download-data', '--exchange', 'binance', '--pairs', 'ETH/BTC', 'XRP/BTC', '--days', '20']
    pargs = get_args(args)
    pargs['config'] = None
    start_download_data(pargs)
    assert dl_mock.call_count == 1
    days_ago: datetime = datetime.now() - timedelta(days=20)
    days_ago = dt_utc(days_ago.year, days_ago.month, days_ago.day)
    assert dl_mock.call_args_list[0][1]['timerange'].startts == days_ago.timestamp()
    dl_mock.reset_mock()
    args = ['download-data', '--exchange', 'binance', '--pairs', 'ETH/BTC', 'XRP/BTC', '--timerange', '20200101-']
    pargs = get_args(args)
    pargs['config'] = None
    start_download_data(pargs)
    assert dl_mock.call_count == 1
    assert dl_mock.call_args_list[0][1]['timerange'].startts == int(dt_utc(2020, 1, 1).timestamp())


def test_download_data_no_exchange(mocker: MockerFixture) -> None:
    mocker.patch('freqtrade.data.history.history_utils.refresh_backtest_ohlcv_data', return_value=['ETH/BTC', 'XRP/BTC'])
    patch_exchange(mocker)
    mocker.patch(f'{EXMS}.get_markets', return_value={})
    args: List[str] = ['download-data']
    pargs = get_args(args)
    pargs['config'] = None
    with pytest.raises(OperationalException, match='This command requires a configured exchange.*'):
        start_download_data(pargs)


def test_download_data_no_pairs(mocker: MockerFixture) -> None:
    mocker.patch('freqtrade.data.history.history_utils.refresh_backtest_ohlcv_data', return_value=['ETH/BTC', 'XRP/BTC'])
    patch_exchange(mocker)
    mocker.patch(f'{EXMS}.markets', new_callable=mocker.PropertyMock, return_value={})
    args: List[str] = ['download-data', '--exchange', 'binance']
    pargs = get_args(args)
    pargs['config'] = None
    with pytest.raises(OperationalException, match='Downloading data requires a list of pairs\\..*'):
        start_download_data(pargs)


def test_download_data_all_pairs(mocker: MockerFixture, markets: Any) -> None:
    dl_mock = mocker.patch('freqtrade.data.history.history_utils.refresh_backtest_ohlcv_data', return_value=['ETH/BTC', 'XRP/BTC'])
    patch_exchange(mocker)
    mocker.patch(f'{EXMS}.markets', new_callable=mocker.PropertyMock, return_value=markets)
    args: List[str] = ['download-data', '--exchange', 'binance', '--pairs', '.*/USDT']
    pargs = get_args(args)
    pargs['config'] = None
    start_download_data(pargs)
    expected = set(['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'NEO/USDT', 'TKN/USDT'])
    assert set(dl_mock.call_args_list[0][1]['pairs']) == expected
    assert dl_mock.call_count == 1
    dl_mock.reset_mock()
    args = ['download-data', '--exchange', 'binance', '--pairs', '.*/USDT', '--include-inactive-pairs']
    pargs = get_args(args)
    pargs['config'] = None
    start_download_data(pargs)
    expected = set(['BTC/USDT', 'ETH/USDT', 'LTC/USDT', 'XRP/USDT', 'NEO/USDT', 'TKN/USDT'])
    assert set(dl_mock.call_args_list[0][1]['pairs']) == expected


def test_download_data_trades(mocker: MockerFixture) -> None:
    dl_mock = mocker.patch('freqtrade.data.history.history_utils.refresh_backtest_trades_data', return_value=[])
    convert_mock = mocker.patch('freqtrade.data.history.history_utils.convert_trades_to_ohlcv', return_value=[])
    patch_exchange(mocker)
    mocker.patch(f'{EXMS}.get_markets', return_value={'ETH/BTC': {}, 'XRP/BTC': {}})
    args: List[str] = ['download-data', '--exchange', 'kraken', '--pairs', 'ETH/BTC', 'XRP/BTC', '--days', '20', '--dl-trades']
    pargs = get_args(args)
    pargs['config'] = None
    start_download_data(pargs)
    assert dl_mock.call_args[1]['timerange'].starttype == 'date'
    assert dl_mock.call_count == 1
    assert convert_mock.call_count == 1
    args = ['download-data', '--exchange', 'kraken', '--pairs', 'ETH/BTC', 'XRP/BTC', '--days', '20', '--trading-mode', 'futures', '--dl-trades']


def test_download_data_data_invalid(mocker: MockerFixture) -> None:
    patch_exchange(mocker, exchange='kraken')
    mocker.patch(f'{EXMS}.get_markets', return_value={'ETH/BTC': {}, 'XRP/BTC': {}})
    args: List[str] = ['download-data', '--exchange', 'kraken', '--pairs', 'ETH/BTC', 'XRP/BTC', '--days', '20']
    pargs = get_args(args)
    pargs['config'] = None
    with pytest.raises(OperationalException, match='Historic klines not available for .*'):
        start_download_data(pargs)


def test_start_convert_trades(mocker: MockerFixture) -> None:
    convert_mock = mocker.patch('freqtrade.data.converter.convert_trades_to_ohlcv', return_value=[])
    patch_exchange(mocker)
    mocker.patch(f'{EXMS}.get_markets')
    mocker.patch(f'{EXMS}.markets', new_callable=mocker.PropertyMock, return_value={})
    args: List[str] = ['trades-to-ohlcv', '--exchange', 'kraken', '--pairs', 'ETH/BTC', 'XRP/BTC']
    start_convert_trades(get_args(args))
    assert convert_mock.call_count == 1


def test_start_list_strategies(capsys: CaptureFixture[str]) -> None:
    args: List[str] = ['list-strategies', '--strategy-path', str(Path(__file__).parent.parent / 'strategy' / 'strats'), '-1']
    pargs = get_args(args)
    start_list_strategies(pargs)
    captured = capsys.readouterr()
    assert 'StrategyTestV2' in captured.out
    assert 'strategy_test_v2.py' not in captured.out
    assert CURRENT_TEST_STRATEGY in captured.out
    args = ['list-strategies', '--strategy-path', str(Path(__file__).parent.parent / 'strategy' / 'strats'), '--no-color']
    pargs = get_args(args)
    start_list_strategies(pargs)
    captured = capsys.readouterr()
    assert 'StrategyTestV2' in captured.out
    assert 'strategy_test_v2.py' in captured.out
    assert CURRENT_TEST_STRATEGY in captured.out
    args = ['list-strategies', '--strategy-path', str(Path(__file__).parent.parent / 'strategy' / 'strats')]
    pargs = get_args(args)
    start_list_strategies(pargs)
    captured = capsys.readouterr()
    assert 'StrategyTestV2' in captured.out
    assert 'strategy_test_v2.py' in captured.out
    assert CURRENT_TEST_STRATEGY in captured.out
    assert 'LOAD FAILED' in captured.out
    assert 'TestStrategyNoImplements' not in captured.out
    args = ['list-strategies', '--strategy-path', str(Path(__file__).parent.parent / 'strategy' / 'strats'), '--no-color', '--recursive-strategy-search']
    pargs = get_args(args)
    start_list_strategies(pargs)
    captured = capsys.readouterr()
    assert 'StrategyTestV2' in captured.out
    assert 'strategy_test_v2.py' in captured.out
    assert 'StrategyTestV2' in captured.out
    assert 'TestStrategyNoImplements' in captured.out
    assert str(Path('broken_strats/broken_futures_strategies.py')) in captured.out


def test_start_list_hyperopt_loss_functions(capsys: CaptureFixture[str]) -> None:
    args: List[str] = ['list-hyperoptloss', '-1']
    pargs = get_args(args)
    pargs['config'] = None
    start_list_hyperopt_loss_functions(pargs)
    captured = capsys.readouterr()
    assert 'CalmarHyperOptLoss' in captured.out
    assert 'MaxDrawDownHyperOptLoss' in captured.out
    assert 'SortinoHyperOptLossDaily' in captured.out
    assert '<builtin>/hyperopt_loss_sortino_daily.py' not in captured.out
    args = ['list-hyperoptloss']
    pargs = get_args(args)
    pargs['config'] = None
    start_list_hyperopt_loss_functions(pargs)
    captured = capsys.readouterr()
    assert 'CalmarHyperOptLoss' in captured.out
    assert 'MaxDrawDownHyperOptLoss' in captured.out
    assert 'SortinoHyperOptLossDaily' in captured.out
    assert '<builtin>/hyperopt_loss_sortino_daily.py' in captured.out


def test_start_list_freqAI_models(capsys: CaptureFixture[str]) -> None:
    args: List[str] = ['list-freqaimodels', '-1']
    pargs = get_args(args)
    pargs['config'] = None
    start_list_freqAI_models(pargs)
    captured = capsys.readouterr()
    assert 'LightGBMClassifier' in captured.out
    assert 'LightGBMRegressor' in captured.out
    assert 'XGBoostRegressor' in captured.out
    assert '<builtin>/LightGBMRegressor.py' not in captured.out
    args = ['list-freqaimodels']
    pargs = get_args(args)
    pargs['config'] = None
    start_list_freqAI_models(pargs)
    captured = capsys.readouterr()
    assert 'LightGBMClassifier' in captured.out
    assert 'LightGBMRegressor' in captured.out
    assert 'XGBoostRegressor' in captured.out
    assert '<builtin>/LightGBMRegressor.py' in captured.out


def test_start_test_pairlist(mocker: MockerFixture, caplog: LogCaptureFixture, tickers: Any, default_conf: Any, capsys: CaptureFixture[str]) -> None:
    patch_exchange(mocker, mock_markets=True)
    mocker.patch.multiple(EXMS, exchange_has=mocker.MagicMock(return_value=True), get_tickers=tickers)
    default_conf['pairlists'] = [
        {'method': 'VolumePairList', 'number_assets': 5, 'sort_key': 'quoteVolume'},
        {'method': 'PrecisionFilter'},
        {'method': 'PriceFilter', 'low_price_ratio': 0.02}
    ]
    patched_configuration_load_config_file(mocker, default_conf)
    args: List[str] = ['test-pairlist', '-c', 'tests/testdata/testconfigs/main_test_config.json']
    start_test_pairlist(get_args(args))
    assert log_has_re('^Using resolved pairlist VolumePairList.*', caplog)
    assert log_has_re('^Using resolved pairlist PrecisionFilter.*', caplog)
    assert log_has_re('^Using resolved pairlist PriceFilter.*', caplog)
    captured = capsys.readouterr()
    assert re.match('Pairs for .*', captured.out)
    assert re.match("['ETH/BTC', 'TKN/BTC', 'BLK/BTC', 'LTC/BTC', 'XRP/BTC']", captured.out)
    args = ['test-pairlist', '-c', 'tests/testdata/testconfigs/main_test_config.json', '--one-column']
    start_test_pairlist(get_args(args))
    captured = capsys.readouterr()
    assert re.match('ETH/BTC\\nTKN/BTC\\nBLK/BTC\\nLTC/BTC\\nXRP/BTC\\n', captured.out)
    args = ['test-pairlist', '-c', 'tests/testdata/testconfigs/main_test_config.json', '--print-json']
    start_test_pairlist(get_args(args))
    captured = capsys.readouterr()
    try:
        json_pairs = json.loads(captured.out)
        assert 'ETH/BTC' in json_pairs
        assert 'TKN/BTC' in json_pairs
        assert 'BLK/BTC' in json_pairs
        assert 'LTC/BTC' in json_pairs
        assert 'XRP/BTC' in json_pairs
    except json.decoder.JSONDecodeError:
        pytest.fail(f'Expected well formed JSON, but failed to parse: {captured.out}')


def test_hyperopt_list(mocker: MockerFixture, capsys: CaptureFixture[str], caplog: LogCaptureFixture, tmp_path: Path) -> None:
    saved_hyperopt_results = hyperopt_test_result()
    csv_file: Path = tmp_path / 'test.csv'
    mocker.patch('freqtrade.optimize.hyperopt_tools.HyperoptTools._test_hyperopt_results_exist', return_value=True)

    def fake_iterator(*args: Any, **kwargs: Any):
        yield from [saved_hyperopt_results]
    mocker.patch('freqtrade.optimize.hyperopt_tools.HyperoptTools._read_results', side_effect=fake_iterator)
    args: List[str] = ['hyperopt-list', '--no-details', '--no-color']
    pargs = get_args(args)
    pargs['config'] = None
    start_hyperopt_list(pargs)
    captured = capsys.readouterr()
    assert all((x in captured.out for x in [' 1/12', ' 2/12', ' 3/12', ' 4/12', ' 5/12', ' 6/12', ' 7/12', ' 8/12', ' 9/12', ' 10/12', ' 11/12', ' 12/12']))
    args = ['hyperopt-list', '--best', '--no-details', '--no-color']
    pargs = get_args(args)
    pargs['config'] = None
    start_hyperopt_list(pargs)
    captured = capsys.readouterr()
    assert all((x in captured.out for x in [' 1/12', ' 5/12', ' 10/12']))
    assert all((x not in captured.out for x in [' 2/12', ' 3/12', ' 4/12', ' 6/12', ' 7/12', ' 8/12', ' 9/12', ' 11/12', ' 12/12']))
    args = ['hyperopt-list', '--profitable', '--no-details', '--no-color']
    pargs = get_args(args)
    pargs['config'] = None
    start_hyperopt_list(pargs)
    captured = capsys.readouterr()
    assert all((x in captured.out for x in [' 2/12', ' 10/12']))
    assert all((x not in captured.out for x in [' 1/12', ' 3/12', ' 4/12', ' 5/12', ' 6/12', ' 7/12', ' 8/12', ' 9/12', ' 11/12', ' 12/12']))
    args = ['hyperopt-list', '--profitable', '--no-color']
    pargs = get_args(args)
    pargs['config'] = None
    start_hyperopt_list(pargs)
    captured = capsys.readouterr()
    assert all((x in captured.out for x in [' 2/12', ' 10/12', 'Best result:', 'Buy hyperspace params', 'Sell hyperspace params', 'ROI table', 'Stoploss']))
    assert all((x not in captured.out for x in [' 1/12', ' 3/12', ' 4/12', ' 5/12', ' 6/12', ' 7/12', ' 8/12', ' 9/12', ' 11/12', ' 12/12']))
    args = ['hyperopt-list', '--no-details', '--no-color', '--min-trades', '20']
    pargs = get_args(args)
    pargs['config'] = None
    start_hyperopt_list(pargs)
    captured = capsys.readouterr()
    assert all((x in captured.out for x in [' 3/12', ' 6/12', ' 7/12', ' 9/12', ' 11/12']))
    assert all((x not in captured.out for x in [' 1/12', ' 2/12', ' 4/12', ' 5/12', ' 8/12', ' 10/12', ' 12/12']))
    args = ['hyperopt-list', '--profitable', '--no-details', '--no-color', '--max-trades', '20']
    pargs = get_args(args)
    pargs['config'] = None
    start_hyperopt_list(pargs)
    captured = capsys.readouterr()
    assert all((x in captured.out for x in [' 2/12', ' 10/12']))
    assert all((x not in captured.out for x in [' 1/12', ' 3/12', ' 4/12', ' 5/12', ' 6/12', ' 7/12', ' 8/12', ' 9/12', ' 11/12', ' 12/12']))
    args = ['hyperopt-list', '--profitable', '--no-details', '--no-color', '--min-avg-profit', '0.11']
    pargs = get_args(args)
    pargs['config'] = None
    start_hyperopt_list(pargs)
    captured = capsys.readouterr()
    assert all((x in captured.out for x in [' 2/12']))
    assert all((x not in captured.out for x in [' 1/12', ' 3/12', ' 4/12', ' 5/12', ' 6/12', ' 7/12', ' 8/12', ' 9/12', ' 10/12', ' 11/12', ' 12/12']))
    args = ['hyperopt-list', '--no-details', '--no-color', '--max-avg-profit', '0.10']
    pargs = get_args(args)
    pargs['config'] = None
    start_hyperopt_list(pargs)
    captured = capsys.readouterr()
    assert all((x in captured.out for x in [' 1/12', ' 3/12', ' 5/12', ' 6/12', ' 7/12', ' 8/12', ' 9/12', ' 11/12']))
    assert all((x not in captured.out for x in [' 2/12', ' 4/12', ' 10/12', ' 12/12']))
    args = ['hyperopt-list', '--no-details', '--no-color', '--min-total-profit', '0.4']
    pargs = get_args(args)
    pargs['config'] = None
    start_hyperopt_list(pargs)
    captured = capsys.readouterr()
    assert all((x in captured.out for x in [' 10/12']))
    assert all((x not in captured.out for x in [' 1/12', ' 2/12', ' 3/12', ' 4/12', ' 5/12', ' 6/12', ' 7/12', ' 8/12', ' 9/12', ' 11/12', ' 12/12']))
    args = ['hyperopt-list', '--no-details', '--no-color', '--max-total-profit', '0.4']
    pargs = get_args(args)
    pargs['config'] = None
    start_hyperopt_list(pargs)
    captured = capsys.readouterr()
    assert all((x in captured.out for x in [' 1/12', ' 2/12', ' 3/12', ' 5/12', ' 6/12', ' 7/12', ' 8/12', ' 9/12', ' 11/12']))
    assert all((x not in captured.out for x in [' 4/12', ' 10/12', ' 12/12']))
    args = ['hyperopt-list', '--no-details', '--no-color', '--min-objective', '0.1']
    pargs = get_args(args)
    pargs['config'] = None
    start_hyperopt_list(pargs)
    captured = capsys.readouterr()
    assert all((x in captured.out for x in [' 10/12']))
    assert all((x not in captured.out for x in [' 1/12', ' 2/12', ' 3/12', ' 4/12', ' 5/12', ' 6/12', ' 7/12', ' 8/12', ' 9/12', ' 11/12', ' 12/12']))
    args = ['hyperopt-list', '--no-details', '--max-objective', '0.1']
    pargs = get_args(args)
    pargs['config'] = None
    start_hyperopt_list(pargs)
    captured = capsys.readouterr()
    assert all((x in captured.out for x in [' 1/12', ' 2/12', ' 3/12', ' 5/12', ' 6/12', ' 7/12', ' 8/12', ' 9/12', ' 11/12']))
    assert all((x not in captured.out for x in [' 4/12', ' 10/12', ' 12/12']))
    args = ['hyperopt-list', '--profitable', '--no-details', '--no-color', '--min-avg-time', '2000']
    pargs = get_args(args)
    pargs['config'] = None
    start_hyperopt_list(pargs)
    captured = capsys.readouterr()
    assert all((x in captured.out for x in [' 10/12']))
    assert all((x not in captured.out for x in [' 1/12', ' 2/12', ' 3/12', ' 4/12', ' 5/12', ' 6/12', ' 7/12', ' 8/12', ' 9/12', ' 11/12', ' 12/12']))
    args = ['hyperopt-list', '--no-details', '--no-color', '--max-avg-time', '1500']
    pargs = get_args(args)
    pargs['config'] = None
    start_hyperopt_list(pargs)
    captured = capsys.readouterr()
    assert all((x in captured.out for x in [' 2/12', ' 6/12']))
    assert all((x not in captured.out for x in [' 1/12', ' 3/12', ' 4/12', ' 5/12', ' 7/12', ' 8/12', ' 9/12', ' 10/12', ' 11/12', ' 12/12']))
    args = ['hyperopt-list', '--no-details', '--no-color', '--export-csv', str(csv_file)]
    pargs = get_args(args)
    pargs['config'] = None
    start_hyperopt_list(pargs)
    captured = capsys.readouterr()
    log_has('CSV file created: test_file.csv', caplog)
    assert csv_file.is_file()
    line = csv_file.read_text()
    assert 'Best,1,2,-1.25%,-1.2222,-0.00125625,BTC,-2.51,"3,930.0 m",-0.00125625,23.00%,0.43662' in line or 'Best,1,2,-1.25%,-1.2222,-0.00125625,BTC,-2.51,2 days 17:30:00,2,0,-0.00125625,23.00%,0.43662' in line
    csv_file.unlink()


def test_hyperopt_show(mocker: MockerFixture, capsys: CaptureFixture[str]) -> None:
    saved_hyperopt_results = hyperopt_test_result()
    mocker.patch('freqtrade.optimize.hyperopt_tools.HyperoptTools._test_hyperopt_results_exist', return_value=True)

    def fake_iterator(*args: Any, **kwargs: Any):
        yield from [saved_hyperopt_results]
    mocker.patch('freqtrade.optimize.hyperopt_tools.HyperoptTools._read_results', side_effect=fake_iterator)
    mocker.patch('freqtrade.optimize.optimize_reports.show_backtest_result')
    args: List[str] = ['hyperopt-show']
    pargs = get_args(args)
    pargs['config'] = None
    start_hyperopt_show(pargs)
    captured = capsys.readouterr()
    assert ' 12/12' in captured.out
    args = ['hyperopt-show', '--best']
    pargs = get_args(args)
    pargs['config'] = None
    start_hyperopt_show(pargs)
    captured = capsys.readouterr()
    assert ' 10/12' in captured.out
    args = ['hyperopt-show', '-n', '1']
    pargs = get_args(args)
    pargs['config'] = None
    start_hyperopt_show(pargs)
    captured = capsys.readouterr()
    assert ' 1/12' in captured.out
    args = ['hyperopt-show', '--best', '-n', '2']
    pargs = get_args(args)
    pargs['config'] = None
    start_hyperopt_show(pargs)
    captured = capsys.readouterr()
    assert ' 5/12' in captured.out
    args = ['hyperopt-show', '--best', '-n', '-1']
    pargs = get_args(args)
    pargs['config'] = None
    start_hyperopt_show(pargs)
    captured = capsys.readouterr()
    assert ' 10/12' in captured.out
    args = ['hyperopt-show', '--best', '-n', '-4']
    pargs = get_args(args)
    pargs['config'] = None
    with pytest.raises(OperationalException, match='The index of the epoch to show should be greater than -4.'):
        start_hyperopt_show(pargs)
    args = ['hyperopt-show', '--best', '-n', '4']
    pargs = get_args(args)
    pargs['config'] = None
    with pytest.raises(OperationalException, match='The index of the epoch to show should be less than 4.'):
        start_hyperopt_show(pargs)


def test_convert_data(mocker: MockerFixture, testdatadir: Path) -> None:
    ohlcv_mock = mocker.patch('freqtrade.data.converter.convert_ohlcv_format')
    trades_mock = mocker.patch('freqtrade.data.converter.convert_trades_format')
    args: List[str] = ['convert-data', '--format-from', 'json', '--format-to', 'jsongz', '--datadir', str(testdatadir)]
    pargs = get_args(args)
    pargs['config'] = None
    start_convert_data(pargs, True)
    assert trades_mock.call_count == 0
    assert ohlcv_mock.call_count == 1
    assert ohlcv_mock.call_args[1]['convert_from'] == 'json'
    assert ohlcv_mock.call_args[1]['convert_to'] == 'jsongz'
    assert ohlcv_mock.call_args[1]['erase'] is False


def test_convert_data_trades(mocker: MockerFixture, testdatadir: Path) -> None:
    ohlcv_mock = mocker.patch('freqtrade.data.converter.convert_ohlcv_format')
    trades_mock = mocker.patch('freqtrade.data.converter.convert_trades_format')
    args: List[str] = ['convert-trade-data', '--format-from', 'jsongz', '--format-to', 'json', '--datadir', str(testdatadir)]
    pargs = get_args(args)
    pargs['config'] = None
    start_convert_data(pargs, False)
    assert ohlcv_mock.call_count == 0
    assert trades_mock.call_count == 1
    assert trades_mock.call_args[1]['convert_from'] == 'jsongz'
    assert trades_mock.call_args[1]['convert_to'] == 'json'
    assert trades_mock.call_args[1]['erase'] is False


def test_start_list_data(testdatadir: Path, capsys: CaptureFixture[str]) -> None:
    args: List[str] = ['list-data', '--datadir', str(testdatadir)]
    pargs = get_args(args)
    pargs['config'] = None
    start_list_data(pargs)
    captured = capsys.readouterr()
    assert 'Found 16 pair / timeframe combinations.' in captured.out
    assert re.search('.*Pair.*Timeframe.*Type.*\\n', captured.out)
    assert re.search('\\n.* UNITTEST/BTC .* 1m, 5m, 8m, 30m .* spot |\\n', captured.out)
    args = ['list-data', '--data-format-ohlcv', 'feather', '--pairs', 'XRP/ETH', '--datadir', str(testdatadir)]
    pargs = get_args(args)
    pargs['config'] = None
    start_list_data(pargs)
    captured = capsys.readouterr()
    assert 'Found 2 pair / timeframe combinations.' in captured.out
    assert re.search('.*Pair.*Timeframe.*Type.*\\n', captured.out)
    assert 'UNITTEST/BTC' not in captured.out
    assert re.search('\\n.* XRP/ETH .* 1m, 5m .* spot |\\n', captured.out)
    args = ['list-data', '--trading-mode', 'futures', '--datadir', str(testdatadir)]
    pargs = get_args(args)
    pargs['config'] = None
    start_list_data(pargs)
    captured = capsys.readouterr()
    assert 'Found 6 pair / timeframe combinations.' in captured.out
    assert re.search('.*Pair.*Timeframe.*Type.*\\n', captured.out)
    assert re.search('\\n.* XRP/USDT:USDT .* 5m, 1h .* futures |\\n', captured.out)
    assert re.search('\\n.* XRP/USDT:USDT .* 1h, 8h .* mark |\\n', captured.out)
    args = ['list-data', '--pairs', 'XRP/ETH', '--datadir', str(testdatadir), '--show-timerange']
    pargs = get_args(args)
    pargs['config'] = None
    start_list_data(pargs)
    captured = capsys.readouterr()
    assert 'Found 2 pair / timeframe combinations.' in captured.out
    assert re.search('.*Pair.*Timeframe.*Type.*From .* To .* Candles .*\\n', captured.out)
    assert 'UNITTEST/BTC' not in captured.out
    assert re.search('\\n.* XRP/USDT .* 1m .* spot .* 2019-10-11 00:00:00 .* 2019-10-13 11:19:00 .* 2469 |\\n', captured.out)


def test_start_list_trades_data(testdatadir: Path, capsys: CaptureFixture[str]) -> None:
    args: List[str] = ['list-data', '--datadir', str(testdatadir), '--trades']
    pargs = get_args(args)
    pargs['config'] = None
    start_list_data(pargs)
    captured = capsys.readouterr()
    assert 'Found trades data for 1 pair.' in captured.out
    assert re.search('.*Pair.*Type.*\\n', captured.out)
    assert re.search('\\n.* XRP/ETH .* spot |\\n', captured.out)
    args = ['list-data', '--datadir', str(testdatadir), '--trades', '--show-timerange']
    pargs = get_args(args)
    pargs['config'] = None
    start_list_data(pargs)
    captured = capsys.readouterr()
    assert 'Found trades data for 1 pair.' in captured.out
    assert re.search('.*Pair.*Type.*From.*To.*Trades.*\\n', captured.out)
    assert re.search('\\n.* XRP/ETH .* spot .* 2019-10-11 00:00:01 .* 2019-10-13 11:19:28 .* 12477 .*|\\n', captured.out)
    args = ['list-data', '--datadir', str(testdatadir), '--trades', '--trading-mode', 'futures']
    pargs = get_args(args)
    pargs['config'] = None
    start_list_data(pargs)
    captured = capsys.readouterr()
    assert 'Found trades data for 0 pairs.' in captured.out


@pytest.mark.usefixtures('init_persistence')
def test_show_trades(mocker: MockerFixture, fee: Any, capsys: CaptureFixture[str], caplog: LogCaptureFixture) -> None:
    mocker.patch('freqtrade.persistence.init_db')
    create_mock_trades(fee, False)
    args: List[str] = ['show-trades', '--db-url', 'sqlite:///']
    pargs = get_args(args)
    pargs['config'] = None
    start_show_trades(pargs)
    assert log_has(f'Printing {MOCK_TRADE_COUNT} Trades: ', caplog)
    captured = capsys.readouterr()
    assert 'Trade(id=1' in captured.out
    assert 'Trade(id=2' in captured.out
    assert 'Trade(id=3' in captured.out
    args = ['show-trades', '--db-url', 'sqlite:///', '--print-json', '--trade-ids', '1', '2']
    pargs = get_args(args)
    pargs['config'] = None
    start_show_trades(pargs)
    captured = capsys.readouterr()
    assert log_has('Printing 2 Trades: ', caplog)
    assert '"trade_id": 1' in captured.out
    assert '"trade_id": 2' in captured.out
    assert '"trade_id": 3' not in captured.out
    args = ['show-trades']
    pargs = get_args(args)
    pargs['config'] = None
    with pytest.raises(OperationalException, match='--db-url is required for this command.'):
        start_show_trades(pargs)


def test_backtesting_show(mocker: MockerFixture, testdatadir: Path, capsys: CaptureFixture[str]) -> None:
    sbr = mocker.patch('freqtrade.optimize.optimize_reports.show_backtest_results')
    args: List[str] = ['backtesting-show', '--export-filename', f'{testdatadir / "backtest_results/backtest-result.json"}', '--show-pair-list']
    pargs = get_args(args)
    pargs['config'] = None
    start_backtesting_show(pargs)
    assert sbr.call_count == 1
    out, _err = capsys.readouterr()
    assert 'Pairs for Strategy' in out


def test_start_convert_db(fee: Any, tmp_path: Path) -> None:
    db_src_file: Path = tmp_path / 'db.sqlite'
    db_from = f'sqlite:///{db_src_file}'
    db_target_file: Path = tmp_path / 'db_target.sqlite'
    db_to = f'sqlite:///{db_target_file}'
    args: List[str] = ['convert-db', '--db-url-from', db_from, '--db-url', db_to]
    assert not db_src_file.is_file()
    init_db(db_from)
    create_mock_trades(fee)
    PairLocks.timeframe = '5m'
    PairLocks.lock_pair('XRP/USDT', datetime.now(), 'Random reason 125', side='long')
    assert db_src_file.is_file()
    assert not db_target_file.is_file()
    pargs = get_args(args)
    pargs['config'] = None
    start_convert_db(pargs)
    assert db_target_file.is_file()


def test_start_strategy_updater(mocker: MockerFixture, tmp_path: Path) -> None:
    sc_mock = mocker.patch('freqtrade.commands.strategy_utils_commands.start_conversion')
    teststrats: Path = Path(__file__).parent.parent / 'strategy/strats'
    args: List[str] = ['strategy-updater', '--userdir', str(tmp_path), '--strategy-path', str(teststrats)]
    pargs = get_args(args)
    pargs['config'] = None
    start_strategy_update(pargs)
    assert sc_mock.call_count == 12
    sc_mock.reset_mock()
    args = ['strategy-updater', '--userdir', str(tmp_path), '--strategy-path', str(teststrats), '--strategy-list', 'StrategyTestV3', 'StrategyTestV2']
    pargs = get_args(args)
    pargs['config'] = None
    start_strategy_update(pargs)
    assert sc_mock.call_count == 2


def test_start_show_config(capsys: CaptureFixture[str], caplog: LogCaptureFixture) -> None:
    args: List[str] = ['show-config', '--config', 'tests/testdata/testconfigs/main_test_config.json']
    pargs = get_args(args)
    start_show_config(pargs)
    captured = capsys.readouterr()
    assert 'Your combined configuration is:' in captured.out
    assert '"max_open_trades":' in captured.out
    assert '"secret": "REDACTED"' in captured.out
    args = ['show-config', '--config', 'tests/testdata/testconfigs/main_test_config.json', '--show-sensitive']
    pargs = get_args(args)
    start_show_config(pargs)
    captured = capsys.readouterr()
    assert 'Your combined configuration is:' in captured.out
    assert '"max_open_trades":' in captured.out
    assert '"secret": "REDACTED"' not in captured.out
    assert log_has_re('Sensitive information will be shown in the upcoming output.*', caplog)
