import json
import re
import shutil
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
from unittest.mock import MagicMock, PropertyMock
from zipfile import ZipFile

import pytest
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
    start_list_strategies,
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
    args = ['list-exchanges', '--config', 'tests/testdata/testconfigs/main_test_config.json']
    config = setup_utils_configuration(get_args(args), RunMode.OTHER)
    assert 'exchange' in config
    assert config['dry_run'] is True
    args = ['list-exchanges', '--config', 'tests/testdata/testconfigs/testconfig.json']
    config = setup_utils_configuration(get_args(args), RunMode.OTHER, set_dry=False)
    assert 'exchange' in config
    assert config['dry_run'] is False


def test_start_trading_fail(mocker: Any, caplog: Any) -> None:
    mocker.patch('freqtrade.worker.Worker.run', MagicMock(side_effect=OperationalException))
    mocker.patch('freqtrade.worker.Worker.__init__', MagicMock(return_value=None))
    exitmock = mocker.patch('freqtrade.worker.Worker.exit', MagicMock())
    args = ['trade', '-c', 'tests/testdata/testconfigs/main_test_config.json']
    with pytest.raises(OperationalException):
        start_trading(get_args(args))
    assert exitmock.call_count == 1
    exitmock.reset_mock()
    caplog.clear()
    mocker.patch('freqtrade.worker.Worker.__init__', MagicMock(side_effect=OperationalException))
    with pytest.raises(OperationalException):
        start_trading(get_args(args))
    assert exitmock.call_count == 0


def test_start_webserver(mocker: Any, caplog: Any) -> None:
    api_server_mock = mocker.patch('freqtrade.rpc.api_server.ApiServer')
    args = ['webserver', '-c', 'tests/testdata/testconfigs/main_test_config.json']
    start_webserver(get_args(args))
    assert api_server_mock.call_count == 1


def test_list_exchanges(capsys: Any) -> None:
    args = ['list-exchanges']
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


def test_list_timeframes(mocker: Any, capsys: Any) -> None:
    api_mock = MagicMock()
    api_mock.timeframes = {'1m': 'oneMin', '5m': 'fiveMin', '30m': 'thirtyMin', '1h': 'hour', '1d': 'day'}
    patch_exchange(mocker, api_mock=api_mock, exchange='bybit')
    args = ['list-timeframes']
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


def test_list_markets(mocker: Any, markets_static: Any, capsys: Any) -> None:
    api_mock = MagicMock()
    patch_exchange(mocker, api_mock=api_mock, exchange='binance', mock_markets=markets_static)
    args = ['list-markets']
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
    mocker.patch(f'{EXMS}.markets', PropertyMock(side_effect=ValueError))
    args = ['list-markets', '--config', 'tests/testdata/testconfigs/main_test_config.json', '--one-column']
    with pytest