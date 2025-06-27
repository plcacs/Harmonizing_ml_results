import random
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import ANY, MagicMock, PropertyMock
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import pytest
from freqtrade import constants
from freqtrade.commands.optimize_commands import setup_optimize_configuration, start_backtesting
from freqtrade.configuration import TimeRange
from freqtrade.data import history
from freqtrade.data.btanalysis import BT_DATA_COLUMNS, evaluate_result_multi
from freqtrade.data.converter import clean_ohlcv_dataframe, ohlcv_fill_up_missing_data
from freqtrade.data.dataprovider import DataProvider
from freqtrade.data.history import get_timerange
from freqtrade.enums import CandleType, ExitType, RunMode
from freqtrade.exceptions import DependencyException, OperationalException
from freqtrade.exchange import timeframe_to_next_date, timeframe_to_prev_date
from freqtrade.optimize.backtest_caching import get_backtest_metadata_filename, get_strategy_run_id
from freqtrade.optimize.backtesting import Backtesting
from freqtrade.persistence import LocalTrade, Trade
from freqtrade.resolvers import StrategyResolver
from freqtrade.util.datetime_helpers import dt_utc
from tests.conftest import CURRENT_TEST_STRATEGY, EXMS, generate_test_data, get_args, log_has, log_has_re, patch_exchange, patched_configuration_load_config_file
ORDER_TYPES: List[Dict[str, Union[str, bool]]] = [{'entry': 'limit', 'exit':
    'limit', 'stoploss': 'limit', 'stoploss_on_exchange': False}, {'entry':
    'limit', 'exit': 'limit', 'stoploss': 'limit', 'stoploss_on_exchange': 
    True}]


def trim_dictlist(dict_list: Dict[str, pd.DataFrame], num) ->Dict[str, pd.
    DataFrame]:
    new: Dict[str, pd.DataFrame] = {}
    for pair, pair_data in dict_list.items():
        new[pair] = pair_data[num:].reset_index()
    return new


def load_data_test(what: str, testdatadir: Path):
    timerange = TimeRange.parse_timerange('1510694220-1510700340')
    data = history.load_pair_history(pair='UNITTEST/BTC', datadir=
        testdatadir, timeframe='1m', timerange=timerange, drop_incomplete=
        False, fill_up_missing=False)
    base = 0.001
    if what == 'raise':
        data.loc[:, 'open'] = data.index * base
        data.loc[:, 'high'] = data.index * base + 0.0001
        data.loc[:, 'low'] = data.index * base - 0.0001
        data.loc[:, 'close'] = data.index * base
    if what == 'lower':
        data.loc[:, 'open'] = 1 - data.index * base
        data.loc[:, 'high'] = 1 - data.index * base + 0.0001
        data.loc[:, 'low'] = 1 - data.index * base - 0.0001
        data.loc[:, 'close'] = 1 - data.index * base
    if what == 'sine':
        hz = 0.1
        data.loc[:, 'open'] = np.sin(data.index * hz) / 1000 + base
        data.loc[:, 'high'] = np.sin(data.index * hz) / 1000 + base + 0.0001
        data.loc[:, 'low'] = np.sin(data.index * hz) / 1000 + base - 0.0001
        data.loc[:, 'close'] = np.sin(data.index * hz) / 1000 + base
    return {'UNITTEST/BTC': clean_ohlcv_dataframe(data, timeframe='1m',
        pair='UNITTEST/BTC', fill_missing=True, drop_incomplete=True)}


def _make_backtest_conf(mocker: Any, datadir: Path, conf: Optional[Dict[str,
    Any]]=None, pair='UNITTEST/BTC') ->Dict[str, Any]:
    data = history.load_data(datadir=datadir, timeframe='1m', pairs=[pair])
    data = trim_dictlist(data, -201)
    patch_exchange(mocker)
    backtesting = Backtesting(conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    processed = backtesting.strategy.advise_all_indicators(data)
    min_date, max_date = get_timerange(processed)
    return {'processed': processed, 'start_date': min_date, 'end_date':
        max_date}


def _trend(signals: Dict[str, np.ndarray], buy_value: float, sell_value: float
    ) ->Dict[str, np.ndarray]:
    n = len(signals['low'])
    buy = np.zeros(n)
    sell = np.zeros(n)
    for i in range(0, len(signals['date'])):
        if random.random() > 0.5:
            buy[i] = buy_value
            sell[i] = sell_value
    signals['enter_long'] = buy
    signals['exit_long'] = sell
    signals['enter_short'] = 0
    signals['exit_short'] = 0
    return signals


def _trend_alternate(dataframe=None, metadata: Optional[Dict[str, Any]]=None
    ) ->pd.DataFrame:
    signals = dataframe
    low = signals['low']
    n = len(low)
    buy = np.zeros(n)
    sell = np.zeros(n)
    for i in range(0, len(buy)):
        if i % 2 == 0:
            buy[i] = 1
        else:
            sell[i] = 1
    signals['enter_long'] = buy
    signals['exit_long'] = sell
    signals['enter_short'] = 0
    signals['exit_short'] = 0
    return dataframe


def test_setup_optimize_configuration_without_arguments(mocker: Any,
    default_conf: Dict[str, Any], caplog: Any) ->None:
    patched_configuration_load_config_file(mocker, default_conf)
    args = ['backtesting', '--config', 'config.json', '--strategy',
        CURRENT_TEST_STRATEGY, '--export', 'none']
    config = setup_optimize_configuration(get_args(args), RunMode.BACKTEST)
    assert 'max_open_trades' in config
    assert 'stake_currency' in config
    assert 'stake_amount' in config
    assert 'exchange' in config
    assert 'pair_whitelist' in config['exchange']
    assert 'datadir' in config
    assert log_has('Using data directory: {} ...'.format(config['datadir']),
        caplog)
    assert 'timeframe' in config
    assert not log_has_re('Parameter -i/--ticker-interval detected .*', caplog)
    assert 'position_stacking' not in config
    assert not log_has('Parameter --enable-position-stacking detected ...',
        caplog)
    assert 'timerange' not in config
    assert 'export' in config
    assert config['export'] == 'none'
    assert 'runmode' in config
    assert config['runmode'] == RunMode.BACKTEST


def test_setup_bt_configuration_with_arguments(mocker: Any, default_conf:
    Dict[str, Any], caplog: Any) ->None:
    patched_configuration_load_config_file(mocker, default_conf)
    mocker.patch('freqtrade.configuration.configuration.create_datadir', lambda
        c, x: x)
    args = ['backtesting', '--config', 'config.json', '--strategy',
        CURRENT_TEST_STRATEGY, '--datadir', '/foo/bar', '--timeframe', '1m',
        '--enable-position-stacking', '--timerange', ':100',
        '--export-filename', 'foo_bar.json', '--fee', '0']
    config = setup_optimize_configuration(get_args(args), RunMode.BACKTEST)
    assert 'max_open_trades' in config
    assert 'stake_currency' in config
    assert 'stake_amount' in config
    assert 'exchange' in config
    assert 'pair_whitelist' in config['exchange']
    assert 'datadir' in config
    assert config['runmode'] == RunMode.BACKTEST
    assert log_has('Using data directory: {} ...'.format(config['datadir']),
        caplog)
    assert 'timeframe' in config
    assert log_has(
        'Parameter -i/--timeframe detected ... Using timeframe: 1m ...', caplog
        )
    assert 'position_stacking' in config
    assert log_has('Parameter --enable-position-stacking detected ...', caplog)
    assert 'timerange' in config
    assert log_has('Parameter --timerange detected: {} ...'.format(config[
        'timerange']), caplog)
    assert 'export' in config
    assert 'exportfilename' in config
    assert isinstance(config['exportfilename'], Path)
    assert log_has('Storing backtest results to {} ...'.format(config[
        'exportfilename']), caplog)
    assert 'fee' in config
    assert log_has('Parameter --fee detected, setting fee to: {} ...'.
        format(config['fee']), caplog)


def test_setup_optimize_configuration_stake_amount(mocker: Any,
    default_conf: Dict[str, Any], caplog: Any) ->None:
    patched_configuration_load_config_file(mocker, default_conf)
    args = ['backtesting', '--config', 'config.json', '--strategy',
        CURRENT_TEST_STRATEGY, '--stake-amount', '1', '--starting-balance', '2'
        ]
    conf = setup_optimize_configuration(get_args(args), RunMode.BACKTEST)
    assert isinstance(conf, dict)
    args = ['backtesting', '--config', 'config.json', '--strategy',
        CURRENT_TEST_STRATEGY, '--stake-amount', '1', '--starting-balance',
        '0.5']
    with pytest.raises(OperationalException, match=
        'Starting balance .* smaller .*'):
        setup_optimize_configuration(get_args(args), RunMode.BACKTEST)


def test_start(mocker: Any, fee: Any, default_conf: Dict[str, Any], caplog: Any
    ) ->None:
    start_mock = MagicMock()
    mocker.patch(f'{EXMS}.get_fee', fee)
    patch_exchange(mocker)
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.start', start_mock
        )
    patched_configuration_load_config_file(mocker, default_conf)
    args = ['backtesting', '--config', 'config.json', '--strategy',
        CURRENT_TEST_STRATEGY]
    pargs = get_args(args)
    start_backtesting(pargs)
    assert log_has('Starting freqtrade in Backtesting mode', caplog)
    assert start_mock.call_count == 1


@pytest.mark.parametrize('order_types', ORDER_TYPES)
def test_backtesting_init(mocker: Any, default_conf: Dict[str, Any],
    order_types: List[Dict[str, Union[str, bool]]]) ->None:
    """
    Check that stoploss_on_exchange is set to False while backtesting
    since backtesting assumes a perfect stoploss anyway.
    """
    default_conf['order_types'] = order_types
    patch_exchange(mocker)
    get_fee = mocker.patch(f'{EXMS}.get_fee', MagicMock(return_value=0.5))
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    assert backtesting.config == default_conf
    assert backtesting.timeframe == '5m'
    assert callable(backtesting.strategy.advise_all_indicators)
    assert callable(backtesting.strategy.advise_entry)
    assert callable(backtesting.strategy.advise_exit)
    assert isinstance(backtesting.strategy.dp, DataProvider)
    get_fee.assert_called()
    assert backtesting.fee == 0.5
    assert not backtesting.strategy.order_types['stoploss_on_exchange']
    assert backtesting.strategy.bot_started is True


def test_backtesting_init_no_timeframe(mocker: Any, default_conf: Dict[str,
    Any], caplog: Any) ->None:
    patch_exchange(mocker)
    del default_conf['timeframe']
    default_conf['strategy_list'] = [CURRENT_TEST_STRATEGY,
        'HyperoptableStrategy']
    mocker.patch(f'{EXMS}.get_fee', MagicMock(return_value=0.5))
    with pytest.raises(OperationalException, match=
        'Timeframe needs to be set in either configuration'):
        Backtesting(default_conf)


def test_data_with_fee(default_conf: Dict[str, Any], mocker: Any) ->None:
    patch_exchange(mocker)
    default_conf['fee'] = 0.01234
    fee_mock = mocker.patch(f'{EXMS}.get_fee', MagicMock(return_value=0.5))
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    assert backtesting.fee == 0.01234
    assert fee_mock.call_count == 0
    default_conf['fee'] = 0.0
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    assert backtesting.fee == 0.0
    assert fee_mock.call_count == 0


def test_data_to_dataframe_bt(default_conf: Dict[str, Any], mocker: Any,
    testdatadir: Path) ->None:
    patch_exchange(mocker)
    timerange = TimeRange.parse_timerange('1510694220-1510700340')
    data = history.load_data(testdatadir, '1m', ['UNITTEST/BTC'], timerange
        =timerange, fill_up_missing=True)
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    processed = backtesting.strategy.advise_all_indicators(data)
    assert len(processed['UNITTEST/BTC']) == 103
    strategy = StrategyResolver.load_strategy(default_conf)
    processed2 = strategy.advise_all_indicators(data)
    assert processed['UNITTEST/BTC'].equals(processed2['UNITTEST/BTC'])


def test_backtest_abort(default_conf: Dict[str, Any], mocker: Any,
    testdatadir: Path) ->None:
    patch_exchange(mocker)
    backtesting = Backtesting(default_conf)
    backtesting.check_abort()
    backtesting.abort = True
    with pytest.raises(DependencyException, match='Stop requested'):
        backtesting.check_abort()
    assert backtesting.abort is False
    assert backtesting.progress.progress == 0


def test_backtesting_start(default_conf: Dict[str, Any], mocker: Any,
    caplog: Any) ->None:

    def get_timerange(input1: Any) ->Tuple[datetime, datetime]:
        return dt_utc(2017, 11, 14, 21, 17), dt_utc(2017, 11, 14, 22, 59)
    mocker.patch('freqtrade.data.history.get_timerange', get_timerange)
    patch_exchange(mocker)
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.backtest')
    mocker.patch('freqtrade.optimize.backtesting.generate_backtest_stats')
    mocker.patch('freqtrade.optimize.backtesting.show_backtest_results')
    sbs = mocker.patch('freqtrade.optimize.backtesting.store_backtest_results')
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist',
        PropertyMock(return_value=['UNITTEST/BTC']))
    default_conf['timeframe'] = '1m'
    default_conf['export'] = 'signals'
    default_conf['exportfilename'] = 'export.txt'
    default_conf['timerange'] = '-1510694220'
    default_conf['runmode'] = RunMode.BACKTEST
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    backtesting.strategy.bot_loop_start = MagicMock()
    backtesting.strategy.bot_start = MagicMock()
    backtesting.start()
    exists = [
        'Backtesting with data from 2017-11-14 21:17:00 up to 2017-11-14 22:59:00 (0 days).'
        ]
    for line in exists:
        assert log_has(line, caplog)
    assert backtesting.strategy.dp._pairlists is not None
    assert backtesting.strategy.bot_start.call_count == 1
    assert backtesting.strategy.bot_loop_start.call_count == 0
    assert sbs.call_count == 1


def test_backtesting_start_no_data(default_conf: Dict[str, Any], mocker:
    Any, caplog: Any, testdatadir: Path) ->None:

    def get_timerange(input1: Any) ->Tuple[datetime, datetime]:
        return dt_utc(2017, 11, 14, 21, 17), dt_utc(2017, 11, 14, 22, 59)
    mocker.patch('freqtrade.data.history.history_utils.load_pair_history',
        MagicMock(return_value=pd.DataFrame()))
    mocker.patch('freqtrade.data.history.get_timerange', get_timerange)
    patch_exchange(mocker)
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.backtest')
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist',
        PropertyMock(return_value=['UNITTEST/BTC']))
    default_conf['timeframe'] = '1m'
    default
