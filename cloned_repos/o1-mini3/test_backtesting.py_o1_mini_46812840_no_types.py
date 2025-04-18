import random
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import ANY, MagicMock, PropertyMock
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import pytest
from pytest_mock import MockerFixture
from pytest import LogCaptureFixture, CaptureFixture
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
ORDER_TYPES: List[Dict[str, Union[str, bool]]] = [{'entry': 'limit', 'exit': 'limit', 'stoploss': 'limit', 'stoploss_on_exchange': False}, {'entry': 'limit', 'exit': 'limit', 'stoploss': 'limit', 'stoploss_on_exchange': True}]

def trim_dictlist(dict_list, num):
    new: Dict[str, pd.DataFrame] = {}
    for pair, pair_data in dict_list.items():
        new[pair] = pair_data[num:].reset_index()
    return new

def load_data_test(what, testdatadir):
    timerange = TimeRange.parse_timerange('1510694220-1510700340')
    data: pd.DataFrame = history.load_pair_history(pair='UNITTEST/BTC', datadir=testdatadir, timeframe='1m', timerange=timerange, drop_incomplete=False, fill_up_missing=False)
    base: float = 0.001
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
        hz: float = 0.1
        data.loc[:, 'open'] = np.sin(data.index * hz) / 1000 + base
        data.loc[:, 'high'] = np.sin(data.index * hz) / 1000 + base + 0.0001
        data.loc[:, 'low'] = np.sin(data.index * hz) / 1000 + base - 0.0001
        data.loc[:, 'close'] = np.sin(data.index * hz) / 1000 + base
    return {'UNITTEST/BTC': clean_ohlcv_dataframe(data, timeframe='1m', pair='UNITTEST/BTC', fill_missing=True, drop_incomplete=True)}

def _make_backtest_conf(mocker, datadir, conf=None, pair='UNITTEST/BTC'):
    data: Dict[str, pd.DataFrame] = history.load_data(datadir=datadir, timeframe='1m', pairs=[pair])
    data = trim_dictlist(data, -201)
    patch_exchange(mocker)
    backtesting: Backtesting = Backtesting(conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    processed: Dict[str, pd.DataFrame] = backtesting.strategy.advise_all_indicators(data)
    min_date: datetime
    max_date: datetime
    min_date, max_date = get_timerange(processed)
    return {'processed': processed, 'start_date': min_date, 'end_date': max_date}

def _trend(signals, buy_value, sell_value):
    n: int = len(signals['low'])
    buy: np.ndarray = np.zeros(n)
    sell: np.ndarray = np.zeros(n)
    for i in range(0, len(signals['date'])):
        if random.random() > 0.5:
            buy[i] = buy_value
            sell[i] = sell_value
    signals['enter_long'] = buy
    signals['exit_long'] = sell
    signals['enter_short'] = 0
    signals['exit_short'] = 0
    return signals

def _trend_alternate(dataframe=None, metadata=None):
    signals: pd.DataFrame = dataframe
    low: pd.Series = signals['low']
    n: int = len(low)
    buy: np.ndarray = np.zeros(n)
    sell: np.ndarray = np.zeros(n)
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

def test_setup_optimize_configuration_without_arguments(mocker, default_conf, caplog):
    patched_configuration_load_config_file(mocker, default_conf)
    args: List[str] = ['backtesting', '--config', 'config.json', '--strategy', CURRENT_TEST_STRATEGY, '--export', 'none']
    config: Dict[str, Any] = setup_optimize_configuration(get_args(args), RunMode.BACKTEST)
    assert 'max_open_trades' in config
    assert 'stake_currency' in config
    assert 'stake_amount' in config
    assert 'exchange' in config
    assert 'pair_whitelist' in config['exchange']
    assert 'datadir' in config
    assert log_has('Using data directory: {} ...'.format(config['datadir']), caplog)
    assert 'timeframe' in config
    assert not log_has_re('Parameter -i/--ticker-interval detected .*', caplog)
    assert 'position_stacking' not in config
    assert not log_has('Parameter --enable-position-stacking detected ...', caplog)
    assert 'timerange' not in config
    assert 'export' in config
    assert config['export'] == 'none'
    assert 'runmode' in config
    assert config['runmode'] == RunMode.BACKTEST

def test_setup_bt_configuration_with_arguments(mocker, default_conf, caplog):
    patched_configuration_load_config_file(mocker, default_conf)
    mocker.patch('freqtrade.configuration.configuration.create_datadir', lambda c, x: x)
    args: List[str] = ['backtesting', '--config', 'config.json', '--strategy', CURRENT_TEST_STRATEGY, '--datadir', '/foo/bar', '--timeframe', '1m', '--enable-position-stacking', '--timerange', ':100', '--export-filename', 'foo_bar.json', '--fee', '0']
    config: Dict[str, Any] = setup_optimize_configuration(get_args(args), RunMode.BACKTEST)
    assert 'max_open_trades' in config
    assert 'stake_currency' in config
    assert 'stake_amount' in config
    assert 'exchange' in config
    assert 'pair_whitelist' in config['exchange']
    assert 'datadir' in config
    assert config['runmode'] == RunMode.BACKTEST
    assert log_has('Using data directory: {} ...'.format(config['datadir']), caplog)
    assert 'timeframe' in config
    assert log_has('Parameter -i/--timeframe detected ... Using timeframe: 1m ...', caplog)
    assert 'position_stacking' in config
    assert log_has('Parameter --enable-position-stacking detected ...', caplog)
    assert 'timerange' in config
    assert log_has('Parameter --timerange detected: {} ...'.format(config['timerange']), caplog)
    assert 'export' in config
    assert 'exportfilename' in config
    assert isinstance(config['exportfilename'], Path)
    assert log_has('Storing backtest results to {} ...'.format(config['exportfilename']), caplog)
    assert 'fee' in config
    assert log_has('Parameter --fee detected, setting fee to: {} ...'.format(config['fee']), caplog)

def test_setup_optimize_configuration_stake_amount(mocker, default_conf, caplog):
    patched_configuration_load_config_file(mocker, default_conf)
    args: List[str] = ['backtesting', '--config', 'config.json', '--strategy', CURRENT_TEST_STRATEGY, '--stake-amount', '1', '--starting-balance', '2']
    conf: Dict[str, Any] = setup_optimize_configuration(get_args(args), RunMode.BACKTEST)
    assert isinstance(conf, dict)
    args = ['backtesting', '--config', 'config.json', '--strategy', CURRENT_TEST_STRATEGY, '--stake-amount', '1', '--starting-balance', '0.5']
    with pytest.raises(OperationalException, match='Starting balance .* smaller .*'):
        setup_optimize_configuration(get_args(args), RunMode.BACKTEST)

def test_start(mocker, fee, default_conf, caplog):
    start_mock: MagicMock = MagicMock()
    mocker.patch(f'{EXMS}.get_fee', fee)
    patch_exchange(mocker)
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.start', start_mock)
    patched_configuration_load_config_file(mocker, default_conf)
    args: List[str] = ['backtesting', '--config', 'config.json', '--strategy', CURRENT_TEST_STRATEGY]
    pargs = get_args(args)
    start_backtesting(pargs)
    assert log_has('Starting freqtrade in Backtesting mode', caplog)
    assert start_mock.call_count == 1

@pytest.mark.parametrize('order_types', ORDER_TYPES)
def test_backtesting_init(mocker, default_conf, order_types):
    """
    Check that stoploss_on_exchange is set to False while backtesting
    since backtesting assumes a perfect stoploss anyway.
    """
    default_conf['order_types'] = order_types
    patch_exchange(mocker)
    get_fee: MagicMock = mocker.patch(f'{EXMS}.get_fee', MagicMock(return_value=0.5))
    backtesting: Backtesting = Backtesting(default_conf)
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

def test_backtesting_init_no_timeframe(mocker, default_conf, caplog):
    patch_exchange(mocker)
    del default_conf['timeframe']
    default_conf['strategy_list'] = [CURRENT_TEST_STRATEGY, 'HyperoptableStrategy']
    mocker.patch(f'{EXMS}.get_fee', MagicMock(return_value=0.5))
    with pytest.raises(OperationalException, match='Timeframe needs to be set in either configuration'):
        Backtesting(default_conf)

def test_data_with_fee(default_conf, mocker):
    patch_exchange(mocker)
    default_conf['fee'] = 0.01234
    fee_mock: MagicMock = mocker.patch(f'{EXMS}.get_fee', MagicMock(return_value=0.5))
    backtesting: Backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    assert backtesting.fee == 0.01234
    assert fee_mock.call_count == 0
    default_conf['fee'] = 0.0
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    assert backtesting.fee == 0.0
    assert fee_mock.call_count == 0

def test_data_to_dataframe_bt(default_conf, mocker, testdatadir):
    patch_exchange(mocker)
    timerange: TimeRange = TimeRange.parse_timerange('1510694220-1510700340')
    data: Dict[str, pd.DataFrame] = history.load_data(testdatadir, '1m', ['UNITTEST/BTC'], timerange=timerange, fill_up_missing=True)
    backtesting: Backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    processed: Dict[str, pd.DataFrame] = backtesting.strategy.advise_all_indicators(data)
    assert len(processed['UNITTEST/BTC']) == 103
    strategy = StrategyResolver.load_strategy(default_conf)
    processed2: Dict[str, pd.DataFrame] = strategy.advise_all_indicators(data)
    assert processed['UNITTEST/BTC'].equals(processed2['UNITTEST/BTC'])

def test_backtest_abort(default_conf, mocker, testdatadir):
    patch_exchange(mocker)
    backtesting: Backtesting = Backtesting(default_conf)
    backtesting.check_abort()
    backtesting.abort = True
    with pytest.raises(DependencyException, match='Stop requested'):
        backtesting.check_abort()
    assert backtesting.abort is False
    assert backtesting.progress.progress == 0

def test_backtesting_start(default_conf, mocker, caplog):

    def get_timerange(input1):
        return (dt_utc(2017, 11, 14, 21, 17), dt_utc(2017, 11, 14, 22, 59))
    mocker.patch('freqtrade.data.history.get_timerange', get_timerange)
    patch_exchange(mocker)
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.backtest')
    mocker.patch('freqtrade.optimize.backtesting.generate_backtest_stats')
    mocker.patch('freqtrade.optimize.backtesting.show_backtest_results')
    sbs: MagicMock = mocker.patch('freqtrade.optimize.backtesting.store_backtest_results')
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist', PropertyMock(return_value=['UNITTEST/BTC']))
    default_conf['timeframe'] = '1m'
    default_conf['export'] = 'signals'
    default_conf['exportfilename'] = 'export.txt'
    default_conf['timerange'] = '-1510694220'
    default_conf['runmode'] = RunMode.BACKTEST
    backtesting: Backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    backtesting.strategy.bot_loop_start = MagicMock()
    backtesting.strategy.bot_start = MagicMock()
    backtesting.start()
    exists: List[str] = ['Backtesting with data from 2017-11-14 21:17:00 up to 2017-11-14 22:59:00 (0 days).']
    for line in exists:
        assert log_has(line, caplog)
    assert backtesting.strategy.dp._pairlists is not None
    assert backtesting.strategy.bot_start.call_count == 1
    assert backtesting.strategy.bot_loop_start.call_count == 0
    assert sbs.call_count == 1

@pytest.mark.parametrize('order_types', ORDER_TYPES)
def test_backtesting_init(mocker, default_conf, order_types):
    """
    Check that stoploss_on_exchange is set to False while backtesting
    since backtesting assumes a perfect stoploss anyway.
    """
    default_conf['order_types'] = order_types
    patch_exchange(mocker)
    get_fee: MagicMock = mocker.patch(f'{EXMS}.get_fee', MagicMock(return_value=0.5))
    backtesting: Backtesting = Backtesting(default_conf)
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

def test_backtesting_init_no_timeframe(mocker, default_conf, caplog):
    patch_exchange(mocker)
    del default_conf['timeframe']
    default_conf['strategy_list'] = [CURRENT_TEST_STRATEGY, 'HyperoptableStrategy']
    mocker.patch(f'{EXMS}.get_fee', MagicMock(return_value=0.5))
    with pytest.raises(OperationalException, match='Timeframe needs to be set in either configuration'):
        Backtesting(default_conf)

def test_data_with_fee(default_conf, mocker):
    patch_exchange(mocker)
    default_conf['fee'] = 0.01234
    fee_mock: MagicMock = mocker.patch(f'{EXMS}.get_fee', MagicMock(return_value=0.5))
    backtesting: Backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    assert backtesting.fee == 0.01234
    assert fee_mock.call_count == 0
    default_conf['fee'] = 0.0
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    assert backtesting.fee == 0.0
    assert fee_mock.call_count == 0

def test_data_to_dataframe_bt(default_conf, mocker, testdatadir):
    patch_exchange(mocker)
    timerange: TimeRange = TimeRange.parse_timerange('1510694220-1510700340')
    data: Dict[str, pd.DataFrame] = history.load_data(testdatadir, '1m', ['UNITTEST/BTC'], timerange=timerange, fill_up_missing=True)
    backtesting: Backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    processed: Dict[str, pd.DataFrame] = backtesting.strategy.advise_all_indicators(data)
    assert len(processed['UNITTEST/BTC']) == 103
    strategy = StrategyResolver.load_strategy(default_conf)
    processed2: Dict[str, pd.DataFrame] = strategy.advise_all_indicators(data)
    assert processed['UNITTEST/BTC'].equals(processed2['UNITTEST/BTC'])

def test_backtest_abort(default_conf, mocker, testdatadir):
    patch_exchange(mocker)
    backtesting: Backtesting = Backtesting(default_conf)
    backtesting.check_abort()
    backtesting.abort = True
    with pytest.raises(DependencyException, match='Stop requested'):
        backtesting.check_abort()
    assert backtesting.abort is False
    assert backtesting.progress.progress == 0

def test_backtesting_start(default_conf, mocker, caplog):

    def get_timerange(input1):
        return (dt_utc(2017, 11, 14, 21, 17), dt_utc(2017, 11, 14, 22, 59))
    mocker.patch('freqtrade.data.history.get_timerange', get_timerange)
    patch_exchange(mocker)
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.backtest')
    mocker.patch('freqtrade.optimize.backtesting.generate_backtest_stats')
    mocker.patch('freqtrade.optimize.backtesting.show_backtest_results')
    sbs: MagicMock = mocker.patch('freqtrade.optimize.backtesting.store_backtest_results')
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist', PropertyMock(return_value=['UNITTEST/BTC']))
    default_conf['timeframe'] = '1m'
    default_conf['export'] = 'signals'
    default_conf['exportfilename'] = 'export.txt'
    default_conf['timerange'] = '-1510694220'
    default_conf['runmode'] = RunMode.BACKTEST
    backtesting: Backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    backtesting.strategy.bot_loop_start = MagicMock()
    backtesting.strategy.bot_start = MagicMock()
    backtesting.start()
    exists: List[str] = ['Backtesting with data from 2017-11-14 21:17:00 up to 2017-11-14 22:59:00 (0 days).']
    for line in exists:
        assert log_has(line, caplog)
    assert backtesting.strategy.dp._pairlists is not None
    assert backtesting.strategy.bot_start.call_count == 1
    assert backtesting.strategy.bot_loop_start.call_count == 0
    assert sbs.call_count == 1

@pytest.mark.parametrize('use_detail', [True, False])
def test_backtest_one_detail(default_conf_usdt, mocker, testdatadir, use_detail):
    default_conf_usdt['use_exit_signal'] = False
    patch_exchange(mocker)
    mocker.patch(f'{EXMS}.get_min_pair_stake_amount', return_value=1e-05)
    mocker.patch(f'{EXMS}.get_max_pair_stake_amount', return_value=float('inf'))
    if use_detail:
        default_conf_usdt['timeframe_detail'] = '1m'

    def advise_entry(df, *args: Any, **kwargs: Any):
        df.loc[df['rsi'] < 40, 'enter_long'] = 1
        return df

    def custom_entry_price(proposed_rate, **kwargs: Any):
        return proposed_rate * 0.997
    default_conf_usdt['max_open_trades'] = 10
    backtesting: Backtesting = Backtesting(default_conf_usdt)
    backtesting._set_strategy(backtesting.strategylist[0])
    backtesting.strategy.populate_entry_trend = advise_entry
    backtesting.strategy.ignore_buying_expired_candle_after = 59
    backtesting.strategy.custom_entry_price = custom_entry_price
    pair: str = 'XRP/ETH'
    timerange: TimeRange = TimeRange.parse_timerange('20191010-20191013')
    data: Dict[str, pd.DataFrame] = history.load_data(datadir=testdatadir, timeframe='5m', pairs=[pair], timerange=timerange)
    if use_detail:
        data_1m: Dict[str, pd.DataFrame] = history.load_data(datadir=testdatadir, timeframe='1m', pairs=[pair], timerange=timerange)
        backtesting.detail_data = data_1m
    processed: Dict[str, pd.DataFrame] = backtesting.strategy.advise_all_indicators(data)
    min_date: datetime
    max_date: datetime
    min_date, max_date = get_timerange(processed)
    result: Dict[str, Any] = backtesting.backtest(processed=deepcopy(processed), start_date=min_date, end_date=max_date)
    results: pd.DataFrame = result['results']
    assert not results.empty
    assert len(results) == (2 if use_detail else 3)
    assert 'orders' in results.columns
    data_pair: pd.DataFrame = processed[pair]
    data_1m_pair: pd.DataFrame = data_1m[pair] if use_detail else pd.DataFrame()
    late_entry: int = 0
    for _, t in results.iterrows():
        assert len(t['orders']) == 2
        entryo: Dict[str, Any] = t['orders'][0]
        entry_ts: datetime = datetime.fromtimestamp(entryo['order_filled_timestamp'] // 1000, tz=timezone.utc)
        if entry_ts > t['open_date']:
            late_entry += 1
        ln: pd.DataFrame = data_1m_pair.loc[data_1m_pair['date'] == entry_ts] if use_detail else data_pair.loc[data_pair['date'] == entry_ts]
        assert not ln.empty
        assert round(ln.iloc[0]['low'], 6) <= round(t['open_rate'], 6) <= round(ln.iloc[0]['high'], 6)
        ln1: pd.DataFrame = data_pair.loc[data_pair['date'] == t['close_date']]
        if use_detail:
            ln1_1m: pd.DataFrame = data_1m_pair.loc[data_1m_pair['date'] == t['close_date']]
            assert not ln1.empty or not ln1_1m.empty
        else:
            assert not ln1.empty
        ln2: pd.DataFrame = ln1_1m if ln1.empty else ln1
        assert round(ln2.iloc[0]['low'], 6) <= round(t['close_rate'], 6) <= round(ln2.iloc[0]['high'], 6)
    assert late_entry > 0

@pytest.mark.parametrize('use_detail,exp_funding_fee, exp_ff_updates', [(True, -0.018054162, 10), (False, -0.01780296, 6)])
def test_backtest_one_detail_futures(default_conf_usdt, mocker, testdatadir, use_detail, exp_funding_fee, exp_ff_updates):
    default_conf_usdt['use_exit_signal'] = False
    default_conf_usdt['trading_mode'] = 'futures'
    default_conf_usdt['margin_mode'] = 'isolated'
    default_conf_usdt['candle_type_def'] = CandleType.FUTURES
    patch_exchange(mocker)
    mocker.patch(f'{EXMS}.get_min_pair_stake_amount', return_value=1e-05)
    mocker.patch(f'{EXMS}.get_max_pair_stake_amount', return_value=float('inf'))
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist', PropertyMock(return_value=['XRP/USDT:USDT']))
    mocker.patch(f'{EXMS}.get_maintenance_ratio_and_amt', return_value=(0.01, 0.01))
    default_conf_usdt['timeframe'] = '1h'
    if use_detail:
        default_conf_usdt['timeframe_detail'] = '5m'

    def advise_entry(df, *args: Any, **kwargs: Any):
        df.loc[df['rsi'] < 40, 'enter_long'] = 1
        return df

    def custom_entry_price(proposed_rate, **kwargs: Any):
        return proposed_rate * 0.997
    default_conf_usdt['max_open_trades'] = 10
    backtesting: Backtesting = Backtesting(default_conf_usdt)
    ff_spy: MagicMock = mocker.spy(backtesting.exchange, 'calculate_funding_fees')
    backtesting._set_strategy(backtesting.strategylist[0])
    backtesting.strategy.populate_entry_trend = advise_entry
    backtesting.strategy.custom_entry_price = custom_entry_price
    pair: str = 'XRP/USDT:USDT'
    timerange: TimeRange = TimeRange.parse_timerange('20211117-20211119')
    data: Dict[str, pd.DataFrame] = history.load_data(datadir=Path(testdatadir), timeframe='1h', pairs=[pair], timerange=timerange, candle_type=CandleType.FUTURES)
    backtesting.load_bt_data_detail()
    processed: Dict[str, pd.DataFrame] = backtesting.strategy.advise_all_indicators(data)
    min_date: datetime
    max_date: datetime
    min_date, max_date = get_timerange(processed)
    result: Dict[str, Any] = backtesting.backtest(processed=deepcopy(processed), start_date=min_date, end_date=max_date)
    results: pd.DataFrame = result['results']
    assert not results.empty
    assert len(results) == (4 if use_detail else 2)
    assert 'orders' in results.columns
    data_pair: pd.DataFrame = processed[pair]
    data_1m_pair: pd.DataFrame = backtesting.detail_data[pair] if use_detail else pd.DataFrame()
    late_entry: int = 0
    for _, t in results.iterrows():
        assert len(t['orders']) == 2
        entryo: Dict[str, Any] = t['orders'][0]
        entry_ts: datetime = datetime.fromtimestamp(entryo['order_filled_timestamp'] // 1000, tz=timezone.utc)
        if entry_ts > t['open_date']:
            late_entry += 1
        ln: pd.DataFrame = data_1m_pair.loc[data_1m_pair['date'] == entry_ts] if use_detail else data_pair.loc[data_pair['date'] == entry_ts]
        assert not ln.empty
        assert round(ln.iloc[0]['low'], 6) <= round(t['open_rate'], 6) <= round(ln.iloc[0]['high'], 6)
        ln1: pd.DataFrame = data_pair.loc[data_pair['date'] == t['close_date']]
        if use_detail:
            ln1_1m: pd.DataFrame = data_1m_pair.loc[data_1m_pair['date'] == t['close_date']]
            assert not ln1.empty or not ln1_1m.empty
        else:
            assert not ln1.empty
        ln2: pd.DataFrame = ln1_1m if ln1.empty else ln1
        assert round(ln2.iloc[0]['low'], 6) <= round(t['close_rate'], 6) <= round(ln2.iloc[0]['high'], 6)
    assert pytest.approx(Trade.bt_trades[1].funding_fees) == exp_funding_fee
    assert ff_spy.call_count == exp_ff_updates

@pytest.mark.parametrize('use_detail,entries,max_stake,ff_updates,expected_ff', [(True, 50, 3000, 55, -1.18038144), (False, 6, 360, 11, -0.14679994)])
def test_backtest_one_detail_futures_funding_fees(default_conf_usdt, mocker, testdatadir, use_detail, entries, max_stake, ff_updates, expected_ff):
    """
    Funding fees are expected to differ, as the maximum position size differs.
    """
    default_conf_usdt['use_exit_signal'] = False
    default_conf_usdt['trading_mode'] = 'futures'
    default_conf_usdt['margin_mode'] = 'isolated'
    default_conf_usdt['candle_type_def'] = CandleType.FUTURES
    default_conf_usdt['minimal_roi'] = {'0': 1}
    default_conf_usdt['dry_run_wallet'] = 100000
    mocker.patch(f'{EXMS}.get_fee', MagicMock(return_value=0.5))
    mocker.patch(f'{EXMS}.get_min_pair_stake_amount', return_value=1e-05)
    mocker.patch(f'{EXMS}.get_max_pair_stake_amount', return_value=float('inf'))
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist', PropertyMock(return_value=['XRP/USDT:USDT']))
    mocker.patch(f'{EXMS}.get_maintenance_ratio_and_amt', return_value=(0.01, 0.01))
    default_conf_usdt['timeframe'] = '1h'
    if use_detail:
        default_conf_usdt['timeframe_detail'] = '5m'
    patch_exchange(mocker)

    def advise_entry(df, *args: Any, **kwargs: Any):
        df.loc[:, 'enter_long'] = 1
        return df

    def adjust_trade_position(trade, current_time, **kwargs: Any):
        if current_time > datetime(2021, 11, 18, 2, 0, 0, tzinfo=timezone.utc):
            return None
        return default_conf_usdt['stake_amount']
    default_conf_usdt['max_open_trades'] = 1
    backtesting: Backtesting = Backtesting(default_conf_usdt)
    ff_spy: MagicMock = mocker.spy(backtesting.exchange, 'calculate_funding_fees')
    backtesting._set_strategy(backtesting.strategylist[0])
    backtesting.strategy.populate_entry_trend = advise_entry
    backtesting.strategy.adjust_trade_position = adjust_trade_position
    backtesting.strategy.leverage = MagicMock(return_value=5.0)
    mocker.patch(f'{EXMS}.get_maintenance_ratio_and_amt', return_value=(0.01, 0.01))
    pair: str = 'XRP/USDT:USDT'
    timerange: TimeRange = TimeRange.parse_timerange('20211117-20211119')
    data: Dict[str, pd.DataFrame] = history.load_data(datadir=Path(testdatadir), timeframe='1h', pairs=[pair], timerange=timerange, candle_type=CandleType.FUTURES)
    backtesting.load_bt_data_detail()
    processed: Dict[str, pd.DataFrame] = backtesting.strategy.advise_all_indicators(data)
    min_date: datetime
    max_date: datetime
    min_date, max_date = get_timerange(processed)
    result: Dict[str, Any] = backtesting.backtest(processed=deepcopy(processed), start_date=min_date, end_date=max_date)
    results: pd.DataFrame = result['results']
    assert not results.empty
    assert len(results) == (4 if use_detail else 2)
    assert 'orders' in results.columns
    data_pair: pd.DataFrame = processed[pair]
    data_1m_pair: pd.DataFrame = backtesting.detail_data[pair] if use_detail else pd.DataFrame()
    late_entry: int = 0
    for _, t in results.iterrows():
        assert len(t['orders']) == 2
        entryo: Dict[str, Any] = t['orders'][0]
        entry_ts: datetime = datetime.fromtimestamp(entryo['order_filled_timestamp'] // 1000, tz=timezone.utc)
        if entry_ts > t['open_date']:
            late_entry += 1
        ln: pd.DataFrame = data_1m_pair.loc[data_1m_pair['date'] == entry_ts] if use_detail else data_pair.loc[data_pair['date'] == entry_ts]
        assert not ln.empty
        assert round(ln.iloc[0]['low'], 6) <= round(t['open_rate'], 6) <= round(ln.iloc[0]['high'], 6)
        ln1: pd.DataFrame = data_pair.loc[data_pair['date'] == t['close_date']]
        if use_detail:
            ln1_1m: pd.DataFrame = data_1m_pair.loc[data_1m_pair['date'] == t['close_date']]
            assert not ln1.empty or not ln1_1m.empty
        else:
            assert not ln1.empty
        ln2: pd.DataFrame = ln1_1m if ln1.empty else ln1
        assert round(ln2.iloc[0]['low'], 6) <= round(t['close_rate'], 6) <= round(ln2.iloc[0]['high'], 6)
    assert pytest.approx(Trade.bt_trades[1].funding_fees) == exp_funding_fee
    assert ff_spy.call_count == exp_ff_updates

@pytest.mark.parametrize('use_detail,contour,expected', [(None, 'sine', 35), (None, 'raise', 19), (None, 'lower', 0), (None, 'sine', 35), (None, 'raise', 19), ([{'method': 'CooldownPeriod', 'stop_duration': 3}], 'sine', 9), ([{'method': 'CooldownPeriod', 'stop_duration': 3}], 'raise', 10), ([{'method': 'CooldownPeriod', 'stop_duration': 3}], 'lower', 0), ([{'method': 'CooldownPeriod', 'stop_duration': 3}], 'sine', 9), ([{'method': 'CooldownPeriod', 'stop_duration': 3}], 'raise', 10)])
def test_backtest_pricecontours(default_conf, mocker, testdatadir, protections, contour, expected):
    if protections:
        default_conf['_strategy_protections'] = protections
        default_conf['enable_protections'] = True
    patch_exchange(mocker)
    mocker.patch(f'{EXMS}.get_min_pair_stake_amount', return_value=1e-05)
    mocker.patch(f'{EXMS}.get_max_pair_stake_amount', return_value=float('inf'))
    default_conf['timeframe'] = '1m'
    backtesting: Backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    backtesting.strategy.max_open_trades = 1
    backtesting.config.update({'max_open_trades': 1})
    backtesting.strategy.max_open_trades = 1
    data: Dict[str, pd.DataFrame] = load_data_test(contour, testdatadir)
    processed: Dict[str, pd.DataFrame] = backtesting.strategy.advise_all_indicators(data)
    min_date: datetime
    max_date: datetime
    min_date, max_date = get_timerange(processed)
    assert isinstance(processed, dict)
    results: Dict[str, Any] = backtesting.backtest(processed=deepcopy(processed), start_date=min_date, end_date=max_date)
    assert len(results['results']) == expected

def test_backtest_clash_buy_sell(mocker, default_conf, testdatadir):

    def fun(dataframe, pair=None):
        buy_value: float = 1
        sell_value: float = 1
        return _trend(dataframe, buy_value, sell_value)
    default_conf['max_open_trades'] = 10
    backtest_conf: Dict[str, Any] = _make_backtest_conf(mocker, conf=default_conf, datadir=testdatadir)
    backtesting: Backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    backtesting.strategy.advise_entry = fun
    backtesting.strategy.advise_exit = fun
    result: Dict[str, Any] = backtesting.backtest(**backtest_conf)
    assert result['results'].empty

def test_backtest_only_sell(mocker, default_conf, testdatadir):

    def fun(dataframe, pair=None):
        buy_value: float = 0
        sell_value: float = 1
        return _trend(dataframe, buy_value, sell_value)
    default_conf['max_open_trades'] = 10
    backtest_conf: Dict[str, Any] = _make_backtest_conf(mocker, conf=default_conf, datadir=testdatadir)
    backtesting: Backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    backtesting.strategy.advise_entry = fun
    backtesting.strategy.advise_exit = fun
    result: Dict[str, Any] = backtesting.backtest(**backtest_conf)
    assert result['results'].empty

def test_backtest_alternate_buy_sell(default_conf, fee, mocker, testdatadir):
    mocker.patch(f'{EXMS}.get_min_pair_stake_amount', return_value=1e-05)
    mocker.patch(f'{EXMS}.get_max_pair_stake_amount', return_value=float('inf'))
    mocker.patch(f'{EXMS}.get_fee', fee)
    default_conf['max_open_trades'] = 10
    default_conf['runmode'] = 'backtest'
    backtest_conf: Dict[str, Any] = _make_backtest_conf(mocker, conf=default_conf, pair='UNITTEST/BTC', datadir=testdatadir)
    default_conf['timeframe'] = '1m'
    backtesting: Backtesting = Backtesting(default_conf)
    backtesting.required_startup = 0
    backtesting._set_strategy(backtesting.strategylist[0])
    backtesting.strategy.advise_entry = _trend_alternate
    backtesting.strategy.advise_exit = _trend_alternate
    result: Dict[str, Any] = backtesting.backtest(**backtest_conf)
    results: pd.DataFrame = result['results']
    assert len(results) == 100
    analyzed_df: pd.DataFrame = backtesting.dataprovider.get_analyzed_dataframe('UNITTEST/BTC', '1m')[0]
    assert len(analyzed_df) == 200
    expected_last_candle_date: datetime = backtest_conf['end_date'] - timedelta(minutes=1)
    assert analyzed_df.iloc[-1]['date'].to_pydatetime() == expected_last_candle_date
    assert len(results.loc[results['is_open']]) == 0

@pytest.mark.parametrize('pair', ['ADA/BTC', 'LTC/BTC'])
@pytest.mark.parametrize('tres', [0, 20, 30])
def test_backtest_multi_pair(default_conf, fee, mocker, tres, pair, testdatadir):

    def _trend_alternate_hold(dataframe, metadata=None):
        """
        Buy every xth candle - sell every other xth -2 (hold on to pairs a bit)
        """
        if metadata['pair'] in ('ETH/BTC', 'LTC/BTC'):
            multi: int = 20
        else:
            multi: int = 18
        dataframe['enter_long'] = np.where(dataframe.index % multi == 0, 1, 0)
        dataframe['exit_long'] = np.where((dataframe.index + multi - 2) % multi == 0, 1, 0)
        dataframe['enter_short'] = 0
        dataframe['exit_short'] = 0
        return dataframe
    default_conf['runmode'] = 'backtest'
    mocker.patch(f'{EXMS}.get_min_pair_stake_amount', return_value=1e-05)
    mocker.patch(f'{EXMS}.get_max_pair_stake_amount', return_value=float('inf'))
    mocker.patch(f'{EXMS}.get_fee', fee)
    patch_exchange(mocker)
    pairs: List[str] = ['ADA/BTC', 'DASH/BTC', 'ETH/BTC', 'LTC/BTC', 'NXT/BTC']
    data: Dict[str, pd.DataFrame] = history.load_data(datadir=testdatadir, timeframe='5m', pairs=pairs)
    data = trim_dictlist(data, -500)
    if tres > 0:
        data[pair] = data[pair][tres:].reset_index()
    default_conf['timeframe'] = '5m'
    default_conf['max_open_trades'] = 3
    backtesting: Backtesting = Backtesting(default_conf)
    vr_spy: MagicMock = mocker.spy(backtesting, 'validate_row')
    calls_per_candle: Dict[str, List[str]] = defaultdict(list)
    backtesting._set_strategy(backtesting.strategylist[0])
    backtesting.strategy.bot_loop_start = MagicMock()
    backtesting.strategy.advise_entry = _trend_alternate_hold
    backtesting.strategy.advise_exit = _trend_alternate_hold
    processed: Dict[str, pd.DataFrame] = backtesting.strategy.advise_all_indicators(data)
    min_date: datetime
    max_date: datetime
    min_date, max_date = get_timerange(processed)
    backtest_conf: Dict[str, Any] = {'processed': deepcopy(processed), 'start_date': min_date, 'end_date': max_date}
    results: Dict[str, Any] = backtesting.backtest(**backtest_conf)
    assert backtesting.strategy.bot_loop_start.call_count == 499
    assert vr_spy.call_count == 2495
    calls_per_candle = defaultdict(list)
    for call in vr_spy.call_args_list:
        calls_per_candle[call[0][3]].append(call[0][1])
    all_orients: List[List[str]] = [x for _, x in calls_per_candle.items()]
    distinct_calls: List[List[str]] = [list(x) for x in set((tuple(x) for x in all_orients))]
    assert all((len(x) == 5 for x in distinct_calls))
    assert not all((x == ['ADA/BTC', 'DASH/BTC', 'ETH/BTC', 'LTC/BTC', 'NXT/BTC'] for x in distinct_calls))
    assert any((x == ['ADA/BTC', 'DASH/BTC', 'ETH/BTC', 'LTC/BTC', 'NXT/BTC'] for x in distinct_calls))
    assert any((x == ['ETH/BTC', 'ADA/BTC', 'DASH/BTC', 'LTC/BTC', 'NXT/BTC'] for x in distinct_calls)) or any((x == ['ETH/BTC', 'LTC/BTC', 'ADA/BTC', 'DASH/BTC', 'NXT/BTC'] for x in distinct_calls))
    assert len(evaluate_result_multi(results['results'], '5m', 2)) > 0
    assert len(evaluate_result_multi(results['results'], '5m', 3)) == 0
    offset: int = 1 if tres == 0 else 0
    removed_candles: int = len(data[pair]) - offset
    assert len(backtesting.dataprovider.get_analyzed_dataframe(pair, '5m')[0]) == removed_candles
    assert len(backtesting.dataprovider.get_analyzed_dataframe('NXT/BTC', '5m')[0]) == len(data['NXT/BTC']) - 1
    backtesting.strategy.max_open_trades = 1
    backtesting.config.update({'max_open_trades': 1})
    backtest_conf = {'processed': deepcopy(processed), 'start_date': min_date, 'end_date': max_date}
    results = backtesting.backtest(**backtest_conf)
    assert len(evaluate_result_multi(results['results'], '5m', 1)) == 0

@pytest.mark.parametrize('use_detail', [True, False])
@pytest.mark.parametrize('pair', ['ADA/USDT', 'LTC/USDT'])
@pytest.mark.parametrize('tres', [0, 20, 30])
def test_backtest_multi_pair_detail(default_conf_usdt, mocker, testdatadir, use_detail, pair, tres):
    """
    literally the same as test_backtest_multi_pair - but with artificial data
    and detail timeframe.
    """

    def _trend_alternate_hold(dataframe, metadata=None):
        """
        Buy every xth candle - sell every other xth -2 (hold on to pairs a bit)
        """
        if metadata['pair'] in ('ETH/USDT', 'LTC/USDT'):
            multi: int = 20
        else:
            multi: int = 18
        dataframe['enter_long'] = np.where(dataframe.index % multi == 0, 1, 0)
        dataframe['exit_long'] = np.where((dataframe.index + multi - 2) % multi == 0, 1, 0)
        dataframe['enter_short'] = 0
        dataframe['exit_short'] = 0
        return dataframe
    default_conf_usdt.update({'runmode': 'backtest', 'stoploss': -1.0, 'minimal_roi': {'0': 1}})
    if use_detail:
        default_conf_usdt['timeframe_detail'] = '5m'
    mocker.patch(f'{EXMS}.get_min_pair_stake_amount', return_value=1e-05)
    mocker.patch(f'{EXMS}.get_max_pair_stake_amount', return_value=float('inf'))
    default_conf_usdt['timeframe'] = '5m'
    backtesting: Backtesting = Backtesting(default_conf_usdt)
    backtesting._set_strategy(backtesting.strategylist[0])
    backtesting.strategy.max_open_trades = 1
    backtesting.config.update({'max_open_trades': 1})
    backtesting.strategy.max_open_trades = 1
    data: Dict[str, pd.DataFrame] = load_data_test(contour='sine', testdatadir=testdatadir)
    processed: Dict[str, pd.DataFrame] = backtesting.strategy.advise_all_indicators(data)
    min_date: datetime
    max_date: datetime
    min_date, max_date = get_timerange(processed)
    assert isinstance(processed, dict)
    backtesting.strategy.max_open_trades = 1
    backtesting.config.update({'max_open_trades': 1})
    results: Dict[str, Any] = backtesting.backtest(processed=deepcopy(processed), start_date=min_date, end_date=max_date)
    assert len(results['results']) == 35
    assert backtesting.strategy.bot_loop_start.call_count == 499
    assert mocker.spy(backtesting, 'validate_row').call_count == 2495
    assert len(evaluate_result_multi(results['results'], '5m', 1)) == 0

@pytest.mark.parametrize('use_detail,exp_funding_fee, exp_ff_updates', [(True, -0.018054162, 10), (False, -0.01780296, 6)])
def test_backtest_one_detail_futures_funding_fees(default_conf_usdt, mocker, testdatadir, use_detail, exp_funding_fee, exp_ff_updates):
    default_conf_usdt['use_exit_signal'] = False
    default_conf_usdt['trading_mode'] = 'futures'
    default_conf_usdt['margin_mode'] = 'isolated'
    default_conf_usdt['candle_type_def'] = CandleType.FUTURES
    patch_exchange(mocker)
    mocker.patch(f'{EXMS}.get_min_pair_stake_amount', return_value=1e-05)
    mocker.patch(f'{EXMS}.get_max_pair_stake_amount', return_value=float('inf'))
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist', PropertyMock(return_value=['XRP/USDT:USDT']))
    mocker.patch(f'{EXMS}.get_maintenance_ratio_and_amt', return_value=(0.01, 0.01))
    default_conf_usdt['timeframe'] = '1h'
    if use_detail:
        default_conf_usdt['timeframe_detail'] = '5m'

    def advise_entry(df, *args: Any, **kwargs: Any):
        df.loc[:, 'enter_long'] = 1
        return df

    def custom_entry_price(proposed_rate, **kwargs: Any):
        return proposed_rate * 0.997
    default_conf_usdt['max_open_trades'] = 10
    backtesting: Backtesting = Backtesting(default_conf_usdt)
    ff_spy: MagicMock = mocker.spy(backtesting.exchange, 'calculate_funding_fees')
    backtesting._set_strategy(backtesting.strategylist[0])
    backtesting.strategy.populate_entry_trend = advise_entry
    backtesting.strategy.custom_entry_price = custom_entry_price
    pair: str = 'XRP/USDT:USDT'
    timerange: TimeRange = TimeRange.parse_timerange('20211117-20211119')
    data: Dict[str, pd.DataFrame] = history.load_data(datadir=Path(testdatadir), timeframe='1h', pairs=[pair], timerange=timerange, candle_type=CandleType.FUTURES)
    backtesting.load_bt_data_detail()
    processed: Dict[str, pd.DataFrame] = backtesting.strategy.advise_all_indicators(data)
    min_date: datetime
    max_date: datetime
    min_date, max_date = get_timerange(processed)
    result: Dict[str, Any] = backtesting.backtest(processed=deepcopy(processed), start_date=min_date, end_date=max_date)
    results: pd.DataFrame = result['results']
    assert not results.empty
    assert len(results) == (4 if use_detail else 2)
    assert 'orders' in results.columns
    data_pair: pd.DataFrame = processed[pair]
    data_1m_pair: pd.DataFrame = backtesting.detail_data[pair] if use_detail else pd.DataFrame()
    late_entry: int = 0
    for _, t in results.iterrows():
        assert len(t['orders']) == 2
        entryo: Dict[str, Any] = t['orders'][0]
        entry_ts: datetime = datetime.fromtimestamp(entryo['order_filled_timestamp'] // 1000, tz=timezone.utc)
        if entry_ts > t['open_date']:
            late_entry += 1
        ln: pd.DataFrame = data_1m_pair.loc[data_1m_pair['date'] == entry_ts] if use_detail else data_pair.loc[data_pair['date'] == entry_ts]
        assert not ln.empty
        assert round(ln.iloc[0]['low'], 6) <= round(t['open_rate'], 6) <= round(ln.iloc[0]['high'], 6)
        ln1: pd.DataFrame = data_pair.loc[data_pair['date'] == t['close_date']]
        if use_detail:
            ln1_1m: pd.DataFrame = data_1m_pair.loc[data_1m_pair['date'] == t['close_date']]
            assert not ln1.empty or not ln1_1m.empty
        else:
            assert not ln1.empty
        ln2: pd.DataFrame = ln1_1m if ln1.empty else ln1
        assert round(ln2.iloc[0]['low'], 6) <= round(t['close_rate'], 6) <= round(ln2.iloc[0]['high'], 6)
    assert pytest.approx(Trade.bt_trades[1].funding_fees) == exp_funding_fee
    assert ff_spy.call_count == exp_ff_updates

def test_backtest_one(default_conf, mocker, fee, testdatadir):
    default_conf['use_exit_signal'] = False
    default_conf['max_open_trades'] = 10
    patch_exchange(mocker)
    mocker.patch(f'{EXMS}.get_min_pair_stake_amount', return_value=1e-05)
    mocker.patch(f'{EXMS}.get_max_pair_stake_amount', return_value=float('inf'))
    backtesting: Backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    pair: str = 'UNITTEST/BTC'
    timerange: TimeRange = TimeRange('date', None, 1517227800, 0)
    data: Dict[str, pd.DataFrame] = history.load_data(datadir=testdatadir, timeframe='5m', pairs=[pair], timerange=timerange)
    processed: Dict[str, pd.DataFrame] = backtesting.strategy.advise_all_indicators(data)
    backtesting.strategy.order_filled = MagicMock()
    min_date: datetime
    max_date: datetime
    min_date, max_date = get_timerange(processed)
    result: Dict[str, Any] = backtesting.backtest(processed=deepcopy(processed), start_date=min_date, end_date=max_date)
    results: pd.DataFrame = result['results']
    assert not results.empty
    assert len(results) == 2
    expected: pd.DataFrame = pd.DataFrame({'pair': [pair, pair], 'stake_amount': [0.001, 0.001], 'max_stake_amount': [0.001, 0.001], 'amount': [0.00957442, 0.0097064], 'open_date': pd.to_datetime([dt_utc(2018, 1, 29, 18, 40, 0), dt_utc(2018, 1, 30, 3, 30, 0)], utc=True), 'close_date': pd.to_datetime([dt_utc(2018, 1, 29, 22, 35, 0), dt_utc(2018, 1, 30, 4, 10, 0)], utc=True), 'open_rate': [0.104445, 0.10302485], 'close_rate': [0.104969, 0.103541], 'fee_open': [0.0025, 0.0025], 'fee_close': [0.0025, 0.0025], 'trade_duration': [235, 40], 'profit_ratio': [0.0, 0.0], 'profit_abs': [0.0, 0.0], 'exit_reason': [ExitType.ROI.value, ExitType.ROI.value], 'initial_stop_loss_abs': [0.0940005, 0.09272236], 'initial_stop_loss_ratio': [-0.1, -0.1], 'stop_loss_abs': [0.0940005, 0.09272236], 'stop_loss_ratio': [-0.1, -0.1], 'min_rate': [0.10370188, 0.10300000000000001], 'max_rate': [0.10501, 0.1038888], 'is_open': [False, False], 'enter_tag': ['', ''], 'leverage': [1.0, 1.0], 'is_short': [False, False], 'open_timestamp': [1517251200000, 1517283000000], 'close_timestamp': [1517265300000, 1517285400000], 'orders': [[{'amount': 0.00957442, 'safe_price': 0.104445, 'ft_order_side': 'buy', 'order_filled_timestamp': 1517251200000, 'ft_is_entry': True, 'ft_order_tag': '', 'cost': ANY}, {'amount': 0.00957442, 'safe_price': 0.10496853383458644, 'ft_order_side': 'sell', 'order_filled_timestamp': 1517265300000, 'ft_is_entry': False, 'ft_order_tag': 'roi', 'cost': ANY}], [{'amount': 0.0097064, 'safe_price': 0.10302485, 'ft_order_side': 'buy', 'order_filled_timestamp': 1517283000000, 'ft_is_entry': True, 'ft_order_tag': '', 'cost': ANY}, {'amount': 0.0097064, 'safe_price': 0.10354126528822055, 'ft_order_side': 'sell', 'order_filled_timestamp': 1517285400000, 'ft_is_entry': False, 'ft_order_tag': 'roi', 'cost': ANY}]]})
    pd.testing.assert_frame_equal(results, expected)
    assert 'orders' in results.columns
    data_pair: pd.DataFrame = processed[pair]
    assert backtesting.strategy.order_filled.call_count == 4
    for _, t in results.iterrows():
        assert len(t['orders']) == 2
        ln: pd.DataFrame = data_pair.loc[data_pair['date'] == t['open_date']]
        assert not ln.empty
        assert round(ln.iloc[0]['open'], 6) == round(t['open_rate'], 6)
        ln1: pd.DataFrame = data_pair.loc[data_pair['date'] == t['close_date']]
        assert round(ln1.iloc[0]['open'], 6) == round(t['close_rate'], 6) or round(ln1.iloc[0]['low'], 6) < round(t['close_rate'], 6) < round(ln1.iloc[0]['high'], 6)

@pytest.mark.parametrize('use_detail', [True, False])
def test_backtest_one_detail_futures(default_conf_usdt, mocker, testdatadir, use_detail):
    pass

def test_backtest_start_timerange(default_conf, mocker, caplog, testdatadir):
    patch_exchange(mocker)
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.backtest')
    mocker.patch('freqtrade.optimize.backtesting.generate_backtest_stats')
    mocker.patch('freqtrade.optimize.backtesting.show_backtest_results')
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist', PropertyMock(return_value=['UNITTEST/BTC']))
    patched_configuration_load_config_file(mocker, default_conf)
    args: List[str] = ['backtesting', '--config', 'config.json', '--strategy', CURRENT_TEST_STRATEGY, '--datadir', str(testdatadir), '--timeframe', '1m', '--timerange', '1510694220-1510700340', '--enable-position-stacking']
    args = get_args(args)
    start_backtesting(args)
    exists: List[str] = ['Parameter -i/--timeframe detected ... Using timeframe: 1m ...', 'Parameter --timerange detected: 1510694220-1510700340 ...', f'Using data directory: {testdatadir} ...', 'Loading data from 2017-11-14 20:57:00 up to 2017-11-14 22:59:00 (0 days).', 'Backtesting with data from 2017-11-14 21:17:00 up to 2017-11-14 22:59:00 (0 days).', 'Parameter --enable-position-stacking detected ...']
    for line in exists:
        assert log_has(line, caplog)

@pytest.mark.filterwarnings('ignore:deprecated')
def test_backtest_start_multi_strat(default_conf, mocker, caplog, testdatadir, capsys):
    default_conf.update({'use_exit_signal': True, 'exit_profit_only': False, 'exit_profit_offset': 0.0, 'ignore_roi_if_entry_signal': False})
    patch_exchange(mocker)
    result1: pd.DataFrame = pd.DataFrame({'pair': ['XRP/BTC', 'LTC/BTC'], 'profit_ratio': [0.0, 0.0], 'profit_abs': [0.0, 0.0], 'open_date': pd.to_datetime(['2018-01-29 18:40:00', '2018-01-30 03:30:00'], utc=True), 'close_date': pd.to_datetime(['2018-01-29 20:45:00', '2018-01-30 05:35:00'], utc=True), 'trade_duration': [235, 40], 'is_open': [False, False], 'stake_amount': [0.01, 0.01], 'open_rate': [0.104445, 0.10302485], 'close_rate': [0.104969, 0.103541], 'is_short': [False, False], 'exit_reason': [ExitType.ROI.value, ExitType.ROI.value]})
    result2: pd.DataFrame = pd.DataFrame({'pair': ['XRP/BTC', 'LTC/BTC', 'ETH/BTC'], 'profit_ratio': [0.03, 0.01, 0.1], 'profit_abs': [0.01, 0.02, 0.2], 'open_date': pd.to_datetime(['2018-01-29 18:40:00', '2018-01-30 03:30:00', '2018-01-30 05:30:00'], utc=True), 'close_date': pd.to_datetime(['2018-01-29 20:45:00', '2018-01-30 05:35:00', '2018-01-30 08:30:00'], utc=True), 'trade_duration': [47, 40, 20], 'is_open': [False, False, False], 'stake_amount': [0.01, 0.01, 0.01], 'open_rate': [0.104445, 0.10302485, 0.122541], 'close_rate': [0.104969, 0.103541, 0.123541], 'is_short': [False, False, False], 'exit_reason': [ExitType.ROI.value, ExitType.ROI.value, ExitType.STOP_LOSS.value]})
    backtestmock: MagicMock = MagicMock(side_effect=[{'results': result1, 'config': default_conf, 'locks': [], 'rejected_signals': 20, 'timedout_entry_orders': 0, 'timedout_exit_orders': 0, 'canceled_trade_entries': 0, 'canceled_entry_orders': 0, 'replaced_entry_orders': 0, 'final_balance': 1000}, {'results': result2, 'config': default_conf, 'locks': [], 'rejected_signals': 20, 'timedout_entry_orders': 0, 'timedout_exit_orders': 0, 'canceled_trade_entries': 0, 'canceled_entry_orders': 0, 'replaced_entry_orders': 0, 'final_balance': 1000}])
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist', PropertyMock(return_value=['UNITTEST/BTC']))
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.backtest', backtestmock)
    text_table_mock: MagicMock = MagicMock()
    tag_metrics_mock: MagicMock = MagicMock()
    strattable_mock: MagicMock = MagicMock()
    strat_summary: MagicMock = MagicMock()
    mocker.patch.multiple('freqtrade.optimize.optimize_reports.bt_output', text_table_bt_results=text_table_mock, text_table_strategy=strattable_mock)
    mocker.patch.multiple('freqtrade.optimize.optimize_reports.optimize_reports', generate_pair_metrics=MagicMock(), generate_tag_metrics=tag_metrics_mock, generate_strategy_comparison=strat_summary, generate_daily_stats=MagicMock())
    patched_configuration_load_config_file(mocker, default_conf)
    args: List[str] = ['backtesting', '--config', 'config.json', '--datadir', str(testdatadir), '--strategy-path', str(Path(__file__).parents[1] / 'strategy/strats'), '--timeframe', '1m', '--timerange', '1510694220-1510700340', '--enable-position-stacking', '--strategy-list', CURRENT_TEST_STRATEGY, 'StrategyTestV2']
    args = get_args(args)
    start_backtesting(args)
    assert backtestmock.call_count == 2
    assert text_table_mock.call_count == 4
    assert strattable_mock.call_count == 1
    assert tag_metrics_mock.call_count == 6
    assert strat_summary.call_count == 1
    exists: List[str] = ['Parameter -i/--timeframe detected ... Using timeframe: 1m ...', 'Parameter --timerange detected: 1510694220-1510700340 ...', f'Using data directory: {testdatadir} ...', 'Loading data from 2017-11-14 20:57:00 up to 2017-11-14 22:59:00 (0 days).', 'Backtesting with data from 2017-11-14 21:17:00 up to 2017-11-14 22:59:00 (0 days).', 'Parameter --enable-position-stacking detected ...', f'Running backtesting for Strategy {CURRENT_TEST_STRATEGY}', 'Running backtesting for Strategy StrategyTestV2']
    for line in exists:
        assert log_has(line, caplog)
    captured: str = capsys.readouterr().out
    assert 'BACKTESTING REPORT' in captured
    assert 'EXIT REASON STATS' in captured
    assert 'DAY BREAKDOWN' in captured
    assert 'LEFT OPEN TRADES REPORT' in captured
    assert '2017-11-14 21:17:00 -> 2017-11-14 22:59:00 | Max open trades : 1' in captured
    assert 'STRATEGY SUMMARY' in captured

def test_backtest_start_multi_strat_nomock_detail(default_conf, mocker, caplog, testdatadir, capsys):
    default_conf.update({'use_exit_signal': True, 'exit_profit_only': False, 'exit_profit_offset': 0.0, 'ignore_roi_if_entry_signal': False})
    patch_exchange(mocker)
    result1: pd.DataFrame = pd.DataFrame({'pair': ['XRP/BTC', 'LTC/BTC'], 'profit_ratio': [0.0, 0.0], 'profit_abs': [0.0, 0.0], 'open_date': pd.to_datetime(['2018-01-29 18:40:00', '2018-01-30 03:30:00'], utc=True), 'close_date': pd.to_datetime(['2018-01-29 20:45:00', '2018-01-30 05:35:00'], utc=True), 'trade_duration': [235, 40], 'is_open': [False, False], 'stake_amount': [0.01, 0.01], 'open_rate': [0.104445, 0.10302485], 'close_rate': [0.104969, 0.103541], 'is_short': [False, False], 'exit_reason': [ExitType.ROI.value, ExitType.ROI.value]})
    result2: pd.DataFrame = pd.DataFrame({'pair': ['XRP/BTC', 'LTC/BTC', 'ETH/BTC'], 'profit_ratio': [0.03, 0.01, 0.1], 'profit_abs': [0.01, 0.02, 0.2], 'open_date': pd.to_datetime(['2018-01-29 18:40:00', '2018-01-30 03:30:00', '2018-01-30 05:30:00'], utc=True), 'close_date': pd.to_datetime(['2018-01-29 20:45:00', '2018-01-30 05:35:00', '2018-01-30 08:30:00'], utc=True), 'trade_duration': [47, 40, 20], 'is_open': [False, False, False], 'is_short': [False, False, False], 'stake_amount': [0.01, 0.01, 0.01], 'open_rate': [0.104445, 0.10302485, 0.122541], 'close_rate': [0.104969, 0.103541, 0.123541], 'is_short': [False, False, False], 'exit_reason': [ExitType.ROI.value, ExitType.ROI.value, ExitType.STOP_LOSS.value]})
    backtestmock: MagicMock = MagicMock(side_effect=[{'results': result1, 'config': default_conf, 'locks': [], 'rejected_signals': 20, 'timedout_entry_orders': 0, 'timedout_exit_orders': 0, 'canceled_trade_entries': 0, 'canceled_entry_orders': 0, 'replaced_entry_orders': 0, 'final_balance': 1000}, {'results': result2, 'config': default_conf, 'locks': [], 'rejected_signals': 20, 'timedout_entry_orders': 0, 'timedout_exit_orders': 0, 'canceled_trade_entries': 0, 'canceled_entry_orders': 0, 'replaced_entry_orders': 0, 'final_balance': 1000}])
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist', PropertyMock(return_value=['XRP/ETH']))
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.backtest', backtestmock)
    patched_configuration_load_config_file(mocker, default_conf)
    args: List[str] = ['backtesting', '--config', 'config.json', '--datadir', str(testdatadir), '--strategy-path', str(Path(__file__).parents[1] / 'strategy/strats'), '--timeframe', '5m', '--timeframe-detail', '1m', '--strategy-list', CURRENT_TEST_STRATEGY]
    args = get_args(args)
    start_backtesting(args)
    exists: List[str] = ['Parameter -i/--timeframe detected ... Using timeframe: 5m ...', 'Parameter --timeframe-detail detected, using 1m for intra-candle backtesting ...', f'Using data directory: {testdatadir} ...', 'Loading data from 2019-10-11 00:00:00 up to 2019-10-13 11:15:00 (2 days).', 'Backtesting with data from 2019-10-11 01:40:00 up to 2019-10-13 11:15:00 (2 days).', f'Running backtesting for Strategy {CURRENT_TEST_STRATEGY}']
    for line in exists:
        assert log_has(line, caplog)
    captured: str = capsys.readouterr().out
    assert 'BACKTESTING REPORT' in captured
    assert 'EXIT REASON STATS' in captured
    assert 'LEFT OPEN TRADES REPORT' in captured

@pytest.mark.filterwarnings('ignore:deprecated')
def test_backtest_start_futures_noliq(default_conf_usdt, mocker, caplog, testdatadir, capsys):
    default_conf_usdt.update({'trading_mode': 'futures', 'margin_mode': 'isolated', 'use_exit_signal': True, 'exit_profit_only': False, 'exit_profit_offset': 0.0, 'ignore_roi_if_entry_signal': False, 'strategy': CURRENT_TEST_STRATEGY})
    patch_exchange(mocker)
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist', PropertyMock(return_value=['HULUMULU/USDT', 'XRP/USDT:USDT']))
    patched_configuration_load_config_file(mocker, default_conf_usdt)
    args: List[str] = ['backtesting', '--config', 'config.json', '--strategy', CURRENT_TEST_STRATEGY, '--datadir', str(testdatadir), '--strategy-path', str(Path(__file__).parents[1] / 'strategy/strats'), '--timeframe', '1h']
    args = get_args(args)
    with pytest.raises(OperationalException, match='Pairs .* got no leverage tiers available\\.'):
        start_backtesting(args)

@pytest.mark.filterwarnings('ignore:deprecated')
def test_backtest_start_nomock_futures(default_conf_usdt, mocker, caplog, testdatadir, capsys):
    default_conf_usdt.update({'trading_mode': 'futures', 'margin_mode': 'isolated', 'use_exit_signal': True, 'exit_profit_only': False, 'exit_profit_offset': 0.0, 'ignore_roi_if_entry_signal': False, 'strategy': CURRENT_TEST_STRATEGY})
    patch_exchange(mocker)
    result1: pd.DataFrame = pd.DataFrame({'pair': ['XRP/USDT:USDT', 'XRP/USDT:USDT'], 'profit_ratio': [0.0, 0.0], 'profit_abs': [0.0, 0.0], 'open_date': pd.to_datetime(['2021-11-18 18:00:00', '2021-11-18 03:00:00'], utc=True), 'close_date': pd.to_datetime(['2021-11-18 20:00:00', '2021-11-18 05:00:00'], utc=True), 'trade_duration': [235, 40], 'is_open': [False, False], 'is_short': [False, False], 'stake_amount': [0.01, 0.01], 'open_rate': [0.104445, 0.10302485], 'close_rate': [0.104969, 0.103541], 'exit_reason': [ExitType.ROI, ExitType.ROI]})
    result2: pd.DataFrame = pd.DataFrame({'pair': ['XRP/USDT:USDT', 'XRP/USDT:USDT', 'XRP/USDT:USDT'], 'profit_ratio': [0.03, 0.01, 0.1], 'profit_abs': [0.01, 0.02, 0.2], 'open_date': pd.to_datetime(['2021-11-19 18:00:00', '2021-11-19 03:00:00', '2021-11-19 05:00:00'], utc=True), 'close_date': pd.to_datetime(['2021-11-19 20:00:00', '2021-11-19 05:00:00', '2021-11-19 08:00:00'], utc=True), 'trade_duration': [47, 40, 20], 'is_open': [False, False, False], 'is_short': [False, False, False], 'stake_amount': [0.01, 0.01, 0.01], 'open_rate': [0.104445, 0.10302485, 0.122541], 'close_rate': [0.104969, 0.103541, 0.123541], 'exit_reason': [ExitType.ROI, ExitType.ROI, ExitType.STOP_LOSS]})
    backtestmock: MagicMock = MagicMock(side_effect=[{'results': result1, 'config': default_conf_usdt, 'locks': [], 'rejected_signals': 20, 'timedout_entry_orders': 0, 'timedout_exit_orders': 0, 'canceled_trade_entries': 0, 'canceled_entry_orders': 0, 'replaced_entry_orders': 0, 'final_balance': 1000}, {'results': result2, 'config': default_conf_usdt, 'locks': [], 'rejected_signals': 20, 'timedout_entry_orders': 0, 'timedout_exit_orders': 0, 'canceled_trade_entries': 0, 'canceled_entry_orders': 0, 'replaced_entry_orders': 0, 'final_balance': 1000}])
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist', PropertyMock(return_value=['XRP/USDT:USDT']))
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.backtest', backtestmock)
    text_table_mock: MagicMock = MagicMock()
    tag_metrics_mock: MagicMock = MagicMock()
    strattable_mock: MagicMock = MagicMock()
    strat_summary: MagicMock = MagicMock()
    mocker.patch.multiple('freqtrade.optimize.optimize_reports.bt_output', text_table_bt_results=text_table_mock, text_table_strategy=strattable_mock)
    mocker.patch.multiple('freqtrade.optimize.optimize_reports.optimize_reports', generate_pair_metrics=MagicMock(), generate_tag_metrics=tag_metrics_mock, generate_strategy_comparison=strat_summary, generate_daily_stats=MagicMock())
    patched_configuration_load_config_file(mocker, default_conf_usdt)
    args: List[str] = ['backtesting', '--config', 'config.json', '--datadir', str(testdatadir), '--strategy-path', str(Path(__file__).parents[1] / 'strategy/strats'), '--timeframe', '1h']
    args = get_args(args)
    start_backtesting(args)
    exists: List[str] = ['Parameter -i/--timeframe detected ... Using timeframe: 1h ...', f'Using data directory: {testdatadir} ...', 'Loading data from 2021-11-17 01:00:00 up to 2021-11-21 04:00:00 (4 days).', 'Backtesting with data from 2021-11-17 21:00:00 up to 2021-11-21 04:00:00 (3 days).', 'XRP/USDT:USDT, funding_rate, 8h, data starts at 2021-11-18 00:00:00', 'XRP/USDT:USDT, mark, 8h, data starts at 2021-11-18 00:00:00', f'Running backtesting for Strategy {CURRENT_TEST_STRATEGY}']
    for line in exists:
        assert log_has(line, caplog)
    captured: str = capsys.readouterr().out
    assert 'BACKTESTING REPORT' in captured
    assert 'EXIT REASON STATS' in captured
    assert 'LEFT OPEN TRADES REPORT' in captured

@pytest.mark.parametrize('run_id,timerange,start_delta', [('2', TimeRange.parse_timerange('1510694220-1510700340'), {'days': 0}), ('changed', TimeRange.parse_timerange('1510694220-1510700340'), {'days': 1})])
@pytest.mark.parametrize('cache', constants.BACKTEST_CACHE_AGE)
def test_backtest_start_multi_strat_caching(default_conf, mocker, caplog, testdatadir, run_id, timerange, start_delta, cache):
    default_conf.update({'use_exit_signal': True, 'exit_profit_only': False, 'exit_profit_offset': 0.0, 'ignore_roi_if_entry_signal': False})
    patch_exchange(mocker)
    backtestmock: MagicMock = MagicMock(return_value={'results': pd.DataFrame(columns=BT_DATA_COLUMNS), 'config': default_conf, 'locks': [], 'rejected_signals': 20, 'timedout_entry_orders': 0, 'timedout_exit_orders': 0, 'canceled_trade_entries': 0, 'canceled_entry_orders': 0, 'replaced_entry_orders': 0, 'final_balance': 1000})
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist', PropertyMock(return_value=['UNITTEST/BTC']))
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.backtest', backtestmock)
    mocker.patch('freqtrade.optimize.backtesting.show_backtest_results', MagicMock())
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist', PropertyMock(return_value=['UNITTEST/BTC']))
    now: datetime = datetime.now(tz=timezone.utc)
    start_time: datetime = now - timedelta(**start_delta) + timedelta(hours=1)
    if cache == 'none':
        min_backtest_date: datetime = now + timedelta(days=1)
    elif cache == 'day':
        min_backtest_date = now - timedelta(days=1)
    elif cache == 'week':
        min_backtest_date = now - timedelta(weeks=1)
    elif cache == 'month':
        min_backtest_date = now - timedelta(weeks=4)
    load_backtest_metadata: MagicMock = MagicMock(return_value={'StrategyTestV2': {'run_id': '1', 'backtest_start_time': now.timestamp()}, 'StrategyTestV3': {'run_id': run_id, 'backtest_start_time': start_time.timestamp()}})
    load_backtest_stats: MagicMock = MagicMock(side_effect=[{'metadata': {'StrategyTestV2': {'run_id': '1'}}, 'strategy': {'StrategyTestV2': {}}, 'strategy_comparison': [{'key': 'StrategyTestV2'}]}, {'metadata': {'StrategyTestV3': {'run_id': '2'}}, 'strategy': {'StrategyTestV3': {}}, 'strategy_comparison': [{'key': 'StrategyTestV3'}]}])
    mocker.patch('pathlib.Path.glob', return_value=[Path(datetime.strftime(datetime.now(), 'backtest-result-%Y-%m-%d_%H-%M-%S.json'))])
    mocker.patch.multiple('freqtrade.data.btanalysis', load_backtest_metadata=load_backtest_metadata, load_backtest_stats=load_backtest_stats)
    mocker.patch('freqtrade.optimize.backtesting.get_strategy_run_id', side_effect=['1', '2', '2'])
    patched_configuration_load_config_file(mocker, default_conf)
    args: List[str] = ['backtesting', '--config', 'config.json', '--datadir', str(testdatadir), '--strategy-path', str(Path(__file__).parents[1] / 'strategy/strats'), '--timeframe', '1m', '--timerange', '1510694220-1510700340', '--enable-position-stacking', '--cache', cache, '--strategy-list', 'StrategyTestV2', 'StrategyTestV3']
    args = get_args(args)
    start_backtesting(args)
    exists: List[str] = ['Parameter -i/--timeframe detected ... Using timeframe: 1m ...', 'Parameter --timerange detected: 1510694220-1510700340 ...', f'Using data directory: {testdatadir} ...', 'Loading data from 2017-11-14 20:57:00 up to 2017-11-14 22:59:00 (0 days).', 'Parameter --enable-position-stacking detected ...']
    for line in exists:
        assert log_has(line, caplog)
    if cache == 'none':
        assert backtestmock.call_count == 2
        exists = ['Running backtesting for Strategy StrategyTestV2', 'Running backtesting for Strategy StrategyTestV3', 'Backtesting with data from 2017-11-14 21:17:00 up to 2017-11-14 22:59:00 (0 days).']
    elif run_id == '2' and datetime.fromtimestamp(min_backtest_date.timestamp(), tz=timezone.utc) < start_time:
        assert backtestmock.call_count == 0
        exists = ['Reusing result of previous backtest for StrategyTestV2', 'Reusing result of previous backtest for StrategyTestV3']
    else:
        exists = ['Reusing result of previous backtest for StrategyTestV2', 'Running backtesting for Strategy StrategyTestV3', 'Backtesting with data from 2017-11-14 21:17:00 up to 2017-11-14 22:59:00 (0 days).']
        assert backtestmock.call_count == 1
    for line in exists:
        assert log_has(line, caplog)

def test_get_strategy_run_id(default_conf_usdt):
    default_conf_usdt.update({'strategy': 'StrategyTestV2', 'max_open_trades': float('inf')})
    strategy = StrategyResolver.load_strategy(default_conf_usdt)
    x: str = get_strategy_run_id(strategy)
    assert isinstance(x, str)

def test_get_backtest_metadata_filename():
    filename: Union[Path, str] = Path('backtest_results.json')
    expected: Path = Path('backtest_results.meta.json')
    assert get_backtest_metadata_filename(filename) == expected
    filename = Path('/path/to/backtest.results.json')
    expected = Path('/path/to/backtest.results.meta.json')
    assert get_backtest_metadata_filename(filename) == expected
    filename = Path('backtest_results.json')
    expected = Path('backtest_results.meta.json')
    assert get_backtest_metadata_filename(filename) == expected
    filename = '/path/to/backtest_results.json'
    expected = Path('/path/to/backtest_results.meta.json')
    assert get_backtest_metadata_filename(filename) == expected
    filename = '/path/to/backtest_results'
    expected = Path('/path/to/backtest_results.meta.json')
    assert get_backtest_metadata_filename(filename) == expected
    filename = '/path/to/backtest.results.json'
    expected = Path('/path/to/backtest.results.meta.json')
    assert get_backtest_metadata_filename(filename) == expected
    filename = 'backtest_results.json'
    expected = Path('backtest_results.meta.json')
    assert get_backtest_metadata_filename(filename) == expected
    filename = 'backtest_results_zip.zip'
    expected = Path('backtest_results_zip.meta.json')
    assert get_backtest_metadata_filename(filename) == expected

def test_backtest_start_futures_noliq(default_conf_usdt, mocker, caplog, testdatadir, capsys):
    default_conf_usdt.update({'trading_mode': 'futures', 'margin_mode': 'isolated', 'use_exit_signal': True, 'exit_profit_only': False, 'exit_profit_offset': 0.0, 'ignore_roi_if_entry_signal': False, 'strategy': CURRENT_TEST_STRATEGY})
    patch_exchange(mocker)
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist', PropertyMock(return_value=['HULUMULU/USDT', 'XRP/USDT:USDT']))
    patched_configuration_load_config_file(mocker, default_conf_usdt)
    args: List[str] = ['backtesting', '--config', 'config.json', '--strategy', CURRENT_TEST_STRATEGY, '--datadir', str(testdatadir), '--strategy-path', str(Path(__file__).parents[1] / 'strategy/strats'), '--timeframe', '1h']
    args = get_args(args)
    with pytest.raises(OperationalException, match='Pairs .* got no leverage tiers available\\.'):
        start_backtesting(args)

def test_backtest_start_multi_strat_nomock_futures(default_conf_usdt, mocker, caplog, testdatadir, capsys):
    default_conf_usdt.update({'runmode': 'backtest', 'trading_mode': 'futures', 'margin_mode': 'isolated', 'use_exit_signal': True, 'exit_profit_only': False, 'exit_profit_offset': 0.0, 'ignore_roi_if_entry_signal': False, 'strategy': CURRENT_TEST_STRATEGY})
    patch_exchange(mocker)
    result1: pd.DataFrame = pd.DataFrame({'pair': ['XRP/USDT:USDT', 'XRP/USDT:USDT'], 'profit_ratio': [0.0, 0.0], 'profit_abs': [0.0, 0.0], 'open_date': pd.to_datetime(['2021-11-18 18:00:00', '2021-11-18 03:00:00'], utc=True), 'close_date': pd.to_datetime(['2021-11-18 20:00:00', '2021-11-18 05:00:00'], utc=True), 'trade_duration': [235, 40], 'is_open': [False, False], 'is_short': [False, False], 'stake_amount': [0.01, 0.01], 'open_rate': [0.104445, 0.10302485], 'close_rate': [0.104969, 0.103541], 'exit_reason': [ExitType.ROI, ExitType.ROI]})
    result2: pd.DataFrame = pd.DataFrame({'pair': ['XRP/USDT:USDT', 'XRP/USDT:USDT', 'XRP/USDT:USDT'], 'profit_ratio': [0.03, 0.01, 0.1], 'profit_abs': [0.01, 0.02, 0.2], 'open_date': pd.to_datetime(['2021-11-19 18:00:00', '2021-11-19 03:00:00', '2021-11-19 05:00:00'], utc=True), 'close_date': pd.to_datetime(['2021-11-19 20:00:00', '2021-11-19 05:00:00', '2021-11-19 08:00:00'], utc=True), 'trade_duration': [47, 40, 20], 'is_open': [False, False, False], 'is_short': [False, False, False], 'stake_amount': [0.01, 0.01, 0.01], 'open_rate': [0.104445, 0.10302485, 0.122541], 'close_rate': [0.104969, 0.103541, 0.123541], 'exit_reason': [ExitType.ROI, ExitType.ROI, ExitType.STOP_LOSS]})
    backtestmock: MagicMock = MagicMock(side_effect=[{'results': result1, 'config': default_conf_usdt, 'locks': [], 'rejected_signals': 20, 'timedout_entry_orders': 0, 'timedout_exit_orders': 0, 'canceled_trade_entries': 0, 'canceled_entry_orders': 0, 'replaced_entry_orders': 0, 'final_balance': 1000}, {'results': result2, 'config': default_conf_usdt, 'locks': [], 'rejected_signals': 20, 'timedout_entry_orders': 0, 'timedout_exit_orders': 0, 'canceled_trade_entries': 0, 'canceled_entry_orders': 0, 'replaced_entry_orders': 0, 'final_balance': 1000}])
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist', PropertyMock(return_value=['XRP/USDT:USDT']))
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.backtest', backtestmock)
    text_table_mock: MagicMock = MagicMock()
    tag_metrics_mock: MagicMock = MagicMock()
    strattable_mock: MagicMock = MagicMock()
    strat_summary: MagicMock = MagicMock()
    mocker.patch.multiple('freqtrade.optimize.optimize_reports.bt_output', text_table_bt_results=text_table_mock, text_table_strategy=strattable_mock)
    mocker.patch.multiple('freqtrade.optimize.optimize_reports.optimize_reports', generate_pair_metrics=MagicMock(), generate_tag_metrics=tag_metrics_mock, generate_strategy_comparison=strat_summary, generate_daily_stats=MagicMock())
    patched_configuration_load_config_file(mocker, default_conf_usdt)
    args: List[str] = ['backtesting', '--config', 'config.json', '--datadir', str(testdatadir), '--strategy-path', str(Path(__file__).parents[1] / 'strategy/strats'), '--timeframe', '5m', '--timeframe-detail', '1m', '--strategy-list', CURRENT_TEST_STRATEGY]
    args = get_args(args)
    start_backtesting(args)
    exists: List[str] = ['Parameter -i/--timeframe detected ... Using timeframe: 5m ...', 'Parameter --timeframe-detail detected, using 1m for intra-candle backtesting ...', f'Using data directory: {testdatadir} ...', 'Loading data from 2019-10-11 00:00:00 up to 2019-10-13 11:15:00 (2 days).', 'Backtesting with data from 2019-10-11 01:40:00 up to 2019-10-13 11:15:00 (2 days).', f'Running backtesting for Strategy {CURRENT_TEST_STRATEGY}']
    for line in exists:
        assert log_has(line, caplog)
    captured: str = capsys.readouterr().out
    assert 'BACKTESTING REPORT' in captured
    assert 'EXIT REASON STATS' in captured
    assert 'LEFT OPEN TRADES REPORT' in captured

@pytest.mark.parametrize('use_detail', [True, False])
@pytest.mark.parametrize('pair', ['ADA/USDT', 'LTC/USDT'])
@pytest.mark.parametrize('tres', [0, 20, 30])
def test_backtest_multi_pair_detail_simplified(default_conf_usdt, mocker, testdatadir, use_detail, pair, tres):
    """
    literally the same as test_backtest_multi_pair_detail
    but with an "always enter" strategy, exiting after about half of the candle duration.
    """

    def _always_buy(dataframe, metadata=None):
        """
        Buy every candle.
        """
        dataframe['enter_long'] = 1
        dataframe['enter_short'] = 0
        dataframe['exit_short'] = 0
        return dataframe

    def custom_exit(trade, current_time, **kwargs: Any):
        if trade.open_date_utc + timedelta(minutes=20) < current_time:
            return 'exit after 20 minutes'
    default_conf_usdt.update({'runmode': 'backtest', 'stoploss': -1.0, 'minimal_roi': {'0': 100}})
    if use_detail:
        default_conf_usdt['timeframe_detail'] = '5m'
    mocker.patch(f'{EXMS}.get_min_pair_stake_amount', return_value=1e-05)
    mocker.patch(f'{EXMS}.get_max_pair_stake_amount', return_value=float('inf'))
    mocker.patch(f'{EXMS}.get_fee', MagicMock(return_value=0.5))
    patch_exchange(mocker)
    raw_candles_5m: pd.DataFrame = generate_test_data('5m', 1000, '2022-01-03 12:00:00+00:00')
    raw_candles: pd.DataFrame = ohlcv_fill_up_missing_data(raw_candles_5m, '1h', 'dummy')
    pairs: List[str] = ['ADA/USDT', 'DASH/USDT', 'ETH/USDT', 'LTC/USDT', 'NXT/USDT']
    data: Dict[str, pd.DataFrame] = {pair: raw_candles for pair in pairs}
    detail_data: Dict[str, pd.DataFrame] = {pair: raw_candles_5m for pair in pairs}
    data = trim_dictlist(data, -200)
    if tres > 0:
        data[pair] = data[pair][tres:].reset_index()
    default_conf_usdt['timeframe'] = '1h'
    default_conf_usdt['max_open_trades'] = 3
    backtesting: Backtesting = Backtesting(default_conf_usdt)
    vr_spy: MagicMock = mocker.spy(backtesting, 'validate_row')
    bl_spy: MagicMock = mocker.spy(backtesting, 'backtest_loop')
    backtesting.detail_data = detail_data
    backtesting.funding_fee_timeframe_secs = 3600 * 8
    backtesting.futures_data = {pair: pd.DataFrame() for pair in pairs}
    backtesting.strategylist[0].can_short = True
    backtesting._set_strategy(backtesting.strategylist[0])
    backtesting.strategy.bot_loop_start = MagicMock()
    backtesting.strategy.advise_entry = _always_buy
    backtesting.strategy.advise_exit = _always_buy
    backtesting.strategy.custom_exit = custom_exit
    processed: Dict[str, pd.DataFrame] = backtesting.strategy.advise_all_indicators(data)
    min_date: datetime
    max_date: datetime
    min_date, max_date = get_timerange(processed)
    backtest_conf: Dict[str, Any] = {'processed': deepcopy(processed), 'start_date': min_date, 'end_date': max_date}
    result: Dict[str, Any] = backtesting.backtest(**backtest_conf)
    results: pd.DataFrame = result['results']
    assert not results.empty
    assert len(results) == 1
    assert 'orders' in results.columns
    assert bl_spy.call_count == 10
    exists: List[str] = ['Backtesting with data from 2017-11-14 21:17:00 up to 2017-11-14 22:59:00 (0 days).']
    for line in exists:
        assert log_has(line, caplog)
    assert backtesting.strategy.dp._pairlists is not None
    assert backtesting.strategy.bot_start.call_count == 1
    assert backtesting.strategy.bot_loop_start.call_count == 0
    assert mocker.spy(backtesting, 'store_backtest_results').call_count == 1
    assert len(results.loc[results['is_open']]) == 0
    assert len(results['orders']) == 2
    assert round(results.iloc[0]['profit_ratio'], 6) == 0.0

@pytest.mark.parametrize('protections,contour,expected', [(None, 'sine', 35), (None, 'raise', 19), (None, 'lower', 0), (None, 'sine', 35), (None, 'raise', 19), ([{'method': 'CooldownPeriod', 'stop_duration': 3}], 'sine', 9), ([{'method': 'CooldownPeriod', 'stop_duration': 3}], 'raise', 10), ([{'method': 'CooldownPeriod', 'stop_duration': 3}], 'lower', 0), ([{'method': 'CooldownPeriod', 'stop_duration': 3}], 'sine', 9), ([{'method': 'CooldownPeriod', 'stop_duration': 3}], 'raise', 10)])
def test_backtest_pricecontours(default_conf, mocker, testdatadir, protections, contour, expected):
    if protections:
        default_conf['_strategy_protections'] = protections
        default_conf['enable_protections'] = True
    patch_exchange(mocker)
    mocker.patch(f'{EXMS}.get_min_pair_stake_amount', return_value=1e-05)
    mocker.patch(f'{EXMS}.get_max_pair_stake_amount', return_value=float('inf'))
    default_conf['timeframe'] = '1m'
    backtesting: Backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    backtesting.strategy.max_open_trades = 1
    backtesting.config.update({'max_open_trades': 1})
    backtesting.strategy.max_open_trades = 1
    data: Dict[str, pd.DataFrame] = load_data_test(contour, testdatadir)
    processed: Dict[str, pd.DataFrame] = backtesting.strategy.advise_all_indicators(data)
    min_date: datetime
    max_date: datetime
    min_date, max_date = get_timerange(processed)
    assert isinstance(processed, dict)
    results: Dict[str, Any] = backtesting.backtest(processed=deepcopy(processed), start_date=min_date, end_date=max_date)
    assert len(results['results']) == expected