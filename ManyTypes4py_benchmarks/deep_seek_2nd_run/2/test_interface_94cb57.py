import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock
from typing import Any, Dict, List, Optional, Tuple, Union

import pytest
from pandas import DataFrame
from freqtrade.configuration import TimeRange
from freqtrade.constants import CUSTOM_TAG_MAX_LENGTH
from freqtrade.data.dataprovider import DataProvider
from freqtrade.data.history import load_data
from freqtrade.enums import ExitCheckTuple, ExitType, HyperoptState, SignalDirection
from freqtrade.exceptions import OperationalException, StrategyError
from freqtrade.optimize.hyperopt_tools import HyperoptStateContainer
from freqtrade.optimize.space import SKDecimal
from freqtrade.persistence import PairLocks, Trade
from freqtrade.resolvers import StrategyResolver
from freqtrade.strategy.hyper import detect_parameters
from freqtrade.strategy.parameters import BaseParameter, BooleanParameter, CategoricalParameter, DecimalParameter, IntParameter, RealParameter
from freqtrade.util import dt_now
from tests.conftest import CURRENT_TEST_STRATEGY, TRADE_SIDES, log_has, log_has_re
from .strats.strategy_test_v3 import StrategyTestV3

_STRATEGY: StrategyTestV3 = StrategyTestV3(config={})
_STRATEGY.dp = DataProvider({}, None, None)

def test_returns_latest_signal(ohlcv_history: DataFrame) -> None:
    ohlcv_history.loc[1, 'date'] = dt_now()
    mocked_history: DataFrame = ohlcv_history.copy()
    mocked_history['enter_long'] = 0
    mocked_history['exit_long'] = 0
    mocked_history['enter_short'] = 0
    mocked_history['exit_short'] = 0
    mocked_history.loc[0, 'enter_tag'] = 'wrong_line'
    mocked_history.loc[0, 'exit_tag'] = 'wrong_line'
    mocked_history.loc[1, 'exit_long'] = 1
    assert _STRATEGY.get_entry_signal('ETH/BTC', '5m', mocked_history) == (None, None)
    assert _STRATEGY.get_exit_signal('ETH/BTC', '5m', mocked_history) == (False, True, None)
    assert _STRATEGY.get_exit_signal('ETH/BTC', '5m', mocked_history, True) == (False, False, None)
    mocked_history.loc[1, 'exit_long'] = 0
    mocked_history.loc[1, 'enter_long'] = 1
    assert _STRATEGY.get_entry_signal('ETH/BTC', '5m', mocked_history) == (SignalDirection.LONG, None)
    assert _STRATEGY.get_exit_signal('ETH/BTC', '5m', mocked_history) == (True, False, None)
    assert _STRATEGY.get_exit_signal('ETH/BTC', '5m', mocked_history, True) == (False, False, None)
    mocked_history.loc[1, 'exit_long'] = 0
    mocked_history.loc[1, 'enter_long'] = 0
    assert _STRATEGY.get_entry_signal('ETH/BTC', '5m', mocked_history) == (None, None)
    assert _STRATEGY.get_exit_signal('ETH/BTC', '5m', mocked_history) == (False, False, None)
    assert _STRATEGY.get_exit_signal('ETH/BTC', '5m', mocked_history, True) == (False, False, None)
    mocked_history.loc[1, 'exit_long'] = 0
    mocked_history.loc[1, 'enter_long'] = 1
    mocked_history.loc[1, 'enter_tag'] = 'buy_signal_01'
    assert _STRATEGY.get_entry_signal('ETH/BTC', '5m', mocked_history) == (SignalDirection.LONG, 'buy_signal_01')
    assert _STRATEGY.get_exit_signal('ETH/BTC', '5m', mocked_history) == (True, False, None)
    assert _STRATEGY.get_exit_signal('ETH/BTC', '5m', mocked_history, True) == (False, False, None)
    mocked_history.loc[1, 'exit_long'] = 0
    mocked_history.loc[1, 'enter_long'] = 0
    mocked_history.loc[1, 'enter_short'] = 1
    mocked_history.loc[1, 'exit_short'] = 0
    mocked_history.loc[1, 'enter_tag'] = 'sell_signal_01'
    assert _STRATEGY.get_entry_signal('ETH/BTC', '5m', mocked_history) == (None, None)
    _STRATEGY.config['trading_mode'] = 'futures'
    assert _STRATEGY.get_entry_signal('ETH/BTC', '5m', mocked_history) == (None, None)
    _STRATEGY.can_short = True
    assert _STRATEGY.get_entry_signal('ETH/BTC', '5m', mocked_history) == (SignalDirection.SHORT, 'sell_signal_01')
    assert _STRATEGY.get_exit_signal('ETH/BTC', '5m', mocked_history) == (False, False, None)
    assert _STRATEGY.get_exit_signal('ETH/BTC', '5m', mocked_history, True) == (True, False, None)
    mocked_history.loc[1, 'enter_short'] = 0
    mocked_history.loc[1, 'exit_short'] = 1
    mocked_history.loc[1, 'exit_tag'] = 'sell_signal_02'
    assert _STRATEGY.get_entry_signal('ETH/BTC', '5m', mocked_history) == (None, None)
    assert _STRATEGY.get_exit_signal('ETH/BTC', '5m', mocked_history) == (False, False, 'sell_signal_02')
    assert _STRATEGY.get_exit_signal('ETH/BTC', '5m', mocked_history, True) == (False, True, 'sell_signal_02')
    _STRATEGY.can_short = False
    _STRATEGY.config['trading_mode'] = 'spot'

def test_analyze_pair_empty(mocker: Any, caplog: Any, ohlcv_history: DataFrame) -> None:
    mocker.patch.object(_STRATEGY.dp, 'ohlcv', return_value=ohlcv_history)
    mocker.patch.object(_STRATEGY, '_analyze_ticker_internal', return_value=DataFrame([]))
    mocker.patch.object(_STRATEGY, 'assert_df')
    _STRATEGY.analyze_pair('ETH/BTC')
    assert log_has('Empty dataframe for pair ETH/BTC', caplog)

def test_get_signal_empty(default_conf: Dict[str, Any], caplog: Any) -> None:
    assert (None, None) == _STRATEGY.get_latest_candle('foo', default_conf['timeframe'], DataFrame())
    assert log_has('Empty candle (OHLCV) data for pair foo', caplog)
    caplog.clear()
    assert (None, None) == _STRATEGY.get_latest_candle('bar', default_conf['timeframe'], None)
    assert log_has('Empty candle (OHLCV) data for pair bar', caplog)
    caplog.clear()
    assert (None, None) == _STRATEGY.get_latest_candle('baz', default_conf['timeframe'], DataFrame([]))
    assert log_has('Empty candle (OHLCV) data for pair baz', caplog)

def test_get_signal_exception_valueerror(mocker: Any, caplog: Any, ohlcv_history: DataFrame) -> None:
    caplog.set_level(logging.INFO)
    mocker.patch.object(_STRATEGY.dp, 'ohlcv', return_value=ohlcv_history)
    mocker.patch.object(_STRATEGY, '_analyze_ticker_internal', side_effect=ValueError('xyz'))
    _STRATEGY.analyze_pair('foo')
    assert log_has_re('Strategy caused the following exception: xyz.*', caplog)
    caplog.clear()
    mocker.patch.object(_STRATEGY, 'analyze_ticker', side_effect=Exception('invalid ticker history '))
    _STRATEGY.analyze_pair('foo')
    assert log_has_re('Strategy caused the following exception: xyz.*', caplog)

def test_get_signal_old_dataframe(default_conf: Dict[str, Any], mocker: Any, caplog: Any, ohlcv_history: DataFrame) -> None:
    ohlcv_history.loc[1, 'date'] = dt_now() - timedelta(minutes=16)
    mocked_history: DataFrame = ohlcv_history.copy()
    mocked_history['exit_long'] = 0
    mocked_history['enter_long'] = 0
    mocked_history.loc[1, 'enter_long'] = 1
    caplog.set_level(logging.INFO)
    mocker.patch.object(_STRATEGY, 'assert_df')
    assert (None, None) == _STRATEGY.get_latest_candle('xyz', default_conf['timeframe'], mocked_history)
    assert log_has('Outdated history for pair xyz. Last tick is 16 minutes old', caplog)

def test_get_signal_no_sell_column(default_conf: Dict[str, Any], mocker: Any, caplog: Any, ohlcv_history: DataFrame) -> None:
    ohlcv_history.loc[1, 'date'] = dt_now()
    mocked_history: DataFrame = ohlcv_history.copy()
    mocked_history['enter_long'] = 0
    mocked_history.loc[1, 'enter_long'] = 1
    caplog.set_level(logging.INFO)
    mocker.patch.object(_STRATEGY, 'assert_df')
    assert (SignalDirection.LONG, None) == _STRATEGY.get_entry_signal('xyz', default_conf['timeframe'], mocked_history)

def test_ignore_expired_candle(default_conf: Dict[str, Any]) -> None:
    strategy = StrategyResolver.load_strategy(default_conf)
    strategy.ignore_buying_expired_candle_after = 60
    latest_date: datetime = datetime(2020, 12, 30, 7, 0, 0, tzinfo=timezone.utc)
    current_time: datetime = latest_date + timedelta(seconds=80 + 300)
    assert strategy.ignore_expired_candle(latest_date=latest_date, current_time=current_time, timeframe_seconds=300, enter=True) is True
    current_time = latest_date + timedelta(seconds=30 + 300)
    assert strategy.ignore_expired_candle(latest_date=latest_date, current_time=current_time, timeframe_seconds=300, enter=True) is not True

def test_assert_df_raise(mocker: Any, caplog: Any, ohlcv_history: DataFrame) -> None:
    ohlcv_history.loc[1, 'date'] = dt_now() - timedelta(minutes=16)
    mocked_history: DataFrame = ohlcv_history.copy()
    mocked_history['sell'] = 0
    mocked_history['buy'] = 0
    mocked_history.loc[1, 'buy'] = 1
    caplog.set_level(logging.INFO)
    mocker.patch.object(_STRATEGY.dp, 'ohlcv', return_value=ohlcv_history)
    mocker.patch.object(_STRATEGY.dp, 'get_analyzed_dataframe', return_value=(mocked_history, 0))
    mocker.patch.object(_STRATEGY, 'assert_df', side_effect=StrategyError('Dataframe returned...'))
    _STRATEGY.analyze_pair('xyz')
    assert log_has('Unable to analyze candle (OHLCV) data for pair xyz: Dataframe returned...', caplog)

def test_assert_df(ohlcv_history: DataFrame, caplog: Any) -> None:
    df_len: int = len(ohlcv_history) - 1
    ohlcv_history.loc[:, 'enter_long'] = 0
    ohlcv_history.loc[:, 'exit_long'] = 0
    _STRATEGY.assert_df(ohlcv_history, len(ohlcv_history), ohlcv_history.loc[df_len, 'close'], ohlcv_history.loc[df_len, 'date'])
    with pytest.raises(StrategyError, match='Dataframe returned from strategy.*length\\.'):
        _STRATEGY.assert_df(ohlcv_history, len(ohlcv_history) + 1, ohlcv_history.loc[df_len, 'close'], ohlcv_history.loc[df_len, 'date'])
    with pytest.raises(StrategyError, match='Dataframe returned from strategy.*last close price\\.'):
        _STRATEGY.assert_df(ohlcv_history, len(ohlcv_history), ohlcv_history.loc[df_len, 'close'] + 0.01, ohlcv_history.loc[df_len, 'date'])
    with pytest.raises(StrategyError, match='Dataframe returned from strategy.*last date\\.'):
        _STRATEGY.assert_df(ohlcv_history, len(ohlcv_history), ohlcv_history.loc[df_len, 'close'], ohlcv_history.loc[0, 'date'])
    with pytest.raises(StrategyError, match='No dataframe returned \\(return statement missing\\?\\).'):
        _STRATEGY.assert_df(None, len(ohlcv_history), ohlcv_history.loc[df_len, 'close'], ohlcv_history.loc[0, 'date'])
    _STRATEGY.disable_dataframe_checks = True
    caplog.clear()
    _STRATEGY.assert_df(ohlcv_history, len(ohlcv_history), ohlcv_history.loc[2, 'close'], ohlcv_history.loc[0, 'date'])
    assert log_has_re('Dataframe returned from strategy.*last date\\.', caplog)
    _STRATEGY.disable_dataframe_checks = False

def test_advise_all_indicators(default_conf: Dict[str, Any], testdatadir: Path) -> None:
    strategy = StrategyResolver.load_strategy(default_conf)
    timerange: TimeRange = TimeRange.parse_timerange('1510694220-1510700340')
    data: Dict[str, DataFrame] = load_data(testdatadir, '1m', ['UNITTEST/BTC'], timerange=timerange, fill_up_missing=True)
    processed: Dict[str, DataFrame] = strategy.advise_all_indicators(data)
    assert len(processed['UNITTEST/BTC']) == 103

def test_freqai_not_initialized(default_conf: Dict[str, Any]) -> None:
    strategy = StrategyResolver.load_strategy(default_conf)
    strategy.ft_bot_start()
    with pytest.raises(OperationalException, match='freqAI is not enabled\\.'):
        strategy.freqai.start()

def test_advise_all_indicators_copy(mocker: Any, default_conf: Dict[str, Any], testdatadir: Path) -> None:
    strategy = StrategyResolver.load_strategy(default_conf)
    aimock = mocker.patch('freqtrade.strategy.interface.IStrategy.advise_indicators')
    timerange: TimeRange = TimeRange.parse_timerange('1510694220-1510700340')
    data: Dict[str, DataFrame] = load_data(testdatadir, '1m', ['UNITTEST/BTC'], timerange=timerange, fill_up_missing=True)
    strategy.advise_all_indicators(data)
    assert aimock.call_count == 1
    assert aimock.call_args_list[0][0][0] is not data

def test_min_roi_reached(default_conf: Dict[str, Any], fee: Any) -> None:
    min_roi_list: List[Dict[int, float]] = [{20: 0.05, 55: 0.01, 0: 0.1}, {0: 0.1, 20: 0.05, 55: 0.01}]
    for roi in min_roi_list:
        strategy = StrategyResolver.load_strategy(default_conf)
        strategy.minimal_roi = roi
        trade = Trade(pair='ETH/BTC', stake_amount=0.001, amount=5, open_date=dt_now() - timedelta(hours=1), fee_open=fee.return_value, fee_close=fee.return_value, exchange='binance', open_rate=1)
        assert not strategy.min_roi_reached(trade, 0.02, dt_now() - timedelta(minutes=56))
        assert strategy.min_roi_reached(trade, 0.12, dt_now() - timedelta(minutes=56))
        assert not strategy.min_roi_reached(trade, 0.04, dt_now() - timedelta(minutes=39))
        assert strategy.min_roi_reached(trade, 0.06, dt_now() - timedelta(minutes=39))
        assert not strategy.min_roi_reached(trade, -0.01, dt_now() - timedelta(minutes=1))
        assert strategy.min_roi_reached(trade, 0.02, dt_now() - timedelta(minutes=1))

def test_min_roi_reached2(default_conf: Dict[str, Any], fee: Any) -> None:
    min_roi_list: List[Dict[int, float]] = [{20: 0.07, 30: 0.05, 55: 0.3, 0: 0.1}, {0: 0.1, 20: 0.07, 30: 0.05, 55: 0.3}]
    for roi in min_roi_list:
        strategy = StrategyResolver.load_strategy(default_conf)
        strategy.minimal_roi = roi
        trade = Trade(pair='ETH/BTC', stake_amount=0.001, amount=5, open_date=dt_now() - timedelta(hours=1), fee_open=fee.return_value, fee_close=fee.return_value, exchange='binance', open_rate=1)
        assert not strategy.min_roi_reached(trade, 0.02, dt_now() - timedelta(minutes=56))
        assert strategy.min_roi_reached(trade, 0.12,