# pragma pylint: disable=missing-docstring, C0103
import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import MagicMock

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
from freqtrade.strategy.parameters import (
    BaseParameter,
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
)
from freqtrade.util import dt_now
from tests.conftest import CURRENT_TEST_STRATEGY, TRADE_SIDES, log_has, log_has_re

from .strats.strategy_test_v3 import StrategyTestV3


# Avoid to reinit the same object again and again
_STRATEGY: StrategyTestV3 = StrategyTestV3(config={})
_STRATEGY.dp: DataProvider = DataProvider({}, None, None)


def test_returns_latest_signal(ohlcv_history: DataFrame) -> None:
    ohlcv_history.loc[1, "date"] = dt_now()
    # Take a copy to correctly modify the call
    mocked_history: DataFrame = ohlcv_history.copy()
    mocked_history["enter_long"] = 0
    mocked_history["exit_long"] = 0
    mocked_history["enter_short"] = 0
    mocked_history["exit_short"] = 0
    # Set tags in lines that don't matter to test nan in the sell line
    mocked_history.loc[0, "enter_tag"] = "wrong_line"
    mocked_history.loc[0, "exit_tag"] = "wrong_line"
    mocked_history.loc[1, "exit_long"] = 1

    assert _STRATEGY.get_entry_signal("ETH/BTC", "5m", mocked_history) == (None, None)
    assert _STRATEGY.get_exit_signal("ETH/BTC", "5m", mocked_history) == (False, True, None)
    assert _STRATEGY.get_exit_signal("ETH/BTC", "5m", mocked_history, True) == (False, False, None)
    mocked_history.loc[1, "exit_long"] = 0
    mocked_history.loc[1, "enter_long"] = 1

    assert _STRATEGY.get_entry_signal("ETH/BTC", "5m", mocked_history) == (
        SignalDirection.LONG,
        None,
    )
    assert _STRATEGY.get_exit_signal("ETH/BTC", "5m", mocked_history) == (True, False, None)
    assert _STRATEGY.get_exit_signal("ETH/BTC", "5m", mocked_history, True) == (False, False, None)
    mocked_history.loc[1, "exit_long"] = 0
    mocked_history.loc[1, "enter_long"] = 0

    assert _STRATEGY.get_entry_signal("ETH/BTC", "5m", mocked_history) == (None, None)
    assert _STRATEGY.get_exit_signal("ETH/BTC", "5m", mocked_history) == (False, False, None)
    assert _STRATEGY.get_exit_signal("ETH/BTC", "5m", mocked_history, True) == (False, False, None)
    mocked_history.loc[1, "exit_long"] = 0
    mocked_history.loc[1, "enter_long"] = 1
    mocked_history.loc[1, "enter_tag"] = "buy_signal_01"

    assert _STRATEGY.get_entry_signal("ETH/BTC", "5m", mocked_history) == (
        SignalDirection.LONG,
        "buy_signal_01",
    )
    assert _STRATEGY.get_exit_signal("ETH/BTC", "5m", mocked_history) == (True, False, None)
    assert _STRATEGY.get_exit_signal("ETH/BTC", "5m", mocked_history, True) == (False, False, None)

    mocked_history.loc[1, "exit_long"] = 0
    mocked_history.loc[1, "enter_long"] = 0
    mocked_history.loc[1, "enter_short"] = 1
    mocked_history.loc[1, "exit_short"] = 0
    mocked_history.loc[1, "enter_tag"] = "sell_signal_01"

    # Don't provide short signal while in spot mode
    assert _STRATEGY.get_entry_signal("ETH/BTC", "5m", mocked_history) == (None, None)

    _STRATEGY.config["trading_mode"] = "futures"
    # Short signal gets ignored as can_short is not set.
    assert _STRATEGY.get_entry_signal("ETH/BTC", "5m", mocked_history) == (None, None)

    _STRATEGY.can_short = True

    assert _STRATEGY.get_entry_signal("ETH/BTC", "5m", mocked_history) == (
        SignalDirection.SHORT,
        "sell_signal_01",
    )
    assert _STRATEGY.get_exit_signal("ETH/BTC", "5m", mocked_history) == (False, False, None)
    assert _STRATEGY.get_exit_signal("ETH/BTC", "5m", mocked_history, True) == (True, False, None)

    mocked_history.loc[1, "enter_short"] = 0
    mocked_history.loc[1, "exit_short"] = 1
    mocked_history.loc[1, "exit_tag"] = "sell_signal_02"
    assert _STRATEGY.get_entry_signal("ETH/BTC", "5m", mocked_history) == (None, None)
    assert _STRATEGY.get_exit_signal("ETH/BTC", "5m", mocked_history) == (
        False,
        False,
        "sell_signal_02",
    )
    assert _STRATEGY.get_exit_signal("ETH/BTC", "5m", mocked_history, True) == (
        False,
        True,
        "sell_signal_02",
    )

    _STRATEGY.can_short = False
    _STRATEGY.config["trading_mode"] = "spot"


def test_analyze_pair_empty(mocker: Any, caplog: Any, ohlcv_history: DataFrame) -> None:
    mocker.patch.object(_STRATEGY.dp, "ohlcv", return_value=ohlcv_history)
    mocker.patch.object(_STRATEGY, "_analyze_ticker_internal", return_value=DataFrame([]))
    mocker.patch.object(_STRATEGY, "assert_df")

    _STRATEGY.analyze_pair("ETH/BTC")

    assert log_has("Empty dataframe for pair ETH/BTC", caplog)


def test_get_signal_empty(default_conf: Dict[str, Any], caplog: Any) -> None:
    assert (None, None) == _STRATEGY.get_latest_candle(
        "foo", default_conf["timeframe"], DataFrame()
    )
    assert log_has("Empty candle (OHLCV) data for pair foo", caplog)
    caplog.clear()

    assert (None, None) == _STRATEGY.get_latest_candle("bar", default_conf["timeframe"], None)
    assert log_has("Empty candle (OHLCV) data for pair bar", caplog)
    caplog.clear()

    assert (None, None) == _STRATEGY.get_latest_candle(
        "baz", default_conf["timeframe"], DataFrame([])
    )
    assert log_has("Empty candle (OHLCV) data for pair baz", caplog)


def test_get_signal_exception_valueerror(mocker: Any, caplog: Any, ohlcv_history: DataFrame) -> None:
    caplog.set_level(logging.INFO)
    mocker.patch.object(_STRATEGY.dp, "ohlcv", return_value=ohlcv_history)
    mocker.patch.object(_STRATEGY, "_analyze_ticker_internal", side_effect=ValueError("xyz"))
    _STRATEGY.analyze_pair("foo")
    assert log_has_re(r"Strategy caused the following exception: xyz.*", caplog)
    caplog.clear()

    mocker.patch.object(
        _STRATEGY, "analyze_ticker", side_effect=Exception("invalid ticker history ")
    )
    _STRATEGY.analyze_pair("foo")
    assert log_has_re(r"Strategy caused the following exception: xyz.*", caplog)


def test_get_signal_old_dataframe(default_conf: Dict[str, Any], mocker: Any, caplog: Any, ohlcv_history: DataFrame) -> None:
    # default_conf defines a 5m interval. we check interval * 2 + 5m
    # this is necessary as the last candle is removed (partial candles) by default
    ohlcv_history.loc[1, "date"] = dt_now() - timedelta(minutes=16)
    # Take a copy to correctly modify the call
    mocked_history: DataFrame = ohlcv_history.copy()
    mocked_history["exit_long"] = 0
    mocked_history["enter_long"] = 0
    mocked_history.loc[1, "enter_long"] = 1

    caplog.set_level(logging.INFO)
    mocker.patch.object(_STRATEGY, "assert_df")

    assert (None, None) == _STRATEGY.get_latest_candle(
        "xyz", default_conf["timeframe"], mocked_history
    )
    assert log_has("Outdated history for pair xyz. Last tick is 16 minutes old", caplog)


def test_get_signal_no_sell_column(default_conf: Dict[str, Any], mocker: Any, caplog: Any, ohlcv_history: DataFrame) -> None:
    # default_conf defines a 5m interval. we check interval * 2 + 5m
    # this is necessary as the last candle is removed (partial candles) by default
    ohlcv_history.loc[1, "date"] = dt_now()
    # Take a copy to correctly modify the call
    mocked_history: DataFrame = ohlcv_history.copy()
    # Intentionally don't set sell column
    # mocked_history['sell'] = 0
    mocked_history["enter_long"] = 0
    mocked_history.loc[1, "enter_long"] = 1

    caplog.set_level(logging.INFO)
    mocker.patch.object(_STRATEGY, "assert_df")

    assert (SignalDirection.LONG, None) == _STRATEGY.get_entry_signal(
        "xyz", default_conf["timeframe"], mocked_history
    )


def test_ignore_expired_candle(default_conf: Dict[str, Any]) -> None:
    strategy = StrategyResolver.load_strategy(default_conf)
    strategy.ignore_buying_expired_candle_after = 60

    latest_date: datetime = datetime(2020, 12, 30, 7, 0, 0, tzinfo=timezone.utc)
    # Add 1 candle length as the "latest date" defines candle open.
    current_time: datetime = latest_date + timedelta(seconds=80 + 300)

    assert (
        strategy.ignore_expired_candle(
            latest_date=latest_date, current_time=current_time, timeframe_seconds=300, enter=True
        )
        is True
    )

    current_time = latest_date + timedelta(seconds=30 + 300)

    assert (
        strategy.ignore_expired_candle(
            latest_date=latest_date, current_time=current_time, timeframe_seconds=300, enter=True
        )
        is not True
    )


def test_assert_df_raise(mocker: Any, caplog: Any, ohlcv_history: DataFrame) -> None:
    ohlcv_history.loc[1, "date"] = dt_now() - timedelta(minutes=16)
    # Take a copy to correctly modify the call
    mocked_history: DataFrame = ohlcv_history.copy()
    mocked_history["sell"] = 0
    mocked_history["buy"] = 0
    mocked_history.loc[1, "buy"] = 1

    caplog.set_level(logging.INFO)
    mocker.patch.object(_STRATEGY.dp, "ohlcv", return_value=ohlcv_history)
    mocker.patch.object(_STRATEGY.dp, "get_analyzed_dataframe", return_value=(mocked_history, 0))
    mocker.patch.object(_STRATEGY, "assert_df", side_effect=StrategyError("Dataframe returned..."))
    _STRATEGY.analyze_pair("xyz")
    assert log_has(
        "Unable to analyze candle (OHLCV) data for pair xyz: Dataframe returned...", caplog
    )


def test_assert_df(ohlcv_history: DataFrame, caplog: Any) -> None:
    df_len: int = len(ohlcv_history) - 1
    ohlcv_history.loc[:, "enter_long"] = 0
    ohlcv_history.loc[:, "exit_long"] = 0
    # Ensure it's running when passed correctly
    _STRATEGY.assert_df(
        ohlcv_history,
        len(ohlcv_history),
        ohlcv_history.loc[df_len, "close"],
        ohlcv_history.loc[df_len, "date"],
    )

    with pytest.raises(StrategyError, match=r"Dataframe returned from strategy.*length\."):
        _STRATEGY.assert_df(
            ohlcv_history,
            len(ohlcv_history) + 1,
            ohlcv_history.loc[df_len, "close"],
            ohlcv_history.loc[df_len, "date"],
        )

    with pytest.raises(
        StrategyError, match=r"Dataframe returned from strategy.*last close price\."
    ):
        _STRATEGY.assert_df(
            ohlcv_history,
            len(ohlcv_history),
            ohlcv_history.loc[df_len, "close"] + 0.01,
            ohlcv_history.loc[df_len, "date"],
        )
    with pytest.raises(StrategyError, match=r"Dataframe returned from strategy.*last date\."):
        _STRATEGY.assert_df(
            ohlcv_history,
            len(ohlcv_history),
            ohlcv_history.loc[df_len, "close"],
            ohlcv_history.loc[0, "date"],
        )
    with pytest.raises(
        StrategyError, match=r"No dataframe returned \(return statement missing\?\)."
    ):
        _STRATEGY.assert_df(
            None,
            len(ohlcv_history),
            ohlcv_history.loc[df_len, "close"],
            ohlcv_history.loc[0, "date"],
        )

    _STRATEGY.disable_dataframe_checks = True
    caplog.clear()
    _STRATEGY.assert_df(
        ohlcv_history,
        len(ohlcv_history),
        ohlcv_history.loc[2, "close"],
        ohlcv_history.loc[0, "date"],
    )
    assert log_has_re(r"Dataframe returned from strategy.*last date\.", caplog)
    # reset to avoid problems in other tests due to test leakage
    _STRATEGY.disable_dataframe_checks = False


def test_advise_all_indicators(default_conf: Dict[str, Any], testdatadir: Path) -> None:
    strategy = StrategyResolver.load_strategy(default_conf)

    timerange: TimeRange = TimeRange.parse_timerange("1510694220-1510700340")
    data: Dict[str, DataFrame] = load_data(testdatadir, "1m", ["UNITTEST/BTC"], timerange=timerange, fill_up_missing=True)
    processed: Dict[str, DataFrame] = strategy.advise_all_indicators(data)
    assert len(processed["UNITTEST/BTC"]) == 103


def test_freqai_not_initialized(default_conf: Dict[str, Any]) -> None:
    strategy = StrategyResolver.load_strategy(default_conf)
    strategy.ft_bot_start()
    with pytest.raises(OperationalException, match=r"freqAI is not enabled\."):
        strategy.freqai.start()


def test_advise_all_indicators_copy(mocker: Any, default_conf: Dict[str, Any], testdatadir: Path) -> None:
    strategy = StrategyResolver.load_strategy(default_conf)
    aimock = mocker.patch("freqtrade.strategy.interface.IStrategy.advise_indicators")
    timerange: TimeRange = TimeRange.parse_timerange("1510694220-1510700340")
    data: Dict[str, DataFrame] = load_data(testdatadir, "1m", ["UNITTEST/BTC"], timerange=timerange, fill_up_missing=True)
    strategy.advise_all_indicators(data)
    assert aimock.call_count == 1
    # Ensure that a copy of the dataframe is passed to advice_indicators
    assert aimock.call_args_list[0][0][0] is not data


def test_min_roi_reached(default_conf: Dict[str, Any], fee: Any) -> None:
    # Use list to confirm sequence does not matter
    min_roi_list: List[Dict[int, float]] = [{20: 0.05, 55: 0.01, 0: 0.1}, {0: 0.1, 20: 0.05, 55: 0.01}]
    for roi in min_roi_list:
        strategy = StrategyResolver.load_strategy(default_conf)
        strategy.minimal_roi = roi
        trade = Trade(
            pair="ETH/BTC",
            stake_amount=0.001,
            amount=5,
            open_date=dt_now() - timedelta(hours=1),
            fee_open=fee.return_value,
            fee_close=fee.return_value,
            exchange="binance",
            open_rate=1,
        )

        assert not strategy.min_roi_reached(trade, 0.02, dt_now() - timedelta(minutes=56))
        assert strategy.min_roi_reached(trade, 0.12, dt_now() - timedelta(minutes=56))

        assert not strategy.min_roi_reached(trade, 0.04, dt_now() - timedelta(minutes=39))
        assert strategy.min_roi_reached(trade, 0.06, dt_now() - timedelta(minutes=39))

        assert not strategy.min_roi_reached(trade, -0.01, dt_now() - timedelta(minutes=1))
        assert strategy.min_roi_reached(trade, 0.02, dt_now() - timedelta(minutes=1))


def test_min_roi_reached2(default_conf: Dict[str, Any], fee: Any) -> None:
    # test with ROI raising after last interval
    min_roi_list: List[Dict[int, float]] = [
        {20: 0.07, 30: 0.05, 55: 0.30, 0: 0.1},
        {0: 0.1, 20: 0.07, 30: 0.05, 55: 0.30},
    ]
    for roi in min_roi_list:
        strategy = StrategyResolver.load_strategy(default_conf)
        strategy.minimal_roi = roi
        trade = Trade(
            pair="ETH/BTC",
            stake_amount=0.001,
            amount=5,
            open_date=dt_now() - timedelta(hours=1),
            fee_open=fee.return_value,
            fee_close=fee.return_value,
            exchange="binance",
            open_rate=1,
        )

        assert not strategy.min_roi_reached(trade, 0.02, dt_now() - timedelta(minutes=56))
        assert strategy.min_roi_reached(trade, 0.12, dt_now() - timedelta(minutes=56))

        assert not strategy.min_roi_reached(trade, 0.04, dt_now() - timedelta(minutes=39))
        assert strategy.min_roi_reached(trade, 0.071, dt_now() - timedelta(minutes=39))

        assert not strategy.min_roi_reached(trade, 0.04, dt_now() - timedelta(minutes=26))
        assert strategy.min_roi_reached(trade, 0.06, dt_now() - timedelta(minutes=26))

        # Should not trigger with 20% profit since after 55 minutes only 30% is active.
        assert not strategy.min_roi_reached(trade, 0.20, dt_now() - timedelta(minutes=2))
        assert strategy.min_roi_reached(trade, 0.31, dt_now() - timedelta(minutes=2))


def test_min_roi_reached3(default_conf: Dict[str, Any], fee: Any) -> None:
    # test for issue #1948
    min_roi: Dict[int, float] = {
        20: 0.07,
        30: 0.05,
        55: 0.30,
    }
    strategy = StrategyResolver.load_strategy(default_conf)
    strategy.minimal_roi = min_roi
    trade = Trade(
        pair="ETH/BTC",
        stake_amount=0.001,
        amount=5,
        open_date=dt_now() - timedelta(hours=1),
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange="binance",
        open_rate=1,
    )

    assert not strategy.min_roi_reached(trade, 0.02, dt_now() - timedelta(minutes=56))
    assert not strategy.min_roi_reached(trade, 0.12, dt_now() - timedelta(minutes=56))

    assert not strategy.min_roi_reached(trade, 0.04, dt_now() - timedelta(minutes=39))
    assert strategy.min_roi_reached(trade, 0.071, dt_now() - timedelta(minutes=39))

    assert not strategy.min_roi_reached(trade, 0.04, dt_now() - timedelta(minutes=26))
    assert strategy.min_roi_reached(trade, 0.06, dt_now() - timedelta(minutes=26))

    # Should not trigger with 20% profit since after 55 minutes only 30% is active.
    assert not strategy.min_roi_reached(trade, 0.20, dt_now() - timedelta(minutes=2))
    assert strategy.min_roi_reached(trade, 0.31, dt_now() - timedelta(minutes=2))


@pytest.mark.parametrize(
    "profit,adjusted,expected,liq,trailing,custom,profit2,adjusted2,expected2,custom_stop",
    [
        # Profit, adjusted stoploss(absolute), profit for 2nd call, enable trailing,
        #   enable custom stoploss, expected after 1st call, expected after 2nd call
        (0.2, 0.9, ExitType.NONE, None, False, False, 0.3, 0.9, ExitType.NONE, None),
        (0.2, 0.9, ExitType.NONE, None, False, False, -0.2, 0.9, ExitType.STOP_LOSS, None),
        (0.2, 0.9, ExitType.NONE, 0.92, False, False, -0.09, 0.9, ExitType.LIQUIDATION, None),
        (
            0.2,
            1.14,
            ExitType.NONE,
            None,
            True,
            False,
            0.05,
            1.14,
            ExitType.TRAILING_STOP_LOSS,
            None,
        ),
        (0.01, 0.96, ExitType.NONE, None, True, False, 0.05, 0.998, ExitType.NONE, None),
        (
            0.05,
            0.998,
            ExitType.NONE,
            None,
            True,
            False,
            -0.01,
            0.998,
            ExitType.TRAILING_STOP_LOSS,
            None,
        ),
        # Default custom case - trails with 10%
        (0.05, 0.945, ExitType.NONE, None, False, True, -0.02, 0.945, ExitType.NONE, None),
        (
            0.05,
            0.945,
            ExitType.NONE,
            None,
            False,
            True,
            -0.06,
            0.945,
            ExitType.TRAILING_STOP_LOSS,
            None,
        ),
        (
            0.05,
            0.998,
            ExitType.NONE,
            None,
            False,
            True,
            -0.06,
            0.998,
            ExitType.TRAILING_STOP_LOSS,
            lambda **kwargs: -0.05,
        ),
        (
            0.05,
            0.998,
            ExitType.NONE,
            None,
            False,
            True,
            0.09,
            1.036,
            ExitType.NONE,
            lambda **kwargs: -0.05,
        ),
        (
            0.05,
            0.945,
            ExitType.NONE,
            None,
            False,
            True,
            0.09,
            0.981,
            ExitType.NONE,
            lambda current_profit, **kwargs: (
                -0.1 if current_profit < 0.6 else -(current_profit * 2)
            ),
        ),
        # Error case - static stoploss in place
        (
            0.05,
            0.9,
            ExitType.NONE,
            None,
            False,
            True,
            0.09,
            0.9,
            ExitType.NONE,
            lambda **kwargs: None,
        ),
        # Error case - Returning inf.
        (
            0.05,
            0.9,
            ExitType.NONE,
            None,
            False,
            True,
            0.09,
            0.9,
            ExitType.NONE,
            lambda **kwargs: math.inf,
        ),
    ],
)
def test_ft_stoploss_reached(
    default_conf: Dict[str, Any],
    fee: Any,
    profit: float,
    adjusted: float,
    expected: ExitType,
    liq: Optional[float],
    trailing: bool,
    custom: bool,
    profit2: float,
    adjusted2: float,
    expected2: ExitType,
    custom_stop: Optional[Any],
) -> None:
    strategy = StrategyResolver.load_strategy(default_conf)
    trade = Trade(
        pair="ETH/BTC",
        stake_amount=0.01,
        amount=1,
        open_date=dt_now() - timedelta(hours=1),
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange="binance",
        open_rate=1,
        liquidation_price=liq,
        price_precision=4,
        precision_mode=2,
        precision_mode_price=2,
    )
    trade.adjust_min_max_rates(trade.open_rate, trade.open_rate)
    strategy.trailing_stop = trailing
    strategy.trailing_stop_positive = -0.05
    strategy.use_custom_stoploss = custom
    original_stopvalue = strategy.custom_stoploss
    if custom_stop:
        strategy.custom_stoploss = custom_stop

    now: datetime = dt_now()
    current_rate: float = trade.open_rate * (1 + profit)
    sl_flag: ExitCheckTuple = strategy.ft_stoploss_reached(
        current_rate=current_rate,
        trade=trade,
        current_time=now,
        current_profit=profit,
        force_stoploss=0,
        high=None,
    )
    assert isinstance(sl_flag, ExitCheckTuple)
    assert sl_flag.exit_type == expected
    if expected == ExitType.NONE:
        assert sl_flag.exit_flag is False
    else:
        assert sl_flag.exit_flag is True
    assert round(trade.stop_loss, 3) == adjusted
    current_rate2: float = trade.open_rate * (1 + profit2)

    sl_flag = strategy.ft_stoploss_reached(
        current_rate=current_rate2,
        trade=trade,
        current_time=now,
        current_profit=profit2,
        force_stoploss=0,
        high=None,
    )
    assert sl_flag.exit_type == expected2
    if expected2 == ExitType.NONE:
        assert sl_flag.exit_flag is False
    else:
        assert sl_flag.exit_flag is True
    assert round(trade.stop_loss, 3) == adjusted2

    strategy.custom_stoploss = original_stopvalue


def test_custom_exit(default_conf: Dict[str, Any], fee: Any, caplog: Any) -> None:
    strategy = StrategyResolver.load_strategy(default_conf)
    trade = Trade(
        pair="ETH/BTC",
        stake_amount=0.01,
        amount=1,
        open_date=dt_now() - timedelta(hours=1),
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange="binance",
        open_rate=1,
        leverage=1.0,
    )

    now: datetime = dt_now()
    res: List[ExitCheckTuple] = strategy.should_exit(trade, 1, now, enter=False, exit_=False, low=None, high=None)

    assert res == []

    strategy.custom_exit = MagicMock(return_value=True)
    res = strategy.should_exit(trade, 1, now, enter=False, exit_=False, low=None, high=None)
    assert res[0].exit_flag is True
    assert res[0].exit_type == ExitType.CUSTOM_EXIT
    assert res[0].exit_reason == "custom_exit"

    strategy.custom_exit = MagicMock(return_value="hello world")

    res = strategy.should_exit(trade, 1, now, enter=False, exit_=False, low=None, high=None)
    assert res[0].exit_type == ExitType.CUSTOM_EXIT
    assert res[0].exit_flag is True
    assert res[0].exit_reason == "hello world"

    caplog.clear()
    strategy.custom_exit = MagicMock(return_value="h" * CUSTOM_TAG_MAX_LENGTH * 2)
    res = strategy.should_exit(trade, 1, now, enter=False, exit_=False, low=None, high=None)
    assert res[0].exit_type == ExitType.CUSTOM_EXIT
    assert res[0].exit_flag is True
    assert res[0].exit_reason == "h" * (CUSTOM_TAG_MAX_LENGTH)
    assert log_has_re("Custom exit reason returned from custom_exit is too long.*", caplog)


def test_should_sell(default_conf: Dict[str, Any], fee: Any) -> None:
    strategy = StrategyResolver.load_strategy(default_conf)
    trade = Trade(
        pair="ETH/BTC",
        stake_amount=0.01,
        amount=1,
        open_date=dt_now() - timedelta(hours=1),
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange="binance",
        open_rate=1,
        leverage=1.0,
    )
    now: datetime = dt_now()
    res: List[ExitCheckTuple] = strategy.should_exit(trade, 1, now, enter=False, exit_=False, low=None, high=None)

    assert res == []
    strategy.min_roi_reached = MagicMock(return_value=True)

    res = strategy.should_exit(trade, 1, now, enter=False, exit_=False, low=None, high=None)
    assert len(res) == 1
    assert res == [ExitCheckTuple(exit_type=ExitType.ROI)]

    strategy.min_roi_reached = MagicMock(return_value=True)
    strategy.ft_stoploss_reached = MagicMock(
        return_value=ExitCheckTuple(exit_type=ExitType.STOP_LOSS)
    )

    res = strategy.should_exit(trade, 1, now, enter=False, exit_=False, low=None, high=None)
    assert len(res) == 2
    assert res == [
        ExitCheckTuple(exit_type=ExitType.STOP_LOSS),
        ExitCheckTuple(exit_type=ExitType.ROI),
    ]

    strategy.custom_exit = MagicMock(return_value="hello world")
    # custom-exit and exit-signal is first
    res = strategy.should_exit(trade, 1, now, enter=False, exit_=False, low=None, high=None)
    assert len(res) == 3
    assert res == [
        ExitCheckTuple(exit_type=ExitType.CUSTOM_EXIT, exit_reason="hello world"),
        ExitCheckTuple(exit_type=ExitType.STOP_LOSS),
        ExitCheckTuple(exit_type=ExitType.ROI),
    ]

    strategy.ft_stoploss_reached = MagicMock(
        return_value=ExitCheckTuple(exit_type=ExitType.TRAILING_STOP_LOSS)
    )
    # Regular exit signal
    res = strategy.should_exit(trade, 1, now, enter=False, exit_=True, low=None, high=None)
    assert len(res) == 3
    assert res == [
        ExitCheckTuple(exit_type=ExitType.EXIT_SIGNAL),
        ExitCheckTuple(exit_type=ExitType.ROI),
        ExitCheckTuple(exit_type=ExitType.TRAILING_STOP_LOSS),
    ]

    # Regular exit signal, no ROI
    strategy.min_roi_reached = MagicMock(return_value=False)
    res = strategy.should_exit(trade, 1, now, enter=False, exit_=True, low=None, high=None)
    assert len(res) == 2
    assert res == [
        ExitCheckTuple(exit_type=ExitType.EXIT_SIGNAL),
        ExitCheckTuple(exit_type=ExitType.TRAILING_STOP_LOSS),
    ]


@pytest.mark.parametrize("side", TRADE_SIDES)
def test_leverage_callback(default_conf: Dict[str, Any], side: str) -> None:
    default_conf["strategy"] = "StrategyTestV2"
    strategy = StrategyResolver.load_strategy(default_conf)

    assert (
        strategy.leverage(
            pair="XRP/USDT",
            current_time=datetime.now(timezone.utc),
            current_rate=2.2,
            proposed_leverage=1.0,
            max_leverage=5.0,
            side=side,
            entry_tag=None,
        )
        == 1
    )

    default_conf["strategy"] = CURRENT_TEST_STRATEGY
    strategy = StrategyResolver.load_strategy(default_conf)
    assert (
        strategy.leverage(
            pair="XRP/USDT",
            current_time=datetime.now(timezone.utc),
            current_rate=2.2,
            proposed_leverage=1.0,
            max_leverage=5.0,
            side=side,
            entry_tag="entry_tag_test",
        )
        == 3
    )


def test_analyze_ticker_default(ohlcv_history: DataFrame, mocker: Any, caplog: Any) -> None:
    caplog.set_level(logging.DEBUG)
    ind_mock