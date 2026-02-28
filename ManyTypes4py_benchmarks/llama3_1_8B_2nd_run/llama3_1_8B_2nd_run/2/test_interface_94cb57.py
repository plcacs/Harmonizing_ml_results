import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
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
from freqtrade.strategy.parameters import BaseParameter, BooleanParameter, CategoricalParameter, DecimalParameter, IntParameter, RealParameter
from freqtrade.util import dt_now
from tests.conftest import CURRENT_TEST_STRATEGY, TRADE_SIDES, log_has, log_has_re
from .strats.strategy_test_v3 import StrategyTestV3

_STRATEGY: StrategyTestV3 = StrategyTestV3(config={})
_STRATEGY.dp = DataProvider({}, None, None)


def test_returns_latest_signal(
    ohlcv_history: DataFrame,
) -> None:
    ohlcv_history.loc[1, "date"] = dt_now()
    mocked_history = ohlcv_history.copy()
    mocked_history["enter_long"] = 0
    mocked_history["exit_long"] = 0
    mocked_history["enter_short"] = 0
    mocked_history["exit_short"] = 0
    mocked_history.loc[0, "enter_tag"] = "wrong_line"
    mocked_history.loc[0, "exit_tag"] = "wrong_line"
    mocked_history.loc[1, "exit_long"] = 1
    assert _STRATEGY.get_entry_signal("ETH/BTC", "5m", mocked_history) == (None, None)
    assert _STRATEGY.get_exit_signal("ETH/BTC", "5m", mocked_history) == (False, True, None)
    assert _STRATEGY.get_exit_signal("ETH/BTC", "5m", mocked_history, True) == (False, False, None)
    mocked_history.loc[1, "exit_long"] = 0
    mocked_history.loc[1, "enter_long"] = 1
    assert _STRATEGY.get_entry_signal("ETH/BTC", "5m", mocked_history) == (SignalDirection.LONG, None)
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
    assert _STRATEGY.get_entry_signal("ETH/BTC", "5m", mocked_history) == (SignalDirection.LONG, "buy_signal_01")
    assert _STRATEGY.get_exit_signal("ETH/BTC", "5m", mocked_history) == (True, False, None)
    assert _STRATEGY.get_exit_signal("ETH/BTC", "5m", mocked_history, True) == (False, False, None)
    mocked_history.loc[1, "exit_long"] = 0
    mocked_history.loc[1, "enter_long"] = 0
    mocked_history.loc[1, "enter_short"] = 1
    mocked_history.loc[1, "exit_short"] = 0
    mocked_history.loc[1, "enter_tag"] = "sell_signal_01"
    assert _STRATEGY.get_entry_signal("ETH/BTC", "5m", mocked_history) == (None, None)
    _STRATEGY.config["trading_mode"] = "futures"
    assert _STRATEGY.get_entry_signal("ETH/BTC", "5m", mocked_history) == (None, None)
    _STRATEGY.can_short = True
    assert _STRATEGY.get_entry_signal("ETH/BTC", "5m", mocked_history) == (SignalDirection.SHORT, "sell_signal_01")
    assert _STRATEGY.get_exit_signal("ETH/BTC", "5m", mocked_history) == (False, False, None)
    assert _STRATEGY.get_exit_signal("ETH/BTC", "5m", mocked_history, True) == (True, False, None)
    mocked_history.loc[1, "enter_short"] = 0
    mocked_history.loc[1, "exit_short"] = 1
    mocked_history.loc[1, "exit_tag"] = "sell_signal_02"
    assert _STRATEGY.get_entry_signal("ETH/BTC", "5m", mocked_history) == (None, None)
    assert _STRATEGY.get_exit_signal("ETH/BTC", "5m", mocked_history) == (False, False, "sell_signal_02")
    assert _STRATEGY.get_exit_signal("ETH/BTC", "5m", mocked_history, True) == (False, True, "sell_signal_02")
    _STRATEGY.can_short = False
    _STRATEGY.config["trading_mode"] = "spot"


def test_analyze_pair_empty(mocker: pytest.Mock, caplog: pytest.LogCaptureFixture, ohlcv_history: DataFrame) -> None:
    mocker.patch.object(_STRATEGY.dp, "ohlcv", return_value=ohlcv_history)
    mocker.patch.object(_STRATEGY, "_analyze_ticker_internal", return_value=DataFrame([]))
    mocker.patch.object(_STRATEGY, "assert_df")
    _STRATEGY.analyze_pair("ETH/BTC")
    assert log_has("Empty dataframe for pair ETH/BTC", caplog)


def test_get_signal_empty(
    default_conf: dict,
    caplog: pytest.LogCaptureFixture,
) -> None:
    assert (None, None) == _STRATEGY.get_latest_candle("foo", default_conf["timeframe"], DataFrame())
    assert log_has("Empty candle (OHLCV) data for pair foo", caplog)
    caplog.clear()
    assert (None, None) == _STRATEGY.get_latest_candle("bar", default_conf["timeframe"], None)
    assert log_has("Empty candle (OHLCV) data for pair bar", caplog)
    caplog.clear()
    assert (None, None) == _STRATEGY.get_latest_candle("baz", default_conf["timeframe"], DataFrame([]))
    assert log_has("Empty candle (OHLCV) data for pair baz", caplog)


def test_get_signal_exception_valueerror(
    mocker: pytest.Mock,
    caplog: pytest.LogCaptureFixture,
    ohlcv_history: DataFrame,
) -> None:
    caplog.set_level(logging.INFO)
    mocker.patch.object(_STRATEGY.dp, "ohlcv", return_value=ohlcv_history)
    mocker.patch.object(_STRATEGY, "_analyze_ticker_internal", side_effect=ValueError("xyz"))
    _STRATEGY.analyze_pair("foo")
    assert log_has_re("Strategy caused the following exception: xyz.*", caplog)
    caplog.clear()
    mocker.patch.object(_STRATEGY, "analyze_ticker", side_effect=Exception("invalid ticker history "))
    _STRATEGY.analyze_pair("foo")
    assert log_has_re("Strategy caused the following exception: xyz.*", caplog)


def test_get_signal_old_dataframe(
    default_conf: dict,
    mocker: pytest.Mock,
    caplog: pytest.LogCaptureFixture,
    ohlcv_history: DataFrame,
) -> None:
    ohlcv_history.loc[1, "date"] = dt_now() - timedelta(minutes=16)
    mocked_history = ohlcv_history.copy()
    mocked_history["exit_long"] = 0
    mocked_history["enter_long"] = 0
    mocked_history.loc[1, "enter_long"] = 1
    caplog.set_level(logging.INFO)
    mocker.patch.object(_STRATEGY, "assert_df")
    assert (None, None) == _STRATEGY.get_latest_candle("xyz", default_conf["timeframe"], mocked_history)
    assert log_has("Outdated history for pair xyz. Last tick is 16 minutes old", caplog)


def test_get_signal_no_sell_column(
    default_conf: dict,
    mocker: pytest.Mock,
    caplog: pytest.LogCaptureFixture,
    ohlcv_history: DataFrame,
) -> None:
    ohlcv_history.loc[1, "date"] = dt_now()
    mocked_history = ohlcv_history.copy()
    mocked_history["enter_long"] = 0
    mocked_history.loc[1, "enter_long"] = 1
    caplog.set_level(logging.INFO)
    mocker.patch.object(_STRATEGY, "assert_df")
    assert (SignalDirection.LONG, None) == _STRATEGY.get_entry_signal("xyz", default_conf["timeframe"], mocked_history)


def test_ignore_expired_candle(
    default_conf: dict,
) -> None:
    strategy = StrategyResolver.load_strategy(default_conf)
    strategy.ignore_buying_expired_candle_after = 60
    latest_date = datetime(2020, 12, 30, 7, 0, 0, tzinfo=timezone.utc)
    current_time = latest_date + timedelta(seconds=80 + 300)
    assert strategy.ignore_expired_candle(
        latest_date=latest_date, current_time=current_time, timeframe_seconds=300, enter=True
    ) is True
    current_time = latest_date + timedelta(seconds=30 + 300)
    assert strategy.ignore_expired_candle(
        latest_date=latest_date, current_time=current_time, timeframe_seconds=300, enter=True
    ) is not True


def test_assert_df_raise(
    mocker: pytest.Mock,
    caplog: pytest.LogCaptureFixture,
    ohlcv_history: DataFrame,
) -> None:
    ohlcv_history.loc[1, "date"] = dt_now() - timedelta(minutes=16)
    mocked_history = ohlcv_history.copy()
    mocked_history["sell"] = 0
    mocked_history["buy"] = 0
    mocked_history.loc[1, "buy"] = 1
    caplog.set_level(logging.INFO)
    mocker.patch.object(_STRATEGY.dp, "ohlcv", return_value=ohlcv_history)
    mocker.patch.object(_STRATEGY.dp, "get_analyzed_dataframe", return_value=(mocked_history, 0))
    mocker.patch.object(_STRATEGY, "assert_df", side_effect=StrategyError("Dataframe returned..."))
    _STRATEGY.analyze_pair("xyz")
    assert log_has(
        "Unable to analyze candle (OHLCV) data for pair xyz: Dataframe returned...",
        caplog,
    )


def test_assert_df(
    ohlcv_history: DataFrame,
    caplog: pytest.LogCaptureFixture,
) -> None:
    df_len = len(ohlcv_history) - 1
    ohlcv_history.loc[:, "enter_long"] = 0
    ohlcv_history.loc[:, "exit_long"] = 0
    _STRATEGY.assert_df(
        ohlcv_history,
        len(ohlcv_history),
        ohlcv_history.loc[df_len, "close"],
        ohlcv_history.loc[df_len, "date"],
    )
    with pytest.raises(StrategyError, match="Dataframe returned from strategy.*length\\."):
        _STRATEGY.assert_df(
            ohlcv_history,
            len(ohlcv_history) + 1,
            ohlcv_history.loc[df_len, "close"],
            ohlcv_history.loc[df_len, "date"],
        )
    with pytest.raises(StrategyError, match="Dataframe returned from strategy.*last close price\\."):
        _STRATEGY.assert_df(
            ohlcv_history,
            len(ohlcv_history),
            ohlcv_history.loc[df_len, "close"] + 0.01,
            ohlcv_history.loc[df_len, "date"],
        )
    with pytest.raises(StrategyError, match="Dataframe returned from strategy.*last date\\."):
        _STRATEGY.assert_df(
            ohlcv_history,
            len(ohlcv_history),
            ohlcv_history.loc[df_len, "close"],
            ohlcv_history.loc[0, "date"],
        )
    with pytest.raises(StrategyError, match="No dataframe returned \\(return statement missing\\?\\)."):
        _STRATEGY.assert_df(None, len(ohlcv_history), ohlcv_history.loc[df_len, "close"], ohlcv_history.loc[0, "date"])
    _STRATEGY.disable_dataframe_checks = True
    caplog.clear()
    _STRATEGY.assert_df(
        ohlcv_history,
        len(ohlcv_history),
        ohlcv_history.loc[2, "close"],
        ohlcv_history.loc[0, "date"],
    )
    assert log_has_re("Dataframe returned from strategy.*last date\\.", caplog)
    _STRATEGY.disable_dataframe_checks = False


def test_advise_all_indicators(
    default_conf: dict,
    testdatadir: str,
) -> None:
    strategy = StrategyResolver.load_strategy(default_conf)
    timerange = TimeRange.parse_timerange("1510694220-1510700340")
    data = load_data(
        testdatadir,
        "1m",
        ["UNITTEST/BTC"],
        timerange=timerange,
        fill_up_missing=True,
    )
    processed = strategy.advise_all_indicators(data)
    assert len(processed["UNITTEST/BTC"]) == 103


def test_freqai_not_initialized(
    default_conf: dict,
) -> None:
    strategy = StrategyResolver.load_strategy(default_conf)
    strategy.ft_bot_start()
    with pytest.raises(OperationalException, match="freqAI is not enabled\\."):
        strategy.freqai.start()


def test_advise_all_indicators_copy(
    mocker: pytest.Mock,
    default_conf: dict,
    testdatadir: str,
) -> None:
    strategy = StrategyResolver.load_strategy(default_conf)
    aimock = mocker.patch("freqtrade.strategy.interface.IStrategy.advise_indicators")
    timerange = TimeRange.parse_timerange("1510694220-1510700340")
    data = load_data(
        testdatadir,
        "1m",
        ["UNITTEST/BTC"],
        timerange=timerange,
        fill_up_missing=True,
    )
    strategy.advise_all_indicators(data)
    assert aimock.call_count == 1
    assert aimock.call_args_list[0][0][0] is not data


def test_min_roi_reached(
    default_conf: dict,
    fee: MagicMock,
) -> None:
    min_roi_list = [{20: 0.05, 55: 0.01, 0: 0.1}, {0: 0.1, 20: 0.05, 55: 0.01}]
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


def test_min_roi_reached2(
    default_conf: dict,
    fee: MagicMock,
) -> None:
    min_roi_list = [{20: 0.07, 30: 0.05, 55: 0.3, 0: 0.1}, {0: 0.1, 20: 0.07, 30: 0.05, 55: 0.3}]
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
        assert not strategy.min_roi_reached(trade, 0.2, dt_now() - timedelta(minutes=2))
        assert strategy.min_roi_reached(trade, 0.31, dt_now() - timedelta(minutes=2))


def test_min_roi_reached3(
    default_conf: dict,
    fee: MagicMock,
) -> None:
    min_roi = {20: 0.07, 30: 0.05, 55: 0.3}
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
    assert not strategy.min_roi_reached(trade, 0.2, dt_now() - timedelta(minutes=2))
    assert strategy.min_roi_reached(trade, 0.31, dt_now() - timedelta(minutes=2))


@pytest.mark.parametrize(
    "profit,adjusted,expected,liq,trailing,custom,profit2,adjusted2,expected2,custom_stop",
    [
        (0.2, 0.9, ExitType.NONE, None, False, False, 0.3, 0.9, ExitType.NONE, None),
        (0.2, 0.9, ExitType.NONE, None, False, False, -0.2, 0.9, ExitType.STOP_LOSS, None),
        (0.2, 0.9, ExitType.NONE, 0.92, False, False, -0.09, 0.9, ExitType.LIQUIDATION, None),
        (0.2, 1.14, ExitType.NONE, None, True, False, 0.05, 1.14, ExitType.TRAILING_STOP_LOSS, None),
        (0.01, 0.96, ExitType.NONE, None, True, False, 0.05, 0.998, ExitType.NONE, None),
        (0.05, 0.998, ExitType.NONE, None, True, False, -0.01, 0.998, ExitType.TRAILING_STOP_LOSS, None),
        (0.05, 0.945, ExitType.NONE, None, False, True, -0.02, 0.945, ExitType.NONE, None),
        (0.05, 0.945, ExitType.NONE, None, False, True, -0.06, 0.945, ExitType.TRAILING_STOP_LOSS, None),
        (0.05, 0.998, ExitType.NONE, None, False, True, -0.06, 0.998, ExitType.TRAILING_STOP_LOSS, None),
        (0.05, 0.998, ExitType.NONE, None, False, True, 0.09, 1.036, ExitType.NONE, lambda **kwargs: -0.05),
        (0.05, 0.945, ExitType.NONE, None, False, True, 0.09, 0.981, ExitType.NONE, lambda current_profit, **kwargs: -0.1 if current_profit < 0.6 else -(current_profit * 2)),
        (0.05, 0.9, ExitType.NONE, None, False, True, 0.09, 0.9, ExitType.NONE, lambda **kwargs: None),
        (0.05, 0.9, ExitType.NONE, None, False, True, 0.09, 0.9, ExitType.NONE, lambda **kwargs: math.inf),
    ],
)
def test_ft_stoploss_reached(
    default_conf: dict,
    fee: MagicMock,
    profit: float,
    adjusted: float,
    expected: ExitType,
    liq: float,
    trailing: bool,
    custom: bool,
    profit2: float,
    adjusted2: float,
    expected2: ExitType,
    custom_stop: callable,
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
    now = dt_now()
    current_rate = trade.open_rate * (1 + profit)
    sl_flag = strategy.ft_stoploss_reached(
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
    current_rate2 = trade.open_rate * (1 + profit2)
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


def test_custom_exit(
    default_conf: dict,
    fee: MagicMock,
    caplog: pytest.LogCaptureFixture,
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
        leverage=1.0,
    )
    now = dt_now()
    res = strategy.should_exit(trade, 1, now, enter=False, exit_=False, low=None, high=None)
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
    assert res[0].exit_reason == "h" * CUSTOM_TAG_MAX_LENGTH
    assert log_has_re(
        "Custom exit reason returned from custom_exit is too long.*",
        caplog,
    )


def test_should_sell(
    default_conf: dict,
    fee: MagicMock,
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
        leverage=1.0,
    )
    now = dt_now()
    res = strategy.should_exit(trade, 1, now, enter=False, exit_=False, low=None, high=None)
    assert res == []
    strategy.min_roi_reached = MagicMock(return_value=True)
    res = strategy.should_exit(trade, 1, now, enter=False, exit_=False, low=None, high=None)
    assert len(res) == 1
    assert res == [ExitCheckTuple(exit_type=ExitType.ROI)]
    strategy.min_roi_reached = MagicMock(return_value=True)
    strategy.ft_stoploss_reached = MagicMock(return_value=ExitCheckTuple(exit_type=ExitType.STOP_LOSS))
    res = strategy.should_exit(trade, 1, now, enter=False, exit_=False, low=None, high=None)
    assert len(res) == 2
    assert res == [
        ExitCheckTuple(exit_type=ExitType.STOP_LOSS),
        ExitCheckTuple(exit_type=ExitType.ROI),
    ]
    strategy.custom_exit = MagicMock(return_value="hello world")
    res = strategy.should_exit(trade, 1, now, enter=False, exit_=False, low=None, high=None)
    assert len(res) == 3
    assert res == [
        ExitCheckTuple(exit_type=ExitType.CUSTOM_EXIT, exit_reason="hello world"),
        ExitCheckTuple(exit_type=ExitType.STOP_LOSS),
        ExitCheckTuple(exit_type=ExitType.ROI),
    ]
    strategy.ft_stoploss_reached = MagicMock(return_value=ExitCheckTuple(exit_type=ExitType.TRAILING_STOP_LOSS))
    res = strategy.should_exit(trade, 1, now, enter=False, exit_=True, low=None, high=None)
    assert len(res) == 3
    assert res == [
        ExitCheckTuple(exit_type=ExitType.EXIT_SIGNAL),
        ExitCheckTuple(exit_type=ExitType.ROI),
        ExitCheckTuple(exit_type=ExitType.TRAILING_STOP_LOSS),
    ]
    strategy.min_roi_reached = MagicMock(return_value=False)
    res = strategy.should_exit(trade, 1, now, enter=False, exit_=True, low=None, high=None)
    assert len(res) == 2
    assert res == [
        ExitCheckTuple(exit_type=ExitType.EXIT_SIGNAL),
        ExitCheckTuple(exit_type=ExitType.TRAILING_STOP_LOSS),
    ]


@pytest.mark.parametrize("side", TRADE_SIDES)
def test_leverage_callback(
    default_conf: dict,
    side: str,
) -> None:
    default_conf["strategy"] = "StrategyTestV2"
    strategy = StrategyResolver.load_strategy(default_conf)
    assert strategy.leverage(
        pair="XRP/USDT",
        current_time=datetime.now(timezone.utc),
        current_rate=2.2,
        proposed_leverage=1.0,
        max_leverage=5.0,
        side=side,
        entry_tag=None,
    ) == 1
    default_conf["strategy"] = CURRENT_TEST_STRATEGY
    strategy = StrategyResolver.load_strategy(default_conf)
    assert strategy.leverage(
        pair="XRP/USDT",
        current_time=datetime.now(timezone.utc),
        current_rate=2.2,
        proposed_leverage=1.0,
        max_leverage=5.0,
        side=side,
        entry_tag="entry_tag_test",
    ) == 3


def test_analyze_ticker_default(
    ohlcv_history: DataFrame,
    mocker: pytest.Mock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    ind_mock = MagicMock(side_effect=lambda x, meta: x)
    entry_mock = MagicMock(side_effect=lambda x, meta: x)
    exit_mock = MagicMock(side_effect=lambda x, meta: x)
    mocker.patch.multiple(
        "freqtrade.strategy.interface.IStrategy",
        advise_indicators=ind_mock,
        advise_entry=entry_mock,
        advise_exit=exit_mock,
    )
    strategy = StrategyTestV3({})
    strategy.analyze_ticker(ohlcv_history, {"pair": "ETH/BTC"})
    assert ind_mock.call_count == 1
    assert entry_mock.call_count == 1
    assert entry_mock.call_count == 1
    assert log_has("TA Analysis Launched", caplog)
    assert not log_has("Skipping TA Analysis for already analyzed candle", caplog)
    caplog.clear()
    strategy.analyze_ticker(ohlcv_history, {"pair": "ETH/BTC"})
    assert ind_mock.call_count == 2
    assert entry_mock.call_count == 2
    assert entry_mock.call_count == 2
    assert log_has("TA Analysis Launched", caplog)
    assert not log_has("Skipping TA Analysis for already analyzed candle", caplog)


def test__analyze_ticker_internal_skip_analyze(
    ohlcv_history: DataFrame,
    mocker: pytest.Mock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    ind_mock = MagicMock(side_effect=lambda x, meta: x)
    entry_mock = MagicMock(side_effect=lambda x, meta: x)
    exit_mock = MagicMock(side_effect=lambda x, meta: x)
    mocker.patch.multiple(
        "freqtrade.strategy.interface.IStrategy",
        advise_indicators=ind_mock,
        advise_entry=entry_mock,
        advise_exit=exit_mock,
    )
    strategy = StrategyTestV3({})
    strategy.dp = DataProvider({}, None, None)
    strategy.process_only_new_candles = True
    ret = strategy._analyze_ticker_internal(ohlcv_history, {"pair": "ETH/BTC"})
    assert "high" in ret.columns
    assert "low" in ret.columns
    assert "close" in ret.columns
    assert isinstance(ret, DataFrame)
    assert ind_mock.call_count == 1
    assert entry_mock.call_count == 1
    assert entry_mock.call_count == 1
    assert log_has("TA Analysis Launched", caplog)
    assert not log_has("Skipping TA Analysis for already analyzed candle", caplog)
    caplog.clear()
    ret = strategy._analyze_ticker_internal(ohlcv_history, {"pair": "ETH/BTC"})
    assert ind_mock.call_count == 1
    assert entry_mock.call_count == 1
    assert entry_mock.call_count == 1
    assert "enter_long" in ret.columns
    assert "exit_long" in ret.columns
    assert ret["enter_long"].sum() == 0
    assert ret["exit_long"].sum() == 0
    assert not log_has("TA Analysis Launched", caplog)
    assert log_has("Skipping TA Analysis for already analyzed candle", caplog)


@pytest.mark.usefixtures("init_persistence")
def test_is_pair_locked(
    default_conf: dict,
) -> None:
    PairLocks.timeframe = default_conf["timeframe"]
    PairLocks.use_db = True
    strategy = StrategyResolver.load_strategy(default_conf)
    assert len(PairLocks.get_pair_locks(None)) == 0
    pair = "ETH/BTC"
    assert not strategy.is_pair_locked(pair)
    strategy.lock_pair(pair, dt_now() + timedelta(minutes=4))
    assert strategy.is_pair_locked(pair)
    pair = "XRP/BTC"
    assert not strategy.is_pair_locked(pair)
    strategy.unlock_pair(pair)
    pair = "ETH/BTC"
    strategy.unlock_pair(pair)
    assert not strategy.is_pair_locked(pair)
    reason = "TestLockR"
    strategy.lock_pair(pair, dt_now() + timedelta(minutes=4), reason)
    assert strategy.is_pair_locked(pair)
    strategy.unlock_reason(reason)
    assert not strategy.is_pair_locked(pair)
    pair = "BTC/USDT"
    lock_time = datetime(2020, 5, 1, 14, 30, 0, tzinfo=timezone.utc)
    strategy.lock_pair(pair, lock_time - timedelta(seconds=2))
    assert not strategy.is_pair_locked(pair)
    assert strategy.is_pair_locked(pair, candle_date=lock_time + timedelta(minutes=-10))
    assert strategy.is_pair_locked(pair, candle_date=lock_time + timedelta(minutes=-50))
    assert not strategy.is_pair_locked(pair, candle_date=lock_time + timedelta(minutes=-4))
    assert not strategy.is_pair_locked(pair, candle_date=lock_time + timedelta(minutes=10))
    strategy.timeframe = "15m"
    assert strategy.is_pair_locked(pair, candle_date=lock_time + timedelta(minutes=-16))
    assert strategy.is_pair_locked(pair, candle_date=lock_time + timedelta(minutes=-15, seconds=-2))
    assert not strategy.is_pair_locked(pair, candle_date=lock_time + timedelta(minutes=-15))


def test_is_informative_pairs_callback(
    default_conf: dict,
) -> None:
    default_conf.update({"strategy": "StrategyTestV2"})
    strategy = StrategyResolver.load_strategy(default_conf)
    assert [] == strategy.gather_informative_pairs()


def test_hyperopt_parameters() -> None:
    HyperoptStateContainer.set_state(HyperoptState.INDICATORS)
    from skopt.space import Categorical, Integer, Real

    with pytest.raises(OperationalException, match="Name is determined.*"):
        IntParameter(low=0, high=5, default=1, name="hello")
    with pytest.raises(OperationalException, match="IntParameter space must be.*"):
        IntParameter(low=0, default=5, space="buy")
    with pytest.raises(OperationalException, match="RealParameter space must be.*"):
        RealParameter(low=0, default=5, space="buy")
    with pytest.raises(OperationalException, match="DecimalParameter space must be.*"):
        DecimalParameter(low=0, default=5, space="buy")
    with pytest.raises(OperationalException, match="IntParameter space invalid\\."):
        IntParameter([0, 10], high=7, default=5, space="buy")
    with pytest.raises(OperationalException, match="RealParameter space invalid\\."):
        RealParameter([0, 10], high=7, default=5, space="buy")
    with pytest.raises(OperationalException, match="DecimalParameter space invalid\\."):
        DecimalParameter([0, 10], high=7, default=5, space="buy")
    with pytest.raises(OperationalException, match="CategoricalParameter space must.*"):
        CategoricalParameter(["aa"], default="aa", space="buy")
    with pytest.raises(TypeError):
        BaseParameter(opt_range=[0, 1], default=1, space="buy")
    intpar = IntParameter(low=0, high=5, default=1, space="buy")
    assert intpar.value == 1
    assert isinstance(intpar.get_space(""), Integer)
    assert isinstance(intpar.range, range)
    assert len(list(intpar.range)) == 1
    assert list(intpar.range) == [intpar.value]
    intpar.in_space = True
    assert len(list(intpar.range)) == 6
    assert list(intpar.range) == [0, 1, 2, 3, 4, 5]
    fltpar = RealParameter(low=0.0, high=5.5, default=1.0, space="buy")
    assert fltpar.value == 1
    assert isinstance(fltpar.get_space(""), Real)
    fltpar = DecimalParameter(low=0.0, high=0.5, default=0.14, decimals=1, space="buy")
    assert fltpar.value == 0.1
    assert isinstance(fltpar.get_space(""), SKDecimal)
    assert isinstance(fltpar.range, list)
    assert len(list(fltpar.range)) == 1
    assert list(fltpar.range) == [fltpar.value]
    fltpar.in_space = True
    assert len(list(fltpar.range)) == 6
    assert list(fltpar.range) == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    catpar = CategoricalParameter(["buy_rsi", "buy_macd", "buy_none"], default="buy_macd", space="buy")
    assert catpar.value == "buy_macd"
    assert isinstance(catpar.get_space(""), Categorical)
    assert isinstance(catpar.range, list)
    assert len(list(catpar.range)) == 1
    assert list(catpar.range) == [catpar.value]
    catpar.in_space = True
    assert len(list(catpar.range)) == 3
    assert list(catpar.range) == ["buy_rsi", "buy_macd", "buy_none"]
    boolpar = BooleanParameter(default=True, space="buy")
    assert boolpar.value is True
    assert isinstance(boolpar.get_space(""), Categorical)
    assert isinstance(boolpar.range, list)
    assert len(list(boolpar.range)) == 1
    boolpar.in_space = True
    assert len(list(boolpar.range)) == 2
    assert list(boolpar.range) == [True, False]
    HyperoptStateContainer.set_state(HyperoptState.OPTIMIZE)
    assert len(list(intpar.range)) == 1
    assert len(list(fltpar.range)) == 1
    assert len(list(catpar.range)) == 1
    assert len(list(boolpar.range)) == 1


def test_auto_hyperopt_interface(
    default_conf: dict,
    mocker: pytest.Mock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    default_conf.update({"strategy": "HyperoptableStrategyV2"})
    PairLocks.timeframe = default_conf["timeframe"]
    strategy = StrategyResolver.load_strategy(default_conf)
    strategy.ft_bot_start()
    with pytest.raises(OperationalException):
        next(strategy.enumerate_parameters("deadBeef"))
    assert strategy.buy_rsi.value == strategy.buy_params["buy_rsi"]
    assert strategy.buy_plusdi.value == 0.5
    assert strategy.sell_rsi.value == strategy.sell_params["sell_rsi"]
    assert repr(strategy.sell_rsi) == "IntParameter(74)"
    assert strategy.sell_minusdi.value == 0.5
    all_params = strategy.detect_all_parameters()
    assert isinstance(all_params, dict)
    assert len(all_params["buy"]) == 1
    assert len(list(detect_parameters(strategy, "buy"))) == 2
    assert len(all_params["sell"]) == 2
    assert all_params["count"] == 5
    strategy.__class__.sell_rsi = IntParameter([0, 10], default=5, space="buy")
    with pytest.raises(OperationalException, match="Inconclusive parameter.*"):
        [x for x in detect_parameters(strategy, "sell")]


def test_auto_hyperopt_interface_loadparams(
    default_conf: dict,
    mocker: pytest.Mock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    default_conf.update({"strategy": "HyperoptableStrategy"})
    del default_conf["stoploss"]
    del default_conf["minimal_roi"]
    mocker.patch.object(Path, "is_file", MagicMock(return_value=True))
    mocker.patch.object(Path, "open")
    expected_result = {
        "strategy_name": "HyperoptableStrategy",
        "params": {"stoploss": {"stoploss": -0.05}, "roi": {"0": 0.2, "1200": 0.01}},
    }
    mocker.patch("freqtrade.strategy.hyper.HyperoptTools.load_params", return_value=expected_result)
    PairLocks.timeframe = default_conf["timeframe"]
    strategy = StrategyResolver.load_strategy(default_conf)
    assert strategy.stoploss == -0.05
    assert strategy.minimal_roi == {0: 0.2, 1200: 0.01}
    expected_result = {
        "strategy_name": "HyperoptableStrategy_No",
        "params": {"stoploss": {"stoploss": -0.05}, "roi": {"0": 0.2, "1200": 0.01}},
    }
    mocker.patch("freqtrade.strategy.hyper.HyperoptTools.load_params", return_value=expected_result)
    with pytest.raises(OperationalException, match="Invalid parameter file provided."):
        StrategyResolver.load_strategy(default_conf)
    mocker.patch("freqtrade.strategy.hyper.HyperoptTools.load_params", MagicMock(side_effect=ValueError()))
    StrategyResolver.load_strategy(default_conf)
    assert log_has("Invalid parameter file format.", caplog)


@pytest.mark.parametrize("function,raises", [("populate_entry_trend", False), ("advise_entry", False), ("populate_exit_trend", False), ("advise_exit", False)])
def test_pandas_warning_direct(
    ohlcv_history: DataFrame,
    function: str,
    raises: bool,
    recwarn: pytest.WarningsRecorder,
) -> None:
    df = strategy.populate_indicators(ohlcv_history, {"pair": "ETH/BTC"})
    if raises:
        assert len(recwarn) == 1
        getattr(strategy, function)(df, {"pair": "ETH/BTC"})
    else:
        assert len(recwarn) == 0
        getattr(strategy, function)(df, {"pair": "ETH/BTC"})


def test_pandas_warning_through_analyze_pair(
    ohlcv_history: DataFrame,
    mocker: pytest.Mock,
    recwarn: pytest.WarningsRecorder,
) -> None:
    mocker.patch.object(strategy.dp, "ohlcv", return_value=ohlcv_history)
    strategy.analyze_pair("ETH/BTC")
    assert len(recwarn) == 0
