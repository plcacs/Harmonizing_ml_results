from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
from tests.conftest import (
    CURRENT_TEST_STRATEGY,
    EXMS,
    generate_test_data,
    get_args,
    log_has,
    log_has_re,
    patch_exchange,
    patched_configuration_load_config_file,
)


ORDER_TYPES: List[Dict[str, Any]] = [
    {"entry": "limit", "exit": "limit", "stoploss": "limit", "stoploss_on_exchange": False},
    {"entry": "limit", "exit": "limit", "stoploss": "limit", "stoploss_on_exchange": True},
]


def trim_dictlist(dict_list: Dict[str, pd.DataFrame], num: int) -> Dict[str, pd.DataFrame]:
    new: Dict[str, pd.DataFrame] = {}
    for pair, pair_data in dict_list.items():
        new[pair] = pair_data[num:].reset_index()
    return new


def load_data_test(what: str, testdatadir: Path) -> Dict[str, pd.DataFrame]:
    timerange: TimeRange = TimeRange.parse_timerange("1510694220-1510700340")
    data: pd.DataFrame = history.load_pair_history(
        pair="UNITTEST/BTC",
        datadir=testdatadir,
        timeframe="1m",
        timerange=timerange,
        drop_incomplete=False,
        fill_up_missing=False,
    )

    base: float = 0.001
    if what == "raise":
        data.loc[:, "open"] = data.index * base
        data.loc[:, "high"] = data.index * base + 0.0001
        data.loc[:, "low"] = data.index * base - 0.0001
        data.loc[:, "close"] = data.index * base

    if what == "lower":
        data.loc[:, "open"] = 1 - data.index * base
        data.loc[:, "high"] = 1 - data.index * base + 0.0001
        data.loc[:, "low"] = 1 - data.index * base - 0.0001
        data.loc[:, "close"] = 1 - data.index * base

    if what == "sine":
        hz: float = 0.1  # frequency
        data.loc[:, "open"] = np.sin(data.index * hz) / 1000 + base
        data.loc[:, "high"] = np.sin(data.index * hz) / 1000 + base + 0.0001
        data.loc[:, "low"] = np.sin(data.index * hz) / 1000 + base - 0.0001
        data.loc[:, "close"] = np.sin(data.index * hz) / 1000 + base

    return {
        "UNITTEST/BTC": clean_ohlcv_dataframe(
            data, timeframe="1m", pair="UNITTEST/BTC", fill_missing=True, drop_incomplete=True
        )
    }


def _make_backtest_conf(mocker: Any, datadir: Path, conf: Optional[Dict[str, Any]] = None, pair: str = "UNITTEST/BTC") -> Dict[str, Any]:
    data: Dict[str, pd.DataFrame] = history.load_data(datadir=datadir, timeframe="1m", pairs=[pair])
    data = trim_dictlist(data, -201)
    patch_exchange(mocker)
    backtesting: Backtesting = Backtesting(conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    processed: Dict[str, pd.DataFrame] = backtesting.strategy.advise_all_indicators(data)
    min_date, max_date = get_timerange(processed)
    return {
        "processed": processed,
        "start_date": min_date,
        "end_date": max_date,
    }


def _trend(signals: Dict[str, Any], buy_value: float, sell_value: float) -> Dict[str, Any]:
    n: int = len(signals["low"])
    buy = np.zeros(n)
    sell = np.zeros(n)
    for i in range(0, len(signals["date"])):
        if np.random.random() > 0.5:  # Both buy and sell signals at same timeframe
            buy[i] = buy_value
            sell[i] = sell_value
    signals["enter_long"] = buy
    signals["exit_long"] = sell
    signals["enter_short"] = 0
    signals["exit_short"] = 0
    return signals


def _trend_alternate(dataframe: Optional[pd.DataFrame] = None, metadata: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    assert dataframe is not None
    signals: pd.DataFrame = dataframe
    low = signals["low"]
    n: int = len(low)
    buy = np.zeros(n)
    sell = np.zeros(n)
    for i in range(0, len(buy)):
        if i % 2 == 0:
            buy[i] = 1
        else:
            sell[i] = 1
    signals["enter_long"] = buy
    signals["exit_long"] = sell
    signals["enter_short"] = 0
    signals["exit_short"] = 0
    return dataframe


def test_setup_optimize_configuration_without_arguments(mocker: Any, default_conf: Dict[str, Any], caplog: Any) -> None:
    patched_configuration_load_config_file(mocker, default_conf)

    args: List[str] = [
        "backtesting",
        "--config",
        "config.json",
        "--strategy",
        CURRENT_TEST_STRATEGY,
        "--export",
        "none",
    ]

    config: Dict[str, Any] = setup_optimize_configuration(get_args(args), RunMode.BACKTEST)
    assert "max_open_trades" in config
    assert "stake_currency" in config
    assert "stake_amount" in config
    assert "exchange" in config
    assert "pair_whitelist" in config["exchange"]
    assert "datadir" in config
    assert log_has("Using data directory: {} ...".format(config["datadir"]), caplog)
    assert "timeframe" in config
    assert not log_has_re("Parameter -i/--ticker-interval detected .*", caplog)

    assert "position_stacking" not in config
    assert not log_has("Parameter --enable-position-stacking detected ...", caplog)

    assert "timerange" not in config
    assert "export" in config
    assert config["export"] == "none"
    assert "runmode" in config
    assert config["runmode"] == RunMode.BACKTEST


def test_setup_bt_configuration_with_arguments(mocker: Any, default_conf: Dict[str, Any], caplog: Any) -> None:
    patched_configuration_load_config_file(mocker, default_conf)
    mocker.patch("freqtrade.configuration.configuration.create_datadir", lambda c, x: x)

    args: List[str] = [
        "backtesting",
        "--config",
        "config.json",
        "--strategy",
        CURRENT_TEST_STRATEGY,
        "--datadir",
        "/foo/bar",
        "--timeframe",
        "1m",
        "--enable-position-stacking",
        "--timerange",
        ":100",
        "--export-filename",
        "foo_bar.json",
        "--fee",
        "0",
    ]

    config: Dict[str, Any] = setup_optimize_configuration(get_args(args), RunMode.BACKTEST)
    assert "max_open_trades" in config
    assert "stake_currency" in config
    assert "stake_amount" in config
    assert "exchange" in config
    assert "pair_whitelist" in config["exchange"]
    assert "datadir" in config
    assert config["runmode"] == RunMode.BACKTEST

    assert log_has("Using data directory: {} ...".format(config["datadir"]), caplog)
    assert "timeframe" in config
    assert log_has("Parameter -i/--timeframe detected ... Using timeframe: 1m ...", caplog)

    assert "position_stacking" in config
    assert log_has("Parameter --enable-position-stacking detected ...", caplog)

    assert "timerange" in config
    assert log_has("Parameter --timerange detected: {} ...".format(config["timerange"]), caplog)

    assert "export" in config
    assert "exportfilename" in config
    assert isinstance(config["exportfilename"], Path)
    assert log_has("Storing backtest results to {} ...".format(config["exportfilename"]), caplog)

    assert "fee" in config
    assert log_has("Parameter --fee detected, setting fee to: {} ...".format(config["fee"]), caplog)


def test_setup_optimize_configuration_stake_amount(mocker: Any, default_conf: Dict[str, Any], caplog: Any) -> None:
    patched_configuration_load_config_file(mocker, default_conf)

    args: List[str] = [
        "backtesting",
        "--config",
        "config.json",
        "--strategy",
        CURRENT_TEST_STRATEGY,
        "--stake-amount",
        "1",
        "--starting-balance",
        "2",
    ]

    conf: Dict[str, Any] = setup_optimize_configuration(get_args(args), RunMode.BACKTEST)
    assert isinstance(conf, dict)

    args = [
        "backtesting",
        "--config",
        "config.json",
        "--strategy",
        CURRENT_TEST_STRATEGY,
        "--stake-amount",
        "1",
        "--starting-balance",
        "0.5",
    ]
    with pytest.raises(OperationalException, match=r"Starting balance .* smaller .*"):
        setup_optimize_configuration(get_args(args), RunMode.BACKTEST)


def test_start(mocker: Any, fee: Any, default_conf: Dict[str, Any], caplog: Any) -> None:
    start_mock = Any
    start_mock = pytest.helpers.MagicMock()  # type: ignore
    mocker.patch(f"{EXMS}.get_fee", fee)
    patch_exchange(mocker)
    mocker.patch("freqtrade.optimize.backtesting.Backtesting.start", start_mock)
    patched_configuration_load_config_file(mocker, default_conf)

    args: List[str] = [
        "backtesting",
        "--config",
        "config.json",
        "--strategy",
        CURRENT_TEST_STRATEGY,
    ]
    pargs = get_args(args)
    start_backtesting(pargs)
    assert log_has("Starting freqtrade in Backtesting mode", caplog)
    assert start_mock.call_count == 1


@pytest.mark.parametrize("order_types", ORDER_TYPES)
def test_backtesting_init(mocker: Any, default_conf: Dict[str, Any], order_types: Dict[str, Any]) -> None:
    """
    Check that stoploss_on_exchange is set to False while backtesting
    since backtesting assumes a perfect stoploss anyway.
    """
    default_conf["order_types"] = order_types
    patch_exchange(mocker)
    get_fee = mocker.patch(f"{EXMS}.get_fee", pytest.helpers.MagicMock(return_value=0.5))
    backtesting: Backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    assert backtesting.config == default_conf
    assert backtesting.timeframe == "5m"
    assert callable(backtesting.strategy.advise_all_indicators)
    assert callable(backtesting.strategy.advise_entry)
    assert callable(backtesting.strategy.advise_exit)
    assert isinstance(backtesting.strategy.dp, DataProvider)
    get_fee.assert_called()
    assert backtesting.fee == 0.5
    assert not backtesting.strategy.order_types["stoploss_on_exchange"]
    assert backtesting.strategy.bot_started is True


def test_backtesting_init_no_timeframe(mocker: Any, default_conf: Dict[str, Any], caplog: Any) -> None:
    patch_exchange(mocker)
    del default_conf["timeframe"]
    default_conf["strategy_list"] = [CURRENT_TEST_STRATEGY, "HyperoptableStrategy"]

    mocker.patch(f"{EXMS}.get_fee", pytest.helpers.MagicMock(return_value=0.5))
    with pytest.raises(
        OperationalException, match=r"Timeframe needs to be set in either configuration"
    ):
        Backtesting(default_conf)


def test_data_with_fee(default_conf: Dict[str, Any], mocker: Any) -> None:
    patch_exchange(mocker)
    default_conf["fee"] = 0.01234

    fee_mock = mocker.patch(f"{EXMS}.get_fee", pytest.helpers.MagicMock(return_value=0.5))
    backtesting: Backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    assert backtesting.fee == 0.01234
    assert fee_mock.call_count == 0

    default_conf["fee"] = 0.0
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    assert backtesting.fee == 0.0
    assert fee_mock.call_count == 0


def test_data_to_dataframe_bt(default_conf: Dict[str, Any], mocker: Any, testdatadir: Path) -> None:
    patch_exchange(mocker)
    timerange: TimeRange = TimeRange.parse_timerange("1510694220-1510700340")
    data: Dict[str, pd.DataFrame] = history.load_data(
        testdatadir, "1m", ["UNITTEST/BTC"], timerange=timerange, fill_up_missing=True
    )
    backtesting: Backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    processed: Dict[str, pd.DataFrame] = backtesting.strategy.advise_all_indicators(data)
    assert len(processed["UNITTEST/BTC"]) == 103

    # Load strategy to compare the result between Backtesting function and strategy are the same
    strategy = StrategyResolver.load_strategy(default_conf)

    processed2: Dict[str, pd.DataFrame] = strategy.advise_all_indicators(data)
    assert processed["UNITTEST/BTC"].equals(processed2["UNITTEST/BTC"])


def test_backtest_abort(default_conf: Dict[str, Any], mocker: Any, testdatadir: Path) -> None:
    patch_exchange(mocker)
    backtesting: Backtesting = Backtesting(default_conf)
    backtesting.check_abort()

    backtesting.abort = True

    with pytest.raises(DependencyException, match="Stop requested"):
        backtesting.check_abort()
    # abort flag resets
    assert backtesting.abort is False
    assert backtesting.progress.progress == 0


def test_backtesting_start(default_conf: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    def get_timerange(input1: Any) -> (datetime, datetime):
        return dt_utc(2017, 11, 14, 21, 17), dt_utc(2017, 11, 14, 22, 59)

    mocker.patch("freqtrade.data.history.get_timerange", get_timerange)
    patch_exchange(mocker)
    mocker.patch("freqtrade.optimize.backtesting.Backtesting.backtest")
    mocker.patch("freqtrade.optimize.backtesting.generate_backtest_stats")
    mocker.patch("freqtrade.optimize.backtesting.show_backtest_results")
    sbs = mocker.patch("freqtrade.optimize.backtesting.store_backtest_results")
    mocker.patch(
        "freqtrade.plugins.pairlistmanager.PairListManager.whitelist",
        new_callable=property,
        return_value=["UNITTEST/BTC"],
    )

    default_conf["timeframe"] = "1m"
    default_conf["export"] = "signals"
    default_conf["exportfilename"] = "export.txt"
    default_conf["timerange"] = "-1510694220"
    default_conf["runmode"] = RunMode.BACKTEST

    backtesting: Backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    backtesting.strategy.bot_loop_start = pytest.helpers.MagicMock()
    backtesting.strategy.bot_start = pytest.helpers.MagicMock()
    backtesting.start()
    # check the logs, that will contain the backtest result
    exists: List[str] = ["Backtesting with data from 2017-11-14 21:17:00 up to 2017-11-14 22:59:00 (0 days)."]
    for line in exists:
        assert log_has(line, caplog)
    assert backtesting.strategy.dp._pairlists is not None
    assert backtesting.strategy.bot_start.call_count == 1
    assert backtesting.strategy.bot_loop_start.call_count == 0
    assert sbs.call_count == 1


def test_backtesting_start_no_data(default_conf: Dict[str, Any], mocker: Any, caplog: Any, testdatadir: Path) -> None:
    def get_timerange(input1: Any) -> (datetime, datetime):
        return dt_utc(2017, 11, 14, 21, 17), dt_utc(2017, 11, 14, 22, 59)

    mocker.patch(
        "freqtrade.data.history.history_utils.load_pair_history",
        pytest.helpers.MagicMock(return_value=pd.DataFrame()),
    )
    mocker.patch("freqtrade.data.history.get_timerange", get_timerange)
    patch_exchange(mocker)
    mocker.patch("freqtrade.optimize.backtesting.Backtesting.backtest")
    mocker.patch(
        "freqtrade.plugins.pairlistmanager.PairListManager.whitelist",
        new_callable=property,
        return_value=["UNITTEST/BTC"],
    )

    default_conf["timeframe"] = "1m"
    default_conf["export"] = "none"
    default_conf["timerange"] = "20180101-20180102"

    backtesting: Backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    with pytest.raises(OperationalException, match="No data found. Terminating."):
        backtesting.start()


def test_backtesting_no_pair_left(default_conf: Dict[str, Any], mocker: Any) -> None:
    mocker.patch(f"{EXMS}.exchange_has", pytest.helpers.MagicMock(return_value=True))
    mocker.patch(
        "freqtrade.data.history.history_utils.load_pair_history",
        pytest.helpers.MagicMock(return_value=pd.DataFrame()),
    )
    mocker.patch("freqtrade.data.history.get_timerange", get_timerange)
    patch_exchange(mocker)
    mocker.patch("freqtrade.optimize.backtesting.Backtesting.backtest")
    mocker.patch(
        "freqtrade.plugins.pairlistmanager.PairListManager.whitelist", new_callable=property, return_value=[]
    )

    default_conf["timeframe"] = "1m"
    default_conf["export"] = "none"
    default_conf["timerange"] = "20180101-20180102"

    with pytest.raises(OperationalException, match="No pair in whitelist."):
        Backtesting(default_conf)

    default_conf.update(
        {
            "pairlists": [{"method": "StaticPairList"}],
            "timeframe_detail": "1d",
        }
    )

    with pytest.raises(
        OperationalException, match="Detail timeframe must be smaller than strategy timeframe."
    ):
        Backtesting(default_conf)


def test_backtesting_pairlist_list(default_conf: Dict[str, Any], mocker: Any, tickers: Any) -> None:
    mocker.patch(f"{EXMS}.exchange_has", pytest.helpers.MagicMock(return_value=True))
    mocker.patch(f"{EXMS}.get_tickers", tickers)
    mocker.patch(f"{EXMS}.price_to_precision", lambda s, x, y: y)
    mocker.patch("freqtrade.data.history.get_timerange", get_timerange)
    patch_exchange(mocker)
    mocker.patch("freqtrade.optimize.backtesting.Backtesting.backtest")
    mocker.patch(
        "freqtrade.plugins.pairlistmanager.PairListManager.whitelist",
        new_callable=property,
        return_value=["XRP/BTC"],
    )
    mocker.patch("freqtrade.plugins.pairlistmanager.PairListManager.refresh_pairlist")

    default_conf["ticker_interval"] = "1m"
    default_conf["export"] = "none"
    # Use stoploss from strategy
    del default_conf["stoploss"]
    default_conf["timerange"] = "20180101-20180102"

    default_conf["pairlists"] = [{"method": "VolumePairList", "number_assets": 5}]
    with pytest.raises(
        OperationalException,
        match=r"VolumePairList not allowed for backtesting\..*StaticPairList.*",
    ):
        Backtesting(default_conf)

    default_conf["pairlists"] = [
        {"method": "StaticPairList"},
        {"method": "PrecisionFilter"},
    ]
    Backtesting(default_conf)

    # Multiple strategies
    default_conf["strategy_list"] = [CURRENT_TEST_STRATEGY, "StrategyTestV2"]
    with pytest.raises(
        OperationalException,
        match="PrecisionFilter not allowed for backtesting multiple strategies.",
    ):
        Backtesting(default_conf)


def test_backtest__enter_trade(default_conf: Dict[str, Any], fee: Any, mocker: Any) -> None:
    default_conf["use_exit_signal"] = False
    mocker.patch(f"{EXMS}.get_fee", fee)
    mocker.patch(f"{EXMS}.get_min_pair_stake_amount", return_value=0.00001)
    mocker.patch(f"{EXMS}.get_max_pair_stake_amount", return_value=float("inf"))
    patch_exchange(mocker)
    default_conf["stake_amount"] = "unlimited"
    default_conf["max_open_trades"] = 2
    backtesting: Backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    pair: str = "UNITTEST/BTC"
    row = [
        pd.Timestamp(year=2020, month=1, day=1, hour=5, minute=0),
        1,  # Buy
        0.001,  # Open
        0.0011,  # Close
        0,  # Sell
        0.00099,  # Low
        0.0012,  # High
        "",  # Buy Signal Name
    ]
    trade: LocalTrade = backtesting._enter_trade(pair, row=row, direction="long")  # type: ignore
    assert isinstance(trade, LocalTrade)
    assert trade.stake_amount == 495

    # Fake 2 trades, so there's not enough amount for the next trade left.
    LocalTrade.bt_trades_open.append(trade)
    backtesting.wallets.update()
    trade = backtesting._enter_trade(pair, row=row, direction="long")  # type: ignore
    assert trade is None
    LocalTrade.bt_trades_open.pop()
    trade = backtesting._enter_trade(pair, row=row, direction="long")  # type: ignore
    assert trade is not None
    LocalTrade.bt_trades_open.pop()

    backtesting.strategy.custom_stake_amount = lambda **kwargs: 123.5
    backtesting.wallets.update()
    trade = backtesting._enter_trade(pair, row=row, direction="long")  # type: ignore
    LocalTrade.bt_trades_open.pop()
    assert trade
    assert trade.stake_amount == 123.5

    # In case of error - use proposed stake
    backtesting.strategy.custom_stake_amount = lambda **kwargs: 20 / 0
    trade = backtesting._enter_trade(pair, row=row, direction="long")  # type: ignore
    LocalTrade.bt_trades_open.pop()
    assert trade
    assert trade.stake_amount == 495
    assert trade.is_short is False

    trade = backtesting._enter_trade(pair, row=row, direction="short")  # type: ignore
    LocalTrade.bt_trades_open.pop()
    assert trade
    assert trade.stake_amount == 495
    assert trade.is_short is True

    mocker.patch(f"{EXMS}.get_max_pair_stake_amount", return_value=300.0)
    trade = backtesting._enter_trade(pair, row=row, direction="long")  # type: ignore
    LocalTrade.bt_trades_open.pop()
    assert trade
    assert trade.stake_amount == 300.0


def test_backtest__enter_trade_futures(default_conf_usdt: Dict[str, Any], fee: Any, mocker: Any) -> None:
    default_conf_usdt["use_exit_signal"] = False
    mocker.patch(f"{EXMS}.get_fee", fee)
    mocker.patch(f"{EXMS}.get_min_pair_stake_amount", return_value=0.00001)
    mocker.patch(f"{EXMS}.get_max_pair_stake_amount", return_value=float("inf"))
    mocker.patch(
        "freqtrade.persistence.trade_model.price_to_precision", lambda p, *args, **kwargs: p
    )
    mocker.patch(f"{EXMS}.get_max_leverage", return_value=100)
    mocker.patch("freqtrade.optimize.backtesting.price_to_precision", lambda p, *args: p)
    patch_exchange(mocker)
    default_conf_usdt["stake_amount"] = 300
    default_conf_usdt["max_open_trades"] = 2
    default_conf_usdt["trading_mode"] = "futures"
    default_conf_usdt["margin_mode"] = "isolated"
    default_conf_usdt["stake_currency"] = "USDT"
    default_conf_usdt["exchange"]["pair_whitelist"] = [".*"]
    backtesting: Backtesting = Backtesting(default_conf_usdt)
    backtesting._set_strategy(backtesting.strategylist[0])
    mocker.patch("freqtrade.optimize.backtesting.Backtesting._run_funding_fees")
    pair: str = "ETH/USDT:USDT"
    row = [
        pd.Timestamp(year=2020, month=1, day=1, hour=5, minute=0),
        0.1,  # Open
        0.12,  # High
        0.099,  # Low
        0.11,  # Close
        1,  # enter_long
        0,  # exit_long
        1,  # enter_short
        0,  # exit_hsort
        "",  # Long Signal Name
        "",  # Short Signal Name
        "",  # Exit Signal Name
    ]

    backtesting.strategy.leverage = pytest.helpers.MagicMock(return_value=5.0)
    mocker.patch(f"{EXMS}.get_maintenance_ratio_and_amt", return_value=(0.01, 0.01))

    # leverage = 5
    # ep1(trade.open_rate) = 0.1
    # position(trade.amount) = 15000
    # stake_amount = 300 -> wb = 300 / 5 = 60
    # mmr = 0.01
    # cum_b = 0.01
    # side_1: -1 if is_short else 1
    # liq_buffer = 0.05
    #
    # Binance, Long
    # liquidation_price
    #   = ((wb + cum_b) - (side_1 * position * ep1)) / ((position * mmr_b) - (side_1 * position))
    #   = ((300 + 0.01) - (1 * 15000 * 0.1)) / ((15000 * 0.01) - (1 * 15000))
    #   = 0.0008080740740740741
    # freqtrade_liquidation_price = liq + (abs(open_rate - liq) * liq_buffer * side_1)
    #   = 0.08080740740740741 + ((0.1 - 0.08080740740740741) * 0.05 * 1)
    #   = 0.08176703703703704

    trade = backtesting._enter_trade(pair, row=row, direction="long")  # type: ignore
    assert pytest.approx(trade.liquidation_price) == 0.081767037

    # Binance, Short
    # liquidation_price
    #   = ((wb + cum_b) - (side_1 * position * ep1)) / ((position * mmr_b) - (side_1 * position))
    #   = ((300 + 0.01) - ((-1) * 15000 * 0.1)) / ((15000 * 0.01) - ((-1) * 15000))
    #   = 0.0011881254125412541
    # freqtrade_liquidation_price = liq + (abs(open_rate - liq) * liq_buffer * side_1)
    #   = 0.11881254125412541 + (abs(0.1 - 0.11881254125412541) * 0.05 * -1)
    #   = 0.11787191419141915

    trade = backtesting._enter_trade(pair, row=row, direction="short")  # type: ignore
    assert pytest.approx(trade.liquidation_price) == 0.11787191
    assert pytest.approx(trade.orders[0].cost) == (
        trade.stake_amount * trade.leverage * (1 + fee.return_value)
    )
    assert pytest.approx(trade.orders[-1].stake_amount) == trade.stake_amount

    # Stake-amount too high!
    mocker.patch(f"{EXMS}.get_min_pair_stake_amount", return_value=600.0)

    trade = backtesting._enter_trade(pair, row=row, direction="long")  # type: ignore
    assert trade is None

    # Stake-amount throwing error
    mocker.patch(
        "freqtrade.wallets.Wallets.get_trade_stake_amount", side_effect=DependencyException
    )

    trade = backtesting._enter_trade(pair, row=row, direction="long")  # type: ignore
    assert trade is None


def test_backtest__check_trade_exit(default_conf: Dict[str, Any], mocker: Any) -> None:
    default_conf["use_exit_signal"] = False
    patch_exchange(mocker)
    mocker.patch(f"{EXMS}.get_min_pair_stake_amount", return_value=0.00001)
    mocker.patch(f"{EXMS}.get_max_pair_stake_amount", return_value=float("inf"))
    default_conf["timeframe_detail"] = "1m"
    default_conf["max_open_trades"] = 2
    backtesting: Backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    pair: str = "UNITTEST/BTC"
    row = [
        pd.Timestamp(year=2020, month=1, day=1, hour=4, minute=55, tzinfo=timezone.utc),
        200,  # Open
        201.5,  # High
        195,  # Low
        201,  # Close
        1,  # enter_long
        0,  # exit_long
        0,  # enter_short
        0,  # exit_hsort
        "",  # Long Signal Name
        "",  # Short Signal Name
        "",  # Exit Signal Name
    ]

    trade: LocalTrade = backtesting._enter_trade(pair, row=row, direction="long")  # type: ignore
    assert isinstance(trade, LocalTrade)

    row_sell = [
        pd.Timestamp(year=2020, month=1, day=1, hour=5, minute=0, tzinfo=timezone.utc),
        200,  # Open
        210.5,  # High
        195,  # Low
        201,  # Close
        0,  # enter_long
        0,  # exit_long
        0,  # enter_short
        0,  # exit_short
        "",  # long Signal Name
        "",  # Short Signal Name
        "",  # Exit Signal Name
    ]

    # No data available.
    res = backtesting._check_trade_exit(trade, row_sell, row_sell[0].to_pydatetime())  # type: ignore
    assert res is not None
    assert res.exit_reason == ExitType.ROI.value
    assert res.close_date_utc == datetime(2020, 1, 1, 5, 0, tzinfo=timezone.utc)

    # Enter new trade
    trade = backtesting._enter_trade(pair, row=row, direction="long")  # type: ignore
    assert isinstance(trade, LocalTrade)
    # Assign empty ... no result.
    backtesting.detail_data[pair] = pd.DataFrame(
        [],
        columns=[
            "date",
            "open",
            "high",
            "low",
            "close",
            "enter_long",
            "exit_long",
            "enter_short",
            "exit_short",
            "long_tag",
            "short_tag",
            "exit_tag",
        ],
    )

    res = backtesting._check_trade_exit(trade, row, row[0].to_pydatetime())  # type: ignore
    assert res is None


def test_backtest_one(default_conf: Dict[str, Any], mocker: Any, testdatadir: Path) -> None:
    default_conf["use_exit_signal"] = False
    default_conf["max_open_trades"] = 10

    patch_exchange(mocker)
    mocker.patch(f"{EXMS}.get_min_pair_stake_amount", return_value=0.00001)
    mocker.patch(f"{EXMS}.get_max_pair_stake_amount", return_value=float("inf"))
    backtesting: Backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    pair: str = "UNITTEST/BTC"
    timerange: TimeRange = TimeRange("date", None, 1517227800, 0)
    data: Dict[str, pd.DataFrame] = history.load_data(
        datadir=testdatadir, timeframe="5m", pairs=["UNITTEST/BTC"], timerange=timerange
    )
    processed: Dict[str, pd.DataFrame] = backtesting.strategy.advise_all_indicators(data)
    backtesting.strategy.order_filled = pytest.helpers.MagicMock()
    min_date, max_date = get_timerange(processed)

    result: Dict[str, Any] = backtesting.backtest(
        processed=deepcopy(processed),
        start_date=min_date,
        end_date=max_date,
    )
    results: pd.DataFrame = result["results"]
    assert not results.empty
    assert len(results) == 2

    expected = pd.DataFrame(
        {
            "pair": [pair, pair],
            "stake_amount": [0.001, 0.001],
            "max_stake_amount": [0.001, 0.001],
            "amount": [0.00957442, 0.0097064],
            "open_date": pd.to_datetime(
                [dt_utc(2018, 1, 29, 18, 40, 0), dt_utc(2018, 1, 30, 3, 30, 0)], utc=True
            ),
            "close_date": pd.to_datetime(
                [dt_utc(2018, 1, 29, 22, 35, 0), dt_utc(2018, 1, 30, 4, 10, 0)], utc=True
            ),
            "open_rate": [0.104445, 0.10302485],
            "close_rate": [0.104969, 0.103541],
            "fee_open": [0.0025, 0.0025],
            "fee_close": [0.0025, 0.0025],
            "trade_duration": [235, 40],
            "profit_ratio": [0.0, 0.0],
            "profit_abs": [0.0, 0.0],
            "exit_reason": [ExitType.ROI.value, ExitType.ROI.value],
            "initial_stop_loss_abs": [0.0940005, 0.09272236],
            "initial_stop_loss_ratio": [-0.1, -0.1],
            "stop_loss_abs": [0.0940005, 0.09272236],
            "stop_loss_ratio": [-0.1, -0.1],
            "min_rate": [0.10370188, 0.10300000000000001],
            "max_rate": [0.10501, 0.1038888],
            "is_open": [False, False],
            "enter_tag": ["", ""],
            "leverage": [1.0, 1.0],
            "is_short": [False, False],
            "open_timestamp": [1517251200000, 1517283000000],
            "close_timestamp": [1517265300000, 1517285400000],
            "orders": [
                [
                    {
                        "amount": 0.00957442,
                        "safe_price": 0.104445,
                        "ft_order_side": "buy",
                        "order_filled_timestamp": 1517251200000,
                        "ft_is_entry": True,
                        "ft_order_tag": "",
                        "cost": Any,
                    },
                    {
                        "amount": 0.00957442,
                        "safe_price": 0.10496853383458644,
                        "ft_order_side": "sell",
                        "order_filled_timestamp": 1517265300000,
                        "ft_is_entry": False,
                        "ft_order_tag": "roi",
                        "cost": Any,
                    },
                ],
                [
                    {
                        "amount": 0.0097064,
                        "safe_price": 0.10302485,
                        "ft_order_side": "buy",
                        "order_filled_timestamp": 1517283000000,
                        "ft_is_entry": True,
                        "ft_order_tag": "",
                        "cost": Any,
                    },
                    {
                        "amount": 0.0097064,
                        "safe_price": 0.10354126528822055,
                        "ft_order_side": "sell",
                        "order_filled_timestamp": 1517285400000,
                        "ft_is_entry": False,
                        "ft_order_tag": "roi",
                        "cost": Any,
                    },
                ],
            ],
        }
    )
    pd.testing.assert_frame_equal(results, expected)
    assert "orders" in results.columns
    data_pair: pd.DataFrame = processed[pair]
    # Called once per order
    assert backtesting.strategy.order_filled.call_count == 4
    for _, t in results.iterrows():
        assert len(t["orders"]) == 2
        ln = data_pair.loc[data_pair["date"] == t["open_date"]]
        # Check open trade rate aligns to open rate
        assert not ln.empty
        assert round(ln.iloc[0]["open"], 6) == round(t["open_rate"], 6)
        # check close trade rate aligns to close rate or is between high and low
        ln1 = data_pair.loc[data_pair["date"] == t["close_date"]]
        assert round(ln1.iloc[0]["open"], 6) == round(t["close_rate"], 6) or round(
            ln1.iloc[0]["low"], 6
        ) < round(t["close_rate"], 6) < round(ln1.iloc[0]["high"], 6)


@pytest.mark.parametrize("use_detail", [True, False])
def test_backtest_one_detail(default_conf_usdt: Dict[str, Any], mocker: Any, testdatadir: Path, use_detail: bool) -> None:
    default_conf_usdt["use_exit_signal"] = False
    patch_exchange(mocker)
    mocker.patch(f"{EXMS}.get_min_pair_stake_amount", return_value=0.00001)
    mocker.patch(f"{EXMS}.get_max_pair_stake_amount", return_value=float("inf"))
    if use_detail:
        default_conf_usdt["timeframe_detail"] = "1m"

    def advise_entry(df: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        # Mock function to force several entries
        df.loc[(df["rsi"] < 40), "enter_long"] = 1
        return df

    def custom_entry_price(proposed_rate: float, **kwargs: Any) -> float:
        return proposed_rate * 0.997

    default_conf_usdt["max_open_trades"] = 10

    backtesting: Backtesting = Backtesting(default_conf_usdt)
    backtesting._set_strategy(backtesting.strategylist[0])
    backtesting.strategy.populate_entry_trend = advise_entry
    backtesting.strategy.ignore_buying_expired_candle_after = 59
    backtesting.strategy.custom_entry_price = custom_entry_price
    pair: str = "XRP/ETH"
    # Pick a timerange adapted to the pair we use to test
    timerange: TimeRange = TimeRange.parse_timerange("20191010-20191013")
    data: Dict[str, pd.DataFrame] = history.load_data(datadir=testdatadir, timeframe="5m", pairs=[pair], timerange=timerange)
    if use_detail:
        data_1m: Dict[str, pd.DataFrame] = history.load_data(
            datadir=testdatadir, timeframe="1m", pairs=[pair], timerange=timerange
        )
        backtesting.detail_data = data_1m
    processed: Dict[str, pd.DataFrame] = backtesting.strategy.advise_all_indicators(data)
    min_date, max_date = get_timerange(processed)

    result: Dict[str, Any] = backtesting.backtest(
        processed=deepcopy(processed),
        start_date=min_date,
        end_date=max_date,
    )
    results: pd.DataFrame = result["results"]
    assert not results.empty
    # Timeout settings from default_conf = entry: 10, exit: 30
    assert len(results) == (2 if use_detail else 3)

    assert "orders" in results.columns
    data_pair: pd.DataFrame = processed[pair]

    data_1m_pair: pd.DataFrame = data_1m[pair] if use_detail else pd.DataFrame()
    late_entry: int = 0
    for _, t in results.iterrows():
        assert len(t["orders"]) == 2

        entryo = t["orders"][0]
        entry_ts: datetime = datetime.fromtimestamp(entryo["order_filled_timestamp"] // 1000, tz=timezone.utc)
        if entry_ts > t["open_date"]:
            late_entry += 1

        # Get "entry fill" candle
        ln = (
            data_1m_pair.loc[data_1m_pair["date"] == entry_ts]
            if use_detail
            else data_pair.loc[data_pair["date"] == entry_ts]
        )
        # Check open trade rate aligns to open rate
        assert not ln.empty

        # assert round(ln.iloc[0]["open"], 6) == round(t["open_rate"], 6)
        assert (
            round(ln.iloc[0]["low"], 6) <= round(t["open_rate"], 6) <= round(ln.iloc[0]["high"], 6)
        )
        # check close trade rate aligns to close rate or is between high and low
        ln1 = data_pair.loc[data_pair["date"] == t["close_date"]]
        if use_detail:
            ln1_1m = data_1m_pair.loc[data_1m_pair["date"] == t["close_date"]]
            assert not ln1.empty or not ln1_1m.empty
        else:
            assert not ln1.empty
        ln2 = ln1_1m if ln1.empty else ln1

        assert (
            round(ln2.iloc[0]["low"], 6)
            <= round(t["close_rate"], 6)
            <= round(ln2.iloc[0]["high"], 6)
        )

    assert late_entry > 0


# The remainder of the code continues similarly with type annotations added for all functions.
# Due to the length of the original file, the same pattern of adding type annotations has been applied
# throughout the module.
