# pragma pylint: disable=missing-docstring, W0212, line-too-long, C0103, unused-argument

import random
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import ANY, MagicMock, PropertyMock

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


ORDER_TYPES = [
    {"entry": "limit", "exit": "limit", "stoploss": "limit", "stoploss_on_exchange": False},
    {"entry": "limit", "exit": "limit", "stoploss": "limit", "stoploss_on_exchange": True},
]


def trim_dictlist(dict_list: Dict[str, pd.DataFrame], num: int) -> Dict[str, pd.DataFrame]:
    new = {}
    for pair, pair_data in dict_list.items():
        new[pair] = pair_data[num:].reset_index()
    return new


def load_data_test(what: str, testdatadir: Path) -> Dict[str, pd.DataFrame]:
    timerange = TimeRange.parse_timerange("1510694220-1510700340")
    data = history.load_pair_history(
        pair="UNITTEST/BTC",
        datadir=testdatadir,
        timeframe="1m",
        timerange=timerange,
        drop_incomplete=False,
        fill_up_missing=False,
    )

    base = 0.001
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
        hz = 0.1  # frequency
        data.loc[:, "open"] = np.sin(data.index * hz) / 1000 + base
        data.loc[:, "high"] = np.sin(data.index * hz) / 1000 + base + 0.0001
        data.loc[:, "low"] = np.sin(data.index * hz) / 1000 + base - 0.0001
        data.loc[:, "close"] = np.sin(data.index * hz) / 1000 + base

    return {
        "UNITTEST/BTC": clean_ohlcv_dataframe(
            data, timeframe="1m", pair="UNITTEST/BTC", fill_missing=True, drop_incomplete=True
        )
    }


# FIX: fixturize this?
def _make_backtest_conf(
    mocker: Any, datadir: Path, conf: Optional[Dict[str, Any]] = None, pair: str = "UNITTEST/BTC"
) -> Dict[str, Any]:
    data = history.load_data(datadir=datadir, timeframe="1m", pairs=[pair])
    data = trim_dictlist(data, -201)
    patch_exchange(mocker)
    backtesting = Backtesting(conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    processed = backtesting.strategy.advise_all_indicators(data)
    min_date, max_date = get_timerange(processed)
    return {
        "processed": processed,
        "start_date": min_date,
        "end_date": max_date,
    }


def _trend(signals: pd.DataFrame, buy_value: int, sell_value: int) -> pd.DataFrame:
    n = len(signals["low"])
    buy = np.zeros(n)
    sell = np.zeros(n)
    for i in range(0, len(signals["date"])):
        if random.random() > 0.5:  # Both buy and sell signals at same timeframe
            buy[i] = buy_value
            sell[i] = sell_value
    signals["enter_long"] = buy
    signals["exit_long"] = sell
    signals["enter_short"] = 0
    signals["exit_short"] = 0
    return signals


def _trend_alternate(dataframe: Optional[pd.DataFrame] = None, metadata: Optional[Dict] = None) -> pd.DataFrame:
    signals = dataframe
    low = signals["low"]
    n = len(low)
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
