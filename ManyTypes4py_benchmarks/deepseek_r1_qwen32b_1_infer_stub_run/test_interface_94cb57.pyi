import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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
from tests.conftest import (
    CURRENT_TEST_STRATEGY,
    TRADE_SIDES,
    log_has,
    log_has_re,
)

from .strats.strategy_test_v3 import StrategyTestV3
from freqtrade.strategy.interface import IStrategy


class StrategyTestV3(IStrategy):
    def populate_indicators(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        ...

    def populate_entry_trend(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        ...

    def populate_exit_trend(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        ...

    def analyze_ticker(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        ...


_STRATEGY: StrategyTestV3 = StrategyTestV3(config={})
_STRATEGY.dp = DataProvider({}, None, None)


def test_returns_latest_signal(ohlcv_history: DataFrame) -> None:
    ...


def test_analyze_pair_empty(mocker: pytest_mock.MockFixture, caplog: pytest.LogCaptureFixture, ohlcv_history: DataFrame) -> None:
    ...


def test_get_signal_empty(default_conf: Dict[str, Any], caplog: pytest.LogCaptureFixture) -> None:
    ...


def test_get_signal_exception_valueerror(
    mocker: pytest_mock.MockFixture,
    caplog: pytest.LogCaptureFixture,
    ohlcv_history: DataFrame,
) -> None:
    ...


def test_get_signal_old_dataframe(
    default_conf: Dict[str, Any],
    mocker: pytest_mock.MockFixture,
    caplog: pytest.LogCaptureFixture,
    ohlcv_history: DataFrame,
) -> None:
    ...


def test_get_signal_no_sell_column(
    default_conf: Dict[str, Any],
    mocker: pytest_mock.MockFixture,
    caplog: pytest.LogCaptureFixture,
    ohlcv_history: DataFrame,
) -> None:
    ...


def test_ignore_expired_candle(default_conf: Dict[str, Any]) -> None:
    ...


def test_assert_df_raise(
    mocker: pytest_mock.MockFixture,
    caplog: pytest.LogCaptureFixture,
    ohlcv_history: DataFrame,
) -> None:
    ...


def test_assert_df(ohlcv_history: DataFrame, caplog: pytest.LogCaptureFixture) -> None:
    ...


def test_advise_all_indicators(default_conf: Dict[str, Any], testdatadir: Path) -> None:
    ...


def test_freqai_not_initialized(default_conf: Dict[str, Any]) -> None:
    ...


def test_advise_all_indicators_copy(
    mocker: pytest_mock.MockFixture,
    default_conf: Dict[str, Any],
    testdatadir: Path,
) -> None:
    ...


def test_min_roi_reached(default_conf: Dict[str, Any], fee: Callable) -> None:
    ...


def test_min_roi_reached2(default_conf: Dict[str, Any], fee: Callable) -> None:
    ...


def test_min_roi_reached3(default_conf: Dict[str, Any], fee: Callable) -> None:
    ...


@pytest.mark.parametrize(
    'profit,adjusted,expected,liq,trailing,custom,profit2,adjusted2,expected2,custom_stop',
    [
        (0.2, 0.9, ExitType.NONE, None, False, False, 0.3, 0.9, ExitType.NONE, None),
        (0.2, 0.9, ExitType.NONE, None, False, False, -0.2, 0.9, ExitType.STOP_LOSS, None),
        (0.2, 1.14, ExitType.NONE, None, True, False, 0.05, 1.14, ExitType.TRAILING_STOP_LOSS, None),
        (0.01, 0.96, ExitType.NONE, None, True, False, 0.05, 0.998, ExitType.NONE, None),
        (0.05, 0.998, ExitType.NONE, None, True, False, -0.01, 0.998, ExitType.TRAILING_STOP_LOSS, None),
        (0.05, 0.945, ExitType.NONE, None, False, True, -0.02, 0.945, ExitType.NONE, None),
        (0.05, 0.945, ExitType.NONE, None, False, True, -0.06, 0.945, ExitType.TRAILING_STOP_LOSS, None),
        (0.05, 0.998, ExitType.NONE, None, False, True, -0.06, 0.998, ExitType.TRAILING_STOP_LOSS, lambda **kwargs: -0.05),
        (0.05, 0.998, ExitType.NONE, None, False, True, 0.09, 1.036, ExitType.NONE, lambda **kwargs: -0.05),
        (0.05, 0.945, ExitType.NONE, None, False, True, 0.09, 0.981, ExitType.NONE, lambda current_profit, **kwargs: -0.1 if current_profit < 0.6 else -(current_profit * 2)),
        (0.05, 0.9, ExitType.NONE, None, False, True, 0.09, 0.9, ExitType.NONE, lambda **kwargs: None),
        (0.05, 0.9, ExitType.NONE, None, False, True, 0.09, 0.9, ExitType.NONE, lambda **kwargs: math.inf),
    ],
)
def test_ft_stoploss_reached(
    default_conf: Dict[str, Any],
    fee: Callable,
    profit: float,
    adjusted: float,
    expected: ExitType,
    liq: Optional[float],
    trailing: bool,
    custom: bool,
    profit2: float,
    adjusted2: float,
    expected2: ExitType,
    custom_stop: Optional[Callable],
) -> None:
    ...


def test_custom_exit(default_conf: Dict[str, Any], fee: Callable, caplog: pytest.LogCaptureFixture) -> None:
    ...


def test_should_sell(default_conf: Dict[str, Any], fee: Callable) -> None:
    ...


@pytest.mark.parametrize('side', TRADE_SIDES)
def test_leverage_callback(default_conf: Dict[str, Any], side: str) -> None:
    ...


def test_analyze_ticker_default(ohlcv_history: DataFrame, mocker: pytest_mock.MockFixture, caplog: pytest.LogCaptureFixture) -> None:
    ...


def test__analyze_ticker_internal_skip_analyze(ohlcv_history: DataFrame, mocker: pytest_mock.MockFixture, caplog: pytest.LogCaptureFixture) -> None:
    ...


@pytest.mark.usefixtures('init_persistence')
def test_is_pair_locked(default_conf: Dict[str, Any]) -> None:
    ...


def test_is_informative_pairs_callback(default_conf: Dict[str, Any]) -> None:
    ...


def test_hyperopt_parameters() -> None:
    ...


def test_auto_hyperopt_interface(default_conf: Dict[str, Any]) -> None:
    ...


def test_auto_hyperopt_interface_loadparams(
    default_conf: Dict[str, Any],
    mocker: pytest_mock.MockFixture,
    caplog: pytest.LogCaptureFixture,
) -> None:
    ...


@pytest.mark.parametrize('function,raises', [('populate_entry_trend', False), ('advise_entry', False), ('populate_exit_trend', False), ('advise_exit', False)])
def test_pandas_warning_direct(ohlcv_history: DataFrame, function: str, raises: bool, recwarn: pytest.PytestRecwarn) -> None:
    ...


def test_pandas_warning_through_analyze_pair(ohlcv_history: DataFrame, mocker: pytest_mock.MockFixture, recwarn: pytest.PytestRecwarn) -> None:
    ...