from __future__ import annotations
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from unittest.mock import MagicMock
from zipfile import ZipFile
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    NamedTuple,
    Iterable,
    Sequence,
)
import pytest
from pandas import DataFrame, DateOffset, Timestamp
from freqtrade.data.btanalysis import (
    BT_DATA_COLUMNS,
    MaxDrawdownResult,
)
from freqtrade.data.history import TimeRange
from freqtrade.exceptions import OperationalException

def test_get_latest_backtest_filename(testdatadir: Path, mocker: MagicMock) -> None:
    ...

def test_get_latest_hyperopt_file(testdatadir: Path) -> None:
    ...

def test_load_backtest_metadata(mocker: MagicMock, testdatadir: Path) -> None:
    ...

def test_load_backtest_data_old_format(testdatadir: Path, mocker: MagicMock) -> None:
    ...

def test_load_backtest_data_new_format(testdatadir: Path) -> None:
    ...

def test_load_backtest_data_multi(testdatadir: Path) -> None:
    ...

@pytest.mark.usefixtures('init_persistence')
@pytest.mark.parametrize('is_short', [False, True])
def test_load_trades_from_db(default_conf: Dict[str, Any], fee: float, is_short: bool, mocker: MagicMock) -> None:
    ...

def test_extract_trades_of_period(testdatadir: Path) -> None:
    ...

def test_analyze_trade_parallelism(testdatadir: Path) -> None:
    ...

def test_load_trades(default_conf: Dict[str, Any], mocker: MagicMock) -> None:
    ...

def test_calculate_market_change(testdatadir: Path) -> None:
    ...

def test_combine_dataframes_with_mean(testdatadir: Path) -> None:
    ...

def test_combined_dataframes_with_rel_mean(testdatadir: Path) -> None:
    ...

def test_combine_dataframes_with_mean_no_data(testdatadir: Path) -> None:
    ...

def test_create_cum_profit(testdatadir: Path) -> None:
    ...

def test_create_cum_profit1(testdatadir: Path) -> None:
    ...

def test_calculate_max_drawdown(testdatadir: Path) -> None:
    ...

def test_calculate_csum(testdatadir: Path) -> None:
    ...

def test_calculate_expectancy(testdatadir: Path) -> None:
    ...

def test_calculate_sortino(testdatadir: Path) -> None:
    ...

def test_calculate_sharpe(testdatadir: Path) -> None:
    ...

def test_calculate_calmar(testdatadir: Path) -> None:
    ...

@pytest.mark.parametrize('start,end,days, expected', [
    (64900, 176000, 3 * 365, 0.3945),
    (64900, 176000, 365, 1.7119),
    (1000, 1000, 365, 0.0),
    (1000, 1500, 365, 0.5),
    (1000, 1500, 100, 3.3927),
    (0.01, 0.01762792, 120, 4.6087),
])
def test_calculate_cagr(start: float, end: float, days: int, expected: float) -> None:
    ...

def test_calculate_max_drawdown2() -> None:
    ...

@pytest.mark.parametrize('profits,relative,highd,lowdays,result,result_rel', [
    ([0.0, -500.0, 500.0, 10000.0, -1000.0], False, 3, 4, 1000.0, 0.090909),
    ([0.0, -500.0, 500.0, 10000.0, -1000.0], True, 0, 1, 500.0, 0.5),
])
def test_calculate_max_drawdown_abs(
    profits: List[float],
    relative: bool,
    highd: int,
    lowdays: int,
    result: float,
    result_rel: float,
) -> None:
    ...

def test_load_file_from_zip(tmp_path: Path) -> None:
    ...