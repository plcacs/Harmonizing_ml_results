from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock
from zipfile import ZipFile
import pytest
from pandas import DataFrame, DateOffset, Timestamp, to_datetime
from freqtrade.configuration import TimeRange
from freqtrade.constants import LAST_BT_RESULT_FN
from freqtrade.data.btanalysis import BT_DATA_COLUMNS, analyze_trade_parallelism, extract_trades_of_period, get_latest_backtest_filename, get_latest_hyperopt_file, load_backtest_data, load_backtest_metadata, load_file_from_zip, load_trades, load_trades_from_db
from freqtrade.data.history import load_data, load_pair_history
from freqtrade.data.metrics import calculate_cagr: (float, float, int) -> float, calculate_calmar: (DataFrame, datetime, datetime, float) -> float, calculate_csum: (DataFrame, float) -> (float, float), calculate_expectancy: (DataFrame) -> (float, float), calculate_market_change: (DataFrame) -> float, calculate_max_drawdown: (DataFrame, datetime, datetime, float) -> Drawdown, calculate_sharpe: (DataFrame, datetime, datetime, float) -> float, calculate_sortino: (DataFrame, datetime, datetime, float) -> float, combine_dataframes_with_mean: (DataFrame) -> DataFrame, combined_dataframes_with_rel_mean: (DataFrame, datetime, datetime) -> DataFrame, create_cum_profit: (DataFrame, DataFrame, str, str) -> DataFrame
from freqtrade.exceptions import OperationalException
from freqtrade.util import dt_utc
from tests.conftest import CURRENT_TEST_STRATEGY: str, create_mock_trades: (float, bool) -> None, MOCK_TRADE_COUNT: int
from tests.conftest_trades import MOCK_TRADE_COUNT: int

def test_get_latest_backtest_filename(testdatadir: Path, mocker: pytest.Mock) -> None:
    # ...

def test_get_latest_hyperopt_file(testdatadir: Path) -> None:
    # ...

def test_load_backtest_metadata(mocker: pytest.Mock, testdatadir: Path) -> None:
    # ...

def test_load_backtest_data_old_format(testdatadir: Path, mocker: pytest.Mock) -> None:
    # ...

def test_load_backtest_data_new_format(testdatadir: Path) -> None:
    # ...

def test_load_trades(default_conf: dict, mocker: pytest.Mock) -> None:
    # ...

def test_calculate_market_change(testdatadir: Path) -> float:
    # ...

def test_combine_dataframes_with_mean(testdatadir: Path) -> DataFrame:
    # ...

def test_combined_dataframes_with_rel_mean(testdatadir: Path) -> DataFrame:
    # ...

def test_create_cum_profit(testdatadir: Path) -> DataFrame:
    # ...

def test_calculate_max_drawdown(testdatadir: Path) -> Drawdown:
    # ...

def test_calculate_csum(testdatadir: Path) -> (float, float):
    # ...

def test_calculate_expectancy(testdatadir: Path) -> (float, float):
    # ...

def test_calculate_sortino(testdatadir: Path) -> float:
    # ...

def test_calculate_sharpe(testdatadir: Path) -> float:
    # ...

def test_calculate_calmar(testdatadir: Path) -> float:
    # ...

def test_calculate_cagr(days: int, start: float, end: float) -> float:
    # ...

def test_calculate_max_drawdown_abs(profits: list, relative: bool, highd: int, lowdays: int, result: float, result_rel: float) -> None:
    # ...

def test_load_file_from_zip(tmp_path: Path) -> None:
    # ...
