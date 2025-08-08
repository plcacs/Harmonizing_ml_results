from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock
from zipfile import ZipFile
import pytest
from pandas import DataFrame, Timestamp, to_datetime
from freqtrade.configuration import TimeRange
from freqtrade.constants import LAST_BT_RESULT_FN
from freqtrade.data.btanalysis import BT_DATA_COLUMNS, analyze_trade_parallelism, extract_trades_of_period, get_latest_backtest_filename, get_latest_hyperopt_file, load_backtest_data, load_backtest_metadata, load_file_from_zip, load_trades, load_trades_from_db
from freqtrade.data.history import load_data, load_pair_history
from freqtrade.data.metrics import calculate_cagr, calculate_calmar, calculate_csum, calculate_expectancy, calculate_market_change, calculate_max_drawdown, calculate_sharpe, calculate_sortino, calculate_underwater, combine_dataframes_with_mean, combined_dataframes_with_rel_mean, create_cum_profit
from freqtrade.exceptions import OperationalException
from freqtrade.util import dt_utc
from tests.conftest import CURRENT_TEST_STRATEGY, create_mock_trades
from tests.conftest_trades import MOCK_TRADE_COUNT

def test_get_latest_backtest_filename(testdatadir: Path, mocker: MagicMock) -> None:
def test_get_latest_hyperopt_file(testdatadir: Path) -> None:
def test_load_backtest_metadata(mocker: MagicMock, testdatadir: Path) -> None:
def test_load_backtest_data_old_format(testdatadir: Path, mocker: MagicMock) -> None:
def test_load_backtest_data_new_format(testdatadir: Path) -> None:
def test_load_backtest_data_multi(testdatadir: Path) -> None:
def test_load_trades_from_db(default_conf: dict, fee: float, is_short: bool, mocker: MagicMock) -> None:
def test_extract_trades_of_period(testdatadir: Path) -> None:
def test_analyze_trade_parallelism(testdatadir: Path) -> None:
def test_load_trades(default_conf: dict, mocker: MagicMock) -> None:
def test_calculate_market_change(testdatadir: Path) -> None:
def test_combine_dataframes_with_mean(testdatadir: Path) -> None:
def test_combined_dataframes_with_rel_mean(testdatadir: Path) -> None:
def test_combine_dataframes_with_mean_no_data(testdatadir: Path) -> None:
def test_create_cum_profit(testdatadir: Path) -> None:
def test_create_cum_profit1(testdatadir: Path) -> None:
def test_calculate_max_drawdown(testdatadir: Path) -> None:
def test_calculate_csum(testdatadir: Path) -> None:
def test_calculate_expectancy(testdatadir: Path) -> None:
def test_calculate_sortino(testdatadir: Path) -> None:
def test_calculate_sharpe(testdatadir: Path) -> None:
def test_calculate_calmar(testdatadir: Path) -> None:
def test_calculate_cagr(start: float, end: float, days: int, expected: float) -> None:
def test_calculate_max_drawdown2() -> None:
def test_calculate_max_drawdown_abs(profits: list, relative: bool, highd: int, lowdays: int, result: float, result_rel: float) -> None:
def test_load_file_from_zip(tmp_path: Path) -> None:
