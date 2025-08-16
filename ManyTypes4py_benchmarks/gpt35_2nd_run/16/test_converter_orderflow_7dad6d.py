from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import pytest
from freqtrade.constants import DEFAULT_TRADES_COLUMNS
from freqtrade.data.converter import populate_dataframe_with_trades
from freqtrade.data.converter.orderflow import ORDERFLOW_ADDED_COLUMNS, stacked_imbalance, timeframe_to_DateOffset, trades_to_volumeprofile_with_total_delta_bid_ask
from freqtrade.data.converter.trade_converter import trades_list_to_df
from freqtrade.data.dataprovider import DataProvider
from tests.strategy.strats.strategy_test_v3 import StrategyTestV3

BIN_SIZE_SCALE: float = 0.5

def read_csv(filename: str) -> pd.DataFrame:
    converter_columns: List[str] = ['side', 'type']
    return pd.read_csv(filename, skipinitialspace=True, index_col=0, parse_dates=True, date_format='ISO8601', converters={col: str.strip for col in converter_columns})

@pytest.fixture
def populate_dataframe_with_trades_dataframe(testdatadir: Path) -> pd.DataFrame:
    return pd.read_feather(testdatadir / 'orderflow/populate_dataframe_with_trades_DF.feather')

@pytest.fixture
def populate_dataframe_with_trades_trades(testdatadir: Path) -> pd.DataFrame:
    return pd.read_feather(testdatadir / 'orderflow/populate_dataframe_with_trades_TRADES.feather')

@pytest.fixture
def candles(testdatadir: Path) -> pd.DataFrame:
    return pd.read_json(testdatadir / 'orderflow/candles.json').copy()

@pytest.fixture
def public_trades_list(testdatadir: Path) -> pd.DataFrame:
    return read_csv(testdatadir / 'orderflow/public_trades_list.csv').copy()

@pytest.fixture
def public_trades_list_simple(testdatadir: Path) -> pd.DataFrame:
    return read_csv(testdatadir / 'orderflow/public_trades_list_simple_example.csv').copy()

def test_public_trades_columns_before_change(populate_dataframe_with_trades_dataframe: pd.DataFrame, populate_dataframe_with_trades_trades: pd.DataFrame) -> None:
    assert populate_dataframe_with_trades_dataframe.columns.tolist() == ['date', 'open', 'high', 'low', 'close', 'volume']
    assert populate_dataframe_with_trades_trades.columns.tolist() == ['timestamp', 'id', 'type', 'side', 'price', 'amount', 'cost', 'date']

def test_public_trades_mock_populate_dataframe_with_trades__check_orderflow(populate_dataframe_with_trades_dataframe: pd.DataFrame, populate_dataframe_with_trades_trades: pd.DataFrame) -> None:
    ...

def test_public_trades_trades_mock_populate_dataframe_with_trades__check_trades(populate_dataframe_with_trades_dataframe: pd.DataFrame, populate_dataframe_with_trades_trades: pd.DataFrame) -> None:
    ...

def test_public_trades_put_volume_profile_into_ohlcv_candles(public_trades_list_simple: pd.DataFrame, candles: pd.DataFrame) -> None:
    ...

def test_public_trades_binned_big_sample_list(public_trades_list: pd.DataFrame) -> None:
    ...

def test_public_trades_config_max_trades(default_conf: Dict[str, Any], populate_dataframe_with_trades_dataframe: pd.DataFrame, populate_dataframe_with_trades_trades: pd.DataFrame) -> None:
    ...

def test_public_trades_testdata_sanity(candles: pd.DataFrame, public_trades_list: pd.DataFrame, public_trades_list_simple: pd.DataFrame, populate_dataframe_with_trades_dataframe: pd.DataFrame, populate_dataframe_with_trades_trades: pd.DataFrame) -> None:
    ...

def test_analyze_with_orderflow(default_conf_usdt: Dict[str, Any], mocker: Any, populate_dataframe_with_trades_dataframe: pd.DataFrame, populate_dataframe_with_trades_trades: pd.DataFrame) -> None:
    ...

def test_stacked_imbalances_multiple_prices() -> None:
    ...

def test_timeframe_to_DateOffset() -> None:
    ...
