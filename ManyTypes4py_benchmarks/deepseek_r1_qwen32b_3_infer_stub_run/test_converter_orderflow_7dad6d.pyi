import pytest
import pandas as pd
from freqtrade.constants import DEFAULT_TRADES_COLUMNS
from freqtrade.data.converter import populate_dataframe_with_trades
from freqtrade.data.converter.orderflow import (
    ORDERFLOW_ADDED_COLUMNS,
    stacked_imbalance,
    timeframe_to_DateOffset,
    trades_to_volumeprofile_with_total_delta_bid_ask,
)
from freqtrade.data.converter.trade_converter import trades_list_to_df
from freqtrade.data.dataprovider import DataProvider
from tests.strategy.strats.strategy_test_v3 import StrategyTestV3

BIN_SIZE_SCALE: float

def read_csv(filename: str) -> pd.DataFrame: ...

@pytest.fixture
def populate_dataframe_with_trades_dataframe(testdatadir: str) -> pd.DataFrame: ...

@pytest.fixture
def populate_dataframe_with_trades_trades(testdatadir: str) -> pd.DataFrame: ...

@pytest.fixture
def candles(testdatadir: str) -> pd.DataFrame: ...

@pytest.fixture
def public_trades_list(testdatadir: str) -> pd.DataFrame: ...

@pytest.fixture
def public_trades_list_simple(testdatadir: str) -> pd.DataFrame: ...

def test_public_trades_columns_before_change(
    populate_dataframe_with_trades_dataframe: pd.DataFrame,
    populate_dataframe_with_trades_trades: pd.DataFrame,
) -> None: ...

def test_public_trades_mock_populate_dataframe_with_trades__check_orderflow(
    populate_dataframe_with_trades_dataframe: pd.DataFrame,
    populate_dataframe_with_trades_trades: pd.DataFrame,
) -> None: ...

def test_public_trades_trades_mock_populate_dataframe_with_trades__check_trades(
    populate_dataframe_with_trades_dataframe: pd.DataFrame,
    populate_dataframe_with_trades_trades: pd.DataFrame,
) -> None: ...

def test_public_trades_put_volume_profile_into_ohlcv_candles(
    public_trades_list_simple: pd.DataFrame,
    candles: pd.DataFrame,
) -> None: ...

def test_public_trades_binned_big_sample_list(public_trades_list: pd.DataFrame) -> None: ...

def test_public_trades_config_max_trades(
    default_conf: dict,
    populate_dataframe_with_trades_dataframe: pd.DataFrame,
    populate_dataframe_with_trades_trades: pd.DataFrame,
) -> None: ...

def test_public_trades_testdata_sanity(
    candles: pd.DataFrame,
    public_trades_list: pd.DataFrame,
    public_trades_list_simple: pd.DataFrame,
    populate_dataframe_with_trades_dataframe: pd.DataFrame,
    populate_dataframe_with_trades_trades: pd.DataFrame,
) -> None: ...

def test_analyze_with_orderflow(
    default_conf_usdt: dict,
    mocker: pytest_mock.MockFixture,
    populate_dataframe_with_trades_dataframe: pd.DataFrame,
    populate_dataframe_with_trades_trades: pd.DataFrame,
) -> None: ...

def test_stacked_imbalances_multiple_prices() -> None: ...

def test_timeframe_to_DateOffset() -> None: ...

def stacked_imbalance(df: pd.DataFrame, side: str, stacked_imbalance_range: int) -> list[float]: ...

def timeframe_to_DateOffset(timeframe: str) -> pd.DateOffset: ...

def trades_to_volumeprofile_with_total_delta_bid_ask(
    trades_df: pd.DataFrame,
    scale: float = ...,
) -> pd.DataFrame: ...