import pandas as pd
import pytest
from pathlib import Path

BIN_SIZE_SCALE: float

def read_csv(filename: str | Path) -> pd.DataFrame: ...

@pytest.fixture
def populate_dataframe_with_trades_dataframe(testdatadir: Path) -> pd.DataFrame: ...

@pytest.fixture
def populate_dataframe_with_trades_trades(testdatadir: Path) -> pd.DataFrame: ...

@pytest.fixture
def candles(testdatadir: Path) -> pd.DataFrame: ...

@pytest.fixture
def public_trades_list(testdatadir: Path) -> pd.DataFrame: ...

@pytest.fixture
def public_trades_list_simple(testdatadir: Path) -> pd.DataFrame: ...

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

def test_public_trades_binned_big_sample_list(
    public_trades_list: pd.DataFrame,
) -> None: ...

def test_public_trades_config_max_trades(
    default_conf: dict[str, object],
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
    default_conf_usdt: dict[str, object],
    mocker: pytest.MonkeyPatch,
    populate_dataframe_with_trades_dataframe: pd.DataFrame,
    populate_dataframe_with_trades_trades: pd.DataFrame,
) -> None: ...

def test_stacked_imbalances_multiple_prices() -> None: ...

def test_timeframe_to_DateOffset() -> None: ...