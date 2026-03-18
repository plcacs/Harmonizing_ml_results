```python
import pandas as pd
import pytest
from typing import Any

BIN_SIZE_SCALE: float = ...

def read_csv(filename: Any) -> pd.DataFrame: ...

@pytest.fixture
def populate_dataframe_with_trades_dataframe(testdatadir: Any) -> pd.DataFrame: ...

@pytest.fixture
def populate_dataframe_with_trades_trades(testdatadir: Any) -> pd.DataFrame: ...

@pytest.fixture
def candles(testdatadir: Any) -> pd.DataFrame: ...

@pytest.fixture
def public_trades_list(testdatadir: Any) -> pd.DataFrame: ...

@pytest.fixture
def public_trades_list_simple(testdatadir: Any) -> pd.DataFrame: ...

def test_public_trades_columns_before_change(
    populate_dataframe_with_trades_dataframe: Any,
    populate_dataframe_with_trades_trades: Any
) -> None: ...

def test_public_trades_mock_populate_dataframe_with_trades__check_orderflow(
    populate_dataframe_with_trades_dataframe: Any,
    populate_dataframe_with_trades_trades: Any
) -> None: ...

def test_public_trades_trades_mock_populate_dataframe_with_trades__check_trades(
    populate_dataframe_with_trades_dataframe: Any,
    populate_dataframe_with_trades_trades: Any
) -> None: ...

def test_public_trades_put_volume_profile_into_ohlcv_candles(
    public_trades_list_simple: Any,
    candles: Any
) -> None: ...

def test_public_trades_binned_big_sample_list(public_trades_list: Any) -> None: ...

def test_public_trades_config_max_trades(
    default_conf: Any,
    populate_dataframe_with_trades_dataframe: Any,
    populate_dataframe_with_trades_trades: Any
) -> None: ...

def test_public_trades_testdata_sanity(
    candles: Any,
    public_trades_list: Any,
    public_trades_list_simple: Any,
    populate_dataframe_with_trades_dataframe: Any,
    populate_dataframe_with_trades_trades: Any
) -> None: ...

def test_analyze_with_orderflow(
    default_conf_usdt: Any,
    mocker: Any,
    populate_dataframe_with_trades_dataframe: Any,
    populate_dataframe_with_trades_trades: Any
) -> None: ...

def test_stacked_imbalances_multiple_prices() -> None: ...

def test_timeframe_to_DateOffset() -> None: ...
```