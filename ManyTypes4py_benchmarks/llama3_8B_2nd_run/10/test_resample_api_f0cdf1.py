from datetime import datetime
import numpy as np
import pandas as pd

@pytest.fixture
def dti() -> pd.DatetimeIndex:
    return pd.date_range(start=datetime(2005, 1, 1), end=datetime(2005, 1, 10), freq='Min')

@pytest.fixture
def _test_series(dti: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(np.random.default_rng(2).random(len(dti)), index=dti)

@pytest.fixture
def test_frame(_test_series: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({'A': _test_series, 'B': _test_series, 'C': np.arange(len(dti))})

def test_str(_test_series: pd.Series) -> None:
    # ...

def test_agg_mixed_column_aggregation(test_frame: pd.DataFrame) -> None:
    # ...

def test_agg_dict_of_dict_specificationerror(test_frame: pd.DataFrame) -> None:
    # ...

def test_agg_specificationerror_invalid_names(test_frame: pd.DataFrame) -> None:
    # ...

def test_agg_nested_dicts(test_frame: pd.DataFrame) -> None:
    # ...

def test_selection_api_validation() -> None:
    # ...

def test_resample_agg_readonly() -> None:
    # ...

def test_end_and_end_day_origin() -> None:
    # ...

def test_frame_downsample_method(method: str, numeric_only: bool, expected_data: dict, using_infer_string: bool) -> None:
    # ...

def test_series_downsample_method(method: str, numeric_only: bool, expected_data: dict, using_infer_string: bool) -> None:
