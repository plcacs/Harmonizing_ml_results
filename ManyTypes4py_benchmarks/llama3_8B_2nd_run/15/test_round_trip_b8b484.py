import datetime
import re
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
from pandas.compat import is_platform_windows
import pandas as pd
from pandas import DataFrame, DatetimeIndex, Index, Series, _testing as tm, bdate_range, date_range, read_hdf
from pandas.tests.io.pytables.common import _maybe_remove, ensure_clean_store
from pandas.util import _test_decorators as td

def test_conv_read_write() -> None:
    with tm.ensure_clean() as path:
        ...

def test_long_strings(setup_path: str) -> None:
    ...

def test_api(tmp_path: str, setup_path: str) -> None:
    ...

def test_api_append(tmp_path: str, setup_path: str) -> None:
    ...

def test_api_2(tmp_path: str, setup_path: str) -> None:
    ...

def test_get(setup_path: str) -> None:
    ...

def test_put_integer(setup_path: str) -> None:
    ...

def test_table_values_dtypes_roundtrip(setup_path: str, using_infer_string: bool) -> None:
    ...

def test_series(setup_path: str) -> None:
    ...

def test_float_index(setup_path: str) -> None:
    ...

def test_tuple_index(setup_path: str, performance_warning: str) -> None:
    ...

def test_index_types(setup_path: str) -> None:
    ...

def test_timeseries_preepoch(setup_path: str, request) -> None:
    ...

def test_frame(compression: bool, setup_path: str) -> None:
    ...

def test_empty_series_frame(setup_path: str) -> None:
    ...

def test_empty_series(dtype: np.dtype, setup_path: str) -> None:
    ...

def test_can_serialize_dates(setup_path: str) -> None:
    ...

def test_store_hierarchical(setup_path: str, using_infer_string: bool, multiindex_dataframe_random_data: DataFrame) -> None:
    ...

def test_store_mixed(compression: bool, setup_path: str) -> None:
    ...

def _check_roundtrip(obj: pd.DataFrame, comparator: callable, path: str, compression: bool = False, **kwargs) -> None:
    ...

def _check_roundtrip_table(obj: pd.DataFrame, comparator: callable, path: str, compression: bool = False) -> None:
    ...

def test_unicode_index(setup_path: str) -> None:
    ...

def test_unicode_longer_encoded(setup_path: str) -> None:
    ...

def test_store_datetime_mixed(setup_path: str) -> None:
    ...

def test_round_trip_equals(tmp_path: str, setup_path: str) -> None:
    ...

def test_infer_string_columns(tmp_path: str, setup_path: str) -> None:
    ...
