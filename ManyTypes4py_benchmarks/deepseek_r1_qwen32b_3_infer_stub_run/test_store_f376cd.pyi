import datetime as dt
import hashlib
import tempfile
import time
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame, DatetimeIndex, Index, MultiIndex, Series, Timestamp
from pandas.io.pytables import HDFStore
from pathlib import Path
from typing import Any, Optional, Union

@pytest.mark.single_cpu
def test_context(setup_path: str) -> None:
    ...

def test_no_track_times(tmp_path: Path, setup_path: str) -> None:
    ...

def test_iter_empty(setup_path: str) -> None:
    ...

def test_repr(setup_path: str, performance_warning: Any, using_infer_string: Any) -> None:
    ...

def test_contains(setup_path: str) -> None:
    ...

def test_versioning(setup_path: str) -> None:
    ...

def test_walk(where: str, expected: dict) -> None:
    ...

def test_getattr(setup_path: str) -> None:
    ...

def test_store_dropna(tmp_path: Path, setup_path: str) -> None:
    ...

def test_to_hdf_with_min_itemsize(tmp_path: Path, setup_path: str) -> None:
    ...

@pytest.mark.xfail
def test_to_hdf_errors(tmp_path: Path, format: str, setup_path: str) -> None:
    ...

def test_create_table_index(setup_path: str) -> None:
    ...

def test_create_table_index_data_columns_argument(setup_path: str) -> None:
    ...

def test_mi_data_columns(setup_path: str) -> None:
    ...

def test_table_mixed_dtypes(setup_path: str) -> None:
    ...

def test_calendar_roundtrip_issue(setup_path: str) -> None:
    ...

def test_remove(setup_path: str) -> None:
    ...

def test_same_name_scoping(setup_path: str) -> None:
    ...

def test_store_index_name(setup_path: str) -> None:
    ...

@pytest.mark.parametrize('tz', [None, str])
@pytest.mark.parametrize('table_format', ['table', 'fixed'])
def test_store_index_name_numpy_str(tmp_path: Path, table_format: str, setup_path: str, unit: Any, tz: Optional[str]) -> None:
    ...

def test_store_series_name(setup_path: str) -> None:
    ...

def test_overwrite_node(setup_path: str) -> None:
    ...

def test_coordinates(setup_path: str) -> None:
    ...

def test_start_stop_table(setup_path: str) -> None:
    ...

def test_start_stop_multiple(setup_path: str) -> None:
    ...

def test_start_stop_fixed(setup_path: str) -> None:
    ...

def test_select_filter_corner(setup_path: str, request: Any) -> None:
    ...

def test_path_pathlib() -> None:
    ...

@pytest.mark.parametrize('start, stop', [(0, 2), (1, 2), (None, None)])
def test_contiguous_mixed_data_table(start: Optional[int], stop: Optional[int], setup_path: str) -> None:
    ...

def test_path_pathlib_hdfstore() -> None:
    ...

def test_pickle_path_localpath() -> None:
    ...

@pytest.mark.parametrize('propindexes', [True, False])
def test_copy(propindexes: bool) -> None:
    ...

def test_duplicate_column_name(tmp_path: Path, setup_path: str) -> None:
    ...

def test_preserve_timedeltaindex_type(setup_path: str) -> None:
    ...

def test_columns_multiindex_modified(tmp_path: Path, setup_path: str) -> None:
    ...

@pytest.mark.filterwarnings
@pytest.mark.parametrize('columns', [Index([0, 1], dtype=np.int64), Index([0.0, 1.0], dtype=np.float64), date_range('2020-01-01', periods=2), timedelta_range('1 day', periods=2), period_range('2020-01-01', periods=2, freq='D')])
def test_to_hdf_with_object_column_names_should_fail(tmp_path: Path, setup_path: str, columns: Union[Index, pd.PeriodIndex, pd.TimedeltaIndex, pd.DatetimeIndex]) -> None:
    ...

@pytest.mark.parametrize('dtype', [None, 'category'])
def test_to_hdf_with_object_column_names_should_run(tmp_path: Path, setup_path: str, dtype: Optional[str]) -> None:
    ...

def test_hdfstore_strides(setup_path: str) -> None:
    ...

def test_store_bool_index(tmp_path: Path, setup_path: str) -> None:
    ...