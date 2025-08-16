import contextlib
import datetime as dt
import hashlib
import tempfile
import time
import numpy as np
import pytest
from pandas._config import using_string_dtype
from pandas.compat import PY312
import pandas as pd
from pandas import DataFrame, DatetimeIndex, Index, MultiIndex, Series, Timestamp, concat, date_range, period_range, timedelta_range
import pandas._testing as tm
from pandas.conftest import has_pyarrow
from pandas.tests.io.pytables.common import _maybe_remove, ensure_clean_store
from pandas.io.pytables import HDFStore, read_hdf
pytestmark = [pytest.mark.single_cpu]
tables = pytest.importorskip('tables')

def test_context(setup_path: str) -> None:
    ...

def test_no_track_times(tmp_path: str, setup_path: str) -> None:
    ...

def test_iter_empty(setup_path: str) -> None:
    ...

def test_repr(setup_path: str, performance_warning: bool, using_infer_string: bool) -> None:
    ...

def test_contains(setup_path: str) -> None:
    ...

def test_versioning(setup_path: str) -> None:
    ...

def test_walk(where: str, expected: dict) -> None:
    ...

def test_getattr(setup_path: str) -> None:
    ...

def test_store_dropna(tmp_path: str, setup_path: str) -> None:
    ...

def test_to_hdf_with_min_itemsize(tmp_path: str, setup_path: str) -> None:
    ...

def test_to_hdf_errors(tmp_path: str, format: str, setup_path: str) -> None:
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

def test_store_index_name_numpy_str(tmp_path: str, table_format: str, setup_path: str, unit: str, tz: str) -> None:
    ...

def test_copy(propindexes: bool) -> None:
    ...

def test_duplicate_column_name(tmp_path: str, setup_path: str) -> None:
    ...

def test_preserve_timedeltaindex_type(setup_path: str) -> None:
    ...

def test_columns_multiindex_modified(tmp_path: str, setup_path: str) -> None:
    ...

def test_to_hdf_with_object_column_names_should_fail(tmp_path: str, setup_path: str, columns: Index) -> None:
    ...

def test_to_hdf_with_object_column_names_should_run(tmp_path: str, setup_path: str, dtype: str) -> None:
    ...

def test_hdfstore_strides(setup_path: str) -> None:
    ...

def test_store_bool_index(tmp_path: str, setup_path: str) -> None:
    ...
