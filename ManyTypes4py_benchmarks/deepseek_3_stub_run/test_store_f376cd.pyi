import contextlib
import datetime as dt
import pathlib
import tempfile
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    Series,
    Timestamp,
)
from pandas.io.pytables import HDFStore

pytestmark: List[Any] = ...

tables: Any = ...


def test_context(setup_path: Union[str, pathlib.Path]) -> None: ...


def test_no_track_times(
    tmp_path: pathlib.Path, setup_path: Union[str, pathlib.Path]
) -> None: ...


def test_iter_empty(setup_path: Union[str, pathlib.Path]) -> None: ...


def test_repr(
    setup_path: Union[str, pathlib.Path],
    performance_warning: Any,
    using_infer_string: bool,
) -> None: ...


def test_contains(setup_path: Union[str, pathlib.Path]) -> None: ...


def test_versioning(setup_path: Union[str, pathlib.Path]) -> None: ...


@pytest.mark.parametrize("where, expected", ...)
def test_walk(
    where: str,
    expected: Dict[
        str, Tuple[Set[str], Set[str]]
    ],
) -> None: ...


def test_getattr(setup_path: Union[str, pathlib.Path]) -> None: ...


def test_store_dropna(
    tmp_path: pathlib.Path, setup_path: Union[str, pathlib.Path]
) -> None: ...


def test_to_hdf_with_min_itemsize(
    tmp_path: pathlib.Path, setup_path: Union[str, pathlib.Path]
) -> None: ...


@pytest.mark.xfail(...)
@pytest.mark.parametrize("format", ...)
def test_to_hdf_errors(
    tmp_path: pathlib.Path,
    format: str,
    setup_path: Union[str, pathlib.Path],
) -> None: ...


def test_create_table_index(setup_path: Union[str, pathlib.Path]) -> None: ...


def test_create_table_index_data_columns_argument(
    setup_path: Union[str, pathlib.Path],
) -> None: ...


def test_mi_data_columns(setup_path: Union[str, pathlib.Path]) -> None: ...


def test_table_mixed_dtypes(setup_path: Union[str, pathlib.Path]) -> None: ...


def test_calendar_roundtrip_issue(setup_path: Union[str, pathlib.Path]) -> None: ...


def test_remove(setup_path: Union[str, pathlib.Path]) -> None: ...


def test_same_name_scoping(setup_path: Union[str, pathlib.Path]) -> None: ...


def test_store_index_name(setup_path: Union[str, pathlib.Path]) -> None: ...


@pytest.mark.parametrize("tz", ...)
@pytest.mark.parametrize("table_format", ...)
def test_store_index_name_numpy_str(
    tmp_path: pathlib.Path,
    table_format: str,
    setup_path: Union[str, pathlib.Path],
    unit: str,
    tz: Optional[str],
) -> None: ...


def test_store_series_name(setup_path: Union[str, pathlib.Path]) -> None: ...


def test_overwrite_node(setup_path: Union[str, pathlib.Path]) -> None: ...


def test_coordinates(setup_path: Union[str, pathlib.Path]) -> None: ...


def test_start_stop_table(setup_path: Union[str, pathlib.Path]) -> None: ...


def test_start_stop_multiple(setup_path: Union[str, pathlib.Path]) -> None: ...


def test_start_stop_fixed(setup_path: Union[str, pathlib.Path]) -> None: ...


def test_select_filter_corner(
    setup_path: Union[str, pathlib.Path], request: Any
) -> None: ...


def test_path_pathlib() -> None: ...


@pytest.mark.parametrize("start, stop", ...)
def test_contiguous_mixed_data_table(
    start: Optional[int],
    stop: Optional[int],
    setup_path: Union[str, pathlib.Path],
) -> None: ...


def test_path_pathlib_hdfstore() -> None: ...


def test_pickle_path_localpath() -> None: ...


@pytest.mark.parametrize("propindexes", ...)
def test_copy(propindexes: bool) -> None: ...


def test_duplicate_column_name(
    tmp_path: pathlib.Path, setup_path: Union[str, pathlib.Path]
) -> None: ...


def test_preserve_timedeltaindex_type(setup_path: Union[str, pathlib.Path]) -> None: ...


def test_columns_multiindex_modified(
    tmp_path: pathlib.Path, setup_path: Union[str, pathlib.Path]
) -> None: ...


@pytest.mark.filterwarnings(...)
@pytest.mark.parametrize("columns", ...)
def test_to_hdf_with_object_column_names_should_fail(
    tmp_path: pathlib.Path,
    setup_path: Union[str, pathlib.Path],
    columns: Index,
) -> None: ...


@pytest.mark.parametrize("dtype", ...)
def test_to_hdf_with_object_column_names_should_run(
    tmp_path: pathlib.Path,
    setup_path: Union[str, pathlib.Path],
    dtype: Optional[str],
) -> None: ...


def test_hdfstore_strides(setup_path: Union[str, pathlib.Path]) -> None: ...


def test_store_bool_index(
    tmp_path: pathlib.Path, setup_path: Union[str, pathlib.Path]
) -> None: ...