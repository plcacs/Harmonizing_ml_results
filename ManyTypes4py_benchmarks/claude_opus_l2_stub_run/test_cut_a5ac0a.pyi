from datetime import datetime
from typing import Any

import numpy as np
import pytest

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    DatetimeIndex,
    Index,
    Interval,
    IntervalIndex,
    Series,
    TimedeltaIndex,
    Timestamp,
    cut,
    date_range,
    interval_range,
    isna,
    qcut,
    timedelta_range,
    to_datetime,
)
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
import pandas.core.reshape.tile as tmod

def test_simple() -> None: ...

@pytest.mark.parametrize("func", [list, np.array])
def test_bins(func: Any) -> None: ...

def test_right() -> None: ...

def test_no_right() -> None: ...

def test_bins_from_interval_index() -> None: ...

def test_bins_from_interval_index_doc_example() -> None: ...

def test_bins_not_overlapping_from_interval_index() -> None: ...

def test_bins_not_monotonic() -> None: ...

@pytest.mark.parametrize("x, bins, expected", [(..., ..., ...)])
def test_bins_monotonic_not_overflowing(x: Any, bins: Any, expected: IntervalIndex) -> None: ...

def test_wrong_num_labels() -> None: ...

@pytest.mark.parametrize("x,bins,msg", [(..., ..., ...)])
def test_cut_corner(x: list[Any], bins: Any, msg: str) -> None: ...

@pytest.mark.parametrize("arg", [2, np.eye(2), DataFrame(np.eye(2))])
@pytest.mark.parametrize("cut_func", [cut, qcut])
def test_cut_not_1d_arg(arg: Any, cut_func: Any) -> None: ...

@pytest.mark.parametrize("data", [...])
def test_int_bins_with_inf(data: list[float]) -> None: ...

def test_cut_out_of_range_more() -> None: ...

@pytest.mark.parametrize("right,breaks,closed", [...])
def test_labels(right: bool, breaks: list[float], closed: str) -> None: ...

def test_cut_pass_series_name_to_factor() -> None: ...

def test_label_precision() -> None: ...

@pytest.mark.parametrize("labels", [None, False])
def test_na_handling(labels: bool | None) -> None: ...

def test_inf_handling() -> None: ...

def test_cut_out_of_bounds() -> None: ...

@pytest.mark.parametrize("get_labels,get_expected", [...])
def test_cut_pass_labels(get_labels: Any, get_expected: Any) -> None: ...

def test_cut_pass_labels_compat() -> None: ...

@pytest.mark.parametrize("x", [np.arange(11.0), np.arange(11.0) / 10000000000.0])
def test_round_frac_just_works(x: np.ndarray) -> None: ...

@pytest.mark.parametrize("val,precision,expected", [...])
def test_round_frac(val: float, precision: int, expected: float) -> None: ...

def test_cut_return_intervals() -> None: ...

def test_series_ret_bins() -> None: ...

@pytest.mark.parametrize("kwargs,msg", [...])
def test_cut_duplicates_bin(kwargs: dict[str, Any], msg: str | None) -> None: ...

@pytest.mark.parametrize("data", [9.0, -9.0, 0.0])
@pytest.mark.parametrize("length", [1, 2])
def test_single_bin(data: float, length: int) -> None: ...

@pytest.mark.parametrize("array_1_writeable,array_2_writeable", [...])
def test_cut_read_only(array_1_writeable: bool, array_2_writeable: bool) -> None: ...

@pytest.mark.parametrize("conv", [...])
def test_datetime_bin(conv: Any) -> None: ...

@pytest.mark.parametrize("box", [Series, Index, np.array, list])
def test_datetime_cut(unit: str, box: Any) -> None: ...

@pytest.mark.parametrize("box", [list, np.array, Index, Series])
def test_datetime_tz_cut_mismatched_tzawareness(box: Any) -> None: ...

@pytest.mark.parametrize("bins", [...])
@pytest.mark.parametrize("box", [list, np.array, Index, Series])
def test_datetime_tz_cut(bins: Any, box: Any) -> None: ...

def test_datetime_nan_error() -> None: ...

def test_datetime_nan_mask() -> None: ...

@pytest.mark.parametrize("tz", [None, "UTC", "US/Pacific"])
def test_datetime_cut_roundtrip(tz: str | None, unit: str) -> None: ...

def test_timedelta_cut_roundtrip() -> None: ...

@pytest.mark.parametrize("bins", [6, 7])
@pytest.mark.parametrize("box, compare", [...])
def test_cut_bool_coercion_to_int(bins: int, box: Any, compare: Any) -> None: ...

@pytest.mark.parametrize("labels", ["foo", 1, True])
def test_cut_incorrect_labels(labels: Any) -> None: ...

@pytest.mark.parametrize("bins", [3, [0, 5, 15]])
@pytest.mark.parametrize("right", [True, False])
@pytest.mark.parametrize("include_lowest", [True, False])
def test_cut_nullable_integer(bins: Any, right: bool, include_lowest: bool) -> None: ...

@pytest.mark.parametrize("data, bins, labels, expected_codes, expected_labels", [...])
def test_cut_non_unique_labels(
    data: list[int],
    bins: list[int],
    labels: list[Any],
    expected_codes: list[int],
    expected_labels: list[Any],
) -> None: ...

@pytest.mark.parametrize("data, bins, labels, expected_codes, expected_labels", [...])
def test_cut_unordered_labels(
    data: list[int],
    bins: list[int],
    labels: list[Any],
    expected_codes: list[int],
    expected_labels: list[Any],
) -> None: ...

def test_cut_unordered_with_missing_labels_raises_error() -> None: ...

def test_cut_unordered_with_series_labels() -> None: ...

def test_cut_no_warnings() -> None: ...

def test_cut_with_duplicated_index_lowest_included() -> None: ...

@pytest.mark.filterwarnings("ignore:invalid value encountered in cast:RuntimeWarning")
def test_cut_with_nonexact_categorical_indices() -> None: ...

def test_cut_with_timestamp_tuple_labels() -> None: ...

def test_cut_bins_datetime_intervalindex() -> None: ...

def test_cut_with_nullable_int64() -> None: ...

def test_cut_datetime_array_no_attributeerror() -> None: ...