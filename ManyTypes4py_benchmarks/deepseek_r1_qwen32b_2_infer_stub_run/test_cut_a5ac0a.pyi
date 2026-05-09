from datetime import datetime, timedelta
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
    overload,
)
import numpy as np
import pandas as pd
from pandas import (
    Categorical,
    CategoricalDtype,
    DataFrame,
    DatetimeIndex,
    Interval,
    IntervalIndex,
    Series,
    TimedeltaIndex,
    Timestamp,
)
import pytest

def test_simple() -> None:
    data: np.ndarray[np.int64]
    result: Categorical
    ...

def test_bins(func: Callable[[Union[List[float], np.ndarray[float]]], Union[List[float], np.ndarray[float]]]) -> None:
    data: Union[List[float], np.ndarray[float]]
    result: Categorical
    bins: np.ndarray[float]
    ...

def test_right() -> None:
    data: np.ndarray[float]
    result: Categorical
    bins: np.ndarray[float]
    ...

def test_no_right() -> None:
    data: np.ndarray[float]
    result: Categorical
    bins: np.ndarray[float]
    ...

def test_bins_from_interval_index() -> None:
    c: Categorical
    expected: Categorical
    result: Categorical
    ...

def test_bins_from_interval_index_doc_example() -> None:
    ages: np.ndarray[int]
    c: Categorical
    expected: IntervalIndex
    result: Categorical
    ...

def test_bins_not_overlapping_from_interval_index() -> None:
    msg: str
    ...

def test_bins_not_monotonic() -> None:
    msg: str
    ...

def test_bins_monotonic_not_overflowing(x: Union[np.ndarray[int], List[datetime], List[timedelta]], bins: Union[List[datetime], List[timedelta], np.ndarray[int]], expected: IntervalIndex) -> None:
    ...

def test_wrong_num_labels() -> None:
    msg: str
    ...

def test_cut_corner(x: Union[List[int], np.ndarray[int]], bins: Union[int, float], msg: str) -> None:
    ...

def test_cut_not_1d_arg(arg: Union[int, np.ndarray[float], DataFrame], cut_func: Callable[[Union[int, np.ndarray[float], DataFrame], int], Any]) -> None:
    msg: str
    ...

def test_int_bins_with_inf(data: List[Union[int, float]]) -> None:
    msg: str
    ...

def test_cut_out_of_range_more() -> None:
    name: str
    ser: Series
    ind: Series
    exp: Series
    ...

def test_labels(right: bool, breaks: List[float], closed: str) -> None:
    arr: np.ndarray[float]
    result: Categorical
    bins: np.ndarray[float]
    ...

def test_cut_pass_series_name_to_factor() -> None:
    name: str
    ser: Series
    factor: Categorical
    ...

def test_label_precision() -> None:
    arr: np.ndarray[float]
    result: Categorical
    ...

def test_na_handling(labels: Union[None, bool]) -> None:
    arr: np.ndarray[float]
    result: Categorical
    ...

def test_inf_handling() -> None:
    data: np.ndarray[int]
    data_ser: Series
    bins: List[Union[int, float]]
    result: Categorical
    result_ser: Categorical
    ...

def test_cut_out_of_bounds() -> None:
    arr: np.ndarray[float]
    result: Categorical
    mask: np.ndarray[bool]
    ex_mask: np.ndarray[bool]
    ...

def test_cut_pass_labels(get_labels: Callable[[List[str]], Union[List[str], Categorical]], get_expected: Callable[[List[str]], Categorical]) -> None:
    bins: List[int]
    arr: List[int]
    labels: List[str]
    result: Categorical
    ...

def test_cut_pass_labels_compat() -> None:
    arr: List[int]
    labels: List[str]
    result: Categorical
    exp: Categorical
    ...

def test_round_frac_just_works(x: np.ndarray[float]) -> None:
    ...

def test_round_frac(val: float, precision: int, expected: Union[int, float]) -> None:
    result: Union[int, float]
    ...

def test_cut_return_intervals() -> None:
    ser: Series
    result: Series
    exp_bins: np.ndarray[float]
    expected: Series
    ...

def test_series_ret_bins() -> None:
    ser: Series
    result: Series
    bins: np.ndarray[float]
    expected: Series
    ...

def test_cut_duplicates_bin(kwargs: Dict[str, Any], msg: Optional[str]) -> None:
    bins: List[int]
    values: Series
    ...

def test_single_bin(data: float, length: int) -> None:
    ser: Series
    result: Series
    expected: Series
    ...

def test_cut_read_only(array_1_writeable: bool, array_2_writeable: bool) -> None:
    array_1: np.ndarray[int]
    array_2: np.ndarray[int]
    hundred_elements: np.ndarray[int]
    ...

def test_datetime_bin(conv: Callable[[str], Union[Timestamp, datetime]]) -> None:
    data: List[np.datetime64]
    bin_data: List[str]
    expected: Series
    bins: List[Union[Timestamp, datetime]]
    result: Series
    ...

def test_datetime_cut(unit: str, box: Callable[[List[np.datetime64]], Union[List[np.datetime64], np.ndarray[np.datetime64], DatetimeIndex]]) -> None:
    data: Union[List[np.datetime64], np.ndarray[np.datetime64]]
    result: Series
    ...

def test_datetime_tz_cut_mismatched_tzawareness(box: Callable[[List[Timestamp]], Union[List[Timestamp], np.ndarray[Timestamp], DatetimeIndex]]) -> None:
    bins: Union[List[Timestamp], np.ndarray[Timestamp]]
    ser: Series
    msg: str
    ...

def test_datetime_tz_cut(bins: Union[int, List[Timestamp]], box: Callable[[List[Timestamp]], Union[List[Timestamp], np.ndarray[Timestamp], DatetimeIndex]]) -> None:
    tz: str
    ser: Series
    result: Series
    ii: IntervalIndex
    ...

def test_datetime_nan_error() -> None:
    msg: str
    ...

def test_datetime_nan_mask() -> None:
    result: Categorical
    mask: np.ndarray[bool]
    ...

def test_datetime_cut_roundtrip(tz: Optional[str], unit: str) -> None:
    ser: Series
    result: Categorical
    result_bins: DatetimeIndex
    expected: Categorical
    ...

def test_timedelta_cut_roundtrip() -> None:
    ser: Series
    result: Categorical
    result_bins: TimedeltaIndex
    ...

def test_cut_bool_coercion_to_int(bins: int, box: Callable[[List[Union[bool, int]]], Union[List[Union[bool, int]], np.ndarray[Union[bool, int]]]], compare: Callable[[Categorical, Categorical], None]) -> None:
    data_expected: Union[List[int], np.ndarray[int]]
    data_result: Union[List[bool], np.ndarray[bool]]
    expected: Categorical
    result: Categorical
    ...

def test_cut_incorrect_labels(labels: Union[str, int, bool]) -> None:
    values: List[int]
    msg: str
    ...

def test_cut_nullable_integer(bins: Union[int, List[int]], right: bool, include_lowest: bool) -> None:
    a: pd.array
    result: Categorical
    expected: Categorical
    ...

def test_cut_non_unique_labels(data: List[int], bins: List[int], labels: List[str], expected_codes: List[int], expected_labels: List[str]) -> None:
    result: Categorical
    expected: Categorical
    ...

def test_cut_unordered_labels(data: List[int], bins: List[int], labels: List[str], expected_codes: List[int], expected_labels: List[str]) -> None:
    result: Categorical
    expected: Categorical
    ...

def test_cut_unordered_with_missing_labels_raises_error() -> None:
    msg: str
    ...

def test_cut_unordered_with_series_labels() -> None:
    ser: Series
    bins: Series
    labels: Series
    result: Series
    ...

def test_cut_no_warnings() -> None:
    df: DataFrame
    labels: List[str]
    ...

def test_cut_with_duplicated_index_lowest_included() -> None:
    expected: Series
    ser: Series
    result: Categorical
    ...

def test_cut_with_nonexact_categorical_indices() -> None:
    ser: Series
    ser1: Series
    ser2: Series
    result: DataFrame
    index: CategoricalIndex
    expected: DataFrame
    ...

def test_cut_with_timestamp_tuple_labels() -> None:
    labels: List[Tuple[Timestamp]]
    result: Categorical
    expected: Categorical
    ...

def test_cut_bins_datetime_intervalindex() -> None:
    bins: IntervalIndex
    result: Categorical
    expected: Categorical
    ...

def test_cut_with_nullable_int64() -> None:
    series: Series
    bins: List[int]
    intervals: IntervalIndex
    expected: Series
    result: Categorical
    ...

def test_cut_datetime_array_no_attributeerror() -> None:
    ser: Series
    result: Categorical
    categories: IntervalIndex
    expected: Categorical
    ...