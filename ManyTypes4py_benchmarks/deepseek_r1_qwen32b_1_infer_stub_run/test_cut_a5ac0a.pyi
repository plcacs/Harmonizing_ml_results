from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
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
)
from pandas.api.types import CategoricalDtype

def test_simple() -> None:
    ...

def test_bins(func: Callable[[List[float]], Union[List[float], np.ndarray]]) -> None:
    ...

def test_right() -> None:
    ...

def test_no_right() -> None:
    ...

def test_bins_from_interval_index() -> None:
    ...

def test_bins_from_interval_index_doc_example() -> None:
    ...

def test_bins_not_overlapping_from_interval_index() -> None:
    ...

def test_bins_not_monotonic() -> None:
    ...

def test_bins_monotonic_not_overflowing(
    x: Union[List[datetime], np.ndarray, List[np.timedelta64]],
    bins: Union[List[datetime], np.ndarray],
    expected: IntervalIndex,
) -> None:
    ...

def test_wrong_num_labels() -> None:
    ...

def test_cut_corner(
    x: List[float],
    bins: Union[int, List[float]],
    msg: str,
) -> None:
    ...

def test_cut_not_1d_arg(
    arg: Union[int, np.ndarray, DataFrame],
    cut_func: Callable,
) -> None:
    ...

def test_int_bins_with_inf(data: List[float]) -> None:
    ...

def test_cut_out_of_range_more() -> None:
    ...

def test_labels(
    right: bool,
    breaks: List[float],
    closed: str,
) -> None:
    ...

def test_cut_pass_series_name_to_factor() -> None:
    ...

def test_label_precision() -> None:
    ...

def test_na_handling(labels: Optional[Union[List[str], Categorical]]) -> None:
    ...

def test_inf_handling() -> None:
    ...

def test_cut_out_of_bounds() -> None:
    ...

def test_cut_pass_labels(
    get_labels: Callable[[List[str]], Union[List[str], Categorical]],
    get_expected: Callable[[List[str]], Categorical],
) -> None:
    ...

def test_cut_pass_labels_compat() -> None:
    ...

def test_round_frac_just_works(x: np.ndarray) -> None:
    ...

def test_round_frac(
    val: float,
    precision: int,
    expected: float,
) -> None:
    ...

def test_cut_return_intervals() -> None:
    ...

def test_series_ret_bins() -> None:
    ...

def test_cut_duplicates_bin(
    kwargs: Dict[str, Any],
    msg: Optional[str],
) -> None:
    ...

def test_single_bin(
    data: float,
    length: int,
) -> None:
    ...

def test_cut_read_only(
    array_1_writeable: bool,
    array_2_writeable: bool,
) -> None:
    ...

def test_datetime_bin(conv: Callable[[str], datetime]) -> None:
    ...

def test_datetime_cut(unit: str, box: Callable) -> None:
    ...

def test_datetime_tz_cut_mismatched_tzawareness(box: Callable) -> None:
    ...

def test_datetime_tz_cut(
    bins: Union[int, List[datetime]],
    box: Callable,
) -> None:
    ...

def test_datetime_nan_error() -> None:
    ...

def test_datetime_nan_mask() -> None:
    ...

def test_datetime_cut_roundtrip(tz: Optional[str], unit: str) -> None:
    ...

def test_timedelta_cut_roundtrip() -> None:
    ...

def test_cut_bool_coercion_to_int(
    bins: int,
    box: Callable,
    compare: Callable,
) -> None:
    ...

def test_cut_incorrect_labels(labels: Union[str, int, bool]) -> None:
    ...

def test_cut_nullable_integer(
    bins: Union[int, List[float]],
    right: bool,
    include_lowest: bool,
) -> None:
    ...

def test_cut_non_unique_labels(
    data: List[float],
    bins: List[float],
    labels: List[str],
    expected_codes: List[int],
    expected_labels: List[str],
) -> None:
    ...

def test_cut_unordered_labels(
    data: List[float],
    bins: List[float],
    labels: List[str],
    expected_codes: List[int],
    expected_labels: List[str],
) -> None:
    ...

def test_cut_unordered_with_missing_labels_raises_error() -> None:
    ...

def test_cut_unordered_with_series_labels() -> None:
    ...

def test_cut_no_warnings() -> None:
    ...

def test_cut_with_duplicated_index_lowest_included() -> None:
    ...

def test_cut_with_nonexact_categorical_indices() -> None:
    ...

def test_cut_with_timestamp_tuple_labels() -> None:
    ...

def test_cut_bins_datetime_intervalindex() -> None:
    ...

def test_cut_with_nullable_int64() -> None:
    ...

def test_cut_datetime_array_no_attributeerror() -> None:
    ...