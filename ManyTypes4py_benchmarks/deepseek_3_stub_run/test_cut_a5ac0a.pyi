import datetime
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from pandas import Categorical, DataFrame, DatetimeIndex, Index, Interval, IntervalIndex, Series, TimedeltaIndex, Timestamp
from pandas.api.types import CategoricalDtype
from pandas.core.dtypes.common import is_list_like

def test_simple() -> None: ...

@pytest.mark.parametrize('func', [list, np.array])
def test_bins(func: Callable[[List[float]], Union[List[float], np.ndarray]]) -> None: ...

def test_right() -> None: ...

def test_no_right() -> None: ...

def test_bins_from_interval_index() -> None: ...

def test_bins_from_interval_index_doc_example() -> None: ...

def test_bins_not_overlapping_from_interval_index() -> None: ...

def test_bins_not_monotonic() -> None: ...

@pytest.mark.parametrize('x,bins,expected', [
    (pd.DatetimeIndex, List[Timestamp], IntervalIndex),
    (List[int], np.ndarray, IntervalIndex),
    (List[np.timedelta64], np.ndarray, IntervalIndex)
])
def test_bins_monotonic_not_overflowing(
    x: Union[pd.DatetimeIndex, List[int], List[np.timedelta64]],
    bins: Union[List[Timestamp], np.ndarray],
    expected: IntervalIndex
) -> None: ...

def test_wrong_num_labels() -> None: ...

@pytest.mark.parametrize('x,bins,msg', [
    (List[float], int, str),
    (List[int], float, str)
])
def test_cut_corner(x: Union[List[float], List[int]], bins: Union[int, float], msg: str) -> None: ...

@pytest.mark.parametrize('arg', [int, np.ndarray, DataFrame])
@pytest.mark.parametrize('cut_func', [Callable[..., Any], Callable[..., Any]])
def test_cut_not_1d_arg(arg: Union[int, np.ndarray, DataFrame], cut_func: Callable[..., Any]) -> None: ...

@pytest.mark.parametrize('data', [
    List[Union[float, np.float64]],
    List[Union[float, np.float64]],
    List[Union[float, np.float64]]
])
def test_int_bins_with_inf(data: List[Union[float, np.float64]]) -> None: ...

def test_cut_out_of_range_more() -> None: ...

@pytest.mark.parametrize('right,breaks,closed', [
    (bool, List[float], str),
    (bool, List[float], str)
])
def test_labels(right: bool, breaks: List[float], closed: str) -> None: ...

def test_cut_pass_series_name_to_factor() -> None: ...

def test_label_precision() -> None: ...

@pytest.mark.parametrize('labels', [None, bool])
def test_na_handling(labels: Optional[bool]) -> None: ...

def test_inf_handling() -> None: ...

def test_cut_out_of_bounds() -> None: ...

@pytest.mark.parametrize('get_labels,get_expected', [
    (Callable[[List[str]], List[str]], Callable[[List[str]], Categorical]),
    (Callable[[List[str]], Categorical], Callable[[List[str]], Categorical])
])
def test_cut_pass_labels(get_labels: Callable[[List[str]], Union[List[str], Categorical]], get_expected: Callable[[List[str]], Categorical]) -> None: ...

def test_cut_pass_labels_compat() -> None: ...

@pytest.mark.parametrize('x', [np.ndarray, np.ndarray])
def test_round_frac_just_works(x: np.ndarray) -> None: ...

@pytest.mark.parametrize('val,precision,expected', [
    (float, int, float),
    (float, int, float),
    (float, int, float),
    (float, int, float)
])
def test_round_frac(val: float, precision: int, expected: float) -> None: ...

def test_cut_return_intervals() -> None: ...

def test_series_ret_bins() -> None: ...

@pytest.mark.parametrize('kwargs,msg', [
    (dict, Optional[str]),
    (dict, Optional[str]),
    (dict, Optional[str]),
    (dict, Optional[str])
])
def test_cut_duplicates_bin(kwargs: dict, msg: Optional[str]) -> None: ...

@pytest.mark.parametrize('data', [float, float, float])
@pytest.mark.parametrize('length', [int, int])
def test_single_bin(data: float, length: int) -> None: ...

@pytest.mark.parametrize('array_1_writeable,array_2_writeable', [
    (bool, bool),
    (bool, bool),
    (bool, bool)
])
def test_cut_read_only(array_1_writeable: bool, array_2_writeable: bool) -> None: ...

@pytest.mark.parametrize('conv', [
    Callable[[str], Timestamp],
    Callable[[str], pd.Timestamp],
    Callable[[str], np.datetime64],
    Callable[[str], datetime.datetime]
])
def test_datetime_bin(conv: Callable[[str], Union[Timestamp, pd.Timestamp, np.datetime64, datetime.datetime]]) -> None: ...

@pytest.mark.parametrize('box', [Series, Index, np.ndarray, list])
def test_datetime_cut(unit: str, box: Callable[[Any], Any]) -> None: ...

@pytest.mark.parametrize('box', [list, np.ndarray, Index, Series])
def test_datetime_tz_cut_mismatched_tzawareness(box: Callable[[List[Timestamp]], Any]) -> None: ...

@pytest.mark.parametrize('bins', [
    int,
    List[Timestamp]
])
@pytest.mark.parametrize('box', [list, np.ndarray, Index, Series])
def test_datetime_tz_cut(bins: Union[int, List[Timestamp]], box: Callable[[Any], Any]) -> None: ...

def test_datetime_nan_error() -> None: ...

def test_datetime_nan_mask() -> None: ...

@pytest.mark.parametrize('tz', [None, str, str])
def test_datetime_cut_roundtrip(tz: Optional[str], unit: str) -> None: ...

def test_timedelta_cut_roundtrip() -> None: ...

@pytest.mark.parametrize('bins', [int, int])
@pytest.mark.parametrize('box,compare', [
    (Series, Callable[[Any, Any], None]),
    (np.ndarray, Callable[[Any, Any], None]),
    (list, Callable[[Any, Any], None])
])
def test_cut_bool_coercion_to_int(bins: int, box: Callable[[Any], Any], compare: Callable[[Any, Any], None]) -> None: ...

@pytest.mark.parametrize('labels', [str, int, bool])
def test_cut_incorrect_labels(labels: Union[str, int, bool]) -> None: ...

@pytest.mark.parametrize('bins', [int, List[int]])
@pytest.mark.parametrize('right', [bool, bool])
@pytest.mark.parametrize('include_lowest', [bool, bool])
def test_cut_nullable_integer(bins: Union[int, List[int]], right: bool, include_lowest: bool) -> None: ...

@pytest.mark.parametrize('data,bins,labels,expected_codes,expected_labels', [
    (List[int], List[int], List[str], List[int], List[str]),
    (List[int], List[int], List[int], List[int], List[int])
])
def test_cut_non_unique_labels(
    data: List[int],
    bins: List[int],
    labels: Union[List[str], List[int]],
    expected_codes: List[int],
    expected_labels: Union[List[str], List[int]]
) -> None: ...

@pytest.mark.parametrize('data,bins,labels,expected_codes,expected_labels', [
    (List[int], List[int], List[str], List[int], List[str]),
    (List[int], List[int], List[int], List[int], List[int])
])
def test_cut_unordered_labels(
    data: List[int],
    bins: List[int],
    labels: Union[List[str], List[int]],
    expected_codes: List[int],
    expected_labels: Union[List[str], List[int]]
) -> None: ...

def test_cut_unordered_with_missing_labels_raises_error() -> None: ...

def test_cut_unordered_with_series_labels() -> None: ...

def test_cut_no_warnings() -> None: ...

def test_cut_with_duplicated_index_lowest_included() -> None: ...

@pytest.mark.filterwarnings('ignore:invalid value encountered in cast:RuntimeWarning')
def test_cut_with_nonexact_categorical_indices() -> None: ...

def test_cut_with_timestamp_tuple_labels() -> None: ...

def test_cut_bins_datetime_intervalindex() -> None: ...

def test_cut_with_nullable_int64() -> None: ...

def test_cut_datetime_array_no_attributeerror() -> None: ...