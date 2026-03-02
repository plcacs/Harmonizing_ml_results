from __future__ import annotations
import operator
from typing import TYPE_CHECKING, Literal, NoReturn, cast
import numpy as np
from pandas._libs import lib
from pandas._libs.missing import is_matching_na
from pandas._libs.sparse import SparseIndex
import pandas._libs.testing as _testing
from pandas._libs.tslibs.np_datetime import compare_mismatched_resolutions
from pandas.core.dtypes.common import is_bool, is_float_dtype, is_integer_dtype, is_number, is_numeric_dtype, needs_i8_conversion
from pandas.core.dtypes.dtypes import CategoricalDtype, DatetimeTZDtype, ExtensionDtype, NumpyEADtype
from pandas.core.dtypes.missing import array_equivalent
import pandas as pd
from pandas import Categorical, DataFrame, DatetimeIndex, Index, IntervalDtype, IntervalIndex, MultiIndex, PeriodIndex, RangeIndex, Series, TimedeltaIndex
from pandas.core.arrays import DatetimeArray, ExtensionArray, IntervalArray, PeriodArray, TimedeltaArray
from pandas.core.arrays.datetimelike import DatetimeLikeArrayMixin
from pandas.core.arrays.string_ import StringDtype
from pandas.core.indexes.api import safe_sort_index
from pandas.io.formats.printing import pprint_thing

if TYPE_CHECKING:
    from pandas._typing import DtypeObj

def assert_almost_equal(
    left: object,
    right: object,
    check_dtype: Literal['equiv', bool] = 'equiv',
    rtol: float = 1e-05,
    atol: float = 1e-08,
    **kwargs: object
) -> None:
    ...

def _check_isinstance(left: object, right: object, cls: type) -> None:
    ...

def assert_dict_equal(left: dict, right: dict, compare_keys: bool = True) -> None:
    ...

def assert_index_equal(
    left: Index,
    right: Index,
    exact: Literal['equiv', bool] = 'equiv',
    check_names: bool = True,
    check_exact: bool = True,
    check_categorical: bool = True,
    check_order: bool = True,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    obj: str = 'Index',
) -> None:
    ...

def assert_class_equal(left: object, right: object, exact: bool = True, obj: str = 'Input') -> None:
    ...

def assert_attr_equal(
    attr: str,
    left: object,
    right: object,
    obj: str = 'Attributes',
) -> None:
    ...

def assert_is_sorted(seq: object) -> None:
    ...

def assert_categorical_equal(
    left: Categorical,
    right: Categorical,
    check_dtype: bool = True,
    check_category_order: bool = True,
    obj: str = 'Categorical',
) -> None:
    ...

def assert_interval_array_equal(
    left: IntervalArray,
    right: IntervalArray,
    exact: Literal['equiv', bool] = 'equiv',
    obj: str = 'IntervalArray',
) -> None:
    ...

def assert_period_array_equal(left: PeriodArray, right: PeriodArray, obj: str = 'PeriodArray') -> None:
    ...

def assert_datetime_array_equal(
    left: DatetimeArray,
    right: DatetimeArray,
    obj: str = 'DatetimeArray',
    check_freq: bool = True,
) -> None:
    ...

def assert_timedelta_array_equal(
    left: TimedeltaArray,
    right: TimedeltaArray,
    obj: str = 'TimedeltaArray',
    check_freq: bool = True,
) -> None:
    ...

def raise_assert_detail(
    obj: str,
    message: str,
    left: object,
    right: object,
    diff: object = None,
    first_diff: object = None,
    index_values: Index | np.ndarray = None,
) -> None:
    ...

def assert_numpy_array_equal(
    left: np.ndarray | object,
    right: np.ndarray | object,
    strict_nan: bool = False,
    check_dtype: bool = True,
    err_msg: str = None,
    check_same: Literal['same', 'copy', None] = None,
    obj: str = 'numpy array',
    index_values: Index | np.ndarray = None,
) -> None:
    ...

def assert_extension_array_equal(
    left: ExtensionArray,
    right: ExtensionArray,
    check_dtype: bool = True,
    index_values: Index | np.ndarray = None,
    check_exact: bool = False,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    obj: str = 'ExtensionArray',
) -> None:
    ...

def assert_series_equal(
    left: Series,
    right: Series,
    check_dtype: bool = True,
    check_index_type: Literal['equiv', bool] = 'equiv',
    check_series_type: bool = True,
    check_names: bool = True,
    check_exact: bool = False,
    check_datetimelike_compat: bool = False,
    check_categorical: bool = True,
    check_category_order: bool = True,
    check_freq: bool = True,
    check_flags: bool = True,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    obj: str = 'Series',
    check_index: bool = True,
    check_like: bool = False,
) -> None:
    ...

def assert_frame_equal(
    left: DataFrame,
    right: DataFrame,
    check_dtype: bool = True,
    check_index_type: Literal['equiv', bool] = 'equiv',
    check_column_type: Literal['equiv', bool] = 'equiv',
    check_frame_type: bool = True,
    check_names: bool = True,
    by_blocks: bool = False,
    check_exact: bool = False,
    check_datetimelike_compat: bool = False,
    check_categorical: bool = True,
    check_like: bool = False,
    check_freq: bool = True,
    check_flags: bool = True,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    obj: str = 'DataFrame',
) -> None:
    ...

def assert_equal(
    left: Index | Series | DataFrame | ExtensionArray | np.ndarray,
    right: Index | Series | DataFrame | ExtensionArray | np.ndarray,
    **kwargs: object,
) -> None:
    ...

def assert_sp_array_equal(left: pd.arrays.SparseArray, right: pd.arrays.SparseArray) -> None:
    ...

def assert_contains_all(iterable: object, dic: dict) -> None:
    ...

def assert_copy(iter1: object, iter2: object, **eql_kwargs: object) -> None:
    ...

def is_extension_array_dtype_and_needs_i8_conversion(left_dtype: DtypeObj, right_dtype: DtypeObj) -> bool:
    ...

def assert_indexing_slices_equivalent(ser: Series, l_slc: object, i_slc: object) -> None:
    ...

def assert_metadata_equivalent(left: object, right: object = None) -> None:
