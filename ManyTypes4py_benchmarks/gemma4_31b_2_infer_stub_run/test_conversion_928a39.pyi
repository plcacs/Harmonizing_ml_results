import numpy as np
import pandas as pd
from pandas import CategoricalIndex, Series, Timedelta, Timestamp, date_range
from pandas.core.arrays import DatetimeArray, IntervalArray, NumpyExtensionArray, PeriodArray, SparseArray, TimedeltaArray
from pandas.core.arrays.string_ import StringArrayNumpySemantics
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
from typing import Any, Callable, Union, Optional, Type, List, Tuple, overload

class TestToIterable:
    dtypes: List[Tuple[str, Type[Any]]]
    def test_iterable(
        self, 
        index_or_series: Type[Union[pd.Index, Series]], 
        method: Callable[[Any], List[Any]], 
        dtype: str, 
        rdtype: Type[Any]
    ) -> None: ...
    def test_iterable_object_and_category(
        self, 
        index_or_series: Type[Union[pd.Index, Series]], 
        method: Callable[[Any], List[Any]], 
        dtype: str, 
        rdtype: Type[Any], 
        obj: Any
    ) -> None: ...
    def test_iterable_items(self, dtype: str, rdtype: Type[Any]) -> None: ...
    def test_iterable_map(
        self, 
        index_or_series: Type[Union[pd.Index, Series]], 
        dtype: str, 
        rdtype: Union[Type[Any], Tuple[Type[Any], ...]]
    ) -> None: ...
    def test_categorial_datetimelike(self, method: Callable[[Any], List[Any]]) -> None: ...
    def test_iter_box_dt64(self, unit: str) -> None: ...
    def test_iter_box_dt64tz(self, unit: str) -> None: ...
    def test_iter_box_timedelta64(self, unit: str) -> None: ...
    def test_iter_box_period(self) -> None: ...

def test_values_consistent(
    arr: Any, 
    expected_type: Type[Any], 
    dtype: Any, 
    using_infer_string: bool
) -> None: ...

def test_numpy_array(arr: np.ndarray) -> None: ...

def test_numpy_array_all_dtypes(any_numpy_dtype: Any) -> None: ...

def test_array(
    arr: Any, 
    attr: Optional[str], 
    index_or_series: Type[Union[pd.Index, Series]]
) -> None: ...

def test_array_multiindex_raises() -> None: ...

def test_to_numpy(
    arr: Any, 
    expected: np.ndarray, 
    zero_copy: bool, 
    index_or_series_or_array: Type[Union[pd.Index, Series, Any]]
) -> None: ...

def test_to_numpy_copy(
    arr: np.ndarray, 
    as_series: bool, 
    using_infer_string: bool
) -> None: ...

def test_to_numpy_dtype(as_series: bool) -> None: ...

def test_to_numpy_na_value_numpy_dtype(
    index_or_series: Type[Union[pd.Index, Series]], 
    values: List[Any], 
    dtype: Optional[Union[str, Type[Any]]], 
    na_value: Any, 
    expected: List[Any]
) -> None: ...

def test_to_numpy_multiindex_series_na_value(
    data: List[Any], 
    multiindex: List[Tuple[Any, ...]], 
    dtype: Optional[Type[Any]], 
    na_value: Any, 
    expected: List[Any]
) -> None: ...

def test_to_numpy_kwargs_raises() -> None: ...

def test_to_numpy_dataframe_na_value(
    data: dict[str, Any], 
    dtype: Type[Any], 
    na_value: Any
) -> None: ...

def test_to_numpy_dataframe_single_block(
    data: dict[str, Any], 
    expected_data: List[List[Any]]
) -> None: ...

def test_to_numpy_dataframe_single_block_no_mutate() -> None: ...

class TestAsArray:
    def test_asarray_object_dt64(self, tz: Optional[str]) -> None: ...
    def test_asarray_tz_naive(self) -> None: ...
    def test_asarray_tz_aware(self) -> None: ...