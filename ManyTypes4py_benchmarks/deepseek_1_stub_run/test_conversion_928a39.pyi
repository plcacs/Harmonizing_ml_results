```python
import numpy as np
import pandas as pd
from pandas import CategoricalIndex, Series, Timedelta, Timestamp
from pandas.core.arrays import DatetimeArray, IntervalArray, NumpyExtensionArray, PeriodArray, SparseArray, TimedeltaArray
from pandas.core.arrays.string_ import StringArrayNumpySemantics
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
from typing import Any, Callable, List, Tuple, Union

dtypes: List[Tuple[str, Any]] = ...
arr: Any = ...
expected_type: Any = ...
dtype: Any = ...
using_infer_string: bool = ...

def test_values_consistent(
    arr: Any,
    expected_type: Any,
    dtype: Any,
    using_infer_string: bool
) -> None: ...

def test_numpy_array(arr: Any) -> None: ...

def test_numpy_array_all_dtypes(any_numpy_dtype: Any) -> None: ...

def test_array(arr: Any, attr: Any, index_or_series: Any) -> None: ...

def test_array_multiindex_raises() -> None: ...

def test_to_numpy(
    arr: Any,
    expected: Any,
    zero_copy: bool,
    index_or_series_or_array: Any
) -> None: ...

def test_to_numpy_copy(
    arr: Any,
    as_series: bool,
    using_infer_string: bool
) -> None: ...

def test_to_numpy_dtype(as_series: bool) -> None: ...

def test_to_numpy_na_value_numpy_dtype(
    index_or_series: Any,
    values: Any,
    dtype: Any,
    na_value: Any,
    expected: Any
) -> None: ...

def test_to_numpy_multiindex_series_na_value(
    data: Any,
    multiindex: Any,
    dtype: Any,
    na_value: Any,
    expected: Any
) -> None: ...

def test_to_numpy_kwargs_raises() -> None: ...

def test_to_numpy_dataframe_na_value(
    data: Any,
    dtype: Any,
    na_value: Any
) -> None: ...

def test_to_numpy_dataframe_single_block(
    data: Any,
    expected_data: Any
) -> None: ...

def test_to_numpy_dataframe_single_block_no_mutate() -> None: ...

class TestToIterable:
    dtypes: List[Tuple[str, Any]] = ...

    def test_iterable(
        self,
        index_or_series: Any,
        method: Callable[[Any], Any],
        dtype: str,
        rdtype: Any
    ) -> None: ...

    def test_iterable_object_and_category(
        self,
        index_or_series: Any,
        method: Callable[[Any], Any],
        dtype: str,
        rdtype: Any,
        obj: Any
    ) -> None: ...

    def test_iterable_items(self, dtype: str, rdtype: Any) -> None: ...

    def test_iterable_map(
        self,
        index_or_series: Any,
        dtype: str,
        rdtype: Any
    ) -> None: ...

    def test_categorial_datetimelike(
        self,
        method: Callable[[Any], Any]
    ) -> None: ...

    def test_iter_box_dt64(self, unit: Any) -> None: ...

    def test_iter_box_dt64tz(self, unit: Any) -> None: ...

    def test_iter_box_timedelta64(self, unit: Any) -> None: ...

    def test_iter_box_period(self) -> None: ...

class TestAsArray:
    def test_asarray_object_dt64(self, tz: Any) -> None: ...

    def test_asarray_tz_naive(self) -> None: ...

    def test_asarray_tz_aware(self) -> None: ...
```