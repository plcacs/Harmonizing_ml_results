from __future__ import annotations
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    Type,
    TYPE_CHECKING,
)
from datetime import datetime as datetime_, timedelta as timedelta_
from pandas import (
    CategoricalIndex,
    Index,
    Series,
    Timedelta,
    Timestamp,
    date_range,
    Period,
    DatetimeTZDtype,
    PeriodDtype,
)
from pandas.core.arrays import (
    DatetimeArray,
    IntervalArray,
    NumpyExtensionArray,
    PeriodArray,
    SparseArray,
    TimedeltaArray,
)
from pandas.core.arrays.string_ import StringArrayNumpySemantics
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics

class TestToIterable:
    dtypes: List[Tuple[str, type]] = ...

    def test_iterable(
        self,
        index_or_series: Type[Union[Series, Index]],
        method: Callable,
        dtype: str,
        rdtype: type,
    ) -> None:
        ...

    def test_iterable_object_and_category(
        self,
        index_or_series: Type[Union[Series, Index]],
        method: Callable,
        dtype: str,
        rdtype: type,
        obj: Any,
    ) -> None:
        ...

    def test_iterable_items(self, dtype: str, rdtype: type) -> None:
        ...

    def test_iterable_map(
        self,
        index_or_series: Type[Union[Series, Index]],
        dtype: str,
        rdtype: Union[type, Tuple[type, ...]],
    ) -> None:
        ...

    def test_categorial_datetimelike(self, method: Callable) -> None:
        ...

    def test_iter_box_dt64(self, unit: str) -> None:
        ...

    def test_iter_box_dt64tz(self, unit: str) -> None:
        ...

    def test_iter_box_timedelta64(self, unit: str) -> None:
        ...

    def test_iter_box_period(self) -> None:
        ...

def test_values_consistent(
    arr: Any,
    expected_type: type,
    dtype: str,
    using_infer_string: bool,
) -> None:
    ...

def test_numpy_array(arr: np.ndarray) -> None:
    ...

def test_numpy_array_all_dtypes(any_numpy_dtype: str) -> None:
    ...

def test_array(
    arr: Any,
    attr: Optional[str],
    index_or_series: Type[Union[Series, Index]],
) -> None:
    ...

def test_array_multiindex_raises() -> None:
    ...

def test_to_numpy(
    arr: Any,
    expected: np.ndarray,
    zero_copy: bool,
    index_or_series_or_array: Type[Union[Series, Index, np.ndarray]],
) -> None:
    ...

def test_to_numpy_copy(
    arr: np.ndarray,
    as_series: bool,
    using_infer_string: bool,
) -> None:
    ...

def test_to_numpy_dtype(as_series: bool) -> None:
    ...

def test_to_numpy_na_value_numpy_dtype(
    index_or_series: Type[Union[Series, Index]],
    values: List[Any],
    dtype: Optional[str],
    na_value: Any,
    expected: List[Any],
) -> None:
    ...

def test_to_numpy_multiindex_series_na_value(
    data: List[Any],
    multiindex: List[Tuple[Any, ...]],
    dtype: Optional[str],
    na_value: Any,
    expected: List[Any],
) -> None:
    ...

def test_to_numpy_kwargs_raises() -> None:
    ...

def test_to_numpy_dataframe_na_value(
    data: Dict[str, List[Any]],
    dtype: str,
    na_value: Any,
) -> None:
    ...

def test_to_numpy_dataframe_single_block(
    data: Dict[str, List[Any]],
    expected_data: List[List[Any]],
) -> None:
    ...

def test_to_numpy_dataframe_single_block_no_mutate() -> None:
    ...

class TestAsArray:
    def test_asarray_object_dt64(self, tz: Optional[str]) -> None:
        ...

    def test_asarray_tz_naive(self) -> None:
        ...

    def test_asarray_tz_aware(self) -> None:
        ...