```pyi
from typing import Any, Callable, Tuple, Type, Union
import numpy as np
import pytest
from pandas.compat import HAS_PYARROW
from pandas.compat.numpy import np_version_gt2
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
from pandas import CategoricalIndex, Series, Timedelta, Timestamp, date_range
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray, IntervalArray, NumpyExtensionArray, PeriodArray, SparseArray, TimedeltaArray
from pandas.core.arrays.string_ import StringArrayNumpySemantics
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics

class TestToIterable:
    dtypes: list[Tuple[str, Type[Any]]]
    
    @pytest.mark.parametrize('dtype, rdtype', dtypes)
    @pytest.mark.parametrize('method', [lambda x: x.tolist(), lambda x: x.to_list(), lambda x: list(x), lambda x: list(x.__iter__())], ids=['tolist', 'to_list', 'list', 'iter'])
    def test_iterable(self, index_or_series: Any, method: Callable[[Any], Any], dtype: str, rdtype: Type[Any]) -> None: ...
    
    @pytest.mark.parametrize('dtype, rdtype, obj', [('object', object, 'a'), ('object', int, 1), ('category', object, 'a'), ('category', int, 1)])
    @pytest.mark.parametrize('method', [lambda x: x.tolist(), lambda x: x.to_list(), lambda x: list(x), lambda x: list(x.__iter__())], ids=['tolist', 'to_list', 'list', 'iter'])
    def test_iterable_object_and_category(self, index_or_series: Any, method: Callable[[Any], Any], dtype: str, rdtype: Type[Any], obj: Any) -> None: ...
    
    @pytest.mark.parametrize('dtype, rdtype', dtypes)
    def test_iterable_items(self, dtype: str, rdtype: Type[Any]) -> None: ...
    
    @pytest.mark.parametrize('dtype, rdtype', dtypes + [('object', int), ('category', int)])
    def test_iterable_map(self, index_or_series: Any, dtype: str, rdtype: Type[Any]) -> None: ...
    
    @pytest.mark.parametrize('method', [lambda x: x.tolist(), lambda x: x.to_list(), lambda x: list(x), lambda x: list(x.__iter__())], ids=['tolist', 'to_list', 'list', 'iter'])
    def test_categorial_datetimelike(self, method: Callable[[Any], Any]) -> None: ...
    
    def test_iter_box_dt64(self, unit: str) -> None: ...
    
    def test_iter_box_dt64tz(self, unit: str) -> None: ...
    
    def test_iter_box_timedelta64(self, unit: str) -> None: ...
    
    def test_iter_box_period(self) -> None: ...

@pytest.mark.parametrize('arr, expected_type, dtype', [(np.array([0, 1], dtype=np.int64), np.ndarray, 'int64'), (np.array(['a', 'b']), np.ndarray, 'object'), (pd.Categorical(['a', 'b']), pd.Categorical, 'category'), (pd.DatetimeIndex(['2017', '2018'], tz='US/Central'), DatetimeArray, 'datetime64[ns, US/Central]'), (pd.PeriodIndex([2018, 2019], freq='Y'), PeriodArray, pd.core.dtypes.dtypes.PeriodDtype('Y-DEC')), (pd.IntervalIndex.from_breaks([0, 1, 2]), IntervalArray, 'interval'), (pd.DatetimeIndex(['2017', '2018']), DatetimeArray, 'datetime64[ns]'), (pd.TimedeltaIndex([10 ** 10]), TimedeltaArray, 'm8[ns]')])
def test_values_consistent(arr: Any, expected_type: Type[Any], dtype: Any, using_infer_string: Any) -> None: ...

@pytest.mark.parametrize('arr', [np.array([1, 2, 3])])
def test_numpy_array(arr: np.ndarray) -> None: ...

def test_numpy_array_all_dtypes(any_numpy_dtype: Any) -> None: ...

@pytest.mark.parametrize('arr, attr', [(pd.Categorical(['a', 'b']), '_codes'), (PeriodArray._from_sequence(['2000', '2001'], dtype='period[D]'), '_ndarray'), (pd.array([0, np.nan], dtype='Int64'), '_data'), (IntervalArray.from_breaks([0, 1]), '_left'), (SparseArray([0, 1]), '_sparse_values'), (DatetimeArray._from_sequence(np.array([1, 2], dtype='datetime64[ns]')), '_ndarray'), (DatetimeArray._from_sequence(np.array(['2000-01-01T12:00:00', '2000-01-02T12:00:00'], dtype='M8[ns]'), dtype=DatetimeTZDtype(tz='US/Central')), '_ndarray')])
def test_array(arr: Any, attr: str, index_or_series: Any) -> None: ...

def test_array_multiindex_raises() -> None: ...

@pytest.mark.parametrize('arr, expected, zero_copy', [(np.array([1, 2], dtype=np.int64), np.array([1, 2], dtype=np.int64), True), (pd.Categorical(['a', 'b']), np.array(['a', 'b'], dtype=object), False), (pd.core.arrays.period_array(['2000', '2001'], freq='D'), np.array([pd.Period('2000', freq='D'), pd.Period('2001', freq='D')]), False), (pd.array([0, np.nan], dtype='Int64'), np.array([0, np.nan]), False), (IntervalArray.from_breaks([0, 1, 2]), np.array([pd.Interval(0, 1), pd.Interval(1, 2)], dtype=object), False), (SparseArray([0, 1]), np.array([0, 1], dtype=np.int64), False), (DatetimeArray._from_sequence(np.array(['2000', '2001'], dtype='M8[ns]')), np.array(['2000', '2001'], dtype='M8[ns]'), True), (DatetimeArray._from_sequence(np.array(['2000-01-01T06:00:00', '2000-01-02T06:00:00'], dtype='M8[ns]')).tz_localize('UTC').tz_convert('US/Central'), np.array([Timestamp('2000-01-01', tz='US/Central'), Timestamp('2000-01-02', tz='US/Central')]), False), (TimedeltaArray._from_sequence(np.array([0, 3600000000000], dtype='i8').view('m8[ns]'), dtype=np.dtype('m8[ns]')), np.array([0, 3600000000000], dtype='m8[ns]'), True), (pd.Categorical(date_range('2016-01-01', periods=2, tz='US/Pacific')), np.array([Timestamp('2016-01-01', tz='US/Pacific'), Timestamp('2016-01-02', tz='US/Pacific')]), False)])
def test_to_numpy(arr: Any, expected: np.ndarray, zero_copy: bool, index_or_series_or_array: Any) -> None: ...

@pytest.mark.parametrize('as_series', [True, False])
@pytest.mark.parametrize('arr', [np.array([1, 2, 3], dtype='int64'), np.array(['a', 'b', 'c'], dtype=object)])
def test_to_numpy_copy(arr: np.ndarray, as_series: bool, using_infer_string: Any) -> None: ...

@pytest.mark.parametrize('as_series', [True, False])
def test_to_numpy_dtype(as_series: bool) -> None: ...

@pytest.mark.parametrize('values, dtype, na_value, expected', [([1, 2, None], 'float64', 0, [1.0, 2.0, 0.0]), ([Timestamp('2000'), Timestamp('2000'), pd.NaT], None, Timestamp('2000'), [np.datetime64('2000-01-01T00:00:00', 's')] * 3)])
def test_to_numpy_na_value_numpy_dtype(index_or_series: Any, values: Any, dtype: Any, na_value: Any, expected: Any) -> None: ...

@pytest.mark.parametrize('data, multiindex, dtype, na_value, expected', [([1, 2, None, 4], [(0, 'a'), (0, 'b'), (1, 'b'), (1, 'c')], float, None, [1.0, 2.0, np.nan, 4.0]), ([1, 2, None, 4], [(0, 'a'), (0, 'b'), (1, 'b'), (1, 'c')], float, np.nan, [1.0, 2.0, np.nan, 4.0]), ([1.0, 2.0, np.nan, 4.0], [('a', 0), ('a', 1), ('a', 2), ('b', 0)], int, 0, [1, 2, 0, 4]), ([Timestamp('2000'), Timestamp('2000'), pd.NaT], [(0, Timestamp('2021')), (0, Timestamp('2022')), (1, Timestamp('2000'))], None, Timestamp('2000'), [np.datetime64('2000-01-01T00:00:00', 's')] * 3)])
def test_to_numpy_multiindex_series_na_value(data: Any, multiindex: Any, dtype: Any, na_value: Any, expected: Any) -> None: ...

def test_to_numpy_kwargs_raises() -> None: ...

@pytest.mark.parametrize('data', [{'a': [1, 2, 3], 'b': [1, 2, None]}, {'a': np.array([1, 2, 3]), 'b': np.array([1, 2, np.nan])}, {'a': pd.array([1, 2, 3]), 'b': pd.array([1, 2, None])}])
@pytest.mark.parametrize('dtype, na_value', [(float, np.nan), (object, None)])
def test_to_numpy_dataframe_na_value(data: Any, dtype: Any, na_value: Any) -> None: ...

@pytest.mark.parametrize('data, expected_data', [({'a': pd.array([1, 2, None])}, [[1.0], [2.0], [np.nan]]), ({'a': [1, 2, 3], 'b': [1, 2, 3]}, [[1, 1], [2, 2], [3, 3]])])
def test_to_numpy_dataframe_single_block(data: Any, expected_data: Any) -> None: ...

def test_to_numpy_dataframe_single_block_no_mutate() -> None: ...

class TestAsArray:
    @pytest.mark.parametrize('tz', [None, 'US/Central'])
    def test_asarray_object_dt64(self, tz: Union[None, str]) -> None: ...
    
    def test_asarray_tz_naive(self) -> None: ...
    
    def test_asarray_tz_aware(self) -> None: ...
```