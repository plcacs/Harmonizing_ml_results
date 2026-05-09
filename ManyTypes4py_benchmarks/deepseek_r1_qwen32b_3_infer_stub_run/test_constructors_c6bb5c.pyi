import pytest
from collections import abc
from datetime import date, datetime, timedelta
from pandas import (
    Categorical,
    CategoricalIndex,
    DataFrame,
    DatetimeIndex,
    Index,
    Interval,
    MultiIndex,
    Period,
    RangeIndex,
    Series,
    Timedelta,
    Timestamp,
    date_range,
    isna,
)
from pandas._testing import tm
from pandas.arrays import (
    DatetimeArray,
    IntervalArray,
    PeriodArray,
    SparseArray,
    TimedeltaArray,
)
from numpy import ma
from numpy import ndarray

class TestDataFrameConstructors:
    def test_constructor_from_ndarray_with_str_dtype(self) -> None:
        ...

    def test_constructor_from_2d_datetimearray(self) -> None:
        ...

    def test_constructor_dict_with_tzaware_scalar(self, dt: Timestamp) -> None:
        ...

    def test_construct_ndarray_with_nas_and_int_dtype(self) -> None:
        ...

    def test_construct_from_list_of_datetimes(self) -> None:
        ...

    def test_constructor_from_tzaware_datetimeindex(self) -> None:
        ...

    def test_columns_with_leading_underscore_work_with_to_dict(self) -> None:
        ...

    def test_columns_with_leading_number_and_underscore_work_with_to_dict(self) -> None:
        ...

    def test_array_of_dt64_nat_with_td64dtype_raises(self, frame_or_series: Union[DataFrame, Series]) -> None:
        ...

    @pytest.mark.parametrize('kind', ['m', 'M'])
    def test_datetimelike_values_with_object_dtype(self, kind: str, frame_or_series: Union[DataFrame, Series]) -> None:
        ...

    def test_series_with_name_not_matching_column(self) -> None:
        ...

    @pytest.mark.parametrize('constructor', [lambda: DataFrame(), lambda: DataFrame(None), lambda: DataFrame(()), lambda: DataFrame([]), lambda: DataFrame((_ for _ in [])), lambda: DataFrame(range(0)), lambda: DataFrame(data=None), lambda: DataFrame(data=()), lambda: DataFrame(data=[]), lambda: DataFrame(data=(_ for _ in [])), lambda: DataFrame(data=range(0))])
    def test_empty_constructor(self, constructor) -> None:
        ...

    def test_empty_constructor_object_index(self) -> None:
        ...

    @pytest.mark.parametrize('emptylike,expected_index,expected_columns', [([[]], RangeIndex(1), RangeIndex(0)), ([[], []], RangeIndex(2), RangeIndex(0)), ([(_ for _ in [])], RangeIndex(1), RangeIndex(0))])
    def test_emptylike_constructor(self, emptylike: list, expected_index: RangeIndex, expected_columns: RangeIndex) -> None:
        ...

    def test_constructor_mixed(self, float_string_frame: DataFrame, using_infer_string: bool) -> None:
        ...

    def test_constructor_cast_failure(self) -> None:
        ...

    def test_constructor_dtype_copy(self) -> None:
        ...

    def test_constructor_dtype_nocast_view_dataframe(self) -> None:
        ...

    def test_constructor_dtype_nocast_view_2d_array(self) -> None:
        ...

    def test_1d_object_array_does_not_copy(self, using_infer_string: bool) -> None:
        ...

    def test_2d_object_array_does_not_copy(self, using_infer_string: bool) -> None:
        ...

    def test_constructor_dtype_list_data(self) -> None:
        ...

    def test_constructor_list_of_2d_raises(self) -> None:
        ...

    @pytest.mark.parametrize('typ, ad', [['float', {}], ['float', {'A': 1, 'B': 'foo', 'C': 'bar'}], ['int', {}]])
    def test_constructor_mixed_dtypes(self, typ: str, ad: dict) -> None:
        ...

    def test_constructor_complex_dtypes(self) -> None:
        ...

    def test_constructor_dtype_str_na_values(self, string_dtype: str) -> None:
        ...

    def test_constructor_rec(self, float_frame: DataFrame) -> None:
        ...

    def test_constructor_bool(self) -> None:
        ...

    def test_constructor_overflow_int64(self) -> None:
        ...

    @pytest.mark.parametrize('values', [np.array([2 ** 64 - i for i in range(1, 10)], dtype=np.uint64), np.array([2 ** 65]), [2 ** 64 + 1], np.array([-2 ** 63 - 4], dtype=np.uint64), np.array([-2 ** 64 - 1]), [-2 ** 65 - 2]])
    def test_constructor_int_overflow(self, values: ndarray) -> None:
        ...

    @pytest.mark.parametrize('values', [np.array([1], dtype=np.uint16), np.array([1], dtype=np.uint32), np.array([1], dtype=np.uint64), [np.uint16(1)], [np.uint32(1)], [np.uint64(1)])
    def test_constructor_numpy_uints(self, values: ndarray) -> None:
        ...

    def test_constructor_ordereddict(self) -> None:
        ...

    def test_constructor_dict(self) -> None:
        ...

    def test_constructor_dict_length1(self) -> None:
        ...

    def test_constructor_dict_with_index(self) -> None:
        ...

    def test_constructor_dict_with_index_and_columns(self) -> None:
        ...

    def test_constructor_dict_of_empty_lists(self) -> None:
        ...

    def test_constructor_dict_with_none(self) -> None:
        ...

    def test_constructor_dict_errors(self) -> None:
        ...

    @pytest.mark.parametrize('scalar', [2, np.nan, None, 'D'])
    def test_constructor_invalid_items_unused(self, scalar: Union[int, float, None, str]) -> None:
        ...

    @pytest.mark.parametrize('value', [4, np.nan, None, float('nan')])
    def test_constructor_dict_nan_key(self, value: Union[int, float, None, str]) -> None:
        ...

    @pytest.mark.parametrize('value', [np.nan, None, float('nan')])
    def test_constructor_dict_nan_tuple_key(self, value: Union[float, None, str]) -> None:
        ...

    def test_constructor_dict_order_insertion(self) -> None:
        ...

    def test_constructor_dict_nan_key_and_columns(self) -> None:
        ...

    def test_constructor_multi_index(self) -> None:
        ...

    def test_constructor_2d_index(self) -> None:
        ...

    def test_constructor_error_msgs(self) -> None:
        ...

    def test_constructor_subclass_dict(self, dict_subclass: Type[dict]) -> None:
        ...

    def test_constructor_defaultdict(self, float_frame: DataFrame) -> None:
        ...

    def test_constructor_dict_block(self) -> None:
        ...

    def test_constructor_dict_cast(self, using_infer_string: bool) -> None:
        ...

    def test_constructor_dict_cast2(self) -> None:
        ...

    def test_constructor_dict_dont_upcast(self) -> None:
        ...

    def test_constructor_dict_dont_upcast2(self) -> None:
        ...

    def test_constructor_dict_of_tuples(self) -> None:
        ...

    def test_constructor_dict_of_ranges(self) -> None:
        ...

    def test_constructor_dict_of_iterators(self) -> None:
        ...

    def test_constructor_dict_of_generators(self) -> None:
        ...

    def test_constructor_dict_multiindex(self) -> None:
        ...

    def test_constructor_dict_datetime64_index(self) -> None:
        ...

    @pytest.mark.parametrize('klass,name', [(lambda x: np.timedelta64(x, 'D'), 'timedelta64'), (lambda x: timedelta(days=x), 'pytimedelta'), (lambda x: Timedelta(x, 'D'), 'Timedelta[ns]'), (lambda x: Timedelta(x, 'D').as_unit('s'), 'Timedelta[s]')])
    def test_constructor_dict_timedelta64_index(self, klass: Callable, name: str) -> None:
        ...

    def test_constructor_period_dict(self) -> None:
        ...

    def test_constructor_extension_scalar_data(self, data: Union[Period, Interval, Timestamp], dtype: Union[PeriodDtype, IntervalDtype, DatetimeTZDtype]) -> None:
        ...

    def test_nested_dict_frame_constructor(self) -> None:
        ...

    def _check_basic_constructor(self, empty: Callable) -> None:
        ...

    def test_constructor_ndarray(self) -> None:
        ...

    def test_constructor_maskedarray(self) -> None:
        ...

    @pytest.mark.filterwarnings('ignore:elementwise comparison failed:DeprecationWarning')
    def test_constructor_maskedarray_nonfloat(self) -> None:
        ...

    def test_constructor_maskedarray_hardened(self) -> None:
        ...

    def test_constructor_maskedrecarray_dtype(self) -> None:
        ...

    def test_constructor_corner_shape(self) -> None:
        ...

    @pytest.mark.parametrize('data, input_dtype, expected_dtype', [(None, list(range(10)), ['a', 'b'], object, np.object_), (None, None, ['a', 'b'], 'int64', np.dtype('int64')), (None, list(range(10)), ['a', 'b'], int, np.dtype('float64')), ({}, None, ['foo', 'bar'], None, np.object_), ({'b': 1}, list(range(10)), list('abc'), int, np.dtype('float64'))])
    def test_constructor_dtype(self, data: Any, input_dtype: Any, expected_dtype: Any) -> None:
        ...

    @pytest.mark.parametrize('data,input_dtype,expected_dtype', (([True, False, None], 'boolean', pd.BooleanDtype), ([1.0, 2.0, None], 'Float64', pd.Float64Dtype), ([1, 2, None], 'Int64', pd.Int64Dtype), (['a', 'b', 'c'], 'string', pd.StringDtype)))
    def test_constructor_dtype_nullable_extension_arrays(self, data: Any, input_dtype: str, expected_dtype: Any) -> None:
        ...

    def test_constructor_scalar_inference(self, using_infer_string: bool) -> None:
        ...

    def test_constructor_arrays_and_scalars(self) -> None:
        ...

    def test_constructor_DataFrame(self, float_frame: DataFrame) -> None:
        ...

    def test_constructor_empty_dataframe(self) -> None:
        ...

    def test_constructor_more(self, float_frame: DataFrame) -> None:
        ...

    def test_constructor_empty_list(self) -> None:
        ...

    def test_constructor_list_of_lists(self, using_infer_string: bool) -> None:
        ...

    def test_nested_pandasarray_matches_nested_ndarray(self) -> None