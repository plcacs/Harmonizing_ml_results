from collections import defaultdict
from datetime import datetime
from functools import partial
import math
import operator
import re
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_any_real_numeric_dtype, is_numeric_dtype, is_object_dtype
import pandas as pd
from pandas import CategoricalIndex, DataFrame, DatetimeIndex, IntervalIndex, PeriodIndex, RangeIndex, Series, TimedeltaIndex, date_range, period_range, timedelta_range
import pandas._testing as tm
from pandas.core.indexes.api import Index, MultiIndex, _get_combined_index, ensure_index, ensure_index_from_sequences

class TestIndex:
    @pytest.fixture
    def simple_index(self) -> Index:
        ...

    def test_can_hold_identifiers(self, simple_index: Index) -> None:
        ...

    @pytest.mark.parametrize('index', ['datetime'], indirect=True)
    def test_new_axis(self, index: DatetimeIndex) -> None:
        ...

    def test_constructor_regular(self, index: Index) -> None:
        ...

    @pytest.mark.parametrize('index', ['string'], indirect=True)
    def test_constructor_casting(self, index: Index) -> None:
        ...

    def test_constructor_copy(self, using_infer_string: bool) -> None:
        ...

    @pytest.mark.parametrize('cast_as_obj', [True, False])
    @pytest.mark.parametrize('index', [date_range('2015-01-01 10:00', freq='D', periods=3, tz='US/Eastern', name='Green Eggs & Ham'), date_range('2015-01-01 10:00', freq='D', periods=3), timedelta_range('1 days', freq='D', periods=3), period_range('2015-01-01', freq='D', periods=3)])
    def test_constructor_from_index_dtlike(self, cast_as_obj: bool, index: Index) -> None:
        ...

    @pytest.mark.parametrize('index,has_tz', [(date_range('2015-01-01 10:00', freq='D', periods=3, tz='US/Eastern'), True), (timedelta_range('1 days', freq='D', periods=3), False), (period_range('2015-01-01', freq='D', periods=3), False)])
    def test_constructor_from_series_dtlike(self, index: Index, has_tz: bool) -> None:
        ...

    def test_constructor_from_frame_series_freq(self, using_infer_string: bool) -> None:
        ...

    def test_constructor_int_dtype_nan(self) -> None:
        ...

    @pytest.mark.parametrize('klass,dtype,na_val', [(Index, np.float64, np.nan), (DatetimeIndex, 'datetime64[s]', pd.NaT)])
    def test_index_ctor_infer_nan_nat(self, klass: type[Index], dtype: str, na_val: float) -> None:
        ...

    @pytest.mark.parametrize('vals,dtype', [([1, 2, 3, 4, 5], 'int'), ([1.1, np.nan, 2.2, 3.0], 'float'), (['A', 'B', 'C', np.nan], 'obj')])
    def test_constructor_simple_new(self, vals: list, dtype: str) -> None:
        ...

    @pytest.mark.parametrize('attr', ['values', 'asi8'])
    @pytest.mark.parametrize('klass', [Index, DatetimeIndex])
    def test_constructor_dtypes_datetime(self, tz_naive_fixture: str | None, attr: str, klass: type[Index]) -> None:
        ...

    @pytest.mark.parametrize('attr', ['values', 'asi8'])
    @pytest.mark.parametrize('klass', [Index, TimedeltaIndex])
    def test_constructor_dtypes_timedelta(self, attr: str, klass: type[Index]) -> None:
        ...

    @pytest.mark.parametrize('value', [[], iter([]), (_ for _ in [])])
    @pytest.mark.parametrize('klass', [Index, CategoricalIndex, DatetimeIndex, TimedeltaIndex])
    def test_constructor_empty(self, value: list, klass: type[Index]) -> None:
        ...

    @pytest.mark.parametrize('empty,klass', [(PeriodIndex([], freq='D'), PeriodIndex), (PeriodIndex(iter([]), freq='D'), PeriodIndex), (PeriodIndex((_ for _ in []), freq='D'), PeriodIndex), (RangeIndex(step=1), RangeIndex), (MultiIndex(levels=[[1, 2], ['blue', 'red']], codes=[[], []]), MultiIndex)])
    def test_constructor_empty_special(self, empty: Index, klass: type[Index]) -> None:
        ...

    @pytest.mark.parametrize('index', ['datetime', 'float64', 'float32', 'int64', 'int32', 'period', 'range', 'repeats', 'timedelta', 'tuples', 'uint64', 'uint32'], indirect=True)
    def test_view_with_args(self, index: Index) -> None:
        ...

    @pytest.mark.parametrize('index', ['string', pytest.param('categorical', marks=pytest.mark.xfail(reason='gh-25464')), 'bool-object', 'bool-dtype', 'empty'], indirect=True)
    def test_view_with_args_object_array_raises(self, index: Index) -> None:
        ...

    @pytest.mark.parametrize('index', ['int64', 'int32', 'range'], indirect=True)
    def test_astype(self, index: Index) -> None:
        ...

    def test_equals_object(self) -> None:
        ...

    @pytest.mark.parametrize('comp', [Index(['a', 'b']), Index(['a', 'b', 'd']), ['a', 'b', 'c']])
    def test_not_equals_object(self, comp: Index | list) -> None:
        ...

    def test_identical(self) -> None:
        ...

    def test_is_(self) -> None:
        ...

    def test_asof_numeric_vs_bool_raises(self) -> None:
        ...

    @pytest.mark.parametrize('index', ['string'], indirect=True)
    def test_booleanindex(self, index: Index) -> None:
        ...

    def test_fancy(self, simple_index: Index) -> None:
        ...

    @pytest.mark.parametrize('index', ['string', 'int64', 'int32', 'uint64', 'uint32', 'float64', 'float32'], indirect=True)
    @pytest.mark.parametrize('dtype', [int, np.bool_])
    def test_empty_fancy(self, index: Index, dtype: type, request: pytest.FixtureRequest, using_infer_string: bool) -> None:
        ...

    @pytest.mark.parametrize('index', ['string', 'int64', 'int32', 'uint64', 'uint32', 'float64', 'float32'], indirect=True)
    def test_empty_fancy_raises(self, index: Index) -> None:
        ...

    def test_union_dt_as_obj(self, simple_index: Index) -> None:
        ...

    def test_map_with_tuples(self) -> None:
        ...

    def test_map_with_tuples_mi(self) -> None:
        ...

    @pytest.mark.parametrize('index', [date_range('2020-01-01', freq='D', periods=10), period_range('2020-01-01', freq='D', periods=10), timedelta_range('1 day', periods=10)])
    def test_map_tseries_indices_return_index(self, index: Index) -> None:
        ...

    def test_map_tseries_indices_accsr_return_index(self) -> None:
        ...

    @pytest.mark.parametrize('mapper', [lambda values, index: {i: e for e, i in zip(values, index)}, lambda values, index: Series(values, index)])
    def test_map_dictlike_simple(self, mapper: callable) -> None:
        ...

    @pytest.mark.parametrize('mapper', [lambda values, index: {i: e for e, i in zip(values, index)}, lambda values, index: Series(values, index)])
    @pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
    def test_map_dictlike(self, index: Index, mapper: callable, request: pytest.FixtureRequest) -> None:
        ...

    @pytest.mark.parametrize('mapper', [Series(['foo', 2.0, 'baz'], index=[0, 2, -1]), {0: 'foo', 2: 2.0, -1: 'baz'}])
    def test_map_with_non_function_missing_values(self, mapper: callable) -> None:
        ...

    def test_map_na_exclusion(self) -> None:
        ...

    def test_map_defaultdict(self) -> None:
        ...

    @pytest.mark.parametrize('name,expected', [('foo', 'foo'), ('bar', None)])
    def test_append_empty_preserve_name(self, name: str, expected: str | None) -> None:
        ...

    @pytest.mark.parametrize('index, expected', [('string', False), ('bool-object', False), ('bool-dtype', False), ('categorical', False), ('int64', True), ('int32', True), ('uint64', True), ('uint32', True), ('datetime', False), ('float64', True), ('float32', True)], indirect=['index'])
    def test_is_numeric(self, index: Index, expected: bool) -> None:
        ...

    @pytest.mark.parametrize('index, expected', [('string', True), ('bool-object', True), ('bool-dtype', False), ('categorical', False), ('int64', False), ('int32', False), ('uint64', False), ('uint32', False), ('datetime', False), ('float64', False), ('float32', False)], indirect=['index'])
    def test_is_object(self, index: Index, expected: bool, using_infer_string: bool) -> None:
        ...

    def test_summary(self, index: Index) -> None:
        ...

    def test_logical_compat(self, all_boolean_reductions: str, simple_index: Index) -> None:
        ...

    @pytest.mark.parametrize('index', ['string', 'int64', 'int32', 'float64', 'float32'], indirect=True)
    def test_drop_by_str_label(self, index: Index) -> None:
        ...

    @pytest.mark.parametrize('index', ['string', 'int64', 'int32', 'float64', 'float32'], indirect=True)
    @pytest.mark.parametrize('keys', [['foo', 'bar'], ['1', 'bar']])
    def test_drop_by_str_label_raises_missing_keys(self, index: Index, keys: list) -> None:
        ...

    @pytest.mark.parametrize('index', ['string', 'int64', 'int32', 'float64', 'float32'], indirect=True)
    def test_drop_by_str_label_errors_ignore(self, index: Index) -> None:
        ...

    def test_drop_by_numeric_label_loc(self) -> None:
        ...

    def test_drop_by_numeric_label_raises_missing_keys(self) -> None:
        ...

    @pytest.mark.parametrize('key,expected', [(4, Index([1, 2, 3])), ([3, 4, 5], Index([1, 2]))])
    def test_drop_by_numeric_label_errors_ignore(self, key: int | list, expected: Index) -> None:
        ...

    @pytest.mark.parametrize('values', [['a', 'b', ('c', 'd')], ['a', ('c', 'd'), 'b'], [('c', 'd'), 'a', 'b']])
    @pytest.mark.parametrize('to_drop', [[('c', 'd'), 'a'], ['a', ('c', 'd')]])
    def test_drop_tuple(self, values: list, to_drop: list) -> None:
        ...

    @pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
    def test_drop_with_duplicates_in_index(self, index: Index) -> None:
        ...

    @pytest.mark.parametrize('attr', ['is_monotonic_increasing', 'is_monotonic_decreasing', '_is_strictly_monotonic_increasing', '_is_strictly_monotonic_decreasing'])
    def test_is_monotonic_incomparable(self, attr: str) -> None:
        ...

    @pytest.mark.parametrize('values', [['foo', 'bar', 'quux'], {'foo', 'bar', 'quux'}])
    @pytest.mark.parametrize('index,expected', [(['qux', 'baz', 'foo', 'bar'], [False, False, True, True]), ([], [])])
    def test_isin(self, values: list | set, index: list, expected: list) -> None:
        ...

    def test_isin_nan_common_object(self, nulls_fixture: Any, nulls_fixture2: Any, using_infer_string: bool) -> None:
        ...

    def test_isin_nan_common_float64(self, nulls_fixture: Any, float_numpy_dtype: type) -> None:
        ...

    @pytest.mark.parametrize('level', [0, -1])
    @pytest.mark.parametrize('index', [['qux', 'baz', 'foo', 'bar'], np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)])
    def test_isin_level_kwarg(self, level: int, index: list | np.ndarray) -> None:
        ...

    def test_isin_level_kwarg_bad_level_raises(self, index: Index) -> None:
        ...

    @pytest.mark.parametrize('label', [1.0, 'foobar', 'xyzzy', np.nan])
    def test_isin_level_kwarg_bad_label_raises(self, label: Any, index: Index) -> None:
        ...

    @pytest.mark.parametrize('empty', [[], Series(dtype=object), np.array([])])
    def test_isin_empty(self, empty: list | Series | np.ndarray) -> None:
        ...

    def test_isin_string_null(self, string_dtype_no_object: str) -> None:
        ...

    @pytest.mark.parametrize('values', [[1, 2, 3, 4], [1.0, 2.0, 3.0, 4.0], [True, True, True, True], ['foo', 'bar', 'baz', 'qux'], date_range('2018-01-01', freq='D', periods=4)])
    def test_boolean_cmp(self, values: list) -> None:
        ...

    @pytest.mark.parametrize('index', ['string'], indirect=True)
    @pytest.mark.parametrize('name,level', [(None, 0), ('a', 'a')])
    def test_get_level_values(self, index: Index, name: str | None, level: int | str) -> None:
        ...

    def test_slice_keep_name(self) -> None:
        ...

    def test_slice_is_unique(self) -> None:
        ...

    def test_slice_is_montonic(self) -> None:
        ...

    @pytest.mark.parametrize('index', ['string', 'datetime', 'int64', 'int32', 'uint64', 'uint32', 'float64', 'float32'], indirect=True)
    def test_join_self(self, index: Index, join_type: str) -> None:
        ...

    @pytest.mark.parametrize('method', ['strip', 'rstrip', 'lstrip'])
    def test_str_attribute(self, method: str) -> None:
        ...

    @pytest.mark.parametrize('index', [Index(range(5)), date_range('2020-01-01', periods=10), MultiIndex.from_tuples([('foo', '1'), ('bar', '3')]), period_range(start='2000', end='2010', freq='Y')])
    def test_str_attribute_raises(self, index: Index) -> None:
        ...

    @pytest.mark.parametrize('expand,expected', [(None, Index([['a', 'b', 'c'], ['d', 'e'], ['f']])), (False, Index([['a', 'b', 'c'], ['d', 'e'], ['f']])), (True, MultiIndex.from_tuples([('a', 'b', 'c'), ('d', 'e', np.nan), ('f', np.nan, np.nan)]))])
    def test_str_split(self, expand: bool | None, expected: Index | MultiIndex) -> None:
        ...

    def test_str_bool_return(self) -> None:
        ...

    def test_str_bool_series_indexing(self) -> None:
        ...

    @pytest.mark.parametrize('index,expected', [(list('abcd'), True), (range(4), False)])
    def test_tab_completion(self, index: list | range, expected: bool) -> None:
        ...

    def test_indexing_doesnt_change_class(self) -> None:
        ...

    def test_outer_join_sort(self) -> None:
        ...

    def test_take_fill_value(self) -> None:
        ...

    def test_take_fill_value_none_raises(self) -> None:
        ...

    def test_take_bad_bounds_raises(self) -> None:
        ...

    @pytest.mark.parametrize('name', [None, 'foobar'])
    @pytest.mark.parametrize('labels', [[], np.array([]), ['A', 'B', 'C'], ['C', 'B', 'A'], np.array(['A', 'B', 'C']), np.array(['C', 'B', 'A']), date_range('20130101', periods=3).values, date_range('20130101', periods=3).tolist()])
    def test_reindex_preserves_name_if_target_is_list_or_ndarray(self, name: str | None, labels: list | np.ndarray) -> None:
        ...

    @pytest.mark.parametrize('labels', [[], np.array([]), np.array([], dtype=np.int64)])
    def test_reindex_preserves_type_if_target_is_empty_list_or_array(self, labels: list | np.ndarray) -> None:
        ...

    def test_reindex_doesnt_preserve_type_if_target_is_empty_index(self) -> None:
        ...

    def test_reindex_doesnt_preserve_type_if_target_is_empty_index_numeric(self, any_real_numpy_dtype: type) -> None:
        ...

    def test_reindex_no_type_preserve_target_empty_mi(self) -> None:
        ...

    def test_reindex_ignoring_level(self) -> None:
        ...

    def test_groupby(self) -> None:
        ...

    @pytest.mark.parametrize('mi,expected', [(MultiIndex.from_tuples([(1, 2), (4, 5)]), np.array([True, True])), (MultiIndex.from_tuples([(1, 2), (4, 6)]), np.array([True, False]))])
    def test_equals_op_multiindex(self, mi: MultiIndex, expected: np.ndarray) -> None:
        ...

    def test_equals_op_multiindex_identify(self) -> None:
        ...

    @pytest.mark.parametrize('index', [MultiIndex.from_tuples([(1, 2), (4, 5), (8, 9)]), Index(['foo', 'bar', 'baz'])])
    def test_equals_op_mismatched_multiindex_raises(self, index: Index) -> None:
        ...

    def test_equals_op_index_vs_mi_same_length(self, using_infer_string: bool) -> None:
        ...

    @pytest.mark.parametrize('dt_conv, arg', [(pd.to_datetime, ['2000-01-01', '2000-01-02']), (pd.to_timedelta, ['01:02:03', '01:02:04'])])
    def test_dt_conversion_preserves_name(self, dt_conv: callable, arg: list) -> None:
        ...

    def test_cached_properties_not_settable(self) -> None:
        ...

    def test_tab_complete_warning(self, ip: Any) -> None:
        ...

    def test_contains_method_removed(self, index: Index) -> None:
        ...

    def test_sortlevel(self) -> None:
        ...

    def test_sortlevel_na_position(self) -> None:
        ...

    @pytest.mark.parametrize('periods, expected_results', [(1, [np.nan, 10, 10, 10, 10]), (2, [np.nan, np.nan, 20, 20, 20]), (3, [np.nan, np.nan, np.nan, 30, 30])])
    def test_index_diff(self, periods: int, expected_results: list) -> None:
        ...

    @pytest.mark.parametrize('decimals, expected_results', [(0, [1.0, 2.0, 3.0]), (1, [1.2, 2.3, 3.5]), (2, [1.23, 2.35, 3.46])])
    def test_index_round(self, decimals: int, expected_results: list) -> None:
        ...

class TestMixedIntIndex:
    @pytest.fixture
    def simple_index(self) -> Index:
        ...

    def test_argsort(self, simple_index: Index) -> None:
        ...

    def test_numpy_argsort(self, simple_index: Index) -> None:
        ...

    def test_copy_name(self, simple_index: Index) -> None:
        ...

    def test_copy_name2(self) -> None:
        ...

    def test_unique_na(self) -> None:
        ...

    def test_logical_compat(self, simple_index: Index) -> None:
        ...

    @pytest.mark.parametrize('how', ['any', 'all'])
    @pytest.mark.parametrize('dtype', [None, object, 'category'])
    @pytest.mark.parametrize('vals,expected', [([1, 2, 3], [1, 2, 3]), ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), ([1.0, 2.0, np.nan, 3.0], [1.0, 2.0, 3.0]), (['A', 'B', 'C'], ['A', 'B', 'C']), (['A', np.nan, 'B', 'C'], ['A', 'B', 'C'])])
    def test_dropna(self, how: str, dtype: type | None, vals: list, expected: list) -> None:
        ...

    @pytest.mark.parametrize('how', ['any', 'all'])
    @pytest.mark.parametrize('index,expected', [(DatetimeIndex(['2011-01-01', '2011-01-02', '2011-01-03']), DatetimeIndex(['2011-01-01', '2011-01-02', '2011-01-03'])), (DatetimeIndex(['2011-01-01', '2011-01-02', '2011-01-03', pd.NaT]), DatetimeIndex(['2011-01-01', '2011-01-02', '2011-01-03'])), (TimedeltaIndex(['1 days', '2 days', '3 days']), TimedeltaIndex(['1 days', '2 days', '3 days'])), (TimedeltaIndex([pd.NaT, '1 days', '2 days', '3 days', pd.NaT]), TimedeltaIndex(['1 days', '2 days', '3 days'])), (PeriodIndex(['2012-02', '2012-04', '2012-05'], freq='M'), PeriodIndex(['2012-02', '2012-04', '2012-05'], freq='M')), (PeriodIndex(['2012-02', '2012-04', 'NaT', '2012-05'], freq='M'), PeriodIndex(['2012-02', '2012-04', '2012-05'], freq='M'))])
    def test_dropna_dt_like(self, how: str, index: Index, expected: Index) -> None:
        ...

    def test_dropna_invalid_how_raises(self) -> None:
        ...

    @pytest.mark.parametrize('index', [Index([np.nan]), Index([np.nan, 1]), Index([1, 2, np.nan]), Index(['a', 'b', np.nan]), pd.to_datetime(['NaT']), pd.to_datetime(['NaT', '2000-01-01']), pd.to_datetime(['2000-01-01', 'NaT', '2000-01-02']), pd.to_timedelta(['1 day', 'NaT'])])
    def test_is_monotonic_na(self, index: Index) -> None:
        ...

    @pytest.mark.parametrize('dtype', ['f8', 'm8[ns]', 'M8[us]'])
    @pytest.mark.parametrize('unique_first', [True, False])
    def test_is_monotonic_unique_na(self, dtype: str, unique_first: bool) -> None:
        ...

    def test_int_name_format(self, frame_or_series: type) -> None:
        ...

    def test_str_to_bytes_raises(self) -> None:
        ...

    @pytest.mark.filterwarnings('ignore:elementwise comparison failed:FutureWarning')
    def test_index_with_tuple_bool(self) -> None:
        ...

class TestIndexUtils:
    @pytest.mark.parametrize('data, names, expected', [([[1, 2, 4]], None, Index([1, 2, 4])), ([[1, 2, 4]], ['name'], Index([1, 2, 4], name='name')), ([[1, 2, 3]], None, RangeIndex(1, 4)), ([[1, 2, 3]], ['name'], RangeIndex(1, 4, name='name')), ([['a', 'a'], ['c', 'd']], None, MultiIndex([['a'], ['c', 'd']], [[0, 0], [0, 1]])), ([['a', 'a'], ['c', 'd']], ['L1', 'L2'], MultiIndex([['a'], ['c', 'd']], [[0, 0], [0, 1]], names=['L1', 'L2']))])
    def test_ensure_index_from_sequences(self, data: list, names: list | None, expected: Index) -> None:
        ...

    def test_ensure_index_mixed_closed_intervals(self) -> None:
        ...

    def test_ensure_index_uint64(self) -> None:
        ...

    def test_get_combined_index(self) -> None:
        ...

@pytest.mark.parametrize('opname', ['eq', 'ne', 'le', 'lt', 'ge', 'gt', 'add', 'radd', 'sub', 'rsub', 'mul', 'rmul', 'truediv', 'rtruediv', 'floordiv', 'rfloordiv', 'pow', 'rpow', 'mod', 'divmod'])
def test_generated_op_names(opname: str, index: Index) -> None:
    ...

@pytest.mark.parametrize('klass', [partial(CategoricalIndex, data=[1]), partial(DatetimeIndex, data=['2020-01-01']), partial(PeriodIndex, data=['2020-01-01']), partial(TimedeltaIndex, data=['1 day']), partial(RangeIndex, start=range(1)), partial(IntervalIndex, data=[pd.Interval(0, 1)]), partial(Index, data=['a'], dtype=object), partial(MultiIndex, levels=[1], codes=[0])])
def test_index_subclass_constructor_wrong_kwargs(klass: type[Index]) -> None:
    ...

def test_deprecated_fastpath() -> None:
    ...

def test_shape_of_invalid_index() -> None:
    ...

@pytest.mark.parametrize('dtype', [None, np.int64, np.uint64, np.float64])
def test_validate_1d_input(dtype: type | None) -> None:
    ...

@pytest.mark.parametrize('klass, extra_kwargs', [[Index, {}], *[[lambda x: Index(x, dtype=dtyp), {}] for dtyp in tm.ALL_REAL_NUMPY_DTYPES], [DatetimeIndex, {}], [TimedeltaIndex, {}], [PeriodIndex, {'freq': 'Y'}]])
def test_construct_from_memoryview(klass: type[Index], extra_kwargs: dict) -> None:
    ...

@pytest.mark.parametrize('op', [operator.lt, operator.gt])
def test_nan_comparison_same_object(op: callable) -> None:
    ...

@td.skip_if_no('pyarrow')
def test_is_monotonic_pyarrow_list_type() -> None:
    ...