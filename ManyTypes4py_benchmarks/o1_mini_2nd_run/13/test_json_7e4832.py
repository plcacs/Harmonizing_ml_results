import collections
import operator
import sys
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.json.array import JSONArray, JSONDtype, make_data

unhashable = pytest.mark.xfail(reason='Unhashable')


@pytest.fixture
def dtype() -> JSONDtype:
    return JSONDtype()


@pytest.fixture
def data() -> JSONArray:
    """Length-100 PeriodArray for semantics test."""
    data = make_data()
    while len(data[0]) == len(data[1]):
        data = make_data()
    return JSONArray(data)


@pytest.fixture
def data_missing() -> JSONArray:
    """Length 2 array with [NA, Valid]"""
    return JSONArray([{}, {'a': 10}])


@pytest.fixture
def data_for_sorting() -> JSONArray:
    return JSONArray([{'b': 1}, {'c': 4}, {'a': 2, 'c': 3}])


@pytest.fixture
def data_missing_for_sorting() -> JSONArray:
    return JSONArray([{'b': 1}, {}, {'a': 4}])


@pytest.fixture
def na_cmp() -> Callable[[Any, Any], bool]:
    return operator.eq


@pytest.fixture
def data_for_grouping() -> JSONArray:
    return JSONArray([
        {'b': 1},
        {'b': 1},
        {},
        {},
        {'a': 0, 'c': 2},
        {'a': 0, 'c': 2},
        {'b': 1},
        {'c': 2}
    ])


class TestJSONArray(base.ExtensionTests):

    @pytest.mark.xfail(reason='comparison method not implemented for JSONArray (GH-37867)')
    def test_contains(self, data: JSONArray) -> None:
        super().test_contains(data)

    @pytest.mark.xfail(reason='not implemented constructor from dtype')
    def test_from_dtype(self, data: JSONArray) -> None:
        super().test_from_dtype(data)

    @pytest.mark.xfail(reason='RecursionError, GH-33900')
    def test_series_constructor_no_data_with_index(self, dtype: JSONDtype, na_value: Any) -> None:
        rec_limit = sys.getrecursionlimit()
        try:
            sys.setrecursionlimit(100)
            super().test_series_constructor_no_data_with_index(dtype, na_value)
        finally:
            sys.setrecursionlimit(rec_limit)

    @pytest.mark.xfail(reason='RecursionError, GH-33900')
    def test_series_constructor_scalar_na_with_index(self, dtype: JSONDtype, na_value: Any) -> None:
        rec_limit = sys.getrecursionlimit()
        try:
            sys.setrecursionlimit(100)
            super().test_series_constructor_scalar_na_with_index(dtype, na_value)
        finally:
            sys.setrecursionlimit(rec_limit)

    @pytest.mark.xfail(reason='collection as scalar, GH-33901')
    def test_series_constructor_scalar_with_index(self, data: JSONArray, dtype: JSONDtype) -> None:
        rec_limit = sys.getrecursionlimit()
        try:
            sys.setrecursionlimit(100)
            super().test_series_constructor_scalar_with_index(data, dtype)
        finally:
            sys.setrecursionlimit(rec_limit)

    @pytest.mark.xfail(reason='Different definitions of NA')
    def test_stack(self) -> None:
        """
        The test does .astype(object).stack(). If we happen to have
        any missing values in `data`, then we'll end up with different
        rows since we consider `{}` NA, but `.astype(object)` doesn't.
        """
        super().test_stack()

    @pytest.mark.xfail(reason='dict for NA')
    def test_unstack(self, data: JSONArray, index: Any) -> None:
        return super().test_unstack(data, index)

    @pytest.mark.xfail(reason='Setting a dict as a scalar')
    def test_fillna_series(self) -> None:
        """We treat dictionaries as a mapping in fillna, not a scalar."""
        super().test_fillna_series()

    @pytest.mark.xfail(reason='Setting a dict as a scalar')
    def test_fillna_frame(self) -> None:
        """We treat dictionaries as a mapping in fillna, not a scalar."""
        super().test_fillna_frame()

    def test_fillna_with_none(self, data_missing: JSONArray) -> None:
        with pytest.raises(AssertionError):
            super().test_fillna_with_none(data_missing)

    @pytest.mark.xfail(reason='fill value is a dictionary, takes incorrect code path')
    def test_fillna_limit_frame(self, data_missing: JSONArray) -> None:
        super().test_fillna_limit_frame(data_missing)

    @pytest.mark.xfail(reason='fill value is a dictionary, takes incorrect code path')
    def test_fillna_limit_series(self, data_missing: JSONArray) -> None:
        super().test_fillna_limit_frame(data_missing)

    @pytest.mark.parametrize(
        'limit_area, input_ilocs, expected_ilocs',
        [
            ('outside', [1, 0, 0, 0, 1], [1, 0, 0, 0, 1]),
            ('outside', [1, 0, 1, 0, 1], [1, 0, 1, 0, 1]),
            ('outside', [0, 1, 1, 1, 0], [0, 1, 1, 1, 1]),
            ('outside', [0, 1, 0, 1, 0], [0, 1, 0, 1, 1]),
            ('inside', [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]),
            ('inside', [1, 0, 1, 0, 1], [1, 1, 1, 1, 1]),
            ('inside', [0, 1, 1, 1, 0], [0, 1, 1, 1, 0]),
            ('inside', [0, 1, 0, 1, 0], [0, 1, 1, 1, 0]),
        ]
    )
    def test_ffill_limit_area(
        self,
        data_missing: JSONArray,
        limit_area: str,
        input_ilocs: List[int],
        expected_ilocs: List[int]
    ) -> None:
        msg = 'JSONArray does not implement limit_area'
        with pytest.raises(NotImplementedError, match=msg):
            super().test_ffill_limit_area(data_missing, limit_area, input_ilocs, expected_ilocs)

    @unhashable
    def test_value_counts(self, all_data: JSONArray, dropna: bool) -> None:
        super().test_value_counts(all_data, dropna)

    @unhashable
    def test_value_counts_with_normalize(self, data: JSONArray) -> None:
        super().test_value_counts_with_normalize(data)

    @unhashable
    def test_sort_values_frame(self) -> None:
        super().test_sort_values_frame()

    @pytest.mark.xfail(reason='combine for JSONArray not supported')
    def test_combine_le(self, data_repeated: JSONArray) -> None:
        super().test_combine_le(data_repeated)

    @pytest.mark.xfail(
        reason='combine for JSONArray not supported - may pass depending on random data',
        strict=False,
        raises=AssertionError
    )
    def test_combine_first(self, data: JSONArray) -> None:
        super().test_combine_first(data)

    @pytest.mark.xfail(reason='broadcasting error')
    def test_where_series(self, data: JSONArray, na_value: Any) -> None:
        super().test_where_series(data, na_value)

    @pytest.mark.xfail(reason="Can't compare dicts.")
    def test_searchsorted(self, data_for_sorting: JSONArray) -> None:
        super().test_searchsorted(data_for_sorting)

    @pytest.mark.xfail(reason="Can't compare dicts.")
    def test_equals(self, data: JSONArray, na_value: Any, as_series: bool) -> None:
        super().test_equals(data, na_value, as_series)

    @pytest.mark.skip(reason='fill-value is interpreted as a dict of values')
    def test_fillna_copy_frame(self, data_missing: JSONArray) -> None:
        super().test_fillna_copy_frame(data_missing)

    @pytest.mark.xfail(reason='Fails with CoW')
    def test_equals_same_data_different_object(self, data: JSONArray) -> None:
        super().test_equals_same_data_different_object(data)

    @pytest.mark.xfail(reason='failing on np.array(self, dtype=str)')
    def test_astype_str(self) -> None:
        """This currently fails in NumPy on np.array(self, dtype=str) with

        *** ValueError: setting an array element with a sequence
        """
        super().test_astype_str()

    @unhashable
    def test_groupby_extension_transform(self) -> None:
        """
        This currently fails in Series.name.setter, since the
        name must be hashable, but the value is a dictionary.
        I think this is what we want, i.e. `.name` should be the original
        values, and not the values for factorization.
        """
        super().test_groupby_extension_transform()

    @unhashable
    def test_groupby_extension_apply(self) -> None:
        """
        This fails in Index._do_unique_check with

        >   hash(val)
        E   TypeError: unhashable type: 'UserDict' with

        I suspect that once we support Index[ExtensionArray],
        we'll be able to dispatch unique.
        """
        super().test_groupby_extension_apply()

    @unhashable
    def test_groupby_extension_agg(self) -> None:
        """
        This fails when we get to tm.assert_series_equal when left.index
        contains dictionaries, which are not hashable.
        """
        super().test_groupby_extension_agg()

    @unhashable
    def test_groupby_extension_no_sort(self) -> None:
        """
        This fails when we get to tm.assert_series_equal when left.index
        contains dictionaries, which are not hashable.
        """
        super().test_groupby_extension_no_sort()

    def test_arith_frame_with_scalar(
        self,
        data: JSONArray,
        all_arithmetic_operators: Any,
        request: pytest.FixtureRequest
    ) -> None:
        if len(data[0]) != 1:
            mark = pytest.mark.xfail(reason='raises in coercing to Series')
            request.applymarker(mark)
        super().test_arith_frame_with_scalar(data, all_arithmetic_operators)

    def test_compare_array(
        self,
        data: JSONArray,
        comparison_op: Callable[[Any, Any], bool],
        request: pytest.FixtureRequest
    ) -> None:
        if comparison_op.__name__ in ['eq', 'ne']:
            mark = pytest.mark.xfail(reason='Comparison methods not implemented')
            request.applymarker(mark)
        super().test_compare_array(data, comparison_op)

    @pytest.mark.xfail(reason='ValueError: Must have equal len keys and value')
    def test_setitem_loc_scalar_mixed(self, data: JSONArray) -> None:
        super().test_setitem_loc_scalar_mixed(data)

    @pytest.mark.xfail(reason='ValueError: Must have equal len keys and value')
    def test_setitem_loc_scalar_multiple_homogoneous(self, data: JSONArray) -> None:
        super().test_setitem_loc_scalar_multiple_homogoneous(data)

    @pytest.mark.xfail(reason='ValueError: Must have equal len keys and value')
    def test_setitem_iloc_scalar_mixed(self, data: JSONArray) -> None:
        super().test_setitem_iloc_scalar_mixed(data)

    @pytest.mark.xfail(reason='ValueError: Must have equal len keys and value')
    def test_setitem_iloc_scalar_multiple_homogoneous(self, data: JSONArray) -> None:
        super().test_setitem_iloc_scalar_multiple_homogoneous(data)

    @pytest.mark.parametrize(
        'mask',
        [
            np.array([True, True, True, False, False]),
            pd.array([True, True, True, False, False], dtype='boolean'),
            pd.array([True, True, True, pd.NA, pd.NA], dtype='boolean')
        ],
        ids=['numpy-array', 'boolean-array', 'boolean-array-na']
    )
    def test_setitem_mask(
        self,
        data: JSONArray,
        mask: Union[np.ndarray, pd.arrays.BooleanArray],
        box_in_series: bool,
        request: pytest.FixtureRequest
    ) -> None:
        if box_in_series:
            mark = pytest.mark.xfail(reason='cannot set using a list-like indexer with a different length')
            request.applymarker(mark)
        elif not isinstance(mask, np.ndarray):
            mark = pytest.mark.xfail(reason='Issues unwanted DeprecationWarning')
            request.applymarker(mark)
        super().test_setitem_mask(data, mask, box_in_series)

    def test_setitem_mask_raises(
        self,
        data: JSONArray,
        box_in_series: bool,
        request: pytest.FixtureRequest
    ) -> None:
        if not box_in_series:
            mark = pytest.mark.xfail(reason='Fails to raise')
            request.applymarker(mark)
        super().test_setitem_mask_raises(data, box_in_series)

    @pytest.mark.xfail(reason='cannot set using a list-like indexer with a different length')
    @pytest.mark.parametrize('setter', ['loc', None])
    def test_setitem_mask_broadcast(
        self,
        data: JSONArray,
        setter: Optional[str]
    ) -> None:
        super().test_setitem_mask_broadcast(data, setter)

    @pytest.mark.xfail(reason='cannot set using a slice indexer with a different length')
    def test_setitem_slice(
        self,
        data: JSONArray,
        box_in_series: bool
    ) -> None:
        super().test_setitem_slice(data, box_in_series)

    @pytest.mark.xfail(reason='slice object is not iterable')
    def test_setitem_loc_iloc_slice(
        self,
        data: JSONArray
    ) -> None:
        super().test_setitem_loc_iloc_slice(data)

    @pytest.mark.xfail(reason='slice object is not iterable')
    def test_setitem_slice_mismatch_length_raises(
        self,
        data: JSONArray
    ) -> None:
        super().test_setitem_slice_mismatch_length_raises(data)

    @pytest.mark.xfail(reason='slice object is not iterable')
    def test_setitem_slice_array(
        self,
        data: JSONArray
    ) -> None:
        super().test_setitem_slice_array(data)

    @pytest.mark.xfail(reason='Fail to raise')
    def test_setitem_invalid(
        self,
        data: JSONArray,
        invalid_scalar: Any
    ) -> None:
        super().test_setitem_invalid(data, invalid_scalar)

    @pytest.mark.xfail(reason='only integer scalar arrays can be converted')
    def test_setitem_2d_values(
        self,
        data: JSONArray
    ) -> None:
        super().test_setitem_2d_values(data)

    @pytest.mark.xfail(reason="data type 'json' not understood")
    @pytest.mark.parametrize('engine', ['c', 'python'])
    def test_EA_types(
        self,
        engine: str,
        data: JSONArray,
        request: Any
    ) -> None:
        super().test_EA_types(engine, data, request)


def custom_assert_series_equal(
    left: pd.Series,
    right: pd.Series,
    *args: Any,
    **kwargs: Any
) -> None:
    if left.dtype.name == 'json':
        assert left.dtype == right.dtype
        left = pd.Series(JSONArray(left.values.astype(object)), index=left.index, name=left.name)
        right = pd.Series(JSONArray(right.values.astype(object)), index=right.index, name=right.name)
    tm.assert_series_equal(left, right, *args, **kwargs)


def custom_assert_frame_equal(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *args: Any,
    **kwargs: Any
) -> None:
    obj_type = kwargs.get('obj', 'DataFrame')
    tm.assert_index_equal(
        left.columns,
        right.columns,
        exact=kwargs.get('check_column_type', 'equiv'),
        check_names=kwargs.get('check_names', True),
        check_exact=kwargs.get('check_exact', False),
        check_categorical=kwargs.get('check_categorical', True),
        obj=f'{obj_type}.columns'
    )
    jsons = (left.dtypes == 'json').index
    for col in jsons:
        custom_assert_series_equal(left[col], right[col], *args, **kwargs)
    left = left.drop(columns=jsons)
    right = right.drop(columns=jsons)
    tm.assert_frame_equal(left, right, *args, **kwargs)


def test_custom_asserts() -> None:
    data = JSONArray([
        collections.UserDict({'a': 1}),
        collections.UserDict({'b': 2}),
        collections.UserDict({'c': 3})
    ])
    a = pd.Series(data)
    custom_assert_series_equal(a, a)
    custom_assert_frame_equal(a.to_frame(), a.to_frame())
    b = pd.Series(data.take([0, 0, 1]))
    msg = 'Series are different'
    with pytest.raises(AssertionError, match=msg):
        custom_assert_series_equal(a, b)
    with pytest.raises(AssertionError, match=msg):
        custom_assert_frame_equal(a.to_frame(), b.to_frame())
