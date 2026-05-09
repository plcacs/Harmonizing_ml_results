import collections
import operator
import sys
import numpy as np
import pytest
import pandas as pd
from pandas import (
    Index,
    Series,
    DataFrame,
    NA,
    Int64Dtype,
    Categorical,
    CategoricalDtype,
    Interval,
    IntervalDtype,
    Period,
    PeriodDtype,
    SparseDtype,
    BooleanDtype,
)
from pandas._testing import (
    tm,
    assert_series_equal,
    assert_frame_equal,
)
from pandas.tests.extension import base
from pandas.tests.extension.json.array import (
    JSONArray,
    JSONDtype,
    make_data,
)

@pytest.fixture
def dtype() -> JSONDtype:
    ...

@pytest.fixture
def data() -> JSONArray:
    ...

@pytest.fixture
def data_missing() -> JSONArray:
    ...

@pytest.fixture
def data_for_sorting() -> JSONArray:
    ...

@pytest.fixture
def data_missing_for_sorting() -> JSONArray:
    ...

@pytest.fixture
def na_cmp() -> operator.methodcaller:
    ...

@pytest.fixture
def data_for_grouping() -> JSONArray:
    ...

class TestJSONArray(base.ExtensionTests):
    @pytest.mark.xfail
    def test_contains(self, data: JSONArray) -> None:
        ...

    @pytest.mark.xfail
    def test_from_dtype(self, data: JSONArray) -> None:
        ...

    @pytest.mark.xfail
    def test_series_constructor_no_data_with_index(self, dtype: JSONDtype, na_value: dict) -> None:
        ...

    @pytest.mark.xfail
    def test_series_constructor_scalar_na_with_index(self, dtype: JSONDtype, na_value: dict) -> None:
        ...

    @pytest.mark.xfail
    def test_series_constructor_scalar_with_index(self, data: JSONArray, dtype: JSONDtype) -> None:
        ...

    @pytest.mark.xfail
    def test_stack(self) -> None:
        ...

    @pytest.mark.xfail
    def test_unstack(self, data: JSONArray, index: Index) -> None:
        ...

    @pytest.mark.xfail
    def test_fillna_series(self) -> None:
        ...

    @pytest.mark.xfail
    def test_fillna_frame(self) -> None:
        ...

    def test_fillna_with_none(self, data_missing: JSONArray) -> None:
        ...

    @pytest.mark.xfail
    def test_fillna_limit_frame(self, data_missing: JSONArray) -> None:
        ...

    @pytest.mark.xfail
    def test_fillna_limit_series(self, data_missing: JSONArray) -> None:
        ...

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
        ],
    )
    def test_ffill_limit_area(
        self,
        data_missing: JSONArray,
        limit_area: str,
        input_ilocs: list[int],
        expected_ilocs: list[int],
    ) -> None:
        ...

    @pytest.mark.xfail
    def test_value_counts(self, all_data: JSONArray, dropna: bool) -> None:
        ...

    @pytest.mark.xfail
    def test_value_counts_with_normalize(self, data: JSONArray) -> None:
        ...

    @pytest.mark.xfail
    def test_sort_values_frame(self) -> None:
        ...

    @pytest.mark.xfail
    def test_combine_le(self, data_repeated: JSONArray) -> None:
        ...

    @pytest.mark.xfail
    def test_combine_first(self, data: JSONArray) -> None:
        ...

    @pytest.mark.xfail
    def test_where_series(self, data: JSONArray, na_value: dict) -> None:
        ...

    @pytest.mark.xfail
    def test_searchsorted(self, data_for_sorting: JSONArray) -> None:
        ...

    @pytest.mark.xfail
    def test_equals(self, data: JSONArray, na_value: dict, as_series: bool) -> None:
        ...

    @pytest.mark.skip
    def test_fillna_copy_frame(self, data_missing: JSONArray) -> None:
        ...

    @pytest.mark.xfail
    def test_equals_same_data_different_object(self, data: JSONArray) -> None:
        ...

    @pytest.mark.xfail
    def test_astype_str(self) -> None:
        ...

    @pytest.mark.xfail
    def test_groupby_extension_transform(self) -> None:
        ...

    @pytest.mark.xfail
    def test_groupby_extension_apply(self) -> None:
        ...

    @pytest.mark.xfail
    def test_groupby_extension_agg(self) -> None:
        ...

    @pytest.mark.xfail
    def test_groupby_extension_no_sort(self) -> None:
        ...

    def test_arith_frame_with_scalar(
        self,
        data: JSONArray,
        all_arithmetic_operators: str,
        request: pytest.FixtureRequest,
    ) -> None:
        ...

    def test_compare_array(
        self,
        data: JSONArray,
        comparison_op: operator.methodcaller,
        request: pytest.FixtureRequest,
    ) -> None:
        ...

    @pytest.mark.xfail
    def test_setitem_loc_scalar_mixed(self, data: JSONArray) -> None:
        ...

    @pytest.mark.xfail
    def test_setitem_loc_scalar_multiple_homogoneous(self, data: JSONArray) -> None:
        ...

    @pytest.mark.xfail
    def test_setitem_iloc_scalar_mixed(self, data: JSONArray) -> None:
        ...

    @pytest.mark.xfail
    def test_setitem_iloc_scalar_multiple_homogoneous(self, data: JSONArray) -> None:
        ...

    @pytest.mark.parametrize(
        'mask',
        [
            np.array([True, True, True, False, False]),
            pd.array([True, True, True, False, False], dtype='boolean'),
            pd.array([True, True, True, pd.NA, pd.NA], dtype='boolean'),
        ],
        ids=['numpy-array', 'boolean-array', 'boolean-array-na'],
    )
    def test_setitem_mask(
        self,
        data: JSONArray,
        mask: np.ndarray | pd.BooleanArray,
        box_in_series: bool,
        request: pytest.FixtureRequest,
    ) -> None:
        ...

    def test_setitem_mask_raises(
        self,
        data: JSONArray,
        box_in_series: bool,
        request: pytest.FixtureRequest,
    ) -> None:
        ...

    @pytest.mark.xfail
    def test_setitem_mask_boolean_array_with_na(
        self,
        data: JSONArray,
        box_in_series: bool,
    ) -> None:
        ...

    @pytest.mark.parametrize(
        'idx',
        [
            [0, 1, 2],
            pd.array([0, 1, 2], dtype='Int64'),
            np.array([0, 1, 2]),
        ],
        ids=['list', 'integer-array', 'numpy-array'],
    )
    def test_setitem_integer_array(
        self,
        data: JSONArray,
        idx: list[int] | pd.IntegerArray | np.ndarray,
        box_in_series: bool,
        request: pytest.FixtureRequest,
    ) -> None:
        ...

    @pytest.mark.xfail
    @pytest.mark.parametrize(
        'idx, box_in_series',
        [
            ([0, 1, 2, pd.NA], False),
            pytest.param([0, 1, 2, pd.NA], True, marks=pytest.mark.xfail(reason='GH-31948')),
            (pd.array([0, 1, 2, pd.NA], dtype='Int64'), False),
            (pd.array([0, 1, 2, pd.NA], dtype='Int64'), True),
        ],
        ids=['list-False', 'list-True', 'integer-array-False', 'integer-array-True'],
    )
    def test_setitem_integer_with_missing_raises(
        self,
        data: JSONArray,
        idx: list[int | pd.NA] | pd.IntegerArray,
        box_in_series: bool,
    ) -> None:
        ...

    @pytest.mark.xfail
    def test_setitem_scalar_key_sequence_raise(self, data: JSONArray) -> None:
        ...

    def test_setitem_with_expansion_dataframe_column(
        self,
        data: JSONArray,
        full_indexer: slice | int | list[int],
        request: pytest.FixtureRequest,
    ) -> None:
        ...

    @pytest.mark.xfail
    def test_setitem_frame_2d_values(self, data: JSONArray) -> None:
        ...

    @pytest.mark.xfail
    @pytest.mark.parametrize('setter', ['loc', None])
    def test_setitem_mask_broadcast(
        self,
        data: JSONArray,
        setter: str | None,
    ) -> None:
        ...

    @pytest.mark.xfail
    def test_setitem_slice(self, data: JSONArray, box_in_series: bool) -> None:
        ...

    @pytest.mark.xfail
    def test_setitem_loc_iloc_slice(self, data: JSONArray) -> None:
        ...

    @pytest.mark.xfail
    def test_setitem_slice_mismatch_length_raises(self, data: JSONArray) -> None:
        ...

    @pytest.mark.xfail
    def test_setitem_slice_array(self, data: JSONArray) -> None:
        ...

    @pytest.mark.xfail
    def test_setitem_invalid(self, data: JSONArray, invalid_scalar: object) -> None:
        ...

    @pytest.mark.xfail
    def test_setitem_2d_values(self, data: JSONArray) -> None:
        ...

    @pytest.mark.xfail
    @pytest.mark.parametrize('engine', ['c', 'python'])
    def test_EA_types(
        self,
        engine: str,
        data: JSONArray,
        request: pytest.FixtureRequest,
    ) -> None:
        ...

def custom_assert_series_equal(
    left: Series,
    right: Series,
    *args: object,
    **kwargs: object,
) -> None:
    ...

def custom_assert_frame_equal(
    left: DataFrame,
    right: DataFrame,
    *args: object,
    **kwargs: object,
) -> None:
    ...

def test_custom_asserts() -> None:
    ...