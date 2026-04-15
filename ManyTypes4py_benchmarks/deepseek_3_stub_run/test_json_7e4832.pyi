import collections
import operator
import sys
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    TypeVar,
    overload,
)

import numpy as np
import pandas as pd
import pandas._testing as tm
import pytest
from pandas.tests.extension import base
from pandas.tests.extension.json.array import JSONArray, JSONDtype

unhashable: Any = ...

_T = TypeVar("_T")

@pytest.fixture
def dtype() -> JSONDtype: ...

@pytest.fixture
def data() -> JSONArray: ...

@pytest.fixture
def data_missing() -> JSONArray: ...

@pytest.fixture
def data_for_sorting() -> JSONArray: ...

@pytest.fixture
def data_missing_for_sorting() -> JSONArray: ...

@pytest.fixture
def na_cmp() -> Callable[[Any, Any], bool]: ...

@pytest.fixture
def data_for_grouping() -> JSONArray: ...

class TestJSONArray(base.ExtensionTests):
    @pytest.mark.xfail(reason="comparison method not implemented for JSONArray (GH-37867)")
    def test_contains(self, data: JSONArray) -> None: ...

    @pytest.mark.xfail(reason="not implemented constructor from dtype")
    def test_from_dtype(self, data: JSONArray) -> None: ...

    @pytest.mark.xfail(reason="RecursionError, GH-33900")
    def test_series_constructor_no_data_with_index(self, dtype: JSONDtype, na_value: Any) -> None: ...

    @pytest.mark.xfail(reason="RecursionError, GH-33900")
    def test_series_constructor_scalar_na_with_index(self, dtype: JSONDtype, na_value: Any) -> None: ...

    @pytest.mark.xfail(reason="collection as scalar, GH-33901")
    def test_series_constructor_scalar_with_index(self, data: JSONArray, dtype: JSONDtype) -> None: ...

    @pytest.mark.xfail(reason="Different definitions of NA")
    def test_stack(self) -> None: ...

    @pytest.mark.xfail(reason="dict for NA")
    def test_unstack(self, data: JSONArray, index: Any) -> None: ...

    @pytest.mark.xfail(reason="Setting a dict as a scalar")
    def test_fillna_series(self) -> None: ...

    @pytest.mark.xfail(reason="Setting a dict as a scalar")
    def test_fillna_frame(self) -> None: ...

    def test_fillna_with_none(self, data_missing: JSONArray) -> None: ...

    @pytest.mark.xfail(reason="fill value is a dictionary, takes incorrect code path")
    def test_fillna_limit_frame(self, data_missing: JSONArray) -> None: ...

    @pytest.mark.xfail(reason="fill value is a dictionary, takes incorrect code path")
    def test_fillna_limit_series(self, data_missing: JSONArray) -> None: ...

    @pytest.mark.parametrize(
        "limit_area, input_ilocs, expected_ilocs",
        [
            ("outside", [1, 0, 0, 0, 1], [1, 0, 0, 0, 1]),
            ("outside", [1, 0, 1, 0, 1], [1, 0, 1, 0, 1]),
            ("outside", [0, 1, 1, 1, 0], [0, 1, 1, 1, 1]),
            ("outside", [0, 1, 0, 1, 0], [0, 1, 0, 1, 1]),
            ("inside", [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]),
            ("inside", [1, 0, 1, 0, 1], [1, 1, 1, 1, 1]),
            ("inside", [0, 1, 1, 1, 0], [0, 1, 1, 1, 0]),
            ("inside", [0, 1, 0, 1, 0], [0, 1, 1, 1, 0]),
        ],
    )
    def test_ffill_limit_area(
        self,
        data_missing: JSONArray,
        limit_area: str,
        input_ilocs: List[int],
        expected_ilocs: List[int],
    ) -> None: ...

    @unhashable
    def test_value_counts(self, all_data: JSONArray, dropna: bool) -> None: ...

    @unhashable
    def test_value_counts_with_normalize(self, data: JSONArray) -> None: ...

    @unhashable
    def test_sort_values_frame(self) -> None: ...

    @pytest.mark.xfail(reason="combine for JSONArray not supported")
    def test_combine_le(self, data_repeated: JSONArray) -> None: ...

    @pytest.mark.xfail(
        reason="combine for JSONArray not supported - may pass depending on random data",
        strict=False,
        raises=AssertionError,
    )
    def test_combine_first(self, data: JSONArray) -> None: ...

    @pytest.mark.xfail(reason="broadcasting error")
    def test_where_series(self, data: JSONArray, na_value: Any) -> None: ...

    @pytest.mark.xfail(reason="Can't compare dicts.")
    def test_searchsorted(self, data_for_sorting: JSONArray) -> None: ...

    @pytest.mark.xfail(reason="Can't compare dicts.")
    def test_equals(self, data: JSONArray, na_value: Any, as_series: bool) -> None: ...

    @pytest.mark.skip("fill-value is interpreted as a dict of values")
    def test_fillna_copy_frame(self, data_missing: JSONArray) -> None: ...

    @pytest.mark.xfail(reason="Fails with CoW")
    def test_equals_same_data_different_object(self, data: JSONArray) -> None: ...

    @pytest.mark.xfail(reason="failing on np.array(self, dtype=str)")
    def test_astype_str(self) -> None: ...

    @unhashable
    def test_groupby_extension_transform(self) -> None: ...

    @unhashable
    def test_groupby_extension_apply(self) -> None: ...

    @unhashable
    def test_groupby_extension_agg(self) -> None: ...

    @unhashable
    def test_groupby_extension_no_sort(self) -> None: ...

    def test_arith_frame_with_scalar(
        self, data: JSONArray, all_arithmetic_operators: str, request: pytest.FixtureRequest
    ) -> None: ...

    def test_compare_array(
        self, data: JSONArray, comparison_op: Callable[[Any, Any], bool], request: pytest.FixtureRequest
    ) -> None: ...

    @pytest.mark.xfail(reason="ValueError: Must have equal len keys and value")
    def test_setitem_loc_scalar_mixed(self, data: JSONArray) -> None: ...

    @pytest.mark.xfail(reason="ValueError: Must have equal len keys and value")
    def test_setitem_loc_scalar_multiple_homogoneous(self, data: JSONArray) -> None: ...

    @pytest.mark.xfail(reason="ValueError: Must have equal len keys and value")
    def test_setitem_iloc_scalar_mixed(self, data: JSONArray) -> None: ...

    @pytest.mark.xfail(reason="ValueError: Must have equal len keys and value")
    def test_setitem_iloc_scalar_multiple_homogoneous(self, data: JSONArray) -> None: ...

    @pytest.mark.parametrize(
        "mask",
        [
            np.array([True, True, True, False, False]),
            pd.array([True, True, True, False, False], dtype="boolean"),
            pd.array([True, True, True, pd.NA, pd.NA], dtype="boolean"),
        ],
        ids=["numpy-array", "boolean-array", "boolean-array-na"],
    )
    def test_setitem_mask(
        self,
        data: JSONArray,
        mask: Union[np.ndarray, pd.array],
        box_in_series: bool,
        request: pytest.FixtureRequest,
    ) -> None: ...

    def test_setitem_mask_raises(
        self, data: JSONArray, box_in_series: bool, request: pytest.FixtureRequest
    ) -> None: ...

    @pytest.mark.xfail(reason="cannot set using a list-like indexer with a different length")
    def test_setitem_mask_boolean_array_with_na(self, data: JSONArray, box_in_series: bool) -> None: ...

    @pytest.mark.parametrize(
        "idx",
        [
            [0, 1, 2],
            pd.array([0, 1, 2], dtype="Int64"),
            np.array([0, 1, 2]),
        ],
        ids=["list", "integer-array", "numpy-array"],
    )
    def test_setitem_integer_array(
        self,
        data: JSONArray,
        idx: Union[List[int], pd.array, np.ndarray],
        box_in_series: bool,
        request: pytest.FixtureRequest,
    ) -> None: ...

    @pytest.mark.xfail(reason="list indices must be integers or slices, not NAType")
    @pytest.mark.parametrize(
        "idx, box_in_series",
        [
            ([0, 1, 2, pd.NA], False),
            pytest.param(
                [0, 1, 2, pd.NA],
                True,
                marks=pytest.mark.xfail(reason="GH-31948"),
            ),
            (pd.array([0, 1, 2, pd.NA], dtype="Int64"), False),
            (pd.array([0, 1, 2, pd.NA], dtype="Int64"), True),
        ],
        ids=["list-False", "list-True", "integer-array-False", "integer-array-True"],
    )
    def test_setitem_integer_with_missing_raises(
        self, data: JSONArray, idx: Union[List[Union[int, pd.NA]], pd.array], box_in_series: bool
    ) -> None: ...

    @pytest.mark.xfail(reason="Fails to raise")
    def test_setitem_scalar_key_sequence_raise(self, data: JSONArray) -> None: ...

    def test_setitem_with_expansion_dataframe_column(
        self, data: JSONArray, full_indexer: Any, request: pytest.FixtureRequest
    ) -> None: ...

    @pytest.mark.xfail(reason="slice is not iterable")
    def test_setitem_frame_2d_values(self, data: JSONArray) -> None: ...

    @pytest.mark.xfail(reason="cannot set using a list-like indexer with a different length")
    @pytest.mark.parametrize("setter", ["loc", None])
    def test_setitem_mask_broadcast(self, data: JSONArray, setter: Optional[str]) -> None: ...

    @pytest.mark.xfail(reason="cannot set using a slice indexer with a different length")
    def test_setitem_slice(self, data: JSONArray, box_in_series: bool) -> None: ...

    @pytest.mark.xfail(reason="slice object is not iterable")
    def test_setitem_loc_iloc_slice(self, data: JSONArray) -> None: ...

    @pytest.mark.xfail(reason="slice object is not iterable")
    def test_setitem_slice_mismatch_length_raises(self, data: JSONArray) -> None: ...

    @pytest.mark.xfail(reason="slice object is not iterable")
    def test_setitem_slice_array(self, data: JSONArray) -> None: ...

    @pytest.mark.xfail(reason="Fail to raise")
    def test_setitem_invalid(self, data: JSONArray, invalid_scalar: Any) -> None: ...

    @pytest.mark.xfail(reason="only integer scalar arrays can be converted")
    def test_setitem_2d_values(self, data: JSONArray) -> None: ...

    @pytest.mark.xfail(reason="data type 'json' not understood")
    @pytest.mark.parametrize("engine", ["c", "python"])
    def test_EA_types(
        self, engine: str, data: JSONArray, request: pytest.FixtureRequest
    ) -> None: ...


def custom_assert_series_equal(
    left: pd.Series,
    right: pd.Series,
    *args: Any,
    **kwargs: Any,
) -> None: ...


def custom_assert_frame_equal(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *args: Any,
    **kwargs: Any,
) -> None: ...


def test_custom_asserts() -> None: ...