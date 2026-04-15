import inspect
import operator
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

import numpy as np
import pandas as pd
import pytest
from pandas._typing import Dtype
from pandas.core.dtypes.dtypes import NumpyEADtype

T = TypeVar("T")
T_BaseMethodsTests = TypeVar("T_BaseMethodsTests", bound="BaseMethodsTests")

class BaseMethodsTests:
    def test_hash_pandas_object(
        self, data: pd.core.arrays.base.ExtensionArray
    ) -> None: ...
    def test_value_counts_default_dropna(
        self, data: pd.core.arrays.base.ExtensionArray
    ) -> None: ...
    @pytest.mark.parametrize("dropna", [True, False])
    def test_value_counts(
        self, all_data: pd.core.arrays.base.ExtensionArray, dropna: bool
    ) -> None: ...
    def test_value_counts_with_normalize(
        self, data: pd.core.arrays.base.ExtensionArray
    ) -> None: ...
    def test_count(self, data_missing: pd.core.arrays.base.ExtensionArray) -> None: ...
    def test_series_count(
        self, data_missing: pd.core.arrays.base.ExtensionArray
    ) -> None: ...
    def test_apply_simple_series(
        self, data: pd.core.arrays.base.ExtensionArray
    ) -> None: ...
    @pytest.mark.parametrize("na_action", [None, "ignore"])
    def test_map(
        self,
        data_missing: pd.core.arrays.base.ExtensionArray,
        na_action: Optional[Literal["ignore"]],
    ) -> None: ...
    def test_argsort(
        self, data_for_sorting: pd.core.arrays.base.ExtensionArray
    ) -> None: ...
    def test_argsort_missing_array(
        self, data_missing_for_sorting: pd.core.arrays.base.ExtensionArray
    ) -> None: ...
    def test_argsort_missing(
        self, data_missing_for_sorting: pd.core.arrays.base.ExtensionArray
    ) -> None: ...
    def test_argmin_argmax(
        self,
        data_for_sorting: pd.core.arrays.base.ExtensionArray,
        data_missing_for_sorting: pd.core.arrays.base.ExtensionArray,
        na_value: Any,
    ) -> None: ...
    @pytest.mark.parametrize("method", ["argmax", "argmin"])
    def test_argmin_argmax_empty_array(
        self, method: Literal["argmax", "argmin"], data: pd.core.arrays.base.ExtensionArray
    ) -> None: ...
    @pytest.mark.parametrize("method", ["argmax", "argmin"])
    def test_argmin_argmax_all_na(
        self,
        method: Literal["argmax", "argmin"],
        data: pd.core.arrays.base.ExtensionArray,
        na_value: Any,
    ) -> None: ...
    @pytest.mark.parametrize(
        "op_name, skipna, expected",
        [
            ("idxmax", True, 0),
            ("idxmin", True, 2),
            ("argmax", True, 0),
            ("argmin", True, 2),
            ("idxmax", False, -1),
            ("idxmin", False, -1),
            ("argmax", False, -1),
            ("argmin", False, -1),
        ],
    )
    def test_argreduce_series(
        self,
        data_missing_for_sorting: pd.core.arrays.base.ExtensionArray,
        op_name: Literal["idxmax", "idxmin", "argmax", "argmin"],
        skipna: bool,
        expected: int,
    ) -> None: ...
    def test_argmax_argmin_no_skipna_notimplemented(
        self, data_missing_for_sorting: pd.core.arrays.base.ExtensionArray
    ) -> None: ...
    @pytest.mark.parametrize(
        "na_position, expected",
        [
            ("last", np.ndarray),
            ("first", np.ndarray),
        ],
    )
    def test_nargsort(
        self,
        data_missing_for_sorting: pd.core.arrays.base.ExtensionArray,
        na_position: Literal["last", "first"],
        expected: np.ndarray,
    ) -> None: ...
    @pytest.mark.parametrize("ascending", [True, False])
    def test_sort_values(
        self,
        data_for_sorting: pd.core.arrays.base.ExtensionArray,
        ascending: bool,
        sort_by_key: Optional[Callable[[pd.Series], pd.Series]],
    ) -> None: ...
    @pytest.mark.parametrize("ascending", [True, False])
    def test_sort_values_missing(
        self,
        data_missing_for_sorting: pd.core.arrays.base.ExtensionArray,
        ascending: bool,
        sort_by_key: Optional[Callable[[pd.Series], pd.Series]],
    ) -> None: ...
    @pytest.mark.parametrize("ascending", [True, False])
    def test_sort_values_frame(
        self,
        data_for_sorting: pd.core.arrays.base.ExtensionArray,
        ascending: bool,
    ) -> None: ...
    @pytest.mark.parametrize("keep", ["first", "last", False])
    def test_duplicated(
        self,
        data: pd.core.arrays.base.ExtensionArray,
        keep: Union[Literal["first", "last"], bool],
    ) -> None: ...
    @pytest.mark.parametrize("box", [pd.Series, Callable[[Any], Any]])
    @pytest.mark.parametrize(
        "method", [Callable[[Any], Any], Callable[[Sequence[Any]], np.ndarray]]
    )
    def test_unique(
        self,
        data: pd.core.arrays.base.ExtensionArray,
        box: Union[Type[pd.Series], Callable[[Any], Any]],
        method: Union[Callable[[Any], Any], Callable[[Sequence[Any]], np.ndarray]],
    ) -> None: ...
    def test_factorize(
        self, data_for_grouping: pd.core.arrays.base.ExtensionArray
    ) -> None: ...
    def test_factorize_equivalence(
        self, data_for_grouping: pd.core.arrays.base.ExtensionArray
    ) -> None: ...
    def test_factorize_empty(
        self, data: pd.core.arrays.base.ExtensionArray
    ) -> None: ...
    def test_fillna_limit_frame(
        self, data_missing: pd.core.arrays.base.ExtensionArray
    ) -> None: ...
    def test_fillna_limit_series(
        self, data_missing: pd.core.arrays.base.ExtensionArray
    ) -> None: ...
    def test_fillna_copy_frame(
        self, data_missing: pd.core.arrays.base.ExtensionArray
    ) -> None: ...
    def test_fillna_copy_series(
        self, data_missing: pd.core.arrays.base.ExtensionArray
    ) -> None: ...
    def test_fillna_length_mismatch(
        self, data_missing: pd.core.arrays.base.ExtensionArray
    ) -> None: ...
    _combine_le_expected_dtype: NumpyEADtype = ...
    def test_combine_le(
        self, data_repeated: Callable[[int], Tuple[pd.core.arrays.base.ExtensionArray, ...]]
    ) -> None: ...
    def test_combine_add(
        self, data_repeated: Callable[[int], Tuple[pd.core.arrays.base.ExtensionArray, ...]]
    ) -> None: ...
    def test_combine_first(
        self, data: pd.core.arrays.base.ExtensionArray
    ) -> None: ...
    @pytest.mark.parametrize("frame", [True, False])
    @pytest.mark.parametrize(
        "periods, indices",
        [
            (-2, Sequence[int]),
            (0, Sequence[int]),
            (2, Sequence[int]),
        ],
    )
    def test_container_shift(
        self,
        data: pd.core.arrays.base.ExtensionArray,
        frame: bool,
        periods: int,
        indices: Sequence[int],
    ) -> None: ...
    def test_shift_0_periods(
        self, data: pd.core.arrays.base.ExtensionArray
    ) -> None: ...
    @pytest.mark.parametrize("periods", [1, -2])
    def test_diff(
        self, data: pd.core.arrays.base.ExtensionArray, periods: int
    ) -> None: ...
    @pytest.mark.parametrize(
        "periods, indices",
        [
            (-4, Sequence[int]),
            (-1, Sequence[int]),
            (0, Sequence[int]),
            (1, Sequence[int]),
            (4, Sequence[int]),
        ],
    )
    def test_shift_non_empty_array(
        self,
        data: pd.core.arrays.base.ExtensionArray,
        periods: int,
        indices: Sequence[int],
    ) -> None: ...
    @pytest.mark.parametrize("periods", [-4, -1, 0, 1, 4])
    def test_shift_empty_array(
        self, data: pd.core.arrays.base.ExtensionArray, periods: int
    ) -> None: ...
    def test_shift_zero_copies(
        self, data: pd.core.arrays.base.ExtensionArray
    ) -> None: ...
    def test_shift_fill_value(
        self, data: pd.core.arrays.base.ExtensionArray
    ) -> None: ...
    def test_not_hashable(
        self, data: pd.core.arrays.base.ExtensionArray
    ) -> None: ...
    def test_hash_pandas_object_works(
        self,
        data: pd.core.arrays.base.ExtensionArray,
        as_frame: bool,
    ) -> None: ...
    def test_searchsorted(
        self,
        data_for_sorting: pd.core.arrays.base.ExtensionArray,
        as_series: bool,
    ) -> None: ...
    def _test_searchsorted_bool_dtypes(
        self,
        data_for_sorting: pd.core.arrays.base.ExtensionArray,
        as_series: bool,
    ) -> None: ...
    def test_where_series(
        self,
        data: pd.core.arrays.base.ExtensionArray,
        na_value: Any,
        as_frame: bool,
    ) -> None: ...
    @pytest.mark.parametrize("repeats", [0, 1, 2, Sequence[int]])
    def test_repeat(
        self,
        data: pd.core.arrays.base.ExtensionArray,
        repeats: Union[int, Sequence[int]],
        as_series: bool,
        use_numpy: bool,
    ) -> None: ...
    @pytest.mark.parametrize(
        "repeats, kwargs, error, msg",
        [
            (2, dict, ValueError, str),
            (-1, dict, ValueError, str),
            (Sequence[int], dict, ValueError, str),
            (2, dict, TypeError, str),
        ],
    )
    def test_repeat_raises(
        self,
        data: pd.core.arrays.base.ExtensionArray,
        repeats: Union[int, Sequence[int]],
        kwargs: dict[str, Any],
        error: Type[Exception],
        msg: str,
        use_numpy: bool,
    ) -> None: ...
    def test_delete(
        self, data: pd.core.arrays.base.ExtensionArray
    ) -> None: ...
    def test_insert(
        self, data: pd.core.arrays.base.ExtensionArray
    ) -> None: ...
    def test_insert_invalid(
        self,
        data: pd.core.arrays.base.ExtensionArray,
        invalid_scalar: Any,
    ) -> None: ...
    def test_insert_invalid_loc(
        self, data: pd.core.arrays.base.ExtensionArray
    ) -> None: ...
    @pytest.mark.parametrize("box", [pd.array, pd.Series, pd.DataFrame])
    def test_equals(
        self,
        data: pd.core.arrays.base.ExtensionArray,
        na_value: Any,
        as_series: bool,
        box: Union[
            Type[pd.core.arrays.base.ExtensionArray],
            Type[pd.Series],
            Type[pd.DataFrame],
        ],
    ) -> None: ...
    def test_equals_same_data_different_object(
        self, data: pd.core.arrays.base.ExtensionArray
    ) -> None: ...