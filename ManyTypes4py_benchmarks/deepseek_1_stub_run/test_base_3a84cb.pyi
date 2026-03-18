```python
from collections import defaultdict
from datetime import datetime
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    Iterator,
    List,
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
from pandas import (
    CategoricalIndex,
    DataFrame,
    DatetimeIndex,
    IntervalIndex,
    MultiIndex,
    PeriodIndex,
    RangeIndex,
    Series,
    TimedeltaIndex,
)
from pandas.core.indexes.api import Index
import pandas._testing as tm
import numpy.typing as npt
from typing_extensions import Literal

_T = TypeVar("_T")
_IndexT = TypeVar("_IndexT", bound=Index)

def _get_combined_index(
    indexes: Sequence[Index],
) -> Index: ...

def ensure_index(
    index_like: Any,
    copy: bool = ...,
) -> Index: ...

def ensure_index_from_sequences(
    sequences: Sequence[Sequence[Any]],
    names: Optional[Sequence[Hashable]] = ...,
) -> Union[Index, MultiIndex]: ...

class TestIndex:
    @pytest.fixture
    def simple_index(self) -> Index: ...
    def test_can_hold_identifiers(self, simple_index: Index) -> None: ...
    @pytest.mark.parametrize("index", ["datetime"], indirect=True)
    def test_new_axis(self, index: Index) -> None: ...
    def test_constructor_regular(self, index: Index) -> None: ...
    @pytest.mark.parametrize("index", ["string"], indirect=True)
    def test_constructor_casting(self, index: Index) -> None: ...
    def test_constructor_copy(self, using_infer_string: bool) -> None: ...
    @pytest.mark.parametrize("cast_as_obj", [True, False])
    @pytest.mark.parametrize(
        "index",
        [
            date_range("2015-01-01 10:00", freq="D", periods=3, tz="US/Eastern", name="Green Eggs & Ham"),
            date_range("2015-01-01 10:00", freq="D", periods=3),
            timedelta_range("1 days", freq="D", periods=3),
            period_range("2015-01-01", freq="D", periods=3),
        ],
    )
    def test_constructor_from_index_dtlike(self, cast_as_obj: bool, index: Index) -> None: ...
    @pytest.mark.parametrize(
        "index,has_tz",
        [
            (date_range("2015-01-01 10:00", freq="D", periods=3, tz="US/Eastern"), True),
            (timedelta_range("1 days", freq="D", periods=3), False),
            (period_range("2015-01-01", freq="D", periods=3), False),
        ],
    )
    def test_constructor_from_series_dtlike(self, index: Index, has_tz: bool) -> None: ...
    def test_constructor_from_series_freq(self) -> None: ...
    def test_constructor_from_frame_series_freq(self, using_infer_string: bool) -> None: ...
    def test_constructor_int_dtype_nan(self) -> None: ...
    @pytest.mark.parametrize(
        "klass,dtype,na_val",
        [(Index, np.float64, np.nan), (DatetimeIndex, "datetime64[s]", pd.NaT)],
    )
    def test_index_ctor_infer_nan_nat(self, klass: Type[Index], dtype: Any, na_val: Any) -> None: ...
    @pytest.mark.parametrize(
        "vals,dtype",
        [
            ([1, 2, 3, 4, 5], "int"),
            ([1.1, np.nan, 2.2, 3.0], "float"),
            (["A", "B", "C", np.nan], "obj"),
        ],
    )
    def test_constructor_simple_new(self, vals: List[Any], dtype: str) -> None: ...
    @pytest.mark.parametrize("attr", ["values", "asi8"])
    @pytest.mark.parametrize("klass", [Index, DatetimeIndex])
    def test_constructor_dtypes_datetime(
        self, tz_naive_fixture: Any, attr: str, klass: Type[Index]
    ) -> None: ...
    @pytest.mark.parametrize("attr", ["values", "asi8"])
    @pytest.mark.parametrize("klass", [Index, TimedeltaIndex])
    def test_constructor_dtypes_timedelta(self, attr: str, klass: Type[Index]) -> None: ...
    @pytest.mark.parametrize("value", [[], iter([]), (_ for _ in [])])
    @pytest.mark.parametrize(
        "klass",
        [Index, CategoricalIndex, DatetimeIndex, TimedeltaIndex],
    )
    def test_constructor_empty(self, value: Any, klass: Type[Index]) -> None: ...
    @pytest.mark.parametrize(
        "empty,klass",
        [
            (PeriodIndex([], freq="D"), PeriodIndex),
            (PeriodIndex(iter([]), freq="D"), PeriodIndex),
            (PeriodIndex((_ for _ in []), freq="D"), PeriodIndex),
            (RangeIndex(step=1), RangeIndex),
            (MultiIndex(levels=[[1, 2], ["blue", "red"]], codes=[[], []]), MultiIndex),
        ],
    )
    def test_constructor_empty_special(self, empty: Index, klass: Type[Index]) -> None: ...
    @pytest.mark.parametrize(
        "index",
        [
            "datetime",
            "float64",
            "float32",
            "int64",
            "int32",
            "period",
            "range",
            "repeats",
            "timedelta",
            "tuples",
            "uint64",
            "uint32",
        ],
        indirect=True,
    )
    def test_view_with_args(self, index: Index) -> None: ...
    @pytest.mark.parametrize(
        "index",
        [
            "string",
            pytest.param("categorical", marks=pytest.mark.xfail(reason="gh-25464")),
            "bool-object",
            "bool-dtype",
            "empty",
        ],
        indirect=True,
    )
    def test_view_with_args_object_array_raises(self, index: Index) -> None: ...
    @pytest.mark.parametrize("index", ["int64", "int32", "range"], indirect=True)
    def test_astype(self, index: Index) -> None: ...
    def test_equals_object(self) -> None: ...
    @pytest.mark.parametrize(
        "comp",
        [Index(["a", "b"]), Index(["a", "b", "d"]), ["a", "b", "c"]],
    )
    def test_not_equals_object(self, comp: Any) -> None: ...
    def test_identical(self) -> None: ...
    def test_is_(self) -> None: ...
    def test_asof_numeric_vs_bool_raises(self) -> None: ...
    @pytest.mark.parametrize("index", ["string"], indirect=True)
    def test_booleanindex(self, index: Index) -> None: ...
    def test_fancy(self, simple_index: Index) -> None: ...
    @pytest.mark.parametrize(
        "index",
        ["string", "int64", "int32", "uint64", "uint32", "float64", "float32"],
        indirect=True,
    )
    @pytest.mark.parametrize("dtype", [int, np.bool_])
    def test_empty_fancy(
        self,
        index: Index,
        dtype: Type[Any],
        request: Any,
        using_infer_string: bool,
    ) -> None: ...
    @pytest.mark.parametrize(
        "index",
        ["string", "int64", "int32", "uint64", "uint32", "float64", "float32"],
        indirect=True,
    )
    def test_empty_fancy_raises(self, index: Index) -> None: ...
    def test_union_dt_as_obj(self, simple_index: Index) -> None: ...
    def test_map_with_tuples(self) -> None: ...
    def test_map_with_tuples_mi(self) -> None: ...
    @pytest.mark.parametrize(
        "index",
        [
            date_range("2020-01-01", freq="D", periods=10),
            period_range("2020-01-01", freq="D", periods=10),
            timedelta_range("1 day", periods=10),
        ],
    )
    def test_map_tseries_indices_return_index(self, index: Index) -> None: ...
    def test_map_tseries_indices_accsr_return_index(self) -> None: ...
    @pytest.mark.parametrize(
        "mapper",
        [
            lambda values, index: {i: e for e, i in zip(values, index)},
            lambda values, index: Series(values, index),
        ],
    )
    def test_map_dictlike_simple(self, mapper: Callable) -> None: ...
    @pytest.mark.parametrize(
        "mapper",
        [
            lambda values, index: {i: e for e, i in zip(values, index)},
            lambda values, index: Series(values, index),
        ],
    )
    @pytest.mark.filterwarnings("ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning")
    def test_map_dictlike(self, index: Index, mapper: Callable, request: Any) -> None: ...
    @pytest.mark.parametrize(
        "mapper",
        [
            Series(["foo", 2.0, "baz"], index=[0, 2, -1]),
            {0: "foo", 2: 2.0, -1: "baz"},
        ],
    )
    def test_map_with_non_function_missing_values(self, mapper: Any) -> None: ...
    def test_map_na_exclusion(self) -> None: ...
    def test_map_defaultdict(self) -> None: ...
    @pytest.mark.parametrize("name,expected", [("foo", "foo"), ("bar", None)])
    def test_append_empty_preserve_name(self, name: Optional[str], expected: Optional[str]) -> None: ...
    @pytest.mark.parametrize(
        "index, expected",
        [
            ("string", False),
            ("bool-object", False),
            ("bool-dtype", False),
            ("categorical", False),
            ("int64", True),
            ("int32", True),
            ("uint64", True),
            ("uint32", True),
            ("datetime", False),
            ("float64", True),
            ("float32", True),
        ],
        indirect=["index"],
    )
    def test_is_numeric(self, index: Index, expected: bool) -> None: ...
    @pytest.mark.parametrize(
        "index, expected",
        [
            ("string", True),
            ("bool-object", True),
            ("bool-dtype", False),
            ("categorical", False),
            ("int64", False),
            ("int32", False),
            ("uint64", False),
            ("uint32", False),
            ("datetime", False),
            ("float64", False),
            ("float32", False),
        ],
        indirect=["index"],
    )
    def test_is_object(self, index: Index, expected: bool, using_infer_string: bool) -> None: ...
    def test_summary(self, index: Index) -> None: ...
    def test_logical_compat(self, all_boolean_reductions: str, simple_index: Index) -> None: ...
    @pytest.mark.parametrize(
        "index",
        ["string", "int64", "int32", "float64", "float32"],
        indirect=True,
    )
    def test_drop_by_str_label(self, index: Index) -> None: ...
    @pytest.mark.parametrize(
        "index",
        ["string", "int64", "int32", "float64", "float32"],
        indirect=True,
    )
    @pytest.mark.parametrize("keys", [["foo", "bar"], ["1", "bar"]])
    def test_drop_by_str_label_raises_missing_keys(self, index: Index, keys: List[str]) -> None: ...
    @pytest.mark.parametrize(
        "index",
        ["string", "int64", "int32", "float64", "float32"],
        indirect=True,
    )
    def test_drop_by_str_label_errors_ignore(self, index: Index) -> None: ...
    def test_drop_by_numeric_label_loc(self) -> None: ...
    def test_drop_by_numeric_label_raises_missing_keys(self) -> None: ...
    @pytest.mark.parametrize(
        "key,expected",
        [(4, Index([1, 2, 3])), ([3, 4, 5], Index([1, 2]))],
    )
    def test_drop_by_numeric_label_errors_ignore(self, key: Any, expected: Index) -> None: ...
    @pytest.mark.parametrize(
        "values",
        [
            ["a", "b", ("c", "d")],
            ["a", ("c", "d"), "b"],
            [("c", "d"), "a", "b"],
        ],
    )
    @pytest.mark.parametrize("to_drop", [[("c", "d"), "a"], ["a", ("c", "d")]])
    def test_drop_tuple(self, values: List[Any], to_drop: List[Any]) -> None: ...
    @pytest.mark.filterwarnings("ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning")
    def test_drop_with_duplicates_in_index(self, index: Index) -> None: ...
    @pytest.mark.parametrize(
        "attr",
        [
            "is_monotonic_increasing",
            "is_monotonic_decreasing",
            "_is_strictly_monotonic_increasing",
            "_is_strictly_monotonic_decreasing",
        ],
    )
    def test_is_monotonic_incomparable(self, attr: str) -> None: ...
    @pytest.mark.parametrize("values", [["foo", "bar", "quux"], {"foo", "bar", "quux"}])
    @pytest.mark.parametrize(
        "index,expected",
        [
            (["qux", "baz", "foo", "bar"], [False, False, True, True]),
            ([], []),
        ],
    )
    def test_isin(self, values: Any, index: List[str], expected: List[bool]) -> None: ...
    def test_isin_nan_common_object(
        self,
        nulls_fixture: Any,
        nulls_fixture2: Any,
        using_infer_string: bool,
    ) -> None: ...
    def test_isin_nan_common_float64(
        self, nulls_fixture: Any, float_numpy_dtype: Any
    ) -> None: ...
    @pytest.mark.parametrize("level", [0, -1])
    @pytest.mark.parametrize(
        "index",
        [
            ["qux", "baz", "foo", "bar"],
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        ],
    )
    def test_isin_level_kwarg(self, level: int, index: Any) -> None: ...
    def test_isin_level_kwarg_bad_level_raises(self, index: Index) -> None: ...
    @pytest.mark.parametrize("label", [1.0, "foobar", "xyzzy", np.nan])
    def test_isin_level_kwarg_bad_label_raises(
        self, label: Any, index: Index
    ) -> None: ...
    @pytest.mark.parametrize(
        "empty", [[], Series(dtype=object), np.array([])]
    )
    def test_isin_empty(self, empty: Any) -> None: ...
    def test_isin_string_null(self, string_dtype_no_object: Any) -> None: ...
    @pytest.mark.parametrize(
        "values",
        [
            [1, 2, 3, 4],
            [1.0, 2.0, 3.0, 4.0],
            [True, True, True, True],
            ["foo", "bar", "baz", "qux"],
            date_range("2018-01-01", freq="D", periods=4),
        ],
    )
    def test_boolean_cmp(self, values: List[Any]) -> None: ...
    @pytest.mark.parametrize("index", ["string"], indirect=True)
    @pytest.mark.parametrize("name,level", [(None, 0), ("a", "a")])
    def test_get_level_values(
        self, index: Index, name: Optional[str], level: Any
    ) -> None: ...
    def test_slice_keep_name(self) -> None: ...
    def test_slice_is_unique(self) -> None: ...
    def test_slice_is_montonic(self) -> None: ...
    @pytest.mark.parametrize(
        "index",
        [
            "string",
            "datetime",
            "int64",
            "int32",
            "uint64",
            "uint32",
            "float64",
            "float32",
        ],
        indirect=True,
    )
    def test_join_self(self, index: Index, join_type: str) -> None: ...
    @pytest.mark.parametrize("method", ["strip", "rstrip", "lstrip"])
    def test_str_attribute(self, method: str) -> None: ...
    @pytest.mark.parametrize(
        "index",
        [
            Index(range(5)),
            date_range("2020-01-01", periods=10),
            MultiIndex.from_tuples([("foo",