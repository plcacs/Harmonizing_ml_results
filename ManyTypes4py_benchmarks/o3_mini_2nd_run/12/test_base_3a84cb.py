#!/usr/bin/env python3
from collections import defaultdict
from datetime import datetime
from functools import partial
import math
import operator
import re
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import (
    is_any_real_numeric_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import pandas as pd
from pandas import (
    CategoricalIndex,
    DataFrame,
    DatetimeIndex,
    IntervalIndex,
    PeriodIndex,
    RangeIndex,
    Series,
    TimedeltaIndex,
    date_range,
    period_range,
    timedelta_range,
)
import pandas._testing as tm
from pandas.core.indexes.api import Index, MultiIndex, _get_combined_index, ensure_index, ensure_index_from_sequences

IndexT = pd.Index

class TestIndex:

    @pytest.fixture
    def simple_index(self) -> IndexT:
        return Index(list("abcde"))

    def test_can_hold_identifiers(self, simple_index: IndexT) -> None:
        index: IndexT = simple_index
        key: Any = index[0]
        assert index._can_hold_identifiers_and_holds_name(key) is True

    @pytest.mark.parametrize("index", ["datetime"], indirect=True)
    def test_new_axis(self, index: IndexT) -> None:
        with pytest.raises(ValueError, match="Multi-dimensional indexing"):
            index[None, :]

    def test_constructor_regular(self, index: IndexT) -> None:
        tm.assert_contains_all(index, index)

    @pytest.mark.parametrize("index", ["string"], indirect=True)
    def test_constructor_casting(self, index: IndexT) -> None:
        arr: np.ndarray = np.array(index)
        new_index: IndexT = Index(arr)
        tm.assert_contains_all(arr, new_index)
        tm.assert_index_equal(index, new_index)

    def test_constructor_copy(self, using_infer_string: bool) -> None:
        index: IndexT = Index(list("abc"), name="name")
        arr: np.ndarray = np.array(index)
        new_index: IndexT = Index(arr, copy=True, name="name")
        assert isinstance(new_index, Index)
        assert new_index.name == "name"
        if using_infer_string:
            tm.assert_extension_array_equal(new_index.values, pd.array(arr, dtype="str"))
        else:
            tm.assert_numpy_array_equal(arr, new_index.values)
        arr[0] = "SOMEBIGLONGSTRING"
        assert new_index[0] != "SOMEBIGLONGSTRING"

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
    def test_constructor_from_index_dtlike(self, cast_as_obj: bool, index: IndexT) -> None:
        if cast_as_obj:
            result: IndexT = Index(index.astype(object))
            assert result.dtype == np.dtype(object)
            if isinstance(index, DatetimeIndex):
                index += pd.Timedelta(nanoseconds=50)
                result = Index(index, dtype=object)
                assert result.dtype == np.object_
                assert list(result) == list(index)
        else:
            result = Index(index)
            tm.assert_index_equal(result, index)

    @pytest.mark.parametrize(
        "index,has_tz",
        [
            (date_range("2015-01-01 10:00", freq="D", periods=3, tz="US/Eastern"), True),
            (timedelta_range("1 days", freq="D", periods=3), False),
            (period_range("2015-01-01", freq="D", periods=3), False),
        ],
    )
    def test_constructor_from_series_dtlike(self, index: IndexT, has_tz: bool) -> None:
        result: IndexT = Index(Series(index))
        tm.assert_index_equal(result, index)
        if has_tz:
            assert result.tz == index.tz

    def test_constructor_from_series_freq(self) -> None:
        dts: List[str] = ["1-1-1990", "2-1-1990", "3-1-1990", "4-1-1990", "5-1-1990"]
        expected: DatetimeIndex = DatetimeIndex(dts, freq="MS")
        s: Series = Series(pd.to_datetime(dts))
        result: DatetimeIndex = DatetimeIndex(s, freq="MS")
        tm.assert_index_equal(result, expected)

    def test_constructor_from_frame_series_freq(self, using_infer_string: bool) -> None:
        dts: List[str] = ["1-1-1990", "2-1-1990", "3-1-1990", "4-1-1990", "5-1-1990"]
        expected: DatetimeIndex = DatetimeIndex(dts, freq="MS")
        df: DataFrame = DataFrame(np.random.default_rng(2).random((5, 3)))
        df["date"] = dts
        result: DatetimeIndex = DatetimeIndex(df["date"], freq="MS")
        dtype: Union[type, str] = object if not using_infer_string else "str"
        assert df["date"].dtype == dtype
        expected.name = "date"
        tm.assert_index_equal(result, expected)
        expected_series: Series = Series(dts, name="date")
        tm.assert_series_equal(df["date"], expected_series)
        if not using_infer_string:
            freq: Optional[str] = pd.infer_freq(df["date"])
            assert freq == "MS"

    def test_constructor_int_dtype_nan(self) -> None:
        data: List[float] = [np.nan]
        expected: IndexT = Index(data, dtype=np.float64)
        result: IndexT = Index(data, dtype="float")
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("klass,dtype,na_val", [(Index, np.float64, np.nan), (DatetimeIndex, "datetime64[s]", pd.NaT)])
    def test_index_ctor_infer_nan_nat(self, klass: Callable, dtype: Any, na_val: Any) -> None:
        na_list: List[Any] = [na_val, na_val]
        expected: IndexT = klass(na_list)
        assert expected.dtype == dtype
        result: IndexT = Index(na_list)
        tm.assert_index_equal(result, expected)
        result = Index(np.array(na_list))
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "vals,dtype", [([1, 2, 3, 4, 5], "int"), ([1.1, np.nan, 2.2, 3.0], "float"), (["A", "B", "C", np.nan], "obj")]
    )
    def test_constructor_simple_new(self, vals: List[Any], dtype: str) -> None:
        index: IndexT = Index(vals, name=dtype)
        result: IndexT = index._simple_new(index.values, dtype)
        tm.assert_index_equal(result, index)

    @pytest.mark.parametrize("attr", ["values", "asi8"])
    @pytest.mark.parametrize("klass", [Index, DatetimeIndex])
    def test_constructor_dtypes_datetime(self, tz_naive_fixture: Optional[str], attr: str, klass: Callable) -> None:
        index: DatetimeIndex = date_range("2011-01-01", periods=5)
        arg: Any = getattr(index, attr)
        index = index.tz_localize(tz_naive_fixture)
        dtype = index.dtype
        err: bool = tz_naive_fixture is not None
        msg: str = "Cannot use .astype to convert from timezone-naive dtype to"
        if attr == "asi8":
            result: DatetimeIndex = DatetimeIndex(arg).tz_localize(tz_naive_fixture)
            tm.assert_index_equal(result, index)
        elif klass is Index:
            with pytest.raises(TypeError, match="unexpected keyword"):
                klass(arg, tz=tz_naive_fixture)
        else:
            result = klass(arg, tz=tz_naive_fixture)
            tm.assert_index_equal(result, index)
        if attr == "asi8":
            if err:
                with pytest.raises(TypeError, match=msg):
                    DatetimeIndex(arg).astype(dtype)
            else:
                result = DatetimeIndex(arg).astype(dtype)
                tm.assert_index_equal(result, index)
        else:
            result = klass(arg, dtype=dtype)
            tm.assert_index_equal(result, index)
        if attr == "asi8":
            result = DatetimeIndex(list(arg)).tz_localize(tz_naive_fixture)
            tm.assert_index_equal(result, index)
        elif klass is Index:
            with pytest.raises(TypeError, match="unexpected keyword"):
                klass(arg, tz=tz_naive_fixture)
        else:
            result = klass(list(arg), tz=tz_naive_fixture)
            tm.assert_index_equal(result, index)
        if attr == "asi8":
            if err:
                with pytest.raises(TypeError, match=msg):
                    DatetimeIndex(list(arg)).astype(dtype)
            else:
                result = DatetimeIndex(list(arg)).astype(dtype)
                tm.assert_index_equal(result, index)
        else:
            result = klass(list(arg), dtype=dtype)
            tm.assert_index_equal(result, index)

    @pytest.mark.parametrize("attr", ["values", "asi8"])
    @pytest.mark.parametrize("klass", [Index, TimedeltaIndex])
    def test_constructor_dtypes_timedelta(self, attr: str, klass: Callable) -> None:
        index: TimedeltaIndex = timedelta_range("1 days", periods=5)
        index = index._with_freq(None)
        dtype = index.dtype
        values: Any = getattr(index, attr)
        result = klass(values, dtype=dtype)
        tm.assert_index_equal(result, index)
        result = klass(list(values), dtype=dtype)
        tm.assert_index_equal(result, index)

    @pytest.mark.parametrize("value", [[], iter([]), (_ for _ in [])])
    @pytest.mark.parametrize("klass", [Index, CategoricalIndex, DatetimeIndex, TimedeltaIndex])
    def test_constructor_empty(self, value: Any, klass: Callable) -> None:
        empty: IndexT = klass(value)
        assert isinstance(empty, klass)
        assert not len(empty)

    @pytest.mark.parametrize(
        "empty,klass",
        [
            (PeriodIndex([], freq="D"), PeriodIndex),
            (PeriodIndex(iter([]), freq="D"), PeriodIndex),
            (PeriodIndex((_ for _ in []), freq="D"), PeriodIndex),
            (RangeIndex(step=1), RangeIndex),
            (
                MultiIndex(levels=[[1, 2], ["blue", "red"]], codes=[[], []]),
                MultiIndex,
            ),
        ],
    )
    def test_constructor_empty_special(self, empty: IndexT, klass: Callable) -> None:
        assert isinstance(empty, klass)
        assert not len(empty)

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
    def test_view_with_args(self, index: IndexT) -> None:
        index.view("i8")

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
    def test_view_with_args_object_array_raises(self, index: IndexT) -> None:
        if index.dtype == bool:
            msg: str = "When changing to a larger dtype"
            with pytest.raises(ValueError, match=msg):
                index.view("i8")
        else:
            msg = (
                "Cannot change data-type for array of references\\.|"
                "Cannot change data-type for object array\\.|"
                "Cannot change data-type for array of strings\\.|"
            )
            with pytest.raises(TypeError, match=msg):
                index.view("i8")

    @pytest.mark.parametrize("index", ["int64", "int32", "range"], indirect=True)
    def test_astype(self, index: IndexT) -> None:
        casted: IndexT = index.astype("i8")
        casted.get_loc(5)
        index.name = "foobar"
        casted = index.astype("i8")
        assert casted.name == "foobar"

    def test_equals_object(self) -> None:
        assert Index(["a", "b", "c"]).equals(Index(["a", "b", "c"]))

    @pytest.mark.parametrize("comp", [Index(["a", "b"]), Index(["a", "b", "d"]), ["a", "b", "c"]])
    def test_not_equals_object(self, comp: Any) -> None:
        assert not Index(["a", "b", "c"]).equals(comp)

    def test_identical(self) -> None:
        i1: IndexT = Index(["a", "b", "c"])
        i2: IndexT = Index(["a", "b", "c"])
        assert i1.identical(i2)
        i1 = i1.rename("foo")
        assert i1.equals(i2)
        assert not i1.identical(i2)
        i2 = i2.rename("foo")
        assert i1.identical(i2)
        i3: IndexT = Index([("a", "a"), ("a", "b"), ("b", "a")])
        i4: IndexT = Index([("a", "a"), ("a", "b"), ("b", "a")], tupleize_cols=False)
        assert not i3.identical(i4)

    def test_is_(self) -> None:
        ind: IndexT = Index(range(10))
        assert ind.is_(ind)
        assert ind.is_(ind.view().view().view().view())
        assert not ind.is_(Index(range(10)))
        assert not ind.is_(ind.copy())
        assert not ind.is_(ind.copy(deep=False))
        assert not ind.is_(ind[:])
        assert not ind.is_(np.array(range(10)))
        assert ind.is_(ind.view())
        ind2: IndexT = ind.view()
        ind2.name = "bob"
        assert ind.is_(ind2)
        assert ind2.is_(ind)
        assert not ind.is_(Index(ind.values))
        arr: np.ndarray = np.array(range(1, 11))
        ind1: IndexT = Index(arr, copy=False)
        ind2 = Index(arr, copy=False)
        assert not ind1.is_(ind2)

    def test_asof_numeric_vs_bool_raises(self) -> None:
        left: IndexT = Index([1, 2, 3])
        right: IndexT = Index([True, False], dtype=object)
        msg: str = "Cannot compare dtypes int64 and bool"
        with pytest.raises(TypeError, match=msg):
            left.asof(right[0])
        with pytest.raises(InvalidIndexError, match=re.escape(str(right))):
            left.asof(right)
        with pytest.raises(InvalidIndexError, match=re.escape(str(left))):
            right.asof(left)

    @pytest.mark.parametrize("index", ["string"], indirect=True)
    def test_booleanindex(self, index: IndexT) -> None:
        bool_index: np.ndarray = np.ones(len(index), dtype=bool)
        bool_index[5:30:2] = False
        sub_index: IndexT = index[bool_index]
        for i, val in enumerate(sub_index):
            assert sub_index.get_loc(val) == i
        sub_index = index[list(bool_index)]
        for i, val in enumerate(sub_index):
            assert sub_index.get_loc(val) == i

    def test_fancy(self, simple_index: IndexT) -> None:
        index: IndexT = simple_index
        sl: IndexT = index[[1, 2, 3]]
        for i in sl:
            assert i == sl[sl.get_loc(i)]

    @pytest.mark.parametrize("index", ["string", "int64", "int32", "uint64", "uint32", "float64", "float32"], indirect=True)
    @pytest.mark.parametrize("dtype", [int, np.bool_])
    def test_empty_fancy(self, index: IndexT, dtype: Any, request: Any, using_infer_string: bool) -> None:
        if dtype is np.bool_ and using_infer_string and (index.dtype == "string"):
            request.applymarker(pytest.mark.xfail(reason="numpy behavior is buggy"))
        empty_arr: np.ndarray = np.array([], dtype=dtype)
        empty_index: IndexT = type(index)([], dtype=index.dtype)
        assert index[[]].identical(empty_index)
        if dtype == np.bool_:
            with pytest.raises(ValueError, match="length of the boolean indexer"):
                _ = index[empty_arr]
        else:
            assert index[empty_arr].identical(empty_index)

    @pytest.mark.parametrize("index", ["string", "int64", "int32", "uint64", "uint32", "float64", "float32"], indirect=True)
    def test_empty_fancy_raises(self, index: IndexT) -> None:
        empty_farr: np.ndarray = np.array([], dtype=np.float64)
        empty_index: IndexT = type(index)([], dtype=index.dtype)
        assert index[[]].identical(empty_index)
        msg: str = "arrays used as indices must be of integer"
        with pytest.raises(IndexError, match=msg):
            _ = index[empty_farr]

    def test_union_dt_as_obj(self, simple_index: IndexT) -> None:
        index: IndexT = simple_index
        date_index: DatetimeIndex = date_range("2019-01-01", periods=10)
        first_cat: IndexT = index.union(date_index)
        second_cat: IndexT = index.union(index)
        appended: IndexT = Index(np.append(index, date_index.astype("O")))
        tm.assert_index_equal(first_cat, appended)
        tm.assert_index_equal(second_cat, index)
        tm.assert_contains_all(index, first_cat)
        tm.assert_contains_all(index, second_cat)
        tm.assert_contains_all(date_index, first_cat)

    def test_map_with_tuples(self) -> None:
        index: IndexT = Index(np.arange(3), dtype=np.int64)
        result: IndexT = index.map(lambda x: (x,))
        expected: IndexT = Index([(i,) for i in index])
        tm.assert_index_equal(result, expected)
        result = index.map(lambda x: (x, x == 1))
        expected = MultiIndex.from_tuples([(i, i == 1) for i in index])
        tm.assert_index_equal(result, expected)

    def test_map_with_tuples_mi(self) -> None:
        first_level: List[str] = ["foo", "bar", "baz"]
        multi_index: MultiIndex = MultiIndex.from_tuples(zip(first_level, [1, 2, 3]))
        reduced_index: IndexT = multi_index.map(lambda x: x[0])
        tm.assert_index_equal(reduced_index, Index(first_level))

    @pytest.mark.parametrize(
        "index",
        [
            date_range("2020-01-01", freq="D", periods=10),
            period_range("2020-01-01", freq="D", periods=10),
            timedelta_range("1 day", periods=10),
        ],
    )
    def test_map_tseries_indices_return_index(self, index: IndexT) -> None:
        expected: IndexT = Index([1] * 10)
        result: IndexT = index.map(lambda x: 1)
        tm.assert_index_equal(expected, result)

    def test_map_tseries_indices_accsr_return_index(self) -> None:
        date_index: DatetimeIndex = DatetimeIndex(date_range("2020-01-01", periods=24, freq="h"), name="hourly")
        result: IndexT = date_index.map(lambda x: x.hour)
        expected: IndexT = Index(np.arange(24, dtype="int64"), name="hourly")
        tm.assert_index_equal(result, expected, exact=True)

    @pytest.mark.parametrize(
        "mapper", [
            lambda values, index: {i: e for e, i in zip(values, index)},
            lambda values, index: Series(values, index)
        ]
    )
    def test_map_dictlike_simple(self, mapper: Callable, ) -> None:
        expected: IndexT = Index(["foo", "bar", "baz"])
        index: IndexT = Index(np.arange(3), dtype=np.int64)
        result: IndexT = index.map(mapper(expected.values, index))
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "mapper", [
            lambda values, index: {i: e for e, i in zip(values, index)},
            lambda values, index: Series(values, index)
        ]
    )
    @pytest.mark.filterwarnings("ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning")
    def test_map_dictlike(self, index: IndexT, mapper: Callable, request: Any) -> None:
        if isinstance(index, CategoricalIndex):
            pytest.skip("Tested in test_categorical")
        elif not index.is_unique:
            pytest.skip("Cannot map duplicated index")
        rng: np.ndarray = np.arange(len(index), 0, -1, dtype=np.int64)
        if index.empty:
            expected: IndexT = Index([])
        elif is_numeric_dtype(index.dtype):
            expected = index._constructor(rng, dtype=index.dtype)
        elif type(index) is Index and index.dtype != object:
            expected = Index(rng, dtype=index.dtype)
        else:
            expected = Index(rng)
        result: IndexT = index.map(mapper(expected, index))
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("mapper", [Series(["foo", 2.0, "baz"], index=[0, 2, -1]), {0: "foo", 2: 2.0, -1: "baz"}])
    def test_map_with_non_function_missing_values(self, mapper: Union[Series, Dict[Any, Any]]) -> None:
        expected: IndexT = Index([2.0, np.nan, "foo"])
        result: IndexT = Index([2, 1, 0]).map(mapper)
        tm.assert_index_equal(expected, result)

    def test_map_na_exclusion(self) -> None:
        index: IndexT = Index([1.5, np.nan, 3, np.nan, 5])
        result: IndexT = index.map(lambda x: x * 2, na_action="ignore")
        expected: IndexT = index * 2
        tm.assert_index_equal(result, expected)

    def test_map_defaultdict(self) -> None:
        index: IndexT = Index([1, 2, 3])
        default_dict: Dict[Any, str] = defaultdict(lambda: "blank")
        default_dict[1] = "stuff"
        result: IndexT = index.map(default_dict)
        expected: IndexT = Index(["stuff", "blank", "blank"])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("name,expected", [("foo", "foo"), ("bar", None)])
    def test_append_empty_preserve_name(self, name: Optional[str], expected: Optional[str]) -> None:
        left: IndexT = Index([], name="foo")
        right: IndexT = Index([1, 2, 3], name=name)
        result: IndexT = left.append(right)
        assert result.name == expected

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
    def test_is_numeric(self, index: IndexT, expected: bool) -> None:
        assert is_any_real_numeric_dtype(index) is expected

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
    def test_is_object(self, index: IndexT, expected: bool, using_infer_string: bool) -> None:
        if using_infer_string and index.dtype == "string" and expected:
            expected = False  # type: ignore[assignment]
        assert is_object_dtype(index) is expected

    def test_summary(self, index: IndexT) -> None:
        index._summary()

    def test_logical_compat(self, all_boolean_reductions: str, simple_index: IndexT) -> None:
        index: IndexT = simple_index
        left: Any = getattr(index, all_boolean_reductions)()
        assert left == getattr(index.values, all_boolean_reductions)()
        right: Any = getattr(index.to_series(), all_boolean_reductions)()
        assert bool(left) == bool(right)

    @pytest.mark.parametrize("index", ["string", "int64", "int32", "float64", "float32"], indirect=True)
    def test_drop_by_str_label(self, index: IndexT) -> None:
        n: int = len(index)
        drop: IndexT = index[list(range(5, 10))]
        dropped: IndexT = index.drop(drop)
        expected: IndexT = index[list(range(5)) + list(range(10, n))]
        tm.assert_index_equal(dropped, expected)
        dropped = index.drop(index[0])
        expected = index[1:]
        tm.assert_index_equal(dropped, expected)

    @pytest.mark.parametrize("index", ["string", "int64", "int32", "float64", "float32"], indirect=True)
    @pytest.mark.parametrize("keys", [["foo", "bar"], ["1", "bar"]])
    def test_drop_by_str_label_raises(self, index: IndexT, keys: List[Any]) -> None:
        with pytest.raises(KeyError, match=""):
            index.drop(keys)

    @pytest.mark.parametrize("index", ["string", "int64", "int32", "float64", "float32"], indirect=True)
    def test_drop_by_str_label_errors_ignore(self, index: IndexT) -> None:
        n: int = len(index)
        drop: IndexT = index[list(range(5, 10))]
        mixed: List[Any] = drop.tolist() + ["foo"]
        dropped: IndexT = index.drop(mixed, errors="ignore")
        expected: IndexT = index[list(range(5)) + list(range(10, n))]
        tm.assert_index_equal(dropped, expected)
        dropped = index.drop(["foo", "bar"], errors="ignore")
        expected = index[list(range(n))]
        tm.assert_index_equal(dropped, expected)

    def test_drop_by_numeric_label_loc(self) -> None:
        index: IndexT = Index([1, 2, 3])
        dropped: IndexT = index.drop(1)
        expected: IndexT = Index([2, 3])
        tm.assert_index_equal(dropped, expected)

    def test_drop_by_numeric_label_raises(self) -> None:
        index: IndexT = Index([1, 2, 3])
        with pytest.raises(KeyError, match=""):
            index.drop([3, 4])

    @pytest.mark.parametrize("key,expected", [(4, Index([1, 2, 3])), ([3, 4, 5], Index([1, 2]))])
    def test_drop_by_numeric_label_errors_ignore(self, key: Union[int, List[int]], expected: IndexT) -> None:
        index: IndexT = Index([1, 2, 3])
        dropped: IndexT = index.drop(key, errors="ignore")
        tm.assert_index_equal(dropped, expected)

    @pytest.mark.parametrize("values", [["a", "b", ("c", "d")], ["a", ("c", "d"), "b"], [("c", "d"), "a", "b"]])
    @pytest.mark.parametrize("to_drop", [[("c", "d"), "a"], ["a", ("c", "d")]])
    def test_drop_tuple(self, values: List[Any], to_drop: List[Any]) -> None:
        index: IndexT = Index(values)
        expected: IndexT = Index(["b"], dtype=object)
        result: IndexT = index.drop(to_drop)
        tm.assert_index_equal(result, expected)
        removed: IndexT = index.drop(to_drop[0])
        for drop_me in (to_drop[1], [to_drop[1]]):
            result = removed.drop(drop_me)
            tm.assert_index_equal(result, expected)
        removed = index.drop(to_drop[1])
        msg: str = f'\\"\\[{re.escape(to_drop[1].__repr__())}\\] not found in axis\\"'
        for drop_me in (to_drop[1], [to_drop[1]]):
            with pytest.raises(KeyError, match=msg):
                removed.drop(drop_me)

    @pytest.mark.filterwarnings("ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning")
    def test_drop_with_duplicates_in_index(self, index: IndexT) -> None:
        if len(index) == 0 or isinstance(index, MultiIndex):
            pytest.skip("Test doesn't make sense for empty MultiIndex")
        if isinstance(index, IntervalIndex) and (not IS64):
            pytest.skip("Cannot test IntervalIndex with int64 dtype on 32 bit platform")
        index = index.unique().repeat(2)
        expected: IndexT = index[2:]
        result: IndexT = index.drop(index[0])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("attr", ["is_monotonic_increasing", "is_monotonic_decreasing", "_is_strictly_monotonic_increasing", "_is_strictly_monotonic_decreasing"])
    def test_is_monotonic_incomparable(self, attr: str) -> None:
        index: IndexT = Index([5, datetime.now(), 7])
        assert not getattr(index, attr)

    @pytest.mark.parametrize("values", [["foo", "bar", "quux"], {"foo", "bar", "quux"}])
    @pytest.mark.parametrize("index,expected", [(["qux", "baz", "foo", "bar"], [False, False, True, True]), ([], [])])
    def test_isin(self, values: Union[List[Any], set], index: Sequence[Any], expected: Sequence[bool]) -> None:
        index_obj: IndexT = Index(index)
        result: np.ndarray = index_obj.isin(values)
        expected_arr: np.ndarray = np.array(expected, dtype=bool)
        tm.assert_numpy_array_equal(result, expected_arr)

    def test_isin_nan_common_object(self, nulls_fixture: Any, nulls_fixture2: Any, using_infer_string: bool) -> None:
        idx: IndexT = Index(["a", nulls_fixture])
        if isinstance(nulls_fixture, float) and isinstance(nulls_fixture2, float) and math.isnan(nulls_fixture) and math.isnan(nulls_fixture2):
            tm.assert_numpy_array_equal(idx.isin([nulls_fixture2]), np.array([False, True]))
        elif nulls_fixture is nulls_fixture2:
            tm.assert_numpy_array_equal(idx.isin([nulls_fixture2]), np.array([False, True]))
        elif using_infer_string and idx.dtype == "string":
            tm.assert_numpy_array_equal(idx.isin([nulls_fixture2]), np.array([False, True]))
        else:
            tm.assert_numpy_array_equal(idx.isin([nulls_fixture2]), np.array([False, False]))

    def test_isin_nan_common_float64(self, nulls_fixture: Any, float_numpy_dtype: Any) -> None:
        dtype: Any = float_numpy_dtype
        if nulls_fixture is pd.NaT or nulls_fixture is pd.NA:
            msg: str = f'float\\(\\) argument must be a string or a (real )?number, not {type(nulls_fixture).__name__!r}'
            with pytest.raises(TypeError, match=msg):
                Index([1.0, nulls_fixture], dtype=dtype)
            idx: IndexT = Index([1.0, np.nan], dtype=dtype)
            assert not idx.isin([nulls_fixture]).any()
            return
        idx: IndexT = Index([1.0, nulls_fixture], dtype=dtype)
        res: np.ndarray = idx.isin([np.nan])
        tm.assert_numpy_array_equal(res, np.array([False, True]))
        res = idx.isin([pd.NaT])
        tm.assert_numpy_array_equal(res, np.array([False, False]))

    @pytest.mark.parametrize("level", [0, -1])
    @pytest.mark.parametrize(
        "index",
        [(["qux", "baz", "foo", "bar"]), np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)],
    )
    def test_isin_level_kwarg(self, level: Union[int, str], index: Union[List[Any], np.ndarray]) -> None:
        index_obj: IndexT = Index(index)
        values: List[Any] = index_obj.tolist()[-2:] + ["nonexisting"]
        expected: np.ndarray = np.array([False, False, True, True])
        tm.assert_numpy_array_equal(expected, index_obj.isin(values, level=level))
        index_obj.name = "foobar"
        tm.assert_numpy_array_equal(expected, index_obj.isin(values, level="foobar"))

    def test_isin_level_kwarg_bad_level_raises(self, index: IndexT) -> None:
        for level in [10, index.nlevels, -(index.nlevels + 1)]:
            with pytest.raises(IndexError, match="Too many levels"):
                index.isin([], level=level)

    @pytest.mark.parametrize("label", [1.0, "foobar", "xyzzy", np.nan])
    def test_isin_level_kwarg_bad_label_raises(self, label: Any, index: IndexT) -> None:
        if isinstance(index, MultiIndex):
            index = index.rename(["foo", "bar"] + index.names[2:])
            msg: str = f"'Level {label} not found'"
        else:
            index = index.rename("foo")
            msg = f"Requested level \\({label}\\) does not match index name \\(foo\\)"
        with pytest.raises(KeyError, match=msg):
            index.isin([], level=label)

    @pytest.mark.parametrize("empty", [[], Series(dtype=object), np.array([])])
    def test_isin_empty(self, empty: Any) -> None:
        index: IndexT = Index(["a", "b"])
        expected: np.ndarray = np.array([False, False])
        result: np.ndarray = index.isin(empty)
        tm.assert_numpy_array_equal(expected, result)

    def test_isin_string_null(self, string_dtype_no_object: Any) -> None:
        index: IndexT = Index(["a", "b"], dtype=string_dtype_no_object)
        result: np.ndarray = index.isin([None])
        expected: np.ndarray = np.array([False, False])
        tm.assert_numpy_array_equal(result, expected)

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
    def test_boolean_cmp(self, values: Any) -> None:
        index: IndexT = Index(values)
        result: np.ndarray = index == values
        expected: np.ndarray = np.array([True, True, True, True], dtype=bool)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("index", ["string"], indirect=True)
    @pytest.mark.parametrize("name,level", [(None, 0), ("a", "a")])
    def test_get_level_values(self, index: IndexT, name: Optional[str], level: Union[int, str]) -> None:
        expected: IndexT = index.copy()
        if name:
            expected.name = name
        result: IndexT = expected.get_level_values(level)
        tm.assert_index_equal(result, expected)

    def test_slice_keep_name(self) -> None:
        index: IndexT = Index(["a", "b"], name="asdf")
        assert index.name == index[1:].name

    def test_slice_is_unique(self) -> None:
        index: IndexT = Index([1, 1, 2, 3, 4])
        assert not index.is_unique
        filtered_index: IndexT = index[2:].copy()
        assert filtered_index.is_unique

    def test_slice_is_montonic(self) -> None:
        """Test that is_monotonic_decreasing is correct on slices."""
        index: IndexT = Index([1, 2, 3, 3])
        assert not index.is_monotonic_decreasing
        filtered_index: IndexT = index[2:].copy()
        assert filtered_index.is_monotonic_decreasing
        assert filtered_index.is_monotonic_increasing
        filtered_index = index[1:].copy()
        assert not filtered_index.is_monotonic_decreasing
        assert filtered_index.is_monotonic_increasing
        filtered_index = index[:].copy()
        assert not filtered_index.is_monotonic_decreasing
        assert filtered_index.is_monotonic_increasing

    @pytest.mark.parametrize(
        "index",
        ["string", "datetime", "int64", "int32", "uint64", "uint32", "float64", "float32"],
        indirect=True,
    )
    def test_join_self(self, index: IndexT, join_type: str) -> None:
        result: IndexT = index.join(index, how=join_type)
        expected: IndexT = index
        if join_type == "outer":
            expected = expected.sort_values()
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("method", ["strip", "rstrip", "lstrip"])
    def test_str_attribute(self, method: str) -> None:
        index: IndexT = Index([" jack", "jill ", " jesse ", "frank"])
        expected: IndexT = Index([getattr(str, method)(x) for x in index.values])
        result: IndexT = getattr(index.str, method)()
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "index",
        [
            Index(range(5)),
            date_range("2020-01-01", periods=10),
            MultiIndex.from_tuples([("foo", "1"), ("bar", "3")]),
            period_range(start="2000", end="2010", freq="Y"),
        ],
    )
    def test_str_attribute_raises(self, index: IndexT) -> None:
        with pytest.raises(AttributeError, match="only use .str accessor"):
            index.str.repeat(2)

    @pytest.mark.parametrize(
        "expand,expected",
        [
            (None, Index([["a", "b", "c"], ["d", "e"], ["f"]])),
            (False, Index([["a", "b", "c"], ["d", "e"], ["f"]])),
            (True, MultiIndex.from_tuples([("a", "b", "c"), ("d", "e", np.nan), ("f", np.nan, np.nan)])),
        ],
    )
    def test_str_split(self, expand: Optional[bool], expected: IndexT) -> None:
        index: IndexT = Index(["a b c", "d e", "f"])
        if expand is not None:
            result: Any = index.str.split(expand=expand)
        else:
            result = index.str.split()
        tm.assert_index_equal(result, expected)

    def test_str_bool_return(self) -> None:
        index: IndexT = Index(["a1", "a2", "b1", "b2"])
        result: np.ndarray = index.str.startswith("a")
        expected: np.ndarray = np.array([True, True, False, False])
        tm.assert_numpy_array_equal(result, expected)
        assert isinstance(result, np.ndarray)

    def test_str_bool_series_indexing(self) -> None:
        index: IndexT = Index(["a1", "a2", "b1", "b2"])
        s: Series = Series(range(4), index=index)
        result: Series = s[s.index.str.startswith("a")]
        expected: Series = Series(range(2), index=["a1", "a2"])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("index,expected", [(["abcd"], True), (range(4), False)])
    def test_tab_completion(self, index: Any, expected: bool) -> None:
        index_obj: IndexT = Index(index)
        result: bool = "str" in dir(index_obj)
        assert result == expected

    def test_indexing_doesnt_change_class(self) -> None:
        index: IndexT = Index([1, 2, 3, "a", "b", "c"])
        assert index[1:3].identical(Index([2, 3], dtype=np.object_))
        assert index[[0, 1]].identical(Index([1, 2], dtype=np.object_))

    def test_outer_join_sort(self) -> None:
        left_index: IndexT = Index(np.random.default_rng(2).permutation(15))
        right_index: DatetimeIndex = date_range("2020-01-01", periods=10)
        with tm.assert_produces_warning(RuntimeWarning, match="not supported between"):
            result: IndexT = left_index.join(right_index, how="outer")
        with tm.assert_produces_warning(RuntimeWarning, match="not supported between"):
            expected: IndexT = left_index.astype(object).union(right_index.astype(object))
        tm.assert_index_equal(result, expected)

    def test_take_fill_value(self) -> None:
        index: IndexT = Index(list("ABC"), name="xxx")
        result: IndexT = index.take(np.array([1, 0, -1]))
        expected: IndexT = Index(list("BAC"), name="xxx")
        tm.assert_index_equal(result, expected)
        result = index.take(np.array([1, 0, -1]), fill_value=True)
        expected = Index(["B", "A", np.nan], name="xxx")
        tm.assert_index_equal(result, expected)
        result = index.take(np.array([1, 0, -1]), allow_fill=False, fill_value=True)
        expected = Index(["B", "A", "C"], name="xxx")
        tm.assert_index_equal(result, expected)

    def test_take_fill_value_none_raises(self) -> None:
        index: IndexT = Index(list("ABC"), name="xxx")
        msg: str = "When allow_fill=True and fill_value is not None, all indices must be >= -1"
        with pytest.raises(ValueError, match=msg):
            index.take(np.array([1, 0, -2]), fill_value=True)
        with pytest.raises(ValueError, match=msg):
            index.take(np.array([1, 0, -5]), fill_value=True)

    def test_take_bad_bounds_raises(self) -> None:
        index: IndexT = Index(list("ABC"), name="xxx")
        with pytest.raises(IndexError, match="out of bounds"):
            index.take(np.array([1, -5]))

    @pytest.mark.parametrize(
        "name,labels",
        [
            (None, []),
            (None, np.array([])),
            (None, ["A", "B", "C"]),
            (None, ["C", "B", "A"]),
            (None, np.array(["A", "B", "C"])),
            (None, np.array(["C", "B", "A"])),
            (None, date_range("20130101", periods=3).values),
            (None, date_range("20130101", periods=3).tolist()),
        ],
    )
    def test_reindex_preserves_name_if_target_is_list_or_ndarray(self, name: Optional[Any], labels: Any) -> None:
        index: IndexT = Index([0, 1, 2])
        index.name = name
        result: Tuple[IndexT, Any] = index.reindex(labels)
        assert result[0].name == name

    @pytest.mark.parametrize("labels", [[], np.array([]), np.array([], dtype=np.int64)])
    def test_reindex_preserves_type_if_target_is_empty_list_or_array(self, labels: Any) -> None:
        index: IndexT = Index(list("abc"))
        assert index.reindex(labels)[0].dtype.type == index.dtype.type

    def test_reindex_doesnt_preserve_type_if_target_is_empty_index(self) -> None:
        index: IndexT = Index(list("abc"))
        labels: DatetimeIndex = DatetimeIndex([])
        dtype = np.datetime64
        assert index.reindex(labels)[0].dtype.type == dtype

    def test_reindex_doesnt_preserve_type_if_target_is_empty_index_numeric(self, any_real_numpy_dtype: Any) -> None:
        dtype: Any = any_real_numpy_dtype
        index: IndexT = Index(list("abc"))
        labels: IndexT = Index([], dtype=dtype)
        assert index.reindex(labels)[0].dtype == dtype

    def test_reindex_no_type_preserve_target_empty_mi(self) -> None:
        index: IndexT = Index(list("abc"))
        result: Index = index.reindex(MultiIndex([Index([], np.int64), Index([], np.float64)], [[], []]))[0]
        assert result.levels[0].dtype.type == np.int64
        assert result.levels[1].dtype.type == np.float64

    def test_reindex_ignoring_level(self) -> None:
        idx: IndexT = Index([1, 2, 3], name="x")
        idx2: IndexT = Index([1, 2, 3, 4], name="x")
        expected: IndexT = Index([1, 2, 3, 4], name="x")
        result, _ = idx.reindex(idx2, level="x")
        tm.assert_index_equal(result, expected)

    def test_groupby(self) -> None:
        index: IndexT = Index(range(5))
        result: Dict[Any, IndexT] = index.groupby(np.array([1, 1, 2, 2, 2]))
        expected: Dict[Any, IndexT] = {1: Index([0, 1]), 2: Index([2, 3, 4])}
        tm.assert_dict_equal(result, expected)

    @pytest.mark.parametrize(
        "mi,expected",
        [
            (MultiIndex.from_tuples([(1, 2), (4, 5)]), np.array([True, True])),
            (MultiIndex.from_tuples([(1, 2), (4, 6)]), np.array([True, False])),
        ],
    )
    def test_equals_op_multiindex(self, mi: MultiIndex, expected: np.ndarray) -> None:
        df: DataFrame = DataFrame([3, 6], columns=["c"], index=MultiIndex.from_arrays([[1, 4], [2, 5]], names=["a", "b"]))
        result: np.ndarray = df.index == mi
        tm.assert_numpy_array_equal(result, expected)

    def test_equals_op_multiindex_identify(self) -> None:
        df: DataFrame = DataFrame([3, 6], columns=["c"], index=MultiIndex.from_arrays([[1, 4], [2, 5]], names=["a", "b"]))
        result: np.ndarray = df.index == df.index
        expected: np.ndarray = np.array([True, True])
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("index", [MultiIndex.from_tuples([(1, 2), (4, 5), (8, 9)]), Index(["foo", "bar", "baz"])])
    def test_equals_op_mismatched_multiindex_raises(self, index: IndexT) -> None:
        df: DataFrame = DataFrame([3, 6], columns=["c"], index=MultiIndex.from_arrays([[1, 4], [2, 5]], names=["a", "b"]))
        with pytest.raises(ValueError, match="Lengths must match"):
            _ = df.index == index

    def test_equals_op_index_vs_mi_same_length(self, using_infer_string: bool) -> None:
        mi: MultiIndex = MultiIndex.from_tuples([(1, 2), (4, 5), (8, 9)])
        index: IndexT = Index(["foo", "bar", "baz"])
        result: np.ndarray = mi == index
        expected: np.ndarray = np.array([False, False, False])
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("dt_conv, arg", [(pd.to_datetime, ["2000-01-01", "2000-01-02"]), (pd.to_timedelta, ["01:02:03", "01:02:04"])])
    def test_dt_conversion_preserves_name(self, dt_conv: Callable, arg: List[str]) -> None:
        index: IndexT = Index(arg, name="label")
        assert index.name == dt_conv(index).name

    def test_cached_properties_not_settable(self) -> None:
        index: IndexT = Index([1, 2, 3])
        with pytest.raises(AttributeError, match="Can't set attribute"):
            index.is_unique = False

    def test_tab_complete_warning(self, ip: Any) -> None:
        pytest.importorskip("IPython", minversion="6.0.0")
        from IPython.core.completer import provisionalcompleter

        code: str = "import pandas as pd; idx = pd.Index([1, 2])"
        ip.run_cell(code)
        with tm.assert_produces_warning(None, raise_on_extra_warnings=False):
            with provisionalcompleter("ignore"):
                _ = list(ip.Completer.completions("idx.", 4))

    def test_contains_method_removed(self, index: IndexT) -> None:
        if isinstance(index, IntervalIndex):
            index.contains(1)
        else:
            msg: str = f"'{type(index).__name__}' object has no attribute 'contains'"
            with pytest.raises(AttributeError, match=msg):
                index.contains(1)

    def test_sortlevel(self) -> None:
        index: IndexT = Index([5, 4, 3, 2, 1])
        with pytest.raises(Exception, match="ascending must be a single bool value or"):
            index.sortlevel(ascending="True")
        with pytest.raises(Exception, match="ascending must be a list of bool values of length 1"):
            index.sortlevel(ascending=[True, True])
        with pytest.raises(Exception, match="ascending must be a bool value"):
            index.sortlevel(ascending=["True"])
        expected: IndexT = Index([1, 2, 3, 4, 5])
        result: Tuple[IndexT, Any] = index.sortlevel(ascending=[True])
        tm.assert_index_equal(result[0], expected)
        expected = Index([1, 2, 3, 4, 5])
        result = index.sortlevel(ascending=True)
        tm.assert_index_equal(result[0], expected)
        expected = Index([5, 4, 3, 2, 1])
        result = index.sortlevel(ascending=False)
        tm.assert_index_equal(result[0], expected)

    def test_sortlevel_na_position(self) -> None:
        idx: IndexT = Index([1, np.nan])
        result: Tuple[IndexT, Any] = idx.sortlevel(na_position="first")
        expected: IndexT = Index([np.nan, 1])
        tm.assert_index_equal(result[0], expected)

    @pytest.mark.parametrize("periods, expected_results", [(1, [np.nan, 10, 10, 10, 10]), (2, [np.nan, np.nan, 20, 20, 20]), (3, [np.nan, np.nan, np.nan, 30, 30])])
    def test_index_diff(self, periods: int, expected_results: List[Any]) -> None:
        idx: IndexT = Index([10, 20, 30, 40, 50])
        result: IndexT = idx.diff(periods)
        expected: IndexT = Index(expected_results)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("decimals, expected_results", [(0, [1.0, 2.0, 3.0]), (1, [1.2, 2.3, 3.5]), (2, [1.23, 2.35, 3.46])])
    def test_index_round(self, decimals: int, expected_results: List[float]) -> None:
        idx: IndexT = Index([1.234, 2.345, 3.456])
        result: IndexT = idx.round(decimals)
        expected: IndexT = Index(expected_results)
        tm.assert_index_equal(result, expected)

class TestMixedIntIndex:

    @pytest.fixture
    def simple_index(self) -> IndexT:
        return Index([0, "a", 1, "b", 2, "c"])

    def test_argsort(self, simple_index: IndexT) -> None:
        index: IndexT = simple_index
        with pytest.raises(TypeError, match="'>|<' not supported"):
            index.argsort()

    def test_numpy_argsort(self, simple_index: IndexT) -> None:
        index: IndexT = simple_index
        with pytest.raises(TypeError, match="'>|<' not supported"):
            np.argsort(index)

    def test_copy_name(self, simple_index: IndexT) -> None:
        index: IndexT = simple_index
        first: IndexT = type(index)(index, copy=True, name="mario")
        second: IndexT = type(first)(first, copy=False)
        assert first is not second
        tm.assert_index_equal(first, second)
        assert first.name == "mario"
        assert second.name == "mario"
        s1: Series = Series(2, index=first)
        s2: Series = Series(3, index=second[:-1])
        s3: Series = s1 * s2
        assert s3.index.name == "mario"

    def test_copy_name2(self) -> None:
        index: IndexT = Index([1, 2], name="MyName")
        index1: IndexT = index.copy()
        tm.assert_index_equal(index, index1)
        index2: IndexT = index.copy(name="NewName")
        tm.assert_index_equal(index, index2, check_names=False)
        assert index.name == "MyName"
        assert index2.name == "NewName"

    def test_unique_na(self) -> None:
        idx: IndexT = Index([2, np.nan, 2, 1], name="my_index")
        expected: IndexT = Index([2, np.nan, 1], name="my_index")
        result: IndexT = idx.unique()
        tm.assert_index_equal(result, expected)

    def test_logical_compat(self, simple_index: IndexT) -> None:
        index: IndexT = simple_index
        assert index.all() == index.values.all()
        assert index.any() == index.values.any()

    @pytest.mark.parametrize("how", ["any", "all"])
    @pytest.mark.parametrize("dtype", [None, object, "category"])
    @pytest.mark.parametrize(
        "vals,expected",
        [
            ([1, 2, 3], [1, 2, 3]),
            ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
            ([1.0, 2.0, np.nan, 3.0], [1.0, 2.0, 3.0]),
            (["A", "B", "C"], ["A", "B", "C"]),
            (["A", np.nan, "B", "C"], ["A", "B", "C"]),
        ],
    )
    def test_dropna(self, how: str, dtype: Optional[Any], vals: List[Any], expected: List[Any]) -> None:
        index: IndexT = Index(vals, dtype=dtype)
        result: IndexT = index.dropna(how=how)
        expected_index: IndexT = Index(expected, dtype=dtype)
        tm.assert_index_equal(result, expected_index)

    @pytest.mark.parametrize("how", ["any", "all"])
    @pytest.mark.parametrize(
        "index,expected",
        [
            (DatetimeIndex(["2011-01-01", "2011-01-02", "2011-01-03"]), DatetimeIndex(["2011-01-01", "2011-01-02", "2011-01-03"])),
            (DatetimeIndex(["2011-01-01", "2011-01-02", "2011-01-03", pd.NaT]), DatetimeIndex(["2011-01-01", "2011-01-02", "2011-01-03"])),
            (TimedeltaIndex(["1 days", "2 days", "3 days"]), TimedeltaIndex(["1 days", "2 days", "3 days"])),
            (TimedeltaIndex([pd.NaT, "1 days", "2 days", "3 days", pd.NaT]), TimedeltaIndex(["1 days", "2 days", "3 days"])),
            (PeriodIndex(["2012-02", "2012-04", "2012-05"], freq="M"), PeriodIndex(["2012-02", "2012-04", "2012-05"], freq="M")),
            (PeriodIndex(["2012-02", "2012-04", "NaT", "2012-05"], freq="M"), PeriodIndex(["2012-02", "2012-04", "2012-05"], freq="M")),
        ],
    )
    def test_dropna_dt_like(self, how: str, index: IndexT, expected: IndexT) -> None:
        result: IndexT = index.dropna(how=how)
        tm.assert_index_equal(result, expected)

    def test_dropna_invalid_how_raises(self) -> None:
        msg: str = "invalid how option: xxx"
        with pytest.raises(ValueError, match=msg):
            Index([1, 2, 3]).dropna(how="xxx")

    @pytest.mark.parametrize(
        "index",
        [
            Index([np.nan]),
            Index([np.nan, 1]),
            Index([1, 2, np.nan]),
            Index(["a", "b", np.nan]),
            pd.to_datetime(["NaT"]),
            pd.to_datetime(["NaT", "2000-01-01"]),
            pd.to_datetime(["2000-01-01", "NaT", "2000-01-02"]),
            pd.to_timedelta(["1 day", "NaT"]),
        ],
    )
    def test_is_monotonic_na(self, index: IndexT) -> None:
        assert index.is_monotonic_increasing is False
        assert index.is_monotonic_decreasing is False
        assert index._is_strictly_monotonic_increasing is False
        assert index._is_strictly_monotonic_decreasing is False

    @pytest.mark.parametrize("dtype", ["f8", "m8[ns]", "M8[us]"])
    @pytest.mark.parametrize("unique_first", [True, False])
    def test_is_monotonic_unique_na(self, dtype: Any, unique_first: bool) -> None:
        index: IndexT = Index([None, 1, 1], dtype=dtype)
        if unique_first:
            assert index.is_unique is False
            assert index.is_monotonic_increasing is False
            assert index.is_monotonic_decreasing is False
        else:
            assert index.is_monotonic_increasing is False
            assert index.is_monotonic_decreasing is False
            assert index.is_unique is False

    def test_int_name_format(self, frame_or_series: Callable) -> None:
        index: IndexT = Index(["a", "b", "c"], name=0)
        result: Any = frame_or_series(list(range(3)), index=index)
        assert "0" in repr(result)

    def test_str_to_bytes_raises(self) -> None:
        index: IndexT = Index([str(x) for x in range(10)])
        msg: str = "^'str' object cannot be interpreted as an integer$"
        with pytest.raises(TypeError, match=msg):
            bytes(index)

    @pytest.mark.filterwarnings("ignore:elementwise comparison failed:FutureWarning")
    def test_index_with_tuple_bool(self) -> None:
        idx: IndexT = Index([("a", "b"), ("b", "c"), ("c", "a")])
        result: np.ndarray = idx == ("c", "a")
        expected: np.ndarray = np.array([False, False, True])
        tm.assert_numpy_array_equal(result, expected)

class TestIndexUtils:

    @pytest.mark.parametrize(
        "data, names, expected",
        [
            ([[1, 2, 4]], None, Index([1, 2, 4])),
            ([[1, 2, 4]], ["name"], Index([1, 2, 4], name="name")),
            ([[1, 2, 3]], None, RangeIndex(1, 4)),
            ([[1, 2, 3]], ["name"], RangeIndex(1, 4, name="name")),
            ([["a", "a"], ["c", "d"]], None, MultiIndex([["a"], ["c", "d"]], [[0, 0], [0, 1]])),
            ([["a", "a"], ["c", "d"]], ["L1", "L2"], MultiIndex([["a"], ["c", "d"]], [[0, 0], [0, 1]], names=["L1", "L2"])),
        ],
    )
    def test_ensure_index_from_sequences(self, data: List[List[Any]], names: Optional[List[Any]], expected: IndexT) -> None:
        result: IndexT = ensure_index_from_sequences(data, names)
        tm.assert_index_equal(result, expected, exact=True)

    def test_ensure_index_mixed_closed_intervals(self) -> None:
        intervals: List[pd.Interval] = [
            pd.Interval(0, 1, closed="left"),
            pd.Interval(1, 2, closed="right"),
            pd.Interval(2, 3, closed="neither"),
            pd.Interval(3, 4, closed="both"),
        ]
        result: IndexT = ensure_index(intervals)
        expected: IndexT = Index(intervals, dtype=object)
        tm.assert_index_equal(result, expected)

    def test_ensure_index_uint64(self) -> None:
        values: List[int] = [0, np.iinfo(np.uint64).max]
        result: IndexT = ensure_index(values)
        assert list(result) == values
        expected: IndexT = Index(values, dtype="uint64")
        tm.assert_index_equal(result, expected)

    def test_get_combined_index(self) -> None:
        result: IndexT = _get_combined_index([])
        expected: RangeIndex = RangeIndex(0)
        tm.assert_index_equal(result, expected)

@pytest.mark.parametrize("opname", ["eq", "ne", "le", "lt", "ge", "gt", "add", "radd", "sub", "rsub", "mul", "rmul", "truediv", "rtruediv", "floordiv", "rfloordiv", "pow", "rpow", "mod", "divmod"])
def test_generated_op_names(opname: str, index: IndexT) -> None:
    opname_attr: str = f"__{opname}__"
    method: Callable = getattr(index, opname_attr)
    assert method.__name__ == opname_attr

@pytest.mark.parametrize(
    "klass, extra_kwargs",
    [
        [Index, {}],
        *[[lambda x, dtyp=dtyp: Index(x, dtype=dtyp), {}] for dtyp in tm.ALL_REAL_NUMPY_DTYPES],
        [DatetimeIndex, {}],
        [TimedeltaIndex, {}],
        [PeriodIndex, {"freq": "Y"}],
        [RangeIndex, {"start": list(range(1))}],
        [IntervalIndex, {"data": [pd.Interval(0, 1)]}],
        [lambda x: Index(x, dtype=object), {}],
        [lambda x: MultiIndex(levels=[1], codes=[0]), {}],
    ],
)
def test_index_subclass_constructor_wrong_kwargs(klass: Callable, extra_kwargs: Dict[str, Any]) -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        klass(foo="bar")

def test_deprecated_fastpath() -> None:
    msg: str = "[Uu]nexpected keyword argument"
    with pytest.raises(TypeError, match=msg):
        Index(np.array(["a", "b"], dtype=object), name="test", fastpath=True)
    with pytest.raises(TypeError, match=msg):
        Index(np.array([1, 2, 3], dtype="int64"), name="test", fastpath=True)
    with pytest.raises(TypeError, match=msg):
        RangeIndex(0, 5, 2, name="test", fastpath=True)
    with pytest.raises(TypeError, match=msg):
        CategoricalIndex(["a", "b", "c"], name="test", fastpath=True)

def test_shape_of_invalid_index() -> None:
    idx: IndexT = Index([0, 1, 2, 3])
    with pytest.raises(ValueError, match="Multi-dimensional indexing"):
        idx[:, None]

@pytest.mark.parametrize("dtype", [None, np.int64, np.uint64, np.float64])
def test_validate_1d_input(dtype: Optional[Any]) -> None:
    msg: str = "Index data must be 1-dimensional"
    arr: np.ndarray = np.arange(8).reshape(2, 2, 2)
    with pytest.raises(ValueError, match=msg):
        Index(arr, dtype=dtype)
    df: DataFrame = DataFrame(arr.reshape(4, 2))
    with pytest.raises(ValueError, match=msg):
        Index(df, dtype=dtype)
    ser: Series = Series(0, index=range(4))
    with pytest.raises(ValueError, match=msg):
        ser.index = np.array([[2, 3]] * 4, dtype=dtype)

@pytest.mark.parametrize(
    "klass, extra_kwargs",
    [
        [Index, {}],
        *[[lambda x, dtyp=dtyp: Index(x, dtype=dtyp), {}] for dtyp in tm.ALL_REAL_NUMPY_DTYPES],
        [DatetimeIndex, {}],
        [TimedeltaIndex, {}],
        [PeriodIndex, {"freq": "Y"}],
    ],
)
def test_construct_from_memoryview(klass: Callable, extra_kwargs: Dict[str, Any]) -> None:
    result: IndexT = klass(memoryview(np.arange(2000, 2005)), **extra_kwargs)
    expected: IndexT = klass(list(range(2000, 2005)), **extra_kwargs)
    tm.assert_index_equal(result, expected, exact=True)

@pytest.mark.parametrize("op", [operator.lt, operator.gt])
def test_nan_comparison_same_object(op: Callable[[Any, Any], Any]) -> None:
    idx: IndexT = Index([np.nan])
    expected: np.ndarray = np.array([False])
    result: np.ndarray = op(idx, idx)
    tm.assert_numpy_array_equal(result, expected)
    result = op(idx, idx.copy())
    tm.assert_numpy_array_equal(result, expected)

@td.skip_if_no("pyarrow")
def test_is_monotonic_pyarrow_list_type() -> None:
    import pyarrow as pa
    idx: IndexT = Index([[1], [2, 3]], dtype=pd.ArrowDtype(pa.list_(pa.int64())))
    assert not idx.is_monotonic_increasing
    assert not idx.is_monotonic_decreasing