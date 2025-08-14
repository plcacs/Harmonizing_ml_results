import builtins
import datetime as dt
from string import ascii_lowercase
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import pytest

from pandas._libs.tslibs import iNaT

from pandas.core.dtypes.common import pandas_dtype
from pandas.core.dtypes.missing import na_value_for_dtype

import pandas as pd
from pandas import (
    DataFrame,
    MultiIndex,
    Series,
    Timestamp,
    date_range,
    isna,
)
import pandas._testing as tm
from pandas.util import _test_decorators as td


@pytest.mark.parametrize("dtype", ["int64", "int32", "float64", "float32"])
def test_basic_aggregations(dtype: str) -> None:
    data = Series(np.arange(9) // 3, index=np.arange(9), dtype=dtype)

    index = np.arange(9)
    np.random.default_rng(2).shuffle(index)
    data = data.reindex(index)

    grouped = data.groupby(lambda x: x // 3, group_keys=False)

    for k, v in grouped:
        assert len(v) == 3

    agged = grouped.aggregate(np.mean)
    assert agged[1] == 1

    expected = grouped.agg(np.mean)
    tm.assert_series_equal(agged, expected)  # shorthand
    tm.assert_series_equal(agged, grouped.mean())
    result = grouped.sum()
    expected = grouped.agg(np.sum)
    if dtype == "int32":
        # NumPy's sum returns int64
        expected = expected.astype("int32")
    tm.assert_series_equal(result, expected)

    expected = grouped.apply(lambda x: x * x.sum())
    transformed = grouped.transform(lambda x: x * x.sum())
    assert transformed[7] == 12
    tm.assert_series_equal(transformed, expected)

    value_grouped = data.groupby(data)
    result = value_grouped.aggregate(np.mean)
    tm.assert_series_equal(result, agged, check_index_type=False)

    # complex agg
    agged = grouped.aggregate([np.mean, np.std])

    msg = r"nested renamer is not supported"
    with pytest.raises(pd.errors.SpecificationError, match=msg):
        grouped.aggregate({"one": np.mean, "two": np.std})

    # corner cases
    msg = "Must produce aggregated value"
    # exception raised is type Exception
    with pytest.raises(Exception, match=msg):
        grouped.aggregate(lambda x: x * 2)


@pytest.mark.parametrize(
    "vals",
    [
        ["foo", "bar", "baz"],
        ["foo", "", ""],
        ["", "", ""],
        [1, 2, 3],
        [1, 0, 0],
        [0, 0, 0],
        [1.0, 2.0, 3.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [True, True, True],
        [True, False, False],
        [False, False, False],
        [np.nan, np.nan, np.nan],
    ],
)
def test_groupby_bool_aggs(skipna: bool, all_boolean_reductions: str, vals: List[Any]) -> None:
    df = DataFrame({"key": ["a"] * 3 + ["b"] * 3, "val": vals * 2})

    # Figure out expectation using Python builtin
    exp = getattr(builtins, all_boolean_reductions)(vals)

    # edge case for missing data with skipna and 'any'
    if skipna and all(isna(vals)) and all_boolean_reductions == "any":
        exp = False

    expected = DataFrame(
        [exp] * 2, columns=["val"], index=pd.Index(["a", "b"], name="key")
    )
    result = getattr(df.groupby("key"), all_boolean_reductions)(skipna=skipna)
    tm.assert_frame_equal(result, expected)


def test_any() -> None:
    df = DataFrame(
        [[1, 2, "foo"], [1, np.nan, "bar"], [3, np.nan, "baz"]],
        columns=["A", "B", "C"],
    )
    expected = DataFrame(
        [[True, True], [False, True]], columns=["B", "C"], index=[1, 3]
    )
    expected.index.name = "A"
    result = df.groupby("A").any()
    tm.assert_frame_equal(result, expected)


def test_bool_aggs_dup_column_labels(all_boolean_reductions: str) -> None:
    # GH#21668
    df = DataFrame([[True, True]], columns=["a", "a"])
    grp_by = df.groupby([0])
    result = getattr(grp_by, all_boolean_reductions)()

    expected = df.set_axis(np.array([0]))
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "data",
    [
        [False, False, False],
        [True, True, True],
        [pd.NA, pd.NA, pd.NA],
        [False, pd.NA, False],
        [True, pd.NA, True],
        [True, pd.NA, False],
    ],
)
def test_masked_kleene_logic(all_boolean_reductions: str, skipna: bool, data: List[Any]) -> None:
    # GH#37506
    ser = Series(data, dtype="boolean")

    # The result should match aggregating on the whole series. Correctness
    # there is verified in test_reductions.py::test_any_all_boolean_kleene_logic
    expected_data = getattr(ser, all_boolean_reductions)(skipna=skipna)
    expected = Series(expected_data, index=np.array([0]), dtype="boolean")

    result = ser.groupby([0, 0, 0]).agg(all_boolean_reductions, skipna=skipna)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "dtype1,dtype2,exp_col1,exp_col2",
    [
        (
            "float",
            "Float64",
            np.array([True], dtype=bool),
            pd.array([pd.NA], dtype="boolean"),
        ),
        (
            "Int64",
            "float",
            pd.array([pd.NA], dtype="boolean"),
            np.array([True], dtype=bool),
        ),
        (
            "Int64",
            "Int64",
            pd.array([pd.NA], dtype="boolean"),
            pd.array([pd.NA], dtype="boolean"),
        ),
        (
            "Float64",
            "boolean",
            pd.array([pd.NA], dtype="boolean"),
            pd.array([pd.NA], dtype="boolean"),
        ),
    ],
)
def test_masked_mixed_types(dtype1: str, dtype2: str, exp_col1: Any, exp_col2: Any) -> None:
    # GH#37506
    data = [1.0, np.nan]
    df = DataFrame(
        {"col1": pd.array(data, dtype=dtype1), "col2": pd.array(data, dtype=dtype2)}
    )
    result = df.groupby([1, 1]).agg("all", skipna=False)

    expected = DataFrame({"col1": exp_col1, "col2": exp_col2}, index=np.array([1]))
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype", ["Int64", "Float64", "boolean"])
def test_masked_bool_aggs_skipna(
    all_boolean_reductions: str, dtype: str, skipna: bool, frame_or_series: Any
) -> None:
    # GH#40585
    obj = frame_or_series([pd.NA, 1], dtype=dtype)
    expected_res = True
    if not skipna and all_boolean_reductions == "all":
        expected_res = pd.NA
    expected = frame_or_series([expected_res], index=np.array([1]), dtype="boolean")

    result = obj.groupby([1, 1]).agg(all_boolean_reductions, skipna=skipna)
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "bool_agg_func,data,expected_res",
    [
        ("any", [pd.NA, np.nan], False),
        ("any", [pd.NA, 1, np.nan], True),
        ("all", [pd.NA, pd.NaT], True),
        ("all", [pd.NA, False, pd.NaT], False),
    ],
)
def test_object_type_missing_vals(bool_agg_func: str, data: List[Any], expected_res: Any, frame_or_series: Any) -> None:
    # GH#37501
    obj = frame_or_series(data, dtype=object)
    result = obj.groupby([1] * len(data)).agg(bool_agg_func)
    expected = frame_or_series([expected_res], index=np.array([1]), dtype="bool")
    tm.assert_equal(result, expected)


def test_object_NA_raises_with_skipna_false(all_boolean_reductions: str) -> None:
    # GH#37501
    ser = Series([pd.NA], dtype=object)
    with pytest.raises(TypeError, match="boolean value of NA is ambiguous"):
        ser.groupby([1]).agg(all_boolean_reductions, skipna=False)


def test_empty(frame_or_series: Any, all_boolean_reductions: str) -> None:
    # GH 45231
    kwargs = {"columns": ["a"]} if frame_or_series is DataFrame else {"name": "a"}
    obj = frame_or_series(**kwargs, dtype=object)
    result = getattr(obj.groupby(obj.index), all_boolean_reductions)()
    expected = frame_or_series(**kwargs, dtype=bool)
    tm.assert_equal(result, expected)


@pytest.mark.parametrize("how", ["idxmin", "idxmax"])
def test_idxmin_idxmax_extremes(how: str, any_real_numpy_dtype: Any) -> None:
    # GH#57040
    if any_real_numpy_dtype is int or any_real_numpy_dtype is float:
        # No need to test
        return
    info = np.iinfo if "int" in any_real_numpy_dtype else np.finfo
    min_value = info(any_real_numpy_dtype).min
    max_value = info(any_real_numpy_dtype).max
    df = DataFrame(
        {"a": [2, 1, 1, 2], "b": [min_value, max_value, max_value, min_value]},
        dtype=any_real_numpy_dtype,
    )
    gb = df.groupby("a")
    result = getattr(gb, how)()
    expected = DataFrame(
        {"b": [1, 0]}, index=pd.Index([1, 2], name="a", dtype=any_real_numpy_dtype)
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("how", ["idxmin", "idxmax"])
def test_idxmin_idxmax_extremes_skipna(skipna: bool, how: str, float_numpy_dtype: Any) -> None:
    # GH#57040
    min_value = np.finfo(float_numpy_dtype).min
    max_value = np.finfo(float_numpy_dtype).max
    df = DataFrame(
        {
            "a": Series(np.repeat(range(1, 6), repeats=2), dtype="intp"),
            "b": Series(
                [
                    np.nan,
                    min_value,
                    np.nan,
                    max_value,
                    min_value,
                    np.nan,
                    max_value,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
                dtype=float_numpy_dtype,
            ),
        },
    )
    gb = df.groupby("a")

    if not skipna:
        msg = f"DataFrameGroupBy.{how} with skipna=False"
        with pytest.raises(ValueError, match=msg):
            getattr(gb, how)(skipna=skipna)
        return
    result = getattr(gb, how)(skipna=skipna)
    expected = DataFrame(
        {"b": [1, 3, 4, 6, np.nan]}, index=pd.Index(range(1, 6), name="a", dtype="intp")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "func, values",
    [
        ("idxmin", {"c_int": [0, 2], "c_float": [1, 3], "c_date": [1, 2]}),
        ("idxmax", {"c_int": [1, 3], "c_float": [0, 2], "c_date": [0, 3]}),
    ],
)
@pytest.mark.parametrize("numeric_only", [True, False])
def test_idxmin_idxmax_returns_int_types(func: str, values: Dict[str, List[int]], numeric_only: bool) -> None:
    # GH 25444
    df = DataFrame(
        {
            "name": ["A", "A", "B", "B"],
            "c_int": [1, 2, 3, 4],
            "c_float": [4.02, 3.03, 2.04, 1.05],
            "c_date": ["2019", "2018", "2016", "2017"],
        }
    )
    df["c_date"] = pd.to_datetime(df["c_date"])
    df["c_date_tz"] = df["c_date"].dt.tz_localize("US/Pacific")
    df["c_timedelta"] = df["c_date"] - df["c_date"].iloc[0]
    df["c_period"] = df["c_date"].dt.to_period("W")
    df["c_Integer"] = df["c_int"].astype("Int64")
    df["c_Floating"] = df["c_float"].astype("Float64")

    result = getattr(df.groupby("name"), func)(numeric_only=numeric_only)

    expected = DataFrame(values, index=pd.Index(["A", "B"], name="name"))
    if numeric_only:
        expected = expected.drop(columns=["c_date"])
    else:
        expected["c_date_tz"] = expected["c_date"]
        expected["c_timedelta"] = expected["c_date"]
        expected["c_period"] = expected["c_date"]
    expected["c_Integer"] = expected["c_int"]
    expected["c_Floating"] = expected["c_float"]

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "data",
    [
        (
            Timestamp("2011-01-15 12:50:28.502376"),
            Timestamp("2011-01-20 12:50:28.593448"),
        ),
        (24650000000000001, 24650000000000002),
    ],
)
@pytest.mark.parametrize("method", ["count", "min", "max", "first", "last"])
def test_groupby_non_arithmetic_agg_int_like_precision(method: str, data: Tuple[Any, Any]) -> None:
    # GH#6620, GH#9311
    df = DataFrame({"a": [1, 1], "b": data})

    grouped = df.groupby("a")
    result = getattr(grouped, method)()
    if method == "count":
        expected_value = 2
    elif method == "first":
        expected_value = data[0]
    elif method == "last":
        expected_value = data[1]
    else:
        expected_value = getattr(df["b"], method)()
    expected = DataFrame({"b": [expected_value]}, index=pd.Index([1], name="a"))

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("how", ["first", "last"])
def test_first_last_skipna(any_real_nullable_dtype: Any, sort: bool, skipna: bool, how: str) -> None:
    # GH#57019
    na_value = na_value_for_dtype(pandas_dtype(any_real_nullable_dtype))
    df = DataFrame(
        {
            "a": [2, 1, 1, 2, 3, 3],
            "b": [na_value, 3.0, na_value, 4.0, np.nan, np.nan],
            "c": [na_value, 3.0, na_value, 4.0, np.nan, np.nan],
        },
        dtype=any_real_nullable_dtype,
    )
    gb = df.groupby("a", sort=sort)
    method = getattr(gb, how)
    result = method(skipna=skipna)

    ilocs = {
        ("first", True): [3, 1, 4],
        ("first", False): [0, 1, 4],
        ("last", True): [3, 1, 5],
        ("last", False): [3, 2, 5],
    }[how, skipna]
    expected = df.iloc[ilocs].set_index("a")
    if sort:
        expected = expected.sort_index()
    tm.assert_frame_equal(result, expected)


def test_groupby_mean_no_overflow() -> None:
    # Regression test for (#22487)
    df = DataFrame(
        {
            "user": ["A", "A", "A", "A", "A"],
            "connections": [4970, 4749, 4719, 4704, 18446744073699999744],
        }
    )
    assert df.groupby("user")["connections"].mean()["A"] == 3689348814740003840


def test_mean_on_timedelta() -> None:
    # GH 17382
    df = DataFrame({"time": pd.to_timedelta(range(10)), "cat": ["A", "B