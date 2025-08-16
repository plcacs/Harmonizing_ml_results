import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    concat,
    date_range,
    timedelta_range,
)
import pandas._testing as tm
from pandas.tests.apply.common import series_transform_kernels
from typing import Any, Callable, Dict, List, Optional, Union, Tuple, cast


@pytest.fixture(params=[False, "compat"])
def by_row(request: pytest.FixtureRequest) -> Union[bool, str]:
    return cast(Union[bool, str], request.param)


def test_series_map_box_timedelta(by_row: Union[bool, str]) -> None:
    # GH#11349
    ser = Series(timedelta_range("1 day 1 s", periods=3, freq="h"))

    def f(x: pd.Timedelta) -> float:
        return x.total_seconds() if by_row else x.dt.total_seconds()

    result = ser.apply(f, by_row=by_row)

    expected = ser.map(lambda x: x.total_seconds())
    tm.assert_series_equal(result, expected)

    expected = Series([86401.0, 90001.0, 93601.0])
    tm.assert_series_equal(result, expected)


def test_apply(datetime_series: Series, by_row: Union[bool, str]) -> None:
    result = datetime_series.apply(np.sqrt, by_row=by_row)
    with np.errstate(all="ignore"):
        expected = np.sqrt(datetime_series)
    tm.assert_series_equal(result, expected)

    # element-wise apply (ufunc)
    result = datetime_series.apply(np.exp, by_row=by_row)
    expected = np.exp(datetime_series)
    tm.assert_series_equal(result, expected)

    # empty series
    s = Series(dtype=object, name="foo", index=Index([], name="bar"))
    rs = s.apply(lambda x: x, by_row=by_row)
    tm.assert_series_equal(s, rs)

    # check all metadata (GH 9322)
    assert s is not rs
    assert s.index is rs.index
    assert s.dtype == rs.dtype
    assert s.name == rs.name

    # index but no data
    s = Series(index=[1, 2, 3], dtype=np.float64)
    rs = s.apply(lambda x: x, by_row=by_row)
    tm.assert_series_equal(s, rs)


def test_apply_map_same_length_inference_bug() -> None:
    s = Series([1, 2])

    def f(x: int) -> Tuple[int, int]:
        return (x, x + 1)

    result = s.apply(f, by_row="compat")
    expected = s.map(f)
    tm.assert_series_equal(result, expected)


def test_apply_args() -> None:
    s = Series(["foo,bar"])

    result = s.apply(str.split, args=(",",))
    assert result[0] == ["foo", "bar"]
    assert isinstance(result[0], list)


@pytest.mark.parametrize(
    "args, kwargs, increment",
    [((), {}, 0), ((), {"a": 1}, 1), ((2, 3), {}, 32), ((1,), {"c": 2}, 201)],
)
def test_agg_args(args: Tuple[Any, ...], kwargs: Dict[str, Any], increment: int) -> None:
    # GH 43357
    def f(x: int, a: int = 0, b: int = 0, c: int = 0) -> int:
        return x + a + 10 * b + 100 * c

    s = Series([1, 2])
    result = s.agg(f, 0, *args, **kwargs)
    expected = s + increment
    tm.assert_series_equal(result, expected)


def test_agg_mapping_func_deprecated() -> None:
    # GH 53325
    s = Series([1, 2, 3])

    def foo1(x: int, a: int = 1, c: int = 0) -> int:
        return x + a + c

    def foo2(x: int, b: int = 2, c: int = 0) -> int:
        return x + b + c

    s.agg(foo1, 0, 3, c=4)
    s.agg([foo1, foo2], 0, 3, c=4)
    s.agg({"a": foo1, "b": foo2}, 0, 3, c=4)


def test_series_apply_map_box_timestamps(by_row: Union[bool, str]) -> None:
    # GH#2689, GH#2627
    ser = Series(date_range("1/1/2000", periods=10))

    def func(x: pd.Timestamp) -> Tuple[int, int, int]:
        return (x.hour, x.day, x.month)

    if not by_row:
        msg = "Series' object has no attribute 'hour'"
        with pytest.raises(AttributeError, match=msg):
            ser.apply(func, by_row=by_row)
        return

    result = ser.apply(func, by_row=by_row)
    expected = ser.map(func)
    tm.assert_series_equal(result, expected)


def test_apply_box_dt64() -> None:
    # ufunc will not be boxed. Same test cases as the test_map_box
    vals = [pd.Timestamp("2011-01-01"), pd.Timestamp("2011-01-02")]
    ser = Series(vals, dtype="M8[ns]")
    assert ser.dtype == "datetime64[ns]"
    # boxed value must be Timestamp instance
    res = ser.apply(lambda x: f"{type(x).__name__}_{x.day}_{x.tz}", by_row="compat")
    exp = Series(["Timestamp_1_None", "Timestamp_2_None"])
    tm.assert_series_equal(res, exp)


def test_apply_box_dt64tz() -> None:
    vals = [
        pd.Timestamp("2011-01-01", tz="US/Eastern"),
        pd.Timestamp("2011-01-02", tz="US/Eastern"),
    ]
    ser = Series(vals, dtype="M8[ns, US/Eastern]")
    assert ser.dtype == "datetime64[ns, US/Eastern]"
    res = ser.apply(lambda x: f"{type(x).__name__}_{x.day}_{x.tz}", by_row="compat")
    exp = Series(["Timestamp_1_US/Eastern", "Timestamp_2_US/Eastern"])
    tm.assert_series_equal(res, exp)


def test_apply_box_td64() -> None:
    # timedelta
    vals = [pd.Timedelta("1 days"), pd.Timedelta("2 days")]
    ser = Series(vals)
    assert ser.dtype == "timedelta64[ns]"
    res = ser.apply(lambda x: f"{type(x).__name__}_{x.days}", by_row="compat")
    exp = Series(["Timedelta_1", "Timedelta_2"])
    tm.assert_series_equal(res, exp)


def test_apply_box_period() -> None:
    # period
    vals = [pd.Period("2011-01-01", freq="M"), pd.Period("2011-01-02", freq="M")]
    ser = Series(vals)
    assert ser.dtype == "Period[M]"
    res = ser.apply(lambda x: f"{type(x).__name__}_{x.freqstr}", by_row="compat")
    exp = Series(["Period_M", "Period_M"])
    tm.assert_series_equal(res, exp)


def test_apply_datetimetz(by_row: Union[bool, str]) -> None:
    values = date_range("2011-01-01", "2011-01-02", freq="h").tz_localize("Asia/Tokyo")
    s = Series(values, name="XX")

    result = s.apply(lambda x: x + pd.offsets.Day(), by_row=by_row)
    exp_values = date_range("2011-01-02", "2011-01-03", freq="h").tz_localize(
        "Asia/Tokyo"
    )
    exp = Series(exp_values, name="XX")
    tm.assert_series_equal(result, exp)

    result = s.apply(lambda x: x.hour if by_row else x.dt.hour, by_row=by_row)
    exp = Series(list(range(24)) + [0], name="XX", dtype="int64" if by_row else "int32")
    tm.assert_series_equal(result, exp)

    # not vectorized
    def f(x: pd.Timestamp) -> str:
        return str(x.tz) if by_row else str(x.dt.tz)

    result = s.apply(f, by_row=by_row)
    if by_row:
        exp = Series(["Asia/Tokyo"] * 25, name="XX")
        tm.assert_series_equal(result, exp)
    else:
        assert result == "Asia/Tokyo"


def test_apply_categorical(by_row: Union[bool, str], using_infer_string: bool) -> None:
    values = pd.Categorical(list("ABBABCD"), categories=list("DCBA"), ordered=True)
    ser = Series(values, name="XX", index=list("abcdefg"))

    if not by_row:
        msg = "Series' object has no attribute 'lower"
        with pytest.raises(AttributeError, match=msg):
            ser.apply(lambda x: x.lower(), by_row=by_row)
        assert ser.apply(lambda x: "A", by_row=by_row) == "A"
        return

    result = ser.apply(lambda x: x.lower(), by_row=by_row)

    # should be categorical dtype when the number of categories are
    # the same
    values = pd.Categorical(list("abbabcd"), categories=list("dcba"), ordered=True)
    exp = Series(values, name="XX", index=list("abcdefg"))
    tm.assert_series_equal(result, exp)
    tm.assert_categorical_equal(result.values, exp.values)

    result = ser.apply(lambda x: "A")
    exp = Series(["A"] * 7, name="XX", index=list("abcdefg"))
    tm.assert_series_equal(result, exp)
    assert result.dtype == object if not using_infer_string else "str"


@pytest.mark.parametrize("series", [["1-1", "1-1", np.nan], ["1-1", "1-2", np.nan]])
def test_apply_categorical_with_nan_values(
    series: List[str], by_row: Union[bool, str]
) -> None:
    # GH 20714 bug fixed in: GH 24275
    s = Series(series, dtype="category")
    if not by_row:
        msg = "'Series' object has no attribute 'split'"
        with pytest.raises(AttributeError, match=msg):
            s.apply(lambda x: x.split("-")[0], by_row=by_row)
        return
    # NaN for cat dtype fixed in (GH 59966)
    result = s.apply(lambda x: x.split("-")[0] if pd.notna(x) else False, by_row=by_row)
    result = result.astype(object)
    expected = Series(["1", "1", False], dtype="category")
    expected = expected.astype(object)
    tm.assert_series_equal(result, expected)


def test_apply_empty_integer_series_with_datetime_index(by_row: Union[bool, str]) -> None:
    # GH 21245
    s = Series([], index=date_range(start="2018-01-01", periods=0), dtype=int)
    result = s.apply(lambda x: x, by_row=by_row)
    tm.assert_series_equal(result, s)


def test_apply_dataframe_iloc() -> None:
    uintDF = DataFrame(np.uint64([1, 2, 3, 4, 5]), columns=["Numbers"])
    indexDF = DataFrame([2, 3, 2, 1, 2], columns=["Indices"])

    def retrieve(targetRow: int, targetDF: DataFrame) -> Any:
        val = targetDF["Numbers"].iloc[targetRow]
        return val

    result = indexDF["Indices"].apply(retrieve, args=(uintDF,))
    expected = Series([3, 4, 3, 2, 3], name="Indices", dtype="uint64")
    tm.assert_series_equal(result, expected)


def test_transform(string_series: Series, by_row: Union[bool, str]) -> None:
    # transforming functions

    with np.errstate(all="ignore"):
        f_sqrt = np.sqrt(string_series)
        f_abs = np.abs(string_series)

        # ufunc
        result = string_series.apply(np.sqrt, by_row=by_row)
        expected = f_sqrt.copy()
        tm.assert_series_equal(result, expected)

        # list-like
        result = string_series.apply([np.sqrt], by_row=by_row)
        expected = f_sqrt.to_frame().copy()
        expected.columns = ["sqrt"]
        tm.assert_frame_equal(result, expected)

        result = string_series.apply(["sqrt"], by_row=by_row)
        tm.assert_frame_equal(result, expected)

        # multiple items in list
        # these are in the order as if we are applying both functions per
        # series and then concatting
        expected = concat([f_sqrt, f_abs], axis=1)
        expected.columns = ["sqrt", "absolute"]
        result = string_series.apply([np.sqrt, np.abs], by_row=by_row)
        tm.assert_frame_equal(result, expected)

        # dict, provide renaming
        expected = concat([f_sqrt, f_abs], axis=1)
        expected.columns = ["foo", "bar"]
        expected = expected.unstack().rename("series")

        result = string_series.apply({"foo": np.sqrt, "bar": np.abs}, by_row=by_row)
        tm.assert_series_equal(result.reindex_like(expected), expected)


@pytest.mark.parametrize("op", series_transform_kernels)
def test_transform_partial_failure(op: str, request: pytest.FixtureRequest) -> None:
    # GH 35964
    if op in ("ffill", "bfill", "shift"):
        request.applymarker(
            pytest.mark.xfail(reason=f"{op} is successful on any dtype")
        )

    # Using object makes most transform kernels fail
    ser = Series(3 * [object])

    if op in ("fillna", "ngroup"):
        error = ValueError
        msg = "Transform function failed"
    else:
        error = TypeError
        msg = "|".join(
            [
                "not supported between instances of 'type' and 'type'",
                "unsupported operand type",
            ]
        )

    with pytest.raises(error, match=msg):
        ser.transform([op, "shift"])

    with pytest.raises(error, match=msg):
        ser.transform({"A": op, "B": "shift"})

    with pytest.raises(error, match=msg):
        ser.transform({"A": [op], "B": ["shift"]})

    with pytest.raises(error, match=msg):
        ser.transform({"A": [op, "shift"], "B": [op]})


def test_transform_partial_failure_valueerror() -> None:
    # GH 40211
    def noop(x: Any) -> Any:
        return x

    def raising_op(_: Any) -> None:
        raise ValueError

    ser = Series(3 * [object])
    msg = "Transform function failed"

    with pytest.raises(ValueError, match=msg):
        ser.transform([noop, raising_op])

    with pytest.raises(ValueError, match=msg):
        ser.transform({"A": raising_op, "B": noop})

    with pytest.raises(ValueError, match=msg):
        ser.transform({"A": [raising_op], "B": [noop]})

    with pytest.raises(ValueError, match=msg):
        ser.transform({"A": [noop, raising_op], "B": [noop]})


def test_demo() -> None:
    # demonstration tests
    s = Series(range(6), dtype="int64", name="series")

    result = s.agg(["min", "max"])
    expected = Series([0, 5], index=["min", "max"], name="series")
    tm.assert_series_equal(result, expected)

    result = s.agg({"foo": "min"})
    expected = Series([0], index=["foo"], name="series")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "func", [str, lambda x: str(x)]
)
def test_apply_map_evaluate_lambdas_the_same(
    string_series: Series, func: Callable[[Any], str], by_row: Union[bool, str]
) -> None:
    # test that we are evaluating row-by-row first if by_row="compat"
    # else vectorized evaluation
    result = string_series.apply(func, by_row=by_row)

    if by_row:
        expected = string_series.map(func)
        tm.assert_series_equal(result, expected)
    else:
        assert result == str(string_series)


def test_agg_evaluate_lambdas(string_series: Series) -> None:
    # GH53325
    result = string_series.agg(lambda x: type(x))
    assert result is Series

    result = string_series.agg(type)
    assert result is Series


@pytest.mark.parametrize("op_name", ["agg", "apply"])
def test_with_nested_series(datetime_series: Series, op_name: str) -> None:
    # GH 2316 & GH52123
    # .agg with a reducer and a transform, what to do
    result = getattr(datetime_series, op_name)(
        lambda x: Series([x, x**2], index=["x", "x^2"])
    )
    if op_name == "apply":
        expected = DataFrame({"x": datetime_series, "x^2": datetime_series**2})
        tm.assert_frame_equal(result, expected)
    else:
        expected = Series([datetime_series, datetime_series