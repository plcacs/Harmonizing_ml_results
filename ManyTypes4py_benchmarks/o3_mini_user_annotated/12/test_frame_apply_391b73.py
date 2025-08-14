#!/usr/bin/env python3
from datetime import datetime
import warnings
from typing import Any, Callable, List, Optional, Union

import numpy as np
import pytest

from pandas.compat import is_platform_arm

from pandas.core.dtypes.dtypes import CategoricalDtype

import pandas as pd
from pandas import (
    DataFrame,
    MultiIndex,
    Series,
    Timestamp,
    date_range,
)
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
from pandas.util.version import Version


@pytest.fixture
def int_frame_const_col() -> DataFrame:
    """
    Fixture for DataFrame of ints which are constant per column

    Columns are ['A', 'B', 'C'], with values (per column): [1, 2, 3]
    """
    df: DataFrame = DataFrame(
        np.tile(np.arange(3, dtype="int64"), 6).reshape(6, -1) + 1,
        columns=["A", "B", "C"],
    )
    return df


@pytest.fixture(params=["python", pytest.param("numba", marks=pytest.mark.single_cpu)])
def engine(request: pytest.FixtureRequest) -> str:
    if request.param == "numba":
        pytest.importorskip("numba")
    return request.param


def test_apply(float_frame: DataFrame, engine: str, request: pytest.FixtureRequest) -> None:
    if engine == "numba":
        mark = pytest.mark.xfail(reason="numba engine not supporting numpy ufunc yet")
        request.node.add_marker(mark)
    with np.errstate(all="ignore"):
        # ufunc
        result: Series = np.sqrt(float_frame["A"])
        expected: Series = float_frame.apply(np.sqrt, engine=engine)["A"]
        tm.assert_series_equal(result, expected)

        # aggregator
        result = float_frame.apply(np.mean, engine=engine)["A"]
        expected = np.mean(float_frame["A"])
        assert result == expected

        d = float_frame.index[0]
        result = float_frame.apply(np.mean, axis=1, engine=engine)
        expected = np.mean(float_frame.xs(d))
        assert result[d] == expected
        assert result.index is float_frame.index


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("raw", [True, False])
@pytest.mark.parametrize("nopython", [True, False])
def test_apply_args(float_frame: DataFrame, axis: Union[int, str], raw: bool, engine: str, nopython: bool) -> None:
    numba = pytest.importorskip("numba")
    if (
        engine == "numba"
        and Version(numba.__version__) == Version("0.61")
        and is_platform_arm()
    ):
        pytest.skip(f"Segfaults on ARM platforms with numba {numba.__version__}")
    engine_kwargs = {"nopython": nopython}
    result = float_frame.apply(
        lambda x, y: x + y,
        axis,
        args=(1,),
        raw=raw,
        engine=engine,
        engine_kwargs=engine_kwargs,
    )
    expected = float_frame + 1
    tm.assert_frame_equal(result, expected)

    # GH:58712
    result = float_frame.apply(
        lambda x, a, b: x + a + b,
        args=(1,),
        b=2,
        raw=raw,
        engine=engine,
        engine_kwargs=engine_kwargs,
    )
    expected = float_frame + 3
    tm.assert_frame_equal(result, expected)

    if engine == "numba":
        # py signature binding
        with pytest.raises(TypeError, match="missing a required argument: 'a'"):
            float_frame.apply(
                lambda x, a: x + a,
                b=2,
                raw=raw,
                engine=engine,
                engine_kwargs=engine_kwargs,
            )

        # keyword-only arguments are not supported in numba
        with pytest.raises(
            pd.errors.NumbaUtilError,
            match="numba does not support keyword-only arguments",
        ):
            float_frame.apply(
                lambda x, a, *, b: x + a + b,
                args=(1,),
                b=2,
                raw=raw,
                engine=engine,
                engine_kwargs=engine_kwargs,
            )

        with pytest.raises(
            pd.errors.NumbaUtilError,
            match="numba does not support keyword-only arguments",
        ):
            float_frame.apply(
                lambda *x, b: x[0] + x[1] + b,
                args=(1,),
                b=2,
                raw=raw,
                engine=engine,
                engine_kwargs=engine_kwargs,
            )


def test_apply_categorical_func() -> None:
    # GH 9573
    df: DataFrame = DataFrame({"c0": ["A", "A", "B", "B"], "c1": ["C", "C", "D", "D"]})
    result: DataFrame = df.apply(lambda ts: ts.astype("category"))
    assert result.shape == (4, 2)
    assert isinstance(result["c0"].dtype, CategoricalDtype)
    assert isinstance(result["c1"].dtype, CategoricalDtype)


def test_apply_axis1_with_ea() -> None:
    # GH#36785
    expected: DataFrame = DataFrame({"A": [Timestamp("2013-01-01", tz="UTC")]})
    result: DataFrame = expected.apply(lambda x: x, axis=1)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "data, dtype",
    [(1, None), (1, CategoricalDtype([1])), (Timestamp("2013-01-01", tz="UTC"), None)],
)
def test_agg_axis1_duplicate_index(data: Any, dtype: Optional[CategoricalDtype]) -> None:
    # GH 42380
    expected: DataFrame = DataFrame([[data], [data]], index=["a", "a"], dtype=dtype)
    result: DataFrame = expected.agg(lambda x: x, axis=1)
    tm.assert_frame_equal(result, expected)


def test_apply_mixed_datetimelike() -> None:
    # mixed datetimelike
    # GH 7778
    expected: DataFrame = DataFrame(
        {
            "A": date_range("20130101", periods=3),
            "B": pd.to_timedelta(np.arange(3), unit="s"),
        }
    )
    result: DataFrame = expected.apply(lambda x: x, axis=1)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("func", [np.sqrt, np.mean])
def test_apply_empty(func: Callable[[Any], Any], engine: str) -> None:
    # empty
    empty_frame: DataFrame = DataFrame()
    result: DataFrame = empty_frame.apply(func, engine=engine)
    assert result.empty


def test_apply_float_frame(float_frame: DataFrame, engine: str) -> None:
    no_rows: DataFrame = float_frame[:0]
    result: Series = no_rows.apply(lambda x: x.mean(), engine=engine)
    expected: Series = Series(np.nan, index=float_frame.columns)
    tm.assert_series_equal(result, expected)

    no_cols: DataFrame = float_frame.loc[:, []]
    result = no_cols.apply(lambda x: x.mean(), axis=1, engine=engine)
    expected = Series(np.nan, index=float_frame.index)
    tm.assert_series_equal(result, expected)


def test_apply_empty_except_index(engine: str) -> None:
    # GH 2476
    expected: DataFrame = DataFrame(index=["a"])
    result: DataFrame = expected.apply(lambda x: x["a"], axis=1, engine=engine)
    tm.assert_frame_equal(result, expected)


def test_apply_with_reduce_empty() -> None:
    # reduce with an empty DataFrame
    empty_frame: DataFrame = DataFrame()

    x: List[Any] = []
    result: DataFrame = empty_frame.apply(x.append, axis=1, result_type="expand")
    tm.assert_frame_equal(result, empty_frame)
    result: Series = empty_frame.apply(x.append, axis=1, result_type="reduce")
    expected: Series = Series([], dtype=np.float64)
    tm.assert_series_equal(result, expected)

    empty_with_cols: DataFrame = DataFrame(columns=["a", "b", "c"])
    result = empty_with_cols.apply(x.append, axis=1, result_type="expand")
    tm.assert_frame_equal(result, empty_with_cols)
    result = empty_with_cols.apply(x.append, axis=1, result_type="reduce")
    expected = Series([], dtype=np.float64)
    tm.assert_series_equal(result, expected)

    # Ensure that x.append hasn't been called
    assert x == []


@pytest.mark.parametrize("func", ["sum", "prod", "any", "all"])
def test_apply_funcs_over_empty(func: str) -> None:
    # GH 28213
    df: DataFrame = DataFrame(columns=["a", "b", "c"])

    result: Series = df.apply(getattr(np, func))
    expected = getattr(df, func)()
    if func in ("sum", "prod"):
        expected = expected.astype(float)
    tm.assert_series_equal(result, expected)


def test_nunique_empty() -> None:
    # GH 28213
    df: DataFrame = DataFrame(columns=["a", "b", "c"])

    result: Series = df.nunique()
    expected: Series = Series(0, index=df.columns)
    tm.assert_series_equal(result, expected)

    result: Series = df.T.nunique()
    expected = Series([], dtype=np.float64)
    tm.assert_series_equal(result, expected)


def test_apply_standard_nonunique() -> None:
    df: DataFrame = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=["a", "a", "c"])

    result: Series = df.apply(lambda s: s[0], axis=1)
    expected: Series = Series([1, 4, 7], index=["a", "a", "c"])
    tm.assert_series_equal(result, expected)

    result = df.T.apply(lambda s: s[0], axis=0)
    tm.assert_series_equal(result, expected)


def test_apply_broadcast_scalars(float_frame: DataFrame) -> None:
    # scalars
    result: DataFrame = float_frame.apply(np.mean, result_type="broadcast")
    expected: DataFrame = DataFrame([float_frame.mean()], index=float_frame.index)
    tm.assert_frame_equal(result, expected)


def test_apply_broadcast_scalars_axis1(float_frame: DataFrame) -> None:
    result: DataFrame = float_frame.apply(np.mean, axis=1, result_type="broadcast")
    m: Series = float_frame.mean(axis=1)
    expected: DataFrame = DataFrame({c: m for c in float_frame.columns})
    tm.assert_frame_equal(result, expected)


def test_apply_broadcast_lists_columns(float_frame: DataFrame) -> None:
    # lists
    result: DataFrame = float_frame.apply(
        lambda x: list(range(len(float_frame.columns))),
        axis=1,
        result_type="broadcast",
    )
    m: List[int] = list(range(len(float_frame.columns)))
    expected: DataFrame = DataFrame(
        [m] * len(float_frame.index),
        dtype="float64",
        index=float_frame.index,
        columns=float_frame.columns,
    )
    tm.assert_frame_equal(result, expected)


def test_apply_broadcast_lists_index(float_frame: DataFrame) -> None:
    result: DataFrame = float_frame.apply(
        lambda x: list(range(len(float_frame.index))), result_type="broadcast"
    )
    m: List[int] = list(range(len(float_frame.index)))
    expected: DataFrame = DataFrame(
        {c: m for c in float_frame.columns},
        dtype="float64",
        index=float_frame.index,
    )
    tm.assert_frame_equal(result, expected)


def test_apply_broadcast_list_lambda_func(int_frame_const_col: DataFrame) -> None:
    # preserve columns
    df: DataFrame = int_frame_const_col
    result: DataFrame = df.apply(lambda x: [1, 2, 3], axis=1, result_type="broadcast")
    tm.assert_frame_equal(result, df)


def test_apply_broadcast_series_lambda_func(int_frame_const_col: DataFrame) -> None:
    df: DataFrame = int_frame_const_col
    result: DataFrame = df.apply(
        lambda x: Series([1, 2, 3], index=list("abc")),
        axis=1,
        result_type="broadcast",
    )
    expected: DataFrame = df.copy()
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("axis", [0, 1])
def test_apply_raw_float_frame(float_frame: DataFrame, axis: Union[int, str], engine: str) -> None:
    if engine == "numba":
        pytest.skip("numba can't handle when UDF returns None.")

    def _assert_raw(x: np.ndarray) -> None:
        assert isinstance(x, np.ndarray)
        assert x.ndim == 1

    float_frame.apply(_assert_raw, axis=axis, engine=engine, raw=True)


@pytest.mark.parametrize("axis", [0, 1])
def test_apply_raw_float_frame_lambda(float_frame: DataFrame, axis: Union[int, str], engine: str) -> None:
    result: Series = float_frame.apply(np.mean, axis=axis, engine=engine, raw=True)
    expected: Series = float_frame.apply(lambda x: x.values.mean(), axis=axis)
    tm.assert_series_equal(result, expected)


def test_apply_raw_float_frame_no_reduction(float_frame: DataFrame, engine: str) -> None:
    # no reduction
    result: DataFrame = float_frame.apply(lambda x: x * 2, engine=engine, raw=True)
    expected: DataFrame = float_frame * 2
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("axis", [0, 1])
def test_apply_raw_mixed_type_frame(axis: Union[int, str], engine: str) -> None:
    if engine == "numba":
        pytest.skip("isinstance check doesn't work with numba")

    def _assert_raw(x: np.ndarray) -> None:
        assert isinstance(x, np.ndarray)
        assert x.ndim == 1

    # Mixed dtype (GH-32423)
    df: DataFrame = DataFrame(
        {
            "a": 1.0,
            "b": 2,
            "c": "foo",
            "float32": np.array([1.0] * 10, dtype="float32"),
            "int32": np.array([1] * 10, dtype="int32"),
        },
        index=np.arange(10),
    )
    df.apply(_assert_raw, axis=axis, engine=engine, raw=True)


def test_apply_axis1(float_frame: DataFrame) -> None:
    d = float_frame.index[0]
    result = float_frame.apply(np.mean, axis=1)[d]
    expected = np.mean(float_frame.xs(d))
    assert result == expected


def test_apply_mixed_dtype_corner() -> None:
    df: DataFrame = DataFrame({"A": ["foo"], "B": [1.0]})
    result: Series = df[:0].apply(np.mean, axis=1)
    expected: Series = Series(dtype=np.float64)
    tm.assert_series_equal(result, expected)


def test_apply_mixed_dtype_corner_indexing() -> None:
    df: DataFrame = DataFrame({"A": ["foo"], "B": [1.0]})
    result: Series = df.apply(lambda x: x["A"], axis=1)
    expected: Series = Series(["foo"], index=range(1))
    tm.assert_series_equal(result, expected)

    result = df.apply(lambda x: x["B"], axis=1)
    expected = Series([1.0], index=range(1))
    tm.assert_series_equal(result, expected)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("ax", ["index", "columns"])
@pytest.mark.parametrize(
    "func", [lambda x: x, lambda x: x.mean()], ids=["identity", "mean"]
)
@pytest.mark.parametrize("raw", [True, False])
@pytest.mark.parametrize("axis", [0, 1])
def test_apply_empty_infer_type(ax: str, func: Callable[[np.ndarray], Any], raw: bool, axis: Union[int, str], engine: str, request: pytest.FixtureRequest) -> None:
    df: DataFrame = DataFrame(**{ax: ["a", "b", "c"]})

    with np.errstate(all="ignore"):
        test_res = func(np.array([], dtype="f8"))
        is_reduction: bool = not isinstance(test_res, np.ndarray)

        result = df.apply(func, axis=axis, engine=engine, raw=raw)
        if is_reduction:
            agg_axis: Any = df._get_agg_axis(axis)
            assert isinstance(result, Series)
            assert result.index is agg_axis
        else:
            assert isinstance(result, DataFrame)


def test_apply_empty_infer_type_broadcast() -> None:
    no_cols: DataFrame = DataFrame(index=["a", "b", "c"])
    result: DataFrame = no_cols.apply(lambda x: x.mean(), result_type="broadcast")
    assert isinstance(result, DataFrame)


def test_apply_with_args_kwds_add_some(float_frame: DataFrame) -> None:
    def add_some(x: Any, howmuch: int = 0) -> Any:
        return x + howmuch

    result: DataFrame = float_frame.apply(add_some, howmuch=2)
    expected: DataFrame = float_frame.apply(lambda x: x + 2)
    tm.assert_frame_equal(result, expected)


def test_apply_with_args_kwds_agg_and_add(float_frame: DataFrame) -> None:
    def agg_and_add(x: Any, howmuch: int = 0) -> Any:
        return x.mean() + howmuch

    result: Series = float_frame.apply(agg_and_add, howmuch=2)
    expected: Series = float_frame.apply(lambda x: x.mean() + 2)
    tm.assert_series_equal(result, expected)


def test_apply_with_args_kwds_subtract_and_divide(float_frame: DataFrame) -> None:
    def subtract_and_divide(x: Any, sub: float, divide: float = 1) -> Any:
        return (x - sub) / divide

    result: DataFrame = float_frame.apply(subtract_and_divide, args=(2,), divide=2)
    expected: DataFrame = float_frame.apply(lambda x: (x - 2.0) / 2.0)
    tm.assert_frame_equal(result, expected)


def test_apply_yield_list(float_frame: DataFrame) -> None:
    result: DataFrame = float_frame.apply(list)
    tm.assert_frame_equal(result, float_frame)


def test_apply_reduce_Series(float_frame: DataFrame) -> None:
    float_frame.iloc[::2, float_frame.columns.get_loc("A")] = np.nan
    expected: Series = float_frame.mean(axis=1)
    result: Series = float_frame.apply(np.mean, axis=1)
    tm.assert_series_equal(result, expected)


def test_apply_reduce_to_dict() -> None:
    # GH 25196 37544
    data: DataFrame = DataFrame([[1, 2], [3, 4]], columns=["c0", "c1"], index=["i0", "i1"])

    result: Series = data.apply(dict, axis=0)
    expected: Series = Series([{"i0": 1, "i1": 3}, {"i0": 2, "i1": 4}], index=data.columns)
    tm.assert_series_equal(result, expected)

    result = data.apply(dict, axis=1)
    expected = Series([{"c0": 1, "c1": 2}, {"c0": 3, "c1": 4}], index=data.index)
    tm.assert_series_equal(result, expected)


def test_apply_differently_indexed() -> None:
    df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((20, 10)))
    result: DataFrame = df.apply(Series.describe, axis=0)
    expected: DataFrame = DataFrame({i: v.describe() for i, v in df.items()}, columns=df.columns)
    tm.assert_frame_equal(result, expected)

    result = df.apply(Series.describe, axis=1)
    expected = DataFrame({i: v.describe() for i, v in df.T.items()}, columns=df.index).T
    tm.assert_frame_equal(result, expected)


def test_apply_bug() -> None:
    # GH 6125
    positions: DataFrame = DataFrame(
        [
            [1, "ABC0", 50],
            [1, "YUM0", 20],
            [1, "DEF0", 20],
            [2, "ABC1", 50],
            [2, "YUM1", 20],
            [2, "DEF1", 20],
        ],
        columns=["a", "market", "position"],
    )

    def f(r: Series) -> Any:
        return r["market"]

    expected: Series = positions.apply(f, axis=1)

    positions = DataFrame(
        [
            [datetime(2013, 1, 1), "ABC0", 50],
            [datetime(2013, 1, 2), "YUM0", 20],
            [datetime(2013, 1, 3), "DEF0", 20],
            [datetime(2013, 1, 4), "ABC1", 50],
            [datetime(2013, 1, 5), "YUM1", 20],
            [datetime(2013, 1, 6), "DEF1", 20],
        ],
        columns=["a", "market", "position"],
    )
    result: Series = positions.apply(f, axis=1)
    tm.assert_series_equal(result, expected)


def test_apply_convert_objects() -> None:
    expected: DataFrame = DataFrame(
        {
            "A": [
                "foo",
                "foo",
                "foo",
                "foo",
                "bar",
                "bar",
                "bar",
                "bar",
                "foo",
                "foo",
                "foo",
            ],
            "B": [
                "one",
                "one",
                "one",
                "two",
                "one",
                "one",
                "one",
                "two",
                "two",
                "two",
                "one",
            ],
            "C": [
                "dull",
                "dull",
                "shiny",
                "dull",
                "dull",
                "shiny",
                "shiny",
                "dull",
                "shiny",
                "shiny",
                "shiny",
            ],
            "D": np.random.default_rng(2).standard_normal(11),
            "E": np.random.default_rng(2).standard_normal(11),
            "F": np.random.default_rng(2).standard_normal(11),
        }
    )

    result: DataFrame = expected.apply(lambda x: x, axis=1)
    tm.assert_frame_equal(result, expected)


def test_apply_attach_name(float_frame: DataFrame) -> None:
    result: Series = float_frame.apply(lambda x: x.name)
    expected: Series = Series(float_frame.columns, index=float_frame.columns)
    tm.assert_series_equal(result, expected)


def test_apply_attach_name_axis1(float_frame: DataFrame) -> None:
    result: Series = float_frame.apply(lambda x: x.name, axis=1)
    expected: Series = Series(float_frame.index, index=float_frame.index)
    tm.assert_series_equal(result, expected)


def test_apply_attach_name_non_reduction(float_frame: DataFrame) -> None:
    # non-reductions
    result: DataFrame = float_frame.apply(lambda x: np.repeat(x.name, len(x)))
    expected: DataFrame = DataFrame(
        np.tile(float_frame.columns, (len(float_frame.index), 1)),
        index=float_frame.index,
        columns=float_frame.columns,
    )
    tm.assert_frame_equal(result, expected)


def test_apply_attach_name_non_reduction_axis1(float_frame: DataFrame) -> None:
    result: Series = float_frame.apply(lambda x: np.repeat(x.name, len(x)), axis=1)
    # Using generator expression for expected Series values
    expected = Series(
        [tuple([row[0]] * len(float_frame.columns)) for row in float_frame.itertuples()],
        index=float_frame.index,
    )
    tm.assert_series_equal(result, expected)


def test_apply_multi_index() -> None:
    index: MultiIndex = MultiIndex.from_arrays([["a", "a", "b"], ["c", "d", "d"]])
    s: DataFrame = DataFrame([[1, 2], [3, 4], [5, 6]], index=index, columns=["col1", "col2"])
    result: DataFrame = s.apply(lambda x: Series({"min": min(x), "max": max(x)}), 1)
    expected: DataFrame = DataFrame([[1, 2], [3, 4], [5, 6]], index=index, columns=["min", "max"])
    tm.assert_frame_equal(result, expected, check_like=True)


@pytest.mark.parametrize(
    "df, dicts",
    [
        [
            DataFrame([["foo", "bar"], ["spam", "eggs"]]),
            Series([{0: "foo", 1: "spam"}, {0: "bar", 1: "eggs"}]),
        ],
        [DataFrame([[0, 1], [2, 3]]), Series([{0: 0, 1: 2}, {0: 1, 1: 3}])],
    ],
)
def test_apply_dict(df: DataFrame, dicts: Series) -> None:
    # GH 8735
    fn: Callable[[Any], dict] = lambda x: x.to_dict()
    reduce_true: Series = df.apply(fn, result_type="reduce")
    reduce_false: DataFrame = df.apply(fn, result_type="expand")
    reduce_none: Series = df.apply(fn)
    tm.assert_series_equal(reduce_true, dicts)
    tm.assert_frame_equal(reduce_false, df)
    tm.assert_series_equal(reduce_none, dicts)


def test_apply_non_numpy_dtype() -> None:
    df: DataFrame = DataFrame({"dt": date_range("2015-01-01", periods=3, tz="Europe/Brussels")})
    result: DataFrame = df.apply(lambda x: x)
    tm.assert_frame_equal(result, df)

    result = df.apply(lambda x: x + pd.Timedelta("1day"))
    expected: DataFrame = DataFrame(
        {"dt": date_range("2015-01-02", periods=3, tz="Europe/Brussels")}
    )
    tm.assert_frame_equal(result, expected)


def test_apply_non_numpy_dtype_category() -> None:
    df: DataFrame = DataFrame({"dt": ["a", "b", "c", "a"]}, dtype="category")
    result: DataFrame = df.apply(lambda x: x)
    tm.assert_frame_equal(result, df)


def test_apply_dup_names_multi_agg() -> None:
    # GH 21063
    df: DataFrame = DataFrame([[0, 1], [2, 3]], columns=["a", "a"])
    expected: DataFrame = DataFrame([[0, 1]], columns=["a", "a"], index=["min"])
    result: DataFrame = df.agg(["min"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("op", ["apply", "agg"])
def test_apply_nested_result_axis_1(op: str) -> None:
    # GH 13820
    def apply_list(row: Series) -> List[float]:
        return [2 * row["A"], 2 * row["C"], 2 * row["B"]]

    df: DataFrame = DataFrame(np.zeros((4, 4)), columns=list("ABCD"))
    result: Series = getattr(df, op)(apply_list, axis=1)
    expected: Series = Series(
        [[0.0, 0.0, 0.0] for _ in range(4)]
    )
    tm.assert_series_equal(result, expected)


def test_apply_noreduction_tzaware_object() -> None:
    # https://github.com/pandas-dev/pandas/issues/31505
    expected: DataFrame = DataFrame(
        {"foo": [Timestamp("2020", tz="UTC")]}, dtype="datetime64[ns, UTC]"
    )
    result: DataFrame = expected.apply(lambda x: x)
    tm.assert_frame_equal(result, expected)
    result = expected.apply(lambda x: x.copy())
    tm.assert_frame_equal(result, expected)


def test_apply_function_runs_once() -> None:
    # https://github.com/pandas-dev/pandas/issues/30815
    df: DataFrame = DataFrame({"a": [1, 2, 3]})
    names: List[Any] = []  # Save row names function is applied to

    def reducing_function(row: Series) -> None:
        names.append(row.name)

    def non_reducing_function(row: Series) -> Series:
        names.append(row.name)
        return row

    for func in [reducing_function, non_reducing_function]:
        del names[:]
        df.apply(func, axis=1)
        assert names == list(df.index)


def test_apply_raw_function_runs_once(engine: str) -> None:
    # https://github.com/pandas-dev/pandas/issues/34506
    if engine == "numba":
        pytest.skip("appending to list outside of numba func is not supported")
    df: DataFrame = DataFrame({"a": [1, 2, 3]})
    values: List[Any] = []  # Save row values function is applied to

    def reducing_function(row: np.ndarray) -> None:
        values.extend(row)

    def non_reducing_function(row: np.ndarray) -> np.ndarray:
        values.extend(row)
        return row

    for func in [reducing_function, non_reducing_function]:
        del values[:]
        df.apply(func, engine=engine, raw=True, axis=1)
        assert values == list(df.a.to_list())


def test_apply_with_byte_string() -> None:
    # GH 34529
    df: DataFrame = DataFrame(np.array([b"abcd", b"efgh"]), columns=["col"])
    expected: DataFrame = DataFrame(np.array([b"abcd", b"efgh"]), columns=["col"], dtype=object)
    result: DataFrame = df.apply(lambda x: x.astype("object"))
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("val", ["asd", 12, None, np.nan])
def test_apply_category_equalness(val: Union[str, int, None, float]) -> None:
    # Check if categorical comparisons on apply, GH 21239
    df_values: List[Any] = ["asd", None, 12, "asd", "cde", np.nan]
    df: DataFrame = DataFrame({"a": df_values}, dtype="category")
    result: Series = df.a.apply(lambda x: x == val)
    expected: Series = Series([False if pd.isnull(x) else x == val for x in df_values], name="a")
    tm.assert_series_equal(result, expected)


def test_infer_row_shape() -> None:
    # GH 17437
    df: DataFrame = DataFrame(np.random.default_rng(2).random((10, 2)))
    result_shape: Any = df.apply(np.fft.fft, axis=0).shape
    assert result_shape == (10, 2)
    result_shape = df.apply(np.fft.rfft, axis=0).shape
    assert result_shape == (6, 2)


@pytest.mark.parametrize(
    "ops, by_row, expected",
    [
        ({"a": lambda x: x + 1}, "compat", DataFrame({"a": [2, 3]})),
        ({"a": lambda x: x + 1}, False, DataFrame({"a": [2, 3]})),
        ({"a": lambda x: x.sum()}, "compat", Series({"a": 3})),
        ({"a": lambda x: x.sum()}, False, Series({"a": 3})),
        (
            {"a": ["sum", np.sum, lambda x: x.sum()]},
            "compat",
            DataFrame({"a": [3, 3, 3]}, index=["sum", "sum", "<lambda>"]),
        ),
        (
            {"a": ["sum", np.sum, lambda x: x.sum()]},
            False,
            DataFrame({"a": [3, 3, 3]}, index=["sum", "sum", "<lambda>"]),
        ),
        ({"a": lambda x: 1}, "compat", DataFrame({"a": [1, 1]})),
        ({"a": lambda x: 1}, False, Series({"a": 1})),
    ],
)
def test_dictlike_lambda(ops: Any, by_row: Union[str, bool], expected: Union[DataFrame, Series]) -> None:
    # GH53601
    df: DataFrame = DataFrame({"a": [1, 2]})
    result = df.apply(ops, by_row=by_row)
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "ops",
    [
        {"a": lambda x: x + 1},
        {"a": lambda x: x.sum()},
        {"a": ["sum", np.sum, lambda x: x.sum()]},
        {"a": lambda x: 1},
    ],
)
def test_dictlike_lambda_raises(ops: Any) -> None:
    # GH53601
    df: DataFrame = DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="by_row=True not allowed"):
        df.apply(ops, by_row=True)


def test_with_dictlike_columns() -> None:
    # GH 17602
    df: DataFrame = DataFrame([[1, 2], [1, 2]], columns=["a", "b"])
    result: Series = df.apply(lambda x: {"s": x["a"] + x["b"]}, axis=1)
    expected: Series = Series([{"s": 3} for _ in df.itertuples()])
    tm.assert_series_equal(result, expected)

    df["tm"] = [
        Timestamp("2017-05-01 00:00:00"),
        Timestamp("2017-05-02 00:00:00"),
    ]
    result = df.apply(lambda x: {"s": x["a"] + x["b"]}, axis=1)
    tm.assert_series_equal(result, expected)

    result = (df["a"] + df["b"]).apply(lambda x: {"s": x})
    expected = Series([{"s": 3}, {"s": 3}])
    tm.assert_series_equal(result, expected)


def test_with_dictlike_columns_with_datetime() -> None:
    # GH 18775
    df: DataFrame = DataFrame()
    df["author"] = ["X", "Y", "Z"]
    df["publisher"] = ["BBC", "NBC", "N24"]
    df["date"] = pd.to_datetime(
        ["17-10-2010 07:15:30", "13-05-2011 08:20:35", "15-01-2013 09:09:09"],
        dayfirst=True,
    )
    result: Series = df.apply(lambda x: {}, axis=1)
    expected: Series = Series([{}, {}, {}])
    tm.assert_series_equal(result, expected)


def test_with_dictlike_columns_with_infer() -> None:
    # GH 17602
    df: DataFrame = DataFrame([[1, 2], [1, 2]], columns=["a", "b"])
    result: DataFrame = df.apply(lambda x: {"s": x["a"] + x["b"]}, axis=1, result_type="expand")
    expected: DataFrame = DataFrame({"s": [3, 3]})
    tm.assert_frame_equal(result, expected)

    df["tm"] = [
        Timestamp("2017-05-01 00:00:00"),
        Timestamp("2017-05-02 00:00:00"),
    ]
    result = df.apply(lambda x: {"s": x["a"] + x["b"]}, axis=1, result_type="expand")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "ops, by_row, expected",
    [
        ([lambda x: x + 1], "compat", DataFrame({("a", "<lambda>"): [2, 3]})),
        ([lambda x: x + 1], False, DataFrame({("a", "<lambda>"): [2, 3]})),
        ([lambda x: x.sum()], "compat", DataFrame({"a": [3]}, index=["<lambda>"])),
        ([lambda x: x.sum()], False, DataFrame({"a": [3]}, index=["<lambda>"])),
        (
            ["sum", np.sum, lambda x: x.sum()],
            "compat",
            DataFrame({"a": [3, 3, 3]}, index=["sum", "sum", "<lambda>"]),
        ),
        (
            ["sum", np.sum, lambda x: x.sum()],
            False,
            DataFrame({"a": [3, 3, 3]}, index=["sum", "sum", "<lambda>"]),
        ),
        (
            [lambda x: x + 1, lambda x: 3],
            "compat",
            DataFrame([[2, 3], [3, 3]], columns=[["a", "a"], ["<lambda>", "<lambda>"]]),
        ),
        (
            [lambda x: 2, lambda x: 3],
            False,
            DataFrame({"a": [2, 3]}, index=["<lambda>", "<lambda>"]),
        ),
    ],
)
def test_listlike_lambda(ops: Any, by_row: Union[str, bool], expected: Union[DataFrame, Series]) -> None:
    # GH53601
    df: DataFrame = DataFrame({"a": [1, 2]})
    result = df.apply(ops, by_row=by_row)
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "ops",
    [
        [lambda x: x + 1],
        [lambda x: x.sum()],
        ["sum", np.sum, lambda x: x.sum()],
        [lambda x: x + 1, lambda x: 3],
    ],
)
def test_listlike_lambda_raises(ops: Any) -> None:
    # GH53601
    df: DataFrame = DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="by_row=True not allowed"):
        df.apply(ops, by_row=True)


def test_with_listlike_columns() -> None:
    # GH 17348
    df: DataFrame = DataFrame(
        {
            "a": Series(np.random.default_rng(2).standard_normal(4)),
            "b": ["a", "list", "of", "words"],
            "ts": date_range("2016-10-01", periods=4, freq="h"),
        }
    )

    result: Series = df[["a", "b"]].apply(tuple, axis=1)
    expected: Series = Series([t[1:] for t in df[["a", "b"]].itertuples()])
    tm.assert_series_equal(result, expected)

    result = df[["a", "ts"]].apply(tuple, axis=1)
    expected = Series([t[1:] for t in df[["a", "ts"]].itertuples()])
    tm.assert_series_equal(result, expected)


def test_with_listlike_columns_returning_list() -> None:
    # GH 18919
    df: DataFrame = DataFrame({"x": Series([["a", "b"], ["q"]]), "y": Series([["z"], ["q", "t"]])})
    df.index = MultiIndex.from_tuples([("i0", "j0"), ("i1", "j1")])
    result: Series = df.apply(lambda row: [el for el in row["x"] if el in row["y"]], axis=1)
    expected: Series = Series([[], ["q"]], index=df.index)
    tm.assert_series_equal(result, expected)


def test_infer_output_shape_columns() -> None:
    # GH 18573
    df: DataFrame = DataFrame(
        {
            "number": [1.0, 2.0],
            "string": ["foo", "bar"],
            "datetime": [
                Timestamp("2017-11-29 03:30:00"),
                Timestamp("2017-11-29 03:45:00"),
            ],
        }
    )
    result: Series = df.apply(lambda row: (row.number, row.string), axis=1)
    expected: Series = Series([(t.number, t.string) for t in df.itertuples()])
    tm.assert_series_equal(result, expected)


def test_infer_output_shape_listlike_columns() -> None:
    # GH 16353
    df: DataFrame = DataFrame(
        np.random.default_rng(2).standard_normal((6, 3)), columns=["A", "B", "C"]
    )
    result: Series = df.apply(lambda x: [1, 2, 3], axis=1)
    expected: Series = Series([[1, 2, 3] for _ in df.itertuples()])
    tm.assert_series_equal(result, expected)

    result = df.apply(lambda x: [1, 2], axis=1)
    expected = Series([[1, 2] for _ in df.itertuples()])
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("val", [1, 2])
def test_infer_output_shape_listlike_columns_np_func(val: int) -> None:
    # GH 17970
    df: DataFrame = DataFrame({"a": [1, 2, 3]}, index=list("abc"))
    result: Series = df.apply(lambda row: np.ones(val), axis=1)
    expected: Series = Series([np.ones(val) for _ in df.itertuples()], index=df.index)
    tm.assert_series_equal(result, expected)


def test_infer_output_shape_listlike_columns_with_timestamp() -> None:
    # GH 17892
    df: DataFrame = DataFrame(
        {
            "a": [
                Timestamp("2010-02-01"),
                Timestamp("2010-02-04"),
                Timestamp("2010-02-05"),
                Timestamp("2010-02-06"),
            ],
            "b": [9, 5, 4, 3],
            "c": [5, 3, 4, 2],
            "d": [1, 2, 3, 4],
        }
    )

    def fun(x: Any) -> tuple:
        return (1, 2)

    result: Series = df.apply(fun, axis=1)
    expected: Series = Series([(1, 2) for _ in df.itertuples()])
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("lst", [[1, 2, 3], [1, 2]])
def test_consistent_coerce_for_shapes(lst: List[int]) -> None:
    # we want column names to NOT be propagated
    df: DataFrame = DataFrame(
        np.random.default_rng(2).standard_normal((4, 3)), columns=["A", "B", "C"]
    )
    result: Series = df.apply(lambda x: lst, axis=1)
    expected: Series = Series([lst for _ in df.itertuples()])
    tm.assert_series_equal(result, expected)


def test_consistent_names(int_frame_const_col: DataFrame) -> None:
    # if a Series is returned, we should use the resulting index names
    df: DataFrame = int_frame_const_col
    result: DataFrame = df.apply(
        lambda x: Series([1, 2, 3], index=["test", "other", "cols"]), axis=1
    )
    expected: DataFrame = int_frame_const_col.rename(
        columns={"A": "test", "B": "other", "C": "cols"}
    )
    tm.assert_frame_equal(result, expected)

    result = df.apply(lambda x: Series([1, 2], index=["test", "other"]), axis=1)
    expected = expected[["test", "other"]]
    tm.assert_frame_equal(result, expected)


def test_result_type(int_frame_const_col: DataFrame) -> None:
    df: DataFrame = int_frame_const_col
    result: DataFrame = df.apply(lambda x: [1, 2, 3], axis=1, result_type="expand")
    expected: DataFrame = df.copy()
    expected.columns = range(3)
    tm.assert_frame_equal(result, expected)


def test_result_type_shorter_list(int_frame_const_col: DataFrame) -> None:
    df: DataFrame = int_frame_const_col
    result: DataFrame = df.apply(lambda x: [1, 2], axis=1, result_type="expand")
    expected: DataFrame = df[["A", "B"]].copy()
    expected.columns = range(2)
    tm.assert_frame_equal(result, expected)


def test_result_type_broadcast(int_frame_const_col: DataFrame, request: pytest.FixtureRequest, engine: str) -> None:
    if engine == "numba":
        mark = pytest.mark.xfail(reason="numba engine doesn't support list return")
        request.node.add_marker(mark)
    df: DataFrame = int_frame_const_col
    result: DataFrame = df.apply(
        lambda x: [1, 2, 3], axis=1, result_type="broadcast", engine=engine
    )
    expected: DataFrame = df.copy()
    tm.assert_frame_equal(result, expected)


def test_result_type_broadcast_series_func(int_frame_const_col: DataFrame, engine: str, request: pytest.FixtureRequest) -> None:
    if engine == "numba":
        mark = pytest.mark.xfail(
            reason="numba Series constructor only support ndarrays not list data"
        )
        request.node.add_marker(mark)
    df: DataFrame = int_frame_const_col
    columns: List[str] = ["other", "col", "names"]
    result: DataFrame = df.apply(
        lambda x: Series([1, 2, 3], index=columns),
        axis=1,
        result_type="broadcast",
        engine=engine,
    )
    expected: DataFrame = df.copy()
    tm.assert_frame_equal(result, expected)


def test_result_type_series_result(int_frame_const_col: DataFrame, engine: str, request: pytest.FixtureRequest) -> None:
    if engine == "numba":
        mark = pytest.mark.xfail(
            reason="numba Series constructor only support ndarrays not list data"
        )
        request.node.add_marker(mark)
    df: DataFrame = int_frame_const_col
    result: DataFrame = df.apply(lambda x: Series([1, 2, 3], index=x.index), axis=1, engine=engine)
    expected: DataFrame = df.copy()
    tm.assert_frame_equal(result, expected)


def test_result_type_series_result_other_index(int_frame_const_col: DataFrame, engine: str, request: pytest.FixtureRequest) -> None:
    if engine == "numba":
        mark = pytest.mark.xfail(
            reason="no support in numba Series constructor for list of columns"
        )
        request.node.add_marker(mark)
    df: DataFrame = int_frame_const_col
    columns: List[str] = ["other", "col", "names"]
    result: DataFrame = df.apply(lambda x: Series([1, 2, 3], index=columns), axis=1, engine=engine)
    expected: DataFrame = df.copy()
    expected.columns = columns
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "box",
    [lambda x: list(x), lambda x: tuple(x), lambda x: np.array(x, dtype="int64")],
    ids=["list", "tuple", "array"],
)
def test_consistency_for_boxed(box: Callable[[Any], Any], int_frame_const_col: DataFrame) -> None:
    df: DataFrame = int_frame_const_col
    result: Series = df.apply(lambda x: box([1, 2]), axis=1)
    expected: Series = Series([box([1, 2]) for _ in df.itertuples()])
    tm.assert_series_equal(result, expected)

    result = df.apply(lambda x: box([1, 2]), axis=1, result_type="expand")
    expected_df: DataFrame = int_frame_const_col[["A", "B"]].rename(columns={"A": 0, "B": 1})
    tm.assert_frame_equal(result, expected_df)


def test_agg_transform(axis: Union[int, str], float_frame: DataFrame) -> None:
    other_axis: Union[int, str] = 1 if axis in {0, "index"} else 0

    with np.errstate(all="ignore"):
        f_abs: DataFrame = np.abs(float_frame)
        f_sqrt: DataFrame = np.sqrt(float_frame)

        expected: DataFrame = f_sqrt.copy()
        result: DataFrame = float_frame.apply(np.sqrt, axis=axis)
        tm.assert_frame_equal(result, expected)

        result = float_frame.apply([np.sqrt], axis=axis)
        expected = f_sqrt.copy()
        if axis in {0, "index"}:
            expected.columns = MultiIndex.from_product([float_frame.columns, ["sqrt"]])
        else:
            expected.index = MultiIndex.from_product([float_frame.index, ["sqrt"]])
        tm.assert_frame_equal(result, expected)

        result = float_frame.apply([np.abs, np.sqrt], axis=axis)
        expected = zip_frames([f_abs, f_sqrt], axis=other_axis)
        if axis in {0, "index"}:
            expected.columns = MultiIndex.from_product(
                [float_frame.columns, ["absolute", "sqrt"]]
            )
        else:
            expected.index = MultiIndex.from_product(
                [float_frame.index, ["absolute", "sqrt"]]
            )
        tm.assert_frame_equal(result, expected)


def test_demo() -> None:
    df: DataFrame = DataFrame({"A": range(5), "B": 5})
    result: DataFrame = df.agg(["min", "max"])
    expected: DataFrame = DataFrame(
        {"A": [0, 4], "B": [5, 5]}, columns=["A", "B"], index=["min", "max"]
    )
    tm.assert_frame_equal(result, expected)


def test_demo_dict_agg() -> None:
    df: DataFrame = DataFrame({"A": range(5), "B": 5})
    result: DataFrame = df.agg({"A": ["min", "max"], "B": ["sum", "max"]})
    expected: DataFrame = DataFrame(
        {"A": [4.0, 0.0, np.nan], "B": [5.0, np.nan, 25.0]},
        columns=["A", "B"],
        index=["max", "min", "sum"],
    )
    tm.assert_frame_equal(result.reindex_like(expected), expected)


def test_agg_with_name_as_column_name() -> None:
    data: dict = {"name": ["foo", "bar"]}
    df: DataFrame = DataFrame(data)
    result: Series = df.agg({"name": "count"})
    expected: Series = Series({"name": 2})
    tm.assert_series_equal(result, expected)
    result = df["name"].agg({"name": "count"})
    expected = Series({"name": 2}, name="name")
    tm.assert_series_equal(result, expected)


def test_agg_multiple_mixed() -> None:
    mdf: DataFrame = DataFrame(
        {
            "A": [1, 2, 3],
            "B": [1.0, 2.0, 3.0],
            "C": ["foo", "bar", "baz"],
        }
    )
    expected: DataFrame = DataFrame(
        {
            "A": [1, 6],
            "B": [1.0, 6.0],
            "C": ["bar", "foobarbaz"],
        },
        index=["min", "sum"],
    )
    result: DataFrame = mdf.agg(["min", "sum"])
    tm.assert_frame_equal(result, expected)
    result = mdf[["C", "B", "A"]].agg(["sum", "min"])
    expected = expected[["C", "B", "A"]].reindex(["sum", "min"])
    tm.assert_frame_equal(result, expected)


def test_agg_multiple_mixed_raises() -> None:
    mdf: DataFrame = DataFrame(
        {
            "A": [1, 2, 3],
            "B": [1.0, 2.0, 3.0],
            "C": ["foo", "bar", "baz"],
            "D": date_range("20130101", periods=3),
        }
    )
    msg: str = "does not support operation"
    with pytest.raises(TypeError, match=msg):
        mdf.agg(["min", "sum"])
    with pytest.raises(TypeError, match=msg):
        mdf[["D", "C", "B", "A"]].agg(["sum", "min"])


def test_agg_reduce(axis: Union[int, str], float_frame: DataFrame) -> None:
    other_axis: Union[int, str] = 1 if axis in {0, "index"} else 0
    name1: Any = sorted(float_frame.axes[other_axis].unique()[:2])[0]
    name2: Any = sorted(float_frame.axes[other_axis].unique()[:2])[1]

    expected: DataFrame = pd.concat(
        [
            float_frame.mean(axis=axis),
            float_frame.max(axis=axis),
            float_frame.sum(axis=axis),
        ],
        axis=1,
    )
    expected.columns = ["mean", "max", "sum"]
    if axis in {0, "index"}:
        expected = expected.T
    result: DataFrame = float_frame.agg(["mean", "max", "sum"], axis=axis)
    tm.assert_frame_equal(result, expected)

    func: dict = {name1: "mean", name2: "sum"}
    result = float_frame.agg(func, axis=axis)
    expected_series: Series = Series(
        [
            float_frame.loc(other_axis)[name1].mean(),
            float_frame.loc(other_axis)[name2].sum(),
        ],
        index=[name1, name2],
    )
    tm.assert_series_equal(result, expected_series)

    func = {name1: ["mean"], name2: ["sum"]}
    result = float_frame.agg(func, axis=axis)
    expected_df: DataFrame = DataFrame(
        {
            name1: Series([float_frame.loc(other_axis)[name1].mean()], index=["mean"]),
            name2: Series([float_frame.loc(other_axis)[name2].sum()], index=["sum"]),
        }
    )
    if axis in {1, "columns"}:
        expected_df = expected_df.T
    tm.assert_frame_equal(result, expected_df)

    func = {name1: ["mean", "sum"], name2: ["sum", "max"]}
    result = float_frame.agg(func, axis=axis)
    expected = pd.concat(
        {
            name1: Series(
                [
                    float_frame.loc(other_axis)[name1].mean(),
                    float_frame.loc(other_axis)[name1].sum(),
                ],
                index=["mean", "sum"],
            ),
            name2: Series(
                [
                    float_frame.loc(other_axis)[name2].sum(),
                    float_frame.loc(other_axis)[name2].max(),
                ],
                index=["sum", "max"],
            ),
        },
        axis=1,
    )
    if axis in {1, "columns"}:
        expected = expected.T
    tm.assert_frame_equal(result, expected)


def test_named_agg_reduce_axis1_raises(float_frame: DataFrame) -> None:
    name1: Any = sorted(float_frame.axes[0].unique()[:2])[0]
    name2: Any = sorted(float_frame.axes[0].unique()[:2])[1]
    msg: str = "Named aggregation is not supported when axis=1."
    for axis in [1, "columns"]:
        with pytest.raises(NotImplementedError, match=msg):
            float_frame.agg(row1=(name1, "sum"), row2=(name2, "max"), axis=axis)


def test_nuiscance_columns() -> None:
    df: DataFrame = DataFrame(
        {
            "A": [1, 2, 3],
            "B": [1.0, 2.0, 3.0],
            "C": ["foo", "bar", "baz"],
            "D": date_range("20130101", periods=3),
        }
    )
    result: Series = df.agg("min")
    expected: Series = Series([1, 1.0, "bar", Timestamp("20130101")], index=df.columns)
    tm.assert_series_equal(result, expected)
    result = df.agg(["min"])
    expected = DataFrame(
        [[1, 1.0, "bar", Timestamp("20130101").as_unit("ns")]],
        index=["min"],
        columns=df.columns,
    )
    tm.assert_frame_equal(result, expected)
    msg: str = "does not support operation"
    with pytest.raises(TypeError, match=msg):
        df.agg("sum")
    result = df[["A", "B", "C"]].agg("sum")
    expected = Series([6, 6.0, "foobarbaz"], index=["A", "B", "C"])
    tm.assert_series_equal(result, expected)
    with pytest.raises(TypeError, match=msg):
        df.agg(["sum"])


@pytest.mark.parametrize("how", ["agg", "apply"])
def test_non_callable_aggregates(how: str) -> None:
    df: DataFrame = DataFrame(
        {"A": [None, 2, 3], "B": [1.0, np.nan, 3.0], "C": ["foo", None, "bar"]}
    )
    result = getattr(df, how)({"A": "count"})
    expected: Series = Series({"A": 2})
    tm.assert_series_equal(result, expected)
    result = getattr(df, how)({"A": "size"})
    expected = Series({"A": 3})
    tm.assert_series_equal(result, expected)
    result1 = getattr(df, how)(["count", "size"])
    result2 = getattr(df, how)(
        {"A": ["count", "size"], "B": ["count", "size"], "C": ["count", "size"]}
    )
    expected = DataFrame(
        {
            "A": {"count": 2, "size": 3},
            "B": {"count": 2, "size": 3},
            "C": {"count": 2, "size": 3},
        }
    )
    tm.assert_frame_equal(result1, result2, check_like=True)
    tm.assert_frame_equal(result2, expected, check_like=True)
    result = getattr(df, how)("count")
    expected = df.count()
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("how", ["agg", "apply"])
def test_size_as_str(how: str, axis: Union[int, str]) -> None:
    df: DataFrame = DataFrame(
        {"A": [None, 2, 3], "B": [1.0, np.nan, 3.0], "C": ["foo", None, "bar"]}
    )
    result = getattr(df, how)("size", axis=axis)
    if axis in (0, "index"):
        expected = Series(df.shape[0], index=df.columns)
    else:
        expected = Series(df.shape[1], index=df.index)
    tm.assert_series_equal(result, expected)


def test_agg_listlike_result() -> None:
    df: DataFrame = DataFrame({"A": [2, 2, 3], "B": [1.5, np.nan, 1.5], "C": ["foo", None, "bar"]})
    def func(group_col: Series) -> List[Any]:
        return list(group_col.dropna().unique())
    result = df.agg(func)
    expected = Series([[2, 3], [1.5], ["foo", "bar"]], index=["A", "B", "C"])
    tm.assert_series_equal(result, expected)
    result = df.agg([func])
    expected = expected.to_frame("func").T
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize(
    "args, kwargs",
    [
        ((1, 2, 3), {}),
        ((8, 7, 15), {}),
        ((1, 2), {}),
        ((1,), {"b": 2}),
        ((), {"a": 1, "b": 2}),
        ((), {"a": 2, "b": 1}),
        ((), {"a": 1, "b": 2, "c": 3}),
    ],
)
def test_agg_args_kwargs(axis: Union[int, str], args: tuple, kwargs: dict) -> None:
    def f(x: Series, a: int, b: int, c: int = 3) -> Any:
        return x.sum() + (a + b) / c
    df: DataFrame = DataFrame([[1, 2], [3, 4]])
    if axis == 0:
        expected = Series([5.0, 7.0])
    else:
        expected = Series([4.0, 8.0])
    result = df.agg(f, axis, *args, **kwargs)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("num_cols", [2, 3, 5])
def test_frequency_is_original(num_cols: int, engine: str, request: pytest.FixtureRequest) -> None:
    if engine == "numba":
        mark = pytest.mark.xfail(reason="numba engine only supports numeric indices")
        request.node.add_marker(mark)
    index: pd.DatetimeIndex = pd.DatetimeIndex(["1950-06-30", "1952-10-24", "1953-05-29"])
    original: pd.DatetimeIndex = index.copy()
    df: DataFrame = DataFrame(1, index=index, columns=range(num_cols))
    df.apply(lambda x: x, engine=engine)
    assert index.freq == original.freq


def test_apply_datetime_tz_issue(engine: str, request: pytest.FixtureRequest) -> None:
    if engine == "numba":
        mark = pytest.mark.xfail(
            reason="numba engine doesn't support non-numeric indexes"
        )
        request.node.add_marker(mark)
    timestamps: List[Timestamp] = [
        Timestamp("2019-03-15 12:34:31.909000+0000", tz="UTC"),
        Timestamp("2019-03-15 12:34:34.359000+0000", tz="UTC"),
        Timestamp("2019-03-15 12:34:34.660000+0000", tz="UTC"),
    ]
    df: DataFrame = DataFrame(data=[0, 1, 2], index=timestamps)
    result: Series = df.apply(lambda x: x.name, axis=1, engine=engine)
    expected: Series = Series(index=timestamps, data=timestamps)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("df", [DataFrame({"A": ["a", None], "B": ["c", "d"]})])
@pytest.mark.parametrize("method", ["min", "max", "sum"])
def test_mixed_column_raises(df: DataFrame, method: str, using_infer_string: bool) -> None:
    if method == "sum":
        msg: str = r'can only concatenate str \(not "int"\) to str|does not support'
    else:
        msg = "not supported between instances of 'str' and 'float'"
    if not using_infer_string:
        with pytest.raises(TypeError, match=msg):
            getattr(df, method)()
    else:
        getattr(df, method)()


@pytest.mark.parametrize("col", [1, 1.0, True, "a", np.nan])
def test_apply_dtype(col: Any) -> None:
    df: DataFrame = DataFrame([[1.0, col]], columns=["a", "b"])
    result: Series = df.apply(lambda x: x.dtype)
    expected: Series = df.dtypes
    tm.assert_series_equal(result, expected)


def test_apply_mutating() -> None:
    df: DataFrame = DataFrame({"a": range(10), "b": range(10, 20)})
    df_orig: DataFrame = df.copy()

    def func(row: Series) -> Series:
        mgr = row._mgr
        row.loc["a"] += 1
        assert row._mgr is not mgr
        return row

    expected: DataFrame = df.copy()
    expected["a"] += 1

    result: DataFrame = df.apply(func, axis=1)
    tm.assert_frame_equal(result, expected)
    tm.assert_frame_equal(df, df_orig)


def test_apply_empty_list_reduce() -> None:
    df: DataFrame = DataFrame([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], columns=["a", "b"])
    result: Series = df.apply(lambda x: [], result_type="reduce")
    expected: Series = Series({"a": [], "b": []}, dtype=object)
    tm.assert_series_equal(result, expected)


def test_apply_no_suffix_index(engine: str, request: pytest.FixtureRequest) -> None:
    if engine == "numba":
        mark = pytest.mark.xfail(
            reason="numba engine doesn't support list-likes/dict-like callables"
        )
        request.node.add_marker(mark)
    pdf: DataFrame = DataFrame([[4, 9]] * 3, columns=["A", "B"])
    result: DataFrame = pdf.apply(["sum", lambda x: x.sum(), lambda x: x.sum()], engine=engine)
    expected: DataFrame = DataFrame(
        {"A": [12, 12, 12], "B": [27, 27, 27]}, index=["sum", "<lambda>", "<lambda>"]
    )
    tm.assert_frame_equal(result, expected)


def test_apply_raw_returns_string(engine: str) -> None:
    if engine == "numba":
        pytest.skip("No object dtype support in numba")
    df: DataFrame = DataFrame({"A": ["aa", "bbb"]})
    result: Series = df.apply(lambda x: x[0], engine=engine, axis=1, raw=True)
    expected: Series = Series(["aa", "bbb"])
    tm.assert_series_equal(result, expected)


def test_aggregation_func_column_order() -> None:
    df: DataFrame = DataFrame(
        [
            (1, 0, 0),
            (2, 0, 0),
            (3, 0, 0),
            (4, 5, 4),
            (5, 6, 6),
            (6, 7, 7),
        ],
        columns=("att1", "att2", "att3"),
    )

    def sum_div2(s: Series) -> float:
        return s.sum() / 2

    aggs: List[Any] = ["sum", sum_div2, "count", "min"]
    result: DataFrame = df.agg(aggs)
    expected: DataFrame = DataFrame(
        {
            "att1": [21.0, 10.5, 6.0, 1.0],
            "att2": [18.0, 9.0, 6.0, 0.0],
            "att3": [17.0, 8.5, 6.0, 0.0],
        },
        index=["sum", "sum_div2", "count", "min"],
    )
    tm.assert_frame_equal(result, expected)


def test_apply_getitem_axis_1(engine: str, request: pytest.FixtureRequest) -> None:
    if engine == "numba":
        mark = pytest.mark.xfail(
            reason="numba engine not supporting duplicate index values"
        )
        request.node.add_marker(mark)
    df: DataFrame = DataFrame({"a": [0, 1, 2], "b": [1, 2, 3]})
    result: Series = df[["a", "a"]].apply(
        lambda x: x.iloc[0] + x.iloc[1], axis=1, engine=engine
    )
    expected: Series = Series([0, 2, 4])
    tm.assert_series_equal(result, expected)


def test_nuiscance_depr_passes_through_warnings() -> None:
    def expected_warning(x: Series) -> Any:
        warnings.warn("Hello, World!")
        return x.sum()
    df: DataFrame = DataFrame({"a": [1, 2, 3]})
    with tm.assert_produces_warning(UserWarning, match="Hello, World!"):
        df.agg([expected_warning])


def test_apply_type() -> None:
    df: DataFrame = DataFrame(
        {"col1": [3, "string", float], "col2": [0.25, datetime(2020, 1, 1), np.nan]},
        index=["a", "b", "c"],
    )
    result: Series = df.apply(type, axis=0)
    expected: Series = Series({"col1": Series, "col2": Series})
    tm.assert_series_equal(result, expected)
    result = df.apply(type, axis=1)
    expected = Series({"a": Series, "b": Series, "c": Series})
    tm.assert_series_equal(result, expected)


def test_apply_on_empty_dataframe(engine: str) -> None:
    df: DataFrame = DataFrame({"a": [1, 2], "b": [3, 0]})
    result: Series = df.head(0).apply(lambda x: max(x["a"], x["b"]), axis=1, engine=engine)
    expected: Series = Series([], dtype=np.float64)
    tm.assert_series_equal(result, expected)


def test_apply_return_list() -> None:
    df: DataFrame = DataFrame({"a": [1, 2], "b": [2, 3]})
    result: DataFrame = df.apply(lambda x: [x.values])
    expected: DataFrame = DataFrame({"a": [[1, 2]], "b": [[2, 3]]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "test, constant",
    [
        ({"a": [1, 2, 3], "b": [1, 1, 1]}, {"a": [1, 2, 3], "b": [1]}),
        ({"a": [2, 2, 2], "b": [1, 1, 1]}, {"a": [2], "b": [1]}),
    ],
)
def test_unique_agg_type_is_series(test: dict, constant: dict) -> None:
    df1: DataFrame = DataFrame(test)
    expected: Series = Series(data=constant, index=["a", "b"], dtype="object")
    aggregation: dict = {"a": "unique", "b": "unique"}
    result: Series = df1.agg(aggregation)
    tm.assert_series_equal(result, expected)


def test_any_apply_keyword_non_zero_axis_regression() -> None:
    df: DataFrame = DataFrame({"A": [1, 2, 0], "B": [0, 2, 0], "C": [0, 0, 0]})
    expected: Series = Series([True, True, False])
    tm.assert_series_equal(df.any(axis=1), expected)
    result: Series = df.apply("any", axis=1)
    tm.assert_series_equal(result, expected)
    result = df.apply("any", 1)
    tm.assert_series_equal(result, expected)


def test_agg_mapping_func_deprecated() -> None:
    df: DataFrame = DataFrame({"x": [1, 2, 3]})
    def foo1(x: Series, a: int = 1, c: int = 0) -> Any:
        return x + a + c
    def foo2(x: Series, b: int = 2, c: int = 0) -> Any:
        return x + b + c
    result: DataFrame = df.agg(foo1, 0, 3, c=4)
    expected: DataFrame = df + 7
    tm.assert_frame_equal(result, expected)
    result = df.agg([foo1, foo2], 0, 3, c=4)
    expected = DataFrame(
        [[8, 8], [9, 9], [10, 10]], columns=[["x", "x"], ["foo1", "foo2"]]
    )
    tm.assert_frame_equal(result, expected)
    result = df.agg({"x": foo1}, 0, 3, c=4)
    expected = DataFrame([2, 3, 4], columns=["x"])
    tm.assert_frame_equal(result, expected)


def test_agg_std() -> None:
    df: DataFrame = DataFrame(np.arange(6).reshape(3, 2), columns=["A", "B"])
    result: Series = df.agg(np.std, ddof=1)
    expected: Series = Series({"A": 2.0, "B": 2.0}, dtype=float)
    tm.assert_series_equal(result, expected)
    result = df.agg([np.std], ddof=1)
    expected = DataFrame({"A": 2.0, "B": 2.0}, index=["std"])
    tm.assert_frame_equal(result, expected)


def test_agg_dist_like_and_nonunique_columns() -> None:
    df: DataFrame = DataFrame(
        {"A": [None, 2, 3], "B": [1.0, np.nan, 3.0], "C": ["foo", None, "bar"]}
    )
    df.columns = ["A", "A", "C"]
    result: Series = df.agg({"A": "count"})
    expected = df["A"].count()
    tm.assert_series_equal(result, expected)


# End of annotated tests.
