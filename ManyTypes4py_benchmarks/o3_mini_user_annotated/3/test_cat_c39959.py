from typing import Any, Callable, List, Optional, Type, Union
import re

import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    _testing as tm,
    concat,
    option_context,
)

IndexOrSeriesFunc = Callable[[List[str]], Union[Series, Index]]


@pytest.fixture
def index_or_series2(index_or_series: IndexOrSeriesFunc) -> IndexOrSeriesFunc:
    return index_or_series


@pytest.mark.parametrize("other", [None, Series, Index])
def test_str_cat_name(index_or_series: IndexOrSeriesFunc, other: Optional[Type[Union[Series, Index]]]) -> None:
    # GH 21053
    box = index_or_series
    values: List[str] = ["a", "b"]
    if other:
        other = other(values)
    else:
        other = values
    result = box(values, name="name").str.cat(other, sep=",")
    assert result.name == "name"


@pytest.mark.parametrize("infer_string", [False, pytest.param(True, marks=td.skip_if_no("pyarrow"))])
def test_str_cat(index_or_series: IndexOrSeriesFunc, infer_string: bool) -> None:
    with option_context("future.infer_string", infer_string):
        box = index_or_series
        # test_cat above tests "str_cat" from ndarray;
        # here testing "str.cat" from Series/Index to ndarray/list
        s = box(["a", "a", "b", "b", "c", np.nan])

        # single array
        result: str = s.str.cat()
        expected: str = "aabbc"
        assert result == expected

        result = s.str.cat(na_rep="-")
        expected = "aabbc-"
        assert result == expected

        result = s.str.cat(sep="_", na_rep="NA")
        expected = "a_a_b_b_c_NA"
        assert result == expected

        t: np.ndarray = np.array(["a", np.nan, "b", "d", "foo", np.nan], dtype=object)
        expected_box = box(["aa", "a-", "bb", "bd", "cfoo", "--"])

        # Series/Index with array
        result = s.str.cat(t, na_rep="-")
        tm.assert_equal(result, expected_box)

        # Series/Index with list
        result = s.str.cat(list(t), na_rep="-")
        tm.assert_equal(result, expected_box)

        # errors for incorrect lengths
        rgx: str = r"If `others` contains arrays or lists \(or other list-likes.*"
        z: Series = Series(["1", "2", "3"])

        with pytest.raises(ValueError, match=rgx):
            s.str.cat(z.values)

        with pytest.raises(ValueError, match=rgx):
            s.str.cat(list(z))


def test_str_cat_raises_intuitive_error(index_or_series: IndexOrSeriesFunc) -> None:
    # GH 11334
    box = index_or_series
    s = box(["a", "b", "c", "d"])
    message: str = "Did you mean to supply a `sep` keyword?"
    with pytest.raises(ValueError, match=message):
        s.str.cat("|")
    with pytest.raises(ValueError, match=message):
        s.str.cat("    ")


@pytest.mark.parametrize(
    "infer_string", [False, pytest.param(True, marks=td.skip_if_no("pyarrow"))]
)
@pytest.mark.parametrize("sep", ["", None])
@pytest.mark.parametrize("dtype_target", ["object", "category"])
@pytest.mark.parametrize("dtype_caller", ["object", "category"])
def test_str_cat_categorical(
    index_or_series: IndexOrSeriesFunc,
    dtype_caller: str,
    dtype_target: str,
    sep: Optional[str],
    infer_string: bool,
) -> None:
    box = index_or_series

    with option_context("future.infer_string", infer_string):
        s = Index(["a", "a", "b", "a"], dtype=dtype_caller)
        s = s if box == Index else Series(s, index=s, dtype=s.dtype)  # type: ignore
        t = Index(["b", "a", "b", "c"], dtype=dtype_target)

        expected = Index(
            ["ab", "aa", "bb", "ac"], dtype=object if dtype_caller == "object" else None
        )
        expected = (
            expected
            if box == Index
            else Series(
                expected, index=Index(s, dtype=dtype_caller), dtype=expected.dtype
            )
        )

        # Series/Index with unaligned Index -> t.values
        result = s.str.cat(t.values, sep=sep)
        tm.assert_equal(result, expected)

        # Series/Index with Series having matching Index
        t_series = Series(t.values, index=Index(s, dtype=dtype_caller))
        result = s.str.cat(t_series, sep=sep)
        tm.assert_equal(result, expected)

        # Series/Index with Series.values
        result = s.str.cat(t_series.values, sep=sep)
        tm.assert_equal(result, expected)

        # Series/Index with Series having different Index
        t_diff = Series(t.values, index=t.values)
        expected = Index(
            ["aa", "aa", "bb", "bb", "aa"],
            dtype=object if dtype_caller == "object" else None,
        )
        dtype_val = object if dtype_caller == "object" else s.dtype.categories.dtype  # type: ignore
        expected = (
            expected
            if box == Index
            else Series(
                expected,
                index=Index(expected.str[:1], dtype=dtype_val),  # type: ignore
                dtype=expected.dtype,
            )
        )

        result = s.str.cat(t_diff, sep=sep)
        tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "data",
    [[1, 2, 3], [0.1, 0.2, 0.3], [1, 2, "b"]],
    ids=["integers", "floats", "mixed"],
)
@pytest.mark.parametrize(
    "box",
    [Series, Index, list, lambda x: np.array(x, dtype=object)],
    ids=["Series", "Index", "list", "np.array"],
)
def test_str_cat_wrong_dtype_raises(box: Callable[[List[Any]], Any], data: List[Union[int, float, str]]) -> None:
    # GH 22722
    s: Series = Series(["a", "b", "c"])
    t = box(data)

    msg: str = "Concatenation requires list-likes containing only strings.*"
    with pytest.raises(TypeError, match=msg):
        # need to use outer and na_rep, as otherwise Index would not raise
        s.str.cat(t, join="outer", na_rep="-")


def test_str_cat_mixed_inputs(index_or_series: IndexOrSeriesFunc) -> None:
    box = index_or_series
    s_val: List[str] = ["a", "b", "c", "d"]
    s = Index(s_val)
    s = s if box == Index else Series(s, index=s)

    t: Series = Series(["A", "B", "C", "D"], index=s.values)
    d: DataFrame = concat([t, Series(s, index=s)], axis=1)

    expected = Index(["aAa", "bBb", "cCc", "dDd"])
    expected = expected if box == Index else Series(expected.values, index=s.values)

    # Series/Index with DataFrame
    result = s.str.cat(d)
    tm.assert_equal(result, expected)

    # Series/Index with two-dimensional ndarray
    result = s.str.cat(d.values)
    tm.assert_equal(result, expected)

    # Series/Index with list of Series
    result = s.str.cat([t, s])
    tm.assert_equal(result, expected)

    # Series/Index with mixed list of Series/array
    result = s.str.cat([t, s.values])
    tm.assert_equal(result, expected)

    # Series/Index with list of Series; different indexes
    t_diff = t.copy()
    t_diff.index = ["b", "c", "d", "a"]
    expected = box(["aDa", "bAb", "cBc", "dCd"])
    expected = expected if box == Index else Series(expected.values, index=s.values)
    result = s.str.cat([t_diff, s])
    tm.assert_equal(result, expected)

    # Series/Index with mixed list; different index
    result = s.str.cat([t_diff, s.values])
    tm.assert_equal(result, expected)

    # Series/Index with DataFrame; different indexes
    d_diff = d.copy()
    d_diff.index = ["b", "c", "d", "a"]
    expected = box(["aDd", "bAa", "cBb", "dCc"])
    expected = expected if box == Index else Series(expected.values, index=s.values)
    result = s.str.cat(d_diff)
    tm.assert_equal(result, expected)

    # errors for incorrect lengths
    rgx: str = r"If `others` contains arrays or lists \(or other list-likes.*"
    z: Series = Series(["1", "2", "3"])
    e: DataFrame = concat([z, z], axis=1)

    # two-dimensional ndarray
    with pytest.raises(ValueError, match=rgx):
        s.str.cat(e.values)

    # list of list-likes
    with pytest.raises(ValueError, match=rgx):
        s.str.cat([z.values, s.values])

    # mixed list of Series/list-like
    with pytest.raises(ValueError, match=rgx):
        s.str.cat([z.values, s])

    # errors for incorrect arguments in list-like
    rgx2: str = "others must be Series, Index, DataFrame,.*"
    # make sure None/NaN do not crash checks in _get_series_list
    u: Series = Series(["a", np.nan, "c", None])

    # mix of string and Series
    with pytest.raises(TypeError, match=rgx2):
        s.str.cat([u, "u"])

    # DataFrame in list
    with pytest.raises(TypeError, match=rgx2):
        s.str.cat([u, d])

    # 2-dim ndarray in list
    with pytest.raises(TypeError, match=rgx2):
        s.str.cat([u, d.values])

    # nested lists
    with pytest.raises(TypeError, match=rgx2):
        s.str.cat([u, [u, d]])

    # forbidden input type: set
    # GH 23009
    with pytest.raises(TypeError, match=rgx2):
        s.str.cat(set(u))

    # forbidden input type: set in list
    # GH 23009
    with pytest.raises(TypeError, match=rgx2):
        s.str.cat([u, set(u)])

    # other forbidden input type, e.g. int
    with pytest.raises(TypeError, match=rgx2):
        s.str.cat(1)

    # nested list-likes
    with pytest.raises(TypeError, match=rgx2):
        s.str.cat(iter([t.values, list(s)]))


def test_str_cat_align_indexed(index_or_series: IndexOrSeriesFunc, join_type: str) -> None:
    # https://github.com/pandas-dev/pandas/issues/18657
    box = index_or_series

    s: Series = Series(["a", "b", "c", "d"], index=["a", "b", "c", "d"])
    t: Series = Series(["D", "A", "E", "B"], index=["d", "a", "e", "b"])
    sa, ta = s.align(t, join=join_type)
    # result after manual alignment of inputs
    expected = sa.str.cat(ta, na_rep="-")

    if box == Index:
        s = Index(s)
        sa = Index(sa)
        expected = Index(expected)

    result = s.str.cat(t, join=join_type, na_rep="-")
    tm.assert_equal(result, expected)


def test_str_cat_align_mixed_inputs(join_type: str) -> None:
    s: Series = Series(["a", "b", "c", "d"])
    t: Series = Series(["d", "a", "e", "b"], index=[3, 0, 4, 1])
    d: DataFrame = concat([t, t], axis=1)

    expected_outer: Series = Series(["aaa", "bbb", "c--", "ddd", "-ee"])
    expected = expected_outer.loc[s.index.join(t.index, how=join_type)]

    # list of Series
    result = s.str.cat([t, t], join=join_type, na_rep="-")
    tm.assert_series_equal(result, expected)

    # DataFrame
    result = s.str.cat(d, join=join_type, na_rep="-")
    tm.assert_series_equal(result, expected)

    # mixed list of indexed/unindexed
    u: np.ndarray = np.array(["A", "B", "C", "D"])
    expected_outer = Series(["aaA", "bbB", "c-C", "ddD", "-e-"])
    # joint index of rhs [t, u]; u will be forced have index of s
    rhs_idx = (
        t.index.intersection(s.index)
        if join_type == "inner"
        else t.index.union(s.index)
        if join_type == "outer"
        else t.index.append(s.index.difference(t.index))
    )

    expected = expected_outer.loc[s.index.join(rhs_idx, how=join_type)]
    result = s.str.cat([t, u], join=join_type, na_rep="-")
    tm.assert_series_equal(result, expected)

    with pytest.raises(TypeError, match="others must be Series,.*"):
        # nested lists are forbidden
        s.str.cat([t, list(u)])

    # errors for incorrect lengths
    rgx: str = r"If `others` contains arrays or lists \(or other list-likes.*"
    z: np.ndarray = Series(["1", "2", "3"]).values

    # unindexed object of wrong length
    with pytest.raises(ValueError, match=rgx):
        s.str.cat(z, join=join_type)

    # unindexed object of wrong length in list
    with pytest.raises(ValueError, match=rgx):
        s.str.cat([t, z], join=join_type)


def test_str_cat_all_na(index_or_series: IndexOrSeriesFunc, index_or_series2: IndexOrSeriesFunc) -> None:
    # GH 24044
    box = index_or_series
    other = index_or_series2

    # check that all NaNs in caller / target work
    s_vals: List[str] = ["a", "b", "c", "d"]
    s = Index(s_vals)
    s = s if box == Index else Series(s, index=s)
    t = other([np.nan] * 4, dtype=object)
    # add index of s for alignment
    t = t if other == Index else Series(t, index=s)

    # all-NA target
    if box == Series:
        expected = Series([np.nan] * 4, index=s.index, dtype=s.dtype)
    else:  # box == Index
        expected = Index([np.nan] * 4, dtype=object)
    result = s.str.cat(t, join="left")
    tm.assert_equal(result, expected)

    # all-NA caller (only for Series)
    if other == Series:
        expected = Series([np.nan] * 4, dtype=object, index=t.index)
        result = t.str.cat(s, join="left")
        tm.assert_series_equal(result, expected)


def test_str_cat_special_cases() -> None:
    s: Series = Series(["a", "b", "c", "d"])
    t: Series = Series(["d", "a", "e", "b"], index=[3, 0, 4, 1])

    # iterator of elements with different types
    expected: Series = Series(["aaa", "bbb", "c-c", "ddd", "-e-"])
    result = s.str.cat(iter([t, s.values]), join="outer", na_rep="-")
    tm.assert_series_equal(result, expected)

    # right-align with different indexes in others
    expected = Series(["aa-", "d-d"], index=[0, 3])
    result = s.str.cat([t.loc[[0]], t.loc[[3]]], join="right", na_rep="-")
    tm.assert_series_equal(result, expected)


def test_cat_on_filtered_index() -> None:
    df: DataFrame = DataFrame(
        index=MultiIndex.from_product(
            [[2011, 2012], [1, 2, 3]], names=["year", "month"]
        )
    )

    df = df.reset_index()
    df = df[df.month > 1]

    str_year: Series = df.year.astype("str")
    str_month: Series = df.month.astype("str")
    str_both = str_year.str.cat(str_month, sep=" ")

    assert str_both.loc[1] == "2011 2"

    str_multiple = str_year.str.cat([str_month, str_month], sep=" ")

    assert str_multiple.loc[1] == "2011 2 2"


@pytest.mark.parametrize("klass", [tuple, list, np.array, Series, Index])
def test_cat_different_classes(klass: Type[Any]) -> None:
    # https://github.com/pandas-dev/pandas/issues/33425
    s: Series = Series(["a", "b", "c"])
    result = s.str.cat(klass(["x", "y", "z"]))
    expected = Series(["ax", "by", "cz"])
    tm.assert_series_equal(result, expected)


def test_cat_on_series_dot_str() -> None:
    # GH 28277
    ps: Series = Series(["AbC", "de", "FGHI", "j", "kLLLm"])

    message = re.escape(
        "others must be Series, Index, DataFrame, np.ndarray "
        "or list-like (either containing only strings or "
        "containing only objects of type Series/Index/"
        "np.ndarray[1-dim])"
    )
    with pytest.raises(TypeError, match=message):
        ps.str.cat(others=ps.str)