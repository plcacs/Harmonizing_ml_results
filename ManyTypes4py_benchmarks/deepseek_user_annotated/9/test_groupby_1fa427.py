from datetime import datetime
import decimal
from decimal import Decimal
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pytest

from pandas.errors import SpecificationError
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    Grouper,
    Index,
    Interval,
    MultiIndex,
    RangeIndex,
    Series,
    Timedelta,
    Timestamp,
    date_range,
    to_datetime,
)
import pandas._testing as tm
from pandas.core.arrays import BooleanArray
import pandas.core.common as com

pytestmark = pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")


def test_repr() -> None:
    # GH18203
    result = repr(Grouper(key="A", level="B"))
    expected = "Grouper(key='A', level='B', sort=False, dropna=True)"
    assert result == expected


def test_groupby_nonobject_dtype(multiindex_dataframe_random_data: DataFrame) -> None:
    key = multiindex_dataframe_random_data.index.codes[0]
    grouped = multiindex_dataframe_random_data.groupby(key)
    result = grouped.sum()

    expected = multiindex_dataframe_random_data.groupby(key.astype("O")).sum()
    assert result.index.dtype == np.int8
    assert expected.index.dtype == np.int64
    tm.assert_frame_equal(result, expected, check_index_type=False)


def test_groupby_nonobject_dtype_mixed() -> None:
    # GH 3911, mixed frame non-conversion
    df = DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
            "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
            "C": np.random.default_rng(2).standard_normal(8),
            "D": np.array(np.random.default_rng(2).standard_normal(8), dtype="float32"),
        }
    )
    df["value"] = range(len(df))

    def max_value(group: DataFrame) -> DataFrame:
        return group.loc[group["value"].idxmax()]

    applied = df.groupby("A").apply(max_value)
    result = applied.dtypes
    expected = df.drop(columns="A").dtypes
    tm.assert_series_equal(result, expected)


def test_pass_args_kwargs(ts: Series) -> None:
    def f(x: np.ndarray, q: Optional[float] = None, axis: int = 0) -> np.ndarray:
        return np.percentile(x, q, axis=axis)

    g = lambda x: np.percentile(x, 80, axis=0)

    # Series
    ts_grouped = ts.groupby(lambda x: x.month)
    agg_result = ts_grouped.agg(np.percentile, 80, axis=0)
    apply_result = ts_grouped.apply(np.percentile, 80, axis=0)
    trans_result = ts_grouped.transform(np.percentile, 80, axis=0)

    agg_expected = ts_grouped.quantile(0.8)
    trans_expected = ts_grouped.transform(g)

    tm.assert_series_equal(apply_result, agg_expected)
    tm.assert_series_equal(agg_result, agg_expected)
    tm.assert_series_equal(trans_result, trans_expected)

    agg_result = ts_grouped.agg(f, q=80)
    apply_result = ts_grouped.apply(f, q=80)
    trans_result = ts_grouped.transform(f, q=80)
    tm.assert_series_equal(agg_result, agg_expected)
    tm.assert_series_equal(apply_result, agg_expected)
    tm.assert_series_equal(trans_result, trans_expected)


def test_pass_args_kwargs_dataframe(tsframe: DataFrame, as_index: bool) -> None:
    def f(x: np.ndarray, q: Optional[float] = None, axis: int = 0) -> np.ndarray:
        return np.percentile(x, q, axis=axis)

    df_grouped = tsframe.groupby(lambda x: x.month, as_index=as_index)
    agg_result = df_grouped.agg(np.percentile, 80, axis=0)
    apply_result = df_grouped.apply(DataFrame.quantile, 0.8)
    expected = df_grouped.quantile(0.8)
    tm.assert_frame_equal(apply_result, expected, check_names=False)
    tm.assert_frame_equal(agg_result, expected)

    apply_result = df_grouped.apply(DataFrame.quantile, [0.4, 0.8])
    expected_seq = df_grouped.quantile([0.4, 0.8])
    if not as_index:
        # apply treats the op as a transform; .quantile knows it's a reduction
        apply_result.index = range(4)
        apply_result.insert(loc=0, column="level_0", value=[1, 1, 2, 2])
        apply_result.insert(loc=1, column="level_1", value=[0.4, 0.8, 0.4, 0.8])
    tm.assert_frame_equal(apply_result, expected_seq, check_names=False)

    agg_result = df_grouped.agg(f, q=80)
    apply_result = df_grouped.apply(DataFrame.quantile, q=0.8)
    tm.assert_frame_equal(agg_result, expected)
    tm.assert_frame_equal(apply_result, expected, check_names=False)


def test_len() -> None:
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    grouped = df.groupby([lambda x: x.year, lambda x: x.month, lambda x: x.day])
    assert len(grouped) == len(df)

    grouped = df.groupby([lambda x: x.year, lambda x: x.month])
    expected = len({(x.year, x.month) for x in df.index})
    assert len(grouped) == expected


def test_len_nan_group() -> None:
    # issue 11016
    df = DataFrame({"a": [np.nan] * 3, "b": [1, 2, 3]})
    assert len(df.groupby("a")) == 0
    assert len(df.groupby("b")) == 3
    assert len(df.groupby(["a", "b"])) == 0


def test_groupby_timedelta_median() -> None:
    # issue 57926
    expected = Series(data=Timedelta("1D"), index=["foo"])
    df = DataFrame({"label": ["foo", "foo"], "timedelta": [pd.NaT, Timedelta("1D")]})
    gb = df.groupby("label")["timedelta"]
    actual = gb.median()
    tm.assert_series_equal(actual, expected, check_names=False)


@pytest.mark.parametrize("keys", [["a"], ["a", "b"]])
def test_len_categorical(dropna: bool, observed: bool, keys: List[str]) -> None:
    # GH#57595
    df = DataFrame(
        {
            "a": Categorical([1, 1, 2, np.nan], categories=[1, 2, 3]),
            "b": Categorical([1, 1, 2, np.nan], categories=[1, 2, 3]),
            "c": 1,
        }
    )
    gb = df.groupby(keys, observed=observed, dropna=dropna)
    result = len(gb)
    if observed and dropna:
        expected = 2
    elif observed and not dropna:
        expected = 3
    elif len(keys) == 1:
        expected = 3 if dropna else 4
    else:
        expected = 9 if dropna else 16
    assert result == expected, f"{result} vs {expected}"


def test_basic_regression() -> None:
    # regression
    result = Series([1.0 * x for x in list(range(1, 10)) * 10])

    data = np.random.default_rng(2).random(1100) * 10.0
    groupings = Series(data)

    grouped = result.groupby(groupings)
    grouped.mean()


def test_indices_concatenation_order() -> None:
    # GH 2808

    def f1(x: DataFrame) -> DataFrame:
        y = x[(x.b % 2) == 1] ** 2
        if y.empty:
            multiindex = MultiIndex(levels=[[]] * 2, codes=[[]] * 2, names=["b", "c"])
            res = DataFrame(columns=["a"], index=multiindex)
            return res
        else:
            y = y.set_index(["b", "c"])
            return y

    def f2(x: DataFrame) -> DataFrame:
        y = x[(x.b % 2) == 1] ** 2
        if y.empty:
            return DataFrame()
        else:
            y = y.set_index(["b", "c"])
            return y

    def f3(x: DataFrame) -> DataFrame:
        y = x[(x.b % 2) == 1] ** 2
        if y.empty:
            multiindex = MultiIndex(
                levels=[[]] * 2, codes=[[]] * 2, names=["foo", "bar"]
            )
            res = DataFrame(columns=["a", "b"], index=multiindex)
            return res
        else:
            return y

    df = DataFrame({"a": [1, 2, 2, 2], "b": range(4), "c": range(5, 9)})

    df2 = DataFrame({"a": [3, 2, 2, 2], "b": range(4), "c": range(5, 9)})

    # correct result
    result1 = df.groupby("a").apply(f1)
    result2 = df2.groupby("a").apply(f1)
    tm.assert_frame_equal(result1, result2)

    # should fail (not the same number of levels)
    msg = "Cannot concat indices that do not have the same number of levels"
    with pytest.raises(AssertionError, match=msg):
        df.groupby("a").apply(f2)
    with pytest.raises(AssertionError, match=msg):
        df2.groupby("a").apply(f2)

    # should fail (incorrect shape)
    with pytest.raises(AssertionError, match=msg):
        df.groupby("a").apply(f3)
    with pytest.raises(AssertionError, match=msg):
        df2.groupby("a").apply(f3)


def test_attr_wrapper(ts: Series) -> None:
    grouped = ts.groupby(lambda x: x.weekday())

    result = grouped.std()
    expected = grouped.agg(lambda x: np.std(x, ddof=1))
    tm.assert_series_equal(result, expected)

    # this is pretty cool
    result = grouped.describe()
    expected = {name: gp.describe() for name, gp in grouped}
    expected = DataFrame(expected).T
    tm.assert_frame_equal(result, expected)

    # get attribute
    result = grouped.dtype
    expected = grouped.agg(lambda x: x.dtype)
    tm.assert_series_equal(result, expected)

    # make sure raises error
    msg = "'SeriesGroupBy' object has no attribute 'foo'"
    with pytest.raises(AttributeError, match=msg):
        grouped.foo


def test_frame_groupby(tsframe: DataFrame) -> None:
    grouped = tsframe.groupby(lambda x: x.weekday())

    # aggregate
    aggregated = grouped.aggregate("mean")
    assert len(aggregated) == 5
    assert len(aggregated.columns) == 4

    # by string
    tscopy = tsframe.copy()
    tscopy["weekday"] = [x.weekday() for x in tscopy.index]
    stragged = tscopy.groupby("weekday").aggregate("mean")
    tm.assert_frame_equal(stragged, aggregated, check_names=False)

    # transform
    grouped = tsframe.head(30).groupby(lambda x: x.weekday())
    transformed = grouped.transform(lambda x: x - x.mean())
    assert len(transformed) == 30
    assert len(transformed.columns) == 4

    # transform propagate
    transformed = grouped.transform(lambda x: x.mean())
    for name, group in grouped:
        mean = group.mean()
        for idx in group.index:
            tm.assert_series_equal(transformed.xs(idx), mean, check_names=False)

    # iterate
    for weekday, group in grouped:
        assert group.index[0].weekday() == weekday

    # groups / group_indices
    groups = grouped.groups
    indices = grouped.indices

    for k, v in groups.items():
        samething = tsframe.index.take(indices[k])
        assert (samething == v).all()


def test_frame_set_name_single(df: DataFrame) -> None:
    grouped = df.groupby("A")

    result = grouped.mean(numeric_only=True)
    assert result.index.name == "A"

    result = df.groupby("A", as_index=False).mean(numeric_only=True)
    assert result.index.name != "A"

    result = grouped[["C", "D"]].agg("mean")
    assert result.index.name == "A"

    result = grouped.agg({"C": "mean", "D": "std"})
    assert result.index.name == "A"

    result = grouped["C"].mean()
    assert result.index.name == "A"
    result = grouped["C"].agg("mean")
    assert result.index.name == "A"
    result = grouped["C"].agg(["mean", "std"])
    assert result.index.name == "A"

    msg = r"nested renamer is not supported"
    with pytest.raises(SpecificationError, match=msg):
        grouped["C"].agg({"foo": "mean", "bar": "std"})


def test_multi_func(df: DataFrame) -> None:
    col1 = df["A"]
    col2 = df["B"]

    grouped = df.groupby([col1.get, col2.get])
    agged = grouped.mean(numeric_only=True)
    expected = df.groupby(["A", "B"]).mean()

    # TODO groupby get drops names
    tm.assert_frame_equal(
        agged.loc[:, ["C", "D"]], expected.loc[:, ["C", "D"]], check_names=False
    )

    # some "groups" with no data
    df = DataFrame(
        {
            "v1": np.random.default_rng(2).standard_normal(6),
            "v2": np.random.default_rng(2).standard_normal(6),
            "k1": np.array(["b", "b", "b", "a", "a", "a"]),
            "k2": np.array(["1", "1", "1", "2", "2", "2"]),
        },
        index=["one", "two", "three", "four", "five", "six"],
    )
    # only verify that it works for now
    grouped = df.groupby(["k1", "k2"])
    grouped.agg("sum")


def test_multi_key_multiple_functions(df: DataFrame) -> None:
    grouped = df.groupby(["A", "B"])["C"]

    agged = grouped.agg(["mean", "std"])
    expected = DataFrame({"mean": grouped.agg("mean"), "std": grouped.agg("std")})
    tm.assert_frame_equal(agged, expected)


def test_frame_multi_key_function_list() -> None:
    data = DataFrame(
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
            "D": np.random.default_rng(2).standard_normal(11),
            "E": np.random.default_rng(2).standard_normal(11),
            "F": np.random.default_rng(2).standard_normal(11),
        }
    )

    grouped = data.groupby(["A", "B"])
    funcs = ["mean", "std"]
    agged = grouped.agg(funcs)
    expected = pd.concat(
        [grouped["D"].agg(funcs), grouped["E"].agg(funcs), grouped["F"].agg(funcs)],
        keys=["D", "E", "F"],
        axis=1,
    )
    assert isinstance(agged.index, MultiIndex)
    assert isinstance(expected.index, MultiIndex)
    tm.assert_frame_equal(agged, expected)


def test_frame_multi_key_function_list_partial_failure(using_infer_string: bool) -> None:
    data = DataFrame(
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

    grouped = data.groupby(["A