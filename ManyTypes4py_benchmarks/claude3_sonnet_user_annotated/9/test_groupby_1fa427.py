from datetime import datetime
import decimal
from decimal import Decimal
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

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

    def max_value(group: DataFrame) -> Series:
        return group.loc[group["value"].idxmax()]

    applied = df.groupby("A").apply(max_value)
    result = applied.dtypes
    expected = df.drop(columns="A").dtypes
    tm.assert_series_equal(result, expected)


def test_pass_args_kwargs(ts: Series) -> None:
    def f(x: np.ndarray, q: Optional[float] = None, axis: int = 0) -> float:
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
    def f(x: np.ndarray, q: Optional[float] = None, axis: int = 0) -> float:
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

    grouped = data.groupby(["A", "B"])
    funcs = ["mean", "std"]
    msg = re.escape("agg function failed [how->mean,dtype->")
    if using_infer_string:
        msg = "dtype 'str' does not support operation 'mean'"
    with pytest.raises(TypeError, match=msg):
        grouped.agg(funcs)


@pytest.mark.parametrize("op", [lambda x: x.sum(), lambda x: x.mean()])
def test_groupby_multiple_columns(df: DataFrame, op: Callable) -> None:
    data = df
    grouped = data.groupby(["A", "B"])

    result1 = op(grouped)

    keys = []
    values = []
    for n1, gp1 in data.groupby("A"):
        for n2, gp2 in gp1.groupby("B"):
            keys.append((n1, n2))
            values.append(op(gp2.loc[:, ["C", "D"]]))

    mi = MultiIndex.from_tuples(keys, names=["A", "B"])
    expected = pd.concat(values, axis=1).T
    expected.index = mi

    # a little bit crude
    for col in ["C", "D"]:
        result_col = op(grouped[col])
        pivoted = result1[col]
        exp = expected[col]
        tm.assert_series_equal(result_col, exp)
        tm.assert_series_equal(pivoted, exp)

    # test single series works the same
    result = data["C"].groupby([data["A"], data["B"]]).mean()
    expected = data.groupby(["A", "B"]).mean()["C"]

    tm.assert_series_equal(result, expected)


def test_as_index_select_column() -> None:
    # GH 5764
    df = DataFrame([[1, 2], [1, 4], [5, 6]], columns=["A", "B"])
    result = df.groupby("A", as_index=False)["B"].get_group(1)
    expected = Series([2, 4], name="B")
    tm.assert_series_equal(result, expected)

    result = df.groupby("A", as_index=False, group_keys=True)["B"].apply(
        lambda x: x.cumsum()
    )
    expected = Series([2, 6, 6], name="B", index=range(3))
    tm.assert_series_equal(result, expected)


def test_groupby_as_index_select_column_sum_empty_df() -> None:
    # GH 35246
    df = DataFrame(columns=Index(["A", "B", "C"], name="alpha"))
    left = df.groupby(by="A", as_index=False)["B"].sum(numeric_only=False)

    expected = DataFrame(columns=df.columns[:2], index=range(0))
    # GH#50744 - Columns after selection shouldn't retain names
    expected.columns.names = [None]
    tm.assert_frame_equal(left, expected)


def test_ops_not_as_index(reduction_func: str) -> None:
    # GH 10355, 21090
    # Using as_index=False should not modify grouped column

    if reduction_func in ("corrwith", "nth", "ngroup"):
        pytest.skip(f"GH 5755: Test not applicable for {reduction_func}")

    df = DataFrame(
        np.random.default_rng(2).integers(0, 5, size=(100, 2)), columns=["a", "b"]
    )
    expected = getattr(df.groupby("a"), reduction_func)()
    if reduction_func == "size":
        expected = expected.rename("size")
    expected = expected.reset_index()

    if reduction_func != "size":
        # 32 bit compat -> groupby preserves dtype whereas reset_index casts to int64
        expected["a"] = expected["a"].astype(df["a"].dtype)

    g = df.groupby("a", as_index=False)

    result = getattr(g, reduction_func)()
    tm.assert_frame_equal(result, expected)

    result = g.agg(reduction_func)
    tm.assert_frame_equal(result, expected)

    result = getattr(g["b"], reduction_func)()
    tm.assert_frame_equal(result, expected)

    result = g["b"].agg(reduction_func)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("columns", ["C", ["C"]])
@pytest.mark.parametrize("keys", [["A"], ["A", "B"]])
@pytest.mark.parametrize(
    "values",
    [
        [True],
        [0],
        [0.0],
        ["a"],
        Categorical([0]),
        [to_datetime(0)],
        date_range(0, 1, 1, tz="US/Eastern"),
        pd.period_range("2016-01-01", periods=3, freq="D"),
        pd.array([0], dtype="Int64"),
        pd.array([0], dtype="Float64"),
        pd.array([False], dtype="boolean"),
    ],
    ids=[
        "bool",
        "int",
        "float",
        "str",
        "cat",
        "dt64",
        "dt64tz",
        "period",
        "Int64",
        "Float64",
        "boolean",
    ],
)
@pytest.mark.parametrize("method", ["attr", "agg", "apply"])
@pytest.mark.parametrize(
    "op", ["idxmax", "idxmin", "min", "max", "sum", "prod", "skew", "kurt"]
)
def test_empty_groupby(
    columns: Union[str, List[str]],
    keys: List[str],
    values: Any,
    method: str,
    op: str,
    dropna: bool,
    using_infer_string: bool,
) -> None:
    # GH8093 & GH26411
    override_dtype = None

    if isinstance(values, BooleanArray) and op in ["sum", "prod"]:
        # We expect to get Int64 back for these
        override_dtype = "Int64"

    if isinstance(values[0], bool) and op in ("prod", "sum"):
        # sum/product of bools is an integer
        override_dtype = "int64"

    df = DataFrame({"A": values, "B": values, "C": values}, columns=list("ABC"))

    if hasattr(values, "dtype"):
        # check that we did the construction right
        assert (df.dtypes == values.dtype).all()

    df = df.iloc[:0]

    gb = df.groupby(keys, group_keys=False, dropna=dropna, observed=False)[columns]

    def get_result(**kwargs: Any) -> DataFrame:
        if method == "attr":
            return getattr(gb, op)(**kwargs)
        else:
            return getattr(gb, method)(op, **kwargs)

    def get_categorical_invalid_expected() -> DataFrame:
        # Categorical is special without 'observed=True', we get an NaN entry
        #  corresponding to the unobserved group. If we passed observed=True
        #  to groupby, expected would just be 'df.set_index(keys)[columns]'
        #  as below
        lev = Categorical([0], dtype=values.dtype)
        if len(keys) != 1:
            idx = MultiIndex.from_product([lev, lev], names=keys)
        else:
            # all columns are dropped, but we end up with one row
            # Categorical is special without 'observed=True'
            idx = Index(lev, name=keys[0])

        if using_infer_string:
            columns_idx = Index([], dtype="str")
        else:
            columns_idx = []
        expected = DataFrame([], columns=columns_idx, index=idx)
        return expected

    is_per = isinstance(df.dtypes.iloc[0], pd.PeriodDtype)
    is_dt64 = df.dtypes.iloc[0].kind == "M"
    is_cat = isinstance(values, Categorical)
    is_str = isinstance(df.dtypes.iloc[0], pd.StringDtype)

    if (
        isinstance(values, Categorical)
        and not values.ordered
        and op in ["min", "max", "idxmin", "idxmax"]
    ):
        if op in ["min", "max"]:
            msg = f"Cannot perform {op} with non-ordered Categorical"
            klass = TypeError
        else:
            msg = f"Can't get {op} of an empty group due to unobserved categories"
            klass = ValueError
        with pytest.raises(klass, match=msg):
            get_result()

        if op in ["min", "max", "idxmin", "idxmax"] and isinstance(columns, list):
            # i.e. DataframeGroupBy, not SeriesGroupBy
            result = get_result(numeric_only=True)
            expected = get_categorical_invalid_expected()
            tm.assert_equal(result, expected)
        return

    if op in ["prod", "sum", "skew", "kurt"]:
        # ops that require more than just ordered-ness
        if is_dt64 or is_cat or is_per or (is_str and op != "sum"):
            # GH#41291
            # datetime64 -> prod and sum are invalid
            if is_dt64:
                msg = "datetime64 type does not support"
            elif is_per:
                msg = "Period type does not support"
            elif is_str:
                msg = f"dtype 'str' does not support operation '{op}'"
            else:
                msg = "category type does not support"
            if op in ["skew", "kurt"]:
                msg = "|".join([msg, f"does not support operation '{op}'"])
            with pytest.raises(TypeError, match=msg):
                get_result()

            if not isinstance(columns, list):
                # i.e. SeriesGroupBy
                return
            elif op in ["skew", "kurt"]:
                # TODO: test the numeric_only=True case
                return
            else:
                # i.e. op in ["prod", "sum"]:
                # i.e. DataFrameGroupBy
                # ops that require more than just ordered-ness
                # GH#41291
                result = get_result(numeric_only=True)

                # with numeric_only=True, these are dropped, and we get
                # an empty DataFrame back
                expected = df.set_index(keys)[[]]
                if is_cat:
                    expected = get_categorical_invalid_expected()
                tm.assert_equal(result, expected)
                return

    result = get_result()
    expected = df.set_index(keys)[columns]
    if op in ["idxmax", "idxmin"]:
        expected = expected.astype(df.index.dtype)
    if override_dtype is not None:
        expected = expected.astype(override_dtype)
    if len(keys) == 1:
        expected.index.name = keys[0]
    tm.assert_equal(result, expected)


def test_empty_groupby_apply_nonunique_columns() -> None:
    # GH#44417
    df = DataFrame(np.random.default_rng(2).standard_normal((0, 4)))
    df[3] = df[3].astype(np.int64)
    df.columns = [0, 1, 2, 0]
    gb = df.groupby(df[1], group_keys=False)
    res = gb.apply(lambda x: x)
    assert (res.dtypes == df.drop(columns=1).dtypes).all()


def test_tuple_as_grouping() -> None:
    # https://github.com/pandas-dev/pandas/issues/18314
    df = DataFrame(
        {
            ("a", "b"): [1, 1, 1, 1],
            "a": [2, 2, 2, 2],
            "b": [2, 2, 2, 2],
            "c": [1, 1, 1, 1],
        }
    )

    with pytest.raises(KeyError, match=r"('a', 'b')"):
        df[["a", "b", "c"]].groupby(("a", "b"))

    result = df.groupby(("a", "b"))["c"].sum()
    expected = Series([4], name="c", index=Index([1], name=("a", "b")))
    tm.assert_series_equal(result, expected)


def test_tuple_correct_keyerror() -> None:
    # https://github.com/pandas-dev/pandas/issues/18798
    df = DataFrame(1, index=range(3), columns=MultiIndex.from_product([[1, 2], [3, 4]]))
    with pytest.raises(KeyError, match=r"^\(7, 8\)$"):
        df.groupby((7, 8)).mean()


def test_groupby_agg_ohlc_non_first() -> None:
    # GH 21716
    df = DataFrame(
        [[1], [1]],
        columns=Index(["foo"], name="mycols"),
        index=date_range("2018-01-01", periods=2, freq="D", name="dti"),
    )

    expected = DataFrame(
        [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
        columns=MultiIndex.from_tuples(
            (
                ("foo", "sum", "foo"),
                ("foo", "ohlc", "open"),
                ("foo", "ohlc", "high"),
                ("foo", "ohlc", "low"),
                ("foo", "ohlc", "close"),
            ),
            names=["mycols", None, None],
        ),
        index=date_range("2018-01-01", periods=2, freq="D", name="dti"),
    )

    result = df.groupby(Grouper(freq="D")).agg(["sum", "ohlc"])

    tm.assert_frame_equal(result, expected)


def test_groupby_multiindex_nat() -> None:
    # GH 9236
    values = [
        (pd.NaT, "a"),
        (datetime(2012, 1, 2), "a"),
        (datetime(2012, 1, 2), "b"),
        (datetime(2012, 1, 3), "a"),
    ]
    mi = MultiIndex.from_tuples(values, names=["date", None])
    ser = Series([3, 2, 2.5, 4], index=mi)

    result = ser.groupby(level=1).mean()
    expected = Series([3.0, 2.5], index=["a", "b"])
    tm.assert_series_equal(result, expected)


def test_groupby_empty_list_raises() -> None:
    # GH 5289
    values = zip(range(10), range(10))
    df = DataFrame(values, columns=["apple", "b"])
    msg = "Grouper and axis must be same length"
    with pytest.raises(ValueError, match=msg):
        df.groupby([[]])


def test_groupby_multiindex_series_keys_len_equal_group_axis() -> None:
    # GH 25704
    index_array = [["x", "x"], ["a", "b"], ["k", "k"]]
    index_names = ["first", "second", "third"]
    ri = MultiIndex.from_arrays(index_array, names=index_names)
    s = Series(data=[1, 2], index=ri)
    result = s.groupby(["first", "third"]).sum()

    index_array = [["x"], ["k"]]
    index_names = ["first", "third"]
    ei = MultiIndex.from_arrays(index_array, names=index_names)
    expected = Series([3], index=ei)

    tm.assert_series_equal(result, expected)


def test_groupby_groups_in_BaseGrouper() -> None:
    # GH 26326
    # Test if DataFrame grouped with a pandas.Grouper has correct groups
    mi = MultiIndex.from_product([["A", "B"], ["C", "D"]], names=["alpha", "beta"])
    df = DataFrame({"foo": [1, 2, 1, 2], "bar": [1, 2, 3, 4]}, index=mi)
    result = df.groupby([Grouper(level="alpha"), "beta"])
    expected = df.groupby(["alpha", "beta"])
    assert result.groups == expected.groups

    result = df.groupby(["beta", Grouper(level="alpha")])
    expected = df.groupby(["beta", "alpha"])
    assert result.groups == expected.groups


def test_groups_sort_dropna(sort: bool, dropna: bool) -> None:
    # GH#56966, GH#56851
    df = DataFrame([[2.0, 1.0], [np.nan, 4.0], [0.0, 3.0]])
    keys = [(2.0, 1.0), (np.nan, 4.0), (0.0, 3.0)]
    values = [
        RangeIndex(0, 1),
        RangeIndex(1, 2),
        RangeIndex(2, 3),
    ]
    if sort:
        taker = [2, 0] if dropna else [2, 0, 1]
    else:
        taker = [0, 2] if dropna else [0, 1, 2]
    expected = {keys[idx]: values[idx] for idx in taker}

    gb = df.groupby([0, 1], sort=sort, dropna=dropna)
    result = gb.groups

    for result_key, expected_key in zip(result.keys(), expected.keys()):
        # Compare as NumPy arrays to handle np.nan
        result_key_arr = np.array(result_key)
        expected_key_arr = np.array(expected_key)
        tm.assert_numpy_array_equal(result_key_arr, expected_key_arr)
    for result_value, expected_value in zip(result.values(), expected.values()):
        tm.assert_index_equal(result_value, expected_value)


@pytest.mark.parametrize(
    "op, expected",
    [
        (
            "shift",
            {
                "time": [
                    None,
                    None,
                    Timestamp("2019-01-01 12:00:00"),
                    Timestamp("2019-01-01 12:30:00"),
                    None,
                    None,
                ]
            },
        ),
        (
            "bfill",
            {
                "time": [
                    Timestamp("2019-01-01 12:00:00"),
                    Timestamp("2019-01-01 12:30:00"),
                    Timestamp("2019-01-01 14:00:00"),
                    Timestamp("2019-01-01 14:30:00"),
                    Timestamp("2019-01-01 14:00:00"),
                    Timestamp("2019-01-01 14:30:00"),
                ]
            },
        ),
        (
            "ffill",
            {
                "time": [
                    Timestamp("2019-01-01 12:00:00"),
                    Timestamp("2019-01-01 12:30:00"),
                    Timestamp("2019-01-01 12:00:00"),
                    Timestamp("2019-01-01 12:30:00"),
                    Timestamp("2019-01-01 14:00:00"),
                    Timestamp("2019-01-01 14:30:00"),
                ]
            },
        ),
    ],
)
def test_shift_bfill_ffill_tz(tz_naive_fixture: Optional[str], op: str, expected: Dict[str, List[Optional[Timestamp]]]) -> None:
    # GH19995, GH27992: Check that timezone does not drop in shift, bfill, and ffill
    tz = tz_naive_fixture
    data = {
        "id": ["A", "B", "A", "B", "A", "B"],
        "time": [
            Timestamp("2019-01-01 12:00:00"),
            Timestamp("2019-01-01 12:30:00"),
            None,
            None,
            Timestamp("2019-01-01 14:00:00"),
            Timestamp("2019-01-01 14:30:00"),
        ],
    }
    df = DataFrame(data).assign(time=lambda x: x.time.dt.tz_localize(tz))

    grouped = df.groupby("id")
    result = getattr(grouped, op)()
    expected_df = DataFrame(expected).assign(time=lambda x: x.time.dt.tz_localize(tz))
    tm.assert_frame_equal(result, expected_df)


def test_groupby_only_none_group() -> None:
    # see GH21624
    # this was crashing with "ValueError: Length of passed values is 1, index implies 0"
    df = DataFrame({"g": [None], "x": 1})
    actual = df.groupby("g")["x"].transform("sum")
    expected = Series([np.nan], name="x")

    tm.assert_series_equal(actual, expected)


def test_groupby_duplicate_index() -> None:
    # GH#29189 the groupby call here used to raise
    ser = Series([2, 5, 6, 8], index=[2.0, 4.0, 4.0, 5.0])
    gb = ser.groupby(level=0)

    result = gb.mean()
    expected = Series([2, 5.5, 8], index=[2.0, 4.0, 5.0])
    tm.assert_series_equal(result, expected)


def test_group_on_empty_multiindex(transformation_func: str, request: Any) -> None:
    # GH 47787
    # With one row, those are transforms so the schema should be the same
    df = DataFrame(
        data=[[1, Timestamp("today"), 3, 4]],
        columns=["col_1", "col_2", "col_3", "col_4"],
    )
    df["col_3"] = df["col_3"].astype(int)
    df["col_4"] = df["col_4"].astype(int)
    df = df.set_index(["col_1", "col_2"])
    result = df.iloc[:0].groupby(["col_1"]).transform(transformation_func)
    expected = df.groupby(["col_1"]).transform(transformation_func).iloc[:0]
    if transformation_func in ("diff", "shift"):
        expected = expected.astype(int)
    tm.assert_equal(result, expected)

    result = df["col_3"].iloc[:0].groupby(["col_1"]).transform(transformation_func)
    expected = df["col_3"].groupby(["col_1"]).transform(transformation_func).iloc[:0]
    if transformation_func in ("diff", "shift"):
        expected = expected.astype(int)
    tm.assert_equal(result, expected)


def test_groupby_crash_on_nunique() -> None:
    # Fix following 30253
    dti = date_range("2016-01-01", periods=2, name="foo")
    df = DataFrame({("A", "B"): [1, 2], ("A", "C"): [1, 3], ("D", "B"): [0, 0]})
    df.columns.names = ("bar", "baz")
    df.index = dti

    df = df.T
    gb = df.groupby(level=0)
    result = gb.nunique()

    expected = DataFrame({"A": [1, 2], "D": [1, 1]}, index=dti)
    expected.columns.name = "bar"
    expected = expected.T

    tm.assert_frame_equal(result, expected)

    # same thing, but empty columns
    gb2 = df[[]].groupby(level=0)
    exp = expected[[]]

    res = gb2.nunique()
    tm.assert_frame_equal(res, exp)


def test_groupby_list_level() -> None:
    # GH 9790
    expected = DataFrame(np.arange(0, 9).reshape(3, 3), dtype=float)
    result = expected.groupby(level=[0]).mean()
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "max_seq_items, expected",
    [
        (5, "{0: [0], 1: [1], 2: [2], 3: [3], 4: [4]}"),
        (4, "{0: [0], 1: [1], 2: [2], 3: [3], ...}"),
        (1, "{0: [0], ...}"),
    ],
)
def test_groups_repr_truncates(max_seq_items: int, expected: str) -> None:
    # GH 1135
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 1)))
    df["a"] = df.index

    with pd.option_context("display.max_seq_items", max_seq_items):
        result = df.groupby("a").groups.__repr__()
        assert result == expected

        result = df.groupby(np.array(df.a)).groups.__repr__()
        assert result == expected


def test_group_on_two_row_multiindex_returns_one_tuple_key() -> None:
    # GH 18451
    df = DataFrame([{"a": 1, "b": 2, "c": 99}, {"a": 1, "b": 2, "c": 88}])
    df = df.set_index(["a", "b"])

    grp = df.groupby(["a", "b"])
    result = grp.indices
    expected = {(1, 2): np.array([0, 1], dtype=np.int64)}

    assert len(result) == 1
    key = (1, 2)
    assert (result[key] == expected[key]).all()


@pytest.mark.parametrize(
    "klass, attr, value",
    [
        (DataFrame, "level", "a"),
        (DataFrame, "as_index", False),
        (DataFrame, "sort", False),
        (DataFrame, "group_keys", False),
        (DataFrame, "observed", True),
        (DataFrame, "dropna", False),
        (Series, "level", "a"),
        (Series, "as_index", False),
        (Series, "sort", False),
        (Series, "group_keys", False),
        (Series, "observed", True),
        (Series, "dropna", False),
    ],
)
def test_subsetting_columns_keeps_attrs(klass: Union[type[DataFrame], type[Series]], attr: str, value: Any) -> None:
    # GH 9959 - When subsetting columns, don't drop attributes
    df = DataFrame({"a": [1], "b": [2], "c": [3]})
    if attr != "axis":
        df = df.set_index("a")

    expected = df.groupby("a", **{attr: value})
    result = expected[["b"]] if klass is DataFrame else expected["b"]
    assert getattr(result, attr) == getattr(expected, attr)


@pytest.mark.parametrize("func", ["sum", "any", "shift"])
def test_groupby_column_index_name_lost(func: str) -> None:
    # GH: 29764 groupby loses index sometimes
    expected = Index(["a"], name="idx")
    df = DataFrame([[1]], columns=expected)
    df_grouped = df.groupby([1])
    result = getattr(df_grouped, func)().columns
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize(
    "infer_string",
    [
        False,
        pytest.param(True, marks=td.skip_if_no("pyarrow")),
    ],
)
def test_groupby_duplicate_columns(infer_string: bool) -> None:
    # GH: 31735
    if infer_string:
        pytest.importorskip("pyarrow")
    df = DataFrame(
        {"A": ["f", "e", "g", "h"], "B": ["a", "b", "c", "d"], "C": [1, 2, 3, 4]}
    ).astype(object)
    df.columns = ["A", "B", "B"]
    with pd.option_context("future.infer_string", infer_string):
        result = df.groupby([0, 0, 0, 0]).min()
    expected = DataFrame(
        [["e", "a", 1]], index=np.array([0]), columns=["A", "B", "B"], dtype=object
    )
    tm.assert_frame_equal(result, expected)


def test_groupby_series_with_tuple_name() -> None:
    # GH 37755
    ser = Series([1, 2, 3, 4], index=[1, 1, 2, 2], name=("a", "a"))
    ser.index.name = ("b", "b")
    result = ser.groupby(level=0).last()
    expected = Series([2, 4], index=[1, 2], name=("a", "a"))
    expected.index.name = ("b", "b")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "func, values", [("sum", [97.0, 98.0]), ("mean", [24.25, 24.5])]
)
def test_groupby_numerical_stability_sum_mean(func: str, values: List[float]) -> None:
    # GH#38778
    data = [1e16, 1e16, 97, 98, -5e15, -5e15, -5e15, -5e15]
    df = DataFrame({"group": [1, 2] * 4, "a": data, "b": data})
    result = getattr(df.groupby("group"), func)()
    expected = DataFrame({"a": values, "b": values}, index=Index([1, 2], name="group"))
    tm.assert_frame_equal(result, expected)


def test_groupby_numerical_stability_cumsum() -> None:
    # GH#38934
    data = [1e16, 1e16, 97, 98, -5e15, -5e15, -5e15, -5e15]
    df = DataFrame({"group": [1, 2] * 4, "a": data, "b": data})
    result = df.groupby("group").cumsum()
    exp_data = (
        [1e16] * 2 + [1e16 + 96, 1e16 + 98] + [5e15 + 97, 5e15 + 98] + [97.0, 98.0]
    )
    expected = DataFrame({"a": exp_data, "b": exp_data})
    tm.assert_frame_equal(result, expected, check_exact=True)


def test_groupby_cumsum_skipna_false() -> None:
    # GH#46216 don't propagate np.nan above the diagonal
    arr = np.random.default_rng(2).standard_normal((5, 5))
    df = DataFrame(arr)
    for i in range(5):
        df.iloc[i, i] = np.nan

    df["A"] = 1
    gb = df.groupby("A")

    res = gb.cumsum(skipna=False)

    expected = df[[0, 1, 2, 3, 4]].cumsum(skipna=False)
    tm.assert_frame_equal(res, expected)


def test_groupby_cumsum_timedelta64() -> None:
    # GH#46216 don't ignore is_datetimelike in libgroupby.group_cumsum
    dti = date_range("2016-01-01", periods=5)
    ser = Series(dti) - dti[0]
    ser[2] = pd.NaT

    df = DataFrame({"A": 1, "B": ser})
    gb = df.groupby("A")

    res = gb.cumsum(numeric_only=False, skipna=True)
    exp = DataFrame({"B": [ser[0], ser[1], pd.NaT, ser[4], ser[4] * 2]})
    tm.assert_frame_equal(res, exp)

    res = gb.cumsum(numeric_only=False, skipna=False)
    exp = DataFrame({"B": [ser[0], ser[1], pd.NaT, pd.NaT, pd.NaT]})
    tm.assert_frame_equal(res, exp)


def test_groupby_mean_duplicate_index(rand_series_with_duplicate_datetimeindex: Series) -> None:
    dups = rand_series_with_duplicate_datetimeindex
    result = dups.groupby(level=0).mean()
    expected = dups.groupby(dups.index).mean()
    tm.assert_series_equal(result, expected)


def test_groupby_all_nan_groups_drop() -> None:
    # GH 15036
    s = Series([1, 2, 3], [np.nan, np.nan, np.nan])
    result = s.groupby(s.index).sum()
    expected = Series([], index=Index([], dtype=np.float64), dtype=np.int64)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("numeric_only", [True, False])
def test_groupby_empty_multi_column(as_index: bool, numeric_only: bool) -> None:
    # GH 15106 & GH 41998
    df = DataFrame(data=[], columns=["A", "B", "C"])
    gb = df.groupby(["A", "B"], as_index=as_index)
    result = gb.sum(numeric_only=numeric_only)
    if as_index:
        index = MultiIndex([[], []], [[], []], names=["A", "B"])
        columns = ["C"] if not numeric_only else Index([], dtype="str")
    else:
        index = RangeIndex(0)
        columns = ["A", "B", "C"] if not numeric_only else ["A", "B"]
    expected = DataFrame([], columns=columns, index=index)
    tm.assert_frame_equal(result, expected)


def test_groupby_aggregation_non_numeric_dtype() -> None:
    # GH #43108
    df = DataFrame(
        [["M", [1]], ["M", [1]], ["W", [10]], ["W", [20]]], columns=["MW", "v"]
    )

    expected = DataFrame(
        {
            "v": [[1, 1], [10, 20]],
        },
        index=Index(["M", "W"], name="MW"),
    )

    gb = df.groupby(by=["MW"])
    result = gb.sum()
    tm.assert_frame_equal(result, expected)


def test_groupby_aggregation_multi_non_numeric_dtype() -> None:
    # GH #42395
    df = DataFrame(
        {
            "x": [1, 0, 1, 1, 0],
            "y": [Timedelta(i, "days") for i in range(1, 6)],
            "z": [Timedelta(i * 10, "days") for i in range(1, 6)],
        }
    )

    expected = DataFrame(
        {
            "y": [Timedelta(i, "days") for i in range(7, 9)],
            "z": [Timedelta(i * 10, "days") for i in range(7, 9)],
        },
        index=Index([0, 1], dtype="int64", name="x"),
    )

    gb = df.groupby(by=["x"])
    result = gb.sum()
    tm.assert_frame_equal(result, expected)


def test_groupby_aggregation_numeric_with_non_numeric_dtype() -> None:
    # GH #43108
    df = DataFrame(
        {
            "x": [1, 0, 1, 1, 0],
            "y": [Timedelta(i, "days") for i in range(1, 6)],
            "z": list(range(1, 6)),
        }
    )

    expected = DataFrame(
        {"y": [Timedelta(7, "days"), Timedelta(8, "days")], "z": [7, 8]},
        index=Index([0, 1], dtype="int64", name="x"),
    )

    gb = df.groupby(by=["x"])
    result = gb.sum()
    tm.assert_frame_equal(result, expected)


def test_groupby_filtered_df_std() -> None:
    # GH 16174
    dicts = [
        {"filter_col": False, "groupby_col": True, "bool_col": True, "float_col": 10.5},
        {"filter_col": True, "groupby_col": True, "bool_col": True, "float_col": 20.5},
        {"filter_col": True, "groupby_col": True, "bool_col": True, "float_col": 30.5},
    ]
    df = DataFrame(dicts)

    df_filter = df[df["filter_col"] == True]  # noqa: E712
    dfgb = df_filter.groupby("groupby_col")
    result = dfgb.std()
    expected = DataFrame(
        [[0.0, 0.0, 7.071068]],
        columns=["filter_col", "bool_col", "float_col"],
        index=Index([True], name="groupby_col"),
    )
    tm.assert_frame_equal(result, expected)


def test_datetime_categorical_multikey_groupby_indices() -> None:
    # GH 26859
    df = DataFrame(
        {
            "a": Series(list("abc")),
            "b": Series(
                to_datetime(["2018-01-01", "2018-02-01", "2018-03-01"]),
                dtype="category",
            ),
            "c": Categorical.from_codes([-1, 0, 1], categories=[0, 1]),
        }
    )
    result = df.groupby(["a", "b"], observed=False).indices
    expected = {
        ("a", Timestamp("2018-01-01 00:00:00")): np.array([0]),
        ("b", Timestamp("2018-02-01 00:00:00")): np.array([1]),
        ("c", Timestamp("2018-03-01 00:00:00")): np.array([2]),
    }
    assert result == expected


def test_rolling_wrong_param_min_period() -> None:
    # GH34037
    name_l = ["Alice"] * 5 + ["Bob"] * 5
    val_l = [np.nan, np.nan, 1, 2, 3] + [np.nan, 1, 2, 3, 4]
    test_df = DataFrame([name_l, val_l]).T
    test_df.columns = ["name", "val"]

    result_error_msg = (
        r"^[a-zA-Z._]*\(\) got an unexpected keyword argument 'min_period'"
    )
    with pytest.raises(TypeError, match=result_error_msg):
        test_df.groupby("name")["val"].rolling(window=2, min_period=1).sum()


def test_by_column_values_with_same_starting_value(any_string_dtype: str) -> None:
    # GH29635
    df = DataFrame(
        {
            "Name": ["Thomas", "Thomas", "Thomas John"],
            "Credit": [1200, 1300, 900],
            "Mood": Series(["sad", "happy", "happy"], dtype=any_string_dtype),
        }
    )
    aggregate_details: Dict[str, Union[str, Callable]] = {"Mood": Series.mode, "Credit": "sum"}

    result = df.groupby(["Name"]).agg(aggregate_details)
    expected_result = DataFrame(
        {
            "Mood": [["happy", "sad"], "happy"],
            "Credit": [2500, 900],
            "Name": ["Thomas", "Thomas John"],
        }
    ).set_index("Name")

    tm.assert_frame_equal(result, expected_result)


def test_groupby_none_in_first_mi_level() -> None:
    # GH#47348
    arr = [[None, 1, 0, 1], [2, 3, 2, 3]]
    ser = Series(1, index=MultiIndex.from_arrays(arr, names=["a", "b"]))
    result = ser.groupby(level=[0, 1]).sum()
    expected = Series(
        [1, 2], MultiIndex.from_tuples([(0.0, 2), (1.0, 3)], names=["a", "b"])
    )
    tm.assert_series_equal(result, expected)


def test_groupby_none_column_name(using_infer_string: bool) -> None:
    # GH#47348
    df = DataFrame({None: [1, 1, 2, 2], "b": [1, 1, 2, 3], "c": [4, 5, 6, 7]})
    by = [np.nan] if using_infer_string else [None]
    gb = df.groupby(by=by)
    result = gb.sum()
    expected = DataFrame({"b": [2, 5], "c": [9, 13]}, index=Index([1, 2], name=by[0]))
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("selection", [None, "a", ["a"]])
def test_single_element_list_grouping(selection: Optional[Union[str, List[str]]]) -> None:
    # GH#42795, GH#53500
    df = DataFrame({"a": [1, 2], "b": [np.nan, 5], "c": [np.nan, 2]}, index=["x", "y"])
    grouped = df.groupby(["a"]) if selection is None else df.groupby(["a"])[selection]
    result = [key for key, _ in grouped]

    expected = [(1,), (2,)]
    assert result == expected


def test_groupby_string_dtype() -> None:
    # GH 40148
    df = DataFrame({"str_col": ["a", "b", "c", "a"], "num_col": [1, 2, 3, 2]})
    df["str_col"] = df["str_col"].astype("string")
    expected = DataFrame(
        {
            "str_col": [
                "a",
                "b",
                "c",
            ],
            "num_col": [1.5, 2.0, 3.0],
        }
    )
    expected["str_col"] = expected["str_col"].astype("string")
    grouped = df.groupby("str_col", as_index=False)
    result = grouped.mean()
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "level_arg, multiindex", [([0], False), ((0,), False), ([0], True), ((0,), True)]
)
def test_single_element_listlike_level_grouping(level_arg: Union[List[int], Tuple[int]], multiindex: bool) -> None:
    # GH 51583
    df = DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]}, index=["x", "y"])
    if multiindex:
        df = df.set_index(["a", "b"])
    result = [key for key, _ in df.groupby(level=level_arg)]
    expected = [(1,), (2,)] if multiindex else [("x",), ("y",)]
    assert result == expected


@pytest.mark.parametrize("func", ["sum", "cumsum", "cumprod", "prod"])
def test_groupby_avoid_casting_to_float(func: str) -> None:
    # GH#37493
    val = 922337203685477580
    df = DataFrame({"a": 1, "b": [val]})
    result = getattr(df.groupby("a"), func)() - val
    expected = DataFrame({"b": [0]}, index=Index([1], name="a"))
    if func in ["cumsum", "cumprod"]:
        expected = expected.reset_index(drop=True)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("func, val", [("sum", 3), ("prod", 2)])
def test_groupby_sum_support_mask(any_numeric_ea_dtype: str, func: str, val: int) -> None:
    # GH#37493
    df = DataFrame({"a": 1, "b": [1, 2, pd.NA]}, dtype=any_numeric_ea_dtype)
    result = getattr(df.groupby("a"), func)()
    expected = DataFrame(
        {"b": [val]},
        index=Index([1], name="a", dtype=any_numeric_ea_dtype),
        dtype=any_numeric_ea_dtype,
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("val, dtype", [(111, "int"), (222, "uint")])
def test_groupby_overflow(val: int, dtype: str) -> None:
    # GH#37493
    df = DataFrame({"a": 1, "b": [val, val]}, dtype=f"{dtype}8")
    result = df.groupby("a").sum()
    expected = DataFrame(
        {"b": [val * 2]},
        index=Index([1], name="a", dtype=f"{dtype}8"),
        dtype=f"{dtype}64",
    )
    tm.assert_frame_equal(result, expected)

    result = df.groupby("a").cumsum()
    expected = DataFrame({"b": [val, val * 2]}, dtype=f"{dtype}64")
    tm.assert_frame_equal(result, expected)

    result = df.groupby("a").prod()
    expected = DataFrame(
        {"b": [val * val]},
        index=Index([1], name="a", dtype=f"{dtype}8"),
        dtype=f"{dtype}64",
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("skipna, val", [(True, 3), (False, pd.NA)])
def test_groupby_cumsum_mask(any_numeric_ea_dtype: str, skipna: bool, val: Union[int, pd._libs.missing.NAType]) -> None:
    # GH#37493
    df = DataFrame({"a": 1, "b": [1, pd.NA, 2]}, dtype=any_numeric_ea_dtype)
    result = df.groupby("a").cumsum(skipna=skipna)
    expected = DataFrame(
        {"b": [1, pd.NA, val]},
        dtype=any_numeric_ea_dtype,
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "val_in, index, val_out",
    [
        (
            [1.0, 2.0, 3.0, 4.0, 5.0],
            ["foo", "foo", "bar", "baz", "blah"],
            [3.0, 4.0, 5.0, 3.0],
        ),
        (
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ["foo", "foo", "bar", "baz", "blah", "blah"],
            [3.0, 4.0, 11.0, 3.0],
        ),
    ],
)
def test_groupby_index_name_in_index_content(val_in: List[float], index: List[str], val_out: List[float]) -> None:
    # GH 48567
    series = Series(data=val_in, name="values", index=Index(index, name="blah"))
    result = series.groupby("blah").sum()
    expected = Series(
        data=val_out,
        name="values",
        index=Index(["bar", "baz", "blah", "foo"], name="blah"),
    )
    tm.assert_series_equal(result, expected)

    result = series.to_frame().groupby("blah").sum()
    expected = expected.to_frame()
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("n", [1, 10, 32, 100, 1000])
def test_sum_of_booleans(n: int) -> None:
    # GH 50347
    df = DataFrame({"groupby_col": 1, "bool": [True] * n})
    df["bool"] = df["bool"].eq(True)
    result = df.groupby("groupby_col").sum()
    expected = DataFrame({"bool": [n]}, index=Index([1], name="groupby_col"))
    tm.assert_frame_equal(result, expected)


@pytest.mark.filterwarnings(
    "ignore:invalid value encountered in remainder:RuntimeWarning"
)
@pytest.mark.parametrize("method", ["head", "tail", "nth", "first", "last"])
def test_groupby_method_drop_na(method: str) -> None:
    # GH 21755
    df = DataFrame({"A": ["a", np.nan, "b", np.nan, "c"], "B": range(5)})

    if method == "nth":
        result = getattr(df.groupby("A"), method)(n=0)
    else:
        result = getattr(df.groupby("A"), method)()

    if method in ["first", "last"]:
        expected = DataFrame({"B": [0, 2, 4]}).set_index(
            Series(["a", "b", "c"], name="A")
        )
    else:
        expected = DataFrame(
            {"A": ["a", "b", "c"], "B": [0, 2, 4]}, index=range(0, 6, 2)
        )
    tm.assert_frame_equal(result, expected)


def test_groupby_reduce_period() -> None:
    # GH#51040
    pi = pd.period_range("2016-01-01", periods=100, freq="D")
    grps = list(range(10)) * 10
    ser = pi.to_series()
    gb = ser.groupby(grps)

    with pytest.raises(TypeError, match="Period type does not support sum operations"):
        gb.sum()
    with pytest.raises(
        TypeError, match="Period type does not support cumsum operations"
    ):
        gb.cumsum()
    with pytest.raises(TypeError, match="Period type does not support prod operations"):
        gb.prod()
    with pytest.raises(
        TypeError, match="Period type does not support cumprod operations"
    ):
        gb.cumprod()

    res = gb.max()
    expected = ser[-10:]
    expected.index = Index(range(10), dtype=int)
    tm.assert_series_equal(res, expected)

    res = gb.min()
    expected = ser[:10]
    expected.index = Index(range(10), dtype=int)
    tm.assert_series_equal(res, expected)


def test_obj_with_exclusions_duplicate_columns() -> None:
    # GH#50806
    df = DataFrame([[0, 1, 2, 3]])
    df.columns = [0, 1, 2, 0]
    gb = df.groupby(df[1])
    result = gb._obj_with_exclusions
    expected = df.take([0, 2, 3], axis=1)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("numeric_only", [True, False])
def test_groupby_numeric_only_std_no_result(numeric_only: bool) -> None:
    # GH 51080
    dicts_non_numeric = [{"a": "foo", "b": "bar"}, {"a": "car", "b": "dar"}]
    df = DataFrame(dicts_non_numeric, dtype=object)
    dfgb = df.groupby("a", as_index=False, sort=False)

    if numeric_only:
        result = dfgb.std(numeric_only=True)
        expected_df = DataFrame(["foo", "car"], columns=["a"])
        tm.assert_frame_equal(result, expected_df)
    else:
        with pytest.raises(
            ValueError, match="could not convert string to float: 'bar'"
        ):
            dfgb.std(numeric_only=numeric_only)


def test_grouping_with_categorical_interval_columns() -> None:
    # GH#34164
    df = DataFrame({"x": [0.1, 0.2, 0.3, -0.4, 0.5], "w": ["a", "b", "a", "c", "a"]})
    qq = pd.qcut(df["x"], q=np.linspace(0, 1, 5))
    result = df.groupby([qq, "w"], observed=False)["x"].agg("mean")
    categorical_index_level_1 = Categorical(
        [
            Interval(-0.401, 0.1, closed="right"),
            Interval(0.1, 0.2, closed="right"),
            Interval(0.2, 0.3, closed="right"),
            Interval(0.3, 0.5, closed="right"),
        ],
        ordered=True,
    )
    index_level_2 = ["a", "b", "c"]
    mi = MultiIndex.from_product(
        [categorical_index_level_1, index_level_2], names=["x", "w"]
    )
    expected = Series(
        np.array(
            [
                0.1,
                np.nan,
                -0.4,
                np.nan,
                0.2,
                np.nan,
                0.3,
                np.nan,
                np.nan,
                0.5,
                np.nan,
                np.nan,
            ]
        ),
        index=mi,
        name="x",
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("bug_var", [1, "a"])
def test_groupby_sum_on_nan_should_return_nan(bug_var: Union[int, str]) -> None:
    # GH 24196
    df = DataFrame({"A": [bug_var, bug_var, bug_var, np.nan]})
    if isinstance(bug_var, str):
        df = df.astype(object)
    dfgb = df.groupby(lambda x: x)
    result = dfgb.sum(min_count=1)

    expected_df = DataFrame(
        [bug_var, bug_var, bug_var, None], columns=["A"], dtype=df["A"].dtype
    )
    tm.assert_frame_equal(result, expected_df)


@pytest.mark.parametrize(
    "method",
    [
        "count",
        "corr",
        "cummax",
        "cummin",
        "cumprod",
        "describe",
        "rank",
        "quantile",
        "diff",
        "shift",
        "all",
        "any",
        "idxmin",
        "idxmax",
        "ffill",
        "bfill",
        "pct_change",
    ],
)
def test_groupby_selection_with_methods(df: DataFrame, method: str) -> None:
    # some methods which require DatetimeIndex
    rng = date_range("2014", periods=len(df))
    df.index = rng

    g = df.groupby(["A"])[["C"]]
    g_exp = df[["C"]].groupby(df["A"])
    # TODO check groupby with > 1 col ?

    res = getattr(g, method)()
    exp = getattr(g_exp, method)()

    # should always be frames!
    tm.assert_frame_equal(res, exp)


def test_groupby_selection_other_methods(df: DataFrame) -> None:
    # some methods which require DatetimeIndex
    rng = date_range("2014", periods=len(df))
    df.columns.name = "foo"
    df.index = rng

    g = df.groupby(["A"])[["C"]]
    g_exp = df[["C"]].groupby(df["A"])

    # methods which aren't just .foo()
    tm.assert_frame_equal(g.apply(lambda x: x.sum()), g_exp.apply(lambda x: x.sum()))

    tm.assert_frame_equal(g.resample("D").mean(), g_exp.resample("D").mean())
    tm.assert_frame_equal(g.resample("D").ohlc(), g_exp.resample("D").ohlc())

    tm.assert_frame_equal(
        g.filter(lambda x: len(x) == 3), g_exp.filter(lambda x: len(x) == 3)
    )


def test_groupby_with_Time_Grouper(unit: str) -> None:
    idx2 = to_datetime(
        [
            "2016-08-31 22:08:12.000",
            "2016-08-31 22:09:12.200",
            "2016-08-31 22:20:12.400",
        ]
    ).as_unit(unit)

    test_data = DataFrame(
        {"quant": [1.0, 1.0, 3.0], "quant2": [1.0, 1.0, 3.0], "time2": idx2}
    )

    time2 = date_range("2016-08-31 22:08:00", periods=13, freq="1min", unit=unit)
    expected_output = DataFrame(
        {
            "time2": time2,
            "quant": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            "quant2": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        }
    )

    gb = test_data.groupby(Grouper(key="time2", freq="1min"))
    result = gb.count().reset_index()

    tm.assert_frame_equal(result, expected_output)


def test_groupby_series_with_datetimeindex_month_name() -> None:
    # GH 48509
    s = Series([0, 1, 0], index=date_range("2022-01-01", periods=3), name="jan")
    result = s.groupby(s).count()
    expected = Series([2, 1], name="jan")
    expected.index.name = "jan"
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("test_series", [True, False])
@pytest.mark.parametrize(
    "kwarg, value, name, warn",
    [
        ("by", "a", 1, None),
        ("by", ["a"], (1,), None),
        ("level", 0, 1, None),
        ("level", [0], (1,), None),
    ],
)
def test_get_group_len_1_list_likes(
    test_series: bool, kwarg: str, value: Union[str, List[str], int, List[int]], name: Union[int, Tuple[int]], warn: None
) -> None:
    # GH#25971
    obj = DataFrame({"b": [3, 4, 5]}, index=Index([1, 1, 2], name="a"))
    if test_series:
        obj = obj["b"]
    gb = obj.groupby(**{kwarg: value})
    result = gb.get_group(name)
    if test_series:
        expected = Series([3, 4], index=Index([1, 1], name="a"), name="b")
    else:
        expected = DataFrame({"b": [3, 4]}, index=Index([1, 1], name="a"))
    tm.assert_equal(result, expected)


def test_groupby_ngroup_with_nan() -> None:
    # GH#50100
    df = DataFrame({"a": Categorical([np.nan]), "b": [1]})
    result = df.groupby(["a", "b"], dropna=False, observed=False).ngroup()
    expected = Series([0])
    tm.assert_series_equal(result, expected)


def test_groupby_ffill_with_duplicated_index() -> None:
    # GH#43412
    df = DataFrame({"a": [1, 2, 3, 4, np.nan, np.nan]}, index=[0, 1, 2, 0, 1, 2])

    result = df.groupby(level=0).ffill()
    expected = DataFrame({"a": [1, 2, 3, 4, 2, 3]}, index=[0, 1, 2, 0, 1, 2])
    tm.assert_frame_equal(result, expected, check_dtype=False)


@pytest.mark.parametrize("test_series", [True, False])
def test_decimal_na_sort(test_series: bool) -> None:
    # GH#54847
    # We catch both TypeError and decimal.InvalidOperation exceptions in safe_sort.
    # If this next assert raises, we can just catch TypeError
    assert not isinstance(decimal.InvalidOperation, TypeError)
    df = DataFrame(
        {
            "key": [Decimal(1), Decimal(1), None, None],
            "value": [Decimal(2), Decimal(3), Decimal(4), Decimal(5)],
        }
    )
    gb = df.groupby("key", dropna=False)
    if test_series:
        gb = gb["value"]
    result = gb._grouper.result_index
    expected = Index([Decimal(1), None], name="key")
    tm.assert_index_equal(result, expected)


def test_groupby_dropna_with_nunique_unique() -> None:
    # GH#42016
    df = [[1, 1, 1, "A"], [1, None, 1, "A"], [1, None, 2, "A"], [1, None, 3, "A"]]
    df_dropna = DataFrame(df, columns=["a", "b", "c", "partner"])
    result = df_dropna.groupby(["a", "b", "c"], dropna=False).agg(
        {"partner": ["nunique", "unique"]}
    )

    index = MultiIndex.from_tuples(
        [(1, 1.0, 1), (1, np.nan, 1), (1, np.nan, 2), (1, np.nan, 3)],
        names=["a", "b", "c"],
    )
    columns = MultiIndex.from_tuples([("partner", "nunique"), ("partner", "unique")])
    expected = DataFrame(
        [(1, ["A"]), (1, ["A"]), (1, ["A"]), (1, ["A"])], index=index, columns=columns
    )

    tm.assert_frame_equal(result, expected)


def test_groupby_agg_namedagg_with_duplicate_columns() -> None:
    # GH#58446
    df = DataFrame(
        {
            "col1": [2, 1, 1, 0, 2, 0],
            "col2": [4, 5, 36, 7, 4, 5],
            "col3": [3.1, 8.0, 12, 10, 4, 1.1],
            "col4": [17, 3, 16, 15, 5, 6],
            "col5": [-1, 3, -1, 3, -2, -1],
        }
    )

    result = df.groupby(by=["col1", "col1", "col2"], as_index=False).agg(
        new_col=pd.NamedAgg(column="col1", aggfunc="min"),
        new_col1=pd.NamedAgg(column="col1", aggfunc="max"),
        new_col2=pd.NamedAgg(column="col2", aggfunc="count"),
    )

    expected = DataFrame(
        {
            "col1": [0, 0, 1, 1, 2],
            "col2": [5, 7, 5, 36, 4],
            "new_col": [0, 0, 1, 1, 2],
            "new_col1": [0, 0, 1, 1, 2],
            "new_col2": [1, 1, 1, 1, 2],
        }
    )

    tm.assert_frame_equal(result, expected)


def test_groupby_multi_index_codes() -> None:
    # GH#54347
    df = DataFrame(
        {"A": [1, 2, 3, 4], "B": [1, float("nan"), 2, float("nan")], "C": [2, 4, 6, 8]}
    )
    df_grouped = df.groupby(["A", "B"], dropna=False).sum()

    index = df_grouped.index
    tm.assert_index_equal(index, MultiIndex.from_frame(index.to_frame()))
