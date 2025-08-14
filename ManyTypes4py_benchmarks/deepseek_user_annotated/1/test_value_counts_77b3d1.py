"""
these are systematically testing all of the args to value_counts
with different size combinations. This is to ensure stability of the sorting
and proper parameter handling
"""

import numpy as np
import pytest

from pandas import (
    Categorical,
    CategoricalIndex,
    DataFrame,
    Grouper,
    Index,
    MultiIndex,
    Series,
    date_range,
    to_datetime,
)
import pandas._testing as tm
from pandas.util.version import Version
from typing import Any, Dict, List, Optional, Tuple, Union


def tests_value_counts_index_names_category_column() -> None:
    # GH44324 Missing name of index category column
    df = DataFrame(
        {
            "gender": ["female"],
            "country": ["US"],
        }
    )
    df["gender"] = df["gender"].astype("category")
    result = df.groupby("country")["gender"].value_counts()

    # Construct expected, very specific multiindex
    df_mi_expected = DataFrame([["US", "female"]], columns=["country", "gender"])
    df_mi_expected["gender"] = df_mi_expected["gender"].astype("category")
    mi_expected = MultiIndex.from_frame(df_mi_expected)
    expected = Series([1], index=mi_expected, name="count")

    tm.assert_series_equal(result, expected)


def seed_df(seed_nans: bool, n: int, m: int) -> DataFrame:
    days = date_range("2015-08-24", periods=10)

    frame = DataFrame(
        {
            "1st": np.random.default_rng(2).choice(list("abcd"), n),
            "2nd": np.random.default_rng(2).choice(days, n),
            "3rd": np.random.default_rng(2).integers(1, m + 1, n),
        }
    )

    if seed_nans:
        # Explicitly cast to float to avoid implicit cast when setting nan
        frame["3rd"] = frame["3rd"].astype("float")
        frame.loc[1::11, "1st"] = np.nan
        frame.loc[3::17, "2nd"] = np.nan
        frame.loc[7::19, "3rd"] = np.nan
        frame.loc[8::19, "3rd"] = np.nan
        frame.loc[9::19, "3rd"] = np.nan

    return frame


@pytest.mark.slow
@pytest.mark.parametrize("seed_nans", [True, False])
@pytest.mark.parametrize("num_rows", [10, 50])
@pytest.mark.parametrize("max_int", [5, 20])
@pytest.mark.parametrize("keys", ["1st", "2nd", ["1st", "2nd"]], ids=repr)
@pytest.mark.parametrize("bins", [None, [0, 5]], ids=repr)
@pytest.mark.parametrize("isort", [True, False])
@pytest.mark.parametrize("normalize, name", [(True, "proportion"), (False, "count")])
def test_series_groupby_value_counts(
    seed_nans: bool,
    num_rows: int,
    max_int: int,
    keys: Union[str, List[str]],
    bins: Optional[List[int]],
    isort: bool,
    normalize: bool,
    name: str,
    sort: bool,
    ascending: Optional[bool],
    dropna: bool,
) -> None:
    df = seed_df(seed_nans, num_rows, max_int)

    def rebuild_index(df: DataFrame) -> DataFrame:
        arr = list(map(df.index.get_level_values, range(df.index.nlevels)))
        df.index = MultiIndex.from_arrays(arr, names=df.index.names)
        return df

    kwargs: Dict[str, Any] = {
        "normalize": normalize,
        "sort": sort,
        "ascending": ascending,
        "dropna": dropna,
        "bins": bins,
    }

    gr = df.groupby(keys, sort=isort)
    left = gr["3rd"].value_counts(**kwargs)

    gr = df.groupby(keys, sort=isort)
    right = gr["3rd"].apply(Series.value_counts, **kwargs)
    right.index.names = right.index.names[:-1] + ["3rd"]
    # https://github.com/pandas-dev/pandas/issues/49909
    right = right.rename(name)

    # have to sort on index because of unstable sort on values
    left, right = map(rebuild_index, (left, right))  # xref GH9212
    tm.assert_series_equal(left.sort_index(), right.sort_index())


@pytest.mark.parametrize("utc", [True, False])
def test_series_groupby_value_counts_with_grouper(utc: bool) -> None:
    # GH28479
    df = DataFrame(
        {
            "Timestamp": [
                1565083561,
                1565083561 + 86400,
                1565083561 + 86500,
                1565083561 + 86400 * 2,
                1565083561 + 86400 * 3,
                1565083561 + 86500 * 3,
                1565083561 + 86400 * 4,
            ],
            "Food": ["apple", "apple", "banana", "banana", "orange", "orange", "pear"],
        }
    ).drop([3])

    df["Datetime"] = to_datetime(df["Timestamp"], utc=utc, unit="s")
    dfg = df.groupby(Grouper(freq="1D", key="Datetime"))

    # have to sort on index because of unstable sort on values xref GH9212
    result = dfg["Food"].value_counts().sort_index()
    expected = dfg["Food"].apply(Series.value_counts).sort_index()
    expected.index.names = result.index.names
    # https://github.com/pandas-dev/pandas/issues/49909
    expected = expected.rename("count")

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("columns", [["A", "B"], ["A", "B", "C"]])
def test_series_groupby_value_counts_empty(columns: List[str]) -> None:
    # GH39172
    df = DataFrame(columns=columns)
    dfg = df.groupby(columns[:-1])

    result = dfg[columns[-1]].value_counts()
    expected = Series([], dtype=result.dtype, name="count")
    expected.index = MultiIndex.from_arrays([[]] * len(columns), names=columns)

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("columns", [["A", "B"], ["A", "B", "C"]])
def test_series_groupby_value_counts_one_row(columns: List[str]) -> None:
    # GH42618
    df = DataFrame(data=[range(len(columns))], columns=columns)
    dfg = df.groupby(columns[:-1])

    result = dfg[columns[-1]].value_counts()
    expected = df.value_counts()

    tm.assert_series_equal(result, expected)


def test_series_groupby_value_counts_on_categorical() -> None:
    # GH38672

    s = Series(Categorical(["a"], categories=["a", "b"]))
    result = s.groupby([0]).value_counts()

    expected = Series(
        data=[1, 0],
        index=MultiIndex.from_arrays(
            [
                np.array([0, 0]),
                CategoricalIndex(
                    ["a", "b"], categories=["a", "b"], ordered=False, dtype="category"
                ),
            ]
        ),
        name="count",
    )

    # Expected:
    # 0  a    1
    #    b    0
    # dtype: int64

    tm.assert_series_equal(result, expected)


def test_series_groupby_value_counts_no_sort() -> None:
    # GH#50482
    df = DataFrame(
        {
            "gender": ["male", "male", "female", "male", "female", "male"],
            "education": ["low", "medium", "high", "low", "high", "low"],
            "country": ["US", "FR", "US", "FR", "FR", "FR"],
        }
    )
    gb = df.groupby(["country", "gender"], sort=False)["education"]
    result = gb.value_counts(sort=False)
    index = MultiIndex(
        levels=[["US", "FR"], ["male", "female"], ["low", "medium", "high"]],
        codes=[[0, 1, 0, 1, 1], [0, 0, 1, 0, 1], [0, 1, 2, 0, 2]],
        names=["country", "gender", "education"],
    )
    expected = Series([1, 1, 1, 2, 1], index=index, name="count")
    tm.assert_series_equal(result, expected)


@pytest.fixture
def education_df() -> DataFrame:
    return DataFrame(
        {
            "gender": ["male", "male", "female", "male", "female", "male"],
            "education": ["low", "medium", "high", "low", "high", "low"],
            "country": ["US", "FR", "US", "FR", "FR", "FR"],
        }
    )


def test_bad_subset(education_df: DataFrame) -> None:
    gp = education_df.groupby("country")
    with pytest.raises(ValueError, match="subset"):
        gp.value_counts(subset=["country"])


def test_basic(education_df: DataFrame, request: Any) -> None:
    # gh43564
    if Version(np.__version__) >= Version("1.25"):
        request.applymarker(
            pytest.mark.xfail(
                reason=(
                    "pandas default unstable sorting of duplicates"
                    "issue with numpy>=1.25 with AVX instructions"
                ),
                strict=False,
            )
        )
    result = education_df.groupby("country")[["gender", "education"]].value_counts(
        normalize=True
    )
    expected = Series(
        data=[0.5, 0.25, 0.25, 0.5, 0.5],
        index=MultiIndex.from_tuples(
            [
                ("FR", "male", "low"),
                ("FR", "male", "medium"),
                ("FR", "female", "high"),
                ("US", "male", "low"),
                ("US", "female", "high"),
            ],
            names=["country", "gender", "education"],
        ),
        name="proportion",
    )
    tm.assert_series_equal(result, expected)


def _frame_value_counts(
    df: DataFrame, keys: List[str], normalize: bool, sort: bool, ascending: Optional[bool]
) -> Series:
    return df[keys].value_counts(normalize=normalize, sort=sort, ascending=ascending)


@pytest.mark.parametrize("groupby", ["column", "array", "function"])
@pytest.mark.parametrize("normalize, name", [(True, "proportion"), (False, "count")])
@pytest.mark.parametrize(
    "sort, ascending",
    [
        (False, None),
        (True, True),
        (True, False),
    ],
)
@pytest.mark.parametrize("frame", [True, False])
def test_against_frame_and_seriesgroupby(
    education_df: DataFrame,
    groupby: str,
    normalize: bool,
    name: str,
    sort: bool,
    ascending: Optional[bool],
    as_index: bool,
    frame: bool,
    request: Any,
    using_infer_string: bool,
) -> None:
    # test all parameters:
    # - Use column, array or function as by= parameter
    # - Whether or not to normalize
    # - Whether or not to sort and how
    # - Whether or not to use the groupby as an index
    # - 3-way compare against:
    #   - apply with :meth:`~DataFrame.value_counts`
    #   - `~SeriesGroupBy.value_counts`
    if Version(np.__version__) >= Version("1.25") and frame and sort and normalize:
        request.applymarker(
            pytest.mark.xfail(
                reason=(
                    "pandas default unstable sorting of duplicates"
                    "issue with numpy>=1.25 with AVX instructions"
                ),
                strict=False,
            )
        )
    by = {
        "column": "country",
        "array": education_df["country"].values,
        "function": lambda x: education_df["country"][x] == "US",
    }[groupby]

    gp = education_df.groupby(by=by, as_index=as_index)
    result = gp[["gender", "education"]].value_counts(
        normalize=normalize, sort=sort, ascending=ascending
    )
    if frame:
        # compare against apply with DataFrame value_counts
        expected = gp.apply(
            _frame_value_counts, ["gender", "education"], normalize, sort, ascending
        )

        if as_index:
            tm.assert_series_equal(result, expected)
        else:
            name = "proportion" if normalize else "count"
            expected = expected.reset_index().rename({0: name}, axis=1)
            if groupby in ["array", "function"] and (not as_index and frame):
                expected.insert(loc=0, column="level_0", value=result["level_0"])
            else:
                expected.insert(loc=0, column="country", value=result["country"])
            tm.assert_frame_equal(result, expected)
    else:
        # compare against SeriesGroupBy value_counts
        education_df["both"] = education_df["gender"] + "-" + education_df["education"]
        expected = gp["both"].value_counts(
            normalize=normalize, sort=sort, ascending=ascending
        )
        expected.name = name
        if as_index:
            index_frame = expected.index.to_frame(index=False)
            index_frame["gender"] = index_frame["both"].str.split("-").str.get(0)
            index_frame["education"] = index_frame["both"].str.split("-").str.get(1)
            del index_frame["both"]
            index_frame2 = index_frame.rename({0: None}, axis=1)
            expected.index = MultiIndex.from_frame(index_frame2)

            if index_frame2.columns.isna()[0]:
                # with using_infer_string, the columns in index_frame as string
                #  dtype, which makes the rename({0: None}) above use np.nan
                #  instead of None, so we need to set None more explicitly.
                expected.index.names = [None] + expected.index.names[1:]
            tm.assert_series_equal(result, expected)
        else:
            expected.insert(1, "gender", expected["both"].str.split("-").str.get(0))
            expected.insert(2, "education", expected["both"].str.split("-").str.get(1))
            if using_infer_string:
                expected = expected.astype({"gender": "str", "education": "str"})
            del expected["both"]
            tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize(
    "sort, ascending, expected_rows, expected_count, expected_group_size",
    [
        (False, None, [0, 1, 2, 3, 4], [1, 1, 1, 2, 1], [1, 3, 1, 3, 1]),
        (True, False, [3, 0, 1, 2, 4], [2, 1, 1, 1, 1], [3, 1, 3, 1, 1]),
        (True, True, [0, 1, 2, 4, 3], [1, 1, 1, 1, 2], [1, 3, 1, 1, 3]),
    ],
)
def test_compound(
    education_df: DataFrame,
    normalize: bool,
    sort: bool,
    ascending: Optional[bool],
    expected_rows: List[int],
    expected_count: List[int],
    expected_group_size: List[int],
    any_string_dtype: str,
    using_infer_string: bool,
) -> None:
    dtype = any_string_dtype
    education_df = education_df.astype(dtype)
    education_df.columns = education_df.columns.astype(dtype)
    # Multiple groupby keys and as_index=False
    gp = education_df.groupby(["country", "gender"], as_index=False, sort=False)
    result = gp["education"].value_counts(
        normalize=normalize, sort=sort, ascending=ascending
    )
    expected = DataFrame()
    for column in ["country", "gender", "education"]:
        expected[column] = [education_df[column][row] for row in expected_rows]
        expected = expected.astype(dtype)
        expected.columns = expected.columns.astype(dtype)
    if normalize:
        expected["proportion"] = expected_count
        expected["proportion"] /= expected_group_size
        if dtype == "string[pyarrow]":
            # TODO(nullable) also string[python] should return nullable dtypes
            expected["proportion"] = expected["proportion"].convert_dtypes()
    else:
        expected["count"] = expected_count
        if dtype == "string[pyarrow]":
            expected["count"] = expected["count"].convert_dtypes()
    if using_infer_string and dtype == object:
        expected = expected.astype(
            {"country": "str", "gender": "str", "education": "str"}
        )

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "sort, ascending, normalize, name, expected_data, expected_index",
    [
        (False, None, False, "count", [1, 1, 1], [(1, 1, 1), (2, 4, 6), (2, 0, 0)]),
        (True, True, False, "count", [1, 1, 2], [(1, 1, 1), (2, 6, 4), (2, 0, 0)]),
        (True, False, False, "count", [2, 1, 1], [(1, 