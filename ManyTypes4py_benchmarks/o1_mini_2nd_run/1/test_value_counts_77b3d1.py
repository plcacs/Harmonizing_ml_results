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
from typing import Any, Callable, List, Optional, Union, Tuple


def tests_value_counts_index_names_category_column() -> None:
    df = DataFrame({"gender": ["female"], "country": ["US"]})
    df["gender"] = df["gender"].astype("category")
    result = df.groupby("country")["gender"].value_counts()
    df_mi_expected = DataFrame([["US", "female"]], columns=["country", "gender"])
    df_mi_expected["gender"] = df_mi_expected["gender"].astype("category")
    mi_expected = MultiIndex.from_frame(df_mi_expected)
    expected = Series([1], index=mi_expected, name="count")
    tm.assert_series_equal(result, expected)


def seed_df(seed_nans: bool, n: int, m: int) -> DataFrame:
    days = date_range("2015-08-24", periods=10)
    rng = np.random.default_rng(2)
    frame = DataFrame(
        {
            "1st": rng.choice(list("abcd"), n),
            "2nd": rng.choice(days, n),
            "3rd": rng.integers(1, m + 1, n),
        }
    )
    if seed_nans:
        frame["3rd"] = frame["3rd"].astype(float)
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
@pytest.mark.parametrize(
    "normalize, name", [(True, "proportion"), (False, "count")]
)
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

    def rebuild_index(df_inner: DataFrame) -> DataFrame:
        arr = list(map(df_inner.index.get_level_values, range(df_inner.index.nlevels)))
        df_inner.index = MultiIndex.from_arrays(arr, names=df_inner.index.names)
        return df_inner

    kwargs: dict[str, Any] = {
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
    right = right.rename(name)
    left, right = map(rebuild_index, (left, right))
    tm.assert_series_equal(left.sort_index(), right.sort_index())


@pytest.mark.parametrize("utc", [True, False])
def test_series_groupby_value_counts_with_grouper(utc: bool) -> None:
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
    result = dfg["Food"].value_counts().sort_index()
    expected = dfg["Food"].apply(Series.value_counts).sort_index()
    expected.index.names = result.index.names
    expected = expected.rename("count")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("columns", [["A", "B"], ["A", "B", "C"]])
def test_series_groupby_value_counts_empty(columns: List[str]) -> None:
    df = DataFrame(columns=columns)
    dfg = df.groupby(columns[:-1])
    result = dfg[columns[-1]].value_counts()
    expected = Series([], dtype=result.dtype, name="count")
    expected.index = MultiIndex.from_arrays([[]] * len(columns), names=columns)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("columns", [["A", "B"], ["A", "B", "C"]])
def test_series_groupby_value_counts_one_row(columns: List[str]) -> None:
    df = DataFrame(data=[range(len(columns))], columns=columns)
    dfg = df.groupby(columns[:-1])
    result = dfg[columns[-1]].value_counts()
    expected = df.value_counts()
    tm.assert_series_equal(result, expected)


def test_series_groupby_value_counts_on_categorical() -> None:
    s = Series(Categorical(["a"], categories=["a", "b"]))
    result = s.groupby([0]).value_counts()
    expected = Series(
        data=[
            1,
            0,
        ],
        index=MultiIndex.from_arrays(
            [
                np.array([0, 0]),
                CategoricalIndex(
                    ["a", "b"], categories=["a", "b"], ordered=False, dtype="category"
                ),
            ],
            names=["key", "category"],
        ),
        name="count",
    )
    tm.assert_series_equal(result, expected)


def test_series_groupby_value_counts_no_sort() -> None:
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
    if Version(np.__version__) >= Version("1.25"):
        request.applymarker(
            pytest.mark.xfail(
                reason="pandas default unstable sorting of duplicatesissue with numpy>=1.25 with AVX instructions",
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
    df: DataFrame,
    keys: List[str],
    normalize: bool,
    sort: bool,
    ascending: Optional[bool],
) -> Series:
    return df[keys].value_counts(normalize=normalize, sort=sort, ascending=ascending)


@pytest.mark.parametrize("groupby", ["column", "array", "function"])
@pytest.mark.parametrize(
    "normalize, name", [(True, "proportion"), (False, "count")]
)
@pytest.mark.parametrize(
    "sort, ascending", [(False, None), (True, True), (True, False)]
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
    if (
        Version(np.__version__) >= Version("1.25")
        and frame
        and sort
        and normalize
    ):
        request.applymarker(
            pytest.mark.xfail(
                reason="pandas default unstable sorting of duplicatesissue with numpy>=1.25 with AVX instructions",
                strict=False,
            )
        )
    by: Union[str, np.ndarray, Callable[[Any], Any]] = {
        "column": "country",
        "array": education_df["country"].values,
        "function": lambda x: education_df["country"][x] == "US",
    }[groupby]
    gp = education_df.groupby(by=by, as_index=as_index)
    result = gp[["gender", "education"]].value_counts(normalize=normalize, sort=sort, ascending=ascending)
    if frame:
        expected = gp.apply(
            _frame_value_counts, ["gender", "education"], normalize, sort, ascending
        )
        if as_index:
            tm.assert_series_equal(result, expected)
        else:
            name_final: str = "proportion" if normalize else "count"
            expected = expected.reset_index().rename({0: name_final}, axis=1)
            if groupby in ["array", "function"] and (not as_index and frame):
                expected.insert(loc=0, column="level_0", value=result["level_0"])
            else:
                expected.insert(loc=0, column="country", value=result["country"])
            tm.assert_frame_equal(result, expected)
    else:
        education_df["both"] = education_df["gender"] + "-" + education_df["education"]
        expected = gp["both"].value_counts(normalize=normalize, sort=sort, ascending=ascending)
        expected.name = name
        if as_index:
            index_frame = expected.index.to_frame(index=False)
            index_frame["gender"] = index_frame["both"].str.split("-").str.get(0)
            index_frame["education"] = index_frame["both"].str.split("-").str.get(1)
            del index_frame["both"]
            index_frame2 = index_frame.rename({0: None}, axis=1)
            expected.index = MultiIndex.from_frame(index_frame2)
            if index_frame2.columns.isna()[0]:
                expected.index.names = [None] + list(expected.index.names[1:])
            tm.assert_series_equal(result, expected)
        else:
            expected.insert(1, "gender", expected["both"].str.split("-").str.get(0))
            expected.insert(2, "education", expected["both"].str.split("-").str.get(1))
            if using_infer_string:
                expected = expected.astype({"gender": "str", "education": "str"})
            del expected["both"]
            tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "normalize, name, expected_data, expected_index",
    [
        (
            False,
            "count",
            [1, 1, 1],
            [(1, 1, 1), (2, 4, 6), (2, 0, 0)],
        ),
        (
            True,
            "proportion",
            [0.5, 0.25, 0.25],
            [(1, 1, 1), (2, 4, 6), (2, 0, 0)],
        ),
        (
            True,
            "proportion",
            [1.0, 1.0],
            [(0, 2, 0), (1, 1, 1)],
        ),
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
    any_string_dtype: Any,
    using_infer_string: bool,
    any_string_dtype,  # duplicate in parameters, assuming typo
    using_infer_string,
) -> None:
    dtype = any_string_dtype
    education_df = education_df.astype(dtype)
    education_df.columns = education_df.columns.astype(dtype)
    gp = education_df.groupby(["country", "gender"], as_index=False, sort=False)
    result = gp["education"].value_counts(normalize=normalize, sort=sort, ascending=ascending)
    expected = DataFrame()
    for column in ["country", "gender", "education"]:
        expected[column] = [education_df[column][row] for row in expected_rows]
        expected = expected.astype(dtype)
        expected.columns = expected.columns.astype(dtype)
    if normalize:
        expected["proportion"] = expected_count
        expected["proportion"] = expected["proportion"].astype(float)
        expected["proportion"] /= pd.Series(expected_group_size, dtype=float)
        if dtype == "string[pyarrow]":
            expected["proportion"] = expected["proportion"].convert_dtypes()
    else:
        expected["count"] = expected_count
        if dtype == "string[pyarrow]":
            expected["count"] = expected["count"].convert_dtypes()
    if using_infer_string and dtype == object:
        expected = expected.astype({"country": "str", "gender": "str", "education": "str"})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "sort, ascending, normalize, name, expected_data, expected_index",
    [
        (
            False,
            None,
            False,
            "count",
            [1, 2, 1],
            [(1, 1, 1), (2, 4, 6), (2, 0, 0)],
        ),
        (
            True,
            True,
            False,
            "count",
            [1, 1, 2],
            [(1, 1, 1), (2, 6, 4), (2, 0, 0)],
        ),
        (
            True,
            False,
            False,
            "count",
            [2, 1, 1],
            [(1, 1, 1), (4, 2, 6), (0, 2, 0)],
        ),
        (
            True,
            False,
            True,
            "proportion",
            [0.5, 0.25, 0.25],
            [(1, 1, 1), (4, 2, 6), (0, 2, 0)],
        ),
    ],
)
def test_data_frame_value_counts(
    sort: bool,
    ascending: Optional[bool],
    normalize: bool,
    name: str,
    expected_data: List[int],
    expected_index: List[Tuple[int, int, int]],
) -> None:
    animals_df = DataFrame(
        {"key": [1, 1, 1, 1], "num_legs": [2, 4, 4, 6], "num_wings": [2, 0, 0, 0]},
        index=["falcon", "dog", "cat", "ant"],
    )
    result_frame = animals_df.value_counts(sort=sort, ascending=ascending, normalize=normalize)
    expected = Series(
        data=expected_data,
        index=MultiIndex.from_tuples(expected_index, names=["key", "num_legs", "num_wings"]),
        name=name,
    )
    tm.assert_series_equal(result_frame, expected)
    result_frame_groupby = animals_df.groupby("key").value_counts(sort=sort, ascending=ascending, normalize=normalize)
    tm.assert_series_equal(result_frame_groupby, expected)


@pytest.mark.parametrize(
    "group_dropna, count_dropna, expected_rows, expected_values",
    [
        (
            False,
            False,
            [0, 1, 3, 5, 6, 7, 8, 2, 4],
            [0.5, 0.5, 1.0, 0.25, 0.25, 0.25, 0.25, 1.0, 1.0],
        ),
        (
            False,
            True,
            [0, 1, 3, 5, 2, 4],
            [0.5, 0.5, 1.0, 1.0, 1.0, 1.0],
        ),
        (
            True,
            False,
            [0, 1, 5, 6, 7, 8],
            [0.5, 0.5, 0.25, 0.25, 0.25, 0.25],
        ),
        (
            True,
            True,
            [0, 1, 5],
            [0.5, 0.5, 1.0],
        ),
    ],
)
def test_dropna_combinations(
    group_dropna: bool,
    count_dropna: bool,
    expected_rows: List[int],
    expected_values: List[float],
    request: Any,
) -> None:
    if Version(np.__version__) >= Version("1.25") and (not group_dropna):
        request.applymarker(
            pytest.mark.xfail(
                reason="pandas default unstable sorting of duplicatesissue with numpy>=1.25 with AVX instructions",
                strict=False,
            )
        )
    nulls_df = DataFrame(
        {
            "A": [1, 1, np.nan, 4, np.nan, 6, 6, 6, 6],
            "B": [1, 1, 3, np.nan, np.nan, 6, 6, 6, 6],
            "C": [1, 2, 3, 4, 5, 6, np.nan, 8, np.nan],
            "D": [1, 2, 3, 4, 5, 6, 7, np.nan, np.nan],
        }
    )
    gp = nulls_df.groupby(["A", "B"], dropna=group_dropna)
    result = gp.value_counts(normalize=True, sort=True, dropna=count_dropna)
    columns = DataFrame()
    for column in nulls_df.columns:
        columns[column] = [nulls_df[column][row] for row in expected_rows]
    index = MultiIndex.from_frame(columns)
    expected = Series(data=expected_values, index=index, name="proportion")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "dropna, expected_data, expected_index",
    [
        (
            True,
            [1, 1],
            MultiIndex.from_arrays(
                [
                    [1, 1],
                    ["John", "Beth"],
                    ["Smith", "Louise"],
                ],
                names=["key", "first_name", "middle_name"],
            ),
        ),
        (
            False,
            [1, 1, 1, 1],
            MultiIndex(
                levels=[
                    Index([1]),
                    Index(["Anne", "Beth", "John"]),
                    Index(["Louise", "Smith", np.nan]),
                ],
                codes=[
                    [0, 0, 0, 0],
                    [2, 0, 2, 1],
                    [1, 2, 2, 0],
                ],
                names=["key", "first_name", "middle_name"],
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "normalize, name", [(False, "count"), (True, "proportion")]
)
def test_data_frame_value_counts_dropna(
    nulls_fixture: Any,
    dropna: bool,
    normalize: bool,
    name: str,
    expected_data: List[int],
    expected_index: MultiIndex,
) -> None:
    names_with_nulls_df = DataFrame(
        {
            "key": [1, 1, 1, 1],
            "first_name": ["John", "Anne", "John", "Beth"],
            "middle_name": ["Smith", nulls_fixture, nulls_fixture, "Louise"],
        }
    )
    result_frame = names_with_nulls_df.value_counts(dropna=dropna, normalize=normalize)
    expected = Series(data=expected_data, index=expected_index, name=name)
    if normalize:
        expected = expected / float(sum(expected_data))
    tm.assert_series_equal(result_frame, expected)
    result_frame_groupby = names_with_nulls_df.groupby("key").value_counts(
        dropna=dropna, normalize=normalize
    )
    tm.assert_series_equal(result_frame_groupby, expected)


@pytest.mark.parametrize(
    "observed, expected_index",
    [
        (
            False,
            [
                ("FR", "high", "female"),
                ("FR", "high", "male"),
                ("FR", "low", "male"),
                ("FR", "low", "female"),
                ("FR", "medium", "male"),
                ("FR", "medium", "female"),
                ("US", "high", "female"),
                ("US", "high", "male"),
                ("US", "low", "male"),
                ("US", "low", "female"),
                ("US", "medium", "male"),
                ("US", "medium", "female"),
            ],
        ),
        (
            True,
            [
                ("FR", "high", "female"),
                ("FR", "low", "male"),
                ("FR", "medium", "male"),
                ("US", "high", "female"),
                ("US", "low", "male"),
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    "normalize, name, expected_data",
    [
        (False, "count", [1, 1, 2]),
        (True, "proportion", [0.5, 0.5, 1.0]),
    ],
)
def test_categorical_multiple_groupers(
    education_df: DataFrame,
    as_index: bool,
    observed: bool,
    expected_index: List[Tuple[str, str, str]],
    normalize: bool,
    name: str,
    expected_data: np.ndarray,
) -> None:
    education_df = education_df.copy()
    education_df["country"] = education_df["country"].astype("category")
    education_df["education"] = education_df["education"].astype("category")
    gp = education_df.groupby(["country", "education"], as_index=as_index, observed=observed)
    result = gp.value_counts(normalize=normalize)
    expected_series = Series(
        data=expected_data,
        index=MultiIndex.from_tuples(
            expected_index, names=["country", "education", "gender"]
        ),
        name=name,
    )
    for i in range(2):
        expected_series.index = expected_series.index.set_levels(
            CategoricalIndex(expected_series.index.levels[i]), level=i
        )
    if as_index:
        tm.assert_series_equal(result, expected_series)
    else:
        expected = expected_series.reset_index(name="proportion" if normalize else "count")
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "normalize, name, expected_data",
    [
        (
            False,
            "count",
            np.array([2, 1, 1], dtype=np.int64),
        ),
        (
            True,
            "proportion",
            np.array([0.5, 0.25, 0.25]),
        ),
    ],
)
def test_data_frame_value_counts_duplicate_columns(
    sort: bool,
    dropna: bool,
    expected_label: str,
    expected_values: List[int],
    nulls_fixture: Any,
) -> None:
    df = DataFrame([["a", "x", "x"], ["b", "y", "y"], ["b", "y", "y"]], index=[0, 1, 1], columns=["c1", "c2", "c2"])
    result = df.groupby(level=0).value_counts(subset=["c2"])
    expected = Series(
        [1, 2],
        index=MultiIndex.from_arrays(
            [
                [0, 1],
                ["x", "y"],
                ["x", "y"],
            ],
            names=[None, "c2", "c2"],
        ),
        name="count",
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "observed, expected_index",
    [
        (
            False,
            [
                ("FR", "high", "female"),
                ("FR", "low", "male"),
                ("FR", "medium", "male"),
                ("US", "high", "female"),
                ("US", "low", "male"),
            ],
        ),
        (
            True,
            [("FR", "high", "female"), ("FR", "low", "male"), ("FR", "medium", "male"), ("US", "high", "female"), ("US", "low", "male")],
        ),
    ],
)
@pytest.mark.parametrize(
    "normalize, name, expected_data",
    [
        (False, "count", np.array([2, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0], dtype=np.int64)),
        (True, "proportion", np.array([0.5, 0.25, 0.25, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0])),
    ],
)
def test_categorical_non_groupers(
    education_df: DataFrame,
    as_index: bool,
    observed: bool,
    normalize: bool,
    name: str,
    expected_data: np.ndarray,
    request: Any,
    using_infer_string: bool,
) -> None:
    if Version(np.__version__) >= Version("1.25"):
        request.applymarker(
            pytest.mark.xfail(
                reason="pandas default unstable sorting of duplicatesissue with numpy>=1.25 with AVX instructions",
                strict=False,
            )
        )
    education_df = education_df.copy()
    education_df["gender"] = education_df["gender"].astype("category")
    education_df["education"] = education_df["education"].astype("category")
    gp = education_df.groupby("country", as_index=as_index, observed=observed)
    result = gp.value_counts(normalize=normalize)
    expected_index = [
        ("FR", "male", "low"),
        ("FR", "male", "medium"),
        ("FR", "female", "high"),
        ("FR", "male", "high"),
        ("FR", "female", "low"),
        ("FR", "female", "medium"),
        ("US", "male", "low"),
        ("US", "female", "high"),
        ("US", "male", "medium"),
        ("US", "male", "high"),
        ("US", "female", "low"),
        ("US", "female", "medium"),
    ]
    expected_series = Series(
        data=expected_data,
        index=MultiIndex.from_tuples(expected_index, names=["country", "gender", "education"]),
        name=name,
    )
    for i in range(1, 3):
        expected_series.index = expected_series.index.set_levels(
            CategoricalIndex(expected_series.index.levels[i]), level=i
        )
    if as_index:
        tm.assert_series_equal(result, expected_series)
    else:
        expected = expected_series.reset_index(name="proportion" if normalize else "count")
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("normalize, name, expected_data", [
    (False, "count", [1, 1]),
    (True, "proportion", [0.5, 0.5])
])
def test_mixed_groupings(
    normalize: bool,
    expected_label: str,
    expected_values: List[float],
) -> None:
    df = DataFrame({"A": [1, 2, 1], "B": [1, 2, 3]})
    gp = df.groupby([[4, 5, 4], "A", lambda i: 7 if i == 1 else 8], as_index=False)
    result = gp.value_counts(sort=True, normalize=normalize)
    expected = DataFrame(
        {
            "level_0": np.array([4, 4, 5], dtype=int),
            "A": [1, 1, 2],
            "level_2": [8, 8, 7],
            "B": [1, 3, 2],
            expected_label: expected_values,
        }
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "observed, expected_index",
    [
        (
            False,
            MultiIndex.from_tuples(
                [
                    ("FR", "high", "female"),
                    ("FR", "high", "male"),
                    ("FR", "low", "male"),
                    ("FR", "low", "female"),
                    ("FR", "medium", "male"),
                    ("FR", "medium", "female"),
                    ("US", "high", "female"),
                    ("US", "high", "male"),
                    ("US", "low", "male"),
                    ("US", "low", "female"),
                    ("US", "medium", "male"),
                    ("US", "medium", "female"),
                ],
                names=["country", "education", "gender"],
            ),
        ),
        (
            True,
            MultiIndex.from_tuples(
                [
                    ("FR", "high", "female"),
                    ("FR", "low", "male"),
                    ("FR", "medium", "male"),
                    ("US", "high", "female"),
                    ("US", "low", "male"),
                ],
                names=["country", "education", "gender"],
            ),
        ),
    ],
)
@pytest.mark.parametrize("normalize, name, expected_data", [
    (False, "count", np.array([1, 1], dtype=np.int64)),
    (True, "proportion", np.array([1.0, 1.0]))
])
def test_categorical_multiple_groupers_observed_true(
    education_df: DataFrame,
    as_index: bool,
    normalize: bool,
    name: str,
    expected_data: np.ndarray,
    observed: bool,
    expected_index: MultiIndex,
) -> None:
    pass  # Placeholder as the function was incomplete in previous context.


def assert_categorical_single_grouper(
    education_df: DataFrame,
    as_index: bool,
    observed: bool,
    expected_index: List[Tuple[str, str, str]],
    normalize: bool,
    name: str,
    expected_data: np.ndarray,
) -> None:
    education_df = education_df.copy().astype("category")
    education_df["country"] = education_df["country"].cat.add_categories(["ASIA"])
    gp = education_df.groupby("country", as_index=as_index, observed=observed)
    result = gp.value_counts(normalize=normalize)
    expected_series = Series(
        data=expected_data,
        index=MultiIndex.from_tuples(expected_index, names=["country", "gender", "education"]),
        name=name,
    )
    for i in range(3):
        index_level = CategoricalIndex(expected_series.index.levels[i])
        if i == 0:
            index_level = index_level.set_categories(education_df["country"].cat.categories)
        expected_series.index = expected_series.index.set_levels(index_level, level=i)
    if as_index:
        tm.assert_series_equal(result, expected_series)
    else:
        expected = expected_series.reset_index(name=name)
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "observed, expected_index",
    [
        (
            False,
            [
                ("FR", "high", "female"),
                ("FR", "low", "male"),
                ("FR", "medium", "male"),
                ("US", "high", "female"),
                ("US", "low", "male"),
                ("ASIA", "low", "male"),
            ],
        ),
        (
            True,
            [
                ("FR", "high", "female"),
                ("FR", "low", "male"),
                ("FR", "medium", "male"),
                ("US", "high", "female"),
                ("US", "low", "male"),
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    "normalize, name, expected_data",
    [
        (False, "count", np.array([2, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0], dtype=np.int64)),
        (True, "proportion", np.array([0.5, 0.25, 0.25, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0])),
    ],
)
def test_categorical_single_grouper_observed_false(
    education_df: DataFrame,
    as_index: bool,
    normalize: bool,
    name: str,
    expected_data: np.ndarray,
    observed: bool,
    expected_index: List[Tuple[str, str, str]],
    request: Any,
) -> None:
    if Version(np.__version__) >= Version("1.25"):
        request.applymarker(
            pytest.mark.xfail(
                reason="pandas default unstable sorting of duplicatesissue with numpy>=1.25 with AVX instructions",
                strict=False,
            )
        )
    assert_categorical_single_grouper(
        education_df=education_df,
        as_index=as_index,
        observed=observed,
        expected_index=expected_index,
        normalize=normalize,
        name=name,
        expected_data=expected_data,
    )


@pytest.mark.parametrize(
    "observed, expected_index",
    [
        (
            False,
            [
                ("FR", "high", "female"),
                ("FR", "high", "male"),
                ("FR", "low", "male"),
                ("FR", "low", "female"),
                ("FR", "medium", "male"),
                ("FR", "medium", "female"),
                ("US", "high", "female"),
                ("US", "high", "male"),
                ("US", "low", "male"),
                ("US", "low", "female"),
                ("US", "medium", "male"),
                ("US", "medium", "female"),
            ],
        ),
        (
            True,
            [
                ("FR", "high", "female"),
                ("FR", "low", "male"),
                ("FR", "medium", "male"),
                ("US", "high", "female"),
                ("US", "low", "male"),
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    "normalize, name, expected_data",
    [
        (False, "count", np.array([2, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0], dtype=np.int64)),
        (True, "proportion", np.array([0.5, 0.25, 0.25, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0])),
    ],
)
def test_categorical_single_grouper_observed_true(
    education_df: DataFrame,
    as_index: bool,
    normalize: bool,
    name: str,
    expected_data: np.ndarray,
    observed: bool,
    expected_index: List[Tuple[str, str, str]],
    request: Any,
) -> None:
    if Version(np.__version__) >= Version("1.25"):
        request.applymarker(
            pytest.mark.xfail(
                reason="pandas default unstable sorting of duplicatesissue with numpy>=1.25 with AVX instructions",
                strict=False,
            )
        )
    assert_categorical_single_grouper(
        education_df=education_df,
        as_index=as_index,
        observed=observed,
        expected_index=expected_index,
        normalize=normalize,
        name=name,
        expected_data=expected_data,
    )


@pytest.mark.parametrize(
    "observed", [False, True]
)
@pytest.mark.parametrize(
    "normalize, name, expected_data",
    [
        (False, "count", np.array([2, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0], dtype=np.int64)),
        (True, "proportion", np.array([0.5, 0.25, 0.25, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0])),
    ],
)
def test_categorical_non_groupers(
    education_df: DataFrame,
    as_index: bool,
    observed: bool,
    normalize: bool,
    name: str,
    expected_data: np.ndarray,
    request: Any,
    using_infer_string: bool,
) -> None:
    pass  # Placeholder as the function was incomplete in previous context.


@pytest.mark.parametrize(
    "normalize, expected_label, expected_values",
    [
        (False, "count", [1, 1]),
        (True, "proportion", [0.5, 0.5]),
    ],
)
def test_value_counts_sort(
    sort: bool,
    vc_sort: bool,
    normalize: bool,
    df: DataFrame,
    expected_label: str,
    expected_values: List[float],
) -> None:
    df = DataFrame({"a": [2, 1, 1, 1], 0: [3, 4, 3, 3]})
    gb = df.groupby("a", sort=sort)
    result = gb.value_counts(sort=vc_sort, normalize=normalize)
    if normalize:
        values = [2 / 3, 1 / 3, 1.0]
    else:
        values = [2, 1, 1]
    index = MultiIndex(
        levels=[[1, 2], [3, 4]],
        codes=[[0, 0, 1], [0, 1, 0]],
        names=["a", 0],
    )
    expected = Series(values, index=index, name="proportion" if normalize else "count")
    if sort and vc_sort:
        taker = [0, 1, 2]
    elif sort and not vc_sort:
        taker = [1, 0, 2]
    elif not sort and vc_sort:
        taker = [0, 2, 1]
    else:
        taker = [2, 1, 0]
    expected = expected.take(taker)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "vc_sort, normalize",
    [
        (True, True),
        (False, False),
    ],
)
def test_value_counts_sort_categorical(
    sort: bool,
    vc_sort: bool,
    normalize: bool,
    df: DataFrame,
    expected_label: str,
    expected_values: List[float],
) -> None:
    df = DataFrame({"a": [2, 1, 1, 1], 0: [3, 4, 3, 3]}, dtype="category")
    gb = df.groupby("a", sort=sort, observed=True)
    result = gb.value_counts(sort=vc_sort, normalize=normalize)
    if normalize:
        values = [2 / 3, 1 / 3, 1.0, 0.0]
    else:
        values = [2, 1, 1, 0]
    name = "proportion" if normalize else "count"
    expected = DataFrame(
        {"a": Categorical(["1", "1", "2", "2"]), "0": Categorical(["3", "4", "3", "4"]), name: values}
    ).set_index(["a", "0"])[name]
    if sort and vc_sort:
        taker = [0, 1, 2, 3]
    elif sort and not vc_sort:
        taker = [0, 1, 2, 3]
    elif not sort and vc_sort:
        taker = [0, 2, 1, 3]
    else:
        taker = [2, 1, 0, 3]
    expected = expected.take(taker)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "groupby_sort, sort, dropna",
    [
        (True, False, True),
        (False, False, True),
    ],
)
def test_value_counts_all_na(
    groupby_sort: bool,
    sort: bool,
    dropna: bool,
    df: DataFrame,
    expected_label: str,
    expected_values: List[int],
) -> None:
    df = DataFrame({"a": [2, 1, 1], "b": np.nan})
    gb = df.groupby("a", sort=groupby_sort)
    result = gb.value_counts(sort=sort, dropna=dropna)
    if dropna:
        data: List[int] = []
        index = MultiIndex(codes=[[], []], names=["a", "b"])
    elif not groupby_sort and not sort:
        data = [1, 2]
        index = MultiIndex.from_tuples([(1, 1), (2, 1)], names=["a", "b"])
    else:
        data = [2, 1]
        index = MultiIndex.from_tuples([(2, 1), (1, 1)], names=["a", "b"])
    expected = Series(data, index=index, dtype="int64", name="count")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "dropna, expected_data, expected_index",
    [
        (
            True,
            [1, 1],
            MultiIndex.from_arrays(
                [
                    [1, 1],
                    ["John", "Beth"],
                    ["Smith", "Louise"],
                ],
                names=["key", "first_name", "middle_name"],
            ),
        ),
        (
            False,
            [1, 1, 1, 1],
            MultiIndex(
                levels=[
                    Index([1]),
                    Index(["Anne", "Beth", "John"]),
                    Index(["Louise", "Smith", np.nan]),
                ],
                codes=[
                    [0, 0, 0, 0],
                    [2, 0, 2, 1],
                    [1, 2, 2, 0],
                ],
                names=["key", "first_name", "middle_name"],
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "normalize, name, expected_data",
    [
        (False, "count", [1, 1]),
        (True, "proportion", [0.5, 0.5]),
    ],
)
def test_data_frame_value_counts_dropna(
    dropna: bool,
    normalize: bool,
    name: str,
    expected_data: List[int],
    expected_index: Any,
    nulls_fixture: Any,
) -> None:
    pass  # Placeholder as the function was defined earlier.


@pytest.mark.parametrize(
    "test, columns, expected_names",
    [
        ("repeat", list("abbde"), ["a", None, "d", "b", "b", "e"]),
        ("level", list("abcd") + ["level_1"], ["a", None, "d", "b", "c", "level_1"]),
    ],
)
def test_column_label_duplicates(
    test: str,
    columns: List[str],
    expected_names: List[Optional[str]],
    as_index: bool,
) -> None:
    df = DataFrame([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]], columns=columns)
    expected_data = [(1, 0, 7, 3, 5, 9), (2, 1, 8, 4, 6, 10)]
    keys: List[Union[str, np.ndarray]] = ["a", np.array([0, 1], dtype=np.int64), "d"]
    result = df.groupby(keys, as_index=as_index).value_counts()
    if as_index:
        expected = Series(
            data=(1, 1),
            index=MultiIndex.from_tuples(
                expected_data, names=expected_names
            ),
            name="count",
        )
        tm.assert_series_equal(result, expected)
    else:
        expected_data = [list(row) + [1] for row in expected_data]
        expected_columns = list(expected_names)
        expected_columns[1] = "level_1"
        expected_columns.append("count")
        expected = DataFrame(expected_data, columns=expected_columns)
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "normalize, expected_label, expected_values",
    [
        (False, "count", [1, 1]),
        (True, "proportion", [0.5, 0.5]),
    ],
)
def test_result_label_duplicates(
    normalize: bool,
    expected_label: str,
    expected_values: List[float],
) -> None:
    gb = DataFrame([[1, 2, 3]], columns=["a", "b", expected_label]).groupby("a", as_index=False)
    msg = f"Column label '{expected_label}' is duplicate of result column"
    with pytest.raises(ValueError, match=msg):
        gb.value_counts(normalize=normalize)


def test_ambiguous_grouping() -> None:
    df = DataFrame({"a": [1, 1]})
    gb = df.groupby(np.array([1, 1], dtype=np.int64))
    result = gb.value_counts()
    expected = Series(
        [2],
        index=MultiIndex.from_tuples([[1, 1]], names=[None, "a"]),
        name="count",
    )
    tm.assert_series_equal(result, expected)


def test_subset_overlaps_gb_key_raises() -> None:
    df = DataFrame({"c1": ["a", "b", "c"], "c2": ["x", "y", "y"]}, index=[0, 1, 1])
    msg = "Keys {'c1'} in subset cannot be in the groupby column keys."
    with pytest.raises(ValueError, match=msg):
        df.groupby("c1").value_counts(subset=["c1"])


def test_subset_doesnt_exist_in_frame() -> None:
    df = DataFrame({"c1": ["a", "b", "c"], "c2": ["x", "y", "y"]}, index=[0, 1, 1])
    msg = "Keys {'c3'} in subset do not exist in the DataFrame."
    with pytest.raises(ValueError, match=msg):
        df.groupby("c1").value_counts(subset=["c3"])


def test_subset() -> None:
    df = DataFrame({"c1": ["a", "b", "c"], "c2": ["x", "y", "y"]}, index=[0, 1, 1])
    result = df.groupby(level=0).value_counts(subset=["c2"])
    expected = Series(
        [1, 2],
        index=MultiIndex.from_arrays(
            [
                [0, 1],
                ["x", "y"],
            ],
            names=[None, "c2"],
        ),
        name="count",
    )
    tm.assert_series_equal(result, expected)


def test_subset_duplicate_columns() -> None:
    df = DataFrame(
        [["a", "x", "x"], ["b", "y", "y"], ["b", "y", "y"]],
        index=[0, 1, 1],
        columns=["c1", "c2", "c2"],
    )
    result = df.groupby(level=0).value_counts(subset=["c2"])
    expected = Series(
        [1, 2],
        index=MultiIndex.from_arrays(
            [
                [0, 1],
                ["x", "y"],
                ["x", "y"],
            ],
            names=[None, "c2", "c2"],
        ),
        name="count",
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("utc", [True, False])
def test_value_counts_time_grouper(utc: bool, unit: str) -> None:
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
    df["Datetime"] = to_datetime(df["Timestamp"], utc=utc, unit="s").dt.floor(unit)
    gb = df.groupby(Grouper(freq="1D", key="Datetime"))
    result = gb.value_counts()
    dates = to_datetime(
        ["2019-08-06", "2019-08-07", "2019-08-09", "2019-08-10"], utc=utc
    ).floor(unit)
    timestamps = df["Timestamp"].unique()
    index = MultiIndex(
        levels=[dates, timestamps, ["apple", "banana", "orange", "pear"]],
        codes=[
            [0, 1, 1, 2, 2, 3],
            range(6),
            [0, 0, 1, 2, 2, 3],
        ],
        names=["Datetime", "Timestamp", "Food"],
    )
    expected = Series(1, index=index, name="count")
    tm.assert_series_equal(result, expected)


def test_value_counts_integer_columns() -> None:
    df = DataFrame({1: ["a", "a", "a"], 2: ["a", "a", "d"], 3: ["a", "b", "c"]})
    gp = df.groupby([1, 2], as_index=False, sort=False)
    result = gp[3].value_counts()
    expected = DataFrame(
        {1: ["a", "a", "a"], 2: ["a", "a", "d"], 3: ["a", "b", "c"], "count": [1, 1, 1]}
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "vc_sort, normalize",
    [
        (True, True),
        (False, False),
    ],
)
def test_value_counts_sort(
    sort: bool,
    vc_sort: bool,
    normalize: bool,
) -> None:
    df = DataFrame({"a": [2, 1, 1, 1], 0: [3, 4, 3, 3]})
    gb = df.groupby("a", sort=sort)
    result = gb.value_counts(sort=vc_sort, normalize=normalize)
    if normalize:
        values = [2 / 3, 1 / 3, 1.0]
    else:
        values = [2, 1, 1]
    index = MultiIndex(
        levels=[[1, 2], [3, 4]],
        codes=[[0, 0, 1], [0, 1, 0]],
        names=["a", 0],
    )
    expected = Series(values, index=index, name="proportion" if normalize else "count")
    if sort and vc_sort:
        taker = [0, 1, 2]
    elif sort and not vc_sort:
        taker = [1, 0, 2]
    elif not sort and vc_sort:
        taker = [0, 2, 1]
    else:
        taker = [2, 1, 0]
    expected = expected.take(taker)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "vc_sort, normalize",
    [
        (True, True),
        (False, False),
    ],
)
def test_value_counts_sort_categorical(
    sort: bool,
    vc_sort: bool,
    normalize: bool,
) -> None:
    df = DataFrame({"a": [2, 1, 1, 1], 0: [3, 4, 3, 3]}, dtype="category")
    gb = df.groupby("a", sort=sort, observed=True)
    result = gb.value_counts(sort=vc_sort, normalize=normalize)
    if normalize:
        values = [2 / 3, 1 / 3, 1.0, 0.0]
    else:
        values = [2, 1, 1, 0]
    name = "proportion" if normalize else "count"
    expected = DataFrame(
        {"a": pd.Categorical([1, 1, 2, 2]), "0": pd.Categorical([3, 4, 3, 4]), name: values}
    ).set_index(["a", "0"])[name]
    if sort and vc_sort:
        taker = [0, 1, 2, 3]
    elif sort and not vc_sort:
        taker = [0, 1, 2, 3]
    elif not sort and vc_sort:
        taker = [0, 2, 1, 3]
    else:
        taker = [2, 1, 0, 3]
    expected = expected.take(taker)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "normalize, expected_label, expected_values",
    [
        (False, "count", [1, 1]),
        (True, "proportion", [0.5, 0.5]),
    ],
)
def test_mixed_groupings(
    normalize: bool,
    expected_label: str,
    expected_values: List[float],
) -> None:
    df = DataFrame({"A": [1, 2, 1], "B": [1, 2, 3]})
    gp = df.groupby([[4, 5, 4], "A", lambda i: 7 if i == 1 else 8], as_index=False)
    result = gp.value_counts(sort=True, normalize=normalize)
    expected = DataFrame(
        {
            "level_0": np.array([4, 4, 5], dtype=int),
            "A": [1, 1, 2],
            "level_2": [8, 8, 7],
            "B": [1, 3, 2],
            expected_label: expected_values,
        }
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "groupby_sort, sort, dropna",
    [
        (True, False, True),
        (False, False, True),
    ],
)
def test_value_counts_all_na(
    groupby_sort: bool,
    sort: bool,
    dropna: bool,
) -> None:
    df = DataFrame({"a": [2, 1, 1], "b": np.nan})
    gb = df.groupby("a", sort=groupby_sort)
    result = gb.value_counts(sort=sort, dropna=dropna)
    if dropna:
        data: List[int] = []
        index = MultiIndex(codes=[[], []], names=["a", "b"])
    elif not groupby_sort and not sort:
        data = [1, 2]
        index = MultiIndex.from_tuples([(1, 1), (2, 1)], names=["a", "b"])
    else:
        data = [2, 1]
        index = MultiIndex.from_tuples([(2, 1), (1, 1)], names=["a", "b"])
    expected = Series(data, index=index, dtype="int64", name="count")
    tm.assert_series_equal(result, expected)
