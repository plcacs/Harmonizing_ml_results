import sys
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)
from typing_extensions import TypeAlias

import numpy as np
import pandas as pd
from pandas import (
    Categorical,
    CategoricalIndex,
    DataFrame,
    Index,
    MultiIndex,
    Series,
    qcut,
)
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.core.groupby.generic import DataFrameGroupBy, SeriesGroupBy

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

_ArrayLike: TypeAlias = Union[np.ndarray, List[Any], Tuple[Any, ...]]
_GroupKey: TypeAlias = Union[Hashable, List[Hashable]]
_ReductionFunc: TypeAlias = str
_TransformFunc: TypeAlias = Union[str, Callable[..., Any]]
_AggFunc: TypeAlias = Union[
    str, Callable[..., Any], List[Union[str, Callable[..., Any]]]
]

def cartesian_product_for_groupers(
    result: DataFrame,
    args: Sequence[Union[Categorical, CategoricalIndex, _ArrayLike]],
    names: List[str],
    fill_value: Any = np.nan,
) -> DataFrame: ...

_results_for_groupbys_with_missing_categories: Dict[str, Any] = ...

def test_apply_use_categorical_name(df: DataFrame) -> None: ...

def test_basic() -> None: ...

def test_basic_single_grouper() -> None: ...

def test_basic_string(using_infer_string: bool) -> None: ...

def test_basic_monotonic() -> None: ...

def test_basic_non_monotonic() -> None: ...

def test_basic_cut_grouping() -> None: ...

def test_more_basic() -> None: ...

def test_level_get_group(observed: bool) -> None: ...

def test_sorting_with_different_categoricals() -> None: ...

@pytest.mark.parametrize("ordered", [True, False])
def test_apply(ordered: bool) -> None: ...

def test_observed(
    request: pytest.FixtureRequest,
    using_infer_string: bool,
    observed: bool,
) -> None: ...

def test_observed_single_column(observed: bool) -> None: ...

def test_observed_two_columns(observed: bool) -> None: ...

def test_observed_with_as_index(observed: bool) -> None: ...

def test_observed_codes_remap(observed: bool) -> None: ...

def test_observed_perf() -> None: ...

def test_observed_groups(observed: bool) -> None: ...

@pytest.mark.parametrize(
    "keys, expected_values, expected_index_levels",
    [
        ("a", [15, 9, 0], CategoricalIndex([1, 2, 3], name="a")),
        (
            ["a", "b"],
            [7, 8, 0, 0, 0, 9, 0, 0, 0],
            [CategoricalIndex([1, 2, 3], name="a"), Index([4, 5, 6])],
        ),
        (
            ["a", "a2"],
            [15, 0, 0, 0, 9, 0, 0, 0, 0],
            [CategoricalIndex([1, 2, 3], name="a"), CategoricalIndex([1, 2, 3], name="a")],
        ),
    ],
)
@pytest.mark.parametrize("test_series", [True, False])
def test_unobserved_in_index(
    keys: Union[str, List[str]],
    expected_values: List[Union[int, float]],
    expected_index_levels: Union[CategoricalIndex, List[Union[CategoricalIndex, Index]]],
    test_series: bool,
) -> None: ...

def test_observed_groups_with_nan(observed: bool) -> None: ...

def test_observed_nth() -> None: ...

def test_dataframe_categorical_with_nan(observed: bool) -> None: ...

@pytest.mark.parametrize("ordered", [True, False])
def test_dataframe_categorical_ordered_observed_sort(
    ordered: bool, observed: bool, sort: bool
) -> None: ...

def test_datetime() -> None: ...

def test_categorical_index() -> None: ...

def test_describe_categorical_columns() -> None: ...

def test_unstack_categorical() -> None: ...

def test_bins_unequal_len() -> None: ...

@pytest.mark.parametrize(
    ["series", "data"],
    [
        (Series(range(4)), {"A": [0, 3], "B": [1, 2]}),
        (Series(range(4)).rename(lambda idx: idx + 1), {"A": [2], "B": [0, 1]}),
        (Series(range(7)), {"A": [0, 3], "B": [1, 2]}),
    ],
)
def test_categorical_series(
    series: Series, data: Dict[str, List[int]]
) -> None: ...

def test_as_index() -> None: ...

def test_preserve_categories() -> None: ...

def test_preserve_categories_ordered_false() -> None: ...

@pytest.mark.parametrize("col", ["C1", "C2"])
def test_preserve_categorical_dtype(col: str) -> None: ...

@pytest.mark.parametrize(
    "func, values",
    [
        ("first", ["second", "first"]),
        ("last", ["fourth", "third"]),
        ("min", ["fourth", "first"]),
        ("max", ["second", "third"]),
    ],
)
def test_preserve_on_ordered_ops(func: str, values: List[str]) -> None: ...

def test_categorical_no_compress() -> None: ...

def test_categorical_no_compress_string() -> None: ...

def test_groupby_empty_with_category() -> None: ...

def test_sort() -> None: ...

@pytest.mark.parametrize("ordered", [True, False])
def test_sort2(sort: bool, ordered: bool) -> None: ...

@pytest.mark.parametrize("ordered", [True, False])
def test_sort_datetimelike(sort: bool, ordered: bool) -> None: ...

def test_empty_sum() -> None: ...

def test_empty_prod() -> None: ...

def test_groupby_multiindex_categorical_datetime() -> None: ...

@pytest.mark.parametrize(
    "as_index, expected",
    [
        (
            True,
            Series(
                index=MultiIndex.from_arrays(
                    [Series([1, 1, 2], dtype="category"), [1, 2, 2]],
                    names=["a", "b"],
                ),
                data=[1, 2, 3],
                name="x",
            ),
        ),
        (
            False,
            DataFrame(
                {"a": Series([1, 1, 2], dtype="category"), "b": [1, 2, 2], "x": [1, 2, 3]}
            ),
        ),
    ],
)
def test_groupby_agg_observed_true_single_column(
    as_index: bool, expected: Union[Series, DataFrame]
) -> None: ...

@pytest.mark.parametrize("fill_value", [None, np.nan, pd.NaT])
def test_shift(fill_value: Any) -> None: ...

@pytest.fixture
def df_cat(df: DataFrame) -> DataFrame: ...

@pytest.mark.parametrize("operation", ["agg", "apply"])
def test_seriesgroupby_observed_true(
    df_cat: DataFrame, operation: Literal["agg", "apply"]
) -> None: ...

@pytest.mark.parametrize("operation", ["agg", "apply"])
@pytest.mark.parametrize("observed", [False, None])
def test_seriesgroupby_observed_false_or_none(
    df_cat: DataFrame, observed: Union[bool, None], operation: Literal["agg", "apply"]
) -> None: ...

@pytest.mark.parametrize(
    "observed, index, data",
    [
        (
            True,
            MultiIndex.from_arrays(
                [
                    Index(["bar"] * 4 + ["foo"] * 4, dtype="category", name="A"),
                    Index(
                        ["one", "one", "three", "three", "one", "one", "two", "two"],
                        dtype="category",
                        name="B",
                    ),
                    Index(["min", "max"] * 4),
                ]
            ),
            [2, 2, 4, 4, 1, 1, 3, 3],
        ),
        (
            False,
            MultiIndex.from_product(
                [
                    CategoricalIndex(["bar", "foo"], ordered=False),
                    CategoricalIndex(["one", "three", "two"], ordered=False),
                    Index(["min", "max"]),
                ],
                names=["A", "B", None],
            ),
            [2, 2, 4, 4, np.nan, np.nan, 1, 1, np.nan, np.nan, 3, 3],
        ),
        (
            None,
            MultiIndex.from_product(
                [
                    CategoricalIndex(["bar", "foo"], ordered=False),
                    CategoricalIndex(["one", "three", "two"], ordered=False),
                    Index(["min", "max"]),
                ],
                names=["A", "B", None],
            ),
            [2, 2, 4, 4, np.nan, np.nan, 1, 1, np.nan, np.nan, 3, 3],
        ),
    ],
)
def test_seriesgroupby_observed_apply_dict(
    df_cat: DataFrame,
    observed: Union[bool, None],
    index: MultiIndex,
    data: List[Union[int, float]],
) -> None: ...

def test_groupby_categorical_series_dataframe_consistent(df_cat: DataFrame) -> None: ...

def test_groupby_cat_preserves_structure(
    observed: bool, ordered: bool
) -> None: ...

def test_get_nonexistent_category() -> None: ...

def test_series_groupby_on_2_categoricals_unobserved(
    reduction_func: str, observed: bool
) -> None: ...

def test_series_groupby_on_2_categoricals_unobserved_zeroes_or_nans(
    reduction_func: str, request: pytest.FixtureRequest
) -> None: ...

def test_dataframe_groupby_on_2_categoricals_when_observed_is_true(
    reduction_func: str,
) -> None: ...

@pytest.mark.parametrize("observed", [False, None])
def test_dataframe_groupby_on_2_categoricals_when_observed_is_false(
    reduction_func: str, observed: Union[bool, None]
) -> None: ...

def test_series_groupby_categorical_aggregation_getitem() -> None: ...

@pytest.mark.parametrize(
    "func, expected_values",
    [(Series.nunique, [1, 1, 2]), (Series.count, [1, 2, 2])],
)
def test_groupby_agg_categorical_columns(
    func: Callable[..., Any], expected_values: List[int]
) -> None: ...

def test_groupby_agg_non_numeric() -> None: ...

@pytest.mark.parametrize("func", ["first", "last"])
def test_groupby_first_returned_categorical_instead_of_dataframe(
    func: Literal["first", "last"]
) -> None: ...

def test_read_only_category_no_sort() -> None: ...

def test_sorted_missing_category_values() -> None: ...

def test_agg_cython_category_not_implemented_fallback() -> None: ...

def test_aggregate_categorical_with_isnan() -> None: ...

def test_categorical_transform() -> None: ...

@pytest.mark.parametrize("func", ["first", "last"])
def test_series_groupby_first_on_categorical_col_grouped_on_2_categoricals(
    func: Literal["first", "last"], observed: bool
) -> None: ...

@pytest.mark.parametrize("func", ["first", "last"])
def test_df_groupby_first_on_categorical_col_grouped_on_2_categoricals(
    func: Literal["first", "last"], observed: bool
) -> None: ...

def test_groupby_categorical_indices_unused_categories() -> None: ...

@pytest.mark.parametrize("func", ["first", "last"])
def test_groupby_last_first_preserve_categoricaldtype(
    func: Literal["first", "last"]
) -> None: ...

def test_groupby_categorical_observed_nunique() -> None: ...

def test_groupby_categorical_aggregate_functions() -> None: ...

def test_groupby_categorical_dropna(observed: bool, dropna: bool) -> None: ...

@pytest.mark.parametrize("index_kind", ["range", "single", "multi"])
@pytest.mark.parametrize("ordered", [True, False])
def test_category_order_reducer(
    request: pytest.FixtureRequest,
    as_index: bool,
    sort: bool,
    observed: bool,
    reduction_func: str,
    index_kind: Literal["range", "single", "multi"],
    ordered: bool,
) -> None: ...

@pytest.mark.parametrize("index_kind", ["single", "multi"])
@pytest.mark.parametrize("ordered", [True, False])
def test_category_order_transformer(
    as_index: bool,
    sort: bool,
    observed: bool,
    transformation_func: str,
    index_kind: Literal["single", "multi"],
    ordered: bool,
) -> None: ...

@pytest.mark.parametrize("index_kind", ["range", "single", "multi"])
@pytest.mark.parametrize("method", ["head", "tail"])
@pytest.mark.parametrize("ordered", [True, False])
def test_category_order_head_tail(
    as_index: bool,
    sort: bool,
    observed: bool,
    method: Literal["head", "tail"],
    index_kind: Literal["range", "single", "multi"],
    ordered: bool,
) -> None: ...

@pytest.mark.parametrize("index_kind", ["range", "single", "multi"])
@pytest.mark.parametrize("method", ["apply", "agg", "transform"])
@pytest.mark.parametrize("ordered", [True, False])
def test_category_order_apply(
    as_index: bool,
    sort: bool,
    observed: bool,
    method: Literal["apply", "agg", "transform"],
    index_kind: Literal["range", "single", "multi"],
    ordered: bool,
) -> None: ...

@pytest.mark.parametrize("index_kind", ["range", "single", "multi"])
def test_many_categories(
    as_index: bool, sort: bool, index_kind: Literal["range", "single", "multi"], ordered: bool
) -> None: ...

@pytest.mark.parametrize("test_series", [True, False])
@pytest.mark.parametrize("keys", [["a1"], ["a1", "a2"]])
def test_agg_list(
    request: pytest.FixtureRequest,
    as_index: bool,
    observed: bool,
    reduction_func: str,
    test_series: bool,
    keys: List[str],
) -> None: ...