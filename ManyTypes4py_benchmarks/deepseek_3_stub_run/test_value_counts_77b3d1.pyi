import numpy as np
import pandas as pd
import pytest
from pandas import Categorical, CategoricalIndex, DataFrame, Grouper, Index, MultiIndex, Series, Timestamp
from pandas._testing import assert_series_equal, assert_frame_equal
from typing import Any, Callable, Iterable, List, Literal, Optional, Sequence, Tuple, Union

def tests_value_counts_index_names_category_column() -> None: ...

def seed_df(seed_nans: bool, n: int, m: int) -> DataFrame: ...

@pytest.mark.slow
@pytest.mark.parametrize('seed_nans', [True, False])
@pytest.mark.parametrize('num_rows', [10, 50])
@pytest.mark.parametrize('max_int', [5, 20])
@pytest.mark.parametrize('keys', ['1st', '2nd', ['1st', '2nd']], ids=repr)
@pytest.mark.parametrize('bins', [None, [0, 5]], ids=repr)
@pytest.mark.parametrize('isort', [True, False])
@pytest.mark.parametrize('normalize, name', [(True, 'proportion'), (False, 'count')])
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
    ascending: bool,
    dropna: bool
) -> None: ...

@pytest.mark.parametrize('utc', [True, False])
def test_series_groupby_value_counts_with_grouper(utc: bool) -> None: ...

@pytest.mark.parametrize('columns', [['A', 'B'], ['A', 'B', 'C']])
def test_series_groupby_value_counts_empty(columns: List[str]) -> None: ...

@pytest.mark.parametrize('columns', [['A', 'B'], ['A', 'B', 'C']])
def test_series_groupby_value_counts_one_row(columns: List[str]) -> None: ...

def test_series_groupby_value_counts_on_categorical() -> None: ...

def test_series_groupby_value_counts_no_sort() -> None: ...

@pytest.fixture
def education_df() -> DataFrame: ...

def test_bad_subset(education_df: DataFrame) -> None: ...

def test_basic(education_df: DataFrame, request: Any) -> None: ...

def _frame_value_counts(
    df: DataFrame,
    keys: List[str],
    normalize: bool,
    sort: bool,
    ascending: bool
) -> Series: ...

@pytest.mark.parametrize('groupby', ['column', 'array', 'function'])
@pytest.mark.parametrize('normalize, name', [(True, 'proportion'), (False, 'count')])
@pytest.mark.parametrize('sort, ascending', [(False, None), (True, True), (True, False)])
@pytest.mark.parametrize('frame', [True, False])
def test_against_frame_and_seriesgroupby(
    education_df: DataFrame,
    groupby: str,
    normalize: bool,
    name: str,
    sort: bool,
    ascending: bool,
    as_index: bool,
    frame: bool,
    request: Any,
    using_infer_string: bool
) -> None: ...

@pytest.mark.parametrize('normalize', [True, False])
@pytest.mark.parametrize(
    'sort, ascending, expected_rows, expected_count, expected_group_size',
    [
        (False, None, [0, 1, 2, 3, 4], [1, 1, 1, 2, 1], [1, 3, 1, 3, 1]),
        (True, False, [3, 0, 1, 2, 4], [2, 1, 1, 1, 1], [3, 1, 3, 1, 1]),
        (True, True, [0, 1, 2, 4, 3], [1, 1, 1, 1, 2], [1, 3, 1, 1, 3])
    ]
)
def test_compound(
    education_df: DataFrame,
    normalize: bool,
    sort: bool,
    ascending: bool,
    expected_rows: List[int],
    expected_count: List[int],
    expected_group_size: List[int],
    any_string_dtype: str,
    using_infer_string: bool
) -> None: ...

@pytest.mark.parametrize(
    'sort, ascending, normalize, name, expected_data, expected_index',
    [
        (False, None, False, 'count', [1, 2, 1], [(1, 1, 1), (2, 4, 6), (2, 0, 0)]),
        (True, True, False, 'count', [1, 1, 2], [(1, 1, 1), (2, 6, 4), (2, 0, 0)]),
        (True, False, False, 'count', [2, 1, 1], [(1, 1, 1), (4, 2, 6), (0, 2, 0)]),
        (True, False, True, 'proportion', [0.5, 0.25, 0.25], [(1, 1, 1), (4, 2, 6), (0, 2, 0)])
    ]
)
def test_data_frame_value_counts(
    sort: bool,
    ascending: bool,
    normalize: bool,
    name: str,
    expected_data: List[Union[int, float]],
    expected_index: List[Tuple[int, int, int]]
) -> None: ...

@pytest.mark.parametrize(
    'group_dropna, count_dropna, expected_rows, expected_values',
    [
        (False, False, [0, 1, 3, 5, 6, 7, 8, 2, 4], [0.5, 0.5, 1.0, 0.25, 0.25, 0.25, 0.25, 1.0, 1.0]),
        (False, True, [0, 1, 3, 5, 2, 4], [0.5, 0.5, 1.0, 1.0, 1.0, 1.0]),
        (True, False, [0, 1, 5, 6, 7, 8], [0.5, 0.5, 0.25, 0.25, 0.25, 0.25]),
        (True, True, [0, 1, 5], [0.5, 0.5, 1.0])
    ]
)
def test_dropna_combinations(
    group_dropna: bool,
    count_dropna: bool,
    expected_rows: List[int],
    expected_values: List[float],
    request: Any
) -> None: ...

@pytest.mark.parametrize('dropna, expected_data, expected_index', [
    (True, [1, 1], MultiIndex.from_arrays([(1, 1), ('John', 'Beth'), ('Smith', 'Louise')], names=['key', 'first_name', 'middle_name'])),
    (False, [1, 1, 1, 1], MultiIndex(levels=[Index([1]), Index(['Anne', 'Beth', 'John']), Index(['Louise', 'Smith', np.nan])], codes=[[0, 0, 0, 0], [2, 0, 2, 1], [1, 2, 2, 0]], names=['key', 'first_name', 'middle_name']))
])
@pytest.mark.parametrize('normalize, name', [(False, 'count'), (True, 'proportion')])
def test_data_frame_value_counts_dropna(
    nulls_fixture: Any,
    dropna: bool,
    normalize: bool,
    name: str,
    expected_data: List[int],
    expected_index: MultiIndex
) -> None: ...

@pytest.mark.parametrize('observed', [False, True])
@pytest.mark.parametrize('normalize, name, expected_data', [
    (False, 'count', np.array([2, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0], dtype=np.int64)),
    (True, 'proportion', np.array([0.5, 0.25, 0.25, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]))
])
def test_categorical_single_grouper_with_only_observed_categories(
    education_df: DataFrame,
    as_index: bool,
    observed: bool,
    normalize: bool,
    name: str,
    expected_data: np.ndarray,
    request: Any
) -> None: ...

def assert_categorical_single_grouper(
    education_df: DataFrame,
    as_index: bool,
    observed: bool,
    expected_index: List[Tuple[str, str, str]],
    normalize: bool,
    name: str,
    expected_data: np.ndarray
) -> None: ...

@pytest.mark.parametrize('normalize, name, expected_data', [
    (False, 'count', np.array([2, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0], dtype=np.int64)),
    (True, 'proportion', np.array([0.5, 0.25, 0.25, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]))
])
def test_categorical_single_grouper_observed_true(
    education_df: DataFrame,
    as_index: bool,
    normalize: bool,
    name: str,
    expected_data: np.ndarray,
    request: Any
) -> None: ...

@pytest.mark.parametrize('normalize, name, expected_data', [
    (False, 'count', np.array([2, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)),
    (True, 'proportion', np.array([0.5, 0.25, 0.25, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
])
def test_categorical_single_grouper_observed_false(
    education_df: DataFrame,
    as_index: bool,
    normalize: bool,
    name: str,
    expected_data: np.ndarray,
    request: Any
) -> None: ...

@pytest.mark.parametrize('observed, expected_index', [
    (False, [('FR', 'high', 'female'), ('FR', 'high', 'male'), ('FR', 'low', 'male'), ('FR', 'low', 'female'), ('FR', 'medium', 'male'), ('FR', 'medium', 'female'), ('US', 'high', 'female'), ('US', 'high', 'male'), ('US', 'low', 'male'), ('US', 'low', 'female'), ('US', 'medium', 'male'), ('US', 'medium', 'female')]),
    (True, [('FR', 'high', 'female'), ('FR', 'low', 'male'), ('FR', 'medium', 'male'), ('US', 'high', 'female'), ('US', 'low', 'male')])
])
@pytest.mark.parametrize('normalize, name, expected_data', [
    (False, 'count', np.array([1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 0, 0], dtype=np.int64)),
    (True, 'proportion', np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
])
def test_categorical_multiple_groupers(
    education_df: DataFrame,
    as_index: bool,
    observed: bool,
    expected_index: List[Tuple[str, str, str]],
    normalize: bool,
    name: str,
    expected_data: np.ndarray
) -> None: ...

@pytest.mark.parametrize('observed', [False, True])
@pytest.mark.parametrize('normalize, name, expected_data', [
    (False, 'count', np.array([2, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0], dtype=np.int64)),
    (True, 'proportion', np.array([0.5, 0.25, 0.25, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]))
])
def test_categorical_non_groupers(
    education_df: DataFrame,
    as_index: bool,
    observed: bool,
    normalize: bool,
    name: str,
    expected_data: np.ndarray,
    request: Any
) -> None: ...

@pytest.mark.parametrize('normalize, expected_label, expected_values', [
    (False, 'count', [1, 1, 1]),
    (True, 'proportion', [0.5, 0.5, 1.0])
])
def test_mixed_groupings(
    normalize: bool,
    expected_label: str,
    expected_values: List[Union[int, float]]
) -> None: ...

@pytest.mark.parametrize('test, columns, expected_names', [
    ('repeat', list('abbde'), ['a', None, 'd', 'b', 'b', 'e']),
    ('level', list('abcd') + ['level_1'], ['a', None, 'd', 'b', 'c', 'level_1'])
])
def test_column_label_duplicates(
    test: str,
    columns: List[str],
    expected_names: List[Optional[str]],
    as_index: bool
) -> None: ...

@pytest.mark.parametrize('normalize, expected_label', [(False, 'count'), (True, 'proportion')])
def test_result_label_duplicates(normalize: bool, expected_label: str) -> None: ...

def test_ambiguous_grouping() -> None: ...

def test_subset_overlaps_gb_key_raises() -> None: ...

def test_subset_doesnt_exist_in_frame() -> None: ...

def test_subset() -> None: ...

def test_subset_duplicate_columns() -> None: ...

@pytest.mark.parametrize('utc', [True, False])
def test_value_counts_time_grouper(utc: bool, unit: str) -> None: ...

def test_value_counts_integer_columns() -> None: ...

@pytest.mark.parametrize('vc_sort', [True, False])
@pytest.mark.parametrize('normalize', [True, False])
def test_value_counts_sort(
    sort: bool,
    vc_sort: bool,
    normalize: bool
) -> None: ...

@pytest.mark.parametrize('vc_sort', [True, False])
@pytest.mark.parametrize('normalize', [True, False])
def test_value_counts_sort_categorical(
    sort: bool,
    vc_sort: bool,
    normalize: bool
) -> None: ...

@pytest.mark.parametrize('groupby_sort', [True, False])
def test_value_counts_all_na(
    sort: bool,
    dropna: bool,
    groupby_sort: bool
) -> None: ...