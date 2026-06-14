from typing import List, Optional, Sequence, Tuple, Union

import pytest
from pandas import DataFrame, MultiIndex, Series


def tests_value_counts_index_names_category_column() -> None: ...
def seed_df(seed_nans: bool, n: int, m: int) -> DataFrame: ...
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
) -> None: ...
def test_series_groupby_value_counts_with_grouper(utc: bool) -> None: ...
def test_series_groupby_value_counts_empty(columns: List[str]) -> None: ...
def test_series_groupby_value_counts_one_row(columns: List[str]) -> None: ...
def test_series_groupby_value_counts_on_categorical() -> None: ...
def test_series_groupby_value_counts_no_sort() -> None: ...
@pytest.fixture
def education_df() -> DataFrame: ...
def test_bad_subset(education_df: DataFrame) -> None: ...
def test_basic(education_df: DataFrame, request: pytest.FixtureRequest) -> None: ...
def _frame_value_counts(
    df: DataFrame, keys: List[str], normalize: bool, sort: bool, ascending: bool
) -> Series: ...
def test_against_frame_and_seriesgroupby(
    education_df: DataFrame,
    groupby: str,
    normalize: bool,
    name: str,
    sort: bool,
    ascending: Optional[bool],
    as_index: bool,
    frame: bool,
    request: pytest.FixtureRequest,
    using_infer_string: bool,
) -> None: ...
def test_compound(
    education_df: DataFrame,
    normalize: bool,
    sort: bool,
    ascending: Optional[bool],
    expected_rows: List[int],
    expected_count: List[int],
    expected_group_size: List[int],
    any_string_dtype: object,
    using_infer_string: bool,
) -> None: ...
def test_data_frame_value_counts(
    sort: bool,
    ascending: bool,
    normalize: bool,
    name: str,
    expected_data: Sequence[Union[int, float]],
    expected_index: Sequence[Sequence[int]],
) -> None: ...
def test_dropna_combinations(
    group_dropna: bool,
    count_dropna: bool,
    expected_rows: List[int],
    expected_values: Sequence[float],
    request: pytest.FixtureRequest,
) -> None: ...
def test_data_frame_value_counts_dropna(
    nulls_fixture: object,
    dropna: bool,
    normalize: bool,
    name: str,
    expected_data: Sequence[int],
    expected_index: MultiIndex,
) -> None: ...
def test_categorical_single_grouper_with_only_observed_categories(
    education_df: DataFrame,
    as_index: bool,
    observed: bool,
    normalize: bool,
    name: str,
    expected_data: Sequence[Union[int, float]],
    request: pytest.FixtureRequest,
) -> None: ...
def assert_categorical_single_grouper(
    education_df: DataFrame,
    as_index: bool,
    observed: bool,
    expected_index: Sequence[Tuple[str, str, str]],
    normalize: bool,
    name: str,
    expected_data: Sequence[Union[int, float]],
) -> None: ...
def test_categorical_single_grouper_observed_true(
    education_df: DataFrame,
    as_index: bool,
    normalize: bool,
    name: str,
    expected_data: Sequence[Union[int, float]],
    request: pytest.FixtureRequest,
) -> None: ...
def test_categorical_single_grouper_observed_false(
    education_df: DataFrame,
    as_index: bool,
    normalize: bool,
    name: str,
    expected_data: Sequence[Union[int, float]],
    request: pytest.FixtureRequest,
) -> None: ...
def test_categorical_multiple_groupers(
    education_df: DataFrame,
    as_index: bool,
    observed: bool,
    expected_index: Sequence[Tuple[str, str, str]],
    normalize: bool,
    name: str,
    expected_data: Sequence[Union[int, float]],
) -> None: ...
def test_categorical_non_groupers(
    education_df: DataFrame,
    as_index: bool,
    observed: bool,
    normalize: bool,
    name: str,
    expected_data: Sequence[Union[int, float]],
    request: pytest.FixtureRequest,
) -> None: ...
def test_mixed_groupings(
    normalize: bool, expected_label: str, expected_values: Sequence[Union[int, float]]
) -> None: ...
def test_column_label_duplicates(
    test: str, columns: List[str], expected_names: List[Optional[str]], as_index: bool
) -> None: ...
def test_result_label_duplicates(normalize: bool, expected_label: str) -> None: ...
def test_ambiguous_grouping() -> None: ...
def test_subset_overlaps_gb_key_raises() -> None: ...
def test_subset_doesnt_exist_in_frame() -> None: ...
def test_subset() -> None: ...
def test_subset_duplicate_columns() -> None: ...
def test_value_counts_time_grouper(utc: bool, unit: str) -> None: ...
def test_value_counts_integer_columns() -> None: ...
def test_value_counts_sort(sort: bool, vc_sort: bool, normalize: bool) -> None: ...
def test_value_counts_sort_categorical(sort: bool, vc_sort: bool, normalize: bool) -> None: ...
def test_value_counts_all_na(sort: bool, dropna: bool, groupby_sort: bool) -> None: ...