import pytest
import numpy as np
from pandas import DataFrame, Index
from pandas._testing import assert_frame_equal as tm_assert_frame_equal

@pytest.fixture
def df() -> DataFrame:
    ...

@pytest.fixture
def df1() -> DataFrame:
    ...

@pytest.fixture
def var_name() -> str:
    ...

@pytest.fixture
def value_name() -> str:
    ...

class TestMelt:
    def test_top_level_method(self, df: DataFrame) -> None:
        ...

    def test_method_signatures(self, df: DataFrame, df1: DataFrame, var_name: str, value_name: str) -> None:
        ...

    def test_default_col_names(self, df: DataFrame) -> None:
        ...

    def test_value_vars(self, df: DataFrame) -> None:
        ...

    @pytest.mark.parametrize('type_', (tuple, list, np.ndarray))
    def test_value_vars_types(self, type_: type, df: DataFrame) -> None:
        ...

    def test_vars_work_with_multiindex(self, df1: DataFrame) -> None:
        ...

    @pytest.mark.parametrize('id_vars, value_vars, col_level, expected', [
        (['A'], ['B'], 0, ...),
        (['a'], ['b'], 1, ...)
    ])
    def test_single_vars_work_with_multiindex(self, id_vars: list[str], value_vars: list[str], col_level: int, expected: DataFrame, df1: DataFrame) -> None:
        ...

    @pytest.mark.parametrize('id_vars, value_vars', [
        [('A', 'a'), [('B', 'b')]],
        [[('A', 'a')], ('B', 'b')],
        [('A', 'a'), ('B', 'b')]
    ])
    def test_tuple_vars_fail_with_multiindex(self, id_vars: list[tuple[str, str]], value_vars: list[tuple[str, str]], df1: DataFrame) -> None:
        ...

    def test_custom_var_name(self, df: DataFrame, var_name: str) -> None:
        ...

    def test_custom_value_name(self, df: DataFrame, value_name: str) -> None:
        ...

    def test_custom_var_and_value_name(self, df: DataFrame, value_name: str, var_name: str) -> None:
        ...

    @pytest.mark.parametrize('col_level', [0, 'CAP'])
    def test_col_level(self, col_level: int | str, df1: DataFrame) -> None:
        ...

    def test_multiindex(self, df1: DataFrame) -> None:
        ...

    @pytest.mark.parametrize('col', [
        pd.DatetimeIndex,
        pd.Categorical,
        list[int]
    ])
    def test_pandas_dtypes(self, col: type) -> None:
        ...

    def test_preserve_category(self) -> None:
        ...

    def test_melt_missing_columns_raises(self) -> None:
        ...

    def test_melt_mixed_int_str_id_vars(self) -> None:
        ...

    def test_melt_mixed_int_str_value_vars(self) -> None:
        ...

    def test_ignore_index(self) -> None:
        ...

    def test_ignore_multiindex(self) -> None:
        ...

    def test_ignore_index_name_and_type(self) -> None:
        ...

    def test_melt_with_duplicate_columns(self) -> None:
        ...

    @pytest.mark.parametrize('dtype', ['Int8', 'Int64'])
    def test_melt_ea_dtype(self, dtype: str) -> None:
        ...

    def test_melt_ea_columns(self) -> None:
        ...

    def test_melt_preserves_datetime(self) -> None:
        ...

    def test_melt_allows_non_scalar_id_vars(self) -> None:
        ...

    def test_melt_allows_non_string_var_name(self) -> None:
        ...

    def test_melt_non_scalar_var_name_raises(self) -> None:
        ...

    def test_melt_multiindex_columns_var_name(self) -> None:
        ...

    def test_melt_multiindex_columns_var_name_too_many(self) -> None:
        ...

class TestLreshape:
    def test_pairs(self) -> None:
        ...

    def test_stubs(self) -> None:
        ...

    def test_separating_character(self) -> None:
        ...

    def test_escapable_characters(self) -> None:
        ...

    def test_unbalanced(self) -> None:
        ...

    def test_character_overlap(self) -> None:
        ...

    def test_invalid_separator(self) -> None:
        ...

    def test_num_string_disambiguation(self) -> None:
        ...

    def test_invalid_suffixtype(self) -> None:
        ...

    def test_multiple_id_columns(self) -> None:
        ...

    def test_non_unique_idvars(self) -> None:
        ...

    def test_cast_j_int(self) -> None:
        ...

    def test_identical_stubnames(self) -> None:
        ...

    def test_nonnumeric_suffix(self) -> None:
        ...

    def test_mixed_type_suffix(self) -> None:
        ...

    def test_float_suffix(self) -> None:
        ...

    def test_col_substring_of_stubname(self) -> None:
        ...

    def test_raise_of_column_name_value(self) -> None:
        ...

    def test_missing_stubname(self, any_string_dtype: str) -> None:
        ...

def test_wide_to_long_string_columns(string_storage: str) -> None:
    ...