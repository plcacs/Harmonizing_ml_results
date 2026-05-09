import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import Categorical, DataFrame, Index, MultiIndex, Series, Timestamp, bdate_range, concat, merge, option_context
import pandas._testing as tm

def get_test_data(ngroups: int = 8, n: int = 50) -> np.ndarray:
    ...

class TestJoin:
    @pytest.fixture
    def df(self) -> pd.DataFrame:
        ...

    @pytest.fixture
    def df2(self) -> pd.DataFrame:
        ...

    @pytest.fixture
    def target_source(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        ...

    def test_left_outer_join(self, df: pd.DataFrame, df2: pd.DataFrame) -> None:
        ...

    def test_right_outer_join(self, df: pd.DataFrame, df2: pd.DataFrame) -> None:
        ...

    def test_full_outer_join(self, df: pd.DataFrame, df2: pd.DataFrame) -> None:
        ...

    def test_inner_join(self, df: pd.DataFrame, df2: pd.DataFrame) -> None:
        ...

    def test_handle_overlap(self, df: pd.DataFrame, df2: pd.DataFrame) -> None:
        ...

    def test_handle_overlap_arbitrary_key(self, df: pd.DataFrame, df2: pd.DataFrame) -> None:
        ...

    @pytest.mark.parametrize('infer_string', [False, pytest.param(True, marks=td.skip_if_no('pyarrow'))])
    def test_join_on(self, target_source: tuple[pd.DataFrame, pd.DataFrame], infer_string: bool) -> None:
        ...

    def test_join_on_fails_with_different_right_index(self) -> None:
        ...

    def test_join_on_fails_with_different_left_index(self) -> None:
        ...

    def test_join_on_fails_with_different_column_counts(self) -> None:
        ...

    @pytest.mark.parametrize('wrong_type', [int, str, None, np.ndarray])
    def test_join_on_fails_with_wrong_object_type(self, wrong_type: type, df: pd.DataFrame) -> None:
        ...

    def test_join_on_pass_vector(self, target_source: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ...

    def test_join_with_len0(self, target_source: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ...

    def test_join_on_inner(self) -> None:
        ...

    def test_join_on_singlekey_list(self) -> None:
        ...

    def test_join_on_series(self, target_source: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ...

    def test_join_on_series_buglet(self) -> None:
        ...

    def test_join_index_mixed(self, join_type: str) -> None:
        ...

    def test_join_index_mixed_overlap(self) -> None:
        ...

    def test_join_empty_bug(self) -> None:
        ...

    def test_join_unconsolidated(self) -> None:
        ...

    def test_join_multiindex(self) -> None:
        ...

    def test_join_inner_multiindex(self, lexsorted_two_level_string_multiindex: MultiIndex) -> None:
        ...

    def test_join_hierarchical_mixed_raises(self) -> None:
        ...

    def test_join_float64_float32(self) -> None:
        ...

    def test_join_many_non_unique_index(self) -> None:
        ...

    @pytest.mark.parametrize('infer_string', [False, pytest.param(True, marks=td.skip_if_no('pyarrow'))])
    def test_join_sort(self, infer_string: bool) -> None:
        ...

    def test_join_mixed_non_unique_index(self) -> None:
        ...

    def test_join_non_unique_period_index(self) -> None:
        ...

    def test_mixed_type_join_with_suffix(self, using_infer_string: bool) -> None:
        ...

    def test_join_many(self) -> None:
        ...

    def test_join_many_mixed(self) -> None:
        ...

    def test_join_dups(self) -> None:
        ...

    def test_join_multi_to_multi(self, join_type: str) -> None:
        ...

    def test_join_on_tz_aware_datetimeindex(self) -> None:
        ...

    def test_join_datetime_string(self) -> None:
        ...

    def test_join_with_categorical_index(self) -> None:
        ...

def _check_join(left: pd.DataFrame, right: pd.DataFrame, result: pd.DataFrame, join_col: list[str], how: str = 'left', lsuffix: str = '_x', rsuffix: str = '_y') -> None:
    ...

def _restrict_to_columns(group: pd.DataFrame, columns: list[str], suffix: str) -> pd.DataFrame:
    ...

def _assert_same_contents(join_chunk: pd.DataFrame, source: pd.DataFrame) -> None:
    ...

def _assert_all_na(join_chunk: pd.DataFrame, source_columns: list[str], join_col: list[str]) -> None:
    ...

def _join_by_hand(a: pd.DataFrame, b: pd.DataFrame, how: str = 'left') -> pd.DataFrame:
    ...

def test_join_inner_multiindex_deterministic_order() -> None:
    ...

@pytest.mark.parametrize(('input_col', 'output_cols'), [('b', ['a', 'b']), ('a', ['a_x', 'a_y'])])
def test_join_cross(input_col: str, output_cols: list[str]) -> None:
    ...

def test_join_multiindex_one_level(join_type: str) -> None:
    ...

@pytest.mark.parametrize('categories, values', [(['Y', 'X'], ['Y', 'X', 'X']), ([2, 1], [2, 1, 1]), ([2.5, 1.5], [2.5, 1.5, 1.5]), ([Timestamp('2020-12-31'), Timestamp('2019-12-31')], [Timestamp('2020-12-31'), Timestamp('2019-12-31'), Timestamp('2019-12-31')])])
def test_join_multiindex_not_alphabetical_categorical(categories: list, values: list) -> None:
    ...

@pytest.mark.parametrize('left_empty, how, exp', [(False, 'left', 'left'), (False, 'right', 'empty'), (False, 'inner', 'empty'), (False, 'outer', 'left'), (False, 'cross', 'empty'), (True, 'left', 'empty'), (True, 'right', 'right'), (True, 'inner', 'empty'), (True, 'outer', 'right'), (True, 'cross', 'empty')])
def test_join_empty(left_empty: bool, how: str, exp: str) -> None:
    ...

def test_join_empty_uncomparable_columns() -> None:
    ...

@pytest.mark.parametrize('how, values', [('inner', [0, 1, 2]), ('outer', [0, 1, 2]), ('left', [0, 1, 2]), ('right', [0, 2, 1])])
def test_join_multiindex_categorical_output_index_dtype(how: str, values: list[int]) -> None:
    ...

def test_join_multiindex_with_none_as_label() -> None:
    ...