import numpy as np
import pytest
from pandas import DataFrame, Series, MultiIndex, Index, RangeIndex, Timestamp
from pandas import option_context, Series, DataFrame
from pandas.core.reshape.merge import merge
from pandas.core.reshape.concat import concat
from typing import List, Optional, Union, Callable, Dict, Any

@pytest.fixture
def left() -> DataFrame:
    ...

@pytest.fixture
def right(multiindex_dataframe_random_data: DataFrame) -> DataFrame:
    ...

@pytest.fixture
def left_multi() -> DataFrame:
    ...

@pytest.fixture
def right_multi() -> DataFrame:
    ...

@pytest.fixture
def on_cols_multi() -> List[str]:
    ...

class TestMergeMulti:
    def test_merge_on_multikey(self, left: DataFrame, right: DataFrame, join_type: str) -> None:
        ...

    @pytest.mark.parametrize('infer_string', [False, pytest.param(True, marks=td.skip_if_no('pyarrow'))])
    def test_left_join_multi_index(self, sort: bool, infer_string: bool) -> None:
        ...

    def test_merge_right_vs_left(self, left: DataFrame, right: DataFrame, sort: bool) -> None:
        ...

    def test_merge_multiple_cols_with_mixed_cols_index(self) -> None:
        ...

    def test_compress_group_combinations(self) -> DataFrame:
        ...

    def test_left_join_index_preserve_order(self) -> None:
        ...

    def test_left_join_index_multi_match_multiindex(self) -> None:
        ...

    def test_left_join_index_multi_match(self) -> None:
        ...

    def test_left_merge_na_buglet(self) -> None:
        ...

    def test_merge_na_keys(self) -> None:
        ...

    @pytest.mark.parametrize('klass', [None, np.asarray, Series, Index])
    def test_merge_datetime_index(self, klass: Union[None, Callable, Series, Index]) -> None:
        ...

    @pytest.mark.parametrize('merge_type', ['left', 'right'])
    def test_merge_datetime_multi_index_empty_df(self, merge_type: str) -> None:
        ...

    @pytest.fixture
    def household(self) -> DataFrame:
        ...

    @pytest.fixture
    def portfolio(self) -> DataFrame:
        ...

    @pytest.fixture
    def expected(self) -> DataFrame:
        ...

    def test_join_multi_levels(self, portfolio: DataFrame, household: DataFrame, expected: DataFrame) -> None:
        ...

    def test_join_multi_levels_merge_equivalence(self, portfolio: DataFrame, household: DataFrame, expected: DataFrame) -> None:
        ...

    def test_join_multi_levels_outer(self, portfolio: DataFrame, household: DataFrame, expected: DataFrame) -> None:
        ...

    def test_join_multi_levels_invalid(self, portfolio: DataFrame, household: DataFrame) -> None:
        ...

    def test_join_multi_levels2(self) -> None:
        ...

class TestJoinMultiMulti:
    def test_join_multi_multi(self, left_multi: DataFrame, right_multi: DataFrame, join_type: str, on_cols_multi: List[str]) -> None:
        ...

    def test_join_multi_empty_frames(self, left_multi: DataFrame, right_multi: DataFrame, join_type: str, on_cols_multi: List[str]) -> None:
        ...

    @pytest.mark.parametrize('box', [None, np.asarray, Series, Index])
    def test_merge_datetime_index(self, box: Union[None, Callable, Series, Index]) -> None:
        ...

    def test_single_common_level(self) -> None:
        ...

    def test_join_multi_wrong_order(self) -> None:
        ...