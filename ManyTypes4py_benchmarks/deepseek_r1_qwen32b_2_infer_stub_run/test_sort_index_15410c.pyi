import numpy as np
import pytest
import pandas as pd
from pandas import (
    CategoricalDtype,
    CategoricalIndex,
    DataFrame,
    IntervalIndex,
    MultiIndex,
    RangeIndex,
    Series,
    Timestamp,
)
import pandas._testing as tm

class TestDataFrameSortIndex:
    def test_sort_index_and_reconstruction_doc_example(self) -> None:
        ...
    
    def test_sort_index_non_existent_label_multiindex(self) -> None:
        ...
    
    def test_sort_index_reorder_on_ops(self) -> None:
        ...
    
    def test_sort_index_nan_multiindex(self) -> None:
        ...
    
    def test_sort_index_nan(self) -> None:
        ...
    
    def test_sort_index_multi_index(self) -> None:
        ...
    
    def test_sort_index_inplace(self) -> None:
        ...
    
    def test_sort_index_different_sortorder(self) -> None:
        ...
    
    def test_sort_index_level(self) -> None:
        ...
    
    def test_sort_index_categorical_index(self) -> None:
        ...
    
    def test_sort_index(self) -> None:
        ...
    
    @pytest.mark.parametrize('level', ['A', 0])
    def test_sort_index_multiindex(self, level: str | int) -> None:
        ...
    
    def test_sort_index_intervalindex(self) -> None:
        ...
    
    @pytest.mark.parametrize('inplace', [True, False])
    @pytest.mark.parametrize('original_dict, sorted_dict, ascending, ignore_index, output_index', [
        ({'A': [1, 2, 3]}, {'A': [2, 3, 1]}, False, True, [0, 1, 2]),
        ({'A': [1, 2, 3]}, {'A': [1, 3, 2]}, True, True, [0, 1, 2]),
        ({'A': [1, 2, 3]}, {'A': [2, 3, 1]}, False, False, [5, 3, 2]),
        ({'A': [1, 2, 3]}, {'A': [1, 3, 2]}, True, False, [2, 3, 5])
    ])
    def test_sort_index_ignore_index(self, inplace: bool, original_dict: dict, sorted_dict: dict, ascending: bool, ignore_index: bool, output_index: list) -> None:
        ...
    
    @pytest.mark.parametrize('inplace', [True, False])
    @pytest.mark.parametrize('ignore_index', [True, False])
    def test_respect_ignore_index(self, inplace: bool, ignore_index: bool) -> None:
        ...
    
    @pytest.mark.parametrize('inplace', [True, False])
    @pytest.mark.parametrize('original_dict, sorted_dict, ascending, ignore_index, output_index', [
        ({'M1': [1, 2], 'M2': [3, 4]}, {'M1': [1, 2], 'M2': [3, 4]}, True, True, [0, 1]),
        ({'M1': [1, 2], 'M2': [3, 4]}, {'M1': [2, 1], 'M2': [4, 3]}, False, True, [0, 1]),
        ({'M1': [1, 2], 'M2': [3, 4]}, {'M1': [1, 2], 'M2': [3, 4]}, True, False, MultiIndex.from_tuples([(2, 1), (3, 4)], names=list('AB'))),
        ({'M1': [1, 2], 'M2': [3, 4]}, {'M1': [2, 1], 'M2': [4, 3]}, False, False, MultiIndex.from_tuples([(3, 4), (2, 1)], names=list('AB')))
    ])
    def test_sort_index_ignore_index_multi_index(self, inplace: bool, original_dict: dict, sorted_dict: dict, ascending: bool, ignore_index: bool, output_index: list | MultiIndex) -> None:
        ...
    
    def test_sort_index_categorical_multiindex(self) -> None:
        ...
    
    def test_sort_index_and_reconstruction(self) -> None:
        ...
    
    def test_sort_index_level2(self, multiindex_dataframe_random_data: DataFrame) -> None:
        ...
    
    def test_sort_index_level_large_cardinality(self) -> None:
        ...
    
    def test_sort_index_level_by_name(self, multiindex_dataframe_random_data: DataFrame) -> None:
        ...
    
    def test_sort_index_level_mixed(self, multiindex_dataframe_random_data: DataFrame) -> None:
        ...
    
    def test_sort_index_preserve_levels(self, multiindex_dataframe_random_data: DataFrame) -> None:
        ...
    
    @pytest.mark.parametrize('gen,extra', [
        ([1.0, 3.0, 2.0, 5.0], 4.0),
        ([1, 3, 2, 5], 4),
        ([Timestamp('20130101'), Timestamp('20130103'), Timestamp('20130102'), Timestamp('20130105')], Timestamp('20130104')),
        (['1one', '3one', '2one', '5one'], '4one')
    ])
    def test_sort_index_multilevel_repr_8017(self, gen: list, extra: float | int | Timestamp | str) -> None:
        ...
    
    @pytest.mark.parametrize('categories', [
        pytest.param(['a', 'b', 'c'], id='str'),
        pytest.param([pd.Interval(0, 1), pd.Interval(1, 2), pd.Interval(2, 3)], id='pd.Interval')
    ])
    def test_sort_index_with_categories(self, categories: list[str] | list[pd.Interval]) -> None:
        ...
    
    @pytest.mark.parametrize('ascending', [None, [True, None], [False, 'True']])
    def test_sort_index_ascending_bad_value_raises(self, ascending: bool | list[bool | str]) -> None:
        ...
    
    @pytest.mark.parametrize('ascending', [(True, False), [True, False]])
    def test_sort_index_ascending_tuple(self, ascending: tuple[bool, bool] | list[bool]) -> None:
        ...

class TestDataFrameSortIndexKey:
    def test_sort_multi_index_key(self) -> None:
        ...
    
    def test_sort_index_key(self) -> None:
        ...
    
    def test_sort_index_key_int(self) -> None:
        ...
    
    def test_sort_multi_index_key_str(self) -> None:
        ...
    
    def test_changes_length_raises(self) -> None:
        ...
    
    def test_sort_index_multiindex_sparse_column(self) -> None:
        ...
    
    def test_sort_index_na_position(self) -> None:
        ...
    
    def test_sort_index_multiindex_sort_remaining(self, ascending: bool) -> None:
        ...

def test_sort_index_with_sliced_multiindex() -> None:
    ...

def test_axis_columns_ignore_index() -> None:
    ...

def test_axis_columns_ignore_index_ascending_false() -> None:
    ...

def test_sort_index_stable_sort() -> None:
    ...