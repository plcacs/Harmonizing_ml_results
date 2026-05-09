from copy import deepcopy
import numpy as np
import pandas as pd
from pandas import DataFrame, Index, MultiIndex, Series
import pytest

class TestIndexConcat:
    def test_concat_ignore_index(self, sort: bool) -> None:
        ...
    
    @pytest.mark.parametrize('name_in1,name_in2,name_in3,name_out', [('idx', 'idx', 'idx', 'idx'), ('idx', 'idx', None, None), ('idx', None, None, None), ('idx1', 'idx2', None, None), ('idx1', 'idx1', 'idx2', None), ('idx1', 'idx2', 'idx3', None), (None, None, None, None)])
    def test_concat_same_index_names(self, name_in1: Optional[str], name_in2: Optional[str], name_in3: Optional[str], name_out: Optional[str]) -> None:
        ...
    
    def test_concat_rename_index(self) -> None:
        ...
    
    def test_concat_copy_index_series(self, axis: Union[int, str]) -> None:
        ...
    
    def test_concat_copy_index_frame(self, axis: Union[int, str]) -> None:
        ...
    
    def test_default_index(self) -> None:
        ...
    
    def test_dups_index(self) -> None:
        ...

class TestMultiIndexConcat:
    def test_concat_multiindex_with_keys(self, multiindex_dataframe_random_data: DataFrame) -> None:
        ...
    
    def test_concat_multiindex_with_none_in_index_names(self) -> None:
        ...
    
    def test_concat_multiindex_rangeindex(self) -> None:
        ...
    
    def test_concat_multiindex_dfs_with_deepcopy(self) -> None:
        ...
    
    @pytest.mark.parametrize('mi1_list,mi2_list', [[[['a'], range(2)], [['a'], range(2)]], [[['b'], np.arange(2.0, 4.0)], [['b'], np.arange(2.0, 4.0)]], [[['c'], ['A', 'B']], [['c'], ['A', 'B']]], [[['d'], pd.date_range(start='2017', end='2018', periods=2)], [['d'], pd.date_range(start='2017', end='2018', periods=2)]]])
    def test_concat_with_various_multiindex_dtypes(self, mi1_list: List[List[Union[str, np.ndarray]]], mi2_list: List[List[Union[str, np.ndarray]]]) -> None:
        ...
    
    def test_concat_multiindex_(self) -> None:
        ...
    
    def test_concat_with_key_not_unique(self, performance_warning: Any) -> None:
        ...
    
    def test_concat_with_duplicated_levels(self) -> None:
        ...
    
    @pytest.mark.parametrize('levels', [[['x', 'y']], [['x', 'y', 'y']]])
    def test_concat_with_levels_with_none_keys(self, levels: List[List[str]]) -> None:
        ...
    
    def test_concat_range_index_result(self) -> None:
        ...
    
    def test_concat_index_keep_dtype(self) -> None:
        ...
    
    def test_concat_index_keep_dtype_ea_numeric(self, any_numeric_ea_dtype: str) -> None:
        ...
    
    @pytest.mark.parametrize('dtype', ['Int8', 'Int16', 'Int32'])
    def test_concat_index_find_common(self, dtype: str) -> None:
        ...
    
    def test_concat_axis_1_sort_false_rangeindex(self, using_infer_string: bool) -> None:
        ...