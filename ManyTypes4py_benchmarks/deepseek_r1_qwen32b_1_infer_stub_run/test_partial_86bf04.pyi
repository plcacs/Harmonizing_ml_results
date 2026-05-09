"""
Stub file for test_partial_86bf04.py
"""

from typing import Any, List, Union, Optional, Tuple, Dict
import numpy as np
import pandas as pd
from pandas import DataFrame, Series, Index, Period, Timestamp, Timedelta
import pytest

class TestEmptyFrameSetitemExpansion:
    def test_empty_frame_setitem_index_name_retained(self) -> None:
        ...
    
    def test_empty_frame_setitem_index_name_inherited(self) -> None:
        ...
    
    def test_loc_setitem_zerolen_series_columns_align(self) -> None:
        ...
    
    def test_loc_setitem_zerolen_list_length_must_match_columns(self) -> None:
        ...
    
    def test_partial_set_empty_frame(self) -> None:
        ...
    
    def test_partial_set_empty_frame2(self) -> None:
        ...
    
    def test_partial_set_empty_frame3(self) -> None:
        ...
    
    def test_partial_set_empty_frame4(self) -> None:
        ...
    
    def test_partial_set_empty_frame5(self) -> None:
        ...
    
    def test_partial_set_empty_frame_no_index(self) -> None:
        ...
    
    def test_partial_set_empty_frame_row(self) -> None:
        ...
    
    def test_partial_set_empty_frame_set_series(self) -> None:
        ...
    
    def test_partial_set_empty_frame_empty_copy_assignment(self) -> None:
        ...
    
    def test_partial_set_empty_frame_empty_consistencies(self, using_infer_string: bool) -> None:
        ...

class TestPartialSetting:
    def test_partial_setting(self) -> None:
        ...
    
    @pytest.mark.filterwarnings('ignore:Setting a value on a view:FutureWarning')
    def test_partial_setting_frame(self) -> None:
        ...
    
    def test_partial_setting2(self) -> None:
        ...
    
    def test_partial_setting_mixed_dtype(self) -> None:
        ...
    
    def test_series_partial_set(self) -> None:
        ...
    
    def test_series_partial_set_with_name(self) -> None:
        ...
    
    @pytest.mark.parametrize('key', [100, 100.0])
    def test_setitem_with_expansion_numeric_into_datetimeindex(self, key: Union[int, float]) -> None:
        ...
    
    def test_partial_set_invalid(self) -> None:
        ...
    
    @pytest.mark.parametrize('idx,labels,expected_idx', [
        (pd.period_range(start='2000', periods=20, freq='D'), ['2000-01-04', '2000-01-08', '2000-01-12'], [Period('2000-01-04', freq='D'), Period('2000-01-08', freq='D'), Period('2000-01-12', freq='D')]),
        (pd.date_range(start='2000', periods=20, freq='D', unit='s'), ['2000-01-04', '2000-01-08', '2000-01-12'], [Timestamp('2000-01-04'), Timestamp('2000-01-08'), Timestamp('2000-01-12')]),
        (pd.timedelta_range(start='1 day', periods=20), ['4D', '8D', '12D'], [Timedelta('4 day'), Timedelta('8 day'), Timedelta('12 day')])
    ])
    def test_loc_with_list_of_strings_representing_datetimes(self, idx: Index, labels: List[str], expected_idx: List[Union[Period, Timestamp, Timedelta]], frame_or_series: Union[DataFrame, Series]) -> None:
        ...
    
    @pytest.mark.parametrize('idx,labels', [
        (pd.period_range(start='2000', periods=20, freq='D'), ['2000-01-04', '2000-01-30']),
        (pd.date_range(start='2000', periods=20, freq='D'), ['2000-01-04', '2000-01-30']),
        (pd.timedelta_range(start='1 day', periods=20), ['3 day', '30 day'])
    ])
    def test_loc_with_list_of_strings_representing_datetimes_missing_value(self, idx: Index, labels: List[str]) -> None:
        ...
    
    @pytest.mark.parametrize('idx,labels,msg', [
        (pd.period_range(start='2000', periods=20, freq='D'), pd.Index(['4D', '8D'], dtype=object), "None of \\[Index\\(\\['4D', '8D'\\], dtype='object'\\)\\] are in the \\[index\\]"),
        (pd.date_range(start='2000', periods=20, freq='D'), pd.Index(['4D', '8D'], dtype=object), "None of \\[Index\\(\\['4D', '8D'\\], dtype='object'\\)\\] are in the \\[index\\]"),
        (pd.timedelta_range(start='1 day', periods=20), pd.Index(['2000-01-04', '2000-01-08'], dtype=object), "None of \\[Index\\(\\['2000-01-04', '2000-01-08'\\], dtype='object'\\)\\] are in the \\[index\\]")
    ])
    def test_loc_with_list_of_strings_representing_datetimes_not_matched_type(self, idx: Index, labels: Index, msg: str) -> None:
        ...

class TestStringSlicing:
    def test_slice_irregular_datetime_index_with_nan(self) -> None:
        ...