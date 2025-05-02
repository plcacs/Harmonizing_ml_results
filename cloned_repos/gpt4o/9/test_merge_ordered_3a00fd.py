import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame, merge_ordered
import pandas._testing as tm
from typing import Dict, List, Union, Any

@pytest.fixture
def left() -> DataFrame:
    return DataFrame({'key': ['a', 'c', 'e'], 'lvalue': [1, 2.0, 3]})

@pytest.fixture
def right() -> DataFrame:
    return DataFrame({'key': ['b', 'c', 'd', 'f'], 'rvalue': [1, 2, 3.0, 4]})

class TestMergeOrdered:

    def test_basic(self, left: DataFrame, right: DataFrame) -> None:
        result: DataFrame = merge_ordered(left, right, on='key')
        expected: DataFrame = DataFrame({'key': ['a', 'b', 'c', 'd', 'e', 'f'], 'lvalue': [1, np.nan, 2, np.nan, 3, np.nan], 'rvalue': [np.nan, 1, 2, 3, np.nan, 4]})
        tm.assert_frame_equal(result, expected)

    def test_ffill(self, left: DataFrame, right: DataFrame) -> None:
        result: DataFrame = merge_ordered(left, right, on='key', fill_method='ffill')
        expected: DataFrame = DataFrame({'key': ['a', 'b', 'c', 'd', 'e', 'f'], 'lvalue': [1.0, 1, 2, 2, 3, 3.0], 'rvalue': [np.nan, 1, 2, 3, 3, 4]})
        tm.assert_frame_equal(result, expected)

    def test_multigroup(self, left: DataFrame, right: DataFrame) -> None:
        left = pd.concat([left, left], ignore_index=True)
        left['group'] = ['a'] * 3 + ['b'] * 3
        result: DataFrame = merge_ordered(left, right, on='key', left_by='group', fill_method='ffill')
        expected: DataFrame = DataFrame({'key': ['a', 'b', 'c', 'd', 'e', 'f'] * 2, 'lvalue': [1.0, 1, 2, 2, 3, 3.0] * 2, 'rvalue': [np.nan, 1, 2, 3, 3, 4] * 2})
        expected['group'] = ['a'] * 6 + ['b'] * 6
        tm.assert_frame_equal(result, expected.loc[:, result.columns])
        result2: DataFrame = merge_ordered(right, left, on='key', right_by='group', fill_method='ffill')
        tm.assert_frame_equal(result, result2.loc[:, result.columns])
        result = merge_ordered(left, right, on='key', left_by='group')
        assert result['group'].notna().all()

    @pytest.mark.filterwarnings('ignore:Passing a BlockManager|Passing a SingleBlockManager:DeprecationWarning')
    def test_merge_type(self, left: DataFrame, right: DataFrame) -> None:

        class NotADataFrame(DataFrame):

            @property
            def _constructor(self) -> Any:
                return NotADataFrame

        nad: NotADataFrame = NotADataFrame(left)
        result: DataFrame = nad.merge(right, on='key')
        assert isinstance(result, NotADataFrame)

    @pytest.mark.parametrize('df_seq, pattern', [((), '[Nn]o objects'), ([], '[Nn]o objects'), ({}, '[Nn]o objects'), ([None], 'objects.*None'), ([None, None], 'objects.*None')])
    def test_empty_sequence_concat(self, df_seq: Union[tuple, list, dict], pattern: str) -> None:
        with pytest.raises(ValueError, match=pattern):
            pd.concat(df_seq)

    @pytest.mark.parametrize('arg', [[DataFrame()], [None, DataFrame()], [DataFrame(), None]])
    def test_empty_sequence_concat_ok(self, arg: List[Union[DataFrame, None]]) -> None:
        pd.concat(arg)

    def test_doc_example(self) -> None:
        left: DataFrame = DataFrame({'group': list('aaabbb'), 'key': ['a', 'c', 'e', 'a', 'c', 'e'], 'lvalue': [1, 2, 3] * 2})
        right: DataFrame = DataFrame({'key': ['b', 'c', 'd'], 'rvalue': [1, 2, 3]})
        result: DataFrame = merge_ordered(left, right, fill_method='ffill', left_by='group')
        expected: DataFrame = DataFrame({'group': list('aaaaabbbbb'), 'key': ['a', 'b', 'c', 'd', 'e'] * 2, 'lvalue': [1, 1, 2, 2, 3] * 2, 'rvalue': [np.nan, 1, 2, 3, 3] * 2})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('left, right, on, left_by, right_by, expected', [({'G': ['g', 'g'], 'H': ['h', 'h'], 'T': [1, 3]}, {'T': [2], 'E': [1]}, ['T'], ['G', 'H'], None, {'G': ['g'] * 3, 'H': ['h'] * 3, 'T': [1, 2, 3], 'E': [np.nan, 1.0, np.nan]}), ({'G': ['g', 'g'], 'H': ['h', 'h'], 'T': [1, 3]}, {'T': [2], 'E': [1]}, 'T', ['G', 'H'], None, {'G': ['g'] * 3, 'H': ['h'] * 3, 'T': [1, 2, 3], 'E': [np.nan, 1.0, np.nan]}), ({'T': [2], 'E': [1]}, {'G': ['g', 'g'], 'H': ['h', 'h'], 'T': [1, 3]}, ['T'], None, ['G', 'H'], {'T': [1, 2, 3], 'E': [np.nan, 1.0, np.nan], 'G': ['g'] * 3, 'H': ['h'] * 3})])
    def test_list_type_by(self, left: Dict[str, List[Union[str, int]]], right: Dict[str, List[Union[str, int]]], on: Union[str, List[str]], left_by: Union[None, List[str]], right_by: Union[None, List[str]], expected: Dict[str, List[Union[str, float]]]) -> None:
        left_df: DataFrame = DataFrame(left)
        right_df: DataFrame = DataFrame(right)
        result: DataFrame = merge_ordered(left=left_df, right=right_df, on=on, left_by=left_by, right_by=right_by)
        expected_df: DataFrame = DataFrame(expected)
        tm.assert_frame_equal(result, expected_df)

    def test_left_by_length_equals_to_right_shape0(self) -> None:
        left: DataFrame = DataFrame([['g', 'h', 1], ['g', 'h', 3]], columns=list('GHE'))
        right: DataFrame = DataFrame([[2, 1]], columns=list('ET'))
        result: DataFrame = merge_ordered(left, right, on='E', left_by=['G', 'H'])
        expected: DataFrame = DataFrame({'G': ['g'] * 3, 'H': ['h'] * 3, 'E': [1, 2, 3], 'T': [np.nan, 1.0, np.nan]})
        tm.assert_frame_equal(result, expected)

    def test_elements_not_in_by_but_in_df(self) -> None:
        left: DataFrame = DataFrame([['g', 'h', 1], ['g', 'h', 3]], columns=list('GHE'))
        right: DataFrame = DataFrame([[2, 1]], columns=list('ET'))
        msg: str = "\\{'h'\\} not found in left columns"
        with pytest.raises(KeyError, match=msg):
            merge_ordered(left, right, on='E', left_by=['G', 'h'])

    @pytest.mark.parametrize('invalid_method', ['linear', 'carrot'])
    def test_ffill_validate_fill_method(self, left: DataFrame, right: DataFrame, invalid_method: str) -> None:
        with pytest.raises(ValueError, match=re.escape("fill_method must be 'ffill' or None")):
            merge_ordered(left, right, on='key', fill_method=invalid_method)

    def test_ffill_left_merge(self) -> None:
        df1: DataFrame = DataFrame({'key': ['a', 'c', 'e', 'a', 'c', 'e'], 'lvalue': [1, 2, 3, 1, 2, 3], 'group': ['a', 'a', 'a', 'b', 'b', 'b']})
        df2: DataFrame = DataFrame({'key': ['b', 'c', 'd'], 'rvalue': [1, 2, 3]})
        result: DataFrame = merge_ordered(df1, df2, fill_method='ffill', left_by='group', how='left')
        expected: DataFrame = DataFrame({'key': ['a', 'c', 'e', 'a', 'c', 'e'], 'lvalue': [1, 2, 3, 1, 2, 3], 'group': ['a', 'a', 'a', 'b', 'b', 'b'], 'rvalue': [np.nan, 2.0, 2.0, np.nan, 2.0, 2.0]})
        tm.assert_frame_equal(result, expected)
