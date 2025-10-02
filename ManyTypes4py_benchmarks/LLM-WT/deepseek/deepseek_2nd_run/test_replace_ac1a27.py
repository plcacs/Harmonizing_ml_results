from __future__ import annotations
from datetime import datetime
import re
from typing import Any, Dict, List, Optional, Union, Tuple, Sequence, TypeVar
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame, Index, Series, Timestamp, date_range
import pandas._testing as tm

T = TypeVar('T')

@pytest.fixture
def mix_ab() -> Dict[str, List[Union[int, str]]]:
    return {'a': list(range(4)), 'b': list('ab..')}

@pytest.fixture
def mix_abc() -> Dict[str, List[Union[int, str, float]]]:
    return {'a': list(range(4)), 'b': list('ab..'), 'c': ['a', 'b', np.nan, 'd']}

class TestDataFrameReplace:

    def test_replace_inplace(self, datetime_frame: DataFrame, float_string_frame: DataFrame) -> None:
        datetime_frame.loc[datetime_frame.index[:5], 'A'] = np.nan
        datetime_frame.loc[datetime_frame.index[-5:], 'A'] = np.nan
        tsframe = datetime_frame.copy()
        return_value = tsframe.replace(np.nan, 0, inplace=True)
        assert return_value is None
        tm.assert_frame_equal(tsframe, datetime_frame.fillna(0))
        mf = float_string_frame
        mf.iloc[5:20, mf.columns.get_loc('foo')] = np.nan
        mf.iloc[-10:, mf.columns.get_loc('A')] = np.nan
        result = float_string_frame.replace(np.nan, 0)
        expected = float_string_frame.copy()
        expected['foo'] = expected['foo'].astype(object)
        expected = expected.fillna(value=0)
        tm.assert_frame_equal(result, expected)
        tsframe = datetime_frame.copy()
        return_value = tsframe.replace([np.nan], [0], inplace=True)
        assert return_value is None
        tm.assert_frame_equal(tsframe, datetime_frame.fillna(0))

    @pytest.mark.parametrize('to_replace,values,expected', [(['\\s*\\.\\s*', 'e|f|g'], [np.nan, 'crap'], {'a': ['a', 'b', np.nan, np.nan], 'b': ['crap'] * 3 + ['h'], 'c': ['h', 'crap', 'l', 'o']}), (['\\s*(\\.)\\s*', '(e|f|g)'], ['\\1\\1', '\\1_crap'], {'a': ['a', 'b', '..', '..'], 'b': ['e_crap', 'f_crap', 'g_crap', 'h'], 'c': ['h', 'e_crap', 'l', 'o']}), (['\\s*(\\.)\\s*', 'e'], ['\\1\\1', 'crap'], {'a': ['a', 'b', '..', '..'], 'b': ['crap', 'f', 'g', 'h'], 'c': ['h', 'crap', 'l', 'o']})])
    @pytest.mark.parametrize('inplace', [True, False])
    @pytest.mark.parametrize('use_value_regex_args', [True, False])
    def test_regex_replace_list_obj(self, to_replace: List[str], values: List[Union[str, float]], expected: Dict[str, List[Union[str, float]]], inplace: bool, use_value_regex_args: bool) -> None:
        df = DataFrame({'a': list('ab..'), 'b': list('efgh'), 'c': list('helo')})
        if use_value_regex_args:
            result = df.replace(value=values, regex=to_replace, inplace=inplace)
        else:
            result = df.replace(to_replace, values, regex=True, inplace=inplace)
        if inplace:
            assert result is None
            result = df
        expected = DataFrame(expected)
        tm.assert_frame_equal(result, expected)

    def test_regex_replace_list_mixed(self, mix_ab: Dict[str, List[Union[int, str]]]) -> None:
        dfmix = DataFrame(mix_ab)
        to_replace_res = ['\\s*\\.\\s*', 'a']
        values = [np.nan, 'crap']
        mix2 = {'a': list(range(4)), 'b': list('ab..'), 'c': list('halo')}
        dfmix2 = DataFrame(mix2)
        res = dfmix2.replace(to_replace_res, values, regex=True)
        expec = DataFrame({'a': mix2['a'], 'b': ['crap', 'b', np.nan, np.nan], 'c': ['h', 'crap', 'l', 'o']})
        tm.assert_frame_equal(res, expec)
        to_replace_res = ['\\s*(\\.)\\s*', '(a|b)']
        values = ['\\1\\1', '\\1_crap']
        res = dfmix.replace(to_replace_res, values, regex=True)
        expec = DataFrame({'a': mix_ab['a'], 'b': ['a_crap', 'b_crap', '..', '..']})
        tm.assert_frame_equal(res, expec)
        to_replace_res = ['\\s*(\\.)\\s*', 'a', '(b)']
        values = ['\\1\\1', 'crap', '\\1_crap']
        res = dfmix.replace(to_replace_res, values, regex=True)
        expec = DataFrame({'a': mix_ab['a'], 'b': ['crap', 'b_crap', '..', '..']})
        tm.assert_frame_equal(res, expec)
        to_replace_res = ['\\s*(\\.)\\s*', 'a', '(b)']
        values = ['\\1\\1', 'crap', '\\1_crap']
        res = dfmix.replace(regex=to_replace_res, value=values)
        expec = DataFrame({'a': mix_ab['a'], 'b': ['crap', 'b_crap', '..', '..']})
        tm.assert_frame_equal(res, expec)

    def test_regex_replace_list_mixed_inplace(self, mix_ab: Dict[str, List[Union[int, str]]]) -> None:
        dfmix = DataFrame(mix_ab)
        to_replace_res = ['\\s*\\.\\s*', 'a']
        values = [np.nan, 'crap']
        res = dfmix.copy()
        return_value = res.replace(to_replace_res, values, inplace=True, regex=True)
        assert return_value is None
        expec = DataFrame({'a': mix_ab['a'], 'b': ['crap', 'b', np.nan, np.nan]})
        tm.assert_frame_equal(res, expec)
        to_replace_res = ['\\s*(\\.)\\s*', '(a|b)']
        values = ['\\1\\1', '\\1_crap']
        res = dfmix.copy()
        return_value = res.replace(to_replace_res, values, inplace=True, regex=True)
        assert return_value is None
        expec = DataFrame({'a': mix_ab['a'], 'b': ['a_crap', 'b_crap', '..', '..']})
        tm.assert_frame_equal(res, expec)
        to_replace_res = ['\\s*(\\.)\\s*', 'a', '(b)']
        values = ['\\1\\1', 'crap', '\\1_crap']
        res = dfmix.copy()
        return_value = res.replace(to_replace_res, values, inplace=True, regex=True)
        assert return_value is None
        expec = DataFrame({'a': mix_ab['a'], 'b': ['crap', 'b_crap', '..', '..']})
        tm.assert_frame_equal(res, expec)
        to_replace_res = ['\\s*(\\.)\\s*', 'a', '(b)']
        values = ['\\1\\1', 'crap', '\\1_crap']
        res = dfmix.copy()
        return_value = res.replace(regex=to_replace_res, value=values, inplace=True)
        assert return_value is None
        expec = DataFrame({'a': mix_ab['a'], 'b': ['crap', 'b_crap', '..', '..']})
        tm.assert_frame_equal(res, expec)

    def test_regex_replace_dict_mixed(self, mix_abc: Dict[str, List[Union[int, str, float]]]) -> None:
        dfmix = DataFrame(mix_abc)
        res = dfmix.replace({'b': '\\s*\\.\\s*'}, {'b': np.nan}, regex=True)
        res2 = dfmix.copy()
        return_value = res2.replace({'b': '\\s*\\.\\s*'}, {'b': np.nan}, inplace=True, regex=True)
        assert return_value is None
        expec = DataFrame({'a': mix_abc['a'], 'b': ['a', 'b', np.nan, np.nan], 'c': mix_abc['c']})
        tm.assert_frame_equal(res, expec)
        tm.assert_frame_equal(res2, expec)
        res = dfmix.replace({'b': '\\s*(\\.)\\s*'}, {'b': '\\1ty'}, regex=True)
        res2 = dfmix.copy()
        return_value = res2.replace({'b': '\\s*(\\.)\\s*'}, {'b': '\\1ty'}, inplace=True, regex=True)
        assert return_value is None
        expec = DataFrame({'a': mix_abc['a'], 'b': ['a', 'b', '.ty', '.ty'], 'c': mix_abc['c']})
        tm.assert_frame_equal(res, expec)
        tm.assert_frame_equal(res2, expec)
        res = dfmix.replace(regex={'b': '\\s*(\\.)\\s*'}, value={'b': '\\1ty'})
        res2 = dfmix.copy()
        return_value = res2.replace(regex={'b': '\\s*(\\.)\\s*'}, value={'b': '\\1ty'}, inplace=True)
        assert return_value is None
        expec = DataFrame({'a': mix_abc['a'], 'b': ['a', 'b', '.ty', '.ty'], 'c': mix_abc['c']})
        tm.assert_frame_equal(res, expec)
        tm.assert_frame_equal(res2, expec)
        expec = DataFrame({'a': mix_abc['a'], 'b': [np.nan, 'b', '.', '.'], 'c': mix_abc['c']})
        res = dfmix.replace('a', {'b': np.nan}, regex=True)
        res2 = dfmix.copy()
        return_value = res2.replace('a', {'b': np.nan}, regex=True, inplace=True)
        assert return_value is None
        tm.assert_frame_equal(res, expec)
        tm.assert_frame_equal(res2, expec)
        res = dfmix.replace('a', {'b': np.nan}, regex=True)
        res2 = dfmix.copy()
        return_value = res2.replace(regex='a', value={'b': np.nan}, inplace=True)
        assert return_value is None
        expec = DataFrame({'a': mix_abc['a'], 'b': [np.nan, 'b', '.', '.'], 'c': mix_abc['c']})
        tm.assert_frame_equal(res, expec)
        tm.assert_frame_equal(res2, expec)

    def test_regex_replace_dict_nested(self, mix_abc: Dict[str, List[Union[int, str, float]]]) -> None:
        dfmix = DataFrame(mix_abc)
        res = dfmix.replace({'b': {'\\s*\\.\\s*': np.nan}}, regex=True)
        res2 = dfmix.copy()
        res4 = dfmix.copy()
        return_value = res2.replace({'b': {'\\s*\\.\\s*': np.nan}}, inplace=True, regex=True)
        assert return_value is None
        res3 = dfmix.replace(regex={'b': {'\\s*\\.\\s*': np.nan}})
        return_value = res4.replace(regex={'b': {'\\s*\\.\\s*': np.nan}}, inplace=True)
        assert return_value is None
        expec = DataFrame({'a': mix_abc['a'], 'b': ['a', 'b', np.nan, np.nan], 'c': mix_abc['c']})
        tm.assert_frame_equal(res, expec)
        tm.assert_frame_equal(res2, expec)
        tm.assert_frame_equal(res3, expec)
        tm.assert_frame_equal(res4, expec)

    def test_regex_replace_dict_nested_non_first_character(self, any_string_dtype: str, using_infer_string: bool) -> None:
        dtype = any_string_dtype
        df = DataFrame({'first': ['abc', 'bca', 'cab']}, dtype=dtype)
        result = df.replace({'a': '.'}, regex=True)
        expected = DataFrame({'first': ['.bc', 'bc.', 'c.b']}, dtype=dtype)
        tm.assert_frame_equal(result, expected)

    def test_regex_replace_dict_nested_gh4115(self) -> None:
        df = DataFrame({'Type': Series(['Q', 'T', 'Q', 'Q', 'T'], dtype=object), 'tmp': 2})
        expected = DataFrame({'Type': Series([0, 1, 0, 0, 1], dtype=object), 'tmp': 2})
        result = df.replace({'Type': {'Q': 0, 'T': 1}})
        tm.assert_frame_equal(result, expected)

    def test_regex_replace_list_to_scalar(self, mix_abc: Dict[str, List[Union[int, str, float]]]) -> None:
        df = DataFrame(mix_abc)
        expec = DataFrame({'a': mix_abc['a'], 'b': Series([np.nan] * 4, dtype='str'), 'c': [np.nan, np.nan, np.nan, 'd']})
        res = df.replace(['\\s*\\.\\s*', 'a|b'], np.nan, regex=True)
        res2 = df.copy()
        res3 = df.copy()
        return_value = res2.replace(['\\s*\\.\\s*', 'a|b'], np.nan, regex=True, inplace=True)
        assert return_value is None
        return_value = res3.replace(regex=['\\s*\\.\\s*', 'a|b'], value=np.nan, inplace=True)
        assert return_value is None
        tm.assert_frame_equal(res, expec)
        tm.assert_frame_equal(res2, expec)
        tm.assert_frame_equal(res3, expec)

    def test_regex_replace_str_to_numeric(self, mix_abc: Dict[str, List[Union[int, str, float]]]) -> None:
        df = DataFrame(mix_abc)
        res = df.replace('\\s*\\.\\s*', 0, regex=True)
        res2 = df.copy()
        return_value = res2.replace('\\s*\\.\\s*', 0, inplace=True, regex=True)
        assert return_value is None
        res3 = df.copy()
        return_value = res3.replace(regex='\\s*\\.\\s*', value=0, inplace=True)
        assert return_value is None
        expec = DataFrame({'a': mix_abc['a'], 'b': ['a', 'b', 0, 0], 'c': mix_abc['c']})
        expec['c'] = expec['c'].astype(object)
        tm.assert_frame_equal(res, expec)
        tm.assert_frame_equal(res2, expec)
        tm.assert_frame_equal(res3, expec)

    def test_regex_replace_regex_list_to_numeric(self, mix_abc: Dict[str, List[Union[int, str, float]]]) -> None:
        df = DataFrame(mix_abc)
        res = df.replace(['\\s*\\.\\s*', 'b'], 0, regex=True)
        res2 = df.copy()
        return_value = res2.replace(['\\s*\\.\\s*', 'b'], 0, regex=True, inplace=True)
        assert return_value is None
        res3 = df.copy()
        return_value = res3.replace(regex=['\\s*\\.\\s*', 'b'], value=0, inplace=True)
        assert return_value is None
        expec = DataFrame({'a': mix_abc['a'], 'b': ['a', 0, 0, 0], 'c': ['a', 0, np.nan, 'd']})
        tm.assert_frame_equal(res, expec)
        tm.assert_frame_equal(res2, expec)
        tm.assert_frame_equal(res3, expec)

    def test_regex_replace_series_of_regexes(self, mix_abc: Dict[str, List[Union[int, str, float]]]) -> None:
        df = DataFrame(mix_abc)
        s1 = Series({'b': '\\s*\\.\\s*'})
        s2 = Series({'b': np.nan})
        res = df.replace(s1, s2, regex=True)
        res2 = df.copy()
        return_value = res2.replace(s1, s2, inplace=True, regex=True)
        assert return_value is None
        res3 = df.copy()
        return_value = res3.replace(regex=s1, value=s2, inplace=True)
        assert return_value is None
        expec = DataFrame({'a': mix_abc['a'], 'b': ['a', 'b', np.nan, np.nan], 'c': mix_abc['c']})
        tm.assert_frame_equal(res, expec)
        tm.assert_frame_equal(res2, expec)
        tm.assert_frame_equal(res3, expec)

    def test_regex_replace_numeric_to_object_conversion(self, mix_abc: Dict[str, List[Union[int, str, float]]]) -> None:
        df = DataFrame(mix_abc)
        expec = DataFrame({'a': ['a', 1, 2, 3], 'b': mix_abc['b'], 'c': mix_abc['c']})
        res = df.replace(0, 'a')
        tm