from __future__ import annotations
from datetime import datetime
import re
from typing import Any, Callable, Dict, List, Pattern, Sequence, Union

import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame, Index, Series, Timestamp, date_range
import pandas._testing as tm

@pytest.fixture
def mix_ab() -> Dict[str, List[Any]]:
    return {'a': list(range(4)), 'b': list('ab..')}

@pytest.fixture
def mix_abc() -> Dict[str, List[Any]]:
    return {'a': list(range(4)), 'b': list('ab..'), 'c': ['a', 'b', np.nan, 'd']}

class TestDataFrameReplace:
    def test_replace_inplace(self, datetime_frame: DataFrame, float_string_frame: DataFrame) -> None:
        datetime_frame.loc[datetime_frame.index[:5], 'A'] = np.nan
        datetime_frame.loc[datetime_frame.index[-5:], 'A'] = np.nan
        tsframe: DataFrame = datetime_frame.copy()
        return_value = tsframe.replace(np.nan, 0, inplace=True)
        assert return_value is None
        tm.assert_frame_equal(tsframe, datetime_frame.fillna(0))
        mf: DataFrame = float_string_frame
        mf.iloc[5:20, mf.columns.get_loc('foo')] = np.nan
        mf.iloc[-10:, mf.columns.get_loc('A')] = np.nan
        result: DataFrame = float_string_frame.replace(np.nan, 0)
        expected: DataFrame = float_string_frame.copy()
        expected['foo'] = expected['foo'].astype(object)
        expected = expected.fillna(value=0)
        tm.assert_frame_equal(result, expected)
        tsframe = datetime_frame.copy()
        return_value = tsframe.replace([np.nan], [0], inplace=True)
        assert return_value is None
        tm.assert_frame_equal(tsframe, datetime_frame.fillna(0))

    @pytest.mark.parametrize(
        "to_replace,values,expected",
        [
            (
                ['\\s*\\.\\s*', 'e|f|g'],
                [np.nan, 'crap'],
                {'a': ['a', 'b', np.nan, np.nan], 'b': ['crap'] * 3 + ['h'], 'c': ['h', 'crap', 'l', 'o']},
            ),
            (
                ['\\s*(\\.)\\s*', '(e|f|g)'],
                ['\\1\\1', '\\1_crap'],
                {'a': ['a', 'b', '..', '..'], 'b': ['e_crap', 'f_crap', 'g_crap', 'h'], 'c': ['h', 'e_crap', 'l', 'o']},
            ),
            (
                ['\\s*(\\.)\\s*', 'e'],
                ['\\1\\1', 'crap'],
                {'a': ['a', 'b', '..', '..'], 'b': ['crap', 'f', 'g', 'h'], 'c': ['h', 'crap', 'l', 'o']},
            )
        ]
    )
    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.parametrize("use_value_regex_args", [True, False])
    def test_regex_replace_list_obj(
        self,
        to_replace: Union[List[str], List[Any]],
        values: Union[List[Any], List[str]],
        expected: Dict[str, List[Any]],
        inplace: bool,
        use_value_regex_args: bool,
    ) -> None:
        df: DataFrame = DataFrame({'a': list('ab..'), 'b': list('efgh'), 'c': list('helo')})
        if use_value_regex_args:
            result = df.replace(value=values, regex=to_replace, inplace=inplace)
        else:
            result = df.replace(to_replace, values, regex=True, inplace=inplace)
        if inplace:
            assert result is None
            result = df
        expected_df: DataFrame = DataFrame(expected)
        tm.assert_frame_equal(result, expected_df)

    def test_regex_replace_list_mixed(self, mix_ab: Dict[str, List[Any]]) -> None:
        dfmix = DataFrame(mix_ab)
        to_replace_res: List[str] = ['\\s*\\.\\s*', 'a']
        values: List[Union[Any, str]] = [np.nan, 'crap']
        mix2: Dict[str, Any] = {'a': list(range(4)), 'b': list('ab..'), 'c': list('halo')}
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

    def test_regex_replace_list_mixed_inplace(self, mix_ab: Dict[str, List[Any]]) -> None:
        dfmix = DataFrame(mix_ab)
        to_replace_res: List[str] = ['\\s*\\.\\s*', 'a']
        values: List[Union[Any, str]] = [np.nan, 'crap']
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

    def test_regex_replace_dict_mixed(self, mix_abc: Dict[str, List[Any]]) -> None:
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

    def test_regex_replace_dict_nested(self, mix_abc: Dict[str, List[Any]]) -> None:
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

    def test_regex_replace_dict_nested_non_first_character(
        self, any_string_dtype: str, using_infer_string: bool
    ) -> None:
        dtype: str = any_string_dtype
        df = DataFrame({'first': ['abc', 'bca', 'cab']}, dtype=dtype)
        result = df.replace({'a': '.'}, regex=True)
        expected = DataFrame({'first': ['.bc', 'bc.', 'c.b']}, dtype=dtype)
        tm.assert_frame_equal(result, expected)

    def test_regex_replace_dict_nested_gh4115(self) -> None:
        df = DataFrame({'Type': Series(['Q', 'T', 'Q', 'Q', 'T'], dtype=object), 'tmp': 2})
        expected = DataFrame({'Type': Series([0, 1, 0, 0, 1], dtype=object), 'tmp': 2})
        result = df.replace({'Type': {'Q': 0, 'T': 1}})
        tm.assert_frame_equal(result, expected)

    def test_regex_replace_list_to_scalar(self, mix_abc: Dict[str, List[Any]]) -> None:
        df = DataFrame(mix_abc)
        expec = DataFrame({
            'a': mix_abc['a'],
            'b': Series([np.nan] * 4, dtype='str'),
            'c': [np.nan, np.nan, np.nan, 'd']
        })
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

    def test_regex_replace_str_to_numeric(self, mix_abc: Dict[str, List[Any]]) -> None:
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

    def test_regex_replace_regex_list_to_numeric(self, mix_abc: Dict[str, List[Any]]) -> None:
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

    def test_regex_replace_series_of_regexes(self, mix_abc: Dict[str, List[Any]]) -> None:
        df = DataFrame(mix_abc)
        s1: Series = Series({'b': '\\s*\\.\\s*'})
        s2: Series = Series({'b': np.nan})
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

    def test_regex_replace_numeric_to_object_conversion(self, mix_abc: Dict[str, List[Any]]) -> None:
        df = DataFrame(mix_abc)
        expec = DataFrame({'a': ['a', 1, 2, 3], 'b': mix_abc['b'], 'c': mix_abc['c']})
        res = df.replace(0, 'a')
        tm.assert_frame_equal(res, expec)
        assert res.a.dtype == np.object_

    @pytest.mark.parametrize("to_replace", [{'': np.nan, ',': ''}, {',': '', '': np.nan}])
    def test_joint_simple_replace_and_regex_replace(self, to_replace: Dict[str, Any]) -> None:
        df = DataFrame({'col1': ['1,000', 'a', '3'], 'col2': ['a', '', 'b'], 'col3': ['a', 'b', 'c']})
        result = df.replace(regex=to_replace)
        expected = DataFrame({'col1': ['1000', 'a', '3'], 'col2': ['a', np.nan, 'b'], 'col3': ['a', 'b', 'c']})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("metachar", ['[]', '()', '\\d', '\\w', '\\s'])
    def test_replace_regex_metachar(self, metachar: str) -> None:
        df = DataFrame({'a': [metachar, 'else']})
        result = df.replace({'a': {metachar: 'paren'}})
        expected = DataFrame({'a': ['paren', 'else']})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "data,to_replace,expected",
        [
            (['xax', 'xbx'], {'a': 'c', 'b': 'd'}, ['xcx', 'xdx']),
            (['d', '', ''], {'^\\s*$': pd.NA}, ['d', pd.NA, pd.NA])
        ]
    )
    def test_regex_replace_string_types(
        self,
        data: List[Any],
        to_replace: Dict[str, Any],
        expected: List[Any],
        frame_or_series: Callable[[List[Any], Any], Any],
        any_string_dtype: str,
        using_infer_string: bool,
        request: Any
    ) -> None:
        dtype: str = any_string_dtype
        obj = frame_or_series(data, dtype=dtype)
        result = obj.replace(to_replace, regex=True)
        expected_obj = frame_or_series(expected, dtype=dtype)
        tm.assert_equal(result, expected_obj)

    def test_replace(self, datetime_frame: DataFrame) -> None:
        datetime_frame.loc[datetime_frame.index[:5], 'A'] = np.nan
        datetime_frame.loc[datetime_frame.index[-5:], 'A'] = np.nan
        zero_filled = datetime_frame.replace(np.nan, -100000000.0)
        tm.assert_frame_equal(zero_filled, datetime_frame.fillna(-100000000.0))
        tm.assert_frame_equal(zero_filled.replace(-100000000.0, np.nan), datetime_frame)
        datetime_frame.loc[datetime_frame.index[:5], 'A'] = np.nan
        datetime_frame.loc[datetime_frame.index[-5:], 'A'] = np.nan
        datetime_frame.loc[datetime_frame.index[:5], 'B'] = -100000000.0
        df = DataFrame(index=['a', 'b'])
        tm.assert_frame_equal(df, df.replace(5, 7))
        df = DataFrame([('-', pd.to_datetime('20150101')), ('a', pd.to_datetime('20150102'))])
        df1 = df.replace('-', np.nan)
        expected_df = DataFrame([(np.nan, pd.to_datetime('20150101')), ('a', pd.to_datetime('20150102'))])
        tm.assert_frame_equal(df1, expected_df)

    def test_replace_list(self) -> None:
        obj: Dict[str, List[Any]] = {'a': list('ab..'), 'b': list('efgh'), 'c': list('helo')}
        dfobj: DataFrame = DataFrame(obj)
        to_replace_res: List[str] = ['.', 'e']
        values: List[Union[Any, str]] = [np.nan, 'crap']
        res = dfobj.replace(to_replace_res, values)
        expec = DataFrame({'a': ['a', 'b', np.nan, np.nan], 'b': ['crap', 'f', 'g', 'h'], 'c': ['h', 'crap', 'l', 'o']})
        tm.assert_frame_equal(res, expec)
        to_replace_res = ['.', 'f']
        values = ['..', 'crap']
        res = dfobj.replace(to_replace_res, values)
        expec = DataFrame({'a': ['a', 'b', '..', '..'], 'b': ['e', 'crap', 'g', 'h'], 'c': ['h', 'e', 'l', 'o']})
        tm.assert_frame_equal(res, expec)

    def test_replace_with_empty_list(self, frame_or_series: Callable[[Any], Any]) -> None:
        ser = Series([['a', 'b'], [], np.nan, [1]])
        obj = DataFrame({'col': ser})
        obj = tm.get_obj(obj, frame_or_series)
        expected = obj
        result = obj.replace([], np.nan)
        tm.assert_equal(result, expected)
        msg: str = 'NumPy boolean array indexing assignment cannot assign {size} input values to the 1 output values where the mask is true'
        with pytest.raises(ValueError, match=msg.format(size=0)):
            obj.replace({np.nan: []})
        with pytest.raises(ValueError, match=msg.format(size=2)):
            obj.replace({np.nan: ['dummy', 'alt']})

    def test_replace_series_dict(self) -> None:
        df = DataFrame({'zero': {'a': 0.0, 'b': 1}, 'one': {'a': 2.0, 'b': 0}})
        result = df.replace(0, {'zero': 0.5, 'one': 1.0})
        expected = DataFrame({'zero': {'a': 0.5, 'b': 1}, 'one': {'a': 2.0, 'b': 1.0}})
        tm.assert_frame_equal(result, expected)
        result = df.replace(0, df.mean())
        tm.assert_frame_equal(result, expected)
        df = DataFrame({'zero': {'a': 0.0, 'b': 1}, 'one': {'a': 2.0, 'b': 0}})
        s: Series = Series({'zero': 0.0, 'one': 2.0})
        result = df.replace(s, {'zero': 0.5, 'one': 1.0})
        expected = DataFrame({'zero': {'a': 0.5, 'b': 1}, 'one': {'a': 1.0, 'b': 0.0}})
        tm.assert_frame_equal(result, expected)
        result = df.replace(s, df.mean())
        tm.assert_frame_equal(result, expected)

    def test_replace_convert(self, any_string_dtype: str) -> None:
        df = DataFrame([['foo', 'bar', 'bah'], ['bar', 'foo', 'bah']], dtype=any_string_dtype)
        m: Dict[str, Union[int, str]] = {'foo': 1, 'bar': 2, 'bah': 3}
        rep = df.replace(m)
        assert (rep.dtypes == object).all()

    def test_replace_mixed(self, float_string_frame: DataFrame) -> None:
        mf: DataFrame = float_string_frame
        mf.iloc[5:20, mf.columns.get_loc('foo')] = np.nan
        mf.iloc[-10:, mf.columns.get_loc('A')] = np.nan
        result = float_string_frame.replace(np.nan, -18)
        expected = float_string_frame.copy()
        expected['foo'] = expected['foo'].astype(object)
        expected = expected.fillna(value=-18)
        tm.assert_frame_equal(result, expected)
        expected2 = float_string_frame.copy()
        expected2['foo'] = expected2['foo'].astype(object)
        tm.assert_frame_equal(result.replace(-18, np.nan), expected2)
        result = float_string_frame.replace(np.nan, -100000000.0)
        expected = float_string_frame.copy()
        expected['foo'] = expected['foo'].astype(object)
        expected = expected.fillna(value=-100000000.0)
        tm.assert_frame_equal(result, expected)
        expected2 = float_string_frame.copy()
        expected2['foo'] = expected2['foo'].astype(object)
        tm.assert_frame_equal(result.replace(-100000000.0, np.nan), expected2)

    def test_replace_mixed_int_block_upcasting(self) -> None:
        df = DataFrame({
            'A': Series([1.0, 2.0], dtype='float64'),
            'B': Series([0, 1], dtype='int64')
        })
        expected = DataFrame({
            'A': Series([1.0, 2.0], dtype='float64'),
            'B': Series([0.5, 1], dtype='float64')
        })
        result = df.replace(0, 0.5)
        tm.assert_frame_equal(result, expected)
        return_value = df.replace(0, 0.5, inplace=True)
        assert return_value is None
        tm.assert_frame_equal(df, expected)

    def test_replace_mixed_int_block_splitting(self) -> None:
        df = DataFrame({
            'A': Series([1.0, 2.0], dtype='float64'),
            'B': Series([0, 1], dtype='int64'),
            'C': Series([1, 2], dtype='int64')
        })
        expected = DataFrame({
            'A': Series([1.0, 2.0], dtype='float64'),
            'B': Series([0.5, 1], dtype='float64'),
            'C': Series([1, 2], dtype='int64')
        })
        result = df.replace(0, 0.5)
        tm.assert_frame_equal(result, expected)

    def test_replace_mixed2(self) -> None:
        df = DataFrame({
            'A': Series([1.0, 2.0], dtype='float64'),
            'B': Series([0, 1], dtype='int64')
        })
        expected = DataFrame({
            'A': Series([1, 'foo'], dtype='object'),
            'B': Series([0, 1], dtype='int64')
        })
        result = df.replace(2, 'foo')
        tm.assert_frame_equal(result, expected)
        expected = DataFrame({
            'A': Series(['foo', 'bar'], dtype='object'),
            'B': Series([0, 'foo'], dtype='object')
        })
        result = df.replace([1, 2], ['foo', 'bar'])
        tm.assert_frame_equal(result, expected)

    def test_replace_mixed3(self) -> None:
        df = DataFrame({
            'A': Series([3, 0], dtype='int64'),
            'B': Series([0, 3], dtype='int64')
        })
        result = df.replace(3, df.mean().to_dict())
        expected = df.copy().astype('float64')
        m = df.mean()
        expected.iloc[0, 0] = m.iloc[0]
        expected.iloc[1, 1] = m.iloc[1]
        tm.assert_frame_equal(result, expected)

    def test_replace_nullable_int_with_string_doesnt_cast(self) -> None:
        df = DataFrame({'a': [1, 2, 3, np.nan], 'b': ['some', 'strings', 'here', 'he']})
        df['a'] = df['a'].astype('Int64')
        res = df.replace('', np.nan)
        tm.assert_series_equal(res['a'], df['a'])

    @pytest.mark.parametrize("dtype", ['boolean', 'Int64', 'Float64'])
    def test_replace_with_nullable_column(self, dtype: str) -> None:
        nullable_ser: Series = Series([1, 0, 1], dtype=dtype)
        df = DataFrame({'A': ['A', 'B', 'x'], 'B': nullable_ser})
        result = df.replace('x', 'X')
        expected = DataFrame({'A': ['A', 'B', 'X'], 'B': nullable_ser})
        tm.assert_frame_equal(result, expected)

    def test_replace_simple_nested_dict(self) -> None:
        df = DataFrame({'col': range(1, 5)})
        expected = DataFrame({'col': ['a', 2, 3, 'b']})
        result = df.replace({'col': {1: 'a', 4: 'b'}})
        tm.assert_frame_equal(expected, result)
        result = df.replace({1: 'a', 4: 'b'})
        tm.assert_frame_equal(expected, result)

    def test_replace_simple_nested_dict_with_nonexistent_value(self) -> None:
        df = DataFrame({'col': range(1, 5)})
        expected = DataFrame({'col': ['a', 2, 3, 'b']})
        result = df.replace({-1: '-', 1: 'a', 4: 'b'})
        tm.assert_frame_equal(expected, result)
        result = df.replace({'col': {-1: '-', 1: 'a', 4: 'b'}})
        tm.assert_frame_equal(expected, result)

    def test_replace_NA_with_None(self) -> None:
        df = DataFrame({'value': [42, None]}).astype({'value': 'Int64'})
        result = df.replace({pd.NA: None})
        expected = DataFrame({'value': [42, None]}, dtype=object)
        tm.assert_frame_equal(result, expected)

    def test_replace_NAT_with_None(self) -> None:
        df = DataFrame([pd.NaT, pd.NaT])
        result = df.replace({pd.NaT: None, np.nan: None})
        expected = DataFrame([None, None])
        tm.assert_frame_equal(result, expected)

    def test_replace_with_None_keeps_categorical(self) -> None:
        cat_series: Series = Series(['b', 'b', 'b', 'd'], dtype='category')
        df = DataFrame({'id': Series([5, 4, 3, 2], dtype='float64'), 'col': cat_series})
        result = df.replace({3: None})
        expected = DataFrame({'id': Series([5.0, 4.0, None, 2.0], dtype='object'), 'col': cat_series})
        tm.assert_frame_equal(result, expected)

    def test_replace_all_NA(self) -> None:
        df = DataFrame({'ticker': ['#1234#'], 'name': [None]})
        result = df.replace({col: {'^#': '$'} for col in df.columns}, regex=True)
        expected = DataFrame({'ticker': ['$1234#'], 'name': [None]})
        tm.assert_frame_equal(result, expected)

    def test_replace_value_is_none(self, datetime_frame: DataFrame) -> None:
        orig_value = datetime_frame.iloc[0, 0]
        orig2 = datetime_frame.iloc[1, 0]
        datetime_frame.iloc[0, 0] = np.nan
        datetime_frame.iloc[1, 0] = 1
        result = datetime_frame.replace(to_replace={np.nan: 0})
        expected = datetime_frame.T.replace(to_replace={np.nan: 0}).T
        tm.assert_frame_equal(result, expected)
        result = datetime_frame.replace(to_replace={np.nan: 0, 1: -100000000.0})
        tsframe = datetime_frame.copy()
        tsframe.iloc[0, 0] = 0
        tsframe.iloc[1, 0] = -100000000.0
        expected = tsframe
        tm.assert_frame_equal(expected, result)
        datetime_frame.iloc[0, 0] = orig_value
        datetime_frame.iloc[1, 0] = orig2

    def test_replace_for_new_dtypes(self, datetime_frame: DataFrame) -> None:
        tsframe: DataFrame = datetime_frame.copy().astype(np.float32)
        tsframe.loc[tsframe.index[:5], 'A'] = np.nan
        tsframe.loc[tsframe.index[-5:], 'A'] = np.nan
        zero_filled = tsframe.replace(np.nan, -100000000.0)
        tm.assert_frame_equal(zero_filled, tsframe.fillna(-100000000.0))
        tm.assert_frame_equal(zero_filled.replace(-100000000.0, np.nan), tsframe)
        tsframe.loc[tsframe.index[:5], 'A'] = np.nan
        tsframe.loc[tsframe.index[-5:], 'A'] = np.nan
        tsframe.loc[tsframe.index[:5], 'B'] = np.nan

    @pytest.mark.parametrize(
        "frame, to_replace, value, expected",
        [
            (DataFrame({'ints': [1, 2, 3]}), 1, 0, DataFrame({'ints': [0, 2, 3]})),
            (DataFrame({'ints': [1, 2, 3]}, dtype=np.int32), 1, 0, DataFrame({'ints': [0, 2, 3]}, dtype=np.int32)),
            (DataFrame({'ints': [1, 2, 3]}, dtype=np.int16), 1, 0, DataFrame({'ints': [0, 2, 3]}, dtype=np.int16)),
            (DataFrame({'bools': [True, False, True]}), False, True, DataFrame({'bools': [True, True, True]})),
            (DataFrame({'complex': [1j, 2j, 3j]}), 1j, 0, DataFrame({'complex': [0j, 2j, 3j]})),
            (DataFrame({'datetime64': Index([datetime(2018, 5, 28), datetime(2018, 7, 28), datetime(2018, 5, 28)])}),
             datetime(2018, 5, 28), datetime(2018, 7, 28),
             DataFrame({'datetime64': Index([datetime(2018, 7, 28)] * 3)})),
            (DataFrame({'dt': [datetime(3017, 12, 20)], 'str': ['foo']}),
             'foo', 'bar',
             DataFrame({'dt': [datetime(3017, 12, 20)], 'str': ['bar']})),
            (DataFrame({'A': date_range('20130101', periods=3, tz='US/Eastern'), 'B': [0, np.nan, 2]}),
             Timestamp('20130102', tz='US/Eastern'), Timestamp('20130104', tz='US/Eastern'),
             DataFrame({'A': pd.DatetimeIndex([Timestamp('20130101', tz='US/Eastern'),
                                                Timestamp('20130104', tz='US/Eastern'),
                                                Timestamp('20130103', tz='US/Eastern')]).as_unit('ns'),
                        'B': [0, np.nan, 2]})),
            (DataFrame([[1, 1.0], [2, 2.0]]), 1.0, 5, DataFrame([[5, 5.0], [2, 2.0]])),
            (DataFrame([[1, 1.0], [2, 2.0]]), 1, 5, DataFrame([[5, 5.0], [2, 2.0]])),
            (DataFrame([[1, 1.0], [2, 2.0]]), 1.0, 5.0, DataFrame([[5, 5.0], [2, 2.0]])),
            (DataFrame([[1, 1.0], [2, 2.0]]), 1, 5.0, DataFrame([[5, 5.0], [2, 2.0]]))
        ]
    )
    def test_replace_dtypes(self, frame: DataFrame, to_replace: Any, value: Any, expected: DataFrame) -> None:
        result = frame.replace(to_replace, value)
        tm.assert_frame_equal(result, expected)

    def test_replace_input_formats_listlike(self) -> None:
        to_rep: Dict[str, Any] = {'A': np.nan, 'B': 0, 'C': ''}
        values: Dict[str, Union[int, str]] = {'A': 0, 'B': -1, 'C': 'missing'}
        df = DataFrame({'A': [np.nan, 0, np.inf], 'B': [0, 2, 5], 'C': ['', 'asdf', 'fd']})
        filled = df.replace(to_rep, values)
        expected = {k: v.replace(to_rep[k], values[k]) for k, v in df.items()}
        tm.assert_frame_equal(filled, DataFrame(expected))
        result = df.replace([0, 2, 5], [5, 2, 0])
        expected = DataFrame({'A': [np.nan, 5, np.inf], 'B': [5, 2, 0], 'C': ['', 'asdf', 'fd']})
        tm.assert_frame_equal(result, expected)
        values = {'A': 0, 'B': -1, 'C': 'missing'}
        df = DataFrame({'A': [np.nan, 0, np.nan], 'B': [0, 2, 5], 'C': ['', 'asdf', 'fd']})
        filled = df.replace(np.nan, values)
        expected = {k: v.replace(np.nan, values[k]) for k, v in df.items()}
        tm.assert_frame_equal(filled, DataFrame(expected))
        to_rep = [np.nan, 0, '']
        values = [-2, -1, 'missing']
        result = df.replace(to_rep, values)
        expected = df.copy()
        for rep, value in zip(to_rep, values):
            return_value = expected.replace(rep, value, inplace=True)
            assert return_value is None
        tm.assert_frame_equal(result, expected)
        msg: str = 'Replacement lists must match in length\\. Expecting 3 got 2'
        with pytest.raises(ValueError, match=msg):
            df.replace(to_rep, values[1:])

    def test_replace_input_formats_scalar(self) -> None:
        df = DataFrame({'A': [np.nan, 0, np.inf], 'B': [0, 2, 5], 'C': ['', 'asdf', 'fd']})
        to_rep: Dict[str, Any] = {'A': np.nan, 'B': 0, 'C': ''}
        filled = df.replace(to_rep, 0)
        expected = {k: v.replace(to_rep[k], 0) for k, v in df.items()}
        tm.assert_frame_equal(filled, DataFrame(expected))
        msg: str = 'value argument must be scalar, dict, or Series'
        with pytest.raises(TypeError, match=msg):
            df.replace(to_rep, [np.nan, 0, ''])
        to_rep = [np.nan, 0, '']
        result = df.replace(to_rep, -1)
        expected = df.copy()
        for rep in to_rep:
            return_value = expected.replace(rep, -1, inplace=True)
            assert return_value is None
        tm.assert_frame_equal(result, expected)

    def test_replace_limit(self) -> None:
        pass

    def test_replace_dict_no_regex(self, any_string_dtype: str) -> None:
        answer = Series({0: 'Strongly Agree', 1: 'Agree', 2: 'Neutral', 3: 'Disagree', 4: 'Strongly Disagree'}, dtype=any_string_dtype)
        weights = {'Agree': 4, 'Disagree': 2, 'Neutral': 3, 'Strongly Agree': 5, 'Strongly Disagree': 1}
        expected = Series({0: 5, 1: 4, 2: 3, 3: 2, 4: 1}, dtype=object)
        result = answer.replace(weights)
        tm.assert_series_equal(result, expected)

    def test_replace_series_no_regex(self, any_string_dtype: str) -> None:
        answer = Series({0: 'Strongly Agree', 1: 'Agree', 2: 'Neutral', 3: 'Disagree', 4: 'Strongly Disagree'}, dtype=any_string_dtype)
        weights = Series({'Agree': 4, 'Disagree': 2, 'Neutral': 3, 'Strongly Agree': 5, 'Strongly Disagree': 1})
        expected = Series({0: 5, 1: 4, 2: 3, 3: 2, 4: 1}, dtype=object)
        result = answer.replace(weights)
        tm.assert_series_equal(result, expected)

    def test_replace_dict_tuple_list_ordering_remains_the_same(self) -> None:
        df = DataFrame({'A': [np.nan, 1]})
        res1 = df.replace(to_replace={np.nan: 0, 1: -100000000.0})
        res2 = df.replace(to_replace=(1, np.nan), value=[-100000000.0, 0])
        res3 = df.replace(to_replace=[1, np.nan], value=[-100000000.0, 0])
        expected = DataFrame({'A': [0, -100000000.0]})
        tm.assert_frame_equal(res1, res2)
        tm.assert_frame_equal(res2, res3)
        tm.assert_frame_equal(res3, expected)

    def test_replace_doesnt_replace_without_regex(self) -> None:
        df = DataFrame({
            'fol': [1, 2, 2, 3],
            'T_opp': ['0', 'vr', '0', '0'],
            'T_Dir': ['0', '0', '0', 'bt'],
            'T_Enh': ['vo', '0', '0', '0']
        })
        res = df.replace({'\\D': 1})
        tm.assert_frame_equal(df, res)

    def test_replace_bool_with_string(self) -> None:
        df = DataFrame({'a': [True, False], 'b': list('ab')})
        result = df.replace(True, 'a')
        expected = DataFrame({'a': ['a', False], 'b': df.b})
        tm.assert_frame_equal(result, expected)

    def test_replace_pure_bool_with_string_no_op(self) -> None:
        df = DataFrame(np.random.default_rng(2).random((2, 2)) > 0.5)
        result = df.replace('asdf', 'fdsa')
        tm.assert_frame_equal(df, result)

    def test_replace_bool_with_bool(self) -> None:
        df = DataFrame(np.random.default_rng(2).random((2, 2)) > 0.5)
        result = df.replace(False, True)
        expected = DataFrame(np.ones((2, 2), dtype=bool))
        tm.assert_frame_equal(result, expected)

    def test_replace_with_dict_with_bool_keys(self) -> None:
        df = DataFrame({0: [True, False], 1: [False, True]})
        result = df.replace({'asdf': 'asdb', True: 'yes'})
        expected = DataFrame({0: ['yes', False], 1: [False, 'yes']})
        tm.assert_frame_equal(result, expected)

    def test_replace_dict_strings_vs_ints(self) -> None:
        df = DataFrame({'Y0': [1, 2], 'Y1': [3, 4]})
        result = df.replace({'replace_string': 'test'})
        tm.assert_frame_equal(result, df)
        result = df['Y0'].replace({'replace_string': 'test'})
        tm.assert_series_equal(result, df['Y0'])

    def test_replace_truthy(self) -> None:
        df = DataFrame({'a': [True, True]})
        r = df.replace([np.inf, -np.inf], np.nan)
        e = df
        tm.assert_frame_equal(r, e)

    def test_nested_dict_overlapping_keys_replace_int(self) -> None:
        df = DataFrame({'a': list(range(1, 5))})
        result = df.replace({'a': dict(zip(range(1, 5), range(2, 6)))})
        expected = df.replace(dict(zip(range(1, 5), range(2, 6))))
        tm.assert_frame_equal(result, expected)

    def test_nested_dict_overlapping_keys_replace_str(self) -> None:
        a = np.arange(1, 5)
        astr = a.astype(str)
        bstr = np.arange(2, 6).astype(str)
        df = DataFrame({'a': astr})
        result = df.replace(dict(zip(astr, bstr)))
        expected = df.replace({'a': dict(zip(astr, bstr))})
        tm.assert_frame_equal(result, expected)

    def test_replace_swapping_bug(self) -> None:
        df = DataFrame({'a': [True, False, True]})
        res = df.replace({'a': {True: 'Y', False: 'N'}})
        expect = DataFrame({'a': ['Y', 'N', 'Y']}, dtype=object)
        tm.assert_frame_equal(res, expect)
        df = DataFrame({'a': [0, 1, 0]})
        res = df.replace({'a': {0: 'Y', 1: 'N'}})
        expect = DataFrame({'a': ['Y', 'N', 'Y']}, dtype=object)
        tm.assert_frame_equal(res, expect)

    def test_replace_datetimetz(self) -> None:
        df = DataFrame({'A': date_range('20130101', periods=3, tz='US/Eastern'), 'B': [0, np.nan, 2]})
        result = df.replace(np.nan, 1)
        expected = DataFrame({'A': date_range('20130101', periods=3, tz='US/Eastern'), 'B': Series([0, 1, 2], dtype='float64')})
        tm.assert_frame_equal(result, expected)
        result = df.fillna(1)
        tm.assert_frame_equal(result, expected)
        result = df.replace(0, np.nan)
        expected = DataFrame({'A': date_range('20130101', periods=3, tz='US/Eastern'), 'B': [np.nan, np.nan, 2]})
        tm.assert_frame_equal(result, expected)
        result = df.replace(Timestamp('20130102', tz='US/Eastern'), Timestamp('20130104', tz='US/Eastern'))
        expected = DataFrame({'A': [Timestamp('20130101', tz='US/Eastern'), Timestamp('20130104', tz='US/Eastern'), Timestamp('20130103', tz='US/Eastern')], 'B': [0, np.nan, 2]})
        expected['A'] = expected['A'].dt.as_unit('ns')
        tm.assert_frame_equal(result, expected)
        result = df.copy()
        result.iloc[1, 0] = np.nan
        result = result.replace({'A': pd.NaT}, Timestamp('20130104', tz='US/Eastern'))
        tm.assert_frame_equal(result, expected)
        result = df.copy()
        result.iloc[1, 0] = np.nan
        result = result.replace({'A': pd.NaT}, Timestamp('20130104', tz='US/Pacific'))
        expected = DataFrame({'A': [Timestamp('20130101', tz='US/Eastern'), Timestamp('20130104', tz='US/Pacific').tz_convert('US/Eastern'), Timestamp('20130103', tz='US/Eastern')], 'B': [0, np.nan, 2]})
        expected['A'] = expected['A'].dt.as_unit('ns')
        tm.assert_frame_equal(result, expected)
        result = df.copy()
        result.iloc[1, 0] = np.nan
        result = result.replace({'A': np.nan}, Timestamp('20130104'))
        expected = DataFrame({'A': [Timestamp('20130101', tz='US/Eastern'), Timestamp('20130104'), Timestamp('20130103', tz='US/Eastern')], 'B': [0, np.nan, 2]})
        tm.assert_frame_equal(result, expected)

    def test_replace_with_empty_dictlike(self, mix_abc: Dict[str, List[Any]]) -> None:
        df = DataFrame(mix_abc)
        tm.assert_frame_equal(df, df.replace({}))
        tm.assert_frame_equal(df, df.replace(Series([], dtype=object)))
        tm.assert_frame_equal(df, df.replace({'b': {}}))
        tm.assert_frame_equal(df, df.replace(Series({'b': {}})))

    @pytest.mark.parametrize(
        "df, to_replace, exp",
        [
            ({'col1': [1, 2, 3], 'col2': [4, 5, 6]}, {'col2': {4: 5, 5: 6, 6: 7}}, {'col1': [1, 2, 3], 'col2': [5, 6, 7]}),
            ({'col1': [1, 2, 3], 'col2': ['4', '5', '6']}, {'col2': {'4': '5', '5': '6', '6': '7'}}, {'col1': [1, 2, 3], 'col2': ['5', '6', '7']})
        ]
    )
    def test_replace_commutative(self, df: Dict[str, List[Any]], to_replace: Dict[str, Any], exp: Dict[str, List[Any]]) -> None:
        df_obj: DataFrame = DataFrame(df)
        expected = DataFrame(exp)
        result = df_obj.replace(to_replace)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("replacer", [Timestamp('20170827'), np.int8(1), np.int16(1), np.float32(1), np.float64(1)])
    def test_replace_replacer_dtype(self, replacer: Union[Timestamp, np.generic]) -> None:
        df = DataFrame(['a'], dtype=object)
        result = df.replace({'a': replacer, 'b': replacer})
        expected = DataFrame([replacer], dtype=object)
        tm.assert_frame_equal(result, expected)

    def test_replace_after_convert_dtypes(self) -> None:
        df = DataFrame({'grp': [1, 2, 3, 4, 5]}, dtype='Int64')
        result = df.replace(1, 10)
        expected = DataFrame({'grp': [10, 2, 3, 4, 5]}, dtype='Int64')
        tm.assert_frame_equal(result, expected)

    def test_replace_invalid_to_replace(self) -> None:
        df = DataFrame({'one': ['a', 'b ', 'c'], 'two': ['d ', 'e ', 'f ']})
        msg: str = "Expecting 'to_replace' to be either a scalar, array-like, dict or None, got invalid type.*"
        with pytest.raises(TypeError, match=msg):
            df.replace(lambda x: x.strip())

    @pytest.mark.parametrize("dtype", ['float', 'float64', 'int64', 'Int64', 'boolean'])
    @pytest.mark.parametrize("value", [np.nan, pd.NA])
    def test_replace_no_replacement_dtypes(self, dtype: str, value: Union[float, pd._libs.missing.NAType]) -> None:
        df = DataFrame(np.eye(2), dtype=dtype)
        result = df.replace(to_replace=[None, -np.inf, np.inf], value=value)
        tm.assert_frame_equal(result, df)

    @pytest.mark.parametrize("replacement", [np.nan, 5])
    def test_replace_with_duplicate_columns(self, replacement: Union[float, int]) -> None:
        result = DataFrame({'A': [1, 2, 3], 'A1': [4, 5, 6], 'B': [7, 8, 9]})
        result.columns = list('AAB')
        expected = DataFrame({'A': [1, 2, 3], 'A1': [4, 5, 6], 'B': [replacement, 8, 9]})
        expected.columns = list('AAB')
        result['B'] = result['B'].replace(7, replacement)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("value", [pd.Period('2020-01'), pd.Interval(0, 5)])
    def test_replace_ea_ignore_float(self, frame_or_series: Callable[[Any], Any], value: Union[pd.Period, pd.Interval]) -> None:
        obj = DataFrame({'Per': [value] * 3})
        obj = tm.get_obj(obj, frame_or_series)
        expected = obj.copy()
        result = obj.replace(1.0, 0.0)
        tm.assert_equal(expected, result)

    @pytest.mark.parametrize(
        "replace_dict, final_data",
        [
            ({'a': 1, 'b': 1}, [[2, 2], [2, 2]]),
            ({'a': 1, 'b': 2}, [[2, 1], [2, 2]])
        ]
    )
    def test_categorical_replace_with_dict(self, replace_dict: Dict[str, int], final_data: List[List[Any]]) -> None:
        df = DataFrame([[1, 1], [2, 2]], columns=['a', 'b'], dtype='category')
        final_data_np = np.array(final_data)
        a = pd.Categorical(final_data_np[:, 0], categories=[1, 2])
        b = pd.Categorical(final_data_np[:, 1], categories=[1, 2])
        expected = DataFrame({'a': a, 'b': b})
        result = df.replace(replace_dict, 2)
        tm.assert_frame_equal(result, expected)
        msg: str = 'DataFrame.iloc\\[:, 0\\] \\(column name=\\"a\\"\\) are different'
        with pytest.raises(AssertionError, match=msg):
            tm.assert_frame_equal(df, expected)
        return_value = df.replace(replace_dict, 2, inplace=True)
        assert return_value is None
        tm.assert_frame_equal(df, expected)

    def test_replace_value_category_type(self) -> None:
        input_dict = {
            'col1': [1, 2, 3, 4],
            'col2': ['a', 'b', 'c', 'd'],
            'col3': [1.5, 2.5, 3.5, 4.5],
            'col4': ['cat1', 'cat2', 'cat3', 'cat4'],
            'col5': ['obj1', 'obj2', 'obj3', 'obj4']
        }
        input_df = DataFrame(data=input_dict).astype({'col2': 'category', 'col4': 'category'})
        input_df['col2'] = input_df['col2'].cat.reorder_categories(['a', 'b', 'c', 'd'], ordered=True)
        input_df['col4'] = input_df['col4'].cat.reorder_categories(['cat1', 'cat2', 'cat3', 'cat4'], ordered=True)
        expected_dict = {
            'col1': [1, 2, 3, 4],
            'col2': ['a', 'b', 'c', 'z'],
            'col3': [1.5, 2.5, 3.5, 4.5],
            'col4': ['cat1', 'catX', 'cat3', 'cat4'],
            'col5': ['obj9', 'obj2', 'obj3', 'obj4']
        }
        expected = DataFrame(data=expected_dict).astype({'col2': 'category', 'col4': 'category'})
        expected['col2'] = expected['col2'].cat.reorder_categories(['a', 'b', 'c', 'z'], ordered=True)
        expected['col4'] = expected['col4'].cat.reorder_categories(['cat1', 'catX', 'cat3', 'cat4'], ordered=True)
        input_df = input_df.apply(lambda x: x.astype('category').cat.rename_categories({'d': 'z'}))
        input_df = input_df.apply(lambda x: x.astype('category').cat.rename_categories({'obj1': 'obj9'}))
        result = input_df.apply(lambda x: x.astype('category').cat.rename_categories({'cat2': 'catX'}))
        result = result.astype({'col1': 'int64', 'col3': 'float64', 'col5': 'str'})
        tm.assert_frame_equal(result, expected)

    def test_replace_dict_category_type(self) -> None:
        input_dict = {'col1': ['a'], 'col2': ['obj1'], 'col3': ['cat1']}
        input_df = DataFrame(data=input_dict).astype({'col1': 'category', 'col2': 'category', 'col3': 'category'})
        expected_dict = {'col1': ['z'], 'col2': ['obj9'], 'col3': ['catX']}
        expected = DataFrame(data=expected_dict).astype({'col1': 'category', 'col2': 'category', 'col3': 'category'})
        result = input_df.apply(lambda x: x.cat.rename_categories({'a': 'z', 'obj1': 'obj9', 'cat1': 'catX'}))
        tm.assert_frame_equal(result, expected)

    def test_replace_with_compiled_regex(self) -> None:
        df = DataFrame(['a', 'b', 'c'])
        regex: Pattern = re.compile('^a$')
        result = df.replace({regex: 'z'}, regex=True)
        expected = DataFrame(['z', 'b', 'c'])
        tm.assert_frame_equal(result, expected)

    def test_replace_intervals(self) -> None:
        df = DataFrame({'a': [pd.Interval(0, 1), pd.Interval(0, 1)]})
        result = df.replace({'a': {pd.Interval(0, 1): 'x'}})
        expected = DataFrame({'a': ['x', 'x']}, dtype=object)
        tm.assert_frame_equal(result, expected)

    def test_replace_unicode(self) -> None:
        columns_values_map: Dict[str, Dict[str, int]] = {'positive': {'正面': 1, '中立': 1, '负面': 0}}
        df1 = DataFrame({'positive': np.ones(3)})
        result = df1.replace(columns_values_map)
        expected = DataFrame({'positive': np.ones(3)})
        tm.assert_frame_equal(result, expected)

    def test_replace_bytes(self, frame_or_series: Callable[[Any], Any]) -> None:
        obj = frame_or_series(['o']).astype('|S')
        expected = obj.copy()
        obj = obj.replace({None: np.nan})
        tm.assert_equal(obj, expected)

    @pytest.mark.parametrize("data, to_replace, value, expected", [
        ([1], [1.0], [0], [0]),
        ([1], [1], [0], [0]),
        ([1.0], [1.0], [0], [0.0]),
        ([1.0], [1], [0], [0.0])
    ])
    @pytest.mark.parametrize("box", [list, tuple, np.array])
    def test_replace_list_with_mixed_type(
        self,
        data: List[Any],
        to_replace: List[Any],
        value: List[Any],
        expected: List[Any],
        box: Callable[[Sequence[Any]], Any],
        frame_or_series: Callable[[List[Any]], Any]
    ) -> None:
        obj = frame_or_series(data)
        expected_obj = frame_or_series(expected)
        result = obj.replace(box(to_replace), value)
        tm.assert_equal(result, expected_obj)

    @pytest.mark.parametrize("val", [2, np.nan, 2.0])
    def test_replace_value_none_dtype_numeric(self, val: Union[int, float]) -> None:
        df = DataFrame({'a': [1, val]})
        result = df.replace(val, None)
        expected = DataFrame({'a': [1, None]}, dtype=object)
        tm.assert_frame_equal(result, expected)
        df = DataFrame({'a': [1, val]})
        result = df.replace({val: None})
        tm.assert_frame_equal(result, expected)

    def test_replace_with_nil_na(self) -> None:
        ser = DataFrame({'a': ['nil', pd.NA]})
        expected = DataFrame({'a': ['anything else', pd.NA]}, index=[0, 1])
        result = ser.replace('nil', 'anything else')
        tm.assert_frame_equal(expected, result)

class TestDataFrameReplaceRegex:
    @pytest.mark.parametrize("data", [{'a': list('ab..'), 'b': list('efgh')}, {'a': list('ab..'), 'b': list(range(4))}])
    @pytest.mark.parametrize("to_replace,value", [('\\s*\\.\\s*', np.nan), ('\\s*(\\.)\\s*', '\\1\\1\\1')])
    @pytest.mark.parametrize("compile_regex", [True, False])
    @pytest.mark.parametrize("regex_kwarg", [True, False])
    @pytest.mark.parametrize("inplace", [True, False])
    def test_regex_replace_scalar(
        self,
        data: Dict[str, List[Any]],
        to_replace: Union[str, Pattern],
        value: Any,
        compile_regex: bool,
        regex_kwarg: bool,
        inplace: bool
    ) -> None:
        df = DataFrame(data)
        expected = df.copy()
        if compile_regex:
            to_replace = re.compile(to_replace)  # type: ignore
        if regex_kwarg:
            regex = to_replace
            to_replace = None
        else:
            regex = True
        result = df.replace(to_replace, value, inplace=inplace, regex=regex)
        if inplace:
            assert result is None
            result = df
        if value is np.nan:
            expected_replace_val = np.nan
        else:
            expected_replace_val = '...'
        expected.loc[expected['a'] == '.', 'a'] = expected_replace_val
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("regex", [False, True])
    @pytest.mark.parametrize("value", [1, '1'])
    def test_replace_regex_dtype_frame(self, regex: bool, value: Union[int, str]) -> None:
        df1 = DataFrame({'A': ['0'], 'B': ['0']})
        dtype = object if value == 1 else None
        expected_df1 = DataFrame({'A': [value], 'B': [value]}, dtype=dtype)
        result_df1 = df1.replace(to_replace='0', value=value, regex=regex)
        tm.assert_frame_equal(result_df1, expected_df1)
        df2 = DataFrame({'A': ['0'], 'B': ['1']})
        if regex:
            expected_df2 = DataFrame({'A': [value], 'B': ['1']}, dtype=dtype)
        else:
            expected_df2 = DataFrame({'A': Series([value], dtype=dtype), 'B': ['1']})
        result_df2 = df2.replace(to_replace='0', value=value, regex=regex)
        tm.assert_frame_equal(result_df2, expected_df2)

    def test_replace_with_value_also_being_replaced(self) -> None:
        df = DataFrame({'A': [0, 1, 2], 'B': [1, 0, 2]})
        result = df.replace({0: 1, 1: np.nan})
        expected = DataFrame({'A': [1, np.nan, 2], 'B': [np.nan, 1, 2]})
        tm.assert_frame_equal(result, expected)

    def test_replace_categorical_no_replacement(self) -> None:
        df = DataFrame({'a': ['one', 'two', None, 'three'], 'b': ['one', None, 'two', 'three']}, dtype='category')
        expected = df.copy()
        result = df.replace(to_replace=['.', 'def'], value=['_', None])
        tm.assert_frame_equal(result, expected)

    def test_replace_object_splitting(self, using_infer_string: bool) -> None:
        df = DataFrame({'a': ['a'], 'b': 'b'})
        if using_infer_string:
            assert len(df._mgr.blocks) == 2
        else:
            assert len(df._mgr.blocks) == 1
        df.replace(to_replace='^\\s*$', value='', inplace=True, regex=True)
        if using_infer_string:
            assert len(df._mgr.blocks) == 2
        else:
            assert len(df._mgr.blocks) == 1