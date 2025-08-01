"""test label based indexing with loc"""
from collections import namedtuple
import contextlib
from datetime import date, datetime, time, timedelta
import re
from dateutil.tz import gettz
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.errors import IndexingError
import pandas as pd
from pandas import Categorical, CategoricalDtype, CategoricalIndex, DataFrame, DatetimeIndex, Index, IndexSlice, MultiIndex, Period, PeriodIndex, Series, SparseDtype, Timedelta, Timestamp, date_range, timedelta_range, to_datetime, to_timedelta
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.core.indexing import _one_ellipsis_message
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises

@pytest.mark.parametrize('series, new_series, expected_ser', [[[np.nan, np.nan, 'b'], ['a', np.nan, np.nan], [False, True, True]], [[np.nan, 'b'], ['a', np.nan], [False, True]]])
def test_not_change_nan_loc(series: list, new_series: list, expected_ser: list) -> None:
    df = DataFrame({'A': series})
    df.loc[:, 'A'] = new_series
    expected = DataFrame({'A': expected_ser})
    tm.assert_frame_equal(df.isna(), expected)
    tm.assert_frame_equal(df.notna(), ~expected)

class TestLoc:

    def test_none_values_on_string_columns(self, using_infer_string: bool) -> None:
        df = DataFrame(['1', '2', None], columns=['a'], dtype=object)
        assert df.loc[2, 'a'] is None
        df = DataFrame(['1', '2', None], columns=['a'], dtype='str')
        if using_infer_string:
            assert np.isnan(df.loc[2, 'a'])
        else:
            assert df.loc[2, 'a'] is None

    def test_loc_getitem_int(self, frame_or_series: type) -> None:
        obj = frame_or_series(range(3), index=Index(list('abc'), dtype=object))
        check_indexing_smoketest_or_raises(obj, 'loc', 2, fails=KeyError)

    def test_loc_getitem_label(self, frame_or_series: type) -> None:
        obj = frame_or_series()
        check_indexing_smoketest_or_raises(obj, 'loc', 'c', fails=KeyError)

    @pytest.mark.parametrize('key', ['f', 20])
    @pytest.mark.parametrize('index', [Index(list('abcd'), dtype=object), Index([2, 4, 'null', 8], dtype=object), date_range('20130101', periods=4), Index(range(0, 8, 2), dtype=np.float64), Index([])])
    def test_loc_getitem_label_out_of_range(self, key: str, index: Index, frame_or_series: type) -> None:
        obj = frame_or_series(range(len(index)), index=index)
        check_indexing_smoketest_or_raises(obj, 'loc', key, fails=KeyError)

    @pytest.mark.parametrize('key', [[0, 1, 2], [1, 3.0, 'A']])
    @pytest.mark.parametrize('dtype', [np.int64, np.uint64, np.float64])
    def test_loc_getitem_label_list(self, key: list, dtype: type, frame_or_series: type) -> None:
        obj = frame_or_series(range(3), index=Index([0, 1, 2], dtype=dtype))
        check_indexing_smoketest_or_raises(obj, 'loc', key, fails=KeyError)

    @pytest.mark.parametrize('index', [None, Index([0, 1, 2], dtype=np.int64), Index([0, 1, 2], dtype=np.uint64), Index([0, 1, 2], dtype=np.float64), MultiIndex.from_arrays([range(3), range(3)])])
    @pytest.mark.parametrize('key', [[0, 1, 2], [0, 2, 10], [3, 6, 7], [(1, 3), (1, 4), (2, 5)]])
    def test_loc_getitem_label_list_with_missing(self, key: list, index: Index, frame_or_series: type) -> None:
        if index is None:
            obj = frame_or_series()
        else:
            obj = frame_or_series(range(len(index)), index=index)
        check_indexing_smoketest_or_raises(obj, 'loc', key, fails=KeyError)

    @pytest.mark.parametrize('dtype', [np.int64, np.uint64])
    def test_loc_getitem_label_list_fails(self, dtype: type, frame_or_series: type) -> None:
        obj = frame_or_series(range(3), Index([0, 1, 2], dtype=dtype))
        check_indexing_smoketest_or_raises(obj, 'loc', [20, 30, 40], axes=1, fails=KeyError)

    def test_loc_getitem_bool(self, frame_or_series: type) -> None:
        obj = frame_or_series()
        b = [True, False, True, False]
        check_indexing_smoketest_or_raises(obj, 'loc', b, fails=IndexError)

    @pytest.mark.parametrize('slc, indexes, axes, fails', [[slice(1, 3), [Index(list('abcd'), dtype=object), Index([2, 4, 'null', 8], dtype=object), None, date_range('20130101', periods=4), Index(range(0, 12, 3), dtype=np.float64)], None, TypeError], [slice('20130102', '20130104'), [date_range('20130101', periods=4)], 1, TypeError], [slice(2, 8), [Index([2, 4, 'null', 8], dtype=object)], 0, TypeError], [slice(2, 8), [Index([2, 4, 'null', 8], dtype=object)], 1, KeyError], [slice(2, 4, 2), [Index([2, 4, 'null', 8], dtype=object)], 0, TypeError]])
    def test_loc_getitem_label_slice(self, slc: slice, indexes: list, axes: int, fails: type, frame_or_series: type) -> None:
        for index in indexes:
            if index is None:
                obj = frame_or_series()
            else:
                obj = frame_or_series(range(len(index)), index=index)
            check_indexing_smoketest_or_raises(obj, 'loc', slc, axes=axes, fails=fails)

    def test_setitem_from_duplicate_axis(self) -> None:
        df = DataFrame([[20, 'a'], [200, 'a'], [200, 'a']], columns=['col1', 'col2'], index=[10, 1, 1])
        df.loc[1, 'col1'] = np.arange(2)
        expected = DataFrame([[20, 'a'], [0, 'a'], [1, 'a']], columns=['col1', 'col2'], index=[10, 1, 1])
        tm.assert_frame_equal(df, expected)

    def test_column_types_consistent(self) -> None:
        df = DataFrame(data={'channel': [1, 2, 3], 'A': ['String 1', np.nan, 'String 2'], 'B': [Timestamp('2019-06-11 11:00:00'), pd.NaT, Timestamp('2019-06-11 12:00:00')]})
        df2 = DataFrame(data={'A': ['String 3'], 'B': [Timestamp('2019-06-11 12:00:00')]})
        df.loc[df['A'].isna(), ['A', 'B']] = df2.values
        expected = DataFrame(data={'channel': [1, 2, 3], 'A': ['String 1', 'String 3', 'String 2'], 'B': [Timestamp('2019-06-11 11:00:00'), Timestamp('2019-06-11 12:00:00'), Timestamp('2019-06-11 12:00:00')]})
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize('obj, key, exp', [(DataFrame([[1]], columns=Index([False])), IndexSlice[:, False], Series([1], name=False)), (Series([1], index=Index([False])), False, [1]), (DataFrame([[1]], index=Index([False])), False, Series([1], name=False))])
    def test_loc_getitem_single_boolean_arg(self, obj: DataFrame, key: IndexSlice, exp: Series) -> None:
        res = obj.loc[key]
        if isinstance(exp, (DataFrame, Series)):
            tm.assert_equal(res, exp)
        else:
            assert res == exp

class TestLocBaseIndependent:

    def test_loc_npstr(self) -> None:
        df = DataFrame(index=date_range('2021', '2022'))
        result = df.loc[np.array(['2021/6/1'])[0]:]
        expected = df.iloc[151:]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('msg, key', [("Period\\('2019', 'Y-DEC'\\), 'foo', 'bar'", (Period(2019), 'foo', 'bar')), ("Period\\('2019', 'Y-DEC'\\), 'y1', 'bar'", (Period(2019), 'y1', 'bar')), ("Period\\('2019', 'Y-DEC'\\), 'foo', 'z1'", (Period(2019), 'foo', 'z1')), ("Period\\('2018', 'Y-DEC'\\), Period\\('2016', 'Y-DEC'\\), 'bar'", (Period(2018), Period(2016), 'bar')), ("Period\\('2018', 'Y-DEC'\\), 'foo', 'y1'", (Period(2018), 'foo', 'y1')), ("Period\\('2017', 'Y-DEC'\\), 'foo', Period\\('2015', 'Y-DEC'\\)", (Period(2017), 'foo', Period(2015))), ("Period\\('2017', 'Y-DEC'\\), 'z1', 'bar'", (Period(2017), 'z1', 'bar'))])
    def test_contains_raise_error_if_period_index_is_in_multi_index(self, msg: str, key: tuple) -> None:
        """
        parse_datetime_string_with_reso return parameter if type not matched.
        PeriodIndex.get_loc takes returned value from parse_datetime_string_with_reso
        as a tuple.
        If first argument is Period and a tuple has 3 items,
        process go on not raise exception
        """
        df = DataFrame({'A': [Period(2019), 'x1', 'x2'], 'B': [Period(2018), Period(2016), 'y1'], 'C': [Period(2017), 'z1', Period(2015)], 'V1': [1, 2, 3], 'V2': [10, 20, 30]}).set_index(['A', 'B', 'C'])
        with pytest.raises(KeyError, match=msg):
            df.loc[key]

    def test_loc_getitem_missing_unicode_key(self) -> None:
        df = DataFrame({'a': [1]})
        with pytest.raises(KeyError, match='א'):
            df.loc[:, 'א']

    def test_loc_getitem_dups(self) -> None:
        df = DataFrame(np.random.default_rng(2).random((20, 5)), index=['ABCDE'[x % 5] for x in range(20)])
        expected = df.loc['A', 0]
        result = df.loc[:, 0].loc['A']
        tm.assert_series_equal(result, expected)

    def test_loc_getitem_dups2(self) -> None:
        df = DataFrame([[1, 2, 'foo', 'bar', Timestamp('20130101')]], columns=['a', 'a', 'a', 'a', 'a'], index=[1])
        expected = Series([1, 2, 'foo', 'bar', Timestamp('20130101')], index=['a', 'a', 'a', 'a', 'a'], name=1)
        result = df.iloc[0]
        tm.assert_series_equal(result, expected)
        result = df.loc[1]
        tm.assert_series_equal(result, expected)

    def test_loc_setitem_dups(self) -> None:
        df_orig = DataFrame({'me': list('rttti'), 'foo': list('aaade'), 'bar': np.arange(5, dtype='float64') * 1.34 + 2, 'bar2': np.arange(5, dtype='float64') * -0.34 + 2}).set_index('me')
        indexer = ('r', ['bar', 'bar2'])
        df = df_orig.copy()
        df.loc[indexer] *= 2.0
        tm.assert_series_equal(df.loc[indexer], 2.0 * df_orig.loc[indexer])
        indexer = ('r', 'bar')
        df = df_orig.copy()
        df.loc[indexer] *= 2.0
        assert df.loc[indexer] == 2.0 * df_orig.loc[indexer]
        indexer = ('t', ['bar', 'bar2'])
        df = df_orig.copy()
        df.loc[indexer] *= 2.0
        tm.assert_frame_equal(df.loc[indexer], 2.0 * df_orig.loc[indexer])

    def test_loc_setitem_slice(self) -> None:
        df1 = DataFrame({'a': [0, 1, 1], 'b': Series([100, 200, 300], dtype='uint32')})
        ix = df1['a'] == 1
        newb1 = df1.loc[ix, 'b'] + 1
        df1.loc[ix, 'b'] = newb1
        expected = DataFrame({'a': [0, 1, 1], 'b': Series([100, 201, 301], dtype='uint32')})
        tm.assert_frame_equal(df1, expected)
        df2 = DataFrame({'a': [0, 1, 1], 'b': [100, 200, 300]}, dtype='uint64')
        ix = df1['a'] == 1
        newb2 = df2.loc[ix, 'b']
        with pytest.raises(TypeError, match='Invalid value'):
            df1.loc[ix, 'b'] = newb2

    def test_loc_setitem_dtype(self) -> None:
        df = DataFrame({'id': ['A'], 'a': [1.2], 'b': [0.0], 'c': [-2.5]})
        cols = ['a', 'b', 'c']
        df.loc[:, cols] = df.loc[:, cols].astype('float32')
        expected = DataFrame({'id': ['A'], 'a': np.array([1.2], dtype='float64'), 'b': np.array([0.0], dtype='float64'), 'c': np.array([-2.5], dtype='float64')})
        tm.assert_frame_equal(df, expected)

    def test_getitem_label_list_with_missing(self) -> None:
        s = Series(range(3), index=['a', 'b', 'c'])
        with pytest.raises(KeyError, match='not in index'):
            s[['a', 'd']]
        s = Series(range(3))
        with pytest.raises(KeyError, match='not in index'):
            s[[0, 3]]

    @pytest.mark.parametrize('index', [[True, False], [True, False, True, False]])
    def test_loc_getitem_bool_diff_len(self, index: list) -> None:
        s = Series([1, 2, 3])
        msg = f'Boolean index has wrong length: {len(index)} instead of {len(s)}'
        with pytest.raises(IndexError, match=msg):
            s.loc[index]

    def test_loc_getitem_int_slice(self) -> None:
        pass

    def test_loc_to_fail(self) -> None:
        df = DataFrame(np.random.default_rng(2).random((3, 3)), index=['a', 'b', 'c'], columns=['e', 'f', 'g'])
        msg = f'''\\"None of \\[Index\\(\\[1, 2\\], dtype='{np.dtype(int)}'\\)\\] are in the \\[index\\]\\"'''
        with pytest.raises(KeyError, match=msg):
            df.loc[[1, 2], [1, 2]]

    def test_loc_to_fail2(self) -> None:
        s = Series(dtype=object)
        s.loc[1] = 1
        s.loc['a'] = 2
        with pytest.raises(KeyError, match='^-1$'):
            s.loc[-1]
        msg = f'''\\"None of \\[Index\\(\\[-1, -2\\], dtype='{np.dtype(int)}'\\)\\] are in the \\[index\\]\\"'''
        with pytest.raises(KeyError, match=msg):
            s.loc[[-1, -2]]
        msg = '\\"None of \\[Index\\(\\[\'4\'\\], dtype=\'object\'\\)\\] are in the \\[index\\]\\"'
        with pytest.raises(KeyError, match=msg):
            s.loc[Index(['4'], dtype=object