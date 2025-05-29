"""test label based indexing with loc"""
from collections import namedtuple
import contextlib
from datetime import date, datetime, time, timedelta
import re
from typing import List, Union, Tuple, Callable, Any
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


@pytest.mark.parametrize('series, new_series, expected_ser', [[[np.nan, np.
    nan, 'b'], ['a', np.nan, np.nan], [False, True, True]], [[np.nan, 'b'],
    ['a', np.nan], [False, True]]])
def test_not_change_nan_loc(series, new_series, expected_ser):
    df = DataFrame({'A': series})
    df.loc[:, 'A'] = new_series
    expected = DataFrame({'A': expected_ser})
    tm.assert_frame_equal(df.isna(), expected)
    tm.assert_frame_equal(df.notna(), ~expected)


class TestLoc:

    def test_none_values_on_string_columns(self, using_infer_string):
        df = DataFrame(['1', '2', None], columns=['a'], dtype=object)
        assert df.loc[2, 'a'] is None
        df = DataFrame(['1', '2', None], columns=['a'], dtype='str')
        if using_infer_string:
            assert np.isnan(df.loc[2, 'a'])
        else:
            assert df.loc[2, 'a'] is None

    def test_loc_getitem_int(self, frame_or_series):
        obj = frame_or_series(range(3), index=Index(list('abc'), dtype=object))
        check_indexing_smoketest_or_raises(obj, 'loc', 2, fails=KeyError)

    def test_loc_getitem_label(self, frame_or_series):
        obj = frame_or_series()
        check_indexing_smoketest_or_raises(obj, 'loc', 'c', fails=KeyError)

    @pytest.mark.parametrize('key', ['f', 20])
    @pytest.mark.parametrize('index', [Index(list('abcd'), dtype=object),
        Index([2, 4, 'null', 8], dtype=object), date_range('20130101',
        periods=4), Index(range(0, 8, 2), dtype=np.float64), Index([])])
    def test_loc_getitem_label_out_of_range(self, key, index, frame_or_series):
        obj = frame_or_series(range(len(index)), index=index)
        check_indexing_smoketest_or_raises(obj, 'loc', key, fails=KeyError)

    @pytest.mark.parametrize('key', [[0, 1, 2], [1, 3.0, 'A']])
    @pytest.mark.parametrize('dtype', [np.int64, np.uint64, np.float64])
    def test_loc_getitem_label_list(self, key, dtype, frame_or_series):
        obj = frame_or_series(range(3), index=Index([0, 1, 2], dtype=dtype))
        check_indexing_smoketest_or_raises(obj, 'loc', key, fails=KeyError)

    @pytest.mark.parametrize('index', [None, Index([0, 1, 2], dtype=np.
        int64), Index([0, 1, 2], dtype=np.uint64), Index([0, 1, 2], dtype=
        np.float64), MultiIndex.from_arrays([range(3), range(3)])])
    @pytest.mark.parametrize('key', [[0, 1, 2], [0, 2, 10], [3, 6, 7], [(1,
        3), (1, 4), (2, 5)]])
    def test_loc_getitem_label_list_with_missing(self, key, index,
        frame_or_series):
        if index is None:
            obj = frame_or_series()
        else:
            obj = frame_or_series(range(len(index)), index=index)
        check_indexing_smoketest_or_raises(obj, 'loc', key, fails=KeyError)

    @pytest.mark.parametrize('dtype', [np.int64, np.uint64])
    def test_loc_getitem_label_list_fails(self, dtype, frame_or_series):
        obj = frame_or_series(range(3), Index([0, 1, 2], dtype=dtype))
        check_indexing_smoketest_or_raises(obj, 'loc', [20, 30, 40], axes=1,
            fails=KeyError)

    def test_loc_getitem_bool(self, frame_or_series):
        obj = frame_or_series()
        b = [True, False, True, False]
        check_indexing_smoketest_or_raises(obj, 'loc', b, fails=IndexError)

    @pytest.mark.parametrize('slc, indexes, axes, fails', [[slice(1, 3), [
        Index(list('abcd'), dtype=object), Index([2, 4, 'null', 8], dtype=
        object), None, date_range('20130101', periods=4), Index(range(0, 12,
        3), dtype=np.float64)], None, TypeError], [slice('20130102',
        '20130104'), [date_range('20130101', periods=4)], 1, TypeError], [
        slice(2, 8), [Index([2, 4, 'null', 8], dtype=object)], 0, TypeError
        ], [slice(2, 8), [Index([2, 4, 'null', 8], dtype=object)], 1,
        KeyError], [slice(2, 4, 2), [Index([2, 4, 'null', 8], dtype=object)
        ], 0, TypeError]])
    def test_loc_getitem_label_slice(self, slc, indexes, axes, fails,
        frame_or_series):
        for index in indexes:
            if index is None:
                obj = frame_or_series()
            else:
                obj = frame_or_series(range(len(index)), index=index)
            check_indexing_smoketest_or_raises(obj, 'loc', slc, axes=axes,
                fails=fails)

    def test_setitem_from_duplicate_axis(self):
        df = DataFrame([[20, 'a'], [200, 'a'], [200, 'a']], columns=['col1',
            'col2'], index=[10, 1, 1])
        df.loc[1, 'col1'] = np.arange(2)
        expected = DataFrame([[20, 'a'], [0, 'a'], [1, 'a']], columns=[
            'col1', 'col2'], index=[10, 1, 1])
        tm.assert_frame_equal(df, expected)

    def test_column_types_consistent(self):
        df = DataFrame(data={'channel': [1, 2, 3], 'A': ['String 1', np.nan,
            'String 2'], 'B': [Timestamp('2019-06-11 11:00:00'), pd.NaT,
            Timestamp('2019-06-11 12:00:00')]})
        df2 = DataFrame(data={'A': ['String 3'], 'B': [Timestamp(
            '2019-06-11 12:00:00')]})
        df.loc[df['A'].isna(), ['A', 'B']] = df2.values
        expected = DataFrame(data={'channel': [1, 2, 3], 'A': ['String 1',
            'String 3', 'String 2'], 'B': [Timestamp('2019-06-11 11:00:00'),
            Timestamp('2019-06-11 12:00:00'), Timestamp(
            '2019-06-11 12:00:00')]})
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize('obj, key, exp', [(DataFrame([[1]], columns=
        Index([False])), IndexSlice[:, False], Series([1], name=False)), (
        Series([1], index=Index([False])), False, [1]), (DataFrame([[1]],
        index=Index([False])), False, Series([1], name=False))])
    def test_loc_getitem_single_boolean_arg(self, obj, key, exp):
        res = obj.loc[key]
        if isinstance(exp, (DataFrame, Series)):
            tm.assert_equal(res, exp)
        else:
            assert res == exp


class TestLocBaseIndependent:

    def test_loc_npstr(self):
        df = DataFrame(index=date_range('2021', '2022'))
        result = df.loc[np.array(['2021/6/1'])[0]:]
        expected = df.iloc[151:]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('msg, key', [(
        "Period\\('2019', 'Y-DEC'\\), 'foo', 'bar'", (Period(2019), 'foo',
        'bar')), ("Period\\('2019', 'Y-DEC'\\), 'y1', 'bar'", (Period(2019),
        'y1', 'bar')), ("Period\\('2019', 'Y-DEC'\\), 'foo', 'z1'", (Period
        (2019), 'foo', 'z1')), (
        "Period\\('2018', 'Y-DEC'\\), Period\\('2016', 'Y-DEC'\\), 'bar'",
        (Period(2018), Period(2016), 'bar')), (
        "Period\\('2018', 'Y-DEC'\\), 'foo', 'y1'", (Period(2018), 'foo',
        'y1')), (
        "Period\\('2017', 'Y-DEC'\\), 'foo', Period\\('2015', 'Y-DEC'\\)",
        (Period(2017), 'foo', Period(2015))), (
        "Period\\('2017', 'Y-DEC'\\), 'z1', 'bar'", (Period(2017), 'z1',
        'bar'))])
    def test_contains_raise_error_if_period_index_is_in_multi_index(self,
        msg, key):
        """
        parse_datetime_string_with_reso return parameter if type not matched.
        PeriodIndex.get_loc takes returned value from parse_datetime_string_with_reso
        as a tuple.
        If first argument is Period and a tuple has 3 items,
        process go on not raise exception
        """
        df = DataFrame({'A': [Period(2019), 'x1', 'x2'], 'B': [Period(2018),
            Period(2016), 'y1'], 'C': [Period(2017), 'z1', Period(2015)],
            'V1': [1, 2, 3], 'V2': [10, 20, 30]}).set_index(['A', 'B', 'C'])
        with pytest.raises(KeyError, match=msg):
            df.loc[key]

    def test_loc_getitem_missing_unicode_key(self):
        df = DataFrame({'a': [1]})
        with pytest.raises(KeyError, match='א'):
            df.loc[:, 'א']

    def test_loc_getitem_dups(self):
        df = DataFrame(np.random.default_rng(2).random((20, 5)), index=[
            'ABCDE'[x % 5] for x in range(20)])
        expected = df.loc['A', 0]
        result = df.loc[:, 0].loc['A']
        tm.assert_series_equal(result, expected)

    def test_loc_getitem_dups2(self):
        df = DataFrame([[1, 2, 'foo', 'bar', Timestamp('20130101')]],
            columns=['a', 'a', 'a', 'a', 'a'], index=[1])
        expected = Series([1, 2, 'foo', 'bar', Timestamp('20130101')],
            index=['a', 'a', 'a', 'a', 'a'], name=1)
        result = df.iloc[0]
        tm.assert_series_equal(result, expected)
        result = df.loc[1]
        tm.assert_series_equal(result, expected)

    def test_loc_setitem_dups(self):
        df_orig = DataFrame({'me': list('rttti'), 'foo': list('aaade'),
            'bar': np.arange(5, dtype='float64') * 1.34 + 2, 'bar2': np.
            arange(5, dtype='float64') * -0.34 + 2}).set_index('me')
        indexer = 'r', ['bar', 'bar2']
        df = df_orig.copy()
        df.loc[indexer] *= 2.0
        tm.assert_series_equal(df.loc[indexer], 2.0 * df_orig.loc[indexer])
        indexer = 'r', 'bar'
        df = df_orig.copy()
        df.loc[indexer] *= 2.0
        assert df.loc[indexer] == 2.0 * df_orig.loc[indexer]
        indexer = 't', ['bar', 'bar2']
        df = df_orig.copy()
        df.loc[indexer] *= 2.0
        tm.assert_frame_equal(df.loc[indexer], 2.0 * df_orig.loc[indexer])

    def test_loc_setitem_slice(self):
        df1 = DataFrame({'a': [0, 1, 1], 'b': Series([100, 200, 300], dtype
            ='uint32')})
        ix = df1['a'] == 1
        newb1 = df1.loc[ix, 'b'] + 1
        df1.loc[ix, 'b'] = newb1
        expected = DataFrame({'a': [0, 1, 1], 'b': Series([100, 201, 301],
            dtype='uint32')})
        tm.assert_frame_equal(df1, expected)
        df2 = DataFrame({'a': [0, 1, 1], 'b': [100, 200, 300]}, dtype='uint64')
        ix = df1['a'] == 1
        newb2 = df2.loc[ix, 'b']
        with pytest.raises(TypeError, match='Invalid value'):
            df1.loc[ix, 'b'] = newb2

    def test_loc_setitem_dtype(self):
        df = DataFrame({'id': ['A'], 'a': [1.2], 'b': [0.0], 'c': [-2.5]})
        cols = ['a', 'b', 'c']
        df.loc[:, cols] = df.loc[:, cols].astype('float32')
        expected = DataFrame({'id': ['A'], 'a': np.array([1.2], dtype=
            'float64'), 'b': np.array([0.0], dtype='float64'), 'c': np.
            array([-2.5], dtype='float64')})
        tm.assert_frame_equal(df, expected)

    def test_getitem_label_list_with_missing(self):
        s = Series(range(3), index=['a', 'b', 'c'])
        with pytest.raises(KeyError, match='not in index'):
            s[['a', 'd']]
        s = Series(range(3))
        with pytest.raises(KeyError, match='not in index'):
            s[[0, 3]]

    @pytest.mark.parametrize('index', [[True, False], [True, False, True, 
        False]])
    def test_loc_getitem_bool_diff_len(self, index):
        s = Series([1, 2, 3])
        msg = (
            f'Boolean index has wrong length: {len(index)} instead of {len(s)}'
            )
        with pytest.raises(IndexError, match=msg):
            s.loc[index]

    def test_loc_getitem_int_slice(self):
        pass

    def test_loc_to_fail(self):
        df = DataFrame(np.random.default_rng(2).random((3, 3)), index=['a',
            'b', 'c'], columns=['e', 'f', 'g'])
        msg = (
            f'\\"None of \\[Index\\(\\[1, 2\\], dtype=\'{np.dtype(int)}\'\\)\\] are in the \\[index\\]\\"'
            )
        with pytest.raises(KeyError, match=msg):
            df.loc[[1, 2], [1, 2]]

    def test_loc_to_fail2(self):
        s = Series(dtype=object)
        s.loc[1] = 1
        s.loc['a'] = 2
        with pytest.raises(KeyError, match='^-1$'):
            s.loc[-1]
        msg = (
            f'\\"None of \\[Index\\(\\[-1, -2\\], dtype=\'{np.dtype(int)}\'\\)\\] are in the \\[index\\]\\"'
            )
        with pytest.raises(KeyError, match=msg):
            s.loc[[-1, -2]]
        msg = (
            '\\"None of \\[Index\\(\\[\'4\'\\], dtype=\'object\'\\)\\] are in the \\[index\\]\\"'
            )
        with pytest.raises(KeyError, match=msg):
            s.loc[Index(['4'], dtype=object)]
        s.loc[-1] = 3
        with pytest.raises(KeyError, match='not in index'):
            s.loc[[-1, -2]]
        s['a'] = 2
        msg = (
            f'\\"None of \\[Index\\(\\[-2\\], dtype=\'{np.dtype(int)}\'\\)\\] are in the \\[index\\]\\"'
            )
        with pytest.raises(KeyError, match=msg):
            s.loc[[-2]]
        del s['a']
        with pytest.raises(KeyError, match=msg):
            s.loc[[-2]] = 0

    def test_loc_to_fail3(self):
        df = DataFrame([['a'], ['b']], index=[1, 2], columns=['value'])
        msg = (
            f'\\"None of \\[Index\\(\\[3\\], dtype=\'{np.dtype(int)}\'\\)\\] are in the \\[index\\]\\"'
            )
        with pytest.raises(KeyError, match=msg):
            df.loc[[3], :]
        with pytest.raises(KeyError, match=msg):
            df.loc[[3]]

    def test_loc_getitem_list_with_fail(self):
        s = Series([1, 2, 3])
        s.loc[[2]]
        msg = (
            'None of [RangeIndex(start=3, stop=4, step=1)] are in the [index]')
        with pytest.raises(KeyError, match=re.escape(msg)):
            s.loc[[3]]
        with pytest.raises(KeyError, match='not in index'):
            s.loc[[2, 3]]

    def test_loc_index(self):
        df = DataFrame(np.random.default_rng(2).random(size=(5, 10)), index
            =['alpha_0', 'alpha_1', 'alpha_2', 'beta_0', 'beta_1'])
        mask = df.index.map(lambda x: 'alpha' in x)
        expected = df.loc[np.array(mask)]
        result = df.loc[mask]
        tm.assert_frame_equal(result, expected)
        result = df.loc[mask.values]
        tm.assert_frame_equal(result, expected)
        result = df.loc[pd.array(mask, dtype='boolean')]
        tm.assert_frame_equal(result, expected)

    def test_loc_general(self):
        df = DataFrame(np.random.default_rng(2).random((4, 4)), columns=[
            'A', 'B', 'C', 'D'], index=['A', 'B', 'C', 'D'])
        result = df.loc[:, 'A':'B'].iloc[0:2, :]
        assert (result.columns == ['A', 'B']).all()
        assert (result.index == ['A', 'B']).all()
        result = DataFrame({'a': [Timestamp('20130101')], 'b': [1]}).iloc[0]
        expected = Series([Timestamp('20130101'), 1], index=['a', 'b'], name=0)
        tm.assert_series_equal(result, expected)
        assert result.dtype == object

    @pytest.fixture
    def frame_for_consistency(self):
        return DataFrame({'date': date_range('2000-01-01', '2000-01-5'),
            'val': Series(range(5), dtype=np.int64)})

    @pytest.mark.parametrize('val', [0, np.array(0, dtype=np.int64), np.
        array([0, 0, 0, 0, 0], dtype=np.int64)])
    def test_loc_setitem_consistency(self, frame_for_consistency, val):
        df = frame_for_consistency.copy()
        with pytest.raises(TypeError, match='Invalid value'):
            df.loc[:, 'date'] = val

    def test_loc_setitem_consistency_dt64_to_str(self, frame_for_consistency):
        df = frame_for_consistency.copy()
        with pytest.raises(TypeError, match='Invalid value'):
            df.loc[:, 'date'] = 'foo'

    def test_loc_setitem_consistency_dt64_to_float(self, frame_for_consistency
        ):
        df = frame_for_consistency.copy()
        with pytest.raises(TypeError, match='Invalid value'):
            df.loc[:, 'date'] = 1.0

    def test_loc_setitem_consistency_single_row(self):
        df = DataFrame({'date': Series([Timestamp('20180101')])})
        with pytest.raises(TypeError, match='Invalid value'):
            df.loc[:, 'date'] = 'string'

    def test_loc_setitem_consistency_empty(self):
        expected = DataFrame(columns=['x', 'y'])
        df = DataFrame(columns=['x', 'y'])
        with tm.assert_produces_warning(None):
            df.loc[:, 'x'] = 1
        tm.assert_frame_equal(df, expected)
        df = DataFrame(columns=['x', 'y'])
        df['x'] = 1
        expected['x'] = expected['x'].astype(np.int64)
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_consistency_slice_column_len(self, using_infer_string
        ):
        levels = [['Region_1'] * 4, ['Site_1', 'Site_1', 'Site_2', 'Site_2'
            ], [3987227376, 3980680971, 3977723249, 3977723089]]
        mi = MultiIndex.from_arrays(levels, names=['Region', 'Site',
            'RespondentID'])
        clevels = [['Respondent', 'Respondent', 'Respondent', 'OtherCat',
            'OtherCat'], ['Something', 'StartDate', 'EndDate', 'Yes/No',
            'SomethingElse']]
        cols = MultiIndex.from_arrays(clevels, names=['Level_0', 'Level_1'])
        values = [['A', '5/25/2015 10:59', '5/25/2015 11:22', 'Yes', np.nan
            ], ['A', '5/21/2015 9:40', '5/21/2015 9:52', 'Yes', 'Yes'], [
            'A', '5/20/2015 8:27', '5/20/2015 8:41', 'Yes', np.nan], ['A',
            '5/20/2015 8:33', '5/20/2015 9:09', 'Yes', 'No']]
        df = DataFrame(values, index=mi, columns=cols)
        ctx = contextlib.nullcontext()
        if using_infer_string:
            ctx = pytest.raises(TypeError, match='Invalid value')
        with ctx:
            df.loc[:, ('Respondent', 'StartDate')] = to_datetime(df.loc[:,
                ('Respondent', 'StartDate')])
        with ctx:
            df.loc[:, ('Respondent', 'EndDate')] = to_datetime(df.loc[:, (
                'Respondent', 'EndDate')])
        if using_infer_string:
            return
        df = df.infer_objects()
        df.loc[:, ('Respondent', 'Duration')] = df.loc[:, ('Respondent',
            'EndDate')] - df.loc[:, ('Respondent', 'StartDate')]
        with pytest.raises(TypeError, match='Invalid value'):
            df.loc[:, ('Respondent', 'Duration')] = df.loc[:, ('Respondent',
                'Duration')] / Timedelta(60000000000)

    @pytest.mark.parametrize('unit', ['Y', 'M', 'D', 'h', 'm', 's', 'ms', 'us']
        )
    def test_loc_assign_non_ns_datetime(self, unit):
        df = DataFrame({'timestamp': [np.datetime64('2017-02-11 12:41:29'),
            np.datetime64('1991-11-07 04:22:37')]})
        df.loc[:, unit] = df.loc[:, 'timestamp'].values.astype(
            f'datetime64[{unit}]')
        df['expected'] = df.loc[:, 'timestamp'].values.astype(
            f'datetime64[{unit}]')
        expected = Series(df.loc[:, 'expected'], name=unit)
        tm.assert_series_equal(df.loc[:, unit], expected)

    def test_loc_modify_datetime(self):
        df = DataFrame.from_dict({'date': [1485264372711, 1485265925110, 
            1540215845888, 1540282121025]})
        df['date_dt'] = to_datetime(df['date'], unit='ms', cache=True
            ).dt.as_unit('ms')
        df.loc[:, 'date_dt_cp'] = df.loc[:, 'date_dt']
        df.loc[[2, 3], 'date_dt_cp'] = df.loc[[2, 3], 'date_dt']
        expected = DataFrame([[1485264372711, '2017-01-24 13:26:12.711',
            '2017-01-24 13:26:12.711'], [1485265925110,
            '2017-01-24 13:52:05.110', '2017-01-24 13:52:05.110'], [
            1540215845888, '2018-10-22 13:44:05.888',
            '2018-10-22 13:44:05.888'], [1540282121025,
            '2018-10-23 08:08:41.025', '2018-10-23 08:08:41.025']], columns
            =['date', 'date_dt', 'date_dt_cp'])
        columns = ['date_dt', 'date_dt_cp']
        expected[columns] = expected[columns].apply(to_datetime)
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame_with_reindex(self):
        df = DataFrame(index=[3, 5, 4], columns=['A'], dtype=float)
        df.loc[[4, 3, 5], 'A'] = np.array([1, 2, 3], dtype='int64')
        ser = Series([2, 3, 1], index=[3, 5, 4], dtype=float)
        expected = DataFrame({'A': ser})
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame_with_reindex_mixed(self):
        df = DataFrame(index=[3, 5, 4], columns=['A', 'B'], dtype=float)
        df['B'] = 'string'
        df.loc[[4, 3, 5], 'A'] = np.array([1, 2, 3], dtype='int64')
        ser = Series([2, 3, 1], index=[3, 5, 4], dtype='int64')
        expected = DataFrame({'A': ser.astype(float)})
        expected['B'] = 'string'
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame_with_inverted_slice(self):
        df = DataFrame(index=[1, 2, 3], columns=['A', 'B'], dtype=float)
        df['B'] = 'string'
        df.loc[slice(3, 0, -1), 'A'] = np.array([1, 2, 3], dtype='int64')
        expected = DataFrame({'A': [3.0, 2.0, 1.0], 'B': 'string'}, index=[
            1, 2, 3])
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_empty_frame(self):
        keys1 = [('@' + str(i)) for i in range(5)]
        val1 = np.arange(5, dtype='int64')
        keys2 = [('@' + str(i)) for i in range(4)]
        val2 = np.arange(4, dtype='int64')
        index = list(set(keys1).union(keys2))
        df = DataFrame(index=index)
        df['A'] = np.nan
        df.loc[keys1, 'A'] = val1
        df['B'] = np.nan
        df.loc[keys2, 'B'] = val2
        sera = Series(val1, index=keys1, dtype=np.float64)
        serb = Series(val2, index=keys2)
        expected = DataFrame({'A': sera, 'B': serb}, columns=Index(['A',
            'B'], dtype=object)).reindex(index=index)
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((4, 4)),
            index=list('abcd'), columns=list('ABCD'))
        result = df.iloc[0, 0]
        df.loc['a', 'A'] = 1
        result = df.loc['a', 'A']
        assert result == 1
        result = df.iloc[0, 0]
        assert result == 1
        df.loc[:, 'B':'D'] = 0
        expected = df.loc[:, 'B':'D']
        result = df.iloc[:, 1:]
        tm.assert_frame_equal(result, expected)

    def test_loc_setitem_frame_nan_int_coercion_invalid(self):
        df = DataFrame({'A': [1, 2, 3], 'B': np.nan})
        df.loc[df.B > df.A, 'B'] = df.A
        expected = DataFrame({'A': [1, 2, 3], 'B': np.nan})
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame_mixed_labels(self):
        df = DataFrame({(1): [1, 2], (2): [3, 4], 'a': ['a', 'b']})
        result = df.loc[0, [1, 2]]
        expected = Series([1, 3], index=Index([1, 2], dtype=object), dtype=
            object, name=0)
        tm.assert_series_equal(result, expected)
        expected = DataFrame({(1): [5, 2], (2): [6, 4], 'a': ['a', 'b']})
        df.loc[0, [1, 2]] = [5, 6]
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame_multiples(self):
        df = DataFrame({'A': ['foo', 'bar', 'baz'], 'B': Series(range(3),
            dtype=np.int64)})
        rhs = df.loc[1:2]
        rhs.index = df.index[0:2]
        df.loc[0:1] = rhs
        expected = DataFrame({'A': ['bar', 'baz', 'baz'], 'B': Series([1, 2,
            2], dtype=np.int64)})
        tm.assert_frame_equal(df, expected)
        df = DataFrame({'date': date_range('2000-01-01', '2000-01-5'),
            'val': Series(range(5), dtype=np.int64)})
        expected = DataFrame({'date': [Timestamp('20000101'), Timestamp(
            '20000102'), Timestamp('20000101'), Timestamp('20000102'),
            Timestamp('20000103')], 'val': Series([0, 1, 0, 1, 2], dtype=np
            .int64)})
        expected['date'] = expected['date'].astype('M8[ns]')
        rhs = df.loc[0:2]
        rhs.index = df.index[2:5]
        df.loc[2:4] = rhs
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize('indexer', [['A'], slice(None, 'A', None), np.
        array(['A'])])
    @pytest.mark.parametrize('value', [['Z'], np.array(['Z'])])
    def test_loc_setitem_with_scalar_index(self, indexer, value):
        df = DataFrame([[1, 2], [3, 4]], columns=['A', 'B']).astype({'A':
            object})
        df.loc[0, indexer] = value
        result = df.loc[0, 'A']
        assert is_scalar(result) and result == 'Z'

    @pytest.mark.parametrize('indexer, expected', [(([0, 2], ['A', 'B', 'C',
        'D']), DataFrame([[7, 8, 9, 7], [3, 4, np.nan, np.nan], [7, 8, 9, 7
        ]], columns=['A', 'B', 'C', 'D'])), ((1, ['C', 'D']), DataFrame([[1,
        2, np.nan, np.nan], [3, 4, 7, 8], [5, 6, np.nan, np.nan]], columns=
        ['A', 'B', 'C', 'D'])), ((1, ['A', 'B', 'C']), DataFrame([[1, 2, np
        .nan], [7, 8, 9], [5, 6, np.nan]], columns=['A', 'B', 'C'])), ((
        slice(1, 3, None), ['B', 'C', 'D']), DataFrame([[1, 2, np.nan, np.
        nan], [3, 7, 8, 9], [5, 10, 11, 12]], columns=['A', 'B', 'C', 'D'])
        ), ((slice(1, 3, None), ['C', 'A', 'D']), DataFrame([[1, 2, np.nan,
        np.nan], [8, 4, 7, 9], [11, 6, 10, 12]], columns=['A', 'B', 'C',
        'D'])), ((slice(None, None, None), ['A', 'C']), DataFrame([[7, 2, 8
        ], [9, 4, 10], [11, 6, 12]], columns=['A', 'B', 'C']))])
    def test_loc_setitem_missing_columns(self, indexer, box, expected):
        df = DataFrame([[1, 2], [3, 4], [5, 6]], columns=['A', 'B'])
        df.loc[indexer] = box
        tm.assert_frame_equal(df, expected)

    def test_loc_coercion(self, frame_or_series):
        df = DataFrame({'date': [Timestamp('20130101').tz_localize('UTC'),
            pd.NaT]})
        expected = df.dtypes
        result = df.iloc[[0]]
        tm.assert_series_equal(result.dtypes, expected)
        result = df.iloc[[1]]
        tm.assert_series_equal(result.dtypes, expected)

    def test_loc_coercion2(self, frame_or_series):
        df = DataFrame({'date': [datetime(2012, 1, 1), datetime(1012, 1, 2)]})
        expected = df.dtypes
        result = df.iloc[[0]]
        tm.assert_series_equal(result.dtypes, expected)
        result = df.iloc[[1]]
        tm.assert_series_equal(result.dtypes, expected)

    def test_loc_coercion3(self, frame_or_series):
        df = DataFrame({'text': ['some words'] + [None] * 9})
        expected = df.dtypes
        result = df.iloc[0:2]
        tm.assert_series_equal(result.dtypes, expected)
        result = df.iloc[3:]
        tm.assert_series_equal(result.dtypes, expected)

    def test_setitem_new_key_tz(self, indexer_sl):
        vals = [to_datetime(42).tz_localize('UTC'), to_datetime(666).
            tz_localize('UTC')]
        expected = Series(vals, index=Index(['foo', 'bar'], dtype=object))
        ser = Series(dtype=object)
        indexer_sl(ser)['foo'] = vals[0]
        indexer_sl(ser)['bar'] = vals[1]
        tm.assert_series_equal(ser, expected)

    def test_loc_non_unique(self):
        df = DataFrame({'A': [1, 2, 3, 4, 5, 6], 'B': [3, 4, 5, 6, 7, 8]},
            index=[0, 1, 0, 1, 2, 3])
        msg = "'Cannot get left slice bound for non-unique label: 1'"
        with pytest.raises(KeyError, match=msg):
            df.loc[1:]
        msg = "'Cannot get left slice bound for non-unique label: 0'"
        with pytest.raises(KeyError, match=msg):
            df.loc[0:]
        msg = "'Cannot get left slice bound for non-unique label: 1'"
        with pytest.raises(KeyError, match=msg):
            df.loc[1:2]
        df = DataFrame({'A': [1, 2, 3, 4, 5, 6], 'B': [3, 4, 5, 6, 7, 8]},
            index=[0, 1, 0, 1, 2, 3]).sort_index(axis=0)
        result = df.loc[1:]
        expected = DataFrame({'A': [2, 4, 5, 6], 'B': [4, 6, 7, 8]}, index=
            [1, 1, 2, 3])
        tm.assert_frame_equal(result, expected)
        result = df.loc[0:]
        tm.assert_frame_equal(result, df)
        result = df.loc[1:2]
        expected = DataFrame({'A': [2, 4, 5], 'B': [4, 6, 7]}, index=[1, 1, 2])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.arm_slow
    @pytest.mark.parametrize('length, l2', [[900, 100], [900000, 100000]])
    def test_loc_non_unique_memory_error(self, length, l2):
        columns = list('ABCDEFG')
        df = pd.concat([DataFrame(np.random.default_rng(2).standard_normal(
            (length, len(columns))), index=np.arange(length), columns=
            columns), DataFrame(np.ones((l2, len(columns))), index=[0] * l2,
            columns=columns)])
        assert df.index.is_unique is False
        mask = np.arange(l2)
        result = df.loc[mask]
        expected = pd.concat([df.take([0]), DataFrame(np.ones((len(mask),
            len(columns))), index=[0] * len(mask), columns=columns), df.
            take(mask[1:])])
        tm.assert_frame_equal(result, expected)

    def test_loc_name(self):
        df = DataFrame([[1, 1], [1, 1]])
        df.index.name = 'index_name'
        result = df.iloc[[0, 1]].index.name
        assert result == 'index_name'
        result = df.loc[[0, 1]].index.name
        assert result == 'index_name'

    def test_loc_empty_list_indexer_is_ok(self):
        df = DataFrame(np.ones((5, 2)), index=Index([f'i-{i}' for i in
            range(5)], name='a'), columns=Index([f'i-{i}' for i in range(2)
            ], name='a'))
        tm.assert_frame_equal(df.loc[:, []], df.iloc[:, :0],
            check_index_type=True, check_column_type=True)
        tm.assert_frame_equal(df.loc[[], :], df.iloc[:0, :],
            check_index_type=True, check_column_type=True)
        tm.assert_frame_equal(df.loc[[]], df.iloc[:0, :], check_index_type=
            True, check_column_type=True)

    def test_identity_slice_returns_new_object(self):
        original_df = DataFrame({'a': [1, 2, 3]})
        sliced_df = original_df.loc[:]
        assert sliced_df is not original_df
        assert original_df[:] is not original_df
        assert original_df.loc[:, :] is not original_df
        assert np.shares_memory(original_df['a']._values, sliced_df['a'].
            _values)
        original_df.loc[:, 'a'] = [4, 4, 4]
        assert (sliced_df['a'] == [1, 2, 3]).all()
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
        assert df[0] is not df.loc[:, 0]
        original_series = Series([1, 2, 3, 4, 5, 6])
        sliced_series = original_series.loc[:]
        assert sliced_series is not original_series
        assert original_series[:] is not original_series
        original_series[:3] = [7, 8, 9]
        assert all(sliced_series[:3] == [1, 2, 3])

    def test_loc_copy_vs_view(self, request):
        x = DataFrame(zip(range(3), range(3)), columns=['a', 'b'])
        y = x.copy()
        q = y.loc[:, 'a']
        q += 2
        tm.assert_frame_equal(x, y)
        z = x.copy()
        q = z.loc[x.index, 'a']
        q += 2
        tm.assert_frame_equal(x, z)

    def test_loc_uint64(self):
        umax = np.iinfo('uint64').max
        ser = Series([1, 2], index=[umax - 1, umax])
        result = ser.loc[umax - 1]
        expected = ser.iloc[0]
        assert result == expected
        result = ser.loc[[umax - 1]]
        expected = ser.iloc[[0]]
        tm.assert_series_equal(result, expected)
        result = ser.loc[[umax - 1, umax]]
        tm.assert_series_equal(result, ser)

    def test_loc_uint64_disallow_negative(self):
        umax = np.iinfo('uint64').max
        ser = Series([1, 2], index=[umax - 1, umax])
        with pytest.raises(KeyError, match='-1'):
            ser.loc[-1]
        with pytest.raises(KeyError, match='-1'):
            ser.loc[[-1]]

    def test_loc_setitem_empty_append_expands_rows(self):
        data = [1, 2, 3]
        expected = DataFrame({'x': data, 'y': np.array([np.nan] * len(data),
            dtype=object)})
        df = DataFrame(columns=['x', 'y'])
        df.loc[:, 'x'] = data
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_empty_append_expands_rows_mixed_dtype(self,
        using_infer_string):
        data = [1, 2, 3]
        expected = DataFrame({'x': data, 'y': np.array([np.nan] * len(data),
            dtype=object)})
        df = DataFrame(columns=['x', 'y'])
        df['x'] = df['x'].astype(np.int64)
        df.loc[:, 'x'] = data
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_empty_append_single_value(self):
        expected = DataFrame({'x': [1.0], 'y': [np.nan]})
        df = DataFrame(columns=['x', 'y'], dtype=float)
        df.loc[0, 'x'] = expected.loc[0, 'x']
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_empty_append_raises(self):
        data = [1, 2]
        df = DataFrame(columns=['x', 'y'])
        df.index = df.index.astype(np.int64)
        msg = 'None of .*Index.* are in the \\[index\\]'
        with pytest.raises(KeyError, match=msg):
            df.loc[[0, 1], 'x'] = data
        msg = 'setting an array element with a sequence.'
        with pytest.raises(ValueError, match=msg):
            df.loc[0:2, 'x'] = data

    def test_loc_setitem_frame_mixed_labels(self):
        df = DataFrame({(1): [1, 2], (2): [3, 4], 'a': ['a', 'b']})
        result = df.loc[0, [1, 2]]
        expected = Series([1, 3], index=Index([1, 2], dtype=object), dtype=
            object, name=0)
        tm.assert_series_equal(result, expected)
        expected = DataFrame({(1): [5, 2], (2): [6, 4], 'a': ['a', 'b']})
        df.loc[0, [1, 2]] = [5, 6]
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame_multiples(self):
        df = DataFrame({'A': ['foo', 'bar', 'baz'], 'B': Series(range(3),
            dtype=np.int64)})
        rhs = df.loc[1:2]
        rhs.index = df.index[0:2]
        df.loc[0:1] = rhs
        expected = DataFrame({'A': ['bar', 'baz', 'baz'], 'B': Series([1, 2,
            2], dtype=np.int64)})
        tm.assert_frame_equal(df, expected)
        df = DataFrame({'date': date_range('2000-01-01', '2000-01-5'),
            'val': Series(range(5), dtype=np.int64)})
        expected = DataFrame({'date': [Timestamp('20000101'), Timestamp(
            '20000102'), Timestamp('20000101'), Timestamp('20000102'),
            Timestamp('20000103')], 'val': Series([0, 1, 0, 1, 2], dtype=np
            .int64)})
        expected['date'] = expected['date'].astype('M8[ns]')
        rhs = df.loc[0:2]
        rhs.index = df.index[2:5]
        df.loc[2:4] = rhs
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize('indexer, expected', [(0, [20, 1, 2, 3, 4, 5, 
        6, 7, 8, 9]), (slice(4, 8), [0, 1, 2, 3, 20, 20, 20, 20, 8, 9]), ([
        3, 5], [0, 1, 2, 20, 4, 20, 6, 7, 8, 9])])
    def test_loc_setitem_listlike_with_timedelta64index(self, indexer,
        expected_slice):
        tdi = to_timedelta(range(10), unit='s')
        df = DataFrame({'x': range(10)}, index=tdi)
        df.loc[df.index[indexer], 'x'] = 20
        expected = DataFrame([0, 1, 2, 20, 4, 20, 6, 7, 8, 9], index=tdi,
            columns=['x'], dtype='int64')
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_categorical_columns_retains_dtype(self, ordered):
        result = DataFrame({'A': [1], 'B': [1]}, dtype='float64')
        result.loc[:, 'B'] = Categorical(['b'], categories=['a', 'b'],
            ordered=ordered)
        expected = DataFrame({'A': [1], 'B': Categorical(['b'], categories=
            ['a', 'b'], ordered=ordered)})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('ordered', [True, False])
    def test_loc_setitem_categorical_columns_retains_dtype_ordered(self,
        ordered):
        result = DataFrame({'A': [1], 'B': [1]}, dtype='float64')
        result.loc[:, 'B'] = Categorical(['b'], categories=['a', 'b'],
            ordered=ordered)
        expected = DataFrame({'A': [1], 'B': Categorical(['b'], categories=
            ['a', 'b'], ordered=ordered)})
        tm.assert_frame_equal(result, expected)


class TestLocWithEllipsis:

    @pytest.fixture
    def indexer(self, indexer_li):
        return indexer_li

    @pytest.fixture
    def obj(self, series_with_simple_index, frame_or_series):
        obj = series_with_simple_index
        if frame_or_series is not Series:
            obj = obj.to_frame()
        return obj

    def test_loc_iloc_getitem_ellipsis(self, obj, indexer):
        result = indexer(obj)[...]
        tm.assert_equal(result, obj)

    @pytest.mark.parametrize('tpl', [(1,), (1, 2)])
    def test_loc_iloc_getitem_leading_ellipses(self,
        series_with_simple_index, indexer, tpl):
        obj = series_with_simple_index
        key = 0 if indexer is tm.iloc or len(obj) == 0 else obj.index[0]
        if indexer is tm.loc and obj.index.inferred_type == 'boolean':
            return
        if indexer is tm.loc and isinstance(obj.index, MultiIndex):
            msg = 'MultiIndex does not support indexing with Ellipsis'
            with pytest.raises(NotImplementedError, match=msg):
                result = indexer(obj)[..., [key]]
        elif len(obj) != 0:
            result = indexer(obj)[..., [key]]
            expected = indexer(obj)[[key]]
            tm.assert_series_equal(result, expected)
        key2 = 0 if indexer is tm.iloc else obj.name
        df = obj.to_frame()
        result = indexer(df)[..., [key2]]
        expected = indexer(df)[:, [key2]]
        tm.assert_frame_equal(result, expected)

    def test_loc_iloc_getitem_ellipses_only_one_ellipsis(self, obj, indexer):
        key = 0 if indexer is tm.iloc or len(obj) == 0 else obj.index[0]
        with pytest.raises(IndexingError, match=_one_ellipsis_message):
            indexer(obj)[..., ...]
        with pytest.raises(IndexingError, match=_one_ellipsis_message):
            indexer(obj)[..., [key], ...]
        with pytest.raises(IndexingError, match=_one_ellipsis_message):
            indexer(obj)[..., ..., key]
        with pytest.raises(IndexingError, match='Too many indexers'):
            indexer(obj)[key, ..., ...]


class TestLocWithMultiIndex:

    @pytest.mark.parametrize('keys, expected', [(['b', 'a'], [['b', 'b',
        'a', 'a'], [1, 2, 1, 2]]), (['a', 'b'], [['a', 'a', 'b', 'b'], [1, 
        2, 1, 2]]), ((['a', 'b'], [1, 2]), [['a', 'a', 'b', 'b'], [1, 2, 1,
        2]]), ((['a', 'b'], [2, 1]), [['a', 'a', 'b', 'b'], [2, 1, 2, 1]]),
        ((['b', 'a'], [2, 1]), [['b', 'b', 'a', 'a'], [2, 1, 2, 1]]), (([
        'b', 'a'], [1, 2]), [['b', 'b', 'a', 'a'], [1, 2, 1, 2]]), ((['c',
        'a'], [2, 1]), [['c', 'a', 'a'], [1, 2, 1]])])
    @pytest.mark.parametrize('dim', ['index', 'columns'])
    def test_loc_getitem_multilevel_index_order(self, dim, keys, expected):
        kwargs: dict = {dim: [['c', 'a', 'a', 'b', 'b'], [1, 1, 2, 1, 2]]}
        df = DataFrame(np.random.default_rng(2).random((5, 4)), **kwargs)
        exp_index = MultiIndex.from_arrays(expected)
        if dim == 'index':
            res = df.loc[keys, :]
            tm.assert_index_equal(res.index, exp_index)
        elif dim == 'columns':
            res = df.loc[:, keys]
            tm.assert_index_equal(res.columns, exp_index)

    def test_loc_preserve_names(self,
        multiindex_year_month_day_dataframe_random_data):
        ymd = multiindex_year_month_day_dataframe_random_data
        result = ymd.loc[2000]
        result2 = ymd['A'].loc[2000]
        assert result.index.names == ymd.index.names[1:]
        assert result2.index.names == ymd.index.names[1:]
        result = ymd.loc[2000, 2]
        result2 = ymd['A'].loc[2000, 2]
        assert result.index.name == ymd.index.names[2]
        assert result2.index.name == ymd.index.names[2]

    def test_loc_getitem_slice_datetime_objs_with_datetimeindex(self):
        df = DataFrame({'col1': ['a', 'b', 'c'], 'col2': [1, 2, 3]}, index=
            to_datetime(['2020-08-01', '2020-08-02', '2020-08-03']))
        expected = DataFrame({'col1': ['a', 'c'], 'col2': [1, 3]}, index=
            to_datetime(['2020-08-01', '2020-08-03']))
        result = df.loc['2020-08']
        tm.assert_frame_equal(result, expected)

    def test_loc_getitem_slice_with_period_index(self):
        df_unique = DataFrame(np.arange(4.0, dtype='float64'), index=[
            datetime(2001, 1, i, 10, 0) for i in [1, 2, 3, 4]])
        df_dups = DataFrame(np.arange(5.0, dtype='float64'), index=[
            datetime(2001, 1, i, 10, 0) for i in [1, 2, 2, 3, 4]])
        for df in [df_unique, df_dups]:
            result = df.loc[datetime(2001, 1, 1, 10):]
            tm.assert_frame_equal(result, df)
            result = df.loc[:datetime(2001, 1, 4, 10)]
            tm.assert_frame_equal(result, df)
            result = df.loc[datetime(2001, 1, 1, 10):datetime(2001, 1, 4, 10)]
            tm.assert_frame_equal(result, df)
            result = df.loc[datetime(2001, 1, 1, 11):]
            expected = df.iloc[1:]
            tm.assert_frame_equal(result, expected)
            result = df.loc['20010101 11':]
            tm.assert_frame_equal(result, expected)

    def test_loc_setitem_label_slice_multiindex(self):
        index = MultiIndex.from_tuples(zip(['bar', 'bar', 'baz', 'baz',
            'foo', 'foo', 'qux', 'qux'], ['one', 'two', 'one', 'two', 'one',
            'two', 'one', 'two']), names=['first', 'second'])
        result = Series([1, 1, 1, 1, 1, 1, 1, 1], index=index)
        result.loc['baz', 'one':'foo', 'two'] = 100
        expected = Series([1, 1, 100, 100, 100, 100, 1, 1], index=index)
        tm.assert_series_equal(result, expected)


class TestLocSetitemWithExpansion:

    def test_loc_setitem_with_expansion_large_dataframe(self, monkeypatch):
        size_cutoff = 50
        with monkeypatch.context():
            monkeypatch.setattr(libindex, '_SIZE_CUTOFF', size_cutoff)
            result = DataFrame({'x': range(size_cutoff)}, dtype='int64')
            result.loc[size_cutoff] = size_cutoff
        expected = DataFrame({'x': range(size_cutoff + 1)}, dtype='int64')
        tm.assert_frame_equal(result, expected)

    def test_loc_setitem_empty_series(self):
        ser = Series(dtype=object)
        ser.loc[1] = 1
        tm.assert_series_equal(ser, Series([1], index=range(1, 2)))
        ser.loc[3] = 3
        tm.assert_series_equal(ser, Series([1, 3], index=[1, 3]))

    def test_loc_setitem_empty_series_float(self):
        ser = Series(dtype=object)
        ser.loc[1] = 1.0
        tm.assert_series_equal(ser, Series([1.0], index=range(1, 2)))
        ser.loc[3] = 3.0
        tm.assert_series_equal(ser, Series([1.0, 3.0], index=[1, 3]))

    def test_loc_setitem_empty_series_str_idx(self):
        ser = Series(dtype=object)
        ser.loc['foo'] = 1
        tm.assert_series_equal(ser, Series([1], index=Index(['foo'], dtype=
            object)))
        ser.loc['bar'] = 3
        tm.assert_series_equal(ser, Series([1, 3], index=Index(['foo',
            'bar'], dtype=object)))
        ser.loc[3] = 4
        tm.assert_series_equal(ser, Series([1, 3, 4], index=Index(['foo',
            'bar', 3], dtype=object)))

    def test_loc_setitem_empty_append_expands_rows_mixed_dtype(self,
        using_infer_string):
        data = [1, 2, 3]
        expected = DataFrame({'x': data, 'y': np.array([np.nan] * len(data),
            dtype=object)})
        df = DataFrame(columns=['x', 'y'])
        df['x'] = df['x'].astype(np.int64)
        df.loc[:, 'x'] = data
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_empty_append_single_value(self):
        expected = DataFrame({'x': [1.0], 'y': [np.nan]})
        df = DataFrame(columns=['x', 'y'], dtype=float)
        df.loc[0, 'x'] = expected.loc[0, 'x']
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_empty_append_raises(self):
        data = [1, 2]
        df = DataFrame(columns=['x', 'y'])
        df.index = df.index.astype(np.int64)
        msg = 'None of .*Index.* are in the \\[index\\]'
        with pytest.raises(KeyError, match=msg):
            df.loc[[0, 1], 'x'] = data
        msg = 'setting an array element with a sequence.'
        with pytest.raises(ValueError, match=msg):
            df.loc[0:2, 'x'] = data

    def test_loc_setitem_frame_mixed_labels(self):
        df = DataFrame({(1): [1, 2], (2): [3, 4], 'a': ['a', 'b']})
        result = df.loc[0, [1, 2]]
        expected = Series([1, 3], index=Index([1, 2], dtype=object), dtype=
            object, name=0)
        tm.assert_series_equal(result, expected)
        expected = DataFrame({(1): [5, 2], (2): [6, 4], 'a': ['a', 'b']})
        df.loc[0, [1, 2]] = [5, 6]
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame_multiples(self):
        df = DataFrame({'A': ['foo', 'bar', 'baz'], 'B': Series(range(3),
            dtype=np.int64)})
        rhs = df.loc[1:2]
        rhs.index = df.index[0:2]
        df.loc[0:1] = rhs
        expected = DataFrame({'A': ['bar', 'baz', 'baz'], 'B': Series([1, 2,
            2], dtype=np.int64)})
        tm.assert_frame_equal(df, expected)
        df = DataFrame({'date': date_range('2000-01-01', '2000-01-5'),
            'val': Series(range(5), dtype=np.int64)})
        expected = DataFrame({'date': [Timestamp('20000101'), Timestamp(
            '20000102'), Timestamp('20000101'), Timestamp('20000102'),
            Timestamp('20000103')], 'val': Series([0, 1, 0, 1, 2], dtype=np
            .int64)})
        expected['date'] = expected['date'].astype('M8[ns]')
        rhs = df.loc[0:2]
        rhs.index = df.index[2:5]
        df.loc[2:4] = rhs
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize('indexer, expected', [(0, [20, 1, 2, 3, 4, 5, 
        6, 7, 8, 9]), (slice(4, 8), [0, 1, 2, 3, 20, 20, 20, 20, 8, 9]), ([
        3, 5], [0, 1, 2, 20, 4, 20, 6, 7, 8, 9])])
    def test_loc_setitem_listlike_with_timedelta64index(self, indexer, expected
        ):
        tdi = to_timedelta(range(10), unit='s')
        df = DataFrame({'x': range(10)}, index=tdi)
        df.loc[df.index[indexer], 'x'] = 20
        expected_df = DataFrame(expected, index=tdi, columns=['x'], dtype=
            'int64')
        tm.assert_frame_equal(df, expected_df)

    def test_loc_setitem_categorical_value(self):
        ser = Series(['a', 'b', 'c'], dtype='category')
        ser.loc[3] = 0
        expected = Series(['a', 'b', 'c', 0], dtype='object')
        tm.assert_series_equal(ser, expected)

    def test_loc_setitem_categorical_additional_element(self, ordered):
        ser = Series(['a', 'b', 'c'], dtype='category')
        ser.loc[3] = 'a'
        expected = Series(['a', 'b', 'c', 'a'], dtype='category')
        tm.assert_series_equal(ser, expected)

    def test_loc_set_nan_in_categorical_series(self, any_numeric_ea_dtype):
        ser = Series([1, 2, 3], dtype=CategoricalDtype(Index([1, 2, 3],
            dtype=any_numeric_ea_dtype)))
        ser.loc[3] = np.nan
        expected = Series([1, 2, 3, np.nan], dtype=CategoricalDtype(Index([
            1, 2, 3], dtype=any_numeric_ea_dtype)))
        tm.assert_series_equal(ser, expected)
        ser.loc[1] = np.nan
        expected = Series([1, np.nan, 3, np.nan], dtype=CategoricalDtype(
            Index([1, 2, 3], dtype=any_numeric_ea_dtype)))
        tm.assert_series_equal(ser, expected)

    @pytest.mark.parametrize('any_numeric_ea_dtype', [pd.Int32Dtype(), pd.
        Int64Dtype()])
    def test_loc_consistency_series_enlarge_set_into(self, any_numeric_ea_dtype
        ):
        srs_enlarge = Series([0, 0, 0, 0], dtype=any_numeric_ea_dtype)
        srs_setinto = Series([0, 0, 0, 0], dtype=any_numeric_ea_dtype)
        expected = Series([0, 0, 0, 0], dtype=any_numeric_ea_dtype)
        srs_enlarge.loc[3:0] = [0, 0, 0, 0]
        tm.assert_series_equal(srs_enlarge, srs_setinto)
        expected = Series([0, 0, 0, 0], dtype=any_numeric_ea_dtype)
        tm.assert_series_equal(srs_enlarge, expected)

    def test_loc_setitem_identity_slice_obj_type_preserved(self):
        df = DataFrame({'a': [1, 2, 3]})
        sliced_df = df.loc[:]
        sliced_df.loc[:, 'a'] = [4, 4, 4]
        expected = DataFrame({'a': [4, 4, 4]})
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_identity_slice_dtype_preserved(self):
        df = DataFrame({'a': [1, 2, 3]}, dtype='float64')
        sliced_df = df.loc[:]
        sliced_df.loc[:, 'a'] = [4, 4, 4]
        expected = DataFrame({'a': [4.0, 4.0, 4.0]}, dtype='float64')
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_identity_slice_dtype_inferred_object(self):
        df = DataFrame({'a': ['foo', 'bar', 'baz']}, dtype=object)
        sliced_df = df.loc[:]
        sliced_df.loc[:, 'a'] = ['a', 'a', 'a']
        expected = DataFrame({'a': ['a', 'a', 'a']}, dtype=object)
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_multiindex_slice_label_missing_levels(self):
        df = DataFrame({'A': [1, 2, 3, 4, 5, 6]}, index=[0, 0, 1, 1, 2, 2])
        df = df.set_index([['a', 'a', 'b', 'b', 'c', 'c'], ['x', 'x', 'y',
            'y', 'z', 'z']])
        df.loc['a', 'A'] = [7, 8]
        expected = DataFrame({'A': [7, 8, 3, 4, 5, 6]}, index=[('a', 'x'),
            ('a', 'x'), ('b', 'y'), ('b', 'y'), ('c', 'z'), ('c', 'z')])
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_empty_frame(self):
        df = DataFrame(columns=['A', 'B'], dtype='object')
        df.loc[:, 'A'] = []
        tm.assert_frame_equal(df, DataFrame(columns=['A', 'B'], dtype='object')
            )

    def test_loc_setitem_empty_series(self, frame_or_series):
        ser = Series(dtype=object)
        ser.loc[:] = []
        expected = Series(dtype=object)
        tm.assert_series_equal(ser, expected)


class TestLocCallable:

    def test_frame_loc_getitem_callable(self, frame_or_series):
        df = DataFrame({'A': [1, 2, 3, 4], 'B': ['a', 'a', 'b', 'b'], 'C':
            [1, 2, 3, 4]})
        res = df.loc[lambda x: x.A > 2]
        tm.assert_frame_equal(res, df.loc[df.A > 2])
        res = df.loc[lambda x: x.B == 'b', :]
        tm.assert_frame_equal(res, df.loc[df.B == 'b', :])
        res = df.loc[lambda x: x.A > 2, lambda x: x.columns == 'B']
        tm.assert_frame_equal(res, df.loc[df.A > 2, [False, True, False]])
        res = df.loc[lambda x: x.A > 2, lambda x: 'B']
        tm.assert_series_equal(res, df.loc[df.A > 2, 'B'])
        res = df.loc[lambda x: x.A > 2, lambda x: ['A', 'B']]
        tm.assert_frame_equal(res, df.loc[df.A > 2, ['A', 'B']])
        res = df.loc[lambda x: x.A == 2, lambda x: ['A', 'B']]
        tm.assert_frame_equal(res, df.loc[df.A == 2, ['A', 'B']])
        res = df.loc[lambda x: 1, lambda x: 'A']
        assert is_scalar(res) and res == df.loc[1, 'A']

    def test_frame_loc_getitem_callable_mixture(self, frame_or_series):
        df = DataFrame({'A': [1, 2, 3, 4], 'B': ['a', 'a', 'b', 'b'], 'C':
            [1, 2, 3, 4]})
        res = df.loc[lambda x: x.A > 2, ['A', 'B']]
        tm.assert_frame_equal(res, df.loc[df.A > 2, ['A', 'B']])
        res = df.loc[[2, 3], lambda x: ['A', 'B']]
        tm.assert_frame_equal(res, df.loc[[2, 3], ['A', 'B']])
        res = df.loc[3, lambda x: ['A', 'B']]
        tm.assert_series_equal(res, df.loc[3, ['A', 'B']])

    def test_frame_loc_getitem_callable_labels(self):
        df = DataFrame({'X': [1, 2, 3, 4], 'Y': ['a', 'a', 'b', 'b']},
            index=list('ABCD'))
        res = df.loc[lambda x: ['A', 'C']]
        tm.assert_frame_equal(res, df.loc[['A', 'C']])
        res = df.loc[lambda x: ['A', 'C'], :]
        tm.assert_frame_equal(res, df.loc[['A', 'C'], :])
        res = df.loc[lambda x: ['A', 'C'], lambda x: 'X']
        tm.assert_series_equal(res, df.loc[['A', 'C'], 'X'])
        res = df.loc[lambda x: ['A', 'C'], lambda x: ['X']]
        tm.assert_frame_equal(res, df.loc[['A', 'C'], ['X']])
        res = df.loc[['A', 'C'], lambda x: 'X']
        tm.assert_series_equal(res, df.loc[['A', 'C'], 'X'])
        res = df.loc[['A', 'C'], lambda x: ['X']]
        tm.assert_frame_equal(res, df.loc[['A', 'C'], ['X']])
        res = df.loc[lambda x: ['A', 'C'], 'X']
        tm.assert_series_equal(res, df.loc[['A', 'C'], 'X'])
        res = df.loc[lambda x: ['A', 'C'], ['X']]
        tm.assert_frame_equal(res, df.loc[['A', 'C'], ['X']])


class TestPartialStringSlicing:

    def test_loc_getitem_partial_string_slicing_datetimeindex(self):
        df = DataFrame({'col1': ['a', 'b', 'c'], 'col2': [1, 2, 3]}, index=
            to_datetime(['2020-08-01', '2020-08-02', '2020-08-05']))
        expected = DataFrame({'col1': ['a', 'c'], 'col2': [1, 3]}, index=
            to_datetime(['2020-08-01', '2020-08-05']))
        result = df.loc['2020-08']
        tm.assert_frame_equal(result, expected)

    def test_loc_getitem_partial_string_slicing_with_periodindex(self):
        pi = pd.period_range(start='2017-01-01', end='2018-01-01', freq='M')
        ser = pi.to_series()
        result = ser.loc[:'2017-12']
        expected = ser.iloc[:-1]
        tm.assert_series_equal(result, expected)

    def test_loc_getitem_partial_string_slicing_with_timedeltaindex(self):
        ix = timedelta_range(start='1 day', end='2 days', freq='1h')
        ser = ix.to_series()
        result = ser.loc[:'1 days']
        expected = ser.iloc[:-1]
        tm.assert_series_equal(result, expected)

    def test_loc_getitem_str_timedeltaindex(self):
        df = DataFrame({'x': [1, 2, 3]}, index=to_timedelta(range(3), unit=
            'days'))
        with pytest.raises(KeyError, match='not in index'):
            df.loc['3 days']

    @pytest.mark.parametrize('start,stop, expected_slice', [(np.timedelta64
        (0, 'ns'), None, slice(0, 11)), (np.timedelta64(1, 'D'), np.
        timedelta64(6, 'D'), slice(1, 7)), (None, np.timedelta64(4, 'D'),
        slice(0, 5))])
    def test_loc_getitem_slice_label_td64obj(self, start, stop, expected_slice
        ):
        ser = Series(range(11), timedelta_range('0 days', '10 days'))
        result = ser.loc[slice(start, stop)]
        expected = ser.iloc[expected_slice]
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('start', ['2018', '2020'])
    def test_loc_getitem_slice_unordered_dt_index(self, start):
        obj = Series(range(3), index=[Timestamp('2016-01-01'), Timestamp(
            '2019-01-01'), Timestamp('2017-01-01')])
        with pytest.raises(KeyError, match=
            'Value based partial slicing on non-monotonic'):
            obj.loc[start:'2022']

    @pytest.mark.parametrize('dtype', ['object', 'string'])
    def test_loc_setitem_multiindex_datetime_label(self, dtype):
        ser = Series(['x', 'y', 'z'], index=pd.period_range('2020', periods
            =3, freq='D'))
        ser.loc[pd.Period('2020', freq='D')] = 'a'
        expected = Series(['a', 'y', 'z'], index=pd.period_range('2020',
            periods=3, freq='D'))
        tm.assert_series_equal(ser, expected)

    @pytest.mark.parametrize('columns, column_key, expected_columns', [([
        2011, 2012, 2013], [2011, 2012], [0, 1]), ([2011, 2012, 'All'], [
        2011, 2012], [0, 1]), ([2011, 2012, 'All'], [2011, 'All'], [0, 2])])
    def test_loc_getitem_label_list_integer_labels(self, columns,
        column_key, expected_columns):
        df = DataFrame(np.random.default_rng(2).random((3, 3)), columns=
            columns, index=list('ABC'))
        expected = df.iloc[:, expected_columns]
        result = df.loc[['A', 'B', 'C'], column_key]
        tm.assert_frame_equal(result, expected, check_column_type=True)

    def test_loc_setitem_float_intindex(self):
        rand_data = np.random.default_rng(2).standard_normal((8, 4))
        result = DataFrame(rand_data)
        result.loc[:, 0.5] = np.nan
        expected_data = np.hstack((rand_data, np.array([np.nan] * 8).
            reshape(8, 1)))
        expected = DataFrame(expected_data, columns=[0.0, 1.0, 2.0, 3.0, 0.5])
        tm.assert_frame_equal(result, expected)
        result = DataFrame(rand_data)
        result.loc[:, 0.5] = np.nan
        tm.assert_frame_equal(result, expected)


class TestLocGetitemMultiIndexTupleLevel:

    def test_loc_getitem_multiindex_tuple_level(self):
        lev1 = ['a', 'b', 'c']
        lev2 = [(0, 1), (1, 0)]
        lev3 = [0, 1]
        cols = MultiIndex.from_product([lev1, lev2, lev3], names=['x', 'y',
            'z'])
        df = DataFrame(6, index=range(5), columns=cols)
        result = df.loc[:, (lev1[0], lev2[0], lev3[0])]
        expected = df.iloc[:, :1]
        tm.assert_frame_equal(result, expected)
        alt = df.xs((lev1[0], lev2[0], lev3[0]), level=[0, 1, 2], axis=1)
        tm.assert_frame_equal(alt, expected)
        ser = df.iloc[0]
        expected2 = ser.iloc[:1]
        alt2 = ser.xs((lev1[0], lev2[0], lev3[0]), level=[0, 1, 2], axis=0)
        tm.assert_series_equal(alt2, expected2)
        result2 = ser.loc[lev1[0], lev2[0], lev3[0]]
        assert result2 == 6


class TestLocBooleanLabelsAndSlices:

    @pytest.mark.parametrize('bool_value', [True, False])
    def test_loc_bool_incompatible_index_raises(self, index,
        frame_or_series, bool_value):
        message = (
            f'{bool_value}: boolean label can not be used without a boolean index'
            )
        if index.inferred_type != 'boolean':
            obj = frame_or_series(index=index, dtype='object')
            with pytest.raises(KeyError, match=message):
                obj.loc[bool_value]

    @pytest.mark.parametrize('bool_value', [True, False])
    def test_loc_bool_should_not_raise(self, frame_or_series, bool_value):
        obj = frame_or_series(index=Index([True, False], dtype='boolean'),
            dtype='object')
        obj.loc[bool_value]

    def test_loc_bool_slice_raises(self, index, frame_or_series):
        message = (
            'slice\\(True, False, None\\): boolean values can not be used in a slice'
            )
        obj = frame_or_series(index=index, dtype='object')
        with pytest.raises(TypeError, match=message):
            obj.loc[True:False]


class TestLocBooleanMask:

    def test_loc_setitem_bool_mask_timedeltaindex(self):
        df = DataFrame({'x': range(10)})
        df.index = to_timedelta(range(10), unit='s')
        conditions = [df['x'] > 3, df['x'] == 3, df['x'] < 3]
        expected_data = [[0, 1, 2, 3, 10, 10, 10, 10, 10, 10], [0, 1, 2, 10,
            4, 5, 6, 7, 8, 9], [10, 10, 10, 3, 4, 5, 6, 7, 8, 9]]
        for cond, data in zip(conditions, expected_data):
            result = df.copy()
            result.loc[cond, 'x'] = 10
            expected = DataFrame(data, index=to_timedelta(range(10), unit=
                's'), columns=['x'], dtype='int64')
            tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('tz', [None, 'UTC'])
    def test_loc_setitem_mask_with_datetimeindex_tz(self, tz):
        mask = np.array([True, False, True, False])
        idx = date_range('2011-01-01', periods=4, tz=tz)
        df = DataFrame({'a': np.arange(4)}, index=idx).astype('float64')
        result = df.copy()
        result.loc[mask, :] = df.loc[mask, :]
        tm.assert_frame_equal(result, df)
        result = df.copy()
        result.loc[mask] = df.loc[mask]
        tm.assert_frame_equal(result, df)

    def test_loc_setitem_mask_td64_series_value(self):
        ser = Series([0, 0, 0, 0], dtype='object')
        ser.loc[[False, False, True, False]] = Series([0, 0, 0], index=[2, 
            1, 0])
        expected = Series([0, 0, 0, 0], dtype='object')
        tm.assert_series_equal(ser, expected)

    def test_loc_setitem_boolean_and_column(self, float_frame):
        mask = float_frame['A'] > 0
        result = float_frame.copy()
        result.loc[mask, 'B'] = 0
        expected = float_frame.copy()
        expected.loc[mask, 'B'] = 0
        tm.assert_frame_equal(float_frame, expected)

    def test_loc_setitem_ndframe_values_alignment(self):
        df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        df.loc[[False, False, True], 'a'] = Series([10, 11, 12], index=[2, 
            1, 0])
        expected = DataFrame({'a': [1, 2, 10], 'b': [4, 5, 6]})
        tm.assert_frame_equal(df, expected)


class TestLocListlike:

    def test_loc_getitem_list_of_labels_categoricalindex_with_na(self):
        ci = CategoricalIndex(['A', 'B', np.nan])
        ser = Series(range(3), index=ci)
        result = ser.loc[list(ci)]
        tm.assert_series_equal(result, ser)
        result = ser[list(ci)]
        tm.assert_series_equal(result, ser)
        result = ser.to_frame().loc[list(ci)]
        tm.assert_frame_equal(result, ser.to_frame())
        ser2 = ser[:-1]
        ci2 = ci[1:]
        msg = 'not in index'
        with pytest.raises(KeyError, match=msg):
            ser2.loc[list(ci2)]
        with pytest.raises(KeyError, match=msg):
            ser2[list(ci2)]
        with pytest.raises(KeyError, match=msg):
            ser2.to_frame().loc[list(ci2)]

    def test_loc_setitem_dict_timedelta_multiple_set(self):
        df = DataFrame({'time': [Timedelta(6, unit='s'), Timedelta(6, unit=
            's')], 'value': ['foo', 'bar']})
        df.loc[1:1, 'time'] = Timedelta(6, unit='s')
        df.loc[1:1, 'value'] = 'bar'
        expected = DataFrame({'time': [Timedelta(6, unit='s'), Timedelta(6,
            unit='s')], 'value': ['foo', 'bar']})
        tm.assert_frame_equal(df, expected)

    def test_getitem_loc_str_periodindex(self):
        msg = 'Period with BDay freq is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            index = pd.period_range(start='2000', periods=20, freq='B')
            ser = Series(range(20), index=index)
            assert ser.loc['2000-01-14'] == 9

    @pytest.mark.parametrize('indexer_end', [None,
        '2020-01-02 23:59:59.999999999'])
    def test_loc_setitem_with_expansion_nonunique_index(self, indexer_end):
        if not len([1, 1, 2, 2, 3, 3]):
            pytest.skip('Not relevant for empty Index')
        index = Index([1, 1, 2, 2, 3, 3], dtype='Int64')
        N = len(index)
        arr = np.arange(N).astype(np.int64)
        df = DataFrame({'A': np.arange(6, dtype='int8')}, index=index)
        df.loc[4, 'A'] = 10
        ser = Series([1, 2], dtype='float64')
        ser.loc[1] = 2
        tm.assert_series_equal(ser, Series([1, 2], index=[1, 1]))

    def test_loc_setitem_with_expansion_preserves_nullable_int(self,
        any_numeric_ea_dtype):
        ser = Series([0, 0, 0, 0], dtype=any_numeric_ea_dtype)
        df = DataFrame({'data': ser})
        result = DataFrame(index=df.index)
        result.loc[df.index, 'data'] = ser
        tm.assert_frame_equal(result, df, check_column_type=False)
        result = DataFrame(index=df.index)
        result.loc[df.index, 'data'] = ser._values
        tm.assert_frame_equal(result, df, check_column_type=False)

    def test_loc_setitem_ea_not_full_column(self):
        df = DataFrame({'A': [1, 1, 1, 1, 1], 'B': ['a', 'a', 'a', 'a', 'a']})
        df.loc[1:2, 'A'] = Categorical([2, 2], categories=[1, 2])
        expected = DataFrame({'A': [1, 2, 2, 1, 1], 'B': ['a', 'a', 'a',
            'a', 'a']})
        tm.assert_frame_equal(df, expected)


class Test_setitem_getitem_related:

    def test_loc_setitem_labels_with_tz(self):
        df = DataFrame({'A': [1, 2, 3]}, index=pd.date_range('2020-01-01',
            periods=3, tz='UTC'))
        df.loc['2020-01-01', 'A'] = 10
        expected = DataFrame({'A': [10, 2, 3]}, index=pd.date_range(
            '2020-01-01', periods=3, tz='UTC'))
        tm.assert_frame_equal(df, expected)

    def test_setitem_roundtrip(self):
        s = Series([1, 2, 3], index=['a', 'b', 'c'])
        s.loc['d'] = 4
        expected = Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
        tm.assert_series_equal(s, expected)


class TestIndexingWithPeriodIndex:

    def test_loc_periodindex_getitem(self):
        df = DataFrame({'A': [Period(2019), 'x1', 'x2'], 'B': [Period(2018),
            Period(2016), 'y1'], 'C': [Period(2017), 'z1', Period(2015)],
            'V1': [1, 2, 3], 'V2': [10, 20, 30]}).set_index(['A', 'B', 'C'])
        result = df.loc[Period(2019), 'foo', 'bar']
        with pytest.raises(KeyError):
            df.loc[Period(2017), 'foo', 'bar']


def test_loc_assign_dict_to_row(dtype='object'):
    df = DataFrame({'A': ['abc', 'def'], 'B': ['ghi', 'jkl']}, dtype=dtype)
    df.loc[0, :] = {'A': 'newA', 'B': 'newB'}
    expected = DataFrame({'A': ['newA', 'def'], 'B': ['newB', 'jkl']},
        dtype=dtype)
    tm.assert_frame_equal(df, expected)


def test_loc_setitem_multiindex_timestamp():
    vals = np.random.default_rng(2).standard_normal((8, 6))
    idx = date_range('1/1/2000', periods=8)
    cols = ['A', 'B', 'C', 'D', 'E', 'F']
    exp = DataFrame(vals, index=idx, columns=cols)
    exp.loc['baz', 'one':'foo', 'two'] = 100
    df = DataFrame(vals, index=idx, columns=cols)
    df = df.set_index([['a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'], ['x', 'y',
        'x', 'y', 'x', 'y', 'x', 'y']])
    df.loc['baz', 'one':'foo', 'two'] = 100
    expected = Series([1, 1, 100, 100, 100, 100, 1, 1], index=[('a', 'x'),
        ('a', 'y'), ('b', 'x'), ('b', 'y'), ('c', 'x'), ('c', 'y'), ('d',
        'x'), ('d', 'y')])
    tm.assert_series_equal(df['A'], expected)


def test_loc_index_alignment_for_series():
    df = DataFrame({'a': [1, 2], 'b': [3, 4]}, index=list(range(2)))
    other = Series([200, 999], index=[1, 0])
    df.loc[:, 'a'] = other
    expected = DataFrame({'a': [999, 200], 'b': [3, 4]})
    tm.assert_frame_equal(expected, df)


def test_loc_reindexing_of_empty_index():
    df = DataFrame({'a': [1, 1, 2, 2]}, index=[0, 0, 1, 1])
    df.loc[[]] = []
    expected = DataFrame({'a': [1, 1, 2, 2]}, index=[0, 0, 1, 1])
    tm.assert_frame_equal(df, expected)


class TestLocNonUniqueIndex:

    def test_loc_non_unique_index_setitem(self):
        ids = list(range(11))
        index = Index(ids * 1000, dtype='Int64')
        df = DataFrame({'val': np.arange(len(index), dtype=np.intp)}, index
            =index)
        result = df.loc[ids]
        expected = DataFrame({'val': index.argsort(kind='stable').astype(np
            .intp)}, index=Index(np.array(ids).repeat(1000), dtype='Int64'))
        tm.assert_frame_equal(result, expected)

    def test_loc_non_unique_index_getitem(self):
        ids = list(range(11))
        index = Index(ids * 1000, dtype='Int64')
        df = DataFrame({'val': np.arange(len(index), dtype=np.intp)}, index
            =index)
        result = df.loc[ids]
        expected = DataFrame({'val': index.argsort(kind='stable').astype(np
            .intp)}, index=Index(np.array(ids).repeat(1000), dtype='Int64'))
        tm.assert_frame_equal(result, expected)


def test_loc_with_period_index():
    pm = pd.period_range('2011-01-01', '2011-01-02', freq='M')
    df = DataFrame(np.random.default_rng(2).standard_normal((24, 10)), index=pm
        )
    tm.assert_frame_equal(df, df.loc[pm])
    tm.assert_frame_equal(df, df.loc[list(pm)])
    tm.assert_frame_equal(df, df.loc[list(pm)])
    tm.assert_frame_equal(df.iloc[0:5], df.loc[pm[0:5]])
    tm.assert_frame_equal(df, df.loc[list(pm)])


def test_loc_setitem_categorical_columns_retains_dtype_ordered_tie(self,
    ordered):
    result = DataFrame({'A': [1], 'B': [1]}, dtype='float64')
    result.loc[:, 'B'] = Categorical(['b'], categories=['a', 'b'], ordered=
        ordered)
    expected = DataFrame({'A': [1], 'B': Categorical(['b'], categories=['a',
        'b'], ordered=ordered)})
    tm.assert_frame_equal(result, expected)


def test_additional_element_to_categorical_series_loc(self):
    ser = Series(['a', 'b', 'c'], dtype='category')
    ser.loc[3] = 0
    expected = Series(['a', 'b', 'c', 0], dtype='object')
    tm.assert_series_equal(ser, expected)


def test_loc_getitem_multiindex_namedtuple(self):
    IndexType = namedtuple('IndexType', ['a', 'b'])
    idx1 = IndexType('foo', 'bar')
    idx2 = IndexType('baz', 'bof')
    index = Index([idx1, idx2], name='composite_index', tupleize_cols=False)
    df = DataFrame([[1, 2], [3, 4]], index=index, columns=['A', 'B'])
    result = df.loc[IndexType('foo', 'bar')]['A']
    assert result == 1


def test_loc_getitem_missing_unicode_key(self):
    df = DataFrame({'a': [1]})
    with pytest.raises(KeyError, match='א'):
        df.loc[:, 'א']


def test_loc_getitem_slicing_datetimes_frame(self):
    df_unique = DataFrame(np.arange(4.0, dtype='float64'), index=[datetime(
        2001, 1, i, 10, 0) for i in [1, 2, 3, 4]])
    df_dups = DataFrame(np.arange(5.0, dtype='float64'), index=[datetime(
        2001, 1, i, 10, 0) for i in [1, 2, 2, 3, 4]])
    for df in [df_unique, df_dups]:
        result = df.loc[datetime(2001, 1, 1, 10):]
        tm.assert_frame_equal(result, df)
        result = df.loc[:datetime(2001, 1, 4, 10)]
        tm.assert_frame_equal(result, df)
        result = df.loc[datetime(2001, 1, 1, 10):datetime(2001, 1, 4, 10)]
        tm.assert_frame_equal(result, df)
        result = df.loc[datetime(2001, 1, 1, 11):]
        expected = df.iloc[1:]
        tm.assert_frame_equal(result, expected)
        result = df.loc['20010101 11':]
        tm.assert_frame_equal(result, expected)


def test_loc_setitem_multiindex_nonunique_index():
    df = DataFrame({'A': [1, 2, 3, 4, 5, 6]}, index=['a', 'a', 'a', 'a',
        'a', 'a'])
    df.loc['a', 'A'] = [10, 20, 30, 40, 50, 60]
    expected = DataFrame({'A': [10, 20, 30, 40, 50, 60]}, index=['a', 'a',
        'a', 'a', 'a', 'a'])
    tm.assert_frame_equal(df, expected)
