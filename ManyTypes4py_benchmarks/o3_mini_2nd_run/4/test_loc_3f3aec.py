#!/usr/bin/env python3
"""
Test label based indexing with loc
"""

from collections import namedtuple
import contextlib
from datetime import date, datetime, time, timedelta
import re
from typing import Any, Callable, List, Union, Optional, Sequence, Tuple
from dateutil.tz import gettz
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.errors import IndexingError
import pandas as pd
from pandas import (
    Categorical, CategoricalDtype, CategoricalIndex, DataFrame, DatetimeIndex,
    Index, IndexSlice, MultiIndex, Period, PeriodIndex, Series, SparseDtype,
    Timedelta, Timestamp, date_range, timedelta_range, to_datetime, to_timedelta
)
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.core.indexing import _one_ellipsis_message
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises

@pytest.mark.parametrize('series, new_series, expected_ser', [
    [[np.nan, np.nan, 'b'], ['a', np.nan, np.nan], [False, True, True]],
    [[np.nan, 'b'], ['a', np.nan], [False, True]]
])
def test_not_change_nan_loc(series: List[Any], new_series: List[Any], expected_ser: List[Any]) -> None:
    df: DataFrame = DataFrame({'A': series})
    df.loc[:, 'A'] = new_series
    expected: DataFrame = DataFrame({'A': expected_ser})
    tm.assert_frame_equal(df.isna(), expected)
    tm.assert_frame_equal(df.notna(), ~expected)


class TestLoc:
    def test_none_values_on_string_columns(self, using_infer_string: bool) -> None:
        df: DataFrame = DataFrame(['1', '2', None], columns=['a'], dtype=object)
        assert df.loc[2, 'a'] is None
        df = DataFrame(['1', '2', None], columns=['a'], dtype='str')
        if using_infer_string:
            assert np.isnan(df.loc[2, 'a'])
        else:
            assert df.loc[2, 'a'] is None

    def test_loc_getitem_int(self, frame_or_series: Callable[..., Any]) -> None:
        obj: Any = frame_or_series(range(3), index=Index(list('abc'), dtype=object))
        check_indexing_smoketest_or_raises(obj, 'loc', 2, fails=KeyError)

    def test_loc_getitem_label(self, frame_or_series: Callable[..., Any]) -> None:
        obj: Any = frame_or_series()
        check_indexing_smoketest_or_raises(obj, 'loc', 'c', fails=KeyError)

    @pytest.mark.parametrize('key', ['f', 20])
    @pytest.mark.parametrize('index', [
        Index(list('abcd'), dtype=object),
        Index([2, 4, 'null', 8], dtype=object),
        date_range('20130101', periods=4),
        Index(range(0, 8, 2), dtype=np.float64),
        Index([])
    ])
    def test_loc_getitem_label_out_of_range(self, key: Any, index: Index, frame_or_series: Callable[..., Any]) -> None:
        obj: Any = frame_or_series(range(len(index)), index=index)
        check_indexing_smoketest_or_raises(obj, 'loc', key, fails=KeyError)

    @pytest.mark.parametrize('key', [[0, 1, 2], [1, 3.0, 'A']])
    @pytest.mark.parametrize('dtype', [np.int64, np.uint64, np.float64])
    def test_loc_getitem_label_list(self, key: List[Any], dtype: Any, frame_or_series: Callable[..., Any]) -> None:
        obj: Any = frame_or_series(range(3), index=Index([0, 1, 2], dtype=dtype))
        check_indexing_smoketest_or_raises(obj, 'loc', key, fails=KeyError)

    @pytest.mark.parametrize('index', [
        None,
        Index([0, 1, 2], dtype=np.int64),
        Index([0, 1, 2], dtype=np.uint64),
        Index([0, 1, 2], dtype=np.float64),
        MultiIndex.from_arrays([range(3), range(3)])
    ])
    @pytest.mark.parametrize('key', [[0, 1, 2], [0, 2, 10], [3, 6, 7], [(1, 3), (1, 4), (2, 5)]])
    def test_loc_getitem_label_list_with_missing(self, key: Any, index: Optional[Index], frame_or_series: Callable[..., Any]) -> None:
        if index is None:
            obj: Any = frame_or_series()
        else:
            obj = frame_or_series(range(len(index)), index=index)
        check_indexing_smoketest_or_raises(obj, 'loc', key, fails=KeyError)

    @pytest.mark.parametrize('dtype', [np.int64, np.uint64])
    def test_loc_getitem_label_list_fails(self, dtype: Any, frame_or_series: Callable[..., Any]) -> None:
        obj: Any = frame_or_series(range(3), Index([0, 1, 2], dtype=dtype))
        check_indexing_smoketest_or_raises(obj, 'loc', [20, 30, 40], axes=1, fails=KeyError)

    def test_loc_getitem_bool(self, frame_or_series: Callable[..., Any]) -> None:
        obj: Any = frame_or_series()
        b: List[bool] = [True, False, True, False]
        check_indexing_smoketest_or_raises(obj, 'loc', b, fails=IndexError)

    @pytest.mark.parametrize('slc, indexes, axes, fails', [
        (slice(1, 3), [Index(list('abcd'), dtype=object),
                       Index([2, 4, 'null', 8], dtype=object),
                       None,
                       date_range('20130101', periods=4),
                       Index(range(0, 12, 3), dtype=np.float64)],
         None, TypeError),
        (slice('20130102', '20130104'), [date_range('20130101', periods=4)], 1, TypeError),
        (slice(2, 8), [Index([2, 4, 'null', 8], dtype=object)], 0, TypeError),
        (slice(2, 8), [Index([2, 4, 'null', 8], dtype=object)], 1, KeyError),
        (slice(2, 4, 2), [Index([2, 4, 'null', 8], dtype=object)], 0, TypeError)
    ])
    def test_loc_getitem_label_slice(self, slc: slice, indexes: List[Optional[Index]], axes: Optional[Any], fails: Exception, frame_or_series: Callable[..., Any]) -> None:
        for index in indexes:
            if index is None:
                obj: Any = frame_or_series()
            else:
                obj = frame_or_series(range(len(index)), index=index)
            check_indexing_smoketest_or_raises(obj, 'loc', slc, axes=axes, fails=fails)

    def test_setitem_from_duplicate_axis(self) -> None:
        df: DataFrame = DataFrame([[20, 'a'], [200, 'a'], [200, 'a']],
                                   columns=['col1', 'col2'], index=[10, 1, 1])
        df.loc[1, 'col1'] = np.arange(2)
        expected: DataFrame = DataFrame([[20, 'a'], [0, 'a'], [1, 'a']],
                                        columns=['col1', 'col2'], index=[10, 1, 1])
        tm.assert_frame_equal(df, expected)

    def test_column_types_consistent(self) -> None:
        df: DataFrame = DataFrame(data={
            'channel': [1, 2, 3],
            'A': ['String 1', np.nan, 'String 2'],
            'B': [Timestamp('2019-06-11 11:00:00'), pd.NaT, Timestamp('2019-06-11 12:00:00')]
        })
        df2: DataFrame = DataFrame(data={
            'A': ['String 3'],
            'B': [Timestamp('2019-06-11 12:00:00')]
        })
        df.loc[df['A'].isna(), ['A', 'B']] = df2.values
        expected: DataFrame = DataFrame(data={
            'channel': [1, 2, 3],
            'A': ['String 1', 'String 3', 'String 2'],
            'B': [Timestamp('2019-06-11 11:00:00'), Timestamp('2019-06-11 12:00:00'),
                  Timestamp('2019-06-11 12:00:00')]
        })
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize('obj, key, exp', [
        (DataFrame([[1]], columns=Index([False])), IndexSlice[:, False], Series([1], name=False)),
        (Series([1], index=Index([False])), False, [1]),
        (DataFrame([[1]], index=Index([False])), False, Series([1], name=False))
    ])
    def test_loc_getitem_single_boolean_arg(self, obj: Union[DataFrame, Series], key: Any, exp: Any) -> None:
        res: Any = obj.loc[key]
        if isinstance(exp, (DataFrame, Series)):
            tm.assert_equal(res, exp)
        else:
            assert res == exp


class TestLocBaseIndependent:
    def test_loc_npstr(self) -> None:
        df: DataFrame = DataFrame(index=date_range('2021', '2022'))
        result: DataFrame = df.loc[np.array(['2021/6/1'])[0]:]
        expected: DataFrame = df.iloc[151:]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('msg, key', [
        (r"Period\('2019', 'Y-DEC'\), 'foo', 'bar'", (Period(2019), 'foo', 'bar')),
        (r"Period\('2019', 'Y-DEC'\), 'y1', 'bar'", (Period(2019), 'y1', 'bar')),
        (r"Period\('2019', 'Y-DEC'\), 'foo', 'z1'", (Period(2019), 'foo', 'z1')),
        (r"Period\('2018', 'Y-DEC'\), Period\('2016', 'Y-DEC'\), 'bar'", (Period(2018), Period(2016), 'bar')),
        (r"Period\('2018', 'Y-DEC'\), 'foo', 'y1'", (Period(2018), 'foo', 'y1')),
        (r"Period\('2017', 'Y-DEC'\), 'foo', Period\('2015', 'Y-DEC'\)", (Period(2017), 'foo', Period(2015))),
        (r"Period\('2017', 'Y-DEC'\), 'z1', 'bar'", (Period(2017), 'z1', 'bar'))
    ])
    def test_contains_raise_error_if_period_index_is_in_multi_index(self, msg: str, key: Tuple[Any, ...]) -> None:
        df: DataFrame = DataFrame({
            'A': [Period(2019), 'x1', 'x2'],
            'B': [Period(2018), Period(2016), 'y1'],
            'C': [Period(2017), 'z1', Period(2015)],
            'V1': [1, 2, 3],
            'V2': [10, 20, 30]
        }).set_index(['A', 'B', 'C'])
        with pytest.raises(KeyError, match=msg):
            df.loc[key]

    def test_loc_getitem_missing_unicode_key(self) -> None:
        df: DataFrame = DataFrame({'a': [1]})
        with pytest.raises(KeyError, match='א'):
            df.loc[:, 'א']

    def test_loc_getitem_dups(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).random((20, 5)),
                                  index=[ 'ABCDE'[x % 5] for x in range(20) ])
        expected: Any = df.loc['A', 0]
        result: Any = df.loc[:, 0].loc['A']
        tm.assert_series_equal(result, expected)

    def test_loc_getitem_dups2(self) -> None:
        df: DataFrame = DataFrame([[1, 2, 'foo', 'bar', Timestamp('20130101')]],
                                  columns=['a', 'a', 'a', 'a', 'a'], index=[1])
        expected: Series = Series([1, 2, 'foo', 'bar', Timestamp('20130101')],
                                  index=['a', 'a', 'a', 'a', 'a'], name=1)
        result: Series = df.iloc[0]
        tm.assert_series_equal(result, expected)
        result = df.loc[1]
        tm.assert_series_equal(result, expected)

    def test_loc_setitem_dups(self) -> None:
        df_orig: DataFrame = DataFrame({
            'me': list('rttti'),
            'foo': list('aaade'),
            'bar': np.arange(5, dtype='float64') * 1.34 + 2,
            'bar2': np.arange(5, dtype='float64') * -0.34 + 2
        }).set_index('me')
        indexer: Tuple[Any, Any] = ('r', ['bar', 'bar2'])
        df: DataFrame = df_orig.copy()
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
        df1: DataFrame = DataFrame({'a': [0, 1, 1], 'b': pd.Series([100, 200, 300], dtype='uint32')})
        ix: Any = df1['a'] == 1
        newb1: Any = df1.loc[ix, 'b'] + 1
        df1.loc[ix, 'b'] = newb1
        expected: DataFrame = DataFrame({'a': [0, 1, 1], 'b': pd.Series([100, 201, 301], dtype='uint32')})
        tm.assert_frame_equal(df1, expected)
        df2: DataFrame = DataFrame({'a': [0, 1, 1], 'b': [100, 200, 300]}, dtype='uint64')
        ix = df1['a'] == 1
        newb2: Any = df2.loc[ix, 'b']
        with pytest.raises(TypeError, match='Invalid value'):
            df1.loc[ix, 'b'] = newb2

    def test_loc_setitem_dtype(self) -> None:
        df: DataFrame = DataFrame({'id': ['A'], 'a': [1.2], 'b': [0.0], 'c': [-2.5]})
        cols: List[str] = ['a', 'b', 'c']
        df.loc[:, cols] = df.loc[:, cols].astype('float32')
        expected: DataFrame = DataFrame({'id': ['A'],
            'a': np.array([1.2], dtype='float64'),
            'b': np.array([0.0], dtype='float64'),
            'c': np.array([-2.5], dtype='float64')})
        tm.assert_frame_equal(df, expected)

    def test_getitem_label_list_with_missing(self) -> None:
        s: Series = Series(range(3), index=['a', 'b', 'c'])
        with pytest.raises(KeyError, match='not in index'):
            s[['a', 'd']]
        s = Series(range(3))
        with pytest.raises(KeyError, match='not in index'):
            s[[0, 3]]

    @pytest.mark.parametrize('index', [[True, False], [True, False, True, False]])
    def test_loc_getitem_bool_diff_len(self, index: List[bool]) -> None:
        s: Series = Series([1, 2, 3])
        msg: str = f'Boolean index has wrong length: {len(index)} instead of {len(s)}'
        with pytest.raises(IndexError, match=msg):
            s.loc[index]

    def test_loc_getitem_int_slice(self) -> None:
        pass

    def test_loc_to_fail(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).random((3, 3)),
                                   index=['a', 'b', 'c'], columns=['e', 'f', 'g'])
        msg: str = f'''\\"None of \\[Index\\(\\[1, 2\\], dtype='{np.dtype(int)}'\\)\\] are in the \\[index\\]\\"'''
        with pytest.raises(KeyError, match=msg):
            df.loc[[1, 2], [1, 2]]

    def test_loc_to_fail2(self) -> None:
        s: Series = Series(dtype=object)
        s.loc[1] = 1
        s.loc['a'] = 2
        with pytest.raises(KeyError, match='^-1$'):
            s.loc[-1]
        msg: str = f'''\\"None of \\[Index\\(\\[-1, -2\\], dtype='{np.dtype(int)}'\\)\\] are in the \\[index\\]\\"'''
        with pytest.raises(KeyError, match=msg):
            s.loc[[-1, -2]]
        msg = '\\"None of \\[Index\\(\\[\'4\'\\], dtype=\'object\'\\)\\] are in the \\[index\\]\\"'
        with pytest.raises(KeyError, match=msg):
            s.loc[pd.Index(['4'], dtype=object)]
        s.loc[-1] = 3
        with pytest.raises(KeyError, match='not in index'):
            s.loc[[-1, -2]]
        s['a'] = 2
        msg = f'''\\"None of \\[Index\\(\\[-2\\], dtype='{np.dtype(int)}'\\)\\] are in the \\[index\\]\\"'''
        with pytest.raises(KeyError, match=msg):
            s.loc[[-2]]
        del s['a']
        with pytest.raises(KeyError, match=msg):
            s.loc[[-2]] = 0

    def test_loc_to_fail3(self) -> None:
        df: DataFrame = DataFrame([['a'], ['b']], index=[1, 2], columns=['value'])
        msg: str = f'''\\"None of \\[Index\\(\\[3\\], dtype='{np.dtype(int)}'\\)\\] are in the \\[index\\]\\"'''
        with pytest.raises(KeyError, match=msg):
            df.loc[[3], :]
        with pytest.raises(KeyError, match=msg):
            df.loc[[3]]

    def test_loc_getitem_list_with_fail(self) -> None:
        s: Series = Series([1, 2, 3])
        s.loc[[2]]
        msg: str = 'None of [RangeIndex(start=3, stop=4, step=1)] are in the [index]'
        with pytest.raises(KeyError, match=re.escape(msg)):
            s.loc[[3]]
        with pytest.raises(KeyError, match='not in index'):
            s.loc[[2, 3]]

    def test_loc_index(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).random(size=(5, 10)),
                                  index=Index([f'alpha_{i}' for i in range(5)], name='a'),
                                  columns=Index([f'i-{i}' for i in range(2)], name='a'))
        mask = df.index.map(lambda x: 'alpha' in x)
        expected: DataFrame = df.loc[np.array(mask)]
        result: DataFrame = df.loc[mask]
        tm.assert_frame_equal(result, expected)
        result = df.loc[mask.values]
        tm.assert_frame_equal(result, expected)
        result = df.loc[pd.array(mask, dtype='boolean')]
        tm.assert_frame_equal(result, expected)

    def test_loc_general(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).random((4, 4)),
                                  columns=['A', 'B', 'C', 'D'], index=['A', 'B', 'C', 'D'])
        result: DataFrame = df.loc[:, 'A':'B'].iloc[0:2, :]
        assert (result.columns == ['A', 'B']).all()
        assert (result.index == ['A', 'B']).all()
        result = DataFrame({'a': [Timestamp('20130101')], 'b': [1]}).iloc[0]
        expected: Series = Series([Timestamp('20130101'), 1], index=['a', 'b'], name=0)
        tm.assert_series_equal(result, expected)
        assert result.dtype == object

    @pytest.fixture
    def frame_for_consistency(self) -> DataFrame:
        return DataFrame({'date': date_range('2000-01-01', '2000-01-05'),
                          'val': pd.Series(range(5), dtype=np.int64)})

    @pytest.mark.parametrize('val', [0, np.array(0, dtype=np.int64), np.array([0, 0, 0, 0, 0], dtype=np.int64)])
    def test_loc_setitem_consistency(self, frame_for_consistency: DataFrame, val: Any) -> None:
        df: DataFrame = frame_for_consistency.copy()
        with pytest.raises(TypeError, match='Invalid value'):
            df.loc[:, 'date'] = val

    def test_loc_setitem_consistency_dt64_to_str(self, frame_for_consistency: DataFrame) -> None:
        df: DataFrame = frame_for_consistency.copy()
        with pytest.raises(TypeError, match='Invalid value'):
            df.loc[:, 'date'] = 'foo'

    def test_loc_setitem_consistency_dt64_to_float(self, frame_for_consistency: DataFrame) -> None:
        df: DataFrame = frame_for_consistency.copy()
        with pytest.raises(TypeError, match='Invalid value'):
            df.loc[:, 'date'] = 1.0

    def test_loc_setitem_consistency_single_row(self) -> None:
        df: DataFrame = DataFrame({'date': pd.Series([Timestamp('20180101')])})
        with pytest.raises(TypeError, match='Invalid value'):
            df.loc[:, 'date'] = 'string'

    def test_loc_setitem_consistency_empty(self) -> None:
        expected: DataFrame = DataFrame(columns=['x', 'y'])
        df: DataFrame = DataFrame(columns=['x', 'y'])
        with tm.assert_produces_warning(None):
            df.loc[:, 'x'] = 1
        tm.assert_frame_equal(df, expected)
        df = DataFrame(columns=['x', 'y'])
        df['x'] = 1
        expected['x'] = expected['x'].astype(np.int64)
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_consistency_slice_column_len(self, using_infer_string: bool) -> None:
        levels: List[List[Any]] = [
            ['Region_1'] * 4,
            ['Site_1', 'Site_1', 'Site_2', 'Site_2'],
            [3987227376, 3980680971, 3977723249, 3977723089]
        ]
        mi: MultiIndex = MultiIndex.from_arrays(levels, names=['Region', 'Site', 'RespondentID'])
        clevels: List[List[Any]] = [
            ['Respondent', 'Respondent', 'Respondent', 'OtherCat', 'OtherCat'],
            ['Something', 'StartDate', 'EndDate', 'Yes/No', 'SomethingElse']
        ]
        cols: MultiIndex = MultiIndex.from_arrays(clevels, names=['Level_0', 'Level_1'])
        values: List[List[Any]] = [
            ['A', '5/25/2015 10:59', '5/25/2015 11:22', 'Yes', np.nan],
            ['A', '5/21/2015 9:40', '5/21/2015 9:52', 'Yes', 'Yes'],
            ['A', '5/20/2015 8:27', '5/20/2015 8:41', 'Yes', np.nan],
            ['A', '5/20/2015 8:33', '5/20/2015 9:09', 'Yes', 'No']
        ]
        df: DataFrame = DataFrame(values, index=mi, columns=cols)
        ctx: Any = contextlib.nullcontext()
        if using_infer_string:
            ctx = pytest.raises(TypeError, match='Invalid value')
        with ctx:
            df.loc[:, ('Respondent', 'StartDate')] = to_datetime(df.loc[:, ('Respondent', 'StartDate')])
        with ctx:
            df.loc[:, ('Respondent', 'EndDate')] = to_datetime(df.loc[:, ('Respondent', 'EndDate')])
        if using_infer_string:
            return
        df = df.infer_objects()
        df.loc[:, ('Respondent', 'Duration')] = df.loc[:, ('Respondent', 'EndDate')] - df.loc[:, ('Respondent', 'StartDate')]
        with pytest.raises(TypeError, match='Invalid value'):
            df.loc[:, ('Respondent', 'Duration')] = df.loc[:, ('Respondent', 'Duration')] / pd.Timedelta(60000000000)

    @pytest.mark.parametrize('unit', ['Y', 'M', 'D', 'h', 'm', 's', 'ms', 'us'])
    def test_loc_assign_non_ns_datetime(self, unit: str) -> None:
        df: DataFrame = DataFrame({'timestamp': [np.datetime64('2017-02-11 12:41:29'), np.datetime64('1991-11-07 04:22:37')]})
        df.loc[:, unit] = df.loc[:, 'timestamp'].values.astype(f'datetime64[{unit}]')
        df['expected'] = df.loc[:, 'timestamp'].values.astype(f'datetime64[{unit}]')
        expected: Series = Series(df.loc[:, 'expected'], name=unit)
        tm.assert_series_equal(df.loc[:, unit], expected)

    def test_loc_modify_datetime(self) -> None:
        df: DataFrame = DataFrame.from_dict({'date': [1485264372711, 1485265925110, 1540215845888, 1540282121025]})
        df['date_dt'] = to_datetime(df['date'], unit='ms', cache=True).dt.as_unit('ms')
        df.loc[:, 'date_dt_cp'] = df.loc[:, 'date_dt']
        df.loc[[2, 3], 'date_dt_cp'] = df.loc[[2, 3], 'date_dt']
        expected: DataFrame = DataFrame([
            [1485264372711, '2017-01-24 13:26:12.711', '2017-01-24 13:26:12.711'],
            [1485265925110, '2017-01-24 13:52:05.110', '2017-01-24 13:52:05.110'],
            [1540215845888, '2018-10-22 13:44:05.888', '2018-10-22 13:44:05.888'],
            [1540282121025, '2018-10-23 08:08:41.025', '2018-10-23 08:08:41.025']
        ], columns=['date', 'date_dt', 'date_dt_cp'])
        columns: List[str] = ['date_dt', 'date_dt_cp']
        expected[columns] = expected[columns].apply(to_datetime)
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame_with_reindex(self) -> None:
        df: DataFrame = DataFrame(index=[3, 5, 4], columns=['A'], dtype=float)
        df.loc[[4, 3, 5], 'A'] = np.array([1, 2, 3], dtype='int64')
        ser: Series = Series([2, 3, 1], index=[3, 5, 4], dtype=float)
        expected: DataFrame = DataFrame({'A': ser})
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame_with_reindex_mixed(self) -> None:
        df: DataFrame = DataFrame(index=[3, 5, 4], columns=['A', 'B'], dtype=float)
        df['B'] = 'string'
        df.loc[[4, 3, 5], 'A'] = np.array([1, 2, 3], dtype='int64')
        ser: Series = Series([2, 3, 1], index=[3, 5, 4], dtype='int64')
        expected: DataFrame = DataFrame({'A': ser.astype(float)})
        expected['B'] = 'string'
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame_with_inverted_slice(self) -> None:
        df: DataFrame = DataFrame(index=[1, 2, 3], columns=['A', 'B'], dtype=float)
        df['B'] = 'string'
        df.loc[slice(3, 0, -1), 'A'] = np.array([1, 2, 3], dtype='int64')
        expected: DataFrame = DataFrame({'A': [3.0, 2.0, 1.0], 'B': 'string'}, index=[1, 2, 3])
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_empty_frame(self) -> None:
        keys1: List[str] = ['@' + str(i) for i in range(5)]
        val1: np.ndarray = np.arange(5, dtype='int64')
        keys2: List[str] = ['@' + str(i) for i in range(4)]
        val2: np.ndarray = np.arange(4, dtype='int64')
        index: List[Any] = list(set(keys1).union(keys2))
        df: DataFrame = DataFrame(index=index)
        df['A'] = np.nan
        df.loc[keys1, 'A'] = val1
        df['B'] = np.nan
        df.loc[keys2, 'B'] = val2
        sera: Series = Series(val1, index=keys1, dtype=np.float64)
        serb: Series = Series(val2, index=keys2)
        expected: DataFrame = DataFrame({'A': sera, 'B': serb}, columns=Index(['A', 'B'], dtype=object)).reindex(index=index)
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((4, 4)),
                                  index=list('abcd'), columns=list('ABCD'))
        result: Any = df.iloc[0, 0]
        df.loc['a', 'A'] = 1
        result = df.loc['a', 'A']
        assert result == 1
        result = df.iloc[0, 0]
        assert result == 1
        df.loc[:, 'B':'D'] = 0
        expected: DataFrame = df.loc[:, 'B':'D']
        result = df.iloc[:, 1:]
        tm.assert_frame_equal(result, expected)

    def test_loc_setitem_frame_nan_int_coercion_invalid(self) -> None:
        df: DataFrame = DataFrame({'A': [1, 2, 3], 'B': np.nan})
        df.loc[df.B > df.A, 'B'] = df.A
        expected: DataFrame = DataFrame({'A': [1, 2, 3], 'B': np.nan})
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame_mixed_labels(self) -> None:
        df: DataFrame = DataFrame({1: [1, 2], 2: [3, 4], 'a': ['a', 'b']})
        result: Series = df.loc[0, [1, 2]]
        expected: Series = Series([1, 3], index=Index([1, 2], dtype=object), dtype=object, name=0)
        tm.assert_series_equal(result, expected)
        expected = DataFrame({1: [5, 2], 2: [6, 4], 'a': ['a', 'b']})
        df.loc[0, [1, 2]] = [5, 6]
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame_multiples(self) -> None:
        df: DataFrame = DataFrame({'A': ['foo', 'bar', 'baz'], 'B': pd.Series(range(3), dtype=np.int64)})
        rhs: DataFrame = df.loc[1:2]
        rhs.index = df.index[0:2]
        df.loc[0:1] = rhs
        expected: DataFrame = DataFrame({'A': ['bar', 'baz', 'baz'], 'B': pd.Series([1, 2, 2], dtype=np.int64)})
        tm.assert_frame_equal(df, expected)
        df = DataFrame({'date': date_range('2000-01-01', '2000-01-05'),
                        'val': pd.Series(range(5), dtype=np.int64)})
        expected = DataFrame({
            'date': [Timestamp('20000101'), Timestamp('20000102'), Timestamp('20000101'), Timestamp('20000102'), Timestamp('20000103')],
            'val': pd.Series([0, 1, 0, 1, 2], dtype=np.int64)
        })
        expected['date'] = expected['date'].astype('M8[ns]')
        rhs = df.loc[0:2]
        rhs.index = df.index[2:5]
        df.loc[2:4] = rhs
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize('indexer', [['A'], slice(None, 'A', None), np.array(['A'])])
    @pytest.mark.parametrize('value', [['Z'], np.array(['Z'])])
    def test_loc_setitem_with_scalar_index(self, indexer: Any, value: Any) -> None:
        df: DataFrame = DataFrame([[1, 2], [3, 4]], columns=['A', 'B']).astype({'A': object})
        df.loc[0, indexer] = value
        result: Any = df.loc[0, 'A']
        assert is_scalar(result) and result == 'Z'

    @pytest.mark.parametrize('index,box,expected', [
        (([0, 2], ['A', 'B', 'C', 'D']), 7, DataFrame([[7, 7, 7, 7], [3, 4, np.nan, np.nan], [7, 7, 7, 7]], columns=['A', 'B', 'C', 'D'])),
        ((1, ['C', 'D']), [7, 8], DataFrame([[1, 2, np.nan, np.nan], [3, 4, 7, 8], [5, 6, np.nan, np.nan]], columns=['A', 'B', 'C', 'D'])),
        ((1, ['A', 'B', 'C']), np.array([7, 8, 9], dtype=np.int64), DataFrame([[1, 2, np.nan], [7, 8, 9], [5, 6, np.nan]], columns=['A', 'B', 'C'])),
        ((slice(1, 3, None), ['B', 'C', 'D']), [[7, 8, 9], [10, 11, 12]], DataFrame([[1, 2, np.nan, np.nan], [3, 7, 8, 9], [5, 10, 11, 12]], columns=['A', 'B', 'C', 'D'])),
        ((slice(1, 3, None), ['C', 'A', 'D']), np.array([[7, 8, 9], [10, 11, 12]], dtype=np.int64), DataFrame([[1, 2, np.nan, np.nan], [8, 4, 7, 9], [11, 6, 10, 12]], columns=['A', 'B', 'C', 'D'])),
        ((slice(None, None, None), ['A', 'C']), DataFrame([[7, 8], [9, 10], [11, 12]], columns=['A', 'C']), DataFrame([[7, 2, 8], [9, 4, 10], [11, 6, 12]], columns=['A', 'B', 'C']))
    ])
    def test_loc_setitem_missing_columns(self, index: Any, box: Any, expected: DataFrame) -> None:
        df: DataFrame = DataFrame([[1, 2], [3, 4], [5, 6]], columns=['A', 'B'])
        df.loc[index] = box
        tm.assert_frame_equal(df, expected)

    def test_loc_coercion(self) -> None:
        df: DataFrame = DataFrame({'date': [Timestamp('20130101').tz_localize('UTC'), pd.NaT]})
        expected: Any = df.dtypes
        result: DataFrame = df.iloc[[0]]
        tm.assert_series_equal(result.dtypes, expected)
        result = df.iloc[[1]]
        tm.assert_series_equal(result.dtypes, expected)

    def test_loc_coercion2(self) -> None:
        df: DataFrame = DataFrame({'date': [datetime(2012, 1, 1), datetime(1012, 1, 2)]})
        expected: Any = df.dtypes
        result: DataFrame = df.iloc[[0]]
        tm.assert_series_equal(result.dtypes, expected)
        result = df.iloc[[1]]
        tm.assert_series_equal(result.dtypes, expected)

    def test_loc_coercion3(self) -> None:
        df: DataFrame = DataFrame({'text': ['some words'] + [None] * 9})
        expected: Any = df.dtypes
        result: DataFrame = df.iloc[0:2]
        tm.assert_series_equal(result.dtypes, expected)
        result = df.iloc[3:]
        tm.assert_series_equal(result.dtypes, expected)

    def test_setitem_new_key_tz(self, indexer_sl: Callable[[Series], Any]) -> None:
        vals: List[Any] = [to_datetime(42).tz_localize('UTC'), to_datetime(666).tz_localize('UTC')]
        expected: Series = Series(vals, index=Index(['foo', 'bar'], dtype=object))
        ser: Series = Series(dtype=object)
        indexer_sl(ser)['foo'] = vals[0]
        indexer_sl(ser)['bar'] = vals[1]
        tm.assert_series_equal(ser, expected)

    def test_loc_non_unique(self) -> None:
        df: DataFrame = DataFrame({'A': [1, 2, 3, 4, 5, 6], 'B': [3, 4, 5, 6, 7, 8]},
                                  index=[0, 1, 0, 1, 2, 3])
        msg: str = "'Cannot get left slice bound for non-unique label: 1'"
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
        result: DataFrame = df.loc[1:]
        expected: DataFrame = DataFrame({'A': [2, 4, 5, 6], 'B': [4, 6, 7, 8]}, index=[1, 1, 2, 3])
        tm.assert_frame_equal(result, expected)
        result = df.loc[0:]
        tm.assert_frame_equal(result, df)
        result = df.loc[1:2]
        expected = DataFrame({'A': [2, 4, 5], 'B': [4, 6, 7]}, index=[1, 1, 2])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.arm_slow
    @pytest.mark.parametrize('length, l2', [[900, 100], [900000, 100000]])
    def test_loc_non_unique_memory_error(self, length: int, l2: int) -> None:
        columns: List[Any] = list('ABCDEFG')
        df: DataFrame = pd.concat([
            DataFrame(np.random.default_rng(2).standard_normal((length, len(columns))), index=np.arange(length), columns=columns),
            DataFrame(np.ones((l2, len(columns))), index=[0] * l2, columns=columns)
        ])
        assert df.index.is_unique is False
        mask: np.ndarray = np.arange(l2)
        result: DataFrame = df.loc[mask]
        expected: DataFrame = pd.concat([
            df.take([0]),
            DataFrame(np.ones((len(mask), len(columns))), index=[0] * len(mask), columns=columns),
            df.take(mask[1:])
        ])
        tm.assert_frame_equal(result, expected)

    def test_loc_name(self) -> None:
        df: DataFrame = DataFrame([[1, 1], [1, 1]])
        df.index.name = 'index_name'
        result: Any = df.iloc[[0, 1]].index.name
        assert result == 'index_name'
        result = df.loc[[0, 1]].index.name
        assert result == 'index_name'

    def test_loc_empty_list_indexer_is_ok(self) -> None:
        df: DataFrame = DataFrame(np.ones((5, 2)),
                                  index=Index([f'i-{i}' for i in range(5)], name='a'),
                                  columns=Index([f'i-{i}' for i in range(2)], name='a'))
        tm.assert_frame_equal(df.loc[:, []], df.iloc[:, :0], check_index_type=True, check_column_type=True)
        tm.assert_frame_equal(df.loc[[], :], df.iloc[:0, :], check_index_type=True, check_column_type=True)
        tm.assert_frame_equal(df.loc[[]], df.iloc[:0, :], check_index_type=True, check_column_type=True)

    def test_identity_slice_returns_new_object(self) -> None:
        original_df: DataFrame = DataFrame({'a': [1, 2, 3]})
        sliced_df: DataFrame = original_df.loc[:]
        assert sliced_df is not original_df
        assert original_df[:] is not original_df
        assert original_df.loc[:, :] is not original_df
        assert np.shares_memory(original_df['a']._values, sliced_df['a']._values)
        original_df.loc[:, 'a'] = [4, 4, 4]
        assert (sliced_df['a'] == [1, 2, 3]).all()
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
        assert df[0] is not df.loc[:, 0]
        original_series: Series = Series([1, 2, 3, 4, 5, 6])
        sliced_series: Series = original_series.loc[:]
        assert sliced_series is not original_series
        assert original_series[:] is not original_series
        original_series[:3] = [7, 8, 9]
        assert all(sliced_series[:3] == [1, 2, 3])

    def test_loc_copy_vs_view(self, request: Any) -> None:
        x: DataFrame = DataFrame(list(zip(range(3), range(3))), columns=['a', 'b'])
        y: DataFrame = x.copy()
        q: Series = y.loc[:, 'a']
        q += 2
        tm.assert_frame_equal(x, y)
        z: DataFrame = x.copy()
        q = z.loc[x.index, 'a']
        q += 2
        tm.assert_frame_equal(x, z)

    def test_loc_uint64(self) -> None:
        umax: Any = np.iinfo('uint64').max
        ser: Series = Series([1, 2], index=[umax - 1, umax])
        result: Any = ser.loc[umax - 1]
        expected: Any = ser.iloc[0]
        assert result == expected
        result = ser.loc[[umax - 1]]
        expected = ser.iloc[[0]]
        tm.assert_series_equal(result, expected)
        result = ser.loc[[umax - 1, umax]]
        tm.assert_series_equal(result, ser)

    def test_loc_uint64_disallow_negative(self) -> None:
        umax: Any = np.iinfo('uint64').max
        ser: Series = Series([1, 2], index=[umax - 1, umax])
        with pytest.raises(KeyError, match='-1'):
            ser.loc[-1]
        with pytest.raises(KeyError, match='-1'):
            ser.loc[[-1]]

    def test_loc_setitem_empty_append_expands_rows(self) -> None:
        data: List[int] = [1, 2, 3]
        expected: DataFrame = DataFrame({'x': data, 'y': np.array([np.nan] * len(data), dtype=object)})
        df: DataFrame = DataFrame(columns=['x', 'y'])
        df.loc[:, 'x'] = data
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_empty_append_expands_rows_mixed_dtype(self) -> None:
        data: List[int] = [1, 2, 3]
        expected: DataFrame = DataFrame({'x': data, 'y': np.array([np.nan] * len(data), dtype=object)})
        df: DataFrame = DataFrame(columns=['x', 'y'])
        df['x'] = df['x'].astype(np.int64)
        df.loc[:, 'x'] = data
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_empty_append_single_value(self) -> None:
        expected: DataFrame = DataFrame({'x': [1.0], 'y': [np.nan]})
        df: DataFrame = DataFrame(columns=['x', 'y'], dtype=float)
        df.loc[0, 'x'] = expected.loc[0, 'x']
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_empty_append_raises(self) -> None:
        data: List[int] = [1, 2]
        df: DataFrame = DataFrame(columns=['x', 'y'])
        df.index = df.index.astype(np.int64)
        msg: str = 'None of .*Index.* are in the \\[index\\]'
        with pytest.raises(KeyError, match=msg):
            df.loc[[0, 1], 'x'] = data
        msg = 'setting an array element with a sequence.'
        with pytest.raises(ValueError, match=msg):
            df.loc[0:2, 'x'] = data

    def test_indexing_zerodim_np_array(self) -> None:
        df: DataFrame = DataFrame([[1, 2], [3, 4]])
        result: Series = df.loc[np.array(0)]
        s: Series = Series([1, 2], name=0)
        tm.assert_series_equal(result, s)

    def test_series_indexing_zerodim_np_array(self) -> None:
        s: Series = Series([1, 2])
        result: Any = s.loc[np.array(0)]
        assert result == 1

    def test_loc_reverse_assignment(self) -> None:
        data: List[Union[int, None]] = [1, 2, 3, 4, 5, 6] + [None] * 4
        expected: Series = Series(data, index=range(2010, 2020))
        result: Series = Series(index=range(2010, 2020), dtype=np.float64)
        result.loc[2015:2010:-1] = [6, 5, 4, 3, 2, 1]
        tm.assert_series_equal(result, expected)

    def test_loc_setitem_str_to_small_float_conversion_type(self, using_infer_string: bool) -> None:
        col_data: List[str] = [str(np.random.default_rng(2).random() * 1e-12) for _ in range(5)]
        result: DataFrame = DataFrame(col_data, columns=['A'])
        expected: DataFrame = DataFrame(col_data, columns=['A'])
        tm.assert_frame_equal(result, expected)
        if using_infer_string:
            with pytest.raises(TypeError, match='Invalid value'):
                result.loc[result.index, 'A'] = [float(x) for x in col_data]
        else:
            result.loc[result.index, 'A'] = [float(x) for x in col_data]
            expected = DataFrame(col_data, columns=['A'], dtype=float).astype(object)
            tm.assert_frame_equal(result, expected)
        result['A'] = [float(x) for x in col_data]
        expected = DataFrame(col_data, columns=['A'], dtype=float)
        tm.assert_frame_equal(result, expected)

    def test_loc_getitem_time_object(self, frame_or_series: Callable[..., Any]) -> None:
        rng: DatetimeIndex = date_range('1/1/2000', '1/5/2000', freq='5min')
        mask: np.ndarray = (rng.hour == 9) & (rng.minute == 30)
        obj: Any = DataFrame(np.random.default_rng(2).standard_normal((len(rng), 3)), index=rng)
        obj = tm.get_obj(obj, frame_or_series)
        result: Any = obj.loc[time(9, 30)]
        exp: Any = obj.loc[mask]
        tm.assert_equal(result, exp)
        chunk: Any = obj.loc['1/4/2000':]
        result = chunk.loc[time(9, 30)]
        expected: Any = result[-1:]
        result.index = result.index._with_freq(None)
        expected.index = expected.index._with_freq(None)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize('spmatrix_t', ['coo_matrix', 'csc_matrix', 'csr_matrix'])
    @pytest.mark.parametrize('dtype', [np.complex128, np.float64, np.int64, bool])
    def test_loc_getitem_range_from_spmatrix(self, spmatrix_t: str, dtype: Any) -> None:
        sp_sparse: Any = pytest.importorskip('scipy.sparse')
        spmatrix_t_obj: Any = getattr(sp_sparse, spmatrix_t)
        rows: int = 5
        cols: int = 7
        spmatrix: Any = spmatrix_t_obj(np.eye(rows, cols, dtype=dtype), dtype=dtype)
        df: DataFrame = DataFrame.sparse.from_spmatrix(spmatrix)
        itr_idx: range = range(2, rows)
        result: np.ndarray = np.nan_to_num(df.loc[itr_idx].values)
        expected: np.ndarray = spmatrix.toarray()[itr_idx]
        tm.assert_numpy_array_equal(result, expected)
        result = df.loc[itr_idx].dtypes.values
        expected = np.full(cols, SparseDtype(dtype))
        tm.assert_numpy_array_equal(result, expected)

    def test_loc_getitem_listlike_all_retains_sparse(self) -> None:
        df: DataFrame = DataFrame({'A': pd.array([0, 0], dtype=SparseDtype('int64'))})
        result: DataFrame = df.loc[[0, 1]]
        tm.assert_frame_equal(result, df)

    def test_loc_getitem_sparse_frame(self) -> None:
        sp_sparse: Any = pytest.importorskip('scipy.sparse')
        df: DataFrame = DataFrame.sparse.from_spmatrix(sp_sparse.eye(5, dtype=np.int64))
        result: DataFrame = df.loc[range(2)]
        expected: DataFrame = DataFrame([[1, 0, 0, 0, 0],
                                         [0, 1, 0, 0, 0]], dtype=SparseDtype(np.int64))
        tm.assert_frame_equal(result, expected)
        result = df.loc[range(2)].loc[range(1)]
        expected = DataFrame([[1, 0, 0, 0, 0]], dtype=SparseDtype(np.int64))
        tm.assert_frame_equal(result, expected)

    def test_loc_getitem_sparse_series(self) -> None:
        s: Series = Series([1.0, 0.0, 0.0, 0.0, 0.0], dtype=SparseDtype('float64', 0.0))
        result: Series = s.loc[range(2)]
        expected: Series = Series([1.0, 0.0], dtype=SparseDtype('float64', 0.0))
        tm.assert_series_equal(result, expected)
        result = s.loc[range(3)].loc[range(2)]
        expected = Series([1.0, 0.0], dtype=SparseDtype('float64', 0.0))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('indexer', ['loc', 'iloc'])
    def test_getitem_single_row_sparse_df(self, indexer: str) -> None:
        df: DataFrame = DataFrame([[1.0, 0.0, 1.5], [0.0, 2.0, 0.0]], dtype=SparseDtype(float))
        result: Series = getattr(df, indexer)[0]
        expected: Series = Series([1.0, 0.0, 1.5], dtype=SparseDtype(float), name=0)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('key_type', [iter, np.array, Series, Index])
    def test_loc_getitem_iterable(self, float_frame: DataFrame, key_type: Callable[[List[Any]], Any]) -> None:
        idx: Any = key_type(['A', 'B', 'C'])
        result: DataFrame = float_frame.loc[:, idx]
        expected: DataFrame = float_frame.loc[:, ['A', 'B', 'C']]
        tm.assert_frame_equal(result, expected)

    def test_loc_getitem_timedelta_0seconds(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).normal(size=(10, 4)))
        df.index = timedelta_range(start='0s', periods=10, freq='s')
        expected: DataFrame = df.loc[Timedelta('0s'):, :]
        result: DataFrame = df.loc['0s':, :]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('val,expected', [(2 ** 63 - 1, 1), (2 ** 63, 2)])
    def test_loc_getitem_uint64_scalar(self, val: int, expected: int) -> None:
        df: DataFrame = DataFrame([1, 2], index=[2 ** 63 - 1, 2 ** 63])
        result: Series = df.loc[val]
        expected_ser: Series = Series([expected])
        expected_ser.name = val
        tm.assert_series_equal(result, expected_ser)

    def test_loc_setitem_int_label_with_float_index(self, float_numpy_dtype: Any) -> None:
        dtype: Any = float_numpy_dtype
        ser: Series = Series(['a', 'b', 'c'], index=Index([0, 0.5, 1], dtype=dtype))
        expected: Series = ser.copy()
        ser.loc[1] = 'zoo'
        expected.iloc[2] = 'zoo'
        tm.assert_series_equal(ser, expected)

    @pytest.mark.parametrize('indexer, expected', [
        (0, [20, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        (slice(4, 8), [0, 1, 2, 3, 20, 20, 20, 20, 8, 9]),
        ([3, 5], [0, 1, 2, 20, 4, 20, 6, 7, 8, 9])
    ])
    def test_loc_setitem_listlike_with_timedelta64index(self, indexer: Any, expected: List[int]) -> None:
        tdi: pd.TimedeltaIndex = to_timedelta(range(10), unit='s')
        df: DataFrame = DataFrame({'x': range(10)}, dtype='int64', index=tdi)
        df.loc[df.index[indexer], 'x'] = 20
        expected_df: DataFrame = DataFrame(expected, index=tdi, columns=['x'], dtype='int64')
        tm.assert_frame_equal(expected_df, df)

    def test_loc_setitem_categorical_values_partial_column_slice(self) -> None:
        df: DataFrame = DataFrame({'a': [1, 1, 1, 1, 1], 'b': list('aaaaa')})
        with pytest.raises(TypeError, match='Invalid value'):
            df.loc[1:2, 'a'] = Categorical(['b', 'b'], categories=['a', 'b'])
            df.loc[2:3, 'b'] = Categorical(['b', 'b'], categories=['a', 'b'])

    def test_loc_setitem_single_row_categorical(self, using_infer_string: bool) -> None:
        df: DataFrame = DataFrame({'Alpha': ['a'], 'Numeric': [0]})
        categories: Categorical = Categorical(df['Alpha'], categories=['a', 'b', 'c'])
        df.loc[:, 'Alpha'] = categories
        result: Series = df['Alpha']
        expected: Series = Series(categories, index=df.index, name='Alpha').astype(object if not using_infer_string else 'str')
        tm.assert_series_equal(result, expected)
        df['Alpha'] = categories
        tm.assert_series_equal(df['Alpha'], Series(categories, name='Alpha'))

    def test_loc_setitem_datetime_coercion(self) -> None:
        df: DataFrame = DataFrame({'c': [Timestamp('2010-10-01')] * 3})
        df.loc[0:1, 'c'] = np.datetime64('2008-08-08')
        assert Timestamp('2008-08-08') == df.loc[0, 'c']
        assert Timestamp('2008-08-08') == df.loc[1, 'c']
        with pytest.raises(TypeError, match='Invalid value'):
            df.loc[2, 'c'] = date(2005, 5, 5)

    @pytest.mark.parametrize('idxer', ['var', ['var']])
    def test_loc_setitem_datetimeindex_tz(self, idxer: Union[str, List[str]], tz_naive_fixture: Any) -> None:
        tz: Any = tz_naive_fixture
        idx: DatetimeIndex = date_range(start='2015-07-12', periods=3, freq='h', tz=tz)
        expected: DataFrame = DataFrame(1.2, index=idx, columns=['var'])
        result: DataFrame = DataFrame(index=idx, columns=['var'], dtype=np.float64)
        if idxer == 'var':
            with pytest.raises(TypeError, match='Invalid value'):
                result.loc[:, idxer] = expected
        else:
            result.loc[:, idxer] = expected
            tm.assert_frame_equal(result, expected)

    def test_loc_setitem_time_key(self) -> None:
        index: DatetimeIndex = date_range('2012-01-01', '2012-01-05', freq='30min')
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((len(index), 5)), index=index)
        akey: time = time(12, 0, 0)
        bkey: slice = slice(time(13, 0, 0), time(14, 0, 0))
        ainds: List[int] = [24, 72, 120, 168]
        binds: List[int] = [26, 27, 28, 74, 75, 76, 122, 123, 124, 170, 171, 172]
        result: DataFrame = df.copy()
        result.loc[akey] = 0
        result = result.loc[akey]
        expected: DataFrame = df.loc[akey].copy()
        expected.loc[:] = 0
        tm.assert_frame_equal(result, expected)
        result = df.copy()
        result.loc[akey] = 0
        result.loc[akey] = df.iloc[ainds]
        tm.assert_frame_equal(result, df)
        result = df.copy()
        result.loc[bkey] = 0
        result = result.loc[bkey]
        expected = df.loc[bkey].copy()
        expected.loc[:] = 0
        tm.assert_frame_equal(result, expected)
        result = df.copy()
        result.loc[bkey] = 0
        result.loc[bkey] = df.iloc[binds]
        tm.assert_frame_equal(result, df)

    @pytest.mark.parametrize('key', ['A', ['A'], ('A', slice(None))])
    def test_loc_setitem_unsorted_multiindex_columns(self, key: Any) -> None:
        mi: MultiIndex = MultiIndex.from_tuples([('A', 4), ('B', '3'), ('A', '2')])
        df: DataFrame = DataFrame([[1, 2, 3], [4, 5, 6]], columns=mi)
        obj: DataFrame = df.copy()
        obj.loc[:, key] = np.zeros((2, 2), dtype='int64')
        expected: DataFrame = DataFrame([[0, 2, 0], [0, 5, 0]], columns=mi)
        tm.assert_frame_equal(obj, expected)
        df = df.sort_index(axis=1)
        df.loc[:, key] = np.zeros((2, 2), dtype='int64')
        expected = expected.sort_index(axis=1)
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_uint_drop(self, any_int_numpy_dtype: Any) -> None:
        series: Series = Series([1, 2, 3], dtype=any_int_numpy_dtype)
        series.loc[0] = 4
        expected: Series = Series([4, 2, 3], dtype=any_int_numpy_dtype)
        tm.assert_series_equal(series, expected)

    def test_loc_setitem_td64_non_nano(self) -> None:
        ser: Series = Series(10 * [np.timedelta64(10, 'm')])
        ser.loc[[1, 2, 3]] = np.timedelta64(20, 'm')
        expected: Series = Series(10 * [np.timedelta64(10, 'm')])
        expected.loc[[1, 2, 3]] = Timedelta(np.timedelta64(20, 'm'))
        tm.assert_series_equal(ser, expected)

    def test_loc_setitem_2d_to_1d_raises(self) -> None:
        data: np.ndarray = np.random.default_rng(2).standard_normal((2, 2))
        ser: Series = Series(range(2), dtype='float64')
        msg: str = 'setting an array element with a sequence.'
        with pytest.raises(ValueError, match=msg):
            ser.loc[range(2)] = data
        with pytest.raises(ValueError, match=msg):
            ser.loc[:] = data

    def test_loc_getitem_interval_index(self) -> None:
        index: pd.IntervalIndex = pd.interval_range(start=0, periods=3)
        df: DataFrame = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=index, columns=['A', 'B', 'C'])
        expected: Any = 1
        result: Any = df.loc[0.5, 'A']
        tm.assert_almost_equal(result, expected)

    def test_loc_getitem_interval_index2(self) -> None:
        index: pd.IntervalIndex = pd.interval_range(start=0, periods=3, closed='both')
        df: DataFrame = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=index, columns=['A', 'B', 'C'])
        index_exp: pd.IntervalIndex = pd.interval_range(start=0, periods=2, freq=1, closed='both')
        expected: Series = Series([1, 4], index=index_exp, name='A')
        result: Any = df.loc[1, 'A']
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('tpl', [(1,), (1, 2)])
    def test_loc_getitem_index_single_double_tuples(self, tpl: Tuple[Any, ...]) -> None:
        idx: Index = Index([(1,), (1, 2)], name='A', tupleize_cols=False)
        df: DataFrame = DataFrame(index=idx)
        result: DataFrame = df.loc[[tpl]]
        idx_exp: Index = Index([tpl], name='A', tupleize_cols=False)
        expected: DataFrame = DataFrame(index=idx_exp)
        tm.assert_frame_equal(result, expected)

    def test_loc_getitem_index_namedtuple(self) -> None:
        IndexType = namedtuple('IndexType', ['a', 'b'])
        idx1: Any = IndexType('foo', 'bar')
        idx2: Any = IndexType('baz', 'bof')
        index: Index = Index([idx1, idx2], name='composite_index', tupleize_cols=False)
        df: DataFrame = DataFrame([(1, 2), (3, 4)], index=index, columns=['A', 'B'])
        result: Any = df.loc[IndexType('foo', 'bar')]['A']
        assert result == 1

    def test_loc_setitem_single_column_mixed(self, using_infer_string: bool) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 3)), 
                                  index=['a', 'b', 'c', 'd', 'e'], columns=['foo', 'bar', 'baz'])
        df['str'] = 'qux'
        df.loc[df.index[::2], 'str'] = np.nan
        expected: Any = Series([np.nan, 'qux', np.nan, 'qux', np.nan], dtype=object if not using_infer_string else 'str').values
        tm.assert_almost_equal(df['str'].values, expected)

    def test_loc_setitem_cast2(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).random((30, 3)), columns=tuple('ABC'))
        df['event'] = np.nan
        with pytest.raises(TypeError, match='Invalid value'):
            df.loc[10, 'event'] = 'foo'

    def test_loc_setitem_cast3(self) -> None:
        df: DataFrame = DataFrame({'one': np.arange(6, dtype=np.int8)})
        df.loc[1, 'one'] = 6
        assert df.dtypes.one == np.dtype(np.int8)
        df.one = np.int8(7)
        assert df.dtypes.one == np.dtype(np.int8)

    def test_loc_setitem_range_key(self, frame_or_series: Callable[..., Any]) -> None:
        obj: Any = frame_or_series(range(5), index=[3, 4, 1, 0, 2])
        values: Any = [9, 10, 11]
        if obj.ndim == 2:
            values = [[9], [10], [11]]
        obj.loc[range(3)] = values
        expected: Any = frame_or_series([0, 1, 10, 9, 11], index=obj.index)
        tm.assert_equal(obj, expected)

    def test_loc_setitem_numpy_frame_categorical_value(self) -> None:
        df: DataFrame = DataFrame({'a': [1, 1, 1, 1, 1], 'b': ['a', 'a', 'a', 'a', 'a']})
        df.loc[1:2, 'a'] = Categorical([2, 2], categories=[1, 2])
        expected: DataFrame = DataFrame({'a': [1, 2, 2, 1, 1], 'b': ['a', 'a', 'a', 'a', 'a']})
        tm.assert_frame_equal(df, expected)


class TestLocWithEllipsis:
    @pytest.fixture
    def indexer(self, indexer_li: Callable[..., Any]) -> Callable[..., Any]:
        return indexer_li

    @pytest.fixture
    def obj(self, series_with_simple_index: Series, frame_or_series: Callable[..., Any]) -> Union[Series, DataFrame]:
        obj: Any = series_with_simple_index
        if frame_or_series is not Series:
            obj = obj.to_frame()
        return obj

    def test_loc_iloc_getitem_ellipsis(self, obj: Union[Series, DataFrame], indexer: Callable[[Any], Any]) -> None:
        result: Any = indexer(obj)[...]
        tm.assert_equal(result, obj)

    @pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
    def test_loc_iloc_getitem_leading_ellipses(self, series_with_simple_index: Series, indexer: Callable[[Any], Any]) -> None:
        obj: Series = series_with_simple_index
        key: Any = 0 if (indexer is tm.iloc or len(obj) == 0) else obj.index[0]
        if indexer is tm.loc and obj.index.inferred_type == 'boolean':
            return
        if indexer is tm.loc and isinstance(obj.index, MultiIndex):
            msg: str = 'MultiIndex does not support indexing with Ellipsis'
            with pytest.raises(NotImplementedError, match=msg):
                _ = indexer(obj)[..., [key]]
        elif len(obj) != 0:
            result: Any = indexer(obj)[..., [key]]
            expected: Any = indexer(obj)[[key]]
            tm.assert_series_equal(result, expected)
        key2: Any = 0 if indexer is tm.iloc else obj.name
        df: DataFrame = obj.to_frame()
        result = indexer(df)[..., [key2]]
        expected = indexer(df)[:, [key2]]
        tm.assert_frame_equal(result, expected)

    def test_loc_iloc_getitem_ellipses_only_one_ellipsis(self, obj: Union[Series, DataFrame], indexer: Callable[[Any], Any]) -> None:
        key: Any = 0 if (indexer is tm.iloc or len(obj) == 0) else obj.index[0]
        with pytest.raises(IndexingError, match=_one_ellipsis_message):
            _ = indexer(obj)[..., ...]
        with pytest.raises(IndexingError, match=_one_ellipsis_message):
            _ = indexer(obj)[..., [key], ...]
        with pytest.raises(IndexingError, match=_one_ellipsis_message):
            _ = indexer(obj)[..., ..., key]
        with pytest.raises(IndexingError, match='Too many indexers'):
            _ = indexer(obj)[key, ..., ...]

class TestLocWithMultiIndex:
    @pytest.mark.parametrize('keys, expected', [
        (['b', 'a'], [['b', 'b', 'a', 'a'], [1, 2, 1, 2]]),
        (['a', 'b'], [['a', 'a', 'b', 'b'], [1, 2, 1, 2]]),
        ((['a', 'b'], [1, 2]), [['a', 'a', 'b', 'b'], [1, 2, 1, 2]]),
        ((['a', 'b'], [2, 1]), [['a', 'a', 'b', 'b'], [2, 1, 2, 1]]),
        ((['b', 'a'], [2, 1]), [['b', 'b', 'a', 'a'], [2, 1, 2, 1]]),
        ((['b', 'a'], [1, 2]), [['b', 'b', 'a', 'a'], [1, 2, 1, 2]]),
        ((['c', 'a'], [2, 1]), [['c', 'a', 'a'], [1, 2, 1]])
    ])
    @pytest.mark.parametrize('dim', ['index', 'columns'])
    def test_loc_getitem_multilevel_index_order(self, dim: str, keys: Any, expected: List[List[Any]]) -> None:
        kwargs: dict = {dim: [['c', 'a', 'a', 'b', 'b'], [1, 1, 2, 1, 2]]}
        df: DataFrame = DataFrame(np.arange(25).reshape(5, 5), **kwargs)
        exp_index: MultiIndex = MultiIndex.from_arrays(expected)
        if dim == 'index':
            res: DataFrame = df.loc[keys, :]
            tm.assert_index_equal(res.index, exp_index)
        elif dim == 'columns':
            res = df.loc[:, keys]
            tm.assert_index_equal(res.columns, exp_index)

    def test_loc_preserve_names(self, multiindex_year_month_day_dataframe_random_data: DataFrame) -> None:
        ymd: DataFrame = multiindex_year_month_day_dataframe_random_data
        result: DataFrame = ymd.loc[2000]
        result2: Any = ymd['A'].loc[2000]
        assert result.index.names == ymd.index.names[1:]
        assert result2.index.names == ymd.index.names[1:]
        result = ymd.loc[2000, 2]
        result2 = ymd['A'].loc[2000, 2]
        assert result.index.name == ymd.index.names[2]
        assert result2.index.name == ymd.index.names[2]

    def test_loc_getitem_multiindex_nonunique_len_zero(self) -> None:
        mi: MultiIndex = MultiIndex.from_product([[0], [1, 1]])
        ser: Series = Series(0, index=mi)
        res: Series = ser.loc[[]]
        expected: Series = ser[:0]
        tm.assert_series_equal(res, expected)
        res2: Series = ser.loc[ser.iloc[0:0]]
        tm.assert_series_equal(res2, expected)

    def test_loc_getitem_access_none_value_in_multiindex(self) -> None:
        ser: Series = Series([None], MultiIndex.from_arrays([['Level1'], ['Level2']]))
        result: Any = ser.loc['Level1', 'Level2']
        assert result is None
        midx: MultiIndex = MultiIndex.from_product([['Level1'], ['Level2_a', 'Level2_b']])
        ser = Series([None] * len(midx), dtype=object, index=midx)
        result = ser.loc['Level1', 'Level2_a']
        assert result is None
        ser = Series([1] * len(midx), dtype=object, index=midx)
        result = ser.loc['Level1', 'Level2_a']
        assert result == 1

    def test_loc_setitem_multiindex_slice(self) -> None:
        index: MultiIndex = MultiIndex.from_tuples(
            list(zip(['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'], ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two'])),
            names=['first', 'second']
        )
        result: Series = Series([1] * 8, index=index)
        result.loc[('baz', 'one'):('foo', 'two')] = 100
        expected: Series = Series([1, 1, 100, 100, 100, 100, 1, 1], index=index)
        tm.assert_series_equal(result, expected)

    def test_loc_getitem_slice_datetime_objs_with_datetimeindex(self) -> None:
        times: DatetimeIndex = date_range('2000-01-01', freq='10min', periods=100000)
        ser: Series = Series(range(100000), times)
        result: Series = ser.loc[datetime(1900, 1, 1):datetime(2100, 1, 1)]
        tm.assert_series_equal(result, ser)

    def test_loc_getitem_datetime_string_with_datetimeindex(self) -> None:
        df: DataFrame = DataFrame({'a': range(10), 'b': range(10)}, index=date_range('2010-01-01', '2010-01-10'))
        result: DataFrame = df.loc[['2010-01-01', '2010-01-05'], ['a', 'b']]
        expected: DataFrame = DataFrame({'a': [0, 4], 'b': [0, 4]}, index=DatetimeIndex(['2010-01-01', '2010-01-05']).as_unit('ns'))
        tm.assert_frame_equal(result, expected)

    def test_loc_getitem_sorted_index_level_with_duplicates(self) -> None:
        mi: MultiIndex = MultiIndex.from_tuples([('foo', 'bar'), ('foo', 'bar'), ('bah', 'bam'), ('bah', 'bam'), ('foo', 'bar'), ('bah', 'bam')], names=['A', 'B'])
        df: DataFrame = DataFrame([[1.0, 1], [2.0, 2], [3.0, 3], [4.0, 4], [5.0, 5], [6.0, 6]], index=mi, columns=['C', 'D'])
        df = df.sort_index(level=0)
        expected: DataFrame = DataFrame([[1.0, 1], [2.0, 2], [5.0, 5]], columns=['C', 'D'], index=mi.take([0, 1, 4]))
        result: Any = df.loc['foo', 'bar']
        tm.assert_frame_equal(result, expected)

    def test_additional_element_to_categorical_series_loc(self) -> None:
        result: Series = Series(['a', 'b', 'c'], dtype='category')
        result.loc[3] = 0
        expected: Series = Series(['a', 'b', 'c', 0], dtype='object')
        tm.assert_series_equal(result, expected)

    def test_additional_categorical_element_loc(self) -> None:
        result: Series = Series(['a', 'b', 'c'], dtype='category')
        result.loc[3] = 'a'
        expected: Series = Series(['a', 'b', 'c', 'a'], dtype='category')
        tm.assert_series_equal(result, expected)

    def test_loc_set_nan_in_categorical_series(self, any_numeric_ea_dtype: Any) -> None:
        srs: Series = Series([1, 2, 3], dtype=CategoricalDtype(Index([1, 2, 3], dtype=any_numeric_ea_dtype)))
        srs.loc[3] = np.nan
        expected: Series = Series([1, 2, 3, np.nan], dtype=CategoricalDtype(Index([1, 2, 3], dtype=any_numeric_ea_dtype)))
        tm.assert_series_equal(srs, expected)
        srs.loc[1] = np.nan
        expected = Series([1, np.nan, 3, np.nan], dtype=CategoricalDtype(Index([1, 2, 3], dtype=any_numeric_ea_dtype)))
        tm.assert_series_equal(srs, expected)

    @pytest.mark.parametrize('na', (np.nan, pd.NA, None, pd.NaT))
    def test_loc_consistency_series_enlarge_set_into(self, na: Any) -> None:
        srs_enlarge: Series = Series(['a', 'b', 'c'], dtype='category')
        srs_enlarge.loc[3] = na
        srs_setinto: Series = Series(['a', 'b', 'c', 'a'], dtype='category')
        srs_setinto.loc[3] = na
        tm.assert_series_equal(srs_enlarge, srs_setinto)
        expected: Series = Series(['a', 'b', 'c', na], dtype='category')
        tm.assert_series_equal(srs_enlarge, expected)

    def test_loc_getitem_preserves_index_level_category_dtype(self) -> None:
        df: DataFrame = DataFrame(
            data=np.arange(2, 22, 2),
            index=MultiIndex(
                levels=[CategoricalIndex(['a', 'b']), range(10)],
                codes=[[0] * 5 + [1] * 5, list(range(10))],
                names=['Index1', 'Index2']
            )
        )
        expected: pd.CategoricalIndex = CategoricalIndex(['a', 'b'], categories=['a', 'b'], ordered=False, name='Index1', dtype='category')
        result: pd.CategoricalIndex = df.index.levels[0]
        tm.assert_index_equal(result, expected)
        result = df.loc[['a']].index.levels[0]
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('lt_value', [30, 10])
    def test_loc_multiindex_levels_contain_values_not_in_index_anymore(self, lt_value: int) -> None:
        df: DataFrame = DataFrame({'a': [12, 23, 34, 45]}, index=[list('aabb'), [0, 1, 2, 3]])
        with pytest.raises(KeyError, match="\\['b'\\] not in index"):
            df.loc[df['a'] < lt_value, :].loc[['b'], :]

    def test_loc_multiindex_null_slice_na_level(self) -> None:
        lev1: np.ndarray = np.array([np.nan, np.nan])
        lev2: List[str] = ['bar', 'baz']
        mi: MultiIndex = MultiIndex.from_arrays([lev1, lev2])
        ser: Series = Series([0, 1], index=mi)
        result: Series = ser.loc[:, 'bar']
        expected: Series = Series([0], index=[np.nan])
        tm.assert_series_equal(result, expected)

    def test_loc_drops_level(self) -> None:
        mi: MultiIndex = MultiIndex.from_product([list('ab'), list('xy'), [1, 2]], names=['ab', 'xy', 'num'])
        ser: Series = Series(range(8), index=mi)
        loc_result: Any = ser.loc['a', :, :]
        expected_index: Index = ser.index.droplevel(0)[:4]
        tm.assert_index_equal(loc_result.index, expected_index)


class TestLocSetitemWithExpansion:
    def test_loc_setitem_with_expansion_large_dataframe(self, monkeypatch: Any) -> None:
        size_cutoff: int = 50
        with monkeypatch.context():
            monkeypatch.setattr(libindex, '_SIZE_CUTOFF', size_cutoff)
            result: DataFrame = DataFrame({'x': range(size_cutoff)}, dtype='int64')
            result.loc[size_cutoff] = size_cutoff
        expected: DataFrame = DataFrame({'x': range(size_cutoff + 1)}, dtype='int64')
        tm.assert_frame_equal(result, expected)

    def test_loc_setitem_empty_series(self) -> None:
        ser: Series = Series(dtype=object)
        ser.loc[1] = 1
        tm.assert_series_equal(ser, Series([1], index=range(1, 2)))
        ser.loc[3] = 3
        tm.assert_series_equal(ser, Series([1, 3], index=[1, 3]))

    def test_loc_setitem_empty_series_float(self) -> None:
        ser: Series = Series(dtype=object)
        ser.loc[1] = 1.0
        tm.assert_series_equal(ser, Series([1.0], index=range(1, 2)))
        ser.loc[3] = 3.0
        tm.assert_series_equal(ser, Series([1.0, 3.0], index=[1, 3]))

    def test_loc_setitem_empty_series_str_idx(self) -> None:
        ser: Series = Series(dtype=object)
        ser.loc['foo'] = 1
        tm.assert_series_equal(ser, Series([1], index=Index(['foo'], dtype=object)))
        ser.loc['bar'] = 3
        tm.assert_series_equal(ser, Series([1, 3], index=Index(['foo', 'bar'], dtype=object)))
        ser.loc[3] = 4
        tm.assert_series_equal(ser, Series([1, 3, 4], index=Index(['foo', 'bar', 3], dtype=object)))

    def test_loc_setitem_incremental_with_dst(self) -> None:
        base: datetime = datetime(2015, 11, 1, tzinfo=gettz('US/Pacific'))
        idxs: List[datetime] = [base + timedelta(seconds=i * 900) for i in range(16)]
        result: Series = Series([0], index=[idxs[0]])
        for ts in idxs:
            result.loc[ts] = 1
        expected: Series = Series(1, index=idxs)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        'conv', [
            lambda x: x,
            lambda x: x.to_datetime64(),
            lambda x: x.to_pydatetime(),
            lambda x: np.datetime64(x)
        ],
        ids=['self', 'to_datetime64', 'to_pydatetime', 'np.datetime64']
    )
    def test_loc_setitem_datetime_keys_cast(self, conv: Callable[[Timestamp], Any]) -> None:
        dt1: Timestamp = Timestamp('20130101 09:00:00')
        dt2: Timestamp = Timestamp('20130101 10:00:00')
        df: DataFrame = DataFrame()
        df.loc[conv(dt1), 'one'] = 100
        df.loc[conv(dt2), 'one'] = 200
        expected: DataFrame = DataFrame({'one': [100.0, 200.0]}, index=Index([conv(dt1), conv(dt2)], dtype=object), columns=Index(['one'], dtype=object))
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_categorical_column_retains_dtype(self, ordered: bool) -> None:
        result: DataFrame = DataFrame({'A': [1]})
        result.loc[:, 'B'] = Categorical(['b'], ordered=ordered)
        expected: DataFrame = DataFrame({'A': [1], 'B': Categorical(['b'], ordered=ordered)})
        tm.assert_frame_equal(result, expected)

    def test_loc_setitem_with_expansion_and_existing_dst(self) -> None:
        start: Timestamp = Timestamp('2017-10-29 00:00:00+0200', tz='Europe/Madrid')
        end: Timestamp = Timestamp('2017-10-29 03:00:00+0100', tz='Europe/Madrid')
        ts: Timestamp = Timestamp('2016-10-10 03:00:00', tz='Europe/Madrid')
        idx: DatetimeIndex = date_range(start, end, inclusive='left', freq='h')
        assert ts not in idx
        result: DataFrame = DataFrame(index=idx, columns=['value'])
        result.loc[ts, 'value'] = 12
        expected: DataFrame = DataFrame([np.nan] * len(idx) + [12], index=idx.append(DatetimeIndex([ts])), columns=['value'], dtype=object)
        tm.assert_frame_equal(result, expected)

    def test_setitem_with_expansion(self) -> None:
        df: DataFrame = DataFrame(data=to_datetime(['2015-03-30 20:12:32', '2015-03-12 00:11:11']), columns=['time'])
        df['new_col'] = ['new', 'old']
        df.time = df.set_index('time').index.tz_localize('UTC')
        v: Any = df[df.new_col == 'new'].set_index('time').index.tz_convert('US/Pacific')
        df2: DataFrame = df.copy()
        df2.loc[df2.new_col == 'new', 'time'] = v
        expected: Series = Series([v[0].tz_convert('UTC'), df.loc[1, 'time']], name='time')
        tm.assert_series_equal(df2.time, expected)
        v = df.loc[df.new_col == 'new', 'time'] + pd.Timedelta('1s').as_unit('s')
        df.loc[df.new_col == 'new', 'time'] = v
        tm.assert_series_equal(df.loc[df.new_col == 'new', 'time'], v)

    def test_loc_setitem_with_expansion_inf_upcast_empty(self) -> None:
        df: DataFrame = DataFrame()
        df.loc[0, 0] = 1
        df.loc[1, 1] = 2
        df.loc[0, np.inf] = 3
        result: Index = df.columns
        expected: Index = Index([0, 1, np.inf], dtype=np.float64)
        tm.assert_index_equal(result, expected)

    @pytest.mark.filterwarnings('ignore:indexing past lexsort depth')
    def test_loc_setitem_with_expansion_nonunique_index(self, index: Any) -> None:
        if not len(index):
            pytest.skip('Not relevant for empty Index')
        index = index.repeat(2)
        N: int = len(index)
        arr: np.ndarray = np.arange(N).astype(np.int64)
        orig: DataFrame = DataFrame(arr, index=index)
        key: Any = 'kapow'
        assert key not in index
        exp_index: Any = index.insert(len(index), key)
        if isinstance(index, MultiIndex):
            assert exp_index[-1][0] == key
        else:
            assert exp_index[-1] == key
        exp_data: np.ndarray = np.arange(N + 1).astype(np.float64)
        expected: DataFrame = DataFrame(exp_data, index=exp_index)
        df: DataFrame = orig.copy()
        df.loc[key, 0] = N
        tm.assert_frame_equal(df, expected)
        ser: Series = orig.copy()[0]
        ser.loc[key] = N
        expected_ser: Series = expected[0].astype(np.int64)
        tm.assert_series_equal(ser, expected_ser)
        df = orig.copy()
        df.loc[key, 1] = N
        expected = DataFrame({0: list(arr) + [np.nan], 1: [np.nan] * N + [float(N)]}, index=exp_index)
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_with_expansion_preserves_nullable_int(self, any_numeric_ea_dtype: Any) -> None:
        ser: Series = Series([0, 1, 2, 3], dtype=any_numeric_ea_dtype)
        df: DataFrame = DataFrame({'data': ser})
        result: DataFrame = DataFrame(index=df.index)
        result.loc[df.index, 'data'] = ser
        tm.assert_frame_equal(result, df, check_column_type=False)
        result = DataFrame(index=df.index)
        result.loc[df.index, 'data'] = ser._values
        tm.assert_frame_equal(result, df, check_column_type=False)

    def test_loc_setitem_ea_not_full_column(self) -> None:
        df: DataFrame = DataFrame({'A': range(5)})
        val: Any = date_range('2016-01-01', periods=3, tz='US/Pacific')
        df.loc[[0, 1, 2], 'B'] = val
        bex: DatetimeIndex = val.append(DatetimeIndex([pd.NaT, pd.NaT], dtype=val.dtype))
        expected: DataFrame = DataFrame({'A': range(5), 'B': bex})
        assert expected.dtypes['B'] == val.dtype
        tm.assert_frame_equal(df, expected)

    class TestLocCallable:
        def test_frame_loc_getitem_callable(self) -> None:
            df: DataFrame = DataFrame({'A': [1, 2, 3, 4], 'B': list('aabb'), 'C': [1, 2, 3, 4]})
            res: DataFrame = df.loc[lambda x: x.A > 2]
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
            assert res == df.loc[1, 'A']

        def test_frame_loc_getitem_callable_mixture(self) -> None:
            df: DataFrame = DataFrame({'A': [1, 2, 3, 4], 'B': list('aabb'), 'C': [1, 2, 3, 4]})
            res: DataFrame = df.loc[lambda x: x.A > 2, ['A', 'B']]
            tm.assert_frame_equal(res, df.loc[df.A > 2, ['A', 'B']])
            res = df.loc[[2, 3], lambda x: ['A', 'B']]
            tm.assert_frame_equal(res, df.loc[[2, 3], ['A', 'B']])
            res = df.loc[3, lambda x: ['A', 'B']]
            tm.assert_series_equal(res, df.loc[3, ['A', 'B']])

        def test_frame_loc_getitem_callable_labels(self) -> None:
            df: DataFrame = DataFrame({'X': [1, 2, 3, 4], 'Y': list('aabb')}, index=list('ABCD'))
            res: DataFrame = df.loc[lambda x: ['A', 'C']]
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

        def test_frame_loc_setitem_callable(self) -> None:
            df: DataFrame = DataFrame({'X': [1, 2, 3, 4], 'Y': Series(list('aabb'), dtype=object)}, index=list('ABCD'))
            res: DataFrame = df.copy()
            res.loc[lambda x: ['A', 'C']] = -20
            exp: DataFrame = df.copy()
            exp.loc[['A', 'C']] = -20
            tm.assert_frame_equal(res, exp)
            res = df.copy()
            res.loc[lambda x: ['A', 'C'], :] = 20
            exp = df.copy()
            exp.loc[['A', 'C'], :] = 20
            tm.assert_frame_equal(res, exp)
            res = df.copy()
            res.loc[lambda x: ['A', 'C'], lambda x: 'X'] = -1
            exp = df.copy()
            exp.loc[['A', 'C'], 'X'] = -1
            tm.assert_frame_equal(res, exp)
            res = df.copy()
            res.loc[lambda x: ['A', 'C'], lambda x: ['X']] = [5, 10]
            exp = df.copy()
            exp.loc[['A', 'C'], ['X']] = [5, 10]
            tm.assert_frame_equal(res, exp)
            res = df.copy()
            res.loc[['A', 'C'], lambda x: 'X'] = np.array([-1, -2])
            exp = df.copy()
            exp.loc[['A', 'C'], 'X'] = np.array([-1, -2])
            tm.assert_frame_equal(res, exp)
            res = df.copy()
            res.loc[['A', 'C'], lambda x: ['X']] = 10
            exp = df.copy()
            exp.loc[['A', 'C'], ['X']] = 10
            tm.assert_frame_equal(res, exp)
            res = df.copy()
            res.loc[lambda x: ['A', 'C'], 'X'] = -2
            exp = df.copy()
            exp.loc[['A', 'C'], 'X'] = -2
            tm.assert_frame_equal(res, exp)
            res = df.copy()
            res.loc[lambda x: ['A', 'C'], ['X']] = -4
            exp = df.copy()
            exp.loc[['A', 'C'], ['X']] = -4
            tm.assert_frame_equal(res, exp)

class TestPartialStringSlicing:
    def test_loc_getitem_partial_string_slicing_datetimeindex(self) -> None:
        df: DataFrame = DataFrame({'col1': ['a', 'b', 'c'], 'col2': [1, 2, 3]}, index=to_datetime(['2020-08-01', '2020-07-02', '2020-08-05']))
        expected: DataFrame = DataFrame({'col1': ['a', 'c'], 'col2': [1, 3]}, index=to_datetime(['2020-08-01', '2020-08-05']))
        result: DataFrame = df.loc['2020-08']
        tm.assert_frame_equal(result, expected)

    def test_loc_getitem_partial_string_slicing_with_periodindex(self) -> None:
        pi: PeriodIndex = pd.period_range(start='2017-01-01', end='2018-01-01', freq='M')
        ser: Series = pi.to_series()
        result: Series = ser.loc[:'2017-12']
        expected: Series = ser.iloc[:-1]
        tm.assert_series_equal(result, expected)

    def test_loc_getitem_partial_string_slicing_with_timedeltaindex(self) -> None:
        ix: pd.TimedeltaIndex = timedelta_range(start='1 day', end='2 days', freq='1h')
        ser: Series = ix.to_series()
        result: Series = ser.loc[:'1 days']
        expected: Series = ser.iloc[:-1]
        tm.assert_series_equal(result, expected)

    def test_loc_getitem_str_timedeltaindex(self) -> None:
        df: DataFrame = DataFrame({'x': range(3)}, index=to_timedelta(range(3), unit='days'))
        expected: Series = df.iloc[0]
        sliced: Series = df.loc['0 days']
        tm.assert_series_equal(sliced, expected)

    @pytest.mark.parametrize('indexer_end', [None, '2020-01-02 23:59:59.999999999'])
    def test_loc_getitem_partial_slice_non_monotonicity(self, tz_aware_fixture: Any, indexer_end: Optional[str], frame_or_series: Callable[..., Any]) -> None:
        obj: Any = frame_or_series([1] * 5, index=DatetimeIndex([Timestamp('2019-12-30'), Timestamp('2020-01-01'), Timestamp('2019-12-25'), Timestamp('2020-01-02 23:59:59.999999999'), Timestamp('2019-12-19')], tz=tz_aware_fixture))
        expected: Any = frame_or_series([1] * 2, index=DatetimeIndex([Timestamp('2020-01-01'), Timestamp('2020-01-02 23:59:59.999999999')], tz=tz_aware_fixture))
        indexer: slice = slice('2020-01-01', indexer_end)
        result = obj[indexer]
        tm.assert_equal(result, expected)
        result = obj.loc[indexer]
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize('value', [1, 1.5])
    def test_loc_getitem_slice_labels_int_in_object_index(self, frame_or_series: Callable[..., Any], value: Union[int, float]) -> None:
        obj: Any = frame_or_series(range(4), index=[value, 'first', 2, 'third'])
        result: Any = obj.loc[value:'third']
        expected: Any = frame_or_series(range(4), index=[value, 'first', 2, 'third'])
        tm.assert_equal(result, expected)

    def test_loc_getitem_slice_columns_mixed_dtype(self) -> None:
        df: DataFrame = DataFrame({'test': 1, 1: 2, 2: 3}, index=[0])
        expected: DataFrame = DataFrame(data=[[2, 3]], index=[0], columns=Index([1, 2], dtype=object))
        tm.assert_frame_equal(df.loc[:, 1:], expected)

class TestLocBooleanLabelsAndSlices:
    @pytest.mark.parametrize('bool_value', [True, False])
    def test_loc_bool_incompatible_index_raises(self, index: Index, frame_or_series: Callable[..., Any], bool_value: bool) -> None:
        message: str = f'{bool_value}: boolean label can not be used without a boolean index'
        if index.inferred_type != 'boolean':
            obj: Any = frame_or_series(index=index, dtype='object')
            with pytest.raises(KeyError, match=message):
                obj.loc[bool_value]

    @pytest.mark.parametrize('bool_value', [True, False])
    def test_loc_bool_should_not_raise(self, frame_or_series: Callable[..., Any], bool_value: bool) -> None:
        obj: Any = frame_or_series(index=Index([True, False], dtype='boolean'), dtype='object')
        obj.loc[bool_value]

    def test_loc_bool_slice_raises(self, index: Index, frame_or_series: Callable[..., Any]) -> None:
        message: str = 'slice\\(True, False, None\\): boolean values can not be used in a slice'
        obj: Any = frame_or_series(index=index, dtype='object')
        with pytest.raises(TypeError, match=message):
            obj.loc[True:False]

class TestLocBooleanMask:
    def test_loc_setitem_bool_mask_timedeltaindex(self) -> None:
        df: DataFrame = DataFrame({'x': range(10)})
        df.index = to_timedelta(range(10), unit='s')
        conditions: List[Any] = [df['x'] > 3, df['x'] == 3, df['x'] < 3]
        expected_data: List[List[Any]] = [[0, 1, 2, 3, 10, 10, 10, 10, 10, 10],
                                          [0, 1, 2, 10, 4, 5, 6, 7, 8, 9],
                                          [10, 10, 10, 3, 4, 5, 6, 7, 8, 9]]
        for cond, data in zip(conditions, expected_data):
            result: DataFrame = df.copy()
            result.loc[cond, 'x'] = 10
            expected: DataFrame = DataFrame(data, index=to_timedelta(range(10), unit='s'), columns=['x'], dtype='int64')
            tm.assert_frame_equal(expected, result)

    @pytest.mark.parametrize('tz', [None, 'UTC'])
    def test_loc_setitem_mask_with_datetimeindex_tz(self, tz: Optional[str]) -> None:
        mask: np.ndarray = np.array([True, False, True, False])
        idx: DatetimeIndex = date_range('20010101', periods=4, tz=tz)
        df: DataFrame = DataFrame({'a': np.arange(4)}, index=idx).astype('float64')
        result: DataFrame = df.copy()
        result.loc[mask, :] = df.loc[mask, :]
        tm.assert_frame_equal(result, df)
        result = df.copy()
        result.loc[mask] = df.loc[mask]
        tm.assert_frame_equal(result, df)

    def test_loc_setitem_mask_and_label_with_datetimeindex(self) -> None:
        df: DataFrame = DataFrame(np.arange(6.0).reshape(3, 2), columns=list('AB'), index=date_range('1/1/2000', periods=3, freq='1h'))
        expected: DataFrame = df.copy()
        expected['C'] = [expected.index[0]] + [pd.NaT, pd.NaT]
        mask: Series = df.A < 1
        df.loc[mask, 'C'] = df.loc[mask].index
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_mask_td64_series_value(self) -> None:
        td1: Timedelta = Timedelta(0)
        td2: Timedelta = Timedelta(28767471428571405)
        df: DataFrame = DataFrame({'col': Series([td1, td2])})
        df_copy: DataFrame = df.copy()
        ser: Series = Series([td1])
        expected: Any = df['col'].iloc[1]._value
        df.loc[[True, False]] = ser
        result: Any = df['col'].iloc[1]._value
        assert expected == result
        tm.assert_frame_equal(df, df_copy)

    def test_loc_setitem_boolean_and_column(self, float_frame: DataFrame) -> None:
        expected: DataFrame = float_frame.copy()
        mask: Series = float_frame['A'] > 0
        float_frame.loc[mask, 'B'] = 0
        values: np.ndarray = expected.values.copy()
        values[mask.values, 1] = 0
        expected = DataFrame(values, index=expected.index, columns=expected.columns)
        tm.assert_frame_equal(float_frame, expected)

    def test_loc_setitem_ndframe_values_alignment(self) -> None:
        df: DataFrame = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        df.loc[[False, False, True], ['a']] = DataFrame({'a': [10, 20, 30]}, index=[2, 1, 0])
        expected: DataFrame = DataFrame({'a': [1, 2, 10], 'b': [4, 5, 6]})
        tm.assert_frame_equal(df, expected)
        df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        df.loc[[False, False, True], ['a']] = Series([10, 11, 12], index=[2, 1, 0])
        tm.assert_frame_equal(df, expected)
        df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        df.loc[[False, False, True], 'a'] = Series([10, 11, 12], index=[2, 1, 0])
        tm.assert_frame_equal(df, expected)
        df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        df_orig: DataFrame = df.copy()
        ser: Series = df['a']
        ser.loc[[False, False, True]] = Series([10, 11, 12], index=[2, 1, 0])
        tm.assert_frame_equal(df, df_orig)

    def test_loc_indexer_empty_broadcast(self) -> None:
        df: DataFrame = DataFrame({'a': [], 'b': []}, dtype=object)
        expected: DataFrame = df.copy()
        df.loc[np.array([], dtype=np.bool_), ['a']] = df['a'].copy()
        tm.assert_frame_equal(df, expected)

    def test_loc_indexer_all_false_broadcast(self) -> None:
        df: DataFrame = DataFrame({'a': ['x'], 'b': ['y']}, dtype=object)
        expected: DataFrame = df.copy()
        df.loc[np.array([False], dtype=np.bool_), ['a']] = df['b'].copy()
        tm.assert_frame_equal(df, expected)

    def test_loc_indexer_length_one(self) -> None:
        df: DataFrame = DataFrame({'a': ['x'], 'b': ['y']}, dtype=object)
        expected: DataFrame = DataFrame({'a': ['y'], 'b': ['y']}, dtype=object)
        df.loc[np.array([True], dtype=np.bool_), ['a']] = df['b'].copy()
        tm.assert_frame_equal(df, expected)

class TestLocListlike:
    @pytest.mark.parametrize('box', [lambda x: x, np.asarray, list])
    def test_loc_getitem_list_of_labels_categoricalindex_with_na(self, box: Callable[[Sequence[Any]], Any]) -> None:
        ci: CategoricalIndex = CategoricalIndex(['A', 'B', np.nan])
        ser: Series = Series(range(3), index=ci)
        result: Series = ser.loc[box(ci)]
        tm.assert_series_equal(result, ser)
        result = ser[box(ci)]
        tm.assert_series_equal(result, ser)
        result = ser.to_frame().loc[box(ci)]
        tm.assert_frame_equal(result, ser.to_frame())
        ser2: Series = ser[:-1]
        ci2: CategoricalIndex = ci[1:]
        msg: str = 'not in index'
        with pytest.raises(KeyError, match=msg):
            _ = ser2.loc[box(ci2)]
        with pytest.raises(KeyError, match=msg):
            _ = ser2[box(ci2)]
        with pytest.raises(KeyError, match=msg):
            _ = ser2.to_frame().loc[box(ci2)]

    def test_loc_getitem_series_label_list_missing_values(self) -> None:
        key: np.ndarray = np.array(['2001-01-04', '2001-01-02', '2001-01-04', '2001-01-14'], dtype='datetime64')
        ser: Series = Series([2, 5, 8, 11], date_range('2001-01-01', freq='D', periods=4))
        with pytest.raises(KeyError, match='not in index'):
            _ = ser.loc[key]

    def test_loc_getitem_series_label_list_missing_integer_values(self) -> None:
        ser: Series = Series(index=np.array([9730701000001104, 10049011000001109]), data=np.array([999000011000001104, 999000011000001104]))
        with pytest.raises(KeyError, match='not in index'):
            _ = ser.loc[np.array([9730701000001104, 10047311000001102])]

    @pytest.mark.parametrize('to_period', [True, False])
    def test_loc_getitem_listlike_of_datetimelike_keys(self, to_period: bool) -> None:
        idx: DatetimeIndex = date_range('2011-01-01', '2011-01-02', freq='D', name='idx')
        if to_period:
            idx = idx.to_period('D')
        ser: Series = Series([0.1, 0.2], index=idx, name='s')
        keys: List[Timestamp] = [Timestamp('2011-01-01'), Timestamp('2011-01-02')]
        if to_period:
            keys = [x.to_period('D') for x in keys]
        result: Series = ser.loc[keys]
        exp: Series = Series([0.1, 0.2], index=idx, name='s')
        if not to_period:
            exp.index = exp.index._with_freq(None)
        tm.assert_series_equal(result, exp, check_index_type=True)
        keys = [Timestamp('2011-01-02'), Timestamp('2011-01-02'), Timestamp('2011-01-01')]
        if to_period:
            keys = [x.to_period('D') for x in keys]
        exp = Series([0.2, 0.2, 0.1], index=Index(keys, name='idx', dtype=idx.dtype), name='s')
        result = ser.loc[keys]
        tm.assert_series_equal(result, exp, check_index_type=True)
        keys = [Timestamp('2011-01-03'), Timestamp('2011-01-02'), Timestamp('2011-01-03')]
        if to_period:
            keys = [x.to_period('D') for x in keys]
        with pytest.raises(KeyError, match='not in index'):
            _ = ser.loc[keys]

    def test_loc_named_index(self) -> None:
        df: DataFrame = DataFrame([[1, 2], [4, 5], [7, 8]], index=['cobra', 'viper', 'sidewinder'], columns=['max_speed', 'shield'])
        expected: DataFrame = df.iloc[:2]
        expected.index.name = 'foo'
        result: DataFrame = df.loc[Index(['cobra', 'viper'], name='foo')]
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize('columns, column_key, expected_columns', [
    ([2011, 2012, 2013], [2011, 2012], [0, 1]),
    ([2011, 2012, 'All'], [2011, 2012], [0, 1]),
    ([2011, 2012, 'All'], [2011, 'All'], [0, 2])
])
def test_loc_getitem_label_list_integer_labels(columns: List[Any], column_key: Any, expected_columns: List[int]) -> None:
    df: DataFrame = DataFrame(np.random.default_rng(2).random((3, 3)), columns=columns, index=list('ABC'))
    expected: DataFrame = df.iloc[:, expected_columns]
    result: DataFrame = df.loc[['A', 'B', 'C'], column_key]
    tm.assert_frame_equal(result, expected, check_column_type=True)

def test_loc_setitem_float_intindex() -> None:
    rand_data: np.ndarray = np.random.default_rng(2).standard_normal((8, 4))
    result: DataFrame = DataFrame(rand_data)
    result.loc[:, 0.5] = np.nan
    expected_data: np.ndarray = np.hstack((rand_data, np.array([np.nan] * 8).reshape(8, 1)))
    expected: DataFrame = DataFrame(expected_data, columns=[0.0, 1.0, 2.0, 3.0, 0.5])
    tm.assert_frame_equal(result, expected)
    result = DataFrame(rand_data)
    result.loc[:, 0.5] = np.nan
    tm.assert_frame_equal(result, expected)

def test_loc_axis_1_slice() -> None:
    cols: List[Tuple[Any, Any]] = [(yr, m) for yr in [2014, 2015] for m in [7, 8, 9, 10]]
    df: DataFrame = DataFrame(np.ones((10, 8)), index=tuple('ABCDEFGHIJ'), columns=MultiIndex.from_tuples(cols))
    result: DataFrame = df.loc(axis=1)[(2014, 9):(2015, 8)]
    expected: DataFrame = DataFrame(np.ones((10, 4)), index=tuple('ABCDEFGHIJ'), columns=MultiIndex.from_tuples([(2014, 9), (2014, 10), (2015, 7), (2015, 8)]))
    tm.assert_frame_equal(result, expected)

def test_loc_set_dataframe_multiindex() -> None:
    expected: DataFrame = DataFrame('a', index=range(2), columns=MultiIndex.from_product([range(2), range(2)]))
    result: DataFrame = expected.copy()
    result.loc[0, [(0, 1)]] = result.loc[0, [(0, 1)]]
    tm.assert_frame_equal(result, expected)

def test_loc_mixed_int_float() -> None:
    ser: Series = Series(range(2), Index([1, 2.0], dtype=object))
    result: Any = ser.loc[1]
    assert result == 0

def test_loc_with_positional_slice_raises() -> None:
    ser: Series = Series(range(4), index=['A', 'B', 'C', 'D'])
    with pytest.raises(TypeError, match='Slicing a positional slice with .loc'):
        ser.loc[:3] = 2

def test_loc_slice_disallows_positional() -> None:
    dti: DatetimeIndex = date_range('2016-01-01', periods=3)
    df: DataFrame = DataFrame(np.random.default_rng(2).random((3, 2)), index=dti)
    ser: Series = df[0]
    msg: str = 'cannot do slice indexing on DatetimeIndex with these indexers \\[1\\] of type int'
    for obj in [df, ser]:
        with pytest.raises(TypeError, match=msg):
            obj.loc[1:3]
        with pytest.raises(TypeError, match='Slicing a positional slice with .loc'):
            obj.loc[1:3] = 1
    with pytest.raises(TypeError, match=msg):
        df.loc[1:3, 1]
    with pytest.raises(TypeError, match='Slicing a positional slice with .loc'):
        df.loc[1:3, 1] = 2

def test_loc_datetimelike_mismatched_dtypes() -> None:
    df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 3)), columns=['a', 'b', 'c'], index=date_range('2012', freq='h', periods=5))
    df = df.iloc[[0, 2, 2, 3]].copy()
    dti: DatetimeIndex = df.index
    tdi: pd.TimedeltaIndex = pd.TimedeltaIndex(dti.asi8)
    msg: str = 'None of \\[TimedeltaIndex.* are in the \\[index\\]'
    with pytest.raises(KeyError, match=msg):
        _ = df.loc[tdi]
    with pytest.raises(KeyError, match=msg):
        _ = df['a'].loc[tdi]

def test_loc_with_period_index_indexer() -> None:
    idx: PeriodIndex = pd.period_range('2002-01', '2003-12', freq='M')
    df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((24, 10)), index=idx)
    tm.assert_frame_equal(df, df.loc[idx])
    tm.assert_frame_equal(df, df.loc[list(idx)])
    tm.assert_frame_equal(df, df.loc[list(idx)])
    tm.assert_frame_equal(df.iloc[0:5], df.loc[idx[0:5]])
    tm.assert_frame_equal(df, df.loc[list(idx)])

def test_loc_setitem_multiindex_timestamp() -> None:
    vals: np.ndarray = np.random.default_rng(2).standard_normal((8, 6))
    idx: DatetimeIndex = date_range('1/1/2000', periods=8)
    cols: List[str] = ['A', 'B', 'C', 'D', 'E', 'F']
    exp: DataFrame = DataFrame(vals, index=idx, columns=cols)
    exp.loc[exp.index[1], ('A', 'B')] = np.nan
    vals[1][0:2] = np.nan
    res: DataFrame = DataFrame(vals, index=idx, columns=cols)
    tm.assert_frame_equal(res, exp)

def test_loc_getitem_multiindex_tuple_level() -> None:
    lev1: List[str] = ['a', 'b', 'c']
    lev2: List[Tuple[Any, Any]] = [(0, 1), (1, 0)]
    lev3: List[int] = [0, 1]
    cols: MultiIndex = MultiIndex.from_product([lev1, lev2, lev3], names=['x', 'y', 'z'])
    df: DataFrame = DataFrame(6, index=range(5), columns=cols)
    result: DataFrame = df.loc[:, (lev1[0], lev2[0], lev3[0])]
    expected: DataFrame = df.iloc[:, :1]
    tm.assert_frame_equal(result, expected)
    alt: DataFrame = df.xs((lev1[0], lev2[0], lev3[0]), level=[0, 1, 2], axis=1)
    tm.assert_frame_equal(alt, expected)
    ser: Series = df.iloc[0]
    expected2: Series = ser.iloc[:1]
    alt2: Series = ser.xs((lev1[0], lev2[0], lev3[0]), level=[0, 1, 2], axis=0)
    tm.assert_series_equal(alt2, expected2)
    result2: Any = ser.loc[lev1[0], lev2[0], lev3[0]]
    assert result2 == 6

def test_loc_getitem_nullable_index_with_duplicates() -> None:
    df: DataFrame = DataFrame(data=np.array([[1, 2, 3, 4], [5, 6, 7, 8], [1, 2, np.nan, np.nan]]).T, columns=['a', 'b', 'c'], dtype='Int64')
    df2: DataFrame = df.set_index('c')
    assert df2.index.dtype == 'Int64'
    res: Series = df2.loc[1]
    expected: Series = Series([1, 5], index=df2.columns, dtype='Int64', name=1)
    tm.assert_series_equal(res, expected)
    df2.index = df2.index.astype(object)
    res = df2.loc[1]
    tm.assert_series_equal(res, expected)

@pytest.mark.parametrize('value', [300, np.uint16(300), np.int16(300)])
def test_loc_setitem_uint8_upcast(value: Union[int, np.uint16, np.int16]) -> None:
    df: DataFrame = DataFrame([1, 2, 3, 4], columns=['col1'], dtype='uint8')
    with pytest.raises(TypeError, match='Invalid value'):
        df.loc[2, 'col1'] = value

@pytest.mark.parametrize('fill_val,exp_dtype', [
    (Timestamp('2022-01-06'), 'datetime64[ns]'),
    (Timestamp('2022-01-07', tz='US/Eastern'), 'datetime64[ns, US/Eastern]')
])
def test_loc_setitem_using_datetimelike_str_as_index(fill_val: Timestamp, exp_dtype: str) -> None:
    data: List[Any] = ['2022-01-02', '2022-01-03', '2022-01-04', fill_val.date()]
    index: DatetimeIndex = DatetimeIndex(data, tz=fill_val.tz, dtype=exp_dtype)
    df: DataFrame = DataFrame([10, 11, 12, 14], columns=['a'], index=index)
    df.loc['2022-01-08', 'a'] = 13
    data.append('2022-01-08')
    expected_index: DatetimeIndex = DatetimeIndex(data, dtype=exp_dtype)
    tm.assert_index_equal(df.index, expected_index, exact=True)

def test_loc_set_int_dtype() -> None:
    df: DataFrame = DataFrame([list('abc')])
    df.loc[:, 'col1'] = 5
    expected: DataFrame = DataFrame({0: ['a'], 1: ['b'], 2: ['c'], 'col1': [5]})
    tm.assert_frame_equal(df, expected)

@pytest.mark.filterwarnings('ignore:Period with BDay freq is deprecated:FutureWarning')
@pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
def test_loc_periodindex_3_levels() -> None:
    p_index: PeriodIndex = PeriodIndex(['20181101 1100', '20181101 1200', '20181102 1300', '20181102 1400'], name='datetime', freq='B')
    mi_series: DataFrame = DataFrame([['A', 'B', 1.0], ['A', 'C', 2.0], ['Z', 'Q', 3.0], ['W', 'F', 4.0]], index=p_index, columns=['ONE', 'TWO', 'VALUES'])
    mi_series = mi_series.set_index(['ONE', 'TWO'], append=True)['VALUES']
    assert mi_series.loc[p_index[0], 'A', 'B'] == 1.0

def test_loc_setitem_pyarrow_strings() -> None:
    pytest.importorskip('pyarrow')
    df: DataFrame = DataFrame({'strings': Series(['A', 'B', 'C'], dtype='string[pyarrow]'),
                                'ids': Series([True, True, False])})
    new_value: Series = Series(['X', 'Y'])
    df.loc[df.ids, 'strings'] = new_value
    expected_df: DataFrame = DataFrame({'strings': Series(['X', 'Y', 'C'], dtype='string[pyarrow]'),
                                          'ids': Series([True, True, False])})
    tm.assert_frame_equal(df, expected_df)

class TestLocSeries:
    @pytest.mark.parametrize('val,expected', [(2 ** 63 - 1, 3), (2 ** 63, 4)])
    def test_loc_uint64(self, val: int, expected: int) -> None:
        ser: Series = Series({2 ** 63 - 1: 3, 2 ** 63: 4})
        assert ser.loc[val] == expected

    def test_loc_getitem(self, string_series: Series, datetime_series: Series) -> None:
        inds: Sequence[Any] = string_series.index[[3, 4, 7]]
        tm.assert_series_equal(string_series.loc[inds], string_series.reindex(inds))
        tm.assert_series_equal(string_series.iloc[5::2], string_series[5::2])
        d1: Any = datetime_series.index[[5, 15]][0]
        d2: Any = datetime_series.index[[5, 15]][1]
        result: Series = datetime_series.loc[d1:d2]
        expected: Series = datetime_series.truncate(d1, d2)
        tm.assert_series_equal(result, expected)
        mask: Series = string_series > string_series.median()
        tm.assert_series_equal(string_series.loc[mask], string_series[mask])
        assert datetime_series.loc[d1] == datetime_series[d1]
        assert datetime_series.loc[d2] == datetime_series[d2]

    def test_loc_getitem_not_monotonic(self, datetime_series: Series) -> None:
        d1: Any = datetime_series.index[[5, 15]][0]
        d2: Any = datetime_series.index[[5, 15]][1]
        ts2: Series = datetime_series[::2].iloc[[1, 2, 0]]
        msg: str = "Timestamp\\('2000-01-10 00:00:00'\\)"
        with pytest.raises(KeyError, match=msg):
            _ = ts2.loc[d1:d2]
        with pytest.raises(KeyError, match=msg):
            ts2.loc[d1:d2] = 0

    def test_loc_getitem_setitem_integer_slice_keyerrors(self) -> None:
        ser: Series = Series(np.random.default_rng(2).standard_normal(10), index=list(range(0, 20, 2)))
        cp: Series = ser.copy()
        cp.iloc[4:10] = 0
        assert (cp.iloc[4:10] == 0).all()
        cp = ser.copy()
        cp.iloc[3:11] = 0
        assert (cp.iloc[3:11] == 0).values.all()
        result: Series = ser.iloc[2:6]
        result2: Series = ser.loc[3:11]
        expected: Series = ser.reindex([4, 6, 8, 10])
        tm.assert_series_equal(result, expected)
        tm.assert_series_equal(result2, expected)
        s2: Series = ser.iloc[list(range(5)) + list(range(9, 4, -1))]
        with pytest.raises(KeyError, match='^3$'):
            _ = s2.loc[3:11]
        with pytest.raises(KeyError, match='^3$'):
            s2.loc[3:11] = 0

    def test_loc_getitem_iterator(self, string_series: Series) -> None:
        idx = iter(string_series.index[:10])
        result: Series = string_series.loc[idx]
        tm.assert_series_equal(result, string_series[:10])

    def test_loc_setitem_boolean(self, string_series: Series) -> None:
        mask: Series = string_series > string_series.median()
        result: Series = string_series.copy()
        result.loc[mask] = 0
        expected: Series = string_series.copy()
        expected[mask] = 0
        tm.assert_series_equal(result, expected)

    def test_loc_setitem_corner(self, string_series: Series) -> None:
        inds: List[Any] = list(string_series.index[[5, 8, 12]])
        string_series.loc[inds] = 5
        msg: str = "\\['foo'\\] not in index"
        with pytest.raises(KeyError, match=msg):
            string_series.loc[inds + ['foo']] = 5

    def test_basic_setitem_with_labels(self, datetime_series: Series) -> None:
        indices: Any = datetime_series.index[[5, 10, 15]]
        cp: Series = datetime_series.copy()
        exp: Series = datetime_series.copy()
        cp[indices] = 0
        exp.loc[indices] = 0
        tm.assert_series_equal(cp, exp)
        cp = datetime_series.copy()
        exp = datetime_series.copy()
        cp[indices[0]:indices[2]] = 0
        exp.loc[indices[0]:indices[2]] = 0
        tm.assert_series_equal(cp, exp)
        datetime_series.loc[indices[0]] = 4
        datetime_series.loc[indices[2]] = 6
        assert datetime_series.loc[indices[0]] == 4
        assert datetime_series.loc[indices[2]] == 6

    def test_loc_setitem_listlike_of_ints(self) -> None:
        ser: Series = Series(np.random.default_rng(2).standard_normal(10), index=list(range(0, 20, 2)))
        inds: List[int] = [0, 4, 6]
        arr_inds: np.ndarray = np.array([0, 4, 6])
        cp: Series = ser.copy()
        exp: Series = ser.copy()
        ser[inds] = 0
        ser.loc[inds] = 0
        tm.assert_series_equal(cp, exp)
        cp = ser.copy()
        exp = ser.copy()
        ser[arr_inds] = 0
        ser.loc[arr_inds] = 0
        tm.assert_series_equal(cp, exp)
        inds_notfound: List[int] = [0, 4, 5, 6]
        arr_inds_notfound: np.ndarray = np.array([0, 4, 5, 6])
        msg: str = '\\[5\\] not in index'
        with pytest.raises(KeyError, match=msg):
            ser[inds_notfound] = 0
        with pytest.raises(Exception, match=msg):
            ser[arr_inds_notfound] = 0

    def test_loc_setitem_dt64tz_values(self) -> None:
        ser: Series = Series(date_range('2011-01-01', periods=3, tz='US/Eastern'), index=['a', 'b', 'c'])
        s2: Series = ser.copy()
        expected: Timestamp = Timestamp('2011-01-03', tz='US/Eastern')
        s2.loc['a'] = expected
        result: Any = s2.loc['a']
        assert result == expected
        s2 = ser.copy()
        s2.iloc[0] = expected
        result = s2.iloc[0]
        assert result == expected
        s2 = ser.copy()
        s2['a'] = expected
        result = s2['a']
        assert result == expected

    @pytest.mark.parametrize('array_fn', [np.array, pd.array, list, tuple])
    @pytest.mark.parametrize('size', [0, 4, 5, 6])
    def test_loc_iloc_setitem_with_listlike(self, size: int, array_fn: Callable[[List[Any]], Any]) -> None:
        arr: Any = array_fn([0] * size)
        expected: Series = Series([arr, 0, 0, 0, 0], index=list('abcde'), dtype=object)
        ser: Series = Series(0, index=list('abcde'), dtype=object)
        ser.loc['a'] = arr
        tm.assert_series_equal(ser, expected)
        ser = Series(0, index=list('abcde'), dtype=object)
        ser.iloc[0] = arr
        tm.assert_series_equal(ser, expected)

    @pytest.mark.parametrize('indexer', [IndexSlice['A', :], ('A', slice(None))])
    def test_loc_series_getitem_too_many_dimensions(self, indexer: Any) -> None:
        ser: Series = Series(index=MultiIndex.from_tuples([('A', '0'), ('A', '1'), ('B', '0')]), data=[21, 22, 23])
        msg: str = 'Too many indexers'
        with pytest.raises(IndexingError, match=msg):
            _ = ser.loc[indexer, :]
        with pytest.raises(IndexingError, match=msg):
            ser.loc[indexer, :] = 1

    def test_loc_setitem(self, string_series: Series) -> None:
        inds: Any = string_series.index[[3, 4, 7]]
        result: Series = string_series.copy()
        result.loc[inds] = 5
        expected: Series = string_series.copy()
        expected.iloc[[3, 4, 7]] = 5
        tm.assert_series_equal(result, expected)
        result.iloc[5:10] = 10
        expected[5:10] = 10
        tm.assert_series_equal(result, expected)
        d1: Any = string_series.index[[5, 15]][0]
        d2: Any = string_series.index[[5, 15]][1]
        result.loc[d1:d2] = 6
        expected[5:16] = 6
        tm.assert_series_equal(result, expected)
        string_series.loc[d1] = 4
        string_series.loc[d2] = 6
        assert string_series[d1] == 4
        assert string_series[d2] == 6

    @pytest.mark.parametrize('dtype', ['object', 'string'])
    def test_loc_assign_dict_to_row(self, dtype: str) -> None:
        df: DataFrame = DataFrame({'A': ['abc', 'def'], 'B': ['ghi', 'jkl']}, dtype=dtype)
        df.loc[0, :] = {'A': 'newA', 'B': 'newB'}
        expected: DataFrame = DataFrame({'A': ['newA', 'def'], 'B': ['newB', 'jkl']}, dtype=dtype)
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_dict_timedelta_multiple_set(self) -> None:
        result: DataFrame = DataFrame(columns=['time', 'value'])
        result.loc[1] = {'time': Timedelta(6, unit='s'), 'value': 'foo'}
        result.loc[1] = {'time': Timedelta(6, unit='s'), 'value': 'foo'}
        expected: DataFrame = DataFrame([[Timedelta(6, unit='s'), 'foo']], columns=['time', 'value'], index=[1])
        tm.assert_frame_equal(result, expected)

    def test_loc_set_multiple_items_in_multiple_new_columns(self) -> None:
        df: DataFrame = DataFrame(index=[1, 2], columns=['a'])
        df.loc[1, ['b', 'c']] = [6, 7]
        expected: DataFrame = DataFrame({'a': Series([np.nan, np.nan], dtype='object'),
                                          'b': [6, np.nan],
                                          'c': [7, np.nan]},
                                         index=[1, 2])
        tm.assert_frame_equal(df, expected)

    def test_getitem_loc_str_periodindex(self) -> None:
        msg: str = 'Period with BDay freq is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            index: PeriodIndex = pd.period_range(start='2000', periods=20, freq='B')
            series: Series = Series(range(20), index=index)
            assert series.loc['2000-01-14'] == 9

    def test_loc_nonunique_masked_index(self) -> None:
        ids: List[int] = list(range(11))
        index: Index = Index(ids * 1000, dtype='Int64')
        df: DataFrame = DataFrame({'val': np.arange(len(index), dtype=np.intp)}, index=index)
        result: DataFrame = df.loc[ids]
        expected: DataFrame = DataFrame({'val': index.argsort(kind='stable').astype(np.intp)}, index=Index(np.array(ids).repeat(1000), dtype='Int64'))
        tm.assert_frame_equal(result, expected)

    def test_loc_index_alignment_for_series(self) -> None:
        df: DataFrame = DataFrame({'a': [1, 2], 'b': [3, 4]})
        other: Series = Series([200, 999], index=[1, 0])
        df.loc[:, 'a'] = other
        expected: DataFrame = DataFrame({'a': [999, 200], 'b': [3, 4]})
        tm.assert_frame_equal(expected, df)

    def test_loc_reindexing_of_empty_index(self) -> None:
        df: DataFrame = DataFrame(index=[1, 1, 2, 2], data=['1', '1', '2', '2'])
        df.loc[pd.Series([False] * 4, index=df.index, name=0), 0] = df[0]
        expected: DataFrame = DataFrame(index=[1, 1, 2, 2], data=['1', '1', '2', '2'])
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_matching_index(self) -> None:
        s: Series = Series(0.0, index=list('abcd'))
        s1: Series = Series(1.0, index=list('ab'))
        s2: Series = Series(2.0, index=list('xy'))
        s.loc[['a', 'b']] = s1
        result: Series = s[['a', 'b']]
        expected: Series = s1
        tm.assert_series_equal(result, expected)
        s.loc[['a', 'b']] = s2
        result = s[['a', 'b']]
        expected = Series([np.nan, np.nan], index=['a', 'b'])
        tm.assert_series_equal(result, expected)

# End of annotated code.
