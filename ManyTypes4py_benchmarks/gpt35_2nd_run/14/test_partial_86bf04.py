import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame, Index, Period, Series, Timestamp, date_range, period_range
import pandas._testing as tm

class TestEmptyFrameSetitemExpansion:

    def test_empty_frame_setitem_index_name_retained(self) -> None:
        df: DataFrame = DataFrame({}, index=pd.RangeIndex(0, name='df_index'))
        series: Series = Series(1.23, index=pd.RangeIndex(4, name='series_index'))
        df['series'] = series
        expected: DataFrame = DataFrame({'series': [1.23] * 4}, index=pd.RangeIndex(4, name='df_index'), columns=Index(['series'], dtype=object))
        tm.assert_frame_equal(df, expected)

    def test_empty_frame_setitem_index_name_inherited(self) -> None:
        df: DataFrame = DataFrame()
        series: Series = Series(1.23, index=pd.RangeIndex(4, name='series_index'))
        df['series'] = series
        expected: DataFrame = DataFrame({'series': [1.23] * 4}, index=pd.RangeIndex(4, name='series_index'), columns=Index(['series'], dtype=object))
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_zerolen_series_columns_align(self) -> None:
        df: DataFrame = DataFrame(columns=['A', 'B'])
        df.loc[0] = Series(1, index=range(4))
        expected: DataFrame = DataFrame(columns=['A', 'B'], index=[0], dtype=np.float64)
        tm.assert_frame_equal(df, expected)
        df: DataFrame = DataFrame(columns=['A', 'B'])
        df.loc[0] = Series(1, index=['B'])
        exp: DataFrame = DataFrame([[np.nan, 1]], columns=['A', 'B'], index=[0], dtype='float64')
        tm.assert_frame_equal(df, exp)

    def test_loc_setitem_zerolen_list_length_must_match_columns(self) -> None:
        df: DataFrame = DataFrame(columns=['A', 'B'])
        msg: str = 'cannot set a row with mismatched columns'
        with pytest.raises(ValueError, match=msg):
            df.loc[0] = [1, 2, 3]
        df: DataFrame = DataFrame(columns=['A', 'B'])
        df.loc[3] = [6, 7]
        exp: DataFrame = DataFrame([[6, 7]], index=[3], columns=['A', 'B'], dtype=np.int64)
        tm.assert_frame_equal(df, exp)

    def test_partial_set_empty_frame() -> None:
        df: DataFrame = DataFrame()
        msg: str = 'cannot set a frame with no defined columns'
        with pytest.raises(ValueError, match=msg):
            df.loc[1] = 1
        with pytest.raises(ValueError, match=msg):
            df.loc[1] = Series([1], index=['foo'])
        msg: str = 'cannot set a frame with no defined index and a scalar'
        with pytest.raises(ValueError, match=msg):
            df.loc[:, 1] = 1

    def test_partial_set_empty_frame2() -> None:
        expected: DataFrame = DataFrame(columns=Index(['foo'], dtype=object), index=Index([], dtype='object'))
        df: DataFrame = DataFrame(index=Index([], dtype='object'))
        df['foo'] = Series([], dtype='object')
        tm.assert_frame_equal(df, expected)
        df: DataFrame = DataFrame(index=Index([]))
        df['foo'] = Series(df.index)
        tm.assert_frame_equal(df, expected)
        df: DataFrame = DataFrame(index=Index([]))
        df['foo'] = df.index
        tm.assert_frame_equal(df, expected)

    def test_partial_set_empty_frame3() -> None:
        expected: DataFrame = DataFrame(columns=Index(['foo'], dtype=object), index=Index([], dtype='int64'))
        expected['foo'] = expected['foo'].astype('float64')
        df: DataFrame = DataFrame(index=Index([], dtype='int64'))
        df['foo'] = []
        tm.assert_frame_equal(df, expected)
        df: DataFrame = DataFrame(index=Index([], dtype='int64'))
        df['foo'] = Series(np.arange(len(df)), dtype='float64')
        tm.assert_frame_equal(df, expected)

    def test_partial_set_empty_frame4() -> None:
        df: DataFrame = DataFrame(index=Index([], dtype='int64'))
        df['foo'] = range(len(df))
        expected: DataFrame = DataFrame(columns=Index(['foo'], dtype=object), index=Index([], dtype='int64'))
        expected['foo'] = expected['foo'].astype('int64')
        tm.assert_frame_equal(df, expected)

    def test_partial_set_empty_frame5() -> None:
        df: DataFrame = DataFrame()
        tm.assert_index_equal(df.columns, pd.RangeIndex(0))
        df2: DataFrame = DataFrame()
        df2[1] = Series([1], index=['foo'])
        df.loc[:, 1] = Series([1], index=['foo'])
        tm.assert_frame_equal(df, DataFrame([[1]], index=['foo'], columns=[1]))
        tm.assert_frame_equal(df, df2)

    def test_partial_set_empty_frame_no_index() -> None:
        expected: DataFrame = DataFrame({0: Series(1, index=range(4))}, columns=['A', 'B', 0])
        df: DataFrame = DataFrame(columns=['A', 'B'])
        df[0] = Series(1, index=range(4))
        tm.assert_frame_equal(df, expected)
        df: DataFrame = DataFrame(columns=['A', 'B'])
        df.loc[:, 0] = Series(1, index=range(4))
        tm.assert_frame_equal(df, expected)

    def test_partial_set_empty_frame_row() -> None:
        expected: DataFrame = DataFrame(columns=['A', 'B', 'New'], index=Index([], dtype='int64'))
        expected['A'] = expected['A'].astype('int64')
        expected['B'] = expected['B'].astype('float64')
        expected['New'] = expected['New'].astype('float64')
        df: DataFrame = DataFrame({'A': [1, 2, 3], 'B': [1.2, 4.2, 5.2]})
        y: DataFrame = df[df.A > 5]
        y['New'] = np.nan
        tm.assert_frame_equal(y, expected)
        expected: DataFrame = DataFrame(columns=['a', 'b', 'c c', 'd'])
        expected['d'] = expected['d'].astype('int64')
        df: DataFrame = DataFrame(columns=['a', 'b', 'c c'])
        df['d'] = 3
        tm.assert_frame_equal(df, expected)
        tm.assert_series_equal(df['c c'], Series(name='c c', dtype=object))
        df: DataFrame = DataFrame({'A': [1, 2, 3], 'B': [1.2, 4.2, 5.2]})
        y: DataFrame = df[df.A > 5]
        result: DataFrame = y.reindex(columns=['A', 'B', 'C'])
        expected: DataFrame = DataFrame(columns=['A', 'B', 'C'])
        expected['A'] = expected['A'].astype('int64')
        expected['B'] = expected['B'].astype('float64')
        expected['C'] = expected['C'].astype('float64')
        tm.assert_frame_equal(result, expected)

    def test_partial_set_empty_frame_set_series() -> None:
        df: DataFrame = DataFrame(Series(dtype=object))
        expected: DataFrame = DataFrame({0: Series(dtype=object)})
        tm.assert_frame_equal(df, expected)
        df: DataFrame = DataFrame(Series(name='foo', dtype=object))
        expected: DataFrame = DataFrame({'foo': Series(dtype=object)})
        tm.assert_frame_equal(df, expected)

    def test_partial_set_empty_frame_empty_copy_assignment() -> None:
        df: DataFrame = DataFrame(index=[0])
        df = df.copy()
        df['a'] = 0
        expected: DataFrame = DataFrame(0, index=[0], columns=Index(['a'], dtype=object))
        tm.assert_frame_equal(df, expected)

    def test_partial_set_empty_frame_empty_consistencies(self, using_infer_string) -> None:
        df: DataFrame = DataFrame(columns=['x', 'y'])
        df['x'] = [1, 2]
        expected: DataFrame = DataFrame({'x': [1, 2], 'y': [np.nan, np.nan]})
        tm.assert_frame_equal(df, expected, check_dtype=False)
        df: DataFrame = DataFrame(columns=['x', 'y'])
        df['x'] = ['1', '2']
        expected: DataFrame = DataFrame({'x': Series(['1', '2'], dtype=object if not using_infer_string else 'str'), 'y': Series([np.nan, np.nan], dtype=object)})
        tm.assert_frame_equal(df, expected)
        df: DataFrame = DataFrame(columns=['x', 'y'])
        df.loc[0, 'x'] = 1
        expected: DataFrame = DataFrame({'x': [1], 'y': [np.nan]})
        tm.assert_frame_equal(df, expected, check_dtype=False)

class TestPartialSetting:

    def test_partial_setting(self) -> None:
        s_orig: Series = Series([1, 2, 3])
        s: Series = s_orig.copy()
        s[5] = 5
        expected: Series = Series([1, 2, 3, 5], index=[0, 1, 2, 5])
        tm.assert_series_equal(s, expected)
        s: Series = s_orig.copy()
        s.loc[5] = 5
        expected: Series = Series([1, 2, 3, 5], index=[0, 1, 2, 5])
        tm.assert_series_equal(s, expected)
        s: Series = s_orig.copy()
        s[5] = 5.0
        expected: Series = Series([1, 2, 3, 5.0], index=[0, 1, 2, 5])
        tm.assert_series_equal(s, expected)
        s: Series = s_orig.copy()
        s.loc[5] = 5.0
        expected: Series = Series([1, 2, 3, 5.0], index=[0, 1, 2, 5])
        tm.assert_series_equal(s, expected)
        s: Series = s_orig.copy()
        msg: str = 'iloc cannot enlarge its target object'
        with pytest.raises(IndexError, match=msg):
            s.iloc[3] = 5.0
        msg: str = 'index 3 is out of bounds for axis 0 with size 3'
        with pytest.raises(IndexError, match=msg):
            s.iat[3] = 5.0

    @pytest.mark.filterwarnings('ignore:Setting a value on a view:FutureWarning')
    def test_partial_setting_frame(self) -> None:
        df_orig: DataFrame = DataFrame(np.arange(6).reshape(3, 2), columns=['A', 'B'], dtype='int64')
        df: DataFrame = df_orig.copy()
        msg: str = 'iloc cannot enlarge its target object'
        with pytest.raises(IndexError, match=msg):
            df.iloc[4, 2] = 5.0
        msg: str = 'index 2 is out of bounds for axis 0 with size 2'
        with pytest.raises(IndexError, match=msg):
            df.iat[4, 2] = 5.0
        expected: DataFrame = DataFrame({'A': [0, 4, 4], 'B': [1, 5, 5]})
        df: DataFrame = df_orig.copy()
        df.iloc[1] = df.iloc[2]
        tm.assert_frame_equal(df, expected)
        expected: DataFrame = DataFrame({'A': [0, 4, 4], 'B': [1, 5, 5]})
        df: DataFrame = df_orig.copy()
        df.loc[1] = df.loc[2]
        tm.assert_frame_equal(df, expected)
        expected: DataFrame = DataFrame({'A': [0, 2, 4, 4], 'B': [1, 3, 5, 5]})
        df: DataFrame = df_orig.copy()
        df.loc[3] = df.loc[2]
        tm.assert_frame_equal(df, expected)
        expected: DataFrame = DataFrame({'A': [0, 2, 4], 'B': [0, 2, 4]})
        df: DataFrame = df_orig.copy()
        df.loc[:, 'B'] = df.loc[:, 'A']
        tm.assert_frame_equal(df, expected)
        expected: DataFrame = DataFrame({'A': [0, 2, 4], 'B': Series([0.0, 2.0, 4.0])})
        df: DataFrame = df_orig.copy()
        df['B'] = df['B'].astype(np.float64)
        df.loc[:, 'B'] = df.loc[:, 'A']
        tm.assert_frame_equal(df, expected)
        expected: DataFrame = df_orig.copy()
        expected['C'] = df['A']
        df: DataFrame = df_orig.copy()
        df.loc[:, 'C'] = df.loc[:, 'A']
        tm.assert_frame_equal(df, expected)
        expected: DataFrame = df_orig.copy()
        expected['C'] = df['A']
        df: DataFrame = df_orig.copy()
        df.loc[:, 'C'] = df.loc[:, 'A']
        tm.assert_frame_equal(df, expected)

    def test_partial_setting2(self) -> None:
        dates: pd.DatetimeIndex = date_range('1/1/2000', periods=8)
        df_orig: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((8, 4)), index=dates, columns=['A', 'B', 'C', 'D'])
        expected: DataFrame = pd.concat([df_orig, DataFrame({'A': 7}, index=dates[-1:] + dates.freq)], sort=True)
        df: DataFrame = df_orig.copy()
        df.loc[dates[-1] + dates.freq, 'A'] = 7
        tm.assert_frame_equal(df, expected)
        df: DataFrame = df_orig.copy()
        df.at[dates[-1] + dates.freq, 'A'] = 7
        tm.assert_frame_equal(df, expected)
        exp_other: DataFrame = DataFrame({0: 7}, index=dates[-1:] + dates.freq)
        expected: DataFrame = pd.concat([df_orig, exp_other], axis=1)
        df: DataFrame = df_orig.copy()
        df.loc[dates[-1] + dates.freq, 0] = 7
        tm.assert_frame_equal(df, expected)
        df: DataFrame = df_orig.copy()
        df.at[dates[-1] + dates.freq, 0] = 7
        tm.assert_frame_equal(df, expected)

    def test_partial_setting_mixed_dtype(self) -> None:
        df: DataFrame = DataFrame([[True, 1], [False, 2]], columns=['female', 'fitness'])
        s: Series = df.loc[1].copy()
        s.name = 2
        expected: DataFrame = pd.concat([df, DataFrame(s).T.infer_objects()])
        df.loc[2] = df.loc[1]
        tm.assert_frame_equal(df, expected)

    def test_series_partial_set(self) -> None:
        ser: Series = Series([0.1, 0.2], index=[1, 2])
        expected: Series = Series([np.nan, 0.2, np.nan], index=[3, 2, 3])
        with pytest.raises(KeyError, match='not in index'):
            ser.loc[[3, 2, 3]]
        result: Series = ser.reindex([3, 2, 3])
        tm.assert_series_equal(result, expected, check_index_type=True)
        expected: Series = Series([np.nan, 0.2, np.nan, np.nan], index=[3, 2, 3, 'x'])
        with pytest.raises(KeyError, match='not in index'):
            ser.loc[[3, 2, 3, 'x']]
        result: Series = ser.reindex([3, 2, 3, 'x'])
        tm.assert_series_equal(result, expected, check_index_type=True)
        expected: Series = Series([0.2, 0.2, 0.1], index=[2, 2, 1])
        result: Series = ser.loc[[2, 2, 1]]
        tm.assert_series_equal(result, expected, check_index_type=True)
        expected: Series = Series([0.2, 0.2, np.nan, 0.1], index=[2, 2, 'x', 1])
        with pytest.raises(KeyError, match='not in index'):
            ser.loc[[2, 2, 'x', 1]]
        result: Series = ser.reindex([2, 2, 'x', 1])
        tm.assert_series_equal(result, expected, check_index_type=True)
        msg: str = f'''\\"None of \\[Index\\(\\[3, 3, 3\\], dtype='{np.dtype(int)}'\\)\\] are in the \\[index\\]\\"'''
        with pytest.raises(KeyError, match=msg):
            ser.loc[[3, 3, 3]]
        expected: Series = Series([0.2, 0.2, np.nan], index=[2, 2, 3])
        with pytest.raises(KeyError, match='not in index'):
            ser.loc[[2, 2, 3]]
        result: Series = ser.reindex([2, 2, 3])
        tm.assert_series_equal(result, expected, check_index_type=True)
        s: Series = Series([0.1, 0.2, 0.3], index=[1, 2, 3])
        expected: Series = Series([0.3, np.nan, np.nan], index=[3, 4, 4])
        with pytest.raises(KeyError, match='not in index'):
            s.loc[[3, 4, 4]]
        result: Series = s.reindex([3, 4, 4])
        tm.assert_series_equal(result, expected, check_index_type=True)
        s: Series = Series([0.1, 0.2, 0.3, 0.4], index=[1, 2, 3, 4])
        expected: Series = Series([np.nan, 0.3, 0.3], index=[5, 3, 3])
        with pytest.raises(KeyError, match='not in index'):
            s.loc[[5, 3, 3]]
        result: Series = s.reindex([5, 3, 3])
        tm.assert_series_equal(result, expected, check_index_type=True)
        s: Series = Series([0.1,