import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame, Index, Series, Timestamp
import pandas._testing as tm
from typing import Any, Callable, List, Union

@pytest.fixture(params=[['linear', 'single'], ['nearest', 'table']], ids=lambda x: '-'.join(x))
def interp_method(request: pytest.FixtureRequest) -> List[str]:
    """(interpolation, method) arguments for quantile"""
    return request.param

class TestDataFrameQuantile:

    @pytest.mark.parametrize('df,expected', [
        [DataFrame({0: Series(pd.arrays.SparseArray([1, 2])), 1: Series(pd.arrays.SparseArray([3, 4]))}),
         Series([1.5, 3.5], name=0.5)],
        [DataFrame(Series([0.0, None, 1.0, 2.0], dtype='Sparse[float]')),
         Series([1.0], name=0.5)]
    ])
    def test_quantile_sparse(self, df: DataFrame, expected: Series) -> None:
        result: Series = df.quantile()
        expected = expected.astype('Sparse[float]')
        tm.assert_series_equal(result, expected)

    def test_quantile(self, datetime_frame: DataFrame, interp_method: List[str], request: Any) -> None:
        interpolation, method = interp_method
        df: DataFrame = datetime_frame
        result: Series = df.quantile(0.1, axis=0, numeric_only=True,
                                     interpolation=interpolation, method=method)
        expected: Series = Series([np.percentile(df[col], 10) for col in df.columns],
                                  index=df.columns, name=0.1)
        if interpolation == 'linear':
            tm.assert_series_equal(result, expected)
        else:
            tm.assert_index_equal(result.index, expected.index)
            assert result.name == expected.name

        result = df.quantile(0.9, axis=1, numeric_only=True,
                             interpolation=interpolation, method=method)
        expected = Series([np.percentile(df.loc[date], 90) for date in df.index],
                          index=df.index, name=0.9)
        if interpolation == 'linear':
            tm.assert_series_equal(result, expected)
        else:
            tm.assert_index_equal(result.index, expected.index)
            assert result.name == expected.name

    def test_empty(self, interp_method: List[str]) -> None:
        interpolation, method = interp_method
        q: Series = DataFrame({'x': [], 'y': []}).quantile(0.1, axis=0, numeric_only=True,
                                                             interpolation=interpolation, method=method)
        assert np.isnan(q['x']) and np.isnan(q['y'])

    def test_non_numeric_exclusion(self, interp_method: List[str], request: Any) -> None:
        interpolation, method = interp_method
        df: DataFrame = DataFrame({'col1': ['A', 'A', 'B', 'B'], 'col2': [1, 2, 3, 4]})
        rs: Series = df.quantile(0.5, numeric_only=True, interpolation=interpolation, method=method)
        xp: Series = df.median(numeric_only=True).rename(0.5)
        if interpolation == 'nearest':
            xp = (xp + 0.5).astype(np.int64)
        tm.assert_series_equal(rs, xp)

    def test_axis(self, interp_method: List[str]) -> None:
        interpolation, method = interp_method
        df: DataFrame = DataFrame({'A': [1, 2, 3], 'B': [2, 3, 4]}, index=[1, 2, 3])
        result: Series = df.quantile(0.5, axis=1, interpolation=interpolation, method=method)
        expected: Series = Series([1.5, 2.5, 3.5], index=[1, 2, 3], name=0.5)
        if interpolation == 'nearest':
            expected = expected.astype(np.int64)
        tm.assert_series_equal(result, expected)

        result = df.quantile([0.5, 0.75], axis=1, interpolation=interpolation, method=method)
        expected_df: DataFrame = DataFrame({1: [1.5, 1.75],
                                             2: [2.5, 2.75],
                                             3: [3.5, 3.75]},
                                            index=[0.5, 0.75])
        if interpolation == 'nearest':
            expected_df.iloc[0, :] -= 0.5
            expected_df.iloc[1, :] += 0.25
            expected_df = expected_df.astype(np.int64)
        tm.assert_frame_equal(result, expected_df, check_index_type=True)

    def test_axis_numeric_only_true(self, interp_method: List[str]) -> None:
        interpolation, method = interp_method
        df: DataFrame = DataFrame([[1, 2, 3], ['a', 'b', 4]])
        result: Series = df.quantile(0.5, axis=1, numeric_only=True, interpolation=interpolation, method=method)
        expected: Series = Series([3.0, 4.0], index=range(2), name=0.5)
        if interpolation == 'nearest':
            expected = expected.astype(np.int64)
        tm.assert_series_equal(result, expected)

    def test_quantile_date_range(self, interp_method: List[str]) -> None:
        interpolation, method = interp_method
        dti: pd.DatetimeIndex = pd.date_range('2016-01-01', periods=3, tz='US/Pacific')
        ser: Series = Series(dti)
        df: DataFrame = DataFrame(ser)
        result: Series = df.quantile(numeric_only=False, interpolation=interpolation, method=method)
        expected: Series = Series(['2016-01-02 00:00:00'], name=0.5, dtype='datetime64[ns, US/Pacific]')
        tm.assert_series_equal(result, expected)

    def test_quantile_axis_mixed(self, interp_method: List[str]) -> None:
        interpolation, method = interp_method
        df: DataFrame = DataFrame({
            'A': [1, 2, 3],
            'B': [2.0, 3.0, 4.0],
            'C': pd.date_range('20130101', periods=3),
            'D': ['foo', 'bar', 'baz']
        })
        result: Series = df.quantile(0.5, axis=1, numeric_only=True, interpolation=interpolation, method=method)
        expected: Series = Series([1.5, 2.5, 3.5], name=0.5)
        if interpolation == 'nearest':
            expected -= 0.5
        tm.assert_series_equal(result, expected)
        msg: str = "'<' not supported between instances of 'Timestamp' and 'float'"
        with pytest.raises(TypeError, match=msg):
            df.quantile(0.5, axis=1, numeric_only=False)

    def test_quantile_axis_parameter(self, interp_method: List[str]) -> None:
        interpolation, method = interp_method
        df: DataFrame = DataFrame({'A': [1, 2, 3], 'B': [2, 3, 4]}, index=[1, 2, 3])
        result: Series = df.quantile(0.5, axis=0, interpolation=interpolation, method=method)
        expected: Series = Series([2.0, 3.0], index=['A', 'B'], name=0.5)
        if interpolation == 'nearest':
            expected = expected.astype(np.int64)
        tm.assert_series_equal(result, expected)
        expected_alt: Series = df.quantile(0.5, axis='index', interpolation=interpolation, method=method)
        if interpolation == 'nearest':
            expected_alt = expected_alt.astype(np.int64)
        tm.assert_series_equal(result, expected_alt)
        result = df.quantile(0.5, axis=1, interpolation=interpolation, method=method)
        expected = Series([1.5, 2.5, 3.5], index=[1, 2, 3], name=0.5)
        if interpolation == 'nearest':
            expected = expected.astype(np.int64)
        tm.assert_series_equal(result, expected)
        result = df.quantile(0.5, axis='columns', interpolation=interpolation, method=method)
        tm.assert_series_equal(result, expected)
        msg: str = 'No axis named -1 for object type DataFrame'
        with pytest.raises(ValueError, match=msg):
            df.quantile(0.1, axis=-1, interpolation=interpolation, method=method)
        msg = 'No axis named column for object type DataFrame'
        with pytest.raises(ValueError, match=msg):
            df.quantile(0.1, axis='column')

    def test_quantile_interpolation(self) -> None:
        df: DataFrame = DataFrame({'A': [1, 2, 3], 'B': [2, 3, 4]}, index=[1, 2, 3])
        result: Series = df.quantile(0.5, axis=1, interpolation='nearest')
        expected: Series = Series([1, 2, 3], index=[1, 2, 3], name=0.5)
        tm.assert_series_equal(result, expected)
        exp = np.percentile(np.array([[1, 2, 3], [2, 3, 4]]), 0.5, axis=0, method='nearest')
        expected = Series(exp, index=[1, 2, 3], name=0.5, dtype='int64')
        tm.assert_series_equal(result, expected)
        df = DataFrame({'A': [1.0, 2.0, 3.0], 'B': [2.0, 3.0, 4.0]}, index=[1, 2, 3])
        result = df.quantile(0.5, axis=1, interpolation='nearest')
        expected = Series([1.0, 2.0, 3.0], index=[1, 2, 3], name=0.5)
        tm.assert_series_equal(result, expected)
        exp = np.percentile(np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]), 0.5, axis=0, method='nearest')
        expected = Series(exp, index=[1, 2, 3], name=0.5, dtype='float64')
        tm.assert_series_equal(result, expected)
        result = df.quantile([0.5, 0.75], axis=1, interpolation='lower')
        expected_df = DataFrame({1: [1.0, 1.0],
                                 2: [2.0, 2.0],
                                 3: [3.0, 3.0]}, index=[0.5, 0.75])
        tm.assert_frame_equal(result, expected_df)
        df = DataFrame({'x': [], 'y': []})
        q: Series = df.quantile(0.1, axis=0, interpolation='higher')
        assert np.isnan(q['x']) and np.isnan(q['y'])
        df = DataFrame([[1, 1, 1], [2, 2, 2], [3, 3, 3]], columns=['a', 'b', 'c'])
        result = df.quantile([0.25, 0.5], interpolation='midpoint')
        expected_df = DataFrame([[1.5, 1.5, 1.5],
                                 [2.0, 2.0, 2.0]],
                                index=[0.25, 0.5], columns=['a', 'b', 'c'])
        tm.assert_frame_equal(result, expected_df)

    def test_quantile_interpolation_datetime(self, datetime_frame: DataFrame) -> None:
        df: DataFrame = datetime_frame
        q: Series = df.quantile(0.1, axis=0, numeric_only=True, interpolation='linear')
        assert q['A'] == np.percentile(df['A'], 10)

    def test_quantile_interpolation_int(self, int_frame: DataFrame) -> None:
        df: DataFrame = int_frame
        q: Series = df.quantile(0.1)
        assert q['A'] == np.percentile(df['A'], 10)
        q1: Series = df.quantile(0.1, axis=0, interpolation='linear')
        assert q1['A'] == np.percentile(df['A'], 10)
        tm.assert_series_equal(q, q1)

    def test_quantile_multi(self, interp_method: List[str]) -> None:
        interpolation, method = interp_method
        df: DataFrame = DataFrame([[1, 1, 1], [2, 2, 2], [3, 3, 3]], columns=['a', 'b', 'c'])
        result: DataFrame = df.quantile([0.25, 0.5], interpolation=interpolation, method=method)
        expected: DataFrame = DataFrame([[1.5, 1.5, 1.5],
                                         [2.0, 2.0, 2.0]],
                                        index=[0.25, 0.5], columns=['a', 'b', 'c'])
        if interpolation == 'nearest':
            expected = expected.astype(np.int64)
        tm.assert_frame_equal(result, expected)

    def test_quantile_multi_axis_1(self, interp_method: List[str]) -> None:
        interpolation, method = interp_method
        df: DataFrame = DataFrame([[1, 1, 1], [2, 2, 2], [3, 3, 3]], columns=['a', 'b', 'c'])
        result: DataFrame = df.quantile([0.25, 0.5], axis=1, interpolation=interpolation, method=method)
        expected: DataFrame = DataFrame([[1.0, 2.0, 3.0]] * 2, index=[0.25, 0.5], columns=[0, 1, 2])
        if interpolation == 'nearest':
            expected = expected.astype(np.int64)
        tm.assert_frame_equal(result, expected)

    def test_quantile_multi_empty(self, interp_method: List[str]) -> None:
        interpolation, method = interp_method
        result: DataFrame = DataFrame({'x': [], 'y': []}).quantile([0.1, 0.9], axis=0,
                                                                     interpolation=interpolation, method=method)
        expected: DataFrame = DataFrame({'x': [np.nan, np.nan], 'y': [np.nan, np.nan]}, index=[0.1, 0.9])
        tm.assert_frame_equal(result, expected)

    def test_quantile_datetime(self, unit: str) -> None:
        dti: pd.DatetimeIndex = pd.to_datetime(['2010', '2011']).as_unit(unit)
        df: DataFrame = DataFrame({'a': dti, 'b': [0, 5]})
        result: Series = df.quantile(0.5, numeric_only=True)
        expected: Series = Series([2.5], index=['b'], name=0.5)
        tm.assert_series_equal(result, expected)
        result = df.quantile(0.5, numeric_only=False)
        expected = Series([Timestamp('2010-07-02 12:00:00'), 2.5], index=['a', 'b'], name=0.5)
        tm.assert_series_equal(result, expected)
        result = df.quantile([0.5], numeric_only=False)
        expected_df = DataFrame({'a': Timestamp('2010-07-02 12:00:00').as_unit(unit), 'b': 2.5}, index=[0.5])
        tm.assert_frame_equal(result, expected_df)
        df['c'] = pd.to_datetime(['2011', '2012']).as_unit(unit)
        result = df[['a', 'c']].quantile(0.5, axis=1, numeric_only=False)
        expected = Series([Timestamp('2010-07-02 12:00:00'), Timestamp('2011-07-02 12:00:00')],
                          index=[0, 1], name=0.5, dtype=f'M8[{unit}]')
        tm.assert_series_equal(result, expected)
        result = df[['a', 'c']].quantile([0.5], axis=1, numeric_only=False)
        expected_df = DataFrame([[Timestamp('2010-07-02 12:00:00'), Timestamp('2011-07-02 12:00:00')]],
                                index=[0.5], columns=[0, 1], dtype=f'M8[{unit}]')
        tm.assert_frame_equal(result, expected_df)
        result = df[['a', 'c']].quantile(0.5, numeric_only=True)
        expected = Series([], index=Index([], dtype='str'), dtype=np.float64, name=0.5)
        tm.assert_series_equal(result, expected)
        result = df[['a', 'c']].quantile([0.5], numeric_only=True)
        expected_df = DataFrame(index=[0.5], columns=Index([], dtype='str'))
        tm.assert_frame_equal(result, expected_df)

    @pytest.mark.parametrize('dtype', ['datetime64[ns]', 'datetime64[ns, US/Pacific]', 'timedelta64[ns]', 'Period[D]'])
    def test_quantile_dt64_empty(self, dtype: str, interp_method: List[str]) -> None:
        interpolation, method = interp_method
        df: DataFrame = DataFrame(columns=['a', 'b'], dtype=dtype)
        res: Series = df.quantile(0.5, axis=1, numeric_only=False,
                                  interpolation=interpolation, method=method)
        expected: Series = Series([], index=Index([], dtype='str'), name=0.5, dtype=dtype)
        tm.assert_series_equal(res, expected)
        res = df.quantile([0.5], axis=1, numeric_only=False,
                          interpolation=interpolation, method=method)
        expected_df = DataFrame(index=[0.5], columns=Index([], dtype='str'))
        tm.assert_frame_equal(res, expected_df)

    @pytest.mark.parametrize('invalid', [-1, 2, [0.5, -1], [0.5, 2]])
    def test_quantile_invalid(self, invalid: Union[int, List[float]], datetime_frame: DataFrame, interp_method: List[str]) -> None:
        msg: str = 'percentiles should all be in the interval \\[0, 1\\]'
        interpolation, method = interp_method
        with pytest.raises(ValueError, match=msg):
            datetime_frame.quantile(invalid, interpolation=interpolation, method=method)

    def test_quantile_box(self, interp_method: List[str]) -> None:
        interpolation, method = interp_method
        df: DataFrame = DataFrame({
            'A': [Timestamp('2011-01-01'), Timestamp('2011-01-02'), Timestamp('2011-01-03')],
            'B': [Timestamp('2011-01-01', tz='US/Eastern'),
                  Timestamp('2011-01-02', tz='US/Eastern'),
                  Timestamp('2011-01-03', tz='US/Eastern')],
            'C': [pd.Timedelta('1 days'), pd.Timedelta('2 days'), pd.Timedelta('3 days')]
        })
        res: Series = df.quantile(0.5, numeric_only=False, interpolation=interpolation, method=method)
        exp: Series = Series([Timestamp('2011-01-02'),
                              Timestamp('2011-01-02', tz='US/Eastern'),
                              pd.Timedelta('2 days')],
                             name=0.5, index=['A', 'B', 'C'])
        tm.assert_series_equal(res, exp)
        res = df.quantile([0.5], numeric_only=False, interpolation=interpolation, method=method)
        exp_df: DataFrame = DataFrame([[Timestamp('2011-01-02'),
                                        Timestamp('2011-01-02', tz='US/Eastern'),
                                        pd.Timedelta('2 days')]],
                                      index=[0.5], columns=['A', 'B', 'C'])
        tm.assert_frame_equal(res, exp_df)

    def test_quantile_box_nat(self) -> None:
        df: DataFrame = DataFrame({
            'A': [Timestamp('2011-01-01'), pd.NaT, Timestamp('2011-01-02'), Timestamp('2011-01-03')],
            'a': [Timestamp('2011-01-01'), Timestamp('2011-01-02'), pd.NaT, Timestamp('2011-01-03')],
            'B': [Timestamp('2011-01-01', tz='US/Eastern'), pd.NaT, Timestamp('2011-01-02', tz='US/Eastern'),
                  Timestamp('2011-01-03', tz='US/Eastern')],
            'b': [Timestamp('2011-01-01', tz='US/Eastern'), Timestamp('2011-01-02', tz='US/Eastern'),
                  pd.NaT, Timestamp('2011-01-03', tz='US/Eastern')],
            'C': [pd.Timedelta('1 days'), pd.Timedelta('2 days'), pd.Timedelta('3 days'), pd.NaT],
            'c': [pd.NaT, pd.Timedelta('1 days'), pd.Timedelta('2 days'), pd.Timedelta('3 days')]
        }, columns=list('AaBbCc'))
        res: Series = df.quantile(0.5, numeric_only=False)
        exp: Series = Series([Timestamp('2011-01-02'),
                                Timestamp('2011-01-02'),
                                Timestamp('2011-01-02', tz='US/Eastern'),
                                Timestamp('2011-01-02', tz='US/Eastern'),
                                pd.Timedelta('2 days'),
                                pd.Timedelta('2 days')],
                               name=0.5, index=list('AaBbCc'))
        tm.assert_series_equal(res, exp)
        res = df.quantile([0.5], numeric_only=False)
        exp_df = DataFrame([[Timestamp('2011-01-02'),
                              Timestamp('2011-01-02'),
                              Timestamp('2011-01-02', tz='US/Eastern'),
                              Timestamp('2011-01-02', tz='US/Eastern'),
                              pd.Timedelta('2 days'),
                              pd.Timedelta('2 days')]],
                             index=[0.5], columns=list('AaBbCc'))
        tm.assert_frame_equal(res, exp_df)

    def test_quantile_nan(self, interp_method: List[str]) -> None:
        interpolation, method = interp_method
        df: DataFrame = DataFrame({'a': np.arange(1, 6.0), 'b': np.arange(1, 6.0)})
        df.iloc[-1, 1] = np.nan
        res: Series = df.quantile(0.5, interpolation=interpolation, method=method)
        exp: Series = Series([3.0, 2.5 if interpolation == 'linear' else 3.0],
                             index=['a', 'b'], name=0.5)
        tm.assert_series_equal(res, exp)
        res = df.quantile([0.5, 0.75], interpolation=interpolation, method=method)
        exp_df = DataFrame({'a': [3.0, 4.0],
                            'b': [2.5, 3.25] if interpolation == 'linear' else [3.0, 4.0]},
                           index=[0.5, 0.75])
        tm.assert_frame_equal(res, exp_df)
        res = df.quantile(0.5, axis=1, interpolation=interpolation, method=method)
        exp = Series(np.arange(1.0, 6.0), name=0.5)
        tm.assert_series_equal(res, exp)
        res = df.quantile([0.5, 0.75], axis=1, interpolation=interpolation, method=method)
        exp_df = DataFrame([np.arange(1.0, 6.0)] * 2, index=[0.5, 0.75])
        if interpolation == 'nearest':
            exp_df.iloc[1, -1] = np.nan
        tm.assert_frame_equal(res, exp_df)
        df['b'] = np.nan
        res = df.quantile(0.5, interpolation=interpolation, method=method)
        exp = Series([3.0, np.nan], index=['a', 'b'], name=0.5)
        tm.assert_series_equal(res, exp)
        res = df.quantile([0.5, 0.75], interpolation=interpolation, method=method)
        exp_df = DataFrame({'a': [3.0, 4.0], 'b': [np.nan, np.nan]}, index=[0.5, 0.75])
        tm.assert_frame_equal(res, exp_df)

    def test_quantile_nat(self, interp_method: List[str], unit: str) -> None:
        interpolation, method = interp_method
        df: DataFrame = DataFrame({'a': [pd.NaT, pd.NaT, pd.NaT]}, dtype=f'M8[{unit}]')
        res: Series = df.quantile(0.5, numeric_only=False, interpolation=interpolation, method=method)
        exp: Series = Series([pd.NaT], index=['a'], name=0.5, dtype=f'M8[{unit}]')
        tm.assert_series_equal(res, exp)
        res = df.quantile([0.5], numeric_only=False, interpolation=interpolation, method=method)
        exp_df = DataFrame({'a': [pd.NaT]}, index=[0.5], dtype=f'M8[{unit}]')
        tm.assert_frame_equal(res, exp_df)
        df = DataFrame({
            'a': [Timestamp('2012-01-01'), Timestamp('2012-01-02'), Timestamp('2012-01-03')],
            'b': [pd.NaT, pd.NaT, pd.NaT]
        }, dtype=f'M8[{unit}]')
        res = df.quantile(0.5, numeric_only=False, interpolation=interpolation, method=method)
        exp = Series([Timestamp('2012-01-02'), pd.NaT], index=['a', 'b'], name=0.5, dtype=f'M8[{unit}]')
        tm.assert_series_equal(res, exp)
        res = df.quantile([0.5], numeric_only=False, interpolation=interpolation, method=method)
        exp_df = DataFrame([[Timestamp('2012-01-02'), pd.NaT]], index=[0.5], columns=['a', 'b'], dtype=f'M8[{unit}]')
        tm.assert_frame_equal(res, exp_df)

    def test_quantile_empty_no_rows_floats(self, interp_method: List[str]) -> None:
        interpolation, method = interp_method
        df: DataFrame = DataFrame(columns=['a', 'b'], dtype='float64')
        res: Series = df.quantile(0.5, interpolation=interpolation, method=method)
        expected: Series = Series([np.nan, np.nan], index=['a', 'b'], name=0.5)
        tm.assert_series_equal(res, expected)
        res = df.quantile([0.5], interpolation=interpolation, method=method)
        expected_df = DataFrame([[np.nan, np.nan]], columns=['a', 'b'], index=[0.5])
        tm.assert_frame_equal(res, expected_df)
        res = df.quantile(0.5, axis=1, interpolation=interpolation, method=method)
        expected = Series([], index=Index([], dtype='str'), dtype='float64', name=0.5)
        tm.assert_series_equal(res, expected)
        res = df.quantile([0.5], axis=1, interpolation=interpolation, method=method)
        expected_df = DataFrame(columns=Index([], dtype='str'), index=[0.5])
        tm.assert_frame_equal(res, expected_df)

    def test_quantile_empty_no_rows_ints(self, interp_method: List[str]) -> None:
        interpolation, method = interp_method
        df: DataFrame = DataFrame(columns=['a', 'b'], dtype='int64')
        res: Series = df.quantile(0.5, interpolation=interpolation, method=method)
        expected: Series = Series([np.nan, np.nan], index=['a', 'b'], name=0.5)
        tm.assert_series_equal(res, expected)

    def test_quantile_empty_no_rows_dt64(self, interp_method: List[str]) -> None:
        interpolation, method = interp_method
        df: DataFrame = DataFrame(columns=['a', 'b'], dtype='datetime64[ns]')
        res: Series = df.quantile(0.5, numeric_only=False, interpolation=interpolation, method=method)
        expected: Series = Series([pd.NaT, pd.NaT], index=['a', 'b'], dtype='datetime64[ns]', name=0.5)
        tm.assert_series_equal(res, expected)
        df['a'] = df['a'].dt.tz_localize('US/Central')
        res = df.quantile(0.5, numeric_only=False, interpolation=interpolation, method=method)
        expected = expected.astype(object)
        if interpolation == 'nearest':
            expected = expected.fillna(np.nan)
        tm.assert_series_equal(res, expected)
        df['b'] = df['b'].dt.tz_localize('US/Central')
        res = df.quantile(0.5, numeric_only=False, interpolation=interpolation, method=method)
        expected = expected.astype(df['b'].dtype)
        tm.assert_series_equal(res, expected)

    def test_quantile_empty_no_columns(self, interp_method: List[str]) -> None:
        interpolation, method = interp_method
        df: DataFrame = DataFrame(pd.date_range('1/1/18', periods=5))
        df.columns.name = 'captain tightpants'
        result: Series = df.quantile(0.5, numeric_only=True, interpolation=interpolation, method=method)
        expected: Series = Series([], name=0.5, dtype=np.float64)
        expected.index.name = 'captain tightpants'
        tm.assert_series_equal(result, expected)
        result = df.quantile([0.5], numeric_only=True, interpolation=interpolation, method=method)
        expected_df = DataFrame([], index=[0.5])
        expected_df.columns.name = 'captain tightpants'
        tm.assert_frame_equal(result, expected_df)

    def test_quantile_item_cache(self, interp_method: List[str]) -> None:
        interpolation, method = interp_method
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((4, 3)),
                                  columns=['A', 'B', 'C'])
        df['D'] = df['A'] * 2
        ser: Series = df['A']
        assert len(df._mgr.blocks) == 2
        df.quantile(numeric_only=False, interpolation=interpolation, method=method)
        ser.iloc[0] = 99
        assert df.iloc[0, 0] == df['A'][0]
        assert df.iloc[0, 0] != 99

    def test_invalid_method(self) -> None:
        with pytest.raises(ValueError, match='Invalid method: foo'):
            DataFrame(range(1)).quantile(0.5, method='foo')

    def test_table_invalid_interpolation(self) -> None:
        with pytest.raises(ValueError, match='Invalid interpolation: foo'):
            DataFrame(range(1)).quantile(0.5, method='table', interpolation='foo')

class TestQuantileExtensionDtype:

    @pytest.fixture(params=[
        pytest.param(pd.IntervalIndex.from_breaks(range(10)), marks=pytest.mark.xfail(reason='raises when trying to add Intervals')),
        pd.period_range('2016-01-01', periods=9, freq='D'),
        pd.date_range('2016-01-01', periods=9, tz='US/Pacific'),
        pd.timedelta_range('1 Day', periods=9),
        pd.array(np.arange(9), dtype='Int64'),
        pd.array(np.arange(9), dtype='Float64')
    ], ids=lambda x: str(x.dtype))
    def index(self, request: pytest.FixtureRequest) -> Index:
        idx: Index = request.param
        idx.name = 'A'
        return idx

    @pytest.fixture
    def obj(self, index: Index, frame_or_series: Any) -> Union[DataFrame, Series]:
        obj = frame_or_series(index).copy()
        if frame_or_series is Series:
            obj.name = 'A'
        else:
            obj.columns = ['A']
        return obj

    def compute_quantile(self, obj: Union[DataFrame, Series], qs: Union[float, List[float]]) -> Union[Series, DataFrame]:
        if isinstance(obj, Series):
            result = obj.quantile(qs)
        else:
            result = obj.quantile(qs, numeric_only=False)
        return result

    def test_quantile_ea(self, request: Any, obj: Union[DataFrame, Series], index: Index) -> None:
        indexer: np.ndarray = np.arange(len(index), dtype=np.intp)
        np.random.default_rng(2).shuffle(indexer)
        obj = obj.iloc[indexer]
        qs: List[float] = [0.5, 0, 1]
        result = self.compute_quantile(obj, qs)
        exp_dtype = index.dtype
        if index.dtype == 'Int64':
            exp_dtype = 'Float64'
        expected: Series = Series([index[4], index[0], index[-1]], dtype=exp_dtype, index=qs, name='A')
        expected = type(obj)(expected)
        tm.assert_equal(result, expected)

    def test_quantile_ea_with_na(self, obj: Union[DataFrame, Series], index: Index) -> None:
        obj.iloc[0] = index._na_value
        obj.iloc[-1] = index._na_value
        indexer: np.ndarray = np.arange(len(index), dtype=np.intp)
        np.random.default_rng(2).shuffle(indexer)
        obj = obj.iloc[indexer]
        qs: List[float] = [0.5, 0, 1]
        result = self.compute_quantile(obj, qs)
        expected: Series = Series([index[4], index[1], index[-2]], dtype=index.dtype, index=qs, name='A')
        expected = type(obj)(expected)
        tm.assert_equal(result, expected)

    def test_quantile_ea_all_na(self, request: Any, obj: Union[DataFrame, Series], index: Index) -> None:
        obj.iloc[:] = index._na_value
        assert np.all(obj.dtypes == index.dtype)
        indexer: np.ndarray = np.arange(len(index), dtype=np.intp)
        np.random.default_rng(2).shuffle(indexer)
        obj = obj.iloc[indexer]
        qs: List[float] = [0.5, 0, 1]
        result = self.compute_quantile(obj, qs)
        expected = index.take([-1, -1, -1], allow_fill=True, fill_value=index._na_value)
        expected = Series(expected, index=qs, name='A')
        expected = type(obj)(expected)
        tm.assert_equal(result, expected)

    def test_quantile_ea_scalar(self, request: Any, obj: Union[DataFrame, Series], index: Index) -> None:
        indexer: np.ndarray = np.arange(len(index), dtype=np.intp)
        np.random.default_rng(2).shuffle(indexer)
        obj = obj.iloc[indexer]
        qs: float = 0.5
        result = self.compute_quantile(obj, qs)
        exp_dtype = index.dtype
        if index.dtype == 'Int64':
            exp_dtype = 'Float64'
        expected: Series = Series({'A': index[4]}, dtype=exp_dtype, name=0.5)
        if isinstance(obj, Series):
            expected_scalar = expected['A']
            assert result == expected_scalar
        else:
            tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('dtype, expected_data, expected_index, axis', [
        ['float64', [], [], 1],
        ['int64', [], [], 1],
        ['float64', [np.nan, np.nan], ['a', 'b'], 0],
        ['int64', [np.nan, np.nan], ['a', 'b'], 0]
    ])
    def test_empty_numeric(self, dtype: str, expected_data: list, expected_index: list, axis: int) -> None:
        df: DataFrame = DataFrame(columns=['a', 'b'], dtype=dtype)
        result: Series = df.quantile(0.5, axis=axis)
        expected: Series = Series(expected_data, name=0.5, index=Index(expected_index, dtype='str'), dtype='float64')
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('dtype, expected_data, expected_index, axis, expected_dtype', [
        ['datetime64[ns]', [], [], 1, 'datetime64[ns]'],
        ['datetime64[ns]', [pd.NaT, pd.NaT], ['a', 'b'], 0, 'datetime64[ns]']
    ])
    def test_empty_datelike(self, dtype: str, expected_data: list, expected_index: list, axis: int, expected_dtype: str) -> None:
        df: DataFrame = DataFrame(columns=['a', 'b'], dtype=dtype)
        result: Series = df.quantile(0.5, axis=axis, numeric_only=False)
        expected: Series = Series(expected_data, name=0.5, index=Index(expected_index, dtype='str'), dtype=expected_dtype)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('expected_data, expected_index, axis', [
        [[np.nan, np.nan], list(range(2)), 1],
        [[], [], 0]
    ])
    def test_datelike_numeric_only(self, expected_data: list, expected_index: list, axis: int) -> None:
        df: DataFrame = DataFrame({
            'a': pd.to_datetime(['2010', '2011']),
            'b': [0, 5],
            'c': pd.to_datetime(['2011', '2012'])
        })
        result: Series = df[['a', 'c']].quantile(0.5, axis=axis, numeric_only=True)
        expected: Series = Series(expected_data, name=0.5, index=Index(expected_index, dtype='str' if axis == 0 else 'int64'), dtype=np.float64)
        tm.assert_series_equal(result, expected)

def test_multi_quantile_numeric_only_retains_columns() -> None:
    df: DataFrame = DataFrame(list('abc'))
    result: DataFrame = df.quantile([0.5, 0.7], numeric_only=True)
    expected: DataFrame = DataFrame(index=[0.5, 0.7])
    tm.assert_frame_equal(result, expected, check_index_type=True, check_column_type=True)