import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame, Index, Series, Timestamp
import pandas._testing as tm
from typing import Any, List, Tuple, Union, Optional, Dict

@pytest.fixture(params=[['linear', 'single'], ['nearest', 'table']], ids=lambda x: '-'.join(x))
def interp_method(request: pytest.FixtureRequest) -> List[str]:
    """(interpolation, method) arguments for quantile"""
    return request.param

class TestDataFrameQuantile:

    @pytest.mark.parametrize('df,expected', [[DataFrame({0: Series(pd.arrays.SparseArray([1, 2])), 1: Series(pd.arrays.SparseArray([3, 4]))}), Series([1.5, 3.5], name=0.5)], [DataFrame(Series([0.0, None, 1.0, 2.0], dtype='Sparse[float]')), Series([1.0], name=0.5)]])
    def test_quantile_sparse(self, df: DataFrame, expected: Series) -> None:
        result: Series = df.quantile()
        expected = expected.astype('Sparse[float]')
        tm.assert_series_equal(result, expected)

    def test_quantile(self, datetime_frame: DataFrame, interp_method: List[str], request: pytest.FixtureRequest) -> None:
        interpolation, method = interp_method
        df: DataFrame = datetime_frame
        result: Series = df.quantile(0.1, axis=0, numeric_only=True, interpolation=interpolation, method=method)
        expected: Series = Series([np.percentile(df[col], 10) for col in df.columns], index=df.columns, name=0.1)
        if interpolation == 'linear':
            tm.assert_series_equal(result, expected)
        else:
            tm.assert_index_equal(result.index, expected.index)
            assert result.name == expected.name
        result = df.quantile(0.9, axis=1, numeric_only=True, interpolation=interpolation, method=method)
        expected = Series([np.percentile(df.loc[date], 90) for date in df.index], index=df.index, name=0.9)
        if interpolation == 'linear':
            tm.assert_series_equal(result, expected)
        else:
            tm.assert_index_equal(result.index, expected.index)
            assert result.name == expected.name

    def test_empty(self, interp_method: List[str]) -> None:
        interpolation, method = interp_method
        q: Series = DataFrame({'x': [], 'y': []}).quantile(0.1, axis=0, numeric_only=True, interpolation=interpolation, method=method)
        assert np.isnan(q['x']) and np.isnan(q['y'])

    def test_non_numeric_exclusion(self, interp_method: List[str], request: pytest.FixtureRequest) -> None:
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
        expected: DataFrame = DataFrame({1: [1.5, 1.75], 2: [2.5, 2.75], 3: [3.5, 3.75]}, index=[0.5, 0.75])
        if interpolation == 'nearest':
            expected.iloc[0, :] -= 0.5
            expected.iloc[1, :] += 0.25
            expected = expected.astype(np.int64)
        tm.assert_frame_equal(result, expected, check_index_type=True)

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
        df: DataFrame = DataFrame({'A': [1, 2, 3], 'B': [2.0, 3.0, 4.0], 'C': pd.date_range('20130101', periods=3), 'D': ['foo', 'bar', 'baz']})
        result: Series = df.quantile(0.5, axis=1, numeric_only=True, interpolation=interpolation, method=method)
        expected: Series = Series([1.5, 2.5, 3.5], name=0.5)
        if interpolation == 'nearest':
            expected -= 0.5
        tm.assert_series_equal(result, expected)
        msg = "'<' not supported between instances of 'Timestamp' and 'float'"
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
        expected = df.quantile(0.5, axis='index', interpolation=interpolation, method=method)
        if interpolation == 'nearest':
            expected = expected.astype(np.int64)
        tm.assert_series_equal(result, expected)
        result = df.quantile(0.5, axis=1, interpolation=interpolation, method=method)
        expected = Series([1.5, 2.5, 3.5], index=[1, 2, 3], name=0.5)
        if interpolation == 'nearest':
            expected = expected.astype(np.int64)
        tm.assert_series_equal(result, expected)
        result = df.quantile(0.5, axis='columns', interpolation=interpolation, method=method)
        tm.assert_series_equal(result, expected)
        msg = 'No axis named -1 for object type DataFrame'
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
        exp: np.ndarray = np.percentile(np.array([[1, 2, 3], [2, 3, 4]]), 0.5, axis=0, method='nearest')
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
        expected = DataFrame({1: [1.0, 1.0], 2: [2.0, 2.0], 3: [3.0, 3.0]}, index=[0.5, 0.75])
        tm.assert_frame_equal(result, expected)
        df = DataFrame({'x': [], 'y': []})
        q: Series = df.quantile(0.1, axis=0, interpolation='higher')
        assert np.isnan(q['x']) and np.isnan(q['y'])
        df = DataFrame([[1, 1, 1], [2, 2, 2], [3, 3, 3]], columns=['a', 'b', 'c'])
        result = df.quantile([0.25, 0.5], interpolation='midpoint')
        expected = DataFrame([[1.5, 1.5, 1.5], [2.0, 2.0, 2.0]], index=[0.25, 0.5], columns=['a', 'b', 'c'])
        tm.assert_frame_equal(result, expected)

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
        expected: DataFrame = DataFrame([[1.5, 1.5, 1.5], [2.0, 2.0, 2.0]], index=[0.25, 0.5], columns=['a', 'b', 'c'])
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
        result: DataFrame = DataFrame({'x': [], 'y': []}).quantile([0.1, 0.9], axis=0, interpolation=interpolation, method=method)
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
        expected = DataFrame({'a': Timestamp('2010-07-02 12:00:00').as_unit(unit), 'b': 2.5}, index=[0.5])
        tm.assert_frame_equal(result, expected)
        df['c'] = pd.to_datetime(['2011', '2012']).as_unit(unit)
        result = df[['a', 'c']].quantile(0.5, axis=1, numeric_only=False)
        expected = Series([Timestamp('2010-07-02 12:00:00'), Timestamp('2011-07-02 12:00:00')], index=[0, 1], name=0.5, dtype=f'M8[{unit}]')
        tm.assert_series_equal(result, expected)
        result = df[['a', 'c']].quantile([0.5], axis=1, numeric_only=False)
        expected = DataFrame([[Timestamp('2010-07-02 12:00:00'), Timestamp('2011-07-02 12:00:00')]], index=[0.5], columns=[0, 1], dtype=f'M8[{unit}]')
        tm.assert_frame_equal(result, expected)
        result = df[['a', 'c']].quantile(0.5, numeric_only=True)
        expected = Series([], index=Index([], dtype='str'), dtype=np.float64, name=0.5)
        tm.assert_series_equal(result, expected)
        result = df[['a', 'c']].quantile([0.5], numeric_only=True)
        expected = DataFrame(index=[0.5], columns=Index([], dtype='str'))
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('dtype', ['datetime64[ns]', 'datetime64[ns, US/Pacific]', 'timedelta64[ns]', 'Period[D]'])
    def test_quantile_dt64_empty(self, dtype: str, interp_method: List[str]) -> None:
        interpolation, method = interp_method
        df: DataFrame = DataFrame(columns=['a', 'b'], dtype=dtype)
        res: Series = df.quantile(0.5, axis=1, numeric_only=False, interpolation=interpolation, method=method)
