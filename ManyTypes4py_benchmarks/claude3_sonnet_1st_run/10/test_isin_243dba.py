import numpy as np
import pytest
import pandas as pd
from pandas import Series, date_range
import pandas._testing as tm
from pandas.core import algorithms
from pandas.core.arrays import PeriodArray
from typing import List, Set, Dict, Any, Union, Iterable, Optional, Tuple

class TestSeriesIsIn:

    def test_isin(self) -> None:
        s = Series(['A', 'B', 'C', 'a', 'B', 'B', 'A', 'C'])
        result = s.isin(['A', 'C'])
        expected = Series([True, False, True, False, False, False, True, True])
        tm.assert_series_equal(result, expected)
        s = Series(list('abcdefghijk' * 10 ** 5))
        in_list: List[Union[int, str]] = [-1, 'a', 'b', 'G', 'Y', 'Z', 'E', 'K', 'E', 'S', 'I', 'R', 'R'] * 6
        assert s.isin(in_list).sum() == 200000

    def test_isin_with_string_scalar(self) -> None:
        s = Series(['A', 'B', 'C', 'a', 'B', 'B', 'A', 'C'])
        msg = 'only list-like objects are allowed to be passed to isin\\(\\), you passed a `str`'
        with pytest.raises(TypeError, match=msg):
            s.isin('a')
        s = Series(['aaa', 'b', 'c'])
        with pytest.raises(TypeError, match=msg):
            s.isin('aaa')

    def test_isin_datetimelike_mismatched_reso(self) -> None:
        expected = Series([True, True, False, False, False])
        ser = Series(date_range('jan-01-2013', 'jan-05-2013'))
        day_values = np.asarray(ser[0:2].values).astype('datetime64[D]')
        result = ser.isin(day_values)
        tm.assert_series_equal(result, expected)
        dta = ser[:2]._values.astype('M8[s]')
        result = ser.isin(dta)
        tm.assert_series_equal(result, expected)

    def test_isin_datetimelike_mismatched_reso_list(self) -> None:
        expected = Series([True, True, False, False, False])
        ser = Series(date_range('jan-01-2013', 'jan-05-2013'))
        dta = ser[:2]._values.astype('M8[s]')
        result = ser.isin(list(dta))
        tm.assert_series_equal(result, expected)

    def test_isin_with_i8(self) -> None:
        expected = Series([True, True, False, False, False])
        expected2 = Series([False, True, False, False, False])
        s = Series(date_range('jan-01-2013', 'jan-05-2013'))
        result = s.isin(s[0:2])
        tm.assert_series_equal(result, expected)
        result = s.isin(s[0:2].values)
        tm.assert_series_equal(result, expected)
        result = s.isin([s[1]])
        tm.assert_series_equal(result, expected2)
        result = s.isin([np.datetime64(s[1])])
        tm.assert_series_equal(result, expected2)
        result = s.isin(set(s[0:2]))
        tm.assert_series_equal(result, expected)
        s = Series(pd.to_timedelta(range(5), unit='D'))
        result = s.isin(s[0:2])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('empty', [[], Series(dtype=object), np.array([])])
    def test_isin_empty(self, empty: Union[List[Any], Series, np.ndarray]) -> None:
        s = Series(['a', 'b'])
        expected = Series([False, False])
        result = s.isin(empty)
        tm.assert_series_equal(expected, result)

    def test_isin_read_only(self) -> None:
        arr = np.array([1, 2, 3])
        arr.setflags(write=False)
        s = Series([1, 2, 3])
        result = s.isin(arr)
        expected = Series([True, True, True])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('dtype', [object, None])
    def test_isin_dt64_values_vs_ints(self, dtype: Optional[type]) -> None:
        dti = date_range('2013-01-01', '2013-01-05')
        ser = Series(dti)
        comps = np.asarray([1356998400000000000], dtype=dtype)
        res = dti.isin(comps)
        expected = np.array([False] * len(dti), dtype=bool)
        tm.assert_numpy_array_equal(res, expected)
        res = ser.isin(comps)
        tm.assert_series_equal(res, Series(expected))
        res = pd.core.algorithms.isin(ser, comps)
        tm.assert_numpy_array_equal(res, expected)

    def test_isin_tzawareness_mismatch(self) -> None:
        dti = date_range('2013-01-01', '2013-01-05')
        ser = Series(dti)
        other = dti.tz_localize('UTC')
        res = dti.isin(other)
        expected = np.array([False] * len(dti), dtype=bool)
        tm.assert_numpy_array_equal(res, expected)
        res = ser.isin(other)
        tm.assert_series_equal(res, Series(expected))
        res = pd.core.algorithms.isin(ser, other)
        tm.assert_numpy_array_equal(res, expected)

    def test_isin_period_freq_mismatch(self) -> None:
        dti = date_range('2013-01-01', '2013-01-05')
        pi = dti.to_period('M')
        ser = Series(pi)
        dtype = dti.to_period('Y').dtype
        other = PeriodArray._simple_new(pi.asi8, dtype=dtype)
        res = pi.isin(other)
        expected = np.array([False] * len(pi), dtype=bool)
        tm.assert_numpy_array_equal(res, expected)
        res = ser.isin(other)
        tm.assert_series_equal(res, Series(expected))
        res = pd.core.algorithms.isin(ser, other)
        tm.assert_numpy_array_equal(res, expected)

    @pytest.mark.parametrize('values', [[-9.0, 0.0], [-9, 0]])
    def test_isin_float_in_int_series(self, values: List[Union[float, int]]) -> None:
        ser = Series(values)
        result = ser.isin([-9, -0.5])
        expected = Series([True, False])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('dtype', ['boolean', 'Int64', 'Float64'])
    @pytest.mark.parametrize('data,values,expected', [
        ([0, 1, 0], [1], [False, True, False]),
        ([0, 1, 0], [1, pd.NA], [False, True, False]),
        ([0, pd.NA, 0], [1, 0], [True, False, True]),
        ([0, 1, pd.NA], [1, pd.NA], [False, True, True]),
        ([0, 1, pd.NA], [1, np.nan], [False, True, False]),
        ([0, pd.NA, pd.NA], [np.nan, pd.NaT, None], [False, False, False])
    ])
    def test_isin_masked_types(self, dtype: str, data: List[Any], values: List[Any], expected: List[bool]) -> None:
        ser = Series(data, dtype=dtype)
        result = ser.isin(values)
        expected = Series(expected, dtype='boolean')
        tm.assert_series_equal(result, expected)

def test_isin_large_series_mixed_dtypes_and_nan(monkeypatch: pytest.MonkeyPatch) -> None:
    min_isin_comp = 5
    ser = Series([1, 2, np.nan] * min_isin_comp)
    with monkeypatch.context() as m:
        m.setattr(algorithms, '_MINIMUM_COMP_ARR_LEN', min_isin_comp)
        result = ser.isin({'foo', 'bar'})
    expected = Series([False] * 3 * min_isin_comp)
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('dtype, data, values, expected', [
    ('boolean', [pd.NA, False, True], [False, pd.NA], [True, True, False]),
    ('Int64', [pd.NA, 2, 1], [1, pd.NA], [True, False, True]),
    ('boolean', [pd.NA, False, True], [pd.NA, True, 'a', 20], [True, False, True]),
    ('boolean', [pd.NA, False, True], [], [False, False, False]),
    ('Float64', [20.0, 30.0, pd.NA], [pd.NA], [False, False, True])
])
def test_isin_large_series_and_pdNA(
    dtype: str, 
    data: List[Any], 
    values: List[Any], 
    expected: List[bool], 
    monkeypatch: pytest.MonkeyPatch
) -> None:
    min_isin_comp = 2
    ser = Series(data, dtype=dtype)
    expected = Series(expected, dtype='boolean')
    with monkeypatch.context() as m:
        m.setattr(algorithms, '_MINIMUM_COMP_ARR_LEN', min_isin_comp)
        result = ser.isin(values)
    tm.assert_series_equal(result, expected)

def test_isin_complex_numbers() -> None:
    array: List[complex] = [0, 1j, 1j, 1, 1 + 1j, 1 + 2j, 1 + 1j]
    result = Series(array).isin([1j, 1 + 1j, 1 + 2j])
    expected = Series([False, True, True, False, True, True, True], dtype=bool)
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('data,is_in', [
    ([1, [2]], [1]),
    (['simple str', [{'values': 3}]], ['simple str'])
])
def test_isin_filtering_with_mixed_object_types(data: List[Any], is_in: List[Any]) -> None:
    ser = Series(data)
    result = ser.isin(is_in)
    expected = Series([True, False])
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('data', [[1, 2, 3], [1.0, 2.0, 3.0]])
@pytest.mark.parametrize('isin', [[1, 2], [1.0, 2.0]])
def test_isin_filtering_on_iterable(data: List[Union[int, float]], isin: List[Union[int, float]]) -> None:
    ser = Series(data)
    result = ser.isin((i for i in isin))
    expected_result = Series([True, True, False])
    tm.assert_series_equal(result, expected_result)
