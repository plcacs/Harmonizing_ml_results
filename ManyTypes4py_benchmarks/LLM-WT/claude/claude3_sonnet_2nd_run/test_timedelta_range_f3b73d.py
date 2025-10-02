import numpy as np
import pytest
from pandas import Timedelta, timedelta_range, to_timedelta
import pandas._testing as tm
from pandas.tseries.offsets import Day, Second
from typing import Any, List, Tuple

class TestTimedeltas:

    def test_timedelta_range_unit(self) -> None:
        tdi = timedelta_range('0 Days', periods=10, freq='100000D', unit='s')
        exp_arr = (np.arange(10, dtype='i8') * 100000).view('m8[D]').astype('m8[s]')
        tm.assert_numpy_array_equal(tdi.to_numpy(), exp_arr)

    def test_timedelta_range(self) -> None:
        expected = to_timedelta(np.arange(5), unit='D')
        result = timedelta_range('0 days', periods=5, freq='D')
        tm.assert_index_equal(result, expected)
        expected = to_timedelta(np.arange(11), unit='D')
        result = timedelta_range('0 days', '10 days', freq='D')
        tm.assert_index_equal(result, expected)
        expected = to_timedelta(np.arange(5), unit='D') + Second(2) + Day()
        result = timedelta_range('1 days, 00:00:02', '5 days, 00:00:02', freq='D')
        tm.assert_index_equal(result, expected)
        expected = to_timedelta([1, 3, 5, 7, 9], unit='D') + Second(2)
        result = timedelta_range('1 days, 00:00:02', periods=5, freq='2D')
        tm.assert_index_equal(result, expected)
        expected = to_timedelta(np.arange(50), unit='min') * 30
        result = timedelta_range('0 days', freq='30min', periods=50)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('depr_unit, unit', [('H', 'hour'), ('S', 'second')])
    def test_timedelta_units_H_S_deprecated(self, depr_unit: str, unit: str) -> None:
        depr_msg = f"'{depr_unit}' is deprecated and will be removed in a future version."
        expected = to_timedelta(np.arange(5), unit=unit)
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            result = to_timedelta(np.arange(5), unit=depr_unit)
            tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('unit', ['T', 't', 'L', 'l', 'U', 'u', 'N', 'n'])
    def test_timedelta_unit_T_L_U_N_raises(self, unit: str) -> None:
        msg = f'invalid unit abbreviation: {unit}'
        with pytest.raises(ValueError, match=msg):
            to_timedelta(np.arange(5), unit=unit)

    @pytest.mark.parametrize('periods, freq', [(3, '2D'), (5, 'D'), (6, '19h12min'), (7, '16h'), (9, '12h')])
    def test_linspace_behavior(self, periods: int, freq: str) -> None:
        result = timedelta_range(start='0 days', end='4 days', periods=periods)
        expected = timedelta_range(start='0 days', end='4 days', freq=freq)
        tm.assert_index_equal(result, expected)

    def test_timedelta_range_H_raises(self) -> None:
        msg = 'Invalid frequency: H'
        with pytest.raises(ValueError, match=msg):
            timedelta_range(start='0 days', end='4 days', freq='19H12min')

    def test_timedelta_range_T_raises(self) -> None:
        msg = 'Invalid frequency: T'
        with pytest.raises(ValueError, match=msg):
            timedelta_range(start='0 days', end='4 days', freq='19h12T')

    def test_errors(self) -> None:
        msg = 'Of the four parameters: start, end, periods, and freq, exactly three must be specified'
        with pytest.raises(ValueError, match=msg):
            timedelta_range(start='0 days')
        with pytest.raises(ValueError, match=msg):
            timedelta_range(end='5 days')
        with pytest.raises(ValueError, match=msg):
            timedelta_range(periods=2)
        with pytest.raises(ValueError, match=msg):
            timedelta_range()
        with pytest.raises(ValueError, match=msg):
            timedelta_range(start='0 days', end='5 days', periods=10, freq='h')

    @pytest.mark.parametrize('start, end, freq, expected_periods', [('1D', '10D', '2D', (10 - 1) // 2 + 1), ('2D', '30D', '3D', (30 - 2) // 3 + 1), ('2s', '50s', '5s', (50 - 2) // 5 + 1), ('4D', '16D', '3D', (16 - 4) // 3 + 1), ('8D', '16D', '40s', (16 * 3600 * 24 - 8 * 3600 * 24) // 40 + 1)])
    def test_timedelta_range_freq_divide_end(self, start: str, end: str, freq: str, expected_periods: int) -> None:
        res = timedelta_range(start=start, end=end, freq=freq)
        assert Timedelta(start) == res[0]
        assert Timedelta(end) >= res[-1]
        assert len(res) == expected_periods

    def test_timedelta_range_infer_freq(self) -> None:
        result = timedelta_range('0s', '1s', periods=31)
        assert result.freq is None

    @pytest.mark.parametrize('freq_depr, start, end', [('3.5l', '05:03:01', '05:03:10'), ('2.5T', '5 hours', '5 hours 8 minutes'), ('3.5S', '05:03:01', '05:03:10')])
    def test_timedelta_range_removed_freq(self, freq_depr: str, start: str, end: str) -> None:
        msg = f'Invalid frequency: {freq_depr}'
        with pytest.raises(ValueError, match=msg):
            timedelta_range(start=start, end=end, freq=freq_depr)
