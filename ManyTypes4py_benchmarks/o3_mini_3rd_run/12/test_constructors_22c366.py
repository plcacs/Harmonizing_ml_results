from __future__ import annotations
from datetime import datetime, timedelta, timezone
from functools import partial
from operator import attrgetter
from typing import Any, Dict, List, Optional, Union, Callable
import zoneinfo
import dateutil
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
from pandas._libs.tslibs import astype_overflowsafe, timezones
import pandas as pd
from pandas import DatetimeIndex, Index, Timestamp, date_range, offsets, to_datetime
import pandas._testing as tm
from pandas.core.arrays import period_array

class TestDatetimeIndex:

    def test_from_dt64_unsupported_unit(self) -> None:
        val: np.datetime64 = np.datetime64(1, 'D')
        result: DatetimeIndex = DatetimeIndex([val], tz='US/Pacific')
        expected: DatetimeIndex = DatetimeIndex([val.astype('M8[s]')], tz='US/Pacific')
        tm.assert_index_equal(result, expected)

    def test_explicit_tz_none(self) -> None:
        dti: DatetimeIndex = date_range('2016-01-01', periods=10, tz='UTC')
        msg: str = "Passed data is timezone-aware, incompatible with 'tz=None'"
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(dti, tz=None)
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(np.array(dti), tz=None)
        msg = 'Cannot pass both a timezone-aware dtype and tz=None'
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex([], dtype='M8[ns, UTC]', tz=None)

    def test_freq_validation_with_nat(self) -> None:
        msg: str = 'Inferred frequency None from passed values does not conform to passed frequency D'
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex([pd.NaT, Timestamp('2011-01-01')], freq='D')
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex([pd.NaT, Timestamp('2011-01-01')._value], freq='D')

    @pytest.mark.parametrize('index', [date_range('2016-01-01', periods=5, tz='US/Pacific'), pd.timedelta_range('1 Day', periods=5)])
    def test_shallow_copy_inherits_array_freq(self, index: Union[DatetimeIndex, pd.TimedeltaIndex]) -> None:
        array = index._data
        arr = array[[0, 3, 2, 4, 1]]
        assert arr.freq is None
        result = index._shallow_copy(arr)
        assert result.freq is None

    def test_categorical_preserves_tz(self) -> None:
        dti: DatetimeIndex = DatetimeIndex([pd.NaT, '2015-01-01', '1999-04-06 15:14:13', '2015-01-01'], tz='US/Eastern')
        for dtobj in [dti, dti._data]:
            ci = pd.CategoricalIndex(dtobj)
            carr = pd.Categorical(dtobj)
            cser = pd.Series(ci)
            for obj in [ci, carr, cser]:
                result = DatetimeIndex(obj)
                tm.assert_index_equal(result, dti)

    def test_dti_with_period_data_raises(self) -> None:
        data = pd.PeriodIndex(['2016Q1', '2016Q2'], freq='Q')
        with pytest.raises(TypeError, match='PeriodDtype data is invalid'):
            DatetimeIndex(data)
        with pytest.raises(TypeError, match='PeriodDtype data is invalid'):
            to_datetime(data)
        with pytest.raises(TypeError, match='PeriodDtype data is invalid'):
            DatetimeIndex(period_array(data))
        with pytest.raises(TypeError, match='PeriodDtype data is invalid'):
            to_datetime(period_array(data))

    def test_dti_with_timedelta64_data_raises(self) -> None:
        data: np.ndarray = np.array([0], dtype='m8[ns]')
        msg: str = 'timedelta64\\[ns\\] cannot be converted to datetime64'
        with pytest.raises(TypeError, match=msg):
            DatetimeIndex(data)
        with pytest.raises(TypeError, match=msg):
            to_datetime(data)
        with pytest.raises(TypeError, match=msg):
            DatetimeIndex(pd.TimedeltaIndex(data))
        with pytest.raises(TypeError, match=msg):
            to_datetime(pd.TimedeltaIndex(data))

    def test_constructor_from_sparse_array(self) -> None:
        values: List[Timestamp] = [Timestamp('2012-05-01T01:00:00.000000'), Timestamp('2016-05-01T01:00:00.000000')]
        arr = pd.arrays.SparseArray(values)
        result = Index(arr)
        assert type(result) is Index
        assert result.dtype == arr.dtype

    def test_construction_caching(self) -> None:
        df = pd.DataFrame({
            'dt': date_range('20130101', periods=3),
            'dttz': date_range('20130101', periods=3, tz=zoneinfo.ZoneInfo('US/Eastern')),
            'dt_with_null': [Timestamp('20130101'), pd.NaT, Timestamp('20130103')],
            'dtns': date_range('20130101', periods=3, freq='ns')
        })
        assert df.dttz.dtype.tz.key == 'US/Eastern'

    @pytest.mark.parametrize('kwargs', [{'tz': 'dtype.tz'}, {'dtype': 'dtype'}, {'dtype': 'dtype', 'tz': 'dtype.tz'}])
    def test_construction_with_alt(self, kwargs: Dict[str, str], tz_aware_fixture: Any) -> None:
        tz = tz_aware_fixture
        i = date_range('20130101', periods=5, freq='h', tz=tz)
        kwargs = {key: attrgetter(val)(i) for key, val in kwargs.items()}
        result = DatetimeIndex(i, **kwargs)
        tm.assert_index_equal(i, result)

    @pytest.mark.parametrize('kwargs', [{'tz': 'dtype.tz'}, {'dtype': 'dtype'}, {'dtype': 'dtype', 'tz': 'dtype.tz'}])
    def test_construction_with_alt_tz_localize(self, kwargs: Dict[str, str], tz_aware_fixture: Any) -> None:
        tz = tz_aware_fixture
        i = date_range('20130101', periods=5, freq='h', tz=tz)
        i = i._with_freq(None)
        kwargs = {key: attrgetter(val)(i) for key, val in kwargs.items()}
        if 'tz' in kwargs:
            result = DatetimeIndex(i.asi8, tz='UTC').tz_convert(kwargs['tz'])
            expected = DatetimeIndex(i, **kwargs)
            tm.assert_index_equal(result, expected)
        i2 = DatetimeIndex(i.tz_localize(None).asi8, tz='UTC')
        expected = i.tz_localize(None).tz_localize('UTC')
        tm.assert_index_equal(i2, expected)
        msg = 'cannot supply both a tz and a dtype with a tz'
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(i.tz_localize(None).asi8, dtype=i.dtype, tz=zoneinfo.ZoneInfo('US/Hawaii'))

    def test_construction_index_with_mixed_timezones(self) -> None:
        result = Index([Timestamp('2011-01-01'), Timestamp('2011-01-02')], name='idx')
        exp = DatetimeIndex([Timestamp('2011-01-01'), Timestamp('2011-01-02')], name='idx')
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is None
        result = Index([Timestamp('2011-01-01 10:00', tz='Asia/Tokyo'),
                        Timestamp('2011-01-02 10:00', tz='Asia/Tokyo')], name='idx')
        exp = DatetimeIndex([Timestamp('2011-01-01 10:00'), Timestamp('2011-01-02 10:00')],
                            tz='Asia/Tokyo', name='idx')
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is not None
        assert result.tz == exp.tz
        result = Index([Timestamp('2011-01-01 10:00', tz='US/Eastern'),
                        Timestamp('2011-08-01 10:00', tz='US/Eastern')], name='idx')
        exp = DatetimeIndex([Timestamp('2011-01-01 10:00'), Timestamp('2011-08-01 10:00')],
                            tz='US/Eastern', name='idx')
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is not None
        assert result.tz == exp.tz
        result = Index([Timestamp('2011-01-01 10:00'), Timestamp('2011-01-02 10:00', tz='US/Eastern')], name='idx')
        exp = Index([Timestamp('2011-01-01 10:00'), Timestamp('2011-01-02 10:00', tz='US/Eastern')],
                    dtype='object', name='idx')
        tm.assert_index_equal(result, exp, exact=True)
        assert not isinstance(result, DatetimeIndex)
        result = Index([Timestamp('2011-01-01 10:00', tz='Asia/Tokyo'),
                        Timestamp('2011-01-02 10:00', tz='US/Eastern')], name='idx')
        exp = Index([Timestamp('2011-01-01 10:00', tz='Asia/Tokyo'),
                     Timestamp('2011-01-02 10:00', tz='US/Eastern')], dtype='object', name='idx')
        tm.assert_index_equal(result, exp, exact=True)
        assert not isinstance(result, DatetimeIndex)
        msg = 'Mixed timezones detected. Pass utc=True in to_datetime'
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(['2013-11-02 22:00-05:00', '2013-11-03 22:00-06:00'])
        result = Index([Timestamp('2011-01-01')], name='idx')
        exp = DatetimeIndex([Timestamp('2011-01-01')], name='idx')
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is None
        result = Index([Timestamp('2011-01-01 10:00', tz='Asia/Tokyo')], name='idx')
        exp = DatetimeIndex([Timestamp('2011-01-01 10:00')], tz='Asia/Tokyo', name='idx')
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is not None
        assert result.tz == exp.tz

    def test_construction_index_with_mixed_timezones_with_NaT(self) -> None:
        result = Index([pd.NaT, Timestamp('2011-01-01'), pd.NaT, Timestamp('2011-01-02')], name='idx')
        exp = DatetimeIndex([pd.NaT, Timestamp('2011-01-01'), pd.NaT, Timestamp('2011-01-02')], name='idx')
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is None
        result = Index([pd.NaT, Timestamp('2011-01-01 10:00', tz='Asia/Tokyo'),
                        pd.NaT, Timestamp('2011-01-02 10:00', tz='Asia/Tokyo')], name='idx')
        exp = DatetimeIndex([pd.NaT, Timestamp('2011-01-01 10:00'), pd.NaT, Timestamp('2011-01-02 10:00')],
                            tz='Asia/Tokyo', name='idx')
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is not None
        assert result.tz == exp.tz
        result = Index([Timestamp('2011-01-01 10:00', tz='US/Eastern'),
                        pd.NaT,
                        Timestamp('2011-08-01 10:00', tz='US/Eastern')], name='idx')
        exp = DatetimeIndex([Timestamp('2011-01-01 10:00'), pd.NaT, Timestamp('2011-08-01 10:00')],
                            tz='US/Eastern', name='idx')
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is not None
        assert result.tz == exp.tz
        result = Index([pd.NaT, Timestamp('2011-01-01 10:00'),
                        pd.NaT, Timestamp('2011-01-02 10:00', tz='US/Eastern')], name='idx')
        exp = Index([pd.NaT, Timestamp('2011-01-01 10:00'),
                     pd.NaT, Timestamp('2011-01-02 10:00', tz='US/Eastern')], dtype='object', name='idx')
        tm.assert_index_equal(result, exp, exact=True)
        assert not isinstance(result, DatetimeIndex)
        result = Index([pd.NaT, Timestamp('2011-01-01 10:00', tz='Asia/Tokyo'),
                        pd.NaT, Timestamp('2011-01-02 10:00', tz='US/Eastern')], name='idx')
        exp = Index([pd.NaT, Timestamp('2011-01-01 10:00', tz='Asia/Tokyo'),
                     pd.NaT, Timestamp('2011-01-02 10:00', tz='US/Eastern')], dtype='object', name='idx')
        tm.assert_index_equal(result, exp, exact=True)
        assert not isinstance(result, DatetimeIndex)
        result = Index([pd.NaT, pd.NaT], name='idx')
        exp = DatetimeIndex([pd.NaT, pd.NaT], name='idx')
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is None

    def test_construction_dti_with_mixed_timezones(self) -> None:
        result = DatetimeIndex([Timestamp('2011-01-01'), Timestamp('2011-01-02')], name='idx')
        exp = DatetimeIndex([Timestamp('2011-01-01'), Timestamp('2011-01-02')], name='idx')
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        result = DatetimeIndex([Timestamp('2011-01-01 10:00', tz='Asia/Tokyo'),
                                Timestamp('2011-01-02 10:00', tz='Asia/Tokyo')], name='idx')
        exp = DatetimeIndex([Timestamp('2011-01-01 10:00'), Timestamp('2011-01-02 10:00')],
                            tz='Asia/Tokyo', name='idx')
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        result = DatetimeIndex([Timestamp('2011-01-01 10:00', tz='US/Eastern'),
                                Timestamp('2011-08-01 10:00', tz='US/Eastern')], name='idx')
        exp = DatetimeIndex([Timestamp('2011-01-01 10:00'), Timestamp('2011-08-01 10:00')],
                            tz='US/Eastern', name='idx')
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        msg = 'cannot be converted to datetime64'
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex([Timestamp('2011-01-01 10:00', tz='Asia/Tokyo'),
                           Timestamp('2011-01-02 10:00', tz='US/Eastern')], name='idx')
        dti = DatetimeIndex([Timestamp('2011-01-01 10:00'),
                             Timestamp('2011-01-02 10:00', tz='US/Eastern')],
                            tz='Asia/Tokyo', name='idx')
        expected = DatetimeIndex([Timestamp('2011-01-01 10:00', tz='Asia/Tokyo'),
                                  Timestamp('2011-01-02 10:00', tz='US/Eastern').tz_convert('Asia/Tokyo')],
                                 tz='Asia/Tokyo', name='idx')
        tm.assert_index_equal(dti, expected)
        dti = DatetimeIndex([Timestamp('2011-01-01 10:00', tz='Asia/Tokyo'),
                             Timestamp('2011-01-02 10:00', tz='US/Eastern')],
                            tz='US/Eastern', name='idx')
        expected = DatetimeIndex([Timestamp('2011-01-01 10:00', tz='Asia/Tokyo').tz_convert('US/Eastern'),
                                  Timestamp('2011-01-02 10:00', tz='US/Eastern')],
                                 tz='US/Eastern', name='idx')
        tm.assert_index_equal(dti, expected)
        dti = DatetimeIndex([Timestamp('2011-01-01 10:00', tz='Asia/Tokyo'),
                             Timestamp('2011-01-02 10:00', tz='US/Eastern')],
                            dtype='M8[s, US/Eastern]', name='idx')
        tm.assert_index_equal(dti, expected)

    def test_construction_base_constructor(self) -> None:
        arr: List[Optional[Timestamp]] = [Timestamp('2011-01-01'), pd.NaT, Timestamp('2011-01-03')]
        tm.assert_index_equal(Index(arr), DatetimeIndex(arr))
        tm.assert_index_equal(Index(np.array(arr)), DatetimeIndex(np.array(arr)))
        arr = [np.nan, pd.NaT, Timestamp('2011-01-03')]
        tm.assert_index_equal(Index(arr), DatetimeIndex(arr))
        tm.assert_index_equal(Index(np.array(arr)), DatetimeIndex(np.array(arr)))

    def test_construction_outofbounds(self) -> None:
        dates: List[datetime] = [datetime(3000, 1, 1), datetime(4000, 1, 1),
                                 datetime(5000, 1, 1), datetime(6000, 1, 1)]
        exp = Index(dates, dtype='M8[us]')
        res = Index(dates)
        tm.assert_index_equal(res, exp)
        DatetimeIndex(dates)

    @pytest.mark.parametrize('data', [['1400-01-01'], [datetime(1400, 1, 1)]])
    def test_dti_date_out_of_range(self, data: List[Any]) -> None:
        DatetimeIndex(data)

    def test_construction_with_ndarray(self) -> None:
        dates: List[datetime] = [datetime(2013, 10, 7), datetime(2013, 10, 8), datetime(2013, 10, 9)]
        data = DatetimeIndex(dates, freq=offsets.BDay()).values
        result = DatetimeIndex(data, freq=offsets.BDay())
        expected = DatetimeIndex(['2013-10-07', '2013-10-08', '2013-10-09'], dtype='M8[us]', freq='B')
        tm.assert_index_equal(result, expected)

    def test_integer_values_and_tz_interpreted_as_utc(self) -> None:
        val: np.datetime64 = np.datetime64('2000-01-01 00:00:00', 'ns')
        values = np.array([val.view('i8')])
        result = DatetimeIndex(values).tz_localize('US/Central')
        expected = DatetimeIndex(['2000-01-01T00:00:00'], dtype='M8[ns, US/Central]')
        tm.assert_index_equal(result, expected)
        with tm.assert_produces_warning(None):
            result = DatetimeIndex(values, tz='UTC')
        expected = DatetimeIndex(['2000-01-01T00:00:00'], dtype='M8[ns, UTC]')
        tm.assert_index_equal(result, expected)

    def test_constructor_coverage(self) -> None:
        msg: str = 'DatetimeIndex\\(\\.\\.\\.\\) must be called with a collection'
        with pytest.raises(TypeError, match=msg):
            DatetimeIndex('1/1/2000')
        gen = (datetime(2000, 1, 1) + timedelta(i) for i in range(10))
        result = DatetimeIndex(gen)
        expected = DatetimeIndex([datetime(2000, 1, 1) + timedelta(i) for i in range(10)])
        tm.assert_index_equal(result, expected)
        strings = np.array(['2000-01-01', '2000-01-02', '2000-01-03'])
        result = DatetimeIndex(strings)
        expected = DatetimeIndex(strings.astype('O'))
        tm.assert_index_equal(result, expected)
        from_ints = DatetimeIndex(expected.as_unit('ns').asi8).as_unit('s')
        tm.assert_index_equal(from_ints, expected)
        strings = np.array(['2000-01-01', '2000-01-02', 'NaT'])
        result = DatetimeIndex(strings)
        expected = DatetimeIndex(strings.astype('O'))
        tm.assert_index_equal(result, expected)
        from_ints = DatetimeIndex(expected.as_unit('ns').asi8).as_unit('s')
        tm.assert_index_equal(from_ints, expected)
        msg = 'Inferred frequency None from passed values does not conform to passed frequency D'
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(['2000-01-01', '2000-01-02', '2000-01-04'], freq='D')

    @pytest.mark.parametrize('freq', ['YS', 'W-SUN'])
    def test_constructor_datetime64_tzformat(self, freq: str) -> None:
        idx = date_range('2013-01-01T00:00:00-05:00', '2016-01-01T23:59:59-05:00', freq=freq)
        expected = date_range('2013-01-01T00:00:00', '2016-01-01T23:59:59', freq=freq,
                              tz=timezone(timedelta(minutes=-300)))
        tm.assert_index_equal(idx, expected)
        expected_i8 = date_range('2013-01-01T00:00:00', '2016-01-01T23:59:59', freq=freq, tz='America/Lima')
        tm.assert_numpy_array_equal(idx.asi8, expected_i8.asi8)
        idx = date_range('2013-01-01T00:00:00+09:00', '2016-01-01T23:59:59+09:00', freq=freq)
        expected = date_range('2013-01-01T00:00:00', '2016-01-01T23:59:59', freq=freq,
                              tz=timezone(timedelta(minutes=540)))
        tm.assert_index_equal(idx, expected)
        expected_i8 = date_range('2013-01-01T00:00:00', '2016-01-01T23:59:59', freq=freq, tz='Asia/Tokyo')
        tm.assert_numpy_array_equal(idx.asi8, expected_i8.asi8)
        idx = date_range('2013/1/1 0:00:00-5:00', '2016/1/1 23:59:59-5:00', freq=freq)
        expected = date_range('2013-01-01T00:00:00', '2016-01-01T23:59:59', freq=freq,
                              tz=timezone(timedelta(minutes=-300)))
        tm.assert_index_equal(idx, expected)
        expected_i8 = date_range('2013-01-01T00:00:00', '2016-01-01T23:59:59', freq=freq, tz='America/Lima')
        tm.assert_numpy_array_equal(idx.asi8, expected_i8.asi8)
        idx = date_range('2013/1/1 0:00:00+9:00', '2016/1/1 23:59:59+09:00', freq=freq)
        expected = date_range('2013-01-01T00:00:00', '2016-01-01T23:59:59', freq=freq,
                              tz=timezone(timedelta(minutes=540)))
        tm.assert_index_equal(idx, expected)
        expected_i8 = date_range('2013-01-01T00:00:00', '2016-01-01T23:59:59', freq=freq, tz='Asia/Tokyo')
        tm.assert_numpy_array_equal(idx.asi8, expected_i8.asi8)

    def test_constructor_dtype(self) -> None:
        idx = DatetimeIndex(['2013-01-01', '2013-01-02'], dtype='datetime64[ns, US/Eastern]')
        expected = DatetimeIndex(['2013-01-01', '2013-01-02']).as_unit('ns').tz_localize('US/Eastern')
        tm.assert_index_equal(idx, expected)
        idx = DatetimeIndex(['2013-01-01', '2013-01-02'], tz='US/Eastern').as_unit('ns')
        tm.assert_index_equal(idx, expected)

    def test_constructor_dtype_tz_mismatch_raises(self) -> None:
        idx = DatetimeIndex(['2013-01-01', '2013-01-02'], dtype='datetime64[ns, US/Eastern]')
        msg: str = 'cannot supply both a tz and a timezone-naive dtype \\(i\\.e\\. datetime64\\[ns\\]\\)'
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(idx, dtype='datetime64[ns]')
        msg = 'data is already tz-aware US/Eastern, unable to set specified tz: CET'
        with pytest.raises(TypeError, match=msg):
            DatetimeIndex(idx, dtype='datetime64[ns, CET]')
        msg = 'cannot supply both a tz and a dtype with a tz'
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(idx, tz='CET', dtype='datetime64[ns, US/Eastern]')
        result = DatetimeIndex(idx, dtype='datetime64[ns, US/Eastern]')
        tm.assert_index_equal(idx, result)

    @pytest.mark.parametrize('dtype', [object, np.int32, np.int64])
    def test_constructor_invalid_dtype_raises(self, dtype: Any) -> None:
        msg: str = "Unexpected value for 'dtype'"
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex([1, 2], dtype=dtype)

    def test_000constructor_resolution(self) -> None:
        t1: Timestamp = Timestamp(1352934390 * 1000000000 + 1000000 + 1000 + 1)
        idx = DatetimeIndex([t1])
        assert idx.nanosecond[0] == t1.nanosecond

    def test_disallow_setting_tz(self) -> None:
        dti = DatetimeIndex(['2010'], tz='UTC')
        msg: str = 'Cannot directly set timezone'
        with pytest.raises(AttributeError, match=msg):
            dti.tz = zoneinfo.ZoneInfo('US/Pacific')

    @pytest.mark.parametrize('tz', [None, 'America/Los_Angeles', zoneinfo.ZoneInfo('America/Los_Angeles'),
                                      Timestamp('2000', tz='America/Los_Angeles').tz])
    def test_constructor_start_end_with_tz(self, tz: Optional[Union[str, zoneinfo.ZoneInfo]]) -> None:
        start: Timestamp = Timestamp('2013-01-01 06:00:00', tz='America/Los_Angeles')
        end: Timestamp = Timestamp('2013-01-02 06:00:00', tz='America/Los_Angeles')
        result = date_range(freq='D', start=start, end=end, tz=tz)
        expected = DatetimeIndex(['2013-01-01 06:00:00', '2013-01-02 06:00:00'],
                                 dtype='M8[ns, America/Los_Angeles]', freq='D')
        tm.assert_index_equal(result, expected)
        assert zoneinfo.ZoneInfo('America/Los_Angeles') is result.tz

    @pytest.mark.parametrize('tz', ['US/Pacific', 'US/Eastern', 'Asia/Tokyo'])
    def test_constructor_with_non_normalized_pytz(self, tz: str) -> None:
        pytz = pytest.importorskip('pytz')
        tz_in = pytz.timezone(tz)
        non_norm_tz = Timestamp('2010', tz=tz_in).tz
        result = DatetimeIndex(['2010'], tz=non_norm_tz)
        assert pytz.timezone(tz) is result.tz

    def test_constructor_timestamp_near_dst(self) -> None:
        ts: List[Timestamp] = [
            Timestamp('2016-10-30 03:00:00+0300', tz='Europe/Helsinki'),
            Timestamp('2016-10-30 03:00:00+0200', tz='Europe/Helsinki')
        ]
        result = DatetimeIndex(ts).as_unit('ns')
        expected = DatetimeIndex([ts[0].to_pydatetime(), ts[1].to_pydatetime()]).as_unit('ns')
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('klass', [Index, DatetimeIndex])
    @pytest.mark.parametrize('box', [np.array, partial(np.array, dtype=object), list])
    @pytest.mark.parametrize('tz, dtype', [('US/Pacific', 'datetime64[ns, US/Pacific]'), (None, 'datetime64[ns]')])
    def test_constructor_with_int_tz(self, klass: Any, box: Callable, tz: Optional[str], dtype: str) -> None:
        ts = Timestamp('2018-01-01', tz=tz).as_unit('ns')
        result = klass(box([ts._value]), dtype=dtype)
        expected = klass([ts])
        assert result == expected

    def test_construction_int_rountrip(self, tz_naive_fixture: Any) -> None:
        tz = tz_naive_fixture
        result = 1293858000000000000
        expected = DatetimeIndex([result], tz=tz).asi8[0]
        assert result == expected

    def test_construction_from_replaced_timestamps_with_dst(self) -> None:
        index = date_range(Timestamp(2000, 12, 31), Timestamp(2005, 12, 31), freq='YE-DEC', tz='Australia/Melbourne')
        result = DatetimeIndex([x.replace(month=6, day=1) for x in index])
        expected = DatetimeIndex(['2000-06-01 00:00:00', '2001-06-01 00:00:00', '2002-06-01 00:00:00',
                                  '2003-06-01 00:00:00', '2004-06-01 00:00:00', '2005-06-01 00:00:00'],
                                 tz='Australia/Melbourne').as_unit('ns')
        tm.assert_index_equal(result, expected)

    def test_construction_with_tz_and_tz_aware_dti(self) -> None:
        dti = date_range('2016-01-01', periods=3, tz='US/Central')
        msg: str = 'data is already tz-aware US/Central, unable to set specified tz'
        with pytest.raises(TypeError, match=msg):
            DatetimeIndex(dti, tz='Asia/Tokyo')

    def test_construction_with_nat_and_tzlocal(self) -> None:
        tz = dateutil.tz.tzlocal()
        result = DatetimeIndex(['2018', 'NaT'], tz=tz).as_unit('ns')
        expected = DatetimeIndex([Timestamp('2018', tz=tz), pd.NaT]).as_unit('ns')
        tm.assert_index_equal(result, expected)

    def test_constructor_with_ambiguous_keyword_arg(self) -> None:
        expected = DatetimeIndex(['2020-11-01 01:00:00', '2020-11-02 01:00:00'],
                                 dtype='datetime64[ns, America/New_York]', freq='D')
        timezone_str = 'America/New_York'
        start = Timestamp(year=2020, month=11, day=1, hour=1).tz_localize(timezone_str, ambiguous=False)
        result = date_range(start=start, periods=2, ambiguous=False)
        tm.assert_index_equal(result, expected)
        timezone_str = 'America/New_York'
        end = Timestamp(year=2020, month=11, day=2, hour=1).tz_localize(timezone_str, ambiguous=False)
        result = date_range(end=end, periods=2, ambiguous=False)
        tm.assert_index_equal(result, expected)

    def test_constructor_with_nonexistent_keyword_arg(self, warsaw: Any) -> None:
        timezone_val = warsaw
        start = Timestamp('2015-03-29 02:30:00').tz_localize(timezone_val, nonexistent='shift_forward')
        result = date_range(start=start, periods=2, freq='h')
        expected = DatetimeIndex([Timestamp('2015-03-29 03:00:00+02:00', tz=timezone_val),
                                  Timestamp('2015-03-29 04:00:00+02:00', tz=timezone_val)]).as_unit('ns')
        tm.assert_index_equal(result, expected)
        end = start
        result = date_range(end=end, periods=2, freq='h')
        expected = DatetimeIndex([Timestamp('2015-03-29 01:00:00+01:00', tz=timezone_val),
                                  Timestamp('2015-03-29 03:00:00+02:00', tz=timezone_val)]).as_unit('ns')
        tm.assert_index_equal(result, expected)

    def test_constructor_no_precision_raises(self) -> None:
        msg: str = 'with no precision is not allowed'
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(['2000'], dtype='datetime64')
        msg = "The 'datetime64' dtype has no unit. Please pass in"
        with pytest.raises(ValueError, match=msg):
            Index(['2000'], dtype='datetime64')

    def test_constructor_wrong_precision_raises(self) -> None:
        dti = DatetimeIndex(['2000'], dtype='datetime64[us]')
        assert dti.dtype == 'M8[us]'
        assert dti[0] == Timestamp(2000, 1, 1)

    def test_index_constructor_with_numpy_object_array_and_timestamp_tz_with_nan(self) -> None:
        result = Index(np.array([Timestamp('2019', tz='UTC'), np.nan], dtype=object))
        expected = DatetimeIndex([Timestamp('2019', tz='UTC'), pd.NaT])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('tz', [zoneinfo.ZoneInfo('US/Eastern'), gettz('US/Eastern')])
    def test_dti_from_tzaware_datetime(self, tz: Any) -> None:
        d: List[datetime] = [datetime(2012, 8, 19, tzinfo=tz)]
        index = DatetimeIndex(d)
        assert timezones.tz_compare(index.tz, tz)

    @pytest.mark.parametrize('tzstr', ['US/Eastern', 'dateutil/US/Eastern'])
    def test_dti_tz_constructors(self, tzstr: str) -> None:
        """Test different DatetimeIndex constructions with timezone
        Follow-up of GH#4229
        """
        arr: List[str] = ['11/10/2005 08:00:00', '11/10/2005 09:00:00']
        idx1 = to_datetime(arr).tz_localize(tzstr)
        idx2 = date_range(start='2005-11-10 08:00:00', freq='h', periods=2, tz=tzstr, unit='s')
        idx2 = idx2._with_freq(None)
        idx3 = DatetimeIndex(arr, tz=tzstr).as_unit('s')
        idx4 = DatetimeIndex(np.array(arr), tz=tzstr).as_unit('s')
        tm.assert_index_equal(idx1, idx2)
        tm.assert_index_equal(idx1, idx3)
        tm.assert_index_equal(idx1, idx4)

    def test_dti_construction_idempotent(self, unit: str) -> None:
        rng = date_range('03/12/2012 00:00', periods=10, freq='W-FRI', tz='US/Eastern', unit=unit)
        rng2 = DatetimeIndex(data=rng, tz='US/Eastern')
        tm.assert_index_equal(rng, rng2)

    @pytest.mark.parametrize('prefix', ['', 'dateutil/'])
    def test_dti_constructor_static_tzinfo(self, prefix: str) -> None:
        index = DatetimeIndex([datetime(2012, 1, 1)], tz=prefix + 'EST')
        index.hour
        index[0]

    @pytest.mark.parametrize('tzstr', ['US/Eastern', 'dateutil/US/Eastern'])
    def test_dti_convert_datetime_list(self, tzstr: str) -> None:
        dr = date_range('2012-06-02', periods=10, tz=tzstr, name='foo')
        dr2 = DatetimeIndex(list(dr), name='foo', freq='D')
        tm.assert_index_equal(dr, dr2)

    @pytest.mark.parametrize('tz', ['pytz/US/Eastern', gettz('US/Eastern')])
    @pytest.mark.parametrize('use_str', [True, False])
    @pytest.mark.parametrize('box_cls', [Timestamp, DatetimeIndex])
    def test_dti_ambiguous_matches_timestamp(self, tz: Union[str, Any], use_str: bool, box_cls: Any, request: Any) -> None:
        if isinstance(tz, str) and tz.startswith('pytz/'):
            pytz = pytest.importorskip('pytz')
            tz = pytz.timezone(tz.removeprefix('pytz/'))
        dtstr: str = '2013-11-03 01:59:59.999999'
        item: Union[str, datetime, Timestamp] = dtstr
        if not use_str:
            item = Timestamp(dtstr).to_pydatetime()
        if box_cls is not Timestamp:
            item = [item]
        if not use_str and isinstance(tz, dateutil.tz.tzfile):
            mark = pytest.mark.xfail(reason='We implicitly get fold=0.')
            request.applymarker(mark)
        with pytest.raises(ValueError, match=dtstr):
            box_cls(item, tz=tz)

    @pytest.mark.parametrize('tz', [None, 'UTC', 'US/Pacific'])
    def test_dti_constructor_with_non_nano_dtype(self, tz: Optional[str]) -> None:
        ts = Timestamp('2999-01-01')
        dtype: str = 'M8[us]'
        if tz is not None:
            dtype = f'M8[us, {tz}]'
        vals: List[Union[Timestamp, str, int]] = [ts, '2999-01-02 03:04:05.678910', 2500]
        result = DatetimeIndex(vals, dtype=dtype)
        pointwise = [vals[0].tz_localize(tz) if isinstance(vals[0], Timestamp) else Timestamp(vals[0], tz=tz),
                     Timestamp(vals[1], tz=tz),
                     to_datetime(vals[2], unit='us', utc=True).tz_convert(tz)]
        exp_vals = [x.as_unit('us').asm8 for x in pointwise]  # type: ignore
        exp_arr = np.array(exp_vals, dtype='M8[us]')
        expected = DatetimeIndex(exp_arr, dtype='M8[us]')
        if tz is not None:
            expected = expected.tz_localize('UTC').tz_convert(tz)
        tm.assert_index_equal(result, expected)
        result2 = DatetimeIndex(np.array(vals, dtype=object), dtype=dtype)
        tm.assert_index_equal(result2, expected)

    def test_dti_constructor_with_non_nano_now_today(self, request: Any) -> None:
        now = Timestamp.now()
        today = Timestamp.today()
        result = DatetimeIndex(['now', 'today'], dtype='M8[s]')
        assert result.dtype == 'M8[s]'
        diff0 = result[0] - now.as_unit('s')
        diff1 = result[1] - today.as_unit('s')
        assert diff1 >= pd.Timedelta(0), f'The difference is {diff0}'
        assert diff0 >= pd.Timedelta(0), f'The difference is {diff0}'
        request.applymarker(pytest.mark.xfail(reason='result may not exactly match [now, today]', strict=False))
        tolerance = pd.Timedelta(seconds=1)
        assert diff0 < tolerance, f'The difference is {diff0}'
        assert diff1 < tolerance, f'The difference is {diff0}'

    def test_dti_constructor_object_float_matches_float_dtype(self) -> None:
        arr = np.array([0, np.nan], dtype=np.float64)
        arr2 = arr.astype(object)
        dti1 = DatetimeIndex(arr, tz='CET')
        dti2 = DatetimeIndex(arr2, tz='CET')
        tm.assert_index_equal(dti1, dti2)

    @pytest.mark.parametrize('dtype', ['M8[us]', 'M8[us, US/Pacific]'])
    def test_dti_constructor_with_dtype_object_int_matches_int_dtype(self, dtype: str) -> None:
        vals1 = np.arange(5, dtype='i8') * 1000
        vals1[0] = pd.NaT.value
        vals2 = vals1.astype(np.float64)
        vals2[0] = np.nan
        vals3 = vals1.astype(object)
        vals3[0] = pd.NaT
        vals4 = vals2.astype(object)
        res1 = DatetimeIndex(vals1, dtype=dtype)
        res2 = DatetimeIndex(vals2, dtype=dtype)
        res3 = DatetimeIndex(vals3, dtype=dtype)
        res4 = DatetimeIndex(vals4, dtype=dtype)
        expected = DatetimeIndex(vals1.view('M8[us]'))
        if res1.tz is not None:
            expected = expected.tz_localize('UTC').tz_convert(res1.tz)
        tm.assert_index_equal(res1, expected)
        tm.assert_index_equal(res2, expected)
        tm.assert_index_equal(res3, expected)
        tm.assert_index_equal(res4, expected)


class TestTimeSeries:

    def test_dti_constructor_preserve_dti_freq(self) -> None:
        rng = date_range('1/1/2000', '1/2/2000', freq='5min')
        rng2 = DatetimeIndex(rng)
        assert rng.freq == rng2.freq

    def test_explicit_none_freq(self) -> None:
        rng = date_range('1/1/2000', '1/2/2000', freq='5min')
        result = DatetimeIndex(rng, freq=None)
        assert result.freq is None
        result = DatetimeIndex(rng._data, freq=None)
        assert result.freq is None

    def test_dti_constructor_small_int(self, any_int_numpy_dtype: Any) -> None:
        exp = DatetimeIndex(['1970-01-01 00:00:00.00000000',
                             '1970-01-01 00:00:00.00000001',
                             '1970-01-01 00:00:00.00000002'])
        arr = np.array([0, 10, 20], dtype=any_int_numpy_dtype)
        tm.assert_index_equal(DatetimeIndex(arr), exp)

    def test_ctor_str_intraday(self) -> None:
        rng = DatetimeIndex(['1-1-2000 00:00:01'])
        assert rng[0].second == 1

    def test_index_cast_datetime64_other_units(self) -> None:
        arr = np.arange(0, 100, 10, dtype=np.int64).view('M8[D]')
        idx = Index(arr)
        assert (idx.values == astype_overflowsafe(arr, dtype=np.dtype('M8[ns]'))).all()

    def test_constructor_int64_nocopy(self) -> None:
        arr = np.arange(1000, dtype=np.int64)
        index = DatetimeIndex(arr)
        arr[50:100] = -1
        assert (index.asi8[50:100] == -1).all()
        arr = np.arange(1000, dtype=np.int64)
        index = DatetimeIndex(arr, copy=True)
        arr[50:100] = -1
        assert (index.asi8[50:100] != -1).all()

    @pytest.mark.parametrize('freq', ['ME', 'QE', 'YE', 'D', 'B', 'bh', 'min', 's', 'ms', 'us', 'h', 'ns', 'C'])
    def test_from_freq_recreate_from_data(self, freq: str) -> None:
        org = date_range(start='2001/02/01 09:00', freq=freq, periods=1)
        idx = DatetimeIndex(org, freq=freq)
        tm.assert_index_equal(idx, org)
        org = date_range(start='2001/02/01 09:00', freq=freq, tz='US/Pacific', periods=1)
        idx = DatetimeIndex(org, freq=freq, tz='US/Pacific')
        tm.assert_index_equal(idx, org)

    def test_datetimeindex_constructor_misc(self) -> None:
        arr = ['1/1/2005', '1/2/2005', 'Jn 3, 2005', '2005-01-04']
        msg: str = r"(\(')?Unknown datetime string format(:', 'Jn 3, 2005'\))?"
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(arr)
        arr = ['1/1/2005', '1/2/2005', '1/3/2005', '2005-01-04']
        idx1 = DatetimeIndex(arr)
        arr = [datetime(2005, 1, 1), '1/2/2005', '1/3/2005', '2005-01-04']
        idx2 = DatetimeIndex(arr)
        arr = [Timestamp(datetime(2005, 1, 1)), '1/2/2005', '1/3/2005', '2005-01-04']
        idx3 = DatetimeIndex(arr)
        arr = np.array(['1/1/2005', '1/2/2005', '1/3/2005', '2005-01-04'], dtype='O')
        idx4 = DatetimeIndex(arr)
        idx5 = DatetimeIndex(['12/05/2007', '25/01/2008'], dayfirst=True)
        idx6 = DatetimeIndex(['2007/05/12', '2008/01/25'], dayfirst=False, yearfirst=True)
        tm.assert_index_equal(idx5, idx6)
        for other in [idx2, idx3, idx4]:
            assert (idx1.values == other.values).all()

    def test_dti_constructor_object_dtype_dayfirst_yearfirst_with_tz(self) -> None:
        val: str = '5/10/16'
        dfirst: Timestamp = Timestamp(2016, 10, 5, tz='US/Pacific')
        yfirst: Timestamp = Timestamp(2005, 10, 16, tz='US/Pacific')
        result1 = DatetimeIndex([val], tz='US/Pacific', dayfirst=True)
        expected1 = DatetimeIndex([dfirst]).as_unit('s')
        tm.assert_index_equal(result1, expected1)
        result2 = DatetimeIndex([val], tz='US/Pacific', yearfirst=True)
        expected2 = DatetimeIndex([yfirst]).as_unit('s')
        tm.assert_index_equal(result2, expected2)
    
# End of annotated code.
