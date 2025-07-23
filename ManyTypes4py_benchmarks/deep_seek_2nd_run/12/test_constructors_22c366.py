from __future__ import annotations
from datetime import datetime, timedelta, timezone
from functools import partial
from operator import attrgetter
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
from typing import Any, List, Optional, Union, Sequence, TypeVar
import numpy.typing as npt

T = TypeVar('T')

class TestDatetimeIndex:

    def test_from_dt64_unsupported_unit(self) -> None:
        val = np.datetime64(1, 'D')
        result = DatetimeIndex([val], tz='US/Pacific')
        expected = DatetimeIndex([val.astype('M8[s]')], tz='US/Pacific')
        tm.assert_index_equal(result, expected)

    def test_explicit_tz_none(self) -> None:
        dti = date_range('2016-01-01', periods=10, tz='UTC')
        msg = "Passed data is timezone-aware, incompatible with 'tz=None'"
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(dti, tz=None)
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(np.array(dti), tz=None)
        msg = 'Cannot pass both a timezone-aware dtype and tz=None'
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex([], dtype='M8[ns, UTC]', tz=None)

    def test_freq_validation_with_nat(self) -> None:
        msg = 'Inferred frequency None from passed values does not conform to passed frequency D'
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
        dti = DatetimeIndex([pd.NaT, '2015-01-01', '1999-04-06 15:14:13', '2015-01-01'], tz='US/Eastern')
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
        data = np.array([0], dtype='m8[ns]')
        msg = 'timedelta64\\[ns\\] cannot be converted to datetime64'
        with pytest.raises(TypeError, match=msg):
            DatetimeIndex(data)
        with pytest.raises(TypeError, match=msg):
            to_datetime(data)
        with pytest.raises(TypeError, match=msg):
            DatetimeIndex(pd.TimedeltaIndex(data))
        with pytest.raises(TypeError, match=msg):
            to_datetime(pd.TimedeltaIndex(data))

    def test_constructor_from_sparse_array(self) -> None:
        values = [Timestamp('2012-05-01T01:00:00.000000'), Timestamp('2016-05-01T01:00:00.000000')]
        arr = pd.arrays.SparseArray(values)
        result = Index(arr)
        assert type(result) is Index
        assert result.dtype == arr.dtype

    def test_construction_caching(self) -> None:
        df = pd.DataFrame({'dt': date_range('20130101', periods=3), 'dttz': date_range('20130101', periods=3, tz=zoneinfo.ZoneInfo('US/Eastern')), 'dt_with_null': [Timestamp('20130101'), pd.NaT, Timestamp('20130103')], 'dtns': date_range('20130101', periods=3, freq='ns')})
        assert df.dttz.dtype.tz.key == 'US/Eastern'

    @pytest.mark.parametrize('kwargs', [{'tz': 'dtype.tz'}, {'dtype': 'dtype'}, {'dtype': 'dtype', 'tz': 'dtype.tz'}])
    def test_construction_with_alt(self, kwargs: dict[str, str], tz_aware_fixture: Any) -> None:
        tz = tz_aware_fixture
        i = date_range('20130101', periods=5, freq='h', tz=tz)
        kwargs = {key: attrgetter(val)(i) for key, val in kwargs.items()}
        result = DatetimeIndex(i, **kwargs)
        tm.assert_index_equal(i, result)

    @pytest.mark.parametrize('kwargs', [{'tz': 'dtype.tz'}, {'dtype': 'dtype'}, {'dtype': 'dtype', 'tz': 'dtype.tz'}])
    def test_construction_with_alt_tz_localize(self, kwargs: dict[str, str], tz_aware_fixture: Any) -> None:
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
        result = Index([Timestamp('2011-01-01 10:00', tz='Asia/Tokyo'), Timestamp('2011-01-02 10:00', tz='Asia/Tokyo')], name='idx')
        exp = DatetimeIndex([Timestamp('2011-01-01 10:00'), Timestamp('2011-01-02 10:00')], tz='Asia/Tokyo', name='idx')
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is not None
        assert result.tz == exp.tz
        result = Index([Timestamp('2011-01-01 10:00', tz='US/Eastern'), Timestamp('2011-08-01 10:00', tz='US/Eastern')], name='idx')
        exp = DatetimeIndex([Timestamp('2011-01-01 10:00'), Timestamp('2011-08-01 10:00')], tz='US/Eastern', name='idx')
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is not None
        assert result.tz == exp.tz
        result = Index([Timestamp('2011-01-01 10:00'), Timestamp('2011-01-02 10:00', tz='US/Eastern')], name='idx')
        exp = Index([Timestamp('2011-01-01 10:00'), Timestamp('2011-01-02 10:00', tz='US/Eastern')], dtype='object', name='idx')
        tm.assert_index_equal(result, exp, exact=True)
        assert not isinstance(result, DatetimeIndex)
        result = Index([Timestamp('2011-01-01 10:00', tz='Asia/Tokyo'), Timestamp('2011-01-02 10:00', tz='US/Eastern')], name='idx')
        exp = Index([Timestamp('2011-01-01 10:00', tz='Asia/Tokyo'), Timestamp('2011-01-02 10:00', tz='US/Eastern')], dtype='object', name='idx')
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
        result = Index([pd.NaT, Timestamp('2011-01-01 10:00', tz='Asia/Tokyo'), pd.NaT, Timestamp('2011-01-02 10:00', tz='Asia/Tokyo')], name='idx')
        exp = DatetimeIndex([pd.NaT, Timestamp('2011-01-01 10:00'), pd.NaT, Timestamp('2011-01-02 10:00')], tz='Asia/Tokyo', name='idx')
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is not None
        assert result.tz == exp.tz
        result = Index([Timestamp('2011-01-01 10:00', tz='US/Eastern'), pd.NaT, Timestamp('2011-08-01 10:00', tz='US/Eastern')], name='idx')
        exp = DatetimeIndex([Timestamp('2011-01-01 10:00'), pd.NaT, Timestamp('2011-08-01 10:00')], tz='US/Eastern', name='idx')
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is not None
        assert result.tz == exp.tz
        result = Index([pd.NaT, Timestamp('2011-01-01 10:00'), pd.NaT, Timestamp('2011-01-02 10:00', tz='US/Eastern')], name='idx')
        exp = Index([pd.NaT, Timestamp('2011-01-01 10:00'), pd.NaT, Timestamp('2011-01-02 10:00', tz='US/Eastern')], dtype='object', name='idx')
        tm.assert_index_equal(result, exp, exact=True)
        assert not isinstance(result, DatetimeIndex)
        result = Index([pd.NaT, Timestamp('2011-01-01 10:00', tz='Asia/Tokyo'), pd.NaT, Timestamp('2011-01-02 10:00', tz='US/Eastern')], name='idx')
        exp = Index([pd.NaT, Timestamp('2011-01-01 10:00', tz='Asia/Tokyo'), pd.NaT, Timestamp('2011-01-02 10:00', tz='US/Eastern')], dtype='object', name='idx')
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
        result = DatetimeIndex([Timestamp('2011-01-01 10:00', tz='Asia/Tokyo'), Timestamp('2011-01-02 10:00', tz='Asia/Tokyo')], name='idx')
        exp = DatetimeIndex([Timestamp('2011-01-01 10:00'), Timestamp('2011-01-02 10:00')], tz='Asia/Tokyo', name='idx')
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        result = DatetimeIndex([Timestamp('2011-01-01 10:00', tz='US/Eastern'), Timestamp('2011-08-01 10:00', tz='US/Eastern')], name='idx')
        exp = DatetimeIndex([Timestamp('2011-01-01 10:00'), Timestamp('2011-08-01 10:00')], tz='US/Eastern', name='idx')
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        msg = 'cannot be converted to datetime64'
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex([Timestamp('2011-01-01 10:00', tz='Asia/Tokyo'), Timestamp('2011-01-02 10:00', tz='US/Eastern')], name='idx')
        dti = DatetimeIndex([Timestamp('2011-01-01 10:00'), Timestamp('2011-01-02 10:00', tz='US/Eastern')], tz='Asia/Tokyo', name='idx')
        expected = DatetimeIndex([Timestamp('2011-01-01 10:00', tz='Asia/Tokyo'), Timestamp('2011-01-02 10:00', tz='US/Eastern').tz_convert('Asia/Tokyo')], tz='Asia/Tokyo', name='idx')
        tm.assert_index_equal(dti, expected)
        dti = DatetimeIndex([Timestamp('2011-01-01 10:00', tz='Asia/Tokyo'), Timestamp('2011-01-02 10:00', tz='US/Eastern')], tz='US/Eastern', name='idx')
        expected = DatetimeIndex([Timestamp('2011-01-01 10:00', tz='Asia/Tokyo').tz_convert('US/Eastern'), Timestamp('2011-01-02 