from datetime import date, datetime, time, timedelta
from typing import Any, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import DatetimeIndex, Index, Timestamp, bdate_range, date_range, notna
import pandas._testing as tm
from pandas.tseries.frequencies import to_offset
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.base import Index
from pandas._libs.tslibs.timestamps import Timestamp
from pandas._libs.tslibs.offsets import BaseOffset

START: datetime = datetime(2009, 1, 1)
END: datetime = datetime(2010, 1, 1)

class TestGetItem:

    def test_getitem_slice_keeps_name(self) -> None:
        st = Timestamp('2013-07-01 00:00:00', tz='America/Los_Angeles')
        et = Timestamp('2013-07-02 00:00:00', tz='America/Los_Angeles')
        dr = date_range(st, et, freq='h', name='timebucket')
        assert dr[1:].name == dr.name

    @pytest.mark.parametrize('tz', [None, 'Asia/Tokyo'])
    def test_getitem(self, tz: Optional[str]) -> None:
        idx = date_range('2011-01-01', '2011-01-31', freq='D', tz=tz, name='idx')
        result = idx[0]
        assert result == Timestamp('2011-01-01', tz=idx.tz)
        result = idx[0:5]
        expected = date_range('2011-01-01', '2011-01-05', freq='D', tz=idx.tz, name='idx')
        tm.assert_index_equal(result, expected)
        assert result.freq == expected.freq
        result = idx[0:10:2]
        expected = date_range('2011-01-01', '2011-01-09', freq='2D', tz=idx.tz, name='idx')
        tm.assert_index_equal(result, expected)
        assert result.freq == expected.freq
        result = idx[-20:-5:3]
        expected = date_range('2011-01-12', '2011-01-24', freq='3D', tz=idx.tz, name='idx')
        tm.assert_index_equal(result, expected)
        assert result.freq == expected.freq
        result = idx[4::-1]
        expected = DatetimeIndex(['2011-01-05', '2011-01-04', '2011-01-03', '2011-01-02', '2011-01-01'], dtype=idx.dtype, freq='-1D', name='idx')
        tm.assert_index_equal(result, expected)
        assert result.freq == expected.freq

    @pytest.mark.parametrize('freq', ['B', 'C'])
    def test_dti_business_getitem(self, freq: str) -> None:
        rng = bdate_range(START, END, freq=freq)
        smaller = rng[:5]
        exp = DatetimeIndex(rng.view(np.ndarray)[:5], freq=freq)
        tm.assert_index_equal(smaller, exp)
        assert smaller.freq == exp.freq
        assert smaller.freq == rng.freq
        sliced = rng[::5]
        assert sliced.freq == to_offset(freq) * 5
        fancy_indexed = rng[[4, 3, 2, 1, 0]]
        assert len(fancy_indexed) == 5
        assert isinstance(fancy_indexed, DatetimeIndex)
        assert fancy_indexed.freq is None
        assert rng[4] == rng[np_long(4)]

    @pytest.mark.parametrize('freq', ['B', 'C'])
    def test_dti_business_getitem_matplotlib_hackaround(self, freq: str) -> None:
        rng = bdate_range(START, END, freq=freq)
        with pytest.raises(ValueError, match='Multi-dimensional indexing'):
            rng[:, None]

    def test_getitem_int_list(self) -> None:
        dti = date_range(start='1/1/2005', end='12/1/2005', freq='ME')
        dti2 = dti[[1, 3, 5]]
        v1 = dti2[0]
        v2 = dti2[1]
        v3 = dti2[2]
        assert v1 == Timestamp('2/28/2005')
        assert v2 == Timestamp('4/30/2005')
        assert v3 == Timestamp('6/30/2005')
        assert dti2.freq is None

class TestWhere:

    def test_where_doesnt_retain_freq(self) -> None:
        dti = date_range('20130101', periods=3, freq='D', name='idx')
        cond = [True, True, False]
        expected = DatetimeIndex([dti[0], dti[1], dti[0]], freq=None, name='idx')
        result = dti.where(cond, dti[::-1])
        tm.assert_index_equal(result, expected)

    def test_where_other(self) -> None:
        i = date_range('20130101', periods=3, tz='US/Eastern')
        for arr in [np.nan, pd.NaT]:
            result = i.where(notna(i), other=arr)
            expected = i
            tm.assert_index_equal(result, expected)
        i2 = i.copy()
        i2 = Index([pd.NaT, pd.NaT] + i[2:].tolist())
        result = i.where(notna(i2), i2)
        tm.assert_index_equal(result, i2)
        i2 = i.copy()
        i2 = Index([pd.NaT, pd.NaT] + i[2:].tolist())
        result = i.where(notna(i2), i2._values)
        tm.assert_index_equal(result, i2)

    def test_where_invalid_dtypes(self) -> None:
        dti = date_range('20130101', periods=3, tz='US/Eastern')
        tail = dti[2:].tolist()
        i2 = Index([pd.NaT, pd.NaT] + tail)
        mask = notna(i2)
        result = dti.where(mask, i2.values)
        expected = Index([pd.NaT.asm8, pd.NaT.asm8] + tail, dtype=object)
        tm.assert_index_equal(result, expected)
        naive = dti.tz_localize(None)
        result = naive.where(mask, i2)
        expected = Index([i2[0], i2[1]] + naive[2:].tolist(), dtype=object)
        tm.assert_index_equal(result, expected)
        pi = i2.tz_localize(None).to_period('D')
        result = dti.where(mask, pi)
        expected = Index([pi[0], pi[1]] + tail, dtype=object)
        tm.assert_index_equal(result, expected)
        tda = i2.asi8.view('timedelta64[ns]')
        result = dti.where(mask, tda)
        expected = Index([tda[0], tda[1]] + tail, dtype=object)
        assert isinstance(expected[0], np.timedelta64)
        tm.assert_index_equal(result, expected)
        result = dti.where(mask, i2.asi8)
        expected = Index([pd.NaT._value, pd.NaT._value] + tail, dtype=object)
        assert isinstance(expected[0], int)
        tm.assert_index_equal(result, expected)
        td = pd.Timedelta(days=4)
        result = dti.where(mask, td)
        expected = Index([td, td] + tail, dtype=object)
        assert expected[0] is td
        tm.assert_index_equal(result, expected)

    def test_where_mismatched_nat(self, tz_aware_fixture: Any) -> None:
        tz = tz_aware_fixture
        dti = date_range('2013-01-01', periods=3, tz=tz)
        cond = np.array([True, False, True])
        tdnat = np.timedelta64('NaT', 'ns')
        expected = Index([dti[0], tdnat, dti[2]], dtype=object)
        assert expected[1] is tdnat
        result = dti.where(cond, tdnat)
        tm.assert_index_equal(result, expected)

    def test_where_tz(self) -> None:
        i = date_range('20130101', periods=3, tz='US/Eastern')
        result = i.where(notna(i))
        expected = i
        tm.assert_index_equal(result, expected)
        i2 = i.copy()
        i2 = Index([pd.NaT, pd.NaT] + i[2:].tolist())
        result = i.where(notna(i2))
        expected = i2
        tm.assert_index_equal(result, expected)

class TestTake:

    @pytest.mark.parametrize('tzstr', ['US/Eastern', 'dateutil/US/Eastern'])
    def test_dti_take_dont_lose_meta(self, tzstr: str) -> None:
        rng = date_range('1/1/2000', periods=20, tz=tzstr)
        result = rng.take(range(5))
        assert result.tz == rng.tz
        assert result.freq == rng.freq

    def test_take_nan_first_datetime(self) -> None:
        index = DatetimeIndex([pd.NaT, Timestamp('20130101'), Timestamp('20130102')])
        result = index.take([-1, 0, 1])
        expected = DatetimeIndex([index[-1], index[0], index[1]])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('tz', [None, 'Asia/Tokyo'])
    def test_take(self, tz: Optional[str]) -> None:
        idx = date_range('2011-01-01', '2011-01-31', freq='D', name='idx', tz=tz)
        result = idx.take([0])
        assert result == Timestamp('2011-01-01', tz=idx.tz)
        result = idx.take([0, 1, 2])
        expected = date_range('2011-01-01', '2011-01-03', freq='D', tz=idx.tz, name='idx')
        tm.assert_index_equal(result, expected)
        assert result.freq == expected.freq
        result = idx.take([0, 2, 4])
        expected = date_range('2011-01-01', '2011-01-05', freq='2D', tz=idx.tz, name='idx')
        tm.assert_index_equal(result, expected)
        assert result.freq == expected.freq
        result = idx.take([7, 4, 1])
        expected = date_range('2011-01-08', '2011-01-02', freq='-3D', tz=idx.tz, name='idx')
        tm.assert_index_equal(result, expected)
        assert result.freq == expected.freq
        result = idx.take([3, 2, 5])
        expected = DatetimeIndex(['2011-01-04', '2011-01-03', '2011-01-06'], dtype=idx.dtype, freq=None, name='idx')
        tm.assert_index_equal(result, expected)
        assert result.freq is None
        result = idx.take([-3, 2, 5])
        expected = DatetimeIndex(['2011-01-29', '2011-01-03', '2011-01-06'], dtype=idx.dtype, freq=None, name='idx')
        tm.assert_index_equal(result, expected)
        assert result.freq is None

    def test_take_invalid_kwargs(self) -> None:
        idx = date_range('2011-01-01', '2011-01-31', freq='D', name='idx')
        indices = [1, 6, 5, 9, 10, 13, 15, 3]
        msg = "take\\(\\) got an unexpected keyword argument 'foo'"
        with pytest.raises(TypeError, match=msg):
            idx.take(indices, foo=2)
        msg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            idx.take(indices, out=indices)
        msg = "the 'mode' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            idx.take(indices, mode='clip')

    @pytest.mark.parametrize('tz', [None, 'US/Eastern', 'Asia/Tokyo'])
    def test_take2(self, tz: Optional[str]) -> None:
        dates = [datetime(2010, 1, 1, 14), datetime(2010, 1, 1, 15), datetime(2010, 1, 1, 17), datetime(2010, 1, 1, 21)]
        idx = date_range(start='2010-01-01 09:00', end='2010-02-01 09:00', freq='h', tz=tz, name='idx')
        expected = DatetimeIndex(dates, freq=None, name='idx', dtype=idx.dtype)
        taken1 = idx.take([5, 6, 8, 12])
        taken2 = idx[[5, 6, 8, 12]]
        for taken in [taken1, taken2]:
            tm.assert_index_equal(taken, expected)
            assert isinstance(taken, DatetimeIndex)
            assert taken.freq is None
            assert taken.tz == expected.tz
            assert taken.name == expected.name

    def test_take_fill_value(self) -> None:
        idx = DatetimeIndex(['2011-01-01', '2011-02-01', '2011-03-01'], name='xxx')
        result = idx.take(np.array([1, 0, -1]))
        expected = DatetimeIndex(['2011-02-01', '2011-01-01', '2011-03-01'], name='xxx')
        tm.assert_index_equal(result, expected)
        result = idx.take(np.array([1, 0, -1]), fill_value=True)
        expected = DatetimeIndex(['2011-02-01', '2011-01-01', 'NaT'], name='xxx')
        tm.assert_index_equal(result, expected)
        result = idx.take(np.array([1, 0, -1]), allow_fill=False, fill_value=True)
        expected = DatetimeIndex(['2011-02-01', '2011-01-01', '2011-03-01'], name='xxx')
        tm.assert_index_equal(result, expected)
        msg = 'When allow_fill=True and fill_value is not None, all indices must be >= -1'
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -2]), fill_value=True)
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -5]), fill_value=True)
        msg = 'out of bounds'
        with pytest.raises(IndexError, match=msg):
            idx.take(np.array([1, -5]))

    def test_take_fill_value_with_timezone(self) -> None:
        idx = DatetimeIndex(['2011-01-01', '2011-02-01', '2011-03-01'], name='xxx', tz='US/Eastern')
        result = idx.take(np.array([1, 0, -1]))
        expected = DatetimeIndex(['2011-02-01', '2011-01-01', '2011-03-01'], name='xxx', tz='US/Eastern')
        tm.assert_index_equal(result, expected)
        result = idx.take(np.array([1, 0, -1]), fill_value=True)
        expected = DatetimeIndex(['2011-02-01', '2011-01-01', 'NaT'], name='xxx', tz='US/Eastern')
        tm.assert_index_equal(result, expected)
        result = idx.take(np.array([1, 0, -1]), allow_fill=False, fill_value=True)
        expected = DatetimeIndex(['2011-02-01', '2011-01-01', '2011-03-01'], name='xxx', tz='US/Eastern')
        tm.assert_index_equal(result, expected)
        msg = 'When allow_fill=True and fill_value is not None, all indices must be >= -1'
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -2]), fill_value=True)
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -5]), fill_value=True)
        msg = 'out of bounds'
        with pytest.raises(IndexError, match=msg):
            idx.take(np.array([1, -5]))

class TestGetLoc:

    def test_get_loc_key_unit_mismatch(self) -> None:
        idx = date_range('2000-01-01', periods=3)
        key = idx[1].as_unit('ms')
        loc = idx.get_loc(key)
        assert loc == 1
        assert key in idx

    def test_get_loc_key_unit_mismatch_not_castable(self) -> None:
        dta = date_range('2000-01-01', periods=3)._data.astype('M8[s]')
        dti = DatetimeIndex(dta)
        key =