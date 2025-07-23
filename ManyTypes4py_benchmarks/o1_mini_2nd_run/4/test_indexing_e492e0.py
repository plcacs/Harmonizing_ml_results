from datetime import date, datetime, time, timedelta
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import DatetimeIndex, Index, Timestamp, bdate_range, date_range, notna
import pandas._testing as tm
from pandas.tseries.frequencies import to_offset

START, END = (datetime(2009, 1, 1), datetime(2010, 1, 1))

class TestGetItem:

    def test_getitem_slice_keeps_name(self) -> None:
        st: Timestamp = Timestamp('2013-07-01 00:00:00', tz='America/Los_Angeles')
        et: Timestamp = Timestamp('2013-07-02 00:00:00', tz='America/Los_Angeles')
        dr: DatetimeIndex = date_range(st, et, freq='h', name='timebucket')
        assert dr[1:].name == dr.name

    @pytest.mark.parametrize('tz', [None, 'Asia/Tokyo'])
    def test_getitem(self, tz: str | None) -> None:
        idx: DatetimeIndex = date_range('2011-01-01', '2011-01-31', freq='D', tz=tz, name='idx')
        result: Timestamp = idx[0]
        assert result == Timestamp('2011-01-01', tz=idx.tz)
        result = idx[0:5]
        expected: DatetimeIndex = date_range('2011-01-01', '2011-01-05', freq='D', tz=idx.tz, name='idx')
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
        expected = DatetimeIndex(['2011-01-05', '2011-01-04', '2011-01-03', '2011-01-02', '2011-01-01'],
                                 dtype=idx.dtype, freq='-1D', name='idx')
        tm.assert_index_equal(result, expected)
        assert result.freq == expected.freq

    @pytest.mark.parametrize('freq', ['B', 'C'])
    def test_dti_business_getitem(self, freq: str) -> None:
        rng: DatetimeIndex = bdate_range(START, END, freq=freq)
        smaller: DatetimeIndex = rng[:5]
        exp: DatetimeIndex = DatetimeIndex(rng.view(np.ndarray)[:5], freq=freq)
        tm.assert_index_equal(smaller, exp)
        assert smaller.freq == exp.freq
        assert smaller.freq == rng.freq
        sliced: DatetimeIndex = rng[::5]
        assert sliced.freq == to_offset(freq) * 5
        fancy_indexed: DatetimeIndex = rng[[4, 3, 2, 1, 0]]
        assert len(fancy_indexed) == 5
        assert isinstance(fancy_indexed, DatetimeIndex)
        assert fancy_indexed.freq is None
        assert rng[4] == rng[np_long(4)]

    @pytest.mark.parametrize('freq', ['B', 'C'])
    def test_dti_business_getitem_matplotlib_hackaround(self, freq: str) -> None:
        rng: DatetimeIndex = bdate_range(START, END, freq=freq)
        with pytest.raises(ValueError, match='Multi-dimensional indexing'):
            rng[:, None]

    def test_getitem_int_list(self) -> None:
        dti: DatetimeIndex = date_range(start='1/1/2005', end='12/1/2005', freq='ME')
        dti2: DatetimeIndex = dti[[1, 3, 5]]
        v1: Timestamp = dti2[0]
        v2: Timestamp = dti2[1]
        v3: Timestamp = dti2[2]
        assert v1 == Timestamp('2/28/2005')
        assert v2 == Timestamp('4/30/2005')
        assert v3 == Timestamp('6/30/2005')
        assert dti2.freq is None

class TestWhere:

    def test_where_doesnt_retain_freq(self) -> None:
        dti: DatetimeIndex = date_range('20130101', periods=3, freq='D', name='idx')
        cond: list[bool] = [True, True, False]
        expected: DatetimeIndex = DatetimeIndex([dti[0], dti[1], dti[0]], freq=None, name='idx')
        result: DatetimeIndex = dti.where(cond, dti[::-1])
        tm.assert_index_equal(result, expected)

    def test_where_other(self) -> None:
        i: DatetimeIndex = date_range('20130101', periods=3, tz='US/Eastern')
        for arr in [np.nan, pd.NaT]:
            result: DatetimeIndex = i.where(notna(i), other=arr)
            expected: DatetimeIndex = i
            tm.assert_index_equal(result, expected)
        i2: Index = i.copy()
        i2 = Index([pd.NaT, pd.NaT] + i[2:].tolist())
        result = i.where(notna(i2), i2)
        tm.assert_index_equal(result, i2)
        i2 = i.copy()
        i2 = Index([pd.NaT, pd.NaT] + i[2:].tolist())
        result = i.where(notna(i2), i2._values)
        tm.assert_index_equal(result, i2)

    def test_where_invalid_dtypes(self) -> None:
        dti: DatetimeIndex = date_range('20130101', periods=3, tz='US/Eastern')
        tail: list[Timestamp] = dti[2:].tolist()
        i2: Index = Index([pd.NaT, pd.NaT] + tail)
        mask: np.ndarray = notna(i2)
        result: Index = dti.where(mask, i2.values)
        expected: Index = Index([pd.NaT.asm8, pd.NaT.asm8] + tail, dtype=object)
        tm.assert_index_equal(result, expected)
        naive: DatetimeIndex = dti.tz_localize(None)
        result = naive.where(mask, i2)
        expected = Index([i2[0], i2[1]] + naive[2:].tolist(), dtype=object)
        tm.assert_index_equal(result, expected)
        pi: pd.PeriodIndex = i2.tz_localize(None).to_period('D')
        result = dti.where(mask, pi)
        expected = Index([pi[0], pi[1]] + tail, dtype=object)
        tm.assert_index_equal(result, expected)
        tda: np.ndarray = i2.asi8.view('timedelta64[ns]')
        result = dti.where(mask, tda)
        expected = Index([tda[0], tda[1]] + tail, dtype=object)
        assert isinstance(expected[0], np.timedelta64)
        tm.assert_index_equal(result, expected)
        result = dti.where(mask, i2.asi8)
        expected = Index([pd.NaT._value, pd.NaT._value] + tail, dtype=object)
        assert isinstance(expected[0], int)
        tm.assert_index_equal(result, expected)
        td: pd.Timedelta = pd.Timedelta(days=4)
        result = dti.where(mask, td)
        expected = Index([td, td] + tail, dtype=object)
        assert expected[0] is td
        tm.assert_index_equal(result, expected)

    def test_where_mismatched_nat(self, tz_aware_fixture: str | None) -> None:
        tz: str | None = tz_aware_fixture
        dti: DatetimeIndex = date_range('2013-01-01', periods=3, tz=tz)
        cond: np.ndarray = np.array([True, False, True])
        tdnat: np.timedelta64 = np.timedelta64('NaT', 'ns')
        expected: Index = Index([dti[0], tdnat, dti[2]], dtype=object)
        assert expected[1] is tdnat
        result: Index = dti.where(cond, tdnat)
        tm.assert_index_equal(result, expected)

    def test_where_tz(self) -> None:
        i: DatetimeIndex = date_range('20130101', periods=3, tz='US/Eastern')
        result: DatetimeIndex = i.where(notna(i))
        expected: DatetimeIndex = i
        tm.assert_index_equal(result, expected)
        i2: Index = i.copy()
        i2 = Index([pd.NaT, pd.NaT] + i[2:].tolist())
        result = i.where(notna(i2))
        expected = i2
        tm.assert_index_equal(result, expected)

class TestTake:

    @pytest.mark.parametrize('tzstr', ['US/Eastern', 'dateutil/US/Eastern'])
    def test_dti_take_dont_lose_meta(self, tzstr: str) -> None:
        rng: DatetimeIndex = date_range('1/1/2000', periods=20, tz=tzstr)
        result: DatetimeIndex = rng.take(range(5))
        assert result.tz == rng.tz
        assert result.freq == rng.freq

    def test_take_nan_first_datetime(self) -> None:
        index: DatetimeIndex = DatetimeIndex([pd.NaT, Timestamp('20130101'), Timestamp('20130102')])
        result: DatetimeIndex = index.take([-1, 0, 1])
        expected: DatetimeIndex = DatetimeIndex([index[-1], index[0], index[1]])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('tz', [None, 'Asia/Tokyo'])
    def test_take(self, tz: str | None) -> None:
        idx: DatetimeIndex = date_range('2011-01-01', '2011-01-31', freq='D', name='idx', tz=tz)
        result: Timestamp = idx.take([0])
        assert result == Timestamp('2011-01-01', tz=idx.tz)
        result = idx.take([0, 1, 2])
        expected: DatetimeIndex = date_range('2011-01-01', '2011-01-03', freq='D', tz=idx.tz, name='idx')
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
        expected = DatetimeIndex(['2011-01-04', '2011-01-03', '2011-01-06'],
                                 dtype=idx.dtype, freq=None, name='idx')
        tm.assert_index_equal(result, expected)
        assert result.freq is None
        result = idx.take([-3, 2, 5])
        expected = DatetimeIndex(['2011-01-29', '2011-01-03', '2011-01-06'],
                                 dtype=idx.dtype, freq=None, name='idx')
        tm.assert_index_equal(result, expected)
        assert result.freq is None

    def test_take_invalid_kwargs(self) -> None:
        idx: DatetimeIndex = date_range('2011-01-01', '2011-01-31', freq='D', name='idx')
        indices: list[int] = [1, 6, 5, 9, 10, 13, 15, 3]
        msg: str = "take\\(\\) got an unexpected keyword argument 'foo'"
        with pytest.raises(TypeError, match=msg):
            idx.take(indices, foo=2)
        msg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            idx.take(indices, out=indices)
        msg = "the 'mode' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            idx.take(indices, mode='clip')

    @pytest.mark.parametrize('tz', [None, 'US/Eastern', 'Asia/Tokyo'])
    def test_take2(self, tz: str | None) -> None:
        dates: list[datetime] = [
            datetime(2010, 1, 1, 14),
            datetime(2010, 1, 1, 15),
            datetime(2010, 1, 1, 17),
            datetime(2010, 1, 1, 21)
        ]
        idx: DatetimeIndex = date_range(start='2010-01-01 09:00', end='2010-02-01 09:00', freq='h', tz=tz, name='idx')
        expected: DatetimeIndex = DatetimeIndex(dates, freq=None, name='idx', dtype=idx.dtype)
        taken1: DatetimeIndex = idx.take([5, 6, 8, 12])
        taken2: DatetimeIndex = idx[[5, 6, 8, 12]]
        for taken in [taken1, taken2]:
            tm.assert_index_equal(taken, expected)
            assert isinstance(taken, DatetimeIndex)
            assert taken.freq is None
            assert taken.tz == expected.tz
            assert taken.name == expected.name

    def test_take_fill_value(self) -> None:
        idx: DatetimeIndex = DatetimeIndex(['2011-01-01', '2011-02-01', '2011-03-01'], name='xxx')
        result: DatetimeIndex = idx.take(np.array([1, 0, -1]))
        expected: DatetimeIndex = DatetimeIndex(['2011-02-01', '2011-01-01', '2011-03-01'], name='xxx')
        tm.assert_index_equal(result, expected)
        result = idx.take(np.array([1, 0, -1]), fill_value=True)
        expected = DatetimeIndex(['2011-02-01', '2011-01-01', 'NaT'], name='xxx')
        tm.assert_index_equal(result, expected)
        result = idx.take(np.array([1, 0, -1]), allow_fill=False, fill_value=True)
        expected = DatetimeIndex(['2011-02-01', '2011-01-01', '2011-03-01'], name='xxx')
        tm.assert_index_equal(result, expected)
        msg: str = 'When allow_fill=True and fill_value is not None, all indices must be >= -1'
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -2]), fill_value=True)
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -5]), fill_value=True)
        msg = 'out of bounds'
        with pytest.raises(IndexError, match=msg):
            idx.take(np.array([1, -5]))

    def test_take_fill_value_with_timezone(self) -> None:
        idx: DatetimeIndex = DatetimeIndex(['2011-01-01', '2011-02-01', '2011-03-01'],
                                          name='xxx', tz='US/Eastern')
        result: DatetimeIndex = idx.take(np.array([1, 0, -1]))
        expected: DatetimeIndex = DatetimeIndex(['2011-02-01', '2011-01-01', '2011-03-01'],
                                              name='xxx', tz='US/Eastern')
        tm.assert_index_equal(result, expected)
        result = idx.take(np.array([1, 0, -1]), fill_value=True)
        expected = DatetimeIndex(['2011-02-01', '2011-01-01', 'NaT'], name='xxx', tz='US/Eastern')
        tm.assert_index_equal(result, expected)
        result = idx.take(np.array([1, 0, -1]), allow_fill=False, fill_value=True)
        expected = DatetimeIndex(['2011-02-01', '2011-01-01', '2011-03-01'], name='xxx', tz='US/Eastern')
        tm.assert_index_equal(result, expected)
        msg: str = 'When allow_fill=True and fill_value is not None, all indices must be >= -1'
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -2]), fill_value=True)
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -5]), fill_value=True)
        msg = 'out of bounds'
        with pytest.raises(IndexError, match=msg):
            idx.take(np.array([1, -5]))

class TestGetLoc:

    def test_get_loc_key_unit_mismatch(self) -> None:
        idx: DatetimeIndex = date_range('2000-01-01', periods=3)
        key: pd.Timestamp = idx[1].as_unit('ms')
        loc: int | slice = idx.get_loc(key)
        assert loc == 1
        assert key in idx

    def test_get_loc_key_unit_mismatch_not_castable(self) -> None:
        dta: np.ndarray = date_range('2000-01-01', periods=3)._data.astype('M8[s]')
        dti: DatetimeIndex = DatetimeIndex(dta)
        key: pd.Timestamp = dta[0].as_unit('ns') + pd.Timedelta(1)
        with pytest.raises(KeyError, match="Timestamp\\('2000-01-01 00:00:00.000000001'\\)"):
            dti.get_loc(key)
        assert key not in dti

    def test_get_loc_time_obj(self) -> None:
        idx: DatetimeIndex = date_range('2000-01-01', periods=24, freq='h')
        result: np.ndarray = idx.get_loc(time(12))
        expected: np.ndarray = np.array([12])
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)
        result = idx.get_loc(time(12, 30))
        expected = np.array([], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)

    @pytest.mark.parametrize('offset', [-10, 10])
    def test_get_loc_time_obj2(self, monkeypatch: pytest.MonkeyPatch, offset: int) -> None:
        size_cutoff: int = 50
        n: int = size_cutoff + offset
        key: time = time(15, 11, 30)
        start: int = key.hour * 3600 + key.minute * 60 + key.second
        step: int = 24 * 3600
        with monkeypatch.context():
            monkeypatch.setattr(libindex, '_SIZE_CUTOFF', size_cutoff)
            idx: DatetimeIndex = date_range('2014-11-26', periods=n, freq='s')
            ts: pd.Series = pd.Series(np.random.default_rng(2).standard_normal(n), index=idx)
            locs: np.ndarray = np.arange(start, n, step, dtype=np.intp)
            result = ts.index.get_loc(key)
            tm.assert_numpy_array_equal(result, locs)
            tm.assert_series_equal(ts[key], ts.iloc[locs])
            left: pd.Series = ts.copy()
            right: pd.Series = ts.copy()
            left[key] *= -10
            right.iloc[locs] *= -10
            tm.assert_series_equal(left, right)

    def test_get_loc_time_nat(self) -> None:
        tic: time = time(minute=12, second=43, microsecond=145224)
        dti: DatetimeIndex = DatetimeIndex([pd.NaT])
        loc: np.ndarray = dti.get_loc(tic)
        expected: np.ndarray = np.array([], dtype=np.intp)
        tm.assert_numpy_array_equal(loc, expected)

    def test_get_loc_nat(self) -> None:
        index: DatetimeIndex = DatetimeIndex(['1/3/2000', 'NaT'])
        assert index.get_loc(pd.NaT) == 1
        assert index.get_loc(None) == 1
        assert index.get_loc(np.nan) == 1
        assert index.get_loc(pd.NA) == 1
        assert index.get_loc(np.datetime64('NaT')) == 1
        with pytest.raises(KeyError, match='NaT'):
            index.get_loc(np.timedelta64('NaT'))

    @pytest.mark.parametrize('key', [pd.Timedelta(0), pd.Timedelta(1), timedelta(0)])
    def test_get_loc_timedelta_invalid_key(self, key: pd.Timedelta | timedelta) -> None:
        dti: DatetimeIndex = date_range('1970-01-01', periods=10)
        msg: str = 'Cannot index DatetimeIndex with [Tt]imedelta'
        with pytest.raises(TypeError, match=msg):
            dti.get_loc(key)

    def test_get_loc_reasonable_key_error(self) -> None:
        index: DatetimeIndex = DatetimeIndex(['1/3/2000'])
        with pytest.raises(KeyError, match='2000'):
            index.get_loc('1/1/2000')

    def test_get_loc_year_str(self) -> None:
        rng: DatetimeIndex = date_range('1/1/2000', '1/1/2010')
        result: slice = rng.get_loc('2009')
        expected: slice = slice(3288, 3653)
        assert result == expected

class TestContains:

    def test_dti_contains_with_duplicates(self) -> None:
        d: datetime = datetime(2011, 12, 5, 20, 30)
        ix: DatetimeIndex = DatetimeIndex([d, d])
        assert d in ix

    @pytest.mark.parametrize('vals', [
        [0, 1, 0],
        [0, 0, -1],
        [0, -1, -1],
        ['2015', '2015', '2016'],
        ['2015', '2015', '2014']
    ])
    def test_contains_nonunique(self, vals: list[Any]) -> None:
        idx: DatetimeIndex = DatetimeIndex(vals)
        assert idx[0] in idx

class TestGetIndexer:

    def test_get_indexer_date_objs(self) -> None:
        rng: DatetimeIndex = date_range('1/1/2000', periods=20)
        target: DatetimeIndex = rng.map(lambda x: x.date())
        result: np.ndarray = rng.get_indexer(target)
        expected: np.ndarray = rng.get_indexer(rng)
        tm.assert_numpy_array_equal(result, expected)

    def test_get_indexer(self) -> None:
        idx: DatetimeIndex = date_range('2000-01-01', periods=3)
        exp: np.ndarray = np.array([0, 1, 2], dtype=np.intp)
        result: np.ndarray = idx.get_indexer(idx)
        tm.assert_numpy_array_equal(result, exp)
        target: np.ndarray = idx[0] + pd.to_timedelta(['-1 hour', '12 hours', '1 day 1 hour'])
        result = idx.get_indexer(target, 'pad')
        tm.assert_numpy_array_equal(result, np.array([-1, 0, 1], dtype=np.intp))
        result = idx.get_indexer(target, 'backfill')
        tm.assert_numpy_array_equal(result, np.array([0, 1, 2], dtype=np.intp))
        result = idx.get_indexer(target, 'nearest')
        tm.assert_numpy_array_equal(result, np.array([0, 1, 1], dtype=np.intp))
        result = idx.get_indexer(target, 'nearest', tolerance=pd.Timedelta('1 hour'))
        tm.assert_numpy_array_equal(result, np.array([0, -1, 1], dtype=np.intp))
        tol_raw: list[np.timedelta64] = [
            pd.Timedelta('1 hour').to_timedelta64(),
            pd.Timedelta('1 hour').to_timedelta64(),
            pd.Timedelta('1 hour').to_timedelta64()
        ]
        result = idx.get_indexer(target, 'nearest', tolerance=[np.timedelta64(x) for x in tol_raw])
        tm.assert_numpy_array_equal(result, np.array([0, -1, 1], dtype=np.intp))
        tol_bad: list[Any] = [
            pd.Timedelta('2 hour').to_timedelta64(),
            pd.Timedelta('1 hour').to_timedelta64(),
            'foo'
        ]
        msg: str = "Could not convert 'foo' to NumPy timedelta"
        with pytest.raises(ValueError, match=msg):
            idx.get_indexer(target, 'nearest', tolerance=tol_bad)
        with pytest.raises(ValueError, match='abbreviation w/o a number'):
            idx.get_indexer(idx[[0]], method='nearest', tolerance='foo')

    @pytest.mark.parametrize('target', [
        [date(2020, 1, 1), Timestamp('2020-01-02')],
        [Timestamp('2020-01-01'), date(2020, 1, 2)]
    ])
    def test_get_indexer_mixed_dtypes(self, target: list[date | Timestamp]) -> None:
        values: DatetimeIndex = DatetimeIndex([Timestamp('2020-01-01'), Timestamp('2020-01-02')])
        result: np.ndarray = values.get_indexer(target)
        expected: np.ndarray = np.array([0, 1], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize('target, positions', [
        ([date(9999, 1, 1), Timestamp('2020-01-01')], [-1, 0]),
        ([Timestamp('2020-01-01'), date(9999, 1, 1)], [0, -1]),
        ([date(9999, 1, 1), date(9999, 1, 1)], [-1, -1])
    ])
    def test_get_indexer_out_of_bounds_date(self, target: list[date | Timestamp], positions: list[int]) -> None:
        values: DatetimeIndex = DatetimeIndex([Timestamp('2020-01-01'), Timestamp('2020-01-02')])
        result: np.ndarray = values.get_indexer(target)
        expected: np.ndarray = np.array(positions, dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

    def test_get_indexer_pad_requires_monotonicity(self) -> None:
        rng: DatetimeIndex = date_range('1/1/2000', '3/1/2000', freq='B')
        rng2: DatetimeIndex = rng[[1, 0, 2]]
        msg: str = 'index must be monotonic increasing or decreasing'
        with pytest.raises(ValueError, match=msg):
            rng2.get_indexer(rng, method='pad')

class TestMaybeCastSliceBound:

    def test_maybe_cast_slice_bounds_empty(self) -> None:
        empty_idx: DatetimeIndex = date_range(freq='1h', periods=0, end='2015')
        right: Timestamp = empty_idx._maybe_cast_slice_bound('2015-01-02', 'right')
        exp: Timestamp = Timestamp('2015-01-02 23:59:59.999999999')
        assert right == exp
        left: Timestamp = empty_idx._maybe_cast_slice_bound('2015-01-02', 'left')
        exp = Timestamp('2015-01-02 00:00:00')
        assert left == exp

    def test_maybe_cast_slice_duplicate_monotonic(self) -> None:
        idx: DatetimeIndex = DatetimeIndex(['2017', '2017'])
        result: Timestamp = idx._maybe_cast_slice_bound('2017-01-01', 'left')
        expected: Timestamp = Timestamp('2017-01-01')
        assert result == expected

class TestGetSliceBounds:

    @pytest.mark.parametrize('box', [date, datetime, Timestamp])
    @pytest.mark.parametrize('side, expected', [('left', 4), ('right', 5)])
    def test_get_slice_bounds_datetime_within(self, box: type, side: str, expected: int, tz_aware_fixture: str | None) -> None:
        tz: str | None = tz_aware_fixture
        index: DatetimeIndex = bdate_range('2000-01-03', '2000-02-11').tz_localize(tz)
        key = box(year=2000, month=1, day=7)
        if tz is not None:
            with pytest.raises(TypeError, match='Cannot compare tz-naive'):
                index.get_slice_bound(key, side=side)
        else:
            result: int = index.get_slice_bound(key, side=side)
            assert result == expected

    @pytest.mark.parametrize('box', [datetime, Timestamp])
    @pytest.mark.parametrize('side', ['left', 'right'])
    @pytest.mark.parametrize('year, expected', [(1999, 0), (2020, 30)])
    def test_get_slice_bounds_datetime_outside(self, box: type, side: str, year: int, expected: int, tz_aware_fixture: str | None) -> None:
        tz: str | None = tz_aware_fixture
        index: DatetimeIndex = bdate_range('2000-01-03', '2000-02-11').tz_localize(tz)
        key = box(year=year, month=1, day=7)
        if tz is not None:
            with pytest.raises(TypeError, match='Cannot compare tz-naive'):
                index.get_slice_bound(key, side=side)
        else:
            result: int = index.get_slice_bound(key, side=side)
            assert result == expected

    @pytest.mark.parametrize('box', [datetime, Timestamp])
    def test_slice_datetime_locs(self, box: type, tz_aware_fixture: str | None) -> None:
        tz: str | None = tz_aware_fixture
        index: DatetimeIndex = DatetimeIndex(['2010-01-01', '2010-01-03']).tz_localize(tz)
        key1 = box(2010, 1, 1)
        key2 = box(2010, 1, 2)
        if tz is not None:
            with pytest.raises(TypeError, match='Cannot compare tz-naive'):
                index.slice_locs(key1, key2)
        else:
            result: tuple[int, int] = index.slice_locs(key1, key2)
            expected: tuple[int, int] = (0, 1)
            assert result == expected

class TestIndexerBetweenTime:

    def test_indexer_between_time(self) -> None:
        rng: DatetimeIndex = date_range('1/1/2000', '1/5/2000', freq='5min')
        msg: str = 'Cannot convert arg \\[datetime\\.datetime\\(2010, 1, 2, 1, 0\\)\\] to a time'
        with pytest.raises(ValueError, match=msg):
            rng.indexer_between_time(datetime(2010, 1, 2, 1), datetime(2010, 1, 2, 5))

    @pytest.mark.parametrize('unit', ['us', 'ms', 's'])
    def test_indexer_between_time_non_nano(self, unit: str) -> None:
        rng: DatetimeIndex = date_range('1/1/2000', '1/5/2000', freq='5min')
        arr_nano: np.ndarray = rng._data._ndarray
        arr: np.ndarray = arr_nano.astype(f'M8[{unit}]')
        dta: pd.DatetimeArray = type(rng._data)._simple_new(arr, dtype=arr.dtype)
        dti: DatetimeIndex = DatetimeIndex(dta)
        assert dti.dtype == arr.dtype
        tic: time = time(1, 25)
        toc: time = time(2, 29)
        result: np.ndarray = dti.indexer_between_time(tic, toc)
        expected: np.ndarray = rng.indexer_between_time(tic, toc)
        tm.assert_numpy_array_equal(result, expected)
        tic = time(1, 25, 0, 45678)
        toc = time(2, 29, 0, 1234)
        result = dti.indexer_between_time(tic, toc)
        expected = rng.indexer_between_time(tic, toc)
        tm.assert_numpy_array_equal(result, expected)
