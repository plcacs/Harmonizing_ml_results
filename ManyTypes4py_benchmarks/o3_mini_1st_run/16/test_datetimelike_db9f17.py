from __future__ import annotations
import re
import warnings
import numpy as np
import pytest
from pandas._libs import NaT, OutOfBoundsDatetime, Timestamp
from pandas._libs.tslibs import to_offset
from pandas.compat.numpy import np_version_gt2
from pandas.core.dtypes.dtypes import PeriodDtype
import pandas as pd
from pandas import DatetimeIndex, Period, PeriodIndex, TimedeltaIndex
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray, NumpyExtensionArray, PeriodArray, TimedeltaArray
from typing import Any, Optional, Union, Tuple, Callable, List

@pytest.fixture(params=['D', 'B', 'W', 'ME', 'QE', 'YE'])
def freqstr(request: pytest.FixtureRequest) -> str:
    """Fixture returning parametrized frequency in string format."""
    return request.param

@pytest.fixture
def period_index(freqstr: str) -> pd.PeriodIndex:
    """
    A fixture to provide PeriodIndex objects with different frequencies.

    Most PeriodArray behavior is already tested in PeriodIndex tests,
    so here we just test that the PeriodArray behavior matches
    the PeriodIndex behavior.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Period with BDay freq', category=FutureWarning)
        freqstr_conv: str = PeriodDtype(to_offset(freqstr))._freqstr
        pi: pd.PeriodIndex = pd.period_range(start=Timestamp('2000-01-01'), periods=100, freq=freqstr_conv)
    return pi

@pytest.fixture
def datetime_index(freqstr: str) -> DatetimeIndex:
    """
    A fixture to provide DatetimeIndex objects with different frequencies.

    Most DatetimeArray behavior is already tested in DatetimeIndex tests,
    so here we just test that the DatetimeArray behavior matches
    the DatetimeIndex behavior.
    """
    dti: DatetimeIndex = pd.date_range(start=Timestamp('2000-01-01'), periods=100, freq=freqstr)
    return dti

@pytest.fixture
def timedelta_index() -> TimedeltaIndex:
    """
    A fixture to provide TimedeltaIndex objects with different frequencies.
     Most TimedeltaArray behavior is already tested in TimedeltaIndex tests,
    so here we just test that the TimedeltaArray behavior matches
    the TimedeltaIndex behavior.
    """
    return TimedeltaIndex(['1 Day', '3 Hours', 'NaT'])

class SharedTests:

    @pytest.fixture
    def arr1d(self) -> NumpyExtensionArray:
        """Fixture returning DatetimeArray with daily frequency."""
        data: np.ndarray = np.arange(10, dtype='i8') * 24 * 3600 * 10 ** 9
        if self.array_cls is PeriodArray:
            arr: NumpyExtensionArray = self.array_cls(data, freq='D')
        else:
            arr = self.index_cls(data, freq='D')._data
        return arr

    def test_compare_len1_raises(self, arr1d: NumpyExtensionArray) -> None:
        arr: NumpyExtensionArray = arr1d
        idx = self.index_cls(arr)  # type: ignore
        with pytest.raises(ValueError, match='Lengths must match'):
            _ = arr == arr[:1]
        with pytest.raises(ValueError, match='Lengths must match'):
            _ = idx <= idx[[0]]

    @pytest.mark.parametrize('result', [pd.date_range('2020', periods=3),
                                         pd.date_range('2020', periods=3, tz='UTC'),
                                         pd.timedelta_range('0 days', periods=3),
                                         pd.period_range('2020Q1', periods=3, freq='Q')])
    def test_compare_with_Categorical(self, result: pd.Index) -> None:
        expected = pd.Categorical(result)
        assert all(result == expected)
        assert not any(result != expected)

    @pytest.mark.parametrize('reverse', [True, False])
    @pytest.mark.parametrize('as_index', [True, False])
    def test_compare_categorical_dtype(self, arr1d: NumpyExtensionArray, as_index: bool, reverse: bool, ordered: bool) -> None:
        other = pd.Categorical(arr1d, ordered=ordered)
        if as_index:
            other = pd.CategoricalIndex(other)
        left, right = (arr1d, other)
        if reverse:
            left, right = (right, left)
        ones: np.ndarray = np.ones(arr1d.shape, dtype=bool)
        zeros: np.ndarray = ~ones
        result = left == right
        tm.assert_numpy_array_equal(result, ones)
        result = left != right
        tm.assert_numpy_array_equal(result, zeros)
        if not reverse and (not as_index):
            result = left < right
            tm.assert_numpy_array_equal(result, zeros)
            result = left <= right
            tm.assert_numpy_array_equal(result, ones)
            result = left > right
            tm.assert_numpy_array_equal(result, zeros)
            result = left >= right
            tm.assert_numpy_array_equal(result, ones)

    def test_take(self) -> None:
        data: np.ndarray = np.arange(100, dtype='i8') * 24 * 3600 * 10 ** 9
        np.random.default_rng(2).shuffle(data)
        if self.array_cls is PeriodArray:
            arr: NumpyExtensionArray = self.array_cls(data, dtype='period[D]')
        else:
            arr = self.index_cls(data, freq='D')._data  # type: ignore
        idx = self.index_cls._simple_new(arr)  # type: ignore
        takers: List[int] = [1, 4, 94]
        result = arr.take(takers)
        expected = idx.take(takers)
        tm.assert_index_equal(self.index_cls(result), expected)  # type: ignore
        takers = np.array([1, 4, 94])
        result = arr.take(takers)
        expected = idx.take(takers)
        tm.assert_index_equal(self.index_cls(result), expected)  # type: ignore

    @pytest.mark.parametrize('fill_value', [2, 2.0, Timestamp(2021, 1, 1, 12).time])
    def test_take_fill_raises(self, fill_value: Any, arr1d: NumpyExtensionArray) -> None:
        msg: str = f"value should be a '{arr1d._scalar_type.__name__}' or 'NaT'. Got"
        with pytest.raises(TypeError, match=msg):
            _ = arr1d.take([0, 1], allow_fill=True, fill_value=fill_value)

    def test_take_fill(self, arr1d: NumpyExtensionArray) -> None:
        arr: NumpyExtensionArray = arr1d
        result = arr.take([-1, 1], allow_fill=True, fill_value=None)
        assert result[0] is NaT
        result = arr.take([-1, 1], allow_fill=True, fill_value=np.nan)
        assert result[0] is NaT
        result = arr.take([-1, 1], allow_fill=True, fill_value=NaT)
        assert result[0] is NaT

    @pytest.mark.filterwarnings('ignore:Period with BDay freq is deprecated:FutureWarning')
    def test_take_fill_str(self, arr1d: NumpyExtensionArray) -> None:
        result = arr1d.take([-1, 1], allow_fill=True, fill_value=str(arr1d[-1]))
        expected = arr1d[[-1, 1]]
        tm.assert_equal(result, expected)
        msg: str = f"value should be a '{arr1d._scalar_type.__name__}' or 'NaT'. Got"
        with pytest.raises(TypeError, match=msg):
            _ = arr1d.take([-1, 1], allow_fill=True, fill_value='foo')

    def test_concat_same_type(self, arr1d: NumpyExtensionArray) -> None:
        arr: NumpyExtensionArray = arr1d
        idx = self.index_cls(arr)  # type: ignore
        idx = idx.insert(0, NaT)
        arr = arr1d
        result = arr._concat_same_type([arr[:-1], arr[1:], arr])
        arr2 = arr.astype(object)
        expected = self.index_cls(np.concatenate([arr2[:-1], arr2[1:], arr2]))  # type: ignore
        tm.assert_index_equal(self.index_cls(result), expected)  # type: ignore

    def test_unbox_scalar(self, arr1d: NumpyExtensionArray) -> None:
        result = arr1d._unbox_scalar(arr1d[0])
        expected_type = arr1d._ndarray.dtype.type  # type: ignore
        assert isinstance(result, expected_type)
        result = arr1d._unbox_scalar(NaT)
        assert isinstance(result, expected_type)
        msg: str = f"'value' should be a {self.scalar_type.__name__}."
        with pytest.raises(ValueError, match=msg):
            _ = arr1d._unbox_scalar('foo')

    def test_check_compatible_with(self, arr1d: NumpyExtensionArray) -> None:
        arr1d._check_compatible_with(arr1d[0])
        arr1d._check_compatible_with(arr1d[:1])
        arr1d._check_compatible_with(NaT)

    def test_scalar_from_string(self, arr1d: NumpyExtensionArray) -> None:
        result = arr1d._scalar_from_string(str(arr1d[0]))
        assert result == arr1d[0]

    def test_reduce_invalid(self, arr1d: NumpyExtensionArray) -> None:
        msg: str = "does not support operation 'not a method'"
        with pytest.raises(TypeError, match=msg):
            _ = arr1d._reduce('not a method')

    @pytest.mark.parametrize('method', ['pad', 'backfill'])
    def test_fillna_method_doesnt_change_orig(self, method: str) -> None:
        data: np.ndarray = np.arange(10, dtype='i8') * 24 * 3600 * 10 ** 9
        if self.array_cls is PeriodArray:
            arr: NumpyExtensionArray = self.array_cls(data, dtype='period[D]')
        else:
            dtype: str = 'M8[ns]' if self.array_cls is DatetimeArray else 'm8[ns]'
            arr = self.array_cls._from_sequence(data, dtype=np.dtype(dtype))  # type: ignore
        arr[4] = NaT
        fill_value = arr[3] if method == 'pad' else arr[5]
        result = arr._pad_or_backfill(method=method)
        assert result[4] == fill_value
        assert arr[4] is NaT

    def test_searchsorted(self) -> None:
        data: np.ndarray = np.arange(10, dtype='i8') * 24 * 3600 * 10 ** 9
        if self.array_cls is PeriodArray:
            arr: NumpyExtensionArray = self.array_cls(data, dtype='period[D]')
        else:
            dtype: str = 'M8[ns]' if self.array_cls is DatetimeArray else 'm8[ns]'
            arr = self.array_cls._from_sequence(data, dtype=np.dtype(dtype))  # type: ignore
        result: int = arr.searchsorted(arr[1])
        assert result == 1
        result = arr.searchsorted(arr[2], side='right')
        assert result == 3
        result_arr: np.ndarray = arr.searchsorted(arr[1:3])
        expected: np.ndarray = np.array([1, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(result_arr, expected)
        result_arr = arr.searchsorted(arr[1:3], side='right')
        expected = np.array([2, 3], dtype=np.intp)
        tm.assert_numpy_array_equal(result_arr, expected)
        result = arr.searchsorted(NaT)
        assert result == 10

    @pytest.mark.parametrize('box', [None, 'index', 'series'])
    def test_searchsorted_castable_strings(self, arr1d: NumpyExtensionArray, box: Optional[Union[str, None]], string_storage: str) -> None:
        arr: Union[NumpyExtensionArray, pd.Index, pd.Series] = arr1d
        if box is None:
            pass
        elif box == 'index':
            arr = self.index_cls(arr)  # type: ignore
        else:
            arr = pd.Series(arr)
        result: int = arr.searchsorted(str(arr[1]))
        assert result == 1
        result = arr.searchsorted(str(arr[2]), side='right')
        assert result == 3
        result_arr = arr.searchsorted([str(x) for x in arr[1:3]])
        expected = np.array([1, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(result_arr, expected)
        with pytest.raises(TypeError, match=re.escape(f"value should be a '{arr1d._scalar_type.__name__}', 'NaT', or array of those. Got 'str' instead.")):
            _ = arr.searchsorted('foo')
        with pd.option_context('string_storage', string_storage):
            with pytest.raises(TypeError, match=re.escape(f"value should be a '{arr1d._scalar_type.__name__}', 'NaT', or array of those. Got string array instead.")):
                _ = arr.searchsorted([str(arr[1]), 'baz'])

    def test_getitem_near_implementation_bounds(self) -> None:
        i8vals: np.ndarray = np.asarray([NaT._value + n for n in range(1, 5)], dtype='i8')
        if self.array_cls is PeriodArray:
            arr: NumpyExtensionArray = self.array_cls(i8vals, dtype='period[ns]')
        else:
            arr = self.index_cls(i8vals, freq='ns')._data  # type: ignore
        _ = arr[0]
        index = pd.Index(arr)
        _ = index[0]
        ser = pd.Series(arr)
        _ = ser[0]

    def test_getitem_2d(self, arr1d: NumpyExtensionArray) -> None:
        expected = type(arr1d)._simple_new(arr1d._ndarray[:, np.newaxis], dtype=arr1d.dtype)  # type: ignore
        result = arr1d[:, np.newaxis]
        tm.assert_equal(result, expected)
        arr2d = expected
        expected = type(arr2d)._simple_new(arr2d._ndarray[:3, 0], dtype=arr2d.dtype)  # type: ignore
        result = arr2d[:3, 0]
        tm.assert_equal(result, expected)
        result_scalar = arr2d[-1, 0]
        expected_scalar = arr1d[-1]
        assert result_scalar == expected_scalar

    def test_iter_2d(self, arr1d: NumpyExtensionArray) -> None:
        data2d: np.ndarray = arr1d._ndarray[:3, np.newaxis]  # type: ignore
        arr2d: NumpyExtensionArray = type(arr1d)._simple_new(data2d, dtype=arr1d.dtype)  # type: ignore
        result = list(arr2d)
        assert len(result) == 3
        for x in result:
            assert isinstance(x, type(arr1d))
            assert x.ndim == 1
            assert x.dtype == arr1d.dtype

    def test_repr_2d(self, arr1d: NumpyExtensionArray) -> None:
        data2d: np.ndarray = arr1d._ndarray[:3, np.newaxis]  # type: ignore
        arr2d: NumpyExtensionArray = type(arr1d)._simple_new(data2d, dtype=arr1d.dtype)  # type: ignore
        result = repr(arr2d)
        if isinstance(arr2d, TimedeltaArray):
            expected = f"<{type(arr2d).__name__}>\n[\n['{arr1d[0]._repr_base()}'],\n['{arr1d[1]._repr_base()}'],\n['{arr1d[2]._repr_base()}']\n]\nShape: (3, 1), dtype: {arr1d.dtype}"
        else:
            expected = f"<{type(arr2d).__name__}>\n[\n['{arr1d[0]}'],\n['{arr1d[1]}'],\n['{arr1d[2]}']\n]\nShape: (3, 1), dtype: {arr1d.dtype}"
        assert result == expected

    def test_setitem(self) -> None:
        data: np.ndarray = np.arange(10, dtype='i8') * 24 * 3600 * 10 ** 9
        if self.array_cls is PeriodArray:
            arr: NumpyExtensionArray = self.array_cls(data, dtype='period[D]')
        else:
            arr = self.index_cls(data, freq='D')._data  # type: ignore
        arr[0] = arr[1]
        expected = np.arange(10, dtype='i8') * 24 * 3600 * 10 ** 9
        expected[0] = expected[1]
        tm.assert_numpy_array_equal(arr.asi8, expected)
        arr[:2] = arr[-2:]
        expected[:2] = expected[-2:]
        tm.assert_numpy_array_equal(arr.asi8, expected)

    @pytest.mark.parametrize('box', [list, np.array, pd.Index, pd.Series, NumpyExtensionArray])
    def test_setitem_object_dtype(self, box: Callable[[Any], Any], arr1d: NumpyExtensionArray) -> None:
        expected = arr1d.copy()[::-1]
        if expected.dtype.kind in ['m', 'M']:
            expected = expected._with_freq(None)
        vals: Any = expected
        if box is list:
            vals = list(vals)
        elif box is np.array:
            vals = np.array(vals.astype(object))
        elif box is NumpyExtensionArray:
            vals = box(np.asarray(vals, dtype=object))
        else:
            vals = box(vals).astype(object)
        arr1d[:] = vals
        tm.assert_equal(arr1d, expected)

    def test_setitem_strs(self, arr1d: NumpyExtensionArray) -> None:
        expected = arr1d.copy()
        expected[[0, 1]] = arr1d[-2:]
        result = arr1d.copy()
        result[:2] = [str(x) for x in arr1d[-2:]]
        tm.assert_equal(result, expected)
        expected = arr1d.copy()
        expected[0] = arr1d[-1]
        result = arr1d.copy()
        result[0] = str(arr1d[-1])
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize('as_index', [True, False])
    def test_setitem_categorical(self, arr1d: NumpyExtensionArray, as_index: bool) -> None:
        expected = arr1d.copy()[::-1]
        if not isinstance(expected, PeriodArray):
            expected = expected._with_freq(None)
        cat = pd.Categorical(arr1d)
        if as_index:
            cat = pd.CategoricalIndex(cat)
        arr1d[:] = cat[::-1]
        tm.assert_equal(arr1d, expected)

    def test_setitem_raises(self, arr1d: NumpyExtensionArray) -> None:
        arr = arr1d[:10]
        val = arr[0]
        with pytest.raises(IndexError, match='index 12 is out of bounds'):
            arr[12] = val
        with pytest.raises(TypeError, match="value should be a.* 'object'"):
            arr[0] = object()
        msg: str = 'cannot set using a list-like indexer with a different length'
        with pytest.raises(ValueError, match=msg):
            arr[[]] = [arr[1]]
        msg = 'cannot set using a slice indexer with a different length than'
        with pytest.raises(ValueError, match=msg):
            arr[1:1] = arr[:3]

    @pytest.mark.parametrize('box', [list, np.array, pd.Index, pd.Series])
    def test_setitem_numeric_raises(self, box: Callable[[Any], Any], arr1d: NumpyExtensionArray) -> None:
        msg: str = f"value should be a '{arr1d._scalar_type.__name__}', 'NaT', or array of those. Got"
        with pytest.raises(TypeError, match=msg):
            arr1d[:2] = box([0, 1])
        with pytest.raises(TypeError, match=msg):
            arr1d[:2] = box([0.0, 1.0])

    def test_inplace_arithmetic(self) -> None:
        data: np.ndarray = np.arange(10, dtype='i8') * 24 * 3600 * 10 ** 9
        if self.array_cls is PeriodArray:
            arr: NumpyExtensionArray = self.array_cls(data, dtype='period[D]')
        else:
            arr = self.index_cls(data, freq='D')._data  # type: ignore
        expected = arr + pd.Timedelta(days=1)
        arr += pd.Timedelta(days=1)
        tm.assert_equal(arr, expected)
        expected = arr - pd.Timedelta(days=1)
        arr -= pd.Timedelta(days=1)
        tm.assert_equal(arr, expected)

    def test_shift_fill_int_deprecated(self, arr1d: NumpyExtensionArray) -> None:
        with pytest.raises(TypeError, match='value should be a'):
            _ = arr1d.shift(1, fill_value=1)

    def test_median(self, arr1d: NumpyExtensionArray) -> None:
        arr = arr1d
        if len(arr) % 2 == 0:
            arr = arr[:-1]
        expected = arr[len(arr) // 2]
        result = arr.median()
        assert type(result) is type(expected)
        assert result == expected
        arr[len(arr) // 2] = NaT
        if not isinstance(expected, Period):
            expected = arr[len(arr) // 2 - 1:len(arr) // 2 + 2].mean()
        assert arr.median(skipna=False) is NaT
        result = arr.median()
        assert type(result) is type(expected)
        assert result == expected
        assert arr[:0].median() is NaT
        assert arr[:0].median(skipna=False) is NaT
        arr2 = arr.reshape(-1, 1)
        result = arr2.median(axis=None)
        assert type(result) is type(expected)
        assert result == expected
        assert arr2.median(axis=None, skipna=False) is NaT
        result = arr2.median(axis=0)
        expected2 = type(arr)._from_sequence([expected], dtype=arr.dtype)  # type: ignore
        tm.assert_equal(result, expected2)
        result = arr2.median(axis=0, skipna=False)
        expected2 = type(arr)._from_sequence([NaT], dtype=arr.dtype)  # type: ignore
        tm.assert_equal(result, expected2)
        result = arr2.median(axis=1)
        tm.assert_equal(result, arr)
        result = arr2.median(axis=1, skipna=False)
        tm.assert_equal(result, arr)

    def test_from_integer_array(self) -> None:
        arr_int: np.ndarray = np.array([1, 2, 3], dtype=np.int64)
        data = pd.array(arr_int, dtype='Int64')
        if self.array_cls is PeriodArray:
            expected = self.array_cls(arr_int, dtype=self.example_dtype)  # type: ignore
            result = self.array_cls(data, dtype=self.example_dtype)  # type: ignore
        else:
            expected = self.array_cls._from_sequence(arr_int, dtype=self.example_dtype)  # type: ignore
            result = self.array_cls._from_sequence(data, dtype=self.example_dtype)  # type: ignore
        tm.assert_extension_array_equal(result, expected)

class TestDatetimeArray(SharedTests):
    index_cls = DatetimeIndex
    array_cls = DatetimeArray
    scalar_type = Timestamp
    example_dtype: str = 'M8[ns]'

    @pytest.fixture
    def arr1d(self, tz_naive_fixture: Any, freqstr: str) -> DatetimeArray:
        """
        Fixture returning DatetimeArray with parametrized frequency and
        timezones
        """
        tz = tz_naive_fixture
        dti: DatetimeIndex = pd.date_range('2016-01-01 01:01:00', periods=5, freq=freqstr, tz=tz)
        dta: DatetimeArray = dti._data
        return dta

    def test_round(self, arr1d: DatetimeArray) -> None:
        dti = self.index_cls(arr1d)  # type: ignore
        result = dti.round(freq='2min')
        expected = dti - pd.Timedelta(minutes=1)
        expected = expected._with_freq(None)
        tm.assert_index_equal(result, expected)
        dta = dti._data  # type: ignore
        result = dta.round(freq='2min')
        expected = expected._data._with_freq(None)  # type: ignore
        tm.assert_datetime_array_equal(result, expected)

    def test_array_interface(self, datetime_index: DatetimeIndex) -> None:
        arr: DatetimeArray = datetime_index._data  # type: ignore
        copy_false = None if np_version_gt2 else False
        result = np.asarray(arr)
        expected = arr._ndarray  # type: ignore
        assert result is expected
        tm.assert_numpy_array_equal(result, expected)
        result = np.array(arr, copy=copy_false)
        assert result is expected
        tm.assert_numpy_array_equal(result, expected)
        result = np.asarray(arr, dtype='datetime64[ns]')
        expected = arr._ndarray  # type: ignore
        assert result is expected
        tm.assert_numpy_array_equal(result, expected)
        result = np.array(arr, dtype='datetime64[ns]', copy=copy_false)
        assert result is expected
        tm.assert_numpy_array_equal(result, expected)
        result = np.array(arr, dtype='datetime64[ns]')
        if not np_version_gt2:
            assert result is not expected
        tm.assert_numpy_array_equal(result, expected)
        result = np.asarray(arr, dtype=object)
        expected = np.array(list(arr), dtype=object)
        tm.assert_numpy_array_equal(result, expected)
        result = np.asarray(arr, dtype='int64')
        assert result is not arr.asi8
        assert not np.may_share_memory(arr, result)
        expected = arr.asi8.copy()
        tm.assert_numpy_array_equal(result, expected)
        for dtype in ['float64', str]:
            result = np.asarray(arr, dtype=dtype)
            expected = np.asarray(arr).astype(dtype)
            tm.assert_numpy_array_equal(result, expected)

    def test_array_object_dtype(self, arr1d: DatetimeArray) -> None:
        arr: DatetimeArray = arr1d
        dti = self.index_cls(arr1d)  # type: ignore
        expected = np.array(list(dti))
        result = np.array(arr, dtype=object)
        tm.assert_numpy_array_equal(result, expected)
        result = np.array(dti, dtype=object)
        tm.assert_numpy_array_equal(result, expected)

    def test_array_tz(self, arr1d: DatetimeArray) -> None:
        arr: DatetimeArray = arr1d
        dti = self.index_cls(arr1d)  # type: ignore
        copy_false = None if np_version_gt2 else False
        expected = dti.asi8.view('M8[ns]')
        result = np.array(arr, dtype='M8[ns]')
        tm.assert_numpy_array_equal(result, expected)
        result = np.array(arr, dtype='datetime64[ns]')
        tm.assert_numpy_array_equal(result, expected)
        result = np.array(arr, dtype='M8[ns]', copy=copy_false)
        assert result.base is expected.base
        assert result.base is not None
        result = np.array(arr, dtype='datetime64[ns]', copy=copy_false)
        assert result.base is expected.base
        assert result.base is not None

    def test_array_i8_dtype(self, arr1d: DatetimeArray) -> None:
        arr: DatetimeArray = arr1d
        dti = self.index_cls(arr1d)  # type: ignore
        copy_false = None if np_version_gt2 else False
        expected = dti.asi8
        result = np.array(arr, dtype='i8')
        tm.assert_numpy_array_equal(result, expected)
        result = np.array(arr, dtype=np.int64)
        tm.assert_numpy_array_equal(result, expected)
        result = np.array(arr, dtype='i8', copy=copy_false)
        assert result.base is not expected.base
        assert result.base is None

    def test_from_array_keeps_base(self) -> None:
        arr: np.ndarray = np.array(['2000-01-01', '2000-01-02'], dtype='M8[ns]')
        dta = DatetimeArray._from_sequence(arr, dtype=arr.dtype)  # type: ignore
        assert dta._ndarray is arr  # type: ignore
        dta = DatetimeArray._from_sequence(arr[:0], dtype=arr.dtype)  # type: ignore
        assert dta._ndarray.base is arr  # type: ignore

    def test_from_dti(self, arr1d: DatetimeArray) -> None:
        arr: DatetimeArray = arr1d
        dti = self.index_cls(arr1d)  # type: ignore
        assert list(dti) == list(arr)
        dti2 = pd.Index(arr)
        assert isinstance(dti2, DatetimeIndex)
        assert list(dti2) == list(arr)

    def test_astype_object(self, arr1d: DatetimeArray) -> None:
        arr: DatetimeArray = arr1d
        dti = self.index_cls(arr1d)  # type: ignore
        asobj = arr.astype('O')
        assert isinstance(asobj, np.ndarray)
        assert asobj.dtype == 'O'
        assert list(asobj) == list(dti)

    @pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
    def test_to_period(self, datetime_index: DatetimeIndex, freqstr: str) -> None:
        dti = datetime_index
        arr: DatetimeArray = dti._data  # type: ignore
        freqstr_conv: str = PeriodDtype(to_offset(freqstr))._freqstr
        expected = dti.to_period(freq=freqstr_conv)
        result = arr.to_period(freq=freqstr_conv)
        assert isinstance(result, PeriodArray)
        tm.assert_equal(result, expected._data)

    def test_to_period_2d(self, arr1d: DatetimeArray) -> None:
        arr2d = arr1d.reshape(1, -1)
        warn: Optional[type] = None if arr1d.tz is None else UserWarning
        with tm.assert_produces_warning(warn, match='will drop timezone information'):
            result = arr2d.to_period('D')
            expected = arr1d.to_period('D').reshape(1, -1)
        tm.assert_period_array_equal(result, expected)

    @pytest.mark.parametrize('propname', DatetimeArray._bool_ops)
    def test_bool_properties(self, arr1d: DatetimeArray, propname: str) -> None:
        dti = self.index_cls(arr1d)  # type: ignore
        arr: DatetimeArray = arr1d
        result = getattr(arr, propname)
        expected = np.array(getattr(dti, propname), dtype=result.dtype)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize('propname', DatetimeArray._field_ops)
    def test_int_properties(self, arr1d: DatetimeArray, propname: str) -> None:
        dti = self.index_cls(arr1d)  # type: ignore
        arr: DatetimeArray = arr1d
        result = getattr(arr, propname)
        expected = np.array(getattr(dti, propname), dtype=result.dtype)
        tm.assert_numpy_array_equal(result, expected)

    def test_take_fill_valid(self, arr1d: DatetimeArray, fixed_now_ts: Timestamp) -> None:
        arr: DatetimeArray = arr1d
        dti = self.index_cls(arr1d)  # type: ignore
        now: Timestamp = fixed_now_ts.tz_localize(dti.tz)
        result = arr.take([-1, 1], allow_fill=True, fill_value=now)
        assert result[0] == now
        msg: str = f"value should be a '{arr1d._scalar_type.__name__}' or 'NaT'. Got"
        with pytest.raises(TypeError, match=msg):
            _ = arr.take([-1, 1], allow_fill=True, fill_value=now - now)
        with pytest.raises(TypeError, match=msg):
            _ = arr.take([-1, 1], allow_fill=True, fill_value=Period('2014Q1'))
        tz: Optional[str] = None if dti.tz is not None else 'US/Eastern'
        now = fixed_now_ts.tz_localize(tz)
        msg = 'Cannot compare tz-naive and tz-aware datetime-like objects'
        with pytest.raises(TypeError, match=msg):
            _ = arr.take([-1, 1], allow_fill=True, fill_value=now)
        value: Any = NaT._value
        msg = f"value should be a '{arr1d._scalar_type.__name__}' or 'NaT'. Got"
        with pytest.raises(TypeError, match=msg):
            _ = arr.take([-1, 1], allow_fill=True, fill_value=value)
        value = np.timedelta64('NaT', 'ns')
        with pytest.raises(TypeError, match=msg):
            _ = arr.take([-1, 1], allow_fill=True, fill_value=value)
        if arr.tz is not None:
            value = fixed_now_ts.tz_localize('Australia/Melbourne')
            result = arr.take([-1, 1], allow_fill=True, fill_value=value)
            expected = arr.take([-1, 1], allow_fill=True, fill_value=value.tz_convert(arr.dtype.tz))
            tm.assert_equal(result, expected)

    def test_concat_same_type_invalid(self, arr1d: DatetimeArray) -> None:
        arr: DatetimeArray = arr1d
        if arr.tz is None:
            other = arr.tz_localize('UTC')
        else:
            other = arr.tz_localize(None)
        with pytest.raises(ValueError, match='to_concat must have the same'):
            _ = arr._concat_same_type([arr, other])

    def test_concat_same_type_different_freq(self, unit: Any) -> None:
        a = pd.date_range('2000', periods=2, freq='D', tz='US/Central', unit=unit)._data  # type: ignore
        b = pd.date_range('2000', periods=2, freq='h', tz='US/Central', unit=unit)._data  # type: ignore
        result = DatetimeArray._concat_same_type([a, b])  # type: ignore
        expected = pd.to_datetime(['2000-01-01 00:00:00', '2000-01-02 00:00:00',
                                   '2000-01-01 00:00:00', '2000-01-01 01:00:00']).tz_localize('US/Central').as_unit(unit)._data  # type: ignore
        tm.assert_datetime_array_equal(result, expected)

    def test_strftime(self, arr1d: DatetimeArray, using_infer_string: bool) -> None:
        arr: DatetimeArray = arr1d
        result = arr.strftime('%Y %b')
        expected = np.array([ts.strftime('%Y %b') for ts in arr], dtype=object)
        if using_infer_string:
            expected = pd.array(expected, dtype=pd.StringDtype(na_value=np.nan))
        tm.assert_equal(result, expected)

    def test_strftime_nat(self, using_infer_string: bool) -> None:
        arr = DatetimeIndex(['2019-01-01', NaT])._data  # type: ignore
        result = arr.strftime('%Y-%m-%d')
        expected = np.array(['2019-01-01', np.nan], dtype=object)
        if using_infer_string:
            expected = pd.array(expected, dtype=pd.StringDtype(na_value=np.nan))
        tm.assert_equal(result, expected)

class TestTimedeltaArray(SharedTests):
    index_cls = TimedeltaIndex
    array_cls = TimedeltaArray
    scalar_type: type = pd.Timedelta
    example_dtype: str = 'm8[ns]'

    def test_from_tdi(self) -> None:
        tdi: TimedeltaIndex = TimedeltaIndex(['1 Day', '3 Hours'])
        arr = tdi._data  # type: ignore
        assert list(arr) == list(tdi)
        tdi2 = pd.Index(arr)
        assert isinstance(tdi2, TimedeltaIndex)
        assert list(tdi2) == list(arr)

    def test_astype_object(self) -> None:
        tdi: TimedeltaIndex = TimedeltaIndex(['1 Day', '3 Hours'])
        arr = tdi._data  # type: ignore
        asobj = arr.astype('O')
        assert isinstance(asobj, np.ndarray)
        assert asobj.dtype == 'O'
        assert list(asobj) == list(tdi)

    def test_to_pytimedelta(self, timedelta_index: TimedeltaIndex) -> None:
        tdi = timedelta_index
        arr = tdi._data  # type: ignore
        expected = tdi.to_pytimedelta()
        result = arr.to_pytimedelta()
        tm.assert_numpy_array_equal(result, expected)

    def test_total_seconds(self, timedelta_index: TimedeltaIndex) -> None:
        tdi = timedelta_index
        arr = tdi._data  # type: ignore
        expected = tdi.total_seconds()
        result = arr.total_seconds()
        tm.assert_numpy_array_equal(result, expected.values)

    @pytest.mark.parametrize('propname', TimedeltaArray._field_ops)
    def test_int_properties(self, timedelta_index: TimedeltaIndex, propname: str) -> None:
        tdi = timedelta_index
        arr = tdi._data  # type: ignore
        result = getattr(arr, propname)
        expected = np.array(getattr(tdi, propname), dtype=result.dtype)
        tm.assert_numpy_array_equal(result, expected)

    def test_array_interface(self, timedelta_index: TimedeltaIndex) -> None:
        arr = timedelta_index._data  # type: ignore
        copy_false = None if np_version_gt2 else False
        result = np.asarray(arr)
        expected = arr._ndarray  # type: ignore
        assert result is expected
        tm.assert_numpy_array_equal(result, expected)
        result = np.array(arr, copy=copy_false)
        assert result is expected
        tm.assert_numpy_array_equal(result, expected)
        result = np.asarray(arr, dtype='timedelta64[ns]')
        expected = arr._ndarray  # type: ignore
        assert result is expected
        tm.assert_numpy_array_equal(result, expected)
        result = np.array(arr, dtype='timedelta64[ns]', copy=copy_false)
        assert result is expected
        tm.assert_numpy_array_equal(result, expected)
        result = np.array(arr, dtype='timedelta64[ns]')
        if not np_version_gt2:
            assert result is not expected
        tm.assert_numpy_array_equal(result, expected)
        result = np.asarray(arr, dtype=object)
        expected = np.array(list(arr), dtype=object)
        tm.assert_numpy_array_equal(result, expected)
        result = np.asarray(arr, dtype='int64')
        assert result is not arr.asi8
        assert not np.may_share_memory(arr, result)
        expected = arr.asi8.copy()
        tm.assert_numpy_array_equal(result, expected)
        for dtype in ['float64', str]:
            result = np.asarray(arr, dtype=dtype)
            expected = np.asarray(arr).astype(dtype)
            tm.assert_numpy_array_equal(result, expected)

    def test_take_fill_valid(self, timedelta_index: TimedeltaIndex, fixed_now_ts: Timestamp) -> None:
        tdi = timedelta_index
        arr = tdi._data  # type: ignore
        td1 = pd.Timedelta(days=1)
        result = arr.take([-1, 1], allow_fill=True, fill_value=td1)
        assert result[0] == td1
        value: Any = fixed_now_ts
        msg: str = f"value should be a '{arr._scalar_type.__name__}' or 'NaT'. Got"
        with pytest.raises(TypeError, match=msg):
            _ = arr.take([0, 1], allow_fill=True, fill_value=value)
        value = fixed_now_ts.to_period('D')
        with pytest.raises(TypeError, match=msg):
            _ = arr.take([0, 1], allow_fill=True, fill_value=value)
        value = np.datetime64('NaT', 'ns')
        with pytest.raises(TypeError, match=msg):
            _ = arr.take([-1, 1], allow_fill=True, fill_value=value)

@pytest.mark.filterwarnings('ignore:Period with BDay freq is deprecated:FutureWarning')
@pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
class TestPeriodArray(SharedTests):
    index_cls = PeriodIndex
    array_cls = PeriodArray
    scalar_type = Period
    example_dtype = PeriodIndex([], freq='W').dtype

    @pytest.fixture
    def arr1d(self, period_index: pd.PeriodIndex) -> PeriodArray:
        """
        Fixture returning DatetimeArray from parametrized PeriodIndex objects
        """
        return period_index._data  # type: ignore

    def test_from_pi(self, arr1d: PeriodArray) -> None:
        pi = self.index_cls(arr1d)  # type: ignore
        arr = arr1d
        assert list(arr) == list(pi)
        pi2 = pd.Index(arr)
        assert isinstance(pi2, PeriodIndex)
        assert list(pi2) == list(arr)

    def test_astype_object(self, arr1d: PeriodArray) -> None:
        pi = self.index_cls(arr1d)  # type: ignore
        arr = arr1d
        asobj = arr.astype('O')
        assert isinstance(asobj, np.ndarray)
        assert asobj.dtype == 'O'
        assert list(asobj) == list(pi)

    def test_take_fill_valid(self, arr1d: PeriodArray) -> None:
        arr = arr1d
        value: Any = NaT._value
        msg: str = f"value should be a '{arr1d._scalar_type.__name__}' or 'NaT'. Got"
        with pytest.raises(TypeError, match=msg):
            _ = arr.take([-1, 1], allow_fill=True, fill_value=value)
        value = np.timedelta64('NaT', 'ns')
        with pytest.raises(TypeError, match=msg):
            _ = arr.take([-1, 1], allow_fill=True, fill_value=value)

    @pytest.mark.parametrize('how', ['S', 'E'])
    def test_to_timestamp(self, how: str, arr1d: PeriodArray) -> None:
        pi = self.index_cls(arr1d)  # type: ignore
        arr = arr1d
        expected = DatetimeIndex(pi.to_timestamp(how=how))._data  # type: ignore
        result = arr.to_timestamp(how=how)
        assert isinstance(result, DatetimeArray)
        tm.assert_equal(result, expected)

    def test_to_timestamp_roundtrip_bday(self) -> None:
        dta = pd.date_range('2021-10-18', periods=3, freq='B')._data  # type: ignore
        parr = dta.to_period()
        result = parr.to_timestamp()
        assert result.freq == 'B'
        tm.assert_extension_array_equal(result, dta)
        dta2 = dta[::2]
        parr2 = dta2.to_period()
        result2 = parr2.to_timestamp()
        assert result2.freq == '2B'
        tm.assert_extension_array_equal(result2, dta2)
        parr3 = dta.to_period('2B')
        result3 = parr3.to_timestamp()
        assert result3.freq == 'B'
        tm.assert_extension_array_equal(result3, dta)

    def test_to_timestamp_out_of_bounds(self) -> None:
        pi = pd.period_range('1500', freq='Y', periods=3)
        msg: str = 'Out of bounds nanosecond timestamp: 1500-01-01 00:00:00'
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            _ = pi.to_timestamp()
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            _ = pi._data.to_timestamp()  # type: ignore

    @pytest.mark.parametrize('propname', PeriodArray._bool_ops)
    def test_bool_properties(self, arr1d: PeriodArray, propname: str) -> None:
        pi = self.index_cls(arr1d)  # type: ignore
        arr = arr1d
        result = getattr(arr, propname)
        expected = np.array(getattr(pi, propname))
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize('propname', PeriodArray._field_ops)
    def test_int_properties(self, arr1d: PeriodArray, propname: str) -> None:
        pi = self.index_cls(arr1d)  # type: ignore
        arr = arr1d
        result = getattr(arr, propname)
        expected = np.array(getattr(pi, propname))
        tm.assert_numpy_array_equal(result, expected)

    def test_array_interface(self, arr1d: PeriodArray) -> None:
        arr = arr1d
        result = np.asarray(arr, dtype=object)
        expected = np.array(list(arr), dtype=object)
        tm.assert_numpy_array_equal(result, expected)
        result = np.asarray(arr, dtype=object)
        tm.assert_numpy_array_equal(result, expected)
        result = np.asarray(arr, dtype='int64')
        tm.assert_numpy_array_equal(result, arr.asi8)
        result2 = np.asarray(arr, dtype='int64')
        assert np.may_share_memory(result, result2)
        result_copy1 = np.array(arr, dtype='int64', copy=True)
        result_copy2 = np.array(arr, dtype='int64', copy=True)
        assert not np.may_share_memory(result_copy1, result_copy2)
        msg: str = "float\\(\\) argument must be a string or a( real)? number, not 'Period'"
        with pytest.raises(TypeError, match=msg):
            _ = np.asarray(arr, dtype='float64')
        result = np.asarray(arr, dtype='S20')
        expected = np.asarray(arr).astype('S20')
        tm.assert_numpy_array_equal(result, expected)

    def test_strftime(self, arr1d: PeriodArray, using_infer_string: bool) -> None:
        arr = arr1d
        result = arr.strftime('%Y')
        expected = np.array([per.strftime('%Y') for per in arr], dtype=object)
        if using_infer_string:
            expected = pd.array(expected, dtype=pd.StringDtype(na_value=np.nan))
        tm.assert_equal(result, expected)

    def test_strftime_nat(self, using_infer_string: bool) -> None:
        arr = PeriodArray(PeriodIndex(['2019-01-01', NaT], dtype='period[D]'))
        result = arr.strftime('%Y-%m-%d')
        expected = np.array(['2019-01-01', np.nan], dtype=object)
        if using_infer_string:
            expected = pd.array(expected, dtype=pd.StringDtype(na_value=np.nan))
        tm.assert_equal(result, expected)

@pytest.mark.parametrize('arr,casting_nats', [
    (TimedeltaIndex(['1 Day', '3 Hours', 'NaT'])._data, (NaT, np.timedelta64('NaT', 'ns'))),
    (pd.date_range('2000-01-01', periods=3, freq='D')._data, (NaT, np.datetime64('NaT', 'ns'))),
    (pd.period_range('2000-01-01', periods=3, freq='D')._data, (NaT,))
], ids=lambda x: type(x).__name__)
def test_casting_nat_setitem_array(arr: NumpyExtensionArray, casting_nats: Tuple[Any, ...]) -> None:
    expected = type(arr)._from_sequence([NaT, arr[1], arr[2]], dtype=arr.dtype)  # type: ignore
    for nat in casting_nats:
        arr_copy = arr.copy()
        arr_copy[0] = nat
        tm.assert_equal(arr_copy, expected)

@pytest.mark.parametrize('arr,non_casting_nats', [
    (TimedeltaIndex(['1 Day', '3 Hours', 'NaT'])._data, (np.datetime64('NaT', 'ns'), NaT._value)),
    (pd.date_range('2000-01-01', periods=3, freq='D')._data, (np.timedelta64('NaT', 'ns'), NaT._value)),
    (pd.period_range('2000-01-01', periods=3, freq='D')._data, (np.datetime64('NaT', 'ns'), np.timedelta64('NaT', 'ns'), NaT._value))
], ids=lambda x: type(x).__name__)
def test_invalid_nat_setitem_array(arr: NumpyExtensionArray, non_casting_nats: Tuple[Any, ...]) -> None:
    msg: str = "value should be a '(Timestamp|Timedelta|Period)', 'NaT', or array of those. Got '(timedelta64|datetime64|int)' instead."
    for nat in non_casting_nats:
        with pytest.raises(TypeError, match=msg):
            arr[0] = nat

@pytest.mark.parametrize('arr', [pd.date_range('2000', periods=4).array, pd.timedelta_range('2000', periods=4).array])
def test_to_numpy_extra(arr: NumpyExtensionArray) -> None:
    arr[0] = NaT
    original = arr.copy()
    result = arr.to_numpy()
    assert np.isnan(result[0])
    result = arr.to_numpy(dtype='int64')
    assert result[0] == -9223372036854775808
    result = arr.to_numpy(dtype='int64', na_value=0)
    assert result[0] == 0
    result = arr.to_numpy(na_value=arr[1].to_numpy())
    assert result[0] == result[1]
    result = arr.to_numpy(na_value=arr[1].to_numpy(copy=False))
    assert result[0] == result[1]
    tm.assert_equal(arr, original)

@pytest.mark.parametrize('as_index', [True, False])
@pytest.mark.parametrize('values', [
    pd.to_datetime(['2020-01-01', '2020-02-01']),
    pd.to_timedelta([1, 2], unit='D'),
    PeriodIndex(['2020-01-01', '2020-02-01'], freq='D')
])
@pytest.mark.parametrize('klass', [list, np.array, pd.array, pd.Series, pd.Index, pd.Categorical, pd.CategoricalIndex])
def test_searchsorted_datetimelike_with_listlike(values: Union[DatetimeIndex, TimedeltaIndex, PeriodIndex], klass: Any, as_index: bool) -> None:
    if not as_index:
        values = values._data  # type: ignore
    result = values.searchsorted(klass(values))
    expected = np.array([0, 1], dtype=result.dtype)
    tm.assert_numpy_array_equal(result, expected)

@pytest.mark.parametrize('values', [
    pd.to_datetime(['2020-01-01', '2020-02-01']),
    pd.to_timedelta([1, 2], unit='D'),
    PeriodIndex(['2020-01-01', '2020-02-01'], freq='D')
])
@pytest.mark.parametrize('arg', [[1, 2], ['a', 'b'], [Timestamp('2020-01-01', tz='Europe/London')] * 2])
def test_searchsorted_datetimelike_with_listlike_invalid_dtype(values: Union[DatetimeIndex, TimedeltaIndex, PeriodIndex], arg: Any) -> None:
    msg: str = '[Unexpected type|Cannot compare]'
    with pytest.raises(TypeError, match=msg):
        _ = values.searchsorted(arg)

@pytest.mark.parametrize('klass', [list, tuple, np.array, pd.Series])
def test_period_index_construction_from_strings(klass: Callable[[Any], Any]) -> None:
    strings: List[str] = ['2020Q1', '2020Q2'] * 2
    data = klass(strings)
    result = PeriodIndex(data, freq='Q')
    expected = PeriodIndex([Period(s) for s in strings])
    tm.assert_index_equal(result, expected)

@pytest.mark.parametrize('dtype', ['M8[ns]', 'm8[ns]'])
def test_from_pandas_array(dtype: str) -> None:
    data = np.array([1, 2, 3], dtype=dtype)
    arr = NumpyExtensionArray(data)
    cls = {'M8[ns]': DatetimeArray, 'm8[ns]': TimedeltaArray}[dtype]
    result = cls._from_sequence(arr, dtype=dtype)  # type: ignore
    expected = cls._from_sequence(data, dtype=dtype)
    tm.assert_extension_array_equal(result, expected)
    func = {'M8[ns]': pd.to_datetime, 'm8[ns]': pd.to_timedelta}[dtype]
    result = func(arr).array
    expected = func(data).array
    tm.assert_equal(result, expected)
    idx_cls = {'M8[ns]': DatetimeIndex, 'm8[ns]': TimedeltaIndex}[dtype]
    result = idx_cls(arr)
    expected = idx_cls(data)
    tm.assert_index_equal(result, expected)