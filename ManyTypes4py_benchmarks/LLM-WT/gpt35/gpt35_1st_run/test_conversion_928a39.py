from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import pytest
from pandas import CategoricalIndex, Series, Timedelta, Timestamp, date_range
from pandas.core.arrays import DatetimeArray, IntervalArray, NumpyExtensionArray, PeriodArray, SparseArray, TimedeltaArray
from pandas.core.arrays.string_ import StringArrayNumpySemantics
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.compat import HAS_PYARROW
from pandas.compat.numpy import np_version_gt2
from pandas._testing import tm

class TestToIterable:
    dtypes: List[Tuple[str, Union[int, float, Timestamp, Timedelta]]] = [('int8', int), ('int16', int), ('int32', int), ('int64', int), ('uint8', int), ('uint16', int), ('uint32', int), ('uint64', int), ('float16', float), ('float32', float), ('float64', float), ('datetime64[ns]', Timestamp), ('datetime64[ns, US/Eastern]', Timestamp), ('timedelta64[ns]', Timedelta)]

    def test_iterable(self, index_or_series, method, dtype, rdtype):
        typ = index_or_series
        if dtype == 'float16' and issubclass(typ, pd.Index):
            with pytest.raises(NotImplementedError, match='float16 indexes are not '):
                typ([1], dtype=dtype)
            return
        s = typ([1], dtype=dtype)
        result = method(s)[0]
        assert isinstance(result, rdtype)

    def test_iterable_object_and_category(self, index_or_series, method, dtype, rdtype, obj):
        typ = index_or_series
        s = typ([obj], dtype=dtype)
        result = method(s)[0]
        assert isinstance(result, rdtype)

    def test_iterable_items(self, dtype, rdtype):
        s = Series([1], dtype=dtype)
        _, result = next(iter(s.items()))
        assert isinstance(result, rdtype)

    def test_iterable_map(self, index_or_series, dtype, rdtype):
        typ = index_or_series
        if dtype == 'float16' and issubclass(typ, pd.Index):
            with pytest.raises(NotImplementedError, match='float16 indexes are not '):
                typ([1], dtype=dtype)
            return
        s = typ([1], dtype=dtype)
        result = s.map(type)[0]
        if not isinstance(rdtype, tuple):
            rdtype = (rdtype,)
        assert result in rdtype

    def test_categorial_datetimelike(self, method):
        i = CategoricalIndex([Timestamp('1999-12-31'), Timestamp('2000-12-31')])
        result = method(i)[0]
        assert isinstance(result, Timestamp)

    def test_iter_box_dt64(self, unit):
        vals = [Timestamp('2011-01-01'), Timestamp('2011-01-02')]
        ser = Series(vals).dt.as_unit(unit)
        assert ser.dtype == f'datetime64[{unit}]'
        for res, exp in zip(ser, vals):
            assert isinstance(res, Timestamp)
            assert res.tz is None
            assert res == exp
            assert res.unit == unit

    def test_iter_box_dt64tz(self, unit):
        vals = [Timestamp('2011-01-01', tz='US/Eastern'), Timestamp('2011-01-02', tz='US/Eastern')]
        ser = Series(vals).dt.as_unit(unit)
        assert ser.dtype == f'datetime64[{unit}, US/Eastern]'
        for res, exp in zip(ser, vals):
            assert isinstance(res, Timestamp)
            assert res.tz == exp.tz
            assert res == exp
            assert res.unit == unit

    def test_iter_box_timedelta64(self, unit):
        vals = [Timedelta('1 days'), Timedelta('2 days')]
        ser = Series(vals).dt.as_unit(unit)
        assert ser.dtype == f'timedelta64[{unit}]'
        for res, exp in zip(ser, vals):
            assert isinstance(res, Timedelta)
            assert res == exp
            assert res.unit == unit

    def test_iter_box_period(self):
        vals = [pd.Period('2011-01-01', freq='M'), pd.Period('2011-01-02', freq='M')]
        s = Series(vals)
        assert s.dtype == 'Period[M]'
        for res, exp in zip(s, vals):
            assert isinstance(res, pd.Period)
            assert res.freq == 'ME'
            assert res == exp

class TestValuesConsistent:
    def test_values_consistent(self, arr, expected_type, dtype, using_infer_string):
        if using_infer_string and dtype == 'object':
            expected_type = ArrowStringArrayNumpySemantics if HAS_PYARROW else StringArrayNumpySemantics
        l_values = Series(arr)._values
        r_values = pd.Index(arr)._values
        assert type(l_values) is expected_type
        assert type(l_values) is type(r_values)
        tm.assert_equal(l_values, r_values)

class TestNumpyArray:
    def test_numpy_array(self, arr):
        ser = Series(arr)
        result = ser.array
        expected = NumpyExtensionArray(arr)
        tm.assert_extension_array_equal(result, expected)

    def test_numpy_array_all_dtypes(self, any_numpy_dtype):
        ser = Series(dtype=any_numpy_dtype)
        result = ser.array
        if np.dtype(any_numpy_dtype).kind == 'M':
            assert isinstance(result, DatetimeArray)
        elif np.dtype(any_numpy_dtype).kind == 'm':
            assert isinstance(result, TimedeltaArray)
        else:
            assert isinstance(result, NumpyExtensionArray)

class TestArray:
    def test_array(self, arr, attr, index_or_series):
        box = index_or_series
        result = box(arr, copy=False).array
        if attr:
            arr = getattr(arr, attr)
            result = getattr(result, attr)
        assert result is arr

    def test_array_multiindex_raises(self):
        idx = pd.MultiIndex.from_product([['A'], ['a', 'b']])
        msg = 'MultiIndex has no single backing array'
        with pytest.raises(ValueError, match=msg):
            idx.array

class TestToNumpy:
    def test_to_numpy(self, arr, expected_type, dtype, using_infer_string):
        if using_infer_string and dtype == 'object':
            expected_type = ArrowStringArrayNumpySemantics if HAS_PYARROW else StringArrayNumpySemantics
        l_values = Series(arr)._values
        r_values = pd.Index(arr)._values
        assert type(l_values) is expected_type
        assert type(l_values) is type(r_values)
        tm.assert_equal(l_values, r_values)

    def test_to_numpy_array(self, arr, expected, zero_copy):
        if using_infer_string and arr.dtype == object and (obj.dtype.storage == 'pyarrow'):
            assert np.shares_memory(arr, result) is False
        else:
            assert np.shares_memory(arr, result) is True
        result = obj.to_numpy(copy=False)
        if using_infer_string and arr.dtype == object and (obj.dtype.storage == 'pyarrow'):
            assert np.shares_memory(arr, result) is False
        else:
            assert np.shares_memory(arr, result) is True
        result = obj.to_numpy(copy=True)
        assert np.shares_memory(arr, result) is False

    def test_to_numpy_dtype(self, as_series):
        tz = 'US/Eastern'
        obj = pd.DatetimeIndex(['2000', '2001'], tz=tz)
        if as_series:
            obj = Series(obj)
        result = obj.to_numpy()
        expected = np.array([Timestamp('2000', tz=tz), Timestamp('2001', tz=tz)], dtype=object)
        tm.assert_numpy_array_equal(result, expected)
        result = obj.to_numpy(dtype='object')
        tm.assert_numpy_array_equal(result, expected)
        result = obj.to_numpy(dtype='M8[ns]')
        expected = np.array(['2000-01-01T05', '2001-01-01T05'], dtype='M8[ns]')
        tm.assert_numpy_array_equal(result, expected)

    def test_to_numpy_na_value_numpy_dtype(self, index_or_series, values, dtype, na_value, expected):
        obj = index_or_series(values)
        result = obj.to_numpy(dtype=dtype, na_value=na_value)
        expected = np.array(expected)
        tm.assert_numpy_array_equal(result, expected)

    def test_to_numpy_multiindex_series_na_value(self, data, multiindex, dtype, na_value, expected):
        index = pd.MultiIndex.from_tuples(multiindex)
        series = Series(data, index=index)
        result = series.to_numpy(dtype=dtype, na_value=na_value)
        expected = np.array(expected)
        tm.assert_numpy_array_equal(result, expected)

    def test_to_numpy_kwargs_raises(self):
        s = Series([1, 2, 3])
        msg = "to_numpy\\(\\) got an unexpected keyword argument 'foo'"
        with pytest.raises(TypeError, match=msg):
            s.to_numpy(foo=True)
        s = Series([1, 2, 3], dtype='Int64')
        with pytest.raises(TypeError, match=msg):
            s.to_numpy(foo=True)

    def test_to_numpy_dataframe_na_value(self, data, dtype, na_value):
        df = pd.DataFrame(data)
        result = df.to_numpy(dtype=dtype, na_value=na_value)
        expected = np.array([[1, 1], [2, 2], [3, na_value]], dtype=dtype)
        tm.assert_numpy_array_equal(result, expected)

    def test_to_numpy_dataframe_single_block(self, data, expected_data):
        df = pd.DataFrame(data)
        result = df.to_numpy(dtype=float, na_value=np.nan)
        expected = np.array(expected_data, dtype=float)
        tm.assert_numpy_array_equal(result, expected)

    def test_to_numpy_dataframe_single_block_no_mutate(self):
        result = pd.DataFrame(np.array([1.0, 2.0, np.nan]))
        expected = pd.DataFrame(np.array([1.0, 2.0, np.nan]))
        result.to_numpy(na_value=0.0)
        tm.assert_frame_equal(result, expected)

class TestAsArray:
    def test_asarray_object_dt64(self, tz):
        ser = Series(date_range('2000', periods=2, tz=tz))
        with tm.assert_produces_warning(None):
            result = np.asarray(ser, dtype=object)
        expected = np.array([Timestamp('2000-01-01', tz=tz), Timestamp('2000-01-02', tz=tz)])
        tm.assert_numpy_array_equal(result, expected)

    def test_asarray_tz_naive(self):
        ser = Series(date_range('2000', periods=2))
        expected = np.array(['2000-01-01', '2000-01-02'], dtype='M8[ns]')
        result = np.asarray(ser)
        tm.assert_numpy_array_equal(result, expected)

    def test_asarray_tz_aware(self):
        tz = 'US/Central'
        ser = Series(date_range('2000', periods=2, tz=tz))
        expected = np.array(['2000-01-01T06', '2000-01-02T06'], dtype='M8[ns]')
        result = np.asarray(ser, dtype='datetime64[ns]')
        tm.assert_numpy_array_equal(result, expected)
        result = np.asarray(ser, dtype='M8[ns]')
        tm.assert_numpy_array_equal(result, expected)
