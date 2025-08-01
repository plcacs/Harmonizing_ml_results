from itertools import product
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pytest
from pandas._libs import lib
import pandas as pd
import pandas._testing as tm

class TestSeriesConvertDtypes:

    @pytest.mark.parametrize('data, maindtype, expected_default, expected_other', [([1, 2, 3], np.dtype('int32'), 'Int32', {('convert_integer', False): np.dtype('int32')}), ([1, 2, 3], np.dtype('int64'), 'Int64', {('convert_integer', False): np.dtype('int64')}), (['x', 'y', 'z'], np.dtype('O'), pd.StringDtype(), {('convert_string', False): np.dtype('O')}), ([True, False, np.nan], np.dtype('O'), pd.BooleanDtype(), {('convert_boolean', False): np.dtype('O')}), (['h', 'i', np.nan], np.dtype('O'), pd.StringDtype(), {('convert_string', False): np.dtype('O')}), (['h', 'i', 1], np.dtype('O'), np.dtype('O'), {}), ([10, np.nan, 20], np.dtype('float'), 'Int64', {('convert_integer', False, 'convert_floating', True): 'Float64', ('convert_integer', False, 'convert_floating', False): np.dtype('float')}), ([np.nan, 100.5, 200], np.dtype('float'), 'Float64', {('convert_floating', False): np.dtype('float')}), ([3, 4, 5], 'Int8', 'Int8', {}), ([[1, 2], [3, 4], [5]], None, np.dtype('O'), {}), ([4, 5, 6], np.dtype('uint32'), 'UInt32', {('convert_integer', False): np.dtype('uint32')}), ([-10, 12, 13], np.dtype('i1'), 'Int8', {('convert_integer', False): np.dtype('i1')}), ([1.2, 1.3], np.dtype('float32'), 'Float32', {('convert_floating', False): np.dtype('float32')}), ([1, 2.0], object, 'Int64', {('convert_integer', False): 'Float64', ('convert_integer', False, 'convert_floating', False): np.dtype('float'), ('infer_objects', False): np.dtype('object')}), ([1, 2.5], object, 'Float64', {('convert_floating', False): np.dtype('float'), ('infer_objects', False): np.dtype('object')}), (['a', 'b'], pd.CategoricalDtype(), pd.CategoricalDtype(), {}), (pd.to_datetime(['2020-01-14 10:00', '2020-01-15 11:11']).as_unit('s'), pd.DatetimeTZDtype(tz='UTC'), pd.DatetimeTZDtype(tz='UTC'), {}), (pd.to_datetime(['2020-01-14 10:00', '2020-01-15 11:11']).as_unit('ms'), pd.DatetimeTZDtype(tz='UTC'), pd.DatetimeTZDtype(tz='UTC'), {}), (pd.to_datetime(['2020-01-14 10:00', '2020-01-15 11:11']).as_unit('us'), pd.DatetimeTZDtype(tz='UTC'), pd.DatetimeTZDtype(tz='UTC'), {}), (pd.to_datetime(['2020-01-14 10:00', '2020-01-15 11:11']).as_unit('ns'), pd.DatetimeTZDtype(tz='UTC'), pd.DatetimeTZDtype(tz='UTC'), {}), (pd.to_datetime(['2020-01-14 10:00', '2020-01-15 11:11']).as_unit('ns'), 'datetime64[ns]', np.dtype('datetime64[ns]'), {}), (pd.to_datetime(['2020-01-14 10:00', '2020-01-15 11:11']).as_unit('ns'), object, np.dtype('datetime64[ns]'), {('infer_objects', False): np.dtype('object')}), (pd.period_range('1/1/2011', freq='M', periods=3), None, pd.PeriodDtype('M'), {}), (pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(1, 5)]), None, pd.IntervalDtype('int64', 'right'), {})])
    @pytest.mark.parametrize('params', product(*[(True, False)] * 5))
    def test_convert_dtypes(self, data: Any, maindtype: Any, expected_default: Any, expected_other: Dict[Tuple, Any], params: Tuple[bool, bool, bool, bool, bool], using_infer_string: bool) -> None:
        if hasattr(data, 'dtype') and lib.is_np_dtype(data.dtype, 'M') and isinstance(maindtype, pd.DatetimeTZDtype):
            msg = 'Cannot use .astype to convert from timezone-naive dtype'
            with pytest.raises(TypeError, match=msg):
                pd.Series(data, dtype=maindtype)
            return
        if maindtype is not None:
            series = pd.Series(data, dtype=maindtype)
        else:
            series = pd.Series(data)
        result = series.convert_dtypes(*params)
        param_names = ['infer_objects', 'convert_string', 'convert_integer', 'convert_boolean', 'convert_floating']
        params_dict = dict(zip(param_names, params))
        expected_dtype = expected_default
        for spec, dtype in expected_other.items():
            if all((params_dict[key] is val for key, val in zip(spec[::2], spec[1::2]))):
                expected_dtype = dtype
        if using_infer_string and expected_default == 'string' and (expected_dtype == object) and params[0] and (not params[1]):
            expected_dtype = pd.StringDtype(na_value=np.nan)
        expected = pd.Series(data, dtype=expected_dtype)
        tm.assert_series_equal(result, expected)
        copy = series.copy(deep=True)
        if result.notna().sum() > 0 and result.dtype in ['interval[int64, right]']:
            with pytest.raises(TypeError, match='Invalid value'):
                result[result.notna()] = np.nan
        else:
            result[result.notna()] = np.nan
        tm.assert_series_equal(series, copy)

    def test_convert_string_dtype(self, nullable_string_dtype: Any) -> None:
        df = pd.DataFrame({'A': ['a', 'b', pd.NA], 'B': ['ä', 'ö', 'ü']}, dtype=nullable_string_dtype)
        result = df.convert_dtypes()
        tm.assert_frame_equal(df, result)

    def test_convert_bool_dtype(self) -> None:
        df = pd.DataFrame({'A': pd.array([True])})
        tm.assert_frame_equal(df, df.convert_dtypes())

    def test_convert_byte_string_dtype(self) -> None:
        byte_str = b'binary-string'
        df = pd.DataFrame(data={'A': byte_str}, index=[0])
        result = df.convert_dtypes()
        expected = df
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('infer_objects, dtype', [(True, 'Int64'), (False, 'object')])
    def test_convert_dtype_object_with_na(self, infer_objects: bool, dtype: str) -> None:
        ser = pd.Series([1, pd.NA])
        result = ser.convert_dtypes(infer_objects=infer_objects)
        expected = pd.Series([1, pd.NA], dtype=dtype)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('infer_objects, dtype', [(True, 'Float64'), (False, 'object')])
    def test_convert_dtype_object_with_na_float(self, infer_objects: bool, dtype: str) -> None:
        ser = pd.Series([1.5, pd.NA])
        result = ser.convert_dtypes(infer_objects=infer_objects)
        expected = pd.Series([1.5, pd.NA], dtype=dtype)
        tm.assert_series_equal(result, expected)

    def test_convert_dtypes_pyarrow_to_np_nullable(self) -> None:
        pytest.importorskip('pyarrow')
        ser = pd.Series(range(2), dtype='int32[pyarrow]')
        result = ser.convert_dtypes(dtype_backend='numpy_nullable')
        expected = pd.Series(range(2), dtype='Int32')
        tm.assert_series_equal(result, expected)

    def test_convert_dtypes_pyarrow_null(self) -> None:
        pa = pytest.importorskip('pyarrow')
        ser = pd.Series([None, None])
        result = ser.convert_dtypes(dtype_backend='pyarrow')
        expected = pd.Series([None, None], dtype=pd.ArrowDtype(pa.null()))
        tm.assert_series_equal(result, expected)
