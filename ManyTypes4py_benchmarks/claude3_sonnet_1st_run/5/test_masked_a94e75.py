"""
This file contains a minimal set of tests for compliance with the extension
array interface test suite, and should contain no other tests.
The test suite for the full functionality of the array is located in
`pandas/tests/arrays/`.

The tests in this file are inherited from the BaseExtensionTests, and only
minimal tweaks should be applied to get the tests passing (by overwriting a
parent method).

Additional tests should either be added to one of the BaseExtensionTests
classes (if they are relevant for the extension interface for all dtypes), or
be added to the array-specific tests in `pandas/tests/arrays/`.

"""
from typing import Any, Callable, List, Optional, Union, Type, cast

import numpy as np
import pytest
from pandas.compat import IS64, is_platform_windows
from pandas.compat.numpy import np_version_gt2
from pandas.core.dtypes.common import is_float_dtype, is_signed_integer_dtype, is_unsigned_integer_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.boolean import BooleanDtype
from pandas.core.arrays.floating import Float32Dtype, Float64Dtype
from pandas.core.arrays.integer import Int8Dtype, Int16Dtype, Int32Dtype, Int64Dtype, UInt8Dtype, UInt16Dtype, UInt32Dtype, UInt64Dtype
from pandas.tests.extension import base
is_windows_or_32bit: bool = is_platform_windows() and (not np_version_gt2) or not IS64
pytestmark = [pytest.mark.filterwarnings('ignore:invalid value encountered in divide:RuntimeWarning'), pytest.mark.filterwarnings('ignore:Mean of empty slice:RuntimeWarning'), pytest.mark.filterwarnings('ignore:overflow encountered in reduce:RuntimeWarning')]

def make_data() -> List[Union[int, Any]]:
    return list(range(1, 9)) + [pd.NA] + list(range(10, 98)) + [pd.NA] + [99, 100]

def make_float_data() -> List[Union[float, Any]]:
    return list(np.arange(0.1, 0.9, 0.1)) + [pd.NA] + list(np.arange(1, 9.8, 0.1)) + [pd.NA] + [9.9, 10.0]

def make_bool_data() -> List[Union[bool, Any]]:
    return [True, False] * 4 + [np.nan] + [True, False] * 44 + [np.nan] + [True, False]

@pytest.fixture(params=[Int8Dtype, Int16Dtype, Int32Dtype, Int64Dtype, UInt8Dtype, UInt16Dtype, UInt32Dtype, UInt64Dtype, Float32Dtype, Float64Dtype, BooleanDtype])
def dtype(request: Any) -> Any:
    return request.param()

@pytest.fixture
def data(dtype: Any) -> pd.arrays.ExtensionArray:
    if dtype.kind == 'f':
        data = make_float_data()
    elif dtype.kind == 'b':
        data = make_bool_data()
    else:
        data = make_data()
    return pd.array(data, dtype=dtype)

@pytest.fixture
def data_for_twos(dtype: Any) -> pd.arrays.ExtensionArray:
    if dtype.kind == 'b':
        return pd.array(np.ones(100), dtype=dtype)
    return pd.array(np.ones(100) * 2, dtype=dtype)

@pytest.fixture
def data_missing(dtype: Any) -> pd.arrays.ExtensionArray:
    if dtype.kind == 'f':
        return pd.array([pd.NA, 0.1], dtype=dtype)
    elif dtype.kind == 'b':
        return pd.array([np.nan, True], dtype=dtype)
    return pd.array([pd.NA, 1], dtype=dtype)

@pytest.fixture
def data_for_sorting(dtype: Any) -> pd.arrays.ExtensionArray:
    if dtype.kind == 'f':
        return pd.array([0.1, 0.2, 0.0], dtype=dtype)
    elif dtype.kind == 'b':
        return pd.array([True, True, False], dtype=dtype)
    return pd.array([1, 2, 0], dtype=dtype)

@pytest.fixture
def data_missing_for_sorting(dtype: Any) -> pd.arrays.ExtensionArray:
    if dtype.kind == 'f':
        return pd.array([0.1, pd.NA, 0.0], dtype=dtype)
    elif dtype.kind == 'b':
        return pd.array([True, np.nan, False], dtype=dtype)
    return pd.array([1, pd.NA, 0], dtype=dtype)

@pytest.fixture
def na_cmp() -> Callable[[Any, Any], bool]:
    return lambda x, y: x is pd.NA and y is pd.NA

@pytest.fixture
def data_for_grouping(dtype: Any) -> pd.arrays.ExtensionArray:
    if dtype.kind == 'f':
        b = 0.1
        a = 0.0
        c = 0.2
    elif dtype.kind == 'b':
        b = True
        a = False
        c = b
    else:
        b = 1
        a = 0
        c = 2
    na = pd.NA
    return pd.array([b, b, na, na, a, a, b, c], dtype=dtype)

class TestMaskedArrays(base.ExtensionTests):

    @pytest.mark.parametrize('na_action', [None, 'ignore'])
    def test_map(self, data_missing: pd.arrays.ExtensionArray, na_action: Optional[str]) -> None:
        result = data_missing.map(lambda x: x, na_action=na_action)
        if data_missing.dtype == Float32Dtype():
            expected = data_missing.to_numpy(dtype='float64', na_value=np.nan)
        else:
            expected = data_missing.to_numpy()
        tm.assert_numpy_array_equal(result, expected)

    def test_map_na_action_ignore(self, data_missing_for_sorting: pd.arrays.ExtensionArray) -> None:
        zero = data_missing_for_sorting[2]
        result = data_missing_for_sorting.map(lambda x: zero, na_action='ignore')
        if data_missing_for_sorting.dtype.kind == 'b':
            expected = np.array([False, pd.NA, False], dtype=object)
        else:
            expected = np.array([zero, np.nan, zero])
        tm.assert_numpy_array_equal(result, expected)

    def _get_expected_exception(self, op_name: str, obj: Any, other: Any) -> Optional[Type[Exception]]:
        try:
            dtype = tm.get_dtype(obj)
        except AttributeError:
            dtype = tm.get_dtype(other)
        if dtype.kind == 'b':
            if op_name.strip('_').lstrip('r') in ['pow', 'truediv', 'floordiv']:
                return NotImplementedError
            elif op_name in ['__sub__', '__rsub__']:
                return TypeError
            return None
        return None

    def _cast_pointwise_result(self, op_name: str, obj: Any, other: Any, pointwise_result: Any) -> Any:
        sdtype = tm.get_dtype(obj)
        expected = pointwise_result
        if op_name in ('eq', 'ne', 'le', 'ge', 'lt', 'gt'):
            return expected.astype('boolean')
        if sdtype.kind in 'iu':
            if op_name in ('__rtruediv__', '__truediv__', '__div__'):
                filled = expected.fillna(np.nan)
                expected = filled.astype('Float64')
            else:
                expected = expected.astype(sdtype)
        elif sdtype.kind == 'b':
            if op_name in ('__floordiv__', '__rfloordiv__', '__pow__', '__rpow__', '__mod__', '__rmod__'):
                expected = expected.astype('Int8')
            elif op_name in ('__truediv__', '__rtruediv__'):
                op = self.get_op_from_name(op_name)
                expected = self._combine(obj.astype(float), other, op)
                expected = expected.astype('Float64')
            if op_name == '__rpow__':
                result = getattr(obj, op_name)(other)
                expected[result.isna()] = np.nan
        else:
            expected = expected.astype(sdtype)
        return expected

    def test_divmod_series_array(self, data: pd.arrays.ExtensionArray, data_for_twos: pd.arrays.ExtensionArray, request: Any) -> None:
        if data.dtype.kind == 'b':
            mark = pytest.mark.xfail(reason='Inconsistency between floordiv and divmod; we raise for floordiv but not for divmod. This matches what we do for non-masked bool dtype.')
            request.applymarker(mark)
        super().test_divmod_series_array(data, data_for_twos)

    def test_combine_le(self, data_repeated: Any) -> None:
        orig_data1, orig_data2 = data_repeated(2)
        if orig_data1.dtype.kind == 'b':
            self._combine_le_expected_dtype = 'boolean'
        else:
            self._combine_le_expected_dtype = object
        super().test_combine_le(data_repeated)

    def _supports_reduction(self, ser: pd.Series, op_name: str) -> bool:
        if op_name in ['any', 'all'] and ser.dtype.kind != 'b':
            pytest.skip(reason='Tested in tests/reductions/test_reductions.py')
        return True

    def check_reduce(self, ser: pd.Series, op_name: str, skipna: bool) -> None:
        cmp_dtype = 'int64'
        if ser.dtype.kind == 'f':
            cmp_dtype = ser.dtype.numpy_dtype
        elif ser.dtype.kind == 'b':
            if op_name in ['min', 'max']:
                cmp_dtype = 'bool'
        alt = ser.dropna().astype(cmp_dtype)
        if op_name == 'count':
            result = getattr(ser, op_name)()
            expected = getattr(alt, op_name)()
        else:
            result = getattr(ser, op_name)(skipna=skipna)
            expected = getattr(alt, op_name)(skipna=skipna)
            if not skipna and ser.isna().any() and (op_name not in ['any', 'all']):
                expected = pd.NA
        tm.assert_almost_equal(result, expected)

    def _get_expected_reduction_dtype(self, arr: pd.arrays.ExtensionArray, op_name: str, skipna: bool) -> str:
        if is_float_dtype(arr.dtype):
            cmp_dtype = arr.dtype.name
        elif op_name in ['mean', 'median', 'var', 'std', 'skew', 'kurt', 'sem']:
            cmp_dtype = 'Float64'
        elif op_name in ['max', 'min']:
            cmp_dtype = arr.dtype.name
        elif arr.dtype in ['Int64', 'UInt64']:
            cmp_dtype = arr.dtype.name
        elif is_signed_integer_dtype(arr.dtype):
            cmp_dtype = 'Int32' if is_platform_windows() and (not np_version_gt2 or not skipna) or not IS64 else 'Int64'
        elif is_unsigned_integer_dtype(arr.dtype):
            cmp_dtype = 'UInt32' if is_platform_windows() and (not np_version_gt2 or not skipna) or not IS64 else 'UInt64'
        elif arr.dtype.kind == 'b':
            if op_name in ['min', 'max']:
                cmp_dtype = 'boolean'
            elif op_name in ['sum', 'prod']:
                cmp_dtype = 'Int32' if is_platform_windows() and (not np_version_gt2 or not skipna) or not IS64 else 'Int64'
            else:
                raise TypeError('not supposed to reach this')
        else:
            raise TypeError('not supposed to reach this')
        return cmp_dtype

    def _supports_accumulation(self, ser: pd.Series, op_name: str) -> bool:
        return True

    def check_accumulate(self, ser: pd.Series, op_name: str, skipna: bool) -> None:
        length = 64
        if is_windows_or_32bit:
            if not ser.dtype.itemsize == 8:
                length = 32
        if ser.dtype.name.startswith('U'):
            expected_dtype = f'UInt{length}'
        elif ser.dtype.name.startswith('I'):
            expected_dtype = f'Int{length}'
        elif ser.dtype.name.startswith('F'):
            expected_dtype = ser.dtype
        elif ser.dtype.kind == 'b':
            if op_name in ('cummin', 'cummax'):
                expected_dtype = 'boolean'
            else:
                expected_dtype = f'Int{length}'
        if expected_dtype == 'Float32' and op_name == 'cumprod' and skipna:
            pytest.skip(f'Float32 precision lead to large differences with op {op_name} and skipna={skipna}')
        if op_name == 'cumsum':
            result = getattr(ser, op_name)(skipna=skipna)
            expected = pd.Series(pd.array(getattr(ser.astype('float64'), op_name)(skipna=skipna), dtype=expected_dtype))
            tm.assert_series_equal(result, expected)
        elif op_name in ['cummax', 'cummin']:
            result = getattr(ser, op_name)(skipna=skipna)
            expected = pd.Series(pd.array(getattr(ser.astype('float64'), op_name)(skipna=skipna), dtype=ser.dtype))
            tm.assert_series_equal(result, expected)
        elif op_name == 'cumprod':
            result = getattr(ser[:12], op_name)(skipna=skipna)
            expected = pd.Series(pd.array(getattr(ser[:12].astype('float64'), op_name)(skipna=skipna), dtype=expected_dtype))
            tm.assert_series_equal(result, expected)
        else:
            raise NotImplementedError(f'{op_name} not supported')

class Test2DCompat(base.Dim2CompatTests):
    pass
