from __future__ import annotations
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from io import BytesIO, StringIO
import operator
import pickle
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import timezones
from pandas.compat import PY311, PY312, is_ci_environment, is_platform_windows, pa_version_under11p0, pa_version_under13p0, pa_version_under14p0
from pandas.core.dtypes.dtypes import ArrowDtype, CategoricalDtypeType
import pandas as pd
import pandas._testing as tm
from pandas.api.extensions import no_default
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_float_dtype, is_integer_dtype, is_numeric_dtype, is_signed_integer_dtype, is_string_dtype, is_unsigned_integer_dtype
from pandas.tests.extension import base
import pyarrow as pa
from pandas.core.arrays.arrow.array import ArrowExtensionArray, get_unit_from_pa_dtype
from pandas.core.arrays.arrow.extension_types import ArrowPeriodType

def _require_timezone_database(request: pytest.FixtureRequest) -> None:
    if is_platform_windows() and is_ci_environment():
        mark = pytest.mark.xfail(raises=pa.ArrowInvalid, reason='TODO: Set ARROW_TIMEZONE_DATABASE environment variable on CI to path to the tzdata for pyarrow.')
        request.applymarker(mark)

@pytest.fixture(params=tm.ALL_PYARROW_DTYPES, ids=str)
def dtype(request: pytest.FixtureRequest) -> ArrowDtype:
    return ArrowDtype(pyarrow_dtype=request.param)

@pytest.fixture
def data(dtype: ArrowDtype) -> ArrowExtensionArray:
    pa_dtype = dtype.pyarrow_dtype
    if pa.types.is_boolean(pa_dtype):
        data = [True, False] * 4 + [None] + [True, False] * 44 + [None] + [True, False]
    elif pa.types.is_floating(pa_dtype):
        data = [1.0, 0.0] * 4 + [None] + [-2.0, -1.0] * 44 + [None] + [0.5, 99.5]
    elif pa.types.is_signed_integer(pa_dtype):
        data = [1, 0] * 4 + [None] + [-2, -1] * 44 + [None] + [1, 99]
    elif pa.types.is_unsigned_integer(pa_dtype):
        data = [1, 0] * 4 + [None] + [2, 1] * 44 + [None] + [1, 99]
    elif pa.types.is_decimal(pa_dtype):
        data = [Decimal('1'), Decimal('0.0')] * 4 + [None] + [Decimal('-2.0'), Decimal('-1.0')] * 44 + [None] + [Decimal('0.5'), Decimal('33.123')]
    elif pa.types.is_date(pa_dtype):
        data = [date(2022, 1, 1), date(1999, 12, 31)] * 4 + [None] + [date(2022, 1, 1), date(2022, 1, 1)] * 44 + [None] + [date(1999, 12, 31), date(1999, 12, 31)]
    elif pa.types.is_timestamp(pa_dtype):
        data = [datetime(2020, 1, 1, 1, 1, 1, 1), datetime(1999, 1, 1, 1, 1, 1, 1)] * 4 + [None] + [datetime(2020, 1, 1, 1), datetime(1999, 1, 1, 1)] * 44 + [None] + [datetime(2020, 1, 1), datetime(1999, 1, 1)]
    elif pa.types.is_duration(pa_dtype):
        data = [timedelta(1), timedelta(1, 1)] * 4 + [None] + [timedelta(-1), timedelta(0)] * 44 + [None] + [timedelta(-10), timedelta(10)]
    elif pa.types.is_time(pa_dtype):
        data = [time(12, 0), time(0, 12)] * 4 + [None] + [time(0, 0), time(1, 1)] * 44 + [None] + [time(0, 5), time(5, 0)]
    elif pa.types.is_string(pa_dtype):
        data = ['a', 'b'] * 4 + [None] + ['1', '2'] * 44 + [None] + ['!', '>']
    elif pa.types.is_binary(pa_dtype):
        data = [b'a', b'b'] * 4 + [None] + [b'1', b'2'] * 44 + [None] + [b'!', b'>']
    else:
        raise NotImplementedError
    return pd.array(data, dtype=dtype)

@pytest.fixture
def data_missing(data: ArrowExtensionArray) -> ArrowExtensionArray:
    """Length-2 array with [NA, Valid]"""
    return type(data)._from_sequence([None, data[0]], dtype=data.dtype)

@pytest.fixture(params=['data', 'data_missing'])
def all_data(request: pytest.FixtureRequest, data: ArrowExtensionArray, data_missing: ArrowExtensionArray) -> ArrowExtensionArray:
    """Parametrized fixture returning 'data' or 'data_missing' integer arrays.

    Used to test dtype conversion with and without missing values.
    """
    if request.param == 'data':
        return data
    elif request.param == 'data_missing':
        return data_missing

@pytest.fixture
def data_for_grouping(dtype: ArrowDtype) -> ArrowExtensionArray:
    """
    Data for factorization, grouping, and unique tests.

    Expected to be like [B, B, NA, NA, A, A, B, C]

    Where A < B < C and NA is missing
    """
    pa_dtype = dtype.pyarrow_dtype
    if pa.types.is_boolean(pa_dtype):
        A = False
        B = True
        C = True
    elif pa.types.is_floating(pa_dtype):
        A = -1.1
        B = 0.0
        C = 1.1
    elif pa.types.is_signed_integer(pa_dtype):
        A = -1
        B = 0
        C = 1
    elif pa.types.is_unsigned_integer(pa_dtype):
        A = 0
        B = 1
        C = 10
    elif pa.types.is_date(pa_dtype):
        A = date(1999, 12, 31)
        B = date(2010, 1, 1)
        C = date(2022, 1, 1)
    elif pa.types.is_timestamp(pa_dtype):
        A = datetime(1999, 1, 1, 1, 1, 1, 1)
        B = datetime(2020, 1, 1)
        C = datetime(2020, 1, 1, 1)
    elif pa.types.is_duration(pa_dtype):
        A = timedelta(-1)
        B = timedelta(0)
        C = timedelta(1, 4)
    elif pa.types.is_time(pa_dtype):
        A = time(0, 0)
        B = time(0, 12)
        C = time(12, 12)
    elif pa.types.is_string(pa_dtype):
        A = 'a'
        B = 'b'
        C = 'c'
    elif pa.types.is_binary(pa_dtype):
        A = b'a'
        B = b'b'
        C = b'c'
    elif pa.types.is_decimal(pa_dtype):
        A = Decimal('-1.1')
        B = Decimal('0.0')
        C = Decimal('1.1')
    else:
        raise NotImplementedError
    return pd.array([B, B, None, None, A, A, B, C], dtype=dtype)

@pytest.fixture
def data_for_sorting(data_for_grouping: ArrowExtensionArray) -> ArrowExtensionArray:
    """
    Length-3 array with a known sort order.

    This should be three items [B, C, A] with
    A < B < C
    """
    return type(data_for_grouping)._from_sequence([data_for_grouping[0], data_for_grouping[7], data_for_grouping[4]], dtype=data_for_grouping.dtype)

@pytest.fixture
def data_missing_for_sorting(data_for_grouping: ArrowExtensionArray) -> ArrowExtensionArray:
    """
    Length-3 array with a known sort order.

    This should be three items [B, NA, A] with
    A < B and NA missing.
    """
    return type(data_for_grouping)._from_sequence([data_for_grouping[0], data_for_grouping[2], data_for_grouping[4]], dtype=data_for_grouping.dtype)

@pytest.fixture
def data_for_twos(data: ArrowExtensionArray) -> ArrowExtensionArray:
    """Length-100 array in which all the elements are two."""
    pa_dtype = data.dtype.pyarrow_dtype
    if pa.types.is_integer(pa_dtype) or pa.types.is_floating(pa_dtype) or pa.types.is_decimal(pa_dtype) or pa.types.is_duration(pa_dtype):
        return pd.array([2] * 100, dtype=data.dtype)
    return data

class TestArrowArray(base.ExtensionTests):

    def test_compare_scalar(self, data: ArrowExtensionArray, comparison_op: Callable) -> None:
        ser = pd.Series(data)
        self._compare_other(ser, data, comparison_op, data[0])

    @pytest.mark.parametrize('na_action', [None, 'ignore'])
    def test_map(self, data_missing: ArrowExtensionArray, na_action: Optional[str]) -> None:
        if data_missing.dtype.kind in 'mM':
            result = data_missing.map(lambda x: x, na_action=na_action)
            expected = data_missing.to_numpy(dtype=object)
            tm.assert_numpy_array_equal(result, expected)
        else:
            result = data_missing.map(lambda x: x, na_action=na_action)
            if data_missing.dtype == 'float32[pyarrow]':
                expected = data_missing.to_numpy(dtype='float64', na_value=np.nan)
            else:
                expected = data_missing.to_numpy()
            tm.assert_numpy_array_equal(result, expected)

    def test_astype_str(self, data: ArrowExtensionArray, request: pytest.FixtureRequest, using_infer_string: bool) -> None:
        pa_dtype = data.dtype.pyarrow_dtype
        if pa.types.is_binary(pa_dtype):
            request.applymarker(pytest.mark.xfail(reason=f'For {pa_dtype} .astype(str) decodes.'))
        elif not using_infer_string and (pa.types.is_timestamp(pa_dtype) and pa_dtype.tz is None or pa.types.is_duration(pa_dtype)):
            request.applymarker(pytest.mark.xfail(reason='pd.Timestamp/pd.Timedelta repr different from numpy repr'))
        super().test_astype_str(data)

    def test_from_dtype(self, data: ArrowExtensionArray, request: pytest.FixtureRequest) -> None:
        pa_dtype = data.dtype.pyarrow_dtype
        if pa.types.is_string(pa_dtype) or pa.types.is_decimal(pa_dtype):
            if pa.types.is_string(pa_dtype):
                reason = "ArrowDtype(pa.string()) != StringDtype('pyarrow')"
            else:
                reason = f'pyarrow.type_for_alias cannot infer {pa_dtype}'
            request.applymarker(pytest.mark.xfail(reason=reason))
        super().test_from_dtype(data)

    def test_from_sequence_pa_array(self, data: ArrowExtensionArray) -> None:
        result = type(data)._from_sequence(data._pa_array, dtype=data.dtype)
        tm.assert_extension_array_equal(result, data)
        assert isinstance(result._pa_array, pa.ChunkedArray)
        result = type(data)._from_sequence(data._pa_array.combine_chunks(), dtype=data.dtype)
        tm.assert_extension_array_equal(result, data)
        assert isinstance(result._pa_array, pa.ChunkedArray)

    def test_from_sequence_pa_array_notimplemented(self, request: pytest.FixtureRequest) -> None:
        dtype = ArrowDtype(pa.month_day_nano_interval())
        with pytest.raises(NotImplementedError, match='Converting strings to'):
            ArrowExtensionArray._from_sequence_of_strings(['12-1'], dtype=dtype)

    def test_from_sequence_of_strings_pa_array(self, data: ArrowExtensionArray, request: pytest.FixtureRequest) -> None:
        pa_dtype = data.dtype.pyarrow_dtype
        if pa.types.is_time64(pa_dtype) and pa_dtype.equals('time64[ns]') and (not PY311):
            request.applymarker(pytest.mark.xfail(reason='Nanosecond time parsing not supported.'))
        elif pa_version_under11p0 and (pa.types.is_duration(pa_dtype) or pa.types.is_decimal(pa_dtype)):
            request.applymarker(pytest.mark.xfail(raises=pa.ArrowNotImplementedError, reason=f"pyarrow doesn't support parsing {pa_dtype}"))
        elif pa.types.is_timestamp(pa_dtype) and pa_dtype.tz is not None:
            _require_timezone_database(request)
        pa_array = data._pa_array.cast(pa.string())
        result = type(data)._from_sequence_of_strings(pa_array, dtype=data.dtype)
        tm.assert_extension_array_equal(result, data)
        pa_array = pa_array.combine_chunks()
        result = type(data)._from_sequence_of_strings(pa_array, dtype=data.dtype)
        tm.assert_extension_array_equal(result, data)

    def check_accumulate(self, ser: pd.Series, op_name: str, skipna: bool) -> None:
        result = getattr(ser, op_name)(skipna=skipna)
        pa_type = ser.dtype.pyarrow_dtype
        if pa.types.is_temporal(pa_type):
            if pa_type.bit_width == 32:
                int_type = 'int32[pyarrow]'
            else:
                int_type = 'int64[pyarrow]'
            ser = ser.astype(int_type)
            result = result.astype(int_type)
        result = result.astype('Float64')
        expected = getattr(ser.astype('Float64'), op_name)(skipna=skipna)
        tm.assert_series_equal(result, expected, check_dtype=False)

    def _supports_accumulation(self, ser: pd.Series, op_name: str) -> bool:
        pa_type = ser.dtype.pyarrow_dtype
        if pa.types.is_binary(pa_type) or pa.types.is_decimal(pa_type):
            if op_name in ['cumsum', 'cumprod', 'cummax', 'cummin']:
                return False
        elif pa.types.is_string(pa_type):
            if op_name == 'cumprod':
                return False
        elif pa.types.is_boolean(pa_type):
            if op_name in ['cumprod', 'cummax', 'cummin']:
                return False
        elif pa.types.is_temporal(pa_type):
            if op_name == 'cumsum' and (not pa.types.is_duration(pa_type)):
                return False
            elif op_name == 'cumprod':
                return False
        return True

    @pytest.mark.parametrize('skipna', [True, False])
    def test_accumulate_series(self, data: ArrowExtensionArray, all_numeric_accumulations: str, skipna: bool, request: pytest.FixtureRequest) -> None:
        pa_type = data.dtype.pyarrow_dtype
        op_name = all_numeric_accumulations
        if pa.types.is_string(pa_type) and op_name in ['cumsum', 'cummin', 'cummax']:
            return
        ser = pd.Series(data)
        if not self._supports_accumulation(ser, op_name):
            return super().test_accumulate_series(data, all_numeric_accumulations, skipna)
        if pa_version_under13p0 and all_numeric_accumulations != 'cumsum':
            opt = request.config.option
            if opt.markexpr and 'not slow' in opt.markexpr:
                pytest.skip(f'{all_numeric_accumulations} not implemented for pyarrow < 9')
            mark = pytest.mark.xfail(reason=f'{all_numeric_accumulations} not implemented for pyarrow < 9')
            request.applymarker(mark)
        elif all_numeric_accumulations == 'cumsum' and (pa.types.is_boolean(pa_type) or pa.types.is_decimal(pa_type)):
            request.applymarker(pytest.mark.xfail(reason=f'{all_numeric_accumulations} not implemented for {pa_type}', raises=TypeError))
        self.check_accumulate(ser, op_name, skipna)

    def _supports_reduction(self, ser: pd.Series, op_name: str) -> bool:
        if op_name in ['kurt', 'skew']:
            return False
        dtype = ser.dtype
        pa_dtype = dtype.pyarrow_dtype
        if pa.types.is_temporal(pa_dtype