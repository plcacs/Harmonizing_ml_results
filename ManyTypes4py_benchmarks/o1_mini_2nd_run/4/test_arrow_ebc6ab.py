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
from __future__ import annotations
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from io import BytesIO, StringIO
import operator
import pickle
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import timezones
from pandas.compat import (
    PY311,
    PY312,
    is_ci_environment,
    is_platform_windows,
    pa_version_under11p0,
    pa_version_under13p0,
    pa_version_under14p0,
)
from pandas.core.dtypes.dtypes import ArrowDtype, CategoricalDtypeType
import pandas as pd
import pandas._testing as tm
from pandas.api.extensions import no_default
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_signed_integer_dtype,
    is_string_dtype,
    is_unsigned_integer_dtype,
)
from pandas.tests.extension import base
pa = pytest.importorskip("pyarrow")
from pandas.core.arrays.arrow.array import ArrowExtensionArray, get_unit_from_pa_dtype
from pandas.core.arrays.arrow.extension_types import ArrowPeriodType

def _require_timezone_database(request: pytest.FixtureRequest) -> None:
    if is_platform_windows() and is_ci_environment():
        mark = pytest.mark.xfail(
            raises=pa.ArrowInvalid,
            reason="TODO: Set ARROW_TIMEZONE_DATABASE environment variable on CI to path to the tzdata for pyarrow.",
        )
        request.applymarker(mark)

@pytest.fixture(params=tm.ALL_PYARROW_DTYPES, ids=str)
def dtype(request: pytest.FixtureRequest) -> ArrowDtype:
    return ArrowDtype(pyarrow_dtype=request.param)

@pytest.fixture
def data(dtype: ArrowDtype) -> ArrowExtensionArray:
    pa_dtype = dtype.pyarrow_dtype
    if pa.types.is_boolean(pa_dtype):
        data: List[Optional[bool]] = [True, False] * 4 + [None] + [True, False] * 44 + [None] + [True, False]
    elif pa.types.is_floating(pa_dtype):
        data = [1.0, 0.0] * 4 + [None] + [-2.0, -1.0] * 44 + [None] + [0.5, 99.5]
    elif pa.types.is_signed_integer(pa_dtype):
        data = [1, 0] * 4 + [None] + [-2, -1] * 44 + [None] + [1, 99]
    elif pa.types.is_unsigned_integer(pa_dtype):
        data = [1, 0] * 4 + [None] + [2, 1] * 44 + [None] + [1, 99]
    elif pa.types.is_decimal(pa_dtype):
        data = [
            Decimal("1"),
            Decimal("0.0"),
        ] * 4 + [None] + [Decimal("-2.0"), Decimal("-1.0")] * 44 + [None] + [Decimal("0.5"), Decimal("33.123")]
    elif pa.types.is_date(pa_dtype):
        data = [date(2022, 1, 1), date(1999, 12, 31)] * 4 + [None] + [date(2022, 1, 1), date(2022, 1, 1)] * 44 + [None] + [date(1999, 12, 31), date(1999, 12, 31)]
    elif pa.types.is_timestamp(pa_dtype):
        data = [
            datetime(2020, 1, 1, 1, 1, 1, 1),
            datetime(1999, 1, 1, 1, 1, 1, 1),
        ] * 4 + [None] + [datetime(2020, 1, 1, 1), datetime(1999, 1, 1, 1)] * 44 + [None] + [datetime(2020, 1, 1), datetime(1999, 1, 1)]
    elif pa.types.is_duration(pa_dtype):
        data = [
            timedelta(1),
            timedelta(1, 1),
        ] * 4 + [None] + [timedelta(-1), timedelta(0)] * 44 + [None] + [timedelta(-10), timedelta(10)]
    elif pa.types.is_time(pa_dtype):
        data = [time(12, 0), time(0, 12)] * 4 + [None] + [time(0, 0), time(1, 1)] * 44 + [None] + [time(0, 5), time(5, 0)]
    elif pa.types.is_string(pa_dtype):
        data = ["a", "b"] * 4 + [None] + ["1", "2"] * 44 + [None] + ["!", ">"]
    elif pa.types.is_binary(pa_dtype):
        data = [b"a", b"b"] * 4 + [None] + [b"1", b"2"] * 44 + [None] + [b"!", b">"]
    else:
        raise NotImplementedError
    return pd.array(data, dtype=dtype)

@pytest.fixture
def data_missing(data: ArrowExtensionArray) -> ArrowExtensionArray:
    """Length-2 array with [NA, Valid]"""
    return type(data)._from_sequence([None, data[0]], dtype=data.dtype)

@pytest.fixture(params=["data", "data_missing"])
def all_data(
    request: pytest.FixtureRequest,
    data: ArrowExtensionArray,
    data_missing: ArrowExtensionArray,
) -> ArrowExtensionArray:
    """Parametrized fixture returning 'data' or 'data_missing' integer arrays.

    Used to test dtype conversion with and without missing values.
    """
    if request.param == "data":
        return data
    elif request.param == "data_missing":
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
        A = "a"
        B = "b"
        C = "c"
    elif pa.types.is_binary(pa_dtype):
        A = b"a"
        B = b"b"
        C = b"c"
    elif pa.types.is_decimal(pa_dtype):
        A = Decimal("-1.1")
        B = Decimal("0.0")
        C = Decimal("1.1")
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
    return type(data_for_grouping)._from_sequence(
        [data_for_grouping[0], data_for_grouping[7], data_for_grouping[4]], dtype=data_for_grouping.dtype
    )

@pytest.fixture
def data_missing_for_sorting(data_for_grouping: ArrowExtensionArray) -> ArrowExtensionArray:
    """
    Length-3 array with a known sort order.

    This should be three items [B, NA, A] with
    A < B and NA missing.
    """
    return type(data_for_grouping)._from_sequence(
        [data_for_grouping[0], data_for_grouping[2], data_for_grouping[4]], dtype=data_for_grouping.dtype
    )

@pytest.fixture
def data_for_twos(data: ArrowExtensionArray) -> ArrowExtensionArray:
    """Length-100 array in which all the elements are two."""
    pa_dtype = data.dtype.pyarrow_dtype
    if pa.types.is_integer(pa_dtype) or pa.types.is_floating(pa_dtype) or pa.types.is_decimal(pa_dtype) or pa.types.is_duration(pa_dtype):
        return pd.array([2] * 100, dtype=data.dtype)
    return data

class TestArrowArray(base.ExtensionTests):

    def test_compare_scalar(self, data: ArrowExtensionArray, comparison_op: Callable[[Any, Any], Any]) -> None:
        ser = pd.Series(data)
        self._compare_other(ser, data, comparison_op, data[0])

    @pytest.mark.parametrize("na_action", [None, "ignore"])
    def test_map(self, data_missing: ArrowExtensionArray, na_action: Optional[str]) -> None:
        if data_missing.dtype.kind in "mM":
            result = data_missing.map(lambda x: x, na_action=na_action)
            expected = data_missing.to_numpy(dtype=object)
            tm.assert_numpy_array_equal(result, expected)
        else:
            result = data_missing.map(lambda x: x, na_action=na_action)
            if data_missing.dtype == "float32[pyarrow]":
                expected = data_missing.to_numpy(dtype="float64", na_value=np.nan)
            else:
                expected = data_missing.to_numpy()
            tm.assert_numpy_array_equal(result, expected)

    def test_astype_str(
        self, data: ArrowExtensionArray, request: pytest.FixtureRequest, using_infer_string: bool
    ) -> None:
        pa_dtype = data.dtype.pyarrow_dtype
        if pa.types.is_binary(pa_dtype):
            request.applymarker(pytest.mark.xfail(reason=f"For {pa_dtype} .astype(str) decodes."))
        elif (
            not using_infer_string
            and (
                pa.types.is_timestamp(pa_dtype) and pa_dtype.tz is None
                or pa.types.is_duration(pa_dtype)
            )
        ):
            request.applymarker(pytest.mark.xfail(reason="pd.Timestamp/pd.Timedelta repr different from numpy repr"))
        super().test_astype_str(data)

    def test_from_dtype(self, data: ArrowExtensionArray, request: pytest.FixtureRequest) -> None:
        pa_dtype = data.dtype.pyarrow_dtype
        if pa.types.is_string(pa_dtype) or pa.types.is_decimal(pa_dtype):
            if pa.types.is_string(pa_dtype):
                reason = "ArrowDtype(pa.string()) != StringDtype('pyarrow')"
            else:
                reason = f"pyarrow.type_for_alias cannot infer {pa_dtype}"
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
        with pytest.raises(NotImplementedError, match="Converting strings to"):
            ArrowExtensionArray._from_sequence_of_strings(["12-1"], dtype=dtype)

    def test_from_sequence_of_strings_pa_array(self, data: ArrowExtensionArray, request: pytest.FixtureRequest) -> None:
        pa_dtype = data.dtype.pyarrow_dtype
        if pa.types.is_time64(pa_dtype) and pa_dtype.equals("time64[ns]") and (not PY311):
            request.applymarker(pytest.mark.xfail(reason="Nanosecond time parsing not supported."))
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
                int_type = "int32[pyarrow]"
            else:
                int_type = "int64[pyarrow]"
            ser = ser.astype(int_type)
            result = result.astype(int_type)
        result = result.astype("Float64")
        expected = getattr(ser.astype("Float64"), op_name)(skipna=skipna)
        tm.assert_series_equal(result, expected, check_dtype=False)

    def _supports_accumulation(self, ser: pd.Series, op_name: str) -> bool:
        pa_type = ser.dtype.pyarrow_dtype
        if pa.types.is_binary(pa_type) or pa.types.is_decimal(pa_type):
            if op_name in ["cumsum", "cumprod", "cummax", "cummin"]:
                return False
        elif pa.types.is_string(pa_type):
            if op_name == "cumprod":
                return False
        elif pa.types.is_boolean(pa_type):
            if op_name in ["cumprod", "cummax", "cummin"]:
                return False
        elif pa.types.is_temporal(pa_type):
            if op_name == "cumsum" and (not pa.types.is_duration(pa_type)):
                return False
            elif op_name == "cumprod":
                return False
        return True

    @pytest.mark.parametrize("skipna", [True, False])
    def test_accumulate_series(
        self,
        data: ArrowExtensionArray,
        all_numeric_accumulations: str,
        skipna: bool,
        request: pytest.FixtureRequest,
    ) -> None:
        pa_type = data.dtype.pyarrow_dtype
        op_name = all_numeric_accumulations
        if pa.types.is_string(pa_type) and op_name in ["cumsum", "cummin", "cummax"]:
            return
        ser = pd.Series(data)
        if not self._supports_accumulation(ser, op_name):
            return super().test_accumulate_series(data, all_numeric_accumulations, skipna)
        if pa_version_under13p0 and all_numeric_accumulations != "cumsum":
            opt = request.config.option
            if opt.markexpr and "not slow" in opt.markexpr:
                pytest.skip(f"{all_numeric_accumulations} not implemented for pyarrow < 9")
            mark = pytest.mark.xfail(reason=f"{all_numeric_accumulations} not implemented for pyarrow < 9")
            request.applymarker(mark)
        elif all_numeric_accumulations == "cumsum" and (
            pa.types.is_boolean(pa_type) or pa.types.is_decimal(pa_type)
        ):
            request.applymarker(pytest.mark.xfail(reason=f"{all_numeric_accumulations} not implemented for {pa_type}", raises=TypeError))
        self.check_accumulate(ser, op_name, skipna)

    def _supports_reduction(self, ser: pd.Series, op_name: str) -> bool:
        if op_name in ["kurt", "skew"]:
            return False
        dtype = ser.dtype
        pa_dtype = dtype.pyarrow_dtype
        if pa.types.is_temporal(pa_dtype) and op_name in ["sum", "var", "prod"]:
            if pa.types.is_duration(pa_dtype) and op_name in ["sum"]:
                pass
            else:
                return False
        elif pa.types.is_binary(pa_dtype) and op_name == "sum":
            return False
        elif (pa.types.is_string(pa_dtype) or pa.types.is_binary(pa_dtype)) and op_name in ["mean", "median", "prod", "std", "sem", "var"]:
            return False
        if pa.types.is_temporal(pa_dtype) and (not pa.types.is_duration(pa_dtype)) and (op_name in ["any", "all"]):
            return False
        if pa.types.is_boolean(pa_dtype) and op_name in ["median", "std", "var", "skew", "kurt", "sem"]:
            return False
        return True

    def check_reduce(self, ser: pd.Series, op_name: str, skipna: bool) -> None:
        pa_dtype = ser.dtype.pyarrow_dtype
        if pa.types.is_integer(pa_dtype) or pa.types.is_floating(pa_dtype):
            alt = ser.astype("Float64")
        else:
            alt = ser
        if op_name == "count":
            result = getattr(ser, op_name)()
            expected = getattr(alt, op_name)()
        else:
            result = getattr(ser, op_name)(skipna=skipna)
            expected = getattr(alt, op_name)(skipna=skipna)
        tm.assert_almost_equal(result, expected)

    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_series_boolean(
        self,
        data: ArrowExtensionArray,
        all_boolean_reductions: str,
        skipna: bool,
        na_value: Any,
        request: pytest.FixtureRequest,
    ) -> None:
        pa_dtype = data.dtype.pyarrow_dtype
        xfail_mark = pytest.mark.xfail(
            raises=TypeError,
            reason=f"{all_boolean_reductions} is not implemented in pyarrow={pa.__version__} for {pa_dtype}",
        )
        if pa.types.is_string(pa_dtype) or pa.types.is_binary(pa_dtype):
            request.applymarker(xfail_mark)
        return super().test_reduce_series_boolean(data, all_boolean_reductions, skipna)

    def _get_expected_reduction_dtype(
        self, arr: ArrowExtensionArray, op_name: str, skipna: bool
    ) -> str:
        pa_type = arr._pa_array.type
        if op_name in ["max", "min"]:
            cmp_dtype = arr.dtype
        elif pa.types.is_temporal(pa_type):
            if op_name in ["std", "sem"]:
                if pa.types.is_duration(pa_type):
                    cmp_dtype = arr.dtype
                elif pa.types.is_date(pa_type):
                    cmp_dtype = ArrowDtype(pa.duration("s"))
                elif pa.types.is_time(pa_type):
                    unit = get_unit_from_pa_dtype(pa_type)
                    cmp_dtype = ArrowDtype(pa.duration(unit))
                else:
                    cmp_dtype = ArrowDtype(pa.duration(pa_type.unit))
            else:
                cmp_dtype = arr.dtype
        elif arr.dtype.name == "decimal128(7, 3)[pyarrow]":
            if op_name not in ["median", "var", "std", "sem"]:
                cmp_dtype = arr.dtype
            else:
                cmp_dtype = "float64[pyarrow]"
        elif op_name in ["median", "var", "std", "mean", "skew", "sem"]:
            cmp_dtype = "float64[pyarrow]"
        elif op_name in ["sum", "prod"] and pa.types.is_boolean(pa_type):
            cmp_dtype = "uint64[pyarrow]"
        elif op_name == "sum" and pa.types.is_string(pa_type):
            cmp_dtype = arr.dtype
        else:
            cmp_dtype = {"i": "int64[pyarrow]", "u": "uint64[pyarrow]", "f": "float64[pyarrow]"}[arr.dtype.kind]
        return cmp_dtype

    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_frame(
        self,
        data: ArrowExtensionArray,
        all_numeric_reductions: str,
        skipna: bool,
        request: pytest.FixtureRequest,
    ) -> None:
        op_name = all_numeric_reductions
        if op_name == "skew":
            if data.dtype._is_numeric:
                mark = pytest.mark.xfail(reason="skew not implemented")
                request.applymarker(mark)
        elif op_name in ["std", "sem"] and pa.types.is_date64(data._pa_array.type) and skipna:
            mark = pytest.mark.xfail(reason="Cannot cast")
            request.applymarker(mark)
        return super().test_reduce_frame(data, all_numeric_reductions, skipna)

    @pytest.mark.parametrize("typ", ["int64", "uint64", "float64"])
    def test_median_not_approximate(
        self, typ: str
    ) -> None:
        result = pd.Series([1, 2], dtype=f"{typ}[pyarrow]").median()
        assert result == 1.5

    def test_construct_from_string_own_name(
        self, dtype: ArrowDtype, request: pytest.FixtureRequest
    ) -> None:
        pa_dtype = dtype.pyarrow_dtype
        if pa.types.is_decimal(pa_dtype):
            request.applymarker(
                pytest.mark.xfail(
                    raises=NotImplementedError,
                    reason=f"pyarrow.type_for_alias cannot infer {pa_dtype}",
                )
            )
        if pa.types.is_string(pa_dtype):
            msg = "string\\[pyarrow\\] should be constructed by StringDtype"
            with pytest.raises(TypeError, match=msg):
                dtype.construct_from_string(dtype.name)
            return
        super().test_construct_from_string_own_name(dtype)

    def test_is_dtype_from_name(
        self, dtype: ArrowDtype, request: pytest.FixtureRequest
    ) -> None:
        pa_dtype = dtype.pyarrow_dtype
        if pa.types.is_string(pa_dtype):
            assert not type(dtype).is_dtype(dtype.name)
        else:
            if pa.types.is_decimal(pa_dtype):
                request.applymarker(
                    pytest.mark.xfail(
                        raises=NotImplementedError,
                        reason=f"pyarrow.type_for_alias cannot infer {pa_dtype}",
                    )
                )
            super().test_is_dtype_from_name(dtype)

    def test_construct_from_string_another_type_raises(
        self, dtype: ArrowDtype
    ) -> None:
        msg = "'another_type' must end with '\\[pyarrow\\]'"
        with pytest.raises(TypeError, match=msg):
            type(dtype).construct_from_string("another_type")

    def test_get_common_dtype(self, dtype: ArrowDtype, request: pytest.FixtureRequest) -> None:
        pa_dtype = dtype.pyarrow_dtype
        if pa.types.is_date(pa_dtype) or pa.types.is_time(pa_dtype) or (
            pa.types.is_timestamp(pa_dtype) and pa_dtype.tz is not None
        ) or pa.types.is_binary(pa_dtype) or pa.types.is_decimal(pa_dtype):
            request.applymarker(
                pytest.mark.xfail(
                    reason=f"{pa_dtype} does not have associated numpy dtype findable by find_common_type"
                )
            )
        super().test_get_common_dtype(dtype)

    def test_is_not_string_type(self, dtype: ArrowDtype) -> None:
        pa_dtype = dtype.pyarrow_dtype
        if pa.types.is_string(pa_dtype):
            assert is_string_dtype(dtype)
        else:
            super().test_is_not_string_type(dtype)

    @pytest.mark.xfail(reason="GH 45419: pyarrow.ChunkedArray does not support views.", run=False)
    def test_view(self, data: ArrowExtensionArray) -> None:
        super().test_view(data)

    def test_fillna_no_op_returns_copy(self, data: ArrowExtensionArray) -> None:
        data = data[~data.isna()]
        valid = data[0]
        result = data.fillna(valid)
        assert result is not data
        tm.assert_extension_array_equal(result, data)

    @pytest.mark.xfail(reason="GH 45419: pyarrow.ChunkedArray does not support views.", run=False)
    def test_transpose(self, data: ArrowExtensionArray) -> None:
        super().test_transpose(data)

    @pytest.mark.xfail(reason="GH 45419: pyarrow.ChunkedArray does not support views.", run=False)
    def test_setitem_preserves_views(self, data: ArrowExtensionArray) -> None:
        super().test_setitem_preserves_views(data)

    @pytest.mark.parametrize("dtype_backend", ["pyarrow", no_default])
    @pytest.mark.parametrize("engine", ["c", "python"])
    def test_EA_types(
        self, engine: str, data: ArrowExtensionArray, dtype_backend: Optional[str], request: pytest.FixtureRequest
    ) -> None:
        pa_dtype = data.dtype.pyarrow_dtype
        if pa.types.is_decimal(pa_dtype):
            request.applymarker(
                pytest.mark.xfail(
                    raises=NotImplementedError,
                    reason=f"Parameterized types {pa_dtype} not supported.",
                )
            )
        elif pa.types.is_timestamp(pa_dtype) and pa_dtype.unit in ("us", "ns"):
            request.applymarker(
                pytest.mark.xfail(
                    raises=ValueError,
                    reason="https://github.com/pandas-dev/pandas/issues/49767",
                )
            )
        elif pa.types.is_binary(pa_dtype):
            request.applymarker(pytest.mark.xfail(reason="CSV parsers don't correctly handle binary"))
        df = pd.DataFrame({"with_dtype": pd.Series(data, dtype=str(data.dtype))})
        csv_output: Union[BytesIO, StringIO]
        csv_output = df.to_csv(index=False, na_rep=np.nan)
        if pa.types.is_binary(pa_dtype):
            csv_output = BytesIO(csv_output.encode())
        else:
            csv_output = StringIO(csv_output)
        result = pd.read_csv(
            csv_output, dtype={"with_dtype": str(data.dtype)}, engine=engine, dtype_backend=dtype_backend
        )
        expected = df
        tm.assert_frame_equal(result, expected)

    def test_invert(self, data: ArrowExtensionArray, request: pytest.FixtureRequest) -> None:
        pa_dtype = data.dtype.pyarrow_dtype
        if not (pa.types.is_boolean(pa_dtype) or pa.types.is_integer(pa_dtype) or pa.types.is_string(pa_dtype)):
            request.applymarker(
                pytest.mark.xfail(
                    raises=pa.ArrowNotImplementedError,
                    reason=f"pyarrow.compute.invert does support {pa_dtype}",
                )
            )
        if PY312 and pa.types.is_boolean(pa_dtype):
            with tm.assert_produces_warning(
                DeprecationWarning, match="Bitwise inversion", check_stacklevel=False
            ):
                super().test_invert(data)
        else:
            super().test_invert(data)

    @pytest.mark.parametrize("periods", [1, -2])
    def test_diff(
        self, data: ArrowExtensionArray, periods: int, request: pytest.FixtureRequest
    ) -> None:
        pa_dtype = data.dtype.pyarrow_dtype
        if pa.types.is_unsigned_integer(pa_dtype) and periods == 1:
            request.applymarker(
                pytest.mark.xfail(
                    raises=pa.ArrowInvalid,
                    reason=f"diff with {pa_dtype} and periods={periods} will overflow",
                )
            )
        super().test_diff(data, periods)

    def test_value_counts_returns_pyarrow_int64(self, data: ArrowExtensionArray) -> None:
        data = data[:10]
        result = data.value_counts()
        assert result.dtype == ArrowDtype(pa.int64())

    _combine_le_expected_dtype: str = "bool[pyarrow]"

    def get_op_from_name(self, op_name: str) -> Callable[[Any, Any], Any]:
        short_opname = op_name.strip("_")
        if short_opname == "rtruediv":

            def rtruediv(x: Any, y: Any) -> Any:
                return np.divide(y, x)

            return rtruediv
        elif short_opname == "rfloordiv":
            return lambda x, y: np.floor_divide(y, x)
        return tm.get_op_from_name(op_name)

    def _cast_pointwise_result(
        self, op_name: str, obj: pd.Series, other: pd.Series, pointwise_result: pd.Series
    ) -> pd.Series:
        expected = pointwise_result
        if op_name in ["eq", "ne", "lt", "le", "gt", "ge"]:
            return pointwise_result.astype("boolean[pyarrow]")
        was_frame = False
        if isinstance(expected, pd.DataFrame):
            was_frame = True
            expected_data = expected.iloc[:, 0]
            original_dtype = obj.iloc[:, 0].dtype
        else:
            expected_data = expected
            original_dtype = obj.dtype
        orig_pa_type = original_dtype.pyarrow_dtype
        if not was_frame and isinstance(other, pd.Series):
            if not (
                pa.types.is_floating(orig_pa_type)
                or (pa.types.is_integer(orig_pa_type) and op_name not in ["__truediv__", "__rtruediv__"])
                or pa.types.is_duration(orig_pa_type)
                or pa.types.is_timestamp(orig_pa_type)
                or pa.types.is_date(orig_pa_type)
                or pa.types.is_decimal(orig_pa_type)
            ):
                return expected
        elif not (
            op_name == "__floordiv__"
            and pa.types.is_integer(orig_pa_type)
            or pa.types.is_duration(orig_pa_type)
            or pa.types.is_timestamp(orig_pa_type)
            or pa.types.is_date(orig_pa_type)
            or pa.types.is_decimal(orig_pa_type)
        ):
            return expected
        pa_expected = pa.array(expected_data._values)
        if pa.types.is_duration(pa_expected.type):
            if pa.types.is_date(orig_pa_type):
                if pa.types.is_date64(orig_pa_type):
                    unit = "ms"
                else:
                    unit = "s"
            else:
                unit = orig_pa_type.unit
                if isinstance(other, (datetime, timedelta)) and unit in ["s", "ms"]:
                    unit = "us"
            pa_expected = pa_expected.cast(f"duration[{unit}]")
        elif pa.types.is_decimal(pa_expected.type) and pa.types.is_decimal(orig_pa_type):
            alt = getattr(obj, op_name)(other)
            alt_dtype = tm.get_dtype(alt)
            assert isinstance(alt_dtype, ArrowDtype)
            if op_name == "__pow__" and isinstance(other, Decimal):
                alt_dtype = ArrowDtype(pa.float64())
            elif op_name == "__pow__" and isinstance(other, pd.Series) and (other.dtype == original_dtype):
                alt_dtype = ArrowDtype(pa.float64())
            else:
                assert pa.types.is_decimal(alt_dtype.pyarrow_dtype)
            return expected.astype(alt_dtype)
        else:
            pa_expected = pa_expected.cast(orig_pa_type)
        pd_expected = type(expected_data._values)(pa_expected)
        if was_frame:
            expected = pd.DataFrame(pd_expected, index=expected.index, columns=expected.columns)
        else:
            expected = pd.Series(pd_expected)
        return expected

    def _is_temporal_supported(self, opname: str, pa_dtype: pa.DataType) -> bool:
        return (
            (opname in ("__add__", "__radd__"))
            or (
                opname in ("__truediv__", "__rtruediv__", "__floordiv__", "__rfloordiv__")
                and (not pa_version_under14p0)
            )
            and pa.types.is_duration(pa_dtype)
        ) or (
            (opname in ("__sub__", "__rsub__"))
            and pa.types.is_temporal(pa_dtype)
        )

    def _get_expected_exception(
        self, op_name: str, obj: pd.Series, other: pd.Series
    ) -> Optional[Union[Tuple[Exception, ...], Exception]]:
        if op_name in ("__divmod__", "__rdivmod__"):
            return (NotImplementedError, TypeError)
        dtype = tm.get_dtype(obj)
        pa_dtype = dtype.pyarrow_dtype
        arrow_temporal_supported = self._is_temporal_supported(op_name, pa_dtype)
        if op_name in {"__mod__", "__rmod__"}:
            exc = (NotImplementedError, TypeError)
        elif arrow_temporal_supported:
            exc = None
        elif op_name in ["__add__", "__radd__"] and (
            pa.types.is_string(pa_dtype) or pa.types.is_binary(pa_dtype)
        ):
            exc = None
        elif not (
            pa.types.is_floating(pa_dtype)
            or pa.types.is_integer(pa_dtype)
            or pa.types.is_decimal(pa_dtype)
        ):
            exc = TypeError
        else:
            exc = None
        return exc

    def _get_arith_xfail_marker(self, opname: str, pa_dtype: pa.DataType) -> Optional[pytest.Mark]:
        mark: Optional[pytest.Mark] = None
        arrow_temporal_supported = self._is_temporal_supported(opname, pa_dtype)
        if (
            opname == "__rpow__"
            and (pa.types.is_floating(pa_dtype) or pa.types.is_integer(pa_dtype) or pa.types.is_decimal(pa_dtype))
        ):
            mark = pytest.mark.xfail(
                reason=f"GH#29997: 1**pandas.NA == 1 while 1**pyarrow.NA == NULL for {pa_dtype}"
            )
        elif arrow_temporal_supported and (
            pa.types.is_time(pa_dtype)
            or (
                opname in ("__truediv__", "__rtruediv__", "__floordiv__", "__rfloordiv__")
                and pa.types.is_duration(pa_dtype)
            )
        ):
            mark = pytest.mark.xfail(
                raises=TypeError,
                reason=f"{opname} not supported betweenpd.NA and {pa_dtype} Python scalar",
            )
        elif (
            opname == "__rfloordiv__"
            and (pa.types.is_integer(pa_dtype) or pa.types.is_decimal(pa_dtype))
        ):
            mark = pytest.mark.xfail(
                raises=pa.ArrowInvalid,
                reason="divide by 0",
            )
        elif (
            opname == "__rtruediv__"
            and pa.types.is_decimal(pa_dtype)
        ):
            mark = pytest.mark.xfail(
                raises=pa.ArrowInvalid,
                reason="divide by 0",
            )
        return mark

    def test_arith_series_with_scalar(
        self, data: ArrowExtensionArray, all_arithmetic_operators: str, request: pytest.FixtureRequest
    ) -> None:
        pa_dtype = data.dtype.pyarrow_dtype
        if all_arithmetic_operators == "__rmod__" and pa.types.is_binary(pa_dtype):
            pytest.skip("Skip testing Python string formatting")
        mark = self._get_arith_xfail_marker(all_arithmetic_operators, pa_dtype)
        if mark is not None:
            request.applymarker(mark)
        super().test_arith_series_with_scalar(data, all_arithmetic_operators)

    def test_arith_frame_with_scalar(
        self, data: ArrowExtensionArray, all_arithmetic_operators: str, request: pytest.FixtureRequest
    ) -> None:
        pa_dtype = data.dtype.pyarrow_dtype
        if all_arithmetic_operators == "__rmod__" and (
            pa.types.is_string(pa_dtype) or pa.types.is_binary(pa_dtype)
        ):
            pytest.skip("Skip testing Python string formatting")
        mark = self._get_arith_xfail_marker(all_arithmetic_operators, pa_dtype)
        if mark is not None:
            request.applymarker(mark)
        super().test_arith_frame_with_scalar(data, all_arithmetic_operators)

    def test_arith_series_with_array(
        self, data: ArrowExtensionArray, all_arithmetic_operators: str, request: pytest.FixtureRequest
    ) -> None:
        pa_dtype = data.dtype.pyarrow_dtype
        if all_arithmetic_operators in ("__sub__", "__rsub__") and pa.types.is_unsigned_integer(pa_dtype):
            request.applymarker(
                pytest.mark.xfail(
                    raises=pa.ArrowInvalid,
                    reason=f"Implemented pyarrow.compute.subtract_checked which raises on overflow for {pa_dtype}",
                )
            )
        mark = self._get_arith_xfail_marker(all_arithmetic_operators, pa_dtype)
        if mark is not None:
            request.applymarker(mark)
        op_name = all_arithmetic_operators
        ser = pd.Series(data)
        other = pd.Series(pd.array([ser.iloc[0]] * len(ser), dtype=data.dtype))
        self.check_opname(ser, op_name, other)

    def test_add_series_with_extension_array(
        self, data: ArrowExtensionArray, request: pytest.FixtureRequest
    ) -> None:
        pa_dtype = data.dtype.pyarrow_dtype
        if pa_dtype.equals("int8"):
            request.applymarker(
                pytest.mark.xfail(
                    raises=pa.ArrowInvalid,
                    reason=f"raises on overflow for {pa_dtype}",
                )
            )
        super().test_add_series_with_extension_array(data)

    def test_invalid_other_comp(
        self, data: ArrowExtensionArray, comparison_op: Callable[[Any, Any], Any]
    ) -> None:
        with pytest.raises(
            NotImplementedError, match=".* not implemented for <class 'object'>"
        ):
            comparison_op(data, object())

    @pytest.mark.parametrize("masked_dtype", ["boolean", "Int64", "Float64"])
    def test_comp_masked_numpy(
        self, masked_dtype: str, comparison_op: Callable[[Any, Any], Any]
    ) -> None:
        data = [1, 0, None]
        ser_masked = pd.Series(data, dtype=masked_dtype)
        ser_pa = pd.Series(data, dtype=f"{masked_dtype.lower()}[pyarrow]")
        result = comparison_op(ser_pa, ser_masked)
        if comparison_op in [operator.lt, operator.gt, operator.ne]:
            exp = [False, False, None]
        else:
            exp = [True, True, None]
        expected = pd.Series(exp, dtype=ArrowDtype(pa.bool_()))
        tm.assert_series_equal(result, expected)

class TestLogicalOps:
    """Various Series and DataFrame logical ops methods."""

    def test_kleene_or(self) -> None:
        a = pd.Series([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean[pyarrow]")
        b = pd.Series([True, False, None] * 3, dtype="boolean[pyarrow]")
        result = a | b
        expected = pd.Series([True, True, True, True, False, None, True, None, None], dtype="boolean[pyarrow]")
        tm.assert_series_equal(result, expected)
        result = b | a
        tm.assert_series_equal(result, expected)
        tm.assert_series_equal(
            a,
            pd.Series([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean[pyarrow]"),
        )
        tm.assert_series_equal(
            b,
            pd.Series([True, False, None] * 3, dtype="boolean[pyarrow]"),
        )

    @pytest.mark.parametrize(
        "other, expected",
        [
            (None, [True, None, None]),
            (pd.NA, [True, None, None]),
            (True, [True, True, True]),
            (np.bool_(True), [True, True, True]),
            (False, [True, False, None]),
            (np.bool_(False), [True, False, None]),
        ],
    )
    def test_kleene_or_scalar(
        self, other: Union[bool, None, np.bool_], expected: List[Optional[bool]]
    ) -> None:
        a = pd.Series([True, False, None], dtype="boolean[pyarrow]")
        result = a | other
        expected = pd.Series(expected, dtype="boolean[pyarrow]")
        tm.assert_series_equal(result, expected)
        result = other | a
        tm.assert_series_equal(result, expected)
        tm.assert_series_equal(
            a, pd.Series([True, False, None], dtype="boolean[pyarrow]")
        )

    def test_kleene_and(self) -> None:
        a = pd.Series([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean[pyarrow]")
        b = pd.Series([True, False, None] * 3, dtype="boolean[pyarrow]")
        result = a & b
        expected = pd.Series([True, False, None, False, False, False, None, False, None], dtype="boolean[pyarrow]")
        tm.assert_series_equal(result, expected)
        result = b & a
        tm.assert_series_equal(result, expected)
        tm.assert_series_equal(
            a,
            pd.Series([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean[pyarrow]"),
        )
        tm.assert_series_equal(
            b,
            pd.Series([True, False, None] * 3, dtype="boolean[pyarrow]"),
        )

    @pytest.mark.parametrize(
        "other, expected",
        [
            (None, [None, False, None]),
            (pd.NA, [None, False, None]),
            (True, [True, False, None]),
            (False, [False, False, False]),
            (np.bool_(True), [True, False, None]),
            (np.bool_(False), [False, False, False]),
        ],
    )
    def test_kleene_and_scalar(
        self, other: Union[bool, None, np.bool_], expected: List[Optional[bool]]
    ) -> None:
        a = pd.Series([True, False, None], dtype="boolean[pyarrow]")
        result = a & other
        expected = pd.Series(expected, dtype="boolean[pyarrow]")
        tm.assert_series_equal(result, expected)
        result = other & a
        tm.assert_series_equal(result, expected)
        tm.assert_series_equal(
            a, pd.Series([True, False, None], dtype="boolean[pyarrow]")
        )

    def test_kleene_xor(self) -> None:
        a = pd.Series([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean[pyarrow]")
        b = pd.Series([True, False, None] * 3, dtype="boolean[pyarrow]")
        result = a ^ b
        expected = pd.Series([False, True, None, True, False, None, None, None, None], dtype="boolean[pyarrow]")
        tm.assert_series_equal(result, expected)
        result = b ^ a
        tm.assert_series_equal(result, expected)
        tm.assert_series_equal(
            a,
            pd.Series([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean[pyarrow]"),
        )
        tm.assert_series_equal(
            b,
            pd.Series([True, False, None] * 3, dtype="boolean[pyarrow]"),
        )

    @pytest.mark.parametrize(
        "other, expected",
        [
            (None, [None, None, None]),
            (pd.NA, [None, None, None]),
            (True, [False, True, None]),
            (np.bool_(True), [False, True, None]),
            (np.bool_(False), [True, False, None]),
        ],
    )
    def test_kleene_xor_scalar(
        self, other: Union[bool, None, np.bool_], expected: List[Optional[bool]]
    ) -> None:
        a = pd.Series([True, False, None], dtype="boolean[pyarrow]")
        result = a ^ other
        expected = pd.Series(expected, dtype="boolean[pyarrow]")
        tm.assert_series_equal(result, expected)
        result = other ^ a
        tm.assert_series_equal(result, expected)
        tm.assert_series_equal(
            a, pd.Series([True, False, None], dtype="boolean[pyarrow]")
        )

    @pytest.mark.parametrize(
        "op, exp", [["__and__", True], ["__or__", True], ["__xor__", False]]
    )
    def test_logical_masked_numpy(
        self, op: str, exp: bool
    ) -> None:
        data = [True, False, None]
        ser_masked = pd.Series(data, dtype="boolean")
        ser_pa = pd.Series(data, dtype="boolean[pyarrow]")
        result = getattr(ser_pa, op)(ser_masked)
        expected = pd.Series([exp, False, None], dtype=ArrowDtype(pa.bool_()))
        tm.assert_series_equal(result, expected)

@pytest.mark.parametrize("pa_type", tm.ALL_INT_PYARROW_DTYPES)
def test_bitwise(pa_type: pa.DataType) -> None:
    dtype = ArrowDtype(pa_type)
    left = pd.Series([1, None, 3, 4], dtype=dtype)
    right = pd.Series([None, 3, 5, 4], dtype=dtype)
    result = left | right
    expected = pd.Series([None, None, 3 | 5, 4 | 4], dtype=dtype)
    tm.assert_series_equal(result, expected)
    result = left & right
    expected = pd.Series([None, None, 3 & 5, 4 & 4], dtype=dtype)
    tm.assert_series_equal(result, expected)
    result = left ^ right
    expected = pd.Series([None, None, 3 ^ 5, 4 ^ 4], dtype=dtype)
    tm.assert_series_equal(result, expected)
    result = ~left
    expected = ~left.fillna(0).to_numpy()
    expected = pd.Series(expected, dtype=dtype).mask(left.isna())
    tm.assert_series_equal(result, expected)

def test_arrowdtype_construct_from_string_type_with_unsupported_parameters() -> None:
    with pytest.raises(NotImplementedError, match="Passing pyarrow type"):
        ArrowDtype.construct_from_string("not_a_real_dype[s, tz=UTC][pyarrow]")
    with pytest.raises(NotImplementedError, match="Passing pyarrow type"):
        ArrowDtype.construct_from_string("decimal(7, 2)[pyarrow]")

def test_arrowdtype_construct_from_string_supports_dt64tz() -> None:
    dtype = ArrowDtype.construct_from_string("timestamp[s, tz=UTC][pyarrow]")
    expected = ArrowDtype(pa.timestamp("s", "UTC"))
    assert dtype == expected

def test_arrowdtype_construct_from_string_type_only_one_pyarrow() -> None:
    invalid = "int64[pyarrow]foobar[pyarrow]"
    msg = "Passing pyarrow type specific parameters \\(\\[pyarrow\\]\\) in the string is not supported\\."
    with pytest.raises(NotImplementedError, match=msg):
        pd.Series(range(3), dtype=invalid)

def test_arrow_string_multiplication() -> None:
    binary = pd.Series(["abc", "defg"], dtype=ArrowDtype(pa.string()))
    repeat = pd.Series([2, -2], dtype="int64[pyarrow]")
    result = binary * repeat
    expected = pd.Series(["abcabc", ""], dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)
    reflected_result = repeat * binary
    tm.assert_series_equal(result, reflected_result)

def test_arrow_string_multiplication_scalar_repeat() -> None:
    binary = pd.Series(["abc", "defg"], dtype=ArrowDtype(pa.string()))
    result = binary * 2
    expected = pd.Series(["abcabc", "defgdefg"], dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)
    reflected_result = 2 * binary
    tm.assert_series_equal(reflected_result, expected)

@pytest.mark.parametrize("interpolation", ["linear", "lower", "higher", "nearest", "midpoint"])
@pytest.mark.parametrize("quantile", [0.5, [0.5, 0.5]])
def test_quantile(
    data: ArrowExtensionArray,
    interpolation: str,
    quantile: Union[float, List[float]],
    request: pytest.FixtureRequest,
) -> None:
    pa_dtype = data.dtype.pyarrow_dtype
    data = data.take([0, 0, 0])
    ser = pd.Series(data)
    if pa.types.is_string(pa_dtype) or pa.types.is_binary(pa_dtype) or pa.types.is_boolean(pa_dtype):
        msg = "Function 'quantile' has no kernel matching input types \\(.*\\)"
        with pytest.raises(pa.ArrowNotImplementedError, match=msg):
            ser.quantile(q=quantile, interpolation=interpolation)
        return
    if pa.types.is_integer(pa_dtype) or pa.types.is_floating(pa_dtype) or pa.types.is_decimal(pa_dtype):
        pass
    elif pa.types.is_temporal(data._pa_array.type):
        pass
    else:
        request.applymarker(
            pytest.mark.xfail(
                raises=pa.ArrowNotImplementedError,
                reason=f"quantile not supported by pyarrow for {pa_dtype}",
            )
        )
    data = data.take([0, 0, 0])
    ser = pd.Series(data)
    result = ser.quantile(q=quantile, interpolation=interpolation)
    if pa.types.is_timestamp(pa_dtype) and interpolation not in ["lower", "higher"]:
        if pa_dtype.tz:
            pd_dtype = f"M8[{pa_dtype.unit}, {pa_dtype.tz}]"
        else:
            pd_dtype = f"M8[{pa_dtype.unit}]"
        ser_np = ser.astype(pd_dtype)
        expected = ser_np.quantile(q=quantile, interpolation=interpolation)
        if quantile == 0.5:
            if pa_dtype.unit == "us":
                expected = expected.to_pydatetime(warn=False)
            assert result == expected
        else:
            if pa_dtype.unit == "us":
                expected = expected.dt.floor("us")
            tm.assert_series_equal(result, expected.astype(data.dtype))
        return
    if quantile == 0.5:
        assert result == data[0]
    else:
        expected = pd.Series(data.take([0, 0]), index=[0.5, 0.5])
        if pa.types.is_integer(pa_dtype) or pa.types.is_floating(pa_dtype) or pa.types.is_decimal(pa_dtype):
            expected = expected.astype("float64[pyarrow]")
            result = result.astype("float64[pyarrow]")
        tm.assert_series_equal(result, expected)

@pytest.mark.parametrize(
    "take_idx, exp_idx",
    [
        ([[0, 0, 2, 2, 4, 4], [4, 0]]),
        ([[0, 0, 0, 2, 4, 4], [0]]),
    ],
    ids=["multi_mode", "single_mode"],
)
def test_mode_dropna_true(
    data_for_grouping: ArrowExtensionArray,
    take_idx: List[int],
    exp_idx: List[int],
) -> None:
    data = data_for_grouping.take(take_idx)
    ser = pd.Series(data)
    result = ser.mode(dropna=True)
    expected = pd.Series(data_for_grouping.take(exp_idx))
    tm.assert_series_equal(result, expected)

def test_mode_dropna_false_mode_na(data: ArrowExtensionArray) -> None:
    more_nans = pd.Series([None, None, data[0]], dtype=data.dtype)
    result = more_nans.mode(dropna=False)
    expected = pd.Series([None], dtype=data.dtype)
    tm.assert_series_equal(result, expected)
    expected = pd.Series([data[0], None], dtype=data.dtype)
    result = expected.mode(dropna=False)
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize(
    "arrow_dtype, expected_type",
    [
        (pa.binary(), bytes),
        (pa.binary(16), bytes),
        (pa.large_binary(), bytes),
        (pa.large_string(), str),
        (pa.list_(pa.int64()), list),
        (pa.large_list(pa.int64()), list),
        (pa.map_(pa.string(), pa.int64()), list),
        (pa.struct([("f1", pa.int8()), ("f2", pa.string())]), dict),
        (pa.dictionary(pa.int64(), pa.int64()), CategoricalDtypeType),
    ],
)
def test_arrow_dtype_type(
    arrow_dtype: pa.DataType, expected_type: type
) -> None:
    assert ArrowDtype(arrow_dtype).type == expected_type

def test_is_bool_dtype() -> None:
    data = ArrowExtensionArray(pa.array([True, False, True]))
    assert is_bool_dtype(data)
    assert pd.core.common.is_bool_indexer(data)
    s = pd.Series(range(len(data)))
    result = s[data]
    expected = s[np.asarray(data)]
    tm.assert_series_equal(result, expected)

def test_is_numeric_dtype(data: ArrowExtensionArray) -> None:
    pa_type = data.dtype.pyarrow_dtype
    if pa.types.is_floating(pa_type) or pa.types.is_integer(pa_type) or pa.types.is_decimal(pa_type):
        assert is_numeric_dtype(data)
    else:
        assert not is_numeric_dtype(data)

def test_is_integer_dtype(data: ArrowExtensionArray) -> None:
    pa_type = data.dtype.pyarrow_dtype
    if pa.types.is_integer(pa_type):
        assert is_integer_dtype(data)
    else:
        assert not is_integer_dtype(data)

def test_is_signed_integer_dtype(data: ArrowExtensionArray) -> None:
    pa_type = data.dtype.pyarrow_dtype
    if pa.types.is_signed_integer(pa_type):
        assert is_signed_integer_dtype(data)
    else:
        assert not is_signed_integer_dtype(data)

def test_is_unsigned_integer_dtype(data: ArrowExtensionArray) -> None:
    pa_type = data.dtype.pyarrow_dtype
    if pa.types.is_unsigned_integer(pa_type):
        assert is_unsigned_integer_dtype(data)
    else:
        assert not is_unsigned_integer_dtype(data)

def test_is_datetime64_any_dtype(data: ArrowExtensionArray) -> None:
    pa_type = data.dtype.pyarrow_dtype
    if pa.types.is_timestamp(pa_type) or pa.types.is_date(pa_type):
        assert is_datetime64_any_dtype(data)
    else:
        assert not is_datetime64_any_dtype(data)

def test_is_float_dtype(data: ArrowExtensionArray) -> None:
    pa_type = data.dtype.pyarrow_dtype
    if pa.types.is_floating(pa_type):
        assert is_float_dtype(data)
    else:
        assert not is_float_dtype(data)

def test_pickle_roundtrip(data: ArrowExtensionArray) -> None:
    expected = pd.Series(data)
    expected_sliced = expected.head(2)
    full_pickled = pickle.dumps(expected)
    sliced_pickled = pickle.dumps(expected_sliced)
    assert len(full_pickled) > len(sliced_pickled)
    result = pickle.loads(full_pickled)
    tm.assert_series_equal(result, expected)
    result_sliced = pickle.loads(sliced_pickled)
    tm.assert_series_equal(result_sliced, expected_sliced)

def test_astype_from_non_pyarrow(data: ArrowExtensionArray) -> None:
    pd_array = data._pa_array.to_pandas().array
    result = pd_array.astype(data.dtype)
    assert not isinstance(pd_array.dtype, ArrowDtype)
    assert isinstance(result.dtype, ArrowDtype)
    tm.assert_extension_array_equal(result, data)

def test_astype_float_from_non_pyarrow_str() -> None:
    ser = pd.Series(["1.0"])
    result = ser.astype("float64[pyarrow]")
    expected = pd.Series([1.0], dtype="float64[pyarrow]")
    tm.assert_series_equal(result, expected)

def test_astype_errors_ignore() -> None:
    expected = pd.DataFrame({"col": [17000000]}, dtype="int32[pyarrow]")
    result = expected.astype("float[pyarrow]", errors="ignore")
    tm.assert_frame_equal(result, expected)

def test_to_numpy_with_defaults(data: ArrowExtensionArray) -> None:
    result = data.to_numpy()
    pa_type = data._pa_array.type
    if pa.types.is_duration(pa_type) or pa.types.is_timestamp(pa_type):
        pytest.skip("Tested in test_to_numpy_temporal")
    elif pa.types.is_date(pa_type):
        expected = np.array(list(data))
    else:
        expected = np.array(data._pa_array)
    if data._hasna and (not is_numeric_dtype(data.dtype)):
        expected = expected.astype(object)
        expected[pd.isna(data)] = pd.NA
    tm.assert_numpy_array_equal(result, expected)

def test_to_numpy_int_with_na() -> None:
    data = [1, None]
    arr = pd.array(data, dtype="int64[pyarrow]")
    result = arr.to_numpy()
    expected = np.array([1, np.nan])
    assert isinstance(result[0], float)
    tm.assert_numpy_array_equal(result, expected)

@pytest.mark.parametrize("na_val, exp", [(lib.no_default, np.nan), (1, 1)])
def test_to_numpy_null_array(na_val: Any, exp: Any) -> None:
    arr = pd.array([pd.NA, pd.NA], dtype="null[pyarrow]")
    result = arr.to_numpy(dtype="float64", na_value=na_val)
    expected = np.array([exp] * 2, dtype="float64")
    tm.assert_numpy_array_equal(result, expected)

def test_to_numpy_null_array_no_dtype() -> None:
    arr = pd.array([pd.NA, pd.NA], dtype="null[pyarrow]")
    result = arr.to_numpy(dtype=None)
    expected = np.array([pd.NA] * 2, dtype="object")
    tm.assert_numpy_array_equal(result, expected)

def test_to_numpy_without_dtype() -> None:
    arr = pd.array([True, pd.NA], dtype="boolean[pyarrow]")
    result = arr.to_numpy(na_value=False)
    expected = np.array([True, False], dtype=np.bool_)
    tm.assert_numpy_array_equal(result, expected)
    arr = pd.array([1.0, pd.NA], dtype="float32[pyarrow]")
    result = arr.to_numpy(na_value=0.0)
    expected = np.array([1.0, 0.0], dtype=np.float32)
    tm.assert_numpy_array_equal(result, expected)

def test_setitem_null_slice(data: ArrowExtensionArray) -> None:
    orig = data.copy()
    result = orig.copy()
    result[:] = data[0]
    expected = ArrowExtensionArray._from_sequence([data[0]] * len(data), dtype=data.dtype)
    tm.assert_extension_array_equal(result, expected)
    result = orig.copy()
    result[:] = data[::-1]
    expected = data[::-1]
    tm.assert_extension_array_equal(result, expected)
    result = orig.copy()
    result[:] = data.tolist()
    expected = data
    tm.assert_extension_array_equal(result, expected)

def test_setitem_invalid_dtype(data: ArrowExtensionArray) -> None:
    pa_type = data.dtype.pyarrow_dtype
    if pa.types.is_string(pa_type) or pa.types.is_binary(pa_type):
        fill_value = 123
        err: Union[Type[TypeError], Type[pa.ArrowInvalid]] = TypeError
        msg = "Invalid value '123' for dtype"
    elif pa.types.is_integer(pa_type) or pa.types.is_floating(pa_type) or pa.types.is_boolean(pa_type):
        fill_value = "foo"
        err: Union[Type[pa.ArrowInvalid], Type[TypeError]] = pa.ArrowInvalid
        msg = "Could not convert"
    else:
        fill_value = "foo"
        err = TypeError
        msg = "Invalid value 'foo' for dtype"
    with pytest.raises(err, match=msg):
        data[:] = fill_value

def test_from_arrow_respecting_given_dtype() -> None:
    date_array = pa.array([pd.Timestamp("2019-12-31"), pd.Timestamp("2019-12-31")], type=pa.date32())
    result = date_array.to_pandas(types_mapper={pa.date32(): ArrowDtype(pa.date64())}.get)
    expected = pd.Series(
        [pd.Timestamp("2019-12-31"), pd.Timestamp("2019-12-31")],
        dtype=ArrowDtype(pa.date64()),
    )
    tm.assert_series_equal(result, expected)

def test_from_arrow_respecting_given_dtype_unsafe() -> None:
    array = pa.array([1.5, 2.5], type=pa.float64())
    with pytest.raises(pa.ArrowInvalid, match="Cannot convert"):
        array.to_pandas(types_mapper={pa.float64(): ArrowDtype(pa.int64())}.get)

def test_round() -> None:
    dtype = "float64[pyarrow]"
    ser = pd.Series([0.0, 1.23, 2.56, pd.NA], dtype=dtype)
    result = ser.round(1)
    expected = pd.Series([0.0, 1.2, 2.6, pd.NA], dtype=dtype)
    tm.assert_series_equal(result, expected)
    ser = pd.Series([123.4, pd.NA, 56.78], dtype=dtype)
    result = ser.round(-1)
    expected = pd.Series([120.0, pd.NA, 60.0], dtype=dtype)
    tm.assert_series_equal(result, expected)

def test_searchsorted_with_na_raises(
    data_for_sorting: ArrowExtensionArray, as_series: bool
) -> None:
    b, c, a = data_for_sorting
    arr = data_for_sorting.take([2, 0, 1])
    arr[-1] = pd.NA
    if as_series:
        arr = pd.Series(arr)
    msg = "searchsorted requires array to be sorted, which is impossible with NAs present."
    with pytest.raises(ValueError, match=msg):
        arr.searchsorted(b)

def test_sort_values_dictionary() -> None:
    df = pd.DataFrame(
        {
            "a": pd.Series(["x", "y"], dtype=ArrowDtype(pa.dictionary(pa.int32(), pa.string()))),
            "b": [1, 2],
        }
    )
    expected = df.copy()
    result = df.sort_values(by=["a", "b"])
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize("pat", ["abc", "a[a-z]{2}"])
def test_str_count(pat: str) -> None:
    ser = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.count(pat)
    expected = pd.Series([1, None], dtype=ArrowDtype(pa.int32()))
    tm.assert_series_equal(result, expected)

def test_str_count_flags_unsupported() -> None:
    ser = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    with pytest.raises(NotImplementedError, match="count not"):
        ser.str.count("abc", flags=1)

@pytest.mark.parametrize(
    "side, str_func",
    [
        ("left", "rjust"),
        ("right", "ljust"),
        ("both", "center"),
    ],
)
def test_str_pad(side: str, str_func: str) -> None:
    ser = pd.Series(["a", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.pad(width=3, side=side, fillchar="x")
    expected = pd.Series(
        [
            getattr("a", str_func)(3, "x"),
            None,
        ],
        dtype=ArrowDtype(pa.string()),
    )
    tm.assert_series_equal(result, expected)

def test_str_pad_invalid_side() -> None:
    ser = pd.Series(["a", None], dtype=ArrowDtype(pa.string()))
    with pytest.raises(ValueError, match="Invalid side: foo"):
        ser.str.pad(3, "foo", "x")

@pytest.mark.parametrize(
    "val",
    ["abc", "aaa"],
)
def test_str_removesuffix(val: str) -> None:
    ser = pd.Series([val, None], dtype=ArrowDtype(pa.string()))
    result = ser.str.removesuffix("123")
    expected = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize(
    "val",
    ["123abc", "aaa"],
)
def test_str_removeprefix(val: str) -> None:
    ser = pd.Series([val, None], dtype=ArrowDtype(pa.string()))
    result = ser.str.removeprefix("123")
    expected = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize(
    "interpolation, quantile",
    [
        ("linear", 0.5),
        ("linear", [0.5, 0.5]),
        ("lower", 0.5),
        ("lower", [0.5, 0.5]),
        ("higher", 0.5),
        ("higher", [0.5, 0.5]),
        ("nearest", 0.5),
        ("nearest", [0.5, 0.5]),
        ("midpoint", 0.5),
        ("midpoint", [0.5, 0.5]),
    ],
)
def test_quantile_e2e(
    data: ArrowExtensionArray,
    interpolation: str,
    quantile: Union[float, List[float]],
    request: pytest.FixtureRequest,
) -> None:
    pa_dtype = data.dtype.pyarrow_dtype
    data = data.take([0, 0, 0])
    ser = pd.Series(data)
    if pa.types.is_string(pa_dtype) or pa.types.is_binary(pa_dtype) or pa.types.is_boolean(pa_dtype):
        msg = "Function 'quantile' has no kernel matching input types \\(.*\\)"
        with pytest.raises(pa.ArrowNotImplementedError, match=msg):
            ser.quantile(q=quantile, interpolation=interpolation)
        return
    if pa.types.is_integer(pa_dtype) or pa.types.is_floating(pa_dtype) or pa.types.is_decimal(pa_dtype):
        pass
    elif pa.types.is_temporal(data._pa_array.type):
        pass
    else:
        request.applymarker(
            pytest.mark.xfail(
                raises=pa.ArrowNotImplementedError,
                reason=f"quantile not supported by pyarrow for {pa_dtype}",
            )
        )
    data = data.take([0, 0, 0])
    ser = pd.Series(data)
    result = ser.quantile(q=quantile, interpolation=interpolation)
    if pa.types.is_timestamp(pa_dtype) and interpolation not in ["lower", "higher"]:
        if pa_dtype.tz:
            pd_dtype = f"M8[{pa_dtype.unit}, {pa_dtype.tz}]"
        else:
            pd_dtype = f"M8[{pa_dtype.unit}]"
        ser_np = ser.astype(pd_dtype)
        expected = ser_np.quantile(q=quantile, interpolation=interpolation)
        if quantile == 0.5:
            if pa_dtype.unit == "us":
                expected = expected.to_pydatetime(warn=False)
            assert result == expected
        else:
            if pa_dtype.unit == "us":
                expected = expected.dt.floor("us")
            tm.assert_series_equal(result, expected.astype(data.dtype))
        return
    if quantile == 0.5:
        assert result == data[0]
    else:
        expected = pd.Series(data.take([0, 0]), index=[0.5, 0.5])
        if pa.types.is_integer(pa_dtype) or pa.types.is_floating(pa_dtype) or pa.types.is_decimal(pa_dtype):
            expected = expected.astype("float64[pyarrow]")
            result = result.astype("float64[pyarrow]")
        tm.assert_series_equal(result, expected)

@pytest.mark.parametrize(
    "arrow_dtype, expected_type",
    [
        (pa.binary(), bytes),
        (pa.binary(16), bytes),
        (pa.large_binary(), bytes),
        (pa.large_string(), str),
        (pa.list_(pa.int64()), list),
        (pa.large_list(pa.int64()), list),
        (pa.map_(pa.string(), pa.int64()), list),
        (pa.struct([("f1", pa.int8()), ("f2", pa.string())]), dict),
        (pa.dictionary(pa.int64(), pa.int64()), CategoricalDtypeType),
    ],
)
@pytest.mark.parametrize(
    "take_idx, exp_idx",
    [
        ([0, 0, 2, 2, 4, 4], [4, 0]),
        ([0, 0, 0, 2, 4, 4], [0]),
    ],
    ids=["multi_mode", "single_mode"],
)
def test_mode_dropna_true(
    data_for_grouping: ArrowExtensionArray,
    take_idx: List[int],
    exp_idx: List[int],
) -> None:
    data = data_for_grouping.take(take_idx)
    ser = pd.Series(data)
    result = ser.mode(dropna=True)
    expected = pd.Series(data_for_grouping.take(exp_idx))
    tm.assert_series_equal(result, expected)

def test_mode_dropna_false_mode_na(data: ArrowExtensionArray) -> None:
    more_nans = pd.Series([None, None, data[0]], dtype=data.dtype)
    result = more_nans.mode(dropna=False)
    expected = pd.Series([None], dtype=data.dtype)
    tm.assert_series_equal(result, expected)
    expected = pd.Series([data[0], None], dtype=data.dtype)
    result = expected.mode(dropna=False)
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize(
    "arrow_dtype, expected_type",
    [
        (pa.binary(), bytes),
        (pa.binary(16), bytes),
        (pa.large_binary(), bytes),
        (pa.large_string(), str),
        (pa.list_(pa.int64()), list),
        (pa.large_list(pa.int64()), list),
        (pa.map_(pa.string(), pa.int64()), list),
        (pa.struct([("f1", pa.int8()), ("f2", pa.string())]), dict),
        (pa.dictionary(pa.int64(), pa.int64()), CategoricalDtypeType),
    ],
)
def test_arrow_dtype_type(
    arrow_dtype: pa.DataType, expected_type: type
) -> None:
    assert ArrowDtype(arrow_dtype).type == expected_type

def test_is_bool_dtype() -> None:
    data = ArrowExtensionArray(pa.array([True, False, True]))
    assert is_bool_dtype(data)
    assert pd.core.common.is_bool_indexer(data)
    s = pd.Series(range(len(data)))
    result = s[data]
    expected = s[np.asarray(data)]
    tm.assert_series_equal(result, expected)

def test_is_numeric_dtype(data: ArrowExtensionArray) -> None:
    pa_type = data.dtype.pyarrow_dtype
    if pa.types.is_floating(pa_type) or pa.types.is_integer(pa_type) or pa.types.is_decimal(pa_type):
        assert is_numeric_dtype(data)
    else:
        assert not is_numeric_dtype(data)

def test_is_integer_dtype(data: ArrowExtensionArray) -> None:
    pa_type = data.dtype.pyarrow_dtype
    if pa.types.is_integer(pa_type):
        assert is_integer_dtype(data)
    else:
        assert not is_integer_dtype(data)

def test_is_signed_integer_dtype(data: ArrowExtensionArray) -> None:
    pa_type = data.dtype.pyarrow_dtype
    if pa.types.is_signed_integer(pa_type):
        assert is_signed_integer_dtype(data)
    else:
        assert not is_signed_integer_dtype(data)

def test_is_unsigned_integer_dtype(data: ArrowExtensionArray) -> None:
    pa_type = data.dtype.pyarrow_dtype
    if pa.types.is_unsigned_integer(pa_type):
        assert is_unsigned_integer_dtype(data)
    else:
        assert not is_unsigned_integer_dtype(data)

def test_is_datetime64_any_dtype(data: ArrowExtensionArray) -> None:
    pa_type = data.dtype.pyarrow_dtype
    if pa.types.is_timestamp(pa_type) or pa.types.is_date(pa_type):
        assert is_datetime64_any_dtype(data)
    else:
        assert not is_datetime64_any_dtype(data)

def test_is_float_dtype(data: ArrowExtensionArray) -> None:
    pa_type = data.dtype.pyarrow_dtype
    if pa.types.is_floating(pa_type):
        assert is_float_dtype(data)
    else:
        assert not is_float_dtype(data)

def test_pickle_roundtrip(data: ArrowExtensionArray) -> None:
    expected = pd.Series(data)
    expected_sliced = expected.head(2)
    full_pickled = pickle.dumps(expected)
    sliced_pickled = pickle.dumps(expected_sliced)
    assert len(full_pickled) > len(sliced_pickled)
    result = pickle.loads(full_pickled)
    tm.assert_series_equal(result, expected)
    result_sliced = pickle.loads(sliced_pickled)
    tm.assert_series_equal(result_sliced, expected_sliced)

def test_astype_from_non_pyarrow(data: ArrowExtensionArray) -> None:
    pd_array = data._pa_array.to_pandas().array
    result = pd_array.astype(data.dtype)
    assert not isinstance(pd_array.dtype, ArrowDtype)
    assert isinstance(result.dtype, ArrowDtype)
    tm.assert_extension_array_equal(result, data)

def test_astype_float_from_non_pyarrow_str() -> None:
    ser = pd.Series(["1.0"])
    result = ser.astype("float64[pyarrow]")
    expected = pd.Series([1.0], dtype="float64[pyarrow]")
    tm.assert_series_equal(result, expected)

def test_astype_errors_ignore() -> None:
    expected = pd.DataFrame({"col": [17000000]}, dtype="int32[pyarrow]")
    result = expected.astype("float[pyarrow]", errors="ignore")
    tm.assert_frame_equal(result, expected)

def test_to_numpy_with_defaults(data: ArrowExtensionArray) -> None:
    result = data.to_numpy()
    pa_type = data._pa_array.type
    if pa.types.is_duration(pa_type) or pa.types.is_timestamp(pa_type):
        pytest.skip("Tested in test_to_numpy_temporal")
    elif pa.types.is_date(pa_type):
        expected = np.array(list(data))
    else:
        expected = np.array(data._pa_array)
    if data._hasna and (not is_numeric_dtype(data.dtype)):
        expected = expected.astype(object)
        expected[pd.isna(data)] = pd.NA
    tm.assert_numpy_array_equal(result, expected)

def test_to_numpy_int_with_na() -> None:
    data = [1, None]
    arr = pd.array(data, dtype="int64[pyarrow]")
    result = arr.to_numpy()
    expected = np.array([1, np.nan])
    assert isinstance(result[0], float)
    tm.assert_numpy_array_equal(result, expected)

@pytest.mark.parametrize("na_val, exp", [(lib.no_default, np.nan), (1, 1)])
def test_to_numpy_null_array(na_val: Any, exp: Any) -> None:
    arr = pd.array([pd.NA, pd.NA], dtype="null[pyarrow]")
    result = arr.to_numpy(dtype="float64", na_value=na_val)
    expected = np.array([exp] * 2, dtype="float64")
    tm.assert_numpy_array_equal(result, expected)

def test_to_numpy_null_array_no_dtype() -> None:
    arr = pd.array([pd.NA, pd.NA], dtype="null[pyarrow]")
    result = arr.to_numpy(dtype=None)
    expected = np.array([pd.NA] * 2, dtype="object")
    tm.assert_numpy_array_equal(result, expected)

def test_to_numpy_without_dtype() -> None:
    arr = pd.array([True, pd.NA], dtype="boolean[pyarrow]")
    result = arr.to_numpy(na_value=False)
    expected = np.array([True, False], dtype=np.bool_)
    tm.assert_numpy_array_equal(result, expected)
    arr = pd.array([1.0, pd.NA], dtype="float32[pyarrow]")
    result = arr.to_numpy(na_value=0.0)
    expected = np.array([1.0, 0.0], dtype=np.float32)
    tm.assert_numpy_array_equal(result, expected)

def test_setitem_null_slice(data: ArrowExtensionArray) -> None:
    orig = data.copy()
    result = orig.copy()
    result[:] = data[0]
    expected = ArrowExtensionArray._from_sequence([data[0]] * len(data), dtype=data.dtype)
    tm.assert_extension_array_equal(result, expected)
    result = orig.copy()
    result[:] = data[::-1]
    expected = data[::-1]
    tm.assert_extension_array_equal(result, expected)
    result = orig.copy()
    result[:] = data.tolist()
    expected = data
    tm.assert_extension_array_equal(result, expected)

def test_setitem_invalid_dtype(data: ArrowExtensionArray) -> None:
    pa_type = data.dtype.pyarrow_dtype
    if pa.types.is_string(pa_type) or pa.types.is_binary(pa_type):
        fill_value = 123
        err: Union[Type[TypeError], Type[pa.ArrowInvalid]] = TypeError
        msg = "Invalid value '123' for dtype"
    elif pa.types.is_integer(pa_type) or pa.types.is_floating(pa_type) or pa.types.is_boolean(pa_type):
        fill_value = "foo"
        err = pa.ArrowInvalid
        msg = "Could not convert"
    else:
        fill_value = "foo"
        err = TypeError
        msg = "Invalid value 'foo' for dtype"
    with pytest.raises(err, match=msg):
        data[:] = fill_value

def test_from_arrow_respecting_given_dtype() -> None:
    date_array = pa.array([pd.Timestamp("2019-12-31"), pd.Timestamp("2019-12-31")], type=pa.date32())
    result = date_array.to_pandas(types_mapper={pa.date32(): ArrowDtype(pa.date64())}.get)
    expected = pd.Series(
        [pd.Timestamp("2019-12-31"), pd.Timestamp("2019-12-31")],
        dtype=ArrowDtype(pa.date64()),
    )
    tm.assert_series_equal(result, expected)

def test_from_arrow_respecting_given_dtype_unsafe() -> None:
    array = pa.array([1.5, 2.5], type=pa.float64())
    with pytest.raises(pa.ArrowInvalid, match="Cannot convert"):
        array.to_pandas(types_mapper={pa.float64(): ArrowDtype(pa.int64())}.get)

def test_round() -> None:
    dtype = "float64[pyarrow]"
    ser = pd.Series([0.0, 1.23, 2.56, pd.NA], dtype=dtype)
    result = ser.round(1)
    expected = pd.Series([0.0, 1.2, 2.6, pd.NA], dtype=dtype)
    tm.assert_series_equal(result, expected)
    ser = pd.Series([123.4, pd.NA, 56.78], dtype=dtype)
    result = ser.round(-1)
    expected = pd.Series([120.0, pd.NA, 60.0], dtype=dtype)
    tm.assert_series_equal(result, expected)

def test_searchsorted_with_na_raises(
    data_for_sorting: ArrowExtensionArray, as_series: bool
) -> None:
    b, c, a = data_for_sorting
    arr = data_for_sorting.take([2, 0, 1])
    arr[-1] = pd.NA
    if as_series:
        arr = pd.Series(arr)
    msg = "searchsorted requires array to be sorted, which is impossible with NAs present."
    with pytest.raises(ValueError, match=msg):
        arr.searchsorted(b)

def test_sort_values_dictionary() -> None:
    df = pd.DataFrame(
        {
            "a": pd.Series(
                ["x", "y"], dtype=ArrowDtype(pa.dictionary(pa.int32(), pa.string()))
            ),
            "b": [1, 2],
        }
    )
    expected = df.copy()
    result = df.sort_values(by=["a", "b"])
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize("pat", ["abc", "a[a-z]{2}"])
def test_str_count(pat: str) -> None:
    ser = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.count(pat)
    expected = pd.Series([1, None], dtype=ArrowDtype(pa.int32()))
    tm.assert_series_equal(result, expected)

def test_str_count_flags_unsupported() -> None:
    ser = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    with pytest.raises(NotImplementedError, match="count not"):
        ser.str.count("abc", flags=1)

@pytest.mark.parametrize(
    "side, str_func",
    [
        ("left", "rjust"),
        ("right", "ljust"),
        ("both", "center"),
    ],
)
def test_str_pad(side: str, str_func: str) -> None:
    ser = pd.Series(["a", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.pad(width=3, side=side, fillchar="x")
    expected = pd.Series(
        [
            getattr("a", str_func)(3, "x"),
            None,
        ],
        dtype=ArrowDtype(pa.string()),
    )
    tm.assert_series_equal(result, expected)

def test_str_pad_invalid_side() -> None:
    ser = pd.Series(["a", None], dtype=ArrowDtype(pa.string()))
    with pytest.raises(ValueError, match="Invalid side: foo"):
        ser.str.pad(3, "foo", "x")

@pytest.mark.parametrize(
    "val",
    ["abc", "aaa"],
)
def test_str_removesuffix(val: str) -> None:
    ser = pd.Series([val, None], dtype=ArrowDtype(pa.string()))
    result = ser.str.removesuffix("123")
    expected = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize(
    "val",
    ["123abc", "aaa"],
)
def test_str_removeprefix(val: str) -> None:
    ser = pd.Series([val, None], dtype=ArrowDtype(pa.string()))
    result = ser.str.removeprefix("123")
    expected = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize(
    "interpolation, quantile",
    [
        ("linear", 0.5),
        ("linear", [0.5, 0.5]),
        ("lower", 0.5),
        ("lower", [0.5, 0.5]),
        ("higher", 0.5),
        ("higher", [0.5, 0.5]),
        ("nearest", 0.5),
        ("nearest", [0.5, 0.5]),
        ("midpoint", 0.5),
        ("midpoint", [0.5, 0.5]),
    ],
)
def test_quantile_e2e(
    data: ArrowExtensionArray,
    interpolation: str,
    quantile: Union[float, List[float]],
    request: pytest.FixtureRequest,
) -> None:
    pa_dtype = data.dtype.pyarrow_dtype
    data = data.take([0, 0, 0])
    ser = pd.Series(data)
    if pa.types.is_string(pa_dtype) or pa.types.is_binary(pa_dtype) or pa.types.is_boolean(pa_dtype):
        msg = "Function 'quantile' has no kernel matching input types \\(.*\\)"
        with pytest.raises(pa.ArrowNotImplementedError, match=msg):
            ser.quantile(q=quantile, interpolation=interpolation)
        return
    if pa.types.is_integer(pa_dtype) or pa.types.is_floating(pa_dtype) or pa.types.is_decimal(pa_dtype):
        pass
    elif pa.types.is_temporal(data._pa_array.type):
        pass
    else:
        request.applymarker(
            pytest.mark.xfail(
                raises=pa.ArrowNotImplementedError,
                reason=f"quantile not supported by pyarrow for {pa_dtype}",
            )
        )
    data = data.take([0, 0, 0])
    ser = pd.Series(data)
    result = ser.quantile(q=quantile, interpolation=interpolation)
    if pa.types.is_timestamp(pa_dtype) and interpolation not in ["lower", "higher"]:
        if pa_dtype.tz:
            pd_dtype = f"M8[{pa_dtype.unit}, {pa_dtype.tz}]"
        else:
            pd_dtype = f"M8[{pa_dtype.unit}]"
        ser_np = ser.astype(pd_dtype)
        expected = ser_np.quantile(q=quantile, interpolation=interpolation)
        if quantile == 0.5:
            if pa_dtype.unit == "us":
                expected = expected.to_pydatetime(warn=False)
            assert result == expected
        else:
            if pa_dtype.unit == "us":
                expected = expected.dt.floor("us")
            tm.assert_series_equal(result, expected.astype(data.dtype))
        return
    if quantile == 0.5:
        assert result == data[0]
    else:
        expected = pd.Series(data.take([0, 0]), index=[0.5, 0.5])
        if pa.types.is_integer(pa_dtype) or pa.types.is_floating(pa_dtype) or pa.types.is_decimal(pa_dtype):
            expected = expected.astype("float64[pyarrow]")
            result = result.astype("float64[pyarrow]")
        tm.assert_series_equal(result, expected)

@pytest.mark.parametrize(
    "arrow_dtype, expected_type",
    [
        (pa.binary(), bytes),
        (pa.binary(16), bytes),
        (pa.large_binary(), bytes),
        (pa.large_string(), str),
        (pa.list_(pa.int64()), list),
        (pa.large_list(pa.int64()), list),
        (pa.map_(pa.string(), pa.int64()), list),
        (pa.struct([("f1", pa.int8()), ("f2", pa.string())]), dict),
        (pa.dictionary(pa.int64(), pa.int64()), CategoricalDtypeType),
    ],
)
@pytest.mark.parametrize(
    "take_idx, exp_idx",
    [
        ([0, 0, 2, 2, 4, 4], [4, 0]),
        ([0, 0, 0, 2, 4, 4], [0]),
    ],
    ids=["multi_mode", "single_mode"],
)
def test_mode_dropna_true(
    data_for_grouping: ArrowExtensionArray,
    take_idx: List[int],
    exp_idx: List[int],
) -> None:
    data = data_for_grouping.take(take_idx)
    ser = pd.Series(data)
    result = ser.mode(dropna=True)
    expected = pd.Series(data_for_grouping.take(exp_idx))
    tm.assert_series_equal(result, expected)

def test_mode_dropna_false_mode_na(data: ArrowExtensionArray) -> None:
    more_nans = pd.Series([None, None, data[0]], dtype=data.dtype)
    result = more_nans.mode(dropna=False)
    expected = pd.Series([None], dtype=data.dtype)
    tm.assert_series_equal(result, expected)
    expected = pd.Series([data[0], None], dtype=data.dtype)
    result = expected.mode(dropna=False)
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize(
    "pa_type, expected_type",
    [
        (pa.binary(), bytes),
        (pa.binary(16), bytes),
        (pa.large_binary(), bytes),
        (pa.large_string(), str),
        (pa.list_(pa.int64()), list),
        (pa.large_list(pa.int64()), list),
        (pa.map_(pa.string(), pa.int64()), list),
        (pa.struct([("f1", pa.int8()), ("f2", pa.string())]), dict),
        (pa.dictionary(pa.int64(), pa.int64()), CategoricalDtypeType),
    ],
)
@pytest.mark.parametrize(
    "take_idx, exp_idx",
    [
        ([0, 0, 2, 2, 4, 4], [4, 0]),
        ([0, 0, 0, 2, 4, 4], [0]),
    ],
    ids=["multi_mode", "single_mode"],
)
def test_mode_dropna_true(
    data_for_grouping: ArrowExtensionArray,
    take_idx: List[int],
    exp_idx: List[int],
) -> None:
    data = data_for_grouping.take(take_idx)
    ser = pd.Series(data)
    result = ser.mode(dropna=True)
    expected = pd.Series(data_for_grouping.take(exp_idx))
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize(
    "arrow_dtype, expected_type",
    [
        (pa.binary(), bytes),
        (pa.binary(16), bytes),
        (pa.large_binary(), bytes),
        (pa.large_string(), str),
        (pa.list_(pa.int64()), list),
        (pa.large_list(pa.int64()), list),
        (pa.map_(pa.string(), pa.int64()), list),
        (pa.struct([("f1", pa.int8()), ("f2", pa.string())]), dict),
        (pa.dictionary(pa.int64(), pa.int64()), CategoricalDtypeType),
    ],
)
def test_arrow_dtype_type(
    arrow_dtype: pa.DataType, expected_type: type
) -> None:
    assert ArrowDtype(arrow_dtype).type == expected_type

def test_is_bool_dtype() -> None:
    data = ArrowExtensionArray(pa.array([True, False, True]))
    assert is_bool_dtype(data)
    assert pd.core.common.is_bool_indexer(data)
    s = pd.Series(range(len(data)))
    result = s[data]
    expected = s[np.asarray(data)]
    tm.assert_series_equal(result, expected)

def test_is_numeric_dtype(data: ArrowExtensionArray) -> None:
    pa_type = data.dtype.pyarrow_dtype
    if pa.types.is_floating(pa_type) or pa.types.is_integer(pa_type) or pa.types.is_decimal(pa_type):
        assert is_numeric_dtype(data)
    else:
        assert not is_numeric_dtype(data)

def test_is_integer_dtype(data: ArrowExtensionArray) -> None:
    pa_type = data.dtype.pyarrow_dtype
    if pa.types.is_integer(pa_type):
        assert is_integer_dtype(data)
    else:
        assert not is_integer_dtype(data)

def test_is_signed_integer_dtype(data: ArrowExtensionArray) -> None:
    pa_type = data.dtype.pyarrow_dtype
    if pa.types.is_signed_integer(pa_type):
        assert is_signed_integer_dtype(data)
    else:
        assert not is_signed_integer_dtype(data)

def test_is_unsigned_integer_dtype(data: ArrowExtensionArray) -> None:
    pa_type = data.dtype.pyarrow_dtype
    if pa.types.is_unsigned_integer(pa_type):
        assert is_unsigned_integer_dtype(data)
    else:
        assert not is_unsigned_integer_dtype(data)

def test_is_datetime64_any_dtype(data: ArrowExtensionArray) -> None:
    pa_type = data.dtype.pyarrow_dtype
    if pa.types.is_timestamp(pa_type) or pa.types.is_date(pa_type):
        assert is_datetime64_any_dtype(data)
    else:
        assert not is_datetime64_any_dtype(data)

def test_is_float_dtype(data: ArrowExtensionArray) -> None:
    pa_type = data.dtype.pyarrow_dtype
    if pa.types.is_floating(pa_type):
        assert is_float_dtype(data)
    else:
        assert not is_float_dtype(data)

def test_pickle_roundtrip(data: ArrowExtensionArray) -> None:
    expected = pd.Series(data)
    expected_sliced = expected.head(2)
    full_pickled = pickle.dumps(expected)
    sliced_pickled = pickle.dumps(expected_sliced)
    assert len(full_pickled) > len(sliced_pickled)
    result = pickle.loads(full_pickled)
    tm.assert_series_equal(result, expected)
    result_sliced = pickle.loads(sliced_pickled)
    tm.assert_series_equal(result_sliced, expected_sliced)

def test_astype_from_non_pyarrow(data: ArrowExtensionArray) -> None:
    pd_array = data._pa_array.to_pandas().array
    result = pd_array.astype(data.dtype)
    assert not isinstance(pd_array.dtype, ArrowDtype)
    assert isinstance(result.dtype, ArrowDtype)
    tm.assert_extension_array_equal(result, data)

def test_astype_float_from_non_pyarrow_str() -> None:
    ser = pd.Series(["1.0"])
    result = ser.astype("float64[pyarrow]")
    expected = pd.Series([1.0], dtype="float64[pyarrow]")
    tm.assert_series_equal(result, expected)

def test_astype_errors_ignore() -> None:
    expected = pd.DataFrame({"col": [17000000]}, dtype="int32[pyarrow]")
    result = expected.astype("float[pyarrow]", errors="ignore")
    tm.assert_frame_equal(result, expected)

def test_to_numpy_with_defaults(data: ArrowExtensionArray) -> None:
    result = data.to_numpy()
    pa_type = data._pa_array.type
    if pa.types.is_duration(pa_type) or pa.types.is_timestamp(pa_type):
        pytest.skip("Tested in test_to_numpy_temporal")
    elif pa.types.is_date(pa_type):
        expected = np.array(list(data))
    else:
        expected = np.array(data._pa_array)
    if data._hasna and (not is_numeric_dtype(data.dtype)):
        expected = expected.astype(object)
        expected[pd.isna(data)] = pd.NA
    tm.assert_numpy_array_equal(result, expected)

def test_to_numpy_int_with_na() -> None:
    data = [1, None]
    arr = pd.array(data, dtype="int64[pyarrow]")
    result = arr.to_numpy()
    expected = np.array([1, np.nan])
    assert isinstance(result[0], float)
    tm.assert_numpy_array_equal(result, expected)

@pytest.mark.parametrize("na_val, exp", [(lib.no_default, np.nan), (1, 1)])
def test_to_numpy_null_array(na_val: Any, exp: Any) -> None:
    arr = pd.array([pd.NA, pd.NA], dtype="null[pyarrow]")
    result = arr.to_numpy(dtype="float64", na_value=na_val)
    expected = np.array([exp] * 2, dtype="float64")
    tm.assert_numpy_array_equal(result, expected)

def test_to_numpy_null_array_no_dtype() -> None:
    arr = pd.array([pd.NA, pd.NA], dtype="null[pyarrow]")
    result = arr.to_numpy(dtype=None)
    expected = np.array([pd.NA] * 2, dtype="object")
    tm.assert_numpy_array_equal(result, expected)

def test_to_numpy_without_dtype() -> None:
    arr = pd.array([True, pd.NA], dtype="boolean[pyarrow]")
    result = arr.to_numpy(na_value=False)
    expected = np.array([True, False], dtype=np.bool_)
    tm.assert_numpy_array_equal(result, expected)
    arr = pd.array([1.0, pd.NA], dtype="float32[pyarrow]")
    result = arr.to_numpy(na_value=0.0)
    expected = np.array([1.0, 0.0], dtype=np.float32)
    tm.assert_numpy_array_equal(result, expected)

def test_setitem_null_slice(data: ArrowExtensionArray) -> None:
    orig = data.copy()
    result = orig.copy()
    result[:] = data[0]
    expected = ArrowExtensionArray._from_sequence([data[0]] * len(data), dtype=data.dtype)
    tm.assert_extension_array_equal(result, expected)
    result = orig.copy()
    result[:] = data[::-1]
    expected = data[::-1]
    tm.assert_extension_array_equal(result, expected)
    result = orig.copy()
    result[:] = data.tolist()
    expected = data
    tm.assert_extension_array_equal(result, expected)

def test_setitem_invalid_dtype(data: ArrowExtensionArray) -> None:
    pa_type = data.dtype.pyarrow_dtype
    if pa.types.is_string(pa_type) or pa.types.is_binary(pa_type):
        fill_value = 123
        err: Union[Type[TypeError], Type[pa.ArrowInvalid]] = TypeError
        msg = "Invalid value '123' for dtype"
    elif pa.types.is_integer(pa_type) or pa.types.is_floating(pa_type) or pa.types.is_boolean(pa_type):
        fill_value = "foo"
        err = pa.ArrowInvalid
        msg = "Could not convert"
    else:
        fill_value = "foo"
        err = TypeError
        msg = "Invalid value 'foo' for dtype"
    with pytest.raises(err, match=msg):
        data[:] = fill_value

def test_from_arrow_respecting_given_dtype() -> None:
    date_array = pa.array([pd.Timestamp("2019-12-31"), pd.Timestamp("2019-12-31")], type=pa.date32())
    result = date_array.to_pandas(types_mapper={pa.date32(): ArrowDtype(pa.date64())}.get)
    expected = pd.Series(
        [pd.Timestamp("2019-12-31"), pd.Timestamp("2019-12-31")],
        dtype=ArrowDtype(pa.date64()),
    )
    tm.assert_series_equal(result, expected)

def test_from_arrow_respecting_given_dtype_unsafe() -> None:
    array = pa.array([1.5, 2.5], type=pa.float64())
    with pytest.raises(pa.ArrowInvalid, match="Cannot convert"):
        array.to_pandas(types_mapper={pa.float64(): ArrowDtype(pa.int64())}.get)

def test_round() -> None:
    dtype = "float64[pyarrow]"
    ser = pd.Series([0.0, 1.23, 2.56, pd.NA], dtype=dtype)
    result = ser.round(1)
    expected = pd.Series([0.0, 1.2, 2.6, pd.NA], dtype=dtype)
    tm.assert_series_equal(result, expected)
    ser = pd.Series([123.4, pd.NA, 56.78], dtype=dtype)
    result = ser.round(-1)
    expected = pd.Series([120.0, pd.NA, 60.0], dtype=dtype)
    tm.assert_series_equal(result, expected)

def test_searchsorted_with_na_raises(
    data_for_sorting: ArrowExtensionArray, as_series: bool
) -> None:
    b, c, a = data_for_sorting
    arr = data_for_sorting.take([2, 0, 1])
    arr[-1] = pd.NA
    if as_series:
        arr = pd.Series(arr)
    msg = "searchsorted requires array to be sorted, which is impossible with NAs present."
    with pytest.raises(ValueError, match=msg):
        arr.searchsorted(b)

def test_sort_values_dictionary() -> None:
    df = pd.DataFrame(
        {
            "a": pd.Series(
                ["x", "y"], dtype=ArrowDtype(pa.dictionary(pa.int32(), pa.string()))
            ),
            "b": [1, 2],
        }
    )
    expected = df.copy()
    result = df.sort_values(by=["a", "b"])
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize("pat", ["abc", "a[a-z]{2}"])
def test_str_count(pat: str) -> None:
    ser = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.count(pat)
    expected = pd.Series([1, None], dtype=ArrowDtype(pa.int32()))
    tm.assert_series_equal(result, expected)

def test_str_count_flags_unsupported() -> None:
    ser = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    with pytest.raises(NotImplementedError, match="count not"):
        ser.str.count("abc", flags=1)

@pytest.mark.parametrize(
    "side, pat, na, exp",
    [
        ("startswith", "ab", None, [True, None, False]),
        ("startswith", "b", False, [False, False, False]),
        ("endswith", "b", True, [False, True, False]),
        ("endswith", "bc", None, [True, None, False]),
        ("startswith", ("a", "e", "g"), None, [True, None, True]),
        ("endswith", ("a", "c", "g"), None, [True, None, True]),
        ("startswith", (), None, [False, None, False]),
        ("endswith", (), None, [False, None, False]),
    ],
)
def test_str_start_ends_with(
    side: str, pat: Union[str, Tuple[str, ...]], na: Optional[bool], exp: List[Optional[bool]]
) -> None:
    ser = pd.Series(["abc", None, "efg"], dtype=ArrowDtype(pa.string()))
    result = getattr(ser.str, side)(pat, na=na)
    expected = pd.Series(exp, dtype=ArrowDtype(pa.bool_()))
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize(
    "method, exp",
    [
        ("days_in_month", "Sunday"),
        ("month_name", "January"),
    ],
)
def test_str_transform_functions(method: str, exp: str) -> None:
    ser = pd.Series(["aBc dEF", None], dtype=ArrowDtype(pa.string()))
    result = getattr(ser.str, method)()
    expected = pd.Series([exp, None], dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)

def test_str_transform_functions(method: str, exp: str) -> None:
    ser = pd.Series(["aBc dEF", None], dtype=ArrowDtype(pa.string()))
    result = getattr(ser.str, method)()
    expected = pd.Series([exp, None], dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize(
    "method, exp",
    [
        ("capitalize", "Abc def"),
        ("title", "Abc Def"),
        ("swapcase", "AbC Def"),
        ("lower", "abc def"),
        ("upper", "ABC DEF"),
        ("casefold", "abc def"),
    ],
)
def test_str_transform_functions(method: str, exp: str) -> None:
    ser = pd.Series(["aBc dEF", None], dtype=ArrowDtype(pa.string()))
    result = getattr(ser.str, method)()
    expected = pd.Series([exp, None], dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)

def test_str_transform_functions(method: str, exp: str) -> None:
    ser = pd.Series(["aBc dEF", None], dtype=ArrowDtype(pa.string()))
    result = getattr(ser.str, method)()
    expected = pd.Series([exp, None], dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize("unit", ["ms", "ns"])
def test_duration_from_strings_with_nat(unit: str) -> None:
    strings = ["1000", "NaT"]
    pa_type = pa.duration(unit)
    dtype = ArrowDtype(pa_type)
    result = ArrowExtensionArray._from_sequence_of_strings(strings, dtype=dtype)
    expected = ArrowExtensionArray(pa.array([1000, None], type=pa_type))
    tm.assert_extension_array_equal(result, expected)

def test_unsupported_dt(data: ArrowExtensionArray) -> None:
    pa_dtype = data.dtype.pyarrow_dtype
    if not pa.types.is_temporal(pa_dtype):
        with pytest.raises(AttributeError, match="Can only use .dt accessor with datetimelike values"):
            pd.Series(data).dt

@pytest.mark.parametrize(
    "prop, expected",
    [
        ("year", 2023),
        ("day", 2),
        ("day_of_week", 0),
        ("dayofweek", 0),
        ("weekday", 0),
        ("day_of_year", 2),
        ("dayofyear", 2),
        ("hour", 3),
        ("minute", 4),
        ("is_leap_year", False),
        ("microsecond", 2000),
        ("month", 1),
        ("nanosecond", 6),
        ("quarter", 1),
        ("second", 7),
        ("date", date(2023, 1, 2)),
        ("time", time(3, 4, 7, 2000)),
    ],
)
def test_dt_properties(prop: str, expected: Union[int, bool, date, time, None]) -> None:
    ser = pd.Series(
        [
            pd.Timestamp(year=2023, month=1, day=2, hour=3, minute=4, second=7, microsecond=2000, nanosecond=6),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    result = getattr(ser.dt, prop)
    exp_type: Optional[pa.DataType] = None
    if isinstance(expected, date):
        exp_type = pa.date32()
    elif isinstance(expected, time):
        exp_type = pa.time64("ns")
    expected = pd.Series(
        ArrowExtensionArray(pa.array([expected, None], type=exp_type))
    )
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize("microsecond", [2000, 5, 0])
def test_dt_microsecond(microsecond: int) -> None:
    ser = pd.Series(
        [
            pd.Timestamp(year=2024, month=7, day=7, second=5, microsecond=microsecond, nanosecond=6),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    result = ser.dt.microsecond
    expected = pd.Series([microsecond, None], dtype="int64[pyarrow]")
    tm.assert_series_equal(result, expected)

def test_dt_is_month_start_end() -> None:
    ser = pd.Series(
        [
            datetime(year=2023, month=12, day=2, hour=3),
            datetime(year=2023, month=1, day=1, hour=3),
            datetime(year=2023, month=3, day=31, hour=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("us")),
    )
    result = ser.dt.is_month_start
    expected = pd.Series([False, True, False, None], dtype=ArrowDtype(pa.bool_()))
    tm.assert_series_equal(result, expected)
    result = ser.dt.is_month_end
    expected = pd.Series([False, False, True, None], dtype=ArrowDtype(pa.bool_()))
    tm.assert_series_equal(result, expected)

def test_dt_is_year_start_end() -> None:
    ser = pd.Series(
        [
            datetime(year=2023, month=12, day=31, hour=3),
            datetime(year=2023, month=1, day=1, hour=3),
            datetime(year=2023, month=3, day=31, hour=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("us")),
    )
    result = ser.dt.is_year_start
    expected = pd.Series([False, True, False, None], dtype=ArrowDtype(pa.bool_()))
    tm.assert_series_equal(result, expected)
    result = ser.dt.is_year_end
    expected = pd.Series([True, False, False, None], dtype=ArrowDtype(pa.bool_()))
    tm.assert_series_equal(result, expected)

def test_dt_is_quarter_start_end() -> None:
    ser = pd.Series(
        [
            datetime(year=2023, month=11, day=30, hour=3),
            datetime(year=2023, month=1, day=1, hour=3),
            datetime(year=2023, month=3, day=31, hour=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("us")),
    )
    result = ser.dt.is_quarter_start
    expected = pd.Series([False, True, False, None], dtype=ArrowDtype(pa.bool_()))
    tm.assert_series_equal(result, expected)
    result = ser.dt.is_quarter_end
    expected = pd.Series([False, False, True, None], dtype=ArrowDtype(pa.bool_()))
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize("method, exp", [["days_in_month", "Sunday"], ["month_name", "January"]])
def test_dt_day_month_name(method: str, exp: str, request: pytest.FixtureRequest) -> None:
    _require_timezone_database(request)
    ser = pd.Series([datetime(2023, 1, 1), None], dtype=ArrowDtype(pa.timestamp("ms")))
    result = getattr(ser.dt, method)()
    expected = pd.Series([exp, None], dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)

def test_dt_strftime(request: pytest.FixtureRequest) -> None:
    _require_timezone_database(request)
    ser = pd.Series([datetime(year=2023, month=1, day=2, hour=3), None], dtype=ArrowDtype(pa.timestamp("ns")))
    result = ser.dt.strftime("%Y-%m-%dT%H:%M:%S")
    expected = pd.Series(["2023-01-02T03:00:00.000000000", None], dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize("method", ["ceil", "floor", "round"])
def test_dt_roundlike_tz_options_not_supported(method: str) -> None:
    ser = pd.Series(
        [datetime(year=2023, month=1, day=2, hour=3), None],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    with pytest.raises(NotImplementedError, match="ambiguous is not supported."):
        getattr(ser.dt, method)("1h", ambiguous="NaT")
    with pytest.raises(NotImplementedError, match="nonexistent is not supported."):
        getattr(ser.dt, method)("1h", nonexistent="NaT")

@pytest.mark.parametrize("method", ["ceil", "floor", "round"])
def test_dt_roundlike_unsupported_freq(method: str) -> None:
    ser = pd.Series(
        [datetime(year=2023, month=1, day=2, hour=3), None],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    with pytest.raises(ValueError, match="freq='1B' is not supported"):
        getattr(ser.dt, method)("1B")
    with pytest.raises(ValueError, match="Must specify a valid frequency: None"):
        getattr(ser.dt, method)(None)

@pytest.mark.parametrize("freq", ["D", "h", "min", "s", "ms", "us", "ns"])
@pytest.mark.parametrize("method", ["ceil", "floor", "round"])
def test_dt_ceil_year_floor(freq: str, method: str) -> None:
    ser = pd.Series(
        [datetime(year=2023, month=1, day=2, hour=3), None]
    )
    pa_dtype = ArrowDtype(pa.timestamp("ns"))
    expected = getattr(ser.dt, method)(f"1{freq}").astype(pa_dtype)
    result = getattr(ser.astype(pa_dtype).dt, method)(f"1{freq}")
    tm.assert_series_equal(result, expected)

def test_dt_to_pydatetime() -> None:
    data = [datetime(2022, 1, 1), datetime(2023, 1, 1)]
    ser = pd.Series(data, dtype=ArrowDtype(pa.timestamp("ns")))
    result = ser.dt.to_pydatetime()
    expected = pd.Series(data, dtype=object)
    tm.assert_series_equal(result, expected)
    assert all((type(expected.iloc[i]) is datetime for i in range(len(expected))))
    expected = ser.astype("datetime64[ns]").dt.to_pydatetime()
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize("date_type", [32, 64])
def test_dt_to_pydatetime_date_error(date_type: int) -> None:
    ser = pd.Series(
        [date(2022, 12, 31)],
        dtype=ArrowDtype(getattr(pa, f"date{date_type}")()),
    )
    with pytest.raises(ValueError, match="to_pydatetime cannot be called with"):
        ser.dt.to_pydatetime()

def test_dt_tz_localize_unsupported_tz_options() -> None:
    ser = pd.Series(
        [datetime(year=2023, month=1, day=2, hour=3), None],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    with pytest.raises(TypeError, match="Cannot convert tz-naive timestamps"):
        ser.dt.tz_convert("UTC")

def test_dt_tz_convert_none() -> None:
    ser = pd.Series(
        [
            datetime(year=2023, month=1, day=2, hour=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("ns", tz="US/Pacific")),
    )
    result = ser.dt.tz_convert(None)
    expected = pd.Series(
        [
            datetime(year=2023, month=1, day=2, hour=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize("unit", ["us", "ns"])
def test_dt_tz_convert(unit: str) -> None:
    ser = pd.Series(
        [
            datetime(year=2023, month=1, day=2, hour=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp(unit, "US/Pacific")),
    )
    result = ser.dt.tz_convert("US/Eastern")
    expected = pd.Series(
        [
            datetime(year=2023, month=1, day=2, hour=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp(unit, "US/Eastern")),
    )
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize("unit", ["us", "ns"])
def test_dt_time_preserve_unit(unit: str) -> None:
    ser = pd.Series(
        [
            datetime(year=2023, month=1, day=2, hour=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp(unit)),
    )
    assert ser.dt.unit == unit
    result = ser.dt.time
    expected = pd.Series(
        pa.array(
            [time(3, 0), None],
            type=pa.time64(unit),
        )
    )
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize("unit", ["us", "ns"])
def test_dt_cast_pointwise_result_temporal(
    self, op_name: str, obj: pd.Series, other: pd.Series, pointwise_result: pd.Series
) -> None:
    # Assuming this is part of TestArrowArray
    pass  # Placeholder for the actual implementation

def test_dt_isocalendar() -> None:
    ser = pd.Series(
        [
            pd.Timedelta(days=1, seconds=2, microseconds=3, nanoseconds=4),
            None,
        ],
        dtype=ArrowDtype(pa.duration("ns")),
    )
    result = ser.dt.isocalendar()
    expected = pd.DataFrame(
        [[1, 0, 0], [None, None, None]],
        columns=["days", "hours", "minutes"],
        dtype="int32[pyarrow]",
    )
    tm.assert_frame_equal(result, expected)

def test_dt_normalize() -> None:
    ser = pd.Series(
        [
            datetime(year=2023, month=3, day=30),
            datetime(year=2023, month=4, day=1, hour=3),
            datetime(year=2023, month=2, day=3, hour=23, minute=59, second=59),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    result = ser.dt.normalize()
    expected = pd.Series(
        [
            datetime(year=2023, month=3, day=30),
            datetime(year=2023, month=4, day=1),
            datetime(year=2023, month=2, day=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize("method", ["ceil", "floor", "round"])
def test_dt_roundlike_tz_options_not_supported(method: str) -> None:
    ser = pd.Series(
        [
            datetime(year=2023, month=1, day=2, hour=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    with pytest.raises(NotImplementedError, match="ambiguous is not supported."):
        getattr(ser.dt, method)("1h", ambiguous="NaT")
    with pytest.raises(NotImplementedError, match="nonexistent is not supported."):
        getattr(ser.dt, method)("1h", nonexistent="NaT")

@pytest.mark.parametrize("method", ["ceil", "floor", "round"])
def test_dt_roundlike_unsupported_freq(method: str) -> None:
    ser = pd.Series(
        [
            datetime(year=2023, month=1, day=2, hour=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    with pytest.raises(ValueError, match="freq='1B' is not supported"):
        getattr(ser.dt, method)("1B")
    with pytest.raises(ValueError, match="Must specify a valid frequency: None"):
        getattr(ser.dt, method)(None)

@pytest.mark.parametrize("freq", ["D", "h", "min", "s", "ms", "us", "ns"])
@pytest.mark.parametrize("method", ["ceil", "floor", "round"])
def test_dt_ceil_year_floor(freq: str, method: str) -> None:
    ser = pd.Series(
        [
            datetime(year=2023, month=1, day=2, hour=3),
            None,
        ]
    )
    pa_dtype = ArrowDtype(pa.timestamp("ns"))
    expected = getattr(ser.dt, method)(f"1{freq}").astype(pa_dtype)
    result = getattr(ser.astype(pa_dtype).dt, method)(f"1{freq}")
    tm.assert_series_equal(result, expected)

def test_dt_to_pydatetime() -> None:
    data = [datetime(2022, 1, 1), datetime(2023, 1, 1)]
    ser = pd.Series(data, dtype=ArrowDtype(pa.timestamp("ns")))
    result = ser.dt.to_pydatetime()
    expected = pd.Series(data, dtype=object)
    tm.assert_series_equal(result, expected)
    assert all((type(expected.iloc[i]) is datetime for i in range(len(expected))))
    expected = ser.astype("datetime64[ns]").dt.to_pydatetime()
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize("date_type", [32, 64])
def test_dt_to_pydatetime_date_error(date_type: int) -> None:
    ser = pd.Series(
        [date(2022, 12, 31)],
        dtype=ArrowDtype(getattr(pa, f"date{date_type}")()),
    )
    with pytest.raises(ValueError, match="to_pydatetime cannot be called with"):
        ser.dt.to_pydatetime()

def test_dt_tz_localize_unsupported_tz_options() -> None:
    ser = pd.Series(
        [
            datetime(year=2023, month=1, day=2, hour=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    with pytest.raises(TypeError, match="Cannot convert tz-naive timestamps"):
        ser.dt.tz_convert("UTC")

def test_dt_tz_convert_none() -> None:
    ser = pd.Series(
        [
            datetime(year=2023, month=1, day=2, hour=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("ns", tz="US/Pacific")),
    )
    result = ser.dt.tz_convert(None)
    expected = pd.Series(
        [
            datetime(year=2023, month=1, day=2, hour=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize("unit", ["us", "ns"])
def test_dt_tz_convert(unit: str) -> None:
    ser = pd.Series(
        [
            datetime(year=2023, month=1, day=2, hour=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp(unit, "US/Pacific")),
    )
    result = ser.dt.tz_convert("US/Eastern")
    expected = pd.Series(
        [
            datetime(year=2023, month=1, day=2, hour=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp(unit, "US/Eastern")),
    )
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize("unit", ["us", "ns"])
def test_dt_time_preserve_unit(unit: str) -> None:
    ser = pd.Series(
        [
            datetime(year=2023, month=1, day=2, hour=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp(unit)),
    )
    assert ser.dt.unit == unit
    result = ser.dt.time
    expected = pd.Series(
        pa.array(
            [time(3, 0), None],
            type=pa.time64(unit),
        )
    )
    tm.assert_series_equal(result, expected)

def test_dt_isocalendar() -> None:
    ser = pd.Series(
        [
            pd.Timedelta(days=1, seconds=2, microseconds=3, nanoseconds=4),
            None,
        ],
        dtype=ArrowDtype(pa.duration("ns")),
    )
    result = ser.dt.isocalendar()
    expected = pd.DataFrame(
        [
            [1, 0, 0],
            [None, None, None],
        ],
        columns=["days", "hours", "minutes"],
        dtype="int32[pyarrow]",
    )
    tm.assert_frame_equal(result, expected)

def test_dt_normalize() -> None:
    ser = pd.Series(
        [
            datetime(year=2023, month=3, day=30),
            datetime(year=2023, month=4, day=1, hour=3),
            datetime(year=2023, month=2, day=3, hour=23, minute=59, second=59),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    result = ser.dt.normalize()
    expected = pd.Series(
        [
            datetime(year=2023, month=3, day=30),
            datetime(year=2023, month=4, day=1),
            datetime(year=2023, month=2, day=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize("method", ["ceil", "floor", "round"])
def test_dt_roundlike_tz_options_not_supported(method: str) -> None:
    ser = pd.Series(
        [
            datetime(year=2023, month=1, day=2, hour=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    with pytest.raises(NotImplementedError, match="ambiguous is not supported."):
        getattr(ser.dt, method)("1h", ambiguous="NaT")
    with pytest.raises(NotImplementedError, match="nonexistent is not supported."):
        getattr(ser.dt, method)("1h", nonexistent="NaT")

@pytest.mark.parametrize("method", ["ceil", "floor", "round"])
def test_dt_roundlike_unsupported_freq(method: str) -> None:
    ser = pd.Series(
        [
            datetime(year=2023, month=1, day=2, hour=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    with pytest.raises(ValueError, match="freq='1B' is not supported"):
        getattr(ser.dt, method)("1B")
    with pytest.raises(ValueError, match="Must specify a valid frequency: None"):
        getattr(ser.dt, method)(None)

@pytest.mark.parametrize("freq", ["D", "h", "min", "s", "ms", "us", "ns"])
@pytest.mark.parametrize("method", ["ceil", "floor", "round"])
def test_dt_ceil_year_floor(freq: str, method: str) -> None:
    ser = pd.Series(
        [
            datetime(year=2023, month=1, day=2, hour=3),
            None,
        ]
    )
    pa_dtype = ArrowDtype(pa.timestamp("ns"))
    expected = getattr(ser.dt, method)(f"1{freq}").astype(pa_dtype)
    result = getattr(ser.astype(pa_dtype).dt, method)(f"1{freq}")
    tm.assert_series_equal(result, expected)

def test_dt_to_pydatetime() -> None:
    data = [datetime(2022, 1, 1), datetime(2023, 1, 1)]
    ser = pd.Series(data, dtype=ArrowDtype(pa.timestamp("ns")))
    result = ser.dt.to_pydatetime()
    expected = pd.Series(data, dtype=object)
    tm.assert_series_equal(result, expected)
    assert all((type(expected.iloc[i]) is datetime for i in range(len(expected))))
    expected = ser.astype("datetime64[ns]").dt.to_pydatetime()
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize("date_type", [32, 64])
def test_dt_to_pydatetime_date_error(date_type: int) -> None:
    ser = pd.Series(
        [date(2022, 12, 31)],
        dtype=ArrowDtype(getattr(pa, f"date{date_type}")()),
    )
    with pytest.raises(ValueError, match="to_pydatetime cannot be called with"):
        ser.dt.to_pydatetime()

def test_dt_tz_localize_unsupported_tz_options() -> None:
    ser = pd.Series(
        [
            datetime(year=2023, month=1, day=2, hour=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    with pytest.raises(TypeError, match="Cannot convert tz-naive timestamps"):
        ser.dt.tz_convert("UTC")

def test_dt_tz_convert_none() -> None:
    ser = pd.Series(
        [
            datetime(year=2023, month=1, day=2, hour=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("ns", tz="US/Pacific")),
    )
    result = ser.dt.tz_convert(None)
    expected = pd.Series(
        [
            datetime(year=2023, month=1, day=2, hour=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize("unit", ["us", "ns"])
def test_dt_tz_convert(unit: str) -> None:
    ser = pd.Series(
        [
            datetime(year=2023, month=1, day=2, hour=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp(unit, "US/Pacific")),
    )
    result = ser.dt.tz_convert("US/Eastern")
    expected = pd.Series(
        [
            datetime(year=2023, month=1, day=2, hour=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp(unit, "US/Eastern")),
    )
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize("unit", ["us", "ns"])
def test_dt_time_preserve_unit(unit: str) -> None:
    ser = pd.Series(
        [
            datetime(year=2023, month=1, day=2, hour=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp(unit)),
    )
    assert ser.dt.unit == unit
    result = ser.dt.time
    expected = pd.Series(
        pa.array(
            [time(3, 0), None],
            type=pa.time64(unit),
        )
    )
    tm.assert_series_equal(result, expected)

def test_dt_isocalendar() -> None:
    ser = pd.Series(
        [
            pd.Timedelta(days=1, seconds=2, microseconds=3, nanoseconds=4),
            None,
        ],
        dtype=ArrowDtype(pa.duration("ns")),
    )
    result = ser.dt.isocalendar()
    expected = pd.DataFrame(
        [
            [1, 0, 0],
            [None, None, None],
        ],
        columns=["days", "hours", "minutes"],
        dtype="int32[pyarrow]",
    )
    tm.assert_frame_equal(result, expected)

def test_dt_normalize() -> None:
    ser = pd.Series(
        [
            datetime(year=2023, month=3, day=30),
            datetime(year=2023, month=4, day=1, hour=3),
            datetime(year=2023, month=2, day=3, hour=23, minute=59, second=59),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    result = ser.dt.normalize()
    expected = pd.Series(
        [
            datetime(year=2023, month=3, day=30),
            datetime(year=2023, month=4, day=1),
            datetime(year=2023, month=2, day=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize(
    "method, exp",
    [
        ("ceil", "2023-01-02T03:00:00.000000000"),
        ("ceil", "2023-01-02T03:00:00.000000000"),
        ("ceil", "2023-01-02T03:00:00.000000000"),
    ],
)
def test_dt_cast_pointwise_result_temporal(
    method: str, exp: str
) -> None:
    # Placeholder for the actual test implementation
    pass

def test_dt_isocalendar() -> None:
    ser = pd.Series(
        [
            pd.Timedelta(days=1, seconds=2, microseconds=3, nanoseconds=4),
            None,
        ],
        dtype=ArrowDtype(pa.duration("ns")),
    )
    result = ser.dt.isocalendar()
    expected = pd.DataFrame(
        [
            [1, 0, 0],
            [None, None, None],
        ],
        columns=["days", "hours", "minutes"],
        dtype="int32[pyarrow]",
    )
    tm.assert_frame_equal(result, expected)

def test_dt_normalize() -> None:
    ser = pd.Series(
        [
            datetime(year=2023, month=3, day=30),
            datetime(year=2023, month=4, day=1, hour=3),
            datetime(year=2023, month=2, day=3, hour=23, minute=59, second=59),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    result = ser.dt.normalize()
    expected = pd.Series(
        [
            datetime(year=2023, month=3, day=30),
            datetime(year=2023, month=4, day=1),
            datetime(year=2023, month=2, day=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize("unit", ["us", "ns"])
def test_dt_time_preserve_unit(unit: str) -> None:
    ser = pd.Series(
        [
            datetime(year=2023, month=1, day=2, hour=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp(unit)),
    )
    assert ser.dt.unit == unit
    result = ser.dt.time
    expected = pd.Series(
        pa.array(
            [time(3, 0), None],
            type=pa.time64(unit),
        )
    )
    tm.assert_series_equal(result, expected)

def test_dt_isocalendar() -> None:
    ser = pd.Series(
        [
            pd.Timedelta(days=1, seconds=2, microseconds=3, nanoseconds=4),
            None,
        ],
        dtype=ArrowDtype(pa.duration("ns")),
    )
    result = ser.dt.isocalendar()
    expected = pd.DataFrame(
        [
            [1, 0, 0],
            [None, None, None],
        ],
        columns=["days", "hours", "minutes"],
        dtype="int32[pyarrow]",
    )
    tm.assert_frame_equal(result, expected)

def test_dt_normalize() -> None:
    ser = pd.Series(
        [
            datetime(year=2023, month=3, day=30),
            datetime(year=2023, month=4, day=1, hour=3),
            datetime(year=2023, month=2, day=3, hour=23, minute=59, second=59),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    result = ser.dt.normalize()
    expected = pd.Series(
        [
            datetime(year=2023, month=3, day=30),
            datetime(year=2023, month=4, day=1),
            datetime(year=2023, month=2, day=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize("unit", ["us", "ns"])
def test_dt_time_preserve_unit(unit: str) -> None:
    ser = pd.Series(
        [
            datetime(year=2023, month=1, day=2, hour=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp(unit)),
    )
    assert ser.dt.unit == unit
    result = ser.dt.time
    expected = pd.Series(
        pa.array(
            [time(3, 0), None],
            type=pa.time64(unit),
        )
    )
    tm.assert_series_equal(result, expected)

def test_dt_isocalendar() -> None:
    ser = pd.Series(
        [
            pd.Timedelta(days=1, seconds=2, microseconds=3, nanoseconds=4),
            None,
        ],
        dtype=ArrowDtype(pa.duration("ns")),
    )
    result = ser.dt.isocalendar()
    expected = pd.DataFrame(
        [
            [1, 0, 0],
            [None, None, None],
        ],
        columns=["days", "hours", "minutes"],
        dtype="int32[pyarrow]",
    )
    tm.assert_frame_equal(result, expected)

def test_dt_normalize() -> None:
    ser = pd.Series(
        [
            datetime(year=2023, month=3, day=30),
            datetime(year=2023, month=4, day=1, hour=3),
            datetime(year=2023, month=2, day=3, hour=23, minute=59, second=59),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    result = ser.dt.normalize()
    expected = pd.Series(
        [
            datetime(year=2023, month=3, day=30),
            datetime(year=2023, month=4, day=1),
            datetime(year=2023, month=2, day=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize("unit", ["us", "ns"])
def test_dt_time_preserve_unit(unit: str) -> None:
    ser = pd.Series(
        [
            datetime(year=2023, month=1, day=2, hour=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp(unit)),
    )
    assert ser.dt.unit == unit
    result = ser.dt.time
    expected = pd.Series(
        pa.array(
            [time(3, 0), None],
            type=pa.time64(unit),
        )
    )
    tm.assert_series_equal(result, expected)

def test_dt_isocalendar() -> None:
    ser = pd.Series(
        [
            pd.Timedelta(days=1, seconds=2, microseconds=3, nanoseconds=4),
            None,
        ],
        dtype=ArrowDtype(pa.duration("ns")),
    )
    result = ser.dt.isocalendar()
    expected = pd.DataFrame(
        [
            [1, 0, 0],
            [None, None, None],
        ],
        columns=["days", "hours", "minutes"],
        dtype="int32[pyarrow]",
    )
    tm.assert_frame_equal(result, expected)

def test_dt_normalize() -> None:
    ser = pd.Series(
        [
            datetime(year=2023, month=3, day=30),
            datetime(year=2023, month=4, day=1, hour=3),
            datetime(year=2023, month=2, day=3, hour=23, minute=59, second=59),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    result = ser.dt.normalize()
    expected = pd.Series(
        [
            datetime(year=2023, month=3, day=30),
            datetime(year=2023, month=4, day=1),
            datetime(year=2023, month=2, day=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    tm.assert_series_equal(result, expected)

def test_str_find_negative_start_no_match() -> None:
    ser = pd.Series(["abcdefg", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.find(sub="d", start=-3, end=-6)
    expected = pd.Series([-1, None], dtype=ArrowDtype(pa.int64()))
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize(
    "method, exp",
    [
        ("__add__", 3),
        ("__sub__", -2),
        ("__mul__", 3),
        ("__truediv__", 0.5),
    ],
)
def test_arrow_arithmetic_methods(method: str, exp: Any) -> None:
    dtype = "int64[pyarrow]"
    ser = pd.Series([2, None], dtype=dtype)
    result = getattr(ser, method)(1)
    expected = pd.Series([exp, None], dtype=dtype)
    tm.assert_series_equal(result, expected)

def test_string_to_datetime_parsing_cast() -> None:
    string_dates = ["2020-01-01 04:30:00", "2020-01-02 00:00:00", "2020-01-03 00:00:00"]
    result = pd.Series(string_dates, dtype="timestamp[s][pyarrow]")
    expected = pd.Series(ArrowExtensionArray(pa.array(pd.to_datetime(string_dates), from_pandas=True)))
    tm.assert_series_equal(result, expected)

def test_str_find_negative_start_negative_end_no_match() -> None:
    ser = pd.Series(["abcdefg", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.find(sub="d", start=-3, end=-6)
    expected = pd.Series([-1, None], dtype=ArrowDtype(pa.int64()))
    tm.assert_series_equal(result, expected)

def test_str_getitem_temporal_and_other_methods() -> None:
    # Placeholder for actual test implementation
    pass

@pytest.mark.parametrize("flags", [0, 2])
def test_str_findall(flags: int) -> None:
    ser = pd.Series(["abc", "efg", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.findall("b", flags=flags)
    expected = pd.Series(
        [
            ["b"],
            [],
            None,
        ],
        dtype=ArrowDtype(pa.list_(pa.string())),
    )
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize(
    "method, exp",
    [
        ("index", True),
        ("rindex", True),
    ],
)
@pytest.mark.parametrize(
    "start, stop",
    [
        (0, None),
        (1, 4),
    ],
)
def test_str_r_index(method: str, start: int, stop: int, exp: bool) -> None:
    ser = pd.Series(["abcba", None], dtype=ArrowDtype(pa.string()))
    result = getattr(ser.str, method)("c", start, stop)
    expected = pd.Series([2, None], dtype=ArrowDtype(pa.int64()))
    tm.assert_series_equal(result, expected)
    with pytest.raises(ValueError, match="substring not found"):
        getattr(ser.str, method)("foo", start, stop)
