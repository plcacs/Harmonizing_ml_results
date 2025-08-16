"""
Tests for DatetimeArray
"""

from __future__ import annotations

from datetime import timedelta
import operator
from typing import Any, Iterator, List, Tuple, Union, cast

import numpy as np
import pytest

from pandas._libs.tslibs import tz_compare

from pandas.core.dtypes.dtypes import DatetimeTZDtype

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import (
    DatetimeArray,
    TimedeltaArray,
)


class TestNonNano:
    @pytest.fixture(params=["s", "ms", "us"])
    def unit(self, request: pytest.FixtureRequest) -> str:
        """Fixture returning parametrized time units"""
        return cast(str, request.param)

    @pytest.fixture
    def dtype(self, unit: str, tz_naive_fixture: Any) -> Union[np.dtype, DatetimeTZDtype]:
        tz = tz_naive_fixture
        if tz is None:
            return np.dtype(f"datetime64[{unit}]")
        else:
            return DatetimeTZDtype(unit=unit, tz=tz)

    @pytest.fixture
    def dta_dti(self, unit: str, dtype: Union[np.dtype, DatetimeTZDtype]) -> Tuple[DatetimeArray, pd.DatetimeIndex]:
        tz = getattr(dtype, "tz", None)

        dti = pd.date_range("2016-01-01", periods=55, freq="D", tz=tz)
        if tz is None:
            arr = np.asarray(dti).astype(f"M8[{unit}]")
        else:
            arr = np.asarray(dti.tz_convert("UTC").tz_localize(None)).astype(
                f"M8[{unit}]"
            )

        dta = DatetimeArray._simple_new(arr, dtype=dtype)
        return dta, dti

    @pytest.fixture
    def dta(self, dta_dti: Tuple[DatetimeArray, pd.DatetimeIndex]) -> DatetimeArray:
        dta, dti = dta_dti
        return dta

    def test_non_nano(self, unit: str, dtype: Union[np.dtype, DatetimeTZDtype]) -> None:
        arr = np.arange(5, dtype=np.int64).view(f"M8[{unit}]")
        dta = DatetimeArray._simple_new(arr, dtype=dtype)

        assert dta.dtype == dtype
        assert dta[0].unit == unit
        assert tz_compare(dta.tz, dta[0].tz)
        assert (dta[0] == dta[:1]).all()

    @pytest.mark.parametrize(
        "field", DatetimeArray._field_ops + DatetimeArray._bool_ops
    )
    def test_fields(self, unit: str, field: str, dtype: Union[np.dtype, DatetimeTZDtype], dta_dti: Tuple[DatetimeArray, pd.DatetimeIndex]) -> None:
        dta, dti = dta_dti

        assert (dti == dta).all()

        res = getattr(dta, field)
        expected = getattr(dti._data, field)
        tm.assert_numpy_array_equal(res, expected)

    def test_normalize(self, unit: str) -> None:
        dti = pd.date_range("2016-01-01 06:00:00", periods=55, freq="D")
        arr = np.asarray(dti).astype(f"M8[{unit}]")

        dta = DatetimeArray._simple_new(arr, dtype=arr.dtype)

        assert not dta.is_normalized

        # TODO: simplify once we can just .astype to other unit
        exp = np.asarray(dti.normalize()).astype(f"M8[{unit}]")
        expected = DatetimeArray._simple_new(exp, dtype=exp.dtype)

        res = dta.normalize()
        tm.assert_extension_array_equal(res, expected)

    def test_simple_new_requires_match(self, unit: str) -> None:
        arr = np.arange(5, dtype=np.int64).view(f"M8[{unit}]")
        dtype = DatetimeTZDtype(unit, "UTC")

        dta = DatetimeArray._simple_new(arr, dtype=dtype)
        assert dta.dtype == dtype

        wrong = DatetimeTZDtype("ns", "UTC")
        with pytest.raises(AssertionError, match=""):
            DatetimeArray._simple_new(arr, dtype=wrong)

    def test_std_non_nano(self, unit: str) -> None:
        dti = pd.date_range("2016-01-01", periods=55, freq="D")
        arr = np.asarray(dti).astype(f"M8[{unit}]")

        dta = DatetimeArray._simple_new(arr, dtype=arr.dtype)

        # we should match the nano-reso std, but floored to our reso.
        res = dta.std()
        assert res._creso == dta._creso
        assert res == dti.std().floor(unit)

    @pytest.mark.filterwarnings("ignore:Converting to PeriodArray.*:UserWarning")
    def test_to_period(self, dta_dti: Tuple[DatetimeArray, pd.DatetimeIndex]) -> None:
        dta, dti = dta_dti
        result = dta.to_period("D")
        expected = dti._data.to_period("D")

        tm.assert_extension_array_equal(result, expected)

    def test_iter(self, dta: DatetimeArray) -> None:
        res = next(iter(dta))
        expected = dta[0]

        assert type(res) is pd.Timestamp
        assert res._value == expected._value
        assert res._creso == expected._creso
        assert res == expected

    def test_astype_object(self, dta: DatetimeArray) -> None:
        result = dta.astype(object)
        assert all(x._creso == dta._creso for x in result)
        assert all(x == y for x, y in zip(result, dta))

    def test_to_pydatetime(self, dta_dti: Tuple[DatetimeArray, pd.DatetimeIndex]) -> None:
        dta, dti = dta_dti

        result = dta.to_pydatetime()
        expected = dti.to_pydatetime()
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("meth", ["time", "timetz", "date"])
    def test_time_date(self, dta_dti: Tuple[DatetimeArray, pd.DatetimeIndex], meth: str) -> None:
        dta, dti = dta_dti

        result = getattr(dta, meth)
        expected = getattr(dti, meth)
        tm.assert_numpy_array_equal(result, expected)

    def test_format_native_types(self, unit: str, dtype: Union[np.dtype, DatetimeTZDtype], dta_dti: Tuple[DatetimeArray, pd.DatetimeIndex]) -> None:
        # In this case we should get the same formatted values with our nano
        #  version dti._data as we do with the non-nano dta
        dta, dti = dta_dti

        res = dta._format_native_types()
        exp = dti._data._format_native_types()
        tm.assert_numpy_array_equal(res, exp)

    def test_repr(self, dta_dti: Tuple[DatetimeArray, pd.DatetimeIndex], unit: str) -> None:
        dta, dti = dta_dti

        assert repr(dta) == repr(dti._data).replace("[ns", f"[{unit}")

    # TODO: tests with td64
    def test_compare_mismatched_resolutions(self, comparison_op: Any) -> None:
        # comparison that numpy gets wrong bc of silent overflows
        op = comparison_op

        iinfo = np.iinfo(np.int64)
        vals = np.array([iinfo.min, iinfo.min + 1, iinfo.max], dtype=np.int64)

        # Construct so that arr2[1] < arr[1] < arr[2] < arr2[2]
        arr = np.array(vals).view("M8[ns]")
        arr2 = arr.view("M8[s]")

        left = DatetimeArray._simple_new(arr, dtype=arr.dtype)
        right = DatetimeArray._simple_new(arr2, dtype=arr2.dtype)

        if comparison_op is operator.eq:
            expected = np.array([False, False, False])
        elif comparison_op is operator.ne:
            expected = np.array([True, True, True])
        elif comparison_op in [operator.lt, operator.le]:
            expected = np.array([False, False, True])
        else:
            expected = np.array([False, True, False])

        result = op(left, right)
        tm.assert_numpy_array_equal(result, expected)

        result = op(left[1], right)
        tm.assert_numpy_array_equal(result, expected)

        if op not in [operator.eq, operator.ne]:
            # check that numpy still gets this wrong; if it is fixed we may be
            #  able to remove compare_mismatched_resolutions
            np_res = op(left._ndarray, right._ndarray)
            tm.assert_numpy_array_equal(np_res[1:], ~expected[1:])

    def test_add_mismatched_reso_doesnt_downcast(self) -> None:
        # https://github.com/pandas-dev/pandas/pull/48748#issuecomment-1260181008
        td = pd.Timedelta(microseconds=1)
        dti = pd.date_range("2016-01-01", periods=3) - td
        dta = dti._data.as_unit("us")

        res = dta + td.as_unit("us")
        # even though the result is an even number of days
        #  (so we _could_ downcast to unit="s"), we do not.
        assert res.unit == "us"

    @pytest.mark.parametrize(
        "scalar",
        [
            timedelta(hours=2),
            pd.Timedelta(hours=2),
            np.timedelta64(2, "h"),
            np.timedelta64(2 * 3600 * 1000, "ms"),
            pd.offsets.Minute(120),
            pd.offsets.Hour(2),
        ],
    )
    def test_add_timedeltalike_scalar_mismatched_reso(self, dta_dti: Tuple[DatetimeArray, pd.DatetimeIndex], scalar: Any) -> None:
        dta, dti = dta_dti

        td = pd.Timedelta(scalar)
        exp_unit = tm.get_finest_unit(dta.unit, td.unit)

        expected = (dti + td)._data.as_unit(exp_unit)
        result = dta + scalar
        tm.assert_extension_array_equal(result, expected)

        result = scalar + dta
        tm.assert_extension_array_equal(result, expected)

        expected = (dti - td)._data.as_unit(exp_unit)
        result = dta - scalar
        tm.assert_extension_array_equal(result, expected)

    def test_sub_datetimelike_scalar_mismatch(self) -> None:
        dti = pd.date_range("2016-01-01", periods=3)
        dta = dti._data.as_unit("us")

        ts = dta[0].as_unit("s")

        result = dta - ts
        expected = (dti - dti[0])._data.as_unit("us")
        assert result.dtype == "m8[us]"
        tm.assert_extension_array_equal(result, expected)

    def test_sub_datetime64_reso_mismatch(self) -> None:
        dti = pd.date_range("2016-01-01", periods=3)
        left = dti._data.as_unit("s")
        right = left.as_unit("ms")

        result = left - right
        exp_values = np.array([0, 0, 0], dtype="m8[ms]")
        expected = TimedeltaArray._simple_new(
            exp_values,
            dtype=exp_values.dtype,
        )
        tm.assert_extension_array_equal(result, expected)
        result2 = right - left
        tm.assert_extension_array_equal(result2, expected)


class TestDatetimeArrayComparisons:
    # TODO: merge this into tests/arithmetic/test_datetime64 once it is
    #  sufficiently robust

    def test_cmp_dt64_arraylike_tznaive(self, comparison_op: Any) -> None:
        # arbitrary tz-naive DatetimeIndex
        op = comparison_op

        dti = pd.date_range("2016-01-1", freq="MS", periods=9, tz=None)
        arr = dti._data
        assert arr.freq == dti.freq
        assert arr.tz == dti.tz

        right = dti

        expected = np.ones(len(arr), dtype=bool)
        if comparison_op.__name__ in ["ne", "gt", "lt"]:
            # for these the comparisons should be all-False
            expected = ~expected

        result = op(arr, arr)
        tm.assert_numpy_array_equal(result, expected)
        for other in [
            right,
            np.array(right),
            list(right),
            tuple(right),
            right.astype(object),
        ]:
            result = op(arr, other)
            tm.assert_numpy_array_equal(result, expected)

            result = op(other, arr)
            tm.assert_numpy_array_equal(result, expected)


class TestDatetimeArray:
    def test_astype_ns_to_ms_near_bounds(self) -> None:
        # GH#55979
        ts = pd.Timestamp("1677-09-21 00:12:43.145225")
        target = ts.as_unit("ms")

        dta = DatetimeArray._from_sequence([ts], dtype="M8[ns]")
        assert (dta.view("i8") == ts.as_unit("ns").value).all()

        result = dta.astype("M8[ms]")
        assert result[0] == target

        expected = DatetimeArray._from_sequence([ts], dtype="M8[ms]")
        assert (expected.view("i8") == target._value).all()

        tm.assert_datetime_array_equal(result, expected)

    def test_astype_non_nano_tznaive(self) -> None:
        dti = pd.date_range("2016-01-01", periods=3)

        res = dti.astype("M8[s]")
        assert res.dtype == "M8[s]"

        dta = dti._data
        res = dta.astype("M8[s]")
        assert res.dtype == "M8[s]"
        assert isinstance(res, pd.core.arrays.DatetimeArray)  # used to be ndarray

    def test_astype_non_nano_tzaware(self) -> None:
        dti = pd.date_range("2016-01-01", periods=3, tz="UTC")

        res = dti.astype("M8[s, US/Pacific]")
        assert res.dtype == "M8[s, US/Pacific]"

        dta = dti._data
        res = dta.astype("M8[s, US/Pacific]")
        assert res.dtype == "M8[s, US/Pacific]"

        # from non-nano to non-nano, preserving reso
        res2 = res.astype("M8[s, UTC]")
        assert res2.dtype == "M8[s, UTC]"
        assert not tm.shares_memory(res2, res)

        res3 = res.astype("M8[s, UTC]", copy=False)
        assert res2.dtype == "M8[s, UTC]"
        assert tm.shares_memory(res3, res)

    def test_astype_to_same(self) -> None:
        arr = DatetimeArray._from_sequence(
            ["2000"], dtype=DatetimeTZDtype(tz="US/Central")
        )
        result = arr.astype(DatetimeTZDtype(tz="US/Central"), copy=False)
        assert result is arr

    @pytest.mark.parametrize("dtype", ["datetime64[ns]", "datetime64[ns, UTC]"])
    @pytest.mark.parametrize(
        "other", ["datetime64[ns]", "datetime64[ns, UTC]", "datetime64[ns, CET]"]
    )
    def test_astype_copies(self, dtype: str, other: str) -> None:
        # https://github.com/pandas-dev/pandas/pull/32490
        ser = pd.Series([1, 2], dtype=dtype)
        orig = ser.copy()

        err = False
        if (dtype == "datetime64[ns]") ^ (other == "datetime64[ns]"):
            # deprecated in favor of tz_localize
            err = True

        if err:
            if dtype == "datetime64[ns]":
                msg = "Use obj.tz_localize instead or series.dt.tz_localize instead"
            else:
                msg = "from timezone-aware dtype to timezone-naive dtype"
            with pytest.raises(TypeError, match=msg):
                ser.astype(other)
        else:
            t = ser.astype(other)
            t[:] = pd.NaT
            tm.assert_series_equal(ser, orig)

    @pytest.mark.parametrize("dtype", [int, np.int32, np.int64, "uint32", "uint64"])
    def test_astype_int(self, dtype: Any) -> None:
        arr = DatetimeArray._from_sequence(
            [pd.Timestamp("2000"), pd.Timestamp("2001")], dtype="M8[ns]"
        )

        if np.dtype(dtype) != np.int64:
            with pytest.raises(TypeError, match=r"Do obj.astype\('int64'\)"):
                arr.astype(dtype)
            return

        result = arr.astype(dtype)
        expected = arr._ndarray.view("i8")
        tm.assert