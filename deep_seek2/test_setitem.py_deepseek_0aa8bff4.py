import contextlib
from datetime import date, datetime
from decimal import Decimal
import os
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pytest

from pandas.compat.numpy import np_version_gte1p24
from pandas.errors import IndexingError

from pandas.core.dtypes.common import is_list_like

from pandas import (
    NA,
    Categorical,
    DataFrame,
    DatetimeIndex,
    Index,
    Interval,
    IntervalIndex,
    MultiIndex,
    NaT,
    Period,
    Series,
    StringDtype,
    Timedelta,
    Timestamp,
    array,
    concat,
    date_range,
    interval_range,
    period_range,
    timedelta_range,
)
import pandas._testing as tm

from pandas.tseries.offsets import BDay


class TestSetitemDT64Values:
    def test_setitem_none_nan(self) -> None:
        series = Series(date_range("1/1/2000", periods=10))
        series[3] = None
        assert series[3] is NaT

        series[3:5] = None
        assert series[4] is NaT

        series[5] = np.nan
        assert series[5] is NaT

        series[5:7] = np.nan
        assert series[6] is NaT

    def test_setitem_multiindex_empty_slice(self) -> None:
        idx = MultiIndex.from_tuples([("a", 1), ("b", 2)])
        result = Series([1, 2], index=idx)
        expected = result.copy()
        result.loc[[]] = 0
        tm.assert_series_equal(result, expected)

    def test_setitem_with_string_index(self) -> None:
        ser = Series([1, 2, 3], index=["Date", "b", "other"], dtype=object)
        ser["Date"] = date.today()
        assert ser.Date == date.today()
        assert ser["Date"] == date.today()

    def test_setitem_tuple_with_datetimetz_values(self) -> None:
        arr = date_range("2017", periods=4, tz="US/Eastern")
        index = [(0, 1), (0, 2), (0, 3), (0, 4)]
        result = Series(arr, index=index)
        expected = result.copy()
        result[(0, 1)] = np.nan
        expected.iloc[0] = np.nan
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("tz", ["US/Eastern", "UTC", "Asia/Tokyo"])
    def test_setitem_with_tz(self, tz: str, indexer_sli: Any) -> None:
        orig = Series(date_range("2016-01-01", freq="h", periods=3, tz=tz))
        assert orig.dtype == f"datetime64[ns, {tz}]"

        exp = Series(
            [
                Timestamp("2016-01-01 00:00", tz=tz),
                Timestamp("2011-01-01 00:00", tz=tz),
                Timestamp("2016-01-01 02:00", tz=tz),
            ],
            dtype=orig.dtype,
        )

        ser = orig.copy()
        indexer_sli(ser)[1] = Timestamp("2011-01-01", tz=tz)
        tm.assert_series_equal(ser, exp)

        vals = Series(
            [Timestamp("2011-01-01", tz=tz), Timestamp("2012-01-01", tz=tz)],
            index=[1, 2],
            dtype=orig.dtype,
        )
        assert vals.dtype == f"datetime64[ns, {tz}]"

        exp = Series(
            [
                Timestamp("2016-01-01 00:00", tz=tz),
                Timestamp("2011-01-01 00:00", tz=tz),
                Timestamp("2012-01-01 00:00", tz=tz),
            ],
            dtype=orig.dtype,
        )

        ser = orig.copy()
        indexer_sli(ser)[[1, 2]] = vals
        tm.assert_series_equal(ser, exp)

    def test_setitem_with_tz_dst(self, indexer_sli: Any) -> None:
        tz = "US/Eastern"
        orig = Series(date_range("2016-11-06", freq="h", periods=3, tz=tz))
        assert orig.dtype == f"datetime64[ns, {tz}]"

        exp = Series(
            [
                Timestamp("2016-11-06 00:00-04:00", tz=tz),
                Timestamp("2011-01-01 00:00-05:00", tz=tz),
                Timestamp("2016-11-06 01:00-05:00", tz=tz),
            ],
            dtype=orig.dtype,
        )

        ser = orig.copy()
        indexer_sli(ser)[1] = Timestamp("2011-01-01", tz=tz)
        tm.assert_series_equal(ser, exp)

        vals = Series(
            [Timestamp("2011-01-01", tz=tz), Timestamp("2012-01-01", tz=tz)],
            index=[1, 2],
            dtype=orig.dtype,
        )
        assert vals.dtype == f"datetime64[ns, {tz}]"

        exp = Series(
            [
                Timestamp("2016-11-06 00:00", tz=tz),
                Timestamp("2011-01-01 00:00", tz=tz),
                Timestamp("2012-01-01 00:00", tz=tz),
            ],
            dtype=orig.dtype,
        )

        ser = orig.copy()
        indexer_sli(ser)[[1, 2]] = vals
        tm.assert_series_equal(ser, exp)

    def test_object_series_setitem_dt64array_exact_match(self) -> None:
        ser = Series({"X": np.nan}, dtype=object)
        indexer = [True]
        value = np.array([4], dtype="M8[ns]")
        ser.iloc[indexer] = value
        expected = Series([value[0]], index=["X"], dtype=object)
        assert all(isinstance(x, np.datetime64) for x in expected.values)
        tm.assert_series_equal(ser, expected)


class TestSetitemScalarIndexer:
    def test_setitem_negative_out_of_bounds(self) -> None:
        ser = Series(["a"] * 10, index=["a"] * 10)
        ser[-11] = "foo"
        exp = Series(["a"] * 10 + ["foo"], index=["a"] * 10 + [-11])
        tm.assert_series_equal(ser, exp)

    @pytest.mark.parametrize("indexer", [tm.loc, tm.at])
    @pytest.mark.parametrize("ser_index", [0, 1])
    def test_setitem_series_object_dtype(self, indexer: Any, ser_index: int) -> None:
        ser = Series([0, 0], dtype="object")
        idxr = indexer(ser)
        idxr[0] = Series([42], index=[ser_index])
        expected = Series([Series([42], index=[ser_index]), 0], dtype="object")
        tm.assert_series_equal(ser, expected)

    @pytest.mark.parametrize("index, exp_value", [(0, 42), (1, np.nan)])
    def test_setitem_series(self, index: int, exp_value: Any) -> None:
        ser = Series([0, 0])
        ser.loc[0] = Series([42], index=[index])
        expected = Series([exp_value, 0])
        tm.assert_series_equal(ser, expected)


class TestSetitemSlices:
    def test_setitem_slice_float_raises(self, datetime_series: Series) -> None:
        msg = (
            "cannot do slice indexing on DatetimeIndex with these indexers "
            r"\[{key}\] of type float"
        )
        with pytest.raises(TypeError, match=msg.format(key=r"4\.0")):
            datetime_series[4.0:10.0] = 0

        with pytest.raises(TypeError, match=msg.format(key=r"4\.5")):
            datetime_series[4.5:10.0] = 0

    def test_setitem_slice(self) -> None:
        ser = Series(range(10), index=list(range(10)))
        ser[-12:] = 0
        assert (ser == 0).all()

        ser[:-12] = 5
        assert (ser == 0).all()

    def test_setitem_slice_integers(self) -> None:
        ser = Series(
            np.random.default_rng(2).standard_normal(8),
            index=[2, 4, 6, 8, 10, 12, 14, 16],
        )
        ser[:4] = 0
        assert (ser[:4] == 0).all()
        assert not (ser[4:] == 0).any()

    def test_setitem_slicestep(self) -> None:
        series = Series(
            np.arange(20, dtype=np.float64), index=np.arange(20, dtype=np.int64)
        )
        series[::2] = 0
        assert (series[::2] == 0).all()

    def test_setitem_multiindex_slice(self, indexer_sli: Any) -> None:
        mi = MultiIndex.from_product(([0, 1], list("abcde")))
        result = Series(np.arange(10, dtype=np.int64), mi)
        indexer_sli(result)[::4] = 100
        expected = Series([100, 1, 2, 3, 100, 5, 6, 7, 100, 9], mi)
        tm.assert_series_equal(result, expected)


class TestSetitemBooleanMask:
    def test_setitem_mask_cast(self) -> None:
        ser = Series([1, 2], index=[1, 2], dtype="int64")
        ser[[True, False]] = Series([0], index=[1], dtype="int64")
        expected = Series([0, 2], index=[1, 2], dtype="int64")
        tm.assert_series_equal(ser, expected)

    def test_setitem_mask_align_and_promote(self) -> None:
        ts = Series(
            np.random.default_rng(2).standard_normal(100), index=np.arange(100, 0, -1)
        ).round(5)
        mask = ts > 0
        left = ts.copy()
        right = ts[mask].copy().map(str)
        with pytest.raises(TypeError, match="Invalid value"):
            left[mask] = right

    def test_setitem_mask_promote_strs(self) -> None:
        ser = Series([0, 1, 2, 0])
        mask = ser > 0
        ser2 = ser[mask].map(str)
        with pytest.raises(TypeError, match="Invalid value"):
            ser[mask] = ser2

    def test_setitem_mask_promote(self) -> None:
        ser = Series([0, "foo", "bar", 0])
        mask = Series([False, True, True, False])
        ser2 = ser[mask]
        ser[mask] = ser2
        expected = Series([0, "foo", "bar", 0])
        tm.assert_series_equal(ser, expected)

    def test_setitem_boolean(self, string_series: Series) -> None:
        mask = string_series > string_series.median()
        result = string_series.copy()
        result[mask] = string_series * 2
        expected = string_series * 2
        tm.assert_series_equal(result[mask], expected[mask])

        result = string_series.copy()
        result[mask] = (string_series * 2)[0:5]
        expected = (string_series * 2)[0:5].reindex_like(string_series)
        expected[-mask] = string_series[mask]
        tm.assert_series_equal(result[mask], expected[mask])

    def test_setitem_boolean_corner(self, datetime_series: Series) -> None:
        ts = datetime_series
        mask_shifted = ts.shift(1, freq=BDay()) > ts.median()
        msg = (
            r"Unalignable boolean Series provided as indexer \(index of "
            r"the boolean Series and of the indexed object do not match"
        )
        with pytest.raises(IndexingError, match=msg):
            ts[mask_shifted] = 1

        with pytest.raises(IndexingError, match=msg):
            ts.loc[mask_shifted] = 1

    def test_setitem_boolean_different_order(self, string_series: Series) -> None:
        ordered = string_series.sort_values()
        copy = string_series.copy()
        copy[ordered > 0] = 0
        expected = string_series.copy()
        expected[expected > 0] = 0
        tm.assert_series_equal(copy, expected)

    @pytest.mark.parametrize("func", [list, np.array, Series])
    def test_setitem_boolean_python_list(self, func: Any) -> None:
        ser = Series([None, "b", None])
        mask = func([True, False, True])
        ser[mask] = ["a", "c"]
        expected = Series(["a", "b", "c"])
        tm.assert_series_equal(ser, expected)

    def test_setitem_boolean_nullable_int_types(self, any_numeric_ea_dtype: Any) -> None:
        ser = Series([5, 6, 7, 8], dtype=any_numeric_ea_dtype)
        ser[ser > 6] = Series(range(4), dtype=any_numeric_ea_dtype)
        expected = Series([5, 6, 2, 3], dtype=any_numeric_ea_dtype)
        tm.assert_series_equal(ser, expected)

        ser = Series([5, 6, 7, 8], dtype=any_numeric_ea_dtype)
        ser.loc[ser > 6] = Series(range(4), dtype=any_numeric_ea_dtype)
        tm.assert_series_equal(ser, expected)

        ser = Series([5, 6, 7, 8], dtype=any_numeric_ea_dtype)
        loc_ser = Series(range(4), dtype=any_numeric_ea_dtype)
        ser.loc[ser > 6] = loc_ser.loc[loc_ser > 1]
        tm.assert_series_equal(ser, expected)

    def test_setitem_with_bool_mask_and_values_matching_n_trues_in_length(self) -> None:
        ser = Series([None] * 10)
        mask = [False] * 3 + [True] * 5 + [False] * 2
        ser[mask] = range(5)
        result = ser
        expected = Series([None] * 3 + list(range(5)) + [None] * 2, dtype=object)
        tm.assert_series_equal(result, expected)

    def test_setitem_nan_with_bool(self) -> None:
        result = Series([True, False, True])
        with pytest.raises(TypeError, match="Invalid value"):
            result[0] = np.nan

    def test_setitem_mask_smallint_upcast(self) -> None:
        orig = Series([1, 2, 3], dtype="int8")
        alt = np.array([999, 1000, 1001], dtype=np.int64)
        mask = np.array([True, False, True])
        ser = orig.copy()
        with pytest.raises(TypeError, match="Invalid value"):
            ser[mask] = Series(alt)

        with pytest.raises(TypeError, match="Invalid value"):
            ser.mask(mask, alt, inplace=True)

        res = ser.where(~mask, Series(alt))
        expected = Series([999, 2, 1001])
        tm.assert_series_equal(res, expected)

    def test_setitem_mask_smallint_no_upcast(self) -> None:
        orig = Series([1, 2, 3], dtype="uint8")
        alt = Series([245, 1000, 246], dtype=np.int64)
        mask = np.array([True, False, True])
        ser = orig.copy()
        ser[mask] = alt
        expected = Series([245, 2, 246], dtype="uint8")
        tm.assert_series_equal(ser, expected)

        ser2 = orig.copy()
        ser2.mask(mask, alt, inplace=True)
        tm.assert_series_equal(ser2, expected)

        ser3 = orig.copy()
        res = ser3.where(~mask, alt)
        tm.assert_series_equal(res, expected, check_dtype=False)


class TestSetitemViewCopySemantics:
    def test_setitem_invalidates_datetime_index_freq(self) -> None:
        dti = date_range("20130101", periods=3, tz="US/Eastern")
        ts = dti[1]
        ser = Series(dti)
        assert ser._values is not dti
        assert ser._values._ndarray.base is dti._data._ndarray.base
        assert dti.freq == "D"
        ser.iloc[1] = NaT
        assert ser._values.freq is None
        assert ser._values is not dti
        assert ser._values._ndarray.base is not dti._data._ndarray.base
        assert dti[1] == ts
        assert dti.freq == "D"

    def test_dt64tz_setitem_does_not_mutate_dti(self) -> None:
        dti = date_range("2016-01-01", periods=10, tz="US/Pacific")
        ts = dti[0]
        ser = Series(dti)
        assert ser._values is not dti
        assert ser._values._ndarray.base is dti._data._ndarray.base
        assert ser._mgr.blocks[0].values._ndarray.base is dti._data._ndarray.base
        assert ser._mgr.blocks[0].values is not dti
        ser[::3] = NaT
        assert ser[0] is NaT
        assert dti[0] == ts


class TestSetitemCallable:
    def test_setitem_callable_key(self) -> None:
        ser = Series([1, 2, 3, 4], index=list("ABCD"))
        ser[lambda x: "A"] = -1
        expected = Series([-1, 2, 3, 4], index=list("ABCD"))
        tm.assert_series_equal(ser, expected)

    def test_setitem_callable_other(self) -> None:
        inc = lambda x: x + 1
        ser = Series([1, 2, -1, 4], dtype=object)
        ser[ser < 0] = inc
        expected = Series([1, 2, inc, 4])
        tm.assert_series_equal(ser, expected)


class TestSetitemWithExpansion:
    def test_setitem_empty_series(self) -> None:
        key = Timestamp("2012-01-01")
        series = Series(dtype=object)
        series[key] = 47
        expected = Series(47, Index([key], dtype=object))
        tm.assert_series_equal(series, expected)

    def test_setitem_empty_series_datetimeindex_preserves_freq(self) -> None:
        dti = DatetimeIndex([], freq="D", dtype="M8[ns]")
        series = Series([], index=dti, dtype=object)
        key = Timestamp("2012-01-01")
        series[key] = 47
        expected = Series(47, DatetimeIndex([key], freq="D").as_unit("ns"))
        tm.assert_series_equal(series, expected)
        assert series.index.freq == expected.index.freq

    def test_setitem_empty_series_timestamp_preserves_dtype(self) -> None:
        timestamp = Timestamp(1412526600000000000)
        series = Series([timestamp], index=["timestamp"], dtype=object)
        expected = series["timestamp"]
        series = Series([], dtype=object)
        series["anything"] = 300.0
        series["timestamp"] = timestamp
        result = series["timestamp"]
        assert result == expected

    @pytest.mark.parametrize(
        "td",
        [
            Timedelta("9 days"),
            Timedelta("9 days").to_timedelta64(),
            Timedelta("9 days").to_pytimedelta(),
        ],
    )
    def test_append_timedelta_does_not_cast(self, td: Any, using_infer_string: bool, request: Any) -> None:
        if using_infer_string and not isinstance(td, Timedelta):
            request.applymarker(pytest.mark.xfail(reason="inferred as string"))
        expected = Series(["x", td], index=[0, "td"], dtype=object)
        ser = Series(["x"])
        ser["td"] = td
        tm.assert_series_equal(ser, expected)
        assert isinstance(ser["td"], Timedelta)
        ser = Series(["x"])
        ser.loc["td"] = Timedelta("9 days")
        tm.assert_series_equal(ser, expected)
        assert isinstance(ser["td"], Timedelta)

    def test_setitem_with_expansion_type_promotion(self) -> None:
        ser = Series(dtype=object)
        ser["a"] = Timestamp("2016-01-01")
        ser["b"] = 3.0
        ser["c"] = "foo"
        expected = Series(
            [Timestamp("2016-01-01"), 3.0, "foo"],
            index=Index(["a", "b", "c"], dtype=object),
        )
        tm.assert_series_equal(ser, expected)

    def test_setitem_not_contained(self, string_series: Series) -> None:
        ser = string_series.copy()
        assert "foobar" not in ser.index
        ser["foobar"] = 1
        app = Series([1], index=["foobar"], name="series")
        expected = concat([string_series, app])
        tm.assert_series_equal(ser, expected)

    def test_setitem_keep_precision(self, any_numeric_ea_dtype: Any) -> None:
        ser = Series([1, 2], dtype=any_numeric_ea_dtype)
        ser[2] = 10
        expected = Series([1, 2, 10], dtype=any_numeric_ea_dtype)
        tm.assert_series_equal(ser, expected)

    @pytest.mark.parametrize(
        "na, target_na, dtype, target_dtype, indexer, raises",
        [
            (NA, NA, "Int64", "Int64", 1, False),
            (NA, NA, "Int64", "Int64", 2, False),
            (NA, np.nan, "int64", "float64", 1, False),
            (NA, np.nan, "int64", "float64", 2, False),
            (NaT, NaT, "int64", "object", 1, True),
            (NaT, NaT, "int64", "object", 2, False),
            (np.nan, NA, "Int64", "Int64", 1, False),
            (np.nan, NA, "Int64", "Int64", 2, False),
            (np.nan, NA, "Float64", "Float64", 1, False),
            (np.nan, NA, "Float64", "Float64", 2, False),
            (np.nan, np.nan, "int64", "float64", 1, False),
            (np.nan, np.nan, "int64", "float64", 2, False),
        ],
    )
    def test_setitem_enlarge_with_na(
        self, na: Any, target_na: Any, dtype: str, target_dtype: str, indexer: int, raises: bool
    ) -> None:
        ser = Series([1, 2], dtype=dtype)
        if raises:
            with pytest.raises(TypeError, match="Invalid value"):
                ser[indexer] = na
        else:
            ser[indexer] = na
            expected_values = [1, target_na] if indexer == 1 else [1, 2, target_na]
            expected = Series(expected_values, dtype=target_dtype)
            tm.assert_series_equal(ser, expected)

    def test_setitem_enlargement_object_none(self, nulls_fixture: Any, using_infer_string: bool) -> None:
        ser = Series(["a", "b"])
        ser[3] = nulls_fixture
        dtype = (
            "str"
            if using_infer_string and not isinstance(nulls_fixture, Decimal)
            else object
        )
        expected = Series(["a", "b", nulls_fixture], index=[0, 1, 3], dtype=dtype)
        tm.assert_series_equal(ser, expected)
        if using_infer_string:
            ser[3] is np.nan
        else:
            assert ser[3] is nulls_fixture


def test_setitem_scalar_into_readonly_backing_data() -> None:
    array = np.zeros(5)
    array.flags.writeable = False
    series = Series(array, copy=False)
    for n in series.index:
        msg = "assignment destination is read-only"
        with pytest.raises(ValueError, match=msg):
            series[n] = 1
        assert array[n] == 0


def test_setitem_slice_into_readonly_backing_data() -> None:
    array = np.zeros(5)
    array.flags.writeable = False
    series = Series(array, copy=False)
    msg = "assignment destination is read-only"
    with pytest.raises(ValueError, match=msg):
        series[1:3] = 1
    assert not array.any()


def test_setitem_categorical_assigning_ops() -> None:
    orig = Series(Categorical(["b", "b"], categories=["a", "b"]))
    ser = orig.copy()
    ser[:] = "a"
    exp = Series(Categorical(["a", "a"], categories=["a", "b"]))
    tm.assert_series_equal(ser, exp)

    ser = orig.copy()
    ser[1] = "a"
    exp = Series(Categorical(["b", "a"], categories=["a", "b"]))
    tm.assert_series_equal(ser, exp)

    ser = orig.copy()
    ser[ser.index > 0] = "a"
    exp = Series(Categorical(["b", "a"], categories=["a", "b"]))
    tm.assert_series_equal(ser, exp)

    ser = orig.copy()
    ser[[False, True]] = "a"
    exp = Series(Categorical(["b", "a"], categories=["a", "b"]))
    tm.assert_series_equal(ser, exp)

    ser = orig.copy()
    ser.index = ["x", "y"]
    ser["y"] = "a"
    exp = Series(Categorical(["b", "a"], categories=["a", "b"]), index=["x", "y"])
    tm.assert_series_equal(ser, exp)


def test_setitem_nan_into_categorical() -> None:
    ser = Series(Categorical([1, 2, 3]))
    exp = Series(Categorical([1, np.nan, 3], categories=[1, 2, 3]))
    ser[1] = np.nan
    tm.assert_series_equal(ser, exp)


class TestSetitemCasting:
    @pytest.mark.parametrize("unique", [True, False])
    @pytest.mark.parametrize("val", [3, 3.0, "3"], ids=type)
    def test_setitem_non_bool_into_bool(self, val: Any, indexer_sli: Any, unique: bool) -> None:
        ser = Series([True, False])
        if not unique:
            ser.index = [1, 1]
        with pytest.raises(TypeError, match="Invalid value"):
            indexer_sli(ser)[1] = val

    def test_setitem_boolean_array_into_npbool(self) -> None:
        ser = Series([True, False, True])
        values = ser._values
        arr = array([True, False, None])
        ser[:2] = arr[:2]
        assert ser._values is values
        with pytest.raises(TypeError, match="Invalid value"):
            ser[1:] = arr[1:]


class SetitemCastingEquivalents:
    @pytest.fixture
    def is_inplace(self, obj: Series, expected: Series) -> bool:
        return expected.dtype == obj.dtype

    def check_indexer(self, obj: Series, key: Any, expected: Series, val: Any, indexer: Any, is_inplace: bool) -> None:
        orig = obj
        obj = obj.copy()
        arr = obj._values
        indexer(obj)[key] = val
        tm.assert_series_equal(obj, expected)
        self._check_inplace(is_inplace, orig, arr, obj)

    def _check_inplace(self, is_inplace: bool, orig: Series, arr: Any, obj: Series) -> None:
        if is_inplace is None:
            pass
        elif is_inplace:
            if arr.dtype.kind in ["m", "M"]:
                assert arr._ndarray is obj._values._ndarray
            else:
                assert obj._values is arr
        else:
            tm.assert_equal(arr, orig._values)

    def test_int_key(self, obj: Series, key: int, expected: Series, raises: bool, val: Any, indexer_sli: Any, is_inplace: bool) -> None:
        if not isinstance(key, int):
            pytest.skip("Not relevant for int key")
        if raises:
            ctx = pytest.raises(TypeError, match="Invalid value")
        else:
            ctx = contextlib.nullcontext()
        with ctx:
            self.check_indexer(obj, key, expected, val, indexer_sli, is_inplace)
        if indexer_sli is tm.loc:
            with ctx:
                self.check_indexer(obj, key, expected, val, tm.at, is_inplace)
        elif indexer_sli is tm.iloc:
            with ctx:
                self.check_indexer(obj, key, expected, val, tm.iat, is_inplace)
        rng = range(key, key + 1)
        with ctx:
            self.check_indexer(obj, rng, expected, val, indexer_sli, is_inplace)
        if indexer_sli is not tm.loc:
            slc = slice(key, key + 1)
            with ctx:
                self.check_indexer(obj, slc, expected, val, indexer_sli, is_inplace)
        ilkey = [key]
        with ctx:
            self.check_indexer(obj, ilkey, expected, val, indexer_sli, is_inplace)
        indkey = np.array(ilkey)
        with ctx:
            self.check_indexer(obj, indkey, expected, val, indexer_sli, is_inplace)
        genkey = (x for x in [key])
        with ctx:
            self.check_indexer(obj, genkey, expected, val, indexer_sli, is_inplace)

    def test_slice_key(self, obj: Series, key: slice, expected: Series, raises: bool, val: Any, indexer_sli: Any, is_inplace: bool) -> None:
        if not isinstance(key, slice):
            pytest.skip("Not relevant for slice key")
        if raises:
            ctx = pytest.raises(TypeError, match="Invalid value")
        else:
            ctx = contextlib.nullcontext()
        if indexer_sli is not tm.loc:
            with ctx:
                self.check_indexer(obj, key, expected, val, indexer_sli, is_inplace)
        ilkey = list(range(len(obj)))[key]
        with ctx:
            self.check_indexer(obj, ilkey, expected, val, indexer_sli, is_inplace)
        indkey = np.array(ilkey)
        with ctx:
            self.check_indexer(obj, indkey, expected, val, indexer_sli, is_inplace)
        genkey = (x for x in indkey)
        with ctx:
            self.check_indexer(obj, genkey, expected, val, indexer_sli, is_inplace)

    def test_mask_key(self, obj: Series, key: Any, expected: Series, raises: bool, val: Any, indexer_sli: Any) -> None:
        mask = np.zeros(obj.shape, dtype=bool)
        mask[key] = True
        obj = obj.copy()
        if is_list_like(val) and len(val) < mask.sum():
            msg = "boolean index did not match indexed array along dimension"
            with pytest.raises(IndexError, match=msg):
                indexer_sli(obj)[mask] = val
            return
        if raises:
            with pytest.raises(TypeError, match="Invalid value"):
                indexer_sli(obj)[mask] = val
        else:
            indexer_sli(obj)[mask] = val

    def test_series_where(self, obj: Series, key: Any, expected: Series, raises: bool, val: Any, is_inplace: bool) -> None:
        mask = np.zeros(obj.shape, dtype=bool)
        mask[key] = True
        if is_list_like(val) and len(val) < len(obj):
            msg = "operands could not be broadcast together with shapes"
            with pytest.raises(ValueError, match=msg):
                obj.where(~mask, val)
            return
        orig = obj
        obj = obj.copy()
        arr = obj._values
        res = obj.where(~mask, val)
        if val is NA and res.dtype == object:
            expected = expected.fillna(NA)
        elif val is None and res.dtype == object:
            assert expected.dtype == object
            expected = expected.copy()
            expected[expected.isna()] = None
        tm.assert_series_equal(res, expected)
        self._check_inplace(is_inplace, orig, arr, obj)

    def test_index_where(self, obj: Series, key: Any, expected: Series, raises: bool, val: Any) -> None:
        mask = np.zeros(obj.shape, dtype=bool)
        mask[key] = True
        res = Index(obj).where(~mask, val)
        expected_idx = Index(expected, dtype=expected.dtype)
        tm.assert_index_equal(res, expected_idx)

    def test_index_putmask(self, obj: Series, key: Any, expected: Series, raises: bool, val: Any) -> None:
        mask = np.zeros(obj.shape, dtype=bool)
        mask[key] = True
        res = Index(obj).putmask(mask, val)
        tm.assert_index_equal(res, Index(expected, dtype=expected.dtype))


@pytest.mark.parametrize(
    "obj,expected,key,raises",
    [
        pytest.param(
            Series(interval_range(1, 5)),
            Series(
                [Interval(1, 2), np.nan