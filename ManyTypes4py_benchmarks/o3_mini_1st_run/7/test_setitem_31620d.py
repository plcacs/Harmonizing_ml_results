#!/usr/bin/env python3
from __future__ import annotations
import contextlib
from datetime import date, datetime
from decimal import Decimal
import os
from typing import Any, Callable, Generator, Iterable, List, Optional, Union

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
        series: Series = Series(date_range("1/1/2000", periods=10))
        series[3] = None
        assert series[3] is NaT
        series[3:5] = None
        assert series[4] is NaT
        series[5] = np.nan
        assert series[5] is NaT
        series[5:7] = np.nan
        assert series[6] is NaT

    def test_setitem_multiindex_empty_slice(self) -> None:
        idx: MultiIndex = MultiIndex.from_tuples([("a", 1), ("b", 2)])
        result: Series = Series([1, 2], index=idx)
        expected: Series = result.copy()
        result.loc[[]] = 0
        tm.assert_series_equal(result, expected)

    def test_setitem_with_string_index(self) -> None:
        ser: Series = Series([1, 2, 3], index=["Date", "b", "other"], dtype=object)
        ser["Date"] = date.today()
        assert ser.Date == date.today()
        assert ser["Date"] == date.today()

    def test_setitem_tuple_with_datetimetz_values(self) -> None:
        arr: DatetimeIndex = date_range("2017", periods=4, tz="US/Eastern")
        index: List[tuple[int, int]] = [(0, 1), (0, 2), (0, 3), (0, 4)]
        result: Series = Series(arr, index=index)
        expected: Series = result.copy()
        result[0, 1] = np.nan
        expected.iloc[0] = np.nan
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("tz", ["US/Eastern", "UTC", "Asia/Tokyo"])
    def test_setitem_with_tz(self, tz: str, indexer_sli: Callable[[Series], Any]) -> None:
        orig: Series = Series(date_range("2016-01-01", freq="h", periods=3, tz=tz))
        assert orig.dtype == f"datetime64[ns, {tz}]"
        exp: Series = Series(
            [
                Timestamp("2016-01-01 00:00", tz=tz),
                Timestamp("2011-01-01 00:00", tz=tz),
                Timestamp("2016-01-01 02:00", tz=tz),
            ],
            dtype=orig.dtype,
        )
        ser: Series = orig.copy()
        indexer_sli(ser)[1] = Timestamp("2011-01-01", tz=tz)
        tm.assert_series_equal(ser, exp)
        vals: Series = Series(
            [
                Timestamp("2011-01-01", tz=tz),
                Timestamp("2012-01-01", tz=tz),
            ],
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

    def test_setitem_with_tz_dst(self, indexer_sli: Callable[[Series], Any]) -> None:
        tz: str = "US/Eastern"
        orig: Series = Series(date_range("2016-11-06", freq="h", periods=3, tz=tz))
        assert orig.dtype == f"datetime64[ns, {tz}]"
        exp: Series = Series(
            [
                Timestamp("2016-11-06 00:00-04:00", tz=tz),
                Timestamp("2011-01-01 00:00-05:00", tz=tz),
                Timestamp("2016-11-06 01:00-05:00", tz=tz),
            ],
            dtype=orig.dtype,
        )
        ser: Series = orig.copy()
        indexer_sli(ser)[1] = Timestamp("2011-01-01", tz=tz)
        tm.assert_series_equal(ser, exp)
        vals: Series = Series(
            [
                Timestamp("2011-01-01", tz=tz),
                Timestamp("2012-01-01", tz=tz),
            ],
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
        ser: Series = Series({"X": np.nan}, dtype=object)
        indexer: List[bool] = [True]
        value: np.ndarray = np.array([4], dtype="M8[ns]")
        ser.iloc[indexer] = value
        expected: Series = Series([value[0]], index=["X"], dtype=object)
        assert all(isinstance(x, np.datetime64) for x in expected.values)
        tm.assert_series_equal(ser, expected)


class TestSetitemScalarIndexer:
    def test_setitem_negative_out_of_bounds(self) -> None:
        ser: Series = Series(["a"] * 10, index=["a"] * 10)
        ser[-11] = "foo"
        exp: Series = Series(["a"] * 10 + ["foo"], index=["a"] * 10 + [-11])
        tm.assert_series_equal(ser, exp)

    @pytest.mark.parametrize("indexer", [tm.loc, tm.at])
    @pytest.mark.parametrize("ser_index", [0, 1])
    def test_setitem_series_object_dtype(
        self, indexer: Callable[[Series], Any], ser_index: int
    ) -> None:
        ser: Series = Series([0, 0], dtype="object")
        idxr: Any = indexer(ser)
        idxr[0] = Series([42], index=[ser_index])
        expected: Series = Series([Series([42], index=[ser_index]), 0], dtype="object")
        tm.assert_series_equal(ser, expected)

    @pytest.mark.parametrize("index, exp_value", [(0, 42), (1, np.nan)])
    def test_setitem_series(self, index: int, exp_value: Any) -> None:
        ser: Series = Series([0, 0])
        ser.loc[0] = Series([42], index=[index])
        expected: Series = Series([exp_value, 0])
        tm.assert_series_equal(ser, expected)


class TestSetitemSlices:
    def test_setitem_slice_float_raises(self, datetime_series: Series) -> None:
        msg: str = (
            "cannot do slice indexing on DatetimeIndex with these indexers \\[{key}\\] of type float"
        )
        with pytest.raises(TypeError, match=msg.format(key="4\\.0")):
            datetime_series[4.0:10.0] = 0
        with pytest.raises(TypeError, match=msg.format(key="4\\.5")):
            datetime_series[4.5:10.0] = 0

    def test_setitem_slice(self) -> None:
        ser: Series = Series(range(10), index=list(range(10)))
        ser[-12:] = 0
        assert (ser == 0).all()
        ser[:-12] = 5
        assert (ser == 0).all()

    def test_setitem_slice_integers(self) -> None:
        ser: Series = Series(
            np.random.default_rng(2).standard_normal(8),
            index=[2, 4, 6, 8, 10, 12, 14, 16],
        )
        ser[:4] = 0
        assert (ser[:4] == 0).all()
        assert not (ser[4:] == 0).any()

    def test_setitem_slicestep(self) -> None:
        series: Series = Series(np.arange(20, dtype=np.float64), index=np.arange(20, dtype=np.int64))
        series[::2] = 0
        assert (series[::2] == 0).all()

    def test_setitem_multiindex_slice(self, indexer_sli: Callable[[Series], Any]) -> None:
        mi: MultiIndex = MultiIndex.from_product(([0, 1], list("abcde")))
        result: Series = Series(np.arange(10, dtype=np.int64), mi)
        indexer_sli(result)[::4] = 100
        expected: Series = Series([100, 1, 2, 3, 100, 5, 6, 7, 100, 9], mi)
        tm.assert_series_equal(result, expected)


class TestSetitemBooleanMask:
    def test_setitem_mask_cast(self) -> None:
        ser: Series = Series([1, 2], index=[1, 2], dtype="int64")
        ser[[True, False]] = Series([0], index=[1], dtype="int64")
        expected: Series = Series([0, 2], index=[1, 2], dtype="int64")
        tm.assert_series_equal(ser, expected)

    def test_setitem_mask_align_and_promote(self) -> None:
        ts: Series = Series(
            np.random.default_rng(2).standard_normal(100), index=np.arange(100, 0, -1)
        ).round(5)
        mask: Series = ts > 0
        left: Series = ts.copy()
        right: Series = ts[mask].copy().map(str)
        with pytest.raises(TypeError, match="Invalid value"):
            left[mask] = right

    def test_setitem_mask_promote_strs(self) -> None:
        ser: Series = Series([0, 1, 2, 0])
        mask: Series = ser > 0
        ser2: Series = ser[mask].map(str)
        with pytest.raises(TypeError, match="Invalid value"):
            ser[mask] = ser2

    def test_setitem_mask_promote(self) -> None:
        ser: Series = Series([0, "foo", "bar", 0])
        mask: Series = Series([False, True, True, False])
        ser2: Series = ser[mask]
        ser[mask] = ser2
        expected: Series = Series([0, "foo", "bar", 0])
        tm.assert_series_equal(ser, expected)

    def test_setitem_boolean(self, string_series: Series) -> None:
        mask: Series = string_series > string_series.median()
        result: Series = string_series.copy()
        result[mask] = string_series * 2
        expected: Series = string_series * 2
        tm.assert_series_equal(result[mask], expected[mask])
        result = string_series.copy()
        result[mask] = (string_series * 2)[0:5]
        expected = (string_series * 2)[0:5].reindex_like(string_series)
        expected[-mask] = string_series[mask]
        tm.assert_series_equal(result[mask], expected[mask])

    def test_setitem_boolean_corner(self, datetime_series: Series) -> None:
        ts: Series = datetime_series
        mask_shifted: Series = ts.shift(1, freq=BDay()) > ts.median()
        msg: str = (
            "Unalignable boolean Series provided as indexer \\(index of the boolean Series and of the indexed object do not match"
        )
        with pytest.raises(IndexingError, match=msg):
            ts[mask_shifted] = 1
        with pytest.raises(IndexingError, match=msg):
            ts.loc[mask_shifted] = 1

    def test_setitem_boolean_different_order(self, string_series: Series) -> None:
        ordered: Series = string_series.sort_values()
        copy: Series = string_series.copy()
        copy[ordered > 0] = 0
        expected: Series = string_series.copy()
        expected[expected > 0] = 0
        tm.assert_series_equal(copy, expected)

    @pytest.mark.parametrize("func", [list, np.array, Series])
    def test_setitem_boolean_python_list(self, func: Callable, ) -> None:
        ser: Series = Series([None, "b", None])
        mask: Any = func([True, False, True])
        ser[mask] = ["a", "c"]
        expected: Series = Series(["a", "b", "c"])
        tm.assert_series_equal(ser, expected)

    def test_setitem_boolean_nullable_int_types(self, any_numeric_ea_dtype: Any) -> None:
        ser: Series = Series([5, 6, 7, 8], dtype=any_numeric_ea_dtype)
        ser[ser > 6] = Series(range(4), dtype=any_numeric_ea_dtype)
        expected: Series = Series([5, 6, 2, 3], dtype=any_numeric_ea_dtype)
        tm.assert_series_equal(ser, expected)
        ser = Series([5, 6, 7, 8], dtype=any_numeric_ea_dtype)
        ser.loc[ser > 6] = Series(range(4), dtype=any_numeric_ea_dtype)
        tm.assert_series_equal(ser, expected)
        ser = Series([5, 6, 7, 8], dtype=any_numeric_ea_dtype)
        loc_ser: Series = Series(range(4), dtype=any_numeric_ea_dtype)
        ser.loc[ser > 6] = loc_ser.loc[loc_ser > 1]
        tm.assert_series_equal(ser, expected)

    def test_setitem_with_bool_mask_and_values_matching_n_trues_in_length(self) -> None:
        ser: Series = Series([None] * 10)
        mask: List[bool] = [False] * 3 + [True] * 5 + [False] * 2
        ser[mask] = range(5)
        result: Series = ser
        expected: Series = Series([None] * 3 + list(range(5)) + [None] * 2, dtype=object)
        tm.assert_series_equal(result, expected)

    def test_setitem_nan_with_bool(self) -> None:
        result: Series = Series([True, False, True])
        with pytest.raises(TypeError, match="Invalid value"):
            result[0] = np.nan

    def test_setitem_mask_smallint_upcast(self) -> None:
        orig: Series = Series([1, 2, 3], dtype="int8")
        alt: np.ndarray = np.array([999, 1000, 1001], dtype=np.int64)
        mask: np.ndarray = np.array([True, False, True])
        ser: Series = orig.copy()
        with pytest.raises(TypeError, match="Invalid value"):
            ser[mask] = Series(alt)
        with pytest.raises(TypeError, match="Invalid value"):
            ser.mask(mask, alt, inplace=True)
        res: Series = ser.where(~mask, Series(alt))
        expected: Series = Series([999, 2, 1001])
        tm.assert_series_equal(res, expected)

    def test_setitem_mask_smallint_no_upcast(self) -> None:
        orig: Series = Series([1, 2, 3], dtype="uint8")
        alt: Series = Series([245, 1000, 246], dtype=np.int64)
        mask: np.ndarray = np.array([True, False, True])
        ser: Series = orig.copy()
        ser[mask] = alt
        expected: Series = Series([245, 2, 246], dtype="uint8")
        tm.assert_series_equal(ser, expected)
        ser2: Series = orig.copy()
        ser2.mask(mask, alt, inplace=True)
        tm.assert_series_equal(ser2, expected)
        ser3: Series = orig.copy()
        res: Series = ser3.where(~mask, alt)
        tm.assert_series_equal(res, expected, check_dtype=False)


class TestSetitemViewCopySemantics:
    def test_setitem_invalidates_datetime_index_freq(self) -> None:
        dti: DatetimeIndex = date_range("20130101", periods=3, tz="US/Eastern")
        ts: Any = dti[1]
        ser: Series = Series(dti)
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
        dti: DatetimeIndex = date_range("2016-01-01", periods=10, tz="US/Pacific")
        ts: Any = dti[0]
        ser: Series = Series(dti)
        assert ser._values is not dti
        assert ser._values._ndarray.base is dti._data._ndarray.base
        assert ser._mgr.blocks[0].values._ndarray.base is dti._data._ndarray.base
        assert ser._mgr.blocks[0].values is not dti
        ser[::3] = NaT
        assert ser[0] is NaT
        assert dti[0] == ts


class TestSetitemCallable:
    def test_setitem_callable_key(self) -> None:
        ser: Series = Series([1, 2, 3, 4], index=list("ABCD"))
        ser[lambda x: "A"] = -1
        expected: Series = Series([-1, 2, 3, 4], index=list("ABCD"))
        tm.assert_series_equal(ser, expected)

    def test_setitem_callable_other(self) -> None:
        inc: Callable[[Any], Any] = lambda x: x + 1
        ser: Series = Series([1, 2, -1, 4], dtype=object)
        ser[ser < 0] = inc
        expected: Series = Series([1, 2, inc, 4])
        tm.assert_series_equal(ser, expected)


class TestSetitemWithExpansion:
    def test_setitem_empty_series(self) -> None:
        key: Timestamp = Timestamp("2012-01-01")
        series: Series = Series(dtype=object)
        series[key] = 47
        expected: Series = Series(47, Index([key], dtype=object))
        tm.assert_series_equal(series, expected)

    def test_setitem_empty_series_datetimeindex_preserves_freq(self) -> None:
        dti: DatetimeIndex = DatetimeIndex([], freq="D", dtype="M8[ns]")
        series: Series = Series([], index=dti, dtype=object)
        key: Timestamp = Timestamp("2012-01-01")
        series[key] = 47
        expected: Series = Series(47, DatetimeIndex([key], freq="D").as_unit("ns"))
        tm.assert_series_equal(series, expected)
        assert series.index.freq == expected.index.freq

    def test_setitem_empty_series_timestamp_preserves_dtype(self) -> None:
        timestamp: Timestamp = Timestamp(1412526600000000000)
        series: Series = Series([timestamp], index=["timestamp"], dtype=object)
        expected: Any = series["timestamp"]
        series = Series([], dtype=object)
        series["anything"] = 300.0
        series["timestamp"] = timestamp
        result: Any = series["timestamp"]
        assert result == expected

    @pytest.mark.parametrize(
        "td",
        [
            Timedelta("9 days"),
            Timedelta("9 days").to_timedelta64(),
            Timedelta("9 days").to_pytimedelta(),
        ],
    )
    def test_append_timedelta_does_not_cast(
        self, td: Union[Timedelta, np.timedelta64, Any], using_infer_string: Any, request: Any
    ) -> None:
        if using_infer_string and (not isinstance(td, Timedelta)):
            request.applymarker(pytest.mark.xfail(reason="inferred as string"))
        expected: Series = Series(["x", td], index=[0, "td"], dtype=object)
        ser: Series = Series(["x"])
        ser["td"] = td
        tm.assert_series_equal(ser, expected)
        assert isinstance(ser["td"], Timedelta)
        ser = Series(["x"])
        ser.loc["td"] = Timedelta("9 days")
        tm.assert_series_equal(ser, expected)
        assert isinstance(ser["td"], Timedelta)

    def test_setitem_with_expansion_type_promotion(self) -> None:
        ser: Series = Series(dtype=object)
        ser["a"] = Timestamp("2016-01-01")
        ser["b"] = 3.0
        ser["c"] = "foo"
        expected: Series = Series(
            [Timestamp("2016-01-01"), 3.0, "foo"], index=Index(["a", "b", "c"], dtype=object)
        )
        tm.assert_series_equal(ser, expected)

    def test_setitem_not_contained(self, string_series: Series) -> None:
        ser: Series = string_series.copy()
        assert "foobar" not in ser.index
        ser["foobar"] = 1
        app: Series = Series([1], index=["foobar"], name="series")
        expected: Series = concat([string_series, app])
        tm.assert_series_equal(ser, expected)

    def test_setitem_keep_precision(self, any_numeric_ea_dtype: Any) -> None:
        ser: Series = Series([1, 2], dtype=any_numeric_ea_dtype)
        ser[2] = 10
        expected: Series = Series([1, 2, 10], dtype=any_numeric_ea_dtype)
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
        self,
        na: Any,
        target_na: Any,
        dtype: Any,
        target_dtype: Any,
        indexer: Union[int, slice],
        raises: bool,
    ) -> None:
        ser: Series = Series([1, 2], dtype=dtype)
        if raises:
            with pytest.raises(TypeError, match="Invalid value"):
                ser[indexer] = na
        else:
            ser[indexer] = na
            expected_values = [1, target_na] if indexer == 1 else [1, 2, target_na]
            expected: Series = Series(expected_values, dtype=target_dtype)
            tm.assert_series_equal(ser, expected)

    def test_setitem_enlargement_object_none(
        self, nulls_fixture: Any, using_infer_string: Any
    ) -> None:
        ser: Series = Series(["a", "b"])
        ser[3] = nulls_fixture
        dtype: str = "str" if using_infer_string and (not isinstance(nulls_fixture, Decimal)) else object
        expected: Series = Series(["a", "b", nulls_fixture], index=[0, 1, 3], dtype=dtype)
        tm.assert_series_equal(ser, expected)
        if using_infer_string:
            _ = ser[3] is np.nan
        else:
            assert ser[3] is nulls_fixture


def test_setitem_scalar_into_readonly_backing_data() -> None:
    array_: np.ndarray = np.zeros(5)
    array_.flags.writeable = False
    series: Series = Series(array_, copy=False)
    for n in series.index:
        msg: str = "assignment destination is read-only"
        with pytest.raises(ValueError, match=msg):
            series[n] = 1
        assert array_[n] == 0


def test_setitem_slice_into_readonly_backing_data() -> None:
    array_: np.ndarray = np.zeros(5)
    array_.flags.writeable = False
    series: Series = Series(array_, copy=False)
    msg: str = "assignment destination is read-only"
    with pytest.raises(ValueError, match=msg):
        series[1:3] = 1
    assert not array_.any()


def test_setitem_categorical_assigning_ops() -> None:
    orig: Series = Series(Categorical(["b", "b"], categories=["a", "b"]))
    ser: Series = orig.copy()
    ser[:] = "a"
    exp: Series = Series(Categorical(["a", "a"], categories=["a", "b"]))
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
    ser: Series = Series(Categorical([1, 2, 3]))
    exp: Series = Series(Categorical([1, np.nan, 3], categories=[1, 2, 3]))
    ser[1] = np.nan
    tm.assert_series_equal(ser, exp)


class TestSetitemCasting:
    @pytest.mark.parametrize("unique", [True, False])
    @pytest.mark.parametrize("val", [3, 3.0, "3"], ids=type)
    def test_setitem_non_bool_into_bool(
        self, val: Any, indexer_sli: Callable[[Series], Any], unique: bool
    ) -> None:
        ser: Series = Series([True, False])
        if not unique:
            ser.index = [1, 1]
        with pytest.raises(TypeError, match="Invalid value"):
            indexer_sli(ser)[1] = val

    def test_setitem_boolean_array_into_npbool(self) -> None:
        ser: Series = Series([True, False, True])
        values: Any = ser._values
        arr: Any = array([True, False, None])
        ser[:2] = arr[:2]
        assert ser._values is values
        with pytest.raises(TypeError, match="Invalid value"):
            ser[1:] = arr[1:]


class SetitemCastingEquivalents:
    """
    Check each of several methods that _should_ be equivalent to `obj[key] = val`

    We assume that
        - obj.index is the default Index(range(len(obj)))
        - the setitem does not expand the obj
    """

    @pytest.fixture
    def is_inplace(self, obj: Series, expected: Series) -> bool:
        """
        Whether we expect the setting to be in-place or not.
        """
        return expected.dtype == obj.dtype

    def check_indexer(
        self, obj: Series, key: Any, expected: Series, val: Any, indexer: Callable[[Series], Any], is_inplace: bool
    ) -> None:
        orig: Series = obj
        obj = obj.copy()
        arr: Any = obj._values
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

    def test_int_key(
        self,
        obj: Series,
        key: Any,
        expected: Series,
        raises: bool,
        val: Any,
        indexer_sli: Callable[[Series], Any],
        is_inplace: bool,
    ) -> None:
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
        rng: range = range(key, key + 1)
        with ctx:
            self.check_indexer(obj, rng, expected, val, indexer_sli, is_inplace)
        if indexer_sli is not tm.loc:
            slc: slice = slice(key, key + 1)
            with ctx:
                self.check_indexer(obj, slc, expected, val, indexer_sli, is_inplace)
        ilkey: List[Any] = [key]
        with ctx:
            self.check_indexer(obj, ilkey, expected, val, indexer_sli, is_inplace)
        indkey: np.ndarray = np.array(ilkey)
        with ctx:
            self.check_indexer(obj, indkey, expected, val, indexer_sli, is_inplace)
        genkey: Generator[Any, None, None] = (x for x in [key])
        with ctx:
            self.check_indexer(obj, genkey, expected, val, indexer_sli, is_inplace)

    def test_slice_key(
        self, obj: Series, key: Any, expected: Series, raises: bool, val: Any, indexer_sli: Callable[[Series], Any], is_inplace: bool
    ) -> None:
        if not isinstance(key, slice):
            pytest.skip("Not relevant for slice key")
        if raises:
            ctx = pytest.raises(TypeError, match="Invalid value")
        else:
            ctx = contextlib.nullcontext()
        if indexer_sli is not tm.loc:
            with ctx:
                self.check_indexer(obj, key, expected, val, indexer_sli, is_inplace)
        ilkey: List[Any] = list(range(len(obj)))[key]
        with ctx:
            self.check_indexer(obj, ilkey, expected, val, indexer_sli, is_inplace)
        indkey: np.ndarray = np.array(ilkey)
        with ctx:
            self.check_indexer(obj, indkey, expected, val, indexer_sli, is_inplace)
        genkey: Generator[Any, None, None] = (x for x in indkey)
        with ctx:
            self.check_indexer(obj, genkey, expected, val, indexer_sli, is_inplace)

    def test_mask_key(
        self, obj: Series, key: Any, expected: Series, raises: bool, val: Any, indexer_sli: Callable[[Series], Any]
    ) -> None:
        mask: np.ndarray = np.zeros(obj.shape, dtype=bool)
        mask[key] = True
        obj = obj.copy()
        if is_list_like(val) and len(val) < mask.sum():
            msg: str = "boolean index did not match indexed array along dimension"
            with pytest.raises(IndexError, match=msg):
                indexer_sli(obj)[mask] = val
            return
        if raises:
            with pytest.raises(TypeError, match="Invalid value"):
                indexer_sli(obj)[mask] = val
        else:
            indexer_sli(obj)[mask] = val

    def test_series_where(
        self, obj: Series, key: Any, expected: Series, raises: bool, val: Any, is_inplace: bool
    ) -> None:
        mask: np.ndarray = np.zeros(obj.shape, dtype=bool)
        mask[key] = True
        if is_list_like(val) and len(val) < len(obj):
            msg: str = "operands could not be broadcast together with shapes"
            with pytest.raises(ValueError, match=msg):
                obj.where(~mask, val)
            return
        orig: Series = obj
        obj = obj.copy()
        arr: Any = obj._values
        res: Series = obj.where(~mask, val)
        if val is NA and res.dtype == object:
            expected = expected.fillna(NA)
        elif val is None and res.dtype == object:
            assert expected.dtype == object
            expected = expected.copy()
            expected[expected.isna()] = None
        tm.assert_series_equal(res, expected)
        self._check_inplace(is_inplace, orig, arr, obj)

    def test_index_where(self, obj: Series, key: Any, expected: Series, raises: bool, val: Any) -> None:
        mask: np.ndarray = np.zeros(obj.shape, dtype=bool)
        mask[key] = True
        res: Index = Index(obj).where(~mask, val)
        expected_idx: Index = Index(expected, dtype=expected.dtype)
        tm.assert_index_equal(res, expected_idx)

    def test_index_putmask(self, obj: Series, key: Any, expected: Series, raises: bool, val: Any) -> None:
        mask: np.ndarray = np.zeros(obj.shape, dtype=bool)
        mask[key] = True
        res: Index = Index(obj).putmask(mask, val)
        tm.assert_index_equal(res, Index(expected, dtype=expected.dtype))


@pytest.mark.parametrize(
    "obj,expected,key,raises",
    [
        pytest.param(
            Series(interval_range(1, 5)),
            Series([Interval(1, 2), np.nan, Interval(3, 4), Interval(4, 5)], dtype="interval[float64]"),
            1,
            True,
            id="interval_int_na_value",
        ),
        pytest.param(
            Series([2, 3, 4, 5, 6, 7, 8, 9, 10]),
            Series([np.nan, 3, np.nan, 5, np.nan, 7, np.nan, 9, np.nan]),
            slice(None, None, 2),
            False,
            id="int_series_slice_key_step",
        ),
        pytest.param(
            Series([True, True, False, False]),
            Series([np.nan, True, np.nan, False], dtype=object),
            slice(None, None, 2),
            True,
            id="bool_series_slice_key_step",
        ),
        pytest.param(
            Series(np.arange(10)),
            Series([np.nan, np.nan, np.nan, np.nan, np.nan, 5, 6, 7, 8, 9]),
            slice(None, 5),
            False,
            id="int_series_slice_key",
        ),
        pytest.param(
            Series([1, 2, 3]),
            Series([np.nan, 2, 3]),
            0,
            False,
            id="int_series_int_key",
        ),
        pytest.param(
            Series([False]),
            Series([np.nan], dtype=object),
            0,
            True,
            id="bool_series_int_key_change_all",
        ),
        pytest.param(
            Series([False, True]),
            Series([np.nan, True], dtype=object),
            0,
            True,
            id="bool_series_int_key",
        ),
    ],
)
class TestSetitemCastingEquivalents(SetitemCastingEquivalents):
    @pytest.fixture(params=[np.nan, np.float64("NaN"), None, NA])
    def val(self, request: Any) -> Any:
        """
        NA values that should generally be valid_na for *all* dtypes.

        Include both python float NaN and np.float64; only np.float64 has a
        `dtype` attribute.
        """
        return request.param


class TestSetitemTimedelta64IntoNumeric(SetitemCastingEquivalents):
    @pytest.fixture
    def val(self) -> np.timedelta64:
        td: np.timedelta64 = np.timedelta64(4, "ns")
        return td

    @pytest.fixture(params=[complex, int, float])
    def dtype(self, request: Any) -> Any:
        return request.param

    @pytest.fixture
    def obj(self, dtype: Any) -> Series:
        arr: np.ndarray = np.arange(5).astype(dtype)
        ser: Series = Series(arr)
        return ser

    @pytest.fixture
    def expected(self, dtype: Any) -> Series:
        arr: np.ndarray = np.arange(5).astype(dtype)
        ser: Series = Series(arr)
        ser = ser.astype(object)
        ser.iloc[0] = np.timedelta64(4, "ns")
        return ser

    @pytest.fixture
    def key(self) -> int:
        return 0

    @pytest.fixture
    def raises(self) -> bool:
        return True


class TestSetitemDT64IntoInt(SetitemCastingEquivalents):
    @pytest.fixture(params=["M8[ns]", "m8[ns]"])
    def dtype(self, request: Any) -> str:
        return request.param

    @pytest.fixture
    def scalar(self, dtype: str) -> Union[np.datetime64, np.timedelta64]:
        val: np.datetime64 = np.datetime64("2021-01-18 13:25:00", "ns")
        if dtype == "m8[ns]":
            val = val - val
        return val

    @pytest.fixture
    def expected(self, scalar: Any) -> Series:
        expected: Series = Series([scalar, scalar, 3], dtype=object)
        assert isinstance(expected[0], type(scalar))
        return expected

    @pytest.fixture
    def obj(self) -> Series:
        return Series([1, 2, 3])

    @pytest.fixture
    def key(self) -> slice:
        return slice(None, -1)

    @pytest.fixture(params=[None, list, np.array])
    def val(self, scalar: Any, request: Any) -> Any:
        box: Optional[Callable[[list[Any]], Any]] = request.param
        if box is None:
            return scalar
        return box([scalar, scalar])

    @pytest.fixture
    def raises(self) -> bool:
        return True


class TestSetitemNAPeriodDtype(SetitemCastingEquivalents):
    @pytest.fixture
    def expected(self, key: Any) -> Series:
        exp: Series = Series(period_range("2000-01-01", periods=10, freq="D"))
        exp._values.view("i8")[key] = NaT._value
        assert exp[key] is NaT or all(x is NaT for x in exp[key])
        return exp

    @pytest.fixture
    def obj(self) -> Series:
        return Series(period_range("2000-01-01", periods=10, freq="D"))

    @pytest.fixture(params=[3, slice(3, 5)])
    def key(self, request: Any) -> Any:
        return request.param

    @pytest.fixture(params=[None, np.nan])
    def val(self, request: Any) -> Any:
        return request.param

    @pytest.fixture
    def raises(self) -> bool:
        return False


class TestSetitemNADatetimeLikeDtype(SetitemCastingEquivalents):
    @pytest.fixture(params=["m8[ns]", "M8[ns]", "datetime64[ns, UTC]", "datetime64[ns, US/Central]"])
    def dtype(self, request: Any) -> str:
        return request.param

    @pytest.fixture
    def obj(self, dtype: str) -> Series:
        i8vals: np.ndarray = date_range("2016-01-01", periods=3).asi8
        idx: Index = Index(i8vals, dtype=dtype)
        assert idx.dtype == dtype
        return Series(idx)

    @pytest.fixture(params=[None, np.nan, NaT, np.timedelta64("NaT", "ns"), np.datetime64("NaT", "ns")])
    def val(self, request: Any) -> Any:
        return request.param

    @pytest.fixture
    def is_inplace(self, val: Any, obj: Series) -> bool:
        return val is NaT or val is None or val is np.nan or (obj.dtype == val.dtype)

    @pytest.fixture
    def expected(self, obj: Series, val: Any, is_inplace: bool) -> Series:
        dtype: Union[str, Any] = obj.dtype if is_inplace else object
        expected: Series = Series([val] + list(obj[1:]), dtype=dtype)
        return expected

    @pytest.fixture
    def key(self) -> int:
        return 0

    @pytest.fixture
    def raises(self, is_inplace: bool) -> bool:
        return False if is_inplace else True


class TestSetitemMismatchedTZCastsToObject(SetitemCastingEquivalents):
    @pytest.fixture
    def obj(self) -> Series:
        return Series(date_range("2000", periods=2, tz="US/Central"))

    @pytest.fixture
    def val(self) -> Timestamp:
        return Timestamp("2000", tz="US/Eastern")

    @pytest.fixture
    def key(self) -> int:
        return 0

    @pytest.fixture
    def expected(self, obj: Series, val: Timestamp) -> Series:
        expected: Series = Series(
            [
                val.tz_convert("US/Central"),
                Timestamp("2000-01-02 00:00:00-06:00", tz="US/Central"),
            ],
            dtype=obj.dtype,
        )
        return expected

    @pytest.fixture
    def raises(self) -> bool:
        return False


@pytest.mark.parametrize(
    "obj,expected",
    [
        (Series([1, 2, 3]), Series([np.nan, 2, 3])),
        (Series([1.0, 2.0, 3.0]), Series([np.nan, 2.0, 3.0])),
        (
            Series([datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)]),
            Series([NaT, datetime(2000, 1, 2), datetime(2000, 1, 3)]),
        ),
        (Series(["foo", "bar", "baz"]), Series([None, "bar", "baz"])),
    ],
)
class TestSeriesNoneCoercion(SetitemCastingEquivalents):
    @pytest.fixture
    def key(self) -> int:
        return 0

    @pytest.fixture
    def val(self) -> None:
        return None

    @pytest.fixture
    def raises(self) -> bool:
        return False


class TestSetitemFloatIntervalWithIntIntervalValues(SetitemCastingEquivalents):
    def test_setitem_example(self) -> None:
        idx: IntervalIndex = IntervalIndex.from_breaks(range(4))
        obj: Series = Series(idx)
        val: Interval = Interval(0.5, 1.5)
        with pytest.raises(TypeError, match="Invalid value"):
            obj[0] = val

    @pytest.fixture
    def obj(self) -> Series:
        """
        Fixture to create a Series [(0, 1], (1, 2], (2, 3]]
        """
        idx: IntervalIndex = IntervalIndex.from_breaks(range(4))
        return Series(idx)

    @pytest.fixture
    def val(self) -> Interval:
        """
        Fixture to get an interval (0.5, 1.5]
        """
        return Interval(0.5, 1.5)

    @pytest.fixture
    def key(self) -> int:
        """
        Fixture to get a key 0
        """
        return 0

    @pytest.fixture
    def expected(self, obj: Series, val: Interval) -> Series:
        """
        Fixture to get a Series [(0.5, 1.5], (1.0, 2.0], (2.0, 3.0]]
        """
        data: List[Any] = [val] + list(obj[1:])
        idx: IntervalIndex = IntervalIndex(data, dtype="Interval[float64]")
        return Series(idx)

    @pytest.fixture
    def raises(self) -> bool:
        """
        Fixture to enable raising pytest exceptions
        """
        return True


class TestSetitemRangeIntoIntegerSeries(SetitemCastingEquivalents):
    @pytest.fixture
    def obj(self, any_int_numpy_dtype: Any) -> Series:
        dtype: np.dtype = np.dtype(any_int_numpy_dtype)
        ser: Series = Series(range(5), dtype=dtype)
        return ser

    @pytest.fixture
    def val(self) -> range:
        return range(2, 4)

    @pytest.fixture
    def key(self) -> slice:
        return slice(0, 2)

    @pytest.fixture
    def expected(self, any_int_numpy_dtype: Any) -> Series:
        dtype: np.dtype = np.dtype(any_int_numpy_dtype)
        exp: Series = Series([2, 3, 2, 3, 4], dtype=dtype)
        return exp

    @pytest.fixture
    def raises(self) -> bool:
        return False


@pytest.mark.parametrize("val, raises", [(np.array([2.0, 3.0]), False), (np.array([2.5, 3.5]), True), (np.array([2 ** 65, 2 ** 65 + 1], dtype=np.float64), True)])
class TestSetitemFloatNDarrayIntoIntegerSeries(SetitemCastingEquivalents):
    @pytest.fixture
    def obj(self) -> Series:
        return Series(range(5), dtype=np.int64)

    @pytest.fixture
    def key(self) -> slice:
        return slice(0, 2)

    @pytest.fixture
    def expected(self, val: np.ndarray) -> Series:
        if val[0] == 2:
            dtype: Any = np.int64
        else:
            dtype = np.float64
        res_values: np.ndarray = np.array(range(5), dtype=dtype)
        res_values[:2] = val
        return Series(res_values)


@pytest.mark.parametrize("val", [512, np.int16(512)])
class TestSetitemIntoIntegerSeriesNeedsUpcast(SetitemCastingEquivalents):
    @pytest.fixture
    def obj(self) -> Series:
        return Series([1, 2, 3], dtype=np.int8)

    @pytest.fixture
    def key(self) -> int:
        return 1

    @pytest.fixture
    def expected(self) -> Series:
        return Series([1, 512, 3], dtype=np.int16)

    @pytest.fixture
    def raises(self) -> bool:
        return True


@pytest.mark.parametrize("val", [2 ** 33 + 1.0, 2 ** 33 + 1.1, 2 ** 62])
class TestSmallIntegerSetitemUpcast(SetitemCastingEquivalents):
    @pytest.fixture
    def obj(self) -> Series:
        return Series([1, 2, 3], dtype="i4")

    @pytest.fixture
    def key(self) -> int:
        return 0

    @pytest.fixture
    def expected(self, val: Any) -> Series:
        if val % 1 != 0:
            dtype: str = "f8"
        else:
            dtype = "i8"
        return Series([val, 2, 3], dtype=dtype)

    @pytest.fixture
    def raises(self) -> bool:
        return True


class CoercionTest(SetitemCastingEquivalents):
    @pytest.fixture
    def key(self) -> int:
        return 1

    @pytest.fixture
    def expected(self, obj: Series, key: int, val: Any, exp_dtype: Any) -> Series:
        vals: list[Any] = list(obj)
        vals[key] = val
        return Series(vals, dtype=exp_dtype)


@pytest.mark.parametrize("val,exp_dtype,raises", [(np.int32(1), np.int8, None), (np.int16(2 ** 9), np.int16, True)])
class TestCoercionInt8(CoercionTest):
    @pytest.fixture
    def obj(self) -> Series:
        return Series([1, 2, 3, 4], dtype=np.int8)


@pytest.mark.parametrize("val", [1, 1.1, 1 + 1j, True])
@pytest.mark.parametrize("exp_dtype", [object])
class TestCoercionObject(CoercionTest):
    @pytest.fixture
    def obj(self) -> Series:
        return Series(["a", "b", "c", "d"], dtype=object)

    @pytest.fixture
    def raises(self) -> bool:
        return False


@pytest.mark.parametrize("val,exp_dtype,raises", [(1, object, True), ("e", StringDtype(na_value=np.nan), False)])
class TestCoercionString(CoercionTest):
    @pytest.fixture
    def obj(self) -> Series:
        return Series(["a", "b", "c", "d"], dtype=StringDtype(na_value=np.nan))


@pytest.mark.parametrize("val,exp_dtype,raises", [(1, np.complex128, False), (1.1, np.complex128, False), (1 + 1j, np.complex128, False), (True, object, True)])
class TestCoercionComplex(CoercionTest):
    @pytest.fixture
    def obj(self) -> Series:
        return Series([1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j])


@pytest.mark.parametrize("val,exp_dtype,raises", [(1, object, True), ("3", object, True), (3, object, True), (1.1, object, True), (1 + 1j, object, True), (True, bool, False)])
class TestCoercionBool(CoercionTest):
    @pytest.fixture
    def obj(self) -> Series:
        return Series([True, False, True, False], dtype=bool)


@pytest.mark.parametrize("val,exp_dtype,raises", [(1, np.float64, False), (1.1, np.float64, False), (1 + 1j, np.complex128, True), (True, object, True)])
class TestCoercionInt64(CoercionTest):
    @pytest.fixture
    def obj(self) -> Series:
        return Series([1, 2, 3, 4])


@pytest.mark.parametrize("val,exp_dtype,raises", [(1, np.float64, False), (1.1, np.float64, False), (1 + 1j, np.complex128, True), (True, object, True)])
class TestCoercionFloat64(CoercionTest):
    @pytest.fixture
    def obj(self) -> Series:
        return Series([1.1, 2.2, 3.3, 4.4])


@pytest.mark.parametrize(
    "val,exp_dtype,raises",
    [
        (1, np.float32, False),
        pytest.param(
            1.1,
            np.float32,
            False,
            marks=pytest.mark.xfail(
                not np_version_gte1p24
                or (np_version_gte1p24 and os.environ.get("NPY_PROMOTION_STATE", "weak") != "weak"),
                reason="np.float32(1.1) ends up as 1.100000023841858, so np_can_hold_element raises and we cast to float64",
            ),
        ),
        (1 + 1j, np.complex128, True),
        (True, object, True),
        (np.uint8(2), np.float32, False),
        (np.uint32(2), np.float32, False),
        (np.uint32(np.iinfo(np.uint32).max), np.float64, True),
        (np.uint64(2), np.float32, False),
        (np.int64(2), np.float32, False),
    ],
)
class TestCoercionFloat32(CoercionTest):
    @pytest.fixture
    def obj(self) -> Series:
        return Series([1.1, 2.2, 3.3, 4.4], dtype=np.float32)

    def test_slice_key(
        self, obj: Series, key: Any, expected: Series, raises: bool, val: Any, indexer_sli: Callable[[Series], Any], is_inplace: bool
    ) -> None:
        super().test_slice_key(obj, key, expected, raises, val, indexer_sli, is_inplace)
        if isinstance(val, float):
            raise AssertionError("xfail not relevant for this test.")


@pytest.mark.parametrize("exp_dtype", ["M8[ms]", "M8[ms, UTC]", "m8[ms]"])
class TestCoercionDatetime64HigherReso(CoercionTest):
    @pytest.fixture
    def obj(self, exp_dtype: str) -> Series:
        idx: DatetimeIndex = date_range("2011-01-01", freq="D", periods=4, unit="s")
        if exp_dtype == "m8[ms]":
            idx = idx - Timestamp("1970-01-01")
            assert idx.dtype == "m8[s]"
        elif exp_dtype == "M8[ms, UTC]":
            idx = idx.tz_localize("UTC")
        return Series(idx)

    @pytest.fixture
    def val(self, exp_dtype: str) -> Any:
        ts: Timestamp = Timestamp("2011-01-02 03:04:05.678").as_unit("ms")
        if exp_dtype == "m8[ms]":
            return ts - Timestamp("1970-01-01")
        elif exp_dtype == "M8[ms, UTC]":
            return ts.tz_localize("UTC")
        return ts

    @pytest.fixture
    def raises(self) -> bool:
        return True


@pytest.mark.parametrize("val,exp_dtype,raises", [(Timestamp("2012-01-01"), "datetime64[ns]", False), (1, object, True), ("x", object, True)])
class TestCoercionDatetime64(CoercionTest):
    @pytest.fixture
    def obj(self) -> Series:
        return Series(date_range("2011-01-01", freq="D", periods=4))

    @pytest.fixture
    def raises(self) -> bool:
        return False


@pytest.mark.parametrize(
    "val,exp_dtype,raises",
    [
        (Timestamp("2012-01-01", tz="US/Eastern"), "datetime64[ns, US/Eastern]", False),
        (Timestamp("2012-01-01", tz="US/Pacific"), "datetime64[ns, US/Eastern]", False),
        (Timestamp("2012-01-01"), object, True),
        (1, object, True),
    ],
)
class TestCoercionDatetime64TZ(CoercionTest):
    @pytest.fixture
    def obj(self) -> Series:
        tz: str = "US/Eastern"
        return Series(date_range("2011-01-01", freq="D", periods=4, tz=tz))

    @pytest.fixture
    def raises(self) -> bool:
        return False


@pytest.mark.parametrize("val,exp_dtype,raises", [(Timedelta("12 day"), "timedelta64[ns]", False), (1, object, True), ("x", object, True)])
class TestCoercionTimedelta64(CoercionTest):
    @pytest.fixture
    def obj(self) -> Series:
        return Series(timedelta_range("1 day", periods=4))

    @pytest.fixture
    def raises(self) -> bool:
        return False


@pytest.mark.parametrize("val", ["foo", Period("2016", freq="Y"), Interval(1, 2, closed="both")])
@pytest.mark.parametrize("exp_dtype", [object])
class TestPeriodIntervalCoercion(CoercionTest):
    @pytest.fixture(params=[period_range("2016-01-01", periods=3, freq="D"), interval_range(1, 5)])
    def obj(self, request: Any) -> Series:
        return Series(request.param)

    @pytest.fixture
    def raises(self) -> bool:
        return True


def test_20643() -> None:
    orig: Series = Series([0, 1, 2], index=["a", "b", "c"])
    ser: Series = orig.copy()
    with pytest.raises(TypeError, match="Invalid value"):
        ser.at["b"] = 2.7
    with pytest.raises(TypeError, match="Invalid value"):
        ser.loc["b"] = 2.7
    with pytest.raises(TypeError, match="Invalid value"):
        ser["b"] = 2.7
    ser = orig.copy()
    with pytest.raises(TypeError, match="Invalid value"):
        ser.iat[1] = 2.7
    with pytest.raises(TypeError, match="Invalid value"):
        ser.iloc[1] = 2.7
    orig_df: DataFrame = orig.to_frame("A")
    df: DataFrame = orig_df.copy()
    with pytest.raises(TypeError, match="Invalid value"):
        df.at["b", "A"] = 2.7
    with pytest.raises(TypeError, match="Invalid value"):
        df.loc["b", "A"] = 2.7
    with pytest.raises(TypeError, match="Invalid value"):
        df.iloc[1, 0] = 2.7
    with pytest.raises(TypeError, match="Invalid value"):
        df.iat[1, 0] = 2.7


def test_20643_comment() -> None:
    orig: Series = Series([0, 1, 2], index=["a", "b", "c"])
    expected: Series = Series([np.nan, 1, 2], index=["a", "b", "c"])
    ser: Series = orig.copy()
    ser.iat[0] = None
    tm.assert_series_equal(ser, expected)
    ser = orig.copy()
    ser.iloc[0] = None
    tm.assert_series_equal(ser, expected)


def test_15413() -> None:
    ser: Series = Series([1, 2, 3])
    with pytest.raises(TypeError, match="Invalid value"):
        ser[ser == 2] += 0.5
    with pytest.raises(TypeError, match="Invalid value"):
        ser[1] += 0.5
    with pytest.raises(TypeError, match="Invalid value"):
        ser.loc[1] += 0.5
    with pytest.raises(TypeError, match="Invalid value"):
        ser.iloc[1] += 0.5
    with pytest.raises(TypeError, match="Invalid value"):
        ser.iat[1] += 0.5
    with pytest.raises(TypeError, match="Invalid value"):
        ser.at[1] += 0.5


def test_32878_int_itemsize() -> None:
    arr: np.ndarray = np.arange(5).astype("i4")
    ser: Series = Series(arr)
    val: np.int64 = np.int64(np.iinfo(np.int64).max)
    with pytest.raises(TypeError, match="Invalid value"):
        ser[0] = val


def test_32878_complex_itemsize() -> None:
    arr: np.ndarray = np.arange(5).astype("c8")
    ser: Series = Series(arr)
    val: Any = np.finfo(np.float64).max
    val = val.astype("c16")
    with pytest.raises(TypeError, match="Invalid value"):
        ser[0] = val


def test_37692(indexer_al: Callable[[Series], Any]) -> None:
    ser: Series = Series([1, 2, 3], index=["a", "b", "c"])
    with pytest.raises(TypeError, match="Invalid value"):
        indexer_al(ser)["b"] = "test"


def test_setitem_bool_int_float_consistency(indexer_sli: Callable[[Series], Any]) -> None:
    for dtype in [np.float64, np.int64]:
        ser: Series = Series(0, index=range(3), dtype=dtype)
        with pytest.raises(TypeError, match="Invalid value"):
            indexer_sli(ser)[0] = True
        ser = Series(0, index=range(3), dtype=bool)
        with pytest.raises(TypeError, match="Invalid value"):
            ser[0] = dtype(1)
    ser = Series(0, index=range(3), dtype=np.int64)
    indexer_sli(ser)[0] = np.float64(1.0)
    assert ser.dtype == np.int64
    ser = Series(0, index=range(3), dtype=np.float64)
    indexer_sli(ser)[0] = np.int64(1)


def test_setitem_positional_with_casting() -> None:
    ser: Series = Series([1, 2, 3], index=["a", "b", "c"])
    ser[0] = "X"
    expected: Series = Series([1, 2, 3, "X"], index=["a", "b", "c", 0], dtype=object)
    tm.assert_series_equal(ser, expected)


def test_setitem_positional_float_into_int_coerces() -> None:
    ser: Series = Series([1, 2, 3], index=["a", "b", "c"])
    ser[0] = 1.5
    expected: Series = Series([1, 2, 3, 1.5], index=["a", "b", "c", 0])
    tm.assert_series_equal(ser, expected)


def test_setitem_int_not_positional() -> None:
    ser: Series = Series([1, 2, 3, 4], index=[1.1, 2.1, 3.0, 4.1])
    assert not ser.index._should_fallback_to_positional
    ser[3] = 10
    expected: Series = Series([1, 2, 10, 4], index=ser.index)
    tm.assert_series_equal(ser, expected)
    ser[5] = 5
    expected = Series([1, 2, 10, 4, 5], index=[1.1, 2.1, 3.0, 4.1, 5.0])
    tm.assert_series_equal(ser, expected)
    ii: IntervalIndex = IntervalIndex.from_breaks(range(10))[::2]
    ser2: Series = Series(range(len(ii)), index=ii)
    exp_index: Index = ii.astype(object).append(Index([4]))
    expected2: Series = Series([0, 1, 2, 3, 4, 9], index=exp_index)
    ser2[4] = 9
    tm.assert_series_equal(ser2, expected2)
    mi: MultiIndex = MultiIndex.from_product([ser.index, ["A", "B"]])
    ser3: Series = Series(range(len(mi)), index=mi)
    expected3: Series = ser3.copy()
    expected3.loc[4] = 99
    ser3[4] = 99
    tm.assert_series_equal(ser3, expected3)


def test_setitem_with_bool_indexer() -> None:
    df: DataFrame = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result: Series = df.pop("b").copy()
    result[[True, False, False]] = 9
    expected: Series = Series(data=[9, 5, 6], name="b")
    tm.assert_series_equal(result, expected)
    df.loc[[True, False, False], "a"] = 10
    expected_df: DataFrame = DataFrame({"a": [10, 2, 3]})
    tm.assert_frame_equal(df, expected_df)


@pytest.mark.parametrize("size", range(2, 6))
@pytest.mark.parametrize("mask", [[True, False, False, False, False], [True, False], [False]])
@pytest.mark.parametrize("item", [2.0, np.nan, np.finfo(float).max, np.finfo(float).min])
@pytest.mark.parametrize("box", [np.array, list, tuple])
def test_setitem_bool_indexer_dont_broadcast_length1_values(
    size: int, mask: list[bool], item: Any, box: Callable[[list[Any]], Any]
) -> None:
    selection: np.ndarray = np.resize(mask, size)
    data: np.ndarray = np.arange(size, dtype=float)
    ser: Series = Series(data)
    if selection.sum() != 1:
        msg: str = "cannot set using a list-like indexer with a different length than the value"
        with pytest.raises(ValueError, match=msg):
            ser[selection] = box([item])
    else:
        ser[selection] = box([item])
        expected: Series = Series(np.arange(size, dtype=float))
        expected[selection] = item
        tm.assert_series_equal(ser, expected)


def test_setitem_empty_mask_dont_upcast_dt64() -> None:
    dti: DatetimeIndex = date_range("2016-01-01", periods=3)
    ser: Series = Series(dti)
    orig: Series = ser.copy()
    mask: np.ndarray = np.zeros(3, dtype=bool)
    ser[mask] = "foo"
    assert ser.dtype == dti.dtype
    tm.assert_series_equal(ser, orig)
    ser.mask(mask, "foo", inplace=True)
    assert ser.dtype == dti.dtype
    tm.assert_series_equal(ser, orig)