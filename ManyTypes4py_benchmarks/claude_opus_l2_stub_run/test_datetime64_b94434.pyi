from datetime import datetime, time, timedelta, timezone
from itertools import product, starmap
import operator

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DateOffset,
    DatetimeIndex,
    NaT,
    Period,
    Series,
    Timedelta,
    TimedeltaIndex,
    Timestamp,
    date_range,
)


class TestDatetime64ArrayLikeComparisons:
    def test_compare_zerodim(self, tz_naive_fixture: Any, box_with_array: Any) -> None: ...

    @pytest.mark.parametrize(
        "other",
        [
            "foo",
            -1,
            99,
            4.0,
            object(),
            timedelta(days=2),
            datetime(2001, 1, 1).date(),
            None,
            np.nan,
        ],
    )
    def test_dt64arr_cmp_scalar_invalid(
        self, other: object, tz_naive_fixture: Any, box_with_array: Any
    ) -> None: ...

    @pytest.mark.parametrize(
        "other",
        [
            list(range(10)),
            np.arange(10),
            np.arange(10).astype(np.float32),
            np.arange(10).astype(object),
            pd.timedelta_range("1ns", periods=10).array,
            np.array(pd.timedelta_range("1ns", periods=10)),
            list(pd.timedelta_range("1ns", periods=10)),
            pd.timedelta_range("1 Day", periods=10).astype(object),
            pd.period_range("1971-01-01", freq="D", periods=10).array,
            pd.period_range("1971-01-01", freq="D", periods=10).astype(object),
        ],
    )
    def test_dt64arr_cmp_arraylike_invalid(
        self, other: object, tz_naive_fixture: Any, box_with_array: Any
    ) -> None: ...

    def test_dt64arr_cmp_mixed_invalid(self, tz_naive_fixture: Any) -> None: ...

    def test_dt64arr_nat_comparison(
        self, tz_naive_fixture: Any, box_with_array: Any
    ) -> None: ...


class TestDatetime64SeriesComparison:
    @pytest.mark.parametrize(
        "pair",
        [
            (
                [Timestamp("2011-01-01"), NaT, Timestamp("2011-01-03")],
                [NaT, NaT, Timestamp("2011-01-03")],
            ),
            (
                [Timedelta("1 days"), NaT, Timedelta("3 days")],
                [NaT, NaT, Timedelta("3 days")],
            ),
            (
                [Period("2011-01", freq="M"), NaT, Period("2011-03", freq="M")],
                [NaT, NaT, Period("2011-03", freq="M")],
            ),
        ],
    )
    @pytest.mark.parametrize("reverse", [True, False])
    @pytest.mark.parametrize("dtype", [None, object])
    @pytest.mark.parametrize(
        "op, expected",
        [
            (operator.eq, [False, False, True]),
            (operator.ne, [True, True, False]),
            (operator.lt, [False, False, False]),
            (operator.gt, [False, False, False]),
            (operator.ge, [False, False, True]),
            (operator.le, [False, False, True]),
        ],
    )
    def test_nat_comparisons(
        self,
        dtype: type | None,
        index_or_series: Any,
        reverse: bool,
        pair: tuple[list[Any], list[Any]],
        op: Any,
        expected: list[bool],
    ) -> None: ...

    @pytest.mark.parametrize(
        "data",
        [
            [Timestamp("2011-01-01"), NaT, Timestamp("2011-01-03")],
            [Timedelta("1 days"), NaT, Timedelta("3 days")],
            [Period("2011-01", freq="M"), NaT, Period("2011-03", freq="M")],
        ],
    )
    @pytest.mark.parametrize("dtype", [None, object])
    def test_nat_comparisons_scalar(
        self, dtype: type | None, data: list[Any], box_with_array: Any
    ) -> None: ...

    @pytest.mark.parametrize(
        "val", [datetime(2000, 1, 4), datetime(2000, 1, 5)]
    )
    def test_series_comparison_scalars(self, val: datetime) -> None: ...

    @pytest.mark.parametrize(
        "left,right",
        [("lt", "gt"), ("le", "ge"), ("eq", "eq"), ("ne", "ne")],
    )
    def test_timestamp_compare_series(self, left: str, right: str) -> None: ...

    def test_dt64arr_timestamp_equality(self, box_with_array: Any) -> None: ...

    @pytest.mark.parametrize(
        "datetimelike",
        [
            Timestamp("20130101"),
            datetime(2013, 1, 1),
            np.datetime64("2013-01-01T00:00", "ns"),
        ],
    )
    @pytest.mark.parametrize(
        "op,expected",
        [
            (operator.lt, [True, False, False, False]),
            (operator.le, [True, True, False, False]),
            (operator.eq, [False, True, False, False]),
            (operator.gt, [False, False, False, True]),
        ],
    )
    def test_dt64_compare_datetime_scalar(
        self, datetimelike: Any, op: Any, expected: list[bool]
    ) -> None: ...

    def test_ts_series_numpy_maximum(self) -> None: ...


class TestDatetimeIndexComparisons:
    def test_comparators(self, comparison_op: Any) -> None: ...

    @pytest.mark.parametrize(
        "other",
        [
            datetime(2016, 1, 1),
            Timestamp("2016-01-01"),
            np.datetime64("2016-01-01"),
        ],
    )
    def test_dti_cmp_datetimelike(
        self, other: Any, tz_naive_fixture: Any
    ) -> None: ...

    @pytest.mark.parametrize("dtype", [None, object])
    def test_dti_cmp_nat(
        self, dtype: type | None, box_with_array: Any
    ) -> None: ...

    def test_dti_cmp_nat_behaves_like_float_cmp_nan(self) -> None: ...

    def test_comparison_tzawareness_compat(
        self, comparison_op: Any, box_with_array: Any
    ) -> None: ...

    def test_comparison_tzawareness_compat_scalars(
        self, comparison_op: Any, box_with_array: Any
    ) -> None: ...

    @pytest.mark.parametrize(
        "other",
        [
            datetime(2016, 1, 1),
            Timestamp("2016-01-01"),
            np.datetime64("2016-01-01"),
        ],
    )
    @pytest.mark.filterwarnings("ignore:elementwise comp:DeprecationWarning")
    def test_scalar_comparison_tzawareness(
        self,
        comparison_op: Any,
        other: Any,
        tz_aware_fixture: Any,
        box_with_array: Any,
    ) -> None: ...

    def test_nat_comparison_tzawareness(self, comparison_op: Any) -> None: ...

    def test_dti_cmp_str(self, tz_naive_fixture: Any) -> None: ...

    def test_dti_cmp_list(self) -> None: ...

    @pytest.mark.parametrize(
        "other",
        [
            pd.timedelta_range("1D", periods=10),
            pd.timedelta_range("1D", periods=10).to_series(),
            pd.timedelta_range("1D", periods=10).asi8.view("m8[ns]"),
        ],
        ids=lambda x: type(x).__name__,
    )
    def test_dti_cmp_tdi_tzawareness(self, other: Any) -> None: ...

    def test_dti_cmp_object_dtype(self) -> None: ...


class TestDatetime64Arithmetic:
    @pytest.mark.arm_slow
    def test_dt64arr_add_timedeltalike_scalar(
        self, tz_naive_fixture: Any, two_hours: Any, box_with_array: Any
    ) -> None: ...

    def test_dt64arr_sub_timedeltalike_scalar(
        self, tz_naive_fixture: Any, two_hours: Any, box_with_array: Any
    ) -> None: ...

    def test_dt64_array_sub_dt_with_different_timezone(
        self, box_with_array: Any
    ) -> None: ...

    def test_dt64_array_sub_dt64_array_with_different_timezone(
        self, box_with_array: Any
    ) -> None: ...

    def test_dt64arr_add_sub_td64_nat(
        self, box_with_array: Any, tz_naive_fixture: Any
    ) -> None: ...

    def test_dt64arr_add_sub_td64ndarray(
        self, tz_naive_fixture: Any, box_with_array: Any
    ) -> None: ...

    @pytest.mark.parametrize(
        "ts",
        [
            Timestamp("2013-01-01"),
            Timestamp("2013-01-01").to_pydatetime(),
            Timestamp("2013-01-01").to_datetime64(),
            np.datetime64("2013-01-01", "D"),
        ],
    )
    def test_dt64arr_sub_dtscalar(
        self, box_with_array: Any, ts: Any
    ) -> None: ...

    def test_dt64arr_sub_timestamp_tzaware(
        self, box_with_array: Any
    ) -> None: ...

    def test_dt64arr_sub_NaT(
        self, box_with_array: Any, unit: str
    ) -> None: ...

    def test_dt64arr_sub_dt64object_array(
        self,
        performance_warning: Any,
        box_with_array: Any,
        tz_naive_fixture: Any,
    ) -> None: ...

    def test_dt64arr_naive_sub_dt64ndarray(
        self, box_with_array: Any
    ) -> None: ...

    def test_dt64arr_aware_sub_dt64ndarray_raises(
        self, tz_aware_fixture: Any, box_with_array: Any
    ) -> None: ...

    def test_dt64arr_add_dtlike_raises(
        self, tz_naive_fixture: Any, box_with_array: Any
    ) -> None: ...

    @pytest.mark.parametrize(
        "freq", ["h", "D", "W", "2ME", "MS", "QE", "B", None]
    )
    @pytest.mark.parametrize("dtype", [None, "uint8"])
    def test_dt64arr_addsub_intlike(
        self,
        dtype: str | None,
        index_or_series_or_array: Any,
        freq: str | None,
        tz_naive_fixture: Any,
    ) -> None: ...

    @pytest.mark.parametrize(
        "other",
        [3.14, np.array([2.0, 3.0]), Period("2011-01-01", freq="D"), time(1, 2, 3)],
    )
    @pytest.mark.parametrize("dti_freq", [None, "D"])
    def test_dt64arr_add_sub_invalid(
        self, dti_freq: str | None, other: object, box_with_array: Any
    ) -> None: ...

    @pytest.mark.parametrize("pi_freq", ["D", "W", "Q", "h"])
    @pytest.mark.parametrize("dti_freq", [None, "D"])
    def test_dt64arr_add_sub_parr(
        self,
        dti_freq: str | None,
        pi_freq: str,
        box_with_array: Any,
        box_with_array2: Any,
    ) -> None: ...

    @pytest.mark.filterwarnings("ignore::pandas.errors.PerformanceWarning")
    def test_dt64arr_addsub_time_objects_raises(
        self, box_with_array: Any, tz_naive_fixture: Any
    ) -> None: ...

    @pytest.mark.parametrize(
        "dt64_series",
        [
            Series([Timestamp("19900315"), Timestamp("19900315")]),
            Series([NaT, Timestamp("19900315")]),
            Series([NaT, NaT], dtype="datetime64[ns]"),
        ],
    )
    @pytest.mark.parametrize("one", [1, 1.0, np.array(1)])
    def test_dt64_mul_div_numeric_invalid(
        self, one: Any, dt64_series: Series, box_with_array: Any
    ) -> None: ...


class TestDatetime64DateOffsetArithmetic:
    def test_dt64arr_series_add_tick_DateOffset(
        self, box_with_array: Any, unit: str
    ) -> None: ...

    def test_dt64arr_series_sub_tick_DateOffset(
        self, box_with_array: Any
    ) -> None: ...

    @pytest.mark.parametrize(
        "cls_name",
        ["Day", "Hour", "Minute", "Second", "Milli", "Micro", "Nano"],
    )
    def test_dt64arr_add_sub_tick_DateOffset_smoke(
        self, cls_name: str, box_with_array: Any
    ) -> None: ...

    def test_dti_add_tick_tzaware(
        self, tz_aware_fixture: Any, box_with_array: Any
    ) -> None: ...

    def test_dt64arr_add_sub_relativedelta_offsets(
        self, box_with_array: Any, unit: str
    ) -> None: ...

    @pytest.mark.filterwarnings("ignore::pandas.errors.PerformanceWarning")
    @pytest.mark.parametrize(
        "cls_and_kwargs",
        [
            "YearBegin",
            ("YearBegin", {"month": 5}),
            "YearEnd",
            ("YearEnd", {"month": 5}),
            "MonthBegin",
            "MonthEnd",
            "SemiMonthEnd",
            "SemiMonthBegin",
            "Week",
            ("Week", {"weekday": 3}),
            ("Week", {"weekday": 6}),
            "BusinessDay",
            "BDay",
            "QuarterEnd",
            "QuarterBegin",
            "CustomBusinessDay",
            "CDay",
            "CBMonthEnd",
            "CBMonthBegin",
            "BMonthBegin",
            "BMonthEnd",
            "BusinessHour",
            "BYearBegin",
            "BYearEnd",
            "BQuarterBegin",
            ("LastWeekOfMonth", {"weekday": 2}),
            (
                "FY5253Quarter",
                {
                    "qtr_with_extra_week": 1,
                    "startingMonth": 1,
                    "weekday": 2,
                    "variation": "nearest",
                },
            ),
            (
                "FY5253",
                {"weekday": 0, "startingMonth": 2, "variation": "nearest"},
            ),
            ("WeekOfMonth", {"weekday": 2, "week": 2}),
            "Easter",
            ("DateOffset", {"day": 4}),
            ("DateOffset", {"month": 5}),
        ],
    )
    @pytest.mark.parametrize("normalize", [True, False])
    @pytest.mark.parametrize("n", [0, 5])
    @pytest.mark.parametrize("tz", [None, "US/Central"])
    def test_dt64arr_add_sub_DateOffsets(
        self,
        box_with_array: Any,
        n: int,
        normalize: bool,
        cls_and_kwargs: str | tuple[str, dict[str, Any]],
        unit: str,
        tz: str | None,
    ) -> None: ...

    @pytest.mark.parametrize(
        "other",
        [
            [pd.offsets.MonthEnd(), pd.offsets.Day(n=2)],
            [pd.offsets.DateOffset(years=1), pd.offsets.MonthEnd()],
            [pd.offsets.DateOffset(years=1), pd.offsets.DateOffset(years=1)],
        ],
    )
    @pytest.mark.parametrize(
        "op", [operator.add, roperator.radd, operator.sub]
    )
    def test_dt64arr_add_sub_offset_array(
        self,
        performance_warning: Any,
        tz_naive_fixture: Any,
        box_with_array: Any,
        op: Any,
        other: list[Any],
    ) -> None: ...

    @pytest.mark.parametrize(
        "op, offset, exp, exp_freq",
        [
            (
                "__add__",
                DateOffset(months=3, days=10),
                [
                    Timestamp("2014-04-11"),
                    Timestamp("2015-04-11"),
                    Timestamp("2016-04-11"),
                    Timestamp("2017-04-11"),
                ],
                None,
            ),
            (
                "__add__",
                DateOffset(months=3),
                [
                    Timestamp("2014-04-01"),
                    Timestamp("2015-04-01"),
                    Timestamp("2016-04-01"),
                    Timestamp("2017-04-01"),
                ],
                "YS-APR",
            ),
            (
                "__sub__",
                DateOffset(months=3, days=10),
                [
                    Timestamp("2013-09-21"),
                    Timestamp("2014-09-21"),
                    Timestamp("2015-09-21"),
                    Timestamp("2016-09-21"),
                ],
                None,
            ),
            (
                "__sub__",
                DateOffset(months=3),
                [
                    Timestamp("2013-10-01"),
                    Timestamp("2014-10-01"),
                    Timestamp("2015-10-01"),
                    Timestamp("2016-10-01"),
                ],
                "YS-OCT",
            ),
        ],
    )
    def test_dti_add_sub_nonzero_mth_offset(
        self,
        op: str,
        offset: DateOffset,
        exp: list[Timestamp],
        exp_freq: str | None,
        tz_aware_fixture: Any,
        box_with_array: Any,
    ) -> None: ...

    def test_dt64arr_series_add_DateOffset_with_milli(self) -> None: ...


class TestDatetime64OverflowHandling:
    def test_dt64_overflow_masking(self, box_with_array: Any) -> None: ...

    def test_dt64_series_arith_overflow(self) -> None: ...

    def test_datetimeindex_sub_timestamp_overflow(self) -> None: ...

    def test_datetimeindex_sub_datetimeindex_overflow(self) -> None: ...


class TestTimestampSeriesArithmetic:
    def test_empty_series_add_sub(self, box_with_array: Any) -> None: ...

    def test_operators_datetimelike(self) -> None: ...

    def test_dt64ser_sub_datetime_dtype(self, unit: str) -> None: ...

    @pytest.mark.parametrize(
        "left, right, op_fail",
        [
            [
                [Timestamp("20111230"), Timestamp("20120101"), NaT],
                [
                    Timestamp("20111231"),
                    Timestamp("20120102"),
                    Timestamp("20120104"),
                ],
                ["__sub__", "__rsub__"],
            ],
            [
                [Timestamp("20111230"), Timestamp("20120101"), NaT],
                [timedelta(minutes=5, seconds=3), timedelta(minutes=5, seconds=3), NaT],
                ["__add__", "__radd__", "__sub__"],
            ],
            [
                [
                    Timestamp("20111230", tz="US/Eastern"),
                    Timestamp("20111230", tz="US/Eastern"),
                    NaT,
                ],
                [timedelta(minutes=5, seconds=3), NaT, timedelta(minutes=5, seconds=3)],
                ["__add__", "__radd__", "__sub__"],
            ],
        ],
    )
    def test_operators_datetimelike_invalid(
        self,
        left: list[Any],
        right: list[Any],
        op_fail: list[str],
        all_arithmetic_operators: str,
    ) -> None: ...

    def test_sub_single_tz(self, unit: str) -> None: ...

    def test_dt64tz_series_sub_dtitz(self) -> None: ...

    def test_sub_datetime_compat(self, unit: str) -> None: ...

    def test_dt64_series_add_mixed_tick_DateOffset(self) -> None: ...

    def test_datetime64_ops_nat(self, unit: str) -> None: ...

    def test_operators_datetimelike_with_timezones(self) -> None: ...


class TestDatetimeIndexArithmetic:
    def test_dti_add_tdi(self, tz_naive_fixture: Any) -> None: ...

    def test_dti_iadd_tdi(self, tz_naive_fixture: Any) -> None: ...

    def test_dti_sub_tdi(self, tz_naive_fixture: Any) -> None: ...

    def test_dti_isub_tdi(
        self, tz_naive_fixture: Any, unit: str
    ) -> None: ...

    def test_dta_add_sub_index(self, tz_naive_fixture: Any) -> None: ...

    def test_sub_dti_dti(self, unit: str) -> None: ...

    @pytest.mark.parametrize("op", [operator.add, operator.sub])
    def test_timedelta64_equal_timedelta_supported_ops(
        self, op: Any, box_with_array: Any
    ) -> None: ...

    def test_ops_nat_mixed_datetime64_timedelta64(self) -> None: ...

    def test_ufunc_coercions(self, unit: str) -> None: ...

    def test_dti_add_series(
        self, tz_naive_fixture: Any, names: tuple[Any, ...]
    ) -> None: ...

    @pytest.mark.parametrize(
        "op", [operator.add, roperator.radd, operator.sub]
    )
    def test_dti_addsub_offset_arraylike(
        self,
        performance_warning: Any,
        tz_naive_fixture: Any,
        names: tuple[Any, ...],
        op: Any,
        index_or_series: Any,
    ) -> None: ...

    @pytest.mark.parametrize("other_box", [pd.Index, np.array])
    def test_dti_addsub_object_arraylike(
        self,
        performance_warning: Any,
        tz_naive_fixture: Any,
        box_with_array: Any,
        other_box: Any,
    ) -> None: ...


@pytest.mark.parametrize("years", [-1, 0, 1])
@pytest.mark.parametrize("months", [-2, 0, 2])
def test_shift_months(years: int, months: int, unit: str) -> None: ...

def test_dt64arr_addsub_object_dtype_2d(performance_warning: Any) -> None: ...

def test_non_nano_dt64_addsub_np_nat_scalars() -> None: ...

def test_non_nano_dt64_addsub_np_nat_scalars_unitless() -> None: ...

def test_non_nano_dt64_addsub_np_nat_scalars_unsupported_unit() -> None: ...