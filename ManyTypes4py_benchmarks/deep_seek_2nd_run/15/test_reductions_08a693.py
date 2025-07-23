from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import pytest
import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    DatetimeIndex,
    Index,
    NaT,
    Period,
    PeriodIndex,
    RangeIndex,
    Series,
    Timedelta,
    TimedeltaIndex,
    Timestamp,
    date_range,
    isna,
    period_range,
    timedelta_range,
    to_timedelta,
)
from pandas._testing import tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics


def get_objs() -> List[Union[Index, Series]]:
    indexes: List[Index] = [
        Index([True, False] * 5, name="a"),
        Index(np.arange(10), dtype=np.int64, name="a"),
        Index(np.arange(10), dtype=np.float64, name="a"),
        DatetimeIndex(date_range("2020-01-01", periods=10), name="a"),
        DatetimeIndex(date_range("2020-01-01", periods=10), name="a").tz_localize(
            tz="US/Eastern"
        ),
        PeriodIndex(period_range("2020-01-01", periods=10, freq="D"), name="a"),
        Index([str(i) for i in range(10)], name="a"),
    ]
    arr: np.ndarray = np.random.default_rng(2).standard_normal(10)
    series: List[Series] = [Series(arr, index=idx, name="a") for idx in indexes]
    objs: List[Union[Index, Series]] = indexes + series
    return objs


class TestReductions:
    @pytest.mark.filterwarnings("ignore:Period with BDay freq is deprecated:FutureWarning")
    @pytest.mark.parametrize("opname", ["max", "min"])
    @pytest.mark.parametrize("obj", get_objs())
    def test_ops(self, opname: str, obj: Union[Index, Series]) -> None:
        result = getattr(obj, opname)()
        if not isinstance(obj, PeriodIndex):
            if isinstance(obj.values, ArrowStringArrayNumpySemantics):
                expected = getattr(np.array(obj.values), opname)()
            else:
                expected = getattr(obj.values, opname)()
        else:
            expected = Period(ordinal=getattr(obj.asi8, opname)(), freq=obj.freq
        if getattr(obj, "tz", None) is not None:
            expected = expected.astype("M8[ns]").astype("int64")
            assert result._value == expected
        else:
            assert result == expected

    @pytest.mark.parametrize("opname", ["max", "min"])
    @pytest.mark.parametrize(
        "dtype, val",
        [
            ("object", 2.0),
            ("float64", 2.0),
            ("datetime64[ns]", datetime(2011, 11, 1)),
            ("Int64", 2),
            ("boolean", True),
        ],
    )
    def test_nanminmax(
        self,
        opname: str,
        dtype: str,
        val: Any,
        index_or_series: Union[Index, Series],
    ) -> None:
        klass = index_or_series

        def check_missing(res: Any) -> bool:
            if dtype == "datetime64[ns]":
                return res is NaT
            elif dtype in ["Int64", "boolean"]:
                return res is pd.NA
            else:
                return isna(res)

        obj = klass([None], dtype=dtype)
        assert check_missing(getattr(obj, opname)())
        assert check_missing(getattr(obj, opname)(skipna=False))
        obj = klass([], dtype=dtype)
        assert check_missing(getattr(obj, opname)())
        assert check_missing(getattr(obj, opname)(skipna=False))
        if dtype == "object":
            return
        obj = klass([None, val], dtype=dtype)
        assert getattr(obj, opname)() == val
        assert check_missing(getattr(obj, opname)(skipna=False))
        obj = klass([None, val, None], dtype=dtype)
        assert getattr(obj, opname)() == val
        assert check_missing(getattr(obj, opname)(skipna=False))

    @pytest.mark.parametrize("opname", ["max", "min"])
    def test_nanargminmax(self, opname: str, index_or_series: Union[Index, Series]) -> None:
        klass = index_or_series
        arg_op = "arg" + opname if klass is Index else "idx" + opname
        obj = klass([NaT, datetime(2011, 11, 1)])
        assert getattr(obj, arg_op)() == 1
        with pytest.raises(ValueError, match="Encountered an NA value"):
            getattr(obj, arg_op)(skipna=False)
        obj = klass([NaT, datetime(2011, 11, 1), NaT])
        assert getattr(obj, arg_op)() == 1
        with pytest.raises(ValueError, match="Encountered an NA value"):
            getattr(obj, arg_op)(skipna=False)

    @pytest.mark.parametrize("opname", ["max", "min"])
    @pytest.mark.parametrize("dtype", ["M8[ns]", "datetime64[ns, UTC]"])
    def test_nanops_empty_object(
        self, opname: str, index_or_series: Union[Index, Series], dtype: str
    ) -> None:
        klass = index_or_series
        arg_op = "arg" + opname if klass is Index else "idx" + opname
        obj = klass([], dtype=dtype)
        assert getattr(obj, opname)() is NaT
        assert getattr(obj, opname)(skipna=False) is NaT
        with pytest.raises(ValueError, match="empty sequence"):
            getattr(obj, arg_op)()
        with pytest.raises(ValueError, match="empty sequence"):
            getattr(obj, arg_op)(skipna=False)

    def test_argminmax(self) -> None:
        obj = Index(np.arange(5, dtype="int64"))
        assert obj.argmin() == 0
        assert obj.argmax() == 4
        obj = Index([np.nan, 1, np.nan, 2])
        assert obj.argmin() == 1
        assert obj.argmax() == 3
        with pytest.raises(ValueError, match="Encountered an NA value"):
            obj.argmin(skipna=False)
        with pytest.raises(ValueError, match="Encountered an NA value"):
            obj.argmax(skipna=False)
        obj = Index([np.nan])
        with pytest.raises(ValueError, match="Encountered all NA values"):
            obj.argmin()
        with pytest.raises(ValueError, match="Encountered all NA values"):
            obj.argmax()
        with pytest.raises(ValueError, match="Encountered an NA value"):
            obj.argmin(skipna=False)
        with pytest.raises(ValueError, match="Encountered an NA value"):
            obj.argmax(skipna=False)
        obj = Index([NaT, datetime(2011, 11, 1), datetime(2011, 11, 2), NaT])
        assert obj.argmin() == 1
        assert obj.argmax() == 2
        with pytest.raises(ValueError, match="Encountered an NA value"):
            obj.argmin(skipna=False)
        with pytest.raises(ValueError, match="Encountered an NA value"):
            obj.argmax(skipna=False)
        obj = Index([NaT])
        with pytest.raises(ValueError, match="Encountered all NA values"):
            obj.argmin()
        with pytest.raises(ValueError, match="Encountered all NA values"):
            obj.argmax()
        with pytest.raises(ValueError, match="Encountered an NA value"):
            obj.argmin(skipna=False)
        with pytest.raises(ValueError, match="Encountered an NA value"):
            obj.argmax(skipna=False)

    @pytest.mark.parametrize("op, expected_col", [["max", "a"], ["min", "b"]])
    def test_same_tz_min_max_axis_1(self, op: str, expected_col: str) -> None:
        df = DataFrame(
            date_range("2016-01-01 00:00:00", periods=3, tz="UTC"), columns=["a"]
        )
        df["b"] = df.a.subtract(Timedelta(seconds=3600))
        result = getattr(df, op)(axis=1)
        expected = df[expected_col].rename(None)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("func", ["maximum", "minimum"])
    def test_numpy_reduction_with_tz_aware_dtype(
        self, tz_aware_fixture: Any, func: str
    ) -> None:
        tz = tz_aware_fixture
        arg = pd.to_datetime(["2019"]).tz_localize(tz)
        expected = Series(arg)
        result = getattr(np, func)(expected, expected)
        tm.assert_series_equal(result, expected)

    def test_nan_int_timedelta_sum(self) -> None:
        df = DataFrame(
            {
                "A": Series([1, 2, NaT], dtype="timedelta64[ns]"),
                "B": Series([1, 2, np.nan], dtype="Int64"),
            }
        )
        expected = Series({"A": Timedelta(3), "B": 3})
        result = df.sum()
        tm.assert_series_equal(result, expected)


class TestIndexReductions:
    @pytest.mark.parametrize(
        "start,stop,step",
        [
            (0, 400, 3),
            (500, 0, -6),
            (-(10**6), 10**6, 4),
            (10**6, -(10**6), -4),
            (0, 10, 20),
        ],
    )
    def test_max_min_range(self, start: int, stop: int, step: int) -> None:
        idx = RangeIndex(start, stop, step)
        expected = idx._values.max()
        result = idx.max()
        assert result == expected
        result2 = idx.max(skipna=False)
        assert result2 == expected
        expected = idx._values.min()
        result = idx.min()
        assert result == expected
        result2 = idx.min(skipna=False)
        assert result2 == expected
        idx = RangeIndex(start, stop, -step)
        assert isna(idx.max())
        assert isna(idx.min())

    def test_minmax_timedelta64(self) -> None:
        idx1 = TimedeltaIndex(["1 days", "2 days", "3 days"])
        assert idx1.is_monotonic_increasing
        idx2 = TimedeltaIndex(["1 days", np.nan, "3 days", "NaT"])
        assert not idx2.is_monotonic_increasing
        for idx in [idx1, idx2]:
            assert idx.min() == Timedelta("1 days")
            assert idx.max() == Timedelta("3 days")
            assert idx.argmin() == 0
            assert idx.argmax() == 2

    @pytest.mark.parametrize("op", ["min", "max"])
    def test_minmax_timedelta_empty_or_na(self, op: str) -> None:
        obj = TimedeltaIndex([])
        assert getattr(obj, op)() is NaT
        obj = TimedeltaIndex([NaT])
        assert getattr(obj, op)() is NaT
        obj = TimedeltaIndex([NaT, NaT, NaT])
        assert getattr(obj, op)() is NaT

    def test_numpy_minmax_timedelta64(self) -> None:
        td = timedelta_range("16815 days", "16820 days", freq="D")
        assert np.min(td) == Timedelta("16815 days")
        assert np.max(td) == Timedelta("16820 days")
        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.min(td, out=0)
        with pytest.raises(ValueError, match=errmsg):
            np.max(td, out=0)
        assert np.argmin(td) == 0
        assert np.argmax(td) == 5
        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.argmin(td, out=0)
        with pytest.raises(ValueError, match=errmsg):
            np.argmax(td, out=0)

    def test_timedelta_ops(self) -> None:
        s = Series(
            [Timestamp("20130101") + timedelta(seconds=i * i) for i in range(10)]
        )
        td = s.diff()
        result = td.mean()
        expected = to_timedelta(timedelta(seconds=9))
        assert result == expected
        result = td.to_frame().mean()
        assert result[0] == expected
        result = td.quantile(0.1)
        expected = Timedelta(np.timedelta64(2600, "ms"))
        assert result == expected
        result = td.median()
        expected = to_timedelta("00:00:09")
        assert result == expected
        result = td.to_frame().median()
        assert result[0] == expected
        result = td.sum()
        expected = to_timedelta("00:01:21")
        assert result == expected
        result = td.to_frame().sum()
        assert result[0] == expected
        result = td.std()
        expected = to_timedelta(Series(td.dropna().values).std())
        assert result == expected
        result = td.to_frame().std()
        assert result[0] == expected
        s = Series([Timestamp("2015-02-03"), Timestamp("2015-02-07")])
        assert s.diff().median() == timedelta(days=4)
        s = Series(
            [Timestamp("2015-02-03"), Timestamp("2015-02-07"), Timestamp("2015-02-15")]
        )
        assert s.diff().median() == timedelta(days=6)

    @pytest.mark.parametrize(
        "opname", ["skew", "kurt", "sem", "prod", "var"]
    )
    def test_invalid_td64_reductions(self, opname: str) -> None:
        s = Series(
            [Timestamp("20130101") + timedelta(seconds=i * i) for i in range(10)]
        )
        td = s.diff()
        msg = "|".join(
            [
                f"reduction operation '{opname}' not allowed for this dtype",
                f"cannot perform {opname} with type timedelta64\\[ns\\]",
                f"does not support operation '{opname}'",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            getattr(td, opname)()
        with pytest.raises(TypeError, match=msg):
            getattr(td.to_frame(), opname)(numeric_only=False)

    def test_minmax_tz(self, tz_naive_fixture: Any) -> None:
        tz = tz_naive_fixture
        idx1 = DatetimeIndex(["2011-01-01", "2011-01-02", "2011-01-03"], tz=tz)
        assert idx1.is_monotonic_increasing
        idx2 = DatetimeIndex(
            ["2011-01-01", NaT, "2011-01-03", "2011-01-02", NaT], tz=tz
        )
        assert not idx2.is_monotonic_increasing
        for idx in [idx1, idx2]:
            assert idx.min() == Timestamp("2011-01-01", tz=tz)
            assert idx.max() == Timestamp("2011-01-03", tz=tz)
            assert idx.argmin() == 0
            assert idx.argmax() == 2

    @pytest.mark.parametrize("op", ["min", "max"])
    def test_minmax_nat_datetime64(self, op: str) -> None:
        obj = DatetimeIndex([])
        assert isna(getattr(obj, op)())
        obj = DatetimeIndex([NaT])
        assert isna(getattr(obj, op)())
        obj = DatetimeIndex([NaT, NaT, NaT])
        assert isna(getattr(obj, op)())

    def test_numpy_minmax_integer(self) -> None:
        idx = Index([1, 2, 3])
        expected = idx.values.max()
        result = np.max(idx)
        assert result == expected
        expected = idx.values.min()
        result = np.min(idx)
        assert result == expected
        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.min(idx, out=0)
        with pytest.raises(ValueError, match=errmsg):
            np.max(idx, out=0)
        expected = idx.values.argmax()
        result = np.argmax(idx)
        assert result == expected
        expected = idx.values.argmin()
        result = np.argmin(idx)
        assert result == expected
        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.argmin(idx, out=0)
        with pytest.raises(