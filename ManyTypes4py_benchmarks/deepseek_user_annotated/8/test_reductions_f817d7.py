from datetime import timedelta
from decimal import Decimal
import re
from typing import Any, Callable, Dict, List, Optional, Pattern, Sequence, Union

from dateutil.tz import tzlocal
import numpy as np
import pytest

from pandas.compat import (
    IS64,
    is_platform_windows,
)
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    Categorical,
    CategoricalDtype,
    DataFrame,
    DatetimeIndex,
    Index,
    PeriodIndex,
    RangeIndex,
    Series,
    Timestamp,
    date_range,
    isna,
    notna,
    to_datetime,
    to_timedelta,
)
import pandas._testing as tm
from pandas.core import (
    algorithms,
    nanops,
)

is_windows_np2_or_is32: bool = (is_platform_windows() and not np_version_gt2) or not IS64
is_windows_or_is32: bool = is_platform_windows() or not IS64


def make_skipna_wrapper(
    alternative: Callable, skipna_alternative: Optional[Callable] = None
) -> Callable:
    """
    Create a function for calling on an array.

    Parameters
    ----------
    alternative : function
        The function to be called on the array with no NaNs.
        Only used when 'skipna_alternative' is None.
    skipna_alternative : function
        The function to be called on the original array

    Returns
    -------
    function
    """
    if skipna_alternative:
        def skipna_wrapper(x: Any) -> Any:
            return skipna_alternative(x.values)
    else:
        def skipna_wrapper(x: Any) -> Any:
            nona = x.dropna()
            if len(nona) == 0:
                return np.nan
            return alternative(nona)

    return skipna_wrapper


def assert_stat_op_calc(
    opname: str,
    alternative: Callable,
    frame: DataFrame,
    has_skipna: bool = True,
    check_dtype: bool = True,
    check_dates: bool = False,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    skipna_alternative: Optional[Callable] = None,
) -> None:
    """
    Check that operator opname works as advertised on frame

    Parameters
    ----------
    opname : str
        Name of the operator to test on frame
    alternative : function
        Function that opname is tested against; i.e. "frame.opname()" should
        equal "alternative(frame)".
    frame : DataFrame
        The object that the tests are executed on
    has_skipna : bool, default True
        Whether the method "opname" has the kwarg "skip_na"
    check_dtype : bool, default True
        Whether the dtypes of the result of "frame.opname()" and
        "alternative(frame)" should be checked.
    check_dates : bool, default false
        Whether opname should be tested on a Datetime Series
    rtol : float, default 1e-5
        Relative tolerance.
    atol : float, default 1e-8
        Absolute tolerance.
    skipna_alternative : function, default None
        NaN-safe version of alternative
    """
    f = getattr(frame, opname)

    if check_dates:
        df = DataFrame({"b": date_range("1/1/2001", periods=2)})
        with tm.assert_produces_warning(None):
            result = getattr(df, opname)()
        assert isinstance(result, Series)

        df["a"] = range(len(df))
        with tm.assert_produces_warning(None):
            result = getattr(df, opname)()
        assert isinstance(result, Series)
        assert len(result)

    if has_skipna:
        def wrapper(x: Any) -> Any:
            return alternative(x.values)

        skipna_wrapper = make_skipna_wrapper(alternative, skipna_alternative)
        result0 = f(axis=0, skipna=False)
        result1 = f(axis=1, skipna=False)
        tm.assert_series_equal(
            result0, frame.apply(wrapper), check_dtype=check_dtype, rtol=rtol, atol=atol
        )
        tm.assert_series_equal(
            result1,
            frame.apply(wrapper, axis=1),
            rtol=rtol,
            atol=atol,
        )
    else:
        skipna_wrapper = alternative

    result0 = f(axis=0)
    result1 = f(axis=1)
    tm.assert_series_equal(
        result0,
        frame.apply(skipna_wrapper),
        check_dtype=check_dtype,
        rtol=rtol,
        atol=atol,
    )

    if opname in ["sum", "prod"]:
        expected = frame.apply(skipna_wrapper, axis=1)
        tm.assert_series_equal(
            result1, expected, check_dtype=False, rtol=rtol, atol=atol
        )

    # check dtypes
    if check_dtype:
        lcd_dtype = frame.values.dtype
        assert lcd_dtype == result0.dtype
        assert lcd_dtype == result1.dtype

    # bad axis
    with pytest.raises(ValueError, match="No axis named 2"):
        f(axis=2)

    # all NA case
    if has_skipna:
        all_na = frame * np.nan
        r0 = getattr(all_na, opname)(axis=0)
        r1 = getattr(all_na, opname)(axis=1)
        if opname in ["sum", "prod"]:
            unit = 1 if opname == "prod" else 0  # result for empty sum/prod
            expected = Series(unit, index=r0.index, dtype=r0.dtype)
            tm.assert_series_equal(r0, expected)
            expected = Series(unit, index=r1.index, dtype=r1.dtype)
            tm.assert_series_equal(r1, expected)


@pytest.fixture
def bool_frame_with_na() -> DataFrame:
    """
    Fixture for DataFrame of booleans with index of unique strings

    Columns are ['A', 'B', 'C', 'D']; some entries are missing
    """
    df = DataFrame(
        np.concatenate(
            [np.ones((15, 4), dtype=bool), np.zeros((15, 4), dtype=bool)], axis=0
        ),
        index=Index([f"foo_{i}" for i in range(30)], dtype=object),
        columns=Index(list("ABCD"), dtype=object),
        dtype=object,
    )
    # set some NAs
    df.iloc[5:10] = np.nan
    df.iloc[15:20, -2:] = np.nan
    return df


@pytest.fixture
def float_frame_with_na() -> DataFrame:
    """
    Fixture for DataFrame of floats with index of unique strings

    Columns are ['A', 'B', 'C', 'D']; some entries are missing
    """
    df = DataFrame(
        np.random.default_rng(2).standard_normal((30, 4)),
        index=Index([f"foo_{i}" for i in range(30)], dtype=object),
        columns=Index(list("ABCD"), dtype=object),
    )
    # set some NAs
    df.iloc[5:10] = np.nan
    df.iloc[15:20, -2:] = np.nan
    return df


class TestDataFrameAnalytics:
    # ---------------------------------------------------------------------
    # Reductions
    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize(
        "opname",
        [
            "count",
            "sum",
            "mean",
            "product",
            "median",
            "min",
            "max",
            "nunique",
            "var",
            "std",
            "sem",
            pytest.param("skew", marks=td.skip_if_no("scipy")),
            pytest.param("kurt", marks=td.skip_if_no("scipy")),
        ],
    )
    def test_stat_op_api_float_string_frame(
        self, float_string_frame: DataFrame, axis: int, opname: str
    ) -> None:
        if (opname in ("sum", "min", "max") and axis == 0) or opname in (
            "count",
            "nunique",
        ):
            getattr(float_string_frame, opname)(axis=axis)
        else:
            if opname in ["var", "std", "sem", "skew", "kurt"]:
                msg = "could not convert string to float: 'bar'"
            elif opname == "product":
                if axis == 1:
                    msg = "can't multiply sequence by non-int of type 'float'"
                else:
                    msg = "can't multiply sequence by non-int of type 'str'"
            elif opname == "sum":
                msg = r"unsupported operand type\(s\) for \+: 'float' and 'str'"
            elif opname == "mean":
                if axis == 0:
                    # different message on different builds
                    msg = "|".join(
                        [
                            r"Could not convert \['.*'\] to numeric",
                            "Could not convert string '(bar){30}' to numeric",
                        ]
                    )
                else:
                    msg = r"unsupported operand type\(s\) for \+: 'float' and 'str'"
            elif opname in ["min", "max"]:
                msg = "'[><]=' not supported between instances of 'float' and 'str'"
            elif opname == "median":
                msg = re.compile(
                    r"Cannot convert \[.*\] to numeric|does not support|Cannot perform",
                    flags=re.S,
                )
            if not isinstance(msg, re.Pattern):
                msg = msg + "|does not support|Cannot perform reduction"
            with pytest.raises(TypeError, match=msg):
                getattr(float_string_frame, opname)(axis=axis)
        if opname != "nunique":
            getattr(float_string_frame, opname)(axis=axis, numeric_only=True)

    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize(
        "opname",
        [
            "count",
            "sum",
            "mean",
            "product",
            "median",
            "min",
            "max",
            "var",
            "std",
            "sem",
            pytest.param("skew", marks=td.skip_if_no("scipy")),
            pytest.param("kurt", marks=td.skip_if_no("scipy")),
        ],
    )
    def test_stat_op_api_float_frame(
        self, float_frame: DataFrame, axis: int, opname: str
    ) -> None:
        getattr(float_frame, opname)(axis=axis, numeric_only=False)

    def test_stat_op_calc(
        self, float_frame_with_na: DataFrame, mixed_float_frame: DataFrame
    ) -> None:
        def count(s: Series) -> int:
            return notna(s).sum()

        def nunique(s: Series) -> int:
            return len(algorithms.unique1d(s.dropna()))

        def var(x: np.ndarray) -> float:
            return np.var(x, ddof=1)

        def std(x: np.ndarray) -> float:
            return np.std(x, ddof=1)

        def sem(x: np.ndarray) -> float:
            return np.std(x, ddof=1) / np.sqrt(len(x))

        assert_stat_op_calc(
            "nunique",
            nunique,
            float_frame_with_na,
            has_skipna=False,
            check_dtype=False,
            check_dates=True,
        )

        # GH#32571: rol needed for flaky CI builds
        # mixed types (with upcasting happening)
        assert_stat_op_calc(
            "sum",
            np.sum,
            mixed_float_frame.astype("float32"),
            check_dtype=False,
            rtol=1e-3,
        )

        assert_stat_op_calc(
            "sum", np.sum, float_frame_with_na, skipna_alternative=np.nansum
        )
        assert_stat_op_calc("mean", np.mean, float_frame_with_na, check_dates=True)
        assert_stat_op_calc(
            "product", np.prod, float_frame_with_na, skipna_alternative=np.nanprod
        )

        assert_stat_op_calc("var", var, float_frame_with_na)
        assert_stat_op_calc("std", std, float_frame_with_na)
        assert_stat_op_calc("sem", sem, float_frame_with_na)

        assert_stat_op_calc(
            "count",
            count,
            float_frame_with_na,
            has_skipna=False,
            check_dtype=False,
            check_dates=True,
        )

    def test_stat_op_calc_skew_kurtosis(self, float_frame_with_na: DataFrame) -> None:
        sp_stats = pytest.importorskip("scipy.stats")

        def skewness(x: np.ndarray) -> float:
            if len(x) < 3:
                return np.nan
            return sp_stats.skew(x, bias=False)

        def kurt(x: np.ndarray) -> float:
            if len(x) < 4:
                return np.nan
            return sp_stats.kurtosis(x, bias=False)

        assert_stat_op_calc("skew", skewness, float_frame_with_na)
        assert_stat_op_calc("kurt", kurt, float_frame_with_na)

    def test_median(
        self, float_frame_with_na: DataFrame, int_frame: DataFrame
    ) -> None:
        def wrapper(x: np.ndarray) -> float:
            if isna(x).any():
                return np.nan
            return np.median(x)

        assert_stat_op_calc("median", wrapper, float_frame_with_na, check_dates=True)
        assert_stat_op_calc(
            "median", wrapper, int_frame, check_dtype=False, check_dates=True
        )

    @pytest.mark.parametrize(
        "method", ["sum", "mean", "prod", "var", "std", "skew", "min", "max"]
    )
    @pytest.mark.parametrize(
        "df",
        [
            DataFrame(
                {
                    "a": [
                        -0.00049987540199591344,
                        -0.0016467257772919831,
                        0.00067695870775883013,
                    ],
                    "b": [-0, -0, 0.0],
                    "c": [
                        0.00031111847529610595,
                        0.0014902627951905339,
                        -0.00094099200035979691,
                    ],
                },
                index=["foo", "bar", "baz"],
                dtype="O",
            ),
            DataFrame({0: [np.nan, 2], 1: [np.nan, 3], 2: [np.nan, 4]}, dtype=object),
        ],
    )
    @pytest.mark.filterwarnings("ignore:Mismatched null-like values:FutureWarning")
    def test_stat_operators_attempt_obj_array(
        self, method: str, df: DataFrame, axis: int
    ) -> None:
        # GH#676
        assert df.values.dtype == np.object_
        result = getattr(df, method)(axis=axis)
        expected = getattr(df.astype("f8"), method)(axis=axis).astype(object)
        if axis in [1, "columns"] and method in ["min", "max"]:
            expected[expected.isna()] = None
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("op", ["mean", "std", "var", "skew", "kurt", "sem"])
    def test_mixed_ops(self, op: str) -> None:
        # GH#16116
        df = DataFrame(
            {
                "int": [1, 2, 3, 4],
                "float": [1.0, 2.0, 3.0, 4.0],
                "str": ["a", "b", "c", "d"],
            }
        )
        msg = "|".join(
            [
                "Could not convert",
                "could not convert",
                "can't multiply sequence by non-int",
                "does not support",
                "Cannot perform",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            getattr(df, op)()

        with pd.option_context("use_bottleneck", False):
            with pytest.raises(TypeError, match=msg):
                getattr(df, op)()

    def test_reduce_mixed_frame(self) -> None:
        # GH 6806
        df = DataFrame(
            {
                "bool_data": [True, True, False, False, False],
                "int_data": [10, 20, 30, 40, 50],
                "string_data": ["a", "b", "c", "d", "e"],
            }
        )
        df.reindex(columns=["bool_data", "int_data", "string_data"])
        test = df.sum(axis=0)
        tm.assert_numpy_array_equal(
            test.values, np.array([2, 150, "abcde"], dtype=object)
        )
        alt = df.T.sum(axis=1)
        tm.assert_series_equal(test, alt)

    def test_nunique(self) -> None:
        df = DataFrame({"A": [1, 1, 1], "B": [1, 2, 3], "C": [1, np.nan, 3]})
        tm.assert_series_equal(df.nunique(), Series({"A": 1, "B": 3, "C": 2}))
        tm.assert_series_equal(
            df.nunique(dropna=False), Series({"A": 1, "B": 3, "C": 3})
        )
        tm.assert_series_equal(df.nunique(axis=1), Series([1, 2, 2]))
        tm.assert_series_equal(df.nunique(axis=1, dropna=False