from datetime import timedelta
from decimal import Decimal
import re
from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple, Union

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

is_windows_np2_or_is32 = (is_platform_windows() and not np_version_gt2) or not IS64
is_windows_or_is32 = is_platform_windows() or not IS64


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

        def skipna_wrapper(x: Series) -> Any:
            return skipna_alternative(x.values)

    else:

        def skipna_wrapper(x: Series) -> Any:
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

        def wrapper(x: Series) -> Any:
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
