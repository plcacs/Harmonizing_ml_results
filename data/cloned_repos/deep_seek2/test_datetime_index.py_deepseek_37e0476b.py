from datetime import datetime
from functools import partial
import zoneinfo
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pytest

from pandas._libs import lib
from pandas._typing import DatetimeNaTType
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    Series,
    Timedelta,
    Timestamp,
    isna,
    notna,
)
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import (
    Period,
    period_range,
)
from pandas.core.resample import (
    DatetimeIndex,
    _get_timestamp_range_edges,
)

from pandas.tseries import offsets
from pandas.tseries.offsets import Minute


@pytest.fixture
def simple_date_range_series() -> Callable[[str, str, str], Series]:
    """
    Series with date range index and random data for test purposes.
    """

    def _simple_date_range_series(start: str, end: str, freq: str = "D") -> Series:
        rng = date_range(start, end, freq=freq)
        return Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)

    return _simple_date_range_series


def test_custom_grouper(unit: str) -> None:
    index = date_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq="Min")
    dti = index.as_unit(unit)
    s = Series(np.array([1] * len(dti)), index=dti, dtype="int64")

    b = Grouper(freq=Minute(5))
    g = s.groupby(b)

    # check all cython functions work
    g.ohlc()  # doesn't use _cython_agg_general
    funcs = ["sum", "mean", "prod", "min", "max", "var"]
    for f in funcs:
        g._cython_agg_general(f, alt=None, numeric_only=True)

    b = Grouper(freq=Minute(5), closed="right", label="right")
    g = s.groupby(b)
    # check all cython functions work
    g.ohlc()  # doesn't use _cython_agg_general
    funcs = ["sum", "mean", "prod", "min", "max", "var"]
    for f in funcs:
        g._cython_agg_general(f, alt=None, numeric_only=True)

    assert g.ngroups == 2593
    assert notna(g.mean()).all()

    # construct expected val
    arr = [1] + [5] * 2592
    idx = dti[0:-1:5]
    idx = idx.append(dti[-1:])
    idx = DatetimeIndex(idx, freq="5min").as_unit(unit)
    expect = Series(arr, index=idx)

    # GH2763 - return input dtype if we can
    result = g.agg("sum")
    tm.assert_series_equal(result, expect)


def test_custom_grouper_df(unit: str) -> None:
    index = date_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq="D")
    b = Grouper(freq=Minute(5), closed="right", label="right")
    dti = index.as_unit(unit)
    df = DataFrame(
        np.random.default_rng(2).random((len(dti), 10)), index=dti, dtype="float64"
    )
    r = df.groupby(b).agg("sum")

    assert len(r.columns) == 10
    assert len(r.index) == 2593


@pytest.mark.parametrize(
    "closed, expected",
    [
        (
            "right",
            lambda s: Series(
                [s.iloc[0], s[1:6].mean(), s[6:11].mean(), s[11:].mean()],
                index=date_range("1/1/2000", periods=4, freq="5min", name="index"),
            ),
        ),
        (
            "left",
            lambda s: Series(
                [s[:5].mean(), s[5:10].mean(), s[10:].mean()],
                index=date_range(
                    "1/1/2000 00:05", periods=3, freq="5min", name="index"
                ),
            ),
        ),
    ],
)
def test_resample_basic(closed: str, expected: Callable[[Series], Series], unit: str) -> None:
    index = date_range("1/1/2000 00:00:00", "1/1/2000 00:13:00", freq="Min")
    s = Series(range(len(index)), index=index)
    s.index.name = "index"
    s.index = s.index.as_unit(unit)
    expected = expected(s)
    expected.index = expected.index.as_unit(unit)
    result = s.resample("5min", closed=closed, label="right").mean()
    tm.assert_series_equal(result, expected)


def test_resample_integerarray(unit: str) -> None:
    # GH 25580, resample on IntegerArray
    ts = Series(
        range(9),
        index=date_range("1/1/2000", periods=9, freq="min").as_unit(unit),
        dtype="Int64",
    )
    result = ts.resample("3min").sum()
    expected = Series(
        [3, 12, 21],
        index=date_range("1/1/2000", periods=3, freq="3min").as_unit(unit),
        dtype="Int64",
    )
    tm.assert_series_equal(result, expected)

    result = ts.resample("3min").mean()
    expected = Series(
        [1, 4, 7],
        index=date_range("1/1/2000", periods=3, freq="3min").as_unit(unit),
        dtype="Float64",
    )
    tm.assert_series_equal(result, expected)


def test_resample_basic_grouper(unit: str) -> None:
    index = date_range("1/1/2000 00:00:00", "1/1/2000 00:13:00", freq="Min")
    s = Series(range(len(index)), index=index)
    s.index.name = "index"
    s.index = s.index.as_unit(unit)
    result = s.resample("5Min").last()
    grouper = Grouper(freq=Minute(5), closed="left", label="left")
    expected = s.groupby(grouper).agg(lambda x: x.iloc[-1])
    tm.assert_series_equal(result, expected)


@pytest.mark.filterwarnings(
    "ignore:The 'convention' keyword in Series.resample:FutureWarning"
)
@pytest.mark.parametrize(
    "keyword,value",
    [("label", "righttt"), ("closed", "righttt"), ("convention", "starttt")],
)
def test_resample_string_kwargs(keyword: str, value: str, unit: str) -> None:
    # see gh-19303
    # Check that wrong keyword argument strings raise an error
    index = date_range("1/1/2000 00:00:00", "1/1/2000 00:13:00", freq="Min")
    series = Series(range(len(index)), index=index)
    series.index.name = "index"
    series.index = series.index.as_unit(unit)
    msg = f"Unsupported value {value} for `{keyword}`"
    with pytest.raises(ValueError, match=msg):
        series.resample("5min", **({keyword: value}))


def test_resample_how(downsample_method: str, unit: str) -> None:
    if downsample_method == "ohlc":
        pytest.skip("covered by test_resample_how_ohlc")
    index = date_range("1/1/2000 00:00:00", "1/1/2000 00:13:00", freq="Min")
    s = Series(range(len(index)), index=index)
    s.index.name = "index"
    s.index = s.index.as_unit(unit)
    grouplist = np.ones_like(s)
    grouplist[0] = 0
    grouplist[1:6] = 1
    grouplist[6:11] = 2
    grouplist[11:] = 3
    expected = s.groupby(grouplist).agg(downsample_method)
    expected.index = date_range(
        "1/1/2000", periods=4, freq="5min", name="index"
    ).as_unit(unit)

    result = getattr(
        s.resample("5min", closed="right", label="right"), downsample_method
    )()
    tm.assert_series_equal(result, expected)


def test_resample_how_ohlc(unit: str) -> None:
    index = date_range("1/1/2000 00:00:00", "1/1/2000 00:13:00", freq="Min")
    s = Series(range(len(index)), index=index)
    s.index.name = "index"
    s.index = s.index.as_unit(unit)
    grouplist = np.ones_like(s)
    grouplist[0] = 0
    grouplist[1:6] = 1
    grouplist[6:11] = 2
    grouplist[11:] = 3

    def _ohlc(group: Series) -> np.ndarray:
        if isna(group).all():
            return np.repeat(np.nan, 4)
        return [group.iloc[0], group.max(), group.min(), group.iloc[-1]]

    expected = DataFrame(
        s.groupby(grouplist).agg(_ohlc).values.tolist(),
        index=date_range("1/1/2000", periods=4, freq="5min", name="index").as_unit(
            unit
        ),
        columns=["open", "high", "low", "close"],
    )

    result = s.resample("5min", closed="right", label="right").ohlc()
    tm.assert_frame_equal(result, expected)


def test_resample_how_callables(unit: str) -> None:
    # GH#7929
    data = np.arange(5, dtype=np.int64)
    msg = "'d' is deprecated and will be removed in a future version."
    with tm.assert_produces_warning(FutureWarning, match=msg):
        ind = date_range(start="2014-01-01", periods=len(data), freq="d").as_unit(unit)
    df = DataFrame({"A": data, "B": data}, index=ind)

    def fn(x: Series, a: int = 1) -> str:
        return str(type(x))

    class FnClass:
        def __call__(self, x: Series) -> str:
            return str(type(x))

    df_standard = df.resample("ME").apply(fn)
    df_lambda = df.resample("ME").apply(lambda x: str(type(x)))
    df_partial = df.resample("ME").apply(partial(fn))
    df_partial2 = df.resample("ME").apply(partial(fn, a=2))
    df_class = df.resample("ME").apply(FnClass())

    tm.assert_frame_equal(df_standard, df_lambda)
    tm.assert_frame_equal(df_standard, df_partial)
    tm.assert_frame_equal(df_standard, df_partial2)
    tm.assert_frame_equal(df_standard, df_class)


def test_resample_rounding(unit: str) -> None:
    # GH 8371
    # odd results when rounding is needed

    ts = [
        "2014-11-08 00:00:01",
        "2014-11-08 00:00:02",
        "2014-11-08 00:00:02",
        "2014-11-08 00:00:03",
        "2014-11-08 00:00:07",
        "2014-11-08 00:00:07",
        "2014-11-08 00:00:08",
        "2014-11-08 00:00:08",
        "2014-11-08 00:00:08",
        "2014-11-08 00:00:09",
        "2014-11-08 00:00:10",
        "2014-11-08 00:00:11",
        "2014-11-08 00:00:11",
        "2014-11-08 00:00:13",
        "2014-11-08 00:00:14",
        "2014-11-08 00:00:15",
        "2014-11-08 00:00:17",
        "2014-11-08 00:00:20",
        "2014-11-08 00:00:21",
    ]
    df = DataFrame({"value": [1] * 19}, index=pd.to_datetime(ts))
    df.index = df.index.as_unit(unit)

    result = df.resample("6s").sum()
    expected = DataFrame(
        {"value": [4, 9, 4, 2]},
        index=date_range("2014-11-08", freq="6s", periods=4).as_unit(unit),
    )
    tm.assert_frame_equal(result, expected)

    result = df.resample("7s").sum()
    expected = DataFrame(
        {"value": [4, 10, 4, 1]},
        index=date_range("2014-11-08", freq="7s", periods=4).as_unit(unit),
    )
    tm.assert_frame_equal(result, expected)

    result = df.resample("11s").sum()
    expected = DataFrame(
        {"value": [11, 8]},
        index=date_range("2014-11-08", freq="11s", periods=2).as_unit(unit),
    )
    tm.assert_frame_equal(result, expected)

    result = df.resample("13s").sum()
    expected = DataFrame(
        {"value": [13, 6]},
        index=date_range("2014-11-08", freq="13s", periods=2).as_unit(unit),
    )
    tm.assert_frame_equal(result, expected)

    result = df.resample("17s").sum()
    expected = DataFrame(
        {"value": [16, 3]},
        index=date_range("2014-11-08", freq="17s", periods=2).as_unit(unit),
    )
    tm.assert_frame_equal(result, expected)


def test_resample_basic_from_daily(unit: str) -> None:
    # from daily
    dti = date_range(
        start=datetime(2005, 1, 1), end=datetime(2005, 1, 10), freq="D", name="index"
    ).as_unit(unit)

    s = Series(np.random.default_rng(2).random(len(dti)), dti)

    # to weekly
    msg = "'w-sun' is deprecated and will be removed in a future version."
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = s.resample("w-sun").last()

    assert len(result) == 3
    assert (result.index.dayofweek == [6, 6, 6]).all()
    assert result.iloc[0] == s["1/2/2005"]
    assert result.iloc[1] == s["1/9/2005"]
    assert result.iloc[2] == s.iloc[-1]

    result = s.resample("W-MON").last()
    assert len(result) == 2
    assert (result.index.dayofweek == [0, 0]).all()
    assert result.iloc[0] == s["1/3/2005"]
    assert result.iloc[1] == s["1/10/2005"]

    result = s.resample("W-TUE").last()
    assert len(result) == 2
    assert (result.index.dayofweek == [1, 1]).all()
    assert result.iloc[0] == s["1/4/2005"]
    assert result.iloc[1] == s["1/10/2005"]

    result = s.resample("W-WED").last()
    assert len(result) == 2
    assert (result.index.dayofweek == [2, 2]).all()
    assert result.iloc[0] == s["1/5/2005"]
    assert result.iloc[1] == s["1/10/2005"]

    result = s.resample("W-THU").last()
    assert len(result) == 2
    assert (result.index.dayofweek == [3, 3]).all()
    assert result.iloc[0] == s["1/6/2005"]
    assert result.iloc[1] == s["1/10/2005"]

    result = s.resample("W-FRI").last()
    assert len(result) == 2
    assert (result.index.dayofweek == [4, 4]).all()
    assert result.iloc[0] == s["1/7/2005"]
    assert result.iloc[1] == s["1/10/2005"]

    # to biz day
    result = s.resample("B").last()
    assert len(result) == 7
    assert (result.index.dayofweek == [4, 0, 1, 2, 3, 4, 0]).all()

    assert result.iloc[0] == s["1/2/2005"]
    assert result.iloc[1] == s["1/3/2005"]
    assert result.iloc[5] == s["1/9/2005"]
    assert result.index.name == "index"


def test_resample_upsampling_picked_but_not_correct(unit: str) -> None:
    # Test for issue #3020
    dates = date_range("01-Jan-2014", "05-Jan-2014", freq="D").as_unit(unit)
    series = Series(1, index=dates)

    result = series.resample("D").mean()
    assert result.index[0] == dates[0]

    # GH 5955
    # incorrect deciding to upsample when the axis frequency matches the
    # resample frequency

    s = Series(
        np.arange(1.0, 6), index=[datetime(1975, 1, i, 12, 0) for i in range(1, 6)]
    )
    s.index = s.index.as_unit(unit)
    expected = Series(
        np.arange(1.0, 6),
        index=date_range("19750101", periods=5, freq="D").as_unit(unit),
    )

    result = s.resample("D").count()
    tm.assert_series_equal(result, Series(1, index=expected.index))

    result1 = s.resample("D").sum()
    result2 = s.resample("D").mean()
    tm.assert_series_equal(result1, expected)
    tm.assert_series_equal(result2, expected)


@pytest.mark.parametrize("f", ["sum", "mean", "prod", "min", "max", "var"])
def test_resample_frame_basic_cy_funcs(f: str, unit: str) -> None:
    df = DataFrame(
        np.random.default_rng(2).standard_normal((50, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=50, freq="B"),
    )
    df.index = df.index.as_unit(unit)

    b = Grouper(freq="ME")
    g = df.groupby(b)

    # check all cython functions work
    g._cython_agg_general(f, alt=None, numeric_only=True)


@pytest.mark.parametrize("freq", ["YE", "ME"])
def test_resample_frame_basic_M_A(freq: str, unit: str) -> None:
    df = DataFrame(
        np.random.default_rng(2).standard_normal((50, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=50, freq="B"),
    )
    df.index = df.index.as_unit(unit)
    result = df.resample(freq).mean()
    tm.assert_series_equal(result["A"], df["A"].resample(freq).mean())


def test_resample_upsample(unit: str) -> None:
    # from daily
    dti = date_range(
        start=datetime(2005, 1, 1), end=datetime(2005, 1, 10), freq="D", name="index"
    ).as_unit(unit)

    s = Series(np.random.default_rng(2).random(len(dti)), dti)

    # to minutely, by padding
    result = s.resample("Min").ffill()
    assert len(result) == 12961
    assert result.iloc[0] == s.iloc[0]
    assert result.iloc[-1] == s.iloc[-1]

    assert result.index.name == "index"


def test_resample_how_method(unit: str) -> None:
    # GH9915
    s = Series(
        [11, 22],
        index=[
            Timestamp("2015-03-31 21:48:52.672000"),
            Timestamp("2015-03-31 21:49:52.739000"),
        ],
    )
    s.index = s.index.as_unit(unit)
    expected = Series(
        [11, np.nan, np.nan, np.nan, np.nan, np.nan, 22],
        index=DatetimeIndex(
            [
                Timestamp("2015-03-31 21:48:50"),
                Timestamp("2015-03-31 21:49:00"),
                Timestamp("2015-03-31 21:49:10"),
                Timestamp("2015-03-31 21:49:20"),
                Timestamp("2015-03-31 21:49:30"),
                Timestamp("2015-03-31 21:49:40"),
                Timestamp("2015-03-31 21:49:50"),
            ],
            freq="10s",
        ),
    )
    expected.index = expected.index.as_unit(unit)
    tm.assert_series_equal(s.resample("10s").mean(), expected)


def test_resample_extra_index_point(unit: str) -> None:
    # GH#9756
    index = date_range(start="20150101", end="20150331", freq="BME").as_unit(unit)
    expected = DataFrame({"A": Series([21, 41, 63], index=index)})

    index = date_range(start="20150101", end="20150331", freq="B").as_unit(unit)
    df = DataFrame({"A": Series(range(len(index)), index=index)}, dtype="int64")
    result = df.resample("BME").last()
    tm.assert_frame_equal(result, expected)


def test_upsample_with_limit(unit: str) -> None:
    rng = date_range("1/1/2000", periods=3, freq="5min").as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)

    result = ts.resample("min").ffill(limit=2)
    expected = ts.reindex(result.index, method="ffill", limit=2)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("freq", ["1D", "10h", "5Min", "10s"])
@pytest.mark.parametrize("rule", ["YE", "3ME", "15D", "30h", "15Min", "30s"])
def test_nearest_upsample_with_limit(tz_aware_fixture: Any, freq: str, rule: str, unit: str) -> None:
    # GH 33939
    rng = date_range("1/1/2000", periods=3, freq=freq, tz=tz_aware_fixture).as_unit(
        unit
    )
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)

    result = ts.resample(rule).nearest(limit=2)
    expected = ts.reindex(result.index, method="nearest", limit=2)
    tm.assert_series_equal(result, expected)


def test_resample_ohlc(unit: str) -> None:
    index = date_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq="Min")
    s = Series(range(len(index)), index=index)
    s.index.name = "index"
    s.index = s.index.as_unit(unit)

    grouper = Grouper(freq=Minute(5))
    expect = s.groupby(grouper).agg(lambda x: x.iloc[-1])
    result = s.resample("5Min").ohlc()

    assert len(result) == len(expect)
    assert len(result.columns) == 4

    xs = result.iloc[-2]
    assert xs["open"] == s.iloc[-6]
    assert xs["high"] == s[-6:-1].max()
    assert xs["low"] == s[-6:-1].min()
    assert xs["close"] == s.iloc[-2]

    xs = result.iloc[0]
    assert xs["open"] == s.iloc[0]
    assert xs["high"] == s[:5].max()
    assert xs["low"] == s[:5].min()
    assert xs["close"] == s.iloc[4]


def test_resample_ohlc_result(unit: str) -> None:
    # GH 12332
    index = date_range("1-1-2000", "2-15-2000", freq="h").as_unit(unit)
    index = index.union(date_range("4-15-2000", "5-15-2000", freq="h").as_unit(unit))
    s = Series(range(len(index)), index=index)

    a = s.loc[:"4-15-2000"].resample("30min").ohlc()
    assert isinstance(a, DataFrame)

    b = s.loc[:"4-14-2000"].resample("30min").ohlc()
    assert isinstance(b, DataFrame)


def test_resample_ohlc_result_odd_period(unit: str) -> None:
    # GH12348
    # raising on odd period
    rng = date_range("2013-12-30", "2014-01-07").as_unit(unit)
    index = rng.drop(
        [
            Timestamp("2014-01-01"),
            Timestamp("2013-12-31"),
            Timestamp("2014-01-04"),
            Timestamp("2014-01-05"),
        ]
    )
    df = DataFrame(data=np.arange(len(index)), index=index)
    result = df.resample("B").mean()
    expected = df.reindex(index=date_range(rng[0], rng[-1], freq="B").as_unit(unit))
    tm.assert_frame_equal(result, expected)


def test_resample_ohlc_dataframe(unit: str) -> None:
    df = (
        DataFrame(
            {
                "PRICE": {
                    Timestamp("2011-01-06 10:59:05", tz=None): 24990,
                    Timestamp("2011-01-06 12:43:33", tz=None): 25499,
                    Timestamp("2011-01-06 12:54:09", tz=None): 25499,
                },
                "VOLUME": {
                    Timestamp("2011-01-06 10:59:05", tz=None): 1500000000,
                    Timestamp("2011-01-06 12:43:33", tz=None): 5000000000,
                    Timestamp("2011-01-06 12:54:09", tz=None): 100000000,
                },
            }
        )
    ).reindex(["VOLUME", "PRICE"], axis=1)
    df.index = df.index.as_unit(unit)
    df.columns.name = "Cols"
    res = df.resample("h").ohlc()
    exp = pd.concat(
        [df["VOLUME"].resample("h").ohlc(), df["PRICE"].resample("h").ohlc()],
        axis=1,
        keys=df.columns,
    )
    assert exp.columns.names[0] == "Cols"
    tm.assert_frame_equal(exp, res)

    df.columns = [["a", "b"], ["c", "d"]]
    res = df.resample("h").ohlc()
    exp.columns = pd.MultiIndex.from_tuples(
        [
            ("a", "c", "open"),
            ("a", "c", "high"),
            ("a", "c", "low"),
            ("a", "c", "close"),
            ("b", "d", "open"),
            ("b", "d", "high"),
            ("b", "d", "low"),
            ("b", "d", "close"),
        ]
    )
    tm.assert_frame_equal(exp, res)

    # dupe columns fail atm
    # df.columns = ['PRICE', 'PRICE']


def test_resample_reresample(unit: str) -> None:
    dti = date_range(
        start=datetime(2005, 1, 1), end=datetime(2005, 1, 10), freq="D"
    ).as_unit(unit)
    s = Series(np.random.default_rng(2).random(len(dti)), dti)
    bs = s.resample("B", closed="right", label="right").mean()
    result = bs.resample("8h").mean()
    assert len(result) == 25
    assert isinstance(result.index.freq, offsets.DateOffset)
    assert result.index.freq == offsets.Hour(8)


@pytest.mark.parametrize(
    "freq, expected_kwargs",
    [
        ["YE-DEC", {"start": "1990", "end": "2000", "freq": "Y-DEC"}],
        ["YE-JUN", {"start": "1990", "end": "2000", "freq": "Y-JUN"}],
        ["ME", {"start": "1990-01", "end": "2000-01", "freq": "M"}],
    ],
)
def test_resample_timestamp_to_period(
    simple_date_range_series: Callable[[str, str, str], Series],
    freq: str,
    expected_kwargs: Dict[str, str],
    unit: str,
) -> None:
    ts = simple_date_range_series("1/1/1990", "1/1/2000")
    ts.index = ts.index.as_unit(unit)

    result = ts.resample(freq).mean().to_period()
    expected = ts.resample(freq).mean()
    expected.index = period_range(**expected_kwargs)
    tm.assert_series_equal(result, expected)


def test_ohlc_5min(unit: str) -> None:
    def _ohlc(group: Series) -> np.ndarray:
        if isna(group).all():
            return np.repeat(np.nan, 4)
        return [group.iloc[0], group.max(), group.min(), group.iloc[-1]]

    rng = date_range("1/1/2000 00:00:00", "1/1/2000 5:59:50", freq="10s").as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)

    resampled = ts.resample("5min", closed="right", label="right").ohlc()

    assert (resampled.loc["1/1/2000 00:00"] == ts.iloc[0]).all()

    exp = _ohlc(ts[1:31])
    assert (resampled.loc["1/1/2000 00:05"] == exp).all()

    exp = _ohlc(ts["1/1/2000 5:55:01":])
    assert (resampled.loc["1/1/2000 6:00:00"] == exp).all()


def test_downsample_non_unique(unit: str) -> None:
    rng = date_range("1/1/2000", "2/29/2000").as_unit(unit)
    rng2 = rng.repeat(5).values
    ts = Series(np.random.default_rng(2).standard_normal(len(rng2)), index=rng2)

    result = ts.resample("ME").mean()

    expected = ts.groupby(lambda x: x.month).mean()
    assert len(result) == 2
    tm.assert_almost_equal(result.iloc[0], expected[1])
    tm.assert_almost_equal(result.iloc[1], expected[2])


def test_asfreq_non_unique(unit: str) -> None:
    # GH #1077
    rng = date_range("1/1/2000", "2/29/2000").as_unit(unit)
    rng2 = rng.repeat(2).values
    ts = Series(np.random.default_rng(2).standard_normal(len(rng2)), index=rng2)

    msg = "cannot reindex on an axis with duplicate labels"
    with pytest.raises(ValueError, match=msg):
        ts.asfreq("B")


@pytest.mark.parametrize("freq", ["min", "5min", "15min", "30min", "4h", "12h"])
def test_resample_anchored_ticks(freq: str, unit: str) -> None:
    # If a fixed delta (5 minute, 4 hour) evenly divides a day, we should
    # "anchor" the origin at midnight so we get regular intervals rather
    # than starting from the first timestamp which might start in the
    # middle of a desired interval

    rng = date_range("1/1/2000 04:00:00", periods=86400, freq="s").as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    ts[:2] = np.nan  # so results are the same

    result = ts[2:].resample(freq, closed="left", label="left").mean()
    expected = ts.resample(freq, closed="left", label="left").mean()
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("end", [1, 2])
def test_resample_single_group(end: int, unit: str) -> None:
    mysum = lambda x: x.sum()

    rng = date_range("2000-1-1", f"2000-{end}-10", freq="D").as_unit(unit)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    tm.assert_series_equal(ts.resample("ME").sum(), ts.resample("ME").apply(mysum))


def test_resample_single_group_std(unit: str) -> None:
    # GH 3849
   