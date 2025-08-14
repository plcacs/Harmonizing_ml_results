from __future__ import annotations
import array
import datetime
import functools
import re
from collections import abc, defaultdict, OrderedDict, namedtuple
from datetime import datetime, date, timedelta
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

import numpy as np
from numpy import ma

import pandas as pd
from pandas import (
    Categorical,
    CategoricalIndex,
    DataFrame,
    DatetimeIndex,
    Index,
    Interval,
    IntervalIndex,
    MultiIndex,
    Period,
    RangeIndex,
    Series,
    Timedelta
)
from pandas._libs import lib
from pandas._libs.tslibs import Timestamp
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.errors import IntCastingNaNError
from pandas.arrays import DatetimeArray, TimedeltaArray
from pandas.tests.extension.arrow.array import IntervalArray  # If available in the namespace
from pandas.tests.extension.period.array import PeriodArray  # If available in the namespace

class TestDataFrameConstructors:
    def test_constructor_ordereddict(self) -> None:
        nitems: int = 100
        nums: List[int] = list(range(nitems))
        np.random.default_rng(2).shuffle(nums)
        expected: List[str] = [f"A{i:d}" for i in nums]
        df: DataFrame = DataFrame(OrderedDict(zip(expected, [[0]] * nitems)))
        assert expected == list(df.columns)

    def test_constructor_dict(self) -> None:
        datetime_series: Series = Series(
            np.arange(30, dtype=np.float64), index=pd.date_range("2020-01-01", periods=30)
        )
        datetime_series_short: Series = datetime_series[5:]
        frame: DataFrame = DataFrame({"col1": datetime_series, "col2": datetime_series_short})
        assert len(datetime_series) == 30
        assert len(datetime_series_short) == 25
        pd.testing.assert_series_equal(frame["col1"], datetime_series.rename("col1"))
        exp: Series = Series(
            np.concatenate([[np.nan] * 5, datetime_series_short.values]),
            index=datetime_series.index,
            name="col2",
        )
        pd.testing.assert_series_equal(exp, frame["col2"])
        frame = DataFrame(
            {"col1": datetime_series, "col2": datetime_series_short},
            columns=["col2", "col3", "col4"],
        )
        assert len(frame) == len(datetime_series_short)
        assert "col1" not in frame
        from pandas.core.dtypes.common import isna
        assert isna(frame["col3"]).all()
        assert len(DataFrame()) == 0
        msg: str = "Mixing dicts with non-Series may lead to ambiguous ordering."
        import pytest
        with pytest.raises(ValueError, match=msg):
            DataFrame({"A": {"a": "a", "b": "b"}, "B": ["a", "b", "c"]})

    def test_constructor_dict_length1(self) -> None:
        frame: DataFrame = DataFrame({"A": {"1": 1, "2": 2}})
        pd.testing.assert_index_equal(frame.index, Index(["1", "2"]))

    def test_constructor_dict_with_index(self) -> None:
        idx: Index = Index([0, 1, 2])
        frame: DataFrame = DataFrame({}, index=idx)
        assert frame.index is idx

    def test_constructor_dict_with_index_and_columns(self) -> None:
        idx: Index = Index([0, 1, 2])
        frame: DataFrame = DataFrame({}, index=idx, columns=idx)
        assert frame.index is idx
        assert frame.columns is idx
        assert len(frame._series) == 3

    def test_constructor_dict_of_empty_lists(self) -> None:
        frame: DataFrame = DataFrame({"A": [], "B": []}, columns=["A", "B"])
        pd.testing.assert_index_equal(frame.index, RangeIndex(0), exact=True)

    def test_constructor_dict_with_none(self) -> None:
        frame_none: DataFrame = DataFrame({"a": None}, index=[0])
        frame_none_list: DataFrame = DataFrame({"a": [None]}, index=[0])
        assert frame_none._get_value(0, "a") is None
        assert frame_none_list._get_value(0, "a") is None
        pd.testing.assert_frame_equal(frame_none, frame_none_list)

    def test_constructor_dict_errors(self) -> None:
        msg: str = "If using all scalar values, you must pass an index"
        import pytest
        with pytest.raises(ValueError, match=msg):
            DataFrame({"a": 0.7})
        with pytest.raises(ValueError, match=msg):
            DataFrame({"a": 0.7}, columns=["a"])

    @pytest.mark.parametrize("scalar", [2, np.nan, None, "D"])
    def test_constructor_invalid_items_unused(self, scalar: Any) -> None:
        result: DataFrame = DataFrame({"a": scalar}, columns=["b"])
        expected: DataFrame = DataFrame(columns=["b"])
        pd.testing.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("value", [4, np.nan, None, float("nan")])
    def test_constructor_dict_nan_key(self, value: Any) -> None:
        cols: List[Any] = [1, value, 3]
        idx: List[Any] = ["a", value]
        values: List[List[int]] = [[0, 3], [1, 4], [2, 5]]
        data: Dict[Any, Series] = {cols[c]: Series(values[c], index=idx) for c in range(3)}
        result: DataFrame = DataFrame(data).sort_values(1).sort_values("a", axis=1)
        expected: DataFrame = DataFrame(
            np.arange(6, dtype="int64").reshape(2, 3), index=idx, columns=cols
        )
        pd.testing.assert_frame_equal(result, expected)
        result = DataFrame(data, index=idx).sort_values("a", axis=1)
        pd.testing.assert_frame_equal(result, expected)
        result = DataFrame(data, index=idx, columns=cols)
        pd.testing.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("value", [np.nan, None, float("nan")])
    def test_constructor_dict_nan_tuple_key(self, value: Any) -> None:
        cols: Index = Index([(11, 21), (value, 22), (13, value)])
        idx: Index = Index([("a", value), (value, 2)])
        values: List[List[int]] = [[0, 3], [1, 4], [2, 5]]
        data: Dict[Any, Series] = {cols[c]: Series(values[c], index=idx) for c in range(3)}
        result: DataFrame = DataFrame(data).sort_values((11, 21)).sort_values(("a", value), axis=1)
        expected: DataFrame = DataFrame(
            np.arange(6, dtype="int64").reshape(2, 3), index=idx, columns=cols
        )
        pd.testing.assert_frame_equal(result, expected)
        result = DataFrame(data, index=idx).sort_values(("a", value), axis=1)
        pd.testing.assert_frame_equal(result, expected)
        result = DataFrame(data, index=idx, columns=cols)
        pd.testing.assert_frame_equal(result, expected)

    def test_constructor_dict_order_insertion(self) -> None:
        datetime_series: Series = Series(
            np.arange(10, dtype=np.float64), index=pd.date_range("2020-01-01", periods=10)
        )
        datetime_series_short: Series = datetime_series[:5]
        d: Dict[str, Series] = {"b": datetime_series_short, "a": datetime_series}
        frame: DataFrame = DataFrame(data=d)
        expected: DataFrame = DataFrame(data=d, columns=list("ba"))
        pd.testing.assert_frame_equal(frame, expected)

    def test_constructor_dict_nan_key_and_columns(self) -> None:
        result: DataFrame = DataFrame({np.nan: [1, 2], 2: [2, 3]}, columns=[np.nan, 2])
        expected: DataFrame = DataFrame([[1, 2], [2, 3]], columns=[np.nan, 2])
        pd.testing.assert_frame_equal(result, expected)

    def test_constructor_multi_index(self) -> None:
        tuples: List[tuple[int, int]] = [(2, 3), (3, 3), (3, 3)]
        mi: MultiIndex = MultiIndex.from_tuples(tuples)
        df: DataFrame = DataFrame(index=mi, columns=mi)
        from pandas.core.dtypes.common import isna
        assert isna(df).values.ravel().all()
        tuples = [(3, 3), (2, 3), (3, 3)]
        mi = MultiIndex.from_tuples(tuples)
        df = DataFrame(index=mi, columns=mi)
        assert isna(df).values.ravel().all()

    def test_constructor_2d_index(self) -> None:
        df: DataFrame = DataFrame([[1]], columns=[[1]], index=[1, 2])
        expected: DataFrame = DataFrame(
            [1, 1],
            index=Index([1, 2], dtype="int64"),
            columns=MultiIndex.from_arrays([[1]]),
        )
        pd.testing.assert_frame_equal(df, expected)
        df = DataFrame([[1]], columns=[[1]], index=[[1, 2]])
        expected = DataFrame(
            [1, 1],
            index=MultiIndex.from_arrays([[1, 2]]),
            columns=MultiIndex.from_arrays([[1]]),
        )
        pd.testing.assert_frame_equal(df, expected)

    def test_constructor_error_msgs(self) -> None:
        import pytest
        msg: str = "Empty data passed with indices specified."
        with pytest.raises(ValueError, match=msg):
            DataFrame(np.empty(0), index=[1])
        msg = "Mixing dicts with non-Series may lead to ambiguous ordering."
        with pytest.raises(ValueError, match=msg):
            DataFrame({"A": {"a": "a", "b": "b"}, "B": ["a", "b", "c"]})
        msg = r"Shape of passed values is \(4, 3\), indices imply \(3, 3\)"
        with pytest.raises(ValueError, match=msg):
            DataFrame(
                np.arange(12).reshape((4, 3)),
                columns=["foo", "bar", "baz"],
                index=pd.date_range("2000-01-01", periods=3),
            )
        arr: np.ndarray = np.array([[4, 5, 6]])
        msg = r"Shape of passed values is \(1, 3\), indices imply \(1, 4\)"
        with pytest.raises(ValueError, match=msg):
            DataFrame(index=[0], columns=range(4), data=arr)
        arr = np.array([4, 5, 6])
        msg = r"Shape of passed values is \(3, 1\), indices imply \(1, 4\)"
        with pytest.raises(ValueError, match=msg):
            DataFrame(index=[0], columns=range(4), data=arr)
        with pytest.raises(ValueError, match="Must pass 2-d input"):
            DataFrame(np.zeros((3, 3, 3)), columns=["A", "B", "C"], index=[1])
        msg = r"Shape of passed values is \(2, 3\), indices imply \(1, 3\)"
        with pytest.raises(ValueError, match=msg):
            DataFrame(
                np.random.default_rng(2).random((2, 3)),
                columns=["A", "B", "C"],
                index=[1],
            )
        msg = r"Shape of passed values is \(2, 3\), indices imply \(2, 2\)"
        with pytest.raises(ValueError, match=msg):
            DataFrame(
                np.random.default_rng(2).random((2, 3)),
                columns=["A", "B"],
                index=[1, 2],
            )
        msg = "2 columns passed, passed data had 10 columns"
        with pytest.raises(ValueError, match=msg):
            DataFrame((range(10), range(10, 20)), columns=("ones", "twos"))
        msg = "If using all scalar values, you must pass an index"
        with pytest.raises(ValueError, match=msg):
            DataFrame({"a": False, "b": True})

    def test_constructor_subclass_dict(self, dict_subclass: Callable[..., Any]) -> None:
        data: Dict[str, Any] = {
            "col1": dict_subclass((x, 10.0 * x) for x in range(10)),
            "col2": dict_subclass((x, 20.0 * x) for x in range(10)),
        }
        df: DataFrame = DataFrame(data)
        refdf: DataFrame = DataFrame({col: dict(val.items()) for col, val in data.items()})
        pd.testing.assert_frame_equal(refdf, df)
        data = dict_subclass(data.items())
        df = DataFrame(data)
        pd.testing.assert_frame_equal(refdf, df)

    def test_constructor_defaultdict(self, float_frame: DataFrame) -> None:
        data: Dict[Any, Any] = {}
        float_frame.loc[: float_frame.index[10], "B"] = np.nan
        for k, v in float_frame.items():
            dct: defaultdict[Any, Any] = defaultdict(dict)
            dct.update(v.to_dict())
            data[k] = dct
        frame: DataFrame = DataFrame(data)
        expected: DataFrame = frame.reindex(index=float_frame.index)
        pd.testing.assert_frame_equal(float_frame, expected)

    def test_constructor_dict_block(self) -> None:
        expected: np.ndarray = np.array([[4.0, 3.0, 2.0, 1.0]])
        df: DataFrame = DataFrame(
            {"d": [4.0], "c": [3.0], "b": [2.0], "a": [1.0]},
            columns=["d", "c", "b", "a"],
        )
        pd.testing.assert_numpy_array_equal(df.values, expected)

    def test_constructor_dict_cast(self, using_infer_string: bool) -> None:
        test_data: Dict[str, Dict[str, Any]] = {"A": {"1": 1, "2": 2}, "B": {"1": "1", "2": "2", "3": "3"}}
        frame: DataFrame = DataFrame(test_data, dtype=float)
        assert len(frame) == 3
        assert frame["B"].dtype == np.float64
        assert frame["A"].dtype == np.float64
        frame = DataFrame(test_data)
        assert len(frame) == 3
        if not using_infer_string:
            assert frame["B"].dtype == np.object_
        else:
            assert frame["B"].dtype == "str"
        assert frame["A"].dtype == np.float64

    def test_constructor_dict_cast2(self) -> None:
        test_data: Dict[str, Dict[Any, Any]] = {
            "A": dict(zip(range(20), [f"word_{i}" for i in range(20)])),
            "B": dict(zip(range(15), np.random.default_rng(2).standard_normal(15))),
        }
        import pytest
        with pytest.raises(ValueError, match="could not convert string"):
            DataFrame(test_data, dtype=float)

    def test_constructor_dict_dont_upcast(self) -> None:
        d: Dict[str, Dict[str, Any]] = {"Col1": {"Row1": "A String", "Row2": np.nan}}
        df: DataFrame = DataFrame(d)
        assert isinstance(df["Col1"]["Row2"], float)

    def test_constructor_dict_dont_upcast2(self) -> None:
        dm: DataFrame = DataFrame([[1, 2], ["a", "b"]], index=[1, 2], columns=[1, 2])
        assert isinstance(dm[1][1], int)

    def test_constructor_dict_of_tuples(self) -> None:
        data: Dict[str, tuple] = {"a": (1, 2, 3), "b": (4, 5, 6)}
        result: DataFrame = DataFrame(data)
        expected: DataFrame = DataFrame({k: list(v) for k, v in data.items()})
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)

    def test_constructor_dict_of_ranges(self) -> None:
        data: Dict[str, range] = {"a": range(3), "b": range(3, 6)}
        result: DataFrame = DataFrame(data)
        expected: DataFrame = DataFrame({"a": [0, 1, 2], "b": [3, 4, 5]})
        pd.testing.assert_frame_equal(result, expected)

    def test_constructor_dict_of_iterators(self) -> None:
        data: Dict[str, Any] = {"a": iter(range(3)), "b": reversed(range(3))}
        result: DataFrame = DataFrame(data)
        expected: DataFrame = DataFrame({"a": [0, 1, 2], "b": [2, 1, 0]})
        pd.testing.assert_frame_equal(result, expected)

    def test_constructor_dict_of_generators(self) -> None:
        data: Dict[str, Any] = {"a": (i for i in range(3)), "b": (i for i in reversed(range(3)))}
        result: DataFrame = DataFrame(data)
        expected: DataFrame = DataFrame({"a": [0, 1, 2], "b": [2, 1, 0]})
        pd.testing.assert_frame_equal(result, expected)

    def test_constructor_dict_multiindex(self) -> None:
        d: Dict[Any, Dict[Any, int]] = {
            ("a", "a"): {("i", "i"): 0, ("i", "j"): 1, ("j", "i"): 2},
            ("b", "a"): {("i", "i"): 6, ("i", "j"): 5, ("j", "i"): 4},
            ("b", "c"): {("i", "i"): 7, ("i", "j"): 8, ("j", "i"): 9},
        }
        _d: List[tuple[Any, Any]] = sorted(d.items())
        df: DataFrame = DataFrame(d)
        expected: DataFrame = DataFrame(
            [x[1] for x in _d], index=MultiIndex.from_tuples([x[0] for x in _d])
        ).T
        expected.index = MultiIndex.from_tuples(expected.index)
        pd.testing.assert_frame_equal(df, expected)
        d["z"] = {"y": 123.0, ("i", "i"): 111, ("i", "j"): 111, ("j", "i"): 111}
        _d.insert(0, ("z", d["z"]))
        expected = DataFrame(
            [x[1] for x in _d], index=Index([x[0] for x in _d], tupleize_cols=False)
        ).T
        expected.index = Index(expected.index, tupleize_cols=False)
        df = DataFrame(d)
        df = df.reindex(columns=expected.columns, index=expected.index)
        pd.testing.assert_frame_equal(df, expected)

    def test_constructor_dict_datetime64_index(self) -> None:
        dates_as_str: List[str] = ["1984-02-19", "1988-11-06", "1989-12-03", "1990-03-15"]
        def create_data(constructor: Callable[[str], Any]) -> Dict[int, Dict[Any, int]]:
            return {i: {constructor(s): 2 * i} for i, s in enumerate(dates_as_str)}
        data_datetime64: Dict[int, Dict[Any, int]] = create_data(np.datetime64)
        data_datetime: Dict[int, Dict[Any, int]] = create_data(lambda x: datetime.strptime(x, "%Y-%m-%d"))
        data_Timestamp: Dict[int, Dict[Any, int]] = create_data(Timestamp)
        expected: DataFrame = DataFrame(
            [
                [0, None, None, None],
                [None, 2, None, None],
                [None, None, 4, None],
                [None, None, None, 6],
            ],
            index=[Timestamp(dt) for dt in dates_as_str],
        )
        result_datetime64: DataFrame = DataFrame(data_datetime64)
        result_datetime: DataFrame = DataFrame(data_datetime)
        assert result_datetime.index.unit == "us"
        result_datetime.index = result_datetime.index.as_unit("s")
        result_Timestamp: DataFrame = DataFrame(data_Timestamp)
        pd.testing.assert_frame_equal(result_datetime64, expected)
        pd.testing.assert_frame_equal(result_datetime, expected)
        pd.testing.assert_frame_equal(result_Timestamp, expected)

    @pytest.mark.parametrize(
        "klass,name",
        [
            (lambda x: np.timedelta64(x, "D"), "timedelta64"),
            (lambda x: timedelta(days=x), "pytimedelta"),
            (lambda x: Timedelta(x, "D"), "Timedelta[ns]"),
            (lambda x: Timedelta(x, "D").as_unit("s"), "Timedelta[s]"),
        ],
    )
    def test_constructor_dict_timedelta64_index(self, klass: Callable[[int], Any], name: str) -> None:
        td_as_int: List[int] = [1, 2, 3, 4]
        data: Dict[int, Dict[Any, int]] = {i: {klass(s): 2 * i} for i, s in enumerate(td_as_int)}
        expected: DataFrame = DataFrame(
            [
                {0: 0, 1: None, 2: None, 3: None},
                {0: None, 1: 2, 2: None, 3: None},
                {0: None, 1: None, 2: 4, 3: None},
                {0: None, 1: None, 2: None, 3: 6},
            ],
            index=[Timedelta(td, "D") for td in td_as_int],
        )
        result: DataFrame = DataFrame(data)
        pd.testing.assert_frame_equal(result, expected)

    def test_constructor_period_dict(self) -> None:
        a = pd.PeriodIndex(["2012-01", "NaT", "2012-04"], freq="M")
        b = pd.PeriodIndex(["2012-02-01", "2012-03-01", "NaT"], freq="D")
        df: DataFrame = DataFrame({"a": a, "b": b})
        assert df["a"].dtype == a.dtype
        assert df["b"].dtype == b.dtype
        df = DataFrame({"a": a.astype(object).tolist(), "b": b.astype(object).tolist()})
        assert df["a"].dtype == a.dtype
        assert df["b"].dtype == b.dtype

    def test_constructor_dict_extension_scalar(self, ea_scalar_and_dtype: Any) -> None:
        ea_scalar, ea_dtype = ea_scalar_and_dtype
        df: DataFrame = DataFrame({"a": ea_scalar}, index=[0])
        assert df["a"].dtype == ea_dtype
        expected: DataFrame = DataFrame(index=[0], columns=["a"], data=ea_scalar)
        pd.testing.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        "data,dtype",
        [
            (Period("2020-01"), PeriodDtype("M")),
            (Interval(left=0, right=5), IntervalDtype("int64", "right")),
            (
                Timestamp("2011-01-01", tz="US/Eastern"),
                DatetimeTZDtype(unit="s", tz="US/Eastern"),
            ),
        ],
    )
    def test_constructor_extension_scalar_data(self, data: Any, dtype: Any) -> None:
        df: DataFrame = DataFrame(index=range(2), columns=["a", "b"], data=data)
        assert df["a"].dtype == dtype
        assert df["b"].dtype == dtype
        arr = pd.array([data] * 2, dtype=dtype)
        expected: DataFrame = DataFrame({"a": arr, "b": arr})
        pd.testing.assert_frame_equal(df, expected)

    def test_nested_dict_frame_constructor(self) -> None:
        rng = pd.period_range("1/1/2000", periods=5)
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 5)), columns=rng)
        data: Dict[Any, Any] = {}
        for col in df.columns:
            for row in df.index:
                data.setdefault(col, {})[row] = df._get_value(row, col)
        result: DataFrame = DataFrame(data, columns=rng)
        pd.testing.assert_frame_equal(result, df)
        data = {}
        for col in df.columns:
            for row in df.index:
                data.setdefault(row, {})[col] = df._get_value(row, col)
        result = DataFrame(data, index=rng).T
        pd.testing.assert_frame_equal(result, df)

    def _check_basic_constructor(self, empty: Callable[..., np.ndarray]) -> None:
        mat: np.ndarray = empty((2, 3), dtype=float)
        frame: DataFrame = DataFrame(mat, columns=["A", "B", "C"], index=[1, 2])
        assert len(frame.index) == 2
        assert len(frame.columns) == 3
        if empty is not np.ones:
            msg: str = r"Cannot convert non-finite values \(NA or inf\) to integer"
            import pytest
            with pytest.raises(IntCastingNaNError, match=msg):
                DataFrame(mat, columns=["A", "B", "C"], index=[1, 2], dtype=np.int64)
            return
        else:
            frame = DataFrame(
                mat, columns=["A", "B", "C"], index=[1, 2], dtype=np.int64
            )
            assert frame.values.dtype == np.int64
        msg = r"Shape of passed values is \(2, 3\), indices imply \(1, 3\)"
        import pytest
        with pytest.raises(ValueError, match=msg):
            DataFrame(mat, columns=["A", "B", "C"], index=[1])
        msg = r"Shape of passed values is \(2, 3\), indices imply \(2, 2\)"
        with pytest.raises(ValueError, match=msg):
            DataFrame(mat, columns=["A", "B"], index=[1, 2])
        with pytest.raises(ValueError, match="Must pass 2-d input"):
            DataFrame(empty((3, 3, 3)), columns=["A", "B", "C"], index=[1])
        frame = DataFrame(mat)
        pd.testing.assert_index_equal(frame.index, Index(range(2)), exact=True)
        pd.testing.assert_index_equal(frame.columns, Index(range(3)), exact=True)
        frame = DataFrame(mat, index=[1, 2])
        pd.testing.assert_index_equal(frame.columns, Index(range(3)), exact=True)
        frame = DataFrame(mat, columns=["A", "B", "C"])
        pd.testing.assert_index_equal(frame.index, Index(range(2)), exact=True)
        frame = DataFrame(empty((0, 3)))
        assert len(frame.index) == 0
        frame = DataFrame(empty((3, 0)))
        assert len(frame.columns) == 0

    def test_constructor_ndarray(self) -> None:
        self._check_basic_constructor(np.ones)
        frame: DataFrame = DataFrame(["foo", "bar"], index=[0, 1], columns=["A"])
        assert len(frame) == 2

    def test_constructor_maskedarray(self) -> None:
        self._check_basic_constructor(ma.masked_all)
        mat: ma.MaskedArray = ma.masked_all((2, 3), dtype=float)
        mat[0, 0] = 1.0
        mat[1, 2] = 2.0
        frame: DataFrame = DataFrame(mat, columns=["A", "B", "C"], index=[1, 2])
        assert 1.0 == frame["A"][1]
        assert 2.0 == frame["C"][2]
        mat = ma.masked_all((2, 3), dtype=float)
        frame = DataFrame(mat, columns=["A", "B", "C"], index=[1, 2])
        import numpy as np
        assert np.all(~np.asarray(frame == frame))

    @pytest.mark.filterwarnings(
        "ignore:elementwise comparison failed:DeprecationWarning"
    )
    def test_constructor_maskedarray_nonfloat(self) -> None:
        mat: ma.MaskedArray = ma.masked_all((2, 3), dtype=int)
        frame: DataFrame = DataFrame(mat, columns=["A", "B", "C"], index=[1, 2])
        assert len(frame.index) == 2
        assert len(frame.columns) == 3
        import numpy as np
        assert np.all(~np.asarray(frame == frame))
        frame = DataFrame(mat, columns=["A", "B", "C"], index=[1, 2], dtype=np.float64)
        assert frame.values.dtype == np.float64
        mat2: ma.MaskedArray = ma.copy(mat)
        mat2[0, 0] = 1
        mat2[1, 2] = 2
        frame = DataFrame(mat2, columns=["A", "B", "C"], index=[1, 2])
        assert 1 == frame["A"][1]
        assert 2 == frame["C"][2]
        mat: ma.MaskedArray = ma.masked_all((2, 3), dtype="M8[ns]")
        frame = DataFrame(mat, columns=["A", "B", "C"], index=[1, 2])
        assert len(frame.index) == 2
        assert len(frame.columns) == 3
        from pandas.core.dtypes.common import isna
        assert isna(frame).values.all()
        msg: str = r"datetime64\[ns\] values and dtype=int64 is not supported"
        import pytest
        with pytest.raises(TypeError, match=msg):
            DataFrame(mat, columns=["A", "B", "C"], index=[1, 2], dtype=np.int64)
        mat2 = ma.copy(mat)
        mat2[0, 0] = 1
        mat2[1, 2] = 2
        frame = DataFrame(mat2, columns=["A", "B", "C"], index=[1, 2])
        assert 1 == frame["A"].astype("i8")[1]
        assert 2 == frame["C"].astype("i8")[2]
        mat: ma.MaskedArray = ma.masked_all((2, 3), dtype=bool)
        frame = DataFrame(mat, columns=["A", "B", "C"], index=[1, 2])
        assert len(frame.index) == 2
        assert len(frame.columns) == 3
        assert np.all(~np.asarray(frame == frame))
        frame = DataFrame(mat, columns=["A", "B", "C"], index=[1, 2], dtype=object)
        assert frame.values.dtype == object
        mat2 = ma.copy(mat)
        mat2[0, 0] = True
        mat2[1, 2] = False
        frame = DataFrame(mat2, columns=["A", "B", "C"], index=[1, 2])
        assert frame["A"][1] is True
        assert frame["C"][2] is False

    def test_constructor_maskedarray_hardened(self) -> None:
        mat_hard: ma.MaskedArray = ma.masked_all((2, 2), dtype=float).harden_mask()
        result: DataFrame = DataFrame(mat_hard, columns=["A", "B"], index=[1, 2])
        expected: DataFrame = DataFrame(
            {"A": [np.nan, np.nan], "B": [np.nan, np.nan]},
            columns=["A", "B"],
            index=[1, 2],
            dtype=float,
        )
        pd.testing.assert_frame_equal(result, expected)
        mat_hard = ma.ones((2, 2), dtype=float).harden_mask()
        result = DataFrame(mat_hard, columns=["A", "B"], index=[1, 2])
        expected = DataFrame(
            {"A": [1.0, 1.0], "B": [1.0, 1.0]},
            columns=["A", "B"],
            index=[1, 2],
            dtype=float,
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_constructor_maskedrecarray_dtype(self) -> None:
        data: np.ma.MaskedArray = np.ma.array(
            np.ma.zeros(5, dtype=[("date", "<f8"), ("price", "<f8")]), mask=[False] * 5
        )
        from numpy.ma import mrecords
        data = data.view(mrecords.mrecarray)
        import pytest
        with pytest.raises(TypeError, match=r"Pass \{name: data\[name\]"):
            DataFrame(data, dtype=int)

    def test_constructor_corner_shape(self) -> None:
        df: DataFrame = DataFrame(index=[])
        assert df.values.shape == (0, 0)

    @pytest.mark.parametrize(
        "data,index,columns,dtype,expected",
        [
            (None, list(range(10)), ["a", "b"], object, np.object_),
            (None, None, ["a", "b"], "int64", np.dtype("int64")),
            (None, list(range(10)), ["a", "b"], int, np.dtype("float64")),
            ({}, None, ["foo", "bar"], None, np.object_),
            ({"b": 1}, list(range(10)), list("abc"), int, np.dtype("float64")),
        ],
    )
    def test_constructor_dtype(self, data: Any, index: Any, columns: Any, dtype: Any, expected: Any) -> None:
        df: DataFrame = DataFrame(data, index, columns, dtype)
        assert df.values.dtype == expected

    @pytest.mark.parametrize(
        "data,input_dtype,expected_dtype",
        (
            ([True, False, None], "boolean", pd.BooleanDtype),
            ([1.0, 2.0, None], "Float64", pd.Float64Dtype),
            ([1, 2, None], "Int64", pd.Int64Dtype),
            (["a", "b", "c"], "string", pd.StringDtype),
        ),
    )
    def test_constructor_dtype_nullable_extension_arrays(
        self, data: Any, input_dtype: Any, expected_dtype: Any
    ) -> None:
        df: DataFrame = DataFrame({"a": data}, dtype=input_dtype)
        assert df["a"].dtype == expected_dtype()

    def test_constructor_scalar_inference(self, using_infer_string: bool) -> None:
        data: Dict[str, Any] = {"int": 1, "bool": True, "float": 3.0, "complex": 4j, "object": "foo"}
        df: DataFrame = DataFrame(data, index=np.arange(10))
        assert df["int"].dtype == np.int64
        assert df["bool"].dtype == np.bool_
        assert df["float"].dtype == np.float64
        assert df["complex"].dtype == np.complex128
        if not using_infer_string:
            assert df["object"].dtype == np.object_
        else:
            assert df["object"].dtype == "str"

    def test_constructor_arrays_and_scalars(self) -> None:
        df: DataFrame = DataFrame({"a": np.random.default_rng(2).standard_normal(10), "b": True})
        exp: DataFrame = DataFrame({"a": df["a"].values, "b": [True] * 10})
        pd.testing.assert_frame_equal(df, exp)
        import pytest
        with pytest.raises(ValueError, match="must pass an index"):
            DataFrame({"a": False, "b": True})

    def test_constructor_DataFrame(self, float_frame: DataFrame) -> None:
        df: DataFrame = DataFrame(float_frame)
        pd.testing.assert_frame_equal(df, float_frame)
        df_casted: DataFrame = DataFrame(float_frame, dtype=np.int64)
        assert df_casted.values.dtype == np.int64

    def test_constructor_empty_dataframe(self) -> None:
        actual: DataFrame = DataFrame(DataFrame(), dtype="object")
        expected: DataFrame = DataFrame([], dtype="object")
        pd.testing.assert_frame_equal(actual, expected)

    def test_constructor_more(self, float_frame: DataFrame) -> None:
        arr: np.ndarray = np.random.default_rng(2).standard_normal(10)
        dm: DataFrame = DataFrame(arr, columns=["A"], index=np.arange(10))
        assert dm.values.ndim == 2
        arr = np.random.default_rng(2).standard_normal(0)
        dm = DataFrame(arr)
        assert dm.values.ndim == 2
        dm = DataFrame(columns=["A", "B"], index=np.arange(10))
        assert dm.values.shape == (10, 2)
        dm = DataFrame(columns=["A", "B"])
        assert dm.values.shape == (0, 2)
        dm = DataFrame(index=np.arange(10))
        assert dm.values.shape == (10, 0)
        mat: np.ndarray = np.array(["foo", "bar"], dtype=object).reshape(2, 1)
        msg: str = "could not convert string to float: 'foo'"
        import pytest
        with pytest.raises(ValueError, match=msg):
            DataFrame(mat, index=[0, 1], columns=[0], dtype=float)
        dm = DataFrame(DataFrame(float_frame._series))
        pd.testing.assert_frame_equal(dm, float_frame)
        dm = DataFrame(
            {"A": np.ones(10, dtype=int), "B": np.ones(10, dtype=np.float64)},
            index=np.arange(10),
        )
        assert len(dm.columns) == 2
        assert dm.values.dtype == np.float64

    def test_constructor_empty_list(self) -> None:
        df: DataFrame = DataFrame([], index=[])
        expected: DataFrame = DataFrame(index=[])
        pd.testing.assert_frame_equal(df, expected)
        df = DataFrame([], columns=["A", "B"])
        expected = DataFrame({}, columns=["A", "B"])
        pd.testing.assert_frame_equal(df, expected)
        def empty_gen() -> Iterator[Any]:
            yield from ()
        df = DataFrame(empty_gen(), columns=["A", "B"])
        pd.testing.assert_frame_equal(df, expected)

    def test_constructor_list_of_lists(self, using_infer_string: bool) -> None:
        df: DataFrame = DataFrame(data=[[1, "a"], [2, "b"]], columns=["num", "str"])
        from pandas.api.types import is_integer_dtype
        assert is_integer_dtype(df["num"])
        if not using_infer_string:
            assert df["str"].dtype == np.object_
        else:
            assert df["str"].dtype == "str"
        expected: DataFrame = DataFrame(np.arange(10))
        data: List[np.ndarray] = [np.array(x) for x in range(10)]
        result: DataFrame = DataFrame(data)
        pd.testing.assert_frame_equal(result, expected)

    def test_nested_pandasarray_matches_nested_ndarray(self) -> None:
        ser: Series = Series([1, 2])
        arr: np.ndarray = np.array([None, None], dtype=object)
        arr[0] = ser
        arr[1] = ser * 2
        df: DataFrame = DataFrame(arr)
        expected: DataFrame = DataFrame(pd.array(arr))
        pd.testing.assert_frame_equal(df, expected)
        assert df.shape == (2, 1)
        pd.testing.assert_numpy_array_equal(df[0].values, arr)

    def test_constructor_list_like_data_nested_list_column(self) -> None:
        arrays: List[List[str]] = [list("abcd"), list("cdef")]
        result: DataFrame = DataFrame([[1, 2, 3, 4], [4, 5, 6, 7]], columns=arrays)
        mi: MultiIndex = MultiIndex.from_arrays(arrays)
        expected: DataFrame = DataFrame([[1, 2, 3, 4], [4, 5, 6, 7]], columns=mi)
        pd.testing.assert_frame_equal(result, expected)

    def test_constructor_wrong_length_nested_list_column(self) -> None:
        arrays: List[List[str]] = [list("abc"), list("cde")]
        msg: str = "3 columns passed, passed data had 4"
        import pytest
        with pytest.raises(ValueError, match=msg):
            DataFrame([[1, 2, 3, 4], [4, 5, 6, 7]], columns=arrays)

    def test_constructor_unequal_length_nested_list_column(self) -> None:
        arrays: List[List[str]] = [list("abcd"), list("cde")]
        msg: str = "all arrays must be same length"
        import pytest
        with pytest.raises(ValueError, match=msg):
            DataFrame([[1, 2, 3, 4], [4, 5, 6, 7]], columns=arrays)

    @pytest.mark.parametrize(
        "data",
        [
            [[Timestamp("2021-01-01")]],
            [{"x": Timestamp("2021-01-01")}],
            {"x": [Timestamp("2021-01-01")]},
            {"x": Timestamp("2021-01-01")},
        ],
    )
    def test_constructor_one_element_data_list(self, data: Any) -> None:
        result: DataFrame = DataFrame(data, index=range(3), columns=["x"])
        expected: DataFrame = DataFrame({"x": [Timestamp("2021-01-01")] * 3})
        pd.testing.assert_frame_equal(result, expected)

    def test_constructor_sequence_like(self) -> None:
        class DummyContainer(abc.Sequence):
            def __init__(self, lst: List[Any]) -> None:
                self._lst: List[Any] = lst
            def __getitem__(self, n: int) -> Any:
                return self._lst.__getitem__(n)
            def __len__(self) -> int:
                return self._lst.__len__()
        lst_containers: List[DummyContainer] = [DummyContainer([1, "a"]), DummyContainer([2, "b"])]
        columns: List[str] = ["num", "str"]
        result: DataFrame = DataFrame(lst_containers, columns=columns)
        expected: DataFrame = DataFrame([[1, "a"], [2, "b"]], columns=columns)
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)

    def test_constructor_stdlib_array(self) -> None:
        result: DataFrame = DataFrame({"A": array.array("i", range(10))})
        expected: DataFrame = DataFrame({"A": list(range(10))})
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)
        expected = DataFrame([list(range(10)), list(range(10))])
        result = DataFrame([array.array("i", range(10)), array.array("i", range(10))])
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)

    def test_constructor_range(self) -> None:
        result: DataFrame = DataFrame(range(10))
        expected: DataFrame = DataFrame(list(range(10)))
        pd.testing.assert_frame_equal(result, expected)

    def test_constructor_list_of_ranges(self) -> None:
        result: DataFrame = DataFrame([range(10), range(10)])
        expected: DataFrame = DataFrame([list(range(10)), list(range(10))])
        pd.testing.assert_frame_equal(result, expected)

    def test_constructor_iterable(self) -> None:
        class Iter:
            def __iter__(self) -> Iterator:
                for i in range(10):
                    yield [1, 2, 3]
        expected: DataFrame = DataFrame([[1, 2, 3]] * 10)
        result: DataFrame = DataFrame(Iter())
        pd.testing.assert_frame_equal(result, expected)

    def test_constructor_iterator(self) -> None:
        result: DataFrame = DataFrame(iter(range(10)))
        expected: DataFrame = DataFrame(list(range(10)))
        pd.testing.assert_frame_equal(result, expected)

    def test_constructor_list_of_iterators(self) -> None:
        result: DataFrame = DataFrame([iter(range(10)), iter(range(10))])
        expected: DataFrame = DataFrame([list(range(10)), list(range(10))])
        pd.testing.assert_frame_equal(result, expected)

    def test_constructor_generator(self) -> None:
        gen1 = (i for i in range(10))
        gen2 = (i for i in range(10))
        expected: DataFrame = DataFrame([list(range(10)), list(range(10))])
        result: DataFrame = DataFrame([gen1, gen2])
        pd.testing.assert_frame_equal(result, expected)
        gen = ([i, "a"] for i in range(10))
        result = DataFrame(gen)
        expected = DataFrame({0: range(10), 1: "a"})
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)

    def test_constructor_list_of_dicts(self) -> None:
        result: DataFrame = DataFrame([{}])
        expected: DataFrame = DataFrame(index=RangeIndex(1), columns=[])
        pd.testing.assert_frame_equal(result, expected)

    def test_constructor_ordered_dict_nested_preserve_order(self) -> None:
        nested1: OrderedDict = OrderedDict([("b", 1), ("a", 2)])
        nested2: OrderedDict = OrderedDict([("b", 2), ("a", 5)])
        data: OrderedDict = OrderedDict([("col2", nested1), ("col1", nested2)])
        result: DataFrame = DataFrame(data)
        data_expected: Dict[str, List[int]] = {"col2": [1, 2], "col1": [2, 5]}
        expected: DataFrame = DataFrame(data=data_expected, index=["b", "a"])
        pd.testing.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("dict_type", [dict, OrderedDict])
    def test_constructor_ordered_dict_preserve_order(self, dict_type: Callable[..., Any]) -> None:
        expected: DataFrame = DataFrame([[2, 1]], columns=["b", "a"])
        data: Any = dict_type()
        data["b"] = [2]
        data["a"] = [1]
        result: DataFrame = DataFrame(data)
        pd.testing.assert_frame_equal(result, expected)
        data = dict_type()
        data["b"] = 2
        data["a"] = 1
        result = DataFrame([data])
        pd.testing.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("dict_type", [dict, OrderedDict])
    def test_constructor_ordered_dict_conflicting_orders(self, dict_type: Callable[..., Any]) -> None:
        row_one: Any = dict_type()
        row_one["b"] = 2
        row_one["a"] = 1
        row_two: Any = dict_type()
        row_two["a"] = 1
        row_two["b"] = 2
        row_three: Dict[str, int] = {"b": 2, "a": 1}
        expected: DataFrame = DataFrame([[2, 1], [2, 1]], columns=["b", "a"])
        result: DataFrame = DataFrame([row_one, row_two])
        pd.testing.assert_frame_equal(result, expected)
        expected = DataFrame([[2, 1], [2, 1], [2, 1]], columns=["b", "a"])
        result = DataFrame([row_one, row_two, row_three])
        pd.testing.assert_frame_equal(result, expected)

    def test_constructor_list_of_series_aligned_index(self) -> None:
        series: List[Series] = [Series(i, index=["b", "a", "c"], name=str(i)) for i in range(3)]
        result: DataFrame = DataFrame(series)
        expected: DataFrame = DataFrame(
            {"b": [0, 1, 2], "a": [0, 1, 2], "c": [0, 1, 2]},
            columns=["b", "a", "c"],
            index=["0", "1", "2"],
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_constructor_list_of_derived_dicts(self) -> None:
        class CustomDict(dict):
            pass
        d: Dict[str, Any] = {"a": 1.5, "b": 3}
        data_custom: List[Dict[str, Any]] = [CustomDict(d)]
        data: List[Dict[str, Any]] = [d]
        result_custom: DataFrame = DataFrame(data_custom)
        result: DataFrame = DataFrame(data)
        pd.testing.assert_frame_equal(result, result_custom)

    def test_constructor_ragged(self) -> None:
        data: Dict[str, np.ndarray] = {
            "A": np.random.default_rng(2).standard_normal(10),
            "B": np.random.default_rng(2).standard_normal(8),
        }
        import pytest
        with pytest.raises(ValueError, match="All arrays must be of the same length"):
            DataFrame(data)

    def test_constructor_scalar(self) -> None:
        idx: Index = Index(range(3))
        df: DataFrame = DataFrame({"a": 0}, index=idx)
        expected: DataFrame = DataFrame({"a": [0, 0, 0]}, index=idx)
        pd.testing.assert_frame_equal(df, expected, check_dtype=False)

    def test_constructor_Series_copy_bug(self, float_frame: DataFrame) -> None:
        df: DataFrame = DataFrame(float_frame["A"], index=float_frame.index, columns=["A"])
        df.copy()

    def test_constructor_mixed_dict_and_Series(self) -> None:
        data: Dict[str, Any] = {}
        data["A"] = {"foo": 1, "bar": 2, "baz": 3}
        data["B"] = Series([4, 3, 2, 1], index=["bar", "qux", "baz", "foo"])
        result: DataFrame = DataFrame(data)
        assert result.index.is_monotonic_increasing
        import pytest
        with pytest.raises(ValueError, match="ambiguous ordering"):
            DataFrame({"A": ["a", "b"], "B": {"a": "a", "b": "b"}})
        result = DataFrame({"A": ["a", "b"], "B": Series(["a", "b"], index=["a", "b"])})
        expected: DataFrame = DataFrame({"A": ["a", "b"], "B": ["a", "b"]}, index=["a", "b"])
        pd.testing.assert_frame_equal(result, expected)

    def test_constructor_mixed_type_rows(self) -> None:
        data: List[Any] = [[1, 2], (3, 4)]
        result: DataFrame = DataFrame(data)
        expected: DataFrame = DataFrame([[1, 2], [3, 4]])
        pd.testing.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "tuples,lists",
        [
            ((), []),
            (((),), [[]]),
            (((), ()), [(), ()]),
            (((), ()), [[], []]),
            (([], []), [[], []]),
            (([1], [2]), [[1], [2]]),
            (([1, 2, 3], [4, 5, 6]), [[1, 2, 3], [4, 5, 6]]),
        ],
    )
    def test_constructor_tuple(self, tuples: Any, lists: Any) -> None:
        result: DataFrame = DataFrame(tuples)
        expected: DataFrame = DataFrame(lists)
        pd.testing.assert_frame_equal(result, expected)

    def test_constructor_list_of_tuples(self) -> None:
        result: DataFrame = DataFrame({"A": [(1, 2), (3, 4)]})
        expected: DataFrame = DataFrame({"A": Series([(1, 2), (3, 4)])})
        pd.testing.assert_frame_equal(result, expected)

    def test_constructor_list_of_namedtuples(self) -> None:
        named_tuple = namedtuple("Pandas", list("ab"))
        tuples: List[Any] = [named_tuple(1, 3), named_tuple(2, 4)]
        expected: DataFrame = DataFrame({"a": [1, 2], "b": [3, 4]})
        result: DataFrame = DataFrame(tuples)
        pd.testing.assert_frame_equal(result, expected)
        expected = DataFrame({"y": [1, 2], "z": [3, 4]})
        result = DataFrame(tuples, columns=["y", "z"])
        pd.testing.assert_frame_equal(result, expected)

    def test_constructor_list_of_dataclasses(self) -> None:
        Point = functools.partial(__import__("dataclasses").make_dataclass, "Point", [("x", int), ("y", int)])
        data: List[Any] = [Point(0, 3), Point(1, 3)]
        expected: DataFrame = DataFrame({"x": [0, 1], "y": [3, 3]})
        result: DataFrame = DataFrame(data)
        pd.testing.assert_frame_equal(result, expected)

    def test_constructor_list_of_dataclasses_with_varying_types(self) -> None:
        Point = functools.partial(__import__("dataclasses").make_dataclass, "Point", [("x", int), ("y", int)])
        HLine = functools.partial(__import__("dataclasses").make_dataclass, "HLine", [("x0", int), ("x1", int), ("y", int)])
        data: List[Any] = [Point(0, 3), HLine(1, 3, 3)]
        expected: DataFrame = DataFrame(
            {"x": [0, np.nan], "y": [3, 3], "x0": [np.nan, 1], "x1": [np.nan, 3]}
        )
        result: DataFrame = DataFrame(data)
        pd.testing.assert_frame_equal(result, expected)

    def test_constructor_list_of_dataclasses_error_thrown(self) -> None:
        Point = functools.partial(__import__("dataclasses").make_dataclass, "Point", [("x", int), ("y", int)])
        msg: str = "asdict() should be called on dataclass instances"
        import pytest
        with pytest.raises(TypeError, match=re.escape(msg)):
            DataFrame([Point(0, 0), {"x": 1, "y": 0}])

    def test_constructor_list_of_dict_order(self) -> None:
        data: List[Dict[str, Any]] = [
            {"First": 1, "Second": 4, "Third": 7, "Fourth": 10},
            {"Second": 5, "First": 2, "Fourth": 11, "Third": 8},
            {"Second": 6, "First": 3, "Fourth": 12, "Third": 9, "YYY": 14, "XXX": 13},
        ]
        expected: DataFrame = DataFrame(
            {
                "First": [1, 2, 3],
                "Second": [4, 5, 6],
                "Third": [7, 8, 9],
                "Fourth": [10, 11, 12],
                "YYY": [None, None, 14],
                "XXX": [None, None, 13],
            }
        )
        result: DataFrame = DataFrame(data)
        pd.testing.assert_frame_equal(result, expected)

    def test_constructor_Series_named(self) -> None:
        a: Series = Series([1, 2, 3], index=["a", "b", "c"], name="x")
        df: DataFrame = DataFrame(a)
        assert df.columns[0] == "x"
        pd.testing.assert_index_equal(df.index, a.index)
        arr: np.ndarray = np.random.default_rng(2).standard_normal(10)
        s: Series = Series(arr, name="x")
        df = DataFrame(s)
        expected: DataFrame = DataFrame({"x": s})
        pd.testing.assert_frame_equal(df, expected)
        s = Series(arr, index=range(3, 13))
        df = DataFrame(s)
        expected = DataFrame({0: s})
        pd.testing.assert_frame_equal(df, expected, check_column_type=False)
        msg: str = r"Shape of passed values is \(10, 1\), indices imply \(10, 2\)"
        import pytest
        with pytest.raises(ValueError, match=msg):
            DataFrame(s, columns=[1, 2])
        a = Series([], name="x", dtype=object)
        df = DataFrame(a)
        assert df.columns[0] == "x"
        s1: Series = Series(arr, name="x")
        df = DataFrame([s1, arr]).T
        expected = DataFrame({"x": s1, "Unnamed 0": arr}, columns=["x", "Unnamed 0"])
        pd.testing.assert_frame_equal(df, expected)
        df = DataFrame([arr, s1]).T
        expected = DataFrame({1: s1, 0: arr}, columns=range(2))
        pd.testing.assert_frame_equal(df, expected)

    def test_constructor_Series_named_and_columns(self) -> None:
        s0: Series = Series(range(5), name=0)
        s1: Series = Series(range(5), name=1)
        pd.testing.assert_frame_equal(DataFrame(s0, columns=[0]), s0.to_frame())
        pd.testing.assert_frame_equal(DataFrame(s1, columns=[1]), s1.to_frame())
        assert DataFrame(s0, columns=[1]).empty
        assert DataFrame(s1, columns=[0]).empty

    def test_constructor_Series_differently_indexed(self) -> None:
        s1: Series = Series([1, 2, 3], index=["a", "b", "c"], name="x")
        s2: Series = Series([1, 2, 3], index=["a", "b", "c"])
        other_index: Index = Index(["a", "b"])
        df1: DataFrame = DataFrame(s1, index=other_index)
        exp1: DataFrame = DataFrame(s1.reindex(other_index))
        assert df1.columns[0] == "x"
        pd.testing.assert_frame_equal(df1, exp1)
        df2: DataFrame = DataFrame(s2, index=other_index)
        exp2: DataFrame = DataFrame(s2.reindex(other_index))
        assert df2.columns[0] == 0
        pd.testing.assert_index_equal(df2.index, other_index)
        pd.testing.assert_frame_equal(df2, exp2)

    @pytest.mark.parametrize(
        "name_in1,name_in2,name_in3,name_out",
        [
            ("idx", "idx", "idx", "idx"),
            ("idx", "idx", None, None),
            ("idx", None, None, None),
            ("idx1", "idx2", None, None),
            ("idx1", "idx1", "idx2", None),
            ("idx1", "idx2", "idx3", None),
            (None, None, None, None),
        ],
    )
    def test_constructor_index_names(self, name_in1: Optional[Any], name_in2: Optional[Any],
                                       name_in3: Optional[Any], name_out: Optional[Any]) -> None:
        indices: List[Index] = [
            Index(["a", "b", "c"], name=name_in1),
            Index(["b", "c", "d"], name=name_in2),
            Index(["c", "d", "e"], name=name_in3),
        ]
        series: Dict[str, Series] = {
            c: Series([0, 1, 2], index=i) for i, c in zip(indices, ["x", "y", "z"])
        }
        result: DataFrame = DataFrame(series)
        exp_ind: Index = Index(["a", "b", "c", "d", "e"], name=name_out)
        expected: DataFrame = DataFrame(
            {
                "x": [0, 1, 2, np.nan, np.nan],
                "y": [np.nan, 0, 1, 2, np.nan],
                "z": [np.nan, np.nan, 0, 1, 2],
            },
            index=exp_ind,
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_constructor_manager_resize(self, float_frame: DataFrame) -> None:
        index: List[Any] = list(float_frame.index[:5])
        columns: List[Any] = list(float_frame.columns[:3])
        msg: str = "Passing a BlockManager to DataFrame"
        import pytest
        with pd.testing.assert_produces_warning(DeprecationWarning, match=msg, check_stacklevel=False):
            result: DataFrame = DataFrame(float_frame._mgr, index=index, columns=columns)
        pd.testing.assert_index_equal(result.index, Index(index))
        pd.testing.assert_index_equal(result.columns, Index(columns))

    def test_constructor_mix_series_nonseries(self, float_frame: DataFrame) -> None:
        df: DataFrame = DataFrame(
            {"A": float_frame["A"], "B": list(float_frame["B"])}, columns=["A", "B"]
        )
        pd.testing.assert_frame_equal(df, float_frame.loc[:, ["A", "B"]])
        msg: str = "does not match index length"
        import pytest
        with pytest.raises(ValueError, match=msg):
            DataFrame({"A": float_frame["A"], "B": list(float_frame["B"])[:-2]})

    def test_constructor_miscast_na_int_dtype(self) -> None:
        msg: str = r"Cannot convert non-finite values \(NA or inf\) to integer"
        import pytest
        with pytest.raises(IntCastingNaNError, match=msg):
            DataFrame([[np.nan, 1], [1, 0]], dtype=np.int64)

    def test_constructor_column_duplicates(self) -> None:
        df: DataFrame = DataFrame([[8, 5]], columns=["a", "a"])
        edf: DataFrame = DataFrame([[8, 5]])
        edf.columns = ["a", "a"]
        pd.testing.assert_frame_equal(df, edf)
        idf: DataFrame = DataFrame.from_records([(8, 5)], columns=["a", "a"])
        pd.testing.assert_frame_equal(idf, edf)

    def test_constructor_empty_with_string_dtype(self, using_infer_string: bool) -> None:
        expected: DataFrame = DataFrame(index=[0, 1], columns=[0, 1], dtype=object)
        expected_str: DataFrame = DataFrame(
            index=[0, 1], columns=[0, 1], dtype=pd.StringDtype(na_value=np.nan)
        )
        df: DataFrame = DataFrame(index=[0, 1], columns=[0, 1], dtype=str)
        if using_infer_string:
            pd.testing.assert_frame_equal(df, expected_str)
        else:
            pd.testing.assert_frame_equal(df, expected)
        df = DataFrame(index=[0, 1], columns=[0, 1], dtype=np.str_)
        pd.testing.assert_frame_equal(df, expected)
        df = DataFrame(index=[0, 1], columns=[0, 1], dtype="U5")
        pd.testing.assert_frame_equal(df, expected)

    def test_constructor_empty_with_string_extension(self, nullable_string_dtype: Any) -> None:
        expected: DataFrame = DataFrame(columns=["c1"], dtype=nullable_string_dtype)
        df: DataFrame = DataFrame(columns=["c1"], dtype=nullable_string_dtype)
        pd.testing.assert_frame_equal(df, expected)

    def test_constructor_single_value(self) -> None:
        df: DataFrame = DataFrame(0.0, index=[1, 2, 3], columns=["a", "b", "c"])
        pd.testing.assert_frame_equal(
            df, DataFrame(np.zeros(df.shape).astype("float64"), df.index, df.columns)
        )
        df = DataFrame(0, index=[1, 2, 3], columns=["a", "b", "c"])
        pd.testing.assert_frame_equal(
            df, DataFrame(np.zeros(df.shape).astype("int64"), df.index, df.columns)
        )
        df = DataFrame("a", index=[1, 2], columns=["a", "c"])
        pd.testing.assert_frame_equal(
            df,
            DataFrame(
                np.array([["a", "a"], ["a", "a"]], dtype=object),
                index=[1, 2],
                columns=["a", "c"],
            ),
        )
        msg: str = "DataFrame constructor not properly called!"
        import pytest
        with pytest.raises(ValueError, match=msg):
            DataFrame("a", [1, 2])
        with pytest.raises(ValueError, match=msg):
            DataFrame("a", columns=["a", "c"])
        msg = "incompatible data and dtype"
        with pytest.raises(TypeError, match=msg):
            DataFrame("a", [1, 2], ["a", "c"], float)

    def test_constructor_with_datetimes(self, using_infer_string: bool) -> None:
        intname: str = np.dtype(int).name
        floatname: str = np.dtype(np.float64).name
        objectname: str = np.dtype(np.object_).name
        df: DataFrame = DataFrame(
            {
                "A": 1,
                "B": "foo",
                "C": "bar",
                "D": Timestamp("20010101"),
                "E": datetime(2001, 1, 2, 0, 0),
            },
            index=np.arange(10),
        )
        result: Series = df.dtypes
        expected: Series = Series(
            [np.dtype("int64")]
            + [
                np.dtype(objectname)
                if not using_infer_string
                else pd.StringDtype(na_value=np.nan)
            ]
            * 2
            + [np.dtype("M8[s]"), np.dtype("M8[us]")],
            index=list("ABCDE"),
        )
        pd.testing.assert_series_equal(result, expected)
        df = DataFrame(
            {
                "a": 1.0,
                "b": 2,
                "c": "foo",
                floatname: np.array(1.0, dtype=floatname),
                intname: np.array(1, dtype=intname),
            },
            index=np.arange(10),
        )
        result = df.dtypes
        expected = Series(
            [np.dtype("float64")]
            + [np.dtype("int64")]
            + [
                np.dtype("object")
                if not using_infer_string
                else pd.StringDtype(na_value=np.nan)
            ]
            + [np.dtype("float64")]
            + [np.dtype(intname)],
            index=["a", "b", "c", floatname, intname],
        )
        pd.testing.assert_series_equal(result, expected)
        df = DataFrame(
            {
                "a": 1.0,
                "b": 2,
                "c": "foo",
                floatname: np.array([1.0] * 10, dtype=floatname),
                intname: np.array([1] * 10, dtype=intname),
            },
            index=np.arange(10),
        )
        result = df.dtypes
        expected = Series(
            [np.dtype("float64")]
            + [np.dtype("int64")]
            + [
                np.dtype("object")
                if not using_infer_string
                else pd.StringDtype(na_value=np.nan)
            ]
            + [np.dtype("float64")]
            + [np.dtype(intname)],
            index=["a", "b", "c", floatname, intname],
        )
        pd.testing.assert_series_equal(result, expected)

    def test_constructor_with_datetimes1(self) -> None:
        ind: DatetimeIndex = pd.date_range(start="2000-01-01", freq="D", periods=10)
        datetimes: List[Any] = [ts.to_pydatetime() for ts in ind]
        datetime_s: Series = Series(datetimes)
        assert datetime_s.dtype == "M8[us]"

    def test_constructor_with_datetimes2(self) -> None:
        ind: DatetimeIndex = pd.date_range(start="2000-01-01", freq="D", periods=10)
        datetimes: List[Any] = [ts.to_pydatetime() for ts in ind]
        dates: List[Any] = [ts.date() for ts in ind]
        df: DataFrame = DataFrame(datetimes, columns=["datetimes"])
        df["dates"] = dates
        result: Series = df.dtypes
        expected: Series = Series(
            [np.dtype("datetime64[us]"), np.dtype("object")],
            index=["datetimes", "dates"],
        )
        pd.testing.assert_series_equal(result, expected)

    def test_constructor_with_datetimes3(self) -> None:
        dt: Timestamp = Timestamp("2012-01-01", tzinfo=pd._libs.tslibs.timezones.gettz("US/Eastern"))
        df: DataFrame = DataFrame({"End Date": dt}, index=[0])
        assert df.iat[0, 0] == dt
        expected: Series = Series({"End Date": "datetime64[us, US/Eastern]"}, dtype=object)
        pd.testing.assert_series_equal(
            df.dtypes, Series({"End Date": "datetime64[us, US/Eastern]"}, dtype=object)
        )
        df = DataFrame([{"End Date": dt}])
        assert df.iat[0, 0] == dt
        pd.testing.assert_series_equal(
            df.dtypes, Series({"End Date": "datetime64[us, US/Eastern]"}, dtype=object)
        )

    def test_constructor_with_datetimes4(self) -> None:
        dr: DatetimeIndex = pd.date_range("20130101", periods=3)
        df: DataFrame = DataFrame({"value": dr})
        assert df.iat[0, 0].tz is None
        dr = pd.date_range("20130101", periods=3, tz="UTC")
        df = DataFrame({"value": dr})
        assert str(df.iat[0, 0].tz) == "UTC"
        dr = pd.date_range("20130101", periods=3, tz="US/Eastern")
        df = DataFrame({"value": dr})
        assert str(df.iat[0, 0].tz) == "US/Eastern"

    @pytest.mark.xfail(using_string_dtype(), reason="TODO(infer_string)")
    def test_constructor_with_datetimes5(self) -> None:
        i: DatetimeIndex = pd.date_range("1/1/2011", periods=5, freq="10s", tz="US/Eastern")
        expected: DataFrame = DataFrame({"a": i.to_series().reset_index(drop=True)})
        df: DataFrame = DataFrame()
        df["a"] = i
        pd.testing.assert_frame_equal(df, expected)
        df = DataFrame({"a": i})
        pd.testing.assert_frame_equal(df, expected)

    def test_constructor_with_datetimes6(self) -> None:
        i: DatetimeIndex = pd.date_range("1/1/2011", periods=5, freq="10s", tz="US/Eastern")
        i_no_tz: DatetimeIndex = pd.date_range("1/1/2011", periods=5, freq="10s")
        df: DataFrame = DataFrame({"a": i, "b": i_no_tz})
        expected: DataFrame = DataFrame({"a": i.to_series().reset_index(drop=True), "b": i_no_tz})
        pd.testing.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        "arr",
        [
            np.array([None, None, None, None, datetime.now(), None]),
            np.array([None, None, datetime.now(), None]),
            [[np.datetime64("NaT")], [None]],
            [[np.datetime64("NaT")], [pd.NaT]],
            [[None], [np.datetime64("NaT")]],
            [[None], [pd.NaT]],
            [[pd.NaT], [np.datetime64("NaT")]],
            [[pd.NaT], [None]],
        ],
    )
    def test_constructor_datetimes_with_nulls(self, arr: Any) -> None:
        result: Series = DataFrame(arr).dtypes
        unit: str = "ns"
        if isinstance(arr, np.ndarray):
            unit = "us"
        elif not any(isinstance(x, np.datetime64) for y in arr for x in y):
            unit = "s"
        expected: Series = Series([np.dtype(f"datetime64[{unit}]")])
        pd.testing.assert_series_equal(result, expected)

    @pytest.mark.parametrize("order", ["K", "A", "C", "F"])
    @pytest.mark.parametrize(
        "unit",
        ["M", "D", "h", "m", "s", "ms", "us", "ns"],
    )
    def test_constructor_datetimes_non_ns(self, order: str, unit: str) -> None:
        dtype: str = f"datetime64[{unit}]"
        na: np.ndarray = np.array(
            [
                ["2015-01-01", "2015-01-02", "2015-01-03"],
                ["2017-01-01", "2017-01-02", "2017-02-03"],
            ],
            dtype=dtype,
            order=order,
        )
        df: DataFrame = DataFrame(na)
        expected: DataFrame = DataFrame(na.astype("M8[ns]"))
        if unit in ["M", "D", "h", "m"]:
            with pytest.raises(TypeError, match="Cannot cast"):
                expected.astype(dtype)
            expected = expected.astype("datetime64[s]")
        else:
            expected = expected.astype(dtype=dtype)
        pd.testing.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("order", ["K", "A", "C", "F"])
    @pytest.mark.parametrize(
        "unit",
        [
            "D",
            "h",
            "m",
            "s",
            "ms",
            "us",
            "ns",
        ],
    )
    def test_constructor_timedelta_non_ns(self, order: str, unit: str) -> None:
        dtype: str = f"timedelta64[{unit}]"
        na: np.ndarray = np.array(
            [
                [np.timedelta64(1, "D"), np.timedelta64(2, "D")],
                [np.timedelta64(4, "D"), np.timedelta64(5, "D")],
            ],
            dtype=dtype,
            order=order,
        )
        df: DataFrame = DataFrame(na)
        exp_unit: str = unit if unit not in ["D", "h", "m"] else "s"
        exp_dtype: np.dtype = np.dtype(f"m8[{exp_unit}]")
        expected: DataFrame = DataFrame(
            [
                [Timedelta(1, "D"), Timedelta(2, "D")],
                [Timedelta(4, "D"), Timedelta(5, "D")],
            ],
            dtype=exp_dtype,
        )
        pd.testing.assert_frame_equal(df, expected)

    def test_constructor_for_list_with_dtypes(self, using_infer_string: bool) -> None:
        df: DataFrame = DataFrame([np.arange(5) for x in range(5)])
        result: Series = df.dtypes
        expected: Series = Series([np.dtype("int")] * 5)
        pd.testing.assert_series_equal(result, expected)
        df = DataFrame([np.array(np.arange(5), dtype="int32") for x in range(5)])
        result = df.dtypes
        expected = Series([np.dtype("int32")] * 5)
        pd.testing.assert_series_equal(result, expected)
        df = DataFrame({"a": [2**31, 2**31 + 1]})
        assert df.dtypes.iloc[0] == np.dtype("int64")
        df = DataFrame([1, 2])
        assert df.dtypes.iloc[0] == np.dtype("int64")
        df = DataFrame([1.0, 2.0])
        assert df.dtypes.iloc[0] == np.dtype("float64")
        df = DataFrame({"a": [1, 2]})
        assert df.dtypes.iloc[0] == np.dtype("int64")
        df = DataFrame({"a": [1.0, 2.0]})
        assert df.dtypes.iloc[0] == np.dtype("float64")
        df = DataFrame({"a": 1}, index=range(3))
        assert df.dtypes.iloc[0] == np.dtype("int64")
        df = DataFrame({"a": 1.0}, index=range(3))
        assert df.dtypes.iloc[0] == np.dtype("float64")
        df = DataFrame(
            {
                "a": [1, 2, 4, 7],
                "b": [1.2, 2.3, 5.1, 6.3],
                "c": list("abcd"),
                "d": [datetime(2000, 1, 1) for i in range(4)],
                "e": [1.0, 2, 4.0, 7],
            }
        )
        result = df.dtypes
        expected = Series(
            [
                np.dtype("int64"),
                np.dtype("float64"),
                np.dtype("object") if not using_infer_string else pd.StringDtype(na_value=np.nan),
                np.dtype("datetime64[us]"),
                np.dtype("float64"),
            ],
            index=list("abcde"),
        )
        pd.testing.assert_series_equal(result, expected)

    def test_constructor_frame_copy(self, float_frame: DataFrame) -> None:
        cop: DataFrame = DataFrame(float_frame, copy=True)
        cop["A"] = 5
        assert (cop["A"] == 5).all()
        assert not (float_frame["A"] == 5).all()

    def test_constructor_frame_shallow_copy(self, float_frame: DataFrame) -> None:
        orig: DataFrame = float_frame.copy()
        cop: DataFrame = DataFrame(float_frame)
        assert cop._mgr is not float_frame._mgr
        cop.index = np.arange(len(cop))
        pd.testing.assert_frame_equal(float_frame, orig)

    def test_constructor_ndarray_copy(self, float_frame: DataFrame) -> None:
        arr: np.ndarray = float_frame.values.copy()
        df: DataFrame = DataFrame(arr)
        arr[5] = 5
        assert not (df.values[5] == 5).all()
        df = DataFrame(arr, copy=True)
        arr[6] = 6
        assert not (df.values[6] == 6).all()

    def test_constructor_series_copy(self, float_frame: DataFrame) -> None:
        series: Any = float_frame._series
        df: DataFrame = DataFrame({"A": series["A"]}, copy=True)
        df.loc[df.index[0] : df.index[-1], "A"] = 5
        assert not (series["A"] == 5).all()

    @pytest.mark.parametrize(
        "df",
        [
            DataFrame([[1, 2, 3], [4, 5, 6]], index=[1, np.nan]),
            DataFrame([[1, 2, 3], [4, 5, 6]], columns=[1.1, 2.2, np.nan]),
            DataFrame([[0, 1, 2, 3], [4, 5, 6, 7]], columns=[np.nan, 1.1, 2.2, np.nan]),
            DataFrame(
                [[0.0, 1, 2, 3.0], [4, 5, 6, 7]], columns=[np.nan, 1.1, 2.2, np.nan]
            ),
            DataFrame([[0.0, 1, 2, 3.0], [4, 5, 6, 7]], columns=[np.nan, 1, 2, 2]),
        ],
    )
    def test_constructor_with_nas(self, df: DataFrame) -> None:
        for i in range(len(df.columns)):
            df.iloc[:, i]
        import numpy as np
        indexer: np.ndarray = np.arange(len(df.columns))[np.isnan(df.columns)]
        if len(indexer) == 0:
            import pytest
            with pytest.raises(KeyError, match="^nan$"):
                df.loc[:, np.nan]
        elif len(indexer) == 1:
            pd.testing.assert_series_equal(df.iloc[:, indexer[0]], df.loc[:, np.nan])
        else:
            pd.testing.assert_frame_equal(df.iloc[:, indexer], df.loc[:, np.nan])

    def test_constructor_lists_to_object_dtype(self) -> None:
        d: DataFrame = DataFrame({"a": [np.nan, False]})
        assert d["a"].dtype == np.object_
        assert not d["a"][1]

    def test_constructor_ndarray_categorical_dtype(self) -> None:
        cat: Categorical = Categorical(["A", "B", "C"])
        arr: np.ndarray = np.array(cat).reshape(-1, 1)
        arr = np.broadcast_to(arr, (3, 4))
        result: DataFrame = DataFrame(arr, dtype=cat.dtype)
        expected: DataFrame = DataFrame({0: cat, 1: cat, 2: cat, 3: cat})
        pd.testing.assert_frame_equal(result, expected)

    def test_constructor_categorical(self) -> None:
        df: DataFrame = DataFrame({"A": list("abc")}, dtype="category")
        expected: Series = Series(list("abc"), dtype="category", name="A")
        pd.testing.assert_series_equal(df["A"], expected)
        s: Series = Series(list("abc"), dtype="category")
        result: DataFrame = s.to_frame()
        expected = Series(list("abc"), dtype="category", name=0)
        pd.testing.assert_series_equal(result[0], expected)
        result = s.to_frame(name="foo")
        expected = Series(list("abc"), dtype="category", name="foo")
        pd.testing.assert_series_equal(result["foo"], expected)
        df = DataFrame(list("abc"), dtype="category")
        expected = Series(list("abc"), dtype="category", name=0)
        pd.testing.assert_series_equal(df[0], expected)

    def test_construct_from_1item_list_of_categorical(self) -> None:
        cat: Categorical = Categorical(list("abc"))
        df: DataFrame = DataFrame([cat])
        expected: DataFrame = DataFrame([cat.astype(object)])
        pd.testing.assert_frame_equal(df, expected)

    def test_construct_from_list_of_categoricals(self) -> None:
        df: DataFrame = DataFrame([Categorical(list("abc")), Categorical(list("abd"))])
        expected: DataFrame = DataFrame([["a", "b", "c"], ["a", "b", "d"]])
        pd.testing.assert_frame_equal(df, expected)

    def test_from_nested_listlike_mixed_types(self) -> None:
        df: DataFrame = DataFrame([Categorical(list("abc")), list("def")])
        expected: DataFrame = DataFrame([["a", "b", "c"], ["d", "e", "f"]])
        pd.testing.assert_frame_equal(df, expected)

    def test_construct_from_listlikes_mismatched_lengths(self) -> None:
        df: DataFrame = DataFrame([Categorical(list("abc")), Categorical(list("abdefg"))])
        expected: DataFrame = DataFrame([list("abc"), list("abdefg")])
        pd.testing.assert_frame_equal(df, expected)

    def test_constructor_categorical_series(self) -> None:
        items: List[int] = [1, 2, 3, 1]
        exp: Series = Series(items).astype("category")
        res: Series = Series(items, dtype="category")
        pd.testing.assert_series_equal(res, exp)
        index: DatetimeIndex = pd.date_range("20000101", periods=3)
        expected: Series = Series(pd.Categorical(values=[np.nan, np.nan, np.nan], categories=["a", "b", "c"]))
        expected.index = index
        expected = DataFrame({"x": expected})
        df: DataFrame = DataFrame({"x": Series(["a", "b", "c"], dtype="category")}, index=index)
        pd.testing.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        "dtype", [d for d in pd.api.types.ALL_NUMERIC_DTYPES] +
        pd.api.types.DATETIME64_DTYPES +
        pd.api.types.TIMEDELTA64_DTYPES +
        pd.api.types.BOOL_DTYPES,
    )
    def test_check_dtype_empty_numeric_column(self, dtype: Any) -> None:
        data: DataFrame = DataFrame({"a": [1, 2]}, columns=["b"], dtype=dtype)
        assert data.b.dtype == dtype

    @pytest.mark.parametrize(
        "dtype", pd.api.types.STRING_DTYPES + pd.api.types.BYTES_DTYPES + pd.api.types.OBJECT_DTYPES
    )
    def test_check_dtype_empty_string_column(self, dtype: Any) -> None:
        data: DataFrame = DataFrame({"a": [1, 2]}, columns=["b"], dtype=dtype)
        assert data.b.dtype.name == "object"

    def test_to_frame_with_falsey_names(self) -> None:
        result: Series = Series(name=0, dtype=object).to_frame().dtypes
        expected: Series = Series({0: object})
        pd.testing.assert_series_equal(result, expected)
        result = DataFrame(Series(name=0, dtype=object)).dtypes
        pd.testing.assert_series_equal(result, expected)

    @pytest.mark.arm_slow
    @pytest.mark.parametrize("dtype", [None, "uint8", "category"])
    def test_constructor_range_dtype(self, dtype: Optional[str]) -> None:
        expected: DataFrame = DataFrame({"A": [0, 1, 2, 3, 4]}, dtype=dtype or "int64")
        result: DataFrame = DataFrame(range(5), columns=["A"], dtype=dtype)
        pd.testing.assert_frame_equal(result, expected)
        result = DataFrame({"A": range(5)}, dtype=dtype)
        pd.testing.assert_frame_equal(result, expected)

    def test_frame_from_list_subclass(self) -> None:
        class ListSubclass(list):
            pass
        expected: DataFrame = DataFrame([[1, 2, 3], [4, 5, 6]])
        result: DataFrame = DataFrame(ListSubclass([ListSubclass([1, 2, 3]), ListSubclass([4, 5, 6])]))
        pd.testing.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "extension_arr",
        [
            Categorical(list("aabbc")),
            pd.arrays.SparseArray([1, np.nan, np.nan, np.nan]),
            IntervalArray([Interval(0, 1), Interval(1, 5)]),
            PeriodArray(pd.period_range(start="1/1/2017", end="1/1/2018", freq="M")),
        ],
    )
    def test_constructor_with_extension_array(self, extension_arr: Any) -> None:
        expected: DataFrame = DataFrame(Series(extension_arr))
        result: DataFrame = DataFrame(extension_arr)
        pd.testing.assert_frame_equal(result, expected)

    def test_datetime_date_tuple_columns_from_dict(self) -> None:
        v: date = date.today()
        tup: tuple[date, date] = (v, v)
        result: DataFrame = DataFrame({tup: Series(range(3), index=range(3))}, columns=[tup])
        expected: DataFrame = DataFrame([0, 1, 2], columns=Index(Series([tup])))
        pd.testing.assert_frame_equal(result, expected)

    def test_construct_with_two_categoricalindex_series(self) -> None:
        s1: Series = Series([39, 6, 4], index=CategoricalIndex(["female", "male", "unknown"]))
        s2: Series = Series(
            [2, 152, 2, 242, 150],
            index=CategoricalIndex(["f", "female", "m", "male", "unknown"]),
        )
        result: DataFrame = DataFrame([s1, s2])
        expected: DataFrame = DataFrame(
            np.array([[39, 6, 4, np.nan, np.nan], [152.0, 242.0, 150.0, 2.0, 2.0]]),
            columns=["female", "male", "unknown", "f", "m"],
        )
        pd.testing.assert_frame_equal(result, expected)

    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered in cast:RuntimeWarning"
    )
    def test_constructor_series_nonexact_categoricalindex(self) -> None:
        ser: Series = Series(range(100))
        ser1: Series = pd.cut(ser, 10).value_counts().head(5)
        ser2: Series = pd.cut(ser, 10).value_counts().tail(5)
        result: DataFrame = DataFrame({"1": ser1, "2": ser2})
        index: CategoricalIndex = CategoricalIndex(
            [
                Interval(-0.099, 9.9, closed="right"),
                Interval(9.9, 19.8, closed="right"),
                Interval(19.8, 29.7, closed="right"),
                Interval(29.7, 39.6, closed="right"),
                Interval(39.6, 49.5, closed="right"),
                Interval(49.5, 59.4, closed="right"),
                Interval(59.4, 69.3, closed="right"),
                Interval(69.3, 79.2, closed="right"),
                Interval(79.2, 89.1, closed="right"),
                Interval(89.1, 99, closed="right"),
            ],
            ordered=True,
        )
        expected: DataFrame = DataFrame(
            {"1": [10] * 5 + [np.nan] * 5, "2": [np.nan] * 5 + [10] * 5}, index=index
        )
        pd.testing.assert_frame_equal(expected, result)

    def test_from_M8_structured(self) -> None:
        dates: List[tuple[datetime, datetime]] = [(datetime(2012, 9, 9, 0, 0), datetime(2012, 9, 8, 15, 10))]
        arr: np.ndarray = np.array(dates, dtype=[("Date", "M8[us]"), ("Forecasting", "M8[us]")])
        df: DataFrame = DataFrame(arr)
        assert df["Date"][0] == dates[0][0]
        assert df["Forecasting"][0] == dates[0][1]
        s: Series = Series(arr["Date"])
        assert isinstance(s[0], Timestamp)
        assert s[0] == dates[0][0]

    def test_from_datetime_subclass(self) -> None:
        class DatetimeSubclass(datetime):
            pass
        data: DataFrame = DataFrame({"datetime": [DatetimeSubclass(2020, 1, 1, 1, 1)]})
        assert data.datetime.dtype == "datetime64[us]"

    def test_with_mismatched_index_length_raises(self) -> None:
        dti: DatetimeIndex = pd.date_range("2016-01-01", periods=3, tz="US/Pacific")
        msg: str = "Shape of passed values|Passed arrays should have the same length"
        import pytest
        with pytest.raises(ValueError, match=msg):
            DataFrame(dti, index=range(4))

    def test_frame_ctor_datetime64_column(self) -> None:
        rng: DatetimeIndex = pd.date_range("1/1/2000 00:00:00", "1/1/2000 1:59:50", freq="10s")
        dates: np.ndarray = np.asarray(rng)
        df: DataFrame = DataFrame(
            {"A": np.random.default_rng(2).standard_normal(len(rng)), "B": dates}
        )
        import numpy as np
        assert np.issubdtype(df["B"].dtype, np.dtype("M8[ns]"))

    def test_dataframe_constructor_infer_multiindex(self) -> None:
        index_lists: List[List[str]] = [["a", "a", "b", "b"], ["x", "y", "x", "y"]]
        multi: DataFrame = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            index=[np.array(x) for x in index_lists],
        )
        from pandas.core.dtypes.inference import is_list_like
        assert isinstance(multi.index, MultiIndex)
        assert not isinstance(multi.columns, MultiIndex)
        multi = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)), columns=index_lists
        )
        assert isinstance(multi.columns, MultiIndex)

    @pytest.mark.parametrize(
        "input_vals",
        [
            ([1, 2]),
            (["1", "2"]),
            (list(pd.date_range("1/1/2011", periods=2, freq="h"))),
            (list(pd.date_range("1/1/2011", periods=2, freq="h", tz="US/Eastern"))),
            ([Interval(left=0, right=5)]),
        ],
    )
    def test_constructor_list_str(self, input_vals: Any, string_dtype: Any) -> None:
        result: DataFrame = DataFrame({"A": input_vals}, dtype=string_dtype)
        expected: DataFrame = DataFrame({"A": input_vals}).astype({"A": string_dtype})
        pd.testing.assert_frame_equal(result, expected)

    def test_constructor_list_str_na(self, string_dtype: Any) -> None:
        result: DataFrame = DataFrame({"A": [1.0, 2.0, None]}, dtype=string_dtype)
        expected: DataFrame = DataFrame({"A": ["1.0", "2.0", None]}, dtype=object)
        pd.testing.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("copy", [False, True])
    def test_dict_nocopy(self, copy: bool, any_numeric_ea_dtype: Any, any_numpy_dtype: Any) -> None:
        a: np.ndarray = np.array([1, 2], dtype=any_numpy_dtype)
        b: np.ndarray = np.array([3, 4], dtype=any_numpy_dtype)
        if b.dtype.kind in ["S", "U"]:
            import pytest
            pytest.skip(f"{b.dtype} get cast, making the checks below more cumbersome")
        c: Any = pd.array([1, 2], dtype=any_numeric_ea_dtype)
        c_orig: Any = c.copy()
        df: DataFrame = DataFrame({"a": a, "b": b, "c": c}, copy=copy)
        def get_base(obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                return obj.base
            elif hasattr(obj, "_ndarray"):
                return obj._ndarray.base
            else:
                raise TypeError
        def check_views(c_only: bool = False) -> None:
            assert sum(x.values is c for x in df._mgr.blocks) == 1
            if c_only:
                return
            assert (
                sum(
                    get_base(x.values) is a
                    for x in df._mgr.blocks
                    if hasattr(x.values.dtype, "kind")
                )
                == 1
            )
            assert (
                sum(
                    get_base(x.values) is b
                    for x in df._mgr.blocks
                    if hasattr(x.values.dtype, "kind")
                )
                == 1
            )
        if not copy:
            check_views()
        should_raise: bool = not lib.is_np_dtype(df.dtypes.iloc[0], "fciuO")
        import pytest
        if should_raise:
            with pytest.raises(TypeError, match="Invalid value"):
                df.iloc[0, 0] = 0
                df.iloc[0, 1] = 0
            return
        else:
            df.iloc[0, 0] = 0
            df.iloc[0, 1] = 0
        if not copy:
            check_views(True)
        df.iloc[:, 2] = pd.array([45, 46], dtype=c.dtype)
        assert df.dtypes.iloc[2] == c.dtype
        if copy:
            if a.dtype.kind == "M":
                assert a[0] == a.dtype.type(1, "ns")
                assert b[0] == b.dtype.type(3, "ns")
            else:
                assert a[0] == a.dtype.type(1)
                assert b[0] == b.dtype.type(3)
            assert c[0] == c_orig[0]
    def test_construct_from_dict_ea_series(self) -> None:
        ser: Series = Series([1, 2, 3], dtype="Int64")
        df: DataFrame = DataFrame({"a": ser})
        import numpy as np
        assert not np.shares_memory(ser.values._data, df["a"].values._data)
    def test_from_series_with_name_with_columns(self) -> None:
        result: DataFrame = DataFrame(Series(1, name="foo"), columns=["bar"])
        expected: DataFrame = DataFrame(columns=["bar"])
        pd.testing.assert_frame_equal(result, expected)
    def test_nested_list_columns(self) -> None:
        result: DataFrame = DataFrame(
            [[1, 2, 3], [4, 5, 6]], columns=[["A", "A", "A"], ["a", "b", "c"]]
        )
        expected: DataFrame = DataFrame(
            [[1, 2, 3], [4, 5, 6]],
            columns=MultiIndex.from_tuples([("A", "a"), ("A", "b"), ("A", "c")]),
        )
        pd.testing.assert_frame_equal(result, expected)
    def test_from_2d_object_array_of_periods_or_intervals(self) -> None:
        pi: pd.PeriodIndex = pd.period_range("2016-04-05", periods=3)
        data: np.ndarray = pi._data.astype(object).reshape(1, -1)
        df: DataFrame = DataFrame(data)
        assert df.shape == (1, 3)
        assert (df.dtypes == pi.dtype).all()
        assert (df == pi).all().all()
        ii: IntervalIndex = pd.IntervalIndex.from_breaks([3, 4, 5, 6])
        data2: np.ndarray = ii._data.astype(object).reshape(1, -1)
        df2: DataFrame = DataFrame(data2)
        assert df2.shape == (1, 3)
        assert (df2.dtypes == ii.dtype).all()
        assert (df2 == ii).all().all()
        data3: np.ndarray = np.r_[data, data2, data, data2].T
        df3: DataFrame = DataFrame(data3)
        expected: DataFrame = DataFrame({0: pi, 1: ii, 2: pi, 3: ii})
        pd.testing.assert_frame_equal(df3, expected)
    @pytest.mark.parametrize(
        "col_a, col_b",
        [
            ([[1], [2]], np.array([[1], [2]])),
            (np.array([[1], [2]]), [[1], [2]]),
            (np.array([[1], [2]]), np.array([[1], [2]])),
        ],
    )
    def test_error_from_2darray(self, col_a: Any, col_b: Any) -> None:
        msg: str = "Per-column arrays must each be 1-dimensional"
        import pytest
        with pytest.raises(ValueError, match=msg):
            DataFrame({"a": col_a, "b": col_b})
    def test_from_dict_with_missing_copy_false(self) -> None:
        df: DataFrame = DataFrame(index=[1, 2, 3], columns=["a", "b", "c"], copy=False)
        import numpy as np
        assert not np.shares_memory(df["a"]._values, df["b"]._values)
        df.iloc[0, 0] = 0
        expected: DataFrame = DataFrame(
            {
                "a": [0, np.nan, np.nan],
                "b": [np.nan, np.nan, np.nan],
                "c": [np.nan, np.nan, np.nan],
            },
            index=[1, 2, 3],
            dtype=object,
        )
        pd.testing.assert_frame_equal(df, expected)
    def test_construction_empty_array_multi_column_raises(self) -> None:
        msg: str = r"Shape of passed values is \(0, 1\), indices imply \(0, 2\)"
        import pytest
        with pytest.raises(ValueError, match=msg):
            DataFrame(data=np.array([]), columns=["a", "b"])
    def test_construct_with_strings_and_none(self) -> None:
        df: DataFrame = DataFrame(["1", "2", None], columns=["a"], dtype="str")
        expected: DataFrame = DataFrame({"a": ["1", "2", None]}, dtype="str")
        pd.testing.assert_frame_equal(df, expected)
    def test_frame_string_inference(self) -> None:
        dtype = pd.StringDtype(na_value=np.nan)
        expected: DataFrame = DataFrame(
            {"a": ["a", "b"]}, dtype=dtype, columns=Index(["a"], dtype=dtype)
        )
        with pd.option_context("future.infer_string", True):
            df: DataFrame = DataFrame({"a": ["a", "b"]})
        pd.testing.assert_frame_equal(df, expected)
        expected = DataFrame(
            {"a": ["a", "b"]},
            dtype=dtype,
            columns=Index(["a"], dtype=dtype),
            index=Index(["x", "y"], dtype=dtype),
        )
        with pd.option_context("future.infer_string", True):
            df = DataFrame({"a": ["a", "b"]}, index=["x", "y"])
        pd.testing.assert_frame_equal(df, expected)
        expected = DataFrame(
            {"a": ["a", 1]}, dtype="object", columns=Index(["a"], dtype=dtype)
        )
        with pd.option_context("future.infer_string", True):
            df = DataFrame({"a": ["a", 1]})
        pd.testing.assert_frame_equal(df, expected)
        expected = DataFrame(
            {"a": ["a", "b"]}, dtype="object", columns=Index(["a"], dtype=dtype)
        )
        with pd.option_context("future.infer_string", True):
            df = DataFrame({"a": ["a", "b"]}, dtype="object")
        pd.testing.assert_frame_equal(df, expected)
    def test_frame_string_inference_array_string_dtype(self) -> None:
        dtype = pd.StringDtype(na_value=np.nan)
        expected: DataFrame = DataFrame(
            {"a": ["a", "b"]}, dtype=dtype, columns=Index(["a"], dtype=dtype)
        )
        with pd.option_context("future.infer_string", True):
            df: DataFrame = DataFrame({"a": np.array(["a", "b"])})
        pd.testing.assert_frame_equal(df, expected)
        expected = DataFrame({0: ["a", "b"], 1: ["c", "d"]}, dtype=dtype)
        with pd.option_context("future.infer_string", True):
            df = DataFrame(np.array([["a", "c"], ["b", "d"]]))
        pd.testing.assert_frame_equal(df, expected)
        expected = DataFrame(
            {"a": ["a", "b"], "b": ["c", "d"]},
            dtype=dtype,
            columns=Index(["a", "b"], dtype=dtype),
        )
        with pd.option_context("future.infer_string", True):
            df = DataFrame(np.array([["a", "c"], ["b", "d"]]), columns=["a", "b"])
        pd.testing.assert_frame_equal(df, expected)
    def test_frame_string_inference_block_dim(self) -> None:
        with pd.option_context("future.infer_string", True):
            df: DataFrame = DataFrame(np.array([["hello", "goodbye"], ["hello", "Hello"]]))
        assert df._mgr.blocks[0].ndim == 2
    @pytest.mark.parametrize("klass", [Series, Index])
    def test_inference_on_pandas_objects(self, klass: Callable[..., Any]) -> None:
        obj = klass([Timestamp("2019-12-31")], dtype=object)
        result: DataFrame = DataFrame(obj, columns=["a"])
        assert result.dtypes.iloc[0] == np.object_
        result = DataFrame({"a": obj})
        assert result.dtypes.iloc[0] == np.object_
    def test_dict_keys_returns_rangeindex(self) -> None:
        result: Index = DataFrame({0: [1], 1: [2]}).columns
        expected: RangeIndex = RangeIndex(2)
        pd.testing.assert_index_equal(result, expected, exact=True)
    @pytest.mark.parametrize(
        "cons", [Series, Index, DatetimeIndex, DataFrame, pd.array, pd.to_datetime]
    )
    def test_construction_datetime_resolution_inference(self, cons: Callable[..., Any]) -> None:
        ts: Timestamp = Timestamp(2999, 1, 1)
        ts2: Timestamp = ts.tz_localize("US/Pacific")
        obj = cons([ts])
        res_dtype: Any = pd.api.types.infer_dtype(obj)
        assert tm.get_dtype(obj) == "M8[us]", res_dtype
        obj2 = cons([ts2])
        res_dtype2: Any = tm.get_dtype(obj2)
        assert res_dtype2 == "M8[us, US/Pacific]", res_dtype2
    def test_construction_nan_value_timedelta64_dtype(self) -> None:
        result: DataFrame = DataFrame([None, 1], dtype="timedelta64[ns]")
        expected: DataFrame = DataFrame(
            ["NaT", "0 days 00:00:00.000000001"], dtype="timedelta64[ns]"
        )
        pd.testing.assert_frame_equal(result, expected)

class TestDataFrameConstructorIndexInference:
    def test_frame_from_dict_of_series_overlapping_monthly_period_indexes(self) -> None:
        rng1: pd.PeriodIndex = pd.period_range("1/1/1999", "1/1/2012", freq="M")
        s1: Series = Series(np.random.default_rng(2).standard_normal(len(rng1)), rng1)
        rng2: pd.PeriodIndex = pd.period_range("1/1/1980", "12/1/2001", freq="M")
        s2: Series = Series(np.random.default_rng(2).standard_normal(len(rng2)), rng2)
        df: DataFrame = DataFrame({"s1": s1, "s2": s2})
        exp: pd.PeriodIndex = pd.period_range("1/1/1980", "1/1/2012", freq="M")
        pd.testing.assert_index_equal(df.index, exp)
    def test_frame_from_dict_with_mixed_tzaware_indexes(self) -> None:
        dti: DatetimeIndex = pd.date_range("2016-01-01", periods=3)
        ser1: Series = Series(range(3), index=dti)
        ser2: Series = Series(range(3), index=dti.tz_localize("UTC"))
        ser3: Series = Series(range(3), index=dti.tz_localize("US/Central"))
        ser4: Series = Series(range(3))
        df1: DataFrame = DataFrame({"A": ser2, "B": ser3, "C": ser4})
        exp_index: Index = Index(list(ser2.index) + list(ser3.index) + list(ser4.index), dtype=object)
        pd.testing.assert_index_equal(df1.index, exp_index)
        df2: DataFrame = DataFrame({"A": ser2, "C": ser4, "B": ser3})
        exp_index3: Index = Index(list(ser2.index) + list(ser4.index) + list(ser3.index), dtype=object)
        pd.testing.assert_index_equal(df2.index, exp_index3)
        df3: DataFrame = DataFrame({"B": ser3, "A": ser2, "C": ser4})
        exp_index3 = Index(list(ser3.index) + list(ser2.index) + list(ser4.index), dtype=object)
        pd.testing.assert_index_equal(df3.index, exp_index3)
        df4: DataFrame = DataFrame({"C": ser4, "B": ser3, "A": ser2})
        exp_index4: Index = Index(list(ser4.index) + list(ser3.index) + list(ser2.index), dtype=object)
        pd.testing.assert_index_equal(df4.index, exp_index4)
        import pytest
        msg: str = "Cannot join tz-naive with tz-aware DatetimeIndex"
        with pytest.raises(TypeError, match=msg):
            DataFrame({"A": ser2, "B": ser3, "C": ser4, "D": ser1})
        with pytest.raises(TypeError, match=msg):
            DataFrame({"A": ser2, "B": ser3, "D": ser1})
        with pytest.raises(TypeError, match=msg):
            DataFrame({"D": ser1, "A": ser2, "B": ser3})
    @pytest.mark.parametrize(
        "key_val, col_vals, col_type",
        [
            ["3", ["3", "4"], "utf8"],
            [3, [3, 4], "int8"],
        ],
    )
    def test_dict_data_arrow_column_expansion(self, key_val: Any, col_vals: List[Any], col_type: str) -> None:
        pa = pytest.importorskip("pyarrow")
        cols = pd.arrays.ArrowExtensionArray(
            pa.array(col_vals, type=pa.dictionary(pa.int8(), getattr(pa, col_type)()))
        )
        result: DataFrame = DataFrame({key_val: [1, 2]}, columns=cols)
        expected: DataFrame = DataFrame([[1, np.nan], [2, np.nan]], columns=cols)
        expected.isetitem(1, expected.iloc[:, 1].astype(object))
        pd.testing.assert_frame_equal(result, expected)

class TestDataFrameConstructorWithDtypeCoercion:
    def test_floating_values_integer_dtype(self) -> None:
        arr: np.ndarray = np.random.default_rng(2).standard_normal((10, 5))
        import pytest
        msg: str = "Trying to coerce float values to integers"
        with pytest.raises(ValueError, match=msg):
            DataFrame(arr, dtype="i8")
        df: DataFrame = DataFrame(arr.round(), dtype="i8")
        assert (df.dtypes == "i8").all()
        arr[0, 0] = np.nan
        msg = r"Cannot convert non-finite values \(NA or inf\) to integer"
        with pytest.raises(IntCastingNaNError, match=msg):
            DataFrame(arr, dtype="i8")
        with pytest.raises(IntCastingNaNError, match=msg):
            Series(arr[0], dtype="i8")
        with pytest.raises(IntCastingNaNError, match=msg):
            DataFrame(arr).astype("i8")
        with pytest.raises(IntCastingNaNError, match=msg):
            Series(arr[0]).astype("i8")

class TestDataFrameConstructorWithDatetimeTZ:
    @pytest.mark.parametrize("tz", ["US/Eastern", "dateutil/US/Eastern"])
    def test_construction_preserves_tzaware_dtypes(self, tz: str) -> None:
        dr: DatetimeIndex = pd.date_range("2011/1/1", "2012/1/1", freq="W-FRI")
        dr_tz: DatetimeIndex = dr.tz_localize(tz)
        df: DataFrame = DataFrame({"A": "foo", "B": dr_tz}, index=dr)
        tz_expected = DatetimeTZDtype("ns", dr_tz.tzinfo)
        assert df["B"].dtype == tz_expected
        datetimes_naive: List[Any] = [ts.to_pydatetime() for ts in dr]
        datetimes_with_tz: List[Any] = [ts.to_pydatetime() for ts in dr_tz]
        df = DataFrame({"dr": dr})
        df["dr_tz"] = dr_tz
        df["datetimes_naive"] = datetimes_naive
        df["datetimes_with_tz"] = datetimes_with_tz
        result: Series = df.dtypes
        expected = Series(
            [
                np.dtype("datetime64[ns]"),
                DatetimeTZDtype(tz=tz),
                np.dtype("datetime64[us]"),
                DatetimeTZDtype(tz=tz, unit="us"),
            ],
            index=["dr", "dr_tz", "datetimes_naive", "datetimes_with_tz"],
        )
        pd.testing.assert_series_equal(result, expected)
    @pytest.mark.parametrize("pydt", [True, False])
    def test_constructor_data_aware_dtype_naive(self, tz_aware_fixture: str, pydt: bool) -> None:
        tz: str = tz_aware_fixture
        ts: Timestamp = Timestamp("2019", tz=tz)
        if pydt:
            ts = ts.to_pydatetime()
        msg: str = (
            "Cannot convert timezone-aware data to timezone-naive dtype. "
            r"Use pd.Series\(values\).dt.tz_localize\(None\) instead."
        )
        import pytest
        with pytest.raises(ValueError, match=msg):
            DataFrame({0: [ts]}, dtype="datetime64[ns]")
        msg2: str = "Cannot unbox tzaware Timestamp to tznaive dtype"
        with pytest.raises(TypeError, match=msg2):
            DataFrame({0: ts}, index=[0], dtype="datetime64[ns]")
        with pytest.raises(ValueError, match=msg):
            DataFrame([ts], dtype="datetime64[ns]")
        with pytest.raises(ValueError, match=msg):
            DataFrame(np.array([ts], dtype=object), dtype="datetime64[ns]")
        with pytest.raises(TypeError, match=msg2):
            DataFrame(ts, index=[0], columns=[0], dtype="datetime64[ns]")
        with pytest.raises(ValueError, match=msg):
            DataFrame([Series([ts])], dtype="datetime64[ns]")
        with pytest.raises(ValueError, match=msg):
            DataFrame([[ts]], columns=[0], dtype="datetime64[ns]")
    def test_from_dict(self) -> None:
        idx: Index = Index(pd.date_range("20130101", periods=3, tz="US/Eastern"), name="foo")
        dr: DatetimeIndex = pd.date_range("20130110", periods=3)
        df: DataFrame = DataFrame({"A": idx, "B": dr})
        assert df["A"].dtype, "M8[ns, US/Eastern"
        assert df["A"].name == "A"
        pd.testing.assert_series_equal(df["A"], Series(idx, name="A"))
        pd.testing.assert_series_equal(df["B"], Series(dr, name="B"))
    def test_from_index(self) -> None:
        idx2: DatetimeIndex = pd.date_range("20130101", periods=3, tz="US/Eastern", name="foo")
        df2: DataFrame = DataFrame(idx2)
        pd.testing.assert_series_equal(df2["foo"], Series(idx2, name="foo"))
        df2 = DataFrame(Series(idx2))
        pd.testing.assert_series_equal(df2["foo"], Series(idx2, name="foo"))
        idx2 = pd.date_range("20130101", periods=3, tz="US/Eastern")
        df2 = DataFrame(idx2)
        pd.testing.assert_series_equal(df2[0], Series(idx2, name=0))
        df2 = DataFrame(Series(idx2))
        pd.testing.assert_series_equal(df2[0], Series(idx2, name=0))
    def test_frame_dict_constructor_datetime64_1680(self) -> None:
        dr: DatetimeIndex = pd.date_range("1/1/2012", periods=10)
        s: Series = Series(dr, index=dr)
        DataFrame({"a": "foo", "b": s}, index=dr)
        DataFrame({"a": "foo", "b": s.values}, index=dr)
    def test_frame_datetime64_mixed_index_ctor_1681(self) -> None:
        dr: DatetimeIndex = pd.date_range("2011/1/1", "2012/1/1", freq="W-FRI")
        ts: Series = Series(dr)
        d: DataFrame = DataFrame({"A": "foo", "B": ts}, index=dr)
        assert d["B"].isna().all()
    def test_frame_timeseries_column(self) -> None:
        dr: DatetimeIndex = pd.date_range(
            start="20130101T10:00:00", periods=3, freq="min", tz="US/Eastern"
        )
        result: DataFrame = DataFrame(dr, columns=["timestamps"])
        expected: DataFrame = DataFrame(
            {
                "timestamps": [
                    Timestamp("20130101T10:00:00", tz="US/Eastern"),
                    Timestamp("20130101T10:01:00", tz="US/Eastern"),
                    Timestamp("20130101T10:02:00", tz="US/Eastern"),
                ]
            },
            dtype="M8[ns, US/Eastern]",
        )
        pd.testing.assert_frame_equal(result, expected)
    def test_nested_dict_construction(self) -> None:
        columns: List[str] = ["Nevada", "Ohio"]
        pop: Dict[str, Dict[int, float]] = {
            "Nevada": {2001: 2.4, 2002: 2.9},
            "Ohio": {2000: 1.5, 2001: 1.7, 2002: 3.6},
        }
        result: DataFrame = DataFrame(pop, index=[2001, 2002, 2003], columns=columns)
        expected: DataFrame = DataFrame(
            [(2.4, 1.7), (2.9, 3.6), (np.nan, np.nan)],
            columns=columns,
            index=Index([2001, 2002, 2003]),
        )
        pd.testing.assert_frame_equal(result, expected)
    def test_from_tzaware_object_array(self) -> None:
        dti: DatetimeIndex = pd.date_range("2016-04-05 04:30", periods=3, tz="UTC")
        data: np.ndarray = dti._data.astype(object).reshape(1, -1)
        df: DataFrame = DataFrame(data)
        assert df.shape == (1, 3)
        assert (df.dtypes == dti.dtype).all()
        assert (df == dti).all().all()
    def test_from_tzaware_mixed_object_array(self) -> None:
        arr: np.ndarray = np.array(
            [
                [
                    Timestamp("2013-01-01 00:00:00"),
                    Timestamp("2013-01-02 00:00:00"),
                    Timestamp("2013-01-03 00:00:00"),
                ],
                [
                    Timestamp("2013-01-01 00:00:00-0500", tz="US/Eastern"),
                    pd.NaT,
                    Timestamp("2013-01-03 00:00:00-0500", tz="US/Eastern"),
                ],
                [
                    Timestamp("2013-01-01 00:00:00+0100", tz="CET"),
                    pd.NaT,
                    Timestamp("2013-01-03 00:00:00+0100", tz="CET"),
                ],
            ],
            dtype=object,
        ).T
        res: DataFrame = DataFrame(arr, columns=["A", "B", "C"])
        expected_dtypes: List[str] = [
            "datetime64[s]",
            "datetime64[s, US/Eastern]",
            "datetime64[s, CET]",
        ]
        assert (res.dtypes == expected_dtypes).all()
    def test_from_2d_ndarray_with_dtype(self) -> None:
        array_dim2: np.ndarray = np.arange(10).reshape((5, 2))
        df: DataFrame = DataFrame(array_dim2, dtype="datetime64[ns, UTC]")
        expected: DataFrame = DataFrame(array_dim2).astype("datetime64[ns, UTC]")
        pd.testing.assert_frame_equal(df, expected)
    @pytest.mark.parametrize("typ", [set, frozenset])
    def test_construction_from_set_raises(self, typ: type) -> None:
        values: Any = typ({1, 2, 3})
        msg: str = f"'{typ.__name__}' type is unordered"
        import pytest
        with pytest.raises(TypeError, match=msg):
            DataFrame({"a": values})
        with pytest.raises(TypeError, match=msg):
            Series(values)
    def test_construction_from_ndarray_datetimelike(self) -> None:
        arr: np.ndarray = np.arange(0, 12, dtype="datetime64[ns]").reshape(4, 3)
        df: DataFrame = DataFrame(arr)
        from pandas.arrays import DatetimeArray
        assert all(isinstance(block.values, DatetimeArray) for block in df._mgr.blocks)
    def test_construction_from_ndarray_with_eadtype_mismatched_columns(self) -> None:
        arr: np.ndarray = np.random.default_rng(2).standard_normal((10, 2))
        dtype = pd.array([2.0]).dtype
        msg: str = r"len\(arrays\) must match len\(columns\)"
        import pytest
        with pytest.raises(ValueError, match=msg):
            DataFrame(arr, columns=["foo"], dtype=dtype)
        arr2: Any = pd.array([2.0, 3.0, 4.0])
        with pytest.raises(ValueError, match=msg):
            DataFrame(arr2, columns=["foo", "bar"])
    def test_columns_indexes_raise_on_sets(self) -> None:
        data: List[List[int]] = [[1, 2, 3], [4, 5, 6]]
        import pytest
        with pytest.raises(ValueError, match="index cannot be a set"):
            DataFrame(data, index={"a", "b"})
        with pytest.raises(ValueError, match="columns cannot be a set"):
            DataFrame(data, columns={"a", "b", "c"})
    def test_from_dict_with_columns_na_scalar(self) -> None:
        result: DataFrame = DataFrame({"a": pd.NaT}, columns=["a"], index=range(2))
        expected: DataFrame = DataFrame({"a": Series([pd.NaT, pd.NaT])})
        pd.testing.assert_frame_equal(result, expected)
    @pytest.mark.skipif(
        not np.__version__ >= "2", reason="StringDType only available in numpy 2 and above"
    )
    @pytest.mark.parametrize(
        "data",
        [
            {"a": ["a", "b", "c"], "b": [1.0, 2.0, 3.0], "c": ["d", "e", "f"]},
        ],
    )
    def test_np_string_array_object_cast(self, data: Dict[str, Any]) -> None:
        from numpy.dtypes import StringDType
        data["a"] = np.array(data["a"], dtype=StringDType())
        res: DataFrame = DataFrame(data)
        assert res["a"].dtype == np.object_
        assert (res["a"] == data["a"]).all()

class TestFromScalar:
    @pytest.fixture(params=[list, dict, None])
    def box(self, request: Any) -> Optional[Union[list, dict]]:
        return request.param
    @pytest.fixture
    def constructor(self, frame_or_series: Callable[..., Any], box: Optional[Union[list, dict]]) -> Callable[..., Any]:
        extra: Dict[str, Any] = {"index": range(2)}
        if frame_or_series is DataFrame:
            extra["columns"] = ["A"]
        if box is None:
            return functools.partial(frame_or_series, **extra)
        elif box is dict:
            if frame_or_series is Series:
                return lambda x, **kwargs: frame_or_series({0: x, 1: x}, **extra, **kwargs)
            else:
                return lambda x, **kwargs: frame_or_series({"A": [x, x]}, **extra, **kwargs)
        elif frame_or_series is Series:
            return lambda x, **kwargs: frame_or_series([x, x], **extra, **kwargs)
        else:
            return lambda x, **kwargs: frame_or_series({"A": [x, x]}, **extra, **kwargs)
    @pytest.mark.parametrize("dtype", ["M8[ns]", "m8[ns]"])
    def test_from_nat_scalar(self, dtype: str, constructor: Callable[..., Any]) -> None:
        obj: Any = constructor(pd.NaT, dtype=dtype)
        assert (obj.dtypes == dtype).all()
        assert (obj.isna()).all().all()
    def test_from_timedelta_scalar_preserves_nanos(self, constructor: Callable[..., Any]) -> None:
        td: Timedelta = Timedelta(1)
        obj: Any = constructor(td, dtype="m8[ns]")
        from pandas._libs.tslibs import Timedelta as LibTimedelta
        assert get1(obj) == td
    def test_from_timestamp_scalar_preserves_nanos(self, constructor: Callable[..., Any], fixed_now_ts: Timestamp) -> None:
        ts: Timestamp = fixed_now_ts + Timedelta(1)
        obj: Any = constructor(ts, dtype="M8[ns]")
        assert get1(obj) == ts
    def test_from_timedelta64_scalar_object(self, constructor: Callable[..., Any]) -> None:
        td: Timedelta = Timedelta(1)
        td64: Any = td.to_timedelta64()
        obj: Any = constructor(td64, dtype=object)
        from numpy import timedelta64 as np_timedelta64
        assert isinstance(get1(obj), np_timedelta64)
    @pytest.mark.parametrize("cls", [np.datetime64, np.timedelta64])
    def test_from_scalar_datetimelike_mismatched(self, constructor: Callable[..., Any], cls: Any) -> None:
        scalar: Any = cls("NaT", "ns")
        dtype: str = {"M8[ns]": "m8[ns]", "m8[ns]": "M8[ns]"}[cls("NaT", "ns").dtype.str]
        if cls is np.datetime64:
            msg1: str = "Invalid type for timedelta scalar: <class 'numpy.datetime64'>"
        else:
            msg1 = "<class 'numpy.timedelta64'> is not convertible to datetime"
        msg: str = "|".join(["Cannot cast", msg1])
        import pytest
        with pytest.raises(TypeError, match=msg):
            constructor(scalar, dtype=dtype)
        scalar = cls(4, "ns")
        with pytest.raises(TypeError, match=msg):
            constructor(scalar, dtype=dtype)
    @pytest.mark.parametrize("cls", [datetime, np.datetime64])
    def test_from_out_of_bounds_ns_datetime(self, constructor: Callable[..., Any], cls: Any, request: Any, box: Optional[Any], frame_or_series: Any) -> None:
        scalar: datetime = datetime(9999, 1, 1)
        exp_dtype: str = "M8[us]"
        if cls is np.datetime64:
            scalar = np.datetime64(scalar, "D")
            exp_dtype = "M8[s]"
        result: Any = constructor(scalar)
        item: Any = get1(result)
        dtype: Any = tm.get_dtype(result)
        assert type(item) is Timestamp
        assert item.asm8.dtype == exp_dtype
        assert dtype == exp_dtype
    @pytest.mark.skip_ubsan
    def test_out_of_s_bounds_datetime64(self, constructor: Callable[..., Any]) -> None:
        scalar: Any = np.datetime64(np.iinfo(np.int64).max, "D")
        result: Any = constructor(scalar)
        item: Any = get1(result)
        assert type(item) is np.datetime64
        dtype: Any = tm.get_dtype(result)
        assert dtype == object
    @pytest.mark.parametrize("cls", [timedelta, np.timedelta64])
    def test_from_out_of_bounds_ns_timedelta(self, constructor: Callable[..., Any], cls: Any, request: Any, box: Optional[Any], frame_or_series: Any) -> None:
        if box is list or (frame_or_series is Series and box is dict):
            mark = pytest.mark.xfail(
                reason="TimedeltaArray constructor has been updated to cast td64 "
                "to non-nano, but TimedeltaArray._from_sequence has not",
                strict=True,
            )
            request.applymarker(mark)
        scalar: timedelta = datetime(9999, 1, 1) - datetime(1970, 1, 1)
        exp_dtype: str = "m8[us]"
        if cls is np.timedelta64:
            scalar = np.timedelta64(scalar, "D")
            exp_dtype = "m8[s]"
        result: Any = constructor(scalar)
        item: Any = get1(result)
        dtype: Any = tm.get_dtype(result)
        assert type(item) is Timedelta
        assert item.asm8.dtype == exp_dtype
        assert dtype == exp_dtype
    @pytest.mark.skip_ubsan
    @pytest.mark.parametrize("cls", [np.datetime64, np.timedelta64])
    def test_out_of_s_bounds_timedelta64(self, constructor: Callable[..., Any], cls: Any) -> None:
        scalar: Any = cls(np.iinfo(np.int64).max, "D")
        result: Any = constructor(scalar)
        item: Any = get1(result)
        assert type(item) is cls
        dtype: Any = tm.get_dtype(result)
        assert dtype == object
    def test_tzaware_data_tznaive_dtype(self, constructor: Callable[..., Any], box: Optional[Any], frame_or_series: Any) -> None:
        tz: str = "US/Eastern"
        ts: Timestamp = Timestamp("2019", tz=tz)
        import pytest
        if box is None or (frame_or_series is DataFrame and box is dict):
            msg: str = "Cannot unbox tzaware Timestamp to tznaive dtype"
            err: type = TypeError
        else:
            msg = (
                "Cannot convert timezone-aware data to timezone-naive dtype. "
                r"Use pd.Series\(values\).dt.tz_localize\(None\) instead."
            )
            err = ValueError
        with pytest.raises(err, match=msg):
            constructor(ts, dtype="M8[ns]")

class TestAllowNonNano:
    @pytest.fixture(params=[True, False])
    def as_td(self, request: Any) -> bool:
        return request.param
    @pytest.fixture
    def arr(self, as_td: bool) -> Union[TimedeltaArray, DatetimeArray]:
        values: np.ndarray = np.arange(5).astype(np.int64).view("M8[s]")
        if as_td:
            values = values - values[0]
            return TimedeltaArray._simple_new(values, dtype=values.dtype)
        else:
            return DatetimeArray._simple_new(values, dtype=values.dtype)
    def test_index_allow_non_nano(self, arr: Union[TimedeltaArray, DatetimeArray]) -> None:
        idx: Index = Index(arr)
        assert idx.dtype == arr.dtype
    def test_dti_tdi_allow_non_nano(self, arr: Union[TimedeltaArray, DatetimeArray], as_td: bool) -> None:
        if as_td:
            idx = pd.TimedeltaIndex(arr)
        else:
            idx = pd.DatetimeIndex(arr)
        assert idx.dtype == arr.dtype
    def test_series_allow_non_nano(self, arr: Union[TimedeltaArray, DatetimeArray]) -> None:
        ser: Series = Series(arr)
        assert ser.dtype == arr.dtype
    def test_frame_allow_non_nano(self, arr: Union[TimedeltaArray, DatetimeArray]) -> None:
        df: DataFrame = DataFrame(arr)
        assert df.dtypes[0] == arr.dtype
    def test_frame_from_dict_allow_non_nano(self, arr: Union[TimedeltaArray, DatetimeArray]) -> None:
        df: DataFrame = DataFrame({0: arr})
        assert df.dtypes[0] == arr.dtype

def get1(obj: Union[Series, DataFrame]) -> Any:
    if isinstance(obj, Series):
        return obj.iloc[0]
    else:
        return obj.iloc[0, 0]