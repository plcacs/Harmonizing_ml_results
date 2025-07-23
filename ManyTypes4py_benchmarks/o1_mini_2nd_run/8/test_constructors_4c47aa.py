from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_unsigned_integer_dtype
from pandas.core.dtypes.dtypes import IntervalDtype
from pandas import (
    Categorical,
    CategoricalDtype,
    CategoricalIndex,
    Index,
    Interval,
    IntervalIndex,
    date_range,
    notna,
    period_range,
    timedelta_range,
)
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
import pandas.core.common as com


class ConstructorTests:
    """
    Common tests for all variations of IntervalIndex construction. Input data
    to be supplied in breaks format, then converted by the subclass method
    get_kwargs_from_breaks to the expected format.
    """

    @pytest.mark.parametrize(
        "breaks_and_expected_subtype",
        [
            ([3, 14, 15, 92, 653], np.int64),
            (np.arange(10, dtype="int64"), np.int64),
            (Index(np.arange(-10, 11, dtype=np.int64)), np.int64),
            (Index(np.arange(10, 31, dtype=np.uint64)), np.uint64),
            (Index(np.arange(20, 30, 0.5), dtype=np.float64), np.float64),
            (date_range("20180101", periods=10), "M8[ns]"),
            (
                date_range("20180101", periods=10, tz="US/Eastern"),
                "datetime64[ns, US/Eastern]",
            ),
            (timedelta_range("1 day", periods=10), "m8[ns]"),
        ],
    )
    @pytest.mark.parametrize("name", [None, "foo"])
    def test_constructor(
        self,
        constructor: Any,
        breaks_and_expected_subtype: Tuple[Union[List[Any], np.ndarray, Index], Union[type, str]],
        closed: str,
        name: Optional[str],
    ) -> None:
        breaks, expected_subtype = breaks_and_expected_subtype
        result_kwargs: Dict[str, Any] = self.get_kwargs_from_breaks(breaks, closed)
        result: IntervalIndex = constructor(closed=closed, name=name, **result_kwargs)
        assert result.closed == closed
        assert result.name == name
        assert result.dtype.subtype == expected_subtype
        tm.assert_index_equal(result.left, Index(breaks[:-1], dtype=expected_subtype))
        tm.assert_index_equal(result.right, Index(breaks[1:], dtype=expected_subtype))

    @pytest.mark.parametrize(
        "breaks, subtype",
        [
            (Index([0, 1, 2, 3, 4], dtype=np.int64), "float64"),
            (Index([0, 1, 2, 3, 4], dtype=np.int64), "datetime64[ns]"),
            (Index([0, 1, 2, 3, 4], dtype=np.int64), "timedelta64[ns]"),
            (Index([0, 1, 2, 3, 4], dtype=np.float64), "int64"),
            (date_range("2017-01-01", periods=5), "int64"),
            (timedelta_range("1 day", periods=5), "int64"),
        ],
    )
    def test_constructor_dtype(
        self,
        constructor: Any,
        breaks: Union[Index, np.ndarray],
        subtype: Union[str, type],
    ) -> None:
        expected_kwargs: Dict[str, Any] = self.get_kwargs_from_breaks(breaks.astype(subtype))
        expected: IntervalIndex = constructor(**expected_kwargs)
        result_kwargs: Dict[str, Any] = self.get_kwargs_from_breaks(breaks)
        iv_dtype: IntervalDtype = IntervalDtype(subtype, "right")
        for dtype in (iv_dtype, str(iv_dtype)):
            result: IntervalIndex = constructor(dtype=dtype, **result_kwargs)
            tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "breaks",
        [
            Index([0, 1, 2, 3, 4], dtype=np.int64),
            Index([0, 1, 2, 3, 4], dtype=np.uint64),
            Index([0, 1, 2, 3, 4], dtype=np.float64),
            date_range("2017-01-01", periods=5),
            timedelta_range("1 day", periods=5),
        ],
    )
    def test_constructor_pass_closed(
        self, constructor: Any, breaks: Union[Index, np.ndarray]
    ) -> None:
        iv_dtype: IntervalDtype = IntervalDtype(breaks.dtype)
        result_kwargs: Dict[str, Any] = self.get_kwargs_from_breaks(breaks)
        for dtype in (iv_dtype, str(iv_dtype)):
            with tm.assert_produces_warning(None):
                result: IntervalIndex = constructor(dtype=dtype, closed="left", **result_kwargs)
            assert result.dtype.closed == "left"

    @pytest.mark.parametrize("breaks", [[np.nan] * 2, [np.nan] * 4, [np.nan] * 50])
    def test_constructor_nan(
        self, constructor: Any, breaks: List[Optional[Any]], closed: str
    ) -> None:
        result_kwargs: Dict[str, Any] = self.get_kwargs_from_breaks(breaks)
        result: IntervalIndex = constructor(closed=closed, **result_kwargs)
        expected_subtype: type = np.float64
        expected_values: np.ndarray = np.array(breaks[:-1], dtype=object)
        assert result.closed == closed
        assert result.dtype.subtype == expected_subtype
        tm.assert_numpy_array_equal(np.array(result), expected_values)

    @pytest.mark.parametrize(
        "breaks",
        [
            [],
            np.array([], dtype="int64"),
            np.array([], dtype="uint64"),
            np.array([], dtype="float64"),
            np.array([], dtype="datetime64[ns]"),
            np.array([], dtype="timedelta64[ns]"),
        ],
    )
    def test_constructor_empty(
        self, constructor: Any, breaks: Union[List[Any], np.ndarray], closed: str
    ) -> None:
        result_kwargs: Dict[str, Any] = self.get_kwargs_from_breaks(breaks)
        result: IntervalIndex = constructor(closed=closed, **result_kwargs)
        expected_values: np.ndarray = np.array([], dtype=object)
        expected_subtype: Union[type, Any] = getattr(breaks, "dtype", np.int64)
        assert result.empty
        assert result.closed == closed
        assert result.dtype.subtype == expected_subtype
        tm.assert_numpy_array_equal(np.array(result), expected_values)

    @pytest.mark.parametrize(
        "breaks",
        [
            tuple("0123456789"),
            list("abcdefghij"),
            np.array(list("abcdefghij"), dtype=object),
            np.array(list("abcdefghij"), dtype="<U1"),
        ],
    )
    def test_constructor_string(
        self, constructor: Any, breaks: Union[List[str], Tuple[str, ...], np.ndarray]
    ) -> None:
        msg = "category, object, and string subtypes are not supported for IntervalIndex"
        with pytest.raises(TypeError, match=msg):
            constructor(**self.get_kwargs_from_breaks(breaks))

    @pytest.mark.parametrize("cat_constructor", [Categorical, CategoricalIndex])
    def test_constructor_categorical_valid(
        self, constructor: Any, cat_constructor: Union[type, Any]
    ) -> None:
        breaks: np.ndarray = np.arange(10, dtype="int64")
        expected: IntervalIndex = IntervalIndex.from_breaks(breaks)
        cat_breaks: Union[Categorical, CategoricalIndex] = cat_constructor(breaks)
        result_kwargs: Dict[str, Any] = self.get_kwargs_from_breaks(cat_breaks)
        result: IntervalIndex = constructor(**result_kwargs)
        tm.assert_index_equal(result, expected)

    def test_generic_errors(self, constructor: Any) -> None:
        filler: Dict[str, Any] = self.get_kwargs_from_breaks(range(10))
        msg = "closed must be one of 'right', 'left', 'both', 'neither'"
        with pytest.raises(ValueError, match=msg):
            constructor(closed="invalid", **filler)
        msg = "dtype must be an IntervalDtype, got int64"
        with pytest.raises(TypeError, match=msg):
            constructor(dtype="int64", **filler)
        msg = 'data type ["\']invalid["\'] not understood'
        with pytest.raises(TypeError, match=msg):
            constructor(dtype="invalid", **filler)
        periods: pd.PeriodIndex = period_range("2000-01-01", periods=10)
        periods_kwargs: Dict[str, Any] = self.get_kwargs_from_breaks(periods)
        msg = "Period dtypes are not supported, use a PeriodIndex instead"
        with pytest.raises(ValueError, match=msg):
            constructor(**periods_kwargs)
        decreasing_kwargs: Dict[str, Any] = self.get_kwargs_from_breaks(range(10, -1, -1))
        msg = "left side of interval must be <= right side"
        with pytest.raises(ValueError, match=msg):
            constructor(**decreasing_kwargs)


class TestFromArrays(ConstructorTests):
    """Tests specific to IntervalIndex.from_arrays"""

    @pytest.fixture
    def constructor(self) -> Any:
        """Fixture for IntervalIndex.from_arrays constructor"""
        return IntervalIndex.from_arrays

    def get_kwargs_from_breaks(
        self, breaks: Union[List[Any], np.ndarray, Index], closed: str = "right"
    ) -> Dict[str, Any]:
        """
        converts intervals in breaks format to a dictionary of kwargs to
        specific to the format expected by IntervalIndex.from_arrays
        """
        return {"left": breaks[:-1], "right": breaks[1:]}

    def test_constructor_errors(self) -> None:
        data: Categorical = Categorical(list("01234abcde"), ordered=True)
        msg = "category, object, and string subtypes are not supported for IntervalIndex"
        with pytest.raises(TypeError, match=msg):
            IntervalIndex.from_arrays(data[:-1], data[1:])
        left: List[int] = [0, 1, 2]
        right: List[int] = [2, 3]
        msg = "left and right must have the same length"
        with pytest.raises(ValueError, match=msg):
            IntervalIndex.from_arrays(left, right)

    @pytest.mark.parametrize("left_subtype_right_subtype", [(np.int64, np.float64), (np.float64, np.int64)])
    def test_mixed_float_int(
        self, left_subtype_right_subtype: Tuple[type, type]
    ) -> None:
        """mixed int/float left/right results in float for both sides"""
        left_subtype, right_subtype = left_subtype_right_subtype
        left: np.ndarray = np.arange(9, dtype=left_subtype)
        right: np.ndarray = np.arange(1, 10, dtype=right_subtype)
        result: IntervalIndex = IntervalIndex.from_arrays(left, right)
        expected_left: Index = Index(left.astype(np.float64), dtype=np.float64)
        expected_right: Index = Index(right.astype(np.float64), dtype=np.float64)
        expected_subtype: type = np.float64
        tm.assert_index_equal(result.left, expected_left)
        tm.assert_index_equal(result.right, expected_right)
        assert result.dtype.subtype == expected_subtype

    @pytest.mark.parametrize("interval_cls", [IntervalArray, IntervalIndex])
    def test_from_arrays_mismatched_datetimelike_resos(
        self, interval_cls: Union[type, Any]
    ) -> None:
        left: pd.DatetimeIndex = date_range("2016-01-01", periods=3, freq="S")
        right: pd.DatetimeIndex = date_range("2017-01-01", periods=3, freq="L")
        result: IntervalIndex = interval_cls.from_arrays(left, right)
        expected: IntervalIndex = interval_cls.from_arrays(left.as_unit("ms"), right)
        tm.assert_equal(result, expected)
        left2: pd.DatetimeIndex = left - left[0]
        right2: pd.DatetimeIndex = right - left[0]
        result2: IntervalIndex = interval_cls.from_arrays(left2, right2)
        expected2: IntervalIndex = interval_cls.from_arrays(left2.as_unit("ms"), right2)
        tm.assert_equal(result2, expected2)
        left3: pd.DatetimeIndex = left.tz_localize("UTC")
        right3: pd.DatetimeIndex = right.tz_localize("UTC")
        result3: IntervalIndex = interval_cls.from_arrays(left3, right3)
        expected3: IntervalIndex = interval_cls.from_arrays(left3.as_unit("ms"), right3)
        tm.assert_equal(result3, expected3)


class TestFromBreaks(ConstructorTests):
    """Tests specific to IntervalIndex.from_breaks"""

    @pytest.fixture
    def constructor(self) -> Any:
        """Fixture for IntervalIndex.from_breaks constructor"""
        return IntervalIndex.from_breaks

    def get_kwargs_from_breaks(
        self, breaks: Union[List[Any], np.ndarray, Index], closed: str = "right"
    ) -> Dict[str, Any]:
        """
        converts intervals in breaks format to a dictionary of kwargs to
        specific to the format expected by IntervalIndex.from_breaks
        """
        return {"breaks": breaks}

    def test_constructor_errors(self) -> None:
        data: Categorical = Categorical(list("01234abcde"), ordered=True)
        msg = "category, object, and string subtypes are not supported for IntervalIndex"
        with pytest.raises(TypeError, match=msg):
            IntervalIndex.from_breaks(data)

    def test_length_one(self) -> None:
        """breaks of length one produce an empty IntervalIndex"""
        breaks: List[int] = [0]
        result: IntervalIndex = IntervalIndex.from_breaks(breaks)
        expected: IntervalIndex = IntervalIndex.from_breaks([])
        tm.assert_index_equal(result, expected)

    def test_left_right_dont_share_data(self) -> None:
        breaks: np.ndarray = np.arange(5)
        result: IntervalIndex = IntervalIndex.from_breaks(breaks)._data
        assert result._left.base is None or result._left.base is not result._right.base


class TestFromTuples(ConstructorTests):
    """Tests specific to IntervalIndex.from_tuples"""

    @pytest.fixture
    def constructor(self) -> Any:
        """Fixture for IntervalIndex.from_tuples constructor"""
        return IntervalIndex.from_tuples

    def get_kwargs_from_breaks(
        self, breaks: Union[List[Any], np.ndarray, Index], closed: str = "right"
    ) -> Dict[str, Any]:
        """
        converts intervals in breaks format to a dictionary of kwargs to
        specific to the format expected by IntervalIndex.from_tuples
        """
        if is_unsigned_integer_dtype(breaks):
            pytest.skip(f"{breaks.dtype} not relevant IntervalIndex.from_tuples tests")
        if len(breaks) == 0:
            return {"data": breaks}
        tuples: List[Tuple[Any, Any]] = list(zip(breaks[:-1], breaks[1:]))
        if isinstance(breaks, (list, tuple)):
            return {"data": tuples}
        elif isinstance(getattr(breaks, "dtype", None), CategoricalDtype):
            return {"data": breaks._constructor(tuples)}
        return {"data": com.asarray_tuplesafe(tuples)}

    def test_constructor_errors(self) -> None:
        tuples: List[Union[Tuple[int, int], int]] = [(0, 1), 2, (3, 4)]
        msg = "IntervalIndex.from_tuples received an invalid item, 2"
        with pytest.raises(TypeError, match=msg):
            IntervalIndex.from_tuples(tuples)
        tuples = [(0, 1), (2,), (3, 4)]
        msg = "IntervalIndex.from_tuples requires tuples of length 2, got [(0, 1), (2,), (3, 4)]"
        with pytest.raises(ValueError, match=msg):
            IntervalIndex.from_tuples(tuples)
        tuples = [(0, 1), (2, 3, 4), (5, 6)]
        msg = "IntervalIndex.from_tuples requires tuples of length 2, got [(0, 1), (2, 3, 4), (5, 6)]"
        with pytest.raises(ValueError, match=msg):
            IntervalIndex.from_tuples(tuples)

    def test_na_tuples(self) -> None:
        na_tuple: List[Union[Tuple[Any, Any], Any]] = [(0, 1), (np.nan, np.nan), (2, 3)]
        idx_na_tuple: IntervalIndex = IntervalIndex.from_tuples(na_tuple)
        idx_na_element: IntervalIndex = IntervalIndex.from_tuples([(0, 1), np.nan, (2, 3)])
        tm.assert_index_equal(idx_na_tuple, idx_na_element)


class TestClassConstructors(ConstructorTests):
    """Tests specific to the IntervalIndex/Index constructors"""

    @pytest.fixture
    def constructor(self) -> Any:
        """Fixture for IntervalIndex class constructor"""
        return IntervalIndex

    def get_kwargs_from_breaks(
        self, breaks: Union[List[Any], np.ndarray, Index], closed: str = "right"
    ) -> Dict[str, Any]:
        """
        converts intervals in breaks format to a dictionary of kwargs to
        specific to the format expected by the IntervalIndex/Index constructors
        """
        if is_unsigned_integer_dtype(breaks):
            pytest.skip(f"{breaks.dtype} not relevant for class constructor tests")
        if len(breaks) == 0:
            return {"data": breaks}
        ivs: List[Optional[Interval]] = [
            Interval(left, right, closed) if notna(left) else left
            for left, right in zip(breaks[:-1], breaks[1:])
        ]
        if isinstance(breaks, list):
            return {"data": ivs}
        elif isinstance(getattr(breaks, "dtype", None), CategoricalDtype):
            return {"data": breaks._constructor(ivs)}
        return {"data": np.array(ivs, dtype=object)}

    def test_generic_errors(self, constructor: Any) -> None:
        """
        override the base class implementation since errors are handled
        differently; checks unnecessary since caught at the Interval level
        """

    def test_constructor_string(self) -> None:
        pass

    @pytest.mark.parametrize(
        "klass",
        [IntervalIndex, partial(Index, dtype="interval")],
        ids=["IntervalIndex", "Index"],
    )
    def test_constructor_errors(self, klass: Union[type, Any]) -> None:
        ivs: List[Interval] = [
            Interval(0, 1, closed="right"),
            Interval(2, 3, closed="left"),
        ]
        msg = "intervals must all be closed on the same side"
        with pytest.raises(ValueError, match=msg):
            klass(ivs)
        msg = "(IntervalIndex|Index)\\(...\\) must be called with a collection of some kind, 5 was passed"
        with pytest.raises(TypeError, match=msg):
            klass(5)
        msg = "type <class 'numpy.int(32|64)'> with value 0 is not an interval"
        with pytest.raises(TypeError, match=msg):
            klass([0, 1])

    @pytest.mark.parametrize(
        "data, closed",
        [
            ([], "both"),
            ([np.nan, np.nan], "neither"),
            (
                [Interval(0, 3, closed="neither"), Interval(2, 5, closed="neither")],
                "left",
            ),
            (
                [Interval(0, 3, closed="left"), Interval(2, 5, closed="right")],
                "neither",
            ),
            (IntervalIndex.from_breaks(range(5), closed="both"), "right"),
        ],
    )
    def test_override_inferred_closed(
        self, constructor: Any, data: Union[List[Optional[Interval]], IntervalIndex], closed: str
    ) -> None:
        if isinstance(data, IntervalIndex):
            tuples: List[Tuple[Any, Any]] = list(data.to_tuples())
        else:
            tuples = [
                (iv.left, iv.right) if notna(iv) else iv for iv in cast(List[Optional[Interval]], data)
            ]
        expected: IntervalIndex = IntervalIndex.from_tuples(tuples, closed=closed)
        result: IntervalIndex = constructor(data, closed=closed)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("values_constructor", [list, np.array, IntervalIndex, IntervalArray])
    def test_index_object_dtype(
        self, values_constructor: Union[type, Any]
    ) -> None:
        intervals: List[Interval] = [Interval(0, 1), Interval(1, 2), Interval(2, 3)]
        values: Union[List[Interval], np.ndarray, IntervalIndex, IntervalArray] = values_constructor(
            intervals
        )
        result: Index = Index(values, dtype=object)
        assert type(result) is Index
        tm.assert_numpy_array_equal(result.values, np.array(values))

    def test_index_mixed_closed(self) -> None:
        intervals: List[Interval] = [
            Interval(0, 1, closed="left"),
            Interval(1, 2, closed="right"),
            Interval(2, 3, closed="neither"),
            Interval(3, 4, closed="both"),
        ]
        result: Index = Index(intervals)
        expected: Index = Index(intervals, dtype=object)
        tm.assert_index_equal(result, expected)


@pytest.mark.parametrize(
    "timezone, inclusive_endpoints_fixture",
    [
        ("UTC", "right"),
        ("US/Pacific", "left"),
        ("GMT", "both"),
    ],
)
def test_interval_index_subtype(
    timezone: str, inclusive_endpoints_fixture: str
) -> None:
    dates: pd.DatetimeIndex = date_range("2022", periods=3, tz=timezone)
    dtype: str = f"interval[datetime64[ns, {timezone}], {inclusive_endpoints_fixture}]"
    result: IntervalIndex = IntervalIndex.from_arrays(
        ["2022-01-01", "2022-01-02"],
        ["2022-01-02", "2022-01-03"],
        closed=inclusive_endpoints_fixture,
        dtype=dtype,
    )
    expected: IntervalIndex = IntervalIndex.from_arrays(
        dates[:-1], dates[1:], closed=inclusive_endpoints_fixture
    )
    tm.assert_index_equal(result, expected)


def test_dtype_closed_mismatch() -> None:
    dtype: IntervalDtype = IntervalDtype(np.int64, "left")
    msg = "closed keyword does not match dtype.closed"
    with pytest.raises(ValueError, match=msg):
        IntervalIndex([], dtype=dtype, closed="neither")
    with pytest.raises(ValueError, match=msg):
        IntervalArray([], dtype=dtype, closed="neither")


@pytest.mark.parametrize(
    "dtype",
    [
        "Float64",
        pytest.param(
            "float64[pyarrow]",
            marks=td.skip_if_no("pyarrow"),
        ),
    ],
)
def test_ea_dtype(dtype: str) -> None:
    bins: List[Tuple[float, float]] = [(0.0, 0.4), (0.4, 0.6)]
    interval_dtype: IntervalDtype = IntervalDtype(subtype=dtype, closed="left")
    result: IntervalIndex = IntervalIndex.from_tuples(
        bins, closed="left", dtype=interval_dtype
    )
    assert result.dtype == interval_dtype
    expected: IntervalIndex = IntervalIndex.from_tuples(bins, closed="left").astype(
        interval_dtype
    )
    tm.assert_index_equal(result, expected)
