from functools import partial
from typing import Any, List, Optional, Sequence, Tuple, Union, cast

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
        breaks_and_expected_subtype: Tuple[Any, Any],
        closed: str,
        name: Optional[str],
    ) -> None:
        breaks, expected_subtype = breaks_and_expected_subtype

        result_kwargs = self.get_kwargs_from_breaks(breaks, closed)

        result = constructor(closed=closed, name=name, **result_kwargs)

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
    def test_constructor_dtype(self, constructor: Any, breaks: Any, subtype: str) -> None:
        # GH 19262: conversion via dtype parameter
        expected_kwargs = self.get_kwargs_from_breaks(breaks.astype(subtype))
        expected = constructor(**expected_kwargs)

        result_kwargs = self.get_kwargs_from_breaks(breaks)
        iv_dtype = IntervalDtype(subtype, "right")
        for dtype in (iv_dtype, str(iv_dtype)):
            result = constructor(dtype=dtype, **result_kwargs)
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
    def test_constructor_pass_closed(self, constructor: Any, breaks: Any) -> None:
        # not passing closed to IntervalDtype, but to IntervalArray constructor
        iv_dtype = IntervalDtype(breaks.dtype)

        result_kwargs = self.get_kwargs_from_breaks(breaks)

        for dtype in (iv_dtype, str(iv_dtype)):
            with tm.assert_produces_warning(None):
                result = constructor(dtype=dtype, closed="left", **result_kwargs)
            assert result.dtype.closed == "left"

    @pytest.mark.parametrize("breaks", [[np.nan] * 2, [np.nan] * 4, [np.nan] * 50])
    def test_constructor_nan(self, constructor: Any, breaks: Any, closed: str) -> None:
        # GH 18421
        result_kwargs = self.get_kwargs_from_breaks(breaks)
        result = constructor(closed=closed, **result_kwargs)

        expected_subtype = np.float64
        expected_values = np.array(breaks[:-1], dtype=object)

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
    def test_constructor_empty(self, constructor: Any, breaks: Any, closed: str) -> None:
        # GH 18421
        result_kwargs = self.get_kwargs_from_breaks(breaks)
        result = constructor(closed=closed, **result_kwargs)

        expected_values = np.array([], dtype=object)
        expected_subtype = getattr(breaks, "dtype", np.int64)

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
    def test_constructor_string(self, constructor: Any, breaks: Any) -> None:
        # GH 19016
        msg = (
            "category, object, and string subtypes are not supported for IntervalIndex"
        )
        with pytest.raises(TypeError, match=msg):
            constructor(**self.get_kwargs_from_breaks(breaks))

    @pytest.mark.parametrize("cat_constructor", [Categorical, CategoricalIndex])
    def test_constructor_categorical_valid(
        self, constructor: Any, cat_constructor: Any
    ) -> None:
        # GH 21243/21253

        breaks = np.arange(10, dtype="int64")
        expected = IntervalIndex.from_breaks(breaks)

        cat_breaks = cat_constructor(breaks)
        result_kwargs = self.get_kwargs_from_breaks(cat_breaks)
        result = constructor(**result_kwargs)
        tm.assert_index_equal(result, expected)

    def test_generic_errors(self, constructor: Any) -> None:
        # filler input data to be used when supplying invalid kwargs
        filler = self.get_kwargs_from_breaks(range(10))

        # invalid closed
        msg = "closed must be one of 'right', 'left', 'both', 'neither'"
        with pytest.raises(ValueError, match=msg):
            constructor(closed="invalid", **filler)

        # unsupported dtype
        msg = "dtype must be an IntervalDtype, got int64"
        with pytest.raises(TypeError, match=msg):
            constructor(dtype="int64", **filler)

        # invalid dtype
        msg = "data type [\"']invalid[\"'] not understood"
        with pytest.raises(TypeError, match=msg):
            constructor(dtype="invalid", **filler)

        # no point in nesting periods in an IntervalIndex
        periods = period_range("2000-01-01", periods=10)
        periods_kwargs = self.get_kwargs_from_breaks(periods)
        msg = "Period dtypes are not supported, use a PeriodIndex instead"
        with pytest.raises(ValueError, match=msg):
            constructor(**periods_kwargs)

        # decreasing values
        decreasing_kwargs = self.get_kwargs_from_breaks(range(10, -1, -1))
        msg = "left side of interval must be <= right side"
        with pytest.raises(ValueError, match=msg):
            constructor(**decreasing_kwargs)


class TestFromArrays(ConstructorTests):
    """Tests specific to IntervalIndex.from_arrays"""

    @pytest.fixture
    def constructor(self) -> Any:
        """Fixture for IntervalIndex.from_arrays constructor"""
        return IntervalIndex.from_arrays

    def get_kwargs_from_breaks(self, breaks: Any, closed: str = "right") -> Any:
        """
        converts intervals in breaks format to a dictionary of kwargs to
        specific to the format expected by IntervalIndex.from_arrays
        """
        return {"left": breaks[:-1], "right": breaks[1:]}

    def test_constructor_errors(self) -> None:
        # GH 19016: categorical data
        data = Categorical(list("01234abcde"), ordered=True)
        msg = (
            "category, object, and string subtypes are not supported for IntervalIndex"
        )
        with pytest.raises(TypeError, match=msg):
            IntervalIndex.from_arrays(data[:-1], data[1:])

        # unequal length
        left = [0, 1, 2]
        right = [2, 3]
        msg = "left and right must have the same length"
        with pytest.raises(ValueError, match=msg):
            IntervalIndex.from_arrays(left, right)

    @pytest.mark.parametrize(
        "left_subtype, right_subtype", [(np.int64, np.float64), (np.float64, np.int64)]
    )
    def test_mixed_float_int(self, left_subtype: Any, right_subtype: Any) -> None:
        """mixed int/float left/right results in float for both sides"""
        left = np.arange(9, dtype=left_subtype)
        right = np.arange(1, 10, dtype=right_subtype)
        result = IntervalIndex.from_arrays(left, right)

        expected_left = Index(left, dtype=np.float64)
        expected_right = Index(right, dtype=np.float64)
        expected_subtype = np.float64

        tm.assert_index_equal(result.left, expected_left)
        tm.assert_index_equal(result.right, expected_right)
        assert result.dtype.subtype == expected_subtype

    @pytest.mark.parametrize("interval_cls", [IntervalArray, IntervalIndex])
    def test_from_arrays_mismatched_datetimelike_resos(self, interval_cls: Any) -> None:
        # GH#55714
        left = date_range("2016-01-01", periods=3, unit="s")
        right = date_range("2017-01-01", periods=3, unit="ms")
        result = interval_cls.from_arrays(left, right)
        expected = interval_cls.from_arrays(left.as_unit("ms"), right)
        tm.assert_equal(result, expected)

        # td64
        left2 = left - left[0]
        right2 = right - left[0]
        result2 = interval_cls.from_arrays(left2, right2)
        expected2 = interval_cls.from_arrays(left2.as_unit("ms"), right2)
        tm.assert_equal(result2, expected2)

        # dt64tz
        left3 = left.tz_localize("UTC")
        right3 = right.tz_localize("UTC")
        result3 = interval_cls.from_arrays(left3, right3)
        expected3 = interval_cls.from_arrays(left3.as_unit("ms"), right3)
        tm.assert_equal(result3, expected3)


class TestFromBreaks(ConstructorTests):
    """Tests specific to IntervalIndex.from_breaks"""

    @pytest.fixture
    def constructor(self) -> Any:
        """Fixture for IntervalIndex.from_breaks constructor"""
        return IntervalIndex.from_breaks

    def get_kwargs_from_breaks(self, breaks: Any, closed: str = "right") -> Any:
        """
        converts intervals in breaks format to a dictionary of kwargs to
        specific to the format expected by IntervalIndex.from_breaks
        """
        return {"breaks": breaks}

    def test_constructor_errors(self) -> None:
        # GH 19016: categorical data
        data = Categorical(list("01234abcde"), ordered=True)
        msg = (
            "category, object, and string subtypes are not supported for IntervalIndex"
        )
        with pytest.raises(TypeError, match=msg):
            IntervalIndex.from_breaks(data)

    def test_length_one(self) -> None:
        """breaks of length one produce an empty IntervalIndex"""
        breaks = [0]
        result = IntervalIndex.from_breaks(breaks)
        expected = IntervalIndex.from_breaks([])
        tm.assert_index_equal(result, expected)

    def test_left_right_dont_share_data(self) -> None:
        # GH#36310
        breaks = np.arange(5)
        result = IntervalIndex.from_breaks(breaks)._data
        assert result._left.base is None or result._left.base is not result._right.base


class TestFromTuples(ConstructorTests):
    """Tests specific to IntervalIndex.from_tuples"""

    @pytest.fixture
    def constructor(self) -> Any:
        """Fixture for IntervalIndex.from_tuples constructor"""
        return IntervalIndex.from_tuples

    def get_kwargs_from_breaks(self, breaks: Any, closed: str = "right") -> Any:
        """
        converts intervals in breaks format to a dictionary of kwargs to
        specific to the format expected by IntervalIndex.from_tuples
        """
        if is_unsigned_integer_dtype(breaks):
            pytest.skip(f"{breaks.dtype} not relevant IntervalIndex.from_tuples tests")

        if len(breaks) == 0:
            return {"data": breaks}

        tuples = list(zip(breaks[:-1], breaks[1:]))
        if isinstance(breaks, (list, tuple)):
            return {"data": tuples}
        elif isinstance(getattr(breaks, "dtype", None), CategoricalDtype):
            return {"data": breaks._constructor(tuples)}
        return {"data": com.asarray_tuplesafe(tuples)}

    def test_constructor_errors(self) -> None:
        # non-tuple
        tuples = [(0, 1), 2, (3, 4)]
        msg = "IntervalIndex.from_tuples received an invalid item, 2"
        with pytest.raises(TypeError, match=msg.format(t=tuples)):
            IntervalIndex.from_tuples(tuples)

        # too few/many items
        tuples = [(0, 1), (2,), (3, 4)]
        msg = "IntervalIndex.from_tuples requires tuples of length 2, got {t}"
        with pytest.raises(ValueError, match=msg.format(t=tuples)):
            IntervalIndex.from_tuples(tuples)

        tuples = [(0, 1), (2, 3, 4), (5, 6)]
        with pytest.raises(ValueError, match=msg.format(t=tuples)):
            IntervalIndex.from_tuples(tuples)

    def test_na_tuples(self) -> None:
        # tuple (NA, NA) evaluates the same as NA as an element
        na_tuple = [(0, 1), (np.nan, np.nan), (2, 3)]
        idx_na_tuple = IntervalIndex.from_tuples(na_tuple)
        idx_na_element = IntervalIndex.from_tuples([(0, 1), np.nan, (2, 3)])
        tm.assert_index_equal(idx_na_tuple, idx_na_element)


class TestClassConstructors(ConstructorTests):
    """Tests specific to the IntervalIndex/Index constructors"""

    @pytest.fixture
    def constructor(self) -> Any:
        """Fixture for IntervalIndex class constructor"""
        return IntervalIndex

    def get_kwargs_from_breaks(self, breaks: Any, closed: str = "right") -> Any:
        """
        converts intervals in breaks format to a dictionary of kwargs to
        specific to the format expected by the IntervalIndex/Index constructors
        """
        if is_unsigned_integer_dtype(breaks):
            pytest.skip(f"{breaks.dtype} not relevant for class constructor tests")

        if len(breaks) == 0:
            return {"data": breaks}

        ivs = [
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
        # GH23013
        # When forming the interval from breaks,
        # the interval of strings is already forbidden.
        pass

    @pytest.mark.parametrize(
       