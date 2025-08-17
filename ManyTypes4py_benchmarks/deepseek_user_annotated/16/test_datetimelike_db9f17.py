from __future__ import annotations

import re
import warnings
from typing import Any, Generator, Literal, Sequence, TypeVar, Union, cast

import numpy as np
import pytest

from pandas._libs import (
    NaT,
    OutOfBoundsDatetime,
    Timestamp,
)
from pandas._libs.tslibs import to_offset
from pandas.compat.numpy import np_version_gt2

from pandas.core.dtypes.dtypes import PeriodDtype

import pandas as pd
from pandas import (
    DatetimeIndex,
    Period,
    PeriodIndex,
    TimedeltaIndex,
)
import pandas._testing as tm
from pandas.core.arrays import (
    DatetimeArray,
    NumpyExtensionArray,
    PeriodArray,
    TimedeltaArray,
)

T = TypeVar('T', bound='SharedTests')

# TODO: more freq variants
@pytest.fixture(params=["D", "B", "W", "ME", "QE", "YE"])
def freqstr(request: pytest.FixtureRequest) -> Generator[str, None, None]:
    """Fixture returning parametrized frequency in string format."""
    yield request.param


@pytest.fixture
def period_index(freqstr: str) -> Generator[PeriodIndex, None, None]:
    """
    A fixture to provide PeriodIndex objects with different frequencies.

    Most PeriodArray behavior is already tested in PeriodIndex tests,
    so here we just test that the PeriodArray behavior matches
    the PeriodIndex behavior.
    """
    # TODO: non-monotone indexes; NaTs, different start dates
    with warnings.catch_warnings():
        # suppress deprecation of Period[B]
        warnings.filterwarnings(
            "ignore", message="Period with BDay freq", category=FutureWarning
        )
        freqstr = PeriodDtype(to_offset(freqstr))._freqstr
        pi = pd.period_range(start=Timestamp("2000-01-01"), periods=100, freq=freqstr)
    yield pi


@pytest.fixture
def datetime_index(freqstr: str) -> Generator[DatetimeIndex, None, None]:
    """
    A fixture to provide DatetimeIndex objects with different frequencies.

    Most DatetimeArray behavior is already tested in DatetimeIndex tests,
    so here we just test that the DatetimeArray behavior matches
    the DatetimeIndex behavior.
    """
    # TODO: non-monotone indexes; NaTs, different start dates, timezones
    dti = pd.date_range(start=Timestamp("2000-01-01"), periods=100, freq=freqstr)
    yield dti


@pytest.fixture
def timedelta_index() -> Generator[TimedeltaIndex, None, None]:
    """
    A fixture to provide TimedeltaIndex objects with different frequencies.
     Most TimedeltaArray behavior is already tested in TimedeltaIndex tests,
    so here we just test that the TimedeltaArray behavior matches
    the TimedeltaIndex behavior.
    """
    # TODO: flesh this out
    yield TimedeltaIndex(["1 Day", "3 Hours", "NaT"])


class SharedTests:
    index_cls: type[DatetimeIndex | PeriodIndex | TimedeltaIndex]
    array_cls: type[DatetimeArray | PeriodArray | TimedeltaArray]
    scalar_type: type[Timestamp | Period | pd.Timedelta]
    example_dtype: str | PeriodDtype

    @pytest.fixture
    def arr1d(self) -> Generator[DatetimeArray | PeriodArray | TimedeltaArray, None, None]:
        """Fixture returning DatetimeArray with daily frequency."""
        data = np.arange(10, dtype="i8") * 24 * 3600 * 10**9
        if self.array_cls is PeriodArray:
            arr = self.array_cls(data, freq="D")
        else:
            arr = self.index_cls(data, freq="D")._data
        yield arr

    def test_compare_len1_raises(self, arr1d: DatetimeArray | PeriodArray | TimedeltaArray) -> None:
        # make sure we raise when comparing with different lengths, specific
        #  to the case where one has length-1, which numpy would broadcast
        arr = arr1d
        idx = self.index_cls(arr)

        with pytest.raises(ValueError, match="Lengths must match"):
            arr == arr[:1]

        # test the index classes while we're at it, GH#23078
        with pytest.raises(ValueError, match="Lengths must match"):
            idx <= idx[[0]]

    @pytest.mark.parametrize(
        "result",
        [
            pd.date_range("2020", periods=3),
            pd.date_range("2020", periods=3, tz="UTC"),
            pd.timedelta_range("0 days", periods=3),
            pd.period_range("2020Q1", periods=3, freq="Q"),
        ],
    )
    def test_compare_with_Categorical(self, result: DatetimeIndex | TimedeltaIndex | PeriodIndex) -> None:
        expected = pd.Categorical(result)
        assert all(result == expected)
        assert not any(result != expected)

    @pytest.mark.parametrize("reverse", [True, False])
    @pytest.mark.parametrize("as_index", [True, False])
    def test_compare_categorical_dtype(
        self, 
        arr1d: DatetimeArray | PeriodArray | TimedeltaArray, 
        as_index: bool, 
        reverse: bool, 
        ordered: bool
    ) -> None:
        other = pd.Categorical(arr1d, ordered=ordered)
        if as_index:
            other = pd.CategoricalIndex(other)

        left, right = arr1d, other
        if reverse:
            left, right = right, left

        ones = np.ones(arr1d.shape, dtype=bool)
        zeros = ~ones

        result = left == right
        tm.assert_numpy_array_equal(result, ones)

        result = left != right
        tm.assert_numpy_array_equal(result, zeros)

        if not reverse and not as_index:
            # Otherwise Categorical raises TypeError bc it is not ordered
            # TODO: we should probably get the same behavior regardless?
            result = left < right
            tm.assert_numpy_array_equal(result, zeros)

            result = left <= right
            tm.assert_numpy_array_equal(result, ones)

            result = left > right
            tm.assert_numpy_array_equal(result, zeros)

            result = left >= right
            tm.assert_numpy_array_equal(result, ones)

    def test_take(self) -> None:
        data = np.arange(100, dtype="i8") * 24 * 3600 * 10**9
        np.random.default_rng(2).shuffle(data)

        if self.array_cls is PeriodArray:
            arr = PeriodArray(data, dtype="period[D]")
        else:
            arr = self.index_cls(data)._data
        idx = self.index_cls._simple_new(arr)

        takers = [1, 4, 94]
        result = arr.take(takers)
        expected = idx.take(takers)

        tm.assert_index_equal(self.index_cls(result), expected)

        takers = np.array([1, 4, 94])
        result = arr.take(takers)
        expected = idx.take(takers)

        tm.assert_index_equal(self.index_cls(result), expected)

    @pytest.mark.parametrize("fill_value", [2, 2.0, Timestamp(2021, 1, 1, 12).time])
    def test_take_fill_raises(
        self, 
        fill_value: Any, 
        arr1d: DatetimeArray | PeriodArray | TimedeltaArray
    ) -> None:
        msg = f"value should be a '{arr1d._scalar_type.__name__}' or 'NaT'. Got"
        with pytest.raises(TypeError, match=msg):
            arr1d.take([0, 1], allow_fill=True, fill_value=fill_value)

    def test_take_fill(self, arr1d: DatetimeArray | PeriodArray | TimedeltaArray) -> None:
        arr = arr1d

        result = arr.take([-1, 1], allow_fill=True, fill_value=None)
        assert result[0] is NaT

        result = arr.take([-1, 1], allow_fill=True, fill_value=np.nan)
        assert result[0] is NaT

        result = arr.take([-1, 1], allow_fill=True, fill_value=NaT)
        assert result[0] is NaT

    @pytest.mark.filterwarnings(
        "ignore:Period with BDay freq is deprecated:FutureWarning"
    )
    def test_take_fill_str(self, arr1d: DatetimeArray | PeriodArray | TimedeltaArray) -> None:
        # Cast str fill_value matching other fill_value-taking methods
        result = arr1d.take([-1, 1], allow_fill=True, fill_value=str(arr1d[-1]))
        expected = arr1d[[-1, 1]]
        tm.assert_equal(result, expected)

        msg = f"value should be a '{arr1d._scalar_type.__name__}' or 'NaT'. Got"
        with pytest.raises(TypeError, match=msg):
            arr1d.take([-1, 1], allow_fill=True, fill_value="foo")

    def test_concat_same_type(self, arr1d: DatetimeArray | PeriodArray | TimedeltaArray) -> None:
        arr = arr1d
        idx = self.index_cls(arr)
        idx = idx.insert(0, NaT)
        arr = arr1d

        result = arr._concat_same_type([arr[:-1], arr[1:], arr])
        arr2 = arr.astype(object)
        expected = self.index_cls(np.concatenate([arr2[:-1], arr2[1:], arr2]))

        tm.assert_index_equal(self.index_cls(result), expected)

    def test_unbox_scalar(self, arr1d: DatetimeArray | PeriodArray | TimedeltaArray) -> None:
        result = arr1d._unbox_scalar(arr1d[0])
        expected = arr1d._ndarray.dtype.type
        assert isinstance(result, expected)

        result = arr1d._unbox_scalar(NaT)
        assert isinstance(result, expected)

        msg = f"'value' should be a {self.scalar_type.__name__}."
        with pytest.raises(ValueError, match=msg):
            arr1d._unbox_scalar("foo")

    def test_check_compatible_with(self, arr1d: DatetimeArray | PeriodArray | TimedeltaArray) -> None:
        arr1d._check_compatible_with(arr1d[0])
        arr1d._check_compatible_with(arr1d[:1])
        arr1d._check_compatible_with(NaT)

    def test_scalar_from_string(self, arr1d: DatetimeArray | PeriodArray | TimedeltaArray) -> None:
        result = arr1d._scalar_from_string(str(arr1d[0]))
        assert result == arr1d[0]

    def test_reduce_invalid(self, arr1d: DatetimeArray | PeriodArray | TimedeltaArray) -> None:
        msg = "does not support operation 'not a method'"
        with pytest.raises(TypeError, match=msg):
            arr1d._reduce("not a method")

    @pytest.mark.parametrize("method", ["pad", "backfill"])
    def test_fillna_method_doesnt_change_orig(self, method: Literal["pad", "backfill"]) -> None:
        data = np.arange(10, dtype="i8") * 24 * 3600 * 10**9
        if self.array_cls is PeriodArray:
            arr = self.array_cls(data, dtype="period[D]")
        else:
            dtype = "M8[ns]" if self.array_cls is DatetimeArray else "m8[ns]"
            arr = self.array_cls._from_sequence(data, dtype=np.dtype(dtype))
        arr[4] = NaT

        fill_value = arr[3] if method == "pad" else arr[5]

        result = arr._pad_or_backfill(method=method)
        assert result[4] == fill_value

        # check that the original was not changed
        assert arr[4] is NaT

    def test_searchsorted(self) -> None:
        data = np.arange(10, dtype="i8") * 24 * 3600 * 10**9
        if self.array_cls is PeriodArray:
            arr = self.array_cls(data, dtype="period[D]")
        else:
            dtype = "M8[ns]" if self.array_cls is DatetimeArray else "m8[ns]"
            arr = self.array_cls._from_sequence(data, dtype=np.dtype(dtype))

        # scalar
        result = arr.searchsorted(arr[1])
        assert result == 1

        result = arr.searchsorted(arr[2], side="right")
        assert result == 3

        # own-type
        result = arr.searchsorted(arr[1:3])
        expected = np.array([1, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

        result = arr.searchsorted(arr[1:3], side="right")
        expected = np.array([2, 3], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

        # GH#29884 match numpy convention on whether NaT goes
        #  at the end or the beginning
        result = arr.searchsorted(NaT)
        assert result == 10

    @pytest.mark.parametrize("box", [None, "index", "series"])
    def test_searchsorted_castable_strings(
        self, 
        arr1d: DatetimeArray | PeriodArray | TimedeltaArray, 
        box: str | None, 
        string_storage: str
    ) -> None:
        arr = arr1d
        if box is None:
            pass
        elif box == "index":
            # Test the equivalent Index.searchsorted method while we're here
            arr = self.index_cls(arr)
        else:
            # Test the equivalent Series.searchsorted method while we're here
            arr = pd.Series(arr)

        # scalar
        result = arr.searchsorted(str(arr[1]))
        assert result == 1

        result = arr.searchsorted(str(arr[2]), side="right")
        assert result == 3

        result = arr.searchsorted([str(x) for x in arr[1:3]])
        expected = np.array([1, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"value should be a '{arr1d._scalar_type.__name__}', 'NaT', "
                "or array of those. Got 'str' instead."
            ),
        ):
            arr.searchsorted("foo")

        with pd.option_context("string_storage", string_storage):
            with pytest.raises(
                TypeError,
                match=re.escape(
                    f"value should be a '{arr1d._scalar_type.__name__}', 'NaT', "
                    "or array of those. Got string array instead."
                ),
            ):
                arr.searchsorted([str(arr[1]), "baz"])

    def test_getitem_near_implementation_bounds(self) -> None:
        # We only check tz-naive for DTA bc the bounds are slightly different
        #  for other tzs
        i8vals = np.asarray([NaT._value + n for n in range(1, 5)], dtype="i8")
        if self.array_cls is PeriodArray:
            arr = self.array_cls(i8vals, dtype="period[ns]")
        else:
            arr = self.index_cls(i8vals, freq="ns")._data
        arr[0]  # should not raise OutOfBoundsDatetime

        index = pd.Index(arr)
        index[0]  # should not raise OutOfBoundsDatetime

        ser = pd.Series(arr)
        ser[0]  # should not raise OutOfBoundsDatetime

    def test_getitem_2d(self, arr1d: DatetimeArray | PeriodArray | TimedeltaArray) -> None:
        # 2d slicing on a 1D array
        expected = type(arr1d)._simple_new(
            arr1d._ndarray[:, np.newaxis], dtype=arr1d.dtype
        )
        result = arr1d[:, np.newaxis]
        tm.assert_equal(result, expected)

        # Lookup on a 2D array
        arr2d = expected
        expected = type(arr2d)._simple_new(arr2d._ndarray[:3, 0], dtype=arr2d.dtype)
        result = arr2d[:3, 0]
        tm.assert_equal(result, expected)

        # Scalar lookup
        result = arr2d[-1, 0]
        expected = arr1d[-1]
        assert result == expected

    def test_iter_2d(self, arr1d: DatetimeArray | PeriodArray | TimedeltaArray) -> None:
        data2d = arr1d._ndarray[:3, np.newaxis]
        arr2d = type(arr1d)._simple_new(data2d, dtype=arr1d.dtype)
        result = list(arr2d)
        assert len(result) == 3
        for x in result:
            assert isinstance(x, type(arr1d))
            assert x.ndim == 1
            assert x.dtype == arr1d.dtype

    def test_repr_2d(self, arr1d: DatetimeArray | PeriodArray | TimedeltaArray) -> None:
        data2d = arr1d._ndarray[:3, np.newaxis]
        arr2d = type(arr1d)._simple_new(data2d, dtype=arr1d.dtype)

        result = repr(arr2d)

        if isinstance(arr2d, TimedeltaArray):
            expected = (
                f"<{type(arr2d).__name__}>\n"
                "[\n"
                f"['{arr1d[0]._repr_base()}'],\n"
                f"['{arr1d[1]._repr_base