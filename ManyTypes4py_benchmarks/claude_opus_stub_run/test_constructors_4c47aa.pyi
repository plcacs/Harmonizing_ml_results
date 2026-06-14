from functools import partial
from typing import Any

import numpy as np
import pytest

from pandas import (
    Categorical,
    CategoricalDtype,
    CategoricalIndex,
    Index,
    Interval,
    IntervalIndex,
)
from pandas.core.arrays import IntervalArray
from pandas.core.dtypes.dtypes import IntervalDtype


class ConstructorTests:
    def get_kwargs_from_breaks(self, breaks: Any, closed: str = "right") -> dict[str, Any]: ...

    @pytest.mark.parametrize("breaks_and_expected_subtype", [
        ([3, 14, 15, 92, 653], np.int64),
        (np.arange(10, dtype="int64"), np.int64),
        (Index(np.arange(-10, 11, dtype=np.int64)), np.int64),
        (Index(np.arange(10, 31, dtype=np.uint64)), np.uint64),
        (Index(np.arange(20, 30, 0.5), dtype=np.float64), np.float64),
    ])
    @pytest.mark.parametrize("name", [None, "foo"])
    def test_constructor(
        self,
        constructor: Any,
        breaks_and_expected_subtype: tuple[Any, Any],
        closed: str,
        name: str | None,
    ) -> None: ...

    @pytest.mark.parametrize("breaks, subtype", [])
    def test_constructor_dtype(
        self, constructor: Any, breaks: Index, subtype: str
    ) -> None: ...

    @pytest.mark.parametrize("breaks", [])
    def test_constructor_pass_closed(
        self, constructor: Any, breaks: Index
    ) -> None: ...

    @pytest.mark.parametrize("breaks", [[np.nan] * 2, [np.nan] * 4, [np.nan] * 50])
    def test_constructor_nan(
        self, constructor: Any, breaks: list[float], closed: str
    ) -> None: ...

    @pytest.mark.parametrize("breaks", [])
    def test_constructor_empty(
        self, constructor: Any, breaks: Any, closed: str
    ) -> None: ...

    @pytest.mark.parametrize("breaks", [])
    def test_constructor_string(self, constructor: Any, breaks: Any) -> None: ...

    @pytest.mark.parametrize("cat_constructor", [Categorical, CategoricalIndex])
    def test_constructor_categorical_valid(
        self, constructor: Any, cat_constructor: type
    ) -> None: ...

    def test_generic_errors(self, constructor: Any) -> None: ...


class TestFromArrays(ConstructorTests):
    @pytest.fixture
    def constructor(self) -> Any: ...

    def get_kwargs_from_breaks(self, breaks: Any, closed: str = "right") -> dict[str, Any]: ...

    def test_constructor_errors(self) -> None: ...

    @pytest.mark.parametrize("left_subtype, right_subtype", [
        (np.int64, np.float64),
        (np.float64, np.int64),
    ])
    def test_mixed_float_int(
        self, left_subtype: type, right_subtype: type
    ) -> None: ...

    @pytest.mark.parametrize("interval_cls", [IntervalArray, IntervalIndex])
    def test_from_arrays_mismatched_datetimelike_resos(
        self, interval_cls: type
    ) -> None: ...


class TestFromBreaks(ConstructorTests):
    @pytest.fixture
    def constructor(self) -> Any: ...

    def get_kwargs_from_breaks(self, breaks: Any, closed: str = "right") -> dict[str, Any]: ...

    def test_constructor_errors(self) -> None: ...

    def test_length_one(self) -> None: ...

    def test_left_right_dont_share_data(self) -> None: ...


class TestFromTuples(ConstructorTests):
    @pytest.fixture
    def constructor(self) -> Any: ...

    def get_kwargs_from_breaks(self, breaks: Any, closed: str = "right") -> dict[str, Any]: ...

    def test_constructor_errors(self) -> None: ...

    def test_na_tuples(self) -> None: ...


class TestClassConstructors(ConstructorTests):
    @pytest.fixture
    def constructor(self) -> Any: ...

    def get_kwargs_from_breaks(self, breaks: Any, closed: str = "right") -> dict[str, Any]: ...

    def test_generic_errors(self, constructor: Any) -> None: ...

    def test_constructor_string(self) -> None: ...  # type: ignore[override]

    @pytest.mark.parametrize("klass", [IntervalIndex, partial(Index, dtype="interval")])
    def test_constructor_errors(self, klass: Any) -> None: ...

    @pytest.mark.parametrize("data, closed", [])
    def test_override_inferred_closed(
        self, constructor: Any, data: Any, closed: str
    ) -> None: ...

    @pytest.mark.parametrize("values_constructor", [list, np.array, IntervalIndex, IntervalArray])
    def test_index_object_dtype(self, values_constructor: type) -> None: ...

    def test_index_mixed_closed(self) -> None: ...


def test_interval_index_subtype(
    timezone: str, inclusive_endpoints_fixture: str
) -> None: ...

def test_dtype_closed_mismatch() -> None: ...

def test_ea_dtype(dtype: str) -> None: ...