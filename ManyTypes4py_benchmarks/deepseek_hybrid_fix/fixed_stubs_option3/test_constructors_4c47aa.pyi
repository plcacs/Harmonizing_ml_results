from __future__ import annotations

from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)
from typing_extensions import Literal

import numpy as np
import pandas as pd
import pandas._testing as tm
import pandas.core.common as com
import pandas.util._test_decorators as td
import pytest
from pandas import (
    Categorical,
    CategoricalDtype,
    CategoricalIndex,
    Index,
    Interval,
    IntervalDtype,
    IntervalIndex,
    date_range,
    notna,
    period_range,
    timedelta_range,
)
from pandas.core.arrays import IntervalArray
from pandas.core.dtypes.common import is_unsigned_integer_dtype


class ConstructorTests:
    """Common tests for all variations of IntervalIndex construction."""

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
        constructor: Callable[..., IntervalIndex],
        breaks_and_expected_subtype: Tuple[
            Union[List[int], np.ndarray, Index, pd.DatetimeIndex, pd.TimedeltaIndex],
            Union[np.dtype, str],
        ],
        closed: Literal["right", "left", "both", "neither"],
        name: Optional[str],
    ) -> None: ...

    def get_kwargs_from_breaks(
        self, breaks: Any, closed: Literal["right", "left", "both", "neither"] = "right"
    ) -> Dict[str, Any]: ...


class TestFromArrays(ConstructorTests):
    @pytest.fixture
    def constructor(self) -> Callable[..., IntervalIndex]: ...

    def get_kwargs_from_breaks(
        self, breaks: Any, closed: Literal["right", "left", "both", "neither"] = "right"
    ) -> Dict[str, Any]: ...


class TestFromBreaks(ConstructorTests):
    @pytest.fixture
    def constructor(self) -> Callable[..., IntervalIndex]: ...

    def get_kwargs_from_breaks(
        self, breaks: Any, closed: Literal["right", "left", "both", "neither"] = "right"
    ) -> Dict[str, Any]: ...


class TestFromTuples(ConstructorTests):
    @pytest.fixture
    def constructor(self) -> Callable[..., IntervalIndex]: ...

    def get_kwargs_from_breaks(
        self, breaks: Any, closed: Literal["right", "left", "both", "neither"] = "right"
    ) -> Dict[str, Any]: ...


class TestClassConstructors(ConstructorTests):
    @pytest.fixture
    def constructor(self) -> Callable[..., IntervalIndex]: ...

    def get_kwargs_from_breaks(
        self, breaks: Any, closed: Literal["right", "left", "both", "neither"] = "right"
    ) -> Dict[str, Any]: ...


@pytest.mark.parametrize("timezone", ["UTC", "US/Pacific", "GMT"])
def test_interval_index_subtype(
    timezone: str,
    inclusive_endpoints_fixture: Literal["right", "left", "both", "neither"],
) -> None: ...


def test_dtype_closed_mismatch() -> None: ...


@pytest.mark.parametrize(
    "dtype",
    [
        "Float64",
        pytest.param("float64[pyarrow]", marks=td.skip_if_no("pyarrow")),
    ],
)
def test_ea_dtype(dtype: str) -> None: ...