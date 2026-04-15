import contextlib
from datetime import date, datetime
from decimal import Decimal
import os
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)
import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p24
from pandas.errors import IndexingError
from pandas.core.dtypes.common import is_list_like
from pandas import (
    NA,
    Categorical,
    DataFrame,
    DatetimeIndex,
    Index,
    Interval,
    IntervalIndex,
    MultiIndex,
    NaT,
    Period,
    Series,
    StringDtype,
    Timedelta,
    Timestamp,
    array,
    concat,
    date_range,
    interval_range,
    period_range,
    timedelta_range,
)
import pandas._testing as tm
from pandas.tseries.offsets import BDay

_T = TypeVar("_T")

class TestSetitemDT64Values:
    def test_setitem_none_nan(self) -> None: ...
    def test_setitem_multiindex_empty_slice(self) -> None: ...
    def test_setitem_with_string_index(self) -> None: ...
    def test_setitem_tuple_with_datetimetz_values(self) -> None: ...
    @pytest.mark.parametrize("tz", ["US/Eastern", "UTC", "Asia/Tokyo"])
    def test_setitem_with_tz(
        self, tz: str, indexer_sli: Callable[[Series], Any]
    ) -> None: ...
    def test_setitem_with_tz_dst(
        self, indexer_sli: Callable[[Series], Any]
    ) -> None: ...
    def test_object_series_setitem_dt64array_exact_match(self) -> None: ...

class TestSetitemScalarIndexer:
    def test_setitem_negative_out_of_bounds(self) -> None: ...
    @pytest.mark.parametrize("indexer", [tm.loc, tm.at])
    @pytest.mark.parametrize("ser_index", [0, 1])
    def test_setitem_series_object_dtype(
        self, indexer: Callable[[Series], Any], ser_index: int
    ) -> None: ...
    @pytest.mark.parametrize("index, exp_value", [(0, 42), (1, np.nan)])
    def test_setitem_series(self, index: int, exp_value: Union[int, float]) -> None: ...

class TestSetitemSlices:
    def test_setitem_slice_float_raises(self, datetime_series: Series) -> None: ...
    def test_setitem_slice(self) -> None: ...
    def test_setitem_slice_integers(self) -> None: ...
    def test_setitem_slicestep(self) -> None: ...
    def test_setitem_multiindex_slice(
        self, indexer_sli: Callable[[Series], Any]
    ) -> None: ...

class TestSetitemBooleanMask:
    def test_setitem_mask_cast(self) -> None: ...
    def test_setitem_mask_align_and_promote(self) -> None: ...
    def test_setitem_mask_promote_strs(self) -> None: ...
    def test_setitem_mask_promote(self) -> None: ...
    def test_setitem_boolean(self, string_series: Series) -> None: ...
    def test_setitem_boolean_corner(self, datetime_series: Series) -> None: ...
    def test_setitem_boolean_different_order(self, string_series: Series) -> None: ...
    @pytest.mark.parametrize("func", [list, np.array, Series])
    def test_setitem_boolean_python_list(self, func: Callable) -> None: ...
    def test_setitem_boolean_nullable_int_types(
        self, any_numeric_ea_dtype: str
    ) -> None: ...
    def test_setitem_with_bool_mask_and_values_matching_n_trues_in_length(
        self,
    ) -> None: ...
    def test_setitem_nan_with_bool(self) -> None: ...
    def test_setitem_mask_smallint_upcast(self) -> None: ...
    def test_setitem_mask_smallint_no_upcast(self) -> None: ...

class TestSetitemViewCopySemantics:
    def test_setitem_invalidates_datetime_index_freq(self) -> None: ...
    def test_dt64tz_setitem_does_not_mutate_dti(self) -> None: ...

class TestSetitemCallable:
    def test_setitem_callable_key(self) -> None: ...
    def test_setitem_callable_other(self) -> None: ...

class TestSetitemWithExpansion:
    def test_setitem_empty_series(self) -> None: ...
    def test_setitem_empty_series_datetimeindex_preserves_freq(self) -> None: ...
    def test_setitem_empty_series_timestamp_preserves_dtype(self) -> None: ...
    @pytest.mark.parametrize(
        "td", [Timedelta("9 days"), Timedelta("9 days").to_timedelta64(), Timedelta("9 days").to_pytimedelta()]
    )
    def test_append_timedelta_does_not_cast(
        self, td: Union[Timedelta, np.timedelta64, Any], using_infer_string: bool, request: Any
    ) -> None: ...
    def test_setitem_with_expansion_type_promotion(self) -> None: ...
    def test_setitem_not_contained(self, string_series: Series) -> None: ...
    def test_setitem_keep_precision(self, any_numeric_ea_dtype: str) -> None: ...
    @pytest.mark.parametrize(
        "na, target_na, dtype, target_dtype, indexer, raises",
        [
            (NA, NA, "Int64", "Int64", 1, False),
            (NA, NA, "Int64", "Int64", 2, False),
            (NA, np.nan, "int64", "float64", 1, False),
            (NA, np.nan, "int64", "float64", 2, False),
            (NaT, NaT, "int64", "object", 1, True),
            (NaT, NaT, "int64", "object", 2, False),
            (np.nan, NA, "Int64", "Int64", 1, False),
            (np.nan, NA, "Int64", "Int64", 2, False),
            (np.nan, NA, "Float64", "Float64", 1, False),
            (np.nan, NA, "Float64", "Float64", 2, False),
            (np.nan, np.nan, "int64", "float64", 1, False),
            (np.nan, np.nan, "int64", "float64", 2, False),
        ],
    )
    def test_setitem_enlarge_with_na(
        self,
        na: Any,
        target_na: Any,
        dtype: str,
        target_dtype: str,
        indexer: int,
        raises: bool,
    ) -> None: ...
    def test_setitem_enlargement_object_none(
        self, nulls_fixture: Any, using_infer_string: bool
    ) -> None: ...

def test_setitem_scalar_into_readonly_backing_data() -> None: ...
def test_setitem_slice_into_readonly_backing_data() -> None: ...
def test_setitem_categorical_assigning_ops() -> None: ...
def test_setitem_nan_into_categorical() -> None: ...

class TestSetitemCasting:
    @pytest.mark.parametrize("unique", [True, False])
    @pytest.mark.parametrize("val", [3, 3.0, "3"])
    def test_setitem_non_bool_into_bool(
        self, val: Union[int, float, str], indexer_sli: Callable[[Series], Any], unique: bool
    ) -> None: ...
    def test_setitem_boolean_array_into_npbool(self) -> None: ...

class SetitemCastingEquivalents:
    @pytest.fixture
    def is_inplace(self, obj: Series, expected: Series) -> bool: ...
    def check_indexer(
        self,
        obj: Series,
        key: Any,
        expected: Series,
        val: Any,
        indexer: Callable[[Series], Any],
        is_inplace: bool,
    ) -> None: ...
    def _check_inplace(
        self, is_inplace: bool, orig: Series, arr: np.ndarray, obj: Series
    ) -> None: ...
    def test_int_key(
        self,
        obj: Series,
        key: int,
        expected: Series,
        raises: bool,
        val: Any,
        indexer_sli: Callable[[Series], Any],
        is_inplace: bool,
    ) -> None: ...
    def test_slice_key(
        self,
        obj: Series,
        key: slice,
        expected: Series,
        raises: bool,
        val: Any,
        indexer_sli: Callable[[Series], Any],
        is_inplace: bool,
    ) -> None: ...
    def test_mask_key(
        self,
        obj: Series,
        key: Any,
        expected: Series,
        raises: bool,
        val: Any,
        indexer_sli: Callable[[Series], Any],
    ) -> None: ...
    def test_series_where(
        self,
        obj: Series,
        key: Any,
        expected: Series,
        raises: bool,
        val: Any,
        is_inplace: bool,
    ) -> None: ...
    def test_index_where(
        self, obj: Series, key: Any, expected: Series, raises: bool, val: Any
    ) -> None: ...
    def test_index_putmask(
        self, obj: Series, key: Any, expected: Series, raises: bool, val: Any
    ) -> None: ...

@pytest.mark.parametrize(
    "obj,expected,key,raises",
    [
        pytest.param(
            Series(interval_range(1, 5)),
            Series(
                [Interval(1, 2), np.nan, Interval(3, 4), Interval(4, 5)],
                dtype="interval[float64]",
            ),
            1,
            True,
            id="interval_int_na_value",
        ),
        pytest.param(
            Series([2, 3, 4, 5, 6, 7, 8, 9, 10]),
            Series([np.nan, 3, np.nan, 5, np.nan, 7, np.nan, 9, np.nan]),
            slice(None, None, 2),
            False,
            id="int_series_slice_key_step",
        ),
        pytest.param(
            Series([True, True, False, False]),
            Series([np.nan, True, np.nan, False], dtype=object),
            slice(None, None, 2),
            True,
            id="bool_series_slice_key_step",
        ),
        pytest.param(
            Series(np.arange(10)),
            Series([np.nan, np.nan, np.nan, np.nan, np.nan, 5, 6, 7, 8, 9]),
            slice(None, 5),
            False,
            id="int_series_slice_key",
        ),
        pytest.param(
            Series([1, 2, 3]),
            Series([np.nan, 2, 3]),
            0,
            False,
            id="int_series_int_key",
        ),
        pytest.param(
            Series([False]),
            Series([np.nan], dtype=object),
            0,
            True,
            id="bool_series_int_key_change_all",
        ),
        pytest.param(
            Series([False, True]),
            Series([np.nan, True], dtype=object),
            0,
            True,
            id="bool_series_int_key",
        ),
    ],
)
class TestSetitemCastingEquivalents(SetitemCastingEquivalents):
    @pytest.fixture(params=[np.nan, np.float64("NaN"), None, NA])
    def val(self, request: Any) -> Any: ...

class TestSetitemTimedelta64IntoNumeric(SetitemCastingEquivalents):
    @pytest.fixture
    def val(self) -> np.timedelta64: ...
    @pytest.fixture(params=[complex, int, float])
    def dtype(self, request: Any) -> Type: ...
    @pytest.fixture
    def obj(self, dtype: Type) -> Series: ...
    @pytest.fixture
    def expected(self, dtype: Type) -> Series: ...
    @pytest.fixture
    def key(self) -> int: ...
    @pytest.fixture
    def raises(self) -> bool: ...

class TestSetitemDT64IntoInt(SetitemCastingEquivalents):
    @pytest.fixture(params=["M8[ns]", "m8[ns]"])
    def dtype(self, request: Any) -> str: ...
    @pytest.fixture
    def scalar(self, dtype: str) -> Union[np.datetime64, np.timedelta64]: ...
    @pytest.fixture
    def expected(self, scalar: Union[np.datetime64, np.timedelta64]) -> Series: ...
    @pytest.fixture
    def obj(self) -> Series: ...
    @pytest.fixture
    def key(self) -> slice: ...
    @pytest.fixture(params=[None, list, np.array])
    def val(
        self, scalar: Union[np.datetime64, np.timedelta64], request: Any
    ) -> Union[Union[np.datetime64, np.timedelta64], List, np.ndarray]: ...
    @pytest.fixture
    def raises(self) -> bool: ...

class TestSetitemNAPeriodDtype(SetitemCastingEquivalents):
    @pytest.fixture
    def expected(self, key: Union[int, slice]) -> Series: ...
    @pytest.fixture
    def obj(self) -> Series: ...
    @pytest.fixture(params=[3, slice(3, 5)])
    def key(self, request: Any) -> Union[int, slice]: ...
    @pytest.fixture(params=[None, np.nan])
    def val(self, request: Any) -> Union[None, float]: ...
    @pytest.fixture
    def raises(self) -> bool: ...

class TestSetitemNADatetimeLikeDtype(SetitemCastingEquivalents):
    @pytest.fixture(
        params=["m8[ns]", "M8[ns]", "datetime64[ns, UTC]", "datetime64[ns, US/Central]"]
    )
    def dtype(self, request: Any) -> str: ...
    @pytest.fixture
    def obj(self, dtype: str) -> Series: ...
    @pytest.fixture(
        params=[None, np.nan, NaT, np.timedelta64("NaT", "ns"), np.datetime64("NaT", "ns")]
    )
    def val(self, request: Any) -> Any: ...
    @pytest.fixture
    def is_inplace(self, val: Any, obj: Series) -> bool: ...
    @pytest.fixture
    def expected(self, obj: Series, val: Any, is_inplace: bool) -> Series: ...
    @pytest.fixture
    def key(self) -> int: ...
    @pytest.fixture
    def raises(self, is_inplace: bool) -> bool: ...

class TestSetitemMismatchedTZCastsToObject(SetitemCastingEquivalents):
    @pytest.fixture
    def obj(self) -> Series: ...
    @pytest.fixture
    def val(self) -> Timestamp: ...
    @pytest.fixture
    def key(self) -> int: ...
    @pytest.fixture
    def expected(self, obj: Series, val: Timestamp) -> Series: ...
    @pytest.fixture
    def raises(self) -> bool: ...

@pytest.mark.parametrize(
    "obj,expected",
    [
        (Series([1, 2, 3]), Series([np.nan, 2, 3])),
        (Series([1.0, 2.0, 3.0]), Series([np.nan, 2.0, 3.0])),
        (
            Series([datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)]),
            Series([NaT, datetime(2000, 1, 2), datetime(2000, 1, 3)]),
        ),
        (Series(["foo", "bar", "baz"]), Series([None, "bar", "baz"])),
    ],
)
class TestSeriesNoneCoercion(SetitemCastingEquivalents):
    @pytest.fixture
    def key(self) -> int: ...
    @pytest.fixture
    def val(self) -> None: ...
    @pytest.fixture
    def raises(self) -> bool: ...

class TestSetitemFloatIntervalWithIntIntervalValues(SetitemCastingEquivalents):
    def test_setitem_example(self) -> None: ...
    @pytest.fixture
    def obj(self) -> Series: ...
    @pytest.fixture
    def val(self) -> Interval: ...
    @pytest.fixture
    def key(self) -> int: ...
    @pytest.fixture
    def expected(self, obj: Series, val: Interval) -> Series: ...
    @pytest.fixture
    def raises(self) -> bool: ...

class TestSetitemRangeIntoIntegerSeries(SetitemCastingEquivalents):
    @pytest.fixture
    def obj(self, any_int_numpy_dtype: str) -> Series: ...
    @pytest.fixture
    def val(self) -> range: ...
    @pytest.fixture
    def key(self) -> slice: ...
    @pytest.fixture
    def expected(self, any_int_numpy_dtype: str) -> Series: ...
    @pytest.fixture
    def raises(self) -> bool: ...

@pytest.mark.parametrize(
    "val, raises",
    [
        (np.array([2.0, 3.