```python
import datetime
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
from typing_extensions import Literal

import numpy as np
import pandas as pd
from pandas import (
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
    Timedelta,
    Timestamp,
)
from pandas._testing import TestCase
from pandas.core.dtypes.base import ExtensionDtype
from pandas.errors import IndexingError
from pandas.tseries.offsets import BDay

class TestSetitemDT64Values:
    def test_setitem_none_nan(self) -> None: ...
    def test_setitem_multiindex_empty_slice(self) -> None: ...
    def test_setitem_with_string_index(self) -> None: ...
    def test_setitem_tuple_with_datetimetz_values(self) -> None: ...
    def test_setitem_with_tz(self, tz: str, indexer_sli: Any) -> None: ...
    def test_setitem_with_tz_dst(self, indexer_sli: Any) -> None: ...
    def test_object_series_setitem_dt64array_exact_match(self) -> None: ...

class TestSetitemScalarIndexer:
    def test_setitem_negative_out_of_bounds(self) -> None: ...
    def test_setitem_series_object_dtype(self, indexer: Any, ser_index: int) -> None: ...
    def test_setitem_series(self, index: int, exp_value: Any) -> None: ...

class TestSetitemSlices:
    def test_setitem_slice_float_raises(self, datetime_series: Series) -> None: ...
    def test_setitem_slice(self) -> None: ...
    def test_setitem_slice_integers(self) -> None: ...
    def test_setitem_slicestep(self) -> None: ...
    def test_setitem_multiindex_slice(self, indexer_sli: Any) -> None: ...

class TestSetitemBooleanMask:
    def test_setitem_mask_cast(self) -> None: ...
    def test_setitem_mask_align_and_promote(self) -> None: ...
    def test_setitem_mask_promote_strs(self) -> None: ...
    def test_setitem_mask_promote(self) -> None: ...
    def test_setitem_boolean(self, string_series: Series) -> None: ...
    def test_setitem_boolean_corner(self, datetime_series: Series) -> None: ...
    def test_setitem_boolean_different_order(self, string_series: Series) -> None: ...
    def test_setitem_boolean_python_list(self, func: Callable) -> None: ...
    def test_setitem_boolean_nullable_int_types(self, any_numeric_ea_dtype: ExtensionDtype) -> None: ...
    def test_setitem_with_bool_mask_and_values_matching_n_trues_in_length(self) -> None: ...
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
    def test_append_timedelta_does_not_cast(self, td: Any, using_infer_string: bool, request: Any) -> None: ...
    def test_setitem_with_expansion_type_promotion(self) -> None: ...
    def test_setitem_not_contained(self, string_series: Series) -> None: ...
    def test_setitem_keep_precision(self, any_numeric_ea_dtype: ExtensionDtype) -> None: ...
    def test_setitem_enlarge_with_na(self, na: Any, target_na: Any, dtype: str, target_dtype: str, indexer: Any, raises: bool) -> None: ...
    def test_setitem_enlargement_object_none(self, nulls_fixture: Any, using_infer_string: bool) -> None: ...

def test_setitem_scalar_into_readonly_backing_data() -> None: ...
def test_setitem_slice_into_readonly_backing_data() -> None: ...
def test_setitem_categorical_assigning_ops() -> None: ...
def test_setitem_nan_into_categorical() -> None: ...

class TestSetitemCasting:
    def test_setitem_non_bool_into_bool(self, val: Any, indexer_sli: Any, unique: bool) -> None: ...
    def test_setitem_boolean_array_into_npbool(self) -> None: ...

class SetitemCastingEquivalents:
    def check_indexer(self, obj: Any, key: Any, expected: Any, val: Any, indexer: Any, is_inplace: bool) -> None: ...
    def _check_inplace(self, is_inplace: bool, orig: Any, arr: Any, obj: Any) -> None: ...
    def test_int_key(self, obj: Any, key: Any, expected: Any, raises: bool, val: Any, indexer_sli: Any, is_inplace: bool) -> None: ...
    def test_slice_key(self, obj: Any, key: Any, expected: Any, raises: bool, val: Any, indexer_sli: Any, is_inplace: bool) -> None: ...
    def test_mask_key(self, obj: Any, key: Any, expected: Any, raises: bool, val: Any, indexer_sli: Any) -> None: ...
    def test_series_where(self, obj: Any, key: Any, expected: Any, raises: bool, val: Any, is_inplace: bool) -> None: ...
    def test_index_where(self, obj: Any, key: Any, expected: Any, raises: bool, val: Any) -> None: ...
    def test_index_putmask(self, obj: Any, key: Any, expected: Any, raises: bool, val: Any) -> None: ...

class TestSetitemCastingEquivalents(SetitemCastingEquivalents):
    val: Any
    def test_slice_key(self, obj: Any, key: Any, expected: Any, raises: bool, val: Any, indexer_sli: Any, is_inplace: bool) -> None: ...

class TestSetitemTimedelta64IntoNumeric(SetitemCastingEquivalents):
    val: np.timedelta64
    dtype: Any
    obj: Series
    expected: Series
    key: int
    raises: bool

class TestSetitemDT64IntoInt(SetitemCastingEquivalents):
    dtype: str
    scalar: Union[np.datetime64, np.timedelta64]
    expected: Series
    obj: Series
    key: slice
    val: Any
    raises: bool

class TestSetitemNAPeriodDtype(SetitemCastingEquivalents):
    expected: Series
    obj: Series
    key: Union[int, slice]
    val: Optional[float]
    raises: bool

class TestSetitemNADatetimeLikeDtype(SetitemCastingEquivalents):
    dtype: str
    obj: Series
    val: Any
    is_inplace: bool
    expected: Series
    key: int
    raises: bool

class TestSetitemMismatchedTZCastsToObject(SetitemCastingEquivalents):
    obj: Series
    val: Timestamp
    key: int
    expected: Series
    raises: bool

class TestSeriesNoneCoercion(SetitemCastingEquivalents):
    key: int
    val: None
    raises: bool

class TestSetitemFloatIntervalWithIntIntervalValues(SetitemCastingEquivalents):
    def test_setitem_example(self) -> None: ...
    obj: Series
    val: Interval
    key: int
    expected: Series
    raises: bool

class TestSetitemRangeIntoIntegerSeries(SetitemCastingEquivalents):
    obj: Series
    val: range
    key: slice
    expected: Series
    raises: bool

class TestSetitemFloatNDarrayIntoIntegerSeries(SetitemCastingEquivalents):
    obj: Series
    key: slice
    expected: Series

class TestSetitemIntoIntegerSeriesNeedsUpcast(SetitemCastingEquivalents):
    obj: Series
    key: int
    expected: Series
    raises: bool

class TestSmallIntegerSetitemUpcast(SetitemCastingEquivalents):
    obj: Series
    key: int
    expected: Series
    raises: bool

class CoercionTest(SetitemCastingEquivalents):
    key: int
    expected: Series

class TestCoercionInt8(CoercionTest):
    obj: Series

class TestCoercionObject(CoercionTest):
    obj: Series
    raises: bool

class TestCoercionString(CoercionTest):
    obj: Series

class TestCoercionComplex(CoercionTest):
    obj: Series

class TestCoercionBool(CoercionTest):
    obj: Series

class TestCoercionInt64(CoercionTest):
    obj: Series

class TestCoercionFloat64(CoercionTest):
    obj: Series

class TestCoercionFloat32(CoercionTest):
    obj: Series
    def test_slice_key(self, obj: Any, key: Any, expected: Any, raises: bool, val: Any, indexer_sli: Any, is_inplace: bool) -> None: ...

class TestCoercionDatetime64HigherReso(CoercionTest):
    obj: Series
    val: Union[Timestamp, Timedelta]
    raises: bool

class TestCoercionDatetime64(CoercionTest):
    obj: Series
    raises: bool

class TestCoercionDatetime64TZ(CoercionTest):
    obj: Series
    raises: bool

class TestCoercionTimedelta64(CoercionTest):
    obj: Series
    raises: bool

class TestPeriodIntervalCoercion(CoercionTest):
    obj: Series
    raises: bool

def test_20643() -> None: ...
def test_20643_comment() -> None: ...
def test_15413() -> None: ...
def test_32878_int_itemsize() -> None: ...
def test_32878_complex_itemsize() -> None: ...
def test_37692(indexer_al: Any) -> None: ...
def test_setitem_bool_int_float_consistency(indexer_sli: Any) -> None: ...
def test_setitem_positional_with_casting() -> None: ...
def test_setitem_positional_float_into_int_coerces() -> None: ...
def test_setitem_int_not_positional() -> None: ...
def test_setitem_with_bool_indexer() -> None: ...
def test_setitem_bool_indexer_dont_broadcast_length1_values(size: int, mask: List[bool], item: float, box: Callable) -> None: ...
def test_setitem_empty_mask_dont_upcast_dt64() -> None: ...
```