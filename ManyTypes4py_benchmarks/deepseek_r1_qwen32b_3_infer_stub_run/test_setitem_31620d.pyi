from datetime import date, datetime, timedelta
from decimal import Decimal
import numpy as np
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
from pandas._testing import tm
from pandas.compat.numpy import np_version_gte1p24
from pandas.errors import IndexingError
from pandas.tseries.offsets import BDay
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Union,
)
import pytest

class TestSetitemDT64Values:
    def test_setitem_none_nan(self) -> None:
        ...
    
    def test_setitem_multiindex_empty_slice(self) -> None:
        ...
    
    def test_setitem_with_string_index(self) -> None:
        ...
    
    def test_setitem_tuple_with_datetimetz_values(self) -> None:
        ...
    
    @pytest.mark.parametrize('tz', ['US/Eastern', 'UTC', 'Asia/Tokyo'])
    def test_setitem_with_tz(self, tz: str, indexer_sli: Callable[[Series], Any]) -> None:
        ...
    
    def test_setitem_with_tz_dst(self, indexer_sli: Callable[[Series], Any]) -> None:
        ...
    
    def test_object_series_setitem_dt64array_exact_match(self) -> None:
        ...

class TestSetitemScalarIndexer:
    def test_setitem_negative_out_of_bounds(self) -> None:
        ...
    
    @pytest.mark.parametrize('indexer', [tm.loc, tm.at])
    @pytest.mark.parametrize('ser_index', [0, 1])
    def test_setitem_series_object_dtype(self, indexer: Callable, ser_index: int) -> None:
        ...
    
    @pytest.mark.parametrize('index, exp_value', [(0, 42), (1, np.nan)])
    def test_setitem_series(self, index: int, exp_value: Union[int, float]) -> None:
        ...

class TestSetitemSlices:
    def test_setitem_slice_float_raises(self, datetime_series: Series) -> None:
        ...
    
    def test_setitem_slice(self) -> None:
        ...
    
    def test_setitem_slice_integers(self) -> None:
        ...
    
    def test_setitem_slicestep(self) -> None:
        ...
    
    def test_setitem_multiindex_slice(self, indexer_sli: Callable[[Series], Any]) -> None:
        ...

class TestSetitemBooleanMask:
    def test_setitem_mask_cast(self) -> None:
        ...
    
    def test_setitem_mask_align_and_promote(self) -> None:
        ...
    
    def test_setitem_mask_promote_strs(self) -> None:
        ...
    
    def test_setitem_mask_promote(self) -> None:
        ...
    
    def test_setitem_boolean(self, string_series: Series) -> None:
        ...
    
    def test_setitem_boolean_corner(self, datetime_series: Series) -> None:
        ...
    
    def test_setitem_boolean_different_order(self, string_series: Series) -> None:
        ...
    
    @pytest.mark.parametrize('func', [list, np.array, Series])
    def test_setitem_boolean_python_list(self, func: Callable) -> None:
        ...
    
    def test_setitem_boolean_nullable_int_types(self, any_numeric_ea_dtype: Any) -> None:
        ...
    
    def test_setitem_with_bool_mask_and_values_matching_n_trues_in_length(self) -> None:
        ...
    
    def test_setitem_nan_with_bool(self) -> None:
        ...
    
    def test_setitem_mask_smallint_upcast(self) -> None:
        ...
    
    def test_setitem_mask_smallint_no_upcast(self) -> None:
        ...

class TestSetitemViewCopySemantics:
    def test_setitem_invalidates_datetime_index_freq(self) -> None:
        ...
    
    def test_dt64tz_setitem_does_not_mutate_dti(self) -> None:
        ...

class TestSetitemCallable:
    def test_setitem_callable_key(self) -> None:
        ...
    
    def test_setitem_callable_other(self) -> None:
        ...

class TestSetitemWithExpansion:
    def test_setitem_empty_series(self) -> None:
        ...
    
    def test_setitem_empty_series_datetimeindex_preserves_freq(self) -> None:
        ...
    
    def test_setitem_empty_series_timestamp_preserves_dtype(self) -> None:
        ...
    
    @pytest.mark.parametrize('td', [Timedelta('9 days'), Timedelta('9 days').to_timedelta64(), Timedelta('9 days').to_pytimedelta()])
    def test_append_timedelta_does_not_cast(self, td: Timedelta, using_infer_string: bool, request: pytest.FixtureRequest) -> None:
        ...
    
    def test_setitem_with_expansion_type_promotion(self) -> None:
        ...
    
    def test_setitem_not_contained(self, string_series: Series) -> None:
        ...
    
    def test_setitem_keep_precision(self, any_numeric_ea_dtype: Any) -> None:
        ...
    
    @pytest.mark.parametrize('na, target_na, dtype, target_dtype, indexer, raises', [
        (NA, NA, 'Int64', 'Int64', 1, False),
        (NA, NA, 'Int64', 'Int64', 2, False),
        (NA, np.nan, 'int64', 'float64', 1, False),
        (NA, np.nan, 'int64', 'float64', 2, False),
        (NaT, NaT, 'int64', 'object', 1, True),
        (NaT, NaT, 'int64', 'object', 2, False),
        (np.nan, NA, 'Int64', 'Int64', 1, False),
        (np.nan, NA, 'Int64', 'Int64', 2, False),
        (np.nan, NA, 'Float64', 'Float64', 1, False),
        (np.nan, NA, 'Float64', 'Float64', 2, False),
        (np.nan, np.nan, 'int64', 'float64', 1, False),
        (np.nan, np.nan, 'int64', 'float64', 2, False)
    ])
    def test_setitem_enlarge_with_na(self, na: Any, target_na: Any, dtype: str, target_dtype: str, indexer: int, raises: bool) -> None:
        ...
    
    def test_setitem_enlargement_object_none(self, nulls_fixture: Any, using_infer_string: bool) -> None:
        ...

def test_setitem_scalar_into_readonly_backing_data() -> None:
    ...

def test_setitem_slice_into_readonly_backing_data() -> None:
    ...

def test_setitem_categorical_assigning_ops() -> None:
    ...

def test_setitem_nan_into_categorical() -> None:
    ...

class TestSetitemCasting:
    @pytest.mark.parametrize('unique', [True, False])
    @pytest.mark.parametrize('val', [3, 3.0, '3'], ids=type)
    def test_setitem_non_bool_into_bool(self, val: Union[int, float, str], indexer_sli: Callable, unique: bool) -> None:
        ...
    
    def test_setitem_boolean_array_into_npbool(self) -> None:
        ...

class SetitemCastingEquivalents:
    def test_int_key(self, obj: Any, key: int, expected: Any, val: Any, indexer_sli: Callable, is_inplace: bool) -> None:
        ...
    
    def test_slice_key(self, obj: Any, key: slice, expected: Any, val: Any, indexer_sli: Callable, is_inplace: bool) -> None:
        ...
    
    def test_mask_key(self, obj: Any, key: Any, expected: Any, val: Any, indexer_sli: Callable) -> None:
        ...
    
    def test_series_where(self, obj: Any, key: Any, expected: Any, val: Any, is_inplace: bool) -> None:
        ...
    
    def test_index_where(self, obj: Any, key: Any, expected: Any, val: Any) -> None:
        ...
    
    def test_index_putmask(self, obj: Any, key: Any, expected: Any, val: Any) -> None:
        ...

@pytest.mark.parametrize('obj,expected,key,raises', [
    (Series(interval_range(1, 5)), Series([Interval(1, 2), np.nan, Interval(3, 4), Interval(4, 5)], dtype='interval[float64]'), 1, True),
    (Series([2, 3, 4, 5, 6, 7, 8, 9, 10]), Series([np.nan, 3, np.nan, 5, np.nan, 7, np.nan, 9, np.nan]), slice(None, None, 2), False),
    (Series([True, True, False, False]), Series([np.nan, True, np.nan, False], dtype=object), slice(None, None, 2), True),
    (Series(np.arange(10)), Series([np.nan, np.nan, np.nan, np.nan, np.nan, 5, 6, 7, 8, 9]), slice(None, 5), False),
    (Series([1, 2, 3]), Series([np.nan, 2, 3]), 0, False),
    (Series([False]), Series([np.nan], dtype=object), 0, True),
    (Series([False, True]), Series([np.nan, True], dtype=object), 0, True)
])
class TestSetitemCastingEquivalents(SetitemCastingEquivalents):
    @pytest.fixture(params=[np.nan, np.float64('NaN'), None, NA])
    def val(self) -> Any:
        ...

class TestSetitemTimedelta64IntoNumeric(SetitemCastingEquivalents):
    @pytest.fixture
    def val(self) -> np.timedelta64:
        ...
    
    @pytest.fixture(params=[complex, int, float])
    def dtype(self, request) -> type:
        ...
    
    @pytest.fixture
    def obj(self, dtype: type) -> Series:
        ...
    
    @pytest.fixture
    def expected(self, dtype: type) -> Series:
        ...
    
    @pytest.fixture
    def key(self) -> int:
        ...
    
    @pytest.fixture
    def raises(self) -> bool:
        ...

class TestSetitemDT64IntoInt(SetitemCastingEquivalents):
    @pytest.fixture(params=['M8[ns]', 'm8[ns]'])
    def dtype(self, request) -> str:
        ...
    
    @pytest.fixture
    def scalar(self, dtype: str) -> np.datetime64:
        ...
    
    @pytest.fixture
    def expected(self, scalar: np.datetime64) -> Series:
        ...
    
    @pytest.fixture
    def obj(self) -> Series:
        ...
    
    @pytest.fixture
    def key(self) -> slice:
        ...
    
    @pytest.fixture(params=[None, list, np.array])
    def val(self, scalar: np.datetime64, request) -> Any:
        ...
    
    @pytest.fixture
    def raises(self) -> bool:
        ...

class TestSetitemNAPeriodDtype(SetitemCastingEquivalents):
    @pytest.fixture
    def expected(self, key: Any) -> Series:
        ...
    
    @pytest.fixture
    def obj(self) -> Series:
        ...
    
    @pytest.fixture(params=[3, slice(3, 5)])
    def key(self, request) -> Any:
        ...
    
    @pytest.fixture(params=[None, np.nan])
    def val(self, request) -> Any:
        ...
    
    @pytest.fixture
    def raises(self) -> bool:
        ...

class TestSetitemNADatetimeLikeDtype(SetitemCastingEquivalents):
    @pytest.fixture(params=['m8[ns]', 'M8[ns]', 'datetime64[ns, UTC]', 'datetime64[ns, US/Central]'])
    def dtype(self, request) -> str:
        ...
    
    @pytest.fixture
    def obj(self, dtype: str) -> Series:
        ...
    
    @pytest.fixture(params=[None, np.nan, NaT, np.timedelta64('NaT', 'ns'), np.datetime64('NaT', 'ns')])
    def val(self, request) -> Any:
        ...
    
    @pytest.fixture
    def is_inplace(self, val: Any, obj: Series) -> bool:
        ...
    
    @pytest.fixture
    def expected(self, obj: Series, val: Any, is_inplace: bool) -> Series:
        ...
    
    @pytest.fixture
    def key(self) -> int:
        ...
    
    @pytest.fixture
    def raises(self, is_inplace: bool) -> bool:
        ...

class TestSetitemMismatchedTZCastsToObject(SetitemCastingEquivalents):
    @pytest.fixture
    def obj(self) -> Series:
        ...
    
    @pytest.fixture
    def val(self) -> Timestamp:
        ...
    
    @pytest.fixture
    def key(self) -> int:
        ...
    
    @pytest.fixture
    def expected(self, obj: Series, val: Timestamp) -> Series:
        ...
    
    @pytest.fixture
    def raises(self) -> bool:
        ...

@pytest.mark.parametrize('obj,expected', [
    (Series([1, 2, 3]), Series([np.nan, 2, 3])),
    (Series([1.0, 2.0, 3.0]), Series([np.nan, 2.0, 3.0])),
    (Series([datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)]), Series([NaT, datetime(2000, 1, 2), datetime(2000, 1, 3)])),
    (Series(['foo', 'bar', 'baz']), Series([None, 'bar', 'baz']))
])
class TestSeriesNoneCoercion(SetitemCastingEquivalents):
    @pytest.fixture
    def key(self) -> int:
        ...
    
    @pytest.fixture
    def val(self) -> None:
        ...
    
    @pytest.fixture
    def raises(self) -> bool:
        ...

class TestSetitemFloatIntervalWithIntIntervalValues(SetitemCastingEquivalents):
    def test_setitem_example(self) -> None:
        ...
    
    @pytest.fixture
    def obj(self) -> Series:
        ...
    
    @pytest.fixture
    def val(self) -> Interval:
        ...
    
    @pytest.fixture
    def key(self) -> int:
        ...
    
    @pytest.fixture
    def expected(self, obj: Series, val: Interval) -> Series:
        ...
    
    @pytest.fixture
    def raises(self) -> bool:
        ...

class TestSetitemRangeIntoIntegerSeries(SetitemCastingEquivalents):
    @pytest.fixture
    def obj(self, any_int_numpy_dtype: Any) -> Series:
        ...
    
    @pytest.fixture
    def val(self) -> range:
        ...
    
    @pytest.fixture
    def key(self) -> slice:
        ...
    
    @pytest.fixture
    def expected(self, any_int_numpy_dtype: Any) -> Series:
        ...
    
    @pytest.fixture
    def raises(self) -> bool:
        ...

@pytest.mark.parametrize('val, raises', [
    (np.array([2.0, 3.0]), False),
    (np.array([2.5, 3.5]), True),
    (np.array([2 ** 65, 2 ** 65 + 1], dtype=np.float64), True)
])
class TestSetitemFloatNDarrayIntoIntegerSeries(SetitemCastingEquivalents):
    @pytest.fixture
    def obj(self) -> Series:
        ...
    
    @pytest.fixture
    def key(self) -> slice:
        ...
    
    @pytest.fixture
    def expected(self, val: np.ndarray) -> Series:
        ...

@pytest.mark.parametrize('val', [512, np.int16(512)])
class TestSetitemIntoIntegerSeriesNeedsUpcast(SetitemCastingEquivalents):
    @pytest.fixture
    def obj(self) -> Series:
        ...
    
    @pytest.fixture
    def key(self) -> int:
        ...
    
    @pytest.fixture
    def expected(self) -> Series:
        ...
    
    @pytest.fixture
    def raises(self) -> bool:
        ...

@pytest.mark.parametrize('val', [2 ** 33 + 1.0, 2 ** 33 + 1.1, 2 ** 62])
class TestSmallIntegerSetitemUpcast(SetitemCastingEquivalents):
    @pytest.fixture
    def obj(self) -> Series:
        ...
    
    @pytest.fixture
    def key(self) -> int:
        ...
    
    @pytest.fixture
    def expected(self, val: Union[int, float]) -> Series:
        ...
    
    @pytest.fixture
    def raises(self) -> bool:
        ...

class CoercionTest(SetitemCastingEquivalents):
    @pytest.fixture
    def key(self) -> int:
        ...
    
    @pytest.fixture
    def expected(self, obj: Any, key: int, val: Any, exp_dtype: Any) -> Series:
        ...

@pytest.mark.parametrize('val,exp_dtype,raises', [
    (np.int32(1), np.int8, None),
    (np.int16(2 ** 9), np.int16, True)
])
class TestCoercionInt8(CoercionTest):
    @pytest.fixture
    def obj(self) -> Series:
        ...

@pytest.mark.parametrize('val', [1, 1.1, 1 + 1j, True])
@pytest.mark.parametrize('exp_dtype', [object])
class TestCoercionObject(CoercionTest):
    @pytest.fixture
    def obj(self) -> Series:
        ...
    
    @pytest.fixture
    def raises(self) -> bool:
        ...

@pytest.mark.parametrize('val,exp_dtype,raises', [
    (1, object, True),
    ('e', StringDtype(na_value=np.nan), False)
])
class TestCoercionString(CoercionTest):
    @pytest.fixture
    def obj(self) -> Series:
        ...

@pytest.mark.parametrize('val,exp_dtype,raises', [
    (1, np.complex128, False),
    (1.1, np.complex128, False),
    (1 + 1j, np.complex128, False),
    (True, object, True)
])
class TestCoercionComplex(CoercionTest):
    @pytest.fixture
    def obj(self) -> Series:
        ...

@pytest.mark.parametrize('val,exp_dtype,raises', [
    (1, object, True),
    ('3', object, True),
    (3, object, True),
    (1.1, object, True),
    (1 + 1j, object, True),
    (True, bool, False)
])
class TestCoercionBool(CoercionTest):
    @pytest.fixture
    def obj(self) -> Series:
        ...

@pytest.mark.parametrize('val,exp_dtype,raises', [
    (1, np.int64, False),
    (1.1, np.float64, True),
    (1 + 1j, np.complex128, True),
    (True, object, True)
])
class TestCoercionInt64(CoercionTest):
    @pytest.fixture
    def obj(self) -> Series:
        ...

@pytest.mark.parametrize('val,exp_dtype,raises', [
    (1, np.float64, False),
    (1.1, np.float64, False),
    (1 + 1j, np.complex128, True),
    (True, object, True)
])
class TestCoercionFloat64(CoercionTest):
    @pytest.fixture
    def obj(self) -> Series:
        ...

@pytest.mark.parametrize('val,exp_dtype,raises', [
    (1, np.float32, False),
    pytest.param(1.1, np.float32, False, marks=pytest.mark.xfail(not np_version_gte1p24 or (np_version_gte1p24 and os.environ.get('NPY_PROMOTION_STATE', 'weak') != 'weak'), reason='np.float32(1.1) ends up as 1.100000023841858, so np_can_hold_element raises and we cast to float64')),
    (1 + 1j, np.complex128, True),
    (True, object, True),
    (np.uint8(2), np.float32, False),
    (np.uint32(2), np.float32, False),
    (np.uint32(np.iinfo(np.uint32).max), np.float64, True),
    (np.uint64(2), np.float32, False),
    (np.int64(2), np.float32, False)
])
class TestCoercionFloat32(CoercionTest):
    @pytest.fixture
    def obj(self) -> Series:
        ...

@pytest.mark.parametrize('exp_dtype', ['M8[ms]', 'M8[ms, UTC]', 'm8[ms]'])
class TestCoercionDatetime64HigherReso(CoercionTest):
    @pytest.fixture
    def obj(self, exp_dtype: str) -> Series:
        ...
    
    @pytest.fixture
    def val(self, exp_dtype: str) -> Union[np.datetime64, np.timedelta64]:
        ...
    
    @pytest.fixture
    def raises(self) -> bool:
        ...

@pytest.mark.parametrize('val,exp_dtype,raises', [
    (Timestamp('2012-01-01'), 'datetime64[ns]', False),
    (1, object, True),
    ('x', object, True)
])
class TestCoercionDatetime64(CoercionTest):
    @pytest.fixture
    def obj(self) -> Series:
        ...

@pytest.mark.parametrize('val,exp_dtype,raises', [
    (Timestamp('2012-01-01', tz='US/Eastern'), 'datetime64[ns, US/Eastern]', False),
    (Timestamp('2012-01-01', tz='US/Pacific'), 'datetime64[ns, US/Eastern]', False),
    (Timestamp('2012-01-01'), object, True),
    (1, object, True)
])
class TestCoercionDatetime64TZ(CoercionTest):
    @pytest.fixture
    def obj(self) -> Series:
        ...

@pytest.mark.parametrize('val,exp_dtype,raises', [
    (Timedelta('12 day'), 'timedelta64[ns]', False),
    (1, object, True),
    ('x', object, True)
])
class TestCoercionTimedelta64(CoercionTest):
    @pytest.fixture
    def obj(self) -> Series:
        ...

@pytest.mark.parametrize('val', ['foo', Period('2016', freq='Y'), Interval(1, 2, closed='both')])
@pytest.mark.parametrize('exp_dtype', [object])
class TestPeriodIntervalCoercion(CoercionTest):
    @pytest.fixture(params=[period_range('2016-01-01', periods=3, freq='D'), interval_range(1, 5)])
    def obj(self, request) -> Series:
        ...

def test_20643() -> None:
    ...

def test_20643_comment() -> None:
    ...

def test_15413() -> None:
    ...

def test_32878_int_itemsize() -> None:
    ...

def test_32878_complex_itemsize() -> None:
    ...

def test_37692(indexer_al: Callable) -> None:
    ...

def test_setitem_bool_int_float_consistency(indexer_sli: Callable) -> None:
    ...

def test_setitem_positional_with_casting() -> None:
    ...

def test_setitem_positional_float_into_int_coerces() -> None:
    ...

def test_setitem_int_not_positional() -> None:
    ...

def test_setitem_with_bool_indexer() -> None:
    ...

@pytest.mark.parametrize('size', range(2, 6))
@pytest.mark.parametrize('mask', [[True, False, False, False, False], [True, False], [False]])
@pytest.mark.parametrize('item', [2.0, np.nan, np.finfo(float).max, np.finfo(float).min])
@pytest.mark.parametrize('box', [np.array, list, tuple])
def test_setitem_bool_indexer_dont_broadcast_length1_values(size: int, mask: List[bool], item: Union[float, np.floating], box: Callable) -> None:
    ...

def test_setitem_empty_mask_dont_upcast_dt64() -> None:
    ...