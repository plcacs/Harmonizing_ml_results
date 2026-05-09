import contextlib
from datetime import date, datetime
from decimal import Decimal
import os
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

class TestSetitemDT64Values:
    def test_setitem_none_nan(self, series: Series['datetime64[ns, UTC]']) -> None:
        ...

    def test_setitem_multiindex_empty_slice(self, result: Series[int], expected: Series[int]) -> None:
        ...

    def test_setitem_with_string_index(self, ser: Series[object], date: date) -> None:
        ...

    def test_setitem_tuple_with_datetimetz_values(self, arr: DatetimeIndex, index: list[tuple[int, int]], result: Series['datetime64[ns]'], expected: Series['datetime64[ns]']) -> None:
        ...

    @pytest.mark.parametrize('tz', ['US/Eastern', 'UTC', 'Asia/Tokyo'])
    def test_setitem_with_tz(self, tz: str, indexer_sli: Callable[[Series], Any], orig: Series['datetime64[ns]'], exp: Series['datetime64[ns]'], vals: Series['datetime64[ns]']) -> None:
        ...

    def test_setitem_with_tz_dst(self, indexer_sli: Callable[[Series], Any], orig: Series['datetime64[ns, US/Eastern]'], exp: Series['datetime64[ns, US/Eastern]'], vals: Series['datetime64[ns, US/Eastern]']) -> None:
        ...

    def test_object_series_setitem_dt64array_exact_match(self, ser: Series[object], indexer: list[bool], value: np.ndarray['datetime64[ns]']) -> None:
        ...

class TestSetitemScalarIndexer:
    def test_setitem_negative_out_of_bounds(self, ser: Series[str], exp: Series[str]) -> None:
        ...

    @pytest.mark.parametrize('indexer', [tm.loc, tm.at])
    @pytest.mark.parametrize('ser_index', [0, 1])
    def test_setitem_series_object_dtype(self, indexer: Callable[[Series], Any], ser_index: int, idxr: Any, expected: Series[object]) -> None:
        ...

    @pytest.mark.parametrize('index, exp_value', [(0, 42), (1, np.nan)])
    def test_setitem_series(self, index: int, exp_value: int | float, ser: Series[int], expected: Series[int]) -> None:
        ...

class TestSetitemSlices:
    def test_setitem_slice_float_raises(self, datetime_series: Series['datetime64[ns]']) -> None:
        ...

    def test_setitem_slice(self, ser: Series[int]) -> None:
        ...

    def test_setitem_slice_integers(self, ser: Series[float]) -> None:
        ...

    def test_setitem_slicestep(self, series: Series[int]) -> None:
        ...

    def test_setitem_multiindex_slice(self, indexer_sli: Callable[[Series], Any], mi: MultiIndex, result: Series[int], expected: Series[int]) -> None:
        ...

class TestSetitemBooleanMask:
    def test_setitem_mask_cast(self, ser: Series[int], mask: Series[bool], expected: Series[int]) -> None:
        ...

    def test_setitem_mask_align_and_promote(self, ts: Series[float], mask: Series[bool], left: Series[float], right: Series[str]) -> None:
        ...

    def test_setitem_mask_promote_strs(self, ser: Series[int], mask: Series[bool], ser2: Series[str]) -> None:
        ...

    def test_setitem_mask_promote(self, ser: Series[object], mask: Series[bool], ser2: Series[object]) -> None:
        ...

    def test_setitem_boolean(self, string_series: Series[str], result: Series[str], expected: Series[str]) -> None:
        ...

    def test_setitem_boolean_corner(self, datetime_series: Series['datetime64[ns]'], mask_shifted: Series[bool]) -> None:
        ...

    def test_setitem_boolean_different_order(self, string_series: Series[str], ordered: Series[str], copy: Series[str], expected: Series[str]) -> None:
        ...

    @pytest.mark.parametrize('func', [list, np.array, Series])
    def test_setitem_boolean_python_list(self, func: Callable, ser: Series[object], mask: list[bool], expected: Series[object]) -> None:
        ...

    def test_setitem_boolean_nullable_int_types(self, any_numeric_ea_dtype: Any, ser: Series[Any], expected: Series[Any]) -> None:
        ...

    def test_setitem_with_bool_mask_and_values_matching_n_trues_in_length(self, ser: Series[object], mask: list[bool], expected: Series[object]) -> None:
        ...

    def test_setitem_nan_with_bool(self, result: Series[bool]) -> None:
        ...

    def test_setitem_mask_smallint_upcast(self, orig: Series['int8'], alt: np.ndarray[int], mask: np.ndarray[bool], ser: Series['int8'], res: Series[int], expected: Series[int]) -> None:
        ...

    def test_setitem_mask_smallint_no_upcast(self, orig: Series['uint8'], alt: Series[int], mask: np.ndarray[bool], ser: Series['uint8'], ser2: Series['uint8'], ser3: Series['uint8'], expected: Series['uint8']) -> None:
        ...

class TestSetitemViewCopySemantics:
    def test_setitem_invalidates_datetime_index_freq(self, dti: DatetimeIndex, ts: Timestamp, ser: Series['datetime64[ns]']) -> None:
        ...

    def test_dt64tz_setitem_does_not_mutate_dti(self, dti: DatetimeIndex, ts: Timestamp, ser: Series['datetime64[ns]']) -> None:
        ...

class TestSetitemCallable:
    def test_setitem_callable_key(self, ser: Series[int], expected: Series[int]) -> None:
        ...

    def test_setitem_callable_other(self, inc: Callable[[int], int], ser: Series[object], expected: Series[object]) -> None:
        ...

class TestSetitemWithExpansion:
    def test_setitem_empty_series(self, series: Series[object], key: Timestamp, expected: Series[object]) -> None:
        ...

    def test_setitem_empty_series_datetimeindex_preserves_freq(self, dti: DatetimeIndex, series: Series[object], key: Timestamp, expected: Series[object]) -> None:
        ...

    def test_setitem_empty_series_timestamp_preserves_dtype(self, timestamp: Timestamp, series: Series[object], expected: Series[object]) -> None:
        ...

    @pytest.mark.parametrize('td', [Timedelta('9 days'), Timedelta('9 days').to_timedelta64(), Timedelta('9 days').to_pytimedelta()])
    def test_append_timedelta_does_not_cast(self, td: Timedelta, using_infer_string: bool, request: Any, series: Series[object], expected: Series[object]) -> None:
        ...

    def test_setitem_with_expansion_type_promotion(self, ser: Series[object], expected: Series[object]) -> None:
        ...

    def test_setitem_not_contained(self, string_series: Series[str], ser: Series[str], expected: Series[str]) -> None:
        ...

    def test_setitem_keep_precision(self, any_numeric_ea_dtype: Any, ser: Series[Any], expected: Series[Any]) -> None:
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
    def test_setitem_enlarge_with_na(self, na: Any, target_na: Any, dtype: str, target_dtype: str, indexer: int, raises: bool, ser: Series[Any], expected: Series[Any]) -> None:
        ...

    def test_setitem_enlargement_object_none(self, nulls_fixture: Any, using_infer_string: bool, ser: Series[object], expected: Series[object]) -> None:
        ...

def test_setitem_scalar_into_readonly_backing_data(array: np.ndarray, series: Series[int]) -> None:
    ...

def test_setitem_slice_into_readonly_backing_data(array: np.ndarray, series: Series[int]) -> None:
    ...

def test_setitem_categorical_assigning_ops(orig: Series[Categorical], ser: Series[Categorical], expected: Series[Categorical]) -> None:
    ...

def test_setitem_nan_into_categorical(ser: Series[Categorical], expected: Series[Categorical]) -> None:
    ...

class TestSetitemCasting:
    @pytest.mark.parametrize('unique', [True, False])
    @pytest.mark.parametrize('val', [3, 3.0, '3'], ids=type)
    def test_setitem_non_bool_into_bool(self, val: int | float | str, indexer_sli: Callable[[Series], Any], unique: bool, ser: Series[bool], expected: Series[object]) -> None:
        ...

    def test_setitem_boolean_array_into_npbool(self, ser: Series[bool], values: Series[bool], arr: array, expected: Series[bool]) -> None:
        ...

class SetitemCastingEquivalents:
    def check_indexer(self, obj: Series, key: Any, expected: Series, val: Any, indexer: Callable[[Series], Any], is_inplace: bool) -> None:
        ...

    def _check_inplace(self, is_inplace: bool, orig: Series, arr: np.ndarray, obj: Series) -> None:
        ...

    def test_int_key(self, obj: Series, key: int, expected: Series, raises: bool, val: Any, indexer_sli: Callable[[Series], Any], is_inplace: bool) -> None:
        ...

    def test_slice_key(self, obj: Series, key: slice, expected: Series, raises: bool, val: Any, indexer_sli: Callable[[Series], Any], is_inplace: bool) -> None:
        ...

    def test_mask_key(self, obj: Series, key: Any, expected: Series, raises: bool, val: Any, indexer_sli: Callable[[Series], Any]) -> None:
        ...

    def test_series_where(self, obj: Series, key: Any, expected: Series, raises: bool, val: Any, is_inplace: bool) -> None:
        ...

    def test_index_where(self, obj: Series, key: Any, expected: Index, raises: bool, val: Any) -> None:
        ...

    def test_index_putmask(self, obj: Series, key: Any, expected: Index, raises: bool, val: Any) -> None:
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
    ...

class TestSetitemTimedelta64IntoNumeric(SetitemCastingEquivalents):
    ...

class TestSetitemDT64IntoInt(SetitemCastingEquivalents):
    ...

class TestSetitemNAPeriodDtype(SetitemCastingEquivalents):
    ...

class TestSetitemNADatetimeLikeDtype(SetitemCastingEquivalents):
    ...

class TestSetitemMismatchedTZCastsToObject(SetitemCastingEquivalents):
    ...

@pytest.mark.parametrize('obj,expected', [
    (Series([1, 2, 3]), Series([np.nan, 2, 3])),
    (Series([1.0, 2.0, 3.0]), Series([np.nan, 2.0, 3.0])),
    (Series([datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)]), Series([NaT, datetime(2000, 1, 2), datetime(2000, 1, 3)])),
    (Series(['foo', 'bar', 'baz']), Series([None, 'bar', 'baz']))
])
class TestSeriesNoneCoercion(SetitemCastingEquivalents):
    ...

class TestSetitemFloatIntervalWithIntIntervalValues(SetitemCastingEquivalents):
    ...

class TestSetitemRangeIntoIntegerSeries(SetitemCastingEquivalents):
    ...

@pytest.mark.parametrize('val, raises', [
    (np.array([2.0, 3.0]), False),
    (np.array([2.5, 3.5]), True),
    (np.array([2 ** 65, 2 ** 65 + 1], dtype=np.float64), True)
])
class TestSetitemFloatNDarrayIntoIntegerSeries(SetitemCastingEquivalents):
    ...

@pytest.mark.parametrize('val', [512, np.int16(512)])
class TestSetitemIntoIntegerSeriesNeedsUpcast(SetitemCastingEquivalents):
    ...

@pytest.mark.parametrize('val', [2 ** 33 + 1.0, 2 ** 33 + 1.1, 2 ** 62])
class TestSmallIntegerSetitemUpcast(SetitemCastingEquivalents):
    ...

class CoercionTest(SetitemCastingEquivalents):
    ...

@pytest.mark.parametrize('val,exp_dtype,raises', [
    (np.int32(1), np.int8, None),
    (np.int16(2 ** 9), np.int16, True)
])
class TestCoercionInt8(CoercionTest):
    ...

@pytest.mark.parametrize('val', [1, 1.1, 1 + 1j, True])
@pytest.mark.parametrize('exp_dtype', [object])
class TestCoercionObject(CoercionTest):
    ...

@pytest.mark.parametrize('val', ['e'], ids=type)
@pytest.mark.parametrize('exp_dtype', [StringDtype(na_value=np.nan)])
class TestCoercionString(CoercionTest):
    ...

@pytest.mark.parametrize('val,exp_dtype,raises', [
    (1, np.complex128, False),
    (1.1, np.complex128, False),
    (1 + 1j, np.complex128, False),
    (True, object, True)
])
class TestCoercionComplex(CoercionTest):
    ...

@pytest.mark.parametrize('val,exp_dtype,raises', [
    (1, object, True),
    ('3', StringDtype(na_value=np.nan), False),
    (3, object, True),
    (1.1, object, True),
    (1 + 1j, object, True),
    (True, bool, False)
])
class TestCoercionBool(CoercionTest):
    ...

@pytest.mark.parametrize('val,exp_dtype,raises', [
    (1, np.int64, False),
    (1.1, np.float64, True),
    (1 + 1j, np.complex128, True),
    (True, object, True)
])
class TestCoercionInt64(CoercionTest):
    ...

@pytest.mark.parametrize('val,exp_dtype,raises', [
    (1, np.float64, False),
    (1.1, np.float64, False),
    (1 + 1j, np.complex128, True),
    (True, object, True)
])
class TestCoercionFloat64(CoercionTest):
    ...

@pytest.mark.parametrize('val,exp_dtype,raises', [
    (1, np.float32, False),
    (1.1, np.float32, False),
    (1 + 1j, np.complex128, True),
    (True, object, True),
    (np.uint8(2), np.float32, False),
    (np.uint32(2), np.float32, False),
    (np.uint32(np.iinfo(np.uint32).max), np.float64, True),
    (np.uint64(2), np.float32, False),
    (np.int64(2), np.float32, False)
])
class TestCoercionFloat32(CoercionTest):
    ...

@pytest.mark.parametrize('exp_dtype', ['M8[ms]', 'M8[ms, UTC]', 'm8[ms]'])
class TestCoercionDatetime64HigherReso(CoercionTest):
    ...

@pytest.mark.parametrize('val,exp_dtype,raises', [
    (Timestamp('2012-01-01'), 'datetime64[ns]', False),
    (1, object, True),
    ('x', object, True)
])
class TestCoercionDatetime64(CoercionTest):
    ...

@pytest.mark.parametrize('val,exp_dtype,raises', [
    (Timestamp('2012-01-01', tz='US/Eastern'), 'datetime64[ns, US/Eastern]', False),
    (Timestamp('2012-01-01', tz='US/Pacific'), 'datetime64[ns, US/Eastern]', False),
    (Timestamp('2012-01-01'), object, True),
    (1, object, True)
])
class TestCoercionDatetime64TZ(CoercionTest):
    ...

@pytest.mark.parametrize('val,exp_dtype,raises', [
    (Timedelta('12 day'), 'timedelta64[ns]', False),
    (1, object, True),
    ('x', object, True)
])
class TestCoercionTimedelta64(CoercionTest):
    ...

@pytest.mark.parametrize('val', ['foo', Period('2016', freq='Y'), Interval(1, 2, closed='both')])
@pytest.mark.parametrize('exp_dtype', [object])
class TestPeriodIntervalCoercion(CoercionTest):
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

def test_37692(indexer_al: Callable[[Series], Any]) -> None:
    ...

def test_setitem_bool_int_float_consistency(indexer_sli: Callable[[Series], Any]) -> None:
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
def test_setitem_bool_indexer_dont_broadcast_length1_values(size: int, mask: list[bool], item: float, box: Callable) -> None:
    ...

def test_setitem_empty_mask_dont_upcast_dt64() -> None:
    ...