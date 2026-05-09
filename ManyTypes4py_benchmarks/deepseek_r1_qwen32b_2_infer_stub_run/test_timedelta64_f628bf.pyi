from datetime import datetime, timedelta
import numpy as np
import pytest
from pandas.compat import WASM
from pandas.errors import OutOfBoundsDatetime
import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    NaT,
    Series,
    Timedelta,
    TimedeltaIndex,
    Timestamp,
    offsets,
    timedelta_range,
)
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
    assert_invalid_addsub_type,
    assert_invalid_comparison,
    get_upcast_box,
)
from typing import (
    Union,
    List,
    Optional,
    Type,
    Callable,
    Any,
    Tuple,
    Dict,
    overload,
    Literal,
)

def assert_dtype(obj: Union[Series, Index, DataFrame], expected_dtype: np.dtype) -> None:
    ...

def get_expected_name(box: Type[Union[DataFrame, tm.to_array, pd.array]], names: List[str]) -> str:
    ...

class TestTimedelta64ArrayLikeComparisons:
    def test_compare_timedelta64_zerodim(
        self, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]]
    ) -> None:
        ...

    @pytest.mark.parametrize(
        'td_scalar',
        [timedelta(days=1), Timedelta(days=1), Timedelta(days=1).to_timedelta64(), offsets.Hour(24)],
    )
    def test_compare_timedeltalike_scalar(
        self, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]], td_scalar: Union[timedelta, Timedelta, np.timedelta64, offsets.Hour]
    ) -> None:
        ...

    @pytest.mark.parametrize(
        'invalid',
        [
            345600000000000,
            'a',
            Timestamp('2021-01-01'),
            Timestamp('2021-01-01').now('UTC'),
            Timestamp('2021-01-01').now().to_datetime64(),
            Timestamp('2021-01-01').now().to_pydatetime(),
            Timestamp('2021-01-01').date(),
            np.array(4),
        ],
    )
    def test_td64_comparisons_invalid(
        self, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]], invalid: Union[int, str, Timestamp, np.datetime64, datetime, date]
    ) -> None:
        ...

    @pytest.mark.parametrize(
        'other',
        [
            list(range(10)),
            np.arange(10),
            np.arange(10).astype(np.float32),
            np.arange(10).astype(object),
            pd.date_range('1970-01-01', periods=10, tz='UTC').array,
            np.array(pd.date_range('1970-01-01', periods=10)),
            list(pd.date_range('1970-01-01', periods=10)),
            pd.date_range('1970-01-01', periods=10).astype(object),
            pd.period_range('1971-01-01', freq='D', periods=10).array,
            pd.period_range('1971-01-01', freq='D', periods=10).astype(object),
        ],
    )
    def test_td64arr_cmp_arraylike_invalid(
        self, other: Union[List[int], np.ndarray, pd.DatetimeArray, pd.PeriodArray], box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]]
    ) -> None:
        ...

    def test_td64arr_cmp_mixed_invalid(self) -> None:
        ...

class TestTimedelta64ArrayComparisons:
    @pytest.mark.parametrize('dtype', [None, object])
    def test_comp_nat(self, dtype: Optional[str]) -> None:
        ...

    @pytest.mark.parametrize(
        'idx2',
        [
            TimedeltaIndex(['2 day', '2 day', NaT, NaT, '1 day 00:00:02', '5 days 00:00:03']),
            np.array(
                [
                    np.timedelta64(2, 'D'),
                    np.timedelta64(2, 'D'),
                    np.timedelta64('nat'),
                    np.timedelta64('nat'),
                    np.timedelta64(1, 'D') + np.timedelta64(2, 's'),
                    np.timedelta64(5, 'D') + np.timedelta64(3, 's'),
                ]
            ),
        ],
    )
    def test_comparisons_nat(self, idx2: Union[TimedeltaIndex, np.ndarray]) -> None:
        ...

    def test_comparisons_coverage(self) -> None:
        ...

class TestTimedelta64ArithmeticUnsorted:
    def test_ufunc_coercions(self) -> None:
        ...

    def test_subtraction_ops(self) -> None:
        ...

    def test_subtraction_ops_with_tz(
        self, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]]
    ) -> None:
        ...

    def test_dti_tdi_numeric_ops(self) -> None:
        ...

    def test_addition_ops(self) -> None:
        ...

    @pytest.mark.parametrize('freq', ['D', 'B'])
    def test_timedelta(self, freq: Literal['D', 'B']) -> None:
        ...

    def test_timedelta_tick_arithmetic(self) -> None:
        ...

    def test_tda_add_sub_index(self) -> None:
        ...

    def test_tda_add_dt64_object_array(
        self, performance_warning: Any, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]], tz_naive_fixture: Optional[str]
    ) -> None:
        ...

    def test_tdi_iadd_timedeltalike(
        self, two_hours: Union[timedelta, Timedelta, np.timedelta64, offsets.Hour], box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]]
    ) -> None:
        ...

    def test_tdi_isub_timedeltalike(
        self, two_hours: Union[timedelta, Timedelta, np.timedelta64, offsets.Hour], box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]]
    ) -> None:
        ...

    def test_tdi_ops_attributes(self) -> None:
        ...

class TestAddSubNaTMasking:
    @pytest.mark.parametrize('str_ts', ['1950-01-01', '1980-01-01'])
    def test_tdarr_add_timestamp_nat_masking(
        self, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]], str_ts: str
    ) -> None:
        ...

    def test_tdi_add_overflow(self) -> None:
        ...

class TestTimedeltaArraylikeAddSubOps:
    def test_sub_nat_retain_unit(self) -> None:
        ...

    def test_timedelta_ops_with_missing_values(self) -> None:
        ...

    def test_operators_timedelta64(self) -> None:
        ...

    def test_timedelta64_ops_nat(self) -> None:
        ...

    @pytest.mark.parametrize('cls', [Timestamp, datetime, np.datetime64])
    def test_td64arr_add_sub_datetimelike_scalar(
        self, cls: Type[Union[Timestamp, datetime, np.datetime64]], box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]], tz_naive_fixture: Optional[str]
    ) -> None:
        ...

    def test_td64arr_add_datetime64_nat(self, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]]) -> None:
        ...

    def test_td64arr_sub_dt64_array(self, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]]) -> None:
        ...

    def test_td64arr_add_dt64_array(self, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]]) -> None:
        ...

    @pytest.mark.parametrize('pi_freq', ['D', 'W', 'Q', 'h'])
    @pytest.mark.parametrize('tdi_freq', [None, 'h'])
    def test_td64arr_sub_periodlike(
        self, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]], box_with_array2: Type[Union[DataFrame, tm.to_array, pd.array]], tdi_freq: Optional[str], pi_freq: str
    ) -> None:
        ...

    @pytest.mark.parametrize('other', ['a', 1, 1.5, np.array(2)])
    def test_td64arr_addsub_numeric_scalar_invalid(
        self, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]], other: Union[str, int, float, np.ndarray]
    ) -> None:
        ...

    @pytest.mark.parametrize(
        'vec',
        [
            np.array([1, 2, 3]),
            Index([1, 2, 3]),
            Series([1, 2, 3]),
            DataFrame([[1, 2, 3]]),
        ],
        ids=lambda x: type(x).__name__,
    )
    def test_td64arr_addsub_numeric_arr_invalid(
        self, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]], vec: Union[np.ndarray, Index, Series, DataFrame], any_real_numpy_dtype: np.dtype
    ) -> None:
        ...

    def test_td64arr_add_sub_int(
        self, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]], one: Union[int, float]
    ) -> None:
        ...

    def test_td64arr_add_sub_integer_array(
        self, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]]
    ) -> None:
        ...

    def test_td64arr_add_sub_integer_array_no_freq(
        self, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]]
    ) -> None:
        ...

    def test_td64arr_add_sub_td64_array(
        self, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]]
    ) -> None:
        ...

    def test_td64arr_add_sub_tdi(
        self, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]], names: List[str]
    ) -> None:
        ...

    @pytest.mark.parametrize('tdnat', [np.timedelta64('NaT'), NaT])
    def test_td64arr_add_sub_td64_nat(
        self, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]], tdnat: Union[np.timedelta64, NaT]
    ) -> None:
        ...

    def test_td64arr_add_timedeltalike(
        self, two_hours: Union[timedelta, Timedelta, np.timedelta64, offsets.Hour], box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]]
    ) -> None:
        ...

    def test_td64arr_sub_timedeltalike(
        self, two_hours: Union[timedelta, Timedelta, np.timedelta64, offsets.Hour], box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]]
    ) -> None:
        ...

    def test_td64arr_add_sub_offset_index(
        self, performance_warning: Any, names: List[str], box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]]
    ) -> None:
        ...

    def test_td64arr_add_sub_offset_array(
        self, performance_warning: Any, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]]
    ) -> None:
        ...

    def test_td64arr_with_offset_series(
        self, performance_warning: Any, names: List[str], box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]]
    ) -> None:
        ...

    @pytest.mark.parametrize('obox', [np.array, Index, Series])
    def test_td64arr_addsub_anchored_offset_arraylike(
        self, performance_warning: Any, obox: Type[Union[np.ndarray, Index, Series]], box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]]
    ) -> None:
        ...

    def test_td64arr_add_sub_object_array(
        self, performance_warning: Any, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]]
    ) -> None:
        ...

class TestTimedeltaArraylikeMulDivOps:
    def test_td64arr_mul_int(self, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]]) -> None:
        ...

    def test_td64arr_mul_tdlike_scalar_raises(
        self, two_hours: Union[timedelta, Timedelta, np.timedelta64, offsets.Hour], box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]]
    ) -> None:
        ...

    def test_tdi_mul_int_array_zerodim(
        self, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]]
    ) -> None:
        ...

    def test_tdi_mul_int_array(self, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]]) -> None:
        ...

    def test_tdi_mul_int_series(
        self, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]]
    ) -> None:
        ...

    def test_tdi_mul_float_series(
        self, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]]
    ) -> None:
        ...

    @pytest.mark.parametrize(
        'other',
        [
            np.arange(1, 11),
            Index(np.arange(1, 11), np.int64),
            Index(range(1, 11), np.uint64),
            Index(range(1, 11), np.float64),
            pd.RangeIndex(1, 11),
        ],
        ids=lambda x: type(x).__name__,
    )
    def test_tdi_rmul_arraylike(
        self, other: Union[np.ndarray, Index, pd.RangeIndex], box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]]
    ) -> None:
        ...

    def test_td64arr_div_nat_invalid(
        self, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]]
    ) -> None:
        ...

    def test_td64arr_div_td64nat(
        self, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]]
    ) -> None:
        ...

    def test_td64arr_div_int(self, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]]) -> None:
        ...

    def test_td64arr_div_tdlike_scalar(
        self, two_hours: Union[timedelta, Timedelta, np.timedelta64, offsets.Hour], box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]]
    ) -> None:
        ...

    @pytest.mark.parametrize('m', [1, 3, 10])
    @pytest.mark.parametrize('unit', ['D', 'h', 'm', 's', 'ms', 'us', 'ns'])
    def test_td64arr_div_td64_scalar(
        self, m: int, unit: str, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]]
    ) -> None:
        ...

    def test_td64arr_div_tdlike_scalar_with_nat(
        self, two_hours: Union[timedelta, Timedelta, np.timedelta64, offsets.Hour], box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]]
    ) -> None:
        ...

    def test_td64arr_div_td64_ndarray(
        self, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]]
    ) -> None:
        ...

    def test_td64_div_object_mixed_result(
        self, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]]
    ) -> None:
        ...

    @pytest.mark.skipif(WASM, reason='no fp exception support in wasm')
    def test_td64arr_floordiv_td64arr_with_nat(
        self, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]]
    ) -> None:
        ...

    @pytest.mark.filterwarnings('ignore:invalid value encountered:RuntimeWarning')
    def test_td64arr_floordiv_tdscalar(
        self, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]], scalar_td: Union[timedelta, Timedelta, np.timedelta64]
    ) -> None:
        ...

    def test_td64arr_mod_tdscalar(
        self, performance_warning: Any, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]], three_days: Union[timedelta, Timedelta, np.timedelta64]
    ) -> None:
        ...

    def test_td64arr_mod_int(
        self, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]]
    ) -> None:
        ...

    def test_td64arr_rmod_tdscalar(
        self, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]], three_days: Union[timedelta, Timedelta, np.timedelta64]
    ) -> None:
        ...

    def test_td64arr_mul_tdscalar_invalid(
        self, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]], scalar_td: Union[timedelta, Timedelta, np.timedelta64]
    ) -> None:
        ...

    def test_td64arr_mul_too_short_raises(
        self, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]]
    ) -> None:
        ...

    def test_td64arr_mul_td64arr_raises(
        self, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]]
    ) -> None:
        ...

    def test_td64arr_mul_numeric_scalar(
        self, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]], one: Union[int, float]
    ) -> None:
        ...

    @pytest.mark.parametrize('two', [2, 2.0, np.array(2), np.array(2.0)])
    def test_td64arr_div_numeric_scalar(
        self, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]], two: Union[int, float, np.ndarray]
    ) -> None:
        ...

    @pytest.mark.parametrize('two', [2, 2.0, np.array(2), np.array(2.0)])
    def test_td64arr_floordiv_numeric_scalar(
        self, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]], two: Union[int, float, np.ndarray]
    ) -> None:
        ...

    @pytest.mark.parametrize(
        'klass', [np.array, Index, Series], ids=lambda x: x.__name__
    )
    def test_td64arr_rmul_numeric_array(
        self, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]], klass: Type[Union[np.ndarray, Index, Series]], any_real_numpy_dtype: np.dtype
    ) -> None:
        ...

    @pytest.mark.parametrize(
        'klass', [np.array, Index, Series], ids=lambda x: x.__name__
    )
    def test_td64arr_div_numeric_array(
        self, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]], klass: Type[Union[np.ndarray, Index, Series]], any_real_numpy_dtype: np.dtype
    ) -> None:
        ...

    def test_td64arr_mul_int_series(
        self, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]], names: List[str]
    ) -> None:
        ...

    def test_float_series_rdiv_td64arr(
        self, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]], names: List[str]
    ) -> None:
        ...

    def test_td64arr_all_nat_div_object_dtype_numeric(
        self, box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]]
    ) -> None:
        ...

class TestTimedelta64ArrayLikeArithmetic:
    def test_td64arr_pow_invalid(
        self, scalar_td: Union[timedelta, Timedelta, np.timedelta64], box_with_array: Type[Union[DataFrame, tm.to_array, pd.array]]
    ) -> None:
        ...

def test_add_timestamp_to_timedelta() -> None:
    ...