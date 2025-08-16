from __future__ import annotations
from datetime import datetime
import weakref
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
from pandas.core.dtypes.common import is_integer_dtype, is_numeric_dtype
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import CategoricalIndex, DatetimeIndex, DatetimeTZDtype, Index, IntervalIndex, MultiIndex, PeriodIndex, RangeIndex, Series, StringDtype, TimedeltaIndex, isna, period_range
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import BaseMaskedArray

class TestBase:

    @pytest.fixture(params=[RangeIndex(start=0, stop=20, step=2), Index(np.arange(5, dtype=np.float64)), Index(np.arange(5, dtype=np.float32)), Index(np.arange(5, dtype=np.uint64)), Index(range(0, 20, 2), dtype=np.int64), Index(range(0, 20, 2), dtype=np.int32), Index(range(0, 20, 2), dtype=np.int16), Index(range(0, 20, 2), dtype=np.int8), Index(list('abcde')), Index([0, 'a', 1, 'b', 2, 'c']), period_range('20130101', periods=5, freq='D'), TimedeltaIndex(['0 days 01:00:00', '1 days 01:00:00', '2 days 01:00:00', '3 days 01:00:00', '4 days 01:00:00'], dtype='timedelta64[ns]', freq='D'), DatetimeIndex(['2013-01-01', '2013-01-02', '2013-01-03', '2013-01-04', '2013-01-05'], dtype='datetime64[ns]', freq='D'), IntervalIndex.from_breaks(range(11), closed='right')])
    def simple_index(self, request) -> Index:
        return request.param

    def test_pickle_compat_construction(self, simple_index: Index) -> None:
        ...

    def test_shift(self, simple_index: Index) -> None:
        ...

    def test_constructor_name_unhashable(self, simple_index: Index) -> None:
        ...

    def test_create_index_existing_name(self, simple_index: Index) -> None:
        ...

    def test_numeric_compat(self, simple_index: Index) -> None:
        ...

    def test_logical_compat(self, simple_index: Index) -> None:
        ...

    def test_repr_roundtrip(self, simple_index: Index) -> None:
        ...

    def test_repr_max_seq_item_setting(self, simple_index: Index) -> None:
        ...

    @pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
    def test_ensure_copied_data(self, index: Index) -> None:
        ...

    def test_memory_usage(self, index: Index) -> None:
        ...

    def test_memory_usage_doesnt_trigger_engine(self, index: Index) -> None:
        ...

    def test_argsort(self, index: Index) -> None:
        ...

    def test_numpy_argsort(self, index: Index) -> None:
        ...

    def test_repeat(self, simple_index: Index) -> None:
        ...

    def test_numpy_repeat(self, simple_index: Index) -> None:
        ...

    def test_where(self, listlike_box, simple_index: Index) -> None:
        ...

    def test_insert_base(self, index: Index) -> None:
        ...

    def test_insert_out_of_bounds(self, index: Index, using_infer_string) -> None:
        ...

    def test_delete_base(self, index: Index) -> None:
        ...

    @pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
    def test_equals(self, index: Index) -> None:
        ...

    def test_equals_op(self, simple_index: Index) -> None:
        ...

    def test_fillna(self, index: Index) -> None:
        ...

    def test_nulls(self, index: Index) -> None:
        ...

    def test_empty(self, simple_index: Index) -> None:
        ...

    def test_join_self_unique(self, join_type, simple_index: Index) -> None:
        ...

    def test_map(self, simple_index: Index) -> None:
        ...

    @pytest.mark.parametrize('mapper', [lambda values, index: {i: e for e, i in zip(values, index)}, lambda values, index: Series(values, index)])
    @pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
    def test_map_dictlike(self, mapper, simple_index: Index, request) -> None:
        ...

    def test_map_str(self, simple_index: Index) -> None:
        ...

    @pytest.mark.parametrize('copy', [True, False])
    @pytest.mark.parametrize('name', [None, 'foo'])
    @pytest.mark.parametrize('ordered', [True, False])
    def test_astype_category(self, copy, name, ordered, simple_index: Index) -> None:
        ...

    def test_is_unique(self, simple_index: Index) -> None:
        ...

    @pytest.mark.arm_slow
    def test_engine_reference_cycle(self, simple_index: Index) -> None:
        ...

    def test_getitem_2d_deprecated(self, simple_index: Index) -> None:
        ...

    def test_copy_shares_cache(self, simple_index: Index) -> None:
        ...

    def test_shallow_copy_shares_cache(self, simple_index: Index) -> None:
        ...

    def test_index_groupby(self, simple_index: Index) -> None:
        ...

    def test_append_preserves_dtype(self, simple_index: Index) -> None:
        ...

    def test_inv(self, simple_index: Index, using_infer_string) -> None:
        ...

    def test_view(self, simple_index: Index) -> None:
        ...

    def test_insert_non_na(self, simple_index: Index) -> None:
        ...

    def test_insert_na(self, nulls_fixture, simple_index: Index) -> None:
        ...

    def test_arithmetic_explicit_conversions(self, simple_index: Index) -> None:
        ...

    @pytest.mark.parametrize('complex_dtype', [np.complex64, np.complex128])
    def test_astype_to_complex(self, complex_dtype, simple_index: Index) -> None:
        ...

    def test_cast_string(self, simple_index: Index) -> None:
        ...

class TestNumericBase:

    @pytest.fixture(params=[RangeIndex(start=0, stop=20, step=2), Index(np.arange(5, dtype=np.float64)), Index(np.arange(5, dtype=np.float32)), Index(np.arange(5, dtype=np.uint64)), Index(range(0, 20, 2), dtype=np.int64), Index(range(0, 20, 2), dtype=np.int32), Index(range(0, 20, 2), dtype=np.int16), Index(range(0, 20, 2), dtype=np.int8)])
    def simple_index(self, request) -> Index:
        return request.param

    def test_constructor_unwraps_index(self, simple_index: Index) -> None:
        ...

    def test_can_hold_identifiers(self, simple_index: Index) -> None:
        ...

    def test_view(self, simple_index: Index) -> None:
        ...

    def test_insert_non_na(self, simple_index: Index) -> None:
        ...

    def test_insert_na(self, nulls_fixture, simple_index: Index) -> None:
        ...

    def test_arithmetic_explicit_conversions(self, simple_index: Index) -> None:
        ...

    @pytest.mark.parametrize('complex_dtype', [np.complex64, np.complex128])
    def test_astype_to_complex(self, complex_dtype, simple_index: Index) -> None:
        ...
