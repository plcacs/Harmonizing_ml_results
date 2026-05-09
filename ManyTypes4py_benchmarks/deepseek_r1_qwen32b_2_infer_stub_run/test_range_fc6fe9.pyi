import numpy as np
import pytest
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import Index, RangeIndex
import pandas._testing as tm
from pandas.core.indexes.range import min_fitting_element

class TestRangeIndex:
    @pytest.fixture
    def simple_index(self) -> RangeIndex:
        ...

    def test_constructor_unwraps_index(self) -> None:
        ...

    def test_can_hold_identifiers(self, simple_index: RangeIndex) -> None:
        ...

    def test_too_many_names(self, index: RangeIndex) -> None:
        ...

    @pytest.mark.parametrize('index, start, stop, step', [(RangeIndex(5), 0, 5, 1), (RangeIndex(0, 5), 0, 5, 1), (RangeIndex(5, step=2), 0, 5, 2), (RangeIndex(1, 5, 2), 1, 5, 2)])
    def test_start_stop_step_attrs(self, index: RangeIndex, start: int, stop: int, step: int) -> None:
        ...

    def test_copy(self) -> None:
        ...

    def test_repr(self) -> None:
        ...

    def test_insert(self) -> None:
        ...

    def test_insert_edges_preserves_rangeindex(self) -> None:
        ...

    def test_insert_middle_preserves_rangeindex(self) -> None:
        ...

    def test_delete(self) -> None:
        ...

    def test_delete_preserves_rangeindex(self) -> None:
        ...

    def test_delete_preserves_rangeindex_middle(self) -> None:
        ...

    def test_delete_preserves_rangeindex_list_at_end(self) -> None:
        ...

    def test_delete_preserves_rangeindex_list_middle(self) -> None:
        ...

    def test_delete_all_preserves_rangeindex(self) -> None:
        ...

    def test_delete_not_preserving_rangeindex(self) -> None:
        ...

    def test_view(self) -> None:
        ...

    def test_dtype(self, simple_index: RangeIndex) -> None:
        ...

    def test_cache(self) -> None:
        ...

    def test_is_monotonic(self) -> None:
        ...

    @pytest.mark.parametrize('left,right', [(RangeIndex(0, 9, 2), RangeIndex(0, 10, 2)), (RangeIndex(0), RangeIndex(1, -1, 3)), (RangeIndex(1, 2, 3), RangeIndex(1, 3, 4)), (RangeIndex(0, -9, -2), RangeIndex(0, -10, -2))])
    def test_equals_range(self, left: RangeIndex, right: RangeIndex) -> None:
        ...

    def test_logical_compat(self, simple_index: RangeIndex) -> None:
        ...

    def test_identical(self, simple_index: RangeIndex) -> None:
        ...

    def test_nbytes(self) -> None:
        ...

    @pytest.mark.parametrize('start,stop,step', [('foo', 'bar', 'baz'), ('0', '1', '2')])
    def test_cant_or_shouldnt_cast(self, start: str, stop: str, step: str) -> None:
        ...

    def test_view_index(self, simple_index: RangeIndex) -> None:
        ...

    def test_prevent_casting(self, simple_index: RangeIndex) -> None:
        ...

    def test_repr_roundtrip(self, simple_index: RangeIndex) -> None:
        ...

    def test_slice_keep_name(self) -> None:
        ...

    @pytest.mark.parametrize('index', [RangeIndex(start=0, stop=20, step=2, name='foo'), RangeIndex(start=18, stop=-1, step=-2, name='bar')], ids=['index_inc', 'index_dec'])
    def test_has_duplicates(self, index: RangeIndex) -> None:
        ...

    def test_extended_gcd(self, simple_index: RangeIndex) -> None:
        ...

    def test_min_fitting_element(self) -> None:
        ...

    def test_slice_specialised(self, simple_index: RangeIndex) -> None:
        ...

    @pytest.mark.parametrize('step', set(range(-5, 6)) - {0})
    def test_len_specialised(self, step: int) -> None:
        ...

    @pytest.mark.parametrize('indices, expected', [([RangeIndex(1, 12, 5)], RangeIndex(1, 12, 5)), ([RangeIndex(0, 6, 4)], RangeIndex(0, 6, 4)), ([RangeIndex(1, 3), RangeIndex(3, 7)], RangeIndex(1, 7)), ([RangeIndex(1, 5, 2), RangeIndex(5, 6)], RangeIndex(1, 6, 2)), ([RangeIndex(1, 3, 2), RangeIndex(4, 7, 3)], RangeIndex(1, 7, 3)), ([RangeIndex(-4, 3, 2), RangeIndex(4, 7, 2)], RangeIndex(-4, 7, 2)), ([RangeIndex(-4, -8), RangeIndex(-8, -12)], RangeIndex(0, 0)), ([RangeIndex(-4, -8), RangeIndex(3, -4)], RangeIndex(0, 0)), ([RangeIndex(-4, -8), RangeIndex(3, 5)], RangeIndex(3, 5)), ([RangeIndex(-4, -2), RangeIndex(3, 5)], Index([-4, -3, 3, 4])), ([RangeIndex(-2), RangeIndex(3, 5)], RangeIndex(3, 5)), ([RangeIndex(2), RangeIndex(2)], Index([0, 1, 0, 1])), ([RangeIndex(2), RangeIndex(2, 5), RangeIndex(5, 8, 4)], RangeIndex(0, 6)), ([RangeIndex(2), RangeIndex(3, 5), RangeIndex(5, 8, 4)], Index([0, 1, 3, 4, 5])), ([RangeIndex(-2, 2), RangeIndex(2, 5), RangeIndex(5, 8, 4)], RangeIndex(-2, 6)), ([RangeIndex(3), Index([-1, 3, 15])], Index([0, 1, 2, -1, 3, 15])), ([RangeIndex(3), Index([-1, 3.1, 15.0])], Index([0, 1, 2, -1, 3.1, 15.0])), ([RangeIndex(3), Index(['a', None, 14])], Index([0, 1, 2, 'a', None, 14])), ([RangeIndex(3, 1), Index(['a', None, 14])], Index(['a', None, 14]))])
    def test_append(self, indices: list[RangeIndex], expected: RangeIndex) -> None:
        ...

    def test_engineless_lookup(self) -> None:
        ...

    @pytest.mark.parametrize('ri', [RangeIndex(0, -1, -1), RangeIndex(0, 1, 1), RangeIndex(1, 3, 2), RangeIndex(0, -1, -2), RangeIndex(-3, -5, -2)])
    def test_append_len_one(self, ri: RangeIndex) -> None:
        ...

    @pytest.mark.parametrize('base', [RangeIndex(0, 2), Index([0, 1])])
    def test_isin_range(self, base: RangeIndex) -> None:
        ...

    def test_sort_values_key(self) -> None:
        ...

    def test_range_index_rsub_by_const(self) -> None:
        ...

@pytest.mark.parametrize('rng, decimals', [[range(5), 0], [range(5), 2], [range(10, 30, 10), -1], [range(30, 10, -10), -1]])
def test_range_round_returns_rangeindex(rng: range, decimals: int) -> None:
    ...

@pytest.mark.parametrize('rng, decimals', [[range(10, 30, 1), -1], [range(30, 10, -1), -1], [range(11, 14), -10]])
def test_range_round_returns_index(rng: range, decimals: int) -> None:
    ...

def test_reindex_1_value_returns_rangeindex() -> None:
    ...

def test_reindex_empty_returns_rangeindex() -> None:
    ...

def test_insert_empty_0_loc() -> None:
    ...

def test_append_non_rangeindex_return_rangeindex() -> None:
    ...

def test_append_non_rangeindex_return_index() -> None:
    ...

def test_reindex_returns_rangeindex() -> None:
    ...

def test_reindex_returns_index() -> None:
    ...

def test_take_return_rangeindex() -> None:
    ...

@pytest.mark.parametrize('rng, exp_rng', [[range(5), range(3, 4)], [range(0, -10, -2), range(-6, -8, -2)], [range(0, 10, 2), range(6, 8, 2)]])
def test_take_1_value_returns_rangeindex(rng: range, exp_rng: range) -> None:
    ...

def test_append_one_nonempty_preserve_step() -> None:
    ...

def test_getitem_boolmask_all_true() -> None:
    ...

def test_getitem_boolmask_all_false() -> None:
    ...

def test_getitem_boolmask_returns_rangeindex() -> None:
    ...

def test_getitem_boolmask_returns_index() -> None:
    ...

def test_getitem_boolmask_wrong_length() -> None:
    ...

def test_pos_returns_rangeindex() -> None:
    ...

def test_neg_returns_rangeindex() -> None:
    ...

@pytest.mark.parametrize('rng, exp_rng', [[range(0), range(0)], [range(10), range(10)], [range(-2, 1, 1), range(2, -1, -1)], [range(0, -10, -1), range(0, 10, 1)]])
def test_abs_returns_rangeindex(rng: range, exp_rng: range) -> None:
    ...

def test_abs_returns_index() -> None:
    ...

@pytest.mark.parametrize('rng', [range(0), range(5), range(0, -5, -1), range(-2, 2, 1), range(2, -2, -2), range(0, 5, 2)])
def test_invert_returns_rangeindex(rng: range) -> None:
    ...

@pytest.mark.parametrize('rng', [range(0, 5, 1), range(0, 5, 2), range(10, 15, 1), range(10, 5, -1), range(10, 5, -2), range(5, 0, -1)])
@pytest.mark.parametrize('meth', ['argmax', 'argmin'])
def test_arg_min_max(rng: range, meth: str) -> None:
    ...

@pytest.mark.parametrize('meth', ['argmin', 'argmax'])
def test_empty_argmin_argmax_raises(meth: str) -> None:
    ...

def test_getitem_integers_return_rangeindex() -> None:
    ...

def test_getitem_empty_return_rangeindex() -> None:
    ...

def test_getitem_integers_return_index() -> None:
    ...

@pytest.mark.parametrize('normalize', [True, False])
@pytest.mark.parametrize('rng', [range(3), range(0), range(0, 3, 2), range(3, -3, -2)])
def test_value_counts(sort: bool, dropna: bool, ascending: bool, normalize: bool, rng: range) -> None:
    ...

@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('value', [0, -5, 5, -3, np.array([-5, -3, 0, 5])])
def test_searchsorted(side: str, value: int | np.ndarray) -> None:
    ...