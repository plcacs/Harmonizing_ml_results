import numpy as np
import pytest
from typing import Any, Tuple, List
from pandas._libs import index as libindex
import pandas as pd
from pandas import Index, NaT
import pandas._testing as tm

class TestGetSliceBounds:
    @pytest.mark.parametrize('side, expected', [('left', 4), ('right', 5)])
    def test_get_slice_bounds_within(self, side: str, expected: int) -> None:
        index: Index = Index(list('abcdef'))
        result: int = index.get_slice_bound('e', side=side)
        assert result == expected

    @pytest.mark.parametrize('side', ['left', 'right'])
    @pytest.mark.parametrize('data, bound, expected', [(list('abcdef'), 'x', 6), (list('bcdefg'), 'a', 0)])
    def test_get_slice_bounds_outside(self, side: str, expected: int, data: List[Any], bound: Any) -> None:
        index: Index = Index(data)
        result: int = index.get_slice_bound(bound, side=side)
        assert result == expected

    def test_get_slice_bounds_invalid_side(self) -> None:
        with pytest.raises(ValueError, match='Invalid value for side kwarg'):
            Index([]).get_slice_bound('a', side='middle')


class TestGetIndexerNonUnique:
    def test_get_indexer_non_unique_dtype_mismatch(self) -> None:
        indexes, missing = Index(['A', 'B']).get_indexer_non_unique(Index([0]))
        tm.assert_numpy_array_equal(np.array([-1], dtype=np.intp), indexes)
        tm.assert_numpy_array_equal(np.array([0], dtype=np.intp), missing)

    @pytest.mark.parametrize('idx_values,idx_non_unique', [
        ([np.nan, 100, 200, 100], [np.nan, 100]),
        ([np.nan, 100.0, 200.0, 100.0], [np.nan, 100.0])
    ])
    def test_get_indexer_non_unique_int_index(self, idx_values: List[Any], idx_non_unique: List[Any]) -> None:
        indexes, missing = Index(idx_values).get_indexer_non_unique(Index([np.nan]))
        tm.assert_numpy_array_equal(np.array([0], dtype=np.intp), indexes)
        tm.assert_numpy_array_equal(np.array([], dtype=np.intp), missing)
        indexes, missing = Index(idx_values).get_indexer_non_unique(Index(idx_non_unique))
        tm.assert_numpy_array_equal(np.array([0, 1, 3], dtype=np.intp), indexes)
        tm.assert_numpy_array_equal(np.array([], dtype=np.intp), missing)


class TestGetLoc:
    @pytest.mark.slow
    def test_get_loc_tuple_monotonic_above_size_cutoff(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with monkeypatch.context() as m:
            m.setattr(libindex, '_SIZE_CUTOFF', 100)
            lev: List[str] = list('ABCD')
            dti: pd.DatetimeIndex = pd.date_range('2016-01-01', periods=10)
            mi: pd.MultiIndex = pd.MultiIndex.from_product([lev, list(range(5)), dti])
            oidx: Index = mi.to_flat_index()
            loc: int = len(oidx) // 2
            tup: Any = oidx[loc]
            res: int = oidx.get_loc(tup)
        assert res == loc

    def test_get_loc_nan_object_dtype_nonmonotonic_nonunique(self) -> None:
        idx: Index = Index(['foo', np.nan, None, 'foo', 1.0, None], dtype=object)
        res = idx.get_loc(np.nan)
        assert res == 1
        res = idx.get_loc(None)
        expected: np.ndarray = np.array([False, False, True, False, False, True])
        tm.assert_numpy_array_equal(res, expected)
        with pytest.raises(KeyError, match='NaT'):
            idx.get_loc(NaT)


def test_getitem_boolean_ea_indexer() -> None:
    ser: pd.Series = pd.Series([True, False, pd.NA], dtype='boolean')
    result: Index = ser.index[ser]
    expected: Index = Index([0])
    tm.assert_index_equal(result, expected)