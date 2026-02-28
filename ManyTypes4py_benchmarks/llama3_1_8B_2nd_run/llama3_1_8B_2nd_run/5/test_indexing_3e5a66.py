import numpy as np
import pytest
from pandas._libs import index as libindex
import pandas as pd
from pandas import Index, NaT
import pandas._testing as tm

class TestGetSliceBounds:
    """Test cases for get_slice_bound method of Index class."""

    @pytest.mark.parametrize('side, expected', [('left', 4), ('right', 5)])
    def test_get_slice_bounds_within(self, side: str, expected: int) -> None:
        """Test get_slice_bound method within bounds."""
        index: Index = Index(list('abcdef'))
        result: int = index.get_slice_bound('e', side=side)
        assert result == expected

    @pytest.mark.parametrize('side', ['left', 'right'])
    @pytest.mark.parametrize('data, bound, expected', [(list('abcdef'), 'x', 6), (list('bcdefg'), 'a', 0)])
    def test_get_slice_bounds_outside(self, side: str, expected: int, data: list, bound: str) -> None:
        """Test get_slice_bound method outside bounds."""
        index: Index = Index(data)
        result: int = index.get_slice_bound(bound, side=side)
        assert result == expected

    def test_get_slice_bounds_invalid_side(self) -> None:
        """Test get_slice_bound method with invalid side."""
        with pytest.raises(ValueError, match='Invalid value for side kwarg'):
            Index([]).get_slice_bound('a', side='middle')

class TestGetIndexerNonUnique:
    """Test cases for get_indexer_non_unique method of Index class."""

    def test_get_indexer_non_unique_dtype_mismatch(self) -> None:
        """Test get_indexer_non_unique method with dtype mismatch."""
        indexes, missing = Index(['A', 'B']).get_indexer_non_unique(Index([0]))
        tm.assert_numpy_array_equal(np.array([-1], dtype=np.intp), indexes)
        tm.assert_numpy_array_equal(np.array([0], dtype=np.intp), missing)

    @pytest.mark.parametrize('idx_values,idx_non_unique', [([np.nan, 100, 200, 100], [np.nan, 100]), ([np.nan, 100.0, 200.0, 100.0], [np.nan, 100.0])])
    def test_get_indexer_non_unique_int_index(self, idx_values: list, idx_non_unique: list) -> None:
        """Test get_indexer_non_unique method with int index."""
        indexes, missing = Index(idx_values).get_indexer_non_unique(Index([np.nan]))
        tm.assert_numpy_array_equal(np.array([0], dtype=np.intp), indexes)
        tm.assert_numpy_array_equal(np.array([], dtype=np.intp), missing)
        indexes, missing = Index(idx_values).get_indexer_non_unique(Index(idx_non_unique))
        tm.assert_numpy_array_equal(np.array([0, 1, 3], dtype=np.intp), indexes)
        tm.assert_numpy_array_equal(np.array([], dtype=np.intp), missing)

class TestGetLoc:
    """Test cases for get_loc method of Index class."""

    @pytest.mark.slow
    def test_get_loc_tuple_monotonic_above_size_cutoff(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test get_loc method with monotonic above size cutoff."""
        with monkeypatch.context():
            monkeypatch.setattr(libindex, '_SIZE_CUTOFF', 100)
            lev = list('ABCD')
            dti = pd.date_range('2016-01-01', periods=10)
            mi = pd.MultiIndex.from_product([lev, range(5), dti])
            oidx = mi.to_flat_index()
            loc: int = len(oidx) // 2
            tup = oidx[loc]
            res = oidx.get_loc(tup)
        assert res == loc

    def test_get_loc_nan_object_dtype_nonmonotonic_nonunique(self) -> None:
        """Test get_loc method with nan object dtype nonmonotonic nonunique."""
        idx = Index(['foo', np.nan, None, 'foo', 1.0, None], dtype=object)
        res = idx.get_loc(np.nan)
        assert res == 1
        res = idx.get_loc(None)
        expected = np.array([False, False, True, False, False, True])
        tm.assert_numpy_array_equal(res, expected)
        with pytest.raises(KeyError, match='NaT'):
            idx.get_loc(NaT)

def test_getitem_boolean_ea_indexer() -> None:
    """Test getitem method with boolean EA indexer."""
    ser = pd.Series([True, False, pd.NA], dtype='boolean')
    result = ser.index[ser]
    expected = Index([0])
    tm.assert_index_equal(result, expected)
