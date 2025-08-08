import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import NA, Index, RangeIndex, Series, Timestamp
import pandas._testing as tm
from pandas.core.arrays import ArrowExtensionArray, FloatingArray

class TestGetLoc:

    def test_get_loc(self) -> None:
        index: Index = Index([0, 1, 2])
        assert index.get_loc(1) == 1

    def test_get_loc_raises_bad_label(self) -> None:
        index: Index = Index([0, 1, 2])
        with pytest.raises(InvalidIndexError, match='\\[1, 2\\]'):
            index.get_loc([1, 2])

    def test_get_loc_float64(self) -> None:
        idx: Index = Index([0.0, 1.0, 2.0], dtype=np.float64)
        with pytest.raises(KeyError, match="^'foo'$"):
            idx.get_loc('foo')
        with pytest.raises(KeyError, match='^1\\.5$'):
            idx.get_loc(1.5)
        with pytest.raises(KeyError, match='^True$'):
            idx.get_loc(True)
        with pytest.raises(KeyError, match='^False$'):
            idx.get_loc(False)

    def test_get_loc_na(self) -> None:
        idx: Index = Index([np.nan, 1, 2], dtype=np.float64)
        assert idx.get_loc(1) == 1
        assert idx.get_loc(np.nan) == 0
        idx: Index = Index([np.nan, 1, np.nan], dtype=np.float64)
        assert idx.get_loc(1) == 1
        msg: str = "'Cannot get left slice bound for non-unique label: nan'"
        with pytest.raises(KeyError, match=msg):
            idx.slice_locs(np.nan)
        idx: Index = Index([np.nan, 1, np.nan, np.nan], dtype=np.float64)
        assert idx.get_loc(1) == 1
        msg: str = "'Cannot get left slice bound for non-unique label: nan"
        with pytest.raises(KeyError, match=msg):
            idx.slice_locs(np.nan)

    def test_get_loc_missing_nan(self) -> None:
        idx: Index = Index([1, 2], dtype=np.float64)
        assert idx.get_loc(1) == 0
        with pytest.raises(KeyError, match='^3$'):
            idx.get_loc(3)
        with pytest.raises(KeyError, match='^nan$'):
            idx.get_loc(np.nan)
        with pytest.raises(InvalidIndexError, match='\\[nan\\]'):
            idx.get_loc([np.nan])

    @pytest.mark.parametrize('vals', [[1], [1.0], [Timestamp('2019-12-31')], ['test']])
    def test_get_loc_float_index_nan_with_method(self, vals: list) -> None:
        idx: Index = Index(vals)
        with pytest.raises(KeyError, match='nan'):
            idx.get_loc(np.nan)

    @pytest.mark.parametrize('dtype', ['f8', 'i8', 'u8'])
    def test_get_loc_numericindex_none_raises(self, dtype: str) -> None:
        arr: np.ndarray = np.arange(10 ** 7, dtype=dtype)
        idx: Index = Index(arr)
        with pytest.raises(KeyError, match='None'):
            idx.get_loc(None)

    def test_get_loc_overflows(self) -> None:
        idx: Index = Index([0, 2, 1])
        val: int = np.iinfo(np.int64).max + 1
        with pytest.raises(KeyError, match=str(val)):
            idx.get_loc(val)
        with pytest.raises(KeyError, match=str(val)):
            idx._engine.get_loc(val)

class TestGetIndexer:

    def test_get_indexer(self) -> None:
        index1: Index = Index([1, 2, 3, 4, 5])
        index2: Index = Index([2, 4, 6])
        r1: np.ndarray = index1.get_indexer(index2)
        e1: np.ndarray = np.array([1, 3, -1], dtype=np.intp)
        tm.assert_almost_equal(r1, e1)

    @pytest.mark.parametrize('reverse', [True, False])
    @pytest.mark.parametrize('expected,method', [([-1, 0, 0, 1, 1], 'pad'), ([-1, 0, 0, 1, 1], 'ffill'), ([0, 0, 1, 1, 2], 'backfill'), ([0, 0, 1, 1, 2], 'bfill')])
    def test_get_indexer_methods(self, reverse: bool, expected: list, method: str) -> None:
        index1: Index = Index([1, 2, 3, 4, 5])
        index2: Index = Index([2, 4, 6])
        expected: np.ndarray = np.array(expected, dtype=np.intp)
        if reverse:
            index1 = index1[::-1]
            expected = expected[::-1]
        result: np.ndarray = index2.get_indexer(index1, method=method)
        tm.assert_almost_equal(result, expected)

    def test_get_indexer_invalid(self) -> None:
        index: Index = Index(np.arange(10))
        with pytest.raises(ValueError, match='tolerance argument'):
            index.get_indexer([1, 0], tolerance=1)
        with pytest.raises(ValueError, match='limit argument'):
            index.get_indexer([1, 0], limit=1)

    @pytest.mark.parametrize('method, tolerance, indexer, expected', [('pad', None, [0, 5, 9], [0, 5, 9]), ('backfill', None, [0, 5, 9], [0, 5, 9]), ('nearest', None, [0, 5, 9], [0, 5, 9]), ('pad', 0, [0, 5, 9], [0, 5, 9]), ('backfill', 0, [0, 5, 9], [0, 5, 9]), ('nearest', 0, [0, 5, 9], [0, 5, 9]), ('pad', None, [0.2, 1.8, 8.5], [0, 1, 8]), ('backfill', None, [0.2, 1.8, 8.5], [1, 2, 9]), ('nearest', None, [0.2, 1.8, 8.5], [0, 2, 9]), ('pad', 1, [0.2, 1.8, 8.5], [0, 1, 8]), ('backfill', 1, [0.2, 1.8, 8.5], [1, 2, 9]), ('nearest', 1, [0.2, 1.8, 8.5], [0, 2, 9]), ('pad', 0.2, [0.2, 1.8, 8.5], [0, -1, -1]), ('backfill', 0.2, [0.2, 1.8, 8.5], [-1, 2, -1]), ('nearest', 0.2, [0.2, 1.8, 8.5], [0, 2, -1])])
    def test_get_indexer_nearest(self, method: str, tolerance: int, indexer: list, expected: list) -> None:
        index: Index = Index(np.arange(10))
        actual: np.ndarray = index.get_indexer(indexer, method=method, tolerance=tolerance)
        tm.assert_numpy_array_equal(actual, np.array(expected, dtype=np.intp))

    @pytest.mark.parametrize('listtype', [list, tuple, Series, np.array])
    @pytest.mark.parametrize('tolerance, expected', [[[0.3, 0.3, 0.1], [0, 2, -1]], [[0.2, 0.1, 0.1], [0, -1, -1]], [[0.1, 0.5, 0.5], [-1, 2, 9]]])
    def test_get_indexer_nearest_listlike_tolerance(self, tolerance: list, expected: list, listtype: type) -> None:
        index: Index = Index(np.arange(10))
        actual: np.ndarray = index.get_indexer([0.2, 1.8, 8.5], method='nearest', tolerance=listtype(tolerance))
        tm.assert_numpy_array_equal(actual, np.array(expected, dtype=np.intp))

    def test_get_indexer_nearest_error(self) -> None:
        index: Index = Index(np.arange(10))
        with pytest.raises(ValueError, match='limit argument'):
            index.get_indexer([1, 0], method='nearest', limit=1)
        with pytest.raises(ValueError, match='tolerance size must match'):
            index.get_indexer([1, 0], method='nearest', tolerance=[1, 2, 3])

    @pytest.mark.parametrize('method, expected', [('pad', [8, 7, 0]), ('backfill', [9, 8, 1]), ('nearest', [9, 7, 0])])
    def test_get_indexer_nearest_decreasing(self, method: str, expected: list) -> None:
        index: Index = Index(np.arange(10))[::-1]
        actual: np.ndarray = index.get_indexer([0, 5, 9], method=method)
        tm.assert_numpy_array_equal(actual, np.array([9, 4, 0], dtype=np.intp))
        actual: np.ndarray = index.get_indexer([0.2, 1.8, 8.5], method=method)
        tm.assert_numpy_array_equal(actual, np.array(expected, dtype=np.intp))

    @pytest.mark.parametrize('idx_dtype', ['int64', 'float64', 'uint64', 'range'])
    @pytest.mark.parametrize('method', ['get_indexer', 'get_indexer_non_unique'])
    def test_get_indexer_numeric_index_boolean_target(self, method: str, idx_dtype: str) -> None:
        if idx_dtype == 'range':
            numeric_index: RangeIndex = RangeIndex(4)
        else:
            numeric_index: Index = Index(np.arange(4, dtype=idx_dtype)
        other: Index = Index([True, False, True])
        result: np.ndarray = getattr(numeric_index, method)(other)
        expected: np.ndarray = np.array([-1, -1, -1], dtype=np.intp)
        if method == 'get_indexer':
            tm.assert_numpy_array_equal(result, expected)
        else:
            missing: np.ndarray = np.arange(3, dtype=np.intp)
            tm.assert_numpy_array_equal(result[0], expected)
            tm.assert_numpy_array_equal(result[1], missing)

    @pytest.mark.parametrize('method', ['pad', 'backfill', 'nearest'])
    def test_get_indexer_with_method_numeric_vs_bool(self, method: str) -> None:
        left: Index = Index([1, 2, 3])
        right: Index = Index([True, False])
        with pytest.raises(TypeError, match='Cannot compare'):
            left.get_indexer(right, method=method)
        with pytest.raises(TypeError, match='Cannot compare'):
            right.get_indexer(left, method=method)

    def test_get_indexer_numeric_vs_bool(self) -> None:
        left: Index = Index([1, 2, 3])
        right: Index = Index([True, False])
        res: np.ndarray = left.get_indexer(right)
        expected: np.ndarray = -1 * np.ones(len(right), dtype=np.intp)
        tm.assert_numpy_array_equal(res, expected)
        res: np.ndarray = right.get_indexer(left)
        expected: np.ndarray = -1 * np.ones(len(left), dtype=np.intp)
        tm.assert_numpy_array_equal(res, expected)
        res: np.ndarray = left.get_indexer_non_unique(right)[0]
        expected: np.ndarray = -1 * np.ones(len(right), dtype=np.intp)
        tm.assert_numpy_array_equal(res, expected)
        res: np.ndarray = right.get_indexer_non_unique(left)[0]
        expected: np.ndarray = -1 * np.ones(len(left), dtype=np.intp)
        tm.assert_numpy_array_equal(res, expected)

    def test_get_indexer_float64(self) -> None:
        idx: Index = Index([0.0, 1.0, 2.0], dtype=np.float64)
        tm.assert_numpy_array_equal(idx.get_indexer(idx), np.array([0, 1, 2], dtype=np.intp))
        target: list = [-0.1, 0.5, 1.1]
        tm.assert_numpy_array_equal(idx.get_indexer(target, 'pad'), np.array([-1, 0, 1], dtype=np.intp))
        tm.assert_numpy_array_equal(idx.get_indexer(target, 'backfill'), np.array([0, 1, 2], dtype=np.intp))
        tm.assert_numpy_array_equal(idx.get_indexer(target, 'nearest'), np.array([0, 1, 1], dtype=np.intp))

    def test_get_indexer_nan(self) -> None:
        result: np.ndarray = Index([1, 2, np.nan], dtype=np.float64).get_indexer([np.nan])
        expected: np.ndarray = np.array([2], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

    def test_get_indexer_int64(self) -> None:
        index: Index = Index(range(0, 20, 2), dtype=np.int64)
        target: Index = Index(np.arange(10), dtype=np.int64)
        indexer: np.ndarray = index.get_indexer(target)
        expected: np.ndarray = np.array([0, -1, 1, -1, 2, -1, 3, -1, 4, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, expected)
        target: Index = Index(np.arange(10), dtype=np.int64)
        indexer: np.ndarray = index.get_indexer(target, method='pad')
        expected: np.ndarray = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, expected)
        target: Index = Index(np.arange(10), dtype=np.int64)
        indexer: np.ndarray = index.get_indexer(target, method='backfill')
        expected: np.ndarray = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, expected)

    def test_get_indexer_uint64(self) -> None:
        index_large: Index = Index([2 ** 63, 2 ** 63 + 10, 2 ** 63 + 15, 2 ** 63 + 20, 2 ** 63 + 25], dtype=np.uint64)
        target: Index = Index(np.arange(10).astype('uint64') * 5 + 2 ** 63)
        indexer: np.ndarray = index_large.get_indexer(target)
        expected: np.ndarray = np.array([0, -1, 1, 2, 3, 4, -1, -1, -1, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, expected)
        target: Index = Index(np.arange(10).astype('uint64') * 5 + 2 ** 63)
        indexer: np.ndarray = index_large.get_indexer(target, method='pad')
        expected: np.ndarray = np.array([0, 0, 1, 2, 3, 4, 4, 4, 4, 4], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, expected)
        target: Index = Index(np.arange(10).astype('uint64') * 5 + 2 ** 63)
        indexer: np.ndarray = index_large.get_indexer(target, method='backfill')
        expected: np.ndarray = np.array([0, 1, 1, 2, 3, 4, -1, -1, -1, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, expected)

    @pytest.mark.parametrize('val, val2', [(4, 5), (4, 4), (4, NA), (NA, NA)])
    def test_get_loc_masked(self, val: int, val2: int, any_numeric_ea_and_arrow_dtype: str) -> None:
        idx: Index = Index([1, 2, 3, val, val2], dtype=any_numeric_ea_and_arrow_dtype)
        result: int = idx.get_loc(2)
        assert result == 1
        with pytest.raises(KeyError, match='9'):
            idx.get_loc(9)

    def test_get_loc_masked_na(self, any_numeric_ea_and_arrow_dtype: str) -> None:
        idx: Index = Index([1, 2, NA], dtype=any_numeric_ea_and_arrow_dtype)
        result: int = idx.get_loc(NA)
        assert result == 2
        idx: Index = Index([1, 2, NA, NA], dtype=any_numeric_ea_and_arrow_dtype)
        result: np.ndarray = idx.get_loc(NA)
        tm.assert_numpy_array_equal(result, np.array([False, False, True, True]))
        idx: Index = Index([1, 2, 3], dtype=any_numeric_ea_and_arrow_dtype)
        with pytest.raises(KeyError, match='NA'):
            idx.get_loc(NA)

    def test_get_loc_masked_na_and_nan(self) -> None:
        idx: Index = Index(FloatingArray(np.array([1, 2, 1, np.nan]), mask=np.array([False, False, True, False]))
        result: int = idx.get_loc(NA)
        assert result == 2
        result: int = idx.get_loc(np.nan)
        assert result == 3
        idx: Index = Index(FloatingArray(np.array([1, 2, 1.0]), mask=np.array([False, False, True]))
        result: int = idx.get_loc(NA)
        assert result == 2
        with pytest.raises(KeyError, match='nan'):
            idx.get_loc(np.nan)
        idx: Index = Index(FloatingArray(np.array([1, 