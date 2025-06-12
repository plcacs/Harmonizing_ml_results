import operator
import numpy as np
import pytest
import pandas._libs.sparse as splib
import pandas.util._test_decorators as td
from pandas import Series
import pandas._testing as tm
from pandas.core.arrays.sparse import BlockIndex, IntIndex, make_sparse_index
from typing import List, Tuple, Any

@pytest.fixture
def test_length() -> int:
    return 20

@pytest.fixture(
    params=[
        [[0, 7, 15], [3, 5, 5], [2, 9, 14], [2, 3, 5], [2, 9, 15], [1, 3, 4]],
        [[0, 5], [4, 4], [1], [4], [1], [3]],
        [[0], [10], [0, 5], [3, 7], [0, 5], [3, 5]],
        [[10], [5], [0, 12], [5, 3], [12], [3]],
        [[0, 10], [4, 6], [5, 17], [4, 2], [], []],
        [[0], [5], [], [], [], []],
    ],
    ids=[
        'plain_case',
        'delete_blocks',
        'split_blocks',
        'skip_block',
        'no_intersect',
        'one_empty'
    ]
)
def cases(request: Any) -> List[List[int]]:
    return request.param

class TestSparseIndexUnion:

    @pytest.mark.parametrize(
        'xloc, xlen, yloc, ylen, eloc, elen',
        [
            [[0], [5], [5], [4], [0], [9]],
            [[0, 10], [5, 5], [2, 17], [5, 2], [0, 10, 17], [7, 5, 2]],
            [[1], [5], [3], [5], [1], [7]],
            [[2, 10], [4, 4], [4], [8], [2], [12]],
            [[0, 5], [3, 5], [0], [7], [0], [10]],
            [[2, 10], [4, 4], [4, 13], [8, 4], [2], [15]],
            [[2], [15], [4, 9, 14], [3, 2, 2], [2], [15]],
            [[0, 10], [3, 3], [5, 15], [2, 2], [0, 5, 10, 15], [3, 2, 3, 2]],
        ]
    )
    def test_index_make_union(
        self,
        xloc: List[int],
        xlen: List[int],
        yloc: List[int],
        ylen: List[int],
        eloc: List[int],
        elen: List[int],
        test_length: int
    ) -> None:
        xindex: BlockIndex = BlockIndex(test_length, xloc, xlen)
        yindex: BlockIndex = BlockIndex(test_length, yloc, ylen)
        bresult: BlockIndex = xindex.make_union(yindex)
        assert isinstance(bresult, BlockIndex)
        tm.assert_numpy_array_equal(
            bresult.blocs, np.array(eloc, dtype=np.int32)
        )
        tm.assert_numpy_array_equal(
            bresult.blengths, np.array(elen, dtype=np.int32)
        )
        ixindex: IntIndex = xindex.to_int_index()
        iyindex: IntIndex = yindex.to_int_index()
        iresult: IntIndex = ixindex.make_union(iyindex)
        assert isinstance(iresult, IntIndex)
        tm.assert_numpy_array_equal(
            iresult.indices, bresult.to_int_index().indices
        )

    def test_int_index_make_union(self) -> None:
        a: IntIndex = IntIndex(5, np.array([0, 3, 4], dtype=np.int32))
        b: IntIndex = IntIndex(5, np.array([0, 2], dtype=np.int32))
        res: IntIndex = a.make_union(b)
        exp: IntIndex = IntIndex(5, np.array([0, 2, 3, 4], dtype=np.int32))
        assert res.equals(exp)
        
        a = IntIndex(5, np.array([], dtype=np.int32))
        b = IntIndex(5, np.array([0, 2], dtype=np.int32))
        res = a.make_union(b)
        exp = IntIndex(5, np.array([0, 2], dtype=np.int32))
        assert res.equals(exp)
        
        a = IntIndex(5, np.array([], dtype=np.int32))
        b = IntIndex(5, np.array([], dtype=np.int32))
        res = a.make_union(b)
        exp = IntIndex(5, np.array([], dtype=np.int32))
        assert res.equals(exp)
        
        a = IntIndex(5, np.array([0, 1, 2, 3, 4], dtype=np.int32))
        b = IntIndex(5, np.array([0, 1, 2, 3, 4], dtype=np.int32))
        res = a.make_union(b)
        exp = IntIndex(5, np.array([0, 1, 2, 3, 4], dtype=np.int32))
        assert res.equals(exp)
        
        a = IntIndex(5, np.array([0, 1], dtype=np.int32))
        b = IntIndex(4, np.array([0, 1], dtype=np.int32))
        msg: str = 'Indices must reference same underlying length'
        with pytest.raises(ValueError, match=msg):
            a.make_union(b)

class TestSparseIndexIntersect:

    @td.skip_if_windows
    def test_intersect(
        self,
        cases: List[List[int]],
        test_length: int
    ) -> None:
        xloc, xlen, yloc, ylen, eloc, elen = cases
        xindex: BlockIndex = BlockIndex(test_length, xloc, xlen)
        yindex: BlockIndex = BlockIndex(test_length, yloc, ylen)
        expected: BlockIndex = BlockIndex(test_length, eloc, elen)
        longer_index: BlockIndex = BlockIndex(test_length + 1, yloc, ylen)
        result: BlockIndex = xindex.intersect(yindex)
        assert result.equals(expected)
        result = xindex.to_int_index().intersect(yindex.to_int_index())
        assert result.equals(expected.to_int_index())
        msg: str = 'Indices must reference same underlying length'
        with pytest.raises(Exception, match=msg):
            xindex.intersect(longer_index)
        with pytest.raises(Exception, match=msg):
            xindex.to_int_index().intersect(longer_index.to_int_index())

    def test_intersect_empty(self) -> None:
        xindex: IntIndex = IntIndex(4, np.array([], dtype=np.int32))
        yindex: IntIndex = IntIndex(4, np.array([2, 3], dtype=np.int32))
        assert xindex.intersect(yindex).equals(xindex)
        assert yindex.intersect(xindex).equals(xindex)
        xindex = xindex.to_block_index()
        yindex = yindex.to_block_index()
        assert xindex.intersect(yindex).equals(xindex)
        assert yindex.intersect(xindex).equals(xindex)

    @pytest.mark.parametrize(
        'case',
        [
            IntIndex(5, np.array([1, 2], dtype=np.int32)),
            IntIndex(5, np.array([0, 2, 4], dtype=np.int32)),
            IntIndex(0, np.array([], dtype=np.int32)),
            IntIndex(5, np.array([], dtype=np.int32)),
        ]
    )
    def test_intersect_identical(
        self,
        case: IntIndex
    ) -> None:
        assert case.intersect(case).equals(case)
        case = case.to_block_index()
        assert case.intersect(case).equals(case)

class TestSparseIndexCommon:

    def test_int_internal(self) -> None:
        idx: IntIndex = make_sparse_index(
            4, np.array([2, 3], dtype=np.int32), kind='integer'
        )
        assert isinstance(idx, IntIndex)
        assert idx.npoints == 2
        tm.assert_numpy_array_equal(
            idx.indices, np.array([2, 3], dtype=np.int32)
        )
        idx = make_sparse_index(4, np.array([], dtype=np.int32), kind='integer')
        assert isinstance(idx, IntIndex)
        assert idx.npoints == 0
        tm.assert_numpy_array_equal(
            idx.indices, np.array([], dtype=np.int32)
        )
        idx = make_sparse_index(
            4, np.array([0, 1, 2, 3], dtype=np.int32), kind='integer'
        )
        assert isinstance(idx, IntIndex)
        assert idx.npoints == 4
        tm.assert_numpy_array_equal(
            idx.indices, np.array([0, 1, 2, 3], dtype=np.int32)
        )

    def test_block_internal(self) -> None:
        idx: BlockIndex = make_sparse_index(
            4, np.array([2, 3], dtype=np.int32), kind='block'
        )
        assert isinstance(idx, BlockIndex)
        assert idx.npoints == 2
        tm.assert_numpy_array_equal(
            idx.blocs, np.array([2], dtype=np.int32)
        )
        tm.assert_numpy_array_equal(
            idx.blengths, np.array([2], dtype=np.int32)
        )
        idx = make_sparse_index(
            4, np.array([], dtype=np.int32), kind='block'
        )
        assert isinstance(idx, BlockIndex)
        assert idx.npoints == 0
        tm.assert_numpy_array_equal(
            idx.blocs, np.array([], dtype=np.int32)
        )
        tm.assert_numpy_array_equal(
            idx.blengths, np.array([], dtype=np.int32)
        )
        idx = make_sparse_index(
            4, np.array([0, 1, 2, 3], dtype=np.int32), kind='block'
        )
        assert isinstance(idx, BlockIndex)
        assert idx.npoints == 4
        tm.assert_numpy_array_equal(
            idx.blocs, np.array([0], dtype=np.int32)
        )
        tm.assert_numpy_array_equal(
            idx.blengths, np.array([4], dtype=np.int32)
        )
        idx = make_sparse_index(
            4, np.array([0, 2, 3], dtype=np.int32), kind='block'
        )
        assert isinstance(idx, BlockIndex)
        assert idx.npoints == 3
        tm.assert_numpy_array_equal(
            idx.blocs, np.array([0, 2], dtype=np.int32)
        )
        tm.assert_numpy_array_equal(
            idx.blengths, np.array([1, 2], dtype=np.int32)
        )

    @pytest.mark.parametrize('kind', ['integer', 'block'])
    def test_lookup(
        self,
        kind: str
    ) -> None:
        idx: Any = make_sparse_index(
            4, np.array([2, 3], dtype=np.int32), kind=kind
        )
        assert idx.lookup(-1) == -1
        assert idx.lookup(0) == -1
        assert idx.lookup(1) == -1
        assert idx.lookup(2) == 0
        assert idx.lookup(3) == 1
        assert idx.lookup(4) == -1
        idx = make_sparse_index(
            4, np.array([], dtype=np.int32), kind=kind
        )
        for i in range(-1, 5):
            assert idx.lookup(i) == -1
        idx = make_sparse_index(
            4, np.array([0, 1, 2, 3], dtype=np.int32), kind=kind
        )
        assert idx.lookup(-1) == -1
        assert idx.lookup(0) == 0
        assert idx.lookup(1) == 1
        assert idx.lookup(2) == 2
        assert idx.lookup(3) == 3
        assert idx.lookup(4) == -1
        idx = make_sparse_index(
            4, np.array([0, 2, 3], dtype=np.int32), kind=kind
        )
        assert idx.lookup(-1) == -1
        assert idx.lookup(0) == 0
        assert idx.lookup(1) == -1
        assert idx.lookup(2) == 1
        assert idx.lookup(3) == 2
        assert idx.lookup(4) == -1

    @pytest.mark.parametrize(
        'idx, expected',
        [
            [0, -1],
            [5, 0],
            [7, 2],
            [8, -1],
            [9, -1],
            [10, -1],
            [11, -1],
            [12, 3],
            [17, 8],
            [18, -1],
        ]
    )
    def test_lookup_basics(
        self,
        idx: int,
        expected: int
    ) -> None:
        bindex: BlockIndex = BlockIndex(20, [5, 12], [3, 6])
        assert bindex.lookup(idx) == expected
        iindex: IntIndex = bindex.to_int_index()
        assert iindex.lookup(idx) == expected

class TestBlockIndex:

    def test_block_internal(self) -> None:
        idx: BlockIndex = make_sparse_index(
            4, np.array([2, 3], dtype=np.int32), kind='block'
        )
        assert isinstance(idx, BlockIndex)
        assert idx.npoints == 2
        tm.assert_numpy_array_equal(
            idx.blocs, np.array([2], dtype=np.int32)
        )
        tm.assert_numpy_array_equal(
            idx.blengths, np.array([2], dtype=np.int32)
        )
        idx = make_sparse_index(
            4, np.array([], dtype=np.int32), kind='block'
        )
        assert isinstance(idx, BlockIndex)
        assert idx.npoints == 0
        tm.assert_numpy_array_equal(
            idx.blocs, np.array([], dtype=np.int32)
        )
        tm.assert_numpy_array_equal(
            idx.blengths, np.array([], dtype=np.int32)
        )
        idx = make_sparse_index(
            4, np.array([0, 1, 2, 3], dtype=np.int32), kind='block'
        )
        assert isinstance(idx, BlockIndex)
        assert idx.npoints == 4
        tm.assert_numpy_array_equal(
            idx.blocs, np.array([0], dtype=np.int32)
        )
        tm.assert_numpy_array_equal(
            idx.blengths, np.array([4], dtype=np.int32)
        )
        idx = make_sparse_index(
            4, np.array([0, 2, 3], dtype=np.int32), kind='block'
        )
        assert isinstance(idx, BlockIndex)
        assert idx.npoints == 3
        tm.assert_numpy_array_equal(
            idx.blocs, np.array([0, 2], dtype=np.int32)
        )
        tm.assert_numpy_array_equal(
            idx.blengths, np.array([1, 2], dtype=np.int32)
        )

    @pytest.mark.parametrize('i', [5, 10, 100, 101])
    def test_make_block_boundary(
        self,
        i: int
    ) -> None:
        idx: BlockIndex = make_sparse_index(
            i, np.arange(0, i, 2, dtype=np.int32), kind='block'
        )
        exp: np.ndarray = np.arange(0, i, 2, dtype=np.int32)
        tm.assert_numpy_array_equal(idx.blocs, exp)
        tm.assert_numpy_array_equal(
            idx.blengths, np.ones(len(exp), dtype=np.int32)
        )

    def test_equals(self) -> None:
        index: BlockIndex = BlockIndex(10, [0, 4], [2, 5])
        assert index.equals(index)
        assert not index.equals(BlockIndex(10, [0, 4], [2, 6]))

    def test_check_integrity(self) -> None:
        locs: List[int] = []
        lengths: List[int] = []
        BlockIndex(0, locs, lengths)
        BlockIndex(1, locs, lengths)
        msg: str = 'Block 0 extends beyond end'
        with pytest.raises(ValueError, match=msg):
            BlockIndex(10, [5], [10])
        msg = 'Block 0 overlaps'
        with pytest.raises(ValueError, match=msg):
            BlockIndex(10, [2, 5], [5, 3])

    def test_to_int_index(self) -> None:
        locs: List[int] = [0, 10]
        lengths: List[int] = [4, 6]
        exp_inds: List[int] = [0, 1, 2, 3, 10, 11, 12, 13, 14, 15]
        block: BlockIndex = BlockIndex(20, locs, lengths)
        dense: IntIndex = block.to_int_index()
        tm.assert_numpy_array_equal(
            dense.indices, np.array(exp_inds, dtype=np.int32)
        )

    def test_to_block_index(self) -> None:
        index: BlockIndex = BlockIndex(10, [0, 5], [4, 5])
        assert index.to_block_index() is index

class TestIntIndex:

    def test_check_integrity(self) -> None:
        msg: str = 'Too many indices'
        with pytest.raises(ValueError, match=msg):
            IntIndex(length=1, indices=[1, 2, 3])
        msg = 'No index can be less than zero'
        with pytest.raises(ValueError, match=msg):
            IntIndex(length=5, indices=[1, -2, 3])
        msg = 'No index can be less than zero'
        with pytest.raises(ValueError, match=msg):
            IntIndex(length=5, indices=[1, -2, 3])
        msg = 'All indices must be less than the length'
        with pytest.raises(ValueError, match=msg):
            IntIndex(length=5, indices=[1, 2, 5])
        with pytest.raises(ValueError, match=msg):
            IntIndex(length=5, indices=[1, 2, 6])
        msg = 'Indices must be strictly increasing'
        with pytest.raises(ValueError, match=msg):
            IntIndex(length=5, indices=[1, 3, 2])
        with pytest.raises(ValueError, match=msg):
            IntIndex(length=5, indices=[1, 3, 3])

    def test_int_internal(self) -> None:
        idx: IntIndex = make_sparse_index(
            4, np.array([2, 3], dtype=np.int32), kind='integer'
        )
        assert isinstance(idx, IntIndex)
        assert idx.npoints == 2
        tm.assert_numpy_array_equal(
            idx.indices, np.array([2, 3], dtype=np.int32)
        )
        idx = make_sparse_index(
            4, np.array([], dtype=np.int32), kind='integer'
        )
        assert isinstance(idx, IntIndex)
        assert idx.npoints == 0
        tm.assert_numpy_array_equal(
            idx.indices, np.array([], dtype=np.int32)
        )
        idx = make_sparse_index(
            4, np.array([0, 1, 2, 3], dtype=np.int32), kind='integer'
        )
        assert isinstance(idx, IntIndex)
        assert idx.npoints == 4
        tm.assert_numpy_array_equal(
            idx.indices, np.array([0, 1, 2, 3], dtype=np.int32)
        )

    def test_equals(self) -> None:
        index: IntIndex = IntIndex(10, [0, 1, 2, 3, 4])
        assert index.equals(index)
        assert not index.equals(IntIndex(10, [0, 1, 2, 3]))

    def test_to_block_index(
        self,
        cases: List[List[int]],
        test_length: int
    ) -> None:
        xloc, xlen, yloc, ylen, _, _ = cases
        xindex: BlockIndex = BlockIndex(test_length, xloc, xlen)
        yindex: BlockIndex = BlockIndex(test_length, yloc, ylen)
        xbindex: BlockIndex = xindex.to_int_index().to_block_index()
        ybindex: BlockIndex = yindex.to_int_index().to_block_index()
        assert isinstance(xbindex, BlockIndex)
        assert xbindex.equals(xindex)
        assert ybindex.equals(yindex)

    def test_to_int_index(self) -> None:
        index: IntIndex = IntIndex(10, [2, 3, 4, 5, 6])
        assert index.to_int_index() is index

class TestSparseOperators:

    @pytest.mark.parametrize(
        'opname',
        ['add', 'sub', 'mul', 'truediv', 'floordiv']
    )
    def test_op(
        self,
        opname: str,
        cases: List[List[int]],
        test_length: int
    ) -> None:
        sparse_op = getattr(splib, f'sparse_{opname}_float64')
        python_op = getattr(operator, opname)
        xloc, xlen, yloc, ylen, _, _ = cases
        xindex: BlockIndex = BlockIndex(test_length, xloc, xlen)
        yindex: BlockIndex = BlockIndex(test_length, yloc, ylen)
        xdindex: IntIndex = xindex.to_int_index()
        ydindex: IntIndex = yindex.to_int_index()
        x: np.ndarray = np.arange(xindex.npoints) * 10.0 + 1
        y: np.ndarray = np.arange(yindex.npoints) * 100.0 + 1
        xfill: float = 0
        yfill: float = 2
        result_block_vals: np.ndarray
        rb_index: Any
        bfill: float
        result_block_vals, rb_index, bfill = sparse_op(
            x, xindex, xfill, y, yindex, yfill
        )
        result_int_vals: np.ndarray
        ri_index: Any
        ifill: float
        result_int_vals, ri_index, ifill = sparse_op(
            x, xdindex, xfill, y, ydindex, yfill
        )
        assert rb_index.to_int_index().equals(ri_index)
        tm.assert_numpy_array_equal(result_block_vals, result_int_vals)
        assert bfill == ifill
        xseries: Series = Series(x, xdindex.indices)
        xseries = xseries.reindex(np.arange(test_length)).fillna(xfill)
        yseries: Series = Series(y, ydindex.indices)
        yseries = yseries.reindex(np.arange(test_length)).fillna(yfill)
        series_result: Series = python_op(xseries, yseries)
        series_result = series_result.reindex(ri_index.indices)
        tm.assert_numpy_array_equal(result_block_vals, series_result.values)
        tm.assert_numpy_array_equal(result_int_vals, series_result.values)
