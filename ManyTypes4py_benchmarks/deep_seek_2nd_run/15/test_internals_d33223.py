from datetime import date, datetime
import itertools
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union, cast
import numpy as np
import pytest
from pandas._libs.internals import BlockPlacement
from pandas.compat import IS64
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import Categorical, DataFrame, DatetimeIndex, Index, IntervalIndex, Series, Timedelta, Timestamp, period_range
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import DatetimeArray, SparseArray, TimedeltaArray
from pandas.core.internals import BlockManager, SingleBlockManager, make_block
from pandas.core.internals.blocks import ensure_block_shape, maybe_coerce_values, new_block
from pandas.core.internals.managers import BlockManager, SingleBlockManager
from pandas.core.indexes.base import Index
from pandas.core.internals.blocks import Block
from numpy.typing import NDArray

@pytest.fixture(params=[new_block, make_block])
def block_maker(request: pytest.FixtureRequest) -> Callable[..., Block]:
    """
    Fixture to test both the internal new_block and pseudo-public make_block.
    """
    return request.param

@pytest.fixture
def mgr() -> BlockManager:
    return create_mgr('a: f8; b: object; c: f8; d: object; e: f8;f: bool; g: i8; h: complex; i: datetime-1; j: datetime-2;k: M8[ns, US/Eastern]; l: M8[ns, CET];')

def assert_block_equal(left: Block, right: Block) -> None:
    tm.assert_numpy_array_equal(left.values, right.values)
    assert left.dtype == right.dtype
    assert isinstance(left.mgr_locs, BlockPlacement)
    assert isinstance(right.mgr_locs, BlockPlacement)
    tm.assert_numpy_array_equal(left.mgr_locs.as_array, right.mgr_locs.as_array)

def get_numeric_mat(shape: Tuple[int, ...]) -> NDArray[np.int64]:
    arr = np.arange(shape[0])
    return np.lib.stride_tricks.as_strided(x=arr, shape=shape, strides=(arr.itemsize,) + (0,) * (len(shape) - 1)).copy()
N = 10

def create_block(typestr: str, placement: Union[Sequence[int], slice], item_shape: Optional[Tuple[int, ...]] = None, num_offset: int = 0, maker: Callable[..., Block] = new_block) -> Block:
    """
    Supported typestr:

        * float, f8, f4, f2
        * int, i8, i4, i2, i1
        * uint, u8, u4, u2, u1
        * complex, c16, c8
        * bool
        * object, string, O
        * datetime, dt, M8[ns], M8[ns, tz]
        * timedelta, td, m8[ns]
        * sparse (SparseArray with fill_value=0.0)
        * sparse_na (SparseArray with fill_value=np.nan)
        * category, category2

    """
    placement_obj = BlockPlacement(placement)
    num_items = len(placement_obj)
    if item_shape is None:
        item_shape = (N,)
    shape = (num_items,) + item_shape
    mat = get_numeric_mat(shape)
    if typestr in ('float', 'f8', 'f4', 'f2', 'int', 'i8', 'i4', 'i2', 'i1', 'uint', 'u8', 'u4', 'u2', 'u1'):
        values = mat.astype(typestr) + num_offset
    elif typestr in ('complex', 'c16', 'c8'):
        values = 1j * (mat.astype(typestr) + num_offset
    elif typestr in ('object', 'string', 'O'):
        values = np.reshape([f'A{i:d}' for i in mat.ravel() + num_offset], shape)
    elif typestr in ('b', 'bool'):
        values = np.ones(shape, dtype=np.bool_)
    elif typestr in ('datetime', 'dt', 'M8[ns]'):
        values = (mat * 1000000000.0).astype('M8[ns]')
    elif typestr.startswith('M8[ns'):
        m = re.search('M8\\[ns,\\s*(\\w+\\/?\\w*)\\]', typestr)
        assert m is not None, f'incompatible typestr -> {typestr}'
        tz = m.groups()[0]
        assert num_items == 1, 'must have only 1 num items for a tz-aware'
        values = DatetimeIndex(np.arange(N) * 10 ** 9, tz=tz)._data
        values = ensure_block_shape(values, ndim=len(shape))
    elif typestr in ('timedelta', 'td', 'm8[ns]'):
        values = (mat * 1).astype('m8[ns]')
    elif typestr in ('category',):
        values = Categorical([1, 1, 2, 2, 3, 3, 3, 3, 4, 4])
    elif typestr in ('category2',):
        values = Categorical(['a', 'a', 'a', 'a', 'b', 'b', 'c', 'c', 'c', 'd'])
    elif typestr in ('sparse', 'sparse_na'):
        if shape[-1] != 10:
            raise NotImplementedError
        assert all((s == 1 for s in shape[:-1]))
        if typestr.endswith('_na'):
            fill_value = np.nan
        else:
            fill_value = 0.0
        values = SparseArray([fill_value, fill_value, 1, 2, 3, fill_value, 4, 5, fill_value, 6], fill_value=fill_value)
        arr = values.sp_values.view()
        arr += num_offset - 1
    else:
        raise ValueError(f'Unsupported typestr: "{typestr}"')
    values = maybe_coerce_values(values)
    return maker(values, placement=placement_obj, ndim=len(shape))

def create_single_mgr(typestr: str, num_rows: Optional[int] = None) -> SingleBlockManager:
    if num_rows is None:
        num_rows = N
    return SingleBlockManager(create_block(typestr, placement=slice(0, num_rows), item_shape=()), Index(np.arange(num_rows)))

def create_mgr(descr: str, item_shape: Optional[Tuple[int, ...]] = None) -> BlockManager:
    """
    Construct BlockManager from string description.

    String description syntax looks similar to np.matrix initializer.  It looks
    like this::

        a,b,c: f8; d,e,f: i8

    Rules are rather simple:

    * see list of supported datatypes in `create_block` method
    * components are semicolon-separated
    * each component is `NAME,NAME,NAME: DTYPE_ID`
    * whitespace around colons & semicolons are removed
    * components with same DTYPE_ID are combined into single block
    * to force multiple blocks with same dtype, use '-SUFFIX'::

        "a:f8-1; b:f8-2; c:f8-foobar"

    """
    if item_shape is None:
        item_shape = (N,)
    offset = 0
    mgr_items = []
    block_placements: Dict[str, List[int]] = {}
    for d in descr.split(';'):
        d = d.strip()
        if not len(d):
            continue
        names, blockstr = d.partition(':')[::2]
        blockstr = blockstr.strip()
        names = names.strip().split(',')
        mgr_items.extend(names)
        placement = list(np.arange(len(names)) + offset)
        try:
            block_placements[blockstr].extend(placement)
        except KeyError:
            block_placements[blockstr] = placement
        offset += len(names)
    mgr_items_idx = Index(mgr_items)
    blocks = []
    num_offset = 0
    for blockstr, placement in block_placements.items():
        typestr = blockstr.split('-')[0]
        blocks.append(create_block(typestr, placement, item_shape=item_shape, num_offset=num_offset))
        num_offset += len(placement)
    sblocks = sorted(blocks, key=lambda b: b.mgr_locs[0])
    return BlockManager(tuple(sblocks), [mgr_items_idx] + [Index(np.arange(n)) for n in item_shape])

@pytest.fixture
def fblock() -> Block:
    return create_block('float', [0, 2, 4])

class TestBlock:

    def test_constructor(self) -> None:
        int32block = create_block('i4', [0])
        assert int32block.dtype == np.int32

    @pytest.mark.parametrize('typ, data', [['float', [0, 2, 4]], ['complex', [7]], ['object', [1, 3]], ['bool', [5]]])
    def test_pickle(self, typ: str, data: List[int]) -> None:
        blk = create_block(typ, data)
        assert_block_equal(tm.round_trip_pickle(blk), blk)

    def test_mgr_locs(self, fblock: Block) -> None:
        assert isinstance(fblock.mgr_locs, BlockPlacement)
        tm.assert_numpy_array_equal(fblock.mgr_locs.as_array, np.array([0, 2, 4], dtype=np.intp))

    def test_attrs(self, fblock: Block) -> None:
        assert fblock.shape == fblock.values.shape
        assert fblock.dtype == fblock.values.dtype
        assert len(fblock) == len(fblock.values)

    def test_copy(self, fblock: Block) -> None:
        cop = fblock.copy()
        assert cop is not fblock
        assert_block_equal(fblock, cop)

    def test_delete(self, fblock: Block) -> None:
        newb = fblock.copy()
        locs = newb.mgr_locs
        nb = newb.delete(0)[0]
        assert newb.mgr_locs is locs
        assert nb is not newb
        tm.assert_numpy_array_equal(nb.mgr_locs.as_array, np.array([2, 4], dtype=np.intp))
        assert not (newb.values[0] == 1).all()
        assert (nb.values[0] == 1).all()
        newb = fblock.copy()
        locs = newb.mgr_locs
        nb = newb.delete(1)
        assert len(nb) == 2
        assert newb.mgr_locs is locs
        tm.assert_numpy_array_equal(nb[0].mgr_locs.as_array, np.array([0], dtype=np.intp))
        tm.assert_numpy_array_equal(nb[1].mgr_locs.as_array, np.array([4], dtype=np.intp))
        assert not (newb.values[1] == 2).all()
        assert (nb[1].values[0] == 2).all()
        newb = fblock.copy()
        nb = newb.delete(2)
        assert len(nb) == 1
        tm.assert_numpy_array_equal(nb[0].mgr_locs.as_array, np.array([0, 2], dtype=np.intp))
        assert (nb[0].values[1] == 1).all()
        newb = fblock.copy()
        with pytest.raises(IndexError, match=None):
            newb.delete(3)

    def test_delete_datetimelike(self) -> None:
        arr = np.arange(20, dtype='i8').reshape(5, 4).view('m8[ns]')
        df = DataFrame(arr)
        blk = df._mgr.blocks[0]
        assert isinstance(blk.values, TimedeltaArray)
        nb = blk.delete(1)
        assert len(nb) == 2
        assert isinstance(nb[0].values, TimedeltaArray)
        assert isinstance(nb[1].values, TimedeltaArray)
        df = DataFrame(arr.view('M8[ns]'))
        blk = df._mgr.blocks[0]
        assert isinstance(blk.values, DatetimeArray)
        nb = blk.delete([1, 3])
        assert len(nb) == 2
        assert isinstance(nb[0].values, DatetimeArray)
        assert isinstance(nb[1].values, DatetimeArray)

    def test_split(self) -> None:
        values = np.random.default_rng(2).standard_normal((3, 4))
        blk = new_block(values, placement=BlockPlacement([3, 1, 6]), ndim=2)
        result = list(blk._split())
        values[:] = -9999
        assert (blk.values == -9999).all()
        assert len(result) == 3
        expected = [new_block(values[[0]], placement=BlockPlacement([3]), ndim=2), new_block(values[[1]], placement=BlockPlacement([1]), ndim=2), new_block(values[[2]], placement=BlockPlacement([6]), ndim=2)]
        for res, exp in zip(result, expected):
            assert_block_equal(res, exp)

class TestBlockManager:

    def test_attrs(self) -> None:
        mgr = create_mgr('a,b,c: f8-1; d,e,f: f8-2')
        assert mgr.nblocks == 2
        assert len(mgr) == 6

    def test_duplicate_ref_loc_failure(self) -> None:
        tmp_mgr = create_mgr('a:bool; a: f8')
        axes, blocks = (tmp_mgr.axes, tmp_mgr.blocks)
        blocks[0].mgr_locs = BlockPlacement(np.array([0]))
        blocks[1].mgr_locs = BlockPlacement(np.array([0]))
        msg = 'Gaps in blk ref_locs'
        mgr = BlockManager(blocks, axes)
        with pytest.raises(AssertionError, match=msg):
            mgr._rebuild_blknos_and_blklocs()
        blocks[0].mgr_locs = BlockPlacement(np.array([0]))
        blocks[1].mgr_locs = BlockPlacement(np.array([1]))
        mgr = BlockManager(blocks, axes)
        mgr.iget(1)

    def test_pickle(self, mgr: BlockManager) -> None:
        mgr2 = tm.round_trip_pickle(mgr)
        tm.assert_frame_equal(DataFrame._from_mgr(mgr, axes=mgr.axes), DataFrame._from_mgr(mgr2, axes=mgr2.axes))
        assert hasattr(mgr2, '_is_consolidated')
        assert hasattr(mgr2, '_known_consolidated')
        assert not mgr2._is_consolidated
        assert not mgr2._known_consolidated

    @pytest.mark.parametrize('mgr_string', ['a,a,a:f8', 'a: f8; a: i8'])
    def test_non_unique_pickle(self, mgr_string: str) -> None:
        mgr = create_mgr(mgr_string)
        mgr2 = tm.round_trip_pickle(mgr)
        tm.assert_frame_equal(DataFrame._from_mgr(mgr, axes=mgr.axes), DataFrame._from_mgr(mgr2, axes=mgr2.axes))

    def test_categorical_block_pickle(self) -> None:
        mgr = create_mgr('a: category')
        mgr2 = tm.round_trip_pickle(mgr)
        tm.assert_frame_equal(DataFrame._from_mgr(mgr, axes=mgr.axes), DataFrame._from_mgr(mgr2, axes=mgr2.axes))
        smgr = create_single_mgr('category')
        smgr2 = tm.round_trip_pickle(smgr)
        tm.assert_series_equal(Series()._constructor_from_mgr(smgr, axes=smgr.axes), Series()._constructor_from_mgr(smgr2, axes=smgr2.axes))

    def test_iget(self) -> None:
        cols = Index(list('abc'))
        values = np.random.default_rng(2).random((3, 3))
        block = new_block(values=values.copy(), placement=BlockPlacement(np.arange(3, dtype=np.intp)), ndim=values.ndim)
        mgr = BlockManager(blocks=(block,), axes=[cols, Index(np.arange(3))])
        tm.assert_almost_equal(mgr.iget(0).internal_values(), values[0])
        tm.assert_almost_equal(mgr.iget(1).internal_values(), values[1])
        tm.assert_almost_equal(mgr.iget(2).internal_values(), values[2])

    def test_set(self) -> None:
        mgr = create_mgr('a,b,c: int', item_shape=(3,))
        mgr.insert(len(mgr.items), 'd', np.array(['foo'] * 3))
        mgr.iset(1, np.array(['bar'] * 3))
        tm.assert_numpy_array_equal(mgr.iget(0).internal_values(), np.array([0] * 3))
        tm.assert_numpy_array_equal(mgr.iget(1).internal_values(), np.array(['bar'] * 3, dtype=np.object_))
        tm.assert_numpy_array_equal(mgr.iget(2).internal_values(), np.array([2] * 3))
        tm.assert_numpy_array_equal(mgr.iget(3).internal_values(), np.array(['foo'] * 3, dtype=np.object_))

    def test_set_change_dtype(self, mgr: BlockManager) -> None:
        mgr.insert(len(mgr.items), 'baz', np.zeros(N, dtype=bool))
        mgr.iset(mgr.items.get_loc('baz'),