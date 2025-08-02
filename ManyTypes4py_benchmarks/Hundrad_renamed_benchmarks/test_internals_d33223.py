from datetime import date, datetime
import itertools
import re
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


@pytest.fixture(params=[new_block, make_block])
def func_js4kqkdz(request):
    """
    Fixture to test both the internal new_block and pseudo-public make_block.
    """
    return request.param


@pytest.fixture
def func_cfznijau():
    return create_mgr(
        'a: f8; b: object; c: f8; d: object; e: f8;f: bool; g: i8; h: complex; i: datetime-1; j: datetime-2;k: M8[ns, US/Eastern]; l: M8[ns, CET];'
        )


def func_zcr89x5q(left, right):
    tm.assert_numpy_array_equal(left.values, right.values)
    assert left.dtype == right.dtype
    assert isinstance(left.mgr_locs, BlockPlacement)
    assert isinstance(right.mgr_locs, BlockPlacement)
    tm.assert_numpy_array_equal(left.mgr_locs.as_array, right.mgr_locs.as_array
        )


def func_p47arflt(shape):
    arr = np.arange(shape[0])
    return np.lib.stride_tricks.as_strided(x=arr, shape=shape, strides=(arr
        .itemsize,) + (0,) * (len(shape) - 1)).copy()


N = 10


def func_xq76nls3(typestr, placement, item_shape=None, num_offset=0, maker=
    new_block):
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
    placement = BlockPlacement(placement)
    num_items = len(placement)
    if item_shape is None:
        item_shape = N,
    shape = (num_items,) + item_shape
    mat = func_p47arflt(shape)
    if typestr in ('float', 'f8', 'f4', 'f2', 'int', 'i8', 'i4', 'i2', 'i1',
        'uint', 'u8', 'u4', 'u2', 'u1'):
        values = mat.astype(typestr) + num_offset
    elif typestr in ('complex', 'c16', 'c8'):
        values = 1.0j * (mat.astype(typestr) + num_offset)
    elif typestr in ('object', 'string', 'O'):
        values = np.reshape([f'A{i:d}' for i in mat.ravel() + num_offset],
            shape)
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
        values = Categorical(['a', 'a', 'a', 'a', 'b', 'b', 'c', 'c', 'c', 'd']
            )
    elif typestr in ('sparse', 'sparse_na'):
        if shape[-1] != 10:
            raise NotImplementedError
        assert all(s == 1 for s in shape[:-1])
        if typestr.endswith('_na'):
            fill_value = np.nan
        else:
            fill_value = 0.0
        values = SparseArray([fill_value, fill_value, 1, 2, 3, fill_value, 
            4, 5, fill_value, 6], fill_value=fill_value)
        arr = values.sp_values.view()
        arr += num_offset - 1
    else:
        raise ValueError(f'Unsupported typestr: "{typestr}"')
    values = maybe_coerce_values(values)
    return maker(values, placement=placement, ndim=len(shape))


def func_hkweebgz(typestr, num_rows=None):
    if num_rows is None:
        num_rows = N
    return SingleBlockManager(func_xq76nls3(typestr, placement=slice(0,
        num_rows), item_shape=()), Index(np.arange(num_rows)))


def func_dewec046(descr, item_shape=None):
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
        item_shape = N,
    offset = 0
    mgr_items = []
    block_placements = {}
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
    mgr_items = Index(mgr_items)
    blocks = []
    num_offset = 0
    for blockstr, placement in block_placements.items():
        typestr = blockstr.split('-')[0]
        blocks.append(func_xq76nls3(typestr, placement, item_shape=
            item_shape, num_offset=num_offset))
        num_offset += len(placement)
    sblocks = sorted(blocks, key=lambda b: b.mgr_locs[0])
    return BlockManager(tuple(sblocks), [mgr_items] + [Index(np.arange(n)) for
        n in item_shape])


@pytest.fixture
def func_70i31cp1():
    return func_xq76nls3('float', [0, 2, 4])


class TestBlock:

    def func_fykitxgd(self):
        int32block = func_xq76nls3('i4', [0])
        assert int32block.dtype == np.int32

    @pytest.mark.parametrize('typ, data', [['float', [0, 2, 4]], ['complex',
        [7]], ['object', [1, 3]], ['bool', [5]]])
    def func_at99ruz2(self, typ, data):
        blk = func_xq76nls3(typ, data)
        func_zcr89x5q(tm.round_trip_pickle(blk), blk)

    def func_pvf3hjky(self, fblock):
        assert isinstance(fblock.mgr_locs, BlockPlacement)
        tm.assert_numpy_array_equal(fblock.mgr_locs.as_array, np.array([0, 
            2, 4], dtype=np.intp))

    def func_yrr8orjl(self, fblock):
        assert fblock.shape == fblock.values.shape
        assert fblock.dtype == fblock.values.dtype
        assert len(fblock) == len(fblock.values)

    def func_z20ykubc(self, fblock):
        cop = func_70i31cp1.copy()
        assert cop is not fblock
        func_zcr89x5q(fblock, cop)

    def func_ghembhio(self, fblock):
        newb = func_70i31cp1.copy()
        locs = newb.mgr_locs
        nb = newb.delete(0)[0]
        assert newb.mgr_locs is locs
        assert nb is not newb
        tm.assert_numpy_array_equal(nb.mgr_locs.as_array, np.array([2, 4],
            dtype=np.intp))
        assert not (newb.values[0] == 1).all()
        assert (nb.values[0] == 1).all()
        newb = func_70i31cp1.copy()
        locs = newb.mgr_locs
        nb = newb.delete(1)
        assert len(nb) == 2
        assert newb.mgr_locs is locs
        tm.assert_numpy_array_equal(nb[0].mgr_locs.as_array, np.array([0],
            dtype=np.intp))
        tm.assert_numpy_array_equal(nb[1].mgr_locs.as_array, np.array([4],
            dtype=np.intp))
        assert not (newb.values[1] == 2).all()
        assert (nb[1].values[0] == 2).all()
        newb = func_70i31cp1.copy()
        nb = newb.delete(2)
        assert len(nb) == 1
        tm.assert_numpy_array_equal(nb[0].mgr_locs.as_array, np.array([0, 2
            ], dtype=np.intp))
        assert (nb[0].values[1] == 1).all()
        newb = func_70i31cp1.copy()
        with pytest.raises(IndexError, match=None):
            newb.delete(3)

    def func_m3zfui33(self):
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

    def func_mfft8fu1(self):
        values = np.random.default_rng(2).standard_normal((3, 4))
        blk = new_block(values, placement=BlockPlacement([3, 1, 6]), ndim=2)
        result = list(blk._split())
        values[:] = -9999
        assert (blk.values == -9999).all()
        assert len(result) == 3
        expected = [new_block(values[[0]], placement=BlockPlacement([3]),
            ndim=2), new_block(values[[1]], placement=BlockPlacement([1]),
            ndim=2), new_block(values[[2]], placement=BlockPlacement([6]),
            ndim=2)]
        for res, exp in zip(result, expected):
            func_zcr89x5q(res, exp)


class TestBlockManager:

    def func_yrr8orjl(self):
        mgr = func_dewec046('a,b,c: f8-1; d,e,f: f8-2')
        assert mgr.nblocks == 2
        assert len(mgr) == 6

    def func_xfodsng8(self):
        tmp_mgr = func_dewec046('a:bool; a: f8')
        axes, blocks = tmp_mgr.axes, tmp_mgr.blocks
        blocks[0].mgr_locs = BlockPlacement(np.array([0]))
        blocks[1].mgr_locs = BlockPlacement(np.array([0]))
        msg = 'Gaps in blk ref_locs'
        mgr = BlockManager(blocks, axes)
        with pytest.raises(AssertionError, match=msg):
            func_cfznijau._rebuild_blknos_and_blklocs()
        blocks[0].mgr_locs = BlockPlacement(np.array([0]))
        blocks[1].mgr_locs = BlockPlacement(np.array([1]))
        mgr = BlockManager(blocks, axes)
        func_cfznijau.iget(1)

    def func_at99ruz2(self, mgr):
        mgr2 = tm.round_trip_pickle(mgr)
        tm.assert_frame_equal(DataFrame._from_mgr(mgr, axes=mgr.axes),
            DataFrame._from_mgr(mgr2, axes=mgr2.axes))
        assert hasattr(mgr2, '_is_consolidated')
        assert hasattr(mgr2, '_known_consolidated')
        assert not mgr2._is_consolidated
        assert not mgr2._known_consolidated

    @pytest.mark.parametrize('mgr_string', ['a,a,a:f8', 'a: f8; a: i8'])
    def func_bcn71u02(self, mgr_string):
        mgr = func_dewec046(mgr_string)
        mgr2 = tm.round_trip_pickle(mgr)
        tm.assert_frame_equal(DataFrame._from_mgr(mgr, axes=mgr.axes),
            DataFrame._from_mgr(mgr2, axes=mgr2.axes))

    def func_x3usysjw(self):
        mgr = func_dewec046('a: category')
        mgr2 = tm.round_trip_pickle(mgr)
        tm.assert_frame_equal(DataFrame._from_mgr(mgr, axes=mgr.axes),
            DataFrame._from_mgr(mgr2, axes=mgr2.axes))
        smgr = func_hkweebgz('category')
        smgr2 = tm.round_trip_pickle(smgr)
        tm.assert_series_equal(Series()._constructor_from_mgr(smgr, axes=
            smgr.axes), Series()._constructor_from_mgr(smgr2, axes=smgr2.axes))

    def func_svr6ndrc(self):
        cols = Index(list('abc'))
        values = np.random.default_rng(2).random((3, 3))
        block = new_block(values=values.copy(), placement=BlockPlacement(np
            .arange(3, dtype=np.intp)), ndim=values.ndim)
        mgr = BlockManager(blocks=(block,), axes=[cols, Index(np.arange(3))])
        tm.assert_almost_equal(func_cfznijau.iget(0).internal_values(),
            values[0])
        tm.assert_almost_equal(func_cfznijau.iget(1).internal_values(),
            values[1])
        tm.assert_almost_equal(func_cfznijau.iget(2).internal_values(),
            values[2])

    def func_2t3m5xv0(self):
        mgr = func_dewec046('a,b,c: int', item_shape=(3,))
        func_cfznijau.insert(len(mgr.items), 'd', np.array(['foo'] * 3))
        func_cfznijau.iset(1, np.array(['bar'] * 3))
        tm.assert_numpy_array_equal(func_cfznijau.iget(0).internal_values(),
            np.array([0] * 3))
        tm.assert_numpy_array_equal(func_cfznijau.iget(1).internal_values(),
            np.array(['bar'] * 3, dtype=np.object_))
        tm.assert_numpy_array_equal(func_cfznijau.iget(2).internal_values(),
            np.array([2] * 3))
        tm.assert_numpy_array_equal(func_cfznijau.iget(3).internal_values(),
            np.array(['foo'] * 3, dtype=np.object_))

    def func_9jbefuzw(self, mgr):
        func_cfznijau.insert(len(mgr.items), 'baz', np.zeros(N, dtype=bool))
        func_cfznijau.iset(mgr.items.get_loc('baz'), np.repeat('foo', N))
        idx = mgr.items.get_loc('baz')
        assert func_cfznijau.iget(idx).dtype == np.object_
        mgr2 = func_cfznijau.consolidate()
        mgr2.iset(mgr2.items.get_loc('baz'), np.repeat('foo', N))
        idx = mgr2.items.get_loc('baz')
        assert mgr2.iget(idx).dtype == np.object_
        mgr2.insert(len(mgr2.items), 'quux', np.random.default_rng(2).
            standard_normal(N).astype(int))
        idx = mgr2.items.get_loc('quux')
        assert mgr2.iget(idx).dtype == np.dtype(int)
        mgr2.iset(mgr2.items.get_loc('quux'), np.random.default_rng(2).
            standard_normal(N))
        assert mgr2.iget(idx).dtype == np.float64

    def func_z20ykubc(self, mgr):
        cp = func_cfznijau.copy(deep=False)
        for blk, cp_blk in zip(mgr.blocks, cp.blocks):
            tm.assert_equal(cp_blk.values, blk.values)
            if isinstance(blk.values, np.ndarray):
                assert cp_blk.values.base is blk.values.base
            else:
                assert cp_blk.values._ndarray.base is blk.values._ndarray.base
        func_cfznijau._consolidate_inplace()
        cp = func_cfznijau.copy(deep=True)
        for blk, cp_blk in zip(mgr.blocks, cp.blocks):
            bvals = blk.values
            cpvals = cp_blk.values
            tm.assert_equal(cpvals, bvals)
            if isinstance(cpvals, np.ndarray):
                lbase = cpvals.base
                rbase = bvals.base
            else:
                lbase = cpvals._ndarray.base
                rbase = bvals._ndarray.base
            if isinstance(cpvals, DatetimeArray):
                assert lbase is None and rbase is None or lbase is not rbase
            elif not isinstance(cpvals, np.ndarray):
                assert lbase is not rbase
            else:
                assert lbase is None and rbase is None

    def func_yx8zm2yi(self):
        mgr = func_dewec046('a: sparse-1; b: sparse-2')
        assert func_cfznijau.as_array().dtype == np.float64

    def func_kyrx0m6l(self):
        mgr = func_dewec046('a: sparse-1; b: sparse-2; c: f8')
        assert len(mgr.blocks) == 3
        assert isinstance(mgr, BlockManager)

    @pytest.mark.parametrize('mgr_string, dtype', [('c: f4; d: f2', np.
        float32), ('c: f4; d: f2; e: f8', np.float64)])
    def func_jvfli84o(self, mgr_string, dtype):
        mgr = func_dewec046(mgr_string)
        assert func_cfznijau.as_array().dtype == dtype

    @pytest.mark.parametrize('mgr_string, dtype', [('a: bool-1; b: bool-2',
        np.bool_), ('a: i8-1; b: i8-2; c: i4; d: i2; e: u1', np.int64), (
        'c: i4; d: i2; e: u1', np.int32)])
    def func_dzjq88dq(self, mgr_string, dtype):
        mgr = func_dewec046(mgr_string)
        assert func_cfznijau.as_array().dtype == dtype

    def func_g8ehhg0c(self):
        mgr = func_dewec046('h: datetime-1; g: datetime-2')
        assert func_cfznijau.as_array().dtype == 'M8[ns]'

    def func_wrv6z8o9(self):
        mgr = func_dewec046('h: M8[ns, US/Eastern]; g: M8[ns, CET]')
        assert func_cfznijau.iget(0).dtype == 'datetime64[ns, US/Eastern]'
        assert func_cfznijau.iget(1).dtype == 'datetime64[ns, CET]'
        assert func_cfznijau.as_array().dtype == 'object'

    @pytest.mark.parametrize('t', ['float16', 'float32', 'float64', 'int32',
        'int64'])
    def func_f3tdml8w(self, t):
        mgr = func_dewec046('c: f4; d: f2; e: f8')
        t = np.dtype(t)
        tmgr = func_cfznijau.astype(t)
        assert tmgr.iget(0).dtype.type == t
        assert tmgr.iget(1).dtype.type == t
        assert tmgr.iget(2).dtype.type == t
        mgr = func_dewec046(
            'a,b: object; c: bool; d: datetime; e: f4; f: f2; g: f8')
        t = np.dtype(t)
        tmgr = func_cfznijau.astype(t, errors='ignore')
        assert tmgr.iget(2).dtype.type == t
        assert tmgr.iget(4).dtype.type == t
        assert tmgr.iget(5).dtype.type == t
        assert tmgr.iget(6).dtype.type == t
        assert tmgr.iget(0).dtype.type == np.object_
        assert tmgr.iget(1).dtype.type == np.object_
        if t != np.int64:
            assert tmgr.iget(3).dtype.type == np.datetime64
        else:
            assert tmgr.iget(3).dtype.type == t

    def func_mc7f5p94(self, using_infer_string):

        def func_ilukurez(old_mgr, new_mgr):
            """compare the blocks, numeric compare ==, object don't"""
            old_blocks = set(old_mgr.blocks)
            new_blocks = set(new_mgr.blocks)
            assert len(old_blocks) == len(new_blocks)
            for b in old_blocks:
                found = False
                for nb in new_blocks:
                    if (b.values == nb.values).all():
                        found = True
                        break
                assert found
            for b in new_blocks:
                found = False
                for ob in old_blocks:
                    if (b.values == ob.values).all():
                        found = True
                        break
                assert found
        mgr = func_dewec046('f: i8; g: f8')
        new_mgr = func_cfznijau.convert()
        func_ilukurez(mgr, new_mgr)
        mgr = func_dewec046('a,b,foo: object; f: i8; g: f8')
        func_cfznijau.iset(0, np.array(['1'] * N, dtype=np.object_))
        func_cfznijau.iset(1, np.array(['2.'] * N, dtype=np.object_))
        func_cfznijau.iset(2, np.array(['foo.'] * N, dtype=np.object_))
        new_mgr = func_cfznijau.convert()
        dtype = 'str' if using_infer_string else np.object_
        assert new_mgr.iget(0).dtype == dtype
        assert new_mgr.iget(1).dtype == dtype
        assert new_mgr.iget(2).dtype == dtype
        assert new_mgr.iget(3).dtype == np.int64
        assert new_mgr.iget(4).dtype == np.float64
        mgr = func_dewec046(
            'a,b,foo: object; f: i4; bool: bool; dt: datetime; i: i8; g: f8; h: f2'
            )
        func_cfznijau.iset(0, np.array(['1'] * N, dtype=np.object_))
        func_cfznijau.iset(1, np.array(['2.'] * N, dtype=np.object_))
        func_cfznijau.iset(2, np.array(['foo.'] * N, dtype=np.object_))
        new_mgr = func_cfznijau.convert()
        assert new_mgr.iget(0).dtype == dtype
        assert new_mgr.iget(1).dtype == dtype
        assert new_mgr.iget(2).dtype == dtype
        assert new_mgr.iget(3).dtype == np.int32
        assert new_mgr.iget(4).dtype == np.bool_
        assert new_mgr.iget(5).dtype.type, np.datetime64
        assert new_mgr.iget(6).dtype == np.int64
        assert new_mgr.iget(7).dtype == np.float64
        assert new_mgr.iget(8).dtype == np.float16

    def func_yhcudotj(self):
        for dtype in ['f8', 'i8', 'object', 'bool', 'complex', 'M8[ns]',
            'm8[ns]']:
            mgr = func_dewec046(f'a: {dtype}')
            assert func_cfznijau.as_array().dtype == dtype
            mgr = func_dewec046(f'a: {dtype}; b: {dtype}')
            assert func_cfznijau.as_array().dtype == dtype

    @pytest.mark.parametrize('mgr_string, dtype', [('a: category', 'i8'), (
        'a: category; b: category', 'i8'), ('a: category; b: category2',
        'object'), ('a: category2', 'object'), (
        'a: category2; b: category2', 'object'), ('a: f8', 'f8'), (
        'a: f8; b: i8', 'f8'), ('a: f4; b: i8', 'f8'), (
        'a: f4; b: i8; d: object', 'object'), ('a: bool; b: i8', 'object'),
        ('a: complex', 'complex'), ('a: f8; b: category', 'object'), (
        'a: M8[ns]; b: category', 'object'), ('a: M8[ns]; b: bool',
        'object'), ('a: M8[ns]; b: i8', 'object'), ('a: m8[ns]; b: bool',
        'object'), ('a: m8[ns]; b: i8', 'object'), ('a: M8[ns]; b: m8[ns]',
        'object')])
    def func_8yuijclz(self, mgr_string, dtype):
        mgr = func_dewec046('a: category')
        assert func_cfznijau.as_array().dtype == 'i8'
        mgr = func_dewec046('a: category; b: category2')
        assert func_cfznijau.as_array().dtype == 'object'
        mgr = func_dewec046('a: category2')
        assert func_cfznijau.as_array().dtype == 'object'
        mgr = func_dewec046('a: f8')
        assert func_cfznijau.as_array().dtype == 'f8'
        mgr = func_dewec046('a: f8; b: i8')
        assert func_cfznijau.as_array().dtype == 'f8'
        mgr = func_dewec046('a: f4; b: i8')
        assert func_cfznijau.as_array().dtype == 'f8'
        mgr = func_dewec046('a: f4; b: i8; d: object')
        assert func_cfznijau.as_array().dtype == 'object'
        mgr = func_dewec046('a: bool; b: i8')
        assert func_cfznijau.as_array().dtype == 'object'
        mgr = func_dewec046('a: complex')
        assert func_cfznijau.as_array().dtype == 'complex'
        mgr = func_dewec046('a: f8; b: category')
        assert func_cfznijau.as_array().dtype == 'f8'
        mgr = func_dewec046('a: M8[ns]; b: category')
        assert func_cfznijau.as_array().dtype == 'object'
        mgr = func_dewec046('a: M8[ns]; b: bool')
        assert func_cfznijau.as_array().dtype == 'object'
        mgr = func_dewec046('a: M8[ns]; b: i8')
        assert func_cfznijau.as_array().dtype == 'object'
        mgr = func_dewec046('a: m8[ns]; b: bool')
        assert func_cfznijau.as_array().dtype == 'object'
        mgr = func_dewec046('a: m8[ns]; b: i8')
        assert func_cfznijau.as_array().dtype == 'object'
        mgr = func_dewec046('a: M8[ns]; b: m8[ns]')
        assert func_cfznijau.as_array().dtype == 'object'

    def func_9w5lpvbk(self, mgr):
        func_cfznijau.iset(mgr.items.get_loc('f'), np.random.default_rng(2)
            .standard_normal(N))
        func_cfznijau.iset(mgr.items.get_loc('d'), np.random.default_rng(2)
            .standard_normal(N))
        func_cfznijau.iset(mgr.items.get_loc('b'), np.random.default_rng(2)
            .standard_normal(N))
        func_cfznijau.iset(mgr.items.get_loc('g'), np.random.default_rng(2)
            .standard_normal(N))
        func_cfznijau.iset(mgr.items.get_loc('h'), np.random.default_rng(2)
            .standard_normal(N))
        cons = func_cfznijau.consolidate()
        assert cons.nblocks == 4
        cons = func_cfznijau.consolidate().get_numeric_data()
        assert cons.nblocks == 1
        assert isinstance(cons.blocks[0].mgr_locs, BlockPlacement)
        tm.assert_numpy_array_equal(cons.blocks[0].mgr_locs.as_array, np.
            arange(len(cons.items), dtype=np.intp))

    def func_ow5sjqx9(self):
        mgr = func_dewec046(
            'a: f8; b: i8; c: f8; d: i8; e: f8; f: bool; g: f8-2')
        reindexed = func_cfznijau.reindex_axis(['g', 'c', 'a', 'd'], axis=0)
        assert not reindexed.is_consolidated()
        tm.assert_index_equal(reindexed.items, Index(['g', 'c', 'a', 'd']))
        tm.assert_almost_equal(func_cfznijau.iget(6).internal_values(),
            reindexed.iget(0).internal_values())
        tm.assert_almost_equal(func_cfznijau.iget(2).internal_values(),
            reindexed.iget(1).internal_values())
        tm.assert_almost_equal(func_cfznijau.iget(0).internal_values(),
            reindexed.iget(2).internal_values())
        tm.assert_almost_equal(func_cfznijau.iget(3).internal_values(),
            reindexed.iget(3).internal_values())

    def func_2r7xomzi(self):
        mgr = func_dewec046(
            'int: int; float: float; complex: complex;str: object; bool: bool; obj: object; dt: datetime'
            , item_shape=(3,))
        func_cfznijau.iset(5, np.array([1, 2, 3], dtype=np.object_))
        numeric = func_cfznijau.get_numeric_data()
        tm.assert_index_equal(numeric.items, Index(['int', 'float',
            'complex', 'bool']))
        tm.assert_almost_equal(func_cfznijau.iget(mgr.items.get_loc('float'
            )).internal_values(), numeric.iget(numeric.items.get_loc(
            'float')).internal_values())
        numeric.iset(numeric.items.get_loc('float'), np.array([100.0, 200.0,
            300.0]), inplace=True)
        tm.assert_almost_equal(func_cfznijau.iget(mgr.items.get_loc('float'
            )).internal_values(), np.array([1.0, 1.0, 1.0]))

    def func_xvjtgej0(self):
        mgr = func_dewec046(
            'int: int; float: float; complex: complex;str: object; bool: bool; obj: object; dt: datetime'
            , item_shape=(3,))
        func_cfznijau.iset(6, np.array([True, False, True], dtype=np.object_))
        bools = func_cfznijau.get_bool_data()
        tm.assert_index_equal(bools.items, Index(['bool']))
        tm.assert_almost_equal(func_cfznijau.iget(mgr.items.get_loc('bool')
            ).internal_values(), bools.iget(bools.items.get_loc('bool')).
            internal_values())
        bools.iset(0, np.array([True, False, True]), inplace=True)
        tm.assert_numpy_array_equal(func_cfznijau.iget(mgr.items.get_loc(
            'bool')).internal_values(), np.array([True, True, True]))

    def func_12wz9lvr(self):
        repr(func_dewec046('b,א: object'))

    @pytest.mark.parametrize('mgr_string', ['a,b,c: i8-1; d,e,f: i8-2',
        'a,a,a: i8-1; b,b,b: i8-2'])
    def func_k2eyljps(self, mgr_string):
        bm1 = func_dewec046(mgr_string)
        bm2 = BlockManager(bm1.blocks[::-1], bm1.axes)
        assert bm1.equals(bm2)

    @pytest.mark.parametrize('mgr_string', ['a:i8;b:f8',
        'a:i8;b:f8;c:c8;d:b', 'a:i8;e:dt;f:td;g:string',
        'a:i8;b:category;c:category2', 'c:sparse;d:sparse_na;b:f8'])
    def func_aws7msr9(self, mgr_string):
        bm = func_dewec046(mgr_string)
        block_perms = itertools.permutations(bm.blocks)
        for bm_perm in block_perms:
            bm_this = BlockManager(tuple(bm_perm), bm.axes)
            assert bm.equals(bm_this)
            assert bm_this.equals(bm)

    def func_ofqxeg6n(self):
        mgr = func_hkweebgz('f8', num_rows=5)
        assert func_cfznijau.external_values().tolist() == [0.0, 1.0, 2.0, 
            3.0, 4.0]

    @pytest.mark.parametrize('value', [1, 'True', [1, 2, 3], 5.0])
    def func_plh41i5q(self, value):
        bm1 = func_dewec046('a,b,c: i8-1; d,e,f: i8-2')
        msg = (
            f'For argument "inplace" expected type bool, received type {type(value).__name__}.'
            )
        with pytest.raises(ValueError, match=msg):
            bm1.replace_list([1], [2], inplace=value)

    def func_4fvm40k3(self):
        bm = func_dewec046('a,b,c: i8; d: f8')
        bm._iset_split_block(0, np.array([0]))
        tm.assert_numpy_array_equal(bm.blklocs, np.array([0, 0, 1, 0],
            dtype='int64' if IS64 else 'int32'))
        tm.assert_numpy_array_equal(bm.blknos, np.array([0, 0, 0, 1], dtype
            ='int64' if IS64 else 'int32'))
        assert len(bm.blocks) == 2

    def func_o13vvafz(self):
        bm = func_dewec046('a,b,c: i8; d: f8')
        bm._iset_split_block(0, np.array([0]), np.array([list(range(10))]))
        tm.assert_numpy_array_equal(bm.blklocs, np.array([0, 0, 1, 0],
            dtype='int64' if IS64 else 'int32'))
        tm.assert_numpy_array_equal(bm.blknos, np.array([0, 2, 2, 1], dtype
            ='int64' if IS64 else 'int32'))
        assert len(bm.blocks) == 3


def func_00metsks(mgr):
    if mgr.ndim == 1:
        return func_cfznijau.external_values()
    return func_cfznijau.as_array().T


class TestIndexing:
    MANAGERS = [func_hkweebgz('f8', N), func_hkweebgz('i8', N),
        func_dewec046('a,b,c,d,e,f: f8', item_shape=(N,)), func_dewec046(
        'a,b,c,d,e,f: i8', item_shape=(N,)), func_dewec046(
        'a,b: f8; c,d: i8; e,f: string', item_shape=(N,)), func_dewec046(
        'a,b: f8; c,d: i8; e,f: f8', item_shape=(N,))]

    @pytest.mark.parametrize('mgr', MANAGERS)
    def func_2jg7gkku(self, mgr):

        def func_70hbgcud(mgr, axis, slobj):
            mat = func_00metsks(mgr)
            if isinstance(slobj, np.ndarray):
                ax = mgr.axes[axis]
                if len(ax) and len(slobj) and len(slobj) != len(ax):
                    slobj = np.concatenate([slobj, np.zeros(len(ax) - len(
                        slobj), dtype=bool)])
            if isinstance(slobj, slice):
                sliced = func_cfznijau.get_slice(slobj, axis=axis)
            elif mgr.ndim == 1 and axis == 0 and isinstance(slobj, np.ndarray
                ) and slobj.dtype == bool:
                sliced = func_cfznijau.get_rows_with_mask(slobj)
            else:
                raise TypeError(slobj)
            mat_slobj = (slice(None),) * axis + (slobj,)
            tm.assert_numpy_array_equal(mat[mat_slobj], func_00metsks(
                sliced), check_dtype=False)
            tm.assert_index_equal(mgr.axes[axis][slobj], sliced.axes[axis])
        assert mgr.ndim <= 2, mgr.ndim
        for ax in range(mgr.ndim):
            func_70hbgcud(mgr, ax, slice(None))
            func_70hbgcud(mgr, ax, slice(3))
            func_70hbgcud(mgr, ax, slice(100))
            func_70hbgcud(mgr, ax, slice(1, 4))
            func_70hbgcud(mgr, ax, slice(3, 0, -2))
            if mgr.ndim < 2:
                func_70hbgcud(mgr, ax, np.ones(mgr.shape[ax], dtype=np.bool_))
                func_70hbgcud(mgr, ax, np.zeros(mgr.shape[ax], dtype=np.bool_))
                if mgr.shape[ax] >= 3:
                    func_70hbgcud(mgr, ax, np.arange(mgr.shape[ax]) % 3 == 0)
                    func_70hbgcud(mgr, ax, np.array([True, True, False],
                        dtype=np.bool_))

    @pytest.mark.parametrize('mgr', MANAGERS)
    def func_9uykdefs(self, mgr):

        def func_roam257a(mgr, axis, indexer):
            mat = func_00metsks(mgr)
            taken = func_cfznijau.take(indexer, axis)
            tm.assert_numpy_array_equal(np.take(mat, indexer, axis),
                func_00metsks(taken), check_dtype=False)
            tm.assert_index_equal(mgr.axes[axis].take(indexer), taken.axes[
                axis])
        for ax in range(mgr.ndim):
            func_roam257a(mgr, ax, indexer=np.array([], dtype=np.intp))
            func_roam257a(mgr, ax, indexer=np.array([0, 0, 0], dtype=np.intp))
            func_roam257a(mgr, ax, indexer=np.array(list(range(mgr.shape[ax
                ])), dtype=np.intp))
            if mgr.shape[ax] >= 3:
                func_roam257a(mgr, ax, indexer=np.array([0, 1, 2], dtype=np
                    .intp))
                func_roam257a(mgr, ax, indexer=np.array([-1, -2, -3], dtype
                    =np.intp))

    @pytest.mark.parametrize('mgr', MANAGERS)
    @pytest.mark.parametrize('fill_value', [None, np.nan, 100.0])
    def func_erypqqw0(self, fill_value, mgr):

        def func_9ckdxf3p(mgr, axis, new_labels, fill_value):
            mat = func_00metsks(mgr)
            indexer = mgr.axes[axis].get_indexer_for(new_labels)
            reindexed = func_cfznijau.reindex_axis(new_labels, axis,
                fill_value=fill_value)
            tm.assert_numpy_array_equal(algos.take_nd(mat, indexer, axis,
                fill_value=fill_value), func_00metsks(reindexed),
                check_dtype=False)
            tm.assert_index_equal(reindexed.axes[axis], new_labels)
        for ax in range(mgr.ndim):
            func_9ckdxf3p(mgr, ax, Index([]), fill_value)
            func_9ckdxf3p(mgr, ax, mgr.axes[ax], fill_value)
            func_9ckdxf3p(mgr, ax, mgr.axes[ax][[0, 0, 0]], fill_value)
            func_9ckdxf3p(mgr, ax, Index(['foo', 'bar', 'baz']), fill_value)
            func_9ckdxf3p(mgr, ax, Index(['foo', mgr.axes[ax][0], 'baz']),
                fill_value)
            if mgr.shape[ax] >= 3:
                func_9ckdxf3p(mgr, ax, mgr.axes[ax][:-3], fill_value)
                func_9ckdxf3p(mgr, ax, mgr.axes[ax][-3::-1], fill_value)
                func_9ckdxf3p(mgr, ax, mgr.axes[ax][[0, 1, 2, 0, 1, 2]],
                    fill_value)

    @pytest.mark.parametrize('mgr', MANAGERS)
    @pytest.mark.parametrize('fill_value', [None, np.nan, 100.0])
    def func_jawvn1y8(self, fill_value, mgr):

        def func_oplhs66x(mgr, axis, new_labels, indexer, fill_value):
            mat = func_00metsks(mgr)
            reindexed_mat = algos.take_nd(mat, indexer, axis, fill_value=
                fill_value)
            reindexed = func_cfznijau.reindex_indexer(new_labels, indexer,
                axis, fill_value=fill_value)
            tm.assert_numpy_array_equal(reindexed_mat, func_00metsks(
                reindexed), check_dtype=False)
            tm.assert_index_equal(reindexed.axes[axis], new_labels)
        for ax in range(mgr.ndim):
            func_oplhs66x(mgr, ax, Index([]), np.array([], dtype=np.intp),
                fill_value)
            func_oplhs66x(mgr, ax, mgr.axes[ax], np.arange(mgr.shape[ax]),
                fill_value)
            func_oplhs66x(mgr, ax, Index(['foo'] * mgr.shape[ax]), np.
                arange(mgr.shape[ax]), fill_value)
            func_oplhs66x(mgr, ax, mgr.axes[ax][::-1], np.arange(mgr.shape[
                ax]), fill_value)
            func_oplhs66x(mgr, ax, mgr.axes[ax], np.arange(mgr.shape[ax])[:
                :-1], fill_value)
            func_oplhs66x(mgr, ax, Index(['foo', 'bar', 'baz']), np.array([
                0, 0, 0]), fill_value)
            func_oplhs66x(mgr, ax, Index(['foo', 'bar', 'baz']), np.array([
                -1, 0, -1]), fill_value)
            func_oplhs66x(mgr, ax, Index(['foo', mgr.axes[ax][0], 'baz']),
                np.array([-1, -1, -1]), fill_value)
            if mgr.shape[ax] >= 3:
                func_oplhs66x(mgr, ax, Index(['foo', 'bar', 'baz']), np.
                    array([0, 1, 2]), fill_value)


class TestBlockPlacement:

    @pytest.mark.parametrize('slc, expected', [(slice(0, 4), 4), (slice(0, 
        4, 2), 2), (slice(0, 3, 2), 2), (slice(0, 1, 2), 1), (slice(1, 0, -
        1), 1)])
    def func_x6wtkztb(self, slc, expected):
        assert len(BlockPlacement(slc)) == expected

    @pytest.mark.parametrize('slc', [slice(1, 1, 0), slice(1, 2, 0)])
    def func_vrugjs2k(self, slc):
        msg = 'slice step cannot be zero'
        with pytest.raises(ValueError, match=msg):
            BlockPlacement(slc)

    def func_kaev56la(self):
        slc = slice(3, -1, -2)
        bp = BlockPlacement(slc)
        assert bp.indexer == slice(3, None, -2)

    @pytest.mark.parametrize('slc', [slice(None, None), slice(10, None),
        slice(None, None, -1), slice(None, 10, -1), slice(-1, None), slice(
        None, -1), slice(-1, -1), slice(-1, None, -1), slice(None, -1, -1),
        slice(-1, -1, -1)])
    def func_xe9lf7po(self, slc):
        msg = 'unbounded slice'
        with pytest.raises(ValueError, match=msg):
            BlockPlacement(slc)

    @pytest.mark.parametrize('slc', [slice(0, 0), slice(100, 0), slice(100,
        100), slice(100, 100, -1), slice(0, 100, -1)])
    def func_mbn9yati(self, slc):
        assert not BlockPlacement(slc).is_slice_like

    @pytest.mark.parametrize('arr, slc', [([0], slice(0, 1, 1)), ([100],
        slice(100, 101, 1)), ([0, 1, 2], slice(0, 3, 1)), ([0, 5, 10],
        slice(0, 15, 5)), ([0, 100], slice(0, 200, 100)), ([2, 1], slice(2,
        0, -1))])
    def func_20rv9b70(self, arr, slc):
        assert BlockPlacement(arr).as_slice == slc

    @pytest.mark.parametrize('arr', [[], [-1], [-1, -2, -3], [-10], [-1, 0,
        1, 2], [-2, 0, 2, 4], [1, 0, -1], [1, 1, 1]])
    def func_v1mzlgpp(self, arr):
        assert not BlockPlacement(arr).is_slice_like

    @pytest.mark.parametrize('slc, expected', [(slice(0, 3), [0, 1, 2]), (
        slice(0, 0), []), (slice(3, 0), [])])
    def func_qfp8a8t6(self, slc, expected):
        assert list(BlockPlacement(slc)) == expected

    @pytest.mark.parametrize('slc, arr', [(slice(0, 3), [0, 1, 2]), (slice(
        0, 0), []), (slice(3, 0), []), (slice(3, 0, -1), [3, 2, 1])])
    def func_bnarz3fs(self, slc, arr):
        tm.assert_numpy_array_equal(BlockPlacement(slc).as_array, np.
            asarray(arr, dtype=np.intp))

    def func_i23drl0h(self):
        bpl = BlockPlacement(slice(0, 5))
        assert bpl.add(1).as_slice == slice(1, 6, 1)
        assert bpl.add(np.arange(5)).as_slice == slice(0, 10, 2)
        assert list(bpl.add(np.arange(5, 0, -1))) == [5, 5, 5, 5, 5]

    @pytest.mark.parametrize('val, inc, expected', [(slice(0, 0), 0, []), (
        slice(1, 4), 0, [1, 2, 3]), (slice(3, 0, -1), 0, [3, 2, 1]), ([1, 2,
        4], 0, [1, 2, 4]), (slice(0, 0), 10, []), (slice(1, 4), 10, [11, 12,
        13]), (slice(3, 0, -1), 10, [13, 12, 11]), ([1, 2, 4], 10, [11, 12,
        14]), (slice(0, 0), -1, []), (slice(1, 4), -1, [0, 1, 2]), ([1, 2, 
        4], -1, [0, 1, 3])])
    def func_xt5y2qa5(self, val, inc, expected):
        assert list(BlockPlacement(val).add(inc)) == expected

    @pytest.mark.parametrize('val', [slice(1, 4), [1, 2, 4]])
    def func_1ngbr6fh(self, val):
        msg = 'iadd causes length change'
        with pytest.raises(ValueError, match=msg):
            BlockPlacement(val).add(-10)


class TestCanHoldElement:

    @pytest.fixture(params=[lambda x: x, lambda x: x.to_series(), lambda x:
        x._data, lambda x: list(x), lambda x: x.astype(object), lambda x:
        np.asarray(x), lambda x: x[0], lambda x: x[:0]])
    def func_3sgfz4yy(self, request):
        """
        Functions that take an Index and return an element that should have
        blk._can_hold_element(element) for a Block with this index's dtype.
        """
        return request.param

    def func_u2amt3tw(self):
        block = func_xq76nls3('datetime', [0])
        assert block._can_hold_element([])
        arr = pd.array(block.values.ravel())
        assert block._can_hold_element(None)
        arr[0] = None
        assert arr[0] is pd.NaT
        vals = [np.datetime64('2010-10-10'), datetime(2010, 10, 10)]
        for val in vals:
            assert block._can_hold_element(val)
            arr[0] = val
        val = date(2010, 10, 10)
        assert not block._can_hold_element(val)
        msg = (
            "value should be a 'Timestamp', 'NaT', or array of those. Got 'date' instead."
            )
        with pytest.raises(TypeError, match=msg):
            arr[0] = val

    @pytest.mark.parametrize('dtype', [np.int64, np.uint64, np.float64])
    def func_9gd5v5hj(self, dtype, element):
        arr = np.array([1, 3, 4], dtype=dtype)
        ii = IntervalIndex.from_breaks(arr)
        blk = new_block(ii._data, BlockPlacement([1]), ndim=2)
        assert blk._can_hold_element([])

    @pytest.mark.parametrize('dtype', [np.int64, np.uint64, np.float64])
    def func_ttgtexem(self, dtype, element):
        arr = np.array([1, 3, 4, 9], dtype=dtype)
        ii = IntervalIndex.from_breaks(arr)
        blk = new_block(ii._data, BlockPlacement([1]), ndim=2)
        elem = func_3sgfz4yy(ii)
        self.check_series_setitem(elem, ii, True)
        assert blk._can_hold_element(elem)
        ii2 = IntervalIndex.from_breaks(arr[:-1], closed='neither')
        elem = func_3sgfz4yy(ii2)
        with pytest.raises(TypeError, match='Invalid value'):
            self.check_series_setitem(elem, ii, False)
        assert not blk._can_hold_element(elem)
        ii3 = IntervalIndex.from_breaks([Timestamp(1), Timestamp(3),
            Timestamp(4)])
        elem = func_3sgfz4yy(ii3)
        with pytest.raises(TypeError, match='Invalid value'):
            self.check_series_setitem(elem, ii, False)
        assert not blk._can_hold_element(elem)
        ii4 = IntervalIndex.from_breaks([Timedelta(1), Timedelta(3),
            Timedelta(4)])
        elem = func_3sgfz4yy(ii4)
        with pytest.raises(TypeError, match='Invalid value'):
            self.check_series_setitem(elem, ii, False)
        assert not blk._can_hold_element(elem)

    def func_27gw7zgv(self):
        pi = period_range('2016', periods=3, freq='Y')
        blk = new_block(pi._data.reshape(1, 3), BlockPlacement([1]), ndim=2)
        assert blk._can_hold_element([])

    def func_f14hgmzn(self, element):
        pi = period_range('2016', periods=3, freq='Y')
        elem = func_3sgfz4yy(pi)
        self.check_series_setitem(elem, pi, True)
        pi2 = pi.asfreq('D')[:-1]
        elem = func_3sgfz4yy(pi2)
        with pytest.raises(TypeError, match='Invalid value'):
            self.check_series_setitem(elem, pi, False)
        dti = pi.to_timestamp('s')[:-1]
        elem = func_3sgfz4yy(dti)
        with pytest.raises(TypeError, match='Invalid value'):
            self.check_series_setitem(elem, pi, False)

    def func_tfviqbt1(self, obj, elem, inplace):
        blk = obj._mgr.blocks[0]
        if inplace:
            assert blk._can_hold_element(elem)
        else:
            assert not blk._can_hold_element(elem)

    def func_7km03ev0(self, elem, index, inplace):
        arr = index._data.copy()
        ser = Series(arr, copy=False)
        self.check_can_hold_element(ser, elem, inplace)
        if is_scalar(elem):
            ser[0] = elem
        else:
            ser[:len(elem)] = elem
        if inplace:
            assert ser.array is arr
        else:
            assert ser.dtype == object


class TestShouldStore:

    def func_j5rgwdov(self):
        cat = Categorical(['A', 'B', 'C'])
        df = DataFrame(cat)
        blk = df._mgr.blocks[0]
        assert blk.should_store(cat)
        assert blk.should_store(cat[:-1])
        assert not blk.should_store(cat.as_ordered())
        assert not blk.should_store(np.asarray(cat))


def func_0qdmmo7y():
    values = np.array([1.0, 2.0])
    placement = BlockPlacement(slice(2))
    msg = 'Wrong number of dimensions. values.ndim != ndim \\[1 != 2\\]'
    depr_msg = 'make_block is deprecated'
    with pytest.raises(ValueError, match=msg):
        with tm.assert_produces_warning(DeprecationWarning, match=depr_msg):
            make_block(values, placement, ndim=2)


def func_g1k1ei41():
    idx = Index([0, 1, 2, 3, 4])
    a = Series([1, 2, 3]).reindex(idx)
    b = Series(Categorical([1, 2, 3])).reindex(idx)
    assert a._mgr.blocks[0].mgr_locs.indexer == b._mgr.blocks[0
        ].mgr_locs.indexer


def func_osyy62e9(block_maker):
    arr = pd.arrays.NumpyExtensionArray(np.array([1, 2]))
    depr_msg = 'make_block is deprecated'
    warn = DeprecationWarning if block_maker is make_block else None
    with tm.assert_produces_warning(warn, match=depr_msg):
        result = func_js4kqkdz(arr, BlockPlacement(slice(len(arr))), ndim=
            arr.ndim)
    assert result.dtype.kind in ['i', 'u']
    if block_maker is make_block:
        assert result.is_extension is False
        with tm.assert_produces_warning(warn, match=depr_msg):
            result = func_js4kqkdz(arr, slice(len(arr)), dtype=arr.dtype,
                ndim=arr.ndim)
        assert result.dtype.kind in ['i', 'u']
        assert result.is_extension is False
        with tm.assert_produces_warning(warn, match=depr_msg):
            result = func_js4kqkdz(arr.to_numpy(), slice(len(arr)), dtype=
                arr.dtype, ndim=arr.ndim)
        assert result.dtype.kind in ['i', 'u']
        assert result.is_extension is False
