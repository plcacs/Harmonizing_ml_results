from collections import namedtuple
from collections.abc import Generator
from contextlib import contextmanager
import re
import struct
import tracemalloc
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type, TypeVar, Union
import numpy as np
import pytest
from pandas._libs import hashtable as ht
import pandas as pd
import pandas._testing as tm
from pandas.core.algorithms import isin

T = TypeVar('T')

@contextmanager
def activated_tracemalloc() -> Generator[None, None, None]:
    tracemalloc.start()
    try:
        yield
    finally:
        tracemalloc.stop()

def get_allocated_khash_memory() -> int:
    snapshot = tracemalloc.take_snapshot()
    snapshot = snapshot.filter_traces((tracemalloc.DomainFilter(True, ht.get_hashtable_trace_domain()),))
    return sum((x.size for x in snapshot.traces))

@pytest.mark.parametrize('table_type, dtype', [(ht.PyObjectHashTable, np.object_), (ht.Complex128HashTable, np.complex128), (ht.Int64HashTable, np.int64), (ht.UInt64HashTable, np.uint64), (ht.Float64HashTable, np.float64), (ht.Complex64HashTable, np.complex64), (ht.Int32HashTable, np.int32), (ht.UInt32HashTable, np.uint32), (ht.Float32HashTable, np.float32), (ht.Int16HashTable, np.int16), (ht.UInt16HashTable, np.uint16), (ht.Int8HashTable, np.int8), (ht.UInt8HashTable, np.uint8), (ht.IntpHashTable, np.intp)])
class TestHashTable:

    def test_get_set_contains_len(self, table_type: Type[ht.HashTable], dtype: np.dtype) -> None:
        index = 5
        table = table_type(55)
        assert len(table) == 0
        assert index not in table
        table.set_item(index, 42)
        assert len(table) == 1
        assert index in table
        assert table.get_item(index) == 42
        table.set_item(index + 1, 41)
        assert index in table
        assert index + 1 in table
        assert len(table) == 2
        assert table.get_item(index) == 42
        assert table.get_item(index + 1) == 41
        table.set_item(index, 21)
        assert index in table
        assert index + 1 in table
        assert len(table) == 2
        assert table.get_item(index) == 21
        assert table.get_item(index + 1) == 41
        assert index + 2 not in table
        table.set_item(index + 1, 21)
        assert index in table
        assert index + 1 in table
        assert len(table) == 2
        assert table.get_item(index) == 21
        assert table.get_item(index + 1) == 21
        with pytest.raises(KeyError, match=str(index + 2)):
            table.get_item(index + 2)

    def test_get_set_contains_len_mask(self, table_type: Type[ht.HashTable], dtype: np.dtype) -> None:
        if table_type == ht.PyObjectHashTable:
            pytest.skip('Mask not supported for object')
        index = 5
        table = table_type(55, uses_mask=True)
        assert len(table) == 0
        assert index not in table
        table.set_item(index, 42)
        assert len(table) == 1
        assert index in table
        assert table.get_item(index) == 42
        with pytest.raises(KeyError, match='NA'):
            table.get_na()
        table.set_item(index + 1, 41)
        table.set_na(41)
        assert pd.NA in table
        assert index in table
        assert index + 1 in table
        assert len(table) == 3
        assert table.get_item(index) == 42
        assert table.get_item(index + 1) == 41
        assert table.get_na() == 41
        table.set_na(21)
        assert index in table
        assert index + 1 in table
        assert len(table) == 3
        assert table.get_item(index + 1) == 41
        assert table.get_na() == 21
        assert index + 2 not in table
        with pytest.raises(KeyError, match=str(index + 2)):
            table.get_item(index + 2)

    def test_map_keys_to_values(self, table_type: Type[ht.HashTable], dtype: np.dtype, writable: bool) -> None:
        if table_type == ht.Int64HashTable:
            N = 77
            table = table_type()
            keys = np.arange(N).astype(dtype)
            vals = np.arange(N).astype(np.int64) + N
            keys.flags.writeable = writable
            vals.flags.writeable = writable
            table.map_keys_to_values(keys, vals)
            for i in range(N):
                assert table.get_item(keys[i]) == i + N

    def test_map_locations(self, table_type: Type[ht.HashTable], dtype: np.dtype, writable: bool) -> None:
        N = 8
        table = table_type()
        keys = (np.arange(N) + N).astype(dtype)
        keys.flags.writeable = writable
        table.map_locations(keys)
        for i in range(N):
            assert table.get_item(keys[i]) == i

    def test_map_locations_mask(self, table_type: Type[ht.HashTable], dtype: np.dtype, writable: bool) -> None:
        if table_type == ht.PyObjectHashTable:
            pytest.skip('Mask not supported for object')
        N = 3
        table = table_type(uses_mask=True)
        keys = (np.arange(N) + N).astype(dtype)
        keys.flags.writeable = writable
        table.map_locations(keys, np.array([False, False, True]))
        for i in range(N - 1):
            assert table.get_item(keys[i]) == i
        with pytest.raises(KeyError, match=re.escape(str(keys[N - 1]))):
            table.get_item(keys[N - 1])
        assert table.get_na() == 2

    def test_lookup(self, table_type: Type[ht.HashTable], dtype: np.dtype, writable: bool) -> None:
        N = 3
        table = table_type()
        keys = (np.arange(N) + N).astype(dtype)
        keys.flags.writeable = writable
        table.map_locations(keys)
        result = table.lookup(keys)
        expected = np.arange(N)
        tm.assert_numpy_array_equal(result.astype(np.int64), expected.astype(np.int64))

    def test_lookup_wrong(self, table_type: Type[ht.HashTable], dtype: np.dtype) -> None:
        if dtype in (np.int8, np.uint8):
            N = 100
        else:
            N = 512
        table = table_type()
        keys = (np.arange(N) + N).astype(dtype)
        table.map_locations(keys)
        wrong_keys = np.arange(N).astype(dtype)
        result = table.lookup(wrong_keys)
        assert np.all(result == -1)

    def test_lookup_mask(self, table_type: Type[ht.HashTable], dtype: np.dtype, writable: bool) -> None:
        if table_type == ht.PyObjectHashTable:
            pytest.skip('Mask not supported for object')
        N = 3
        table = table_type(uses_mask=True)
        keys = (np.arange(N) + N).astype(dtype)
        mask = np.array([False, True, False])
        keys.flags.writeable = writable
        table.map_locations(keys, mask)
        result = table.lookup(keys, mask)
        expected = np.arange(N)
        tm.assert_numpy_array_equal(result.astype(np.int64), expected.astype(np.int64))
        result = table.lookup(np.array([1 + N]).astype(dtype), np.array([False]))
        tm.assert_numpy_array_equal(result.astype(np.int64), np.array([-1], dtype=np.int64))

    def test_unique(self, table_type: Type[ht.HashTable], dtype: np.dtype, writable: bool) -> None:
        if dtype in (np.int8, np.uint8):
            N = 88
        else:
            N = 1000
        table = table_type()
        expected = (np.arange(N) + N).astype(dtype)
        keys = np.repeat(expected, 5)
        keys.flags.writeable = writable
        unique = table.unique(keys)
        tm.assert_numpy_array_equal(unique, expected)

    def test_tracemalloc_works(self, table_type: Type[ht.HashTable], dtype: np.dtype) -> None:
        if dtype in (np.int8, np.uint8):
            N = 256
        else:
            N = 30000
        keys = np.arange(N).astype(dtype)
        with activated_tracemalloc():
            table = table_type()
            table.map_locations(keys)
            used = get_allocated_khash_memory()
            my_size = table.sizeof()
            assert used == my_size
            del table
            assert get_allocated_khash_memory() == 0

    def test_tracemalloc_for_empty(self, table_type: Type[ht.HashTable], dtype: np.dtype) -> None:
        with activated_tracemalloc():
            table = table_type()
            used = get_allocated_khash_memory()
            my_size = table.sizeof()
            assert used == my_size
            del table
            assert get_allocated_khash_memory() == 0

    def test_get_state(self, table_type: Type[ht.HashTable], dtype: np.dtype) -> None:
        table = table_type(1000)
        state = table.get_state()
        assert state['size'] == 0
        assert state['n_occupied'] == 0
        assert 'n_buckets' in state
        assert 'upper_bound' in state

    @pytest.mark.parametrize('N', range(1, 110))
    def test_no_reallocation(self, table_type: Type[ht.HashTable], dtype: np.dtype, N: int) -> None:
        keys = np.arange(N).astype(dtype)
        preallocated_table = table_type(N)
        n_buckets_start = preallocated_table.get_state()['n_buckets']
        preallocated_table.map_locations(keys)
        n_buckets_end = preallocated_table.get_state()['n_buckets']
        assert n_buckets_start == n_buckets_end
        clean_table = table_type()
        clean_table.map_locations(keys)
        assert n_buckets_start == clean_table.get_state()['n_buckets']

class TestHashTableUnsorted:

    def test_string_hashtable_set_item_signature(self) -> None:
        tbl = ht.StringHashTable()
        tbl.set_item('key', 1)
        assert tbl.get_item('key') == 1
        with pytest.raises(TypeError, match="'key' has incorrect type"):
            tbl.set_item(4, 6)
        with pytest.raises(TypeError, match="'val' has incorrect type"):
            tbl.get_item(4)

    def test_lookup_nan(self, writable: bool) -> None:
        xs = np.array([2.718, 3.14, np.nan, -7, 5, 2, 3])
        xs.setflags(write=writable)
        m = ht.Float64HashTable()
        m.map_locations(xs)
        tm.assert_numpy_array_equal(m.lookup(xs), np.arange(len(xs), dtype=np.intp))

    def test_add_signed_zeros(self) -> None:
        N = 4
        m = ht.Float64HashTable(N)
        m.set_item(0.0, 0)
        m.set_item(-0.0, 0)
        assert len(m) == 1

    def test_add_different_nans(self) -> None:
        NAN1 = struct.unpack('d', struct.pack('=Q', 9221120237041090560))[0]
        NAN2 = struct.unpack('d', struct.pack('=Q', 9221120237041090561))[0]
        assert NAN1 != NAN1
        assert NAN2 != NAN2
        m = ht.Float64HashTable()
        m.set_item(NAN1, 0)
        m.set_item(NAN2, 0)
        assert len(m) == 1

    def test_lookup_overflow(self, writable: bool) -> None:
        xs = np.array([1, 2, 2 ** 63], dtype=np.uint64)
        xs.setflags(write=writable)
        m = ht.UInt64HashTable()
        m.map_locations(xs)
        tm.assert_numpy_array_equal(m.lookup(xs), np.arange(len(xs), dtype=np.intp))

    @pytest.mark.parametrize('nvals', [0, 10])
    @pytest.mark.parametrize('htable, uniques, dtype, safely_resizes', [(ht.PyObjectHashTable, ht.ObjectVector, 'object', False), (ht.StringHashTable, ht.ObjectVector, 'object', True), (ht.Float64HashTable, ht.Float64Vector, 'float64', False), (ht.Int64HashTable, ht.Int64Vector, 'int64', False), (ht.Int32HashTable, ht.Int32Vector, 'int32', False), (ht.UInt64HashTable, ht.UInt64Vector, 'uint64', False)])
    def test_vector_resize(self, writable: bool, htable: Type[ht.HashTable], uniques: Type[ht.Vector], dtype: str, safely_resizes: bool, nvals: int) -> None:
        vals = np.array(range(1000), dtype=dtype)
        vals.setflags(write=writable)
        htable = htable()
        uniques = uniques()
        htable.get_labels(vals[:nvals], uniques, 0, -1)
        tmp = uniques.to_array()
        oldshape = tmp.shape
        if safely_resizes:
            htable.get_labels(vals, uniques, 0, -1)
        else:
            with pytest.raises(ValueError, match='external reference.*'):
                htable.get_labels(vals, uniques, 0, -1)
        uniques.to_array()
        assert tmp.shape == oldshape

    @pytest.mark.parametrize('hashtable', [ht.PyObjectHashTable, ht.StringHashTable, ht.Float64HashTable, ht.Int64HashTable, ht.Int32HashTable, ht.UInt64HashTable])
    def test_hashtable_large_sizehint(self, hashtable: Type[ht.HashTable]) -> None:
        size_hint = np.iinfo(np.uint32).max + 1
        hashtable(size_hint=size_hint)

class TestPyObjectHashTableWithNans:

    def test_nan_float(self) -> None:
        nan1 = float('nan')
        nan2 = float('nan')
        assert nan1 is not nan2
        table = ht.PyObjectHashTable()
        table.set_item(nan1, 42)
        assert table.get_item(nan2) == 42

    def test_nan_complex_both(self) -> None:
        nan1 = complex(float('nan'), float('nan'))
        nan2 = complex(float('nan'), float('nan'))
        assert nan1 is not nan2
        table = ht.PyObjectHashTable()
        table.set_item(nan1, 42)
        assert table.get_item(nan2) == 42

    def test_nan_complex_real(self) -> None:
        nan1 = complex(float('nan'), 1)
        nan2 = complex(float('nan'), 1)
        other = complex(float('nan'), 2)
        assert nan1 is not nan2
        table = ht.PyObjectHashTable()
        table.set_item(nan1, 42)
        assert table.get_item(nan2) == 42
        with pytest.raises(KeyError, match=re.escape(repr(other))):
            table.get_item(other)

    def test_nan_complex_imag(self) -> None:
        nan1 = complex(1, float('nan'))
        nan2 = complex(1, float('nan'))
        other = complex(2, float('nan'))
        assert nan1 is not nan2
        table = ht.PyObjectHashTable()
        table.set_item(nan1, 42)
        assert table.get_item(nan2) == 42
        with pytest.raises(KeyError, match=re.escape(repr(other))):
            table.get_item(other)

    def test_nan_in_tuple(self) -> None:
        nan1 = (float('nan'),)
        nan2 = (float('nan'),)
        assert nan1[0] is not nan2[0]
        table = ht.PyObjectHashTable()
        table.set_item(nan1, 42)
        assert table.get_item(nan2) == 42

    def test_nan_in_nested_tuple(self) -> None:
        nan1 = (1, (2, (float('nan'),)))
        nan2 = (1, (2, (float('nan'),)))
        other = (1, 2)
        table = ht.PyObjectHashTable()
        table.set_item(nan1, 42)
        assert table.get_item(nan2) == 42
        with pytest.raises(KeyError, match=re.escape(repr(other))):
            table.get_item(other)

    def test_nan_in_namedtuple(self) -> None:
        T = namedtuple('T', ['x'])
        nan1 = T(float('nan'))
        nan2 = T(float('nan'))
        assert nan1.x is not nan2.x
        table = ht.PyObjectHashTable()
        table.set_item