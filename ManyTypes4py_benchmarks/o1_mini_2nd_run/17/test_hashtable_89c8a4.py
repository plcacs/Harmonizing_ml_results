from collections import namedtuple
from collections.abc import Generator
from contextlib import contextmanager
import re
import struct
import tracemalloc
import numpy as np
import pytest
from pandas._libs import hashtable as ht
import pandas as pd
import pandas._testing as tm
from pandas.core.algorithms import isin
from typing import Any, Dict, Iterator, List, Tuple, Type, TypeVar, Union

T = TypeVar('T')


@contextmanager
def activated_tracemalloc() -> Iterator[None]:
    tracemalloc.start()
    try:
        yield
    finally:
        tracemalloc.stop()


def get_allocated_khash_memory() -> int:
    snapshot = tracemalloc.take_snapshot()
    snapshot = snapshot.filter_traces((tracemalloc.DomainFilter(True, ht.get_hashtable_trace_domain()),))
    return sum((x.size for x in snapshot.traces))


@pytest.mark.parametrize(
    'table_type, dtype',
    [
        (ht.PyObjectHashTable, np.object_),
        (ht.Complex128HashTable, np.complex128),
        (ht.Int64HashTable, np.int64),
        (ht.UInt64HashTable, np.uint64),
        (ht.Float64HashTable, np.float64),
        (ht.Complex64HashTable, np.complex64),
        (ht.Int32HashTable, np.int32),
        (ht.UInt32HashTable, np.uint32),
        (ht.Float32HashTable, np.float32),
        (ht.Int16HashTable, np.int16),
        (ht.UInt16HashTable, np.uint16),
        (ht.Int8HashTable, np.int8),
        (ht.UInt8HashTable, np.uint8),
        (ht.IntpHashTable, np.intp),
    ],
)
class TestHashTable:

    def test_get_set_contains_len(self, table_type: Type[ht.BaseHashTable], dtype: np.dtype) -> None:
        index: int = 5
        table: ht.BaseHashTable = table_type(55)
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

    def test_get_set_contains_len_mask(self, table_type: Type[ht.BaseHashTable], dtype: np.dtype) -> None:
        if table_type == ht.PyObjectHashTable:
            pytest.skip('Mask not supported for object')
        index: int = 5
        table: ht.BaseHashTable = table_type(55, uses_mask=True)
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

    def test_map_keys_to_values(
        self, table_type: Type[ht.BaseHashTable], dtype: np.dtype, writable: bool
    ) -> None:
        if table_type == ht.Int64HashTable:
            N: int = 77
            table: ht.BaseHashTable = table_type()
            keys: np.ndarray = np.arange(N).astype(dtype)
            vals: np.ndarray = np.arange(N).astype(np.int64) + N
            keys.flags.writeable = writable
            vals.flags.writeable = writable
            table.map_keys_to_values(keys, vals)
            for i in range(N):
                assert table.get_item(keys[i]) == i + N

    def test_map_locations(
        self, table_type: Type[ht.BaseHashTable], dtype: np.dtype, writable: bool
    ) -> None:
        N: int = 8
        table: ht.BaseHashTable = table_type()
        keys: np.ndarray = (np.arange(N) + N).astype(dtype)
        keys.flags.writeable = writable
        table.map_locations(keys)
        for i in range(N):
            assert table.get_item(keys[i]) == i

    def test_map_locations_mask(
        self, table_type: Type[ht.BaseHashTable], dtype: np.dtype, writable: bool
    ) -> None:
        if table_type == ht.PyObjectHashTable:
            pytest.skip('Mask not supported for object')
        N: int = 3
        table: ht.BaseHashTable = table_type(uses_mask=True)
        keys: np.ndarray = (np.arange(N) + N).astype(dtype)
        mask: np.ndarray = np.array([False, False, True])
        keys.flags.writeable = writable
        table.map_locations(keys, mask)
        for i in range(N - 1):
            assert table.get_item(keys[i]) == i
        with pytest.raises(KeyError, match=re.escape(str(keys[N - 1]))):
            table.get_item(keys[N - 1])
        assert table.get_na() == 2

    def test_lookup(
        self, table_type: Type[ht.BaseHashTable], dtype: np.dtype, writable: bool
    ) -> None:
        N: int = 3
        table: ht.BaseHashTable = table_type()
        keys: np.ndarray = (np.arange(N) + N).astype(dtype)
        keys.flags.writeable = writable
        table.map_locations(keys)
        result: np.ndarray = table.lookup(keys)
        expected: np.ndarray = np.arange(N)
        tm.assert_numpy_array_equal(result.astype(np.int64), expected.astype(np.int64))

    def test_lookup_wrong(self, table_type: Type[ht.BaseHashTable], dtype: np.dtype) -> None:
        if dtype in (np.int8, np.uint8):
            N: int = 100
        else:
            N = 512
        table: ht.BaseHashTable = table_type()
        keys: np.ndarray = (np.arange(N) + N).astype(dtype)
        table.map_locations(keys)
        wrong_keys: np.ndarray = np.arange(N).astype(dtype)
        result: np.ndarray = table.lookup(wrong_keys)
        assert np.all(result == -1)

    def test_lookup_mask(
        self, table_type: Type[ht.BaseHashTable], dtype: np.dtype, writable: bool
    ) -> None:
        if table_type == ht.PyObjectHashTable:
            pytest.skip('Mask not supported for object')
        N: int = 3
        table: ht.BaseHashTable = table_type(uses_mask=True)
        keys: np.ndarray = (np.arange(N) + N).astype(dtype)
        mask: np.ndarray = np.array([False, True, False])
        keys.flags.writeable = writable
        table.map_locations(keys, mask)
        result: np.ndarray = table.lookup(keys, mask)
        expected: np.ndarray = np.arange(N)
        tm.assert_numpy_array_equal(result.astype(np.int64), expected.astype(np.int64))
        result = table.lookup(np.array([1 + N]).astype(dtype), np.array([False]))
        tm.assert_numpy_array_equal(result.astype(np.int64), np.array([-1], dtype=np.int64))

    def test_unique(
        self, table_type: Type[ht.BaseHashTable], dtype: np.dtype, writable: bool
    ) -> None:
        if dtype in (np.int8, np.uint8):
            N: int = 88
        else:
            N = 1000
        table: ht.BaseHashTable = table_type()
        expected: np.ndarray = (np.arange(N) + N).astype(dtype)
        keys: np.ndarray = np.repeat(expected, 5)
        keys.flags.writeable = writable
        unique: np.ndarray = table.unique(keys)
        tm.assert_numpy_array_equal(unique, expected)

    def test_tracemalloc_works(
        self, table_type: Type[ht.BaseHashTable], dtype: np.dtype
    ) -> None:
        if dtype in (np.int8, np.uint8):
            N: int = 256
        else:
            N = 30000
        keys: np.ndarray = np.arange(N).astype(dtype)
        with activated_tracemalloc():
            table: ht.BaseHashTable = table_type()
            table.map_locations(keys)
            used: int = get_allocated_khash_memory()
            my_size: int = table.sizeof()
            assert used == my_size
            del table
            assert get_allocated_khash_memory() == 0

    def test_tracemalloc_for_empty(
        self, table_type: Type[ht.BaseHashTable], dtype: np.dtype
    ) -> None:
        with activated_tracemalloc():
            table: ht.BaseHashTable = table_type()
            used: int = get_allocated_khash_memory()
            my_size: int = table.sizeof()
            assert used == my_size
            del table
            assert get_allocated_khash_memory() == 0

    def test_get_state(
        self, table_type: Type[ht.BaseHashTable], dtype: np.dtype
    ) -> None:
        table: ht.BaseHashTable = table_type(1000)
        state: Dict[str, Any] = table.get_state()
        assert state['size'] == 0
        assert state['n_occupied'] == 0
        assert 'n_buckets' in state
        assert 'upper_bound' in state

    @pytest.mark.parametrize('N', range(1, 110))
    def test_no_reallocation(
        self, table_type: Type[ht.BaseHashTable], dtype: np.dtype, N: int
    ) -> None:
        keys: np.ndarray = np.arange(N).astype(dtype)
        preallocated_table: ht.BaseHashTable = table_type(N)
        n_buckets_start: int = preallocated_table.get_state()['n_buckets']
        preallocated_table.map_locations(keys)
        n_buckets_end: int = preallocated_table.get_state()['n_buckets']
        assert n_buckets_start == n_buckets_end
        clean_table: ht.BaseHashTable = table_type()
        clean_table.map_locations(keys)
        assert n_buckets_start == clean_table.get_state()['n_buckets']


class TestHashTableUnsorted:

    def test_string_hashtable_set_item_signature(self) -> None:
        tbl: ht.StringHashTable = ht.StringHashTable()
        tbl.set_item('key', 1)
        assert tbl.get_item('key') == 1
        with pytest.raises(TypeError, match="'key' has incorrect type"):
            tbl.set_item(4, 6)
        with pytest.raises(TypeError, match="'val' has incorrect type"):
            tbl.get_item(4)

    def test_lookup_nan(self, writable: bool) -> None:
        xs: np.ndarray = np.array([2.718, 3.14, np.nan, -7, 5, 2, 3])
        xs.setflags(write=writable)
        m: ht.Float64HashTable = ht.Float64HashTable()
        m.map_locations(xs)
        tm.assert_numpy_array_equal(m.lookup(xs), np.arange(len(xs), dtype=np.intp))

    def test_add_signed_zeros(self) -> None:
        N: int = 4
        m: ht.Float64HashTable = ht.Float64HashTable(N)
        m.set_item(0.0, 0)
        m.set_item(-0.0, 0)
        assert len(m) == 1

    def test_add_different_nans(self) -> None:
        NAN1: float = struct.unpack('d', struct.pack('=Q', 9221120237041090560))[0]
        NAN2: float = struct.unpack('d', struct.pack('=Q', 9221120237041090561))[0]
        assert NAN1 != NAN1
        assert NAN2 != NAN2
        m: ht.Float64HashTable = ht.Float64HashTable()
        m.set_item(NAN1, 0)
        m.set_item(NAN2, 0)
        assert len(m) == 1

    def test_lookup_overflow(self, writable: bool) -> None:
        xs: np.ndarray = np.array([1, 2, 2 ** 63], dtype=np.uint64)
        xs.setflags(write=writable)
        m: ht.UInt64HashTable = ht.UInt64HashTable()
        m.map_locations(xs)
        tm.assert_numpy_array_equal(m.lookup(xs), np.arange(len(xs), dtype=np.intp))

    @pytest.mark.parametrize(
        'nvals',
        [0, 10],
    )
    @pytest.mark.parametrize(
        'htable, uniques, dtype, safely_resizes',
        [
            (ht.PyObjectHashTable, ht.ObjectVector, 'object', False),
            (ht.StringHashTable, ht.ObjectVector, 'object', True),
            (ht.Float64HashTable, ht.Float64Vector, 'float64', False),
            (ht.Int64HashTable, ht.Int64Vector, 'int64', False),
            (ht.Int32HashTable, ht.Int32Vector, 'int32', False),
            (ht.UInt64HashTable, ht.UInt64Vector, 'uint64', False),
        ],
    )
    def test_vector_resize(
        self,
        writable: bool,
        htable: Type[ht.BaseHashTable],
        uniques: Type[Any],
        dtype: str,
        safely_resizes: bool,
        nvals: int,
    ) -> None:
        vals: np.ndarray = np.array(range(1000), dtype=dtype)
        vals.setflags(write=writable)
        htable_instance: ht.BaseHashTable = htable()
        uniques_instance: Any = uniques()
        htable_instance.get_labels(vals[:nvals], uniques_instance, 0, -1)
        tmp: np.ndarray = uniques_instance.to_array()
        oldshape: Tuple[int, ...] = tmp.shape
        if safely_resizes:
            htable_instance.get_labels(vals, uniques_instance, 0, -1)
        else:
            with pytest.raises(ValueError, match='external reference.*'):
                htable_instance.get_labels(vals, uniques_instance, 0, -1)
        uniques_instance.to_array()
        assert tmp.shape == oldshape

    @pytest.mark.parametrize(
        'hashtable',
        [
            ht.PyObjectHashTable,
            ht.StringHashTable,
            ht.Float64HashTable,
            ht.Int64HashTable,
            ht.Int32HashTable,
            ht.UInt64HashTable,
        ],
    )
    def test_hashtable_large_sizehint(self, hashtable: Type[ht.BaseHashTable]) -> None:
        size_hint: int = np.iinfo(np.uint32).max + 1
        hashtable(size_hint=size_hint)


class TestPyObjectHashTableWithNans:

    def test_nan_float(self) -> None:
        nan1: float = float('nan')
        nan2: float = float('nan')
        assert nan1 is not nan2
        table: ht.PyObjectHashTable = ht.PyObjectHashTable()
        table.set_item(nan1, 42)
        assert table.get_item(nan2) == 42

    def test_nan_complex_both(self) -> None:
        nan1: complex = complex(float('nan'), float('nan'))
        nan2: complex = complex(float('nan'), float('nan'))
        assert nan1 is not nan2
        table: ht.PyObjectHashTable = ht.PyObjectHashTable()
        table.set_item(nan1, 42)
        assert table.get_item(nan2) == 42

    def test_nan_complex_real(self) -> None:
        nan1: complex = complex(float('nan'), 1)
        nan2: complex = complex(float('nan'), 1)
        other: complex = complex(float('nan'), 2)
        assert nan1 is not nan2
        table: ht.PyObjectHashTable = ht.PyObjectHashTable()
        table.set_item(nan1, 42)
        assert table.get_item(nan2) == 42
        with pytest.raises(KeyError, match=re.escape(repr(other))):
            table.get_item(other)

    def test_nan_complex_imag(self) -> None:
        nan1: complex = complex(1, float('nan'))
        nan2: complex = complex(1, float('nan'))
        other: complex = complex(2, float('nan'))
        assert nan1 is not nan2
        table: ht.PyObjectHashTable = ht.PyObjectHashTable()
        table.set_item(nan1, 42)
        assert table.get_item(nan2) == 42
        with pytest.raises(KeyError, match=re.escape(repr(other))):
            table.get_item(other)

    def test_nan_in_tuple(self) -> None:
        nan1: Tuple[float, ...] = (float('nan'),)
        nan2: Tuple[float, ...] = (float('nan'),)
        assert nan1[0] is not nan2[0]
        table: ht.PyObjectHashTable = ht.PyObjectHashTable()
        table.set_item(nan1, 42)
        assert table.get_item(nan2) == 42

    def test_nan_in_nested_tuple(self) -> None:
        nan1: Tuple[int, Tuple[int, Tuple[float, ...]]] = (1, (2, (float('nan'),)))
        nan2: Tuple[int, Tuple[int, Tuple[float, ...]]] = (1, (2, (float('nan'),)))
        other: Tuple[int, ...] = (1, 2)
        table: ht.PyObjectHashTable = ht.PyObjectHashTable()
        table.set_item(nan1, 42)
        assert table.get_item(nan2) == 42
        with pytest.raises(KeyError, match=re.escape(repr(other))):
            table.get_item(other)

    def test_nan_in_namedtuple(self) -> None:
        T = namedtuple('T', ['x'])
        nan1: T = T(float('nan'))
        nan2: T = T(float('nan'))
        assert nan1.x is not nan2.x
        table: ht.PyObjectHashTable = ht.PyObjectHashTable()
        table.set_item(nan1, 42)
        assert table.get_item(nan2) == 42

    def test_nan_in_nested_namedtuple(self) -> None:
        T = namedtuple('T', ['x', 'y'])
        nan1: T = T(1, (2, (float('nan'),)))
        nan2: T = T(1, (2, (float('nan'),)))
        other: T = T(1, 2)
        table: ht.PyObjectHashTable = ht.PyObjectHashTable()
        table.set_item(nan1, 42)
        assert table.get_item(nan2) == 42
        with pytest.raises(KeyError, match=re.escape(repr(other))):
            table.get_item(other)


def test_hash_equal_tuple_with_nans() -> None:
    a: Tuple[Any, ...] = (float('nan'), (float('nan'), float('nan')))
    b: Tuple[Any, ...] = (float('nan'), (float('nan'), float('nan')))
    assert ht.object_hash(a) == ht.object_hash(b)
    assert ht.objects_are_equal(a, b)


def test_hash_equal_namedtuple_with_nans() -> None:
    T = namedtuple('T', ['x', 'y'])
    a: Any = T(float('nan'), (float('nan'), float('nan')))
    b: Any = T(float('nan'), (float('nan'), float('nan')))
    assert ht.object_hash(a) == ht.object_hash(b)
    assert ht.objects_are_equal(a, b)


def test_hash_equal_namedtuple_and_tuple() -> None:
    T = namedtuple('T', ['x', 'y'])
    a: Any = T(1, (2, 3))
    b: Tuple[Any, ...] = (1, (2, 3))
    assert ht.object_hash(a) == ht.object_hash(b)
    assert ht.objects_are_equal(a, b)


def test_get_labels_groupby_for_Int64(writable: bool) -> None:
    table: ht.Int64HashTable = ht.Int64HashTable()
    vals: np.ndarray = np.array([1, 2, -1, 2, 1, -1], dtype=np.int64)
    vals.flags.writeable = writable
    arr: np.ndarray
    unique: ht.Int64Vector
    arr, unique = table.get_labels_groupby(vals)
    expected_arr: np.ndarray = np.array([0, 1, -1, 1, 0, -1], dtype=np.intp)
    expected_unique: np.ndarray = np.array([1, 2], dtype=np.int64)
    tm.assert_numpy_array_equal(arr, expected_arr)
    tm.assert_numpy_array_equal(unique, expected_unique)


def test_tracemalloc_works_for_StringHashTable() -> None:
    N: int = 1000
    keys: np.ndarray = np.arange(N).astype(np.str_).astype(object)
    with activated_tracemalloc():
        table: ht.StringHashTable = ht.StringHashTable()
        table.map_locations(keys)
        used: int = get_allocated_khash_memory()
        my_size: int = table.sizeof()
        assert used == my_size
        del table
        assert get_allocated_khash_memory() == 0


def test_tracemalloc_for_empty_StringHashTable() -> None:
    with activated_tracemalloc():
        table: ht.StringHashTable = ht.StringHashTable()
        used: int = get_allocated_khash_memory()
        my_size: int = table.sizeof()
        assert used == my_size
        del table
        assert get_allocated_khash_memory() == 0


@pytest.mark.parametrize('N', range(1, 110))
def test_no_reallocation_StringHashTable(N: int) -> None:
    keys: np.ndarray = np.arange(N).astype(np.str_).astype(object)
    preallocated_table: ht.StringHashTable = ht.StringHashTable(N)
    n_buckets_start: int = preallocated_table.get_state()['n_buckets']
    preallocated_table.map_locations(keys)
    n_buckets_end: int = preallocated_table.get_state()['n_buckets']
    assert n_buckets_start == n_buckets_end
    clean_table: ht.StringHashTable = ht.StringHashTable()
    clean_table.map_locations(keys)
    assert n_buckets_start == clean_table.get_state()['n_buckets']


@pytest.mark.parametrize(
    'table_type, dtype',
    [
        (ht.Float64HashTable, np.float64),
        (ht.Float32HashTable, np.float32),
        (ht.Complex128HashTable, np.complex128),
        (ht.Complex64HashTable, np.complex64),
    ],
)
class TestHashTableWithNans:

    def test_get_set_contains_len(self, table_type: Type[ht.BaseHashTable], dtype: np.dtype) -> None:
        index: float = float('nan')
        table: ht.BaseHashTable = table_type()
        assert index not in table
        table.set_item(index, 42)
        assert len(table) == 1
        assert index in table
        assert table.get_item(index) == 42
        table.set_item(index, 41)
        assert len(table) == 1
        assert index in table
        assert table.get_item(index) == 41

    def test_map_locations(self, table_type: Type[ht.BaseHashTable], dtype: np.dtype) -> None:
        N: int = 10
        table: ht.BaseHashTable = table_type()
        keys: np.ndarray = np.full(N, np.nan, dtype=dtype)
        table.map_locations(keys)
        assert len(table) == 1
        assert table.get_item(np.nan) == N - 1

    def test_unique(self, table_type: Type[ht.BaseHashTable], dtype: np.dtype) -> None:
        N: int = 1020
        table: ht.BaseHashTable = table_type()
        keys: np.ndarray = np.full(N, np.nan, dtype=dtype)
        unique: np.ndarray = table.unique(keys)
        assert np.all(np.isnan(unique)) and len(unique) == 1


def test_unique_for_nan_objects_floats() -> None:
    table: ht.PyObjectHashTable = ht.PyObjectHashTable()
    keys: np.ndarray = np.array([float('nan') for _ in range(50)], dtype=object)
    unique: np.ndarray = table.unique(keys)
    assert len(unique) == 1


def test_unique_for_nan_objects_complex() -> None:
    table: ht.PyObjectHashTable = ht.PyObjectHashTable()
    keys: np.ndarray = np.array([complex(float('nan'), 1.0) for _ in range(50)], dtype=object)
    unique: np.ndarray = table.unique(keys)
    assert len(unique) == 1


def test_unique_for_nan_objects_tuple() -> None:
    table: ht.PyObjectHashTable = ht.PyObjectHashTable()
    keys: np.ndarray = np.array([1] + [(1.0, (float('nan'), 1.0)) for _ in range(50)], dtype=object)
    unique: np.ndarray = table.unique(keys)
    assert len(unique) == 2


@pytest.mark.parametrize(
    'dtype',
    [
        np.object_,
        np.complex128,
        np.int64,
        np.uint64,
        np.float64,
        np.complex64,
        np.int32,
        np.uint32,
        np.float32,
        np.int16,
        np.uint16,
        np.int8,
        np.uint8,
        np.intp,
    ],
)
class TestHelpFunctions:

    def test_value_count(
        self, dtype: np.dtype, writable: bool
    ) -> None:
        N: int = 43
        expected: np.ndarray = (np.arange(N) + N).astype(dtype)
        values: np.ndarray = np.repeat(expected, 5)
        values.flags.writeable = writable
        keys: np.ndarray
        counts: np.ndarray
        _unused: Any
        keys, counts, _unused = ht.value_count(values, False)
        tm.assert_numpy_array_equal(np.sort(keys), expected)
        assert np.all(counts == 5)

    def test_value_count_mask(self, dtype: np.dtype) -> None:
        if dtype == np.object_:
            pytest.skip('mask not implemented for object dtype')
        values: np.ndarray = np.array([1] * 5, dtype=dtype)
        mask: np.ndarray = np.zeros((5,), dtype=bool)
        mask[1] = True
        mask[4] = True
        keys: np.ndarray
        counts: np.ndarray
        na_counter: int
        keys, counts, na_counter = ht.value_count(values, False, mask=mask)
        assert len(keys) == 2
        assert na_counter == 2

    def test_value_count_stable(
        self, dtype: np.dtype, writable: bool
    ) -> None:
        values: np.ndarray = np.array([2, 1, 5, 22, 3, -1, 8]).astype(dtype)
        values.flags.writeable = writable
        keys: np.ndarray
        counts: np.ndarray
        _unused: Any
        keys, counts, _unused = ht.value_count(values, False)
        tm.assert_numpy_array_equal(keys, values)
        assert np.all(counts == 1)

    def test_duplicated_first(
        self, dtype: np.dtype, writable: bool
    ) -> None:
        N: int = 100
        values: np.ndarray = np.repeat(np.arange(N).astype(dtype), 5)
        values.flags.writeable = writable
        result: np.ndarray = ht.duplicated(values)
        expected: np.ndarray = np.ones_like(values, dtype=bool)
        expected[::5] = False
        tm.assert_numpy_array_equal(result, expected)

    def test_ismember_yes(
        self, dtype: np.dtype, writable: bool
    ) -> None:
        N: int = 127
        arr: np.ndarray = np.arange(N).astype(dtype)
        values: np.ndarray = np.arange(N).astype(dtype)
        arr.flags.writeable = writable
        values.flags.writeable = writable
        result: np.ndarray = ht.ismember(arr, values)
        expected: np.ndarray = np.ones_like(values, dtype=bool)
        tm.assert_numpy_array_equal(result, expected)

    def test_ismember_no(self, dtype: np.dtype) -> None:
        N: int = 17
        arr: np.ndarray = np.arange(N).astype(dtype)
        values: np.ndarray = (np.arange(N) + N).astype(dtype)
        result: np.ndarray = ht.ismember(arr, values)
        expected: np.ndarray = np.zeros_like(values, dtype=bool)
        tm.assert_numpy_array_equal(result, expected)

    def test_mode(
        self, dtype: np.dtype, writable: bool
    ) -> None:
        if dtype in (np.int8, np.uint8):
            N: int = 53
        else:
            N = 11111
        values: np.ndarray = np.repeat(np.arange(N).astype(dtype), 5)
        values[0] = 42
        values.flags.writeable = writable
        result: int = ht.mode(values, False)[0]
        assert result == 42

    def test_mode_stable(
        self, dtype: np.dtype, writable: bool
    ) -> None:
        values: np.ndarray = np.array([2, 1, 5, 22, 3, -1, 8]).astype(dtype)
        values.flags.writeable = writable
        keys: np.ndarray = ht.mode(values, False)[0]
        tm.assert_numpy_array_equal(keys, values)


def test_modes_with_nans() -> None:
    nulls: List[Union[pd.NA, float, pd._libs.tslibs.nattype.NaTType, None]] = [pd.NA, np.nan, pd.NaT, None]
    values: np.ndarray = np.array([True] + nulls * 2, dtype=object)
    modes: np.ndarray = ht.mode(values, False)[0]
    assert modes.size == len(nulls)


def test_unique_label_indices_intp(writable: bool) -> None:
    keys: np.ndarray = np.array([1, 2, 2, 2, 1, 3], dtype=np.intp)
    keys.flags.writeable = writable
    result: np.ndarray = ht.unique_label_indices(keys)
    expected: np.ndarray = np.array([0, 1, 5], dtype=np.intp)
    tm.assert_numpy_array_equal(result, expected)


def test_unique_label_indices() -> None:
    rng = np.random.default_rng(2)
    a: np.ndarray = rng.integers(1, 1 << 10, 1 << 15).astype(np.intp)
    left: np.ndarray = ht.unique_label_indices(a)
    right: np.ndarray = np.unique(a, return_index=True)[1]
    tm.assert_numpy_array_equal(left, right, check_dtype=False)
    a[rng.choice(len(a), 10)] = -1
    left = ht.unique_label_indices(a)
    right = np.unique(a, return_index=True)[1][1:]
    tm.assert_numpy_array_equal(left, right, check_dtype=False)


@pytest.mark.parametrize(
    'dtype',
    [np.float64, np.float32, np.complex128, np.complex64],
)
class TestHelpFunctionsWithNans:

    def test_value_count(
        self, dtype: np.dtype
    ) -> None:
        values: np.ndarray = np.array([np.nan, np.nan, np.nan], dtype=dtype)
        keys: np.ndarray
        counts: np.ndarray
        _unused: Any
        keys, counts, _unused = ht.value_count(values, True)
        assert len(keys) == 0
        keys, counts, _unused = ht.value_count(values, False)
        assert len(keys) == 1 and np.all(np.isnan(keys))
        assert counts[0] == 3

    def test_duplicated_first(
        self, dtype: np.dtype
    ) -> None:
        values: np.ndarray = np.array([np.nan, np.nan, np.nan], dtype=dtype)
        result: np.ndarray = ht.duplicated(values)
        expected: np.ndarray = np.array([False, True, True])
        tm.assert_numpy_array_equal(result, expected)

    def test_ismember_yes(
        self, dtype: np.dtype
    ) -> None:
        arr: np.ndarray = np.array([np.nan, np.nan, np.nan], dtype=dtype)
        values: np.ndarray = np.array([np.nan, np.nan], dtype=dtype)
        result: np.ndarray = ht.ismember(arr, values)
        expected: np.ndarray = np.array([True, True, True], dtype=bool)
        tm.assert_numpy_array_equal(result, expected)

    def test_ismember_no(
        self, dtype: np.dtype
    ) -> None:
        arr: np.ndarray = np.array([np.nan, np.nan, np.nan], dtype=dtype)
        values: np.ndarray = np.array([1], dtype=dtype)
        result: np.ndarray = ht.ismember(arr, values)
        expected: np.ndarray = np.array([False, False, False], dtype=bool)
        tm.assert_numpy_array_equal(result, expected)

    def test_mode(
        self, dtype: np.dtype
    ) -> None:
        values: np.ndarray = np.array([42, np.nan, np.nan, np.nan], dtype=dtype)
        assert ht.mode(values, True)[0] == 42
        assert np.isnan(ht.mode(values, False)[0])


def test_ismember_tuple_with_nans() -> None:
    values: np.ndarray = np.empty(2, dtype=object)
    values[:] = [('a', float('nan')), ('b', 1)]
    comps: List[Tuple[Any, ...]] = [('a', float('nan'))]
    result: np.ndarray = isin(values, comps)
    expected: np.ndarray = np.array([True, False], dtype=bool)
    tm.assert_numpy_array_equal(result, expected)


def test_float_complex_int_are_equal_as_objects() -> None:
    values: List[Union[str, int, float, complex]] = ['a', 5, 5.0, 5.0 + 0j]
    comps: List[int] = list(range(129))
    result: np.ndarray = isin(np.array(values, dtype=object), np.asarray(comps))
    expected: np.ndarray = np.array([False, True, True, True], dtype=bool)
    tm.assert_numpy_array_equal(result, expected)
