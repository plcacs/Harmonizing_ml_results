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

@contextmanager
def activated_tracemalloc() -> Generator[None, None, None]:
    ...

def get_allocated_khash_memory() -> int:
    ...

@pytest.mark.parametrize('table_type, dtype', [
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
    (ht.IntpHashTable, np.intp)
])
class TestHashTable:
    def test_get_set_contains_len(self, table_type: type[ht.HashTable], dtype: np.dtype) -> None:
        ...

    def test_get_set_contains_len_mask(self, table_type: type[ht.HashTable], dtype: np.dtype) -> None:
        ...

    def test_map_keys_to_values(self, table_type: type[ht.HashTable], dtype: np.dtype, writable: bool) -> None:
        ...

    def test_map_locations(self, table_type: type[ht.HashTable], dtype: np.dtype, writable: bool) -> None:
        ...

    def test_map_locations_mask(self, table_type: type[ht.HashTable], dtype: np.dtype, writable: bool) -> None:
        ...

    def test_lookup(self, table_type: type[ht.HashTable], dtype: np.dtype, writable: bool) -> None:
        ...

    def test_lookup_wrong(self, table_type: type[ht.HashTable], dtype: np.dtype) -> None:
        ...

    def test_lookup_mask(self, table_type: type[ht.HashTable], dtype: np.dtype, writable: bool) -> None:
        ...

    def test_unique(self, table_type: type[ht.HashTable], dtype: np.dtype, writable: bool) -> None:
        ...

    def test_tracemalloc_works(self, table_type: type[ht.HashTable], dtype: np.dtype) -> None:
        ...

    def test_tracemalloc_for_empty(self, table_type: type[ht.HashTable], dtype: np.dtype) -> None:
        ...

    def test_get_state(self, table_type: type[ht.HashTable], dtype: np.dtype) -> None:
        ...

    @pytest.mark.parametrize('N', range(1, 110))
    def test_no_reallocation(self, table_type: type[ht.HashTable], dtype: np.dtype, N: int) -> None:
        ...

class TestHashTableUnsorted:
    def test_string_hashtable_set_item_signature(self) -> None:
        ...

    def test_lookup_nan(self, writable: bool) -> None:
        ...

    def test_add_signed_zeros(self) -> None:
        ...

    def test_add_different_nans(self) -> None:
        ...

    def test_lookup_overflow(self, writable: bool) -> None:
        ...

    @pytest.mark.parametrize('nvals', [0, 10])
    @pytest.mark.parametrize('htable, uniques, dtype, safely_resizes', [
        (ht.PyObjectHashTable, ht.ObjectVector, 'object', False),
        (ht.StringHashTable, ht.ObjectVector, 'object', True),
        (ht.Float64HashTable, ht.Float64Vector, 'float64', False),
        (ht.Int64HashTable, ht.Int64Vector, 'int64', False),
        (ht.Int32HashTable, ht.Int32Vector, 'int32', False),
        (ht.UInt64HashTable, ht.UInt64Vector, 'uint64', False)
    ])
    def test_vector_resize(self, writable: bool, htable: type[ht.HashTable], uniques: type[ht.Vector], dtype: str, safely_resizes: bool, nvals: int) -> None:
        ...

    @pytest.mark.parametrize('hashtable', [
        ht.PyObjectHashTable,
        ht.StringHashTable,
        ht.Float64HashTable,
        ht.Int64HashTable,
        ht.Int32HashTable,
        ht.UInt64HashTable
    ])
    def test_hashtable_large_sizehint(self, hashtable: type[ht.HashTable]) -> None:
        ...

class TestPyObjectHashTableWithNans:
    def test_nan_float(self) -> None:
        ...

    def test_nan_complex_both(self) -> None:
        ...

    def test_nan_complex_real(self) -> None:
        ...

    def test_nan_complex_imag(self) -> None:
        ...

    def test_nan_in_tuple(self) -> None:
        ...

    def test_nan_in_nested_tuple(self) -> None:
        ...

    def test_nan_in_namedtuple(self) -> None:
        ...

    def test_nan_in_nested_namedtuple(self) -> None:
        ...

def test_hash_equal_tuple_with_nans() -> None:
    ...

def test_hash_equal_namedtuple_with_nans() -> None:
    ...

def test_hash_equal_namedtuple_and_tuple() -> None:
    ...

def test_get_labels_groupby_for_Int64(writable: bool) -> None:
    ...

def test_tracemalloc_works_for_StringHashTable() -> None:
    ...

def test_tracemalloc_for_empty_StringHashTable() -> None:
    ...

@pytest.mark.parametrize('N', range(1, 110))
def test_no_reallocation_StringHashTable(N: int) -> None:
    ...

@pytest.mark.parametrize('table_type, dtype', [
    (ht.Float64HashTable, np.float64),
    (ht.Float32HashTable, np.float32),
    (ht.Complex128HashTable, np.complex128),
    (ht.Complex64HashTable, np.complex64)
])
class TestHashTableWithNans:
    def test_get_set_contains_len(self, table_type: type[ht.HashTable], dtype: np.dtype) -> None:
        ...

    def test_map_locations(self, table_type: type[ht.HashTable], dtype: np.dtype) -> None:
        ...

    def test_unique(self, table_type: type[ht.HashTable], dtype: np.dtype) -> None:
        ...

def test_unique_for_nan_objects_floats() -> None:
    ...

def test_unique_for_nan_objects_complex() -> None:
    ...

def test_unique_for_nan_objects_tuple() -> None:
    ...

@pytest.mark.parametrize('dtype', [
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
    np.intp
])
class TestHelpFunctions:
    def test_value_count(self, dtype: np.dtype, writable: bool) -> None:
        ...

    def test_value_count_mask(self, dtype: np.dtype) -> None:
        ...

    def test_value_count_stable(self, dtype: np.dtype, writable: bool) -> None:
        ...

    def test_duplicated_first(self, dtype: np.dtype, writable: bool) -> None:
        ...

    def test_ismember_yes(self, dtype: np.dtype, writable: bool) -> None:
        ...

    def test_ismember_no(self, dtype: np.dtype) -> None:
        ...

    def test_mode(self, dtype: np.dtype, writable: bool) -> None:
        ...

    def test_mode_stable(self, dtype: np.dtype, writable: bool) -> None:
        ...

def test_modes_with_nans() -> None:
    ...

def test_unique_label_indices_intp(writable: bool) -> None:
    ...

def test_unique_label_indices() -> None:
    ...

@pytest.mark.parametrize('dtype', [
    np.float64,
    np.float32,
    np.complex128,
    np.complex64
])
class TestHelpFunctionsWithNans:
    def test_value_count(self, dtype: np.dtype) -> None:
        ...

    def test_duplicated_first(self, dtype: np.dtype) -> None:
        ...

    def test_ismember_yes(self, dtype: np.dtype) -> None:
        ...

    def test_ismember_no(self, dtype: np.dtype) -> None:
        ...

    def test_mode(self, dtype: np.dtype) -> None:
        ...

def test_ismember_tuple_with_nans() -> None:
    ...

def test_float_complex_int_are_equal_as_objects() -> None:
    ...