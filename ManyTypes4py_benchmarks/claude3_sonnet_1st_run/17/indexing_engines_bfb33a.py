"""
Benchmarks in this file depend mostly on code in _libs/

We have to created masked arrays to test the masked engine though. The
array is unpacked on the Cython level.

If a PR does not edit anything in _libs, it is very unlikely that benchmarks
in this file will be affected.
"""
from typing import Any, List, Tuple, Type, TypeVar, Union, cast

import numpy as np
from pandas._libs import index as libindex
from pandas.core.arrays import BaseMaskedArray

EngineType = TypeVar('EngineType')

def _get_numeric_engines() -> List[Tuple[Any, np.dtype]]:
    engine_names: List[Tuple[str, np.dtype]] = [
        ('Int64Engine', np.int64), ('Int32Engine', np.int32), 
        ('Int16Engine', np.int16), ('Int8Engine', np.int8), 
        ('UInt64Engine', np.uint64), ('UInt32Engine', np.uint32), 
        ('UInt16engine', np.uint16), ('UInt8Engine', np.uint8), 
        ('Float64Engine', np.float64), ('Float32Engine', np.float32)
    ]
    return [(getattr(libindex, engine_name), dtype) for engine_name, dtype in engine_names if hasattr(libindex, engine_name)]

def _get_masked_engines() -> List[Tuple[Any, str]]:
    engine_names: List[Tuple[str, str]] = [
        ('MaskedInt64Engine', 'Int64'), ('MaskedInt32Engine', 'Int32'), 
        ('MaskedInt16Engine', 'Int16'), ('MaskedInt8Engine', 'Int8'), 
        ('MaskedUInt64Engine', 'UInt64'), ('MaskedUInt32Engine', 'UInt32'), 
        ('MaskedUInt16engine', 'UInt16'), ('MaskedUInt8Engine', 'UInt8'), 
        ('MaskedFloat64Engine', 'Float64'), ('MaskedFloat32Engine', 'Float32')
    ]
    return [(getattr(libindex, engine_name), dtype) for engine_name, dtype in engine_names if hasattr(libindex, engine_name)]

class NumericEngineIndexing:
    params: List[Union[List[Tuple[Any, np.dtype]], List[str], List[bool], List[int]]] = [
        _get_numeric_engines(), 
        ['monotonic_incr', 'monotonic_decr', 'non_monotonic'], 
        [True, False], 
        [10 ** 5, 2 * 10 ** 6]
    ]
    param_names: List[str] = ['engine_and_dtype', 'index_type', 'unique', 'N']

    def setup(self, engine_and_dtype: Tuple[Any, np.dtype], index_type: str, unique: bool, N: int) -> None:
        engine, dtype = engine_and_dtype
        if index_type == 'monotonic_incr':
            if unique:
                arr = np.arange(N * 3, dtype=dtype)
            else:
                arr = np.array([1, 2, 3], dtype=dtype).repeat(N)
        elif index_type == 'monotonic_decr':
            if unique:
                arr = np.arange(N * 3, dtype=dtype)[::-1]
            else:
                arr = np.array([3, 2, 1], dtype=dtype).repeat(N)
        else:
            assert index_type == 'non_monotonic'
            if unique:
                arr = np.empty(N * 3, dtype=dtype)
                arr[:N] = np.arange(N * 2, N * 3, dtype=dtype)
                arr[N:] = np.arange(N * 2, dtype=dtype)
            else:
                arr = np.array([1, 2, 3], dtype=dtype).repeat(N)
        self.data = engine(arr)
        self.data.get_loc(2)
        self.key_middle = arr[len(arr) // 2]
        self.key_early = arr[2]

    def time_get_loc(self, engine_and_dtype: Tuple[Any, np.dtype], index_type: str, unique: bool, N: int) -> None:
        self.data.get_loc(self.key_early)

    def time_get_loc_near_middle(self, engine_and_dtype: Tuple[Any, np.dtype], index_type: str, unique: bool, N: int) -> None:
        self.data.get_loc(self.key_middle)

class MaskedNumericEngineIndexing:
    params: List[Union[List[Tuple[Any, str]], List[str], List[bool], List[int]]] = [
        _get_masked_engines(), 
        ['monotonic_incr', 'monotonic_decr', 'non_monotonic'], 
        [True, False], 
        [10 ** 5, 2 * 10 ** 6]
    ]
    param_names: List[str] = ['engine_and_dtype', 'index_type', 'unique', 'N']

    def setup(self, engine_and_dtype: Tuple[Any, str], index_type: str, unique: bool, N: int) -> None:
        engine, dtype = engine_and_dtype
        dtype = dtype.lower()
        if index_type == 'monotonic_incr':
            if unique:
                arr = np.arange(N * 3, dtype=dtype)
            else:
                arr = np.array([1, 2, 3], dtype=dtype).repeat(N)
            mask = np.zeros(N * 3, dtype=np.bool_)
        elif index_type == 'monotonic_decr':
            if unique:
                arr = np.arange(N * 3, dtype=dtype)[::-1]
            else:
                arr = np.array([3, 2, 1], dtype=dtype).repeat(N)
            mask = np.zeros(N * 3, dtype=np.bool_)
        else:
            assert index_type == 'non_monotonic'
            if unique:
                arr = np.zeros(N * 3, dtype=dtype)
                arr[:N] = np.arange(N * 2, N * 3, dtype=dtype)
                arr[N:] = np.arange(N * 2, dtype=dtype)
            else:
                arr = np.array([1, 2, 3], dtype=dtype).repeat(N)
            mask = np.zeros(N * 3, dtype=np.bool_)
            mask[-1] = True
        self.data = engine(BaseMaskedArray(arr, mask))
        self.data.get_loc(2)
        self.key_middle = arr[len(arr) // 2]
        self.key_early = arr[2]

    def time_get_loc(self, engine_and_dtype: Tuple[Any, str], index_type: str, unique: bool, N: int) -> None:
        self.data.get_loc(self.key_early)

    def time_get_loc_near_middle(self, engine_and_dtype: Tuple[Any, str], index_type: str, unique: bool, N: int) -> None:
        self.data.get_loc(self.key_middle)

class ObjectEngineIndexing:
    params: List[Tuple[str, ...]] = [('monotonic_incr', 'monotonic_decr', 'non_monotonic')]
    param_names: List[str] = ['index_type']

    def setup(self, index_type: str) -> None:
        N = 10 ** 5
        values = list('a' * N + 'b' * N + 'c' * N)
        arr = {'monotonic_incr': np.array(values, dtype=object), 
               'monotonic_decr': np.array(list(reversed(values)), dtype=object), 
               'non_monotonic': np.array(list('abc') * N, dtype=object)}[index_type]
        self.data = libindex.ObjectEngine(arr)
        self.data.get_loc('b')

    def time_get_loc(self, index_type: str) -> None:
        self.data.get_loc('b')
