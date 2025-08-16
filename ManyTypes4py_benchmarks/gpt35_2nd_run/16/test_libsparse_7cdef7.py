from typing import List, Tuple
import numpy as np
import pytest
import pandas._libs.sparse as splib
import pandas.util._test_decorators as td
from pandas import Series
import pandas._testing as tm
from pandas.core.arrays.sparse import BlockIndex, IntIndex, make_sparse_index

def test_length() -> int:
    return 20

def cases(request) -> List[List[List[int]]]:
    return request.param

class TestSparseIndexUnion:

    def test_index_make_union(self, xloc: List[int], xlen: List[int], yloc: List[int], ylen: List[int], eloc: List[int], elen: List[int], test_length: int) -> None:
    
    def test_int_index_make_union(self) -> None:

class TestSparseIndexIntersect:

    def test_intersect(self, cases: List[List[List[int]]], test_length: int) -> None:
    
    def test_intersect_empty(self) -> None:

    def test_intersect_identical(self, case: IntIndex) -> None:

class TestSparseIndexCommon:

    def test_int_internal(self) -> None:

    def test_block_internal(self) -> None:

    def test_lookup(self, kind: str) -> None:

    def test_lookup_array(self, kind: str) -> None:

    def test_lookup_basics(self, idx: int, expected: int) -> None:

class TestBlockIndex:

    def test_block_internal(self) -> None:

    def test_make_block_boundary(self, i: int) -> None:

    def test_equals(self) -> None:

    def test_check_integrity(self) -> None:

    def test_to_int_index(self) -> None:

    def test_to_block_index(self) -> None:

class TestIntIndex:

    def test_check_integrity(self) -> None:

    def test_int_internal(self) -> None:

    def test_equals(self) -> None:

    def test_to_block_index(self, cases: List[List[List[int]]], test_length: int) -> None:

    def test_to_int_index(self) -> None:

class TestSparseOperators:

    def test_op(self, opname: str, cases: List[List[List[int]]], test_length: int) -> None:
