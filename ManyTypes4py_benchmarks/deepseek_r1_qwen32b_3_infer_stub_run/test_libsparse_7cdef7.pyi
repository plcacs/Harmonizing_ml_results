import operator
import numpy as np
import pytest
import pandas._libs.sparse as splib
from pandas.core.arrays.sparse import BlockIndex, IntIndex

@pytest.fixture
def test_length() -> int:
    ...

@pytest.fixture
def cases() -> list[list[list[int]]]:
    ...

class TestSparseIndexUnion:
    @pytest.mark.parametrize('xloc, xlen, yloc, ylen, eloc, elen', [
        [[list[int]], [list[int]], [list[int]], [list[int]], [list[int]], [list[int]]],
        ...
    ])
    def test_index_make_union(
        self,
        xloc: list[int],
        xlen: list[int],
        yloc: list[int],
        ylen: list[int],
        eloc: list[int],
        elen: list[int],
        test_length: int
    ) -> None:
        ...

    def test_int_index_make_union(self) -> None:
        ...

class TestSparseIndexIntersect:
    @td.skip_if_windows
    def test_intersect(
        self,
        cases: list[list[list[int]]],
        test_length: int
    ) -> None:
        ...

    def test_intersect_empty(self) -> None:
        ...

    @pytest.mark.parametrize('case', [
        IntIndex(5, np.array([], dtype=np.int32)),
        ...
    ])
    def test_intersect_identical(self, case: IntIndex) -> None:
        ...

class TestSparseIndexCommon:
    def test_int_internal(self) -> None:
        ...

    def test_block_internal(self) -> None:
        ...

    @pytest.mark.parametrize('kind', ['integer', 'block'])
    def test_lookup(self, kind: str) -> None:
        ...

    @pytest.mark.parametrize('kind', ['integer', 'block'])
    def test_lookup_array(self, kind: str) -> None:
        ...

    @pytest.mark.parametrize('idx, expected', [
        [int, int],
        ...
    ])
    def test_lookup_basics(self, idx: int, expected: int) -> None:
        ...

class TestBlockIndex:
    def test_block_internal(self) -> None:
        ...

    @pytest.mark.parametrize('i', [int])
    def test_make_block_boundary(self, i: int) -> None:
        ...

    def test_equals(self) -> None:
        ...

    def test_check_integrity(self) -> None:
        ...

    def test_to_int_index(self) -> None:
        ...

    def test_to_block_index(self) -> None:
        ...

class TestIntIndex:
    def test_check_integrity(self) -> None:
        ...

    def test_int_internal(self) -> None:
        ...

    def test_equals(self) -> None:
        ...

    def test_to_block_index(
        self,
        cases: list[list[list[int]]],
        test_length: int
    ) -> None:
        ...

    def test_to_int_index(self) -> None:
        ...

class TestSparseOperators:
    @pytest.mark.parametrize('opname', ['add', 'sub', 'mul', 'truediv', 'floordiv'])
    def test_op(
        self,
        opname: str,
        cases: list[list[list[int]]],
        test_length: int
    ) -> None:
        ...