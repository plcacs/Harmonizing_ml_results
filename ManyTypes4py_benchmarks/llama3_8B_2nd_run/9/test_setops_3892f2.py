from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import Index, Series
import pandas._testing as tm
from pandas.core.algorithms import safe_sort

def equal_contents(arr1: np.ndarray, arr2: np.ndarray) -> bool:
    """
    Checks if the set of unique elements of arr1 and arr2 are equivalent.
    """
    return frozenset(arr1) == frozenset(arr2)

class TestIndexSetOps:
    @pytest.mark.parametrize('method: str', ['union', 'intersection', 'difference', 'symmetric_difference'])
    def test_setops_sort_validation(self, method: str) -> None:
        idx1: Index = Index(['a', 'b'])
        idx2: Index = Index(['b', 'c'])
        with pytest.raises(ValueError, match="The 'sort' keyword only takes"):
            getattr(idx1, method)(idx2, sort=2)
        getattr(idx1, method)(idx2, sort=True)

    def test_setops_preserve_object_dtype(self) -> None:
        idx: Index = Index([1, 2, 3], dtype=object)
        result: Index = idx.intersection(idx[1:])
        expected: Index = idx[1:]
        tm.assert_index_equal(result, expected)
        result: Index = idx.intersection(idx[1:][::-1])
        tm.assert_index_equal(result, expected)
        result: Index = idx._union(idx[1:], sort=None)
        expected: Index = idx
        tm.assert_numpy_array_equal(result, expected.values)
        result: Index = idx.union(idx[1:], sort=None)
        tm.assert_index_equal(result, expected)
        result: Index = idx._union(idx[1:][::-1], sort=None)
        tm.assert_numpy_array_equal(result, expected.values)
        result: Index = idx.union(idx[1:][::-1], sort=None)
        tm.assert_index_equal(result, expected)

    # ... and so on for the rest of the methods
