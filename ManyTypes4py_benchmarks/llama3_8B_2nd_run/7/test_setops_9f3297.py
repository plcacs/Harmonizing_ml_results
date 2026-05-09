import numpy as np
import pytest
import pandas as pd
from pandas import CategoricalIndex, DataFrame, Index, IntervalIndex, MultiIndex, Series
import pandas._testing as tm
from pandas.api.types import is_float_dtype, is_unsigned_integer_dtype

@pytest.mark.parametrize('case', [0.5, 'xxx'])
@pytest.mark.parametrize('method', ['intersection', 'union', 'difference', 'symmetric_difference'])
def test_set_ops_error_cases(idx: MultiIndex, case: object, sort: bool, method: str) -> None:
    # ...

@pytest.mark.parametrize('klass', [MultiIndex, np.array, Series, list])
def test_intersection_base(idx: MultiIndex, sort: bool, klass: type) -> None:
    # ...

def test_difference_base(idx: MultiIndex, sort: bool) -> None:
    # ...

def test_symmetric_difference(idx: MultiIndex, sort: bool) -> None:
    # ...

def test_union(idx: MultiIndex, sort: bool) -> None:
    # ...

def test_intersection(idx: MultiIndex, sort: bool) -> None:
    # ...

@pytest.mark.parametrize('method', ['union', 'intersection', 'difference', 'symmetric_difference'])
def test_setops_sort_validation(method: str) -> None:
    # ...

def test_intersection_equal_different_names() -> None:
    # ...

def test_intersection_with_missing_values_on_both_sides(nulls_fixture: pd.NA) -> None:
    # ...

def test_union_with_missing_values_on_both_sides(nulls_fixture: pd.NA) -> None:
    # ...

@pytest.mark.parametrize('dupe_val', [3, pd.NA])
def test_union_with_duplicates_keep_ea_dtype(dupe_val: object, any_numeric_ea_dtype: type) -> None:
    # ...

def test_union_duplicates(index: Index) -> None:
    # ...

def test_union_keep_dtype_precision(any_real_numeric_dtype: type) -> None:
    # ...

def test_union_keep_ea_dtype_with_na(any_numeric_ea_dtype: type) -> None:
    # ...

@pytest.mark.parametrize('levels1, levels2, codes1, codes2, names', [([['a', 'b', 'c'], [0, '']], [['c', 'd', 'b'], ['']], [[0, 1, 2], [1, 1, 1]], [[0, 1, 2], [0, 0, 0]], ['name1', 'name2'])])
def test_intersection_lexsort_depth(levels1: list, levels2: list, codes1: list, codes2: list, names: list) -> None:
    # ...

@pytest.mark.parametrize('a', [pd.Categorical(['a', 'b'], categories=['a', 'b']), ['a', 'b']])
@pytest.mark.parametrize('b_ordered', [True, False])
def test_intersection_with_non_lex_sorted_categories(a: object, b_ordered: bool) -> None:
    # ...

@pytest.mark.parametrize('val', [pd.NA, 100])
def test_intersection_keep_ea_dtypes(val: object, any_numeric_ea_dtype: type) -> None:
    # ...

def test_union_with_na_when_constructing_dataframe() -> None:
    # ...
