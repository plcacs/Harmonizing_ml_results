```python
import numpy as np
import pandas as pd
from pandas import CategoricalIndex, DataFrame, Index, IntervalIndex, MultiIndex, Series
from typing import Any, List, Optional, Tuple, Union

def test_set_ops_error_cases(
    idx: Any,
    case: Union[float, str],
    sort: Any,
    method: str
) -> None: ...

def test_intersection_base(
    idx: Any,
    sort: Any,
    klass: Any
) -> None: ...

def test_union_base(
    idx: Any,
    sort: Any,
    klass: Any
) -> None: ...

def test_difference_base(
    idx: Any,
    sort: Any
) -> None: ...

def test_symmetric_difference(
    idx: Any,
    sort: Any
) -> None: ...

def test_multiindex_symmetric_difference() -> None: ...

def test_empty(
    idx: Any
) -> None: ...

def test_difference(
    idx: Any,
    sort: Any
) -> None: ...

def test_difference_sort_special() -> None: ...

def test_difference_sort_special_true() -> None: ...

def test_difference_sort_incomparable() -> None: ...

def test_difference_sort_incomparable_true() -> None: ...

def test_union(
    idx: Any,
    sort: Any
) -> None: ...

def test_union_with_regular_index(
    idx: Any,
    using_infer_string: Any
) -> None: ...

def test_intersection(
    idx: Any,
    sort: Any
) -> None: ...

def test_setop_with_categorical(
    idx: Any,
    sort: Any,
    method: str
) -> None: ...

def test_intersection_non_object(
    idx: Any,
    sort: Any
) -> None: ...

def test_intersect_equal_sort() -> None: ...

def test_intersect_equal_sort_true() -> None: ...

def test_union_sort_other_empty(
    slice_: slice
) -> None: ...

def test_union_sort_other_empty_sort() -> None: ...

def test_union_sort_other_incomparable() -> None: ...

def test_union_sort_other_incomparable_sort() -> None: ...

def test_union_non_object_dtype_raises() -> None: ...

def test_union_empty_self_different_names() -> None: ...

def test_union_multiindex_empty_rangeindex() -> None: ...

def test_setops_sort_validation(
    method: str
) -> None: ...

def test_difference_keep_ea_dtypes(
    any_numeric_ea_dtype: Any,
    val: Any
) -> None: ...

def test_symmetric_difference_keeping_ea_dtype(
    any_numeric_ea_dtype: Any,
    val: Any
) -> None: ...

def test_intersect_with_duplicates(
    tuples: List[Tuple[str, str]],
    exp_tuples: List[Tuple[str, str]]
) -> None: ...

def test_maybe_match_names(
    data: Tuple[Any, ...],
    names: Optional[List[Optional[str]]],
    expected: List[Optional[str]]
) -> None: ...

def test_intersection_equal_different_names() -> None: ...

def test_intersection_different_names() -> None: ...

def test_intersection_with_missing_values_on_both_sides(
    nulls_fixture: Any
) -> None: ...

def test_union_with_missing_values_on_both_sides(
    nulls_fixture: Any
) -> None: ...

def test_union_nan_got_duplicated(
    dtype: str,
    sort: Optional[bool]
) -> None: ...

def test_union_keep_ea_dtype(
    any_numeric_ea_dtype: Any,
    val: int
) -> None: ...

def test_union_with_duplicates_keep_ea_dtype(
    dupe_val: Any,
    any_numeric_ea_dtype: Any
) -> None: ...

def test_union_duplicates(
    index: Any,
    request: Any
) -> None: ...

def test_union_keep_dtype_precision(
    any_real_numeric_dtype: Any
) -> None: ...

def test_union_keep_ea_dtype_with_na(
    any_numeric_ea_dtype: Any
) -> None: ...

def test_intersection_lexsort_depth(
    levels1: List[List[Any]],
    levels2: List[List[Any]],
    codes1: List[List[int]],
    codes2: List[List[int]],
    names: List[str]
) -> None: ...

def test_intersection_with_non_lex_sorted_categories(
    a: Any,
    b_ordered: bool
) -> None: ...

def test_intersection_keep_ea_dtypes(
    val: Any,
    any_numeric_ea_dtype: Any
) -> None: ...

def test_union_with_na_when_constructing_dataframe() -> None: ...
```