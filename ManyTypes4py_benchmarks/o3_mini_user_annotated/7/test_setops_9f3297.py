from typing import Any, List, Optional, Union
import numpy as np
import pytest

import pandas as pd
from pandas import (
    CategoricalIndex,
    DataFrame,
    Index,
    IntervalIndex,
    MultiIndex,
    Series,
)
import pandas._testing as tm
from pandas.api.types import (
    is_float_dtype,
    is_unsigned_integer_dtype,
)


@pytest.mark.parametrize("case", [0.5, "xxx"])
@pytest.mark.parametrize(
    "method", ["intersection", "union", "difference", "symmetric_difference"]
)
def test_set_ops_error_cases(idx: MultiIndex, case: Union[float, str], sort: Optional[bool], method: str) -> None:
    # non-iterable input
    msg = "Input must be Index or array-like"
    with pytest.raises(TypeError, match=msg):
        getattr(idx, method)(case, sort=sort)


@pytest.mark.parametrize("klass", [MultiIndex, np.array, Series, list])
def test_intersection_base(idx: MultiIndex, sort: Optional[bool], klass: Any) -> None:
    first: MultiIndex = idx[2::-1]  # first 3 elements reversed
    second: Any = idx[:5]

    if klass is not MultiIndex:
        second = klass(second.values)

    intersect = first.intersection(second, sort=sort)
    if sort is None:
        expected = first.sort_values()
    else:
        expected = first
    tm.assert_index_equal(intersect, expected)

    msg = "other must be a MultiIndex or a list of tuples"
    with pytest.raises(TypeError, match=msg):
        first.intersection([1, 2, 3], sort=sort)


@pytest.mark.arm_slow
@pytest.mark.parametrize("klass", [MultiIndex, np.array, Series, list])
def test_union_base(idx: MultiIndex, sort: Optional[bool], klass: Any) -> None:
    first: MultiIndex = idx[::-1]
    second: Any = idx[:5]

    if klass is not MultiIndex:
        second = klass(second.values)

    union = first.union(second, sort=sort)
    if sort is None:
        expected = first.sort_values()
    else:
        expected = first
    tm.assert_index_equal(union, expected)

    msg = "other must be a MultiIndex or a list of tuples"
    with pytest.raises(TypeError, match=msg):
        first.union([1, 2, 3], sort=sort)


def test_difference_base(idx: MultiIndex, sort: Optional[bool]) -> None:
    second: MultiIndex = idx[4:]
    answer: MultiIndex = idx[:4]
    result: MultiIndex = idx.difference(second, sort=sort)

    if sort is None:
        answer = answer.sort_values()

    assert result.equals(answer)
    tm.assert_index_equal(result, answer)

    # GH 10149
    cases: List[Any] = [klass(second.values) for klass in [np.array, Series, list]]
    for case in cases:
        result = idx.difference(case, sort=sort)
        tm.assert_index_equal(result, answer)

    msg = "other must be a MultiIndex or a list of tuples"
    with pytest.raises(TypeError, match=msg):
        idx.difference([1, 2, 3], sort=sort)


def test_symmetric_difference(idx: MultiIndex, sort: Optional[bool]) -> None:
    first: MultiIndex = idx[1:]
    second: MultiIndex = idx[:-1]
    answer: MultiIndex = idx[[-1, 0]]
    result: MultiIndex = first.symmetric_difference(second, sort=sort)

    if sort is None:
        answer = answer.sort_values()

    tm.assert_index_equal(result, answer)

    # GH 10149
    cases: List[Any] = [klass(second.values) for klass in [np.array, Series, list]]
    for case in cases:
        result = first.symmetric_difference(case, sort=sort)
        tm.assert_index_equal(result, answer)

    msg = "other must be a MultiIndex or a list of tuples"
    with pytest.raises(TypeError, match=msg):
        first.symmetric_difference([1, 2, 3], sort=sort)


def test_multiindex_symmetric_difference() -> None:
    # GH 13490
    idx: MultiIndex = MultiIndex.from_product([["a", "b"], ["A", "B"]], names=["a", "b"])
    result: MultiIndex = idx.symmetric_difference(idx)
    assert result.names == idx.names

    idx2: MultiIndex = idx.copy().rename(["A", "B"])
    result = idx.symmetric_difference(idx2)
    assert result.names == [None, None]


def test_empty(idx: MultiIndex) -> None:
    # GH 15270
    assert not idx.empty
    assert idx[:0].empty


def test_difference(idx: MultiIndex, sort: Optional[bool]) -> None:
    first: MultiIndex = idx
    result: MultiIndex = first.difference(idx[-3:], sort=sort)
    vals = idx[:-3].values

    if sort is None:
        vals = sorted(vals)

    expected: MultiIndex = MultiIndex.from_tuples(vals, sortorder=0, names=idx.names)

    assert isinstance(result, MultiIndex)
    assert result.equals(expected)
    assert result.names == idx.names
    tm.assert_index_equal(result, expected)

    # empty difference: reflexive
    result = idx.difference(idx, sort=sort)
    expected = idx[:0]
    assert result.equals(expected)
    assert result.names == idx.names

    # empty difference: superset
    result = idx[-3:].difference(idx, sort=sort)
    expected = idx[:0]
    assert result.equals(expected)
    assert result.names == idx.names

    # empty difference: degenerate
    result = idx[:0].difference(idx, sort=sort)
    expected = idx[:0]
    assert result.equals(expected)
    assert result.names == idx.names

    # names not the same
    chunklet: MultiIndex = idx[-3:]
    chunklet.names = ["foo", "baz"]
    result = first.difference(chunklet, sort=sort)
    assert result.names == (None, None)

    # empty, but non-equal
    result = idx.difference(idx.sortlevel(1)[0], sort=sort)
    assert len(result) == 0

    # raise Exception called with non-MultiIndex
    result = first.difference(first.values, sort=sort)
    assert result.equals(first[:0])

    # name from empty array
    result = first.difference([], sort=sort)
    assert first.equals(result)
    assert first.names == result.names

    # name from non-empty array
    result = first.difference([("foo", "one")], sort=sort)
    expected = MultiIndex.from_tuples(
        [("bar", "one"), ("baz", "two"), ("foo", "two"), ("qux", "one"), ("qux", "two")]
    )
    expected.names = first.names
    assert first.names == result.names

    msg = "other must be a MultiIndex or a list of tuples"
    with pytest.raises(TypeError, match=msg):
        first.difference([1, 2, 3, 4, 5], sort=sort)


def test_difference_sort_special() -> None:
    # GH-24959
    idx: MultiIndex = MultiIndex.from_product([[1, 0], ["a", "b"]])
    # sort=None, the default
    result: MultiIndex = idx.difference([])
    tm.assert_index_equal(result, idx)


def test_difference_sort_special_true() -> None:
    idx: MultiIndex = MultiIndex.from_product([[1, 0], ["a", "b"]])
    result: MultiIndex = idx.difference([], sort=True)
    expected: MultiIndex = MultiIndex.from_product([[0, 1], ["a", "b"]])
    tm.assert_index_equal(result, expected)


def test_difference_sort_incomparable() -> None:
    # GH-24959
    idx: MultiIndex = MultiIndex.from_product([[1, pd.Timestamp("2000"), 2], ["a", "b"]])

    other: MultiIndex = MultiIndex.from_product([[3, pd.Timestamp("2000"), 4], ["c", "d"]])
    # sort=None, the default
    msg = "sort order is undefined for incomparable objects"
    with tm.assert_produces_warning(RuntimeWarning, match=msg):
        result: MultiIndex = idx.difference(other)
    tm.assert_index_equal(result, idx)

    # sort=False
    result = idx.difference(other, sort=False)
    tm.assert_index_equal(result, idx)


def test_difference_sort_incomparable_true() -> None:
    idx: MultiIndex = MultiIndex.from_product([[1, pd.Timestamp("2000"), 2], ["a", "b"]])
    other: MultiIndex = MultiIndex.from_product([[3, pd.Timestamp("2000"), 4], ["c", "d"]])

    msg = "'values' is not ordered, please explicitly specify the categories order "
    with pytest.raises(TypeError, match=msg):
        idx.difference(other, sort=True)


def test_union(idx: MultiIndex, sort: Optional[bool]) -> None:
    piece1: MultiIndex = idx[:5][::-1]
    piece2: MultiIndex = idx[3:]

    the_union: MultiIndex = piece1.union(piece2, sort=sort)

    if sort in (None, False):
        tm.assert_index_equal(the_union.sort_values(), idx.sort_values())
    else:
        tm.assert_index_equal(the_union, idx)

    # corner case, pass self or empty thing:
    the_union = idx.union(idx, sort=sort)
    tm.assert_index_equal(the_union, idx)

    the_union = idx.union(idx[:0], sort=sort)
    tm.assert_index_equal(the_union, idx)

    tuples = idx.values
    result = idx[:4].union(tuples[4:], sort=sort)
    if sort is None:
        tm.assert_index_equal(result.sort_values(), idx.sort_values())
    else:
        assert result.equals(idx)


def test_union_with_regular_index(idx: MultiIndex, using_infer_string: bool) -> None:
    other: Index = Index(["A", "B", "C"])

    result: Index = other.union(idx)
    assert ("foo", "one") in result
    assert "B" in result

    if using_infer_string:
        with pytest.raises(NotImplementedError, match="Can only union"):
            idx.union(other)
    else:
        msg = "The values in the array are unorderable"
        with tm.assert_produces_warning(RuntimeWarning, match=msg):
            result2: Index = idx.union(other)
        assert not result.equals(result2)


def test_intersection(idx: MultiIndex, sort: Optional[bool]) -> None:
    piece1: MultiIndex = idx[:5][::-1]
    piece2: MultiIndex = idx[3:]

    the_int: MultiIndex = piece1.intersection(piece2, sort=sort)

    if sort in (None, True):
        tm.assert_index_equal(the_int, idx[3:5])
    else:
        tm.assert_index_equal(the_int.sort_values(), idx[3:5])

    # corner case, pass self
    the_int = idx.intersection(idx, sort=sort)
    tm.assert_index_equal(the_int, idx)

    # empty intersection: disjoint
    empty: MultiIndex = idx[:2].intersection(idx[2:], sort=sort)
    expected = idx[:0]
    assert empty.equals(expected)

    tuples = idx.values
    result = idx.intersection(tuples)
    assert result.equals(idx)


@pytest.mark.parametrize(
    "method", ["intersection", "union", "difference", "symmetric_difference"]
)
def test_setop_with_categorical(idx: MultiIndex, sort: Optional[bool], method: str) -> None:
    other = idx.to_flat_index().astype("category")
    res_names: List[Optional[Any]] = [None] * idx.nlevels

    result = getattr(idx, method)(other, sort=sort)
    expected = getattr(idx, method)(idx, sort=sort).rename(res_names)
    tm.assert_index_equal(result, expected)

    result = getattr(idx, method)(other[:5], sort=sort)
    expected = getattr(idx, method)(idx[:5], sort=sort).rename(res_names)
    tm.assert_index_equal(result, expected)


def test_intersection_non_object(idx: MultiIndex, sort: Optional[bool]) -> None:
    other: Index = Index(range(3), name="foo")

    result = idx.intersection(other, sort=sort)
    expected = MultiIndex(levels=idx.levels, codes=[[]] * idx.nlevels, names=None)
    tm.assert_index_equal(result, expected, exact=True)

    result = idx.intersection(np.asarray(other)[:0], sort=sort)
    expected = MultiIndex(levels=idx.levels, codes=[[]] * idx.nlevels, names=idx.names)
    tm.assert_index_equal(result, expected, exact=True)

    msg = "other must be a MultiIndex or a list of tuples"
    with pytest.raises(TypeError, match=msg):
        idx.intersection(np.asarray(other), sort=sort)


def test_intersect_equal_sort() -> None:
    # GH-24959
    idx: MultiIndex = MultiIndex.from_product([[1, 0], ["a", "b"]])
    tm.assert_index_equal(idx.intersection(idx, sort=False), idx)
    tm.assert_index_equal(idx.intersection(idx, sort=None), idx)


def test_intersect_equal_sort_true() -> None:
    idx: MultiIndex = MultiIndex.from_product([[1, 0], ["a", "b"]])
    expected: MultiIndex = MultiIndex.from_product([[0, 1], ["a", "b"]])
    result: MultiIndex = idx.intersection(idx, sort=True)
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("slice_", [slice(None), slice(0)])
def test_union_sort_other_empty(slice_: slice) -> None:
    # https://github.com/pandas-dev/pandas/issues/24959
    idx: MultiIndex = MultiIndex.from_product([[1, 0], ["a", "b"]])

    # default, sort=None
    other: MultiIndex = idx[slice_]
    tm.assert_index_equal(idx.union(other), idx)
    tm.assert_index_equal(other.union(idx), idx)

    # sort=False
    tm.assert_index_equal(idx.union(other, sort=False), idx)


def test_union_sort_other_empty_sort() -> None:
    idx: MultiIndex = MultiIndex.from_product([[1, 0], ["a", "b"]])
    other: MultiIndex = idx[:0]
    result: MultiIndex = idx.union(other, sort=True)
    expected: MultiIndex = MultiIndex.from_product([[0, 1], ["a", "b"]])
    tm.assert_index_equal(result, expected)


def test_union_sort_other_incomparable() -> None:
    # https://github.com/pandas-dev/pandas/issues/24959
    idx: MultiIndex = MultiIndex.from_product([[1, pd.Timestamp("2000")], ["a", "b"]])

    # default, sort=None
    with tm.assert_produces_warning(RuntimeWarning, match="are unorderable"):
        result: MultiIndex = idx.union(idx[:1])
    tm.assert_index_equal(result, idx)

    # sort=False
    result = idx.union(idx[:1], sort=False)
    tm.assert_index_equal(result, idx)


def test_union_sort_other_incomparable_sort() -> None:
    idx: MultiIndex = MultiIndex.from_product([[1, pd.Timestamp("2000")], ["a", "b"]])
    msg = "'<' not supported between instances of 'Timestamp' and 'int'"
    with pytest.raises(TypeError, match=msg):
        idx.union(idx[:1], sort=True)


def test_union_non_object_dtype_raises() -> None:
    # GH#32646 raise NotImplementedError instead of less-informative error
    mi: MultiIndex = MultiIndex.from_product([["a", "b"], [1, 2]])

    idx_nonobject = mi.levels[1]

    msg = "Can only union MultiIndex with MultiIndex or Index of tuples"
    with pytest.raises(NotImplementedError, match=msg):
        mi.union(idx_nonobject)


def test_union_empty_self_different_names() -> None:
    # GH#38423
    mi: MultiIndex = MultiIndex.from_arrays([[]])
    mi2: MultiIndex = MultiIndex.from_arrays([[1, 2], [3, 4]], names=["a", "b"])
    result: MultiIndex = mi.union(mi2)
    expected: MultiIndex = MultiIndex.from_arrays([[1, 2], [3, 4]])
    tm.assert_index_equal(result, expected)


def test_union_multiindex_empty_rangeindex() -> None:
    # GH#41234
    mi: MultiIndex = MultiIndex.from_arrays([[1, 2], [3, 4]], names=["a", "b"])
    ri: pd.RangeIndex = pd.RangeIndex(0)

    result_left = mi.union(ri)
    tm.assert_index_equal(mi, result_left, check_names=False)

    result_right = ri.union(mi)
    tm.assert_index_equal(mi, result_right, check_names=False)


@pytest.mark.parametrize(
    "method", ["union", "intersection", "difference", "symmetric_difference"]
)
def test_setops_sort_validation(method: str) -> None:
    idx1: MultiIndex = MultiIndex.from_product([["a", "b"], [1, 2]])
    idx2: MultiIndex = MultiIndex.from_product([["b", "c"], [1, 2]])

    with pytest.raises(ValueError, match="The 'sort' keyword only takes"):
        getattr(idx1, method)(idx2, sort=2)

    getattr(idx1, method)(idx2, sort=True)


@pytest.mark.parametrize("val", [pd.NA, 100])
def test_difference_keep_ea_dtypes(any_numeric_ea_dtype: Any, val: Union[int, Any]) -> None:
    # GH#48606
    midx: MultiIndex = MultiIndex.from_arrays(
        [Series([1, 2], dtype=any_numeric_ea_dtype), [2, 1]], names=["a", None]
    )
    midx2: MultiIndex = MultiIndex.from_arrays(
        [Series([1, 2, val], dtype=any_numeric_ea_dtype), [1, 1, 3]]
    )
    result: MultiIndex = midx.difference(midx2)
    expected: MultiIndex = MultiIndex.from_arrays([Series([1], dtype=any_numeric_ea_dtype), [2]])
    tm.assert_index_equal(result, expected)

    result = midx.difference(midx.sort_values(ascending=False))
    expected = MultiIndex.from_arrays(
        [Series([], dtype=any_numeric_ea_dtype), Series([], dtype=np.int64)],
        names=["a", None],
    )
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("val", [pd.NA, 5])
def test_symmetric_difference_keeping_ea_dtype(any_numeric_ea_dtype: Any, val: Union[int, Any]) -> None:
    # GH#48607
    midx: MultiIndex = MultiIndex.from_arrays(
        [Series([1, 2], dtype=any_numeric_ea_dtype), [2, 1]], names=["a", None]
    )
    midx2: MultiIndex = MultiIndex.from_arrays(
        [Series([1, 2, val], dtype=any_numeric_ea_dtype), [1, 1, 3]]
    )
    result: MultiIndex = midx.symmetric_difference(midx2)
    expected: MultiIndex = MultiIndex.from_arrays(
        [Series([1, 1, val], dtype=any_numeric_ea_dtype), [1, 2, 3]]
    )
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize(
    "tuples, exp_tuples",
    [
        ([("val1", "test1")], [("val1", "test1")]),
        ([("val1", "test1"), ("val1", "test1")], [("val1", "test1")]),
        (
            [("val2", "test2"), ("val1", "test1")],
            [("val2", "test2"), ("val1", "test1")],
        ),
    ],
)
def test_intersect_with_duplicates(tuples: List[tuple], exp_tuples: List[tuple]) -> None:
    # GH#36915
    left: MultiIndex = MultiIndex.from_tuples(tuples, names=["first", "second"])
    right: MultiIndex = MultiIndex.from_tuples(
        [("val1", "test1"), ("val1", "test1"), ("val2", "test2")],
        names=["first", "second"],
    )
    result: MultiIndex = left.intersection(right)
    expected: MultiIndex = MultiIndex.from_tuples(exp_tuples, names=["first", "second"])
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize(
    "data, names, expected",
    [
        ((1,), None, [None, None]),
        ((1,), ["a"], [None, None]),
        ((1,), ["b"], [None, None]),
        ((1, 2), ["c", "d"], [None, None]),
        ((1, 2), ["b", "a"], [None, None]),
        ((1, 2, 3), ["a", "b", "c"], [None, None]),
        ((1, 2), ["a", "c"], ["a", None]),
        ((1, 2), ["c", "b"], [None, "b"]),
        ((1, 2), ["a", "b"], ["a", "b"]),
        ((1, 2), [None, "b"], [None, "b"]),
    ],
)
def test_maybe_match_names(data: tuple, names: Optional[List[Any]], expected: List[Any]) -> None:
    # GH#38323
    mi: MultiIndex = MultiIndex.from_tuples([], names=["a", "b"])
    mi2: MultiIndex = MultiIndex.from_tuples([data], names=names)
    result = mi._maybe_match_names(mi2)
    assert result == expected


def test_intersection_equal_different_names() -> None:
    # GH#30302
    mi1: MultiIndex = MultiIndex.from_arrays([[1, 2], [3, 4]], names=["c", "b"])
    mi2: MultiIndex = MultiIndex.from_arrays([[1, 2], [3, 4]], names=["a", "b"])

    result: MultiIndex = mi1.intersection(mi2)
    expected: MultiIndex = MultiIndex.from_arrays([[1, 2], [3, 4]], names=[None, "b"])
    tm.assert_index_equal(result, expected)


def test_intersection_different_names() -> None:
    # GH#38323
    mi: MultiIndex = MultiIndex.from_arrays([[1], [3]], names=["c", "b"])
    mi2: MultiIndex = MultiIndex.from_arrays([[1], [3]])
    result: MultiIndex = mi.intersection(mi2)
    tm.assert_index_equal(result, mi2)


def test_intersection_with_missing_values_on_both_sides(nulls_fixture: Any) -> None:
    # GH#38623
    mi1: MultiIndex = MultiIndex.from_arrays([[3, nulls_fixture, 4, nulls_fixture], [1, 2, 4, 2]])
    mi2: MultiIndex = MultiIndex.from_arrays([[3, nulls_fixture, 3], [1, 2, 4]])
    result: MultiIndex = mi1.intersection(mi2)
    expected: MultiIndex = MultiIndex.from_arrays([[3, nulls_fixture], [1, 2]])
    tm.assert_index_equal(result, expected)


def test_union_with_missing_values_on_both_sides(nulls_fixture: Any) -> None:
    # GH#38623
    mi1: MultiIndex = MultiIndex.from_arrays([[1, nulls_fixture]])
    mi2: MultiIndex = MultiIndex.from_arrays([[1, nulls_fixture, 3]])
    result: MultiIndex = mi1.union(mi2)
    expected: MultiIndex = MultiIndex.from_arrays([[1, 3, nulls_fixture]])
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("dtype", ["float64", "Float64"])
@pytest.mark.parametrize("sort", [None, False])
def test_union_nan_got_duplicated(dtype: str, sort: Optional[bool]) -> None:
    # GH#38977, GH#49010
    mi1: MultiIndex = MultiIndex.from_arrays([pd.array([1.0, np.nan], dtype=dtype), [2, 3]])
    mi2: MultiIndex = MultiIndex.from_arrays([pd.array([1.0, np.nan, 3.0], dtype=dtype), [2, 3, 4]])
    result: MultiIndex = mi1.union(mi2, sort=sort)
    if sort is None:
        expected: MultiIndex = MultiIndex.from_arrays(
            [pd.array([1.0, 3.0, np.nan], dtype=dtype), [2, 4, 3]]
        )
    else:
        expected = mi2
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("val", [4, 1])
def test_union_keep_ea_dtype(any_numeric_ea_dtype: Any, val: int) -> None:
    # GH#48505
    arr1: Series = Series([val, 2], dtype=any_numeric_ea_dtype)
    arr2: Series = Series([2, 1], dtype=any_numeric_ea_dtype)
    midx: MultiIndex = MultiIndex.from_arrays([arr1, [1, 2]], names=["a", None])
    midx2: MultiIndex = MultiIndex.from_arrays([arr2, [2, 1]])
    result: MultiIndex = midx.union(midx2)
    if val == 4:
        expected: MultiIndex = MultiIndex.from_arrays(
            [Series([1, 2, 4], dtype=any_numeric_ea_dtype), [1, 2, 1]]
        )
    else:
        expected = MultiIndex.from_arrays(
            [Series([1, 2], dtype=any_numeric_ea_dtype), [1, 2]]
        )
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("dupe_val", [3, pd.NA])
def test_union_with_duplicates_keep_ea_dtype(dupe_val: Union[int, Any], any_numeric_ea_dtype: Any) -> None:
    # GH48900
    mi1: MultiIndex = MultiIndex.from_arrays(
        [
            Series([1, dupe_val, 2], dtype=any_numeric_ea_dtype),
            Series([1, dupe_val, 2], dtype=any_numeric_ea_dtype),
        ]
    )
    mi2: MultiIndex = MultiIndex.from_arrays(
        [
            Series([2, dupe_val, dupe_val], dtype=any_numeric_ea_dtype),
            Series([2, dupe_val, dupe_val], dtype=any_numeric_ea_dtype),
        ]
    )
    result: MultiIndex = mi1.union(mi2)
    expected: MultiIndex = MultiIndex.from_arrays(
        [
            Series([1, 2, dupe_val, dupe_val], dtype=any_numeric_ea_dtype),
            Series([1, 2, dupe_val, dupe_val], dtype=any_numeric_ea_dtype),
        ]
    )
    tm.assert_index_equal(result, expected)


@pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
def test_union_duplicates(index: Any, request: Any) -> None:
    # GH#38977
    if index.empty or isinstance(index, (IntervalIndex, CategoricalIndex)):
        pytest.skip(f"No duplicates in an empty {type(index).__name__}")

    values: List[Any] = index.unique().values.tolist()
    mi1: MultiIndex = MultiIndex.from_arrays([values, [1] * len(values)])
    mi2: MultiIndex = MultiIndex.from_arrays([[values[0]] + values, [1] * (len(values) + 1)])
    result: MultiIndex = mi2.union(mi1)
    expected: MultiIndex = mi2.sort_values()
    tm.assert_index_equal(result, expected)

    if (
        is_unsigned_integer_dtype(mi2.levels[0])
        and (mi2.get_level_values(0) < 2**63).all()
    ):
        expected = expected.set_levels(
            [expected.levels[0].astype(np.int64), expected.levels[1]]
        )
    elif is_float_dtype(mi2.levels[0]):
        expected = expected.set_levels(
            [expected.levels[0].astype(float), expected.levels[1]]
        )

    result = mi1.union(mi2)
    tm.assert_index_equal(result, expected)


def test_union_keep_dtype_precision(any_real_numeric_dtype: Any) -> None:
    # GH#48498
    arr1: Series = Series([4, 1, 1], dtype=any_real_numeric_dtype)
    arr2: Series = Series([1, 4], dtype=any_real_numeric_dtype)
    midx: MultiIndex = MultiIndex.from_arrays([arr1, [2, 1, 1]], names=["a", None])
    midx2: MultiIndex = MultiIndex.from_arrays([arr2, [1, 2]], names=["a", None])

    result: MultiIndex = midx.union(midx2)
    expected: MultiIndex = MultiIndex.from_arrays(
        ([Series([1, 1, 4], dtype=any_real_numeric_dtype), [1, 1, 2]]),
        names=["a", None],
    )
    tm.assert_index_equal(result, expected)


def test_union_keep_ea_dtype_with_na(any_numeric_ea_dtype: Any) -> None:
    # GH#48498
    arr1: Series = Series([4, pd.NA], dtype=any_numeric_ea_dtype)
    arr2: Series = Series([1, pd.NA], dtype=any_numeric_ea_dtype)
    midx: MultiIndex = MultiIndex.from_arrays([arr1, [2, 1]], names=["a", None])
    midx2: MultiIndex = MultiIndex.from_arrays([arr2, [1, 2]])
    result: MultiIndex = midx.union(midx2)
    expected: MultiIndex = MultiIndex.from_arrays(
        [Series([1, 4, pd.NA, pd.NA], dtype=any_numeric_ea_dtype), [1, 2, 1, 2]]
    )
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize(
    "levels1, levels2, codes1, codes2, names",
    [
        (
            [["a", "b", "c"], [0, ""]],
            [["c", "d", "b"], [""]],
            [[0, 1, 2], [1, 1, 1]],
            [[0, 1, 2], [0, 0, 0]],
            ["name1", "name2"],
        ),
    ],
)
def test_intersection_lexsort_depth(
    levels1: List[List[Any]],
    levels2: List[List[Any]],
    codes1: List[List[int]],
    codes2: List[List[int]],
    names: List[Optional[str]],
) -> None:
    # GH#25169
    mi1: MultiIndex = MultiIndex(levels=levels1, codes=codes1, names=names)
    mi2: MultiIndex = MultiIndex(levels=levels2, codes=codes2, names=names)
    mi_int: MultiIndex = mi1.intersection(mi2)
    assert mi_int._lexsort_depth == 2


@pytest.mark.parametrize(
    "a",
    [pd.Categorical(["a", "b"], categories=["a", "b"]), ["a", "b"]],
)
@pytest.mark.parametrize("b_ordered", [True, False])
def test_intersection_with_non_lex_sorted_categories(a: Union[pd.Categorical, List[str]], b_ordered: bool) -> None:
    # GH#49974
    other: List[str] = ["1", "2"]
    b = pd.Categorical(["a", "b"], categories=["b", "a"], ordered=b_ordered)
    df1: DataFrame = DataFrame({"x": a, "y": other})
    df2: DataFrame = DataFrame({"x": b, "y": other})

    expected: MultiIndex = MultiIndex.from_arrays([a, other], names=["x", "y"])

    res1: MultiIndex = MultiIndex.from_frame(df1).intersection(
        MultiIndex.from_frame(df2.sort_values(["x", "y"]))
    )
    res2: MultiIndex = MultiIndex.from_frame(df1).intersection(MultiIndex.from_frame(df2))
    res3: MultiIndex = MultiIndex.from_frame(df1.sort_values(["x", "y"])).intersection(
        MultiIndex.from_frame(df2)
    )
    res4: MultiIndex = MultiIndex.from_frame(df1.sort_values(["x", "y"])).intersection(
        MultiIndex.from_frame(df2.sort_values(["x", "y"]))
    )

    tm.assert_index_equal(res1, expected)
    tm.assert_index_equal(res2, expected)
    tm.assert_index_equal(res3, expected)
    tm.assert_index_equal(res4, expected)


@pytest.mark.parametrize("val", [pd.NA, 100])
def test_intersection_keep_ea_dtypes(val: Union[int, Any], any_numeric_ea_dtype: Any) -> None:
    # GH#48604
    midx: MultiIndex = MultiIndex.from_arrays(
        [Series([1, 2], dtype=any_numeric_ea_dtype), [2, 1]], names=["a", None]
    )
    midx2: MultiIndex = MultiIndex.from_arrays(
        [Series([1, 2, val], dtype=any_numeric_ea_dtype), [1, 1, 3]]
    )
    result: MultiIndex = midx.intersection(midx2)
    expected: MultiIndex = MultiIndex.from_arrays([Series([2], dtype=any_numeric_ea_dtype), [1]])
    tm.assert_index_equal(result, expected)


def test_union_with_na_when_constructing_dataframe() -> None:
    # GH43222
    series1: Series = Series(
        (1,),
        index=MultiIndex.from_arrays(
            [Series([None], dtype="str"), Series([None], dtype="str")]
        ),
    )
    series2: Series = Series((10, 20), index=MultiIndex.from_tuples(((None, None), ("a", "b"))))
    result: DataFrame = DataFrame([series1, series2])
    expected: DataFrame = DataFrame({(np.nan, np.nan): [1.0, 10.0], ("a", "b"): [np.nan, 20.0]})
    tm.assert_frame_equal(result, expected)