import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas import (
    Categorical,
    CategoricalIndex,
    Index,
    IntervalIndex,
    Series,
    Timestamp,
)
import pandas._testing as tm
from typing import Any, List, Optional, Tuple


class TestCategoricalDtypes:

    def test_categories_match_up_to_permutation(self) -> None:
        c1: Categorical = Categorical(list("aabca"), categories=list("abc"), ordered=False)
        c2: Categorical = Categorical(list("aabca"), categories=list("cab"), ordered=False)
        c3: Categorical = Categorical(list("aabca"), categories=list("cab"), ordered=True)
        assert c1._categories_match_up_to_permutation(c1)
        assert c2._categories_match_up_to_permutation(c2)
        assert c3._categories_match_up_to_permutation(c3)
        assert c1._categories_match_up_to_permutation(c2)
        assert not c1._categories_match_up_to_permutation(c3)
        assert not c1._categories_match_up_to_permutation(Index(list("aabca")))
        assert not c1._categories_match_up_to_permutation(c1.astype(object))
        categorical_index1: CategoricalIndex = CategoricalIndex(c1)
        categorical_index2: CategoricalIndex = CategoricalIndex(c1, categories=list("cab"))
        assert c1._categories_match_up_to_permutation(categorical_index1)
        assert c1._categories_match_up_to_permutation(categorical_index2)
        categorical_index_ordered: CategoricalIndex = CategoricalIndex(c1, ordered=True)
        assert not c1._categories_match_up_to_permutation(categorical_index_ordered)
        s1: Series = Series(c1)
        s2: Series = Series(c2)
        s3: Series = Series(c3)
        assert c1._categories_match_up_to_permutation(s1)
        assert c2._categories_match_up_to_permutation(s2)
        assert c3._categories_match_up_to_permutation(s3)
        assert c1._categories_match_up_to_permutation(s2)
        assert not c1._categories_match_up_to_permutation(s3)
        assert not c1._categories_match_up_to_permutation(s1.astype(object))

    def test_set_dtype_same(self) -> None:
        c: Categorical = Categorical(["a", "b", "c"])
        dtype: CategoricalDtype = CategoricalDtype(["a", "b", "c"])
        result: Categorical = c._set_dtype(dtype)
        tm.assert_categorical_equal(result, c)

    def test_set_dtype_new_categories(self) -> None:
        c: Categorical = Categorical(["a", "b", "c"])
        new_categories: List[str] = list("abcd")
        dtype: CategoricalDtype = CategoricalDtype(new_categories)
        result: Categorical = c._set_dtype(dtype)
        tm.assert_numpy_array_equal(result.codes, c.codes)
        tm.assert_index_equal(result.dtype.categories, Index(new_categories))

    @pytest.mark.parametrize(
        "values, categories, new_categories",
        [
            (["a", "b", "a"], ["a", "b"], ["a", "b"]),
            (["a", "b", "a"], ["a", "b"], ["b", "a"]),
            (["b", "a", "a"], ["a", "b"], ["a", "b"]),
            (["b", "a", "a"], ["a", "b"], ["b", "a"]),
            (["a", "b", "c"], ["a", "b"], ["a", "b"]),
            (["a", "b", "c"], ["a", "b"], ["b", "a"]),
            (["b", "a", "c"], ["a", "b"], ["a", "b"]),
            (["b", "a", "c"], ["a", "b"], ["b", "a"]),
            (["a", "b", "c"], ["a", "b"], ["a"]),
            (["a", "b", "c"], ["a", "b"], ["b"]),
            (["b", "a", "c"], ["a", "b"], ["a"]),
            (["b", "a", "c"], ["a", "b"], ["b"]),
            (["a", "b", "c"], ["a", "b"], ["d", "e"]),
        ],
    )
    def test_set_dtype_many(
        self,
        values: List[str],
        categories: List[str],
        new_categories: List[str],
    ) -> None:
        ordered: Optional[bool] = None  # Assuming ordered is handled elsewhere
        c: Categorical = Categorical(values, categories=categories)
        expected_dtype: CategoricalDtype = CategoricalDtype(new_categories, ordered=ordered)
        expected: Categorical = Categorical(values, categories=new_categories, ordered=ordered)
        result: Categorical = c._set_dtype(expected_dtype)
        tm.assert_categorical_equal(result, expected)

    def test_set_dtype_no_overlap(self) -> None:
        c: Categorical = Categorical(["a", "b", "c"], ["d", "e"])
        new_dtype: CategoricalDtype = CategoricalDtype(["a", "b"])
        result: Categorical = c._set_dtype(new_dtype)
        expected: Categorical = Categorical([None, None, None], categories=["a", "b"])
        tm.assert_categorical_equal(result, expected)

    def test_codes_dtypes(self) -> None:
        result: Categorical = Categorical(["foo", "bar", "baz"])
        assert result.codes.dtype == np.int8
        result = Categorical([f"foo{i:05d}" for i in range(400)])
        assert result.codes.dtype == np.int16
        result = Categorical([f"foo{i:05d}" for i in range(40000)])
        assert result.codes.dtype == np.int32
        result = Categorical(["foo", "bar", "baz"])
        assert result.codes.dtype == np.int8
        result = result.add_categories([f"foo{i:05d}" for i in range(400)])
        assert result.codes.dtype == np.int16
        result = result.remove_categories([f"foo{i:05d}" for i in range(300)])
        assert result.codes.dtype == np.int8

    def test_iter_python_types(self) -> None:
        cat: Categorical = Categorical([1, 2])
        item: Any = next(iter(cat))
        assert isinstance(item, int)
        first_item: Any = cat.tolist()[0]
        assert isinstance(first_item, int)

    def test_iter_python_types_datetime(self) -> None:
        cat: Categorical = Categorical([Timestamp("2017-01-01"), Timestamp("2017-01-02")])
        item: Any = next(iter(cat))
        assert isinstance(item, Timestamp)
        first_item: Any = cat.tolist()[0]
        assert isinstance(first_item, Timestamp)

    def test_interval_index_category(self) -> None:
        index: IntervalIndex = IntervalIndex.from_breaks(np.arange(3, dtype="uint64"))
        result: Index = CategoricalIndex(index).dtype.categories
        expected: IntervalIndex = IntervalIndex.from_arrays(
            [0, 1], [1, 2], dtype="interval[uint64, right]"
        )
        tm.assert_index_equal(result, expected)
