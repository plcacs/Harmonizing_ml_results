import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas import Categorical, CategoricalIndex, Index, IntervalIndex, Series, Timestamp
import pandas._testing as tm

class TestCategoricalDtypes:

    def test_categories_match_up_to_permutation(self) -> None:
        c1: Categorical = Categorical(list('aabca'), categories=list('abc'), ordered=False)
        c2: Categorical = Categorical(list('aabca'), categories=list('cab'), ordered=False)
        c3: Categorical = Categorical(list('aabca'), categories=list('cab'), ordered=True)
        assert c1._categories_match_up_to_permutation(c1)
        assert c2._categories_match_up_to_permutation(c2)
        assert c3._categories_match_up_to_permutation(c3)
        assert c1._categories_match_up_to_permutation(c2)
        assert not c1._categories_match_up_to_permutation(c3)
        assert not c1._categories_match_up_to_permutation(Index(list('aabca')))
        assert not c1._categories_match_up_to_permutation(c1.astype(object))
        assert c1._categories_match_up_to_permutation(CategoricalIndex(c1))
        assert c1._categories_match_up_to_permutation(CategoricalIndex(c1, categories=list('cab')))
        assert not c1._categories_match_up_to_permutation(CategoricalIndex(c1, ordered=True))
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
        c: Categorical = Categorical(['a', 'b', 'c'])
        result: Categorical = c._set_dtype(CategoricalDtype(['a', 'b', 'c']))
        tm.assert_categorical_equal(result, c)

    def test_set_dtype_new_categories(self) -> None:
        c: Categorical = Categorical(['a', 'b', 'c'])
        result: Categorical = c._set_dtype(CategoricalDtype(list('abcd'))
        tm.assert_numpy_array_equal(result.codes, c.codes)
        tm.assert_index_equal(result.dtype.categories, Index(list('abcd')))

    @pytest.mark.parametrize('values, categories, new_categories', [(['a', 'b', 'a'], ['a', 'b'], ['a', 'b']), (['a', 'b', 'a'], ['a', 'b'], ['b', 'a']), (['b', 'a', 'a'], ['a', 'b'], ['a', 'b']), (['b', 'a', 'a'], ['a', 'b'], ['b', 'a']), (['a', 'b', 'c'], ['a', 'b'], ['a', 'b']), (['a', 'b', 'c'], ['a', 'b'], ['b', 'a']), (['b', 'a', 'c'], ['a', 'b'], ['a', 'b']), (['b', 'a', 'c'], ['a', 'b'], ['b', 'a']), (['a', 'b', 'c'], ['a', 'b'], ['a']), (['a', 'b', 'c'], ['a', 'b'], ['b']), (['b', 'a', 'c'], ['a', 'b'], ['a']), (['b', 'a', 'c'], ['a', 'b'], ['b']), (['a', 'b', 'c'], ['a', 'b'], ['d', 'e'])])
    def test_set_dtype_many(self, values: list, categories: list, new_categories: list, ordered: bool) -> None:
        c: Categorical = Categorical(values, categories)
        expected: Categorical = Categorical(values, new_categories, ordered)
        result: Categorical = c._set_dtype(expected.dtype)
        tm.assert_categorical_equal(result, expected)

    def test_set_dtype_no_overlap(self) -> None:
        c: Categorical = Categorical(['a', 'b', 'c'], ['d', 'e'])
        result: Categorical = c._set_dtype(CategoricalDtype(['a', 'b']))
        expected: Categorical = Categorical([None, None, None], categories=['a', 'b'])
        tm.assert_categorical_equal(result, expected)

    def test_codes_dtypes(self) -> None:
        result: Categorical = Categorical(['foo', 'bar', 'baz'])
        assert result.codes.dtype == 'int8'
        result: Categorical = Categorical([f'foo{i:05d}' for i in range(400)])
        assert result.codes.dtype == 'int16'
        result: Categorical = Categorical([f'foo{i:05d}' for i in range(40000)])
        assert result.codes.dtype == 'int32'
        result: Categorical = Categorical(['foo', 'bar', 'baz'])
        assert result.codes.dtype == 'int8'
        result: Categorical = result.add_categories([f'foo{i:05d}' for i in range(400)])
        assert result.codes.dtype == 'int16'
        result: Categorical = result.remove_categories([f'foo{i:05d}' for i in range(300)])
        assert result.codes.dtype == 'int8'

    def test_iter_python_types(self) -> None:
        cat: Categorical = Categorical([1, 2])
        assert isinstance(next(iter(cat)), int)
        assert isinstance(cat.tolist()[0], int)

    def test_iter_python_types_datetime(self) -> None:
        cat: Categorical = Categorical([Timestamp('2017-01-01'), Timestamp('2017-01-02')])
        assert isinstance(next(iter(cat)), Timestamp)
        assert isinstance(cat.tolist()[0], Timestamp)

    def test_interval_index_category(self) -> None:
        index: IntervalIndex = IntervalIndex.from_breaks(np.arange(3, dtype='uint64'))
        result: Index = CategoricalIndex(index).dtype.categories
        expected: IntervalIndex = IntervalIndex.from_arrays([0, 1], [1, 2], dtype='interval[uint64, right]')
        tm.assert_index_equal(result, expected)
