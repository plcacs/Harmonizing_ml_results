import re
import numpy as np
import pytest
from typing import List, Union, Callable, Any, Optional
from pandas.compat import PY311
from pandas import Categorical, CategoricalIndex, DataFrame, Index, Series, StringDtype
import pandas._testing as tm
from pandas.core.arrays.categorical import recode_for_categories


class TestCategoricalAPI:

    def test_ordered_api(self) -> None:
        cat1: Categorical = Categorical(list('acb'), ordered=False)
        tm.assert_index_equal(cat1.categories, Index(['a', 'b', 'c']))
        assert not cat1.ordered
        cat2: Categorical = Categorical(list('acb'), categories=list('bca'), ordered=False)
        tm.assert_index_equal(cat2.categories, Index(['b', 'c', 'a']))
        assert not cat2.ordered
        cat3: Categorical = Categorical(list('acb'), ordered=True)
        tm.assert_index_equal(cat3.categories, Index(['a', 'b', 'c']))
        assert cat3.ordered
        cat4: Categorical = Categorical(list('acb'), categories=list('bca'), ordered=True)
        tm.assert_index_equal(cat4.categories, Index(['b', 'c', 'a']))
        assert cat4.ordered

    def test_set_ordered(self) -> None:
        cat: Categorical = Categorical(['a', 'b', 'c', 'a'], ordered=True)
        cat2: Categorical = cat.as_unordered()
        assert not cat2.ordered
        cat2 = cat.as_ordered()
        assert cat2.ordered
        assert cat2.set_ordered(True).ordered
        assert not cat2.set_ordered(False).ordered
        msg: str = "property 'ordered' of 'Categorical' object has no setter" if PY311 else "can't set attribute"
        with pytest.raises(AttributeError, match=msg):
            cat.ordered = True
        with pytest.raises(AttributeError, match=msg):
            cat.ordered = False

    def test_rename_categories(self) -> None:
        cat: Categorical = Categorical(['a', 'b', 'c', 'a'])
        res: Categorical = cat.rename_categories([1, 2, 3])
        tm.assert_numpy_array_equal(res.__array__(), np.array([1, 2, 3, 1], dtype=np.int64))
        tm.assert_index_equal(res.categories, Index([1, 2, 3]))
        exp_cat: np.ndarray = np.array(['a', 'b', 'c', 'a'], dtype=np.object_)
        tm.assert_numpy_array_equal(cat.__array__(), exp_cat)
        exp_cat = Index(['a', 'b', 'c'])
        tm.assert_index_equal(cat.categories, exp_cat)
        result: Categorical = cat.rename_categories(lambda x: x.upper())
        expected: Categorical = Categorical(['A', 'B', 'C', 'A'])
        tm.assert_categorical_equal(result, expected)

    @pytest.mark.parametrize('new_categories', [[1, 2, 3, 4], [1, 2]])
    def test_rename_categories_wrong_length_raises(self, new_categories: List[int]) -> None:
        cat: Categorical = Categorical(['a', 'b', 'c', 'a'])
        msg: str = 'new categories need to have the same number of items as the old categories!'
        with pytest.raises(ValueError, match=msg):
            cat.rename_categories(new_categories)

    def test_rename_categories_series(self) -> None:
        c: Categorical = Categorical(['a', 'b'])
        result: Categorical = c.rename_categories(Series([0, 1], index=['a', 'b']))
        expected: Categorical = Categorical([0, 1])
        tm.assert_categorical_equal(result, expected)

    def test_rename_categories_dict(self) -> None:
        cat: Categorical = Categorical(['a', 'b', 'c', 'd'])
        res: Categorical = cat.rename_categories({'a': 4, 'b': 3, 'c': 2, 'd': 1})
        expected: Index = Index([4, 3, 2, 1])
        tm.assert_index_equal(res.categories, expected)
        cat = Categorical(['a', 'b', 'c', 'd'])
        res = cat.rename_categories({'a': 1, 'c': 3})
        expected = Index([1, 'b', 3, 'd'])
        tm.assert_index_equal(res.categories, expected)
        cat = Categorical(['a', 'b', 'c', 'd'])
        res = cat.rename_categories({'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6})
        expected = Index([1, 2, 3, 4])
        tm.assert_index_equal(res.categories, expected)
        cat = Categorical(['a', 'b', 'c', 'd'])
        res = cat.rename_categories({'f': 1, 'g': 3})
        expected = Index(['a', 'b', 'c', 'd'])
        tm.assert_index_equal(res.categories, expected)

    def test_reorder_categories(self) -> None:
        cat: Categorical = Categorical(['a', 'b', 'c', 'a'], ordered=True)
        old: Categorical = cat.copy()
        new: Categorical = Categorical(['a', 'b', 'c', 'a'], categories=['c', 'b', 'a'], ordered=True)
        res: Categorical = cat.reorder_categories(['c', 'b', 'a'])
        tm.assert_categorical_equal(cat, old)
        tm.assert_categorical_equal(res, new)

    @pytest.mark.parametrize('new_categories', [['a'], ['a', 'b', 'd'], ['a', 'b', 'c', 'd']])
    def test_reorder_categories_raises(self, new_categories: Union[List[str], str]) -> None:
        cat: Categorical = Categorical(['a', 'b', 'c', 'a'], ordered=True)
        msg: str = 'items in new_categories are not the same as in old categories'
        with pytest.raises(ValueError, match=msg):
            cat.reorder_categories(new_categories)

    def test_add_categories(self) -> None:
        cat: Categorical = Categorical(['a', 'b', 'c', 'a'], ordered=True)
        old: Categorical = cat.copy()
        new: Categorical = Categorical(['a', 'b', 'c', 'a'], categories=['a', 'b', 'c', 'd'], ordered=True)
        res: Categorical = cat.add_categories('d')
        tm.assert_categorical_equal(cat, old)
        tm.assert_categorical_equal(res, new)
        res = cat.add_categories(['d'])
        tm.assert_categorical_equal(cat, old)
        tm.assert_categorical_equal(res, new)
        cat = Categorical(list('abc'), ordered=True)
        expected: Categorical = Categorical(list('abc'), categories=list('abcde'), ordered=True)
        res = cat.add_categories(Series(['d', 'e']))
        tm.assert_categorical_equal(res, expected)
        res = cat.add_categories(np.array(['d', 'e']))
        tm.assert_categorical_equal(res, expected)
        res = cat.add_categories(Index(['d', 'e']))
        tm.assert_categorical_equal(res, expected)
        res = cat.add_categories(['d', 'e'])
        tm.assert_categorical_equal(res, expected)

    def test_add_categories_existing_raises(self) -> None:
        cat: Categorical = Categorical(['a', 'b', 'c', 'd'], ordered=True)
        msg: str = re.escape("new categories must not include old categories: {'d'}")
        with pytest.raises(ValueError, match=msg):
            cat.add_categories(['d'])

    def test_add_categories_losing_dtype_information(self) -> None:
        cat: Categorical = Categorical(Series([1, 2], dtype='Int64'))
        ser: Series = Series([4], dtype='Int64')
        result: Categorical = cat.add_categories(ser)
        expected: Categorical = Categorical(Series([1, 2], dtype='Int64'), categories=Series([1, 2, 4], dtype='Int64'))
        tm.assert_categorical_equal(result, expected)
        cat = Categorical(Series(['a', 'b', 'a'], dtype=StringDtype()))
        ser = Series(['d'], dtype=StringDtype())
        result = cat.add_categories(ser)
        expected = Categorical(Series(['a', 'b', 'a'], dtype=StringDtype()), categories=Series(['a', 'b', 'd'], dtype=StringDtype()))
        tm.assert_categorical_equal(result, expected)

    def test_set_categories(self) -> None:
        cat: Categorical = Categorical(['a', 'b', 'c', 'a'], ordered=True)
        exp_categories: Index = Index(['c', 'b', 'a'])
        exp_values: np.ndarray = np.array(['a', 'b', 'c', 'a'], dtype=np.object_)
        cat = cat.set_categories(['c', 'b', 'a'])
        res: Categorical = cat.set_categories(['a', 'b', 'c'])
        tm.assert_index_equal(cat.categories, exp_categories)
        tm.assert_numpy_array_equal(cat.__array__(), exp_values)
        exp_categories_back: Index = Index(['a', 'b', 'c'])
        tm.assert_index_equal(res.categories, exp_categories_back)
        tm.assert_numpy_array_equal(res.__array__(), exp_values)
        cat = Categorical(['a', 'b', 'c', 'a'], ordered=True)
        res = cat.set_categories(['a'])
        tm.assert_numpy_array_equal(res.codes, np.array([0, -1, -1, 0], dtype=np.int8))
        res = cat.set_categories(['a', 'b', 'd'])
        tm.assert_numpy_array_equal(res.codes, np.array([0, 1, -1, 0], dtype=np.int8))
        tm.assert_index_equal(res.categories, Index(['a', 'b', 'd']))
        cat = cat.set_categories(['a', 'b', 'c', 'd'])
        exp_categories = Index(['a', 'b', 'c', 'd'])
        tm.assert_index_equal(cat.categories, exp_categories)
        c: Categorical = Categorical([1, 2, 3, 4, 1], categories=[1, 2, 3, 4], ordered=True)
        tm.assert_numpy_array_equal(c._codes, np.array([0, 1, 2, 3, 0], dtype=np.int8))
        tm.assert_index_equal(c.categories, Index([1, 2, 3, 4]))
        exp: np.ndarray = np.array([1, 2, 3, 4, 1], dtype=np.int64)
        tm.assert_numpy_array_equal(np.asarray(c), exp)
        c = c.set_categories([4, 3, 2, 1])
        tm.assert_numpy_array_equal(c._codes, np.array([3, 2, 1, 0, 3], dtype=np.int8))
        tm.assert_index_equal(c.categories, Index([4, 3, 2, 1]))
        exp = np.array([1, 2, 3, 4, 1], dtype=np.int64)
        tm.assert_numpy_array_equal(np.asarray(c), exp)
        assert c.min() == 4
        assert c.max() == 1
        c2: Categorical = c.set_categories([4, 3, 2, 1], ordered=False)
        assert not c2.ordered
        tm.assert_numpy_array_equal(np.asarray(c), np.asarray(c2))
        c2 = c.set_ordered(False).set_categories([4, 3, 2, 1])
        assert not c2.ordered
        tm.assert_numpy_array_equal(np.asarray(c), np.asarray(c2))

    @pytest.mark.parametrize(
        'values, categories, new_categories', [
            (['a', 'b', 'a'], ['a', 'b'], ['a', 'b']),
            (['a', 'b', 'a'], ['a', 'b'], ['b', 'a']),
            (['b', 'a', 'a'], ['a', 'b'], ['a', 'b']),
            (['b', 'a', 'a'], ['a', 'b'], ['b', 'a']),
            (['a', 'b', 'c'], ['a', 'b'], ['a', 'b']),
            (['a', 'b', 'c'], ['a', 'b'], ['b', 'a']),
            (['b', 'a', 'c'], ['a', 'b'], ['a', 'b']),
            (['b', 'a', 'c'], ['a', 'b'], ['b', 'a']),
            (['a', 'b', 'c'], ['a', 'b'], ['a']),
            (['a', 'b', 'c'], ['a', 'b'], ['b']),
            (['b', 'a', 'c'], ['a', 'b'], ['a']),
            (['b', 'a', 'c'], ['a', 'b'], ['b']),
            (['a', 'b', 'c'], ['a', 'b'], ['d', 'e'])
        ]
    )
    def test_set_categories_many(
        self,
        values: List[str],
        categories: List[str],
        new_categories: List[str],
        ordered: Optional[bool] = None
    ) -> None:
        c: Categorical = Categorical(values, categories)
        expected: Categorical = Categorical(values, new_categories, ordered=ordered)
        result: Categorical = c.set_categories(new_categories, ordered=ordered)
        tm.assert_categorical_equal(result, expected)

    def test_set_categories_rename_less(self) -> None:
        cat: Categorical = Categorical(['A', 'B'])
        result: Categorical = cat.set_categories(['A'], rename=True)
        expected: Categorical = Categorical(['A', np.nan])
        tm.assert_categorical_equal(result, expected)

    def test_set_categories_private(self) -> None:
        cat: Categorical = Categorical(['a', 'b', 'c'], categories=['a', 'b', 'c', 'd'])
        cat._set_categories(['a', 'c', 'd', 'e'])
        expected: Categorical = Categorical(['a', 'c', 'd'], categories=['a', 'c', 'd', 'e'])
        tm.assert_categorical_equal(cat, expected)
        cat = Categorical(['a', 'b', 'c'], categories=['a', 'b', 'c', 'd'])
        cat._set_categories(['a', 'c', 'd', 'e'], fastpath=True)
        expected = Categorical(['a', 'c', 'd'], categories=['a', 'c', 'd', 'e'])
        tm.assert_categorical_equal(cat, expected)

    def test_remove_categories(self) -> None:
        cat: Categorical = Categorical(['a', 'b', 'c', 'a'], ordered=True)
        old: Categorical = cat.copy()
        new: Categorical = Categorical(['a', 'b', np.nan, 'a'], categories=['a', 'b'], ordered=True)
        res: Categorical = cat.remove_categories('c')
        tm.assert_categorical_equal(cat, old)
        tm.assert_categorical_equal(res, new)
        res = cat.remove_categories(['c'])
        tm.assert_categorical_equal(cat, old)
        tm.assert_categorical_equal(res, new)

    @pytest.mark.parametrize('removals', [['c'], ['c', np.nan], 'c', ['c', 'c']])
    def test_remove_categories_raises(self, removals: Union[List[str], str]) -> None:
        cat: Categorical = Categorical(['a', 'b', 'a'])
        message: str = re.escape("removals must all be in old categories: {'c'}")
        with pytest.raises(ValueError, match=message):
            cat.remove_categories(removals)

    def test_remove_unused_categories(self) -> None:
        c: Categorical = Categorical(['a', 'b', 'c', 'd', 'a'], categories=['a', 'b', 'c', 'd', 'e'])
        exp_categories_all: Index = Index(['a', 'b', 'c', 'd', 'e'])
        exp_categories_dropped: Index = Index(['a', 'b', 'c', 'd'])
        tm.assert_index_equal(c.categories, exp_categories_all)
        res: Categorical = c.remove_unused_categories()
        tm.assert_index_equal(res.categories, exp_categories_dropped)
        tm.assert_index_equal(c.categories, exp_categories_all)
        c = Categorical(['a', 'b', 'c', np.nan], categories=['a', 'b', 'c', 'd', 'e'])
        res = c.remove_unused_categories()
        tm.assert_index_equal(res.categories, Index(np.array(['a', 'b', 'c'])))
        exp_codes: np.ndarray = np.array([0, 1, 2, -1], dtype=np.int8)
        tm.assert_numpy_array_equal(res.codes, exp_codes)
        tm.assert_index_equal(c.categories, exp_categories_all)
        val: List[Optional[str]] = ['F', np.nan, 'D', 'B', 'D', 'F', np.nan]
        cat: Categorical = Categorical(values=val, categories=list('ABCDEFG'))
        out: Categorical = cat.remove_unused_categories()
        tm.assert_index_equal(out.categories, Index(['B', 'D', 'F']))
        exp_codes = np.array([2, -1, 1, 0, 1, 2, -1], dtype=np.int8)
        tm.assert_numpy_array_equal(out.codes, exp_codes)
        assert out.tolist() == val
        alpha: List[str] = list('abcdefghijklmnopqrstuvwxyz')
        val = np.random.default_rng(2).choice(alpha[::2], 10000).astype('object')
        val[np.random.default_rng(2).choice(len(val), 100)] = np.nan
        cat = Categorical(values=val, categories=alpha)
        out = cat.remove_unused_categories()
        assert out.tolist() == val.tolist()


class TestCategoricalAPIWithFactor:

    def test_describe(self) -> None:
        factor: Categorical = Categorical(['a', 'b', 'b', 'a', 'a', 'c', 'c', 'c'], ordered=True)
        desc: DataFrame = factor.describe()
        assert factor.ordered
        exp_index: CategoricalIndex = CategoricalIndex(['a', 'b', 'c'], name='categories', ordered=factor.ordered)
        expected: DataFrame = DataFrame({'counts': [3, 2, 3], 'freqs': [3 / 8.0, 2 / 8.0, 3 / 8.0]}, index=exp_index)
        tm.assert_frame_equal(desc, expected)
        cat: Categorical = factor.copy()
        cat = cat.set_categories(['a', 'b', 'c', 'd'])
        desc = cat.describe()
        exp_index = CategoricalIndex(list('abcd'), ordered=factor.ordered, name='categories')
        expected = DataFrame({'counts': [3, 2, 3, 0], 'freqs': [3 / 8.0, 2 / 8.0, 3 / 8.0, 0]}, index=exp_index)
        tm.assert_frame_equal(desc, expected)
        cat = Categorical([1, 2, 3, 1, 2, 3, 3, 2, 1, 1, 1])
        desc = cat.describe()
        exp_index = CategoricalIndex([1, 2, 3], ordered=cat.ordered, name='categories')
        expected = DataFrame({'counts': [5, 3, 3], 'freqs': [5 / 11.0, 3 / 11.0, 3 / 11.0]}, index=exp_index)
        tm.assert_frame_equal(desc, expected)
        cat = Categorical([np.nan, 1, 2, 2])
        desc = cat.describe()
        expected = DataFrame(
            {'counts': [1, 2, 1], 'freqs': [1 / 4.0, 2 / 4.0, 1 / 4.0]},
            index=CategoricalIndex([1, 2, np.nan], categories=[1, 2], name='categories')
        )
        tm.assert_frame_equal(desc, expected)


class TestPrivateCategoricalAPI:

    def test_codes_immutable(self) -> None:
        c: Categorical = Categorical(['a', 'b', 'c', 'a', np.nan])
        exp: np.ndarray = np.array([0, 1, 2, 0, -1], dtype='int8')
        tm.assert_numpy_array_equal(c.codes, exp)
        msg: str = "property 'codes' of 'Categorical' object has no setter" if PY311 else "can't set attribute"
        with pytest.raises(AttributeError, match=msg):
            c.codes = np.array([0, 1, 2, 0, 1], dtype='int8')
        codes: np.ndarray = c.codes
        with pytest.raises(ValueError, match='assignment destination is read-only'):
            codes[4] = 1
        c[4] = 'a'
        exp = np.array([0, 1, 2, 0, 0], dtype='int8')
        tm.assert_numpy_array_equal(c.codes, exp)
        c._codes[4] = 2
        exp = np.array([0, 1, 2, 0, 2], dtype='int8')
        tm.assert_numpy_array_equal(c.codes, exp)

    @pytest.mark.parametrize(
        'codes, old, new, expected', [
            ([0, 1], ['a', 'b'], ['a', 'b'], [0, 1]),
            ([0, 1], ['b', 'a'], ['b', 'a'], [0, 1]),
            ([0, 1], ['a', 'b'], ['b', 'a'], [1, 0]),
            ([0, 1], ['b', 'a'], ['a', 'b'], [1, 0]),
            ([0, 1, 0, 1], ['a', 'b'], ['a', 'b', 'c'], [0, 1, 0, 1]),
            ([0, 1, 2, 2], ['a', 'b', 'c'], ['a', 'b'], [0, 1, -1, -1]),
            ([0, 1, -1], ['a', 'b', 'c'], ['a', 'b', 'c'], [0, 1, -1]),
            ([0, 1, -1], ['a', 'b', 'c'], ['b'], [-1, 0, -1]),
            ([0, 1, -1], ['a', 'b', 'c'], ['d'], [-1, -1, -1]),
            ([0, 1, -1], ['a', 'b', 'c'], [], [-1, -1, -1]),
            ([-1, -1], [], ['a', 'b'], [-1, -1]),
            ([1, 0], ['b', 'a'], ['a', 'b'], [0, 1])
        ]
    )
    def test_recode_to_categories(
        self,
        codes: List[int],
        old: List[str],
        new: List[str],
        expected: List[int]
    ) -> None:
        codes_np: np.ndarray = np.asanyarray(codes, dtype=np.int8)
        expected_np: np.ndarray = np.asanyarray(expected, dtype=np.int8)
        old_index: Index = Index(old)
        new_index: Index = Index(new)
        result: np.ndarray = recode_for_categories(codes_np, old_index, new_index)
        tm.assert_numpy_array_equal(result, expected_np)

    def test_recode_to_categories_large(self) -> None:
        N: int = 1000
        codes: np.ndarray = np.arange(N, dtype=np.int8)
        old: Index = Index(codes)
        expected: np.ndarray = np.arange(N - 1, -1, -1, dtype=np.int16)
        new: Index = Index(expected)
        result: np.ndarray = recode_for_categories(codes, old, new)
        tm.assert_numpy_array_equal(result, expected)
