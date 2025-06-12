import numpy as np
import pytest
from pandas.core.dtypes.concat import union_categoricals
import pandas as pd
from pandas import Categorical, CategoricalIndex, Series
import pandas._testing as tm
from typing import List, Tuple, Union, Any, Optional, TypeVar

T = TypeVar('T')

class TestUnionCategoricals:

    @pytest.mark.parametrize('a, b, combined', [
        (list('abc'), list('abd'), list('abcabd')),
        ([0, 1, 2], [2, 3, 4], [0, 1, 2, 2, 3, 4]),
        ([0, 1.2, 2], [2, 3.4, 4], [0, 1.2, 2, 2, 3.4, 4]),
        (['b', 'b', np.nan, 'a'], ['a', np.nan, 'c'], ['b', 'b', np.nan, 'a', 'a', np.nan, 'c']),
        (pd.date_range('2014-01-01', '2014-01-05'), pd.date_range('2014-01-06', '2014-01-07'), pd.date_range('2014-01-01', '2014-01-07')),
        (pd.date_range('2014-01-01', '2014-01-05', tz='US/Central'), pd.date_range('2014-01-06', '2014-01-07', tz='US/Central'), pd.date_range('2014-01-01', '2014-01-07', tz='US/Central')),
        (pd.period_range('2014-01-01', '2014-01-05'), pd.period_range('2014-01-06', '2014-01-07'), pd.period_range('2014-01-01', '2014-01-07'))
    ])
    @pytest.mark.parametrize('box', [Categorical, CategoricalIndex, Series])
    def test_union_categorical(self, a: List[T], b: List[T], combined: List[T], box: Any) -> None:
        result: Categorical = union_categoricals([box(Categorical(a)), box(Categorical(b))])
        expected: Categorical = Categorical(combined)
        tm.assert_categorical_equal(result, expected)

    def test_union_categorical_ordered_appearance(self) -> None:
        s: Categorical = Categorical(['x', 'y', 'z'])
        s2: Categorical = Categorical(['a', 'b', 'c'])
        result: Categorical = union_categoricals([s, s2])
        expected: Categorical = Categorical(['x', 'y', 'z', 'a', 'b', 'c'], categories=['x', 'y', 'z', 'a', 'b', 'c'])
        tm.assert_categorical_equal(result, expected)

    def test_union_categorical_ordered_true(self) -> None:
        s: Categorical = Categorical([0, 1.2, 2], ordered=True)
        s2: Categorical = Categorical([0, 1.2, 2], ordered=True)
        result: Categorical = union_categoricals([s, s2])
        expected: Categorical = Categorical([0, 1.2, 2, 0, 1.2, 2], ordered=True)
        tm.assert_categorical_equal(result, expected)

    def test_union_categorical_match_types(self) -> None:
        s: Categorical = Categorical([0, 1.2, 2])
        s2: Categorical = Categorical([2, 3, 4])
        msg: str = 'dtype of categories must be the same'
        with pytest.raises(TypeError, match=msg):
            union_categoricals([s, s2])

    def test_union_categorical_empty(self) -> None:
        msg: str = 'No Categoricals to union'
        with pytest.raises(ValueError, match=msg):
            union_categoricals([])

    def test_union_categoricals_nan(self) -> None:
        res: Categorical = union_categoricals([Categorical([1, 2, np.nan]), Categorical([3, 2, np.nan])])
        exp: Categorical = Categorical([1, 2, np.nan, 3, 2, np.nan])
        tm.assert_categorical_equal(res, exp)
        res = union_categoricals([Categorical(['A', 'B']), Categorical(['B', 'B', np.nan])])
        exp = Categorical(['A', 'B', 'B', 'B', np.nan])
        tm.assert_categorical_equal(res, exp)
        val1: List[Union[pd.Timestamp, pd.NaT]] = [pd.Timestamp('2011-01-01'), pd.Timestamp('2011-03-01'), pd.NaT]
        val2: List[Union[pd.Timestamp, pd.NaT]] = [pd.NaT, pd.Timestamp('2011-01-01'), pd.Timestamp('2011-02-01')]
        res = union_categoricals([Categorical(val1), Categorical(val2)])
        exp = Categorical(val1 + val2, categories=[pd.Timestamp('2011-01-01'), pd.Timestamp('2011-03-01'), pd.Timestamp('2011-02-01')])
        tm.assert_categorical_equal(res, exp)
        res = union_categoricals([Categorical(np.array([np.nan, np.nan], dtype=object)), Categorical(['X'], categories=pd.Index(['X'], dtype=object))])
        exp = Categorical([np.nan, np.nan, 'X'])
        tm.assert_categorical_equal(res, exp)
        res = union_categoricals([Categorical([np.nan, np.nan]), Categorical([np.nan, np.nan])])
        exp = Categorical([np.nan, np.nan, np.nan, np.nan])
        tm.assert_categorical_equal(res, exp)

    @pytest.mark.parametrize('val', [[], ['1']])
    def test_union_categoricals_empty(self, val: List[str], request: Any, using_infer_string: bool) -> None:
        if using_infer_string and val == ['1']:
            request.applymarker(pytest.mark.xfail(reason='TDOD(infer_string) object and strings dont match'))
        res: Categorical = union_categoricals([Categorical([]), Categorical(val)])
        exp: Categorical = Categorical(val)
        tm.assert_categorical_equal(res, exp)

    def test_union_categorical_same_category(self) -> None:
        c1: Categorical = Categorical([1, 2, 3, 4], categories=[1, 2, 3, 4])
        c2: Categorical = Categorical([3, 2, 1, np.nan], categories=[1, 2, 3, 4])
        res: Categorical = union_categoricals([c1, c2])
        exp: Categorical = Categorical([1, 2, 3, 4, 3, 2, 1, np.nan], categories=[1, 2, 3, 4])
        tm.assert_categorical_equal(res, exp)

    def test_union_categorical_same_category_str(self) -> None:
        c1: Categorical = Categorical(['z', 'z', 'z'], categories=['x', 'y', 'z'])
        c2: Categorical = Categorical(['x', 'x', 'x'], categories=['x', 'y', 'z'])
        res: Categorical = union_categoricals([c1, c2])
        exp: Categorical = Categorical(['z', 'z', 'z', 'x', 'x', 'x'], categories=['x', 'y', 'z'])
        tm.assert_categorical_equal(res, exp)

    def test_union_categorical_same_categories_different_order(self) -> None:
        c1: Categorical = Categorical(['a', 'b', 'c'], categories=['a', 'b', 'c'])
        c2: Categorical = Categorical(['a', 'b', 'c'], categories=['b', 'a', 'c'])
        result: Categorical = union_categoricals([c1, c2])
        expected: Categorical = Categorical(['a', 'b', 'c', 'a', 'b', 'c'], categories=['a', 'b', 'c'])
        tm.assert_categorical_equal(result, expected)

    def test_union_categoricals_ordered(self) -> None:
        c1: Categorical = Categorical([1, 2, 3], ordered=True)
        c2: Categorical = Categorical([1, 2, 3], ordered=False)
        msg: str = 'Categorical.ordered must be the same'
        with pytest.raises(TypeError, match=msg):
            union_categoricals([c1, c2])
        res: Categorical = union_categoricals([c1, c1])
        exp: Categorical = Categorical([1, 2, 3, 1, 2, 3], ordered=True)
        tm.assert_categorical_equal(res, exp)
        c1 = Categorical([1, 2, 3, np.nan], ordered=True)
        c2 = Categorical([3, 2], categories=[1, 2, 3], ordered=True)
        res = union_categoricals([c1, c2])
        exp = Categorical([1, 2, 3, np.nan, 3, 2], ordered=True)
        tm.assert_categorical_equal(res, exp)
        c1 = Categorical([1, 2, 3], ordered=True)
        c2 = Categorical([1, 2, 3], categories=[3, 2, 1], ordered=True)
        msg = 'to union ordered Categoricals, all categories must be the same'
        with pytest.raises(TypeError, match=msg):
            union_categoricals([c1, c2])

    def test_union_categoricals_ignore_order(self) -> None:
        c1: Categorical = Categorical([1, 2, 3], ordered=True)
        c2: Categorical = Categorical([1, 2, 3], ordered=False)
        res: Categorical = union_categoricals([c1, c2], ignore_order=True)
        exp: Categorical = Categorical([1, 2, 3, 1, 2, 3])
        tm.assert_categorical_equal(res, exp)
        msg: str = 'Categorical.ordered must be the same'
        with pytest.raises(TypeError, match=msg):
            union_categoricals([c1, c2], ignore_order=False)
        res = union_categoricals([c1, c1], ignore_order=True)
        exp = Categorical([1, 2, 3, 1, 2, 3])
        tm.assert_categorical_equal(res, exp)
        res = union_categoricals([c1, c1], ignore_order=False)
        exp = Categorical([1, 2, 3, 1, 2, 3], categories=[1, 2, 3], ordered=True)
        tm.assert_categorical_equal(res, exp)
        c1 = Categorical([1, 2, 3, np.nan], ordered=True)
        c2 = Categorical([3, 2], categories=[1, 2, 3], ordered=True)
        res = union_categoricals([c1, c2], ignore_order=True)
        exp = Categorical([1, 2, 3, np.nan, 3, 2])
        tm.assert_categorical_equal(res, exp)
        c1 = Categorical([1, 2, 3], ordered=True)
        c2 = Categorical([1, 2, 3], categories=[3, 2, 1], ordered=True)
        res = union_categoricals([c1, c2], ignore_order=True)
        exp = Categorical([1, 2, 3, 1, 2, 3])
        tm.assert_categorical_equal(res, exp)
        res = union_categoricals([c2, c1], ignore_order=True, sort_categories=True)
        exp = Categorical([1, 2, 3, 1, 2, 3], categories=[1, 2, 3])
        tm.assert_categorical_equal(res, exp)
        c1 = Categorical([1, 2, 3], ordered=True)
        c2 = Categorical([4, 5, 6], ordered=True)
        result = union_categoricals([c1, c2], ignore_order=True)
        expected = Categorical([1, 2, 3, 4, 5, 6])
        tm.assert_categorical_equal(result, expected)
        msg = 'to union ordered Categoricals, all categories must be the same'
        with pytest.raises(TypeError, match=msg):
            union_categoricals([c1, c2], ignore_order=False)
        with pytest.raises(TypeError, match=msg):
            union_categoricals([c1, c2])

    def test_union_categoricals_sort(self) -> None:
        c1: Categorical = Categorical(['x', 'y', 'z'])
        c2: Categorical = Categorical(['a', 'b', 'c'])
        result: Categorical = union_categoricals([c1, c2], sort_categories=True)
        expected: Categorical = Categorical(['x', 'y', 'z', 'a', 'b', 'c'], categories=['a', 'b', 'c', 'x', 'y', 'z'])
        tm.assert_categorical_equal(result, expected)
        c1 = Categorical(['a', 'b'], categories=['b', 'a', 'c'])
        c2 = Categorical(['b', 'c'], categories=['b', 'a', 'c'])
        result = union_categoricals([c1, c2], sort_categories=True)
        expected = Categorical(['a', 'b', 'b', 'c'], categories=['a', 'b', 'c'])
        tm.assert_categorical_equal(result, expected)
        c1 = Categorical(['a', 'b'], categories=['c', 'a', 'b'])
        c2 = Categorical(['b', 'c'], categories=['c', 'a', 'b'])
        result = union_categoricals([c1, c2], sort_categories=True)
        expected = Categorical(['a', 'b', 'b', 'c'], categories=['a', 'b', 'c'])
        tm.assert_categorical_equal(result, expected)
        c1 = Categorical(['a', 'b'], categories=['a', 'b', 'c'])
        c2 = Categorical(['b', 'c'], categories=['a', 'b', 'c'])
        result = union_categoricals([c1, c2], sort_categories=True)
        expected = Categorical(['a', 'b', 'b', 'c'], categories=['a', 'b', 'c'])
        tm.assert_categorical_equal(result, expected)
        c1 = Categorical(['x', np.nan])
        c2 = Categorical([np.nan, 'b'])
        result = union_categoricals([c1, c2], sort_categories=True)
        expected = Categorical(['x', np.nan, np.nan, 'b'], categories=['b', 'x'])
        tm.assert_categorical_equal(result, expected)
        c1 = Categorical([np.nan])
        c2 = Categorical([np.nan])
        result = union_categoricals([c1, c2], sort_categories=True)
        expected = Categorical([np.nan, np.nan])
        tm.assert_categorical_equal(result, expected)
        c1 = Categorical([])
        c2 = Categorical([])
        result = union_categoricals([c1, c2], sort_categories=True)
        expected = Categorical([])
        tm.assert_categorical_equal(result, expected)
        c1 = Categorical(['b', 'a'], categories=['b', 'a', 'c'], ordered=True)
        c2 = Categorical(['a', 'c'], categories=['b', 'a', 'c'], ordered=True)
        msg = 'Cannot use sort_categories=True with ordered Categoricals'
        with pytest.raises(TypeError, match=msg):
            union_categoricals([c1, c2], sort_categories=True)

    def test_union_categoricals_sort_false(self) -> None:
        c1: Categorical = Categorical(['x', 'y', 'z'])
        c2: Categorical = Categorical(['a', 'b', 'c'])
        result: Categorical = union_categoricals([c1, c2], sort_categories=False)
        expected: Categorical = Categorical(['x', 'y', 'z', 'a', 'b', 'c'], categories=['x', 'y', 'z', 'a', 'b', 'c'])
        tm.assert_categorical_equal(result, expected)

    def test_union_categoricals_sort_false_fastpath(self) -> None:
        c1: Categorical = Categorical(['a', 'b'], categories=['b', 'a', 'c'])
        c2: Categorical = Categorical(['b', 'c'], categories=['b', 'a', 'c'])
        result: Categorical = union_categoricals([c1, c2], sort_categories=False)
        expected: Categorical = Categorical(['a', 'b', 'b', 'c'], categories=['b', 'a', 'c'])
        tm.assert_categorical_equal(result, expected)

    def test_union_categoricals_sort_false_skipresort(self) -> None:
        c1: Categorical = Categorical(['a', 'b'], categories=['a', 'b', 'c'])
        c2: Categorical = Categorical(['b', 'c'], categories=['a', 'b', 'c'])
        result: Categorical = union_categoricals([c1, c2], sort_categories=False)
        expected: Categorical = Categorical(['a', 'b', 'b', 'c'], categories=['a', 'b', 'c'])
        tm.assert_categorical_equal(result, expected)

    def test_union_categoricals_sort_false_one_nan(self) -> None:
        c1: