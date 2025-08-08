import re
import numpy as np
import pytest
from pandas import DataFrame, Index, MultiIndex, Series, concat, option_context
from pandas.util._test_decorators import skip_if_no
from pandas.core import testing as tm

@pytest.fixture
def index_or_series2(index_or_series: Series) -> Series:
    return index_or_series

@pytest.mark.parametrize('other', [None, Series, Index])
def test_str_cat_name(index_or_series: Series, other: Series) -> None:
    box: Series = index_or_series
    values: list = ['a', 'b']
    if other:
        other = other(values)
    else:
        other = values
    result = box(values, name='name').str.cat(other, sep=',')
    assert result.name == 'name'

@pytest.mark.parametrize('infer_string', [False, pytest.param(True, marks=skip_if_no('pyarrow'))])
def test_str_cat(index_or_series: Series, infer_string: bool) -> None:
    with option_context('future.infer_string', infer_string):
        box: Series = index_or_series
        s = box(['a', 'a', 'b', 'b', 'c', np.nan])
        result = s.str.cat()
        expected = 'aabbc'
        assert result == expected
        result = s.str.cat(na_rep='-')
        expected = 'aabbc-'
        assert result == expected
        result = s.str.cat(sep='_', na_rep='NA')
        expected = 'a_a_b_b_c_NA'
        assert result == expected
        t = np.array(['a', np.nan, 'b', 'd', 'foo', np.nan], dtype=object)
        expected = box(['aa', 'a-', 'bb', 'bd', 'cfoo', '--'])
        result = s.str.cat(t, na_rep='-')
        tm.assert_equal(result, expected)
        result = s.str.cat(list(t), na_rep='-')
        tm.assert_equal(result, expected)
        rgx = 'If `others` contains arrays or lists \\(or other list-likes.*'
        z = Series(['1', '2', '3'])
        with pytest.raises(ValueError, match=rgx):
            s.str.cat(z.values)
        with pytest.raises(ValueError, match=rgx):
            s.str.cat(list(z))

def test_str_cat_raises_intuitive_error(index_or_series: Series) -> None:
    box: Series = index_or_series
    s = box(['a', 'b', 'c', 'd'])
    message = 'Did you mean to supply a `sep` keyword?'
    with pytest.raises(ValueError, match=message):
        s.str.cat('|')
    with pytest.raises(ValueError, match=message):
        s.str.cat('    ')
