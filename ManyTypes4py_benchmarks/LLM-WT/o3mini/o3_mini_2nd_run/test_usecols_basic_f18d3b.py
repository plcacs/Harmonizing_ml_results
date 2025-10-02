from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserError
from pandas import DataFrame, Index, array
import pandas._testing as tm
from typing import Any, Callable, List, Tuple, Union, Optional, Dict

pytestmark = pytest.mark.filterwarnings('ignore:Passing a BlockManager to DataFrame:DeprecationWarning')
_msg_validate_usecols_arg: str = "'usecols' must either be list-like of all strings, all unicode, all integers or a callable."
_msg_validate_usecols_names: str = 'Usecols do not match columns, columns expected but not found: {0}'
_msg_pyarrow_requires_names: str = "The pyarrow engine does not allow 'usecols' to be integer column positions. Pass a list of string column names instead."
xfail_pyarrow = pytest.mark.usefixtures('pyarrow_xfail')
skip_pyarrow = pytest.mark.usefixtures('pyarrow_skip')
pytestmark = pytest.mark.filterwarnings('ignore:Passing a BlockManager to DataFrame is deprecated:DeprecationWarning')


def test_raise_on_mixed_dtype_usecols(all_parsers: Any) -> None:
    data: str = 'a,b,c\n        1000,2000,3000\n        4000,5000,6000\n        '
    usecols: List[Union[int, str]] = [0, 'b', 2]
    parser: Any = all_parsers
    with pytest.raises(ValueError, match=_msg_validate_usecols_arg):
        parser.read_csv(StringIO(data), usecols=usecols)


@pytest.mark.parametrize('usecols', [(1, 2), ('b', 'c')])
def test_usecols(all_parsers: Any, usecols: Union[Tuple[int, int], Tuple[str, str]], request: Any) -> None:
    data: str = 'a,b,c\n1,2,3\n4,5,6\n7,8,9\n10,11,12'
    parser: Any = all_parsers
    if parser.engine == 'pyarrow' and isinstance(usecols[0], int):
        with pytest.raises(ValueError, match=_msg_pyarrow_requires_names):
            parser.read_csv(StringIO(data), usecols=usecols)
        return
    result: DataFrame = parser.read_csv(StringIO(data), usecols=usecols)
    expected: DataFrame = DataFrame([[2, 3], [5, 6], [8, 9], [11, 12]], columns=['b', 'c'])
    tm.assert_frame_equal(result, expected)


def test_usecols_with_names(all_parsers: Any) -> None:
    data: str = 'a,b,c\n1,2,3\n4,5,6\n7,8,9\n10,11,12'
    parser: Any = all_parsers
    names: List[str] = ['foo', 'bar']
    if parser.engine == 'pyarrow':
        with pytest.raises(ValueError, match=_msg_pyarrow_requires_names):
            parser.read_csv(StringIO(data), names=names, usecols=[1, 2], header=0)
        return
    result: DataFrame = parser.read_csv(StringIO(data), names=names, usecols=[1, 2], header=0)
    expected: DataFrame = DataFrame([[2, 3], [5, 6], [8, 9], [11, 12]], columns=names)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize('names,usecols', [
    (['b', 'c'], [1, 2]),
    (['a', 'b', 'c'], ['b', 'c'])
])
def test_usecols_relative_to_names(all_parsers: Any, names: List[str], usecols: List[Union[int, str]]) -> None:
    data: str = '1,2,3\n4,5,6\n7,8,9\n10,11,12'
    parser: Any = all_parsers
    if parser.engine == 'pyarrow' and (not isinstance(usecols[0], int)):
        pytest.skip(reason='https://github.com/apache/arrow/issues/38676')
    result: DataFrame = parser.read_csv(StringIO(data), names=names, header=None, usecols=usecols)
    expected: DataFrame = DataFrame([[2, 3], [5, 6], [8, 9], [11, 12]], columns=['b', 'c'])
    tm.assert_frame_equal(result, expected)


def test_usecols_relative_to_names2(all_parsers: Any) -> None:
    data: str = '1,2,3\n4,5,6\n7,8,9\n10,11,12'
    parser: Any = all_parsers
    result: DataFrame = parser.read_csv(StringIO(data), names=['a', 'b'], header=None, usecols=[0, 1])
    expected: DataFrame = DataFrame([[1, 2], [4, 5], [7, 8], [10, 11]], columns=['a', 'b'])
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow
def test_usecols_name_length_conflict(all_parsers: Any) -> None:
    data: str = '1,2,3\n4,5,6\n7,8,9\n10,11,12'
    parser: Any = all_parsers
    msg: str = 'Number of passed names did not match number of header fields in the file'
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), names=['a', 'b'], header=None, usecols=[1])


def test_usecols_single_string(all_parsers: Any) -> None:
    parser: Any = all_parsers
    data: str = 'foo, bar, baz\n1000, 2000, 3000\n4000, 5000, 6000'
    with pytest.raises(ValueError, match=_msg_validate_usecols_arg):
        parser.read_csv(StringIO(data), usecols='foo')


@skip_pyarrow
@pytest.mark.parametrize('data', ['a,b,c,d\n1,2,3,4\n5,6,7,8', 'a,b,c,d\n1,2,3,4,\n5,6,7,8,'])
def test_usecols_index_col_false(all_parsers: Any, data: str) -> None:
    parser: Any = all_parsers
    usecols: List[str] = ['a', 'c', 'd']
    expected: DataFrame = DataFrame({'a': [1, 5], 'c': [3, 7], 'd': [4, 8]})
    result: DataFrame = parser.read_csv(StringIO(data), usecols=usecols, index_col=False)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize('index_col', ['b', 0])
@pytest.mark.parametrize('usecols', [['b', 'c'], [1, 2]])
def test_usecols_index_col_conflict(all_parsers: Any, usecols: List[Union[int, str]], index_col: Union[str, int], request: Any) -> None:
    parser: Any = all_parsers
    data: str = 'a,b,c,d\nA,a,1,one\nB,b,2,two'
    if parser.engine == 'pyarrow' and isinstance(usecols[0], int):
        with pytest.raises(ValueError, match=_msg_pyarrow_requires_names):
            parser.read_csv(StringIO(data), usecols=usecols, index_col=index_col)
        return
    expected: DataFrame = DataFrame({'c': [1, 2]}, index=Index(['a', 'b'], name='b'))
    result: DataFrame = parser.read_csv(StringIO(data), usecols=usecols, index_col=index_col)
    tm.assert_frame_equal(result, expected)


def test_usecols_index_col_conflict2(all_parsers: Any) -> None:
    parser: Any = all_parsers
    data: str = 'a,b,c,d\nA,a,1,one\nB,b,2,two'
    expected: DataFrame = DataFrame({'b': ['a', 'b'], 'c': [1, 2], 'd': ('one', 'two')})
    expected = expected.set_index(['b', 'c'])
    result: DataFrame = parser.read_csv(StringIO(data), usecols=['b', 'c', 'd'], index_col=['b', 'c'])
    tm.assert_frame_equal(result, expected)


@skip_pyarrow
def test_usecols_implicit_index_col(all_parsers: Any) -> None:
    parser: Any = all_parsers
    data: str = 'a,b,c\n4,apple,bat,5.7\n8,orange,cow,10'
    result: DataFrame = parser.read_csv(StringIO(data), usecols=['a', 'b'])
    expected: DataFrame = DataFrame({'a': ['apple', 'orange'], 'b': ['bat', 'cow']}, index=[4, 8])
    tm.assert_frame_equal(result, expected)


def test_usecols_index_col_middle(all_parsers: Any) -> None:
    parser: Any = all_parsers
    data: str = 'a,b,c,d\n1,2,3,4\n'
    result: DataFrame = parser.read_csv(StringIO(data), usecols=['b', 'c', 'd'], index_col='c')
    expected: DataFrame = DataFrame({'b': [2], 'd': [4]}, index=Index([3], name='c'))
    tm.assert_frame_equal(result, expected)


def test_usecols_index_col_end(all_parsers: Any) -> None:
    parser: Any = all_parsers
    data: str = 'a,b,c,d\n1,2,3,4\n'
    result: DataFrame = parser.read_csv(StringIO(data), usecols=['b', 'c', 'd'], index_col='d')
    expected: DataFrame = DataFrame({'b': [2], 'c': [3]}, index=Index([4], name='d'))
    tm.assert_frame_equal(result, expected)


def test_usecols_regex_sep(all_parsers: Any) -> None:
    parser: Any = all_parsers
    data: str = 'a  b  c\n4  apple  bat  5.7\n8  orange  cow  10'
    if parser.engine == 'pyarrow':
        msg: str = "the 'pyarrow' engine does not support regex separators"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), sep='\\s+', usecols=('a', 'b'))
        return
    result: DataFrame = parser.read_csv(StringIO(data), sep='\\s+', usecols=('a', 'b'))
    expected: DataFrame = DataFrame({'a': ['apple', 'orange'], 'b': ['bat', 'cow']}, index=[4, 8])
    tm.assert_frame_equal(result, expected)


@skip_pyarrow
def test_usecols_with_whitespace(all_parsers: Any) -> None:
    parser: Any = all_parsers
    data: str = 'a  b  c\n4  apple  bat  5.7\n8  orange  cow  10'
    result: DataFrame = parser.read_csv(StringIO(data), sep='\\s+', usecols=('a', 'b'))
    expected: DataFrame = DataFrame({'a': ['apple', 'orange'], 'b': ['bat', 'cow']}, index=[4, 8])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize('usecols,expected', [
    ([0, 1], DataFrame(data=[[1000, 2000], [4000, 5000]], columns=['2', '0'])),
    (['0', '1'], DataFrame(data=[[2000, 3000], [5000, 6000]], columns=['0', '1']))
])
def test_usecols_with_integer_like_header(all_parsers: Any, usecols: Union[List[int], List[str]], expected: DataFrame, request: Any) -> None:
    parser: Any = all_parsers
    data: str = '2,0,1\n1000,2000,3000\n4000,5000,6000'
    if parser.engine == 'pyarrow' and isinstance(usecols[0], int):
        with pytest.raises(ValueError, match=_msg_pyarrow_requires_names):
            parser.read_csv(StringIO(data), usecols=usecols)
        return
    result: DataFrame = parser.read_csv(StringIO(data), usecols=usecols)
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow
def test_empty_usecols(all_parsers: Any) -> None:
    data: str = 'a,b,c\n1,2,3\n4,5,6'
    expected: DataFrame = DataFrame(columns=Index([]))
    parser: Any = all_parsers
    result: DataFrame = parser.read_csv(StringIO(data), usecols=set())
    tm.assert_frame_equal(result, expected)


def test_np_array_usecols(all_parsers: Any) -> None:
    parser: Any = all_parsers
    data: str = 'a,b,c\n1,2,3'
    usecols: np.ndarray = np.array(['a', 'b'])
    expected: DataFrame = DataFrame([[1, 2]], columns=usecols)
    result: DataFrame = parser.read_csv(StringIO(data), usecols=usecols)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize('usecols,expected', [
    (lambda x: x.upper() in ['AAA', 'BBB', 'DDD'],
     DataFrame({'AaA': {0: 0.056674973, 1: 2.6132309819999997, 2: 3.5689350380000002},
                'bBb': {0: 8, 1: 2, 2: 7},
                'ddd': {0: 'a', 1: 'b', 2: 'a'}})),
    (lambda x: False, DataFrame(columns=Index([])))
])
def test_callable_usecols(all_parsers: Any, usecols: Callable[[str], bool], expected: DataFrame) -> None:
    data: str = 'AaA,bBb,CCC,ddd\n0.056674973,8,True,a\n2.613230982,2,False,b\n3.568935038,7,False,a'
    parser: Any = all_parsers
    if parser.engine == 'pyarrow':
        msg: str = "The pyarrow engine does not allow 'usecols' to be a callable"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), usecols=usecols)
        return
    result: DataFrame = parser.read_csv(StringIO(data), usecols=usecols)
    tm.assert_frame_equal(result, expected)


@skip_pyarrow
@pytest.mark.parametrize('usecols', [['a', 'c'], lambda x: x in ['a', 'c']])
def test_incomplete_first_row(all_parsers: Any, usecols: Union[List[str], Callable[[str], bool]]) -> None:
    data: str = '1,2\n1,2,3'
    parser: Any = all_parsers
    names: List[str] = ['a', 'b', 'c']
    expected: DataFrame = DataFrame({'a': [1, 1], 'c': [np.nan, 3]})
    result: DataFrame = parser.read_csv(StringIO(data), names=names, usecols=usecols)
    tm.assert_frame_equal(result, expected)


@skip_pyarrow
@pytest.mark.parametrize('data,usecols,kwargs,expected', [
    ('19,29,39\n' * 2 + '10,20,30,40', [0, 1, 2], {'header': None}, [[19, 29, 39], [19, 29, 39], [10, 20, 30]]),
    ('A,B,C\n1,2,3\n3,4,5\n1,2,4,5,1,6\n1,2,3,,,1,\n1,2,3\n5,6,7', ['A', 'B', 'C'], {}, {'A': [1, 3, 1, 1, 1, 5], 'B': [2, 4, 2, 2, 2, 6], 'C': [3, 5, 4, 3, 3, 7]})
])
def test_uneven_length_cols(all_parsers: Any, data: str, usecols: List[Union[int, str]], kwargs: Dict[str, Any], expected: Any) -> None:
    parser: Any = all_parsers
    result: DataFrame = parser.read_csv(StringIO(data), usecols=usecols, **kwargs)
    expected_df: DataFrame = DataFrame(expected)
    tm.assert_frame_equal(result, expected_df)


@pytest.mark.parametrize('usecols,kwargs,expected,msg', [
    (['a', 'b', 'c', 'd'], {}, DataFrame({'a': [1, 5], 'b': [2, 6], 'c': [3, 7], 'd': [4, 8]}), None),
    (['a', 'b', 'c', 'f'], {}, None, _msg_validate_usecols_names.format("\\['f'\\]")),
    (['a', 'b', 'f'], {}, None, _msg_validate_usecols_names.format("\\['f'\\]")),
    (['a', 'b', 'f', 'g'], {}, None, _msg_validate_usecols_names.format("\\[('f', 'g'|'g', 'f')\\]")),
    (None, {'header': 0, 'names': ['A', 'B', 'C', 'D']}, DataFrame({'A': [1, 5], 'B': [2, 6], 'C': [3, 7], 'D': [4, 8]}), None),
    (['A', 'B', 'C', 'f'], {'header': 0, 'names': ['A', 'B', 'C', 'D']}, None, _msg_validate_usecols_names.format("\\['f'\\]")),
    (['A', 'B', 'f'], {'names': ['A', 'B', 'C', 'D']}, None, _msg_validate_usecols_names.format("\\['f'\\]"))
])
def test_raises_on_usecols_names_mismatch(all_parsers: Any, usecols: Any, kwargs: Dict[str, Any], expected: Optional[DataFrame], msg: Optional[str], request: Any) -> None:
    data: str = 'a,b,c,d\n1,2,3,4\n5,6,7,8'
    kwargs.update(usecols=usecols)
    parser: Any = all_parsers
    if parser.engine == 'pyarrow' and (not (usecols is not None and expected is not None)):
        pytest.skip(reason='https://github.com/apache/arrow/issues/38676')
    if expected is None:
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), **kwargs)
    else:
        result: DataFrame = parser.read_csv(StringIO(data), **kwargs)
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize('usecols', [['A', 'C'], [0, 2]])
def test_usecols_subset_names_mismatch_orig_columns(all_parsers: Any, usecols: Union[List[Union[int, str]], Callable[[str], bool]], request: Any) -> None:
    data: str = 'a,b,c,d\n1,2,3,4\n5,6,7,8'
    names: List[str] = ['A', 'B', 'C', 'D']
    parser: Any = all_parsers
    if parser.engine == 'pyarrow':
        if isinstance(usecols, list) and isinstance(usecols[0], int):
            with pytest.raises(ValueError, match=_msg_pyarrow_requires_names):
                parser.read_csv(StringIO(data), header=0, names=names, usecols=usecols)
            return
        pytest.skip(reason='https://github.com/apache/arrow/issues/38676')
    result: DataFrame = parser.read_csv(StringIO(data), header=0, names=names, usecols=usecols)
    expected: DataFrame = DataFrame({'A': [1, 5], 'C': [3, 7]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize('names', [None, ['a', 'b']])
def test_usecols_indices_out_of_bounds(all_parsers: Any, names: Optional[List[str]]) -> None:
    parser: Any = all_parsers
    data: str = '\na,b\n1,2\n    '
    err: Any = ParserError
    msg: str = 'Defining usecols with out-of-bounds'
    if parser.engine == 'pyarrow':
        err = ValueError
        msg = _msg_pyarrow_requires_names
    with pytest.raises(err, match=msg):
        parser.read_csv(StringIO(data), usecols=[0, 2], names=names, header=0)


def test_usecols_additional_columns(all_parsers: Any) -> None:
    parser: Any = all_parsers
    usecols: Callable[[str], bool] = lambda header: header.strip() in ['a', 'b', 'c']
    if parser.engine == 'pyarrow':
        msg: str = "The pyarrow engine does not allow 'usecols' to be a callable"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO('a,b\nx,y,z'), index_col=False, usecols=usecols)
        return
    result: DataFrame = parser.read_csv(StringIO('a,b\nx,y,z'), index_col=False, usecols=usecols)
    expected: DataFrame = DataFrame({'a': ['x'], 'b': 'y'})
    tm.assert_frame_equal(result, expected)


def test_usecols_additional_columns_integer_columns(all_parsers: Any) -> None:
    parser: Any = all_parsers
    usecols: Callable[[str], bool] = lambda header: header.strip() in ['0', '1']
    if parser.engine == 'pyarrow':
        msg: str = "The pyarrow engine does not allow 'usecols' to be a callable"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO('0,1\nx,y,z'), index_col=False, usecols=usecols)
        return
    result: DataFrame = parser.read_csv(StringIO('0,1\nx,y,z'), index_col=False, usecols=usecols)
    expected: DataFrame = DataFrame({'0': ['x'], '1': 'y'})
    tm.assert_frame_equal(result, expected)


def test_usecols_dtype(all_parsers: Any) -> None:
    parser: Any = all_parsers
    data: str = '\ncol1,col2,col3\na,1,x\nb,2,y\n'
    result: DataFrame = parser.read_csv(StringIO(data), usecols=['col1', 'col2'], dtype={'col1': 'string', 'col2': 'uint8', 'col3': 'string'})
    expected: DataFrame = DataFrame({'col1': array(['a', 'b']), 'col2': np.array([1, 2], dtype='uint8')})
    tm.assert_frame_equal(result, expected)