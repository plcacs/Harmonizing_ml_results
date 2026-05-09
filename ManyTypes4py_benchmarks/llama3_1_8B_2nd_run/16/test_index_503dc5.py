"""
Tests that work on both the Python and C engines but do not have a
specific classification into the other test modules.
"""
from datetime import datetime
from io import StringIO
import os
import pytest
from pandas import DataFrame, Index, MultiIndex
import pandas._testing as tm

pytestmark: pytest.Mark = pytest.mark.filterwarnings('ignore:Passing a BlockManager to DataFrame:DeprecationWarning')
xfail_pyarrow: pytest.Mark = pytest.mark.usefixtures('pyarrow_xfail')
skip_pyarrow: pytest.Mark = pytest.mark.usefixtures('pyarrow_skip')

@pytest.mark.parametrize('data, kwargs, expected', [
    ('foo,2,3,4,5\nbar,7,8,9,10\nbaz,12,13,14,15\nqux,12,13,14,15\nfoo2,12,13,14,15\nbar2,12,13,14,15\n', 
     {'index_col': 0, 'names': ['index', 'A', 'B', 'C', 'D']}, 
     DataFrame([[2, 3, 4, 5], [7, 8, 9, 10], [12, 13, 14, 15], [12, 13, 14, 15], [12, 13, 14, 15], [12, 13, 14, 15]], 
                index=Index(['foo', 'bar', 'baz', 'qux', 'foo2', 'bar2'], name='index'), columns=['A', 'B', 'C', 'D'])),
    ('foo,one,2,3,4,5\nfoo,two,7,8,9,10\nfoo,three,12,13,14,15\nbar,one,12,13,14,15\nbar,two,12,13,14,15\n', 
     {'index_col': [0, 1], 'names': ['index1', 'index2', 'A', 'B', 'C', 'D']}, 
     DataFrame([[2, 3, 4, 5], [7, 8, 9, 10], [12, 13, 14, 15], [12, 13, 14, 15], [12, 13, 14, 15]], 
                index=MultiIndex.from_tuples([('foo', 'one'), ('foo', 'two'), ('foo', 'three'), ('bar', 'one'), ('bar', 'two')], 
                                             names=['index1', 'index2']), columns=['A', 'B', 'C', 'D']))
])
def test_pass_names_with_index(all_parsers: object, data: str, kwargs: dict, expected: DataFrame) -> None:
    parser: object = all_parsers
    result: DataFrame = parser.read_csv(StringIO(data), **kwargs)
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('index_col', [[0, 1], [1, 0]])
def test_multi_index_no_level_names(request: pytest.FixtureRequest, all_parsers: object, index_col: list, using_infer_string: str) -> None:
    data: str = 'index1,index2,A,B,C,D\nfoo,one,2,3,4,5\nfoo,two,7,8,9,10\nfoo,three,12,13,14,15\nbar,one,12,13,14,15\nbar,two,12,13,14,15\n'
    headless_data: str = '\n'.join(data.split('\n')[1:])
    names: list = ['A', 'B', 'C', 'D']
    parser: object = all_parsers
    result: DataFrame = parser.read_csv(StringIO(headless_data), index_col=index_col, header=None, names=names)
    expected: DataFrame = parser.read_csv(StringIO(data), index_col=index_col)
    expected.index.names = [None] * 2
    tm.assert_frame_equal(result, expected)

@skip_pyarrow
def test_multi_index_no_level_names_implicit(all_parsers: object) -> None:
    parser: object = all_parsers
    data: str = 'A,B,C,D\nfoo,one,2,3,4,5\nfoo,two,7,8,9,10\nfoo,three,12,13,14,15\nbar,one,12,13,14,15\nbar,two,12,13,14,15\n'
    result: DataFrame = parser.read_csv(StringIO(data))
    expected: DataFrame = DataFrame([[2, 3, 4, 5], [7, 8, 9, 10], [12, 13, 14, 15], [12, 13, 14, 15], [12, 13, 14, 15]], 
                                     columns=['A', 'B', 'C', 'D'], index=MultiIndex.from_tuples([('foo', 'one'), ('foo', 'two'), ('foo', 'three'), ('bar', 'one'), ('bar', 'two')]))
    tm.assert_frame_equal(result, expected)

@xfail_pyarrow
@pytest.mark.parametrize('data, columns, header', [
    ('a,b', ['a', 'b'], [0]),
    ('a,b\nc,d', MultiIndex.from_tuples([('a', 'c'), ('b', 'd')]), [0, 1])
])
@pytest.mark.parametrize('round_trip', [True, False])
def test_multi_index_blank_df(all_parsers: object, data: str, columns: list, header: list, round_trip: bool) -> None:
    parser: object = all_parsers
    expected: DataFrame = DataFrame(columns=columns)
    data: str = expected.to_csv(index=False) if round_trip else data
    result: DataFrame = parser.read_csv(StringIO(data), header=header)
    tm.assert_frame_equal(result, expected)

@xfail_pyarrow
def test_no_unnamed_index(all_parsers: object) -> None:
    parser: object = all_parsers
    data: str = ' id c0 c1 c2\n0 1 0 a b\n1 2 0 c d\n2 2 2 e f\n'
    result: DataFrame = parser.read_csv(StringIO(data), sep=' ')
    expected: DataFrame = DataFrame([[0, 1, 0, 'a', 'b'], [1, 2, 0, 'c', 'd'], [2, 2, 2, 'e', 'f']], 
                                     columns=['Unnamed: 0', 'id', 'c0', 'c1', 'c2'])
    tm.assert_frame_equal(result, expected)

def test_read_duplicate_index_explicit(all_parsers: object) -> None:
    data: str = 'index,A,B,C,D\nfoo,2,3,4,5\nbar,7,8,9,10\nbaz,12,13,14,15\nqux,12,13,14,15\nfoo,12,13,14,15\nbar,12,13,14,15\n'
    parser: object = all_parsers
    result: DataFrame = parser.read_csv(StringIO(data), index_col=0)
    expected: DataFrame = DataFrame([[2, 3, 4, 5], [7, 8, 9, 10], [12, 13, 14, 15], [12, 13, 14, 15], [12, 13, 14, 15], [12, 13, 14, 15]], 
                                     columns=['A', 'B', 'C', 'D'], index=Index(['foo', 'bar', 'baz', 'qux', 'foo', 'bar'], name='index'))
    tm.assert_frame_equal(result, expected)

@skip_pyarrow
def test_read_duplicate_index_implicit(all_parsers: object) -> None:
    data: str = 'A,B,C,D\nfoo,2,3,4,5\nbar,7,8,9,10\nbaz,12,13,14,15\nqux,12,13,14,15\nfoo,12,13,14,15\nbar,12,13,14,15\n'
    parser: object = all_parsers
    result: DataFrame = parser.read_csv(StringIO(data))
    expected: DataFrame = DataFrame([[2, 3, 4, 5], [7, 8, 9, 10], [12, 13, 14, 15], [12, 13, 14, 15], [12, 13, 14, 15], [12, 13, 14, 15]], 
                                     columns=['A', 'B', 'C', 'D'], index=Index(['foo', 'bar', 'baz', 'qux', 'foo', 'bar']))
    tm.assert_frame_equal(result, expected)

@skip_pyarrow
def test_read_csv_no_index_name(all_parsers: object, csv_dir_path: str) -> None:
    parser: object = all_parsers
    csv2: str = os.path.join(csv_dir_path, 'test2.csv')
    result: DataFrame = parser.read_csv(csv2, index_col=0, parse_dates=True)
    expected: DataFrame = DataFrame([[0.980269, 3.685731, -0.364216805298, -1.159738, 'foo'], 
                                      [1.047916, -0.041232, -0.16181208307, 0.212549, 'bar'], 
                                      [0.498581, 0.731168, -0.537677223318, 1.34627, 'baz'], 
                                      [1.120202, 1.567621, 0.00364077397681, 0.675253, 'qux'], 
                                      [-0.487094, 0.571455, -1.6116394093, 0.103469, 'foo2']], 
                                     columns=['A', 'B', 'C', 'D', 'E'], 
                                     index=Index([datetime(2000, 1, 3), datetime(2000, 1, 4), datetime(2000, 1, 5), datetime(2000, 1, 6), datetime(2000, 1, 7)], dtype='M8[s]'))
    tm.assert_frame_equal(result, expected)

@skip_pyarrow
def test_empty_with_index(all_parsers: object) -> None:
    data: str = 'x,y'
    parser: object = all_parsers
    result: DataFrame = parser.read_csv(StringIO(data), index_col=0)
    expected: DataFrame = DataFrame(columns=['y'], index=Index([], name='x'))
    tm.assert_frame_equal(result, expected)

@skip_pyarrow
def test_empty_with_multi_index(all_parsers: object) -> None:
    data: str = 'x,y,z'
    parser: object = all_parsers
    result: DataFrame = parser.read_csv(StringIO(data), index_col=['x', 'y'])
    expected: DataFrame = DataFrame(columns=['z'], index=MultiIndex.from_arrays([[]] * 2, names=['x', 'y']))
    tm.assert_frame_equal(result, expected)

@skip_pyarrow
def test_empty_with_reversed_multi_index(all_parsers: object) -> None:
    data: str = 'x,y,z'
    parser: object = all_parsers
    result: DataFrame = parser.read_csv(StringIO(data), index_col=[1, 0])
    expected: DataFrame = DataFrame(columns=['z'], index=MultiIndex.from_arrays([[]] * 2, names=['y', 'x']))
    tm.assert_frame_equal(result, expected)
