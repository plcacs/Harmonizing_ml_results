from io import StringIO
from typing import Any, Dict, List, Union
import numpy as np
import pytest
from pandas._libs.parsers import STR_NA_VALUES
from pandas import DataFrame, Index, MultiIndex
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings('ignore:Passing a BlockManager to DataFrame:DeprecationWarning')
xfail_pyarrow = pytest.mark.usefixtures('pyarrow_xfail')
skip_pyarrow = pytest.mark.usefixtures('pyarrow_skip')

def test_string_nas(all_parsers: Any) -> None:
    parser: Any = all_parsers
    data: str = 'A,B,C\na,b,c\nd,,f\n,g,h\n'
    result: DataFrame = parser.read_csv(StringIO(data))
    expected: DataFrame = DataFrame([['a', 'b', 'c'], ['d', np.nan, 'f'], [np.nan, 'g', 'h']], columns=['A', 'B', 'C'])
    if parser.engine == 'pyarrow':
        expected.loc[2, 'A'] = None
        expected.loc[1, 'B'] = None
    tm.assert_frame_equal(result, expected)

def test_detect_string_na(all_parsers: Any) -> None:
    parser: Any = all_parsers
    data: str = 'A,B\nfoo,bar\nNA,baz\nNaN,nan\n'
    expected: DataFrame = DataFrame([['foo', 'bar'], [np.nan, 'baz'], [np.nan, np.nan]], columns=['A', 'B'])
    if parser.engine == 'pyarrow':
        expected.loc[[1, 2], 'A'] = None
        expected.loc[2, 'B'] = None
    result: DataFrame = parser.read_csv(StringIO(data))
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize(
    'na_values',
    [
        ['-999.0', '-999'],
        [-999, -999.0],
        [-999.0, -999],
        ['-999.0'],
        ['-999'],
        [-999.0],
        [-999]
    ]
)
@pytest.mark.parametrize(
    'data',
    [
        'A,B\n-999,1.2\n2,-999\n3,4.5\n',
        'A,B\n-999,1.200\n2,-999.000\n3,4.500\n'
    ]
)
def test_non_string_na_values(all_parsers: Any, data: str, na_values: Any, request: Any) -> None:
    parser: Any = all_parsers
    expected: DataFrame = DataFrame([[np.nan, 1.2], [2.0, np.nan], [3.0, 4.5]], columns=['A', 'B'])
    if parser.engine == 'pyarrow' and (not all((isinstance(x, str) for x in na_values))):
        msg: str = "The 'pyarrow' engine requires all na_values to be strings"
        with pytest.raises(TypeError, match=msg):
            parser.read_csv(StringIO(data), na_values=na_values)
        return
    elif parser.engine == 'pyarrow' and '-999.000' in data:
        mark = pytest.mark.xfail(reason='pyarrow engined does not recognize equivalent floats')
        request.applymarker(mark)
    result: DataFrame = parser.read_csv(StringIO(data), na_values=na_values)
    tm.assert_frame_equal(result, expected)

def test_default_na_values(all_parsers: Any) -> None:
    _NA_VALUES: set = {'-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A', 'N/A', 'n/a', 'NA', '<NA>', '#NA', 'NULL', 'null', 'NaN', 'nan', '-NaN', '-nan', '#N/A N/A', '', 'None'}
    assert _NA_VALUES == STR_NA_VALUES
    parser: Any = all_parsers
    nv: int = len(_NA_VALUES)

    def f(i: int, v: str) -> str:
        if i == 0:
            buf: str = ''
        elif i > 0:
            buf = ''.join([','] * i)
        buf = f'{buf}{v}'
        if i < nv - 1:
            joined: str = ''.join([','] * (nv - i - 1))
            buf = f'{buf}{joined}'
        return buf

    data: StringIO = StringIO('\n'.join([f(i, v) for i, v in enumerate(_NA_VALUES)]))
    expected: DataFrame = DataFrame(np.nan, columns=range(nv), index=range(nv))
    result: DataFrame = parser.read_csv(data, header=None)
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('na_values', ['baz', ['baz']])
def test_custom_na_values(all_parsers: Any, na_values: Union[str, List[str]]) -> None:
    parser: Any = all_parsers
    data: str = 'A,B,C\nignore,this,row\n1,NA,3\n-1.#IND,5,baz\n7,8,NaN\n'
    expected: DataFrame = DataFrame([[1.0, np.nan, 3], [np.nan, 5, np.nan], [7, 8, np.nan]], columns=['A', 'B', 'C'])
    if parser.engine == 'pyarrow':
        msg: str = "skiprows argument must be an integer when using engine='pyarrow'"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), na_values=na_values, skiprows=[1])
        return
    result: DataFrame = parser.read_csv(StringIO(data), na_values=na_values, skiprows=[1])
    tm.assert_frame_equal(result, expected)

def test_bool_na_values(all_parsers: Any) -> None:
    data: str = 'A,B,C\nTrue,False,True\nNA,True,False\nFalse,NA,True'
    parser: Any = all_parsers
    result: DataFrame = parser.read_csv(StringIO(data))
    expected: DataFrame = DataFrame({
        'A': np.array([True, np.nan, False], dtype=object),
        'B': np.array([False, True, np.nan], dtype=object),
        'C': [True, False, True]
    })
    if parser.engine == 'pyarrow':
        expected.loc[1, 'A'] = None
        expected.loc[2, 'B'] = None
    tm.assert_frame_equal(result, expected)

def test_na_value_dict(all_parsers: Any) -> None:
    data: str = 'A,B,C\nfoo,bar,NA\nbar,foo,foo\nfoo,bar,NA\nbar,foo,foo'
    parser: Any = all_parsers
    if parser.engine == 'pyarrow':
        msg: str = "pyarrow engine doesn't support passing a dict for na_values"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), na_values={'A': ['foo'], 'B': ['bar']})
        return
    df: DataFrame = parser.read_csv(StringIO(data), na_values={'A': ['foo'], 'B': ['bar']})
    expected: DataFrame = DataFrame({'A': [np.nan, 'bar', np.nan, 'bar'], 'B': [np.nan, 'foo', np.nan, 'foo'], 'C': [np.nan, 'foo', np.nan, 'foo']})
    tm.assert_frame_equal(df, expected)

@pytest.mark.parametrize(
    'index_col,expected',
    [
        ([0], DataFrame({'b': [np.nan], 'c': [1], 'd': [5]}, index=Index([0], name='a'))),
        ([0, 2], DataFrame({'b': [np.nan], 'd': [5]}, index=MultiIndex.from_tuples([(0, 1)], names=['a', 'c']))),
        (['a', 'c'], DataFrame({'b': [np.nan], 'd': [5]}, index=MultiIndex.from_tuples([(0, 1)], names=['a', 'c'])))
    ]
)
def test_na_value_dict_multi_index(all_parsers: Any, index_col: Any, expected: DataFrame) -> None:
    data: str = 'a,b,c,d\n0,NA,1,5\n'
    parser: Any = all_parsers
    result: DataFrame = parser.read_csv(StringIO(data), na_values=set(), index_col=index_col)
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize(
    'kwargs,expected',
    [
        (
            {},
            {'A': ['a', 'b', np.nan, 'd', 'e', np.nan, 'g'], 'B': [1, 2, 3, 4, 5, 6, 7], 'C': ['one', 'two', 'three', np.nan, 'five', np.nan, 'seven']}
        ),
        (
            {'na_values': {'A': [], 'C': []}, 'keep_default_na': False},
            {'A': ['a', 'b', '', 'd', 'e', 'nan', 'g'], 'B': [1, 2, 3, 4, 5, 6, 7], 'C': ['one', 'two', 'three', 'nan', 'five', '', 'seven']}
        ),
        (
            {'na_values': ['a'], 'keep_default_na': False},
            {'A': [np.nan, 'b', '', 'd', 'e', 'nan', 'g'], 'B': [1, 2, 3, 4, 5, 6, 7], 'C': ['one', 'two', 'three', 'nan', 'five', '', 'seven']}
        ),
        (
            {'na_values': {'A': [], 'C': []}},
            {'A': ['a', 'b', np.nan, 'd', 'e', np.nan, 'g'], 'B': [1, 2, 3, 4, 5, 6, 7], 'C': ['one', 'two', 'three', np.nan, 'five', np.nan, 'seven']}
        )
    ]
)
def test_na_values_keep_default(all_parsers: Any, kwargs: Dict[str, Any], expected: Any, request: Any, using_infer_string: bool) -> None:
    data: str = 'A,B,C\na,1,one\nb,2,two\n,3,three\nd,4,nan\ne,5,five\nnan,6,\ng,7,seven\n'
    parser: Any = all_parsers
    if parser.engine == 'pyarrow':
        if 'na_values' in kwargs and isinstance(kwargs['na_values'], dict):
            msg: str = "The pyarrow engine doesn't support passing a dict for na_values"
            with pytest.raises(ValueError, match=msg):
                parser.read_csv(StringIO(data), **kwargs)
            return
        if not using_infer_string or 'na_values' in kwargs:
            mark = pytest.mark.xfail()
            request.applymarker(mark)
    result: DataFrame = parser.read_csv(StringIO(data), **kwargs)
    expected = DataFrame(expected)
    tm.assert_frame_equal(result, expected)

def test_no_na_values_no_keep_default(all_parsers: Any) -> None:
    data: str = 'A,B,C\na,1,None\nb,2,two\n,3,None\nd,4,nan\ne,5,five\nnan,6,\ng,7,seven\n'
    parser: Any = all_parsers
    result: DataFrame = parser.read_csv(StringIO(data), keep_default_na=False)
    expected: DataFrame = DataFrame({'A': ['a', 'b', '', 'd', 'e', 'nan', 'g'], 'B': [1, 2, 3, 4, 5, 6, 7], 'C': ['None', 'two', 'None', 'nan', 'five', '', 'seven']})
    tm.assert_frame_equal(result, expected)

def test_no_keep_default_na_dict_na_values(all_parsers: Any) -> None:
    data: str = 'a,b\n,2'
    parser: Any = all_parsers
    if parser.engine == 'pyarrow':
        msg: str = "The pyarrow engine doesn't support passing a dict for na_values"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), na_values={'b': ['2']}, keep_default_na=False)
        return
    result: DataFrame = parser.read_csv(StringIO(data), na_values={'b': ['2']}, keep_default_na=False)
    expected: DataFrame = DataFrame({'a': [''], 'b': [np.nan]})
    tm.assert_frame_equal(result, expected)

def test_no_keep_default_na_dict_na_scalar_values(all_parsers: Any) -> None:
    data: str = 'a,b\n1,2'
    parser: Any = all_parsers
    if parser.engine == 'pyarrow':
        msg: str = "The pyarrow engine doesn't support passing a dict for na_values"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), na_values={'b': 2}, keep_default_na=False)
        return
    df: DataFrame = parser.read_csv(StringIO(data), na_values={'b': 2}, keep_default_na=False)
    expected: DataFrame = DataFrame({'a': [1], 'b': [np.nan]})
    tm.assert_frame_equal(df, expected)

@pytest.mark.parametrize('col_zero_na_values', [113125, '113125'])
def test_no_keep_default_na_dict_na_values_diff_reprs(all_parsers: Any, col_zero_na_values: Union[int, str]) -> None:
    data: str = '113125,"blah","/blaha",kjsdkj,412.166,225.874,214.008\n729639,"qwer","",asdfkj,466.681,,252.373\n'
    parser: Any = all_parsers
    expected: DataFrame = DataFrame({
        0: [np.nan, 729639.0],
        1: [np.nan, 'qwer'],
        2: ['/blaha', np.nan],
        3: ['kjsdkj', 'asdfkj'],
        4: [412.166, 466.681],
        5: ['225.874', ''],
        6: [np.nan, 252.373]
    })
    if parser.engine == 'pyarrow':
        msg: str = "The pyarrow engine doesn't support passing a dict for na_values"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), header=None, keep_default_na=False, na_values={2: '', 6: '214.008', 1: 'blah', 0: col_zero_na_values})
        return
    result: DataFrame = parser.read_csv(StringIO(data), header=None, keep_default_na=False, na_values={2: '', 6: '214.008', 1: 'blah', 0: col_zero_na_values})
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('na_filter,row_data', [(True, [[1, 'A'], [np.nan, np.nan], [3, 'C']]), (False, [['1', 'A'], ['nan', 'B'], ['3', 'C']])])
def test_na_values_na_filter_override(request: Any, all_parsers: Any, na_filter: bool, row_data: List[List[Any]], using_infer_string: bool) -> None:
    parser: Any = all_parsers
    if parser.engine == 'pyarrow':
        if not (using_infer_string and na_filter):
            mark = pytest.mark.xfail(reason="pyarrow doesn't support this.")
            request.applymarker(mark)
    data: str = 'A,B\n1,A\nnan,B\n3,C\n'
    result: DataFrame = parser.read_csv(StringIO(data), na_values=['B'], na_filter=na_filter)
    expected: DataFrame = DataFrame(row_data, columns=['A', 'B'])
    tm.assert_frame_equal(result, expected)

@skip_pyarrow
def test_na_trailing_columns(all_parsers: Any) -> None:
    parser: Any = all_parsers
    data: str = 'Date,Currency,Symbol,Type,Units,UnitPrice,Cost,Tax\n2012-03-14,USD,AAPL,BUY,1000\n2012-05-12,USD,SBUX,SELL,500'
    result: DataFrame = parser.read_csv(StringIO(data))
    expected: DataFrame = DataFrame([
        ['2012-03-14', 'USD', 'AAPL', 'BUY', 1000, np.nan, np.nan, np.nan],
        ['2012-05-12', 'USD', 'SBUX', 'SELL', 500, np.nan, np.nan, np.nan]
    ], columns=['Date', 'Currency', 'Symbol', 'Type', 'Units', 'UnitPrice', 'Cost', 'Tax'])
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('na_values,row_data', [(1, [[np.nan, 2.0], [2.0, np.nan]]), ({'a': 2, 'b': 1}, [[1.0, 2.0], [np.nan, np.nan]])])
def test_na_values_scalar(all_parsers: Any, na_values: Any, row_data: List[List[Any]]) -> None:
    parser: Any = all_parsers
    names: List[str] = ['a', 'b']
    data: str = '1,2\n2,1'
    if parser.engine == 'pyarrow' and isinstance(na_values, dict):
        err = ValueError
        msg: str = "The pyarrow engine doesn't support passing a dict for na_values"
        with pytest.raises(err, match=msg):
            parser.read_csv(StringIO(data), names=names, na_values=na_values)
        return
    elif parser.engine == 'pyarrow':
        msg = "The 'pyarrow' engine requires all na_values to be strings"
        with pytest.raises(TypeError, match=msg):
            parser.read_csv(StringIO(data), names=names, na_values=na_values)
        return
    result: DataFrame = parser.read_csv(StringIO(data), names=names, na_values=na_values)
    expected: DataFrame = DataFrame(row_data, columns=names)
    tm.assert_frame_equal(result, expected)

def test_na_values_dict_aliasing(all_parsers: Any) -> None:
    parser: Any = all_parsers
    na_values: Dict[str, Any] = {'a': 2, 'b': 1}
    na_values_copy: Dict[str, Any] = na_values.copy()
    names: List[str] = ['a', 'b']
    data: str = '1,2\n2,1'
    expected: DataFrame = DataFrame([[1.0, 2.0], [np.nan, np.nan]], columns=names)
    if parser.engine == 'pyarrow':
        msg: str = "The pyarrow engine doesn't support passing a dict for na_values"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), names=names, na_values=na_values)
        return
    result: DataFrame = parser.read_csv(StringIO(data), names=names, na_values=na_values)
    tm.assert_frame_equal(result, expected)
    tm.assert_dict_equal(na_values, na_values_copy)

def test_na_values_dict_null_column_name(all_parsers: Any) -> None:
    parser: Any = all_parsers
    data: str = ',x,y\n\nMA,1,2\nNA,2,1\nOA,,3'
    names: List[Any] = [None, 'x', 'y']
    na_values: Dict[Any, Any] = {name: STR_NA_VALUES for name in names}
    dtype: Dict[Any, str] = {None: 'object', 'x': 'float64', 'y': 'float64'}
    if parser.engine == 'pyarrow':
        msg: str = "The pyarrow engine doesn't support passing a dict for na_values"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), index_col=0, header=0, dtype=dtype, names=names, na_values=na_values, keep_default_na=False)
        return
    expected: DataFrame = DataFrame({'x': [1.0, 2.0, np.nan], 'y': [2.0, 1.0, 3.0]}, index=Index(['MA', 'NA', 'OA'], dtype=object))
    result: DataFrame = parser.read_csv(StringIO(data), index_col=0, header=0, dtype=dtype, names=names, na_values=na_values, keep_default_na=False)
    tm.assert_frame_equal(result, expected)

def test_na_values_dict_col_index(all_parsers: Any) -> None:
    data: str = 'a\nfoo\n1'
    parser: Any = all_parsers
    na_values: Dict[int, Any] = {0: 'foo'}
    if parser.engine == 'pyarrow':
        msg: str = "The pyarrow engine doesn't support passing a dict for na_values"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), na_values=na_values)
        return
    result: DataFrame = parser.read_csv(StringIO(data), na_values=na_values)
    expected: DataFrame = DataFrame({'a': [np.nan, 1]})
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize(
    'data,kwargs,expected',
    [
        (str(2 ** 63) + '\n' + str(2 ** 63 + 1), {'na_values': [2 ** 63]}, [str(2 ** 63), str(2 ** 63 + 1)]),
        (str(2 ** 63) + ',1' + '\n,2', {}, [[str(2 ** 63), 1], ['', 2]]),
        (str(2 ** 63) + '\n1', {'na_values': [2 ** 63]}, [np.nan, 1])
    ]
)
def test_na_values_uint64(all_parsers: Any, data: str, kwargs: Dict[str, Any], expected: Any, request: Any) -> None:
    parser: Any = all_parsers
    if parser.engine == 'pyarrow' and 'na_values' in kwargs:
        msg: str = "The 'pyarrow' engine requires all na_values to be strings"
        with pytest.raises(TypeError, match=msg):
            parser.read_csv(StringIO(data), header=None, **kwargs)
        return
    elif parser.engine == 'pyarrow':
        mark = pytest.mark.xfail(reason='Returns float64 instead of object')
        request.applymarker(mark)
    result: DataFrame = parser.read_csv(StringIO(data), header=None, **kwargs)
    expected_df: DataFrame = DataFrame(expected)
    tm.assert_frame_equal(result, expected_df)

def test_empty_na_values_no_default_with_index(all_parsers: Any) -> None:
    data: str = 'a,1\nb,2'
    parser: Any = all_parsers
    expected: DataFrame = DataFrame({'1': [2]}, index=Index(['b'], name='a'))
    result: DataFrame = parser.read_csv(StringIO(data), index_col=0, keep_default_na=False)
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('na_filter,index_data', [(False, ['', '5']), (True, [np.nan, 5.0])])
def test_no_na_filter_on_index(all_parsers: Any, na_filter: bool, index_data: List[Any], request: Any) -> None:
    parser: Any = all_parsers
    data: str = 'a,b,c\n1,,3\n4,5,6'
    if parser.engine == 'pyarrow' and na_filter is False:
        mark = pytest.mark.xfail(reason='mismatched index result')
        request.applymarker(mark)
    expected: DataFrame = DataFrame({'a': [1, 4], 'c': [3, 6]}, index=Index(index_data, name='b'))
    result: DataFrame = parser.read_csv(StringIO(data), index_col=[1], na_filter=na_filter)
    tm.assert_frame_equal(result, expected)

def test_inf_na_values_with_int_index(all_parsers: Any) -> None:
    parser: Any = all_parsers
    data: str = 'idx,col1,col2\n1,3,4\n2,inf,-inf'
    out: DataFrame = parser.read_csv(StringIO(data), index_col=[0], na_values=['inf', '-inf'])
    expected: DataFrame = DataFrame({'col1': [3, np.nan], 'col2': [4, np.nan]}, index=Index([1, 2], name='idx'))
    tm.assert_frame_equal(out, expected)

@xfail_pyarrow
@pytest.mark.parametrize('na_filter', [True, False])
def test_na_values_with_dtype_str_and_na_filter(all_parsers: Any, na_filter: bool) -> None:
    parser: Any = all_parsers
    data: str = 'a,b,c\n1,,3\n4,5,6'
    empty: Any = np.nan if na_filter else ''
    expected: DataFrame = DataFrame({'a': ['1', '4'], 'b': [empty, '5'], 'c': ['3', '6']})
    result: DataFrame = parser.read_csv(StringIO(data), na_filter=na_filter, dtype=str)
    tm.assert_frame_equal(result, expected)

@xfail_pyarrow
@pytest.mark.parametrize('data, na_values', [
    ('false,1\n,1\ntrue', None),
    ('false,1\nnull,1\ntrue', None),
    ('false,1\nnan,1\ntrue', None),
    ('false,1\nfoo,1\ntrue', 'foo'),
    ('false,1\nfoo,1\ntrue', ['foo']),
    ('false,1\nfoo,1\ntrue', {'a': 'foo'})
])
def test_cast_NA_to_bool_raises_error(all_parsers: Any, data: str, na_values: Any) -> None:
    parser: Any = all_parsers
    msg: str = '|'.join(['Bool column has NA values in column [0a]', 'cannot safely convert passed user dtype of bool for object dtyped data in column 0'])
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), header=None, names=['a', 'b'], dtype={'a': 'bool'}, na_values=na_values)

@xfail_pyarrow
def test_str_nan_dropped(all_parsers: Any) -> None:
    parser: Any = all_parsers
    data: str = 'File: small.csv,,\n10010010233,0123,654\nfoo,,bar\n01001000155,4530,898'
    result: DataFrame = parser.read_csv(
        StringIO(data),
        header=None,
        names=['col1', 'col2', 'col3'],
        dtype={'col1': str, 'col2': str, 'col3': str}
    ).dropna()
    expected: DataFrame = DataFrame({'col1': ['10010010233', '01001000155'], 'col2': ['0123', '4530'], 'col3': ['654', '898']}, index=[1, 3])
    tm.assert_frame_equal(result, expected)

def test_nan_multi_index(all_parsers: Any) -> None:
    parser: Any = all_parsers
    data: str = 'A,B,B\nX,Y,Z\n1,2,inf'
    if parser.engine == 'pyarrow':
        msg: str = "The pyarrow engine doesn't support passing a dict for na_values"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), header=list(range(2)), na_values={('B', 'Z'): 'inf'})
        return
    result: DataFrame = parser.read_csv(StringIO(data), header=list(range(2)), na_values={('B', 'Z'): 'inf'})
    expected: DataFrame = DataFrame({('A', 'X'): [1], ('B', 'Y'): [2], ('B', 'Z'): [np.nan]})
    tm.assert_frame_equal(result, expected)

@xfail_pyarrow
def test_bool_and_nan_to_bool(all_parsers: Any) -> None:
    parser: Any = all_parsers
    data: str = '0\nNaN\nTrue\nFalse\n'
    with pytest.raises(ValueError, match='NA values'):
        parser.read_csv(StringIO(data), dtype='bool')

def test_bool_and_nan_to_int(all_parsers: Any) -> None:
    parser: Any = all_parsers
    data: str = '0\nNaN\nTrue\nFalse\n'
    with pytest.raises(ValueError, match='convert|NoneType'):
        parser.read_csv(StringIO(data), dtype='int')

def test_bool_and_nan_to_float(all_parsers: Any) -> None:
    parser: Any = all_parsers
    data: str = '0\nNaN\nTrue\nFalse\n'
    result: DataFrame = parser.read_csv(StringIO(data), dtype='float')
    expected: DataFrame = DataFrame.from_dict({'0': [np.nan, 1.0, 0.0]})
    tm.assert_frame_equal(result, expected)

@xfail_pyarrow
@pytest.mark.parametrize('na_values', [[-99.0, -99], [-99, -99.0]])
def test_na_values_dict_without_dtype(all_parsers: Any, na_values: Any, row_data: List[List[Any]] = [[np.nan], [np.nan], [np.nan], [np.nan]]) -> None:
    parser: Any = all_parsers
    data: str = 'A\n-99\n-99\n-99.0\n-99.0'
    result: DataFrame = parser.read_csv(StringIO(data), na_values=na_values)
    expected: DataFrame = DataFrame({'A': [np.nan, np.nan, np.nan, np.nan]})
    tm.assert_frame_equal(result, expected)

def test_na_values_dict_aliasing(all_parsers: Any) -> None:
    parser: Any = all_parsers
    na_values: Dict[str, Any] = {'a': 2, 'b': 1}
    na_values_copy: Dict[str, Any] = na_values.copy()
    names: List[str] = ['a', 'b']
    data: str = '1,2\n2,1'
    expected: DataFrame = DataFrame([[1.0, 2.0], [np.nan, np.nan]], columns=names)
    if parser.engine == 'pyarrow':
        msg: str = "The pyarrow engine doesn't support passing a dict for na_values"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), names=names, na_values=na_values)
        return
    result: DataFrame = parser.read_csv(StringIO(data), names=names, na_values=na_values)
    tm.assert_frame_equal(result, expected)
    tm.assert_dict_equal(na_values, na_values_copy)

def test_na_values_dict_null_column_name(all_parsers: Any) -> None:
    parser: Any = all_parsers
    data: str = ',x,y\n\nMA,1,2\nNA,2,1\nOA,,3'
    names: List[Any] = [None, 'x', 'y']
    na_values: Dict[Any, Any] = {name: STR_NA_VALUES for name in names}
    dtype: Dict[Any, str] = {None: 'object', 'x': 'float64', 'y': 'float64'}
    if parser.engine == 'pyarrow':
        msg: str = "The pyarrow engine doesn't support passing a dict for na_values"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), index_col=0, header=0, dtype=dtype, names=names, na_values=na_values, keep_default_na=False)
        return
    expected: DataFrame = DataFrame({'x': [1.0, 2.0, np.nan], 'y': [2.0, 1.0, 3.0]}, index=Index(['MA', 'NA', 'OA'], dtype=object))
    result: DataFrame = parser.read_csv(StringIO(data), index_col=0, header=0, dtype=dtype, names=names, na_values=na_values, keep_default_na=False)
    tm.assert_frame_equal(result, expected)

def test_na_values_dict_col_index(all_parsers: Any) -> None:
    data: str = 'a\nfoo\n1'
    parser: Any = all_parsers
    na_values: Dict[int, Any] = {0: 'foo'}
    if parser.engine == 'pyarrow':
        msg: str = "The pyarrow engine doesn't support passing a dict for na_values"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), na_values=na_values)
        return
    result: DataFrame = parser.read_csv(StringIO(data), na_values=na_values)
    expected: DataFrame = DataFrame({'a': [np.nan, 1]})
    tm.assert_frame_equal(result, expected)