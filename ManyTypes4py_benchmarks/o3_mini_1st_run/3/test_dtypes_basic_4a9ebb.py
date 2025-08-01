#!/usr/bin/env python3
"""
Tests dtype specification during parsing
for all of the parsers defined in parsers.py
"""
from collections import defaultdict
from io import StringIO
from typing import Any, DefaultDict, Tuple, Optional
import numpy as np
import pytest
from pandas.errors import ParserWarning
import pandas as pd
from pandas import DataFrame, Timestamp
import pandas._testing as tm
from pandas.core.arrays import IntegerArray

pytestmark = pytest.mark.filterwarnings('ignore:Passing a BlockManager to DataFrame:DeprecationWarning')
xfail_pyarrow = pytest.mark.usefixtures('pyarrow_xfail')


@pytest.mark.parametrize('dtype', [str, object])
@pytest.mark.parametrize('check_orig', [True, False])
@pytest.mark.usefixtures('pyarrow_xfail')
def test_dtype_all_columns(all_parsers: Any, dtype: Any, check_orig: bool, using_infer_string: bool) -> None:
    parser: Any = all_parsers
    df: DataFrame = DataFrame(np.random.default_rng(2).random((5, 2)).round(4),
                                columns=list('AB'),
                                index=['1A', '1B', '1C', '1D', '1E'])
    with tm.ensure_clean('__passing_str_as_dtype__.csv') as path:
        df.to_csv(path)
        result: DataFrame = parser.read_csv(path, dtype=dtype, index_col=0)
        if check_orig:
            expected: DataFrame = df.copy()
            result = result.astype(float)
        elif using_infer_string and dtype is str:
            expected = df.astype(str)
        else:
            expected = df.astype(str).astype(object)
        tm.assert_frame_equal(result, expected)


@pytest.mark.usefixtures('pyarrow_xfail')
def test_dtype_per_column(all_parsers: Any) -> None:
    parser: Any = all_parsers
    data: str = 'one,two\n1,2.5\n2,3.5\n3,4.5\n4,5.5'
    expected: DataFrame = DataFrame([[1, '2.5'], [2, '3.5'], [3, '4.5'], [4, '5.5']],
                                      columns=['one', 'two'])
    expected['one'] = expected['one'].astype(np.float64)
    result: DataFrame = parser.read_csv(StringIO(data), dtype={'one': np.float64, 1: str})
    tm.assert_frame_equal(result, expected)


def test_invalid_dtype_per_column(all_parsers: Any) -> None:
    parser: Any = all_parsers
    data: str = 'one,two\n1,2.5\n2,3.5\n3,4.5\n4,5.5'
    with pytest.raises(TypeError, match='data type ["\']foo["\'] not understood'):
        parser.read_csv(StringIO(data), dtype={'one': 'foo', 1: 'int'})


def test_raise_on_passed_int_dtype_with_nas(all_parsers: Any) -> None:
    parser: Any = all_parsers
    data: str = 'YEAR, DOY, a\n2001,106380451,10\n2001,,11\n2001,106380451,67'
    if parser.engine == 'c':
        msg: str = 'Integer column has NA values'
    elif parser.engine == 'pyarrow':
        msg = "The 'skipinitialspace' option is not supported with the 'pyarrow' engine"
    else:
        msg = 'Unable to convert column DOY'
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), dtype={'DOY': np.int64}, skipinitialspace=True)


def test_dtype_with_converters(all_parsers: Any) -> None:
    parser: Any = all_parsers
    data: str = 'a,b\n1.1,2.2\n1.2,2.3'
    if parser.engine == 'pyarrow':
        msg: str = "The 'converters' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), dtype={'a': 'i8'}, converters={'a': lambda x: str(x)})
        return
    result: DataFrame = parser.read_csv_check_warnings(ParserWarning,
                                                       'Both a converter and dtype were specified for column a - only the converter will be used.',
                                                       StringIO(data),
                                                       dtype={'a': 'i8'},
                                                       converters={'a': lambda x: str(x)})
    expected: DataFrame = DataFrame({'a': ['1.1', '1.2'], 'b': [2.2, 2.3]})
    tm.assert_frame_equal(result, expected)


def test_numeric_dtype(all_parsers: Any, any_real_numpy_dtype: Any) -> None:
    data: str = '0\n1'
    parser: Any = all_parsers
    expected: DataFrame = DataFrame([0, 1], dtype=any_real_numpy_dtype)
    result: DataFrame = parser.read_csv(StringIO(data), header=None, dtype=any_real_numpy_dtype)
    tm.assert_frame_equal(expected, result, check_column_type=False)


@pytest.mark.usefixtures('pyarrow_xfail')
def test_boolean_dtype(all_parsers: Any) -> None:
    parser: Any = all_parsers
    data: str = '\n'.join(['a', 'True', 'TRUE', 'true', '1', '1.0',
                           'False', 'FALSE', 'false', '0', '0.0', 'NaN', 'nan', 'NA', 'null', 'NULL'])
    result: DataFrame = parser.read_csv(StringIO(data), dtype='boolean')
    expected: DataFrame = DataFrame({'a': pd.array([True, True, True, True, True,
                                                      False, False, False, False, False,
                                                      None, None, None, None, None], dtype='boolean')})
    tm.assert_frame_equal(result, expected)


@pytest.mark.usefixtures('pyarrow_xfail')
def test_delimiter_with_usecols_and_parse_dates(all_parsers: Any) -> None:
    result: DataFrame = all_parsers.read_csv(StringIO('"dump","-9,1","-9,1",20101010'),
                                              engine='python',
                                              names=['col', 'col1', 'col2', 'col3'],
                                              usecols=['col1', 'col2', 'col3'],
                                              parse_dates=['col3'],
                                              decimal=',')
    expected: DataFrame = DataFrame({'col1': [-9.1], 'col2': [-9.1], 'col3': [Timestamp('2010-10-10')]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize('thousands', ['_', None])
def test_decimal_and_exponential(request: Any, python_parser_only: Any, numeric_decimal: Tuple[str, Any], thousands: Optional[str]) -> None:
    decimal_number_check(request, python_parser_only, numeric_decimal, thousands, None)


@pytest.mark.parametrize('thousands', ['_', None])
@pytest.mark.parametrize('float_precision', [None, 'legacy', 'high', 'round_trip'])
def test_1000_sep_decimal_float_precision(request: Any, c_parser_only: Any, numeric_decimal: Tuple[str, Any],
                                            float_precision: Optional[str], thousands: Optional[str]) -> None:
    decimal_number_check(request, c_parser_only, numeric_decimal, thousands, float_precision)
    text: str = numeric_decimal[0]
    value: Any = numeric_decimal[1]
    text = ' ' + text + ' '
    if isinstance(value, str):
        value = ' ' + value + ' '
    decimal_number_check(request, c_parser_only, (text, value), thousands, float_precision)


def decimal_number_check(request: Any, parser: Any, numeric_decimal: Tuple[str, Any],
                         thousands: Optional[str], float_precision: Optional[str]) -> None:
    value: str = numeric_decimal[0]
    if thousands is None and value in ('1_,', '1_234,56', '1_234,56e0'):
        request.applymarker(pytest.mark.xfail(reason=f'thousands={thousands} and sep is in {value}'))
    df: DataFrame = parser.read_csv(StringIO(value), float_precision=float_precision, sep='|',
                                     thousands=thousands, decimal=',', header=None)
    val: Any = df.iloc[0, 0]
    assert val == numeric_decimal[1]


@pytest.mark.parametrize('float_precision', [None, 'legacy', 'high', 'round_trip'])
def test_skip_whitespace(c_parser_only: Any, float_precision: Optional[str]) -> None:
    DATA: str = 'id\tnum\t\n1\t1.2 \t\n1\t 2.1\t\n2\t 1\t\n2\t 1.2 \t\n'
    df: DataFrame = c_parser_only.read_csv(StringIO(DATA), float_precision=float_precision, sep='\t', header=0, dtype={1: np.float64})
    tm.assert_series_equal(df.iloc[:, 1], pd.Series([1.2, 2.1, 1.0, 1.2], name='num'))


@pytest.mark.usefixtures('pyarrow_xfail')
def test_true_values_cast_to_bool(all_parsers: Any) -> None:
    text: str = 'a,b\nyes,xxx\nno,yyy\n1,zzz\n0,aaa\n    '
    parser: Any = all_parsers
    result: DataFrame = parser.read_csv(StringIO(text), true_values=['yes'], false_values=['no'], dtype={'a': 'boolean'})
    expected: DataFrame = DataFrame({'a': [True, False, True, False], 'b': ['xxx', 'yyy', 'zzz', 'aaa']})
    expected['a'] = expected['a'].astype('boolean')
    tm.assert_frame_equal(result, expected)


@pytest.mark.usefixtures('pyarrow_xfail')
@pytest.mark.parametrize('dtypes, exp_value', [({}, '1'), ({'a.1': 'int64'}, 1)])
def test_dtype_mangle_dup_cols(all_parsers: Any, dtypes: dict, exp_value: Any) -> None:
    parser: Any = all_parsers
    data: str = 'a,a\n1,1'
    dtype_dict: dict = {'a': str, **dtypes}
    dtype_dict_copy: dict = dtype_dict.copy()
    result: DataFrame = parser.read_csv(StringIO(data), dtype=dtype_dict)
    expected: DataFrame = DataFrame({'a': ['1'], 'a.1': [exp_value]})
    assert dtype_dict == dtype_dict_copy, 'dtype dict changed'
    tm.assert_frame_equal(result, expected)


@pytest.mark.usefixtures('pyarrow_xfail')
def test_dtype_mangle_dup_cols_single_dtype(all_parsers: Any) -> None:
    parser: Any = all_parsers
    data: str = 'a,a\n1,1'
    result: DataFrame = parser.read_csv(StringIO(data), dtype=str)
    expected: DataFrame = DataFrame({'a': ['1'], 'a.1': ['1']})
    tm.assert_frame_equal(result, expected)


@pytest.mark.usefixtures('pyarrow_xfail')
def test_dtype_multi_index(all_parsers: Any) -> None:
    parser: Any = all_parsers
    data: str = 'A,B,B\nX,Y,Z\n1,2,3'
    result: DataFrame = parser.read_csv(StringIO(data),
                                          header=list(range(2)),
                                          dtype={('A', 'X'): np.int32,
                                                 ('B', 'Y'): np.int32,
                                                 ('B', 'Z'): np.float32})
    expected: DataFrame = DataFrame({('A', 'X'): np.int32([1]),
                                     ('B', 'Y'): np.int32([2]),
                                     ('B', 'Z'): np.float32([3])})
    tm.assert_frame_equal(result, expected)


def test_nullable_int_dtype(all_parsers: Any, any_int_ea_dtype: Any) -> None:
    parser: Any = all_parsers
    dtype: Any = any_int_ea_dtype
    data: str = 'a,b,c\n,3,5\n1,,6\n2,4,'
    expected: DataFrame = DataFrame({'a': pd.array([pd.NA, 1, 2], dtype=dtype),
                                     'b': pd.array([3, pd.NA, 4], dtype=dtype),
                                     'c': pd.array([5, 6, pd.NA], dtype=dtype)})
    actual: DataFrame = parser.read_csv(StringIO(data), dtype=dtype)
    tm.assert_frame_equal(actual, expected)


@pytest.mark.usefixtures('pyarrow_xfail')
@pytest.mark.parametrize('default', ['float', 'float64'])
def test_dtypes_defaultdict(all_parsers: Any, default: str) -> None:
    data: str = 'a,b\n1,2\n'
    dtype: DefaultDict[str, Any] = defaultdict(lambda: default, a='int64')
    parser: Any = all_parsers
    result: DataFrame = parser.read_csv(StringIO(data), dtype=dtype)
    expected: DataFrame = DataFrame({'a': [1], 'b': 2.0})
    tm.assert_frame_equal(result, expected)


@pytest.mark.usefixtures('pyarrow_xfail')
def test_dtypes_defaultdict_mangle_dup_cols(all_parsers: Any) -> None:
    data: str = 'a,b,a,b,b.1\n1,2,3,4,5\n'
    dtype: DefaultDict[str, Any] = defaultdict(lambda: 'float64', a='int64')
    dtype['b.1'] = 'int64'
    parser: Any = all_parsers
    result: DataFrame = parser.read_csv(StringIO(data), dtype=dtype)
    expected: DataFrame = DataFrame({'a': [1], 'b': [2.0], 'a.1': [3], 'b.2': [4.0], 'b.1': [5]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.usefixtures('pyarrow_xfail')
def test_dtypes_defaultdict_invalid(all_parsers: Any) -> None:
    data: str = 'a,b\n1,2\n'
    dtype: DefaultDict[str, Any] = defaultdict(lambda: 'invalid_dtype', a='int64')
    parser: Any = all_parsers
    with pytest.raises(TypeError, match='not understood'):
        parser.read_csv(StringIO(data), dtype=dtype)


def test_dtype_backend(all_parsers: Any) -> None:
    parser: Any = all_parsers
    data: str = 'a,b,c,d,e,f,g,h,i,j\n1,2.5,True,a,,,,,12-31-2019,\n3,4.5,False,b,6,7.5,True,a,12-31-2019,\n'
    result: DataFrame = parser.read_csv(StringIO(data), dtype_backend='numpy_nullable', parse_dates=['i'])
    expected: DataFrame = DataFrame({
        'a': pd.Series([1, 3], dtype='Int64'),
        'b': pd.Series([2.5, 4.5], dtype='Float64'),
        'c': pd.Series([True, False], dtype='boolean'),
        'd': pd.Series(['a', 'b'], dtype='string'),
        'e': pd.Series([pd.NA, 6], dtype='Int64'),
        'f': pd.Series([pd.NA, 7.5], dtype='Float64'),
        'g': pd.Series([pd.NA, True], dtype='boolean'),
        'h': pd.Series([pd.NA, 'a'], dtype='string'),
        'i': pd.Series([Timestamp('2019-12-31')] * 2),
        'j': pd.Series([pd.NA, pd.NA], dtype='Int64')
    })
    tm.assert_frame_equal(result, expected)


def test_dtype_backend_and_dtype(all_parsers: Any) -> None:
    parser: Any = all_parsers
    data: str = 'a,b\n1,2.5\n,\n'
    result: DataFrame = parser.read_csv(StringIO(data), dtype_backend='numpy_nullable', dtype='float64')
    expected: DataFrame = DataFrame({'a': [1.0, np.nan], 'b': [2.5, np.nan]})
    tm.assert_frame_equal(result, expected)


def test_dtype_backend_string(all_parsers: Any, string_storage: str) -> None:
    with pd.option_context('mode.string_storage', string_storage):
        parser: Any = all_parsers
        data: str = 'a,b\na,x\nb,\n'
        result: DataFrame = parser.read_csv(StringIO(data), dtype_backend='numpy_nullable')
        expected: DataFrame = DataFrame({
            'a': pd.array(['a', 'b'], dtype=pd.StringDtype(string_storage)),
            'b': pd.array(['x', pd.NA], dtype=pd.StringDtype(string_storage))
        })
    tm.assert_frame_equal(result, expected)


def test_dtype_backend_ea_dtype_specified(all_parsers: Any) -> None:
    data: str = 'a,b\n1,2\n'
    parser: Any = all_parsers
    result: DataFrame = parser.read_csv(StringIO(data), dtype='Int64', dtype_backend='numpy_nullable')
    expected: DataFrame = DataFrame({'a': [1], 'b': 2}, dtype='Int64')
    tm.assert_frame_equal(result, expected)


def test_dtype_backend_pyarrow(all_parsers: Any, request: Any) -> None:
    pa = pytest.importorskip('pyarrow')
    parser: Any = all_parsers
    data: str = 'a,b,c,d,e,f,g,h,i,j\n1,2.5,True,a,,,,,12-31-2019,\n3,4.5,False,b,6,7.5,True,a,12-31-2019,\n'
    result: DataFrame = parser.read_csv(StringIO(data), dtype_backend='pyarrow', parse_dates=['i'])
    expected: DataFrame = DataFrame({
        'a': pd.Series([1, 3], dtype='int64[pyarrow]'),
        'b': pd.Series([2.5, 4.5], dtype='float64[pyarrow]'),
        'c': pd.Series([True, False], dtype='bool[pyarrow]'),
        'd': pd.Series(['a', 'b'], dtype=pd.ArrowDtype(pa.string())),
        'e': pd.Series([pd.NA, 6], dtype='int64[pyarrow]'),
        'f': pd.Series([pd.NA, 7.5], dtype='float64[pyarrow]'),
        'g': pd.Series([pd.NA, True], dtype='bool[pyarrow]'),
        'h': pd.Series([pd.NA, 'a'], dtype=pd.ArrowDtype(pa.string())),
        'i': pd.Series([Timestamp('2019-12-31')] * 2),
        'j': pd.Series([pd.NA, pd.NA], dtype='null[pyarrow]')
    })
    tm.assert_frame_equal(result, expected)


@pytest.mark.usefixtures('pyarrow_xfail')
def test_ea_int_avoid_overflow(all_parsers: Any) -> None:
    parser: Any = all_parsers
    data: str = 'a,b\n1,1\n,1\n1582218195625938945,1\n'
    result: DataFrame = parser.read_csv(StringIO(data), dtype={'a': 'Int64'})
    expected: DataFrame = DataFrame({'a': IntegerArray(np.array([1, 1, 1582218195625938945]),
                                                     np.array([False, True, False])),
                                     'b': 1})
    tm.assert_frame_equal(result, expected)


def test_string_inference(all_parsers: Any) -> None:
    dtype_obj: pd.StringDtype = pd.StringDtype(na_value=np.nan)
    data: str = 'a,b\nx,1\ny,2\n,3'
    parser: Any = all_parsers
    with pd.option_context('future.infer_string', True):
        result: DataFrame = parser.read_csv(StringIO(data))
    expected: DataFrame = DataFrame({'a': pd.Series(['x', 'y', None], dtype=dtype_obj),
                                     'b': [1, 2, 3]},
                                    columns=pd.Index(['a', 'b'], dtype=dtype_obj))
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize('dtype', ['O', object, 'object', np.object_, str, np.str_])
def test_string_inference_object_dtype(all_parsers: Any, dtype: Any, using_infer_string: bool) -> None:
    data: str = 'a,b\nx,a\ny,a\nz,a'
    parser: Any = all_parsers
    with pd.option_context('future.infer_string', True):
        result: DataFrame = parser.read_csv(StringIO(data), dtype=dtype)
    expected_dtype: Any = pd.StringDtype(na_value=np.nan) if dtype is str else object
    expected: DataFrame = DataFrame({'a': pd.Series(['x', 'y', 'z'], dtype=expected_dtype),
                                     'b': pd.Series(['a', 'a', 'a'], dtype=expected_dtype)},
                                    columns=pd.Index(['a', 'b'], dtype=pd.StringDtype(na_value=np.nan)))
    tm.assert_frame_equal(result, expected)
    with pd.option_context('future.infer_string', True):
        result: DataFrame = parser.read_csv(StringIO(data), dtype={'a': dtype})
    expected = DataFrame({'a': pd.Series(['x', 'y', 'z'], dtype=expected_dtype),
                          'b': pd.Series(['a', 'a', 'a'], dtype=pd.StringDtype(na_value=np.nan))},
                         columns=pd.Index(['a', 'b'], dtype=pd.StringDtype(na_value=np.nan)))
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow
def test_accurate_parsing_of_large_integers(all_parsers: Any) -> None:
    data: str = ('SYMBOL,MOMENT,ID,ID_DEAL\n'
                 'AAPL,20230301181139587,1925036343869802844,\n'
                 'AAPL,20230301181139587,2023552585717889863,2023552585717263358\n'
                 'NVDA,20230301181139587,2023552585717889863,2023552585717263359\n'
                 'AMC,20230301181139587,2023552585717889863,2023552585717263360\n'
                 'AMZN,20230301181139587,2023552585717889759,2023552585717263360\n'
                 'MSFT,20230301181139587,2023552585717889863,2023552585717263361\n'
                 'NVDA,20230301181139587,2023552585717889827,2023552585717263361')
    orders: DataFrame = all_parsers.read_csv(StringIO(data), dtype={'ID_DEAL': pd.Int64Dtype()})
    assert len(orders.loc[orders['ID_DEAL'] == 2023552585717263358, 'ID_DEAL']) == 1
    assert len(orders.loc[orders['ID_DEAL'] == 2023552585717263359, 'ID_DEAL']) == 1
    assert len(orders.loc[orders['ID_DEAL'] == 2023552585717263360, 'ID_DEAL']) == 2
    assert len(orders.loc[orders['ID_DEAL'] == 2023552585717263361, 'ID_DEAL']) == 2


def test_dtypes_with_usecols(all_parsers: Any) -> None:
    parser: Any = all_parsers
    data: str = 'a,b,c\n1,2,3\n4,5,6'
    result: DataFrame = parser.read_csv(StringIO(data), usecols=['a', 'c'], dtype={'a': object})
    if parser.engine == 'pyarrow':
        values = [1, 4]
    else:
        values = ['1', '4']
    expected: DataFrame = DataFrame({'a': pd.Series(values, dtype=object), 'c': [3, 6]})
    tm.assert_frame_equal(result, expected)


def test_index_col_with_dtype_no_rangeindex(all_parsers: Any) -> None:
    data: StringIO = StringIO('345.5,519.5,0\n519.5,726.5,1')
    result_index: Any = all_parsers.read_csv(data, header=None, names=['start', 'stop', 'bin_id'],
                                               dtype={'start': np.float32, 'stop': np.float32, 'bin_id': np.uint32},
                                               index_col='bin_id').index
    expected_index: Any = pd.Index([0, 1], dtype=np.uint32, name='bin_id')
    tm.assert_index_equal(result_index, expected_index)