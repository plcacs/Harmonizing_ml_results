from datetime import datetime, timedelta, timezone
from io import StringIO
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame, DatetimeIndex, Index, MultiIndex, Series, Timestamp
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
from pandas.core.tools.datetimes import start_caching_at
from pandas.io.parsers import read_csv

def test_date_col_as_index_col(all_parsers: pd.io.parsers.ParserBase):
    data: str = 'KORD,19990127 19:00:00, 18:56:00, 0.8100, 2.8100, 7.2000, 0.0000, 280.0000\nKORD,19990127 20:00:00, 19:56:00, 0.0100, 2.2100, 7.2000, 0.0000, 260.0000\nKORD,19990127 21:00:00, 20:56:00, -0.5900, 2.2100, 5.7000, 0.0000, 280.0000\nKORD,19990127 21:00:00, 21:18:00, -0.9900, 2.0100, 3.6000, 0.0000, 270.0000\nKORD,19990127 22:00:00, 21:56:00, -0.5900, 1.7100, 5.1000, 0.0000, 290.0000\n'
    parser: pd.io.parsers.ParserBase = all_parsers
    kwds: dict = {'header': None, 'parse_dates': [1], 'index_col': 1, 'names': ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7']}
    result: DataFrame = parser.read_csv(StringIO(data), **kwds)
    index: Index = Index([datetime(1999, 1, 27, 19, 0), datetime(1999, 1, 27, 20, 0), datetime(1999, 1, 27, 21, 0), datetime(1999, 1, 27, 21, 0), datetime(1999, 1, 27, 22, 0)], dtype='M8[s]', name='X1')
    expected: DataFrame = DataFrame([['KORD', ' 18:56:00', 0.81, 2.81, 7.2, 0.0, 280.0], ['KORD', ' 19:56:00', 0.01, 2.21, 7.2, 0.0, 260.0], ['KORD', ' 20:56:00', -0.59, 2.21, 5.7, 0.0, 280.0], ['KORD', ' 21:18:00', -0.99, 2.01, 3.6, 0.0, 270.0], ['KORD', ' 21:56:00', -0.59, 1.71, 5.1, 0.0, 290.0]], columns=['X0', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7'], index=index)
    if parser.engine == 'pyarrow':
        expected['X2'] = pd.to_datetime('1970-01-01' + expected['X2']).dt.time
    tm.assert_frame_equal(result, expected)

@xfail_pyarrow
def test_nat_parse(all_parsers: pd.io.parsers.ParserBase):
    parser: pd.io.parsers.ParserBase = all_parsers
    df: DataFrame = DataFrame({'A': np.arange(10, dtype='float64'), 'B': Timestamp('20010101')})
    df.iloc[3:6, :] = np.nan
    with tm.ensure_clean('__nat_parse_.csv') as path:
        df.to_csv(path)
        result: DataFrame = parser.read_csv(path, index_col=0, parse_dates=['B'])
        tm.assert_frame_equal(result, df)

@skip_pyarrow
def test_parse_dates_implicit_first_col(all_parsers: pd.io.parsers.ParserBase):
    data: str = 'A,B,C\n20090101,a,1,2\n20090102,b,3,4\n20090103,c,4,5\n'
    parser: pd.io.parsers.ParserBase = all_parsers
    result: DataFrame = parser.read_csv(StringIO(data), parse_dates=True)
    expected: DataFrame = parser.read_csv(StringIO(data), index_col=0, parse_dates=True)
    tm.assert_frame_equal(result, expected)

@xfail_pyarrow
def test_parse_dates_string(all_parsers: pd.io.parsers.ParserBase):
    data: str = 'date,A,B,C\n20090101,a,1,2\n20090102,b,3,4\n20090103,c,4,5\n'
    parser: pd.io.parsers.ParserBase = all_parsers
    result: DataFrame = parser.read_csv(StringIO(data), index_col='date', parse_dates=['date'])
    index: DatetimeIndex = date_range('1/1/2009', periods=3, name='date', unit='s')._with_freq(None)
    expected: DataFrame = DataFrame({'A': ['a', 'b', 'c'], 'B': [1, 3, 4], 'C': [2, 4, 5]}, index=index)
    tm.assert_frame_equal(result, expected)

@xfail_pyarrow
def test_parse_dates_column_list(all_parsers: pd.io.parsers.ParserBase, parse_dates):
    data: str = 'a,b,c\n01/01/2010,1,15/02/2010'
    parser: pd.io.parsers.ParserBase = all_parsers
    expected: DataFrame = DataFrame({'a': [datetime(2010, 1, 1)], 'b': [1], 'c': [datetime(2010, 2, 15)])
    expected['a'] = expected['a'].astype('M8[s]')
    expected['c'] = expected['c'].astype('M8[s]')
    expected: DataFrame = expected.set_index(['a', 'b'])
    result: DataFrame = parser.read_csv(StringIO(data), index_col=[0, 1], parse_dates=parse_dates, dayfirst=True)
    tm.assert_frame_equal(result, expected)

@xfail_pyarrow
def test_multi_index_parse_dates(all_parsers: pd.io.parsers.ParserBase, index_col):
    data: str = 'index1,index2,A,B,C\n20090101,one,a,1,2\n20090101,two,b,3,4\n20090101,three,c,4,5\n20090102,one,a,1,2\n20090102,two,b,3,4\n20090102,three,c,4,5\n20090103,one,a,1,2\n20090103,two,b,3,4\n20090103,three,c,4,5\n'
    parser: pd.io.parsers.ParserBase = all_parsers
    dti: DatetimeIndex = date_range('2009-01-01', periods=3, freq='D', unit='s')
    index: MultiIndex = MultiIndex.from_product([dti, ('one', 'two', 'three')], names=['index1', 'index2'])
    if index_col == [1, 0]:
        index: MultiIndex = index.swaplevel(0, 1)
    expected: DataFrame = DataFrame([['a', 1, 2], ['b', 3, 4], ['c', 4, 5], ['a', 1, 2], ['b', 3, 4], ['c', 4, 5], ['a', 1, 2], ['b', 3, 4], ['c', 4, 5]], columns=['A', 'B', 'C'], index=index)
    result: DataFrame = parser.read_csv_check_warnings(UserWarning, 'Could not infer format', StringIO(data), index_col=index_col, parse_dates=True)
    tm.assert_frame_equal(result, expected)

def test_parse_tz_aware(all_parsers: pd.io.parsers.ParserBase):
    parser: pd.io.parsers.ParserBase = all_parsers
    data: str = 'Date,x\n2012-06-13T01:39:00Z,0.5'
    result: DataFrame = parser.read_csv(StringIO(data), index_col=0, parse_dates=True)
    expected_data: dict = {'x': [0.5]}
    expected: DataFrame = DataFrame(expected_data, index=Index([Timestamp('2012-06-13 01:39:00+00:00')], name='Date'))
    if parser.engine == 'pyarrow':
        pytz = pytest.importorskip('pytz')
        expected_tz = pytz.utc
    else:
        expected_tz = timezone.utc
    tm.assert_frame_equal(result, expected)
    assert result.index.tz is expected_tz

@pytest.mark.parametrize('kwargs', [{}, {'index_col': 'C'}])
def test_read_with_parse_dates_scalar_non_bool(all_parsers: pd.io.parsers.ParserBase, kwargs):
    parser: pd.io.parsers.ParserBase = all_parsers
    msg: str = "Only booleans and lists are accepted for the 'parse_dates' parameter"
    data: str = 'A,B,C\n    1,2,2003-11-1'
    with pytest.raises(TypeError, match=msg):
        parser.read_csv(StringIO(data), parse_dates='C', **kwargs)

@pytest.mark.parametrize('parse_dates', [(1,), np.array([4, 5]), {1, 3}])
def test_read_with_parse_dates_invalid_type(all_parsers: pd.io.parsers.ParserBase, parse_dates):
    parser: pd.io.parsers.ParserBase = all_parsers
    msg: str = "Only booleans and lists are accepted for the 'parse_dates' parameter"
    data: str = 'A,B,C\n    1,2,2003-11-1'
    with pytest.raises(TypeError, match=msg):
        parser.read_csv(StringIO(data), parse_dates=parse_dates)

@pytest.mark.parametrize('value', ['nan', ''])
def test_bad_date_parse(all_parsers: pd.io.parsers.ParserBase, cache, value):
    parser: pd.io.parsers.ParserBase = all_parsers
    s: StringIO = StringIO(f'{value},\n' * (start_caching_at + 1))
    parser.read_csv(s, header=None, names=['foo', 'bar'], parse_dates=['foo'], cache_dates=cache)

@pytest.mark.parametrize('value', ['0'])
def test_bad_date_parse_with_warning(all_parsers: pd.io.parsers.ParserBase, cache, value):
    parser: pd.io.parsers.ParserBase = all_parsers
    s: StringIO = StringIO(f'{value},\n' * 50000)
    if parser.engine == 'pyarrow':
        warn = None
    elif cache:
        warn = None
    else:
        warn = UserWarning
    parser.read_csv_check_warnings(warn, 'Could not infer format', s, header=None, names=['foo', 'bar'], parse_dates=['foo'], cache_dates=cache, raise_on_extra_warnings=False)

def test_parse_dates_empty_string(all_parsers: pd.io.parsers.ParserBase):
    parser: pd.io.parsers.ParserBase = all_parsers
    data: str = 'Date,test\n2012-01-01,1\n,2'
    result: DataFrame = parser.read_csv(StringIO(data), parse_dates=['Date'], na_filter=False)
    expected_data: list = [[datetime(2012, 1, 1), 1], [pd.NaT, 2]]
    expected: DataFrame = DataFrame(expected_data, columns=['Date', 'test'])
    expected['Date'] = expected['Date'].astype('M8[s]')
    tm.assert_frame_equal(result, expected)

@xfail_pyarrow
@pytest.mark.parametrize('data,kwargs,expected', [('a\n04.15.2016', {'parse_dates': ['a']}, DataFrame([datetime(2016, 4, 15)], columns=['a'], dtype='M8[s]')), ('a\n04.15.2016', {'parse_dates': True, 'index_col': 0}, DataFrame(index=DatetimeIndex(['2016-04-15'], dtype='M8[s]', name='a'), columns=[])), ('a,b\n04.15.2016,09.16.2013', {'parse_dates': ['a', 'b']}, DataFrame([[datetime(2016, 4, 15), datetime(2013, 9, 16)]], dtype='M8[s]', columns=['a', 'b'])), ('a,b\n04.15.2016,09.16.2013', {'parse_dates': True, 'index_col': [0, 1]}, DataFrame(index=MultiIndex.from_tuples([(Timestamp(2016, 4, 15).as_unit('s'), Timestamp(2013, 9, 16).as_unit('s'))], names=['a', 'b']), columns=[]))])
def test_parse_dates_no_convert_thousands(all_parsers: pd.io.parsers.ParserBase, data, kwargs, expected):
    parser: pd.io.parsers.ParserBase = all_parsers
    result: DataFrame = parser.read_csv(StringIO(data), thousands='.', **kwargs)
    tm.assert_frame_equal(result, expected)

def test_parse_date_column_with_empty_string(all_parsers: pd.io.parsers.ParserBase):
    parser: pd.io.parsers.ParserBase = all_parsers
    data: str = 'case,opdate\n7,10/18/2006\n7,10/18/2008\n621, '
    result: DataFrame = parser.read_csv(StringIO(data), parse_dates=['opdate'])
    expected_data: list = [[7, '10/18/2006'], [7, '10/18/2008'], [621, ' ']]
    expected: DataFrame = DataFrame(expected_data, columns=['case', 'opdate'])
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('data,expected', [('a\n135217135789158401\n1352171357E+5', [135217135789158401, 135217135700000]), ('a\n99999999999\n123456789012345\n1234E+0', [99999999999, 123456789012345, 1234])])
@pytest.mark.parametrize('parse_dates', [True, False])
def test_parse_date_float(all_parsers: pd.io.parsers.ParserBase, data, expected, parse_dates):
    parser: pd.io.parsers.ParserBase = all_parsers
    result: DataFrame = parser.read_csv(StringIO(data), parse_dates=parse_dates)
    expected: DataFrame = DataFrame({'a': expected}, dtype='float64')
    tm.assert_frame_equal(result, expected)

def test_parse_timezone(all_parsers: pd.io.parsers.ParserBase):
    parser: pd.io.parsers.ParserBase = all_parsers
    data: str = 'dt,val\n              2018-01-04 09:01:00+09:00,23350\n              2018-01-04 09:02:00+09:00,23400\n              2018-01-04 09:03:00+09:00,23400\n              2018-01-04 09:04:00+09:00,23400\n              2018-01-04 09:05:00+09:00,23400'
    result: DataFrame = parser.read_csv(StringIO(data), parse_dates=['dt'])
    dti: DatetimeIndex = date_range(start='2018-01-04 09:01:00', end='2018-01-04 09:05:00', freq='1min', tz=timezone(timedelta(minutes=540)), unit='s')._with_freq(None)
    expected_data: dict = {'dt': dti, 'val': [23350, 23400, 23400, 23400, 23400]}
    expected: DataFrame = DataFrame(expected_data)
    tm.assert_frame_equal(result, expected)

@skip_pyarrow
@pytest.mark.parametrize('date_string', ['32/32/2019', '02/30/2019', '13/13/2019', '13/2019', 'a3/11/2018', '10/11/2o17'])
def test_invalid_parse_delimited_date(all_parsers: pd.io.parsers.ParserBase, date_string):
    parser: pd.io.parsers.ParserBase = all_parsers
    expected: DataFrame = DataFrame({0: [date_string]}, dtype='str')
    result: DataFrame = parser.read_csv(StringIO(date_string), header=None, parse_dates=[0])
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('date_string, dayfirst, expected', [('13/02/2019', True, datetime(2019, 2, 13)), ('02/13/2019', False, datetime(2019, 2, 13)), ('04/02/2019', True, datetime(2019, 2, 4))])
def test_parse_delimited_date_swap_no_warning(all_parsers: pd.io.parsers.ParserBase, date_string, dayfirst, expected, request):
    parser: pd.io.parsers.ParserBase = all_parsers
    expected: DataFrame = DataFrame({0: [expected]}, dtype='datetime64[s]')
    if parser.engine == 'pyarrow':
        if not dayfirst:
            pytest.skip(reason='https://github.com/apache/arrow/issues/38676')
        msg: str = "The 'dayfirst' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(date_string), header=None, dayfirst=dayfirst, parse_dates=[0])
        return
    result: DataFrame = parser.read_csv(StringIO(date_string), header=None, dayfirst=dayfirst, parse_dates=[0])
    tm.assert_frame_equal(result, expected)

@skip_pyarrow
@pytest.mark.parametrize('date_string, dayfirst, expected', [('13/02/2019', False, datetime(2019, 2, 13)), ('02/13/2019', True, datetime(2019, 2, 13))])
def test_parse_delimited_date_swap_with_warning(all_parsers: pd.io.parsers.ParserBase, date_string, dayfirst, expected):
    parser: pd.io.parsers.ParserBase = all_parsers
    expected: DataFrame = DataFrame({0: [expected]}, dtype='datetime64[s]')
    warning_msg: str = 'Parsing dates in .* format when dayfirst=.* was specified. Pass `dayfirst=.*` or specify a format to silence this warning