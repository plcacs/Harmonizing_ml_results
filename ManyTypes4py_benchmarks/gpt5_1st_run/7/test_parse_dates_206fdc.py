"""
Tests date parsing functionality for all of the
parsers defined in parsers.py
"""
from datetime import datetime, timedelta, timezone
from io import StringIO
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Protocol, runtime_checkable

import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame, DatetimeIndex, Index, MultiIndex, Series, Timestamp
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
from pandas.core.tools.datetimes import start_caching_at
from pandas.io.parsers import read_csv

try:
    from pytest import FixtureRequest
except Exception:  # pragma: no cover - fallback for older pytest typing
    FixtureRequest = Any  # type: ignore[assignment]

pytestmark: Any = pytest.mark.filterwarnings('ignore:Passing a BlockManager to DataFrame:DeprecationWarning')
xfail_pyarrow: Any = pytest.mark.usefixtures('pyarrow_xfail')
skip_pyarrow: Any = pytest.mark.usefixtures('pyarrow_skip')


@runtime_checkable
class ParserLike(Protocol):
    engine: str

    def read_csv(self, *args: Any, **kwargs: Any) -> DataFrame: ...
    def read_csv_check_warnings(
        self,
        expected_warning: Optional[Union[type, Tuple[type, ...]]],
        message: str,
        *args: Any,
        **kwargs: Any,
    ) -> DataFrame: ...


def test_date_col_as_index_col(all_parsers: ParserLike) -> None:
    data: str = 'KORD,19990127 19:00:00, 18:56:00, 0.8100, 2.8100, 7.2000, 0.0000, 280.0000\nKORD,19990127 20:00:00, 19:56:00, 0.0100, 2.2100, 7.2000, 0.0000, 260.0000\nKORD,19990127 21:00:00, 20:56:00, -0.5900, 2.2100, 5.7000, 0.0000, 280.0000\nKORD,19990127 21:00:00, 21:18:00, -0.9900, 2.0100, 3.6000, 0.0000, 270.0000\nKORD,19990127 22:00:00, 21:56:00, -0.5900, 1.7100, 5.1000, 0.0000, 290.0000\n'
    parser: ParserLike = all_parsers
    kwds: Dict[str, Any] = {'header': None, 'parse_dates': [1], 'index_col': 1, 'names': ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7']}
    result = parser.read_csv(StringIO(data), **kwds)
    index = Index([datetime(1999, 1, 27, 19, 0), datetime(1999, 1, 27, 20, 0), datetime(1999, 1, 27, 21, 0), datetime(1999, 1, 27, 21, 0), datetime(1999, 1, 27, 22, 0)], dtype='M8[s]', name='X1')
    expected = DataFrame([['KORD', ' 18:56:00', 0.81, 2.81, 7.2, 0.0, 280.0], ['KORD', ' 19:56:00', 0.01, 2.21, 7.2, 0.0, 260.0], ['KORD', ' 20:56:00', -0.59, 2.21, 5.7, 0.0, 280.0], ['KORD', ' 21:18:00', -0.99, 2.01, 3.6, 0.0, 270.0], ['KORD', ' 21:56:00', -0.59, 1.71, 5.1, 0.0, 290.0]], columns=['X0', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7'], index=index)
    if parser.engine == 'pyarrow':
        expected['X2'] = pd.to_datetime('1970-01-01' + expected['X2']).dt.time
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow
def test_nat_parse(all_parsers: ParserLike) -> None:
    parser: ParserLike = all_parsers
    df = DataFrame({'A': np.arange(10, dtype='float64'), 'B': Timestamp('20010101')})
    df.iloc[3:6, :] = np.nan
    with tm.ensure_clean('__nat_parse_.csv') as path:
        df.to_csv(path)
        result = parser.read_csv(path, index_col=0, parse_dates=['B'])
        tm.assert_frame_equal(result, df)


@skip_pyarrow
def test_parse_dates_implicit_first_col(all_parsers: ParserLike) -> None:
    data: str = 'A,B,C\n20090101,a,1,2\n20090102,b,3,4\n20090103,c,4,5\n'
    parser: ParserLike = all_parsers
    result = parser.read_csv(StringIO(data), parse_dates=True)
    expected = parser.read_csv(StringIO(data), index_col=0, parse_dates=True)
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow
def test_parse_dates_string(all_parsers: ParserLike) -> None:
    data: str = 'date,A,B,C\n20090101,a,1,2\n20090102,b,3,4\n20090103,c,4,5\n'
    parser: ParserLike = all_parsers
    result = parser.read_csv(StringIO(data), index_col='date', parse_dates=['date'])
    index = date_range('1/1/2009', periods=3, name='date', unit='s')._with_freq(None)
    expected = DataFrame({'A': ['a', 'b', 'c'], 'B': [1, 3, 4], 'C': [2, 4, 5]}, index=index)
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow
@pytest.mark.parametrize('parse_dates', [[0, 2], ['a', 'c']])
def test_parse_dates_column_list(all_parsers: ParserLike, parse_dates: List[Union[int, str]]) -> None:
    data: str = 'a,b,c\n01/01/2010,1,15/02/2010'
    parser: ParserLike = all_parsers
    expected = DataFrame({'a': [datetime(2010, 1, 1)], 'b': [1], 'c': [datetime(2010, 2, 15)]})
    expected['a'] = expected['a'].astype('M8[s]')
    expected['c'] = expected['c'].astype('M8[s]')
    expected = expected.set_index(['a', 'b'])
    result = parser.read_csv(StringIO(data), index_col=[0, 1], parse_dates=parse_dates, dayfirst=True)
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow
@pytest.mark.parametrize('index_col', [[0, 1], [1, 0]])
def test_multi_index_parse_dates(all_parsers: ParserLike, index_col: List[int]) -> None:
    data: str = 'index1,index2,A,B,C\n20090101,one,a,1,2\n20090101,two,b,3,4\n20090101,three,c,4,5\n20090102,one,a,1,2\n20090102,two,b,3,4\n20090102,three,c,4,5\n20090103,one,a,1,2\n20090103,two,b,3,4\n20090103,three,c,4,5\n'
    parser: ParserLike = all_parsers
    dti = date_range('2009-01-01', periods=3, freq='D', unit='s')
    index = MultiIndex.from_product([dti, ('one', 'two', 'three')], names=['index1', 'index2'])
    if index_col == [1, 0]:
        index = index.swaplevel(0, 1)
    expected = DataFrame([['a', 1, 2], ['b', 3, 4], ['c', 4, 5], ['a', 1, 2], ['b', 3, 4], ['c', 4, 5], ['a', 1, 2], ['b', 3, 4], ['c', 4, 5]], columns=['A', 'B', 'C'], index=index)
    result = parser.read_csv_check_warnings(UserWarning, 'Could not infer format', StringIO(data), index_col=index_col, parse_dates=True)
    tm.assert_frame_equal(result, expected)


def test_parse_tz_aware(all_parsers: ParserLike) -> None:
    parser: ParserLike = all_parsers
    data: str = 'Date,x\n2012-06-13T01:39:00Z,0.5'
    result = parser.read_csv(StringIO(data), index_col=0, parse_dates=True)
    expected = DataFrame({'x': [0.5]}, index=Index([Timestamp('2012-06-13 01:39:00+00:00')], name='Date'))
    if parser.engine == 'pyarrow':
        pytz = pytest.importorskip('pytz')
        expected_tz = pytz.utc  # type: ignore[assignment]
    else:
        expected_tz = timezone.utc
    tm.assert_frame_equal(result, expected)
    assert result.index.tz is expected_tz


@pytest.mark.parametrize('kwargs', [{}, {'index_col': 'C'}])
def test_read_with_parse_dates_scalar_non_bool(all_parsers: ParserLike, kwargs: Dict[str, Any]) -> None:
    parser: ParserLike = all_parsers
    msg: str = "Only booleans and lists are accepted for the 'parse_dates' parameter"
    data: str = 'A,B,C\n    1,2,2003-11-1'
    with pytest.raises(TypeError, match=msg):
        parser.read_csv(StringIO(data), parse_dates='C', **kwargs)


@pytest.mark.parametrize('parse_dates', [(1,), np.array([4, 5]), {1, 3}])
def test_read_with_parse_dates_invalid_type(all_parsers: ParserLike, parse_dates: object) -> None:
    parser: ParserLike = all_parsers
    msg: str = "Only booleans and lists are accepted for the 'parse_dates' parameter"
    data: str = 'A,B,C\n    1,2,2003-11-1'
    with pytest.raises(TypeError, match=msg):
        parser.read_csv(StringIO(data), parse_dates=parse_dates)


@pytest.mark.parametrize('value', ['nan', ''])
def test_bad_date_parse(all_parsers: ParserLike, cache: bool, value: str) -> None:
    parser: ParserLike = all_parsers
    s = StringIO(f'{value},\n' * (start_caching_at + 1))
    parser.read_csv(s, header=None, names=['foo', 'bar'], parse_dates=['foo'], cache_dates=cache)


@pytest.mark.parametrize('value', ['0'])
def test_bad_date_parse_with_warning(all_parsers: ParserLike, cache: bool, value: str) -> None:
    parser: ParserLike = all_parsers
    s = StringIO(f'{value},\n' * 50000)
    if parser.engine == 'pyarrow':
        warn: Optional[Union[type, Tuple[type, ...]]] = None
    elif cache:
        warn = None
    else:
        warn = UserWarning
    parser.read_csv_check_warnings(warn, 'Could not infer format', s, header=None, names=['foo', 'bar'], parse_dates=['foo'], cache_dates=cache, raise_on_extra_warnings=False)


def test_parse_dates_empty_string(all_parsers: ParserLike) -> None:
    parser: ParserLike = all_parsers
    data: str = 'Date,test\n2012-01-01,1\n,2'
    result = parser.read_csv(StringIO(data), parse_dates=['Date'], na_filter=False)
    expected = DataFrame([[datetime(2012, 1, 1), 1], [pd.NaT, 2]], columns=['Date', 'test'])
    expected['Date'] = expected['Date'].astype('M8[s]')
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow
@pytest.mark.parametrize(
    'data,kwargs,expected',
    [
        (
            'a\n04.15.2016',
            {'parse_dates': ['a']},
            DataFrame([datetime(2016, 4, 15)], columns=['a'], dtype='M8[s]'),
        ),
        (
            'a\n04.15.2016',
            {'parse_dates': True, 'index_col': 0},
            DataFrame(index=DatetimeIndex(['2016-04-15'], dtype='M8[s]', name='a'), columns=[]),
        ),
        (
            'a,b\n04.15.2016,09.16.2013',
            {'parse_dates': ['a', 'b']},
            DataFrame([[datetime(2016, 4, 15), datetime(2013, 9, 16)]], dtype='M8[s]', columns=['a', 'b']),
        ),
        (
            'a,b\n04.15.2016,09.16.2013',
            {'parse_dates': True, 'index_col': [0, 1]},
            DataFrame(index=MultiIndex.from_tuples([(Timestamp(2016, 4, 15).as_unit('s'), Timestamp(2013, 9, 16).as_unit('s'))], names=['a', 'b']), columns=[]),
        ),
    ],
)
def test_parse_dates_no_convert_thousands(all_parsers: ParserLike, data: str, kwargs: Dict[str, Any], expected: DataFrame) -> None:
    parser: ParserLike = all_parsers
    result = parser.read_csv(StringIO(data), thousands='.', **kwargs)
    tm.assert_frame_equal(result, expected)


def test_parse_date_column_with_empty_string(all_parsers: ParserLike) -> None:
    parser: ParserLike = all_parsers
    data: str = 'case,opdate\n7,10/18/2006\n7,10/18/2008\n621, '
    result = parser.read_csv(StringIO(data), parse_dates=['opdate'])
    expected_data: List[List[Union[int, str]]] = [[7, '10/18/2006'], [7, '10/18/2008'], [621, ' ']]
    expected = DataFrame(expected_data, columns=['case', 'opdate'])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    'data,expected',
    [
        ('a\n135217135789158401\n1352171357E+5', [135217135789158401, 135217135700000]),
        ('a\n99999999999\n123456789012345\n1234E+0', [99999999999, 123456789012345, 1234]),
    ],
)
@pytest.mark.parametrize('parse_dates', [True, False])
def test_parse_date_float(all_parsers: ParserLike, data: str, expected: List[int], parse_dates: bool) -> None:
    parser: ParserLike = all_parsers
    result = parser.read_csv(StringIO(data), parse_dates=parse_dates)
    expected_df = DataFrame({'a': expected}, dtype='float64')
    tm.assert_frame_equal(result, expected_df)


def test_parse_timezone(all_parsers: ParserLike) -> None:
    parser: ParserLike = all_parsers
    data: str = 'dt,val\n              2018-01-04 09:01:00+09:00,23350\n              2018-01-04 09:02:00+09:00,23400\n              2018-01-04 09:03:00+09:00,23400\n              2018-01-04 09:04:00+09:00,23400\n              2018-01-04 09:05:00+09:00,23400'
    result = parser.read_csv(StringIO(data), parse_dates=['dt'])
    dti = date_range(start='2018-01-04 09:01:00', end='2018-01-04 09:05:00', freq='1min', tz=timezone(timedelta(minutes=540)), unit='s')._with_freq(None)
    expected_data = {'dt': dti, 'val': [23350, 23400, 23400, 23400, 23400]}
    expected = DataFrame(expected_data)
    tm.assert_frame_equal(result, expected)


@skip_pyarrow
@pytest.mark.parametrize('date_string', ['32/32/2019', '02/30/2019', '13/13/2019', '13/2019', 'a3/11/2018', '10/11/2o17'])
def test_invalid_parse_delimited_date(all_parsers: ParserLike, date_string: str) -> None:
    parser: ParserLike = all_parsers
    expected = DataFrame({0: [date_string]}, dtype='str')
    result = parser.read_csv(StringIO(date_string), header=None, parse_dates=[0])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    'date_string,dayfirst,expected',
    [
        ('13/02/2019', True, datetime(2019, 2, 13)),
        ('02/13/2019', False, datetime(2019, 2, 13)),
        ('04/02/2019', True, datetime(2019, 2, 4)),
    ],
)
def test_parse_delimited_date_swap_no_warning(
    all_parsers: ParserLike,
    date_string: str,
    dayfirst: bool,
    expected: datetime,
    request: FixtureRequest,
) -> None:
    parser: ParserLike = all_parsers
    expected_df = DataFrame({0: [expected]}, dtype='datetime64[s]')
    if parser.engine == 'pyarrow':
        if not dayfirst:
            pytest.skip(reason='https://github.com/apache/arrow/issues/38676')
        msg = "The 'dayfirst' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(date_string), header=None, dayfirst=dayfirst, parse_dates=[0])
        return
    result = parser.read_csv(StringIO(date_string), header=None, dayfirst=dayfirst, parse_dates=[0])
    tm.assert_frame_equal(result, expected_df)


@skip_pyarrow
@pytest.mark.parametrize(
    'date_string,dayfirst,expected',
    [
        ('13/02/2019', False, datetime(2019, 2, 13)),
        ('02/13/2019', True, datetime(2019, 2, 13)),
    ],
)
def test_parse_delimited_date_swap_with_warning(all_parsers: ParserLike, date_string: str, dayfirst: bool, expected: datetime) -> None:
    parser: ParserLike = all_parsers
    expected_df = DataFrame({0: [expected]}, dtype='datetime64[s]')
    warning_msg = 'Parsing dates in .* format when dayfirst=.* was specified. Pass `dayfirst=.*` or specify a format to silence this warning.'
    result = parser.read_csv_check_warnings(UserWarning, warning_msg, StringIO(date_string), header=None, dayfirst=dayfirst, parse_dates=[0])
    tm.assert_frame_equal(result, expected_df)


def test_parse_multiple_delimited_dates_with_swap_warnings() -> None:
    with pytest.raises(ValueError, match='^time data "31/05/2000" doesn\\\'t match format "%m/%d/%Y". You might want to try:'):
        pd.to_datetime(['01/01/2000', '31/05/2000', '31/05/2001', '01/02/2000'])


@skip_pyarrow
@pytest.mark.parametrize(
    'names, usecols, parse_dates, missing_cols',
    [
        (None, ['val'], ['date', 'time'], 'date, time'),
        (None, ['val'], [0, 'time'], 'time'),
        (['date1', 'time1', 'temperature'], None, ['date', 'time'], 'date, time'),
        (['date1', 'time1', 'temperature'], ['date1', 'temperature'], ['date1', 'time'], 'time'),
    ],
)
def test_missing_parse_dates_column_raises(
    all_parsers: ParserLike,
    names: Optional[List[str]],
    usecols: Optional[List[str]],
    parse_dates: List[Union[int, str]],
    missing_cols: str,
) -> None:
    parser: ParserLike = all_parsers
    content = StringIO('date,time,val\n2020-01-31,04:20:32,32\n')
    msg = f"Missing column provided to 'parse_dates': '{missing_cols}'"
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(content, sep=',', names=names, usecols=usecols, parse_dates=parse_dates)


@xfail_pyarrow
def test_date_parser_and_names(all_parsers: ParserLike) -> None:
    parser: ParserLike = all_parsers
    data = StringIO('x,y\n1,2')
    warn: Union[type, Tuple[type, ...]]
    warn = UserWarning
    if parser.engine == 'pyarrow':
        warn = (UserWarning, DeprecationWarning)
    result = parser.read_csv_check_warnings(warn, 'Could not infer format', data, parse_dates=['B'], names=['B'])
    expected = DataFrame({'B': ['y', '2']}, index=['x', '1'])
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow
def test_date_parser_multiindex_columns(all_parsers: ParserLike) -> None:
    parser: ParserLike = all_parsers
    data: str = 'a,b\n1,2\n2019-12-31,6'
    result = parser.read_csv(StringIO(data), parse_dates=[('a', '1')], header=[0, 1])
    expected = DataFrame({('a', '1'): Timestamp('2019-12-31'), ('b', '2'): [6]})
    tm.assert_frame_equal(result, expected)


def test_date_parser_usecols_thousands(all_parsers: ParserLike) -> None:
    data: str = 'A,B,C\n    1,3,20-09-01-01\n    2,4,20-09-01-01\n    '
    parser: ParserLike = all_parsers
    if parser.engine == 'pyarrow':
        msg = "The 'thousands' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), parse_dates=[1], usecols=[1, 2], thousands='-')
        return
    result = parser.read_csv_check_warnings(UserWarning, 'Could not infer format', StringIO(data), parse_dates=[1], usecols=[1, 2], thousands='-')
    expected = DataFrame({'B': [3, 4], 'C': [Timestamp('20-09-2001 01:00:00')] * 2})
    expected['C'] = expected['C'].astype('M8[s]')
    tm.assert_frame_equal(result, expected)


def test_dayfirst_warnings() -> None:
    input: str = 'date\n31/12/2014\n10/03/2011'
    expected = DatetimeIndex(['2014-12-31', '2011-03-10'], dtype='datetime64[s]', freq=None, name='date')
    warning_msg = 'Parsing dates in .* format when dayfirst=.* was specified. Pass `dayfirst=.*` or specify a format to silence this warning.'
    res1 = read_csv(StringIO(input), parse_dates=['date'], dayfirst=True, index_col='date').index
    tm.assert_index_equal(expected, res1)
    with tm.assert_produces_warning(UserWarning, match=warning_msg):
        res2 = read_csv(StringIO(input), parse_dates=['date'], dayfirst=False, index_col='date').index
    tm.assert_index_equal(expected, res2)
    input = 'date\n31/12/2014\n03/30/2011'
    expected2 = Index(['31/12/2014', '03/30/2011'], dtype='str', name='date')
    res5 = read_csv(StringIO(input), parse_dates=['date'], dayfirst=True, index_col='date').index
    tm.assert_index_equal(expected2, res5)
    with tm.assert_produces_warning(UserWarning, match=warning_msg):
        res6 = read_csv(StringIO(input), parse_dates=['date'], dayfirst=False, index_col='date').index
    tm.assert_index_equal(expected2, res6)


@pytest.mark.parametrize(
    'date_string, dayfirst',
    [
        pytest.param('31/1/2014', False, id='second date is single-digit'),
        pytest.param('1/31/2014', True, id='first date is single-digit'),
    ],
)
def test_dayfirst_warnings_no_leading_zero(date_string: str, dayfirst: bool) -> None:
    initial_value = f'date\n{date_string}'
    expected = DatetimeIndex(['2014-01-31'], dtype='datetime64[s]', freq=None, name='date')
    warning_msg = 'Parsing dates in .* format when dayfirst=.* was specified. Pass `dayfirst=.*` or specify a format to silence this warning.'
    with tm.assert_produces_warning(UserWarning, match=warning_msg):
        res = read_csv(StringIO(initial_value), parse_dates=['date'], index_col='date', dayfirst=dayfirst).index
    tm.assert_index_equal(expected, res)


@skip_pyarrow
def test_infer_first_column_as_index(all_parsers: ParserLike) -> None:
    parser: ParserLike = all_parsers
    data: str = 'a,b,c\n1970-01-01,2,3,4'
    result = parser.read_csv(StringIO(data), parse_dates=['a'])
    expected = DataFrame({'a': '2', 'b': 3, 'c': 4}, index=['1970-01-01'])
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow
def test_replace_nans_before_parsing_dates(all_parsers: ParserLike) -> None:
    parser: ParserLike = all_parsers
    data: str = 'Test\n2012-10-01\n0\n2015-05-15\n#\n2017-09-09\n'
    result = parser.read_csv(StringIO(data), na_values={'Test': ['#', '0']}, parse_dates=['Test'], date_format='%Y-%m-%d')
    expected = DataFrame({'Test': [Timestamp('2012-10-01'), pd.NaT, Timestamp('2015-05-15'), pd.NaT, Timestamp('2017-09-09')]}, dtype='M8[s]')
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow
def test_parse_dates_and_string_dtype(all_parsers: ParserLike) -> None:
    parser: ParserLike = all_parsers
    data: str = 'a,b\n1,2019-12-31\n'
    result = parser.read_csv(StringIO(data), dtype='string', parse_dates=['b'])
    expected = DataFrame({'a': ['1'], 'b': [Timestamp('2019-12-31')]})
    expected['a'] = expected['a'].astype('string')
    expected['b'] = expected['b'].astype('M8[s]')
    tm.assert_frame_equal(result, expected)


def test_parse_dot_separated_dates(all_parsers: ParserLike) -> None:
    parser: ParserLike = all_parsers
    data: str = 'a,b\n27.03.2003 14:55:00.000,1\n03.08.2003 15:20:00.000,2'
    if parser.engine == 'pyarrow':
        expected_index: Index = Index(['27.03.2003 14:55:00.000', '03.08.2003 15:20:00.000'], dtype='str', name='a')
        warn: Optional[Union[type, Tuple[type, ...]]] = None
    else:
        expected_index = DatetimeIndex(['2003-03-27 14:55:00', '2003-08-03 15:20:00'], dtype='datetime64[ms]', name='a')
        warn = UserWarning
    msg = 'when dayfirst=False \\(the default\\) was specified'
    result = parser.read_csv_check_warnings(warn, msg, StringIO(data), parse_dates=True, index_col=0, raise_on_extra_warnings=False)
    expected = DataFrame({'b': [1, 2]}, index=expected_index)
    tm.assert_frame_equal(result, expected)


def test_parse_dates_dict_format(all_parsers: ParserLike) -> None:
    parser: ParserLike = all_parsers
    data: str = 'a,b\n2019-12-31,31-12-2019\n2020-12-31,31-12-2020'
    result = parser.read_csv(StringIO(data), date_format={'a': '%Y-%m-%d', 'b': '%d-%m-%Y'}, parse_dates=['a', 'b'])
    expected = DataFrame({'a': [Timestamp('2019-12-31'), Timestamp('2020-12-31')], 'b': [Timestamp('2019-12-31'), Timestamp('2020-12-31')]}, dtype='M8[s]')
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow
def test_parse_dates_dict_format_index(all_parsers: ParserLike) -> None:
    parser: ParserLike = all_parsers
    data: str = 'a,b\n2019-12-31,31-12-2019\n2020-12-31,31-12-2020'
    result = parser.read_csv(StringIO(data), date_format={'a': '%Y-%m-%d'}, parse_dates=True, index_col=0)
    expected = DataFrame({'b': ['31-12-2019', '31-12-2020']}, index=Index([Timestamp('2019-12-31'), Timestamp('2020-12-31')], name='a'))
    tm.assert_frame_equal(result, expected)


def test_parse_dates_arrow_engine(all_parsers: ParserLike) -> None:
    parser: ParserLike = all_parsers
    data: str = 'a,b\n2000-01-01 00:00:00,1\n2000-01-01 00:00:01,1'
    result = parser.read_csv(StringIO(data), parse_dates=['a'])
    expected = DataFrame({'a': [Timestamp('2000-01-01 00:00:00'), Timestamp('2000-01-01 00:00:01')], 'b': 1})
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow
def test_from_csv_with_mixed_offsets(all_parsers: ParserLike) -> None:
    parser: ParserLike = all_parsers
    data: str = 'a\n2020-01-01T00:00:00+01:00\n2020-01-01T00:00:00+00:00'
    result = parser.read_csv(StringIO(data), parse_dates=['a'])['a']
    expected = Series(['2020-01-01T00:00:00+01:00', '2020-01-01T00:00:00+00:00'], name='a', index=[0, 1])
    tm.assert_series_equal(result, expected)