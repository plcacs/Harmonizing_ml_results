"""
Tests the 'read_fwf' function in parsers.py. This
test suite is independent of the others because the
engine is set to 'python-fwf' internally.
"""
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pytest
from pandas.errors import EmptyDataError
import pandas as pd
from pandas import DataFrame, DatetimeIndex
import pandas._testing as tm
from pandas.io.common import urlopen
from pandas.io.parsers import read_csv, read_fwf

def test_basic() -> None:
    data: str = 'A         B            C            D\n201158    360.242940   149.910199   11950.7\n201159    444.953632   166.985655   11788.4\n201160    364.136849   183.628767   11806.2\n201161    413.836124   184.375703   11916.8\n201162    502.953953   173.237159   12468.3\n'
    result: DataFrame = read_fwf(StringIO(data))
    expected: DataFrame = DataFrame([[201158, 360.24294, 149.910199, 11950.7], [201159, 444.953632, 166.985655, 11788.4], [201160, 364.136849, 183.628767, 11806.2], [201161, 413.836124, 184.375703, 11916.8], [201162, 502.953953, 173.237159, 12468.3]], columns=['A', 'B', 'C', 'D'])
    tm.assert_frame_equal(result, expected)

def test_colspecs() -> None:
    data: str = 'A   B     C            D            E\n201158    360.242940   149.910199   11950.7\n201159    444.953632   166.985655   11788.4\n201160    364.136849   183.628767   11806.2\n201161    413.836124   184.375703   11916.8\n201162    502.953953   173.237159   12468.3\n'
    colspecs: List[Tuple[int, int]] = [(0, 4), (4, 8), (8, 20), (21, 33), (34, 43)]
    result: DataFrame = read_fwf(StringIO(data), colspecs=colspecs)
    expected: DataFrame = DataFrame([[2011, 58, 360.24294, 149.910199, 11950.7], [2011, 59, 444.953632, 166.985655, 11788.4], [2011, 60, 364.136849, 183.628767, 11806.2], [2011, 61, 413.836124, 184.375703, 11916.8], [2011, 62, 502.953953, 173.237159, 12468.3]], columns=['A', 'B', 'C', 'D', 'E'])
    tm.assert_frame_equal(result, expected)

def test_widths() -> None:
    data: str = 'A    B    C            D            E\n2011 58   360.242940   149.910199   11950.7\n2011 59   444.953632   166.985655   11788.4\n2011 60   364.136849   183.628767   11806.2\n2011 61   413.836124   184.375703   11916.8\n2011 62   502.953953   173.237159   12468.3\n'
    result: DataFrame = read_fwf(StringIO(data), widths=[5, 5, 13, 13, 7])
    expected: DataFrame = DataFrame([[2011, 58, 360.24294, 149.910199, 11950.7], [2011, 59, 444.953632, 166.985655, 11788.4], [2011, 60, 364.136849, 183.628767, 11806.2], [2011, 61, 413.836124, 184.375703, 11916.8], [2011, 62, 502.953953, 173.237159, 12468.3]], columns=['A', 'B', 'C', 'D', 'E'])
    tm.assert_frame_equal(result, expected)

def test_non_space_filler() -> None:
    data: str = 'A~~~~B~~~~C~~~~~~~~~~~~D~~~~~~~~~~~~E\n201158~~~~360.242940~~~149.910199~~~11950.7\n201159~~~~444.953632~~~166.985655~~~11788.4\n201160~~~~364.136849~~~183.628767~~~11806.2\n201161~~~~413.836124~~~184.375703~~~11916.8\n201162~~~~502.953953~~~173.237159~~~12468.3\n'
    colspecs: List[Tuple[int, int]] = [(0, 4), (4, 8), (8, 20), (21, 33), (34, 43)]
    result: DataFrame = read_fwf(StringIO(data), colspecs=colspecs, delimiter='~')
    expected: DataFrame = DataFrame([[2011, 58, 360.24294, 149.910199, 11950.7], [2011, 59, 444.953632, 166.985655, 11788.4], [2011, 60, 364.136849, 183.628767, 11806.2], [2011, 61, 413.836124, 184.375703, 11916.8], [2011, 62, 502.953953, 173.237159, 12468.3]], columns=['A', 'B', 'C', 'D', 'E'])
    tm.assert_frame_equal(result, expected)

def test_over_specified() -> None:
    data: str = 'A   B     C            D            E\n201158    360.242940   149.910199   11950.7\n201159    444.953632   166.985655   11788.4\n201160    364.136849   183.628767   11806.2\n201161    413.836124   184.375703   11916.8\n201162    502.953953   173.237159   12468.3\n'
    colspecs: List[Tuple[int, int]] = [(0, 4), (4, 8), (8, 20), (21, 33), (34, 43)]
    with pytest.raises(ValueError, match='must specify only one of'):
        read_fwf(StringIO(data), colspecs=colspecs, widths=[6, 10, 10, 7])

def test_under_specified() -> None:
    data: str = 'A   B     C            D            E\n201158    360.242940   149.910199   11950.7\n201159    444.953632   166.985655   11788.4\n201160    364.136849   183.628767   11806.2\n201161    413.836124   184.375703   11916.8\n201162    502.953953   173.237159   12468.3\n'
    with pytest.raises(ValueError, match='Must specify either'):
        read_fwf(StringIO(data), colspecs=None, widths=None)

def test_read_csv_compat() -> None:
    csv_data: str = 'A,B,C,D,E\n2011,58,360.242940,149.910199,11950.7\n2011,59,444.953632,166.985655,11788.4\n2011,60,364.136849,183.628767,11806.2\n2011,61,413.836124,184.375703,11916.8\n2011,62,502.953953,173.237159,12468.3\n'
    expected: DataFrame = read_csv(StringIO(csv_data), engine='python')
    fwf_data: str = 'A   B     C            D            E\n201158    360.242940   149.910199   11950.7\n201159    444.953632   166.985655   11788.4\n201160    364.136849   183.628767   11806.2\n201161    413.836124   184.375703   11916.8\n201162    502.953953   173.237159   12468.3\n'
    colspecs: List[Tuple[int, int]] = [(0, 4), (4, 8), (8, 20), (21, 33), (34, 43)]
    result: DataFrame = read_fwf(StringIO(fwf_data), colspecs=colspecs)
    tm.assert_frame_equal(result, expected)

def test_bytes_io_input() -> None:
    data: BytesIO = BytesIO('שלום\nשלום'.encode())
    result: DataFrame = read_fwf(data, widths=[2, 2], encoding='utf8')
    expected: DataFrame = DataFrame([['של', 'ום']], columns=['של', 'ום'])
    tm.assert_frame_equal(result, expected)

def test_fwf_colspecs_is_list_or_tuple() -> None:
    data: str = 'index,A,B,C,D\nfoo,2,3,4,5\nbar,7,8,9,10\nbaz,12,13,14,15\nqux,12,13,14,15\nfoo2,12,13,14,15\nbar2,12,13,14,15\n'
    msg: str = 'column specifications must be a list or tuple.+'
    with pytest.raises(TypeError, match=msg):
        read_fwf(StringIO(data), colspecs={'a': 1}, delimiter=',')

def test_fwf_colspecs_is_list_or_tuple_of_two_element_tuples() -> None:
    data: str = 'index,A,B,C,D\nfoo,2,3,4,5\nbar,7,8,9,10\nbaz,12,13,14,15\nqux,12,13,14,15\nfoo2,12,13,14,15\nbar2,12,13,14,15\n'
    msg: str = 'Each column specification must be.+'
    with pytest.raises(TypeError, match=msg):
        read_fwf(StringIO(data), colspecs=[('a', 1)])

@pytest.mark.parametrize('colspecs,exp_data', [([(0, 3), (3, None)], [[123, 456], [456, 789]]), ([(None, 3), (3, 6)], [[123, 456], [456, 789]]), ([(0, None), (3, None)], [[123456, 456], [456789, 789]]), ([(None, None), (3, 6)], [[123456, 456], [456789, 789]])])
def test_fwf_colspecs_none(colspecs: List[Tuple[Optional[int], Optional[int]]], exp_data: List[List[int]]) -> None:
    data: str = '123456\n456789\n'
    expected: DataFrame = DataFrame(exp_data)
    result: DataFrame = read_fwf(StringIO(data), colspecs=colspecs, header=None)
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('infer_nrows,exp_data', [(1, [[1, 2], [3, 8]]), (10, [[1, 2], [123, 98]])])
def test_fwf_colspecs_infer_nrows(infer_nrows: int, exp_data: List[List[int]]) -> None:
    data: str = '  1  2\n123 98\n'
    expected: DataFrame = DataFrame(exp_data)
    result: DataFrame = read_fwf(StringIO(data), infer_nrows=infer_nrows, header=None)
    tm.assert_frame_equal(result, expected)

def test_fwf_regression() -> None:
    tz_list: List[int] = [1, 10, 20, 30, 60, 80, 100]
    widths: List[int] = [16] + [8] * len(tz_list)
    names: List[str] = ['SST'] + [f'T{z:03d}' for z in tz_list[1:]]
    data: str = '  2009164202000   9.5403  9.4105  8.6571  7.8372  6.0612  5.8843  5.5192\n2009164203000   9.5435  9.2010  8.6167  7.8176  6.0804  5.8728  5.4869\n2009164204000   9.5873  9.1326  8.4694  7.5889  6.0422  5.8526  5.4657\n2009164205000   9.5810  9.0896  8.4009  7.4652  6.0322  5.8189  5.4379\n2009164210000   9.6034  9.0897  8.3822  7.4905  6.0908  5.7904  5.4039\n'
    expected: DataFrame = DataFrame([[9.5403, 9.4105, 8.6571, 7.8372, 6.0612, 5.8843, 5.5192], [9.5435, 9.201, 8.6167, 7.8176, 6.0804, 5.8728, 5.4869], [9.5873, 9.1326, 8.4694, 7.5889, 6.0422, 5.8526, 5.4657], [9.581, 9.0896, 8.4009, 7.4652, 6.0322, 5.8189, 5.4379], [9.6034, 9.0897, 8.3822, 7.4905, 6.0908, 5.7904, 5.4039]], index=DatetimeIndex(['2009-06-13 20:20:00', '2009-06-13 20:30:00', '2009-06-13 20:40:00', '2009-06-13 20:50:00', '2009-06-13 21:00:00'], dtype='M8[us]'), columns=['SST', 'T010', 'T020', 'T030', 'T060', 'T080', 'T100'])
    result: DataFrame = read_fwf(StringIO(data), index_col=0, header=None, names=names, widths=widths, parse_dates=True, date_format='%Y%j%H%M%S')
    expected.index = expected.index.astype('M8[s]')
    tm.assert_frame_equal(result, expected)

def test_fwf_for_uint8() -> None:
    data: str = '1421302965.213420    PRI=3 PGN=0xef00      DST=0x17 SRC=0x28    04 154 00 00 00 00 00 127\n1421302964.226776    PRI=6 PGN=0xf002               SRC=0x47    243 00 00 255 247 00 00 71'
    df: DataFrame = read_fwf(StringIO(data), colspecs=[(0, 17), (25, 26), (33, 37), (49, 51), (58, 62), (63, 1000)], names=['time', 'pri', 'pgn', 'dst', 'src', 'data'], converters={'pgn': lambda x: int(x, 16), 'src': lambda x: int(x, 16), 'dst': lambda x: int(x, 16), 'data': lambda x: len(x.split(' '))})
    expected: DataFrame = DataFrame([[1421302965.21342, 3, 61184, 23, 40, 8], [1421302964.226776, 6, 61442, None, 71, 8]], columns=['time', 'pri', 'pgn', 'dst', 'src', 'data'])
    expected['dst'] = expected['dst'].astype(object)
    tm.assert_frame_equal(df, expected)

@pytest.mark.parametrize('comment', ['#', '~', '!'])
def test_fwf_comment(comment: str) -> None:
    data: str = '  1   2.   4  #hello world\n  5  NaN  10.0\n'
    data = data.replace('#', comment)
    colspecs: List[Tuple[int, int]] = [(0, 3), (4, 9), (9, 25)]
    expected: DataFrame = DataFrame([[1, 2.0, 4], [5, np.nan, 10.0]])
    result: DataFrame = read_fwf(StringIO(data), colspecs=colspecs, header=None, comment=comment)
    tm.assert_almost_equal(result, expected)

def test_fwf_skip_blank_lines() -> None:
    data: str = '\n\nA         B            C            D\n\n201158    360.242940   149.910199   11950.7\n201159    444.953632   166.985655   11788.4\n\n\n201162    502.953953   173.237159   12468.3\n\n'
    result: DataFrame = read_fwf(StringIO(data), skip_blank_lines=True)
    expected: DataFrame = DataFrame([[201158, 360.24294, 149.910199, 