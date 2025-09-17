#!/usr/bin/env python3
from __future__ import annotations
import contextlib
import csv
from datetime import date, datetime, time, timedelta
from io import StringIO
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, Index, Series, Timestamp, concat, date_range, isna, to_datetime, to_timedelta
import pandas._testing as tm

# Fixtures and helper functions

@pytest.fixture
def sql_strings() -> dict[str, Any]:
    return {
        'read_parameters': {
            'postgresql': 'SELECT * FROM iris WHERE "SepalLength" = %s',
            'sqlite': 'SELECT * FROM iris WHERE "SepalLength" = ?'
        },
        'read_named_parameters': {
            'postgresql': 'SELECT * FROM iris WHERE "SepalLength" = %(sepal_length)s',
            'sqlite': 'SELECT * FROM iris WHERE "SepalLength" = :sepal_length'
        },
        'read_no_parameters_with_percent': {
            'postgresql': 'SELECT * FROM iris WHERE "SepalLength" LIKE \'%%\'',
            'sqlite': 'SELECT * FROM iris WHERE "SepalLength" LIKE \'%%\''
        }
    }

def flavor(conn_name: str) -> str:
    if 'postgresql' in conn_name:
        return 'postgresql'
    elif 'sqlite' in conn_name:
        return 'sqlite'
    elif 'mysql' in conn_name:
        return 'mysql'
    raise ValueError(f'unsupported connection: {conn_name}')

def format_query(sql: str, *args: Any) -> str:
    _formatters: dict[type, Callable[[Any], str]] = {
        datetime: lambda x: "'{}'".format(x),
        str: lambda x: "'{}'".format(x),
        np.str_: lambda x: "'{}'".format(x),
        bytes: lambda x: "'{}'".format(x),
        float: lambda x: '{:.8f}'.format(x),
        int: lambda x: '{:d}'.format(x),
        type(None): lambda x: 'NULL',
        np.float64: lambda x: '{:.10f}'.format(x),
        bool: lambda x: "'{!s}'".format(x)
    }
    processed_args: List[str] = []
    for arg in args:
        if isinstance(arg, float) and isna(arg):
            arg = None
        formatter = _formatters[type(arg)]
        processed_args.append(formatter(arg))
    return sql % tuple(processed_args)

def tquery(query: str, con: Optional[Any] = None) -> Optional[List[Any]]:
    """Replace removed sql.tquery function"""
    from pandas.io import sql
    with sql.pandasSQL_builder(con) as pandas_sql:
        res = pandas_sql.execute(query).fetchall()
    return None if res is None else list(res)

def get_sqlite_column_type(conn: Any, table: str, column: str) -> str:
    recs = conn.execute(f'PRAGMA table_info({table})')
    for cid, name, ctype, not_null, default, pk in recs:
        if name == column:
            return ctype
    raise ValueError(f'Table {table}, column {column} not found')

# Test functions

@pytest.mark.parametrize('conn', all_connectable)
def test_dataframe_to_sql(conn: Any, test_frame1: pd.DataFrame, request: pytest.FixtureRequest) -> None:
    conn = request.getfixturevalue(conn)
    test_frame1.to_sql(name='test', con=conn, if_exists='append', index=False)

@pytest.mark.parametrize('conn', all_connectable)
def test_dataframe_to_sql_empty(conn: Any, test_frame1: pd.DataFrame, request: pytest.FixtureRequest) -> None:
    if conn == 'postgresql_adbc_conn' and (not pd._config.config.get_bool("mode.use_inf_as_na")):
        request.node.add_marker(pytest.mark.xfail(reason='postgres ADBC driver < 1.2 cannot insert index with null type'))
    conn = request.getfixturevalue(conn)
    empty_df = test_frame1.iloc[:0]
    empty_df.to_sql(name='test', con=conn, if_exists='append', index=False)

@pytest.mark.parametrize('conn', all_connectable)
def test_dataframe_to_sql_arrow_dtypes(conn: Any, request: pytest.FixtureRequest) -> None:
    pytest.importorskip('pyarrow')
    df = DataFrame({
        'int': pd.array([1], dtype='int8[pyarrow]'),
        'datetime': pd.array([datetime(2023, 1, 1)], dtype='timestamp[ns][pyarrow]'),
        'date': pd.array([date(2023, 1, 1)], dtype='date32[day][pyarrow]'),
        'timedelta': pd.array([timedelta(1)], dtype='duration[ns][pyarrow]'),
        'string': pd.array(['a'], dtype='string[pyarrow]')
    })
    if 'adbc' in conn:
        if conn == 'sqlite_adbc_conn':
            df = df.drop(columns=['timedelta'])
        if pd.compat.pa_version_under14p1:
            exp_warning = DeprecationWarning
            msg = 'is_sparse is deprecated'
        else:
            exp_warning = None
            msg = ''
    else:
        exp_warning = UserWarning
        msg = "the 'timedelta'"
    conn = request.getfixturevalue(conn)
    with tm.assert_produces_warning(exp_warning, match=msg, check_stacklevel=False):
        df.to_sql(name='test_arrow', con=conn, if_exists='replace', index=False)

@pytest.mark.parametrize('conn', all_connectable)
def test_dataframe_to_sql_arrow_dtypes_missing(conn: Any, request: pytest.FixtureRequest, nulls_fixture: Any) -> None:
    pytest.importorskip('pyarrow')
    df = DataFrame({'datetime': pd.array([datetime(2023, 1, 1), nulls_fixture], dtype='timestamp[ns][pyarrow]')})
    conn = request.getfixturevalue(conn)
    df.to_sql(name='test_arrow', con=conn, if_exists='replace', index=False)

@pytest.mark.parametrize('conn', all_connectable)
@pytest.mark.parametrize('method', [None, 'multi'])
def test_to_sql(conn: Any, method: Optional[str], test_frame1: pd.DataFrame, request: pytest.FixtureRequest) -> None:
    if method == 'multi' and 'adbc' in conn:
        request.node.add_marker(pytest.mark.xfail(reason="'method' not implemented for ADBC drivers", strict=True))
    conn = request.getfixturevalue(conn)
    from pandas.io import sql as pd_sql
    with pd_sql.pandasSQL_builder(conn, need_transaction=True) as pandasSQL:
        pandasSQL.to_sql(test_frame1, 'test_frame', method=method)
        assert pandasSQL.has_table('test_frame')
    from pandas.io import sql
    assert count_rows(conn, 'test_frame') == len(test_frame1)

@pytest.mark.parametrize('conn', all_connectable)
@pytest.mark.parametrize('mode, num_row_coef', [('replace', 1), ('append', 2)])
def test_to_sql_exist(conn: Any, mode: str, num_row_coef: Union[int, str], test_frame1: pd.DataFrame, request: pytest.FixtureRequest) -> None:
    conn = request.getfixturevalue(conn)
    from pandas.io import sql as pd_sql
    with pd_sql.pandasSQL_builder(conn, need_transaction=True) as pandasSQL:
        pandasSQL.to_sql(test_frame1, 'test_frame', if_exists='fail')
        pandasSQL.to_sql(test_frame1, 'test_frame', if_exists=mode)
        assert pandasSQL.has_table('test_frame')
    from pandas.io import sql
    assert count_rows(conn, 'test_frame') == num_row_coef * len(test_frame1)

@pytest.mark.parametrize('conn', all_connectable)
def test_to_sql_exist_fail(conn: Any, test_frame1: pd.DataFrame, request: pytest.FixtureRequest) -> None:
    conn = request.getfixturevalue(conn)
    from pandas.io import sql as pd_sql
    with pd_sql.pandasSQL_builder(conn, need_transaction=True) as pandasSQL:
        pandasSQL.to_sql(test_frame1, 'test_frame', if_exists='fail')
        assert pandasSQL.has_table('test_frame')
        msg = "Table 'test_frame' already exists"
        with pytest.raises(ValueError, match=msg):
            pandasSQL.to_sql(test_frame1, 'test_frame', if_exists='fail')

@pytest.mark.parametrize('conn', all_connectable_iris)
def test_read_iris_query(conn: Any, request: pytest.FixtureRequest) -> None:
    conn = request.getfixturevalue(conn)
    from pandas.io import sql
    iris_frame = sql.read_sql_query('SELECT * FROM iris', conn)
    check_iris_frame(iris_frame)
    iris_frame = pd.read_sql('SELECT * FROM iris', conn)
    check_iris_frame(iris_frame)
    iris_frame = pd.read_sql('SELECT * FROM iris where 0=1', conn)
    assert iris_frame.shape == (0, 5)
    assert 'SepalWidth' in iris_frame.columns

@pytest.mark.parametrize('conn', all_connectable_iris)
def test_read_iris_query_chunksize(conn: Any, request: pytest.FixtureRequest) -> None:
    if 'adbc' in conn:
        request.node.add_marker(pytest.mark.xfail(reason="'chunksize' not implemented for ADBC drivers", strict=True))
    conn = request.getfixturevalue(conn)
    from pandas.io import sql
    iris_frame = concat(sql.read_sql_query('SELECT * FROM iris', conn, chunksize=7))
    check_iris_frame(iris_frame)
    iris_frame = concat(pd.read_sql('SELECT * FROM iris', conn, chunksize=7))
    check_iris_frame(iris_frame)
    iris_frame = concat(pd.read_sql('SELECT * FROM iris where 0=1', conn, chunksize=7))
    assert iris_frame.shape == (0, 5)
    assert 'SepalWidth' in iris_frame.columns

@pytest.mark.parametrize('conn', all_connectable_iris)
def test_read_iris_query_string_with_parameter(conn: Any, request: pytest.FixtureRequest, sql_strings: dict[str, Any]) -> None:
    if 'adbc' in conn:
        request.node.add_marker(pytest.mark.xfail(reason="'chunksize' not implemented for ADBC drivers", strict=True))
    conn_name: str = conn  # Assuming conn name is a string key here.
    for db, query in sql_strings['read_parameters'].items():
        if db in conn_name:
            break
    else:
        raise KeyError(f"No part of {conn} found in sql_strings['read_parameters']")
    conn = request.getfixturevalue(conn)
    from pandas.io import sql
    iris_frame = sql.read_sql_query(query, conn, params=('Iris-setosa', 5.1))
    check_iris_frame(iris_frame)

@pytest.mark.parametrize('conn', all_connectable_iris)
def test_api_read_sql_view(conn: Any, request: pytest.FixtureRequest) -> None:
    conn = request.getfixturevalue(conn)
    from pandas.io import sql
    iris_frame = sql.read_sql_query('SELECT * FROM iris_view', conn)
    check_iris_frame(iris_frame)

@pytest.mark.parametrize('conn', all_connectable_iris)
def test_api_read_sql_with_chunksize_no_result(conn: Any, request: pytest.FixtureRequest) -> None:
    if 'adbc' in conn:
        request.node.add_marker(pytest.mark.xfail(reason='chunksize argument NotImplemented with ADBC'))
    conn = request.getfixturevalue(conn)
    query = 'SELECT * FROM iris_view WHERE "SepalLength" < 0.0'
    with_batch = sql.read_sql_query(query, conn, chunksize=5)
    without_batch = sql.read_sql_query(query, conn)
    tm.assert_frame_equal(concat(with_batch), without_batch)

@pytest.mark.parametrize('conn', all_connectable)
def test_api_to_sql(conn: Any, request: pytest.FixtureRequest, test_frame1: pd.DataFrame) -> None:
    conn = request.getfixturevalue(conn)
    from pandas.io import sql
    if sql.has_table('test_frame1', conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table('test_frame1')
    sql.to_sql(test_frame1, 'test_frame1', conn)
    assert sql.has_table('test_frame1', conn)

@pytest.mark.parametrize('conn', all_connectable)
def test_api_to_sql_fail(conn: Any, request: pytest.FixtureRequest, test_frame1: pd.DataFrame) -> None:
    conn = request.getfixturevalue(conn)
    from pandas.io import sql
    if sql.has_table('test_frame2', conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table('test_frame2')
    sql.to_sql(test_frame1, 'test_frame2', conn, if_exists='fail')
    assert sql.has_table('test_frame2', conn)
    msg = "Table 'test_frame2' already exists"
    with pytest.raises(ValueError, match=msg):
        sql.to_sql(test_frame1, 'test_frame2', conn, if_exists='fail')

@pytest.mark.parametrize('conn', all_connectable)
def test_api_to_sql_replace(conn: Any, request: pytest.FixtureRequest, test_frame1: pd.DataFrame) -> None:
    conn = request.getfixturevalue(conn)
    from pandas.io import sql
    if sql.has_table('test_frame3', conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table('test_frame3')
    sql.to_sql(test_frame1, 'test_frame3', conn, if_exists='fail')
    sql.to_sql(test_frame1, 'test_frame3', conn, if_exists='replace')
    assert sql.has_table('test_frame3', conn)
    num_entries = len(test_frame1)
    num_rows = count_rows(conn, 'test_frame3')
    assert num_rows == num_entries

@pytest.mark.parametrize('conn', all_connectable)
def test_api_to_sql_append(conn: Any, request: pytest.FixtureRequest, test_frame1: pd.DataFrame) -> None:
    conn = request.getfixturevalue(conn)
    from pandas.io import sql
    if sql.has_table('test_frame4', conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table('test_frame4')
    assert sql.to_sql(test_frame1, 'test_frame4', conn, if_exists='fail') == 4
    assert sql.to_sql(test_frame1, 'test_frame4', conn, if_exists='append') == 4
    assert sql.has_table('test_frame4', conn)
    num_entries = 2 * len(test_frame1)
    num_rows = count_rows(conn, 'test_frame4')
    assert num_rows == num_entries

@pytest.mark.parametrize('conn', all_connectable)
def test_api_to_sql_type_mapping(conn: Any, request: pytest.FixtureRequest, test_frame3: pd.DataFrame) -> None:
    conn = request.getfixturevalue(conn)
    from pandas.io import sql
    if sql.has_table('test_frame5', conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table('test_frame5')
    sql.to_sql(test_frame3, 'test_frame5', conn, index=False)
    result = sql.read_sql('SELECT * FROM test_frame5', conn)
    tm.assert_frame_equal(test_frame3, result)

@pytest.mark.parametrize('conn', all_connectable)
def test_api_to_sql_series(conn: Any, request: pytest.FixtureRequest) -> None:
    conn = request.getfixturevalue(conn)
    from pandas.io import sql
    if sql.has_table('test_series', conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table('test_series')
    s = Series(np.arange(5, dtype='int64'), name='series')
    sql.to_sql(s, 'test_series', conn, index=False)
    s2 = sql.read_sql_query('SELECT * FROM test_series', conn)
    tm.assert_frame_equal(s.to_frame(), s2)

@pytest.mark.parametrize('conn', all_connectable)
def test_api_roundtrip(conn: Any, request: pytest.FixtureRequest, test_frame1: pd.DataFrame) -> None:
    conn_name: str = conn
    conn = request.getfixturevalue(conn)
    if pd.io.sql.has_table('test_frame_roundtrip', conn):
        with pd.io.sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table('test_frame_roundtrip')
    pd.io.sql.to_sql(test_frame1, 'test_frame_roundtrip', con=conn)
    result = pd.io.sql.read_sql_query('SELECT * FROM test_frame_roundtrip', con=conn)
    if 'adbc' in conn_name:
        result = result.drop(columns='__index_level_0__')
    else:
        result = result.drop(columns='level_0')
    tm.assert_frame_equal(result, test_frame1)

@pytest.mark.parametrize('conn', all_connectable)
def test_api_roundtrip_chunksize(conn: Any, request: pytest.FixtureRequest, test_frame1: pd.DataFrame) -> None:
    if 'adbc' in conn:
        request.node.add_marker(pytest.mark.xfail(reason='chunksize argument NotImplemented with ADBC'))
    conn = request.getfixturevalue(conn)
    if pd.io.sql.has_table('test_frame_roundtrip', conn):
        with pd.io.sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table('test_frame_roundtrip')
    pd.io.sql.to_sql(test_frame1, 'test_frame_roundtrip', con=conn, index=False, chunksize=2)
    result = pd.io.sql.read_sql_query('SELECT * FROM test_frame_roundtrip', con=conn)
    tm.assert_frame_equal(result, test_frame1)

@pytest.mark.parametrize('conn', all_connectable_iris)
def test_api_execute_sql(conn: Any, request: pytest.FixtureRequest) -> None:
    conn = request.getfixturevalue(conn)
    from pandas.io import sql
    with sql.pandasSQL_builder(conn) as pandas_sql:
        with pandas_sql.run_transaction():
            iris_results = pandas_sql.execute('SELECT * FROM iris')
            row = iris_results.fetchone()
            iris_results.close()
    assert list(row) == [5.1, 3.5, 1.4, 0.2, 'Iris-setosa']

@pytest.mark.parametrize('conn', all_connectable_types)
def test_api_date_parsing(conn: Any, request: pytest.FixtureRequest) -> None:
    conn_name: str = conn
    conn = request.getfixturevalue(conn)
    df = pd.io.sql.read_sql_query('SELECT * FROM types', conn)
    if not ('mysql' in conn_name or 'postgres' in conn_name):
        assert not issubclass(df.DateCol.dtype.type, np.datetime64)
    df = pd.io.sql.read_sql_query('SELECT * FROM types', conn, parse_dates=['DateCol'])
    assert issubclass(df.DateCol.dtype.type, np.datetime64)
    assert df.DateCol.tolist() == [Timestamp(2000, 1, 3, 0, 0, 0), Timestamp(2000, 1, 4, 0, 0, 0)]
    df = pd.io.sql.read_sql_query('SELECT * FROM types', conn, parse_dates={'DateCol': '%Y-%m-%d %H:%M:%S'})
    assert issubclass(df.DateCol.dtype.type, np.datetime64)
    assert df.DateCol.tolist() == [Timestamp(2000, 1, 3, 0, 0, 0), Timestamp(2000, 1, 4, 0, 0, 0)]
    df = pd.io.sql.read_sql_query('SELECT * FROM types', conn, parse_dates=['IntDateCol'])
    assert issubclass(df.IntDateCol.dtype.type, np.datetime64)
    assert df.IntDateCol.tolist() == [Timestamp(1986, 12, 25, 0, 0, 0), Timestamp(2013, 1, 1, 0, 0, 0)]
    df = pd.io.sql.read_sql_query('SELECT * FROM types', conn, parse_dates={'IntDateCol': 's'})
    assert issubclass(df.IntDateCol.dtype.type, np.datetime64)
    assert df.IntDateCol.tolist() == [Timestamp(1986, 12, 25, 0, 0, 0), Timestamp(2013, 1, 1, 0, 0, 0)]
    df = pd.io.sql.read_sql_query('SELECT * FROM types', conn, parse_dates={'IntDateOnlyCol': '%Y%m%d'})
    assert issubclass(df.IntDateOnlyCol.dtype.type, np.datetime64)
    assert df.IntDateOnlyCol.tolist() == [Timestamp('2010-10-10'), Timestamp('2010-12-12')]

@pytest.mark.parametrize('conn', all_connectable_types)
@pytest.mark.parametrize('error', ['raise', 'coerce'])
@pytest.mark.parametrize('read_sql, text, mode', [
    (pd.io.sql.read_sql, 'SELECT * FROM types', ('sqlalchemy', 'fallback')),
    (pd.io.sql.read_sql, 'types', 'sqlalchemy'),
    (pd.io.sql.read_sql_query, 'SELECT * FROM types', ('sqlalchemy', 'fallback')),
    (pd.io.sql.read_sql_table, 'types', 'sqlalchemy')
])
def test_api_custom_dateparsing_error(conn: Any, request: pytest.FixtureRequest, read_sql: Callable[..., pd.DataFrame], text: str, mode: Any, error: str, types_data_frame: pd.DataFrame) -> None:
    conn_name: str = conn
    conn = request.getfixturevalue(conn)
    if text == 'types' and conn_name == 'sqlite_buildin_types':
        request.applymarker(pytest.mark.xfail(reason='failing combination of arguments'))
    expected = types_data_frame.astype({'DateCol': 'datetime64[s]'})
    result = read_sql(text, con=conn, parse_dates={'DateCol': {'errors': error}})
    if 'postgres' in conn_name:
        result['BoolCol'] = result['BoolCol'].astype(int)
        result['BoolColWithNull'] = result['BoolColWithNull'].astype(float)
    if conn_name == 'postgresql_adbc_types':
        expected = expected.astype({'IntDateCol': 'int32', 'IntDateOnlyCol': 'int32', 'IntCol': 'int32'})
    if conn_name == 'postgresql_adbc_types' and pd.compat.pa_version_under14p1:
        expected['DateCol'] = expected['DateCol'].astype('datetime64[ns]')
    elif 'postgres' in conn_name or 'mysql' in conn_name:
        expected['DateCol'] = expected['DateCol'].astype('datetime64[us]')
    else:
        expected['DateCol'] = expected['DateCol'].astype('datetime64[s]')
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('conn', all_connectable_types)
def test_api_date_and_index(conn: Any, request: pytest.FixtureRequest) -> None:
    conn = request.getfixturevalue(conn)
    df = pd.io.sql.read_sql_query('SELECT * FROM types', conn, index_col='DateCol', parse_dates=['DateCol', 'IntDateCol'])
    assert issubclass(df.index.dtype.type, np.datetime64)
    assert issubclass(df.IntDateCol.dtype.type, np.datetime64)

@pytest.mark.parametrize('conn', all_connectable)
def test_api_timedelta(conn: Any, request: pytest.FixtureRequest) -> None:
    conn_name: str = conn
    conn = request.getfixturevalue(conn)
    if pd.io.sql.has_table('test_timedelta', conn):
        with pd.io.sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table('test_timedelta')
    df = to_timedelta(Series(['00:00:01', '00:00:03'], name='foo')).to_frame()
    if conn_name == 'sqlite_adbc_conn':
        request.node.add_marker(pytest.mark.xfail(reason="sqlite ADBC driver doesn't implement timedelta"))
    if 'adbc' in conn_name:
        if pd.compat.pa_version_under14p1:
            exp_warning = DeprecationWarning
        else:
            exp_warning = None
    else:
        exp_warning = UserWarning
    with tm.assert_produces_warning(exp_warning, check_stacklevel=False):
        result_count = df.to_sql(name='test_timedelta', con=conn)
    assert result_count == 2
    result = pd.io.sql.read_sql_query('SELECT * FROM test_timedelta', conn)
    if conn_name == 'postgresql_adbc_conn':
        expected = Series([pd.DateOffset(months=0, days=0, microseconds=1000000, nanoseconds=0), pd.DateOffset(months=0, days=0, microseconds=3000000, nanoseconds=0)], name='foo')
    else:
        expected = df['foo'].astype('int64')
    tm.assert_series_equal(result['foo'], expected)

@pytest.mark.parametrize('conn', all_connectable)
def test_api_complex_raises(conn: Any, request: pytest.FixtureRequest) -> None:
    conn_name: str = conn
    conn = request.getfixturevalue(conn)
    df = DataFrame({'a': [1 + 1j, 2j]})
    if 'adbc' in conn_name:
        msg = 'datatypes not supported'
    else:
        msg = 'Complex datatypes not supported'
    with pytest.raises(ValueError, match=msg):
        _ = df.to_sql('test_complex', con=conn)

@pytest.mark.parametrize('conn', all_connectable)
@pytest.mark.parametrize('index_name,index_label,expected', [
    (None, None, 'index'),
    (None, 'other_label', 'other_label'),
    ('index_name', None, 'index_name'),
    ('index_name', 'other_label', 'other_label'),
    (0, None, '0'),
    (None, 0, '0')
])
def test_api_to_sql_index_label(conn: Any, request: pytest.FixtureRequest, index_name: Union[str, int, None], index_label: Union[str, int, None], expected: Union[str, int]) -> None:
    if 'adbc' in conn:
        request.node.add_marker(pytest.mark.xfail(reason='index_label argument NotImplemented with ADBC'))
    conn = request.getfixturevalue(conn)
    from pandas.io import sql
    if sql.has_table('test_index_label', conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table('test_index_label')
    temp_frame = DataFrame({'col1': range(4)})
    temp_frame.index.name = index_name  # type: ignore
    sql.to_sql(temp_frame, 'test_index_label', conn, index_label=index_label)
    frame = pd.io.sql.read_sql_query('SELECT * FROM test_index_label', conn)
    assert frame.columns[0] == expected

@pytest.mark.parametrize('conn', all_connectable)
def test_api_to_sql_index_label_multiindex(conn: Any, request: pytest.FixtureRequest) -> None:
    conn_name: str = conn
    if 'mysql' in conn_name:
        request.applymarker(pytest.mark.xfail(reason='MySQL can fail using TEXT without length as key', strict=False))
    elif 'adbc' in conn_name:
        request.node.add_marker(pytest.mark.xfail(reason='index_label argument NotImplemented with ADBC'))
    conn = request.getfixturevalue(conn)
    from pandas.io import sql
    if sql.has_table('test_index_label', conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table('test_index_label')
    expected_row_count: int = 4
    temp_frame = DataFrame({'col1': range(4)}, index=pd.MultiIndex.from_product([('A0', 'A1'), ('B0', 'B1')]))
    result = sql.to_sql(temp_frame, 'test_index_label', conn)
    assert result == expected_row_count
    frame = pd.io.sql.read_sql_query('SELECT * FROM test_index_label', conn)
    assert frame.columns[0] == 'level_0'
    assert frame.columns[1] == 'level_1'
    result = sql.to_sql(temp_frame, 'test_index_label', conn, if_exists='replace', index_label=['A', 'B'])
    assert result == expected_row_count
    frame = pd.io.sql.read_sql_query('SELECT * FROM test_index_label', conn)
    assert frame.columns[:2].tolist() == ['A', 'B']
    temp_frame.index.names = ['A', 'B']
    result = sql.to_sql(temp_frame, 'test_index_label', conn, if_exists='replace')
    assert result == expected_row_count
    frame = pd.io.sql.read_sql_query('SELECT * FROM test_index_label', conn)
    assert frame.columns[:2].tolist() == ['A', 'B']
    result = sql.to_sql(temp_frame, 'test_index_label', conn, if_exists='replace', index_label=['C', 'D'])
    assert result == expected_row_count
    frame = pd.io.sql.read_sql_query('SELECT * FROM test_index_label', conn)
    assert frame.columns[:2].tolist() == ['C', 'D']
    msg = "Length of 'index_label' should match number of levels, which is 2"
    with pytest.raises(ValueError, match=msg):
        sql.to_sql(temp_frame, 'test_index_label', conn, if_exists='replace', index_label='C')

@pytest.mark.parametrize('conn', all_connectable)
def test_api_multiindex_roundtrip(conn: Any, request: pytest.FixtureRequest) -> None:
    conn = request.getfixturevalue(conn)
    if pd.io.sql.has_table('test_multiindex_roundtrip', conn):
        with pd.io.sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table('test_multiindex_roundtrip')
    df = DataFrame.from_records([(1, 2.1, 'line1'), (2, 1.5, 'line2')], columns=['A', 'B', 'C'], index=['A', 'B'])
    pd.io.sql.to_sql(df, name='test_multiindex_roundtrip', con=conn)
    result = pd.io.sql.read_sql_query('SELECT * FROM test_multiindex_roundtrip', conn, index_col=['A', 'B'])
    tm.assert_frame_equal(df, result, check_index_type=True)

@pytest.mark.parametrize('conn', all_connectable)
@pytest.mark.parametrize('dtype', [None, int, float, {'A': int, 'B': float}])
def test_api_dtype_argument(conn: Any, request: pytest.FixtureRequest, dtype: Union[None, int, float, dict],) -> None:
    conn_name: str = conn
    conn = request.getfixturevalue(conn)
    if pd.io.sql.has_table('test_dtype_argument', conn):
        with pd.io.sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table('test_dtype_argument')
    df = DataFrame([[1.2, 3.4], [5.6, 7.8]], columns=['A', 'B'])
    assert df.to_sql(name='test_dtype_argument', con=conn) == 2
    expected = df.astype(dtype) if dtype is not None else df
    if 'postgres' in conn_name:
        query = 'SELECT "A", "B" FROM test_dtype_argument'
    else:
        query = 'SELECT A, B FROM test_dtype_argument'
    result = pd.io.sql.read_sql_query(query, con=conn, dtype=dtype)
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('conn', all_connectable)
def test_api_integer_col_names(conn: Any, request: pytest.FixtureRequest) -> None:
    conn = request.getfixturevalue(conn)
    df = DataFrame([[1, 2], [3, 4]], columns=[0, 1])
    pd.io.sql.to_sql(df, 'test_frame_integer_col_names', conn, if_exists='replace')

@pytest.mark.parametrize('conn', all_connectable)
def test_api_get_schema(conn: Any, request: pytest.FixtureRequest, test_frame1: pd.DataFrame) -> None:
    if 'adbc' in conn:
        request.node.add_marker(pytest.mark.xfail(reason="'get_schema' not implemented for ADBC drivers", strict=True))
    conn = request.getfixturevalue(conn)
    create_sql = pd.io.sql.get_schema(test_frame1, 'test', con=conn)
    assert 'CREATE' in create_sql

@pytest.mark.parametrize('conn', all_connectable)
def test_api_get_schema_with_schema(conn: Any, request: pytest.FixtureRequest, test_frame1: pd.DataFrame) -> None:
    if 'adbc' in conn:
        request.node.add_marker(pytest.mark.xfail(reason="'get_schema' not implemented for ADBC drivers", strict=True))
    conn = request.getfixturevalue(conn)
    create_sql = pd.io.sql.get_schema(test_frame1, 'test', con=conn, schema='pypi')
    assert 'CREATE TABLE pypi.' in create_sql

@pytest.mark.parametrize('conn', all_connectable)
def test_api_get_schema_dtypes(conn: Any, request: pytest.FixtureRequest) -> None:
    if 'adbc' in conn:
        request.node.add_marker(pytest.mark.xfail(reason="'get_schema' not implemented for ADBC drivers", strict=True))
    conn_name: str = conn
    conn = request.getfixturevalue(conn)
    float_frame = DataFrame({'a': [1.1, 1.2], 'b': [2.1, 2.2]})
    if conn_name == 'sqlite_buildin':
        dtype: Any = 'INTEGER'
    else:
        from sqlalchemy import Integer
        dtype = Integer
    create_sql = pd.io.sql.get_schema(float_frame, 'test', con=conn, dtype={'b': dtype})
    assert 'CREATE' in create_sql
    assert 'INTEGER' in create_sql

@pytest.mark.parametrize('conn', all_connectable)
def test_api_get_schema_keys(conn: Any, request: pytest.FixtureRequest, test_frame1: pd.DataFrame) -> None:
    if 'adbc' in conn:
        request.node.add_marker(pytest.mark.xfail(reason="'get_schema' not implemented for ADBC drivers", strict=True))
    conn_name: str = conn
    conn = request.getfixturevalue(conn)
    frame = DataFrame({'Col1': [1.1, 1.2], 'Col2': [2.1, 2.2]})
    create_sql = pd.io.sql.get_schema(frame, 'test', con=conn, keys='Col1')
    if 'mysql' in conn_name:
        constraint_sentence = 'CONSTRAINT test_pk PRIMARY KEY (`Col1`)'
    else:
        constraint_sentence = 'CONSTRAINT test_pk PRIMARY KEY ("Col1")'
    assert constraint_sentence in create_sql
    create_sql = pd.io.sql.get_schema(test_frame1, 'test', con=conn, keys=['A', 'B'])
    if 'mysql' in conn_name:
        constraint_sentence = 'CONSTRAINT test_pk PRIMARY KEY (`A`, `B`)'
    else:
        constraint_sentence = 'CONSTRAINT test_pk PRIMARY KEY ("A", "B")'
    assert constraint_sentence in create_sql

@pytest.mark.parametrize('conn', all_connectable)
def test_api_chunksize_read(conn: Any, request: pytest.FixtureRequest) -> None:
    if 'adbc' in conn:
        request.node.add_marker(pytest.mark.xfail(reason='chunksize argument NotImplemented with ADBC'))
    conn_name: str = conn
    conn = request.getfixturevalue(conn)
    if pd.io.sql.has_table('test_chunksize', conn):
        with pd.io.sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table('test_chunksize')
    df = DataFrame(np.random.default_rng(2).standard_normal((22, 5)), columns=list('abcde'))
    df.to_sql(name='test_chunksize', con=conn, index=False)
    res1 = pd.io.sql.read_sql_query('select * from test_chunksize', conn)
    res2 = DataFrame()
    i = 0
    sizes = [5, 5, 5, 5, 2]
    for chunk in pd.io.sql.read_sql_query('select * from test_chunksize', conn, chunksize=5):
        res2 = concat([res2, chunk], ignore_index=True)
        assert len(chunk) == sizes[i]
        i += 1
    tm.assert_frame_equal(res1, res2)
    if conn_name == 'sqlite_buildin':
        with pytest.raises(NotImplementedError, match=''):
            pd.io.sql.read_sql_table('test_chunksize', conn, chunksize=5)
    else:
        res3 = DataFrame()
        i = 0
        sizes = [5, 5, 5, 5, 2]
        for chunk in pd.io.sql.read_sql_table('test_chunksize', conn, chunksize=5):
            res3 = concat([res3, chunk], ignore_index=True)
            assert len(chunk) == sizes[i]
            i += 1
        tm.assert_frame_equal(res1, res3)

@pytest.mark.parametrize('conn', all_connectable)
def test_api_categorical(conn: Any, request: pytest.FixtureRequest) -> None:
    if conn == 'postgresql_adbc_conn':
        adbc = pytest.importorskip('adbc_driver_postgresql', errors='ignore')
        if adbc is not None and pd.core.dtypes.common.Version(adbc.__version__) < pd.core.dtypes.common.Version('0.9.0'):
            request.node.add_marker(pytest.mark.xfail(reason='categorical dtype not implemented for ADBC postgres driver', strict=True))
    conn = request.getfixturevalue(conn)
    from pandas.io import sql
    if sql.has_table('test_categorical', conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table('test_categorical')
    df = DataFrame({'person_id': [1, 2, 3], 'person_name': ['John P. Doe', 'Jane Dove', 'John P. Doe']})
    df2 = df.copy()
    df2['person_name'] = df2['person_name'].astype('category')
    df2.to_sql(name='test_categorical', con=conn, index=False)
    res = pd.io.sql.read_sql_query('SELECT * FROM test_categorical', conn)
    tm.assert_frame_equal(res, df)

@pytest.mark.parametrize('conn', all_connectable)
def test_api_unicode_column_name(conn: Any, request: pytest.FixtureRequest) -> None:
    conn = request.getfixturevalue(conn)
    from pandas.io import sql
    if sql.has_table('test_unicode', conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table('test_unicode')
    df = DataFrame([[1, 2], [3, 4]], columns=['é', 'b'])
    df.to_sql(name='test_unicode', con=conn, index=False)

@pytest.mark.parametrize('conn', all_connectable)
def test_api_escaped_table_name(conn: Any, request: pytest.FixtureRequest) -> None:
    conn_name: str = conn
    conn = request.getfixturevalue(conn)
    from pandas.io import sql
    if sql.has_table('d1187b08-4943-4c8d-a7f6', conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table('d1187b08-4943-4c8d-a7f6')
    df = DataFrame({'A': [0, 1, 2], 'B': [0.2, np.nan, 5.6]})
    df.to_sql(name='d1187b08-4943-4c8d-a7f6', con=conn, index=False)
    if 'postgres' in conn_name:
        query = 'SELECT * FROM "d1187b08-4943-4c8d-a7f6"'
    else:
        query = 'SELECT * FROM `d1187b08-4943-4c8d-a7f6`'
    res = pd.io.sql.read_sql_query(query, conn)
    tm.assert_frame_equal(res, df)

@pytest.mark.parametrize('conn', all_connectable)
def test_api_read_sql_duplicate_columns(conn: Any, request: pytest.FixtureRequest) -> None:
    if 'adbc' in conn:
        pa = pytest.importorskip('pyarrow')
        if not (pd.core.dtypes.common.Version(pa.__version__) >= pd.core.dtypes.common.Version('16.0') and conn in ['sqlite_adbc_conn', 'postgresql_adbc_conn']):
            request.node.add_marker(pytest.mark.xfail(reason='pyarrow->pandas throws ValueError', strict=True))
    conn = request.getfixturevalue(conn)
    from pandas.io import sql
    if sql.has_table('test_table', conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table('test_table')
    df = DataFrame({'a': [1, 2, 3], 'b': [0.1, 0.2, 0.3], 'c': 1})
    pd.io.sql.to_sql(df, 'test_table', con=conn, index=False)
    result = pd.read_sql('SELECT a, b, a +1 as a, c FROM test_table', conn)
    expected = DataFrame([[1, 0.1, 2, 1], [2, 0.2, 3, 1], [3, 0.3, 4, 1]], columns=['a', 'b', 'a', 'c'])
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('conn', all_connectable)
def test_read_table_columns(conn: Any, request: pytest.FixtureRequest, test_frame1: pd.DataFrame) -> None:
    conn_name: str = conn
    if conn_name == 'sqlite_buildin':
        request.applymarker(pytest.mark.xfail(reason='Not Implemented'))
    conn = request.getfixturevalue(conn)
    pd.io.sql.to_sql(test_frame1, 'test_frame', conn)
    cols = ['A', 'B']
    result = pd.io.sql.read_sql_table('test_frame', conn, columns=cols)
    assert result.columns.tolist() == cols

@pytest.mark.parametrize('conn', all_connectable)
def test_read_table_index_col(conn: Any, request: pytest.FixtureRequest, test_frame1: pd.DataFrame) -> None:
    conn_name: str = conn
    if conn_name == 'sqlite_buildin':
        request.applymarker(pytest.mark.xfail(reason='Not Implemented'))
    conn = request.getfixturevalue(conn)
    pd.io.sql.to_sql(test_frame1, 'test_frame', conn)
    result = pd.io.sql.read_sql_table('test_frame', conn, index_col='index')
    assert result.index.names == ['index']
    result = pd.io.sql.read_sql_table('test_frame', conn, index_col=['A', 'B'])
    assert result.index.names == ['A', 'B']
    result = pd.io.sql.read_sql_table('test_frame', conn, index_col=['A', 'B'], columns=['C', 'D'])
    assert result.index.names == ['A', 'B']
    assert result.columns.tolist() == ['C', 'D']

@pytest.mark.parametrize('conn', all_connectable_iris)
def test_read_sql_delegate(conn: Any, request: pytest.FixtureRequest) -> None:
    if conn == 'sqlite_buildin_iris':
        request.applymarker(pytest.mark.xfail(reason='sqlite_buildin connection does not implement read_sql_table'))
    conn = request.getfixturevalue(conn)
    iris_frame1 = pd.io.sql.read_sql_query('SELECT * FROM iris', conn)
    iris_frame2 = pd.io.sql.read_sql('SELECT * FROM iris', conn)
    tm.assert_frame_equal(iris_frame1, iris_frame2)
    iris_frame1 = pd.io.sql.read_sql_table('iris', conn)
    iris_frame2 = pd.io.sql.read_sql('iris', conn)
    tm.assert_frame_equal(iris_frame1, iris_frame2)

def test_not_reflect_all_tables(sqlite_conn: Any) -> None:
    conn = sqlite_conn
    from sqlalchemy import text
    query_list = [text('CREATE TABLE invalid (x INTEGER, y UNKNOWN);'), text('CREATE TABLE other_table (x INTEGER, y INTEGER);')]
    for query in query_list:
        conn.execute(query)
    with tm.assert_produces_warning(None):
        pd.io.sql.read_sql_table('other_table', conn)
        pd.io.sql.read_sql_query('SELECT * FROM other_table', conn)

@pytest.mark.parametrize('conn', all_connectable)
def test_warning_case_insensitive_table_name(conn: Any, request: pytest.FixtureRequest, test_frame1: pd.DataFrame) -> None:
    conn_name: str = conn
    if conn_name == 'sqlite_buildin' or 'adbc' in conn_name:
        request.applymarker(pytest.mark.xfail(reason='Does not raise warning'))
    conn = request.getfixturevalue(conn)
    from pandas.io import sql
    with tm.assert_produces_warning(UserWarning, match="The provided table name 'TABLE1' is not found exactly as such in the database after writing the table, possibly due to case sensitivity issues. Consider using lower case table names."):
        with sql.SQLDatabase(conn) as db:
            db.check_case_sensitive('TABLE1', '')
    with tm.assert_produces_warning(None):
        test_frame1.to_sql(name='CaseSensitive', con=conn)

@pytest.mark.parametrize('conn', sqlalchemy_connectable)
def test_sqlalchemy_type_mapping(conn: Any, request: pytest.FixtureRequest) -> None:
    conn = request.getfixturevalue(conn)
    from sqlalchemy import TIMESTAMP
    df = DataFrame({'time': to_datetime(['2014-12-12 01:54', '2014-12-11 02:54'], utc=True)})
    with pd.io.sql.SQLDatabase(conn) as db:
        table = pd.io.sql.SQLTable('test_type', db, frame=df)
        assert isinstance(table.table.c['time'].type, TIMESTAMP)

@pytest.mark.parametrize('conn', sqlalchemy_connectable)
@pytest.mark.parametrize('integer, expected', [
    ('int8', 'SMALLINT'),
    ('Int8', 'SMALLINT'),
    ('uint8', 'SMALLINT'),
    ('UInt8', 'SMALLINT'),
    ('int16', 'SMALLINT'),
    ('Int16', 'SMALLINT'),
    ('uint16', 'INTEGER'),
    ('UInt16', 'INTEGER'),
    ('int32', 'INTEGER'),
    ('Int32', 'INTEGER'),
    ('uint32', 'BIGINT'),
    ('UInt32', 'BIGINT'),
    ('int64', 'BIGINT'),
    ('Int64', 'BIGINT'),
    (int, 'BIGINT' if np.dtype(int).name == 'int64' else 'INTEGER')
])
def test_sqlalchemy_integer_mapping(conn: Any, request: pytest.FixtureRequest, integer: Any, expected: str) -> None:
    conn = request.getfixturevalue(conn)
    df = DataFrame([0, 1], columns=['a'], dtype=integer)
    with pd.io.sql.SQLDatabase(conn) as db:
        table = pd.io.sql.SQLTable('test_type', db, frame=df)
        result = str(table.table.c.a.type)
    assert result == expected

@pytest.mark.parametrize('conn', sqlalchemy_connectable)
@pytest.mark.parametrize('integer', ['uint64', 'UInt64'])
def test_sqlalchemy_integer_overload_mapping(conn: Any, request: pytest.FixtureRequest, integer: str) -> None:
    conn = request.getfixturevalue(conn)
    df = DataFrame([0, 1], columns=['a'], dtype=integer)
    with pd.io.sql.SQLDatabase(conn) as db:
        with pytest.raises(ValueError, match='Unsigned 64 bit integer datatype is not supported'):
            pd.io.sql.SQLTable('test_type', db, frame=df)

@pytest.mark.parametrize('conn', all_connectable)
def test_database_uri_string(conn: Any, request: pytest.FixtureRequest, test_frame1: pd.DataFrame) -> None:
    pytest.importorskip('sqlalchemy')
    conn = request.getfixturevalue(conn)
    with tm.ensure_clean() as name:
        db_uri = 'sqlite:///' + name
        table = 'iris'
        test_frame1.to_sql(name=table, con=db_uri, if_exists='replace', index=False)
        test_frame2 = pd.io.sql.read_sql(table, db_uri)
        test_frame3 = pd.io.sql.read_sql_table(table, db_uri)
        query = 'SELECT * FROM iris'
        test_frame4 = pd.io.sql.read_sql_query(query, db_uri)
    tm.assert_frame_equal(test_frame1, test_frame2)
    tm.assert_frame_equal(test_frame1, test_frame3)
    tm.assert_frame_equal(test_frame1, test_frame4)

@td.skip_if_installed('pg8000')
@pytest.mark.parametrize('conn', all_connectable)
def test_pg8000_sqlalchemy_passthrough_error(conn: Any, request: pytest.FixtureRequest) -> None:
    pytest.importorskip('sqlalchemy')
    conn = request.getfixturevalue(conn)
    db_uri = 'postgresql+pg8000://user:pass@host/dbname'
    with pytest.raises(ImportError, match='pg8000'):
        pd.io.sql.read_sql('select * from table', db_uri)

@pytest.mark.parametrize('conn', sqlalchemy_connectable_iris)
def test_query_by_text_obj(conn: Any, request: pytest.FixtureRequest) -> None:
    conn_name: str = conn
    conn = request.getfixturevalue(conn)
    from sqlalchemy import text
    if 'postgres' in conn_name:
        name_text = text('select * from iris where "Name"=:name')
    else:
        name_text = text('select * from iris where name=:name')
    iris_df = pd.io.sql.read_sql(name_text, conn, params={'name': 'Iris-versicolor'})
    all_names = set(iris_df['Name'])
    assert all_names == {'Iris-versicolor'}

@pytest.mark.parametrize('conn', sqlalchemy_connectable_iris)
def test_query_by_select_obj(conn: Any, request: pytest.FixtureRequest) -> None:
    conn = request.getfixturevalue(conn)
    from sqlalchemy import bindparam, select
    iris = iris_table_metadata()  # Assuming iris_table_metadata is defined somewhere.
    name_select = select(iris).where(iris.c.Name == bindparam('name'))
    iris_df = pd.io.sql.read_sql(name_select, conn, params={'name': 'Iris-setosa'})
    all_names = set(iris_df['Name'])
    assert all_names == {'Iris-setosa'}

@pytest.mark.parametrize('conn', all_connectable)
def test_column_with_percentage(conn: Any, request: pytest.FixtureRequest) -> None:
    conn_name: str = conn
    if conn_name == 'sqlite_buildin':
        request.applymarker(pytest.mark.xfail(reason='Not Implemented'))
    conn = request.getfixturevalue(conn)
    df = DataFrame({'A': [0, 1, 2], '%_variation': [3, 4, 5]})
    df.to_sql(name='test_column_percentage', con=conn, index=False)
    res = pd.io.sql.read_sql_table('test_column_percentage', conn)
    tm.assert_frame_equal(res, df)

def test_sql_open_close(test_frame3: pd.DataFrame) -> None:
    from contextlib import closing
    import sqlite3
    with tm.ensure_clean() as name:
        with closing(sqlite3.connect(name)) as conn:
            assert pd.io.sql.to_sql(test_frame3, 'test_frame3_legacy', conn, index=False) == 4
        with closing(sqlite3.connect(name)) as conn:
            result = pd.io.sql.read_sql_query('SELECT * FROM test_frame3_legacy;', conn)
    tm.assert_frame_equal(test_frame3, result)

@td.skip_if_installed('sqlalchemy')
def test_con_string_import_error() -> None:
    from pandas.io import sql
    conn = 'mysql://root@localhost/pandas'
    msg = 'Using URI string without sqlalchemy installed'
    with pytest.raises(ImportError, match=msg):
        sql.read_sql('SELECT * FROM iris', conn)

@td.skip_if_installed('sqlalchemy')
def test_con_unknown_dbapi2_class_does_not_error_without_sql_alchemy_installed() -> None:

    class MockSqliteConnection:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.conn = sqlite3.Connection(*args, **kwargs)

        def __getattr__(self, name: str) -> Any:
            return getattr(self.conn, name)

        def close(self) -> None:
            self.conn.close()

    with contextlib.closing(MockSqliteConnection(':memory:')) as conn:
        with tm.assert_produces_warning(UserWarning, match='only supports SQLAlchemy'):
            pd.io.sql.read_sql('SELECT 1', conn)

def test_sqlite_read_sql_delegate(sqlite_buildin_iris: Any) -> None:
    conn = sqlite_buildin_iris
    iris_frame1 = pd.io.sql.read_sql_query('SELECT * FROM iris', conn)
    iris_frame2 = pd.io.sql.read_sql('SELECT * FROM iris', conn)
    tm.assert_frame_equal(iris_frame1, iris_frame2)
    msg = 'Execution failed on sql \'iris\': near "iris": syntax error'
    with pytest.raises(pd.io.sql.DatabaseError, match=msg):
        pd.io.sql.read_sql('iris', conn)

def test_get_schema2(test_frame1: pd.DataFrame) -> None:
    create_sql = pd.io.sql.get_schema(test_frame1, 'test')
    assert 'CREATE' in create_sql

def test_sqlite_type_mapping(sqlite_buildin: Any) -> None:
    conn = sqlite_buildin
    df = DataFrame({'time': to_datetime(['2014-12-12 01:54', '2014-12-11 02:54'], utc=True)})
    db = pd.io.sql.SQLiteDatabase(conn)
    table = pd.io.sql.SQLiteTable('test_type', db, frame=df)
    schema = table.sql_schema()
    for col in schema.split('\n'):
        if col.split()[0].strip('"') == 'time':
            assert col.split()[1] == 'TIMESTAMP'

@pytest.mark.parametrize('conn', sqlalchemy_connectable)
def test_create_table(conn: Any, request: pytest.FixtureRequest) -> None:
    if conn == 'sqlite_str':
        pytest.skip('sqlite_str has no inspection system')
    conn = request.getfixturevalue(conn)
    from sqlalchemy import inspect
    temp_frame = DataFrame({'one': [1.0, 2.0, 3.0, 4.0], 'two': [4.0, 3.0, 2.0, 1.0]})
    with pd.io.sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
        assert pandasSQL.to_sql(temp_frame, 'temp_frame') == 4
    insp = inspect(conn)
    assert insp.has_table('temp_frame')
    with pd.io.sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
        pandasSQL.drop_table('temp_frame')

@pytest.mark.parametrize('conn', sqlalchemy_connectable)
def test_drop_table(conn: Any, request: pytest.FixtureRequest) -> None:
    if conn == 'sqlite_str':
        pytest.skip('sqlite_str has no inspection system')
    conn = request.getfixturevalue(conn)
    from sqlalchemy import inspect
    temp_frame = DataFrame({'one': [1.0, 2.0, 3.0, 4.0], 'two': [4.0, 3.0, 2.0, 1.0]})
    with pd.io.sql.SQLDatabase(conn) as pandasSQL:
        with pandasSQL.run_transaction():
            assert pandasSQL.to_sql(temp_frame, 'temp_frame') == 4
        insp = inspect(conn)
        assert insp.has_table('temp_frame')
        with pandasSQL.run_transaction():
            pandasSQL.drop_table('temp_frame')
        try:
            insp.clear_cache()
        except AttributeError:
            pass
        assert not insp.has_table('temp_frame')

@pytest.mark.parametrize('conn', sqlalchemy_connectable)
def test_roundtrip(conn: Any, request: pytest.FixtureRequest, test_frame1: pd.DataFrame) -> None:
    if conn == 'sqlite_str':
        pytest.skip('sqlite_str has no inspection system')
    conn_name: str = conn
    conn = request.getfixturevalue(conn)
    pandasSQL = pd.io.sql.pandasSQL_builder(conn)
    with pandasSQL.run_transaction():
        assert pandasSQL.to_sql(test_frame1, 'test_frame_roundtrip') == 4
        result = pandasSQL.read_query('SELECT * FROM test_frame_roundtrip')
    if 'adbc' in conn_name:
        result = result.rename(columns={'__index_level_0__': 'level_0'})
    result.set_index('level_0', inplace=True)
    result.index.name = None
    tm.assert_frame_equal(result, test_frame1)

@pytest.mark.parametrize('conn', sqlalchemy_connectable_iris)
def test_execute_sql(conn: Any, request: pytest.FixtureRequest) -> None:
    conn = request.getfixturevalue(conn)
    with pd.io.sql.pandasSQL_builder(conn) as pandasSQL:
        with pandasSQL.run_transaction():
            iris_results = pandasSQL.execute('SELECT * FROM iris')
            row = iris_results.fetchone()
            iris_results.close()
    assert list(row) == [5.1, 3.5, 1.4, 0.2, 'Iris-setosa']

@pytest.mark.parametrize('conn', sqlalchemy_connectable_iris)
def test_sqlalchemy_read_table(conn: Any, request: pytest.FixtureRequest) -> None:
    conn = request.getfixturevalue(conn)
    iris_frame = pd.io.sql.read_sql_table('iris', con=conn)
    check_iris_frame(iris_frame)

@pytest.mark.parametrize('conn', sqlalchemy_connectable_iris)
def test_sqlalchemy_read_table_columns(conn: Any, request: pytest.FixtureRequest) -> None:
    conn = request.getfixturevalue(conn)
    iris_frame = pd.io.sql.read_sql_table('iris', con=conn, columns=['SepalLength', 'SepalLength'])
    tm.assert_index_equal(iris_frame.columns, Index(['SepalLength', 'SepalLength__1']))

@pytest.mark.parametrize('conn', sqlalchemy_connectable_iris)
def test_read_table_absent_raises(conn: Any, request: pytest.FixtureRequest) -> None:
    conn = request.getfixturevalue(conn)
    msg = 'Table this_doesnt_exist not found'
    with pytest.raises(ValueError, match=msg):
        pd.io.sql.read_sql_table('this_doesnt_exist', con=conn)

@pytest.mark.parametrize('conn', sqlalchemy_connectable_types)
def test_sqlalchemy_default_type_conversion(conn: Any, request: pytest.FixtureRequest) -> None:
    conn_name: str = conn
    if conn_name == 'sqlite_str':
        pytest.skip('types tables not created in sqlite_str fixture')
    elif 'mysql' in conn_name or 'sqlite' in conn_name:
        request.applymarker(pytest.mark.xfail(reason='boolean dtype not inferred properly'))
    conn = request.getfixturevalue(conn)
    df = pd.io.sql.read_sql_table('types', conn)
    assert issubclass(df.FloatCol.dtype.type, np.floating)
    assert issubclass(df.IntCol.dtype.type, np.integer)
    assert issubclass(df.BoolCol.dtype.type, np.bool_)
    assert issubclass(df.IntColWithNull.dtype.type, np.floating)
    assert issubclass(df.BoolColWithNull.dtype.type, object)

@pytest.mark.parametrize('conn', sqlalchemy_connectable)
def test_bigint(conn: Any, request: pytest.FixtureRequest) -> None:
    conn = request.getfixturevalue(conn)
    df = DataFrame(data={'i64': [2 ** 62]})
    assert df.to_sql(name='test_bigint', con=conn, index=False) == 1
    result = pd.io.sql.read_sql_table('test_bigint', conn)
    tm.assert_frame_equal(df, result)

@pytest.mark.parametrize('conn', sqlalchemy_connectable_types)
def test_default_date_load(conn: Any, request: pytest.FixtureRequest) -> None:
    conn_name: str = conn
    if conn_name == 'sqlite_str':
        pytest.skip('types tables not created in sqlite_str fixture')
    elif 'sqlite' in conn_name:
        request.applymarker(pytest.mark.xfail(reason='sqlite does not read date properly'))
    conn = request.getfixturevalue(conn)
    df = pd.io.sql.read_sql_table('types', conn)
    assert issubclass(df.DateCol.dtype.type, np.datetime64)

@pytest.mark.parametrize('conn', postgresql_connectable)
@pytest.mark.parametrize('parse_dates', [None, ['DateColWithTz']])
def test_datetime_with_timezone_query(conn: Any, request: pytest.FixtureRequest, parse_dates: Optional[List[str]]) -> None:
    conn = request.getfixturevalue(conn)
    expected = create_and_load_postgres_datetz(conn)
    df = pd.io.sql.read_sql_query('select * from datetz', conn, parse_dates=parse_dates)
    col = df.DateColWithTz
    tm.assert_series_equal(col, expected)

@pytest.mark.parametrize('conn', postgresql_connectable)
def test_datetime_with_timezone_query_chunksize(conn: Any, request: pytest.FixtureRequest) -> None:
    conn = request.getfixturevalue(conn)
    expected = create_and_load_postgres_datetz(conn)
    df = concat(list(pd.io.sql.read_sql_query('select * from datetz', conn, chunksize=1)), ignore_index=True)
    col = df.DateColWithTz
    tm.assert_series_equal(col, expected)

@pytest.mark.parametrize('conn', postgresql_connectable)
def test_datetime_with_timezone_table(conn: Any, request: pytest.FixtureRequest) -> None:
    conn = request.getfixturevalue(conn)
    expected = create_and_load_postgres_datetz(conn)
    result = pd.io.sql.read_sql_table('datetz', conn)
    exp_frame = expected.to_frame()
    tm.assert_frame_equal(result, exp_frame)

@pytest.mark.parametrize('conn', sqlalchemy_connectable)
def test_datetime_with_timezone_roundtrip(conn: Any, request: pytest.FixtureRequest) -> None:
    conn_name: str = conn
    conn = request.getfixturevalue(conn)
    expected = DataFrame({'A': date_range('2013-01-01 09:00:00', periods=3, tz='US/Pacific', unit='us')})
    assert expected.to_sql(name='test_datetime_tz', con=conn, index=False) == 3
    if 'postgresql' in conn_name:
        expected['A'] = expected['A'].dt.tz_convert('UTC')
    else:
        expected['A'] = expected['A'].dt.tz_localize(None)
    result = pd.io.sql.read_sql_table('test_datetime_tz', conn)
    tm.assert_frame_equal(result, expected)
    result = pd.io.sql.read_sql_query('SELECT * FROM test_datetime_tz', conn)
    if 'sqlite' in conn_name:
        assert isinstance(result.loc[0, 'A'], str)
        result['A'] = to_datetime(result['A']).dt.as_unit('us')
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('conn', sqlalchemy_connectable)
def test_out_of_bounds_datetime(conn: Any, request: pytest.FixtureRequest) -> None:
    conn = request.getfixturevalue(conn)
    data = DataFrame({'date': datetime(9999, 1, 1)}, index=[0])
    assert data.to_sql(name='test_datetime_obb', con=conn, index=False) == 1
    result = pd.io.sql.read_sql_table('test_datetime_obb', conn)
    expected = DataFrame(np.array([datetime(9999, 1, 1)], dtype='M8[us]'), columns=['date'])
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('conn', sqlalchemy_connectable)
def test_naive_datetimeindex_roundtrip(conn: Any, request: pytest.FixtureRequest) -> None:
    conn = request.getfixturevalue(conn)
    dates = date_range('2018-01-01', periods=5, freq='6h', unit='us')._with_freq(None)
    expected = DataFrame({'nums': range(5)}, index=dates)
    assert expected.to_sql(name='foo_table', con=conn, index_label='info_date') == 5
    result = pd.io.sql.read_sql_table('foo_table', conn, index_col='info_date')
    tm.assert_frame_equal(result, expected, check_names=False)

@pytest.mark.parametrize('conn', sqlalchemy_connectable_types)
def test_date_parsing(conn: Any, request: pytest.FixtureRequest) -> None:
    conn_name: str = conn
    conn = request.getfixturevalue(conn)
    df = pd.io.sql.read_sql_table('types', conn)
    expected_type = object if 'sqlite' in conn_name else np.datetime64
    assert issubclass(df.DateCol.dtype.type, expected_type)
    df = pd.io.sql.read_sql_table('types', conn, parse_dates=['DateCol'])
    assert issubclass(df.DateCol.dtype.type, np.datetime64)
    df = pd.io.sql.read_sql_table('types', conn, parse_dates={'DateCol': '%Y-%m-%d %H:%M:%S'})
    assert issubclass(df.DateCol.dtype.type, np.datetime64)
    df = pd.io.sql.read_sql_table('types', conn, parse_dates=['IntDateCol'])
    assert issubclass(df.IntDateCol.dtype.type, np.datetime64)
    df = pd.io.sql.read_sql_table('types', conn, parse_dates={'IntDateCol': 's'})
    assert issubclass(df.IntDateCol.dtype.type, np.datetime64)

@pytest.mark.parametrize('conn', sqlalchemy_connectable)
def test_datetime(conn: Any, request: pytest.FixtureRequest) -> None:
    conn_name: str = conn
    conn = request.getfixturevalue(conn)
    df = DataFrame({'A': date_range('2013-01-01 09:00:00', periods=3), 'B': np.arange(3.0)})
    assert df.to_sql(name='test_datetime', con=conn) == 3
    result = pd.io.sql.read_sql_table('test_datetime', conn)
    result = result.drop('index', axis=1)
    expected = df.copy()
    expected['A'] = expected['A'].astype('M8[us]')
    tm.assert_frame_equal(result, expected)
    result = pd.io.sql.read_sql_query('SELECT * FROM test_datetime', conn)
    result = result.drop('index', axis=1)
    if 'sqlite' in conn_name:
        assert isinstance(result.loc[0, 'A'], str)
        result['A'] = to_datetime(result['A'])
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('conn', sqlalchemy_connectable)
def test_datetime_NaT(conn: Any, request: pytest.FixtureRequest) -> None:
    conn_name: str = conn
    conn = request.getfixturevalue(conn)
    df = DataFrame({'A': date_range('2013-01-01 09:00:00', periods=3), 'B': np.arange(3.0)})
    df.loc[1, 'A'] = np.nan
    assert df.to_sql(name='test_datetime', con=conn, index=False) == 3
    result = pd.io.sql.read_sql_table('test_datetime', conn)
    expected = df.copy()
    expected['A'] = expected['A'].astype('M8[us]')
    tm.assert_frame_equal(result, expected)
    result = pd.io.sql.read_sql_query('SELECT * FROM test_datetime', conn)
    if 'sqlite' in conn_name:
        assert isinstance(result.loc[0, 'A'], str)
        result['A'] = to_datetime(result['A'], errors='coerce')
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('conn', sqlalchemy_connectable)
def test_datetime_date(conn: Any, request: pytest.FixtureRequest) -> None:
    conn = request.getfixturevalue(conn)
    df = DataFrame([date(2014, 1, 1), date(2014, 1, 2)], columns=['a'])
    assert df.to_sql(name='test_date', con=conn, index=False) == 2
    res = pd.io.sql.read_sql_table('test_date', conn)
    result = res['a']
    expected = to_datetime(df['a'])
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('conn', sqlalchemy_connectable)
def test_datetime_time(conn: Any, request: pytest.FixtureRequest, sqlite_buildin: Any) -> None:
    conn_name: str = conn
    conn = request.getfixturevalue(conn)
    df = DataFrame([time(9, 0, 0), time(9, 1, 30)], columns=['a'])
    assert df.to_sql(name='test_time', con=conn, index=False) == 2
    res = pd.io.sql.read_sql_table('test_time', conn)
    tm.assert_frame_equal(res, df)
    sqlite_conn = sqlite_buildin
    assert pd.io.sql.to_sql(df, 'test_time2', sqlite_conn, index=False) == 2
    res = pd.io.sql.read_sql_query('SELECT * FROM test_time2', sqlite_conn)
    ref = df.map(lambda _: _.strftime('%H:%M:%S.%f'))
    tm.assert_frame_equal(ref, res)
    assert pd.io.sql.to_sql(df, 'test_time3', conn, index=False) == 2
    if 'sqlite' in conn_name:
        res = pd.io.sql.read_sql_query('SELECT * FROM test_time3', conn)
        ref = df.map(lambda _: _.strftime('%H:%M:%S.%f'))
        tm.assert_frame_equal(ref, res)
    res = pd.io.sql.read_sql_table('test_time3', conn)
    tm.assert_frame_equal(df, res)

@pytest.mark.parametrize('conn', sqlalchemy_connectable)
def test_mixed_dtype_insert(conn: Any, request: pytest.FixtureRequest) -> None:
    conn = request.getfixturevalue(conn)
    s1 = Series(2 ** 25 + 1, dtype=np.int32)
    s2 = Series(0.0, dtype=np.float32)
    df = DataFrame({'s1': s1, 's2': s2})
    assert df.to_sql(name='test_read_write', con=conn, index=False) == 1
    df2 = pd.io.sql.read_sql_table('test_read_write', conn)
    tm.assert_frame_equal(df, df2, check_dtype=False, check_exact=True)

@pytest.mark.parametrize('conn', sqlalchemy_connectable)
def test_nan_numeric(conn: Any, request: pytest.FixtureRequest) -> None:
    conn = request.getfixturevalue(conn)
    df = DataFrame({'A': [0, 1, 2], 'B': [0.2, np.nan, 5.6]})
    assert df.to_sql(name='test_nan', con=conn, index=False) == 3
    result = pd.io.sql.read_sql_table('test_nan', conn)
    tm.assert_frame_equal(result, df)
    result = pd.io.sql.read_sql_query('SELECT * FROM test_nan', conn)
    tm.assert_frame_equal(result, df)

@pytest.mark.parametrize('conn', sqlalchemy_connectable)
def test_nan_fullcolumn(conn: Any, request: pytest.FixtureRequest) -> None:
    conn = request.getfixturevalue(conn)
    df = DataFrame({'A': [0, 1, 2], 'B': [np.nan, np.nan, np.nan]})
    assert df.to_sql(name='test_nan', con=conn, index=False) == 3
    result = pd.io.sql.read_sql_table('test_nan', conn)
    tm.assert_frame_equal(result, df)
    df['B'] = df['B'].astype('object')
    df['B'] = None
    result = pd.io.sql.read_sql_query('SELECT * FROM test_nan', conn)
    tm.assert_frame_equal(result, df)

@pytest.mark.parametrize('conn', sqlalchemy_connectable)
def test_nan_string(conn: Any, request: pytest.FixtureRequest) -> None:
    conn = request.getfixturevalue(conn)
    df = DataFrame({'A': [0, 1, 2], 'B': ['a', 'b', np.nan]})
    assert df.to_sql(name='test_nan', con=conn, index=False) == 3
    df.loc[2, 'B'] = None
    result = pd.io.sql.read_sql_table('test_nan', conn)
    tm.assert_frame_equal(result, df)
    result = pd.io.sql.read_sql_query('SELECT * FROM test_nan', conn)
    tm.assert_frame_equal(result, df)

@pytest.mark.parametrize('conn', sqlalchemy_connectable)
def test_to_sql_save_index(conn: Any, request: pytest.FixtureRequest) -> None:
    if 'adbc' in conn:
        request.node.add_marker(pytest.mark.xfail(reason='ADBC implementation does not create index', strict=True))
    conn_name: str = conn
    conn = request.getfixturevalue(conn)
    df = DataFrame.from_records([(1, 2.1, 'line1'), (2, 1.5, 'line2')], columns=['A', 'B', 'C'], index=['A'])
    tbl_name = 'test_to_sql_saves_index'
    with pd.io.sql.pandasSQL_builder(conn) as pandasSQL:
        with pandasSQL.run_transaction():
            assert pandasSQL.to_sql(df, tbl_name) == 2
    if conn_name in {'sqlite_buildin', 'sqlite_str'}:
        ixs = pd.io.sql.read_sql_query(f"SELECT * FROM sqlite_master WHERE type = 'index' AND tbl_name = '{tbl_name}'", conn)
        ix_cols: List[List[str]] = []
        for ix_name in ixs.name:
            ix_info = pd.io.sql.read_sql_query(f'PRAGMA index_info({ix_name})', conn)
            ix_cols.append(ix_info.name.tolist())
    else:
        from sqlalchemy import inspect
        insp = inspect(conn)
        ixs = insp.get_indexes(tbl_name)
        ix_cols = [i['column_names'] for i in ixs]
    assert ix_cols == [['A']]

@pytest.mark.parametrize('conn', sqlalchemy_connectable)
def test_transactions(conn: Any, request: pytest.FixtureRequest) -> None:
    conn_name: str = conn
    conn = request.getfixturevalue(conn)
    stmt = 'CREATE TABLE test_trans (A INT, B TEXT)'
    if conn_name != 'sqlite_buildin' and 'adbc' not in conn_name:
        from sqlalchemy import text
        stmt = text(stmt)
    with pd.io.sql.pandasSQL_builder(conn) as pandasSQL:
        with pandasSQL.run_transaction() as trans:
            trans.execute(stmt)

@pytest.mark.parametrize('conn', sqlalchemy_connectable)
def test_transaction_rollback(conn: Any, request: pytest.FixtureRequest) -> None:
    conn_name: str = conn
    conn = request.getfixturevalue(conn)
    with pd.io.sql.pandasSQL_builder(conn) as pandasSQL:
        with pandasSQL.run_transaction() as trans:
            stmt = 'CREATE TABLE test_trans (A INT, B TEXT)'
            if 'adbc' in conn_name or isinstance(pandasSQL, pd.io.sql.SQLiteDatabase):
                trans.execute(stmt)
            else:
                from sqlalchemy import text
                stmt = text(stmt)
                trans.execute(stmt)
        class DummyException(Exception):
            pass
        ins_sql = "INSERT INTO test_trans (A,B) VALUES (1, 'blah')"
        if isinstance(pandasSQL, pd.io.sql.SQLDatabase):
            from sqlalchemy import text
            ins_sql = text(ins_sql)
        try:
            with pandasSQL.run_transaction() as trans:
                trans.execute(ins_sql)
                raise DummyException('error')
        except DummyException:
            pass
        with pandasSQL.run_transaction():
            res = pandasSQL.read_query('SELECT * FROM test_trans')
        assert len(res) == 0
        with pandasSQL.run_transaction() as trans:
            trans.execute(ins_sql)
            res2 = pandasSQL.read_query('SELECT * FROM test_trans')
        assert len(res2) == 1

@pytest.mark.parametrize('conn', sqlalchemy_connectable)
def test_get_schema_create_table(conn: Any, request: pytest.FixtureRequest, test_frame3: pd.DataFrame) -> None:
    if conn == 'sqlite_str':
        request.applymarker(pytest.mark.xfail(reason='test does not support sqlite_str fixture'))
    conn = request.getfixturevalue(conn)
    from sqlalchemy import text
    from sqlalchemy.engine import Engine
    tbl = 'test_get_schema_create_table'
    create_sql = pd.io.sql.get_schema(test_frame3, tbl, con=conn)
    blank_test_df = test_frame3.iloc[:0]
    create_sql = text(create_sql)
    if isinstance(conn, Engine):
        with conn.connect() as newcon:
            with newcon.begin():
                newcon.execute(create_sql)
    else:
        conn.execute(create_sql)
    returned_df = pd.io.sql.read_sql_table(tbl, conn)
    tm.assert_frame_equal(returned_df, blank_test_df, check_index_type=False)

@pytest.mark.parametrize('conn', sqlalchemy_connectable)
def test_dtype(conn: Any, request: pytest.FixtureRequest) -> None:
    if conn == 'sqlite_str':
        pytest.skip('sqlite_str has no inspection system')
    conn = request.getfixturevalue(conn)
    from sqlalchemy import TEXT, String
    from sqlalchemy.schema import MetaData
    cols = ['A', 'B']
    data = [(0.8, True), (0.9, None)]
    df = DataFrame(data, columns=cols)
    assert df.to_sql(name='dtype_test', con=conn) == 2
    assert df.to_sql(name='dtype_test2', con=conn, dtype={'B': TEXT}) == 2
    meta = MetaData()
    meta.reflect(bind=conn)
    sqltype = meta.tables['dtype_test2'].columns['B'].type
    assert isinstance(sqltype, TEXT)
    msg = 'The type of B is not a SQLAlchemy type'
    with pytest.raises(ValueError, match=msg):
        df.to_sql(name='error', con=conn, dtype={'B': str})
    assert df.to_sql(name='dtype_test3', con=conn, dtype={'B': String(10)}) == 2
    meta.reflect(bind=conn)
    sqltype = meta.tables['dtype_test3'].columns['B'].type
    assert isinstance(sqltype, String)
    assert sqltype.length == 10
    assert df.to_sql(name='single_dtype_test', con=conn, dtype=TEXT) == 2
    meta.reflect(bind=conn)
    sqltypea = meta.tables['single_dtype_test'].columns['A'].type
    sqltypeb = meta.tables['single_dtype_test'].columns['B'].type
    assert isinstance(sqltypea, TEXT)
    assert isinstance(sqltypeb, TEXT)

@pytest.mark.parametrize('conn', sqlalchemy_connectable)
def test_notna_dtype(conn: Any, request: pytest.FixtureRequest) -> None:
    if conn == 'sqlite_str':
        pytest.skip('sqlite_str has no inspection system')
    conn_name: str = conn
    conn = request.getfixturevalue(conn)
    from sqlalchemy import Boolean, DateTime, Float, Integer
    from sqlalchemy.schema import MetaData
    cols = {
        'Bool': Series([True, None]),
        'Date': Series([datetime(2012, 5, 1), None]),
        'Int': Series([1, None], dtype='object'),
        'Float': Series([1.1, None])
    }
    df = DataFrame(cols)
    tbl = 'notna_dtype_test'
    assert df.to_sql(name=tbl, con=conn) == 2
    _ = pd.io.sql.read_sql_table(tbl, conn)
    meta = MetaData()
    meta.reflect(bind=conn)
    my_type = Integer if 'mysql' in conn_name else Boolean
    col_dict = meta.tables[tbl].columns
    assert isinstance(col_dict['Bool'].type, my_type)
    assert isinstance(col_dict['Date'].type, DateTime)
    assert isinstance(col_dict['Int'].type, Integer)
    assert isinstance(col_dict['Float'].type, Float)

@pytest.mark.parametrize('conn', sqlalchemy_connectable)
def test_double_precision(conn: Any, request: pytest.FixtureRequest) -> None:
    if conn == 'sqlite_str':
        pytest.skip('sqlite_str has no inspection system')
    conn = request.getfixturevalue(conn)
    from sqlalchemy import BigInteger, Float, Integer
    from sqlalchemy.schema import MetaData
    V = 1.2345678910111213
    df = DataFrame({
        'f32': Series([V], dtype='float32'),
        'f64': Series([V], dtype='float64'),
        'f64_as_f32': Series([V], dtype='float64'),
        'i32': Series([5], dtype='int32'),
        'i64': Series([5], dtype='int64')
    })
    assert df.to_sql(name='test_dtypes', con=conn, index=False, if_exists='replace', dtype={'f64_as_f32': Float(precision=23)}) == 1
    res = pd.io.sql.read_sql_table('test_dtypes', conn)
    assert np.round(df['f64'].iloc[0], 14) == np.round(res['f64'].iloc[0], 14)
    meta = MetaData()
    meta.reflect(bind=conn)
    col_dict = meta.tables['test_dtypes'].columns
    assert str(col_dict['f32'].type) == str(col_dict['f64_as_f32'].type)
    assert isinstance(col_dict['f32'].type, Float)
    assert isinstance(col_dict['f64'].type, Float)
    assert isinstance(col_dict['i32'].type, Integer)
    assert isinstance(col_dict['i64'].type, BigInteger)

@pytest.mark.parametrize('conn', sqlalchemy_connectable)
def test_connectable_issue_example(conn: Any, request: pytest.FixtureRequest) -> None:
    conn = request.getfixturevalue(conn)
    from sqlalchemy.engine import Engine
    def test_select(connection: Any) -> pd.DataFrame:
        query = 'SELECT test_foo_data FROM test_foo_data'
        return pd.io.sql.read_sql_query(query, con=connection)
    def test_append(connection: Any, data: pd.DataFrame) -> None:
        data.to_sql(name='test_foo_data', con=connection, if_exists='append')
    def test_connectable(conn_obj: Any) -> None:
        foo_data = test_select(conn_obj)
        test_append(conn_obj, foo_data)
    def main(connectable: Any) -> None:
        if isinstance(connectable, Engine):
            with connectable.connect() as conn_obj:
                with conn_obj.begin():
                    test_connectable(conn_obj)
        else:
            test_connectable(connectable)
    assert DataFrame({'test_foo_data': [0, 1, 2]}).to_sql(name='test_foo_data', con=conn) == 3
    main(conn)

@pytest.mark.parametrize('conn', sqlalchemy_connectable)
@pytest.mark.parametrize('input', [{'foo': [np.inf]}, {'foo': [-np.inf]}, {'foo': [-np.inf], 'infe0': ['bar']}])
def test_to_sql_with_negative_npinf(conn: Any, request: pytest.FixtureRequest, input: dict[str, Any]) -> None:
    df = DataFrame(input)
    conn_name: str = conn
    conn = request.getfixturevalue(conn)
    if 'mysql' in conn_name:
        import pymysql  # type: ignore
        if pd.core.dtypes.common.Version(pymysql.__version__) < pd.core.dtypes.common.Version('1.0.3') and 'infe0' in df.columns:
            mark = pytest.mark.xfail(reason='GH 36465')
            request.applymarker(mark)
        msg = 'inf cannot be used with MySQL'
        with pytest.raises(ValueError, match=msg):
            df.to_sql(name='foobar', con=conn, index=False)
    else:
        assert df.to_sql(name='foobar', con=conn, index=False) == 1
        res = pd.io.sql.read_sql_table('foobar', conn)
        tm.assert_equal(df, res)

@pytest.mark.parametrize('conn', sqlalchemy_connectable)
def test_temporary_table(conn: Any, request: pytest.FixtureRequest) -> None:
    if conn == 'sqlite_str':
        pytest.skip('test does not work with str connection')
    conn = request.getfixturevalue(conn)
    from sqlalchemy import Column, Integer, Unicode, select
    from sqlalchemy.orm import Session, declarative_base
    test_data: str = 'Hello, World!'
    expected = DataFrame({'spam': [test_data]})
    Base = declarative_base()
    class Temporary(Base):
        __tablename__ = 'temp_test'
        __table_args__ = {'prefixes': ['TEMPORARY']}
        id = Column(Integer, primary_key=True)
        spam = Column(Unicode(30), nullable=False)
    from sqlalchemy.orm import sessionmaker
    with Session(conn) as session:
        with session.begin():
            conn_obj = session.connection()
            Temporary.__table__.create(conn_obj)
            session.add(Temporary(spam=test_data))
            session.flush()
            df = pd.io.sql.read_sql_query(select(Temporary.spam), con=conn_obj)
    tm.assert_frame_equal(df, expected)

@pytest.mark.parametrize('conn', all_connectable)
def test_invalid_engine(conn: Any, request: pytest.FixtureRequest, test_frame1: pd.DataFrame) -> None:
    if conn == 'sqlite_buildin' or 'adbc' in conn:
        request.applymarker(pytest.mark.xfail(reason='SQLiteDatabase/ADBCDatabase does not raise for bad engine'))
    conn = request.getfixturevalue(conn)
    msg = "engine must be one of 'auto', 'sqlalchemy'"
    with pd.io.sql.pandasSQL_builder(conn) as pandasSQL:
        with pytest.raises(ValueError, match=msg):
            pandasSQL.to_sql(test_frame1, 'test_frame1', engine='bad_engine')

@pytest.mark.parametrize('conn', all_connectable)
def test_to_sql_with_sql_engine(conn: Any, request: pytest.FixtureRequest, test_frame1: pd.DataFrame) -> None:
    conn = request.getfixturevalue(conn)
    with pd.io.sql.pandasSQL_builder(conn) as pandasSQL:
        with pandasSQL.run_transaction():
            assert pandasSQL.to_sql(test_frame1, 'test_frame1', engine='auto') == 4
            assert pandasSQL.has_table('test_frame1')
    num_entries = len(test_frame1)
    num_rows = count_rows(conn, 'test_frame1')
    assert num_rows == num_entries

@pytest.mark.parametrize('conn', sqlalchemy_connectable)
def test_options_sqlalchemy(conn: Any, request: pytest.FixtureRequest, test_frame1: pd.DataFrame) -> None:
    conn = request.getfixturevalue(conn)
    with pd.option_context('io.sql.engine', 'sqlalchemy'):
        with pd.io.sql.pandasSQL_builder(conn) as pandasSQL:
            with pandasSQL.run_transaction():
                assert pandasSQL.to_sql(test_frame1, 'test_frame1') == 4
                assert pandasSQL.has_table('test_frame1')
        num_entries = len(test_frame1)
        num_rows = count_rows(conn, 'test_frame1')
        assert num_rows == num_entries

@pytest.mark.parametrize('conn', all_connectable)
def test_options_auto(conn: Any, request: pytest.FixtureRequest, test_frame1: pd.DataFrame) -> None:
    conn = request.getfixturevalue(conn)
    with pd.option_context('io.sql.engine', 'auto'):
        with pd.io.sql.pandasSQL_builder(conn) as pandasSQL:
            with pandasSQL.run_transaction():
                assert pandasSQL.to_sql(test_frame1, 'test_frame1') == 4
                assert pandasSQL.has_table('test_frame1')
        num_entries = len(test_frame1)
        num_rows = count_rows(conn, 'test_frame1')
        assert num_rows == num_entries

def test_options_get_engine() -> None:
    pytest.importorskip('sqlalchemy')
    from pandas.io.sql import get_engine, SQLAlchemyEngine
    assert isinstance(get_engine('sqlalchemy'), SQLAlchemyEngine)
    with pd.option_context('io.sql.engine', 'sqlalchemy'):
        assert isinstance(get_engine('auto'), SQLAlchemyEngine)
        assert isinstance(get_engine('sqlalchemy'), SQLAlchemyEngine)
    with pd.option_context('io.sql.engine', 'auto'):
        assert isinstance(get_engine('auto'), SQLAlchemyEngine)
        assert isinstance(get_engine('sqlalchemy'), SQLAlchemyEngine)

def test_get_engine_auto_error_message() -> None:
    pass

@pytest.mark.parametrize('conn', all_connectable)
@pytest.mark.parametrize('func', ['read_sql', 'read_sql_query'])
def test_read_sql_dtype_backend(conn: Any, request: pytest.FixtureRequest, string_storage: str, func: str, dtype_backend: str, dtype_backend_data: pd.DataFrame, dtype_backend_expected: Callable[[str, str, str], pd.DataFrame]) -> None:
    conn_name: str = conn
    conn = request.getfixturevalue(conn)
    table = 'test'
    df = dtype_backend_data
    df.to_sql(name=table, con=conn, index=False, if_exists='replace')
    with pd.option_context('mode.string_storage', string_storage):
        result = getattr(pd, func)(f'Select * from {table}', conn, dtype_backend=dtype_backend)
        expected = dtype_backend_expected(string_storage, dtype_backend, conn_name)
    tm.assert_frame_equal(result, expected)
    if 'adbc' in conn_name:
        request.applymarker(pytest.mark.xfail(reason='adbc does not support chunksize argument'))
    with pd.option_context('mode.string_storage', string_storage):
        iterator = getattr(pd, func)(f'Select * from {table}', con=conn, dtype_backend=dtype_backend, chunksize=3)
        expected = dtype_backend_expected(string_storage, dtype_backend, conn_name)
        for result in iterator:
            tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('conn', all_connectable)
@pytest.mark.parametrize('func', ['read_sql', 'read_sql_table'])
def test_read_sql_dtype_backend_table(conn: Any, request: pytest.FixtureRequest, string_storage: str, func: str, dtype_backend: str, dtype_backend_data: pd.DataFrame, dtype_backend_expected: Callable[[str, str, str], pd.DataFrame]) -> None:
    if 'sqlite' in conn and 'adbc' not in conn:
        request.applymarker(pytest.mark.xfail(reason='SQLite actually returns proper boolean values via read_sql_table, but before pytest refactor was skipped'))
    conn_name: str = conn
    conn = request.getfixturevalue(conn)
    table = 'test'
    df = dtype_backend_data
    df.to_sql(name=table, con=conn, index=False, if_exists='replace')
    with pd.option_context('mode.string_storage', string_storage):
        result = getattr(pd, func)(table, conn, dtype_backend=dtype_backend)
        expected = dtype_backend_expected(string_storage, dtype_backend, conn_name)
    tm.assert_frame_equal(result, expected)
    if 'adbc' in conn_name:
        return
    with pd.option_context('mode.string_storage', string_storage):
        iterator = getattr(pd, func)(table, conn, dtype_backend=dtype_backend, chunksize=3)
        expected = dtype_backend_expected(string_storage, dtype_backend, conn_name)
        for result in iterator:
            tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('conn', all_connectable)
@pytest.mark.parametrize('func', ['read_sql', 'read_sql_table', 'read_sql_query'])
def test_read_sql_invalid_dtype_backend_table(conn: Any, request: pytest.FixtureRequest, func: str, dtype_backend_data: pd.DataFrame) -> None:
    conn = request.getfixturevalue(conn)
    table = 'test'
    df = dtype_backend_data
    df.to_sql(name=table, con=conn, index=False, if_exists='replace')
    msg = "dtype_backend numpy is invalid, only 'numpy_nullable' and 'pyarrow' are allowed."
    with pytest.raises(ValueError, match=msg):
        getattr(pd, func)(table, conn, dtype_backend='numpy')

@pytest.mark.parametrize('conn', all_connectable)
def test_chunksize_empty_dtypes(conn: Any, request: pytest.FixtureRequest) -> None:
    if 'adbc' in conn:
        request.node.add_marker(pytest.mark.xfail(reason='chunksize argument NotImplemented with ADBC'))
    conn = request.getfixturevalue(conn)
    dtypes = {'a': 'int64', 'b': 'object'}
    df = DataFrame(columns=['a', 'b']).astype(dtypes)
    expected = df.copy()
    df.to_sql(name='test', con=conn, index=False, if_exists='replace')
    for result in pd.io.sql.read_sql_query('SELECT * FROM test', conn, dtype=dtypes, chunksize=1):
        tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('conn', all_connectable)
@pytest.mark.parametrize('dtype_backend', [pd.io.sql.lib.no_default, 'numpy_nullable'])
@pytest.mark.parametrize('func', ['read_sql', 'read_sql_query'])
def test_read_sql_dtype(conn: Any, request: pytest.FixtureRequest, func: str, dtype_backend: Union[str, Any]) -> None:
    conn = request.getfixturevalue(conn)
    table = 'test'
    df = DataFrame({'a': [1, 2, 3], 'b': 5})
    df.to_sql(name=table, con=conn, index=False, if_exists='replace')
    result = getattr(pd, func)(f'Select * from {table}', conn, dtype={'a': np.float64}, dtype_backend=dtype_backend)
    expected_dtype = 'int64' if dtype_backend != 'numpy_nullable' else 'Int64'
    expected = DataFrame({
        'a': Series([1, 2, 3], dtype=np.float64),
        'b': Series([5, 5, 5], dtype=expected_dtype)
    })
    tm.assert_frame_equal(result, expected)

def test_bigint_warning(sqlite_engine: Any) -> None:
    conn = sqlite_engine
    df = DataFrame({'a': [1, 2]}, dtype='int64')
    assert df.to_sql(name='test_bigintwarning', con=conn, index=False) == 2
    with tm.assert_produces_warning(None):
        pd.io.sql.read_sql_table('test_bigintwarning', conn)

def test_valueerror_exception(sqlite_engine: Any) -> None:
    conn = sqlite_engine
    df = DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    with pytest.raises(ValueError, match='Empty table name specified'):
        df.to_sql(name='', con=conn, if_exists='replace', index=False)

def test_row_object_is_named_tuple(sqlite_engine: Any) -> None:
    conn = sqlite_engine
    from sqlalchemy import Column, Integer, String
    from sqlalchemy.orm import declarative_base, sessionmaker
    BaseModel = declarative_base()
    class Test(BaseModel):
        __tablename__ = 'test_frame'
        id = Column(Integer, primary_key=True)
        string_column = Column(String(50))
    with conn.begin():
        BaseModel.metadata.create_all(conn)
    Session = sessionmaker(bind=conn)
    with Session() as session:
        df = DataFrame({'id': [0, 1], 'string_column': ['hello', 'world']})
        assert df.to_sql(name='test_frame', con=conn, index=False, if_exists='replace') == 2
        session.commit()
        test_query = session.query(Test.id, Test.string_column)
        df = DataFrame(test_query)
    assert list(df.columns) == ['id', 'string_column']

def test_read_sql_string_inference(sqlite_engine: Any) -> None:
    conn = sqlite_engine
    table = 'test'
    df = DataFrame({'a': ['x', 'y']})
    df.to_sql(table, con=conn, index=False, if_exists='replace')
    with pd.option_context('future.infer_string', True):
        result = pd.io.sql.read_sql_table(table, conn)
    dtype = pd.StringDtype(na_value=np.nan)
    expected = DataFrame({'a': ['x', 'y']}, dtype=dtype, columns=Index(['a'], dtype=dtype))
    tm.assert_frame_equal(result, expected)

def test_roundtripping_datetimes(sqlite_engine: Any) -> None:
    conn = sqlite_engine
    df = DataFrame({'t': [datetime(2020, 12, 31, 12)]}, dtype='datetime64[ns]')
    df.to_sql('test', conn, if_exists='replace', index=False)
    result = pd.read_sql('select * from test', conn).iloc[0, 0]
    assert result == '2020-12-31 12:00:00.000000'

@pytest.fixture
def sqlite_builtin_detect_types() -> Any:
    with contextlib.closing(sqlite3.connect(':memory:', detect_types=sqlite3.PARSE_DECLTYPES)) as closing_conn:
        with closing_conn as conn:
            yield conn

def test_roundtripping_datetimes_detect_types(sqlite_builtin_detect_types: Any) -> None:
    conn = sqlite_builtin_detect_types
    df = DataFrame({'t': [datetime(2020, 12, 31, 12)]}, dtype='datetime64[ns]')
    df.to_sql('test', conn, if_exists='replace', index=False)
    result = pd.read_sql('select * from test', conn).iloc[0, 0]
    assert result == Timestamp('2020-12-31 12:00:00.000000')

@pytest.mark.db
def test_psycopg2_schema_support(postgresql_psycopg2_engine: Any) -> None:
    conn = postgresql_psycopg2_engine
    df = DataFrame({'col1': [1, 2], 'col2': [0.1, 0.2], 'col3': ['a', 'n']})
    with conn.connect() as con:
        with con.begin():
            con.exec_driver_sql('DROP SCHEMA IF EXISTS other CASCADE;')
            con.exec_driver_sql('CREATE SCHEMA other;')
    assert df.to_sql(name='test_schema_public', con=conn, index=False) == 2
    assert df.to_sql(name='test_schema_public_explicit', con=conn, index=False, schema='public') == 2
    assert df.to_sql(name='test_schema_other', con=conn, index=False, schema='other') == 2
    res1 = pd.io.sql.read_sql_table('test_schema_public', conn)
    tm.assert_frame_equal(df, res1)
    res2 = pd.io.sql.read_sql_table('test_schema_public_explicit', conn)
    tm.assert_frame_equal(df, res2)
    res3 = pd.io.sql.read_sql_table('test_schema_public_explicit', conn, schema='public')
    tm.assert_frame_equal(df, res3)
    res4 = pd.io.sql.read_sql_table('test_schema_other', conn, schema='other')
    tm.assert_frame_equal(df, res4)
    msg = 'Table test_schema_other not found'
    with pytest.raises(ValueError, match=msg):
        pd.io.sql.read_sql_table('test_schema_other', conn, schema='public')
    with conn.connect() as con:
        with con.begin():
            con.exec_driver_sql('DROP SCHEMA IF EXISTS other CASCADE;')
            con.exec_driver_sql('CREATE SCHEMA other;')
    assert df.to_sql(name='test_schema_other', con=conn, schema='other', index=False) == 2
    df.to_sql(name='test_schema_other', con=conn, schema='other', index=False, if_exists='replace')
    assert df.to_sql(name='test_schema_other', con=conn, schema='other', index=False, if_exists='append') == 2
    res = pd.io.sql.read_sql_table('test_schema_other', conn, schema='other')
    tm.assert_frame_equal(concat([df, df], ignore_index=True), res)

@pytest.mark.db
def test_self_join_date_columns(postgresql_psycopg2_engine: Any) -> None:
    conn = postgresql_psycopg2_engine
    from sqlalchemy.sql import text
    create_table = text("""
    CREATE TABLE person
    (
        id serial constraint person_pkey primary key,
        created_dt timestamp with time zone
    );

    INSERT INTO person
        VALUES (1, '2021-01-01T00:00:00Z');
    """)
    with conn.connect() as con:
        with con.begin():
            con.execute(create_table)
    sql_query = 'SELECT * FROM "person" AS p1 INNER JOIN "person" AS p2 ON p1.id = p2.id;'
    result = pd.read_sql(sql_query, conn)
    expected = DataFrame([[1, Timestamp('2021', tz='UTC')] * 2], columns=['id', 'created_dt'] * 2)
    expected['created_dt'] = expected['created_dt'].astype('M8[us, UTC]')
    tm.assert_frame_equal(result, expected)
    with pd.io.sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
        pandasSQL.drop_table('person')

def test_create_and_drop_table(sqlite_engine: Any) -> None:
    conn = sqlite_engine
    temp_frame = DataFrame({'one': [1.0, 2.0, 3.0, 4.0], 'two': [4.0, 3.0, 2.0, 1.0]})
    with pd.io.sql.SQLDatabase(conn) as pandasSQL:
        with pandasSQL.run_transaction():
            assert pandasSQL.to_sql(temp_frame, 'drop_test_frame') == 4
        assert pandasSQL.has_table('drop_test_frame')
        with pandasSQL.run_transaction():
            pandasSQL.drop_table('drop_test_frame')
        assert not pandasSQL.has_table('drop_test_frame')
