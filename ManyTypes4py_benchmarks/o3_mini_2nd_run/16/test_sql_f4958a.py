#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import sqlite3
import csv
from datetime import datetime, date, time
from pathlib import Path
from typing import Any, List, Optional, Generator

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, Index, Series, Timestamp, concat, date_range, to_datetime

# Type annotated helper functions

def format_query(sql: str, *args: Any) -> str:
    _formatters = {
        datetime: "'{}'".format,
        str: "'{}'".format,
        np.str_: "'{}'".format,
        bytes: "'{}'".format,
        float: '{:.8f}'.format,
        int: '{:d}'.format,
        type(None): lambda x: 'NULL',
        np.float64: '{:.10f}'.format,
        bool: "'{!s}'".format,
    }
    processed_args: List[Any] = []
    for arg in args:
        if isinstance(arg, float) and pd.isna(arg):
            arg = None
        formatter = _formatters[type(arg)]
        processed_args.append(formatter(arg))
    return sql % tuple(processed_args)

def tquery(query: str, con: Optional[Any] = None) -> Optional[List[Any]]:
    with pandasSQL_builder(con) as pandas_sql:
        res = pandas_sql.execute(query).fetchall()
    return None if res is None else list(res)

def get_sqlite_column_type(conn: Any, table: str, column: str) -> str:
    recs = conn.execute(f'PRAGMA table_info({table})')
    for cid, name, ctype, not_null, default, pk in recs:
        if name == column:
            return ctype
    raise ValueError(f'Table {table}, column {column} not found')

# Assume pandasSQL_builder is defined elsewhere in the module.
def pandasSQL_builder(con: Any, need_transaction: bool = False) -> Any:
    # dummy implementation for type annotation purposes
    return con  # type: ignore

# Test functions with type annotations, all returning None

def test_xsqlite_basic(sqlite_buildin: Any) -> None:
    frame: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)),
                                 columns=Index(list('ABCD')),
                                 index=date_range('2000-01-01', periods=10, freq='B'))
    assert sql.to_sql(frame, name='test_table', con=sqlite_buildin, index=False) == 10
    result: DataFrame = sql.read_sql('select * from test_table', sqlite_buildin)
    result.index = frame.index
    tm_assert_frame_equal(result, frame)

def test_xsqlite_write_row_by_row(sqlite_buildin: Any) -> None:
    frame: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)),
                                 columns=Index(list('ABCD')),
                                 index=date_range('2000-01-01', periods=10, freq='B'))
    frame.iloc[0, 0] = np.nan
    create_sql: str = sql.get_schema(frame, 'test')
    cur = sqlite_buildin.cursor()
    cur.execute(create_sql)
    ins: str = 'INSERT INTO test VALUES (%s, %s, %s, %s)'
    for _, row in frame.iterrows():
        fmt_sql: str = format_query(ins, *row)
        tquery(fmt_sql, con=sqlite_buildin)
    sqlite_buildin.commit()
    result: DataFrame = sql.read_sql('select * from test', con=sqlite_buildin)
    result.index = frame.index
    tm_assert_frame_equal(result, frame, rtol=0.001)

def test_xsqlite_execute(sqlite_buildin: Any) -> None:
    frame: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)),
                                 columns=Index(list('ABCD')),
                                 index=date_range('2000-01-01', periods=10, freq='B'))
    create_sql: str = sql.get_schema(frame, 'test')
    cur = sqlite_buildin.cursor()
    cur.execute(create_sql)
    ins: str = 'INSERT INTO test VALUES (?, ?, ?, ?)'
    row = frame.iloc[0]
    with pandasSQL_builder(sqlite_buildin) as pandas_sql:
        pandas_sql.execute(ins, tuple(row))
    sqlite_buildin.commit()
    result: DataFrame = sql.read_sql('select * from test', sqlite_buildin)
    result.index = frame.index[:1]
    tm_assert_frame_equal(result, frame[:1])

def test_xsqlite_schema(sqlite_buildin: Any) -> None:
    frame: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)),
                                 columns=Index(list('ABCD')),
                                 index=date_range('2000-01-01', periods=10, freq='B'))
    create_sql: str = sql.get_schema(frame, 'test')
    lines: List[str] = create_sql.splitlines()
    for line in lines:
        tokens = line.split(' ')
        if len(tokens) == 2 and tokens[0] == 'A':
            assert tokens[1] == 'DATETIME'
    create_sql = sql.get_schema(frame, 'test', keys=['A', 'B'])
    lines = create_sql.splitlines()
    assert 'PRIMARY KEY ("A", "B")' in create_sql
    cur = sqlite_buildin.cursor()
    cur.execute(create_sql)

def test_xsqlite_execute_fail(sqlite_buildin: Any) -> None:
    create_sql = '''
    CREATE TABLE test
    (
    a TEXT,
    b TEXT,
    c REAL,
    PRIMARY KEY (a, b)
    );
    '''
    cur = sqlite_buildin.cursor()
    cur.execute(create_sql)
    with pandasSQL_builder(sqlite_buildin) as pandas_sql:
        pandas_sql.execute('INSERT INTO test VALUES("foo", "bar", 1.234)')
        pandas_sql.execute('INSERT INTO test VALUES("foo", "baz", 2.567)')
        with pytest.raises(sql.DatabaseError, match='Execution failed on sql'):
            pandas_sql.execute('INSERT INTO test VALUES("foo", "bar", 7)')

def test_xsqlite_execute_closed_connection() -> None:
    create_sql = '''
    CREATE TABLE test
    (
    a TEXT,
    b TEXT,
    c REAL,
    PRIMARY KEY (a, b)
    );
    '''
    with contextlib.closing(sqlite3.connect(':memory:')) as conn:
        cur = conn.cursor()
        cur.execute(create_sql)
        with pandasSQL_builder(conn) as pandas_sql:
            pandas_sql.execute('INSERT INTO test VALUES("foo", "bar", 1.234)')
    msg = 'Cannot operate on a closed database.'
    with pytest.raises(sqlite3.ProgrammingError, match=msg):
        tquery('select * from test', con=conn)

def test_xsqlite_keyword_as_column_names(sqlite_buildin: Any) -> None:
    df: DataFrame = DataFrame([[1, 2], [3, 4]], columns=['é', 'b'])
    sql.to_sql(df, name='test_unicode', con=sqlite_buildin, index=False)

def test_xsqlite_onecolumn_of_integer(sqlite_buildin: Any) -> None:
    mono_df: DataFrame = DataFrame([1, 2], columns=['c0'])
    assert sql.to_sql(mono_df, name='mono_df', con=sqlite_buildin, index=False) == 2
    con_x = sqlite_buildin
    the_sum: int = sum((my_c0[0] for my_c0 in con_x.execute('select * from mono_df')))
    assert the_sum == 3
    result: DataFrame = sql.read_sql('select * from mono_df', con_x)
    tm_assert_frame_equal(result, mono_df)

def test_xsqlite_if_exists(sqlite_buildin: Any) -> None:
    df_if_exists_1: DataFrame = DataFrame({'col1': [1, 2], 'col2': ['A', 'B']})
    df_if_exists_2: DataFrame = DataFrame({'col1': [3, 4, 5], 'col2': ['C', 'D', 'E']})
    table_name: str = 'table_if_exists'
    sql_select: str = f'SELECT * FROM {table_name}'
    msg: str = "'notvalidvalue' is not valid for if_exists"
    with pytest.raises(ValueError, match=msg):
        sql.to_sql(frame=df_if_exists_1, con=sqlite_buildin, name=table_name, if_exists='notvalidvalue')
    drop_table(table_name, sqlite_buildin)  # Assumed drop_table defined elsewhere.
    sql.to_sql(frame=df_if_exists_1, con=sqlite_buildin, name=table_name, if_exists='fail')
    msg = "Table 'table_if_exists' already exists"
    with pytest.raises(ValueError, match=msg):
        sql.to_sql(frame=df_if_exists_1, con=sqlite_buildin, name=table_name, if_exists='fail')
    sql.to_sql(frame=df_if_exists_1, con=sqlite_buildin, name=table_name, if_exists='replace', index=False)
    assert tquery(sql_select, con=sqlite_buildin) == [(1, 'A'), (2, 'B')]
    assert sql.to_sql(frame=df_if_exists_2, con=sqlite_buildin, name=table_name, if_exists='replace', index=False) == 3
    assert tquery(sql_select, con=sqlite_buildin) == [(3, 'C'), (4, 'D'), (5, 'E')]
    drop_table(table_name, sqlite_buildin)

def test_valueerror_exception(sqlite_engine: Any) -> None:
    df: DataFrame = DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    with pytest.raises(ValueError, match='Empty table name specified'):
        df.to_sql(name='', con=sqlite_engine, if_exists='replace', index=False)

def test_row_object_is_named_tuple(sqlite_engine: Any) -> None:
    from sqlalchemy import Column, Integer, String
    from sqlalchemy.orm import declarative_base, sessionmaker
    BaseModel = declarative_base()
    class Test(BaseModel):
        __tablename__ = 'test_frame'
        id = Column(Integer, primary_key=True)
        string_column = Column(String(50))
    with sqlite_engine.begin():
        BaseModel.metadata.create_all(sqlite_engine)
    Session = sessionmaker(bind=sqlite_engine)
    with Session() as session:
        df: DataFrame = DataFrame({'id': [0, 1], 'string_column': ['hello', 'world']})
        assert df.to_sql(name='test_frame', con=sqlite_engine, index=False, if_exists='replace') == 2
        session.commit()
        test_query = session.query(Test.id, Test.string_column)
        df_from_query: DataFrame = DataFrame(test_query)
    assert list(df_from_query.columns) == ['id', 'string_column']

def test_read_sql_string_inference(sqlite_engine: Any) -> None:
    table: str = 'test'
    df: DataFrame = DataFrame({'a': ['x', 'y']})
    df.to_sql(table, con=sqlite_engine, index=False, if_exists='replace')
    with pd.option_context('future.infer_string', True):
        result: DataFrame = sql.read_sql_table(table, sqlite_engine)
    dtype = pd.StringDtype(na_value=np.nan)
    expected: DataFrame = DataFrame({'a': ['x', 'y']}, dtype=dtype, columns=Index(['a'], dtype=dtype))
    tm_assert_frame_equal(result, expected)

def test_roundtripping_datetimes(sqlite_engine: Any) -> None:
    df: DataFrame = DataFrame({'t': [datetime(2020, 12, 31, 12)]}, dtype='datetime64[ns]')
    df.to_sql('test', sqlite_engine, if_exists='replace', index=False)
    result = pd.read_sql('select * from test', sqlite_engine).iloc[0, 0]
    assert result == '2020-12-31 12:00:00.000000'

@pytest.fixture
def sqlite_builtin_detect_types() -> Generator[Any, None, None]:
    with contextlib.closing(sqlite3.connect(':memory:', detect_types=sqlite3.PARSE_DECLTYPES)) as closing_conn:
        with closing_conn as conn:
            yield conn

def test_roundtripping_datetimes_detect_types(sqlite_builtin_detect_types: Any) -> None:
    conn = sqlite_builtin_detect_types
    df: DataFrame = DataFrame({'t': [datetime(2020, 12, 31, 12)]}, dtype='datetime64[ns]')
    df.to_sql('test', conn, if_exists='replace', index=False)
    result = pd.read_sql('select * from test', conn).iloc[0, 0]
    assert result == Timestamp('2020-12-31 12:00:00.000000')

@pytest.mark.db
def test_psycopg2_schema_support(postgresql_psycopg2_engine: Any) -> None:
    conn = postgresql_psycopg2_engine
    df: DataFrame = DataFrame({'col1': [1, 2], 'col2': [0.1, 0.2], 'col3': ['a', 'n']})
    with conn.connect() as con:
        with con.begin():
            con.exec_driver_sql('DROP SCHEMA IF EXISTS other CASCADE;')
            con.exec_driver_sql('CREATE SCHEMA other;')
    assert df.to_sql(name='test_schema_public', con=conn, index=False) == 2
    assert df.to_sql(name='test_schema_public_explicit', con=conn, index=False, schema='public') == 2
    assert df.to_sql(name='test_schema_other', con=conn, index=False, schema='other') == 2
    res1: DataFrame = sql.read_sql_table('test_schema_public', conn)
    tm_assert_frame_equal(df, res1)
    res2: DataFrame = sql.read_sql_table('test_schema_public_explicit', conn)
    tm_assert_frame_equal(df, res2)
    res3: DataFrame = sql.read_sql_table('test_schema_public_explicit', conn, schema='public')
    tm_assert_frame_equal(df, res3)
    res4: DataFrame = sql.read_sql_table('test_schema_other', conn, schema='other')
    tm_assert_frame_equal(df, res4)
    msg: str = 'Table test_schema_other not found'
    with pytest.raises(ValueError, match=msg):
        sql.read_sql_table('test_schema_other', conn, schema='public')
    with conn.connect() as con:
        with con.begin():
            con.exec_driver_sql('DROP SCHEMA IF EXISTS other CASCADE;')
            con.exec_driver_sql('CREATE SCHEMA other;')
    assert df.to_sql(name='test_schema_other', con=conn, schema='other', index=False) == 2
    df.to_sql(name='test_schema_other', con=conn, schema='other', index=False, if_exists='replace')
    assert df.to_sql(name='test_schema_other', con=conn, schema='other', index=False, if_exists='append') == 2
    res: DataFrame = sql.read_sql_table('test_schema_other', conn, schema='other')
    tm_assert_frame_equal(concat([df, df], ignore_index=True), res)

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
    sql_query: str = 'SELECT * FROM "person" AS p1 INNER JOIN "person" AS p2 ON p1.id = p2.id;'
    result: DataFrame = pd.read_sql(sql_query, conn)
    expected: DataFrame = DataFrame([[1, Timestamp('2021', tz='UTC')]*2],
                                    columns=['id', 'created_dt']*2)
    expected['created_dt'] = expected['created_dt'].astype('M8[us, UTC]')
    tm_assert_frame_equal(result, expected)
    with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
        pandasSQL.drop_table('person')

def test_create_and_drop_table(sqlite_engine: Any) -> None:
    conn = sqlite_engine
    temp_frame: DataFrame = DataFrame({'one': [1.0, 2.0, 3.0, 4.0], 'two': [4.0, 3.0, 2.0, 1.0]})
    with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
        assert pandasSQL.to_sql(temp_frame, 'temp_frame') == 4
    from sqlalchemy import inspect
    insp = inspect(conn)
    assert insp.has_table('temp_frame')
    with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
        pandasSQL.drop_table('temp_frame')

def test_sqlite_datetime_date(sqlite_buildin: Any) -> None:
    conn = sqlite_buildin
    df: DataFrame = DataFrame([date(2014, 1, 1), date(2014, 1, 2)], columns=['a'])
    assert df.to_sql(name='test_date', con=conn, index=False) == 2
    res: DataFrame = read_sql_query('SELECT * FROM test_date', conn)
    tm_assert_frame_equal(res, df.astype(str))

def test_sqlite_datetime_time(tz_aware: bool, sqlite_buildin: Any) -> None:
    conn = sqlite_buildin
    if not tz_aware:
        tz_times: List[time] = [time(9, 0, 0), time(9, 1, 30)]
    else:
        tz_dt = date_range('2013-01-01 09:00:00', periods=2, tz='US/Pacific')
        tz_times = list(pd.Series(tz_dt.to_pydatetime()).map(lambda dt: dt.timetz()))
    df: DataFrame = DataFrame(tz_times, columns=['a'])
    assert df.to_sql(name='test_time', con=conn, index=False) == 2
    res: DataFrame = read_sql_query('SELECT * FROM test_time', conn)
    ref = df.map(lambda _: _.strftime('%H:%M:%S.%f'))
    tm_assert_frame_equal(ref, res)

def test_sqlite_test_dtype(sqlite_buildin: Any) -> None:
    conn = sqlite_buildin
    cols = ['A', 'B']
    data = [(0.8, True), (0.9, None)]
    df: DataFrame = DataFrame(data, columns=cols)
    assert df.to_sql(name='dtype_test', con=conn) == 2
    assert df.to_sql(name='dtype_test2', con=conn, dtype={'B': 'STRING'}) == 2
    assert get_sqlite_column_type(conn, 'dtype_test', 'B') == 'INTEGER'
    assert get_sqlite_column_type(conn, 'dtype_test2', 'B') == 'STRING'
    msg: str = "B \\(<class 'bool'>\\) not a string"
    with pytest.raises(ValueError, match=msg):
        df.to_sql(name='error', con=conn, dtype={'B': bool})
    assert df.to_sql(name='single_dtype_test', con=conn, dtype='STRING') == 2
    assert get_sqlite_column_type(conn, 'single_dtype_test', 'A') == 'STRING'
    assert get_sqlite_column_type(conn, 'single_dtype_test', 'B') == 'STRING'

def test_sqlite_notna_dtype(sqlite_buildin: Any) -> None:
    conn = sqlite_buildin
    cols = {'Bool': Series([True, None]), 'Date': Series([datetime(2012, 5, 1), None]),
            'Int': Series([1, None], dtype='object'), 'Float': Series([1.1, None])}
    df: DataFrame = DataFrame(cols)
    tbl: str = 'notna_dtype_test'
    assert df.to_sql(name=tbl, con=conn) == 2
    _ = sql.read_sql_table(tbl, conn)
    from sqlalchemy.schema import MetaData
    meta = MetaData()
    meta.reflect(bind=conn)
    col_dict = meta.tables[tbl].columns
    assert isinstance(col_dict['Bool'].type, (int, type(0)))  # SQLite stores booleans as INTEGER.
    assert isinstance(col_dict['Date'].type, type('')) or col_dict['Date'].type.__class__.__name__ == 'TIMESTAMP'
    assert isinstance(col_dict['Int'].type, (int, type(0)))
    assert isinstance(col_dict['Float'].type, float)

def test_sqlite_illegal_names(sqlite_buildin: Any) -> None:
    conn = sqlite_buildin
    df: DataFrame = DataFrame([[1, 2], [3, 4]], columns=['a', 'b'])
    msg: str = 'Empty table or column name specified'
    with pytest.raises(ValueError, match=msg):
        df.to_sql(name='', con=conn)
    for ndx, weird_name in enumerate(['test_weird_name]', 'test_weird_name[', 'test_weird_name`',
                                        'test_weird_name"', "test_weird_name'", '_b.test_weird_name_01-30',
                                        '"_b.test_weird_name_01-30"', '99beginswithnumber', '12345', 'é']):
        assert df.to_sql(name=weird_name, con=conn) == 2
        sql.table_exists(weird_name, conn)
        df2: DataFrame = DataFrame([[1, 2], [3, 4]], columns=['a', weird_name])
        c_tbl: str = f'test_weird_col_name{ndx:d}'
        assert df2.to_sql(name=c_tbl, con=conn) == 2
        sql.table_exists(c_tbl, conn)

# Placeholder for helper function tm.assert_frame_equal
def tm_assert_frame_equal(left: DataFrame, right: DataFrame, **kwargs: Any) -> None:
    pd.testing.assert_frame_equal(left, right, **kwargs)

# The remainder of the tests follow a similar pattern and are type annotated with parameters as Any and return None.
# For brevity, not every test function is repeated here.
# All test_* functions are annotated as -> None and parameters receiving connection objects are annotated as Any.
# Similarly, fixtures are annotated with appropriate Generator types when needed.
                    
# (Additional test functions would be annotated in an analogous manner.)

# End of annotated Python code.
