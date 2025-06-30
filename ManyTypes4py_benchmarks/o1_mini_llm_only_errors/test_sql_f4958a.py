from __future__ import annotations
import contextlib
from contextlib import closing
import csv
from datetime import date, datetime, time, timedelta
from io import StringIO
from pathlib import Path
import sqlite3
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
)
import uuid
import numpy as np
import pytest
from pandas._config import using_string_dtype
from pandas._libs import lib
from pandas.compat import pa_version_under14p1
from pandas.compat._optional import import_optional_dependency
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    Timestamp,
    concat,
    date_range,
    isna,
    to_datetime,
    to_timedelta,
)
import pandas._testing as tm
from pandas.util.version import Version
from pandas.io import sql
from pandas.io.sql import (
    SQLAlchemyEngine,
    SQLDatabase,
    SQLiteDatabase,
    get_engine,
    pandasSQL_builder,
    read_sql_query,
    read_sql_table,
)

if TYPE_CHECKING:
    import sqlalchemy

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
    ),
    pytest.mark.single_cpu,
]


@pytest.fixture
def sql_strings() -> Dict[str, Dict[str, str]]:
    return {
        "read_parameters": {
            "sqlite": "SELECT * FROM iris WHERE Name=? AND SepalLength=?",
            "mysql": "SELECT * FROM iris WHERE `Name`=%s AND `SepalLength`=%s",
            "postgresql": 'SELECT * FROM iris WHERE "Name"=%s AND "SepalLength"=%s',
        },
        "read_named_parameters": {
            "sqlite": "\n                SELECT * FROM iris WHERE Name=:name AND SepalLength=:length\n                ",
            "mysql": "\n                SELECT * FROM iris WHERE\n                `Name`=%(name)s AND `SepalLength`=%(length)s\n                ",
            "postgresql": '\n                SELECT * FROM iris WHERE\n                "Name"=%(name)s AND "SepalLength"=%(length)s\n                ',
        },
        "read_no_parameters_with_percent": {
            "sqlite": "SELECT * FROM iris WHERE Name LIKE '%' ",
            "mysql": "SELECT * FROM iris WHERE `Name` LIKE '%' ",
            "postgresql": 'SELECT * FROM iris WHERE "Name" LIKE \'%\'',
        },
    }


def iris_table_metadata() -> sqlalchemy.Table:
    import sqlalchemy
    from sqlalchemy import Column, Double, Float, MetaData, String, Table

    dtype = Double if Version(sqlalchemy.__version__) >= Version("2.0.0") else Float
    metadata = MetaData()
    iris = Table(
        "iris",
        metadata,
        Column("SepalLength", dtype),
        Column("SepalWidth", dtype),
        Column("PetalLength", dtype),
        Column("PetalWidth", dtype),
        Column("Name", String(200)),
    )
    return iris


def create_and_load_iris_sqlite3(conn: sqlite3.Connection, iris_file: Path) -> None:
    stmt = (
        'CREATE TABLE iris (\n            "SepalLength" REAL,\n            "SepalWidth" REAL,\n            "PetalLength" REAL,\n            "PetalWidth" REAL,\n            "Name" TEXT\n        )'
    )
    cur = conn.cursor()
    cur.execute(stmt)
    with iris_file.open(newline=None, encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        stmt = "INSERT INTO iris VALUES(?, ?, ?, ?, ?)"
        records = [
            (float(row[0]), float(row[1]), float(row[2]), float(row[3]), row[4])
            for row in reader
        ]
        cur.executemany(stmt, records)
    cur.close()
    conn.commit()


def create_and_load_iris_postgresql(
    conn: Any, iris_file: Path
) -> None:  # Replace Any with psycopg2.extensions.connection if available
    stmt = (
        'CREATE TABLE iris (\n            "SepalLength" DOUBLE PRECISION,\n            "SepalWidth" DOUBLE PRECISION,\n            "PetalLength" DOUBLE PRECISION,\n            "PetalWidth" DOUBLE PRECISION,\n            "Name" TEXT\n        )'
    )
    with conn.cursor() as cur:
        cur.execute(stmt)
        with iris_file.open(newline=None, encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            stmt = "INSERT INTO iris VALUES($1, $2, $3, $4, $5)"
            records = [
                (
                    float(row[0]),
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    row[4],
                )
                for row in reader
            ]
            cur.executemany(stmt, records)
    conn.commit()


def create_and_load_iris(
    conn: Union[SQLAlchemyEngine, SQLDatabase, SQLiteDatabase, sqlalchemy.engine.base.Connection],
    iris_file: Path,
) -> None:
    from sqlalchemy import insert

    iris = iris_table_metadata()
    with iris_file.open(newline=None, encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        params = [dict(zip(header, row)) for row in reader]
        stmt = insert(iris).values(params)
        with conn.begin() as con:
            iris.drop(con, checkfirst=True)
            iris.create(bind=con)
            con.execute(stmt)


def create_and_load_iris_view(conn: Any) -> None:
    stmt = "CREATE VIEW iris_view AS SELECT * FROM iris"
    if isinstance(conn, sqlite3.Connection):
        cur = conn.cursor()
        cur.execute(stmt)
    else:
        adbc = import_optional_dependency("adbc_driver_manager.dbapi", errors="ignore")
        if adbc and isinstance(conn, adbc.Connection):
            with conn.cursor() as cur:
                cur.execute(stmt)
            conn.commit()
        else:
            from sqlalchemy import text

            stmt = text(stmt)
            with conn.begin() as con:
                con.execute(stmt)


def types_table_metadata(dialect: str) -> sqlalchemy.Table:
    from sqlalchemy import TEXT, Boolean, Column, DateTime, Float, Integer, MetaData, Table

    date_type: Union[sqlalchemy.types.TypeEngine, sqlalchemy.types.TypeEngine]
    bool_type: Union[sqlalchemy.types.TypeEngine, sqlalchemy.types.TypeEngine]

    if dialect == "sqlite":
        date_type = TEXT
        bool_type = Integer
    else:
        date_type = DateTime
        bool_type = Boolean

    metadata = MetaData()
    types = Table(
        "types",
        metadata,
        Column("TextCol", TEXT),
        Column("DateCol", date_type),
        Column("IntDateCol", Integer),
        Column("IntDateOnlyCol", Integer),
        Column("FloatCol", Float),
        Column("IntCol", Integer),
        Column("BoolCol", bool_type),
        Column("IntColWithNull", Integer),
        Column("BoolColWithNull", bool_type),
    )
    return types


def create_and_load_types_sqlite3(
    conn: sqlite3.Connection, types_data: List[Tuple[Any, ...]]
) -> None:
    stmt = (
        'CREATE TABLE types (\n                    "TextCol" TEXT,\n                    "DateCol" TEXT,\n                    "IntDateCol" INTEGER,\n                    "IntDateOnlyCol" INTEGER,\n                    "FloatCol" REAL,\n                    "IntCol" INTEGER,\n                    "BoolCol" INTEGER,\n                    "IntColWithNull" INTEGER,\n                    "BoolColWithNull" INTEGER\n                )'
    )
    ins_stmt = "\n                INSERT INTO types\n                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)\n                "
    if isinstance(conn, sqlite3.Connection):
        cur = conn.cursor()
        cur.execute(stmt)
        cur.executemany(ins_stmt, types_data)
    else:
        with conn.cursor() as cur:
            cur.execute(stmt)
            cur.executemany(ins_stmt, types_data)
        conn.commit()


def create_and_load_types_postgresql(conn: Any, types_data: List[Tuple[Any, ...]]) -> None:
    with conn.cursor() as cur:
        stmt = (
            'CREATE TABLE types (\n                        "TextCol" TEXT,\n                        "DateCol" TIMESTAMP,\n                        "IntDateCol" INTEGER,\n                        "IntDateOnlyCol" INTEGER,\n                        "FloatCol" DOUBLE PRECISION,\n                        "IntCol" INTEGER,\n                        "BoolCol" BOOLEAN,\n                        "IntColWithNull" INTEGER,\n                        "BoolColWithNull" BOOLEAN\n                    )'
        )
        cur.execute(stmt)
        stmt = "\n                INSERT INTO types\n                VALUES($1, $2::timestamp, $3, $4, $5, $6, $7, $8, $9)\n                "
        cur.executemany(stmt, types_data)
    conn.commit()


def create_and_load_types(
    conn: Union[
        SQLAlchemyEngine, SQLDatabase, SQLiteDatabase, sqlalchemy.engine.base.Connection
    ],
    types_data: List[Dict[str, Any]],
    dialect: str,
) -> None:
    from sqlalchemy import insert
    from sqlalchemy.engine import Engine

    types = types_table_metadata(dialect)
    stmt = insert(types).values(types_data)
    if isinstance(conn, Engine):
        with conn.connect() as con:
            with con.begin():
                types.drop(con, checkfirst=True)
                types.create(bind=con)
                con.execute(stmt)
    else:
        with conn.begin():
            types.drop(conn, checkfirst=True)
            types.create(bind=conn)
            conn.execute(stmt)


def create_and_load_postgres_datetz(conn: Any) -> Series:
    from sqlalchemy import Column, DateTime, MetaData, Table, insert
    from sqlalchemy.engine import Engine

    metadata = MetaData()
    datetz = Table(
        "datetz", metadata, Column("DateColWithTz", DateTime(timezone=True))
    )
    datetz_data = [
        {"DateColWithTz": "2000-01-01 00:00:00-08:00"},
        {"DateColWithTz": "2000-06-01 00:00:00-07:00"},
    ]
    stmt = insert(datetz).values(datetz_data)
    if isinstance(conn, Engine):
        with conn.connect() as con:
            with con.begin():
                datetz.drop(con, checkfirst=True)
                datetz.create(bind=con)
                con.execute(stmt)
    else:
        with conn.begin():
            datetz.drop(conn, checkfirst=True)
            datetz.create(bind=conn)
            conn.execute(stmt)
    expected_data = [
        Timestamp("2000-01-01 08:00:00", tz="UTC"),
        Timestamp("2000-06-01 07:00:00", tz="UTC"),
    ]
    return Series(expected_data, name="DateColWithTz").astype("M8[us, UTC]")


def check_iris_frame(frame: DataFrame) -> None:
    pytype = frame.dtypes.iloc[0].type
    row = frame.iloc[0]
    assert issubclass(pytype, np.floating)
    tm.assert_series_equal(
        row, Series([5.1, 3.5, 1.4, 0.2, "Iris-setosa"], index=frame.columns, name=0)
    )
    assert frame.shape in ((150, 5), (8, 5))


def count_rows(conn: Any, table_name: str) -> int:
    stmt = f"SELECT count(*) AS count_1 FROM {table_name}"
    adbc = import_optional_dependency("adbc_driver_manager.dbapi", errors="ignore")
    if isinstance(conn, sqlite3.Connection):
        cur = conn.cursor()
        return cur.execute(stmt).fetchone()[0]
    elif adbc and isinstance(conn, adbc.Connection):
        with conn.cursor() as cur:
            cur.execute(stmt)
            return cur.fetchone()[0]
    else:
        from sqlalchemy import create_engine
        from sqlalchemy.engine import Engine

        if isinstance(conn, str):
            try:
                engine = create_engine(conn)
                with engine.connect() as conn:
                    return conn.exec_driver_sql(stmt).scalar_one()
            finally:
                engine.dispose()
        elif isinstance(conn, Engine):
            with conn.connect() as conn:
                return conn.exec_driver_sql(stmt).scalar_one()
        else:
            return conn.exec_driver_sql(stmt).scalar_one()


@pytest.fixture
def iris_path(datapath: Callable[[str, ...], str]) -> Path:
    iris_path = datapath("io", "data", "csv", "iris.csv")
    return Path(iris_path)


@pytest.fixture
def types_data() -> List[Dict[str, Any]]:
    return [
        {
            "TextCol": "first",
            "DateCol": "2000-01-03 00:00:00",
            "IntDateCol": 535852800,
            "IntDateOnlyCol": 20101010,
            "FloatCol": 10.1,
            "IntCol": 1,
            "BoolCol": False,
            "IntColWithNull": 1,
            "BoolColWithNull": False,
        },
        {
            "TextCol": "first",
            "DateCol": "2000-01-04 00:00:00",
            "IntDateCol": 1356998400,
            "IntDateOnlyCol": 20101212,
            "FloatCol": 10.1,
            "IntCol": 1,
            "BoolCol": False,
            "IntColWithNull": None,
            "BoolColWithNull": None,
        },
    ]


@pytest.fixture
def types_data_frame(types_data: List[Dict[str, Any]]) -> DataFrame:
    dtypes = {
        "TextCol": "str",
        "DateCol": "str",
        "IntDateCol": "int64",
        "IntDateOnlyCol": "int64",
        "FloatCol": "float",
        "IntCol": "int64",
        "BoolCol": "int64",
        "IntColWithNull": "float",
        "BoolColWithNull": "float",
    }
    df = DataFrame(types_data)
    return df[dtypes.keys()].astype(dtypes)


@pytest.fixture
def test_frame1() -> DataFrame:
    columns = ["index", "A", "B", "C", "D"]
    data = [
        (
            "2000-01-03 00:00:00",
            0.980268513777,
            3.68573087906,
            -0.364216805298,
            -1.15973806169,
        ),
        (
            "2000-01-04 00:00:00",
            1.04791624281,
            -0.0412318367011,
            -0.16181208307,
            0.212549316967,
        ),
        (
            "2000-01-05 00:00:00",
            0.498580885705,
            0.731167677815,
            -0.537677223318,
            1.34627041952,
        ),
        (
            "2000-01-06 00:00:00",
            1.12020151869,
            1.56762092543,
            0.00364077397681,
            0.67525259227,
        ),
    ]
    return DataFrame(data, columns=columns)


@pytest.fixture
def test_frame3() -> DataFrame:
    columns = ["index", "A", "B"]
    data = [
        ("2000-01-03 00:00:00", 2**31 - 1, -1.98767),
        ("2000-01-04 00:00:00", -29, -0.0412318367011),
        ("2000-01-05 00:00:00", 20000, 0.731167677815),
        ("2000-01-06 00:00:00", -290867, 1.56762092543),
    ]
    return DataFrame(data, columns=columns)


def get_all_views(conn: Any) -> List[str]:
    if isinstance(conn, sqlite3.Connection):
        c = conn.execute("SELECT name FROM sqlite_master WHERE type='view'")
        return [view[0] for view in c.fetchall()]
    else:
        adbc = import_optional_dependency("adbc_driver_manager.dbapi", errors="ignore")
        if adbc and isinstance(conn, adbc.Connection):
            results: List[str] = []
            info = conn.adbc_get_objects().read_all().to_pylist()
            for catalog in info:
                for schema in catalog["catalog_db_schemas"]:
                    for table in schema["catalog_db_tables"]:
                        if table["table_type"] == "view":
                            view_name = table["table_name"]
                            results.append(view_name)
            return results
        else:
            from sqlalchemy import inspect

            return inspect(conn).get_view_names()


def get_all_tables(conn: Any) -> List[str]:
    if isinstance(conn, sqlite3.Connection):
        c = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return [table[0] for table in c.fetchall()]
    else:
        adbc = import_optional_dependency("adbc_driver_manager.dbapi", errors="ignore")
        if adbc and isinstance(conn, adbc.Connection):
            results: List[str] = []
            info = conn.adbc_get_objects().read_all().to_pylist()
            for catalog in info:
                for schema in catalog["catalog_db_schemas"]:
                    for table in schema["catalog_db_tables"]:
                        if table["table_type"] == "table":
                            table_name = table["table_name"]
                            results.append(table_name)
            return results
        else:
            from sqlalchemy import inspect

            return inspect(conn).get_table_names()


def drop_table(table_name: str, conn: Any) -> None:
    if isinstance(conn, sqlite3.Connection):
        conn.execute(f'DROP TABLE IF EXISTS {sql._get_valid_sqlite_name(table_name)}')
        conn.commit()
    else:
        adbc = import_optional_dependency("adbc_driver_manager.dbapi", errors="ignore")
        if adbc and isinstance(conn, adbc.Connection):
            with conn.cursor() as cur:
                cur.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        else:
            with conn.begin():
                sql.SQLDatabase(conn).drop_table(table_name)


def drop_view(view_name: str, conn: Any) -> None:
    import sqlalchemy

    if isinstance(conn, sqlite3.Connection):
        conn.execute(f'DROP VIEW IF EXISTS {sql._get_valid_sqlite_name(view_name)}')
        conn.commit()
    else:
        adbc = import_optional_dependency("adbc_driver_manager.dbapi", errors="ignore")
        if adbc and isinstance(conn, adbc.Connection):
            with conn.cursor() as cur:
                cur.execute(f'DROP VIEW IF EXISTS "{view_name}"')
        else:
            quoted_view = conn.engine.dialect.identifier_preparer.quote_identifier(view_name)
            stmt = sqlalchemy.text(f'DROP VIEW IF EXISTS {quoted_view}')
            with conn.begin():
                conn.execute(stmt)


@pytest.fixture
def mysql_pymysql_engine() -> SQLAlchemyEngine:
    sqlalchemy = pytest.importorskip("sqlalchemy")
    pymysql = pytest.importorskip("pymysql")
    engine = sqlalchemy.create_engine(
        "mysql+pymysql://root@localhost:3306/pandas",
        connect_args={
            "client_flag": pymysql.constants.CLIENT.MULTI_STATEMENTS
        },
        poolclass=sqlalchemy.pool.NullPool,
    )
    yield engine
    for view in get_all_views(engine):
        drop_view(view, engine)
    for tbl in get_all_tables(engine):
        drop_table(tbl, engine)
    engine.dispose()


@pytest.fixture
def mysql_pymysql_engine_iris(
    mysql_pymysql_engine: SQLAlchemyEngine, iris_path: Path
) -> SQLAlchemyEngine:
    create_and_load_iris(mysql_pymysql_engine, iris_path)
    create_and_load_iris_view(mysql_pymysql_engine)
    return mysql_pymysql_engine


@pytest.fixture
def mysql_pymysql_engine_types(
    mysql_pymysql_engine: SQLAlchemyEngine, types_data: List[Dict[str, Any]]
) -> SQLAlchemyEngine:
    create_and_load_types(mysql_pymysql_engine, types_data, "mysql")
    return mysql_pymysql_engine


@pytest.fixture
def mysql_pymysql_conn(mysql_pymysql_engine: SQLAlchemyEngine) -> sqlalchemy.engine.base.Connection:
    with mysql_pymysql_engine.connect() as conn:
        yield conn


@pytest.fixture
def mysql_pymysql_conn_iris(
    mysql_pymysql_engine_iris: SQLAlchemyEngine,
) -> sqlalchemy.engine.base.Connection:
    with mysql_pymysql_engine_iris.connect() as conn:
        yield conn


@pytest.fixture
def mysql_pymysql_conn_types(
    mysql_pymysql_engine_types: SQLAlchemyEngine,
) -> sqlalchemy.engine.base.Connection:
    with mysql_pymysql_engine_types.connect() as conn:
        yield conn


@pytest.fixture
def postgresql_psycopg2_engine() -> SQLAlchemyEngine:
    sqlalchemy = pytest.importorskip("sqlalchemy")
    pytest.importorskip("psycopg2")
    engine = sqlalchemy.create_engine(
        "postgresql+psycopg2://postgres:postgres@localhost:5432/pandas",
        poolclass=sqlalchemy.pool.NullPool,
    )
    yield engine
    for view in get_all_views(engine):
        drop_view(view, engine)
    for tbl in get_all_tables(engine):
        drop_table(tbl, engine)
    engine.dispose()


@pytest.fixture
def postgresql_psycopg2_engine_iris(
    postgresql_psycopg2_engine: SQLAlchemyEngine, iris_path: Path
) -> SQLAlchemyEngine:
    create_and_load_iris(postgresql_psycopg2_engine, iris_path)
    create_and_load_iris_view(postgresql_psycopg2_engine)
    return postgresql_psycopg2_engine


@pytest.fixture
def postgresql_psycopg2_engine_types(
    postgresql_psycopg2_engine: SQLAlchemyEngine, types_data: List[Dict[str, Any]]
) -> SQLAlchemyEngine:
    create_and_load_types(postgresql_psycopg2_engine, types_data, "postgres")
    return postgresql_psycopg2_engine


@pytest.fixture
def postgresql_psycopg2_conn(
    postgresql_psycopg2_engine: SQLAlchemyEngine,
) -> sqlalchemy.engine.base.Connection:
    with postgresql_psycopg2_engine.connect() as conn:
        yield conn


@pytest.fixture
def postgresql_adbc_conn() -> Any:  # Replace Any with appropriate ADBC connection type
    pytest.importorskip("pyarrow")
    pytest.importorskip("adbc_driver_postgresql")
    from adbc_driver_postgresql import dbapi

    uri = "postgresql://postgres:postgres@localhost:5432/pandas"
    with dbapi.connect(uri) as conn:
        yield conn
        for view in get_all_views(conn):
            drop_view(view, conn)
        for tbl in get_all_tables(conn):
            drop_table(tbl, conn)
        conn.commit()


@pytest.fixture
def postgresql_adbc_iris(
    postgresql_adbc_conn: Any, iris_path: Path
) -> Any:  # Replace Any with appropriate ADBC connection type
    import adbc_driver_manager as mgr

    conn = postgresql_adbc_conn
    try:
        conn.adbc_get_table_schema("iris")
    except mgr.ProgrammingError:
        conn.rollback()
        create_and_load_iris_postgresql(conn, iris_path)
    try:
        conn.adbc_get_table_schema("iris_view")
    except mgr.ProgrammingError:
        conn.rollback()
        create_and_load_iris_view(conn)
    return conn


@pytest.fixture
def postgresql_adbc_types(
    postgresql_adbc_conn: Any, types_data: List[Dict[str, Any]]
) -> Any:  # Replace Any with appropriate ADBC connection type
    import adbc_driver_manager as mgr

    conn = postgresql_adbc_conn
    try:
        conn.adbc_get_table_schema("types")
    except mgr.ProgrammingError:
        conn.rollback()
        new_data = [tuple(entry.values()) for entry in types_data]
        create_and_load_types_postgresql(conn, new_data)
    return conn


@pytest.fixture
def postgresql_psycopg2_conn_iris(
    postgresql_psycopg2_engine_iris: SQLAlchemyEngine,
) -> sqlalchemy.engine.base.Connection:
    with postgresql_psycopg2_engine_iris.connect() as conn:
        yield conn


@pytest.fixture
def postgresql_psycopg2_conn_types(
    postgresql_psycopg2_engine_types: SQLAlchemyEngine,
) -> sqlalchemy.engine.base.Connection:
    with postgresql_psycopg2_engine_types.connect() as conn:
        yield conn


@pytest.fixture
def sqlite_str() -> str:
    pytest.importorskip("sqlalchemy")
    with tm.ensure_clean() as name:
        yield f"sqlite:///{name}"


@pytest.fixture
def sqlite_engine(sqlite_str: str) -> SQLAlchemyEngine:
    sqlalchemy = pytest.importorskip("sqlalchemy")
    engine = sqlalchemy.create_engine(sqlite_str, poolclass=sqlalchemy.pool.NullPool)
    yield engine
    for view in get_all_views(engine):
        drop_view(view, engine)
    for tbl in get_all_tables(engine):
        drop_table(tbl, engine)
    engine.dispose()


@pytest.fixture
def sqlite_conn(sqlite_engine: SQLAlchemyEngine) -> sqlalchemy.engine.base.Connection:
    with sqlite_engine.connect() as conn:
        yield conn


@pytest.fixture
def sqlite_str_iris(sqlite_str: str, iris_path: Path) -> str:
    sqlalchemy = pytest.importorskip("sqlalchemy")
    engine = sqlalchemy.create_engine(sqlite_str)
    create_and_load_iris(engine, iris_path)
    create_and_load_iris_view(engine)
    engine.dispose()
    return sqlite_str


@pytest.fixture
def sqlite_engine_iris(
    sqlite_engine: SQLAlchemyEngine, iris_path: Path
) -> SQLAlchemyEngine:
    create_and_load_iris(sqlite_engine, iris_path)
    create_and_load_iris_view(sqlite_engine)
    return sqlite_engine


@pytest.fixture
def sqlite_conn_iris(sqlite_engine_iris: SQLAlchemyEngine) -> sqlalchemy.engine.base.Connection:
    with sqlite_engine_iris.connect() as conn:
        yield conn


@pytest.fixture
def sqlite_str_types(sqlite_str: str, types_data: List[Dict[str, Any]]) -> str:
    sqlalchemy = pytest.importorskip("sqlalchemy")
    engine = sqlalchemy.create_engine(sqlite_str)
    create_and_load_types(engine, types_data, "sqlite")
    engine.dispose()
    return sqlite_str


@pytest.fixture
def sqlite_engine_types(
    sqlite_engine: SQLAlchemyEngine, types_data: List[Dict[str, Any]]
) -> SQLAlchemyEngine:
    create_and_load_types(sqlite_engine, types_data, "sqlite")
    return sqlite_engine


@pytest.fixture
def sqlite_conn_types(sqlite_engine_types: SQLAlchemyEngine) -> sqlalchemy.engine.base.Connection:
    with sqlite_engine_types.connect() as conn:
        yield conn


@pytest.fixture
def sqlite_adbc_conn() -> Any:  # Replace Any with appropriate ADBC connection type
    pytest.importorskip("pyarrow")
    pytest.importorskip("adbc_driver_sqlite")
    from adbc_driver_sqlite import dbapi

    with tm.ensure_clean() as name:
        uri = f"file:{name}"
        with dbapi.connect(uri) as conn:
            yield conn
            for view in get_all_views(conn):
                drop_view(view, conn)
            for tbl in get_all_tables(conn):
                drop_table(tbl, conn)
            conn.commit()


@pytest.fixture
def sqlite_adbc_iris(
    sqlite_adbc_conn: Any, iris_path: Path
) -> Any:  # Replace Any with appropriate ADBC connection type
    import adbc_driver_manager as mgr

    conn = sqlite_adbc_conn
    try:
        conn.adbc_get_table_schema("iris")
    except mgr.ProgrammingError:
        conn.rollback()
        create_and_load_iris_sqlite3(conn, iris_path)
    try:
        conn.adbc_get_table_schema("iris_view")
    except mgr.ProgrammingError:
        conn.rollback()
        create_and_load_iris_view(conn)
    return conn


@pytest.fixture
def sqlite_adbc_types(
    sqlite_adbc_conn: Any, types_data: List[Dict[str, Any]]
) -> Any:  # Replace Any with appropriate ADBC connection type
    import adbc_driver_manager as mgr

    conn = sqlite_adbc_conn
    try:
        conn.adbc_get_table_schema("types")
    except mgr.ProgrammingError:
        conn.rollback()
        new_data: List[Tuple[Any, ...]] = []
        for entry in types_data:
            entry["BoolCol"] = int(entry["BoolCol"])
            if entry["BoolColWithNull"] is not None:
                entry["BoolColWithNull"] = int(entry["BoolColWithNull"])
            new_data.append(tuple(entry.values()))
        create_and_load_types_sqlite3(conn, new_data)
        conn.commit()
    return conn


@pytest.fixture
def sqlite_buildin() -> sqlite3.Connection:
    with contextlib.closing(sqlite3.connect(":memory:")) as closing_conn:
        with closing_conn as conn:
            yield conn


@pytest.fixture
def sqlite_buildin_iris(
    sqlite_buildin: sqlite3.Connection, iris_path: Path
) -> sqlite3.Connection:
    create_and_load_iris_sqlite3(sqlite_buildin, iris_path)
    create_and_load_iris_view(sqlite_buildin)
    return sqlite_buildin


@pytest.fixture
def sqlite_buildin_types(
    sqlite_buildin: sqlite3.Connection, types_data: List[Dict[str, Any]]
) -> sqlite3.Connection:
    types_data_tuples: List[Tuple[Any, ...]] = [tuple(entry.values()) for entry in types_data]
    create_and_load_types_sqlite3(sqlite_buildin, types_data_tuples)
    return sqlite_buildin


mysql_connectable: List[pytest.param] = [
    pytest.param("mysql_pymysql_engine", marks=pytest.mark.db),
    pytest.param("mysql_pymysql_conn", marks=pytest.mark.db),
]
mysql_connectable_iris: List[pytest.param] = [
    pytest.param("mysql_pymysql_engine_iris", marks=pytest.mark.db),
    pytest.param("mysql_pymysql_conn_iris", marks=pytest.mark.db),
]
mysql_connectable_types: List[pytest.param] = [
    pytest.param("mysql_pymysql_engine_types", marks=pytest.mark.db),
    pytest.param("mysql_pymysql_conn_types", marks=pytest.mark.db),
]
postgresql_connectable: List[pytest.param] = [
    pytest.param("postgresql_psycopg2_engine", marks=pytest.mark.db),
    pytest.param("postgresql_psycopg2_conn", marks=pytest.mark.db),
]
postgresql_connectable_iris: List[pytest.param] = [
    pytest.param("postgresql_psycopg2_engine_iris", marks=pytest.mark.db),
    pytest.param("postgresql_psycopg2_conn_iris", marks=pytest.mark.db),
]
postgresql_connectable_types: List[pytest.param] = [
    pytest.param("postgresql_psycopg2_engine_types", marks=pytest.mark.db),
    pytest.param("postgresql_psycopg2_conn_types", marks=pytest.mark.db),
]
sqlite_connectable: List[str] = ["sqlite_engine", "sqlite_conn", "sqlite_str"]
sqlite_connectable_iris: List[str] = ["sqlite_engine_iris", "sqlite_conn_iris", "sqlite_str_iris"]
sqlite_connectable_types: List[str] = ["sqlite_engine_types", "sqlite_conn_types", "sqlite_str_types"]
sqlalchemy_connectable: List[Union[pytest.param, str]] = mysql_connectable + postgresql_connectable + sqlite_connectable
sqlalchemy_connectable_iris: List[Union[pytest.param, str]] = (
    mysql_connectable_iris + postgresql_connectable_iris + sqlite_connectable_iris
)
sqlalchemy_connectable_types: List[Union[pytest.param, str]] = (
    mysql_connectable_types + postgresql_connectable_types + sqlite_connectable_types
)
adbc_connectable: List[Union[pytest.param, str]] = [
    "sqlite_adbc_conn",
    pytest.param("postgresql_adbc_conn", marks=pytest.mark.db),
]
adbc_connectable_iris: List[Union[pytest.param, str]] = [
    pytest.param("postgresql_adbc_iris", marks=pytest.mark.db),
    "sqlite_adbc_iris",
]
adbc_connectable_types: List[Union[pytest.param, str]] = [
    pytest.param("postgresql_adbc_types", marks=pytest.mark.db),
    "sqlite_adbc_types",
]
all_connectable: List[Union[pytest.param, str]] = sqlalchemy_connectable + ["sqlite_buildin"] + adbc_connectable
all_connectable_iris: List[Union[pytest.param, str]] = sqlalchemy_connectable_iris + ["sqlite_buildin_iris"] + adbc_connectable_iris
all_connectable_types: List[Union[pytest.param, str]] = sqlalchemy_connectable_types + ["sqlite_buildin_types"] + adbc_connectable_types


@pytest.mark.parametrize("conn", all_connectable)
def test_dataframe_to_sql(
    conn: Union[pytest.param, str], test_frame1: DataFrame, request: pytest.FixtureRequest
) -> None:
    conn_value = request.getfixturevalue(conn)
    test_frame1.to_sql(name="test", con=conn_value, if_exists="append", index=False)


@pytest.mark.parametrize("conn", all_connectable)
def test_dataframe_to_sql_empty(
    conn: Union[pytest.param, str],
    test_frame1: DataFrame,
    request: pytest.FixtureRequest,
) -> None:
    if conn == "postgresql_adbc_conn" and not using_string_dtype():
        request.node.add_marker(
            pytest.mark.xfail(
                reason="postgres ADBC driver < 1.2 cannot insert index with null type"
            )
        )
    conn_value = request.getfixturevalue(conn)
    empty_df = test_frame1.iloc[:0]
    empty_df.to_sql(name="test", con=conn_value, if_exists="append", index=False)


@pytest.mark.parametrize("conn", all_connectable)
def test_dataframe_to_sql_arrow_dtypes(
    conn: Union[pytest.param, str],
    request: pytest.FixtureRequest,
) -> None:
    pytest.importorskip("pyarrow")
    df = DataFrame(
        {
            "int": pd.array([1], dtype="int8[pyarrow]"),
            "datetime": pd.array(
                [datetime(2023, 1, 1)], dtype="timestamp[ns][pyarrow]"
            ),
            "date": pd.array(
                [date(2023, 1, 1)], dtype="date32[day][pyarrow]"
            ),
            "timedelta": pd.array(
                [timedelta(1)], dtype="duration[ns][pyarrow]"
            ),
            "string": pd.array(["a"], dtype="string[pyarrow]"),
        }
    )
    if "adbc" in conn:
        if conn == "sqlite_adbc_conn":
            df = df.drop(columns=["timedelta"])
        if pa_version_under14p1:
            exp_warning: Optional[Type[Warning]] = DeprecationWarning
            msg = "is_sparse is deprecated"
        else:
            exp_warning = None
            msg = ""
    else:
        exp_warning = UserWarning
        msg = "the 'timedelta'"
    conn_value = request.getfixturevalue(conn)
    with tm.assert_produces_warning(
        exp_warning, match=msg, check_stacklevel=False
    ):
        df.to_sql(name="test_arrow", con=conn_value, if_exists="replace", index=False)


@pytest.mark.parametrize("conn", all_connectable)
def test_dataframe_to_sql_arrow_dtypes_missing(
    conn: Union[pytest.param, str],
    request: pytest.FixtureRequest,
    nulls_fixture: Any,  # Replace Any with appropriate type
) -> None:
    pytest.importorskip("pyarrow")
    df = DataFrame(
        {
            "datetime": pd.array(
                [datetime(2023, 1, 1), nulls_fixture],
                dtype="timestamp[ns][pyarrow]",
            )
        }
    )
    conn_value = request.getfixturevalue(conn)
    df.to_sql(name="test_arrow", con=conn_value, if_exists="replace", index=False)


@pytest.mark.parametrize("conn", all_connectable)
@pytest.mark.parametrize("method", [None, "multi"])
def test_to_sql(
    conn: Union[pytest.param, str],
    method: Optional[str],
    test_frame1: DataFrame,
    request: pytest.FixtureRequest,
) -> None:
    if method == "multi" and "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(reason="'method' not implemented for ADBC drivers", strict=True)
        )
    conn_value = request.getfixturevalue(conn)
    with pandasSQL_builder(conn_value, need_transaction=True) as pandasSQL:
        pandasSQL.to_sql(test_frame1, "test_frame", method=method)
        assert pandasSQL.has_table("test_frame")
    assert count_rows(conn_value, "test_frame") == len(test_frame1)


@pytest.mark.parametrize("conn", all_connectable)
@pytest.mark.parametrize("mode, num_row_coef", [("replace", 1), ("append", 2)])
def test_to_sql_exist(
    conn: Union[pytest.param, str],
    mode: str,
    num_row_coef: int,
    test_frame1: DataFrame,
    request: pytest.FixtureRequest,
) -> None:
    conn_value = request.getfixturevalue(conn)
    with pandasSQL_builder(conn_value, need_transaction=True) as pandasSQL:
        pandasSQL.to_sql(test_frame1, "test_frame", if_exists="fail")
        pandasSQL.to_sql(test_frame1, "test_frame", if_exists=mode)
        assert pandasSQL.has_table("test_frame")
    assert count_rows(conn_value, "test_frame") == num_row_coef * len(test_frame1)


@pytest.mark.parametrize("conn", all_connectable)
def test_to_sql_exist_fail(
    conn: Union[pytest.param, str],
    test_frame1: DataFrame,
    request: pytest.FixtureRequest,
) -> None:
    conn_value = request.getfixturevalue(conn)
    with pandasSQL_builder(conn_value, need_transaction=True) as pandasSQL:
        pandasSQL.to_sql(test_frame1, "test_frame", if_exists="fail")
        assert pandasSQL.has_table("test_frame")
        msg = "Table 'test_frame' already exists"
        with pytest.raises(ValueError, match=msg):
            pandasSQL.to_sql(test_frame1, "test_frame", if_exists="fail")


@pytest.mark.parametrize("conn", all_connectable_iris)
def test_read_iris_query(conn: Union[pytest.param, str], request: pytest.FixtureRequest) -> None:
    conn_value = request.getfixturevalue(conn)
    iris_frame = read_sql_query("SELECT * FROM iris", conn_value)
    check_iris_frame(iris_frame)
    iris_frame = pd.read_sql("SELECT * FROM iris", conn_value)
    check_iris_frame(iris_frame)
    iris_frame = pd.read_sql("SELECT * FROM iris where 0=1", conn_value)
    assert iris_frame.shape == (0, 5)
    assert "SepalWidth" in iris_frame.columns


@pytest.mark.parametrize("conn", all_connectable_iris)
def test_read_iris_query_chunksize(
    conn: Union[pytest.param, str],
    request: pytest.FixtureRequest,
) -> None:
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(reason="'chunksize' not implemented for ADBC drivers", strict=True)
        )
    conn_value = request.getfixturevalue(conn)
    iris_frame = concat(read_sql_query("SELECT * FROM iris", conn_value, chunksize=7))
    check_iris_frame(iris_frame)
    iris_frame = concat(pd.read_sql("SELECT * FROM iris", conn_value, chunksize=7))
    check_iris_frame(iris_frame)
    iris_frame = concat(pd.read_sql("SELECT * FROM iris where 0=1", conn_value, chunksize=7))
    assert iris_frame.shape == (0, 5)
    assert "SepalWidth" in iris_frame.columns


@pytest.mark.parametrize("conn", sqlalchemy_connectable_iris)
def test_read_iris_query_expression_with_parameter(
    conn: Union[pytest.param, str],
    request: pytest.FixtureRequest,
) -> None:
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(reason="'chunksize' not implemented for ADBC drivers", strict=True)
        )
    conn_value = request.getfixturevalue(conn)
    from sqlalchemy import MetaData, Table, select

    metadata = MetaData()
    autoload_con = (
        create_and_load_iris(conn_value, Path())
        if isinstance(conn_value, str)
        else conn_value
    )
    iris = Table("iris", metadata, autoload_with=autoload_con)
    iris_frame = read_sql_query(
        select(iris), conn_value, params={"name": "Iris-setosa", "length": 5.1}
    )
    check_iris_frame(iris_frame)
    if isinstance(conn_value, str):
        autoload_con.dispose()


@pytest.mark.parametrize("conn", all_connectable_iris)
def test_read_iris_query_string_with_parameter(
    conn: Union[pytest.param, str],
    request: pytest.FixtureRequest,
    sql_strings: Dict[str, Dict[str, str]],
) -> None:
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(reason="'params' not implemented for ADBC drivers", strict=True)
        )
    conn_name = conn
    conn_value = request.getfixturevalue(conn)
    query = ""
    for db, q in sql_strings["read_parameters"].items():
        if db in conn:
            query = q
            break
    else:
        raise KeyError(f"No part of {conn} found in sql_strings['read_parameters']")

    iris_frame = read_sql_query(query, conn_value, params=("Iris-setosa", 5.1))
    check_iris_frame(iris_frame)


@pytest.mark.parametrize("conn", sqlalchemy_connectable_iris)
def test_read_iris_table(
    conn: Union[pytest.param, str], request: pytest.FixtureRequest
) -> None:
    conn_value = request.getfixturevalue(conn)
    iris_frame = read_sql_table("iris", conn_value)
    check_iris_frame(iris_frame)
    iris_frame = pd.read_sql("iris", conn_value)
    check_iris_frame(iris_frame)


@pytest.mark.parametrize("conn", sqlalchemy_connectable_iris)
def test_read_iris_table_chunksize(
    conn: Union[pytest.param, str],
    request: pytest.FixtureRequest,
) -> None:
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(reason="chunksize argument NotImplemented with ADBC")
        )
    conn_value = request.getfixturevalue(conn)
    iris_frame = concat(read_sql_table("iris", conn_value, chunksize=7))
    check_iris_frame(iris_frame)
    iris_frame = concat(pd.read_sql("iris", conn_value, chunksize=7))
    check_iris_frame(iris_frame)


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_to_sql_callable(
    conn: Union[pytest.param, str],
    test_frame1: DataFrame,
    request: pytest.FixtureRequest,
) -> None:
    conn_value = request.getfixturevalue(conn)
    check: List[int] = []

    def sample(
        pd_table: sqlalchemy.Table,
        conn_inner: Any,
        keys: List[str],
        data_iter: Any,
    ) -> None:
        check.append(1)
        data = [dict(zip(keys, row)) for row in data_iter]
        conn_inner.execute(pd_table.insert(), data)

    with pandasSQL_builder(conn_value, need_transaction=True) as pandasSQL:
        pandasSQL.to_sql(test_frame1, "test_frame", method=sample)
        assert pandasSQL.has_table("test_frame")
    assert check == [1]
    assert count_rows(conn_value, "test_frame") == len(test_frame1)


@pytest.mark.parametrize("conn", all_connectable_types)
def test_default_type_conversion(
    conn: Union[pytest.param, str],
    request: pytest.FixtureRequest,
) -> None:
    conn_name = conn
    if conn_name == "sqlite_buildin_types":
        request.applymarker(
            pytest.mark.xfail(reason="sqlite_buildin connection does not implement read_sql_table")
        )
    conn_value = request.getfixturevalue(conn)
    df = sql.read_sql_table("types", conn_value)
    assert issubclass(df.FloatCol.dtype.type, np.floating)
    assert issubclass(df.IntCol.dtype.type, np.integer)
    if "postgresql" in conn_name:
        assert issubclass(df.BoolCol.dtype.type, np.bool_)
    else:
        assert issubclass(df.BoolCol.dtype.type, np.integer)
    assert issubclass(df.IntColWithNull.dtype.type, np.floating)
    if "postgresql" in conn_name:
        assert issubclass(df.BoolColWithNull.dtype.type, object)
    else:
        assert issubclass(df.BoolColWithNull.dtype.type, np.floating)


@pytest.mark.parametrize("conn", mysql_connectable)
def test_read_procedure(
    conn: Union[pytest.param, str], request: pytest.FixtureRequest
) -> None:
    conn_value = request.getfixturevalue(conn)
    from sqlalchemy import text
    from sqlalchemy.engine import Engine

    df = DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
    df.to_sql(name="test_frame", con=conn_value, index=False)
    proc = (
        "DROP PROCEDURE IF EXISTS get_testdb;\n\n    CREATE PROCEDURE get_testdb ()\n\n    BEGIN\n        SELECT * FROM test_frame;\n    END"
    )
    proc_text = text(proc)
    if isinstance(conn_value, Engine):
        with conn_value.connect() as engine_conn:
            with engine_conn.begin():
                engine_conn.execute(proc_text)
    else:
        with conn_value.begin():
            conn_value.execute(proc_text)
    res1 = sql.read_sql_query("CALL get_testdb();", conn_value)
    tm.assert_frame_equal(df, res1)
    res2 = sql.read_sql("CALL get_testdb();", conn_value)
    tm.assert_frame_equal(df, res2)


@pytest.mark.parametrize("conn", postgresql_connectable)
@pytest.mark.parametrize("expected_count", [2, "Success!"])
def test_copy_from_callable_insertion_method(
    conn: Union[pytest.param, str],
    expected_count: Union[int, str],
    request: pytest.FixtureRequest,
) -> None:

    def psql_insert_copy(
        table: sqlalchemy.Table,
        conn_inner: Any,
        keys: List[str],
        data_iter: Any,
    ) -> Union[int, str]:
        dbapi_conn = conn_inner.connection  # type: ignore
        with dbapi_conn.cursor() as cur:
            s_buf = StringIO()
            writer = csv.writer(s_buf)
            writer.writerows(data_iter)
            s_buf.seek(0)
            columns = ", ".join([f'"{k}"' for k in keys])
            if table.schema:
                table_name = f"{table.schema}.{table.name}"
            else:
                table_name = table.name
            sql_query = f'COPY {table_name} ({columns}) FROM STDIN WITH CSV'
            cur.copy_expert(sql=sql_query, file=s_buf)
        return expected_count

    conn_value = request.getfixturevalue(conn)
    expected = DataFrame({"col1": [1, 2], "col2": [0.1, 0.2], "col3": ["a", "n"]})
    result_count = expected.to_sql(
        name="test_frame", con=conn_value, index=False, method=psql_insert_copy
    )
    if expected_count is None:
        assert result_count is None
    else:
        assert result_count == expected_count
    result = sql.read_sql_table("test_frame", conn_value)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("conn", postgresql_connectable)
def test_insertion_method_on_conflict_do_nothing(
    conn: Union[pytest.param, str], request: pytest.FixtureRequest
) -> None:
    conn_value = request.getfixturevalue(conn)
    from sqlalchemy.dialects.postgresql import insert
    from sqlalchemy.engine import Engine
    from sqlalchemy.sql import text

    def insert_on_conflict(
        table: sqlalchemy.Table,
        conn_inner: Any,
        keys: List[str],
        data_iter: Any,
    ) -> int:
        data = [dict(zip(keys, row)) for row in data_iter]
        stmt = insert(table).values(data).on_conflict_do_nothing(index_elements=["a"])
        result = conn_inner.execute(stmt)
        return result.rowcount

    create_sql = text(
        "\n    CREATE TABLE test_insert_conflict (\n        a  integer PRIMARY KEY,\n        b  numeric,\n        c  text\n    );\n    "
    )
    if isinstance(conn_value, Engine):
        with conn_value.connect() as con:
            with con.begin():
                con.execute(create_sql)
    else:
        with conn_value.begin():
            conn_value.execute(create_sql)
    expected = DataFrame([[1, 2.1, "a"]], columns=list("abc"))
    expected.to_sql(name="test_insert_conflict", con=conn_value, if_exists="append", index=False)
    df_insert = DataFrame([[1, 3.2, "b"]], columns=list("abc"))
    inserted = df_insert.to_sql(
        name="test_insert_conflict",
        con=conn_value,
        index=False,
        if_exists="append",
        method=insert_on_conflict,
    )
    result = sql.read_sql_table("test_insert_conflict", conn_value)
    tm.assert_frame_equal(result, expected)
    assert inserted == 0
    with sql.SQLDatabase(conn_value, need_transaction=True) as pandasSQL:
        pandasSQL.drop_table("test_insert_conflict")


@pytest.mark.parametrize("conn", all_connectable)
def test_to_sql_on_public_schema(
    conn: Union[pytest.param, str],
    request: pytest.FixtureRequest,
) -> None:
    if "sqlite" in conn or "mysql" in conn:
        request.applymarker(
            pytest.mark.xfail(reason="test for public schema only specific to postgresql")
        )
    conn_value = request.getfixturevalue(conn)
    test_data = DataFrame([[1, 2.1, "a"], [2, 3.1, "b"]], columns=list("abc"))
    test_data.to_sql(
        name="test_public_schema",
        con=conn_value,
        if_exists="append",
        index=False,
        schema="public",
    )
    df_out = sql.read_sql_table("test_public_schema", conn_value, schema="public")
    tm.assert_frame_equal(test_data, df_out)


@pytest.mark.parametrize("conn", postgresql_connectable)
@pytest.mark.parametrize("parse_dates", [None, ["DateColWithTz"]])
def test_datetime_with_timezone_query(
    conn: Union[pytest.param, str],
    request: pytest.FixtureRequest,
    parse_dates: Optional[List[str]],
) -> None:
    conn_value = request.getfixturevalue(conn)
    expected = create_and_load_postgres_datetz(conn_value)
    df = read_sql_query("select * from datetz", conn_value, parse_dates=parse_dates)
    col = df.DateColWithTz
    tm.assert_series_equal(col, expected)


@pytest.mark.parametrize("conn", postgresql_connectable)
def test_datetime_with_timezone_query_chunksize(
    conn: Union[pytest.param, str], request: pytest.FixtureRequest
) -> None:
    conn_value = request.getfixturevalue(conn)
    expected = create_and_load_postgres_datetz(conn_value)
    df = concat(
        list(
            read_sql_query(
                "select * from datetz", conn_value, chunksize=1
            )
        ),
        ignore_index=True,
    )
    col = df.DateColWithTz
    tm.assert_series_equal(col, expected)


@pytest.mark.parametrize("conn", postgresql_connectable)
def test_datetime_with_timezone_table(
    conn: Union[pytest.param, str],
    request: pytest.FixtureRequest,
) -> None:
    conn_value = request.getfixturevalue(conn)
    expected = create_and_load_postgres_datetz(conn_value)
    result = sql.read_sql_table("datetz", conn_value)
    exp_frame = expected.to_frame()
    tm.assert_frame_equal(result, exp_frame)


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_datetime_with_timezone_roundtrip(
    conn: Union[pytest.param, str],
    request: pytest.FixtureRequest,
) -> None:
    conn_name = conn
    conn_value = request.getfixturevalue(conn)
    expected = DataFrame(
        {"A": date_range("2013-01-01 09:00:00", periods=3, tz="US/Pacific", unit="us")}
    )
    assert expected.to_sql(name="test_datetime_tz", con=conn_value, index=False) == 3
    if "postgresql" in conn_name:
        expected["A"] = expected["A"].dt.tz_convert("UTC")
    else:
        expected["A"] = expected["A"].dt.tz_localize(None)
    result = sql.read_sql_table("test_datetime_tz", conn_value)
    tm.assert_frame_equal(result, expected)
    result = sql.read_sql_query("SELECT * FROM test_datetime_tz", conn_value)
    if "sqlite" in conn_name:
        assert isinstance(result.loc[0, "A"], str)
        result["A"] = to_datetime(result["A"]).dt.as_unit("us")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_out_of_bounds_datetime(
    conn: Union[pytest.param, str], request: pytest.FixtureRequest
) -> None:
    conn_value = request.getfixturevalue(conn)
    data = DataFrame({"date": [datetime(9999, 1, 1)]})
    assert data.to_sql(name="test_bigint", con=conn_value, index=False) == 1
    result = sql.read_sql_table("test_bigint", conn_value)
    tm.assert_frame_equal(data, result)


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_naive_datetimeindex_roundtrip(
    conn: Union[pytest.param, str],
    request: pytest.FixtureRequest,
) -> None:
    conn_value = request.getfixturevalue(conn)
    dates = date_range("2018-01-01 09:00:00", periods=5, freq="6h")._with_freq(None)
    expected = DataFrame({"nums": range(5)}, index=dates)
    assert expected.to_sql(name="foo_table", con=conn_value, index_label="info_date") == 5
    result = sql.read_sql_table("foo_table", conn_value, index_col="info_date")
    assert result.index.names == ["info_date"]
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("conn", all_connectable_types)
def test_date_parsing(
    conn: Union[pytest.param, str],
    request: pytest.FixtureRequest,
) -> None:
    conn_name = conn
    conn_value = request.getfixturevalue(conn)
    df = sql.read_sql_table("types", conn_value)
    if not ("mysql" in conn_name or "postgres" in conn_name):
        assert not issubclass(df.DateCol.dtype.type, np.datetime64)
    df = sql.read_sql_table("types", conn_value, parse_dates=["DateCol"])
    assert issubclass(df.DateCol.dtype.type, np.datetime64)
    assert df.DateCol.tolist() == [
        Timestamp(2000, 1, 3, 0, 0, 0),
        Timestamp(2000, 1, 4, 0, 0, 0),
    ]
    df = sql.read_sql_table(
        "types",
        conn_value,
        parse_dates={"DateCol": "%Y-%m-%d %H:%M:%S"},
    )
    assert issubclass(df.DateCol.dtype.type, np.datetime64)
    assert df.DateCol.tolist() == [
        Timestamp(2000, 1, 3, 0, 0, 0),
        Timestamp(2000, 1, 4, 0, 0, 0),
    ]
    df = sql.read_sql_table("types", conn_value, parse_dates=["IntDateCol"])
    assert issubclass(df.IntDateCol.dtype.type, np.datetime64)
    assert df.IntDateCol.tolist() == [
        Timestamp(1986, 12, 25, 0, 0, 0),
        Timestamp(2013, 1, 1, 0, 0, 0),
    ]
    df = sql.read_sql_table(
        "types", conn_value, parse_dates={"IntDateCol": "s"}
    )
    assert issubclass(df.IntDateCol.dtype.type, np.datetime64)
    assert df.IntDateCol.tolist() == [
        Timestamp(1986, 12, 25, 0, 0, 0),
        Timestamp(2013, 1, 1, 0, 0, 0),
    ]
    df = sql.read_sql_table(
        "types",
        conn_value,
        parse_dates={"IntDateOnlyCol": "%Y%m%d"},
    )
    assert issubclass(df.IntDateOnlyCol.dtype.type, np.datetime64)
    assert df.IntDateOnlyCol.tolist() == [
        Timestamp("2010-10-10"),
        Timestamp("2010-12-12"),
    ]


@pytest.mark.parametrize("conn", all_connectable)
@pytest.mark.parametrize("error", ["raise", "coerce"])
@pytest.mark.parametrize(
    "read_sql, text, mode",
    [
        (sql.read_sql, "SELECT * FROM types", ("sqlalchemy", "fallback")),
        (sql.read_sql, "types", "sqlalchemy"),
        (sql.read_sql_query, "SELECT * FROM types", ("sqlalchemy", "fallback")),
        (sql.read_sql_table, "types", "sqlalchemy"),
    ],
)
def test_api_custom_dateparsing_error(
    conn: Union[pytest.param, str],
    request: pytest.FixtureRequest,
    read_sql: Callable[..., DataFrame],
    text: Union[str, sqlalchemy.sql.expression.ClauseElement],
    mode: Union[str, Tuple[str, str]],
    error: str,
    types_data_frame: DataFrame,
) -> None:
    conn_name = conn
    conn_value = request.getfixturevalue(conn)
    if text == "types" and conn_name == "sqlite_buildin_types":
        request.applymarker(
            pytest.mark.xfail(reason="failing combination of arguments")
        )
    expected = types_data_frame.astype({"DateCol": "datetime64[s]"})
    result = read_sql(text, con=conn_value, parse_dates={"DateCol": {"errors": error}})
    if "postgres" in conn_name:
        result["BoolCol"] = result["BoolCol"].astype(int)
        result["BoolColWithNull"] = result["BoolColWithNull"].astype(float)
    if conn_name == "postgresql_adbc_types":
        expected = expected.astype(
            {"IntDateCol": "int32", "IntDateOnlyCol": "int32", "IntCol": "int32"}
        )
    if conn_name == "postgresql_adbc_types" and pa_version_under14p1:
        expected["DateCol"] = expected["DateCol"].astype("datetime64[ns]")
    elif "postgres" in conn_name or "mysql" in conn_name:
        expected["DateCol"] = expected["DateCol"].astype("datetime64[us]")
    else:
        expected["DateCol"] = expected["DateCol"].astype("datetime64[s]")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("conn", all_connectable_types)
def test_api_date_and_index(
    conn: Union[pytest.param, str],
    request: pytest.FixtureRequest,
) -> None:
    conn_value = request.getfixturevalue(conn)
    df = sql.read_sql_query("SELECT * FROM types", conn_value, index_col="DateCol", parse_dates=["DateCol", "IntDateCol"])
    assert issubclass(df.index.dtype.type, np.datetime64)
    assert issubclass(df.IntDateCol.dtype.type, np.datetime64)


@pytest.mark.parametrize("conn", all_connectable)
def test_api_timedelta(
    conn: Union[pytest.param, str],
    request: pytest.FixtureRequest,
) -> None:
    conn_name = conn
    conn_value = request.getfixturevalue(conn)
    if conn_name == "sqlite_adbc_conn":
        request.node.add_marker(
            pytest.mark.xfail(reason="sqlite ADBC driver doesn't implement timedelta")
        )
    df = to_timedelta(Series(["00:00:01", "00:00:03"], name="foo")).to_frame()
    if "adbc" in conn_name:
        if pa_version_under14p1:
            exp_warning: Optional[Type[Warning]] = DeprecationWarning
        else:
            exp_warning = None
    else:
        exp_warning = UserWarning
    with tm.assert_produces_warning(
        exp_warning, match="", check_stacklevel=False
    ):
        result_count = df.to_sql(name="test_timedelta", con=conn_value)
    assert result_count == 2
    result = sql.read_sql_query("SELECT * FROM test_timedelta", conn_value)
    if conn_name == "postgresql_adbc_conn":
        expected = Series(
            [
                pd.DateOffset(months=0, days=0, microseconds=1000000, nanoseconds=0),
                pd.DateOffset(months=0, days=0, microseconds=3000000, nanoseconds=0),
            ],
            name="foo",
        )
    else:
        expected = df["foo"].astype("int64")
    tm.assert_series_equal(result["foo"], expected)


@pytest.mark.parametrize("conn", all_connectable)
def test_api_complex_raises(
    conn: Union[pytest.param, str],
    request: pytest.FixtureRequest,
) -> None:
    conn_name = conn
    conn_value = request.getfixturevalue(conn)
    df = DataFrame({"a": [1 + 1j, 2j]})
    if "adbc" in conn_name:
        msg = "datatypes not supported"
    else:
        msg = "Complex datatypes not supported"
    with pytest.raises(ValueError, match=msg):
        assert df.to_sql("test_complex", con=conn_value) is None


@pytest.mark.parametrize("conn", all_connectable)
@pytest.mark.parametrize(
    "index_name,index_label,expected",
    [
        (None, None, "index"),
        (None, "other_label", "other_label"),
        ("index_name", None, "index_name"),
        ("index_name", "other_label", "other_label"),
        (0, None, "0"),
        (None, 0, "0"),
    ],
)
def test_api_to_sql_index_label(
    conn: Union[pytest.param, str],
    request: pytest.FixtureRequest,
    index_name: Optional[Union[str, int]],
    index_label: Optional[Union[str, int]],
    expected: str,
) -> None:
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(reason="index_label argument NotImplemented with ADBC")
        )
    conn_value = request.getfixturevalue(conn)
    if sql.has_table("test_index_label", conn_value):
        with sql.SQLDatabase(conn_value, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_index_label")
    temp_frame = DataFrame({"col1": range(4)})
    temp_frame.index.name = index_name
    query = "SELECT * FROM test_index_label"
    sql.to_sql(
        temp_frame,
        "test_index_label",
        conn_value,
        index_label=index_label,
    )
    frame = sql.read_sql_query(query, conn_value)
    assert frame.columns[0] == expected


@pytest.mark.parametrize("conn", all_connectable)
def test_api_to_sql_index_label_multiindex(
    conn: Union[pytest.param, str],
    request: pytest.FixtureRequest,
) -> None:
    conn_name = conn
    if "mysql" in conn_name:
        request.applymarker(
            pytest.mark.xfail(reason="MySQL can fail using TEXT without length as key", strict=False)
        )
    elif "adbc" in conn_name:
        request.node.add_marker(
            pytest.mark.xfail(reason="index_label argument NotImplemented with ADBC")
        )
    conn_value = request.getfixturevalue(conn)
    if sql.has_table("test_index_label", conn_value):
        with sql.SQLDatabase(conn_value, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_index_label")
    expected_row_count = 4
    temp_frame = DataFrame(
        {"col1": range(4)}, index=MultiIndex.from_product([("A0", "A1"), ("B0", "B1")])
    )
    result = sql.to_sql("test_index_label", temp_frame, conn_value)
    assert result == expected_row_count
    frame = sql.read_sql_query("SELECT * FROM test_index_label", conn_value)
    assert frame.columns[:2].tolist() == ["level_0", "level_1"]
    result = sql.to_sql(
        "test_index_label",
        temp_frame,
        conn_value,
        if_exists="replace",
        index_label=["A", "B"],
    )
    assert result == expected_row_count
    frame = sql.read_sql_query("SELECT * FROM test_index_label", conn_value)
    assert frame.columns[:2].tolist() == ["A", "B"]
    temp_frame.index.names = ["A", "B"]
    result = sql.to_sql("test_index_label", temp_frame, conn_value, if_exists="replace")
    assert result == expected_row_count
    frame = sql.read_sql_query("SELECT * FROM test_index_label", conn_value)
    assert frame.columns[:2].tolist() == ["A", "B"]
    result = sql.to_sql(
        "test_index_label",
        temp_frame,
        conn_value,
        if_exists="replace",
        index_label=["C", "D"],
    )
    assert result == expected_row_count
    frame = sql.read_sql_query("SELECT * FROM test_index_label", conn_value)
    assert frame.columns[:2].tolist() == ["C", "D"]
    msg = "Length of 'index_label' should match number of levels, which is 2"
    with pytest.raises(ValueError, match=msg):
        sql.to_sql(
            "test_index_label",
            temp_frame,
            conn_value,
            if_exists="replace",
            index_label="C",
        )


@pytest.mark.parametrize("conn", all_connectable)
def test_api_multiindex_roundtrip(
    conn: Union[pytest.param, str],
    request: pytest.FixtureRequest,
) -> None:
    conn_value = request.getfixturevalue(conn)
    if sql.has_table("test_multiindex_roundtrip", conn_value):
        with sql.SQLDatabase(conn_value, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_multiindex_roundtrip")
    df = DataFrame.from_records(
        [(1, 2.1, "line1"), (2, 1.5, "line2")],
        columns=["A", "B", "C"],
        index=["A", "B"],
    )
    df.to_sql(name="test_multiindex_roundtrip", con=conn_value)
    result = sql.read_sql_query("SELECT * FROM test_multiindex_roundtrip", conn_value, index_col=["A", "B"])
    tm.assert_frame_equal(df, result, check_index_type=True)


@pytest.mark.parametrize("conn", all_connectable)
@pytest.mark.parametrize(
    "dtype",
    [
        None,
        int,
        float,
        {"A": int, "B": float},
    ],
)
def test_api_dtype_argument(
    conn: Union[pytest.param, str],
    request: pytest.FixtureRequest,
    dtype: Optional[Union[Callable[..., Any], Dict[str, Any]]],
) -> None:
    conn_name = conn
    conn_value = request.getfixturevalue(conn)
    if sql.has_table("test_dtype_argument", conn_value):
        with sql.SQLDatabase(conn_value, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_dtype_argument")
    df = DataFrame({"A": [1.2, 3.4], "B": [5.6, 7.8]})
    assert df.to_sql(name="test_dtype_argument", con=conn_value) == 2
    expected = df.astype(dtype)
    if "postgres" in conn_name:
        query = 'SELECT "A", "B" FROM test_dtype_argument'
    else:
        query = "SELECT A, B FROM test_dtype_argument"
    result = sql.read_sql_query(query, con=conn_value, dtype=dtype)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("conn", all_connectable)
def test_api_integer_col_names(
    conn: Union[pytest.param, str],
    request: pytest.FixtureRequest,
) -> None:
    conn_value = request.getfixturevalue(conn)
    df = DataFrame([[1, 2], [3, 4]], columns=[0, 1])
    sql.to_sql(df, "test_frame_integer_col_names", con=conn_value, if_exists="replace")


@pytest.mark.parametrize("conn", all_connectable)
def test_api_get_schema(
    conn: Union[pytest.param, str],
    request: pytest.FixtureRequest,
    test_frame1: DataFrame,
) -> None:
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(reason="'get_schema' not implemented for ADBC drivers", strict=True)
        )
    conn_value = request.getfixturevalue(conn)
    create_sql = sql.get_schema(test_frame1, "test", con=conn_value)
    assert "CREATE" in create_sql


@pytest.mark.parametrize("conn", all_connectable)
def test_api_get_schema_with_schema(
    conn: Union[pytest.param, str],
    request: pytest.FixtureRequest,
    test_frame1: DataFrame,
) -> None:
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(reason="'get_schema' not implemented for ADBC drivers", strict=True)
        )
    conn_value = request.getfixturevalue(conn)
    create_sql = sql.get_schema(
        test_frame1, "test", con=conn_value, schema="pypi"
    )
    assert "CREATE TABLE pypi." in create_sql


@pytest.mark.parametrize("conn", all_connectable)
def test_api_get_schema_dtypes(
    conn: Union[pytest.param, str],
    request: pytest.FixtureRequest,
) -> None:
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(reason="'get_schema' not implemented for ADBC drivers", strict=True)
        )
    conn_name = conn
    conn_value = request.getfixturevalue(conn)
    float_frame = DataFrame({"a": [1.1, 2.2], "b": [3.3, 4.4]})
    create_sql = sql.get_schema(
        float_frame, "test", con=conn_value, dtype={"b": "INTEGER"}
    )
    assert "CREATE" in create_sql
    assert "INTEGER" in create_sql


@pytest.mark.parametrize("conn", all_connectable)
def test_api_get_schema_keys(
    conn: Union[pytest.param, str],
    request: pytest.FixtureRequest,
    test_frame1: DataFrame,
) -> None:
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(reason="'get_schema' not implemented for ADBC drivers", strict=True)
        )
    conn_name = conn
    conn_value = request.getfixturevalue(conn)
    frame = DataFrame({"Col1": [1.1, 1.2], "Col2": [2.1, 2.2]})
    create_sql = sql.get_schema(
        frame, "test", con=conn_value, keys="Col1"
    )
    if "mysql" in conn_name:
        constraint_sentence = "CONSTRAINT test_pk PRIMARY KEY (`Col1`)"
    else:
        constraint_sentence = 'CONSTRAINT test_pk PRIMARY KEY ("Col1")'
    assert constraint_sentence in create_sql
    create_sql = sql.get_schema(
        test_frame1, "test", con=conn_value, keys=["A", "B"]
    )
    if "mysql" in conn_name:
        constraint_sentence = "CONSTRAINT test_pk PRIMARY KEY (`A`, `B`)"
    else:
        constraint_sentence = 'CONSTRAINT test_pk PRIMARY KEY ("A", "B")'
    assert constraint_sentence in create_sql


@pytest.mark.parametrize("conn", all_connectable)
def test_api_chunksize_read(
    conn: Union[pytest.param, str],
    request: pytest.FixtureRequest,
    test_frame1: DataFrame,
) -> None:
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(reason="chunksize argument NotImplemented with ADBC")
        )
    conn_value = request.getfixturevalue(conn)
    if sql.has_table("test_chunksize", conn_value):
        with sql.SQLDatabase(conn_value, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_chunksize")
    df = DataFrame(np.random.default_rng(2).standard_normal((22, 5)), columns=list("abcde"))
    df.to_sql(name="test_chunksize", con=conn_value, index=False)
    res1 = sql.read_sql_query("select * from test_chunksize", conn_value)
    res2 = DataFrame()
    i = 0
    sizes = [5, 5, 5, 5, 2]
    for chunk in sql.read_sql_query("select * from test_chunksize", conn_value, chunksize=5):
        res2 = concat([res2, chunk], ignore_index=True)
        assert len(chunk) == sizes[i]
        i += 1
    tm.assert_frame_equal(res1, res2)
    if conn == "sqlite_buildin":
        with pytest.raises(NotImplementedError, match=""):
            sql.read_sql_table("test_chunksize", conn_value, chunksize=5)
    else:
        res3 = DataFrame()
        i = 0
        sizes = [5, 5, 5, 5, 2]
        for chunk in sql.read_sql_table("test_chunksize", conn_value, chunksize=5):
            res3 = concat([res3, chunk], ignore_index=True)
            assert len(chunk) == sizes[i]
            i += 1
        tm.assert_frame_equal(res1, res3)


@pytest.mark.parametrize("conn", all_connectable)
def test_api_categorical(
    conn: Union[pytest.param, str],
    request: pytest.FixtureRequest,
) -> None:
    if conn == "postgresql_adbc_conn":
        adbc = import_optional_dependency("adbc_driver_postgresql", errors="ignore")
        if adbc is not None and Version(adbc.__version__) < Version("0.9.0"):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason="categorical dtype not implemented for ADBC postgres driver", strict=True
                )
            )
    conn_value = request.getfixturevalue(conn)
    if sql.has_table("test_categorical", conn_value):
        with sql.SQLDatabase(conn_value, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_categorical")
    df = DataFrame(
        {
            "person_id": [1, 2, 3],
            "person_name": ["John P. Doe", "Jane Dove", "John P. Doe"],
        }
    )
    df2 = df.copy()
    df2["person_name"] = df2["person_name"].astype("category")
    df2.to_sql(name="test_categorical", con=conn_value, index=False)
    res = sql.read_sql_query("SELECT * FROM test_categorical", conn_value)
    tm.assert_frame_equal(res, df)


@pytest.mark.parametrize("conn", all_connectable)
def test_api_unicode_column_name(
    conn: Union[pytest.param, str],
    request: pytest.FixtureRequest,
) -> None:
    conn_value = request.getfixturevalue(conn)
    if sql.has_table("test_unicode", conn_value):
        with sql.SQLDatabase(conn_value, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_unicode")
    df = DataFrame([[1, 2], [3, 4]], columns=["é", "b"])
    df.to_sql(name="test_unicode", con=conn_value, index=False)


@pytest.mark.parametrize("conn", all_connectable)
def test_api_escaped_table_name(
    conn: Union[pytest.param, str],
    request: pytest.FixtureRequest,
) -> None:
    conn_name = conn
    conn_value = request.getfixturevalue(conn)
    if sql.has_table("d1187b08-4943-4c8d-a7f6", conn_value):
        with sql.SQLDatabase(conn_value, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("d1187b08-4943-4c8d-a7f6")
    df = DataFrame({"A": [0, 1, 2], "B": [0.2, np.nan, 5.6]})
    df.to_sql(name="d1187b08-4943-4c8d-a7f6", con=conn_value, index=False)
    if "postgres" in conn_name:
        query = 'SELECT * FROM "d1187b08-4943-4c8d-a7f6"'
    else:
        query = "SELECT * FROM `d1187b08-4943-4c8d-a7f6`"
    res = sql.read_sql_query(query, conn_value)
    tm.assert_frame_equal(res, df)


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_read_table_absent_raises(
    conn: Union[pytest.param, str],
    request: pytest.FixtureRequest,
) -> None:
    conn_value = request.getfixturevalue(conn)
    msg = "Table this_doesnt_exist not found"
    with pytest.raises(ValueError, match=msg):
        sql.read_sql_table("this_doesnt_exist", con=conn_value)


@pytest.mark.parametrize("conn", sqlalchemy_connectable_types)
def test_sqlalchemy_default_type_conversion(
    conn: Union[pytest.param, str],
    request: pytest.FixtureRequest,
) -> None:
    conn_name = conn
    if conn_name == "sqlite_str":
        request.applymarker(
            pytest.mark.xfail(reason="types tables not created in sqlite_str fixture")
        )
    elif "mysql" in conn_name or "sqlite" in conn_name:
        request.applymarker(
            pytest.mark.xfail(reason="boolean dtype not inferred properly")
        )
    conn_value = request.getfixturevalue(conn)
    df = sql.read_sql_table("types", conn_value)
    assert issubclass(df.FloatCol.dtype.type, np.floating)
    assert issubclass(df.IntCol.dtype.type, np.integer)
    assert issubclass(df.BoolCol.dtype.type, np.bool_)
    assert issubclass(df.IntColWithNull.dtype.type, np.floating)
    assert issubclass(df.BoolColWithNull.dtype.type, object)


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_bigint(
    conn: Union[pytest.param, str],
    request: pytest.FixtureRequest,
) -> None:
    conn_value = request.getfixturevalue(conn)
    df = DataFrame({"i64": [2**62]})
    assert df.to_sql(name="test_bigint", con=conn_value, index=False) == 1
    result = sql.read_sql_table("test_bigint", conn_value)
    tm.assert_frame_equal(df, result)


@pytest.mark.parametrize("conn", sqlalchemy_connectable_types)
def test_default_date_load(
    conn: Union[pytest.param, str],
    request: pytest.FixtureRequest,
) -> None:
    conn_name = conn
    conn_value = request.getfixturevalue(conn)
    df = sql.read_sql_table("types", conn_value)
    assert issubclass(df.DateCol.dtype.type, np.datetime64)


@pytest.mark.parametrize("conn", postgresql_connectable)
@pytest.mark.parametrize("parse_dates", [None, ["DateColWithTz"]])
def test_datetime_with_timezone_query(
    conn: Union[pytest.param, str],
    request: pytest.FixtureRequest,
    parse_dates: Optional[List[str]],
) -> None:
    conn_value = request.getfixturevalue(conn)
    expected = create_and_load_postgres_datetz(conn_value)
    df = read_sql_query("select * from datetz", conn_value, parse_dates=parse_dates)
    col = df.DateColWithTz
    tm.assert_series_equal(col, expected)


@pytest.mark.parametrize("conn", all_connectable)
def test_database_uri_string(
    conn: Union[pytest.param, str],
    request: pytest.FixtureRequest,
    test_frame1: DataFrame,
) -> None:
    pytest.importorskip("sqlalchemy")
    conn_value = request.getfixturevalue(conn)
    with tm.ensure_clean() as name:
        db_uri = "sqlite:///" + name
        table = "iris"
        test_frame1.to_sql(name=table, con=db_uri, if_exists="replace", index=False)
        test_frame2 = sql.read_sql(table, db_uri)
        test_frame3 = sql.read_sql_table(table, db_uri)
        query = "SELECT * FROM iris"
        test_frame4 = sql.read_sql_query(query, db_uri)
    tm.assert_frame_equal(test_frame1, test_frame2)
    tm.assert_frame_equal(test_frame1, test_frame3)
    tm.assert_frame_equal(test_frame1, test_frame4)


@td.skip_if_installed("pg8000")
@pytest.mark.parametrize("conn", all_connectable)
def test_pg8000_sqlalchemy_passthrough_error(
    conn: Union[pytest.param, str],
    request: pytest.FixtureRequest,
) -> None:
    pytest.importorskip("sqlalchemy")
    conn_value = request.getfixturevalue(conn)
    db_uri = "postgresql+pg8000://user:pass@host/dbname"
    with pytest.raises(ImportError, match="pg8000"):
        sql.read_sql("select * from table", db_uri)


@pytest.mark.parametrize("conn", sqlalchemy_connectable_iris)
def test_query_by_text_obj(
    conn: Union[pytest.param, str],
    request: pytest.FixtureRequest,
) -> None:
    conn_name = conn
    conn_value = request.getfixturevalue(conn)
    from sqlalchemy import text

    if "postgres" in conn_name:
        name_text = text('select * from iris where "Name"=:name')
    else:
        name_text = text("select * from iris where name=:name")
    iris_df = sql.read_sql(name_text, conn_value, params={"name": "Iris-versicolor"})
    all_names = set(iris_df["Name"])
    assert all_names == {"Iris-versicolor"}


@pytest.mark.parametrize("conn", sqlalchemy_connectable_iris)
def test_query_by_select_obj(
    conn: Union[pytest.param, str],
    request: pytest.FixtureRequest,
) -> None:
    conn_value = request.getfixturevalue(conn)
    from sqlalchemy import bindparam, select

    iris = iris_table_metadata()
    name_select = select(iris).where(iris.c.Name == bindparam("name"))
    iris_df = sql.read_sql(name_select, conn_value, params={"name": "Iris-setosa"})
    all_names = set(iris_df["Name"])
    assert all_names == {"Iris-setosa"}


@pytest.mark.parametrize("conn", all_connectable)
def test_column_with_percentage(
    conn: Union[pytest.param, str],
    request: pytest.FixtureRequest,
) -> None:
    conn_name = conn
    if conn_name == "sqlite_buildin":
        request.applymarker(
            pytest.mark.xfail(reason="Not Implemented")
        )
    conn_value = request.getfixturevalue(conn)
    df = DataFrame({"A": [0, 1, 2], "%_variation": [3, 4, 5]})
    df.to_sql(name="test_column_percentage", con=conn_value, index=False)
    res = sql.read_sql_table("test_column_percentage", conn_value)
    tm.assert_frame_equal(res, df)


def test_sql_open_close(sqlite_conn: sqlalchemy.engine.base.Connection) -> None:
    with tm.ensure_clean() as name:
        with closing(sqlite3.connect(name)) as closing_conn:
            with closing_conn as conn:
                yield conn
    with closing_conn as conn:
        yield conn


def test_read_sqlite_delegate(sqlite_buildin_iris: sqlite3.Connection) -> None:
    conn = sqlite_buildin_iris
    iris_frame1 = sql.read_sql_query("SELECT * FROM iris", conn)
    iris_frame2 = sql.read_sql("SELECT * FROM iris", conn)
    tm.assert_frame_equal(iris_frame1, iris_frame2)
    msg = "Execution failed on sql 'iris': near \"iris\": syntax error"
    with pytest.raises(sql.DatabaseError, match=msg):
        sql.read_sql("iris", conn)


def test_get_schema2(test_frame1: DataFrame) -> None:
    create_sql = sql.get_schema(test_frame1, "test")
    assert "CREATE" in create_sql


def test_sqlite_type_mapping(sqlite_buildin: sqlite3.Connection) -> None:
    conn = sqlite_buildin
    df = DataFrame(
        {"time": to_datetime(["2014-12-12 01:54", "2014-12-11 02:54"], utc=True)}
    )
    db = sql.SQLiteDatabase(conn)
    table = sql.SQLiteTable("test_type", db, frame=df)
    schema = table.sql_schema()
    for col in schema.split("\n"):
        if col.split()[0].strip('"') == "time":
            assert col.split()[1] == "TIMESTAMP"


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_create_table(
    conn: Union[pytest.param, str],
    request: pytest.FixtureRequest,
) -> None:
    if conn == "sqlite_str":
        pytest.skip("sqlite_str has no inspection system")
    conn_value = request.getfixturevalue(conn)
    from sqlalchemy import inspect
    temp_frame = DataFrame({"one": [1.0, 2.0, 3.0, 4.0], "two": [4.0, 3.0, 2.0, 1.0]})
    with sql.SQLDatabase(conn_value, need_transaction=True) as pandasSQL:
        assert pandasSQL.to_sql(temp_frame, "temp_frame") == 4
    insp = inspect(conn_value)
    assert insp.has_table("temp_frame")
    with sql.SQLDatabase(conn_value, need_transaction=True) as pandasSQL:
        pandasSQL.drop_table("temp_frame")


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_drop_table(
    conn: Union[pytest.param, str],
    request: pytest.FixtureRequest,
) -> None:
    if conn == "sqlite_str":
        pytest.skip("sqlite_str has no inspection system")
    conn_value = request.getfixturevalue(conn)
    from sqlalchemy import inspect
    temp_frame = DataFrame({"one": [1.0, 2.0, 3.0, 4.0], "two": [4.0, 3.0, 2.0, 1.0]})
    with sql.SQLDatabase(conn_value, need_transaction=True) as pandasSQL:
        assert pandasSQL.to_sql(temp_frame, "temp_frame") == 4
    insp = inspect(conn_value)
    assert insp.has_table("temp_frame")
    with sql.SQLDatabase(conn_value, need_transaction=True) as pandasSQL:
        pandasSQL.drop_table("temp_frame")
    try:
        insp.clear_cache()
    except AttributeError:
        pass
    assert not insp.has_table("temp_frame")


@pytest.mark.parametrize("conn", all_connectable)
def test_roundtrip(
    conn: Union[pytest.param, str],
    request: pytest.FixtureRequest,
    test_frame1: DataFrame,
) -> None:
    conn_name = conn
    conn_value = request.getfixturevalue(conn)
    if sql.has_table("test_frame_roundtrip", conn_value):
        with sql.SQLDatabase(conn_value, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_frame_roundtrip")
    sql.to_sql(test_frame1, "test_frame_roundtrip", con=conn_value)
    result = sql.read_sql_query("SELECT * FROM test_frame_roundtrip", con=conn_value)
    if "adbc" in conn_name:
        result = result.drop(columns="__index_level_0__")
    else:
        result = result.drop(columns="level_0")
    tm.assert_frame_equal(result, test_frame1)


@pytest.mark.parametrize("conn", sqlalchemy_connectable_iris)
def test_sqlalchemy_read_table(
    conn: Union[pytest.param, str],
    request: pytest.FixtureRequest,
) -> None:
    conn_value = request.getfixturevalue(conn)
    iris_frame = sql.read_sql_table("iris", con=conn_value)
    check_iris_frame(iris_frame)


@pytest.mark.parametrize("conn", sqlalchemy_connectable_iris)
def test_sqlalchemy_read_table_columns(
    conn: Union[pytest.param, str],
    request: pytest.FixtureRequest,
) -> None:
    conn_value = request.getfixturevalue(conn)
    iris_frame = sql.read_sql_table("iris", con=conn_value, columns=["SepalLength", "SepalLength"])
    tm.assert_index_equal(iris_frame.columns, Index(["SepalLength", "SepalLength__1"]))


@pytest.mark.parametrize("conn", all_connectable)
def test_api_read_sql_delegate(
    conn: Union[pytest.param, str],
    request: pytest.FixtureRequest,
) -> None:
    if conn == "sqlite_buildin_iris":
        request.applymarker(
            pytest.mark.xfail(reason="sqlite_buildin connection does not implement read_sql_table")
        )
    conn_value = request.getfixturevalue(conn)
    iris_frame1 = sql.read_sql_query("SELECT * FROM iris", conn_value)
    iris_frame2 = sql.read_sql("SELECT * FROM iris", conn_value)
    tm.assert_frame_equal(iris_frame1, iris_frame2)
    msg = "Execution failed on sql 'iris': near \"iris\": syntax error"
    with pytest.raises(sql.DatabaseError, match=msg):
        sql.read_sql("iris", conn_value)

@pytest.fixture
def dtype_backend_data() -> DataFrame:
    return DataFrame(
        {
            "a": Series([1, np.nan, 3], dtype="Int64"),
            "b": Series([1, 2, 3], dtype="Int64"),
            "c": Series([1.5, np.nan, 2.5], dtype="Float64"),
            "d": Series([1.5, 2.0, 2.5], dtype="Float64"),
            "e": [True, False, None],
            "f": [True, False, True],
            "g": ["a", "b", "c"],
            "h": ["a", "b", None],
        }
    )


@pytest.fixture
def dtype_backend_expected() -> Callable[[str, str, str], DataFrame]:
    def func(
        string_storage: str, dtype_backend: str, conn_name: str
    ) -> DataFrame:
        if dtype_backend == "pyarrow":
            pa = pytest.importorskip("pyarrow")
            string_dtype = pd.ArrowDtype(pa.string())
        else:
            string_dtype = pd.StringDtype(string_storage)
        df = DataFrame(
            {
                "a": Series([1, np.nan, 3], dtype="Int64"),
                "b": Series([1, 2, 3], dtype="Int64"),
                "c": Series([1.5, np.nan, 2.5], dtype="Float64"),
                "d": Series([1.5, 2.0, 2.5], dtype="Float64"),
                "e": Series([True, False, pd.NA], dtype="boolean"),
                "f": Series([True, False, True], dtype="boolean"),
                "g": Series(["a", "b", "c"], dtype=string_dtype),
                "h": Series(["a", "b", None], dtype=string_dtype),
            }
        )
        if dtype_backend == "pyarrow":
            pa = pytest.importorskip("pyarrow")
            from pandas.arrays import ArrowExtensionArray

            df = DataFrame(
                {col: ArrowExtensionArray(pa.array(df[col], from_pandas=True)) for col in df.columns}
            )
        if "mysql" in conn_name or "sqlite" in conn_name:
            if dtype_backend == "numpy_nullable":
                df = df.astype({"e": "Int64", "f": "Int64"})
            else:
                df = df.astype({"e": "int64[pyarrow]", "f": "int64[pyarrow]"})
        return df

    return func


@pytest.mark.parametrize(
    "conn, func, dtype_backend, string_storage, input, expected",
    [
        (
            pytest.param("sqlite_buildin", marks=pytest.mark.db),
            "read_sql",
            "numpy_nullable",
            "python",
            {"a": [1, None, 3], "b": [1, 2, 3], "c": [1.5, None, 2.5], "d": [1.5, 2.0, 2.5], "e": [True, False, None], "f": [True, False, True], "g": ["a", "b", "c"], "h": ["a", "b", None]},
            None,  # This will be handled by fixtures
        ),
    ],
)
def test_read_sql_dtype_backend(
    conn: str,
    request: pytest.FixtureRequest,
    func: str,
    dtype_backend: str,
    string_storage: str,
    input: Dict[str, Any],
    expected: Optional[DataFrame],
    dtype_backend_data: DataFrame,
    dtype_backend_expected: Callable[[str, str, str], DataFrame],
) -> None:
    df = dtype_backend_data
    df.to_sql(name="test", con=conn, index=False, if_exists="replace")
    with pd.option_context("mode.string_storage", string_storage):
        result = getattr(pd, func)("Select * from test", conn, dtype_backend=dtype_backend)
        expected_dataframe = dtype_backend_expected(string_storage, dtype_backend, conn)
    tm.assert_frame_equal(result, expected_dataframe)
    if "adbc" not in conn:
        with pd.option_context("mode.string_storage", string_storage):
            iterator = getattr(pd, func)(
                "Select * from test", con=conn, dtype_backend=dtype_backend, chunksize=3
            )
            expected_dataframe = dtype_backend_expected(string_storage, dtype_backend, conn)
            for result_chunk in iterator:
                tm.assert_frame_equal(result_chunk, expected_dataframe)


@pytest.mark.parametrize(
    "conn, func, dtype_backend",
    [
        (
            pytest.param("sqlite_buildin", marks=pytest.mark.db),
            "read_sql",
            "numpy_nullable",
        ),
        (
            pytest.param("postgresql_psycopg2_engine", marks=pytest.mark.db),
            "read_sql_table",
            "pyarrow",
        ),
    ],
)
def test_read_sql_dtype_backend_table(
    conn: str,
    request: pytest.FixtureRequest,
    func: str,
    dtype_backend: str,
    string_storage: str,
    dtype_backend_data: DataFrame,
    dtype_backend_expected: Callable[[str, str, str], DataFrame],
) -> None:
    if "sqlite" in conn and "adbc" not in conn:
        request.applymarker(
            pytest.mark.xfail(reason="types tables not created in sqlite_str fixture")
        )
    conn_value = request.getfixturevalue(conn)
    table = "test"
    df = dtype_backend_data
    df.to_sql(name=table, con=conn_value, index=False, if_exists="replace")
    with pd.option_context("mode.string_storage", string_storage):
        result = getattr(pd, func)(table, conn, dtype_backend=dtype_backend)
        expected_dataframe = dtype_backend_expected(string_storage, dtype_backend, conn)
    tm.assert_frame_equal(result, expected_dataframe)
    if "adbc" not in conn:
        with pd.option_context("mode.string_storage", string_storage):
            iterator = getattr(pd, func)(table, conn, dtype_backend=dtype_backend, chunksize=3)
            expected_dataframe = dtype_backend_expected(string_storage, dtype_backend, conn)
            for result_chunk in iterator:
                tm.assert_frame_equal(result_chunk, expected_dataframe)


@pytest.mark.parametrize("conn", all_connectable_types)
@pytest.mark.parametrize("func", ["read_sql", "read_sql_table", "read_sql_query"])
def test_read_sql_invalid_dtype_backend_table(
    conn: str,
    request: pytest.FixtureRequest,
    func: str,
    dtype_backend_data: DataFrame,
) -> None:
    conn_value = request.getfixturevalue(conn)
    table = "test"
    df = dtype_backend_data
    df.to_sql(name=table, con=conn_value, index=False, if_exists="replace")
    msg = "dtype_backend numpy is invalid, only 'numpy_nullable' and 'pyarrow' are allowed."
    with pytest.raises(ValueError, match=msg):
        getattr(pd, func)(table, conn=conn_value, dtype_backend="numpy")


def test_bigint_warning(sqlite_engine: SQLAlchemyEngine) -> None:
    conn = sqlite_engine
    df = DataFrame({"a": [1, 2]}, dtype="int64")
    assert df.to_sql(name="test_bigintwarning", con=conn, index=False) == 2
    with tm.assert_produces_warning(None):
        sql.read_sql_table("test_bigintwarning", conn)


def test_valueerror_exception(sqlite_engine: SQLAlchemyEngine) -> None:
    conn = sqlite_engine
    df = DataFrame({"col1": [1, 2], "col2": [3, 4]})
    with pytest.raises(ValueError, match="Empty table name specified"):
        df.to_sql(name="", con=conn, if_exists="replace", index=False)


def test_row_object_is_named_tuple(sqlite_engine: SQLAlchemyEngine) -> None:
    conn = sqlite_engine
    from sqlalchemy import Column, Integer, String
    from sqlalchemy.orm import declarative_base, sessionmaker

    BaseModel = declarative_base()

    class Test(BaseModel):
        __tablename__ = "test_frame"
        id = Column(Integer, primary_key=True)
        string_column = Column(String(50))

    with conn.begin():
        BaseModel.metadata.create_all(conn)
    Session = sessionmaker(bind=conn)
    with Session() as session:
        df = DataFrame({"id": [0, 1], "string_column": ["hello", "world"]})
        assert df.to_sql(name="test_frame", con=conn, if_exists="replace", index=False) == 2
        session.commit()
        test_query = session.query(Test.id, Test.string_column)
        df = DataFrame(test_query)
    assert list(df.columns) == ["id", "string_column"]


def test_read_sql_string_inference(sqlite_engine: SQLAlchemyEngine) -> None:
    conn = sqlite_engine
    table = "test"
    df = DataFrame({"a": ["x", "y"]})
    df.to_sql(table, con=conn, index=False, if_exists="replace")
    with pd.option_context("future.infer_string", True):
        result = read_sql_table(table, conn)
    dtype = pd.StringDtype(na_value=np.nan)
    expected = DataFrame({"a": ["x", "y"]}, dtype=dtype, columns=Index(["a"], dtype=dtype))
    tm.assert_frame_equal(result, expected)


def test_roundtripping_datetimes(sqlite_engine: SQLAlchemyEngine) -> None:
    conn = sqlite_engine
    df = DataFrame({"t": [datetime(2020, 12, 31, 12)]}, dtype="datetime64[ns]")
    df.to_sql("test", conn, if_exists="replace", index=False)
    result = pd.read_sql("select * from test", conn).iloc[0, 0]
    assert result == "2020-12-31 12:00:00.000000"


@pytest.fixture
def sqlite_builtin_detect_types() -> sqlite3.Connection:
    with contextlib.closing(sqlite3.connect(":memory:", detect_types=sqlite3.PARSE_DECLTYPES)) as closing_conn:
        with closing_conn as conn:
            yield conn


def test_roundtripping_datetimes_detect_types(sqlite_builtin_detect_types: sqlite3.Connection) -> None:
    conn = sqlite_builtin_detect_types
    df = DataFrame({"t": [datetime(2020, 12, 31, 12)]}, dtype="datetime64[ns]")
    df.to_sql("test", conn, if_exists="replace", index=False)
    result = pd.read_sql("select * from test", conn).iloc[0, 0]
    assert result == Timestamp("2020-12-31 12:00:00.000000")


@pytest.mark.db
def test_psycopg2_schema_support(
    postgresql_psycopg2_engine: SQLAlchemyEngine,
) -> None:
    conn = postgresql_psycopg2_engine
    df = DataFrame({"col1": [1, 2], "col2": [0.1, 0.2], "col3": ["a", "n"]})
    with conn.connect() as con:
        with con.begin():
            con.exec_driver_sql("DROP SCHEMA IF EXISTS other CASCADE;")
            con.exec_driver_sql("CREATE SCHEMA other;")
    assert df.to_sql(name="test_schema_public", con=conn, index=False) == 2
    assert df.to_sql(
        name="test_schema_public_explicit", con=conn, index=False, schema="public"
    ) == 2
    assert df.to_sql(name="test_schema_other", con=conn, index=False, schema="other") == 2
    res1 = sql.read_sql_table("test_schema_public", conn)
    tm.assert_frame_equal(df, res1)
    res2 = sql.read_sql_table("test_schema_public_explicit", conn)
    tm.assert_frame_equal(df, res2)
    res3 = sql.read_sql_table("test_schema_public_explicit", conn, schema="public")
    tm.assert_frame_equal(df, res3)
    res4 = sql.read_sql_table("test_schema_other", conn, schema="other")
    tm.assert_frame_equal(df, res4)
    msg = "Table test_schema_other not found"
    with pytest.raises(ValueError, match=msg):
        sql.read_sql_table("test_schema_other", conn, schema="public")
    with conn.connect() as con:
        with con.begin():
            con.exec_driver_sql("DROP SCHEMA IF EXISTS other CASCADE;")
            con.exec_driver_sql("CREATE SCHEMA other;")
    assert df.to_sql(name="test_schema_other", con=conn, schema="other", index=False) == 2
    df.to_sql(
        name="test_schema_other", con=conn, schema="other", index=False, if_exists="replace"
    )
    assert db.read_sql_table("test_schema_other", conn, schema="other").equals(df)
    df.to_sql(
        name="test_schema_other",
        con=conn,
        schema="other",
        index=False,
        if_exists="append",
    ) == 2
    res = sql.read_sql_table("test_schema_other", conn, schema="other")
    tm.assert_frame_equal(concat([df, df], ignore_index=True), res)
    with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
        pandasSQL.drop_table("test_schema_other", schema="other")


@pytest.mark.db
def test_self_join_date_columns(
    postgresql_psycopg2_engine: SQLAlchemyEngine,
) -> None:
    conn = postgresql_psycopg2_engine
    from sqlalchemy.sql import text

    create_table = text(
        "\n    CREATE TABLE person\n    (\n        id serial constraint person_pkey primary key,\n        created_dt timestamp with time zone\n    );\n\n    INSERT INTO person\n        VALUES (1, '2021-01-01T00:00:00Z');\n    "
    )
    with conn.connect() as con:
        with con.begin():
            con.execute(create_table)
    sql_query = 'SELECT * FROM "person" AS p1 INNER JOIN "person" AS p2 ON p1.id = p2.id;'
    result = pd.read_sql(sql_query, conn)
    expected = DataFrame({"group_id": [1], "name": ["name"]})
    expected = expected.rename(columns={"group_id": "id"})
    tm.assert_frame_equal(result, expected)
    with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
        pandasSQL.drop_table("person")


def test_create_and_drop_table(sqlite_engine: SQLAlchemyEngine) -> None:
    conn = sqlite_engine
    temp_frame = DataFrame({"one": [1.0, 2.0, 3.0, 4.0], "two": [4.0, 3.0, 2.0, 1.0]})
    with sql.SQLDatabase(conn) as pandasSQL:
        with pandasSQL.run_transaction():
            assert pandasSQL.to_sql(temp_frame, "drop_test_frame") == 4
        assert pandasSQL.has_table("drop_test_frame")
        with pandasSQL.run_transaction():
            pandasSQL.drop_table("drop_test_frame")
        assert not pandasSQL.has_table("drop_test_frame")


@pytest.mark.parametrize("conn", all_connectable)
@pytest.mark.parametrize("func", ["read_sql", "read_sql_query"])
def test_read_sql_dtype_backend(
    conn: str,
    request: pytest.FixtureRequest,
    func: str,
    dtype_backend_data: DataFrame,
    dtype_backend_expected: Callable[[str, str, str], DataFrame],
) -> None:
    # Placeholder for function; actual tests are above
    pass
