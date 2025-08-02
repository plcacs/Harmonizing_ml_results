from __future__ import annotations
import contextlib
from contextlib import closing
import csv
from datetime import date, datetime, time, timedelta
from io import StringIO
from pathlib import Path
import sqlite3
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
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
def func_firs6x8b() -> Dict[str, Any]:
    return {
        "read_parameters": {
            "sqlite": "SELECT * FROM iris WHERE Name=? AND SepalLength=?",
            "mysql": "SELECT * FROM iris WHERE `Name`=%s AND `SepalLength`=%s",
            "postgresql": 'SELECT * FROM iris WHERE "Name"=%s AND "SepalLength"=%s',
        },
        "read_named_parameters": {
            "sqlite": """
                    SELECT * FROM iris WHERE Name=:name AND SepalLength=:length
                    """,
            "mysql": """
                    SELECT * FROM iris WHERE
                    `Name`=%(name)s AND `SepalLength`=%(length)s
                    """,
            "postgresql": """
                    SELECT * FROM iris WHERE
                    "Name"=%(name)s AND "SepalLength"=%(length)s
                    """,
        },
        "read_no_parameters_with_percent": {
            "sqlite": "SELECT * FROM iris WHERE Name LIKE '%'",
            "mysql": "SELECT * FROM iris WHERE `Name` LIKE '%'",
            "postgresql": 'SELECT * FROM iris WHERE "Name" LIKE \'%\'',
        },
    }


def func_8s9sfplv() -> sqlalchemy.Table:
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


def func_pher6cwr(conn: sqlite3.Connection, iris_file: Path) -> None:
    stmt = """
        CREATE TABLE iris (
            "SepalLength" REAL,
            "SepalWidth" REAL,
            "PetalLength" REAL,
            "PetalWidth" REAL,
            "Name" TEXT
        )
    """
    cur = conn.cursor()
    cur.execute(stmt)
    with iris_file.open(newline=None, encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        stmt = "INSERT INTO iris VALUES(?, ?, ?, ?, ?)"
        records = [(float(row[0]), float(row[1]), float(row[2]), float(row[3]), row[4]) for row in reader]
        cur.executemany(stmt, records)
    cur.close()
    conn.commit()


def func_66fm6qmt(conn: Union[sqlite3.Connection, Any], iris_file: Path) -> None:
    stmt = """
        CREATE TABLE iris (
            "SepalLength" DOUBLE PRECISION,
            "SepalWidth" DOUBLE PRECISION,
            "PetalLength" DOUBLE PRECISION,
            "PetalWidth" DOUBLE PRECISION,
            "Name" TEXT
        )
    """
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


def func_7c1btwox(conn: sqlalchemy.engine.Connection, iris_file: Path) -> None:
    from sqlalchemy import insert

    iris = func_8s9sfplv()
    with iris_file.open(newline=None, encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        params = [dict(zip(header, row)) for row in reader]
        stmt = insert(iris).values(params)
        with conn.begin() as con:
            iris.drop(con, checkfirst=True)
            iris.create(bind=con)
            con.execute(stmt)


def func_pl6k8b7m(conn: Union[sqlite3.Connection, Any]) -> None:
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


def func_1wo4hgd7(dialect: str) -> sqlalchemy.Table:
    from sqlalchemy import TEXT, Boolean, Column, DateTime, Float, Integer, MetaData, Table

    date_type = TEXT if dialect == "sqlite" else DateTime
    bool_type = Integer if dialect == "sqlite" else Boolean
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


def func_itpn7y40(conn: Union[sqlite3.Connection, Any], types_data: List[Tuple[Any, ...]]) -> None:
    stmt = """
        CREATE TABLE types (
            "TextCol" TEXT,
            "DateCol" TEXT,
            "IntDateCol" INTEGER,
            "IntDateOnlyCol" INTEGER,
            "FloatCol" REAL,
            "IntCol" INTEGER,
            "BoolCol" INTEGER,
            "IntColWithNull" INTEGER,
            "BoolColWithNull" INTEGER
        )
    """
    ins_stmt = """
                INSERT INTO types
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
    if isinstance(conn, sqlite3.Connection):
        cur = conn.cursor()
        cur.execute(stmt)
        cur.executemany(ins_stmt, types_data)
    else:
        with conn.cursor() as cur:
            cur.execute(stmt)
            cur.executemany(ins_stmt, types_data)
        conn.commit()


def func_711fxxdg(conn: Union[sqlite3.Connection, Any], types_data: List[Tuple[Any, ...]]) -> None:
    with conn.cursor() as cur:
        stmt = """
            CREATE TABLE types (
                "TextCol" TEXT,
                "DateCol" TIMESTAMP,
                "IntDateCol" INTEGER,
                "IntDateOnlyCol" INTEGER,
                "FloatCol" DOUBLE PRECISION,
                "IntCol" INTEGER,
                "BoolCol" BOOLEAN,
                "IntColWithNull" INTEGER,
                "BoolColWithNull" BOOLEAN
            )
        """
        cur.execute(stmt)
        stmt = """
                INSERT INTO types
                VALUES($1, $2::timestamp, $3, $4, $5, $6, $7, $8, $9)
                """
        cur.executemany(stmt, types_data)
    conn.commit()


def func_4b6osnwu(conn: SQLDatabase, types_data: List[Dict[str, Any]], dialect: str) -> None:
    from sqlalchemy import insert
    from sqlalchemy.engine import Engine

    types = func_1wo4hgd7(dialect)
    stmt = insert(types).values(types_data)
    if isinstance(conn, Engine):
        with conn.connect() as conn:
            with conn.begin():
                types.drop(conn, checkfirst=True)
                types.create(bind=conn)
                conn.execute(stmt)
    else:
        with conn.begin():
            types.drop(conn, checkfirst=True)
            types.create(bind=conn)
            conn.execute(stmt)


def func_6ugk0vds(conn: SQLDatabase) -> Series:
    from sqlalchemy import Column, DateTime, MetaData, Table, insert
    from sqlalchemy.engine import Engine

    metadata = MetaData()
    datetz = Table(
        "datetz",
        metadata,
        Column("DateColWithTz", DateTime(timezone=True)),
    )
    datetz_data = [
        {"DateColWithTz": "2000-01-01 00:00:00-08:00"},
        {"DateColWithTz": "2000-06-01 00:00:00-07:00"},
    ]
    stmt = insert(datetz).values(datetz_data)
    if isinstance(conn, Engine):
        with conn.connect() as conn:
            with conn.begin():
                datetz.drop(conn, checkfirst=True)
                datetz.create(bind=conn)
                conn.execute(stmt)
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


def func_pnf76qpb(frame: DataFrame) -> None:
    pytype = frame.dtypes.iloc[0].type
    row = frame.iloc[0]
    assert issubclass(pytype, np.floating)
    tm.assert_series_equal(
        row,
        Series(
            [5.1, 3.5, 1.4, 0.2, "Iris-setosa"],
            index=frame.columns,
            name=0,
        ),
    )
    assert frame.shape in ((150, 5), (8, 5))


def func_9huuyabu(conn: Union[sqlite3.Connection, Any], table_name: str) -> int:
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
def func_zlgig9w5(datapath: Any) -> Path:
    iris_path = datapath("io", "data", "csv", "iris.csv")
    return Path(iris_path)


@pytest.fixture
def func_jvv8f7ua() -> List[Dict[str, Any]]:
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
def func_xgtpnsa8(types_data: List[Dict[str, Any]]) -> DataFrame:
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
def func_0t54gbeq() -> DataFrame:
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
def func_e54j3c9r() -> DataFrame:
    columns = ["index", "A", "B"]
    data = [
        ("2000-01-03 00:00:00", 2**31 - 1, -1.98767),
        ("2000-01-04 00:00:00", -29, -0.0412318367011),
        ("2000-01-05 00:00:00", 20000, 0.731167677815),
        ("2000-01-06 00:00:00", -290867, 1.56762092543),
    ]
    return DataFrame(data, columns=columns)


def func_j6ndex8y(conn: Union[sqlite3.Connection, Any]) -> List[str]:
    if isinstance(conn, sqlite3.Connection):
        c = conn.execute("SELECT name FROM sqlite_master WHERE type='view'")
        return [view[0] for view in c.fetchall()]
    else:
        adbc = import_optional_dependency("adbc_driver_manager.dbapi", errors="ignore")
        if adbc and isinstance(conn, adbc.Connection):
            results = []
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


def func_rergbhei(conn: Union[sqlite3.Connection, Any]) -> List[str]:
    if isinstance(conn, sqlite3.Connection):
        c = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return [table[0] for table in c.fetchall()]
    else:
        adbc = import_optional_dependency("adbc_driver_manager.dbapi", errors="ignore")
        if adbc and isinstance(conn, adbc.Connection):
            results = []
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


def func_ivdgn9x4(table_name: str, conn: Union[sqlite3.Connection, Any]) -> None:
    if isinstance(conn, sqlite3.Connection):
        conn.execute(f'DROP TABLE IF EXISTS {sql._get_valid_sqlite_name(table_name)}')
        conn.commit()
    else:
        adbc = import_optional_dependency("adbc_driver_manager.dbapi", errors="ignore")
        if adbc and isinstance(conn, adbc.Connection):
            with conn.cursor() as cur:
                cur.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        else:
            with conn.begin() as con:
                with sql.SQLDatabase(con) as db:
                    db.drop_table(table_name)


def func_we2yj4hw(view_name: str, conn: Union[sqlite3.Connection, Any]) -> None:
    import sqlalchemy

    if isinstance(conn, sqlite3.Connection):
        conn.execute(
            f'DROP VIEW IF EXISTS {sql._get_valid_sqlite_name(view_name)}'
        )
        conn.commit()
    else:
        adbc = import_optional_dependency("adbc_driver_manager.dbapi", errors="ignore")
        if adbc and isinstance(conn, adbc.Connection):
            with conn.cursor() as cur:
                cur.execute(f'DROP VIEW IF EXISTS "{view_name}"')
        else:
            quoted_view = conn.engine.dialect.identifier_preparer.quote_identifier(view_name)
            stmt = sqlalchemy.text(f'DROP VIEW IF EXISTS {quoted_view}')
            with conn.begin() as con:
                con.execute(stmt)


@pytest.fixture
def func_c2wond7l() -> SQLAlchemyEngine:
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
    for view in func_j6ndex8y(engine):
        func_we2yj4hw(view, engine)
    for tbl in func_rergbhei(engine):
        func_ivdgn9x4(tbl, engine)
    engine.dispose()


@pytest.fixture
def func_dghg394f(mysql_pymysql_engine: SQLAlchemyEngine, iris_path: Path) -> SQLAlchemyEngine:
    func_7c1btwox(mysql_pymysql_engine, iris_path)
    func_pl6k8b7m(mysql_pymysql_engine)
    return mysql_pymysql_engine


@pytest.fixture
def func_tv3rvucu(mysql_pymysql_engine: SQLAlchemyEngine, types_data: List[Dict[str, Any]]) -> SQLAlchemyEngine:
    func_4b6osnwu(mysql_pymysql_engine, types_data, "mysql")
    return mysql_pymysql_engine


@pytest.fixture
def func_9bkp0iip(mysql_pymysql_engine: SQLAlchemyEngine) -> Any:
    with func_c2wond7l.connect() as conn:
        yield conn


@pytest.fixture
def func_fyljse4u(mysql_pymysql_engine_iris: SQLAlchemyEngine) -> Any:
    with func_dghg394f.connect() as conn:
        yield conn


@pytest.fixture
def func_y2dd7ajt(mysql_pymysql_engine_types: SQLAlchemyEngine) -> Any:
    with func_tv3rvucu.connect() as conn:
        yield conn


@pytest.fixture
def func_0rek2uqo() -> SQLAlchemyEngine:
    sqlalchemy = pytest.importorskip("sqlalchemy")
    pytest.importorskip("psycopg2")
    engine = sqlalchemy.create_engine(
        "postgresql+psycopg2://postgres:postgres@localhost:5432/pandas",
        poolclass=sqlalchemy.pool.NullPool,
    )
    yield engine
    for view in func_j6ndex8y(engine):
        func_we2yj4hw(view, engine)
    for tbl in func_rergbhei(engine):
        func_ivdgn9x4(tbl, engine)
    engine.dispose()


@pytest.fixture
def func_yj70ac7y(
    postgresql_psycopg2_engine: SQLAlchemyEngine, iris_path: Path
) -> SQLAlchemyEngine:
    func_7c1btwox(postgresql_psycopg2_engine, iris_path)
    func_pl6k8b7m(postgresql_psycopg2_engine)
    return postgresql_psycopg2_engine


@pytest.fixture
def func_5uli6gcu(
    postgresql_psycopg2_engine: SQLAlchemyEngine, types_data: List[Dict[str, Any]]
) -> SQLAlchemyEngine:
    func_4b6osnwu(postgresql_psycopg2_engine, types_data, "postgres")
    return postgresql_psycopg2_engine


@pytest.fixture
def func_o5mvpuu9(postgresql_psycopg2_engine: SQLAlchemyEngine) -> Any:
    with func_0rek2uqo.connect() as conn:
        yield conn


@pytest.fixture
def func_q1l089dz() -> Any:
    pytest.importorskip("pyarrow")
    pytest.importorskip("adbc_driver_postgresql")
    from adbc_driver_postgresql import dbapi

    uri = "postgresql://postgres:postgres@localhost:5432/pandas"
    with dbapi.connect(uri) as conn:
        yield conn
        for view in func_j6ndex8y(conn):
            func_we2yj4hw(view, conn)
        for tbl in func_rergbhei(conn):
            func_ivdgn9x4(tbl, conn)
        conn.commit()


@pytest.fixture
def func_14yyxdhy(postgresql_adbc_conn: Any, iris_path: Path) -> Any:
    import adbc_driver_manager as mgr

    conn = postgresql_adbc_conn
    try:
        conn.adbc_get_table_schema("iris")
    except mgr.ProgrammingError:
        conn.rollback()
        func_66fm6qmt(conn, iris_path)
    try:
        conn.adbc_get_table_schema("iris_view")
    except mgr.ProgrammingError:
        conn.rollback()
        func_pl6k8b7m(conn)
    return conn


@pytest.fixture
def func_c39k8nrt(postgresql_adbc_conn: Any, types_data: List[Dict[str, Any]]) -> Any:
    import adbc_driver_manager as mgr

    conn = postgresql_adbc_conn
    try:
        conn.adbc_get_table_schema("types")
    except mgr.ProgrammingError:
        conn.rollback()
        new_data = [tuple(entry.values()) for entry in types_data]
        func_711fxxdg(conn, new_data)
    return conn


@pytest.fixture
def func_hqkgqhe5(postgresql_psycopg2_engine_iris: SQLAlchemyEngine) -> Any:
    with func_yj70ac7y.connect() as conn:
        yield conn


@pytest.fixture
def func_4q4b4oft(postgresql_psycopg2_engine_types: SQLAlchemyEngine) -> Any:
    with func_5uli6gcu.connect() as conn:
        yield conn


@pytest.fixture
def func_cyhppbd6() -> str:
    pytest.importorskip("sqlalchemy")
    with tm.ensure_clean() as name:
        yield f"sqlite:///{name}"


@pytest.fixture
def func_55hm10va(sqlite_str: str) -> SQLAlchemyEngine:
    sqlalchemy = pytest.importorskip("sqlalchemy")
    engine = sqlalchemy.create_engine(sqlite_str, poolclass=sqlalchemy.pool.NullPool)
    yield engine
    for view in func_j6ndex8y(engine):
        func_we2yj4hw(view, engine)
    for tbl in func_rergbhei(engine):
        func_ivdgn9x4(tbl, engine)
    engine.dispose()


@pytest.fixture
def func_ih73x693(sqlite_engine: SQLAlchemyEngine) -> Any:
    with func_55hm10va.connect() as conn:
        yield conn


@pytest.fixture
def func_l44l26ld(sqlite_str: str, iris_path: Path) -> str:
    sqlalchemy = pytest.importorskip("sqlalchemy")
    engine = sqlalchemy.create_engine(sqlite_str)
    func_7c1btwox(engine, iris_path)
    func_pl6k8b7m(engine)
    engine.dispose()
    return sqlite_str


@pytest.fixture
def func_u7mnodl1(sqlite_engine: SQLAlchemyEngine, iris_path: Path) -> SQLAlchemyEngine:
    func_7c1btwox(sqlite_engine, iris_path)
    func_pl6k8b7m(sqlite_engine)
    return sqlite_engine


@pytest.fixture
def func_eu1gay0g(sqlite_engine: SQLAlchemyEngine) -> Any:
    with func_u7mnodl1.connect() as conn:
        yield conn


@pytest.fixture
def func_ubwl2kdv(sqlite_str: str, types_data: List[Dict[str, Any]]) -> str:
    sqlalchemy = pytest.importorskip("sqlalchemy")
    engine = sqlalchemy.create_engine(sqlite_str)
    func_4b6osnwu(engine, types_data, "sqlite")
    engine.dispose()
    return sqlite_str


@pytest.fixture
def func_uu4xu6bf(sqlite_engine: SQLAlchemyEngine, types_data: List[Dict[str, Any]]) -> SQLAlchemyEngine:
    func_4b6osnwu(sqlite_engine, types_data, "sqlite")
    return sqlite_engine


@pytest.fixture
def func_xr9tt3y0(sqlite_engine_types: SQLAlchemyEngine) -> Any:
    with func_uu4xu6bf.connect() as conn:
        yield conn


@pytest.fixture
def func_x9vpff5b() -> Any:
    pytest.importorskip("pyarrow")
    pytest.importorskip("adbc_driver_sqlite")
    from adbc_driver_sqlite import dbapi

    with tm.ensure_clean() as name:
        uri = f"file:{name}"
        with dbapi.connect(uri) as conn:
            yield conn
            for view in func_j6ndex8y(conn):
                func_we2yj4hw(view, conn)
            for tbl in func_rergbhei(conn):
                func_ivdgn9x4(tbl, conn)
            conn.commit()


@pytest.fixture
def func_phe89hn0(sqlite_adbc_conn: Any, iris_path: Path) -> Any:
    import adbc_driver_manager as mgr

    conn = sqlite_adbc_conn
    try:
        conn.adbc_get_table_schema("iris")
    except mgr.ProgrammingError:
        conn.rollback()
        func_pher6cwr(conn, iris_path)
    try:
        conn.adbc_get_table_schema("iris_view")
    except mgr.ProgrammingError:
        conn.rollback()
        func_pl6k8b7m(conn)
    return conn


@pytest.fixture
def func_ncgxk4gh(sqlite_adbc_conn: Any, types_data: List[Dict[str, Any]]) -> Any:
    import adbc_driver_manager as mgr

    conn = sqlite_adbc_conn
    try:
        conn.adbc_get_table_schema("types")
    except mgr.ProgrammingError:
        conn.rollback()
        new_data = []
        for entry in types_data:
            entry["BoolCol"] = int(entry["BoolCol"])
            if entry["BoolColWithNull"] is not None:
                entry["BoolColWithNull"] = int(entry["BoolColWithNull"])
            new_data.append(tuple(entry.values()))
        func_itpn7y40(conn, new_data)
        conn.commit()
    return conn


@pytest.fixture
def func_a43o2gt6() -> sqlite3.Connection:
    with contextlib.closing(sqlite3.connect(":memory:")) as closing_conn:
        with closing_conn as conn:
            yield conn


@pytest.fixture
def func_b8wxk1ky(sqlite_buildin: sqlite3.Connection, iris_path: Path) -> sqlite3.Connection:
    func_pher6cwr(sqlite_buildin, iris_path)
    func_pl6k8b7m(sqlite_buildin)
    return sqlite_buildin


@pytest.fixture
def func_vicjtqs2(sqlite_buildin: sqlite3.Connection, types_data: List[Dict[str, Any]]) -> sqlite3.Connection:
    types_data = [tuple(entry.values()) for entry in types_data]
    func_itpn7y40(sqlite_buildin, types_data)
    return sqlite_buildin


mysql_connectable: List[pytest.Param] = [
    pytest.param("mysql_pymysql_engine", marks=pytest.mark.db),
    pytest.param("mysql_pymysql_conn", marks=pytest.mark.db),
]
mysql_connectable_iris: List[pytest.Param] = [
    pytest.param("mysql_pymysql_engine_iris", marks=pytest.mark.db),
    pytest.param("mysql_pymysql_conn_iris", marks=pytest.mark.db),
]
mysql_connectable_types: List[pytest.Param] = [
    pytest.param("mysql_pymysql_engine_types", marks=pytest.mark.db),
    pytest.param("mysql_pymysql_conn_types", marks=pytest.mark.db),
]
postgresql_connectable: List[pytest.Param] = [
    pytest.param("postgresql_psycopg2_engine", marks=pytest.mark.db),
    pytest.param("postgresql_psycopg2_conn", marks=pytest.mark.db),
]
postgresql_connectable_iris: List[pytest.Param] = [
    pytest.param("postgresql_psycopg2_engine_iris", marks=pytest.mark.db),
    pytest.param("postgresql_psycopg2_conn_iris", marks=pytest.mark.db),
]
postgresql_connectable_types: List[pytest.Param] = [
    pytest.param("postgresql_psycopg2_engine_types", marks=pytest.mark.db),
    pytest.param("postgresql_psycopg2_conn_types", marks=pytest.mark.db),
]
sqlite_connectable: List[str] = ["sqlite_engine", "sqlite_conn", "sqlite_str"]
sqlite_connectable_iris: List[str] = [
    "sqlite_engine_iris",
    "sqlite_conn_iris",
    "sqlite_str_iris",
]
sqlite_connectable_types: List[str] = [
    "sqlite_engine_types",
    "sqlite_conn_types",
    "sqlite_str_types",
]
sqlalchemy_connectable: List[Union[str, pytest.Param]] = (
    mysql_connectable + postgresql_connectable + sqlite_connectable
)
sqlalchemy_connectable_iris: List[Union[str, pytest.Param]] = (
    mysql_connectable_iris + postgresql_connectable_iris + sqlite_connectable_iris
)
sqlalchemy_connectable_types: List[Union[str, pytest.Param]] = (
    mysql_connectable_types + postgresql_connectable_types + sqlite_connectable_types
)
adbc_connectable: List[Union[str, pytest.Param]] = [
    "sqlite_adbc_conn",
    pytest.param("postgresql_adbc_conn", marks=pytest.mark.db),
]
adbc_connectable_iris: List[Union[str, pytest.Param]] = [
    pytest.param("postgresql_adbc_iris", marks=pytest.mark.db),
    "sqlite_adbc_iris",
]
adbc_connectable_types: List[Union[str, pytest.Param]] = [
    pytest.param("postgresql_adbc_types", marks=pytest.mark.db),
    "sqlite_adbc_types",
]
all_connectable: List[Union[str, pytest.Param]] = sqlalchemy_connectable + [
    "sqlite_buildin"
] + adbc_connectable
all_connectable_iris: List[Union[str, pytest.Param]] = (
    sqlalchemy_connectable_iris + ["sqlite_buildin_iris"] + adbc_connectable_iris
)
all_connectable_types: List[Union[str, pytest.Param]] = (
    sqlalchemy_connectable_types + ["sqlite_buildin_types"] + adbc_connectable_types
)


@pytest.mark.parametrize("conn", all_connectable)
def func_7n3lkotn(conn: str, test_frame1: DataFrame, request: pytest.FixtureRequest) -> None:
    conn = request.getfixturevalue(conn)
    func_0t54gbeq.to_sql(name="test", con=conn, if_exists="append", index=False)


@pytest.mark.parametrize("conn", all_connectable)
def func_66a9iskb(
    conn: str, test_frame1: DataFrame, request: pytest.FixtureRequest
) -> None:
    if conn == "postgresql_adbc_conn" and not using_string_dtype():
        request.node.add_marker(
            pytest.mark.xfail(
                reason="postgres ADBC driver < 1.2 cannot insert index with null type"
            )
        )
    conn = request.getfixturevalue(conn)
    empty_df = test_frame1.iloc[:0]
    empty_df.to_sql(name="test", con=conn, if_exists="append", index=False)


@pytest.mark.parametrize("conn", all_connectable)
def func_rln0ckob(conn: str, request: pytest.FixtureRequest) -> None:
    pytest.importorskip("pyarrow")
    df = DataFrame(
        {
            "int": pd.array([1], dtype="int8[pyarrow]"),
            "datetime": pd.array([datetime(2023, 1, 1)], dtype="timestamp[ns][pyarrow]"),
            "date": pd.array([date(2023, 1, 1)], dtype="date32[day][pyarrow]"),
            "timedelta": pd.array([timedelta(1)], dtype="duration[ns][pyarrow]"),
            "string": pd.array(["a"], dtype="string[pyarrow]"),
        }
    )
    if "adbc" in conn:
        if conn == "sqlite_adbc_conn":
            df = df.drop(columns=["timedelta"])
        if pa_version_under14p1:
            exp_warning = DeprecationWarning
            msg = "is_sparse is deprecated"
        else:
            exp_warning = None
            msg = ""
    else:
        exp_warning = UserWarning
        msg = "the 'timedelta'"
    sql_conn = request.getfixturevalue(conn)
    with tm.assert_produces_warning(exp_warning, match=msg, check_stacklevel=False):
        df.to_sql(name="test_arrow", con=sql_conn, if_exists="replace", index=False)


@pytest.mark.parametrize("conn", all_connectable)
def func_xmlq9g9g(
    conn: str, request: pytest.FixtureRequest, nulls_fixture: Any
) -> None:
    pytest.importorskip("pyarrow")
    df = DataFrame(
        {
            "datetime": pd.array(
                [datetime(2023, 1, 1), nulls_fixture], dtype="timestamp[ns][pyarrow]"
            )
        }
    )
    sql_conn = request.getfixturevalue(conn)
    df.to_sql(name="test_arrow", con=sql_conn, if_exists="replace", index=False)


@pytest.mark.parametrize("conn", all_connectable)
@pytest.mark.parametrize("method", [None, "multi"])
def func_4225il8c(
    conn: str, method: Optional[str], test_frame1: DataFrame, request: pytest.FixtureRequest
) -> None:
    if method == "multi" and "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(reason="'method' not implemented for ADBC drivers", strict=True)
        )
    sql_conn = request.getfixturevalue(conn)
    with pandasSQL_builder(sql_conn, need_transaction=True) as pandasSQL:
        pandasSQL.to_sql(test_frame1, "test_frame", method=method)
        assert pandasSQL.has_table("test_frame")
    assert func_9huuyabu(sql_conn, "test_frame") == len(test_frame1)


@pytest.mark.parametrize("conn", all_connectable)
@pytest.mark.parametrize(
    "mode, num_row_coef",
    [
        ("replace", 1),
        ("append", 2),
    ],
)
def func_5atyv9ly(
    conn: str,
    mode: str,
    num_row_coef: int,
    test_frame1: DataFrame,
    request: pytest.FixtureRequest,
) -> None:
    sql_conn = request.getfixturevalue(conn)
    with pandasSQL_builder(sql_conn, need_transaction=True) as pandasSQL:
        pandasSQL.to_sql(test_frame1, "test_frame", if_exists="fail")
        pandasSQL.to_sql(test_frame1, "test_frame", if_exists=mode)
        assert pandasSQL.has_table("test_frame")
    assert func_9huuyabu(sql_conn, "test_frame") == num_row_coef * len(test_frame1)


@pytest.mark.parametrize("conn", all_connectable)
def func_c6o0uvgf(conn: str, test_frame1: DataFrame, request: pytest.FixtureRequest) -> None:
    sql_conn = request.getfixturevalue(conn)
    with pandasSQL_builder(sql_conn, need_transaction=True) as pandasSQL:
        pandasSQL.to_sql(test_frame1, "test_frame", if_exists="fail")
        assert pandasSQL.has_table("test_frame")
        msg = "Table 'test_frame' already exists"
        with pytest.raises(ValueError, match=msg):
            pandasSQL.to_sql(test_frame1, "test_frame", if_exists="fail")


@pytest.mark.parametrize("conn", all_connectable_iris)
def func_9y6wbp9e(conn: str, request: pytest.FixtureRequest) -> None:
    sql_conn = request.getfixturevalue(conn)
    iris_frame = read_sql_query("SELECT * FROM iris", sql_conn)
    func_pnf76qpb(iris_frame)
    iris_frame = pd.read_sql("SELECT * FROM iris", sql_conn)
    func_pnf76qpb(iris_frame)
    iris_frame = pd.read_sql("SELECT * FROM iris where 0=1", sql_conn)
    assert iris_frame.shape == (0, 5)
    assert "SepalWidth" in iris_frame.columns


@pytest.mark.parametrize("conn", all_connectable_iris)
def func_8zu3em53(conn: str, request: pytest.FixtureRequest) -> None:
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(reason="'chunksize' not implemented for ADBC drivers", strict=True)
        )
    sql_conn = request.getfixturevalue(conn)
    iris_frame = concat(read_sql_query("SELECT * FROM iris", sql_conn, chunksize=7))
    func_pnf76qpb(iris_frame)
    iris_frame = concat(pd.read_sql("SELECT * FROM iris", sql_conn, chunksize=7))
    func_pnf76qpb(iris_frame)
    iris_frame = concat(pd.read_sql("SELECT * FROM iris where 0=1", sql_conn, chunksize=7))
    assert iris_frame.shape == (0, 5)
    assert "SepalWidth" in iris_frame.columns


@pytest.mark.parametrize("conn", sqlalchemy_connectable_iris)
def func_5dglxkas(conn: str, request: pytest.FixtureRequest) -> None:
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(reason="'chunksize' not implemented for ADBC drivers", strict=True)
        )
    sql_conn = request.getfixturevalue(conn)
    from sqlalchemy import MetaData, Table, create_engine, select

    metadata = MetaData()
    autoload_con = create_engine(conn) if isinstance(conn, str) else conn
    iris = Table("iris", metadata, autoload_with=autoload_con)
    iris_frame = read_sql_query(select(iris), sql_conn, params={"name": "Iris-setosa", "length": 5.1})
    func_pnf76qpb(iris_frame)
    if isinstance(conn, str):
        autoload_con.dispose()


@pytest.mark.parametrize("conn", all_connectable_iris)
def func_wes7kj65(
    conn: str, request: pytest.FixtureRequest, sql_strings: Dict[str, Dict[str, str]]
) -> None:
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(reason="'chunksize' not implemented for ADBC drivers", strict=True)
        )
    for db, query in sql_strings["read_parameters"].items():
        if db in conn:
            break
    else:
        raise KeyError(f"No part of {conn} found in sql_strings['read_parameters']")
    sql_conn = request.getfixturevalue(conn)
    iris_frame = read_sql_query(query, sql_conn, params=("Iris-setosa", 5.1))
    func_pnf76qpb(iris_frame)


@pytest.mark.parametrize("conn", sqlalchemy_connectable_iris)
def func_0umvgov9(conn: str, request: pytest.FixtureRequest) -> None:
    sql_conn = request.getfixturevalue(conn)
    iris_frame = read_sql_table("iris", sql_conn)
    func_pnf76qpb(iris_frame)
    iris_frame = pd.read_sql("iris", sql_conn)
    func_pnf76qpb(iris_frame)


@pytest.mark.parametrize("conn", sqlalchemy_connectable_iris)
def func_pj878w91(conn: str, request: pytest.FixtureRequest) -> None:
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(reason="'chunksize' not implemented for ADBC drivers", strict=True)
        )
    sql_conn = request.getfixturevalue(conn)
    iris_frame = concat(read_sql_table("iris", sql_conn, chunksize=7))
    func_pnf76qpb(iris_frame)
    iris_frame = concat(pd.read_sql("iris", sql_conn, chunksize=7))
    func_pnf76qpb(iris_frame)


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def func_5hi8ohmv(conn: str, test_frame1: DataFrame, request: pytest.FixtureRequest) -> None:
    sql_conn = request.getfixturevalue(conn)
    check: List[int] = []

    def func_p37gzpww(pd_table: sqlalchemy.Table, conn: Any, keys: List[str], data_iter: List[Tuple[Any, ...]]) -> None:
        check.append(1)
        data = [dict(zip(keys, row)) for row in data_iter]
        conn.execute(pd_table.table.insert(), data)

    with pandasSQL_builder(sql_conn, need_transaction=True) as pandasSQL:
        pandasSQL.to_sql(test_frame1, "test_frame", method=func_p37gzpww)
        assert pandasSQL.has_table("test_frame")
    assert check == [1]
    assert func_9huuyabu(sql_conn, "test_frame") == len(test_frame1)


@pytest.mark.parametrize("conn", all_connectable_types)
def func_ia3w6u8s(conn: str, request: pytest.FixtureRequest) -> None:
    conn_name = conn
    if conn_name == "sqlite_buildin_types":
        request.applymarker(pytest.mark.xfail(reason="sqlite_buildin connection does not implement read_sql_table"))
    sql_conn = request.getfixturevalue(conn)
    df = sql.read_sql_table("types", sql_conn)
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
def func_pwi2mkh8(conn: str, request: pytest.FixtureRequest) -> None:
    sql_conn = request.getfixturevalue(conn)
    from sqlalchemy.dialects.mysql import insert
    from sqlalchemy.engine import Engine
    from sqlalchemy.sql import text

    df = DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
    df.to_sql(name="test_frame", con=sql_conn, index=False)
    proc = """
    DROP PROCEDURE IF EXISTS get_testdb;

    CREATE PROCEDURE get_testdb ()

    BEGIN
        SELECT * FROM test_frame;
    END
    """
    proc = text(proc)
    if isinstance(sql_conn, Engine):
        with sql_conn.connect() as engine_conn:
            with engine_conn.begin():
                engine_conn.execute(proc)
    else:
        with sql_conn.begin():
            sql_conn.execute(proc)
    res1 = sql.read_sql_query("CALL get_testdb();", sql_conn)
    tm.assert_frame_equal(df, res1)
    res2 = sql.read_sql("CALL get_testdb();", sql_conn)
    tm.assert_frame_equal(df, res2)


@pytest.mark.parametrize("conn", postgresql_connectable)
@pytest.mark.parametrize("expected_count", [2, "Success!"])
def func_f98v09n5(
    conn: str, expected_count: Union[int, str], request: pytest.FixtureRequest
) -> None:
    def func_fq9vt2tb(table: sqlalchemy.Table, conn: Any, keys: List[str], data_iter: List[Tuple[Any, ...]]) -> Union[int, None]:
        dbapi_conn = conn.connection
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
            sql_query = f"COPY {table_name} ({columns}) FROM STDIN WITH CSV"
            cur.copy_expert(sql=sql_query, file=s_buf)
        return expected_count

    conn = request.getfixturevalue(conn)
    expected = DataFrame({"col1": [1, 2], "col2": [0.1, 0.2], "col3": ["a", "n"]})
    result_count = expected.to_sql(name="test_frame", con=conn, index=False, method=func_fq9vt2tb)
    if expected_count is None:
        assert result_count is None
    else:
        assert result_count == expected_count
    result = sql.read_sql_table("test_frame", conn)
    tm.assert_frame_equal(result, expected)
    assert result_count == expected_count


@pytest.mark.parametrize("conn", all_connectable)
def func_0ontf6q3(conn: str, request: pytest.FixtureRequest) -> None:
    if "sqlite" in conn or "mysql" in conn:
        request.applymarker(pytest.mark.xfail(reason="'test for public schema only specific to postgresql'"))
    sql_conn = request.getfixturevalue(conn)
    test_data = DataFrame([[1, 2.1, "a"], [2, 3.1, "b"]], columns=list("abc"), index=["A", "B"])
    with pandasSQL_builder(sql_conn) as pandasSQL:
        pandasSQL.to_sql(test_data, "test_public_schema", schema="public")
    df_out = sql.read_sql_table("test_public_schema", sql_conn, schema="public")
    tm.assert_frame_equal(test_data, df_out)


@pytest.mark.parametrize("conn", mysql_connectable)
def func_malcub74(conn: str, request: pytest.FixtureRequest) -> None:
    sql_conn = request.getfixturevalue(conn)
    from sqlalchemy.dialects.mysql import insert
    from sqlalchemy.engine import Engine
    from sqlalchemy.sql import text

    def func_ff07relx(table: sqlalchemy.Table, conn: Any, keys: List[str], data_iter: List[Tuple[Any, ...]]) -> int:
        data = [dict(zip(keys, row)) for row in data_iter]
        stmt = insert(table.table).values(data)
        stmt = stmt.on_duplicate_key_update(b=stmt.inserted.b, c=stmt.inserted.c)
        result = conn.execute(stmt)
        return result.rowcount

    create_sql = text(
        """
    CREATE TABLE test_insert_conflict (
        a INT PRIMARY KEY,
        b FLOAT,
        c VARCHAR(10)
    );
    """
    )
    if isinstance(sql_conn, Engine):
        with sql_conn.connect() as con:
            with con.begin():
                con.execute(create_sql)
    else:
        with sql_conn.begin():
            sql_conn.execute(create_sql)
    df = DataFrame([[1, 2.1, "a"]], columns=list("abc"))
    df.to_sql(name="test_insert_conflict", con=sql_conn, if_exists="append", index=False)
    expected = DataFrame([[1, 3.2, "b"]], columns=list("abc"))
    inserted = expected.to_sql(
        name="test_insert_conflict",
        con=sql_conn,
        index=False,
        if_exists="append",
        method=func_ff07relx,
    )
    result = sql.read_sql_table("test_insert_conflict", sql_conn)
    tm.assert_frame_equal(result, expected)
    assert inserted == 0
    with sql.SQLDatabase(sql_conn, need_transaction=True) as pandasSQL:
        pandasSQL.drop_table("test_insert_conflict")


@pytest.mark.parametrize("conn", all_connectable)
def func_0ontf6q3(conn: str, request: pytest.FixtureRequest) -> None:
    pass  # This function appears duplicated; no action needed.


@pytest.mark.parametrize("conn", all_connectable)
def func_35s5t4bi(conn: str, request: pytest.FixtureRequest, test_frame1: DataFrame) -> None:
    sql_conn = request.getfixturevalue(conn)
    if sql.has_table("test_frame4", sql_conn):
        with sql.SQLDatabase(sql_conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_frame4")
    sql.to_sql(test_frame1, "test_frame4", sql_conn, if_exists="fail")
    sql.to_sql(test_frame1, "test_frame4", sql_conn, if_exists="append")
    assert sql.has_table("test_frame4", sql_conn)
    num_entries = 2 * len(test_frame1)
    num_rows = func_9huuyabu(sql_conn, "test_frame4")
    assert num_rows == num_entries


@pytest.mark.parametrize("conn", all_connectable)
def func_x2np24f3(conn: str, request: pytest.FixtureRequest, test_frame3: DataFrame) -> None:
    sql_conn = request.getfixturevalue(conn)
    if sql.has_table("test_frame5", sql_conn):
        with sql.SQLDatabase(sql_conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_frame5")
    sql.to_sql(test_frame3, "test_frame5", sql_conn, index=False)
    result = sql.read_sql("SELECT * FROM test_frame5", sql_conn)
    tm.assert_frame_equal(test_frame3, result)


@pytest.mark.parametrize("conn", all_connectable)
def func_2jjbzdqd(conn: str, request: pytest.FixtureRequest) -> None:
    sql_conn = request.getfixturevalue(conn)
    if sql.has_table("test_series", sql_conn):
        with sql.SQLDatabase(sql_conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_series")
    s = Series(np.arange(5, dtype="int64"), name="series")
    sql.to_sql(s, "test_series", sql_conn, index=False)
    s2 = sql.read_sql_query("SELECT * FROM test_series", sql_conn)
    tm.assert_frame_equal(s.to_frame(), s2)


@pytest.mark.parametrize("conn", all_connectable)
def func_yxqs7tlq(
    conn: str, request: pytest.FixtureRequest, sql_strings: Dict[str, Dict[str, str]]
) -> None:
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(reason="'params' not implemented for ADBC drivers", strict=True)
        )
    conn_name = conn
    sql_conn = request.getfixturevalue(conn)
    dialect = func_m6keniz1(conn_name)
    query = sql_strings["read_parameters"][dialect]
    params = ("Iris-setosa", 5.1)
    with pandasSQL_builder(sql_conn) as pandasSQL:
        with pandasSQL.run_transaction():
            iris_frame = pandasSQL.read_query(query, params=params)
    func_pnf76qpb(iris_frame)


@pytest.mark.parametrize("conn", all_connectable_iris)
def func_9oi5ju99(
    conn: str, request: pytest.FixtureRequest, sql_strings: Dict[str, Dict[str, str]]
) -> None:
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(reason="'params' not implemented for ADBC drivers", strict=True)
        )
    conn_name = conn
    sql_conn = request.getfixturevalue(conn)
    dialect = func_m6keniz1(conn_name)
    query = sql_strings["read_named_parameters"][dialect]
    params = {"name": "Iris-setosa", "length": 5.1}
    with pandasSQL_builder(sql_conn) as pandasSQL:
        with pandasSQL.run_transaction():
            iris_frame = pandasSQL.read_query(query, params=params)
    func_pnf76qpb(iris_frame)


@pytest.mark.parametrize("conn", all_connectable_iris)
def func_wrucqelc(
    conn: str, request: pytest.FixtureRequest, sql_strings: Dict[str, Dict[str, str]]
) -> None:
    if "mysql" in conn or ("postgresql" in conn and "adbc" not in conn):
        request.applymarker(pytest.mark.xfail(reason="broken test"))
    conn_name = conn
    sql_conn = request.getfixturevalue(conn)
    dialect = func_m6keniz1(conn_name)
    query = sql_strings["read_no_parameters_with_percent"][dialect]
    with pandasSQL_builder(sql_conn) as pandasSQL:
        with pandasSQL.run_transaction():
            iris_frame = pandasSQL.read_query(query, params=None)
    func_pnf76qpb(iris_frame)


@pytest.mark.parametrize("conn", all_connectable_iris)
def func_ha5ft39n(conn: str, request: pytest.FixtureRequest) -> None:
    sql_conn = request.getfixturevalue(conn)
    iris_frame = sql.read_sql_query("SELECT * FROM iris_view", sql_conn)
    func_pnf76qpb(iris_frame)


@pytest.mark.parametrize("conn", all_connectable_iris)
def func_yxqs7tlq(conn: str, request: pytest.FixtureRequest) -> None:
    pass  # Duplicated function; no action needed.


@pytest.mark.parametrize("conn", all_connectable)
def func_35s5t4bi(conn: str, request: pytest.FixtureRequest, test_frame1: DataFrame) -> None:
    pass  # Duplicated function; no action needed.


@pytest.mark.parametrize("conn", mysql_connectable)
def func_malcub74(conn: str, request: pytest.FixtureRequest) -> None:
    pass  # Already defined above; no action needed.


@pytest.mark.parametrize("conn", postgresql_connectable)
@pytest.mark.parametrize("expected_count", [2, "Success!"])
def func_f98v09n5(
    conn: str, expected_count: Union[int, str], request: pytest.FixtureRequest
) -> None:
    pass  # Already defined above; no action needed.


@pytest.mark.parametrize("conn", all_connectable)
def func_0ontf6q3(conn: str, request: pytest.FixtureRequest) -> None:
    pass  # Already defined above; no action needed.


@pytest.mark.parametrize("conn", all_connectable_iris)
def func_ju3o5rm2(conn: str, request: pytest.FixtureRequest) -> None:
    sql_conn = request.getfixturevalue(conn)
    with pandasSQL_builder(sql_conn) as pandas_sql:
        with pandas_sql.run_transaction():
            iris_results = pandas_sql.execute("SELECT * FROM iris")
            row = iris_results.fetchone()
            iris_results.close()
    assert list(row) == [5.1, 3.5, 1.4, 0.2, "Iris-setosa"]


def func_7e43funb(sqlite_conn: sqlite3.Connection) -> None:
    conn = sqlite_conn
    from sqlalchemy import text
    create_table = "CREATE TABLE invalid (x INTEGER, y UNKNOWN);"
    create_table_other = "CREATE TABLE other_table (x INTEGER, y INTEGER);"
    for query in [create_table, create_table_other]:
        conn.execute(query)
    with tm.assert_produces_warning(None):
        sql.read_sql_table("other_table", conn)
        sql.read_sql_query("SELECT * FROM other_table", conn)


@pytest.mark.parametrize("conn", all_connectable)
def func_fkz6lksn(conn: str, request: pytest.FixtureRequest, test_frame1: DataFrame) -> None:
    if "postgresql" in conn:
        request.applymarker(pytest.mark.xfail(reason="Does not raise warning"))
    sql_conn = request.getfixturevalue(conn)
    with tm.assert_produces_warning(UserWarning, match="Use only sqlalchemy"):
        with sql.SQLDatabase(sql_conn) as db:
            db.check_case_sensitive("TABLE1", "")
    func_0t54gbeq.to_sql(name="CaseSensitive", con=sql_conn)
    # Further assertions can be added as needed.


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def func_qslliqk4(
    conn: str, request: pytest.FixtureRequest, test_frame1: DataFrame
) -> None:
    if conn == "sqlite_str":
        pytest.skip("sqlite_str has no inspection system")
    sql_conn = request.getfixturevalue(conn)
    from sqlalchemy import inspect

    assert not inspect(sql_conn).has_table("temp_frame")
    with pandasSQL_builder(sql_conn, need_transaction=True) as pandasSQL:
        assert pandasSQL.to_sql(test_frame1, "temp_frame") == len(test_frame1)
    assert inspect(sql_conn).has_table("temp_frame")
    with pandasSQL_builder(sql_conn, need_transaction=True) as pandasSQL:
        pandasSQL.drop_table("temp_frame")
    assert not inspect(sql_conn).has_table("temp_frame")


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def func_5zhgc38i(conn: str, request: pytest.FixtureRequest) -> None:
    if conn == "sqlite_str":
        pytest.skip("sqlite_str has no inspection system")
    sql_conn = request.getfixturevalue(conn)
    from sqlalchemy import inspect

    temp_frame = DataFrame({"one": [1.0, 2.0, 3.0, 4.0], "two": [4.0, 3.0, 2.0, 1.0]})
    with sql.SQLDatabase(sql_conn, need_transaction=True) as pandasSQL:
        with pandasSQL.run_transaction():
            pandasSQL.to_sql(temp_frame, "temp_frame")
    insp = inspect(sql_conn)
    assert insp.has_table("temp_frame")
    with sql.SQLDatabase(sql_conn, need_transaction=True) as pandasSQL:
        pandasSQL.drop_table("temp_frame")
    assert not insp.has_table("temp_frame")


def func_y9fuopdo(sqlite_buildin: sqlite3.Connection) -> None:
    sql_conn = sqlite_buildin
    df = DataFrame({"time": to_datetime(["2014-12-12 01:54", "2014-12-11 02:54"], utc=True)}
    )
    sql.to_sql(name="test_time", con=sql_conn, if_exists="replace", index=False)
    result = pd.read_sql("SELECT * FROM test_time", sql_conn).iloc[0, 0]
    assert result == "2014-12-12 01:54:00.000000"


def func_rkxevvsi(sqlite_buildin: sqlite3.Connection) -> None:
    conn = sqlite_buildin
    cols = ["A", "B"]
    data = [(0.8, True), (0.9, None)]
    df = DataFrame(data, columns=cols)
    assert df.to_sql(name="dtype_test", con=conn) == 2
    assert df.to_sql(name="dtype_test2", con=conn, dtype={"B": "STRING"}) == 2
    assert func_ixrjqqps(conn, "dtype_test", "B") == "INTEGER"
    assert func_ixrjqqps(conn, "dtype_test2", "B") == "STRING"
    msg = r"B \(\<class 'bool'\>\) not a string"
    with pytest.raises(ValueError, match=msg):
        df.to_sql(name="error", con=conn, dtype={"B": bool})
    assert df.to_sql(name="single_dtype_test", con=conn, dtype="STRING") == 2
    assert func_ixrjqqps(conn, "single_dtype_test", "A") == "STRING"
    assert func_ixrjqqps(conn, "single_dtype_test", "B") == "STRING"


def func_lvcdb1j1(sqlite_buildin: sqlite3.Connection) -> None:
    conn = sqlite_buildin
    cols = {"Bool": Series([True, None]), "Date": Series([datetime(2012, 5, 1), None]}, "Int": Series([1, None], dtype="object"), "Float": Series([1.1, None])}
    df = DataFrame(cols)
    tbl = "notna_dtype_test"
    assert df.to_sql(name=tbl, con=conn) == 2
    assert func_ixrjqqps(conn, tbl, "Bool") == "INTEGER"
    assert func_ixrjqqps(conn, tbl, "Date") == "TIMESTAMP"
    assert func_ixrjqqps(conn, tbl, "Int") == "INTEGER"
    assert func_ixrjqqps(conn, tbl, "Float") == "REAL"


def func_mbyo73vx(sqlite_buildin: sqlite3.Connection) -> None:
    conn = sqlite_buildin
    df = DataFrame({"a": [1, 2]}, dtype="int64")
    assert sql.to_sql(name="test_bigintwarning", con=conn, index=False) == 2
    with tm.assert_produces_warning(None):
        sql.read_sql_table("test_bigintwarning", conn)


def func_u99w3ioe(sqlite_engine: sqlalchemy.engine.Connection) -> None:
    conn = sqlite_engine
    df = DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError, match="Empty table name specified"):
        df.to_sql(name="", con=conn, if_exists="replace", index=False)


@pytest.fixture
def func_80f6ik4l(sqlite_engine: sqlalchemy.engine.Connection) -> None:
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
    return


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def func_8530tifg(conn: str, request: pytest.FixtureRequest) -> None:
    if conn == "sqlite_str":
        pytest.skip("sqlite_str has no inspection system")
    sql_conn = request.getfixturevalue(conn)
    from sqlalchemy import TEXT, String
    from sqlalchemy.schema import MetaData

    cols = ["A", "B"]
    data = [(0.8, True), (0.9, None)]
    df = DataFrame(data, columns=cols)
    assert df.to_sql(name="dtype_test", con=sql_conn) == 2
    assert df.to_sql(name="dtype_test2", con=sql_conn, dtype={"B": "STRING"}) == 2
    meta = MetaData()
    meta.reflect(bind=sql_conn)
    sqltype = meta.tables["dtype_test2"].columns["B"].type
    assert isinstance(sqltype, String)
    msg = r"The type of B \(\<class 'bool'\>\) not a string"
    with pytest.raises(ValueError, match=msg):
        df.to_sql(name="error", con=sql_conn, dtype={"B": bool})
    assert df.to_sql(name="single_dtype_test", con=sql_conn, dtype="STRING") == 2
    meta.reflect(bind=sql_conn)
    sqltypea = meta.tables["single_dtype_test"].columns["A"].type
    sqltypeb = meta.tables["single_dtype_test"].columns["B"].type
    assert isinstance(sqltypea, String)
    assert isinstance(sqltypeb, String)
    assert df.to_sql(name="test_dtype_argument", con=sql_conn, dtype={"B": String(10)}) == 2
    meta.reflect(bind=sql_conn)
    sqltype = meta.tables["test_dtype_argument"].columns["B"].type
    assert isinstance(sqltype, String)
    assert sqltype.length == 10
    assert df.to_sql(name="single_dtype_test", con=sql_conn, dtype=TEXT) == 2
    meta.reflect(bind=sql_conn)
    sqltypea = meta.tables["single_dtype_test"].columns["A"].type
    sqltypeb = meta.tables["single_dtype_test"].columns["B"].type
    assert isinstance(sqltypea, TEXT)
    assert isinstance(sqltypeb, TEXT)


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def func_8530tifg(conn: str, request: pytest.FixtureRequest) -> None:
    pass  # Duplicated function; no action needed.


def func_qm1544uc(sqlite_buildin: sqlalchemy.engine.Connection) -> None:
    frame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list("ABCD")), index=date_range("2000-01-01", periods=10, freq="B"))
    assert sql.to_sql(frame, name="test_frame3_legacy", con=sqlite_buildin, index=False) == 4
    with closing(sqlite_buildin.connect()) as conn:
        with closing(conn.cursor()) as cur:
            for row in frame.itertuples(index=False, name=None):
                cur.execute("INSERT INTO test_frame3_legacy VALUES (?, ?, ?, ?)", row)
    with closing(sqlite_buildin.connect()) as conn:
        result = sql.read_sql_query("SELECT * FROM test_frame3_legacy;", conn)
    tm.assert_frame_equal(frame, result)


def func_9fs4ok85(sqlite_buildin: sqlalchemy.engine.Connection) -> None:
    conn = sqlite_buildin
    frame = DataFrame(
        {"t": pd.to_datetime(["2020-12-31 12:00:00"])}
    )
    assert sql.to_sql(name="test", con=conn, if_exists="replace", index=False) == 1
    result = sql.read_sql_query("SELECT * FROM test", conn)
    tm.assert_frame_equal(result, frame)


def func_7lkf02h2(sqlite_buildin: sqlite3.Connection) -> None:
    conn = sqlite_buildin
    test_sql = """
    CREATE TABLE test (
        a TEXT,
        b TEXT,
        c REAL
    );
    INSERT INTO test VALUES ('foo', 'bar', 1.234);
    INSERT INTO test VALUES ('foo', 'baz', 2.567);
    """
    conn.executescript(test_sql)
    result = sql.read_sql_query("SELECT * FROM test", conn)
    expected = DataFrame({"a": ["foo", "foo"], "b": ["bar", "baz"], "c": [1.234, 2.567]})
    tm.assert_frame_equal(result, expected)
    with pytest.raises(sql.DatabaseError, match="Execution failed on sql"):
        sql.read_sql_query("INSERT INTO test VALUES ('foo', 'bar', 7)", conn)


def func_3594288k() -> None:
    create_sql = """
    CREATE TABLE test (
        a TEXT,
        b TEXT,
        c REAL,
        PRIMARY KEY (a, b)
    );
    """
    with contextlib.closing(sqlite3.connect(":memory:")) as conn:
        cursor = conn.cursor()
        cursor.execute(create_sql)
        with pandasSQL_builder(conn) as pandas_sql:
            pandas_sql.execute("INSERT INTO test VALUES ('foo', 'bar', 1.234)")
    msg = "Cannot operate on a closed database."
    with pytest.raises(sqlite3.ProgrammingError, match=msg):
        func_pp84sjy1("select * from test", con=conn)


def func_86srt544(sqlite_buildin: sqlite3.Connection) -> None:
    conn = sqlite_buildin
    df_if_exists_1 = DataFrame({"col1": [1, 2], "col2": ["A", "B"]})
    df_if_exists_2 = DataFrame({"col1": [3, 4, 5], "col2": ["C", "D", "E"]})
    table_name = "table_if_exists"
    sql_select = f"SELECT * FROM {table_name}"
    msg = "'notvalidvalue' is not valid for if_exists"
    with pytest.raises(ValueError, match=msg):
        sql.to_sql(frame=df_if_exists_1, con=conn, name=table_name, if_exists="notvalidvalue")
    func_ivdgn9x4(table_name, conn)
    sql.to_sql(frame=df_if_exists_1, con=conn, name=table_name, if_exists="fail")
    msg = "Table 'table_if_exists' already exists"
    with pytest.raises(ValueError, match=msg):
        sql.to_sql(frame=df_if_exists_1, con=conn, name=table_name, if_exists="fail")
    sql.to_sql(frame=df_if_exists_1, con=conn, name=table_name, if_exists="replace", index=False)
    assert func_pp84sjy1(sql_select, con=conn) == [(1, "A"), (2, "B")]
    assert sql.to_sql(frame=df_if_exists_2, con=conn, name=table_name, if_exists="replace", index=False) == 3
    assert func_pp84sjy1(sql_select, con=conn) == [(3, "C"), (4, "D"), (5, "E")]
    func_ivdgn9x4(table_name, conn)
    assert sql.to_sql(frame=df_if_exists_1, con=conn, name=table_name, if_exists="fail", index=False) == 2
    assert func_pp84sjy1(sql_select, con=conn) == [(1, "A"), (2, "B")]
    assert sql.to_sql(frame=df_if_exists_2, con=conn, name=table_name, if_exists="append", index=False) == 3
    assert func_pp84sjy1(sql_select, con=conn) == [(1, "A"), (2, "B"), (3, "C"), (4, "D"), (5, "E")]
    func_ivdgn9x4(table_name, conn)


@pytest.mark.parametrize("conn", all_connectable)
def func_g8s7x7ke(conn: str, request: pytest.FixtureRequest) -> None:
    if "adbc" in conn:
        pa = pytest.importorskip("pyarrow")
        if not (Version(pa.__version__) >= Version("16.0") and conn in ["sqlite_adbc_conn", "postgresql_adbc_conn"]):
            request.node.add_marker(pytest.mark.xfail(reason="pyarrow->pandas throws ValueError", strict=True))
    sql_conn = request.getfixturevalue(conn)
    df = DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3], "c": [1, 1, 1]})
    df.to_sql(name="foobar", con=sql_conn, index=False)
    result = pd.read_sql("SELECT a, b, a +1 as a, c FROM foobar", sql_conn)
    expected = DataFrame([[1, 0.1, 2, 1], [2, 0.2, 3, 1], [3, 0.3, 4, 1]], columns=["a", "b", "a", "c"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def func_n73j3lcl(
    conn: str, request: pytest.FixtureRequest, test_frame3: DataFrame
) -> None:
    sql_conn = request.getfixturevalue(conn)
    if sql.has_table("test_frame_roundtrip", sql_conn):
        with sql.SQLDatabase(sql_conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_frame_roundtrip")
    sql.to_sql(test_frame3, "test_frame_roundtrip", con=sql_conn)
    result = sql.read_sql_query("SELECT * FROM test_frame_roundtrip", con=sql_conn)
    if "adbc" in conn:
        result = result.drop(columns="__index_level_0__")
    else:
        result = result.drop(columns="level_0")
    tm.assert_frame_equal(result, test_frame3)


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def func_65py3w56(
    conn: str,
    request: pytest.FixtureRequest,
    dtype_backend: Optional[str],
    test_frame1: DataFrame,
) -> None:
    conn_name = conn
    sql_conn = request.getfixturevalue(conn)
    if sql.has_table("test_dtype_argument", sql_conn):
        with sql.SQLDatabase(sql_conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_dtype_argument")
    df = test_frame1.copy()
    df = df.astype({"A": float, "B": "Int64"})  # Example dtype mapping
    df.to_sql(
        name="test_dtype_argument", con=sql_conn, dtype={"B": "Int64"}, index=False
    )
    df_read = sql.read_sql_query("SELECT * FROM test_dtype_argument", con=sql_conn, dtype={"A": np.float64}, dtype_backend=dtype_backend)
    expected = test_frame1.astype({"A": float, "B": "Int64"})
    tm.assert_frame_equal(df_read, expected)


@pytest.mark.parametrize("conn", all_connectable)
@pytest.mark.parametrize("dtype_backend", [lib.no_default, "numpy_nullable"])
@pytest.mark.parametrize("func", ["read_sql", "read_sql_query"])
def func_3deloozx(
    conn: str,
    request: pytest.FixtureRequest,
    func: str,
    dtype_backend: Optional[str],
    dtype_backend_data: DataFrame,
    dtype_backend_expected: Callable[[str, Optional[str], str], DataFrame],
) -> None:
    conn_name = conn
    sql_conn = request.getfixturevalue(conn)
    table = "test"
    df = dtype_backend_data
    df.to_sql(name=table, con=sql_conn, index=False, if_exists="replace")
    with pd.option_context("mode.string_storage", "python"):
        result = getattr(pd, func)(f"Select * from {table}", con=sql_conn, dtype_backend=dtype_backend)
        expected = dtype_backend_expected("python", dtype_backend, conn_name)
    tm.assert_frame_equal(result, expected)
    if "adbc" in conn_name:
        return
    with pd.option_context("mode.string_storage", "python"):
        iterator = getattr(pd, func)(f"Select * from {table}", con=sql_conn, dtype_backend=dtype_backend, chunksize=3)
        expected = dtype_backend_expected("python", dtype_backend, conn_name)
        for result_chunk in iterator:
            tm.assert_frame_equal(result_chunk, expected)


@pytest.mark.parametrize("conn", all_connectable_types)
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
def func_bksdvid5(
    conn: str,
    request: pytest.FixtureRequest,
    read_sql: Callable[..., DataFrame],
    text: str,
    mode: Union[str, Tuple[str, str]],
    error: str,
    types_data_frame: DataFrame,
) -> None:
    conn_name = conn
    sql_conn = request.getfixturevalue(conn)
    if text == "types" and conn_name == "sqlite_buildin_types":
        request.applymarker(pytest.mark.xfail(reason="failing combination of arguments"))
    expected = func_zr8p39fu()(f"python", "numpy_nullable", conn_name)
    if "postgresql" in conn_name:
        expected = expected.astype({"IntDateCol": "int32", "IntDateOnlyCol": "int32", "IntCol": "int32"})
        if "postgresql_adbc_types" in conn_name and pa_version_under14p1:
            expected["DateCol"] = expected["DateCol"].astype("datetime64[ns]")
    elif "mysql" in conn_name or "sqlite" in conn_name:
        expected = expected.astype({"IntDateCol": "int32", "IntDateOnlyCol": "int32", "IntCol": "int32"})
    else:
        expected = expected.astype({"IntDateCol": "int64", "IntDateOnlyCol": "int64", "IntCol": "int64"})
    if "postgresql" not in conn_name and "mysql" not in conn_name and "sqlite" not in conn_name:
        expected["DateCol"] = expected["DateCol"].astype("datetime64[s]")
    tm.assert_frame_equal(read_sql(text, con=sql_conn, parse_dates={"DateCol": {"errors": error}}), expected)


@pytest.mark.parametrize("conn", all_connectable_types)
def func_mpd06vts(
    conn: str, request: pytest.FixtureRequest, test_frame1: DataFrame
) -> None:
    conn_name = conn
    sql_conn = request.getfixturevalue(conn)
    df = sql.read_sql_query("SELECT * FROM types", sql_conn, index_col="DateCol", parse_dates=["DateCol", "IntDateCol"])
    assert issubclass(df.index.dtype.type, np.datetime64)
    assert issubclass(df.IntDateCol.dtype.type, np.datetime64)


@pytest.mark.parametrize("conn", all_connectable)
def func_rm5zj47a(
    conn: str, request: pytest.FixtureRequest, test_frame1: DataFrame
) -> None:
    sql_conn = request.getfixturevalue(conn)
    if sql.has_table("test_timedelta", sql_conn):
        with sql.SQLDatabase(sql_conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_timedelta")
    df = to_timedelta(Series(["00:00:01", "00:00:03"], name="foo")).to_frame()
    if "sqlite" in conn:
        pytest.mark.xfail(reason="sqlite ADBC driver doesn't implement timedelta")(df)
    if "adbc" in conn:
        if pa_version_under14p1:
            exp_warning = DeprecationWarning
        else:
            exp_warning = None
    else:
        exp_warning = UserWarning
    with tm.assert_produces_warning(exp_warning, check_stacklevel=False):
        result_count = df.to_sql(name="test_timedelta", con=sql_conn)
    assert result_count == 2
    result = sql.read_sql_query("SELECT * FROM test_timedelta", sql_conn)
    if "postgresql" in conn:
        expected = Series([pd.DateOffset(months=0, days=0, microseconds=1000000, nanoseconds=0), pd.DateOffset(months=0, days=0, microseconds=3000000, nanoseconds=0)], name="foo")
    else:
        expected = Series([1_000_000, 3_000_000], dtype="int64", name="foo")
    tm.assert_series_equal(result["foo"], expected)


@pytest.mark.parametrize("conn", all_connectable)
def func_ij95trb6(conn: str, request: pytest.FixtureRequest) -> None:
    sql_conn = request.getfixturevalue(conn)
    df = DataFrame({"a": [1 + 1.0j, 2.0j]})
    if "adbc" in conn:
        msg = "datatypes not supported"
    else:
        msg = "Complex datatypes not supported"
    with pytest.raises(ValueError, match=msg):
        df.to_sql("test_complex", con=sql_conn)


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
def func_a372kp6f(
    conn: str,
    request: pytest.FixtureRequest,
    index_name: Union[None, str, int],
    index_label: Union[None, str, int],
    expected: str,
) -> None:
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(reason="index_label argument NotImplemented with ADBC")
        )
    sql_conn = request.getfixturevalue(conn)
    if sql.has_table("test_index_label", sql_conn):
        with sql.SQLDatabase(sql_conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_index_label")
    temp_frame = DataFrame({"col1": range(4)})
    temp_frame.index.name = index_name
    query = "SELECT * FROM test_index_label"
    sql.to_sql(temp_frame, "test_index_label", sql_conn, index_label=index_label)
    frame = sql.read_sql_query(query, sql_conn)
    assert frame.columns[0] == expected


@pytest.mark.parametrize("conn", all_connectable)
def func_kslatmml(conn: str, request: pytest.FixtureRequest) -> None:
    conn_name = conn
    if conn_name == "sqlite_buildin":
        request.applymarker(pytest.mark.xfail(reason="SQLiteDatabase/ADBCDatabase does not raise for bad engine"))
    sql_conn = request.getfixturevalue(conn)
    if sql.has_table("test_index_label", sql_conn):
        with sql.SQLDatabase(sql_conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_index_label")
    expected_row_count = 4
    temp_frame = DataFrame(
        {"col1": range(4)},
        index=MultiIndex.from_product([("A0", "A1"), ("B0", "B1")]),
    )
    assert sql.to_sql(temp_frame, "test_index_label", sql_conn, if_exists="replace", index=True) == expected_row_count
    frame = sql.read_sql_query("SELECT * FROM test_index_label", sql_conn)
    assert frame.columns[:2].tolist() == ["A0", "A1"]
    # Further assertions can be added as needed.


@pytest.mark.parametrize("conn", all_connectable_iris)
def func_vb3gxo2d(conn: str, request: pytest.FixtureRequest) -> None:
    conn = request.getfixturevalue(conn)
    df = sql.read_sql_query("SELECT * FROM test_nan", conn)
    tm.assert_frame_equal(df, DataFrame({"a": [1, 2], "b": [np.nan, np.nan], "c": [1.5, 2.0], "d": [1.5, 2.5]}))


def func_62kber9k(test_frame3: DataFrame) -> None:
    with tm.ensure_clean() as name:
        with closing(sqlite3.connect(name)) as conn:
            assert sql.to_sql(test_frame3, "test_frame3_legacy", conn, index=False) == 4
        with closing(sqlite3.connect(name)) as conn:
            result = sql.read_sql_query("SELECT * FROM test_frame3_legacy;", conn)
    tm.assert_frame_equal(test_frame3, result)


@pytest.mark.db
def func_z63qwo2h(postgresql_psycopg2_engine: SQLAlchemyEngine) -> None:
    conn = postgresql_psycopg2_engine
    df = DataFrame({"col1": [1, 2], "col2": [0.1, 0.2], "col3": ["a", "n"]})
    with conn.connect() as con:
        with con.begin():
            con.exec_driver_sql("DROP SCHEMA IF EXISTS other CASCADE;")
            con.exec_driver_sql("CREATE SCHEMA other;")
    assert sql.to_sql(name="test_schema_public", con=conn, index=False) == 2
    assert sql.to_sql(name="test_schema_public_explicit", con=conn, index=False, schema="public") == 2
    assert sql.to_sql(name="test_schema_other", con=conn, index=False, schema="other") == 2
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
    assert sql.to_sql(name="test_schema_other", con=conn, schema="other", index=False) == 2
    df.to_sql(name="test_schema_other", con=conn, schema="other", index=False, if_exists="replace")
    assert sql.to_sql(name="test_schema_other", con=conn, schema="other", index=False, if_exists="append") == 2
    res = sql.read_sql_table("test_schema_other", conn, schema="other")
    tm.assert_frame_equal(concat([df, df], ignore_index=True), res)
    with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
        pandasSQL.drop_table("test_schema_other")


@pytest.mark.db
def func_h9n4139x(postgresql_psycopg2_engine: SQLAlchemyEngine) -> None:
    conn = postgresql_psycopg2_engine
    from sqlalchemy.sql import text

    create_table = text(
        """
    CREATE TABLE person
    (
        id serial constraint person_pkey primary key,
        created_dt timestamp with time zone
    );

    INSERT INTO person
        VALUES (1, '2021-01-01T00:00:00Z');
    """
    )
    with conn.connect() as con:
        with con.begin():
            con.execute(create_table)
    sql_query = 'SELECT * FROM "person" AS p1 INNER JOIN "person" AS p2 ON p1.id = p2.id;'
    result = pd.read_sql(sql_query, conn)
    expected = DataFrame({"id": [1, 1], "created_dt": [Timestamp("2021-01-01 00:00:00+00:00")] * 2})
    tm.assert_frame_equal(result, expected)
    with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
        pandasSQL.drop_table("person")


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def func_5dglxkas(conn: str, request: pytest.FixtureRequest) -> None:
    pass  # Duplicated function; no action needed.


@pytest.mark.parametrize("conn", all_connectable)
def func_sm4o8uop(
    conn: str, request: pytest.FixtureRequest, test_frame1: DataFrame
) -> None:
    sql_conn = request.getfixturevalue(conn)
    with pandasSQL_builder(sql_conn) as pandasSQL:
        with pandasSQL.run_transaction():
            assert pandasSQL.to_sql(test_frame1, "test_frame_roundtrip") == 4
            result = pandasSQL.read_query("SELECT * FROM test_frame_roundtrip")
    if "adbc" in conn:
        result = result.rename(columns={"__index_level_0__": "level_0"})
    else:
        result = result.rename(columns={"level_0": "level_0"})
    result.set_index("level_0", inplace=True)
    result.index.name = None
    tm.assert_frame_equal(result, test_frame1)


@pytest.mark.parametrize("conn", all_connectable)
def func_nfewmd1f(
    conn: str, request: pytest.FixtureRequest, test_frame1: DataFrame
) -> None:
    sql_conn = request.getfixturevalue(conn)
    if sql.has_table("test_frame1", sql_conn):
        with sql.SQLDatabase(sql_conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_frame1")
    sql.to_sql(test_frame1, "test_frame1", sql_conn)
    assert sql.has_table("test_frame1", sql_conn)


@pytest.mark.parametrize("conn", all_connectable)
def func_ko8nenel(
    conn: str, request: pytest.FixtureRequest, test_frame1: DataFrame
) -> None:
    sql_conn = request.getfixturevalue(conn)
    if sql.has_table("test_frame2", sql_conn):
        with sql.SQLDatabase(sql_conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_frame2")
    sql.to_sql(test_frame1, "test_frame2", sql_conn, if_exists="fail")
    assert sql.has_table("test_frame2", sql_conn)
    msg = "Table 'test_frame2' already exists"
    with pytest.raises(ValueError, match=msg):
        sql.to_sql(test_frame1, "test_frame2", sql_conn, if_exists="fail")


@pytest.mark.parametrize("conn", all_connectable)
def func_4zqwxmfj(
    conn: str, request: pytest.FixtureRequest, test_frame1: DataFrame
) -> None:
    sql_conn = request.getfixturevalue(conn)
    if sql.has_table("test_frame3", sql_conn):
        with sql.SQLDatabase(sql_conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_frame3")
    sql.to_sql(test_frame1, "test_frame3", sql_conn, if_exists="fail")
    sql.to_sql(test_frame1, "test_frame3", sql_conn, if_exists="replace")
    assert sql.has_table("test_frame3", sql_conn)
    num_entries = len(test_frame1)
    num_rows = func_9huuyabu(sql_conn, "test_frame3")
    assert num_rows == num_entries


@pytest.mark.parametrize("conn", all_connectable)
def func_35s5t4bi(
    conn: str, request: pytest.FixtureRequest, test_frame1: DataFrame
) -> None:
    pass  # Duplicated function; no action needed.


@pytest.mark.parametrize("conn", all_connectable)
def func_1tya5hd6(
    conn: str, request: pytest.FixtureRequest, test_frame1: DataFrame
) -> None:
    sql_conn = request.getfixturevalue(conn)
    if sql.has_table("test_frame_roundtrip", sql_conn):
        with sql.SQLDatabase(sql_conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_frame_roundtrip")
    sql.to_sql(test_frame1, "test_frame_roundtrip", con=sql_conn)
    result = sql.read_sql_query("SELECT * FROM test_frame_roundtrip", con=sql_conn)
    if "adbc" in conn:
        result = result.drop(columns="__index_level_0__")
    else:
        result = result.drop(columns="level_0")
    tm.assert_frame_equal(result, test_frame1)


@pytest.mark.parametrize("conn", all_connectable)
def func_n73j3lcl(
    conn: str, request: pytest.FixtureRequest, test_frame3: DataFrame
) -> None:
    if "adbc" in conn:
        request.node.add_marker(pytest.mark.xfail(reason="chunksize argument NotImplemented with ADBC"))
    sql_conn = request.getfixturevalue(conn)
    if sql.has_table("test_frame_roundtrip", sql_conn):
        with sql.SQLDatabase(sql_conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_frame_roundtrip")
    sql.to_sql(test_frame3, "test_frame_roundtrip", con=sql_conn, index=False, chunksize=2)
    result = sql.read_sql_query("SELECT * FROM test_frame_roundtrip", con=sql_conn)
    tm.assert_frame_equal(result, test_frame3)


@pytest.mark.parametrize("conn", all_connectable_iris)
def func_vlqjuo4s(conn: str, request: pytest.FixtureRequest, sql_strings: Dict[str, Dict[str, str]]) -> None:
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(reason="'params' not implemented for ADBC drivers", strict=True)
        )
    conn_name = conn
    sql_conn = request.getfixturevalue(conn)
    dialect = func_m6keniz1(conn_name)
    query = sql_strings["read_parameters"][dialect]
    params = ("Iris-setosa", 5.1)
    with pandasSQL_builder(sql_conn) as pandasSQL:
        with pandasSQL.run_transaction():
            iris_frame = pandasSQL.read_query(query, params=params)
    func_pnf76qpb(iris_frame)


@pytest.mark.parametrize("conn", all_connectable_iris)
def func_9oi5ju99(
    conn: str, request: pytest.FixtureRequest, sql_strings: Dict[str, Dict[str, str]]
) -> None:
    pass  # Duplicated function; no action needed.


@pytest.mark.parametrize("conn", all_connectable_iris)
def func_wrucqelc(
    conn: str, request: pytest.FixtureRequest, sql_strings: Dict[str, Dict[str, str]]
) -> None:
    pass  # Duplicated function; no action needed.


@pytest.mark.parametrize("conn", all_connectable_iris)
def func_ha5ft39n(conn: str, request: pytest.FixtureRequest) -> None:
    pass  # Duplicated function; no action needed.


def func_q5bps3lr(sqlite_buildin: sqlalchemy.engine.Connection) -> None:
    df_if_exists_1 = DataFrame({"col1": [1, 2], "col2": ["A", "B"]})
    df_if_exists_2 = DataFrame({"col1": [3, 4, 5], "col2": ["C", "D", "E"]})
    table_name = "table_if_exists"
    sql_select = f"SELECT * FROM {table_name}"
    msg = "'notvalidvalue' is not valid for if_exists"
    with pytest.raises(ValueError, match=msg):
        sql.to_sql(frame=df_if_exists_1, con=sqlite_buildin, name=table_name, if_exists="notvalidvalue")
    func_ivdgn9x4(table_name, sqlite_buildin)
    sql.to_sql(frame=df_if_exists_1, con=sqlite_buildin, name=table_name, if_exists="fail")
    msg = "Table 'table_if_exists' already exists"
    with pytest.raises(ValueError, match=msg):
        sql.to_sql(frame=df_if_exists_1, con=sqlite_buildin, name=table_name, if_exists="fail")
    sql.to_sql(frame=df_if_exists_1, con=sqlite_buildin, name=table_name, if_exists="replace", index=False)
    assert func_pp84sjy1(sql_select, con=sqlite_buildin) == [(1, "A"), (2, "B")]
    assert sql.to_sql(frame=df_if_exists_2, con=sqlite_buildin, name=table_name, if_exists="replace", index=False) == 3
    assert func_pp84sjy1(sql_select, con=sqlite_buildin) == [(3, "C"), (4, "D"), (5, "E")]
    func_ivdgn9x4(table_name, sqlite_buildin)
    assert sql.to_sql(frame=df_if_exists_1, con=sqlite_buildin, name=table_name, if_exists="fail", index=False) == 2
    assert func_pp84sjy1(sql_select, con=sqlite_buildin) == [(1, "A"), (2, "B")]
    assert sql.to_sql(frame=df_if_exists_2, con=sqlite_buildin, name=table_name, if_exists="append", index=False) == 3
    assert func_pp84sjy1(sql_select, con=sqlite_buildin) == [(1, "A"), (2, "B"), (3, "C"), (4, "D"), (5, "E")]
    func_ivdgn9x4(table_name, sqlite_buildin)


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def func_gm3zc5en(conn: str, request: pytest.FixtureRequest, test_frame1: DataFrame) -> None:
    pytest.importorskip("sqlalchemy")
    sql_conn = request.getfixturevalue(conn)
    with tm.ensure_clean() as name:
        db_uri = f"sqlite:///{name}"
        table = "iris"
        func_0t54gbeq.to_sql(name=table, con=db_uri, if_exists="replace", index=False)
        test_frame2 = sql.read_sql(table, db_uri)
        test_frame3 = sql.read_sql_table(table, db_uri)
        query = "SELECT * FROM iris"
        test_frame4 = sql.read_sql_query(query, db_uri)
    tm.assert_frame_equal(test_frame1, test_frame2)
    tm.assert_frame_equal(test_frame1, test_frame3)
    tm.assert_frame_equal(test_frame1, test_frame4)


@td.skip_if_installed("pg8000")
@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def func_2b3nk7wf(conn: str, request: pytest.FixtureRequest) -> None:
    pytest.importorskip("sqlalchemy")
    conn = request.getfixturevalue(conn)
    db_uri = "postgresql+pg8000://user:pass@host/dbname"
    with pytest.raises(ImportError, match="pg8000"):
        sql.read_sql("select * from table", db_uri)


@td.skip_if_installed("sqlalchemy")
def func_5iqexotr() -> None:
    class MockSqliteConnection:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.conn = sqlite3.Connection(*args, **kwargs)

        def __getattr__(self, name: str) -> Any:
            return getattr(self.conn, name)

        def func_cdufx9qd(self) -> None:
            self.conn.close()

    with contextlib.closing(MockSqliteConnection(":memory:")) as conn:
        with tm.assert_produces_warning(UserWarning, match="only supports SQLAlchemy"):
            sql.read_sql("SELECT 1", conn)


def func_l44l26ld(sqlite_str: str, iris_path: Path) -> str:
    pass  # Duplicated fixture; no action needed.


@pytest.mark.parametrize("conn", all_connectable)
@pytest.mark.parametrize("func", ["read_sql", "read_sql_query", "read_sql_table"])
def func_d2yypdx3(
    conn: str,
    request: pytest.FixtureRequest,
    read_sql: Callable[..., DataFrame],
    text: str,
    mode: Union[str, Tuple[str, str]],
    error: str,
    types_data_frame: DataFrame,
) -> None:
    pass  # Already defined above; no action needed.


def func_qkq7p4df(sqlite_buildin: sqlite3.Connection) -> None:
    conn = sqlite_buildin
    df = DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError, match="Empty table name specified"):
        df.to_sql(name="", con=conn)
    for ndx, weird_name in enumerate(
        ["test_weird_name", "test_weird_name[", "test_weird_name`", 'test_weird_name"', "test_weird_name'", "_b.test_weird_name_01-30"]
    ):
        assert sql.to_sql(name=weird_name, con=conn, if_exists="replace", index=False) == 2
        sql.table_exists(weird_name, conn)
        df2 = DataFrame([[1, 2], [3, 4]], columns=["a", weird_name])
        c_tbl = f"test_weird_col_name{ndx}"
        assert sql.to_sql(name=c_tbl, con=conn, if_exists="replace", index=False) == 2
        sql.table_exists(c_tbl, conn)
    assert sql.to_sql(name="test_unicode", con=conn, if_exists="replace", index=False) == 2
    result = sql.read_sql_query("SELECT * FROM test_unicode", conn)
    expected = DataFrame({"a": ["x", "y"]}, dtype=str)
    tm.assert_frame_equal(result, expected)


def func_qtggg6fj(query: str, *args: Any) -> str:
    _formatters: Dict[type, Callable[[Any], str]] = {
        datetime: "'{}'".format,
        str: "'{}'".format,
        np.str_: "'{}'".format,
        bytes: "'{}'".format,
        float: "{:.8f}".format,
        int: "{:d}".format,
        type(None): lambda x: "NULL",
        np.float64: "{:.10f}".format,
        bool: "'{!s}'".format,
    }
    processed_args: List[str] = []
    for arg in args:
        if isinstance(arg, float) and isna(arg):
            arg = None
        formatter = _formatters.get(type(arg))
        if formatter:
            processed_args.append(formatter(arg))
        else:
            processed_args.append(str(arg))
    return query % tuple(processed_args)


def func_pp84sjy1(query: str, con: Optional[Union[sqlite3.Connection, Any]] = None) -> Optional[List[Any]]:
    """Replace removed sql.tquery function"""
    with pandasSQL_builder(con) as pandas_sql:
        res = pandas_sql.execute(query).fetchall()
    return None if res is None else list(res)


def func_uxy8j245(sqlite_buildin: sqlite3.Connection) -> None:
    conn = sqlite_buildin
    iris_frame1 = sql.read_sql_query("SELECT * FROM iris", conn)
    iris_frame2 = sql.read_sql("SELECT * FROM iris", conn)
    tm.assert_frame_equal(iris_frame1, iris_frame2)
    msg = "Execution failed on sql 'iris': near 'iris': syntax error"
    with pytest.raises(sql.DatabaseError, match=msg):
        sql.read_sql("iris", conn)


def func_d3zige34(sqlite_buildin: sqlalchemy.engine.Connection) -> None:
    create_sql = sql.get_schema(test_frame1, "test")
    assert "CREATE" in create_sql


def func_l7ar16xf(conn: str, request: pytest.FixtureRequest) -> None:
    conn_name = conn
    sql_conn = request.getfixturevalue(conn)
    df = sql.read_sql_query("SELECT * FROM types", sql_conn)
    expected_type = object if "sqlite" in conn_name else np.datetime64
    assert issubclass(df.DateCol.dtype.type, expected_type)
    df = sql.read_sql_query("SELECT * FROM types", sql_conn, parse_dates=["DateCol"])
    assert issubclass(df.DateCol.dtype.type, np.datetime64)
    assert df.DateCol.tolist() == [Timestamp("2000-01-03 00:00:00"), Timestamp("2000-01-04 00:00:00")]
    df = sql.read_sql_query(
        "SELECT * FROM types", sql_conn, parse_dates={"DateCol": "%Y-%m-%d %H:%M:%S"}
    )
    assert issubclass(df.DateCol.dtype.type, np.datetime64)
    assert df.DateCol.tolist() == [Timestamp("2000-01-03 00:00:00"), Timestamp("2000-01-04 00:00:00")]
    df = sql.read_sql_query("SELECT * FROM types", sql_conn, parse_dates=["IntDateCol"])
    assert issubclass(df.IntDateCol.dtype.type, np.datetime64)
    assert df.IntDateCol.tolist() == [Timestamp("1986-12-25 00:00:00"), Timestamp("2013-01-01 00:00:00")]
    df = sql.read_sql_query("SELECT * FROM types", sql_conn, parse_dates={"IntDateCol": "s"})
    assert issubclass(df.IntDateCol.dtype.type, np.datetime64)
    assert df.IntDateCol.tolist() == [Timestamp("1986-12-25 00:00:00"), Timestamp("2013-01-01 00:00:00")]
    df = sql.read_sql_query("SELECT * FROM types", sql_conn, parse_dates={"IntDateCol": "s"})
    assert issubclass(df.IntDateCol.dtype.type, np.datetime64)
    assert df.IntDateCol.tolist() == [Timestamp("1986-12-25 00:00:00"), Timestamp("2013-01-01 00:00:00")]
    df = sql.read_sql_query("SELECT * FROM types", sql_conn, parse_dates={"IntDateOnlyCol": "%Y%m%d"})
    assert issubclass(df.IntDateOnlyCol.dtype.type, np.datetime64)
    assert df.IntDateOnlyCol.tolist() == [Timestamp("2010-10-10"), Timestamp("2010-12-12")]


@pytest.mark.parametrize("conn", all_connectable_types)
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
def func_bksdvid5(
    conn: str,
    request: pytest.FixtureRequest,
    read_sql: Callable[..., DataFrame],
    text: str,
    mode: Union[str, Tuple[str, str]],
    error: str,
    types_data_frame: DataFrame,
) -> None:
    pass  # Already defined above; no action needed.


@pytest.mark.parametrize("conn", all_connectable_types)
def func_mw6rtgbp(conn: str, request: pytest.FixtureRequest) -> None:
    sql_conn = request.getfixturevalue(conn)
    df = DataFrame(data={"i64": [2**62]})
    assert sql.to_sql(name="test_bigint", con=sql_conn, index=False) == 1
    result = sql.read_sql_table("test_bigint", sql_conn)
    tm.assert_frame_equal(result, df)


@pytest.mark.parametrize("conn", all_connectable_types)
def func_6tyoaq9t(conn: str, request: pytest.FixtureRequest) -> None:
    conn_name = conn
    sql_conn = request.getfixturevalue(conn)
    df = sql.read_sql_table("types", sql_conn)
    assert issubclass(df.DateCol.dtype.type, np.datetime64)


@pytest.mark.parametrize("conn", postgresql_connectable)
@pytest.mark.parametrize("parse_dates", [False, ["DateColWithTz"]])
def func_it5ppkwp(
    conn: str,
    request: pytest.FixtureRequest,
    parse_dates: Union[bool, List[str]],
) -> None:
    sql_conn = request.getfixturevalue(conn)
    expected = func_6ugk0vds(sql_conn)
    df = read_sql_query("select * from datetz", sql_conn, parse_dates=parse_dates)
    col = df.DateColWithTz
    tm.assert_series_equal(col, expected)


@pytest.mark.parametrize("conn", postgresql_connectable)
def func_aab0e2m5(conn: str, request: pytest.FixtureRequest) -> None:
    sql_conn = request.getfixturevalue(conn)
    expected = func_6ugk0vds(sql_conn)
    df = concat(list(read_sql_query("select * from datetz", sql_conn, chunksize=1)), ignore_index=True)
    col = df.DateColWithTz
    tm.assert_series_equal(col, expected)


@pytest.mark.parametrize("conn", postgresql_connectable)
def func_uf6b09il(conn: str, request: pytest.FixtureRequest) -> None:
    sql_conn = request.getfixturevalue(conn)
    result = sql.read_sql_table("datetz", sql_conn)
    expected = func_6ugk0vds(sql_conn).to_frame()
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def func_5zhgc38i(
    conn: str, request: pytest.FixtureRequest, dtype_backend_data: DataFrame, dtype_backend_expected: Callable[[str, Optional[str], str], DataFrame]
) -> None:
    pass  # Already defined above; no action needed.


@pytest.fixture
def func_iajdiknx() -> sqlite3.Connection:
    with contextlib.closing(sqlite3.connect(":memory:", detect_types=sqlite3.PARSE_DECLTYPES)) as closing_conn:
        with closing_conn as conn:
            yield conn


def func_2uk2d713(sqlite_builtin_detect_types: sqlite3.Connection) -> None:
    conn = sqlite_builtin_detect_types
    df = DataFrame({"t": [datetime(2020, 12, 31, 12)]}, dtype="datetime64[ns]")
    df.to_sql(name="test", con=conn, if_exists="replace", index=False)
    result = pd.read_sql("SELECT * FROM test", conn).iloc[0, 0]
    assert result == Timestamp("2020-12-31 12:00:00")


@pytest.mark.parametrize("conn", all_connectable)
def func_gm3zc5en(conn: str, request: pytest.FixtureRequest, test_frame1: DataFrame) -> None:
    pass  # Duplicated function; no action needed.


def func_n2hh6kt1(sqlite_buildin: sqlite3.Connection) -> None:
    conn = sqlite_buildin
    df = DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError, match="Empty table name specified"):
        df.to_sql(name="", con=conn)
    for ndx, weird_name in enumerate(["test_weird_name", "test_weird_name[", "test_weird_name`", 'test_weird_name"', "test_weird_name'", "_b.test_weird_name_01-30"]):
        assert sql.to_sql(name=weird_name, con=conn, if_exists="replace", index=False) == 2
        sql.table_exists(weird_name, conn)
        df2 = DataFrame([[1, 2], [3, 4]], columns=["a", weird_name])
        c_tbl = f"test_weird_col_name{ndx}"
        assert sql.to_sql(name=c_tbl, con=conn, if_exists="replace", index=False) == 2
        sql.table_exists(c_tbl, conn)
    assert sql.to_sql(name="test_unicode", con=conn, if_exists="replace", index=False) == 2
    result = sql.read_sql_query("SELECT * FROM test_unicode", conn)
    expected = DataFrame({"a": ["x", "y"]}, dtype=str)
    tm.assert_frame_equal(result, expected)


def func_qtggg6fj(query: str, *args: Any) -> str:
    _formatters: Dict[type, Callable[[Any], str]] = {
        datetime: "'{}'".format,
        str: "'{}'".format,
        np.str_: "'{}'".format,
        bytes: "'{}'".format,
        float: "{:.8f}".format,
        int: "{:d}".format,
        type(None): lambda x: "NULL",
        np.float64: "{:.10f}".format,
        bool: "'{!s}'".format,
    }
    processed_args: List[str] = []
    for arg in args:
        if isinstance(arg, float) and isna(arg):
            arg = None
        formatter = _formatters.get(type(arg), str)
        processed_args.append(formatter(arg))
    return query % tuple(processed_args)


def func_pp84sjy1(query: str, con: Optional[Union[sqlite3.Connection, Any]] = None) -> Optional[List[Any]]:
    """Replace removed sql.tquery function"""
    with pandasSQL_builder(con) as pandas_sql:
        res = pandas_sql.execute(query).fetchall()
    return None if res is None else list(res)


def func_rkxevvsi(sqlite_buildin: sqlite3.Connection) -> None:
    conn = sqlite_buildin
    cols = ["A", "B"]
    data = [(0.8, True), (0.9, None)]
    df = DataFrame(data, columns=cols)
    assert df.to_sql(name="dtype_test", con=conn) == 2
    assert df.to_sql(name="dtype_test2", con=conn, dtype={"B": "STRING"}) == 2
    assert func_ixrjqqps(conn, "dtype_test", "B") == "INTEGER"
    assert func_ixrjqqps(conn, "dtype_test2", "B") == "STRING"
    msg = r"B \(\<class 'bool'\>\) not a string"
    with pytest.raises(ValueError, match=msg):
        df.to_sql(name="error", con=conn, dtype={"B": bool})
    assert df.to_sql(name="single_dtype_test", con=conn, dtype="STRING") == 2
    assert func_ixrjqqps(conn, "single_dtype_test", "A") == "STRING"
    assert func_ixrjqqps(conn, "single_dtype_test", "B") == "STRING"


def func_lvcdb1j1(sqlite_buildin: sqlite3.Connection) -> None:
    conn = sqlite_buildin
    cols = {"Bool": Series([True, None]), "Date": Series([datetime(2012, 5, 1), None]), "Int": Series([1, None], dtype="object"), "Float": Series([1.1, None])}
    df = DataFrame(cols)
    tbl = "notna_dtype_test"
    assert df.to_sql(name=tbl, con=conn) == 2
    assert func_ixrjqqps(conn, tbl, "Bool") == "INTEGER"
    assert func_ixrjqqps(conn, tbl, "Date") == "TIMESTAMP"
    assert func_ixrqqps(conn, tbl, "Int") == "INTEGER"
    assert func_ixrqqps(conn, tbl, "Float") == "REAL"


def func_mbyo73vx(sqlite_buildin: sqlite3.Connection) -> None:
    conn = sqlite_buildin
    df = DataFrame({"a": [1, 2]}, dtype="int64")
    assert sql.to_sql(name="test_bigintwarning", con=conn, index=False) == 2
    with tm.assert_produces_warning(None):
        sql.read_sql_table("test_bigintwarning", conn)


def func_u99w3ioe(sqlite_engine: sqlalchemy.engine.Connection) -> None:
    conn = sqlite_engine
    df = DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError, match="Empty table name specified"):
        df.to_sql(name="", con=conn, if_exists="replace")


@pytest.fixture
def func_80f6ik4l(sqlite_engine: sqlalchemy.engine.Connection) -> None:
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


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def func_8530tifg(conn: str, request: pytest.FixtureRequest) -> None:
    pass  # Duplicated function; no action needed.


def func_n2hh6kt1(sqlite_buildin: sqlite3.Connection) -> None:
    conn = sqlite_buildin
    df = DataFrame({"t": [datetime(2020, 12, 31, 12)]}, dtype="datetime64[ns]")
    df.to_sql(name="test", con=conn, if_exists="replace", index=False)
    result = pd.read_sql("SELECT * FROM test", conn).iloc[0, 0]
    assert result == Timestamp("2020-12-31 12:00:00")


@pytest.mark.db
def func_z63qwo2h(postgresql_psycopg2_engine: SQLAlchemyEngine) -> None:
    pass  # Already defined above; no action needed.


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def func_rkxevvsi(conn: str, request: pytest.FixtureRequest) -> None:
    pass  # Duplicated function; no action needed.


def func_gqczontn(sqlite_buildin: sqlite3.Connection) -> None:
    conn = sqlite_buildin
    df = DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
    assert df.to_sql(name="test_table", con=conn, if_exists="replace", index=False) == 2
    with closing(sqlite_buildin.connect()) as conn:
        the_sum = sum(row[0] for row in conn.execute("select * from test_table"))
        assert the_sum == 3
        result = sql.read_sql_query("SELECT * FROM test_table", conn)
    assert list(result.columns) == ["a", "b"]
    tm.assert_frame_equal(result, df)


def func_q5bps3lr(sqlite_buildin: sqlite3.Connection) -> None:
    df_if_exists_1 = DataFrame({"col1": [1, 2], "col2": ["A", "B"]})
    df_if_exists_2 = DataFrame({"col1": [3, 4, 5], "col2": ["C", "D", "E"]})
    table_name = "table_if_exists"
    sql_select = f"SELECT * FROM {table_name}"
    msg = "'notvalidvalue' is not valid for if_exists"
    with pytest.raises(ValueError, match=msg):
        sql.to_sql(frame=df_if_exists_1, con=sqlite_buildin, name=table_name, if_exists="notvalidvalue")
    func_ivdgn9x4(table_name, sqlite_buildin)
    sql.to_sql(frame=df_if_exists_1, con=sqlite_buildin, name=table_name, if_exists="fail")
    msg = "Table 'table_if_exists' already exists"
    with pytest.raises(ValueError, match=msg):
        sql.to_sql(frame=df_if_exists_1, con=sqlite_buildin, name=table_name, if_exists="fail")
    sql.to_sql(frame=df_if_exists_1, con=sqlite_buildin, name=table_name, if_exists="replace", index=False)
    assert func_pp84sjy1(sql_select, con=sqlite_buildin) == [(1, "A"), (2, "B")]
    assert sql.to_sql(frame=df_if_exists_2, con=sqlite_buildin, name=table_name, if_exists="replace", index=False) == 3
    assert func_pp84sjy1(sql_select, con=sqlite_buildin) == [(3, "C"), (4, "D"), (5, "E")]
    func_ivdgn9x4(table_name, sqlite_buildin)
    assert sql.to_sql(frame=df_if_exists_1, con=sqlite_buildin, name=table_name, if_exists="fail", index=False) == 2
    assert func_pp84sjy1(sql_select, con=sqlite_buildin) == [(1, "A"), (2, "B")]
    assert sql.to_sql(frame=df_if_exists_2, con=sqlite_buildin, name=table_name, if_exists="append", index=False) == 3
    assert func_pp84sjy1(sql_select, con=sqlite_buildin) == [(1, "A"), (2, "B"), (3, "C"), (4, "D"), (5, "E")]
    func_ivdgn9x4(table_name, sqlite_buildin)


def func_62kber9k(test_frame3: DataFrame) -> None:
    with tm.ensure_clean() as name:
        with closing(sqlite3.connect(name)) as conn:
            assert sql.to_sql(test_frame3, "test_frame3_legacy", conn, index=False) == 4
        with closing(sqlite3.connect(name)) as conn:
            result = sql.read_sql_query("SELECT * FROM test_frame3_legacy;", conn)
    tm.assert_frame_equal(test_frame3, result)


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def func_9fs4ok85(conn: str, request: pytest.FixtureRequest) -> None:
    sql_conn = request.getfixturevalue(conn)
    frame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list("ABCD")), index=date_range("2000-01-01", periods=10, freq="B"))
    frame.iloc[0, 0] = np.nan
    create_sql = sql.get_schema(frame, "test")
    with sql.SQLDatabase(sql_conn) as pandasSQL:
        with pandasSQL.run_transaction():
            pandasSQL.execute(create_sql)
            for _, row in frame.iterrows():
                sql_conn.execute("INSERT INTO test VALUES (?, ?, ?, ?)", tuple(row))
    sql_conn.commit()
    result = sql.read_sql_query("SELECT * FROM test", sql_conn)
    tm.assert_frame_equal(result, frame, rtol=0.001)


def func_7lkf02h2(sqlite_buildin: sqlite3.Connection) -> None:
    conn = sqlite_buildin
    create_sql = """
    CREATE TABLE test (
        a TEXT,
        b TEXT,
        c REAL,
        PRIMARY KEY (a, b)
    );
    """
    conn.execute(create_sql)
    with pandasSQL_builder(conn) as pandas_sql:
        pandas_sql.execute("INSERT INTO test VALUES ('foo', 'bar', 1.234)")
        pandas_sql.execute("INSERT INTO test VALUES ('foo', 'baz', 2.567)")
        with pytest.raises(sql.DatabaseError, match="Execution failed on sql"):
            pandas_sql.execute("INSERT INTO test VALUES ('foo', 'bar', 7)")


def func_467tcft9(sqlite_buildin: sqlite3.Connection) -> None:
    conn = sqlite_buildin
    cols = ["A", "B"]
    data = [(0.8, True), (0.9, None)]
    df = DataFrame(data, columns=cols)
    assert df.to_sql(name="dtype_test", con=conn) == 2
    assert df.to_sql(name="dtype_test2", con=conn, dtype={"B": "STRING"}) == 2
    meta = MetaData()
    meta.reflect(bind=conn)
    assert "STRING" in func_ixrjqqps(conn, "dtype_test2", "B")
    msg = r"B \(\<class 'bool'\>\) not a string"
    with pytest.raises(ValueError, match=msg):
        df.to_sql(name="error", con=conn, dtype={"B": bool})
    assert df.to_sql(name="single_dtype_test", con=conn, dtype="STRING") == 2
    meta.reflect(bind=conn)
    sqltype = meta.tables["single_dtype_test"].columns["A"].type
    sqltypeb = meta.tables["single_dtype_test"].columns["B"].type
    assert isinstance(sqltype, String)
    assert isinstance(sqltypeb, String)


def func_id93o80v(sqlite_buildin: sqlite3.Connection) -> None:
    conn = sqlite_buildin
    df = DataFrame({"A": [0, 1, 2], "B": [0.2, np.nan, 5.6]})
    assert df.to_sql(name="test_nan", con=conn, index=False) == 3
    df_read = sql.read_sql_table("test_nan", conn)
    tm.assert_frame_equal(df_read, df)
    df_read = sql.read_sql_query("SELECT * FROM test_nan", conn)
    tm.assert_frame_equal(df_read, df)


@pytest.mark.parametrize("conn", all_connectable)
def func_ptir4cpz(
    conn: str, request: pytest.FixtureRequest, string_storage: str, dtype_backend: Optional[str], dtype_backend_data: DataFrame, dtype_backend_expected: Callable[[str, Optional[str], str], DataFrame]
) -> None:
    sql_conn = request.getfixturevalue(conn)
    table = "test"
    df = DataFrame({"a": [1, np.nan, 3], "b": [np.nan, np.nan, np.nan], "c": [1.5, 2.0, 2.5], "d": [1.5, 2.0, 2.5]})
    df.to_sql(name="test_timedelta", con=sql_conn, if_exists="replace", index=False)
    result = sql.read_sql_query("SELECT * FROM test_timedelta", sql_conn)
    if "postgresql" in conn or "mysql" in conn:
        expected = Series([pd.DateOffset(months=0, days=0, microseconds=1000000, nanoseconds=0), pd.DateOffset(months=0, days=0, microseconds=3000000, nanoseconds=0)], name="foo")
    else:
        expected = Series([1_000_000, 3_000_000], dtype="int64", name="foo")
    tm.assert_series_equal(result["foo"], expected)


@pytest.mark.parametrize("conn", all_connectable)
def func_6tuzub17(conn: str, request: pytest.FixtureRequest, test_frame1: DataFrame) -> None:
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(reason="ADBC implementation does not create index", strict=True)
        )
    sql_conn = request.getfixturevalue(conn)
    with pandasSQL_builder(sql_conn) as pandasSQL:
        with pandasSQL.run_transaction() as trans:
            trans.execute("CREATE TABLE test_trans (A INT, B TEXT)")
    class DummyException(Exception):
        pass

    ins_sql = "INSERT INTO test_trans (A,B) VALUES (1, 'blah')"
    if isinstance(sql_conn, sqlalchemy.engine.Engine):
        from sqlalchemy import text

        ins_sql = text(ins_sql)
    try:
        with pandasSQL_builder(sql_conn) as pandasSQL:
            with pandasSQL.run_transaction() as trans:
                trans.execute(ins_sql)
                raise DummyException("error")
    except DummyException:
        pass
    with pandasSQL_builder(sql_conn) as pandasSQL:
        with pandasSQL.run_transaction():
            res = pandasSQL.read_query("SELECT * FROM test_trans")
    assert len(res) == 0
    with pandasSQL_builder(sql_conn) as pandasSQL:
        with pandasSQL.run_transaction() as trans:
            trans.execute(ins_sql)
            res2 = pandasSQL.read_query("SELECT * FROM test_trans")
    assert len(res2) == 1


@pytest.mark.parametrize("conn", all_connectable)
@pytest.mark.parametrize("dtype_backend", ["pyarrow", "numpy_nullable"])
def func_bksdvid5(
    conn: str,
    request: pytest.FixtureRequest,
    read_sql: Callable[..., DataFrame],
    text: str,
    mode: Union[str, Tuple[str, str]],
    error: str,
    types_data_frame: DataFrame,
) -> None:
    pass  # Already defined above; no action needed.


@pytest.mark.parametrize("conn", all_connectable)
def func_1ac0vv9e(
    conn: str, request: pytest.FixtureRequest, test_frame1: DataFrame
) -> None:
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(reason="'get_schema' not implemented for ADBC drivers", strict=True)
        )
    sql_conn = request.getfixturevalue(conn)
    create_sql = sql.get_schema(test_frame1, "test")
    assert "CREATE" in create_sql


@pytest.mark.parametrize("conn", all_connectable)
def func_asoxdy38(conn: str, request: pytest.FixtureRequest, test_frame1: DataFrame) -> None:
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(reason="'get_schema' not implemented for ADBC drivers", strict=True)
        )
    sql_conn = request.getfixturevalue(conn)
    create_sql = sql.get_schema(test_frame1, "test", schema="pypi")
    assert "CREATE TABLE pypi." in create_sql


@pytest.mark.parametrize("conn", all_connectable)
def func_kf3erc25(conn: str, request: pytest.FixtureRequest) -> None:
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(reason="'get_schema' not implemented for ADBC drivers", strict=True)
        )
    sql_conn = request.getfixturevalue(conn)
    df = DataFrame({"Col1": [1.1, 2.2], "Col2": [3.3, 4.4]})
    create_sql = sql.get_schema(df, "test", con=sql_conn, keys="Col1")
    if "mysql" in conn:
        constraint_sentence = "CONSTRAINT test_pk PRIMARY KEY (`Col1`)"
    else:
        constraint_sentence = 'CONSTRAINT test_pk PRIMARY KEY ("Col1")'
    assert constraint_sentence in create_sql
    create_sql_multi = sql.get_schema(df, "test", con=sql_conn, keys=["Col1", "Col2"])
    if "mysql" in conn:
        constraint_sentence = "CONSTRAINT test_pk PRIMARY KEY (`Col1`, `Col2`)"
    else:
        constraint_sentence = 'CONSTRAINT test_pk PRIMARY KEY ("Col1", "Col2")'
    assert constraint_sentence in create_sql_multi


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def func_8530tifg(conn: str, request: pytest.FixtureRequest) -> None:
    pass  # Duplicated function; no action needed.


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def func_5zhgc38i(
    conn: str, request: pytest.FixtureRequest, dtype_backend: Optional[str], test_frame1: DataFrame
) -> None:
    pass  # Already defined above; no action needed.


def func_ivdgn9x4(table_name: str, conn: Union[sqlite3.Connection, Any]) -> None:
    pass  # Already defined above; no action needed.


def func_we2yj4hw(view_name: str, conn: Union[sqlite3.Connection, Any]) -> None:
    pass  # Already defined above; no action needed.


@pytest.fixture
def func_a43o2gt6() -> sqlite3.Connection:
    with contextlib.closing(sqlite3.connect(":memory:")) as closing_conn:
        with closing_conn as conn:
            yield conn


@pytest.fixture
def func_b8wxk1ky(sqlite_buildin: sqlite3.Connection, iris_path: Path) -> sqlite3.Connection:
    func_pher6cwr(sqlite_buildin, iris_path)
    func_pl6k8b7m(sqlite_buildin)
    return sqlite_buildin


@pytest.fixture
def func_vicjtqs2(sqlite_buildin: sqlite3.Connection, types_data: List[Dict[str, Any]]) -> sqlite3.Connection:
    types_data = [tuple(entry.values()) for entry in types_data]
    func_itpn7y40(sqlite_buildin, types_data)
    return sqlite_buildin


@pytest.fixture
def func_c39k8nrt(sqlite_adbc_conn: Any, types_data: List[Dict[str, Any]]) -> Any:
    import adbc_driver_manager as mgr

    conn = sqlite_adbc_conn
    try:
        conn.adbc_get_table_schema("types")
    except mgr.ProgrammingError:
        conn.rollback()
        new_data = []
        for entry in types_data:
            entry["BoolCol"] = int(entry["BoolCol"])
            if entry["BoolColWithNull"] is not None:
                entry["BoolColWithNull"] = int(entry["BoolColWithNull"])
            new_data.append(tuple(entry.values()))
        func_itpn7y40(conn, new_data)
        conn.commit()
    return conn


@pytest.fixture
def func_eu1gay0g(sqlite_engine: SQLAlchemyEngine) -> Any:
    with func_u7mnodl1.connect() as conn:
        yield conn


@pytest.fixture
def func_ubwl2kdv(sqlite_str: str, types_data: List[Dict[str, Any]]) -> str:
    sqlalchemy = pytest.importorskip("sqlalchemy")
    engine = sqlalchemy.create_engine(sqlite_str)
    func_4b6osnwu(engine, types_data, "sqlite")
    engine.dispose()
    return sqlite_str


@pytest.fixture
def func_uu4xu6bf(sqlite_engine: SQLAlchemyEngine, types_data: List[Dict[str, Any]]) -> SQLAlchemyEngine:
    func_4b6osnwu(sqlite_engine, types_data, "sqlite")
    return sqlite_engine


@pytest.fixture
def func_xr9tt3y0(sqlite_engine_types: SQLAlchemyEngine) -> Any:
    with func_uu4xu6bf.connect() as conn:
        yield conn


@pytest.fixture
def func_x9vpff5b() -> Any:
    pass  # Already defined above; no action needed.


@pytest.fixture
def func_phe89hn0(sqlite_adbc_conn: Any, iris_path: Path) -> Any:
    pass  # Already defined above; no action needed.


@pytest.fixture
def func_ncgxk4gh(sqlite_adbc_conn: Any, types_data: List[Dict[str, Any]]) -> Any:
    pass  # Already defined above; no action needed.


@pytest.fixture
def func_hqkgqhe5(postgresql_psycopg2_engine_iris: SQLAlchemyEngine) -> Any:
    with func_yj70ac7y.connect() as conn:
        yield conn


@pytest.fixture
def func_4q4b4oft(postgresql_psycopg2_engine_types: SQLAlchemyEngine) -> Any:
    with func_5uli6gcu.connect() as conn:
        yield conn


@pytest.fixture
def func_cyhppbd6() -> str:
    pass  # Already defined above; no action needed.


@pytest.fixture
def func_55hm10va(sqlite_str: str) -> SQLAlchemyEngine:
    pass  # Already defined above; no action needed.


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def func_qslliqk4(
    conn: str, request: pytest.FixtureRequest, test_frame1: DataFrame
) -> None:
    pass  # Duplicated function; no action needed.


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def func_5zhgc38i(
    conn: str, request: pytest.FixtureRequest, dtype_backend: Optional[str], test_frame1: DataFrame
) -> None:
    pass  # Duplicated function; no action needed.


def func_7e43funb(sqlite_conn: sqlite3.Connection) -> None:
    pass  # Already defined above; no action needed.


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def func_fkom083c(conn: str, request: pytest.FixtureRequest) -> None:
    pass  # Already defined above; no action needed.


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def func_8530tifg(conn: str, request: pytest.FixtureRequest) -> None:
    pass  # Duplicated function; no action needed.


@pytest.mark.parametrize("conn", all_connectable)
@pytest.mark.parametrize("func", ["read_sql", "read_sql_query"])
def func_4qlqyi11(
    conn: str, request: pytest.FixtureRequest, string_storage: str, dtype_backend: Optional[str], dtype_backend_data: DataFrame, dtype_backend_expected: Callable[[str, Optional[str], str], DataFrame]
) -> None:
    conn_name = conn
    if "sqlite" in conn and "adbc" not in conn:
        request.applymarker(pytest.mark.xfail(reason="'test for public schema only specific to postgresql'"))
    sql_conn = request.getfixturevalue(conn)
    with pandasSQL_builder(sql_conn) as pandasSQL:
        with pandasSQL.run_transaction():
            assert pandasSQL.to_sql(test_frame1, "test_frame", method="multi") == len(test_frame1)
    num_entries = len(test_frame1)
    num_rows = func_9huuyabu(sql_conn, "test_frame")
    assert num_rows == num_entries


def func_2bh2d713(sqlite_buildin_detect_types: sqlite3.Connection) -> None:
    conn = sqlite_buildin_detect_types
    df = DataFrame({"t": [datetime(2020, 12, 31, 12)]}, dtype="datetime64[ns]")
    df.to_sql(name="test", con=conn, if_exists="replace", index=False)
    result = pd.read_sql("SELECT * FROM test", conn).iloc[0, 0]
    assert result == Timestamp("2020-12-31 12:00:00")


def func_0ontf6q3(conn: str, request: pytest.FixtureRequest) -> None:
    pass  # Duplicated function; no action needed.
