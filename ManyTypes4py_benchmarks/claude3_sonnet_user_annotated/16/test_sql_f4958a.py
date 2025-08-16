from __future__ import annotations

import contextlib
from contextlib import closing
import csv
from datetime import (
    date,
    datetime,
    time,
    timedelta,
)
from io import StringIO
from pathlib import Path
import sqlite3
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast
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
    from sqlalchemy.engine import Connection, Engine
    from sqlalchemy.sql.expression import TextClause
    from adbc_driver_manager.dbapi import Connection as ADBCConnection


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
            "postgresql": "SELECT * FROM iris WHERE \"Name\" LIKE '%'",
        },
    }


def iris_table_metadata() -> "sqlalchemy.Table":
    import sqlalchemy
    from sqlalchemy import (
        Column,
        Double,
        Float,
        MetaData,
        String,
        Table,
    )

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


def create_and_load_iris_sqlite3(conn: Union[sqlite3.Connection, "ADBCConnection"], iris_file: Path) -> None:
    stmt = """CREATE TABLE iris (
            "SepalLength" REAL,
            "SepalWidth" REAL,
            "PetalLength" REAL,
            "PetalWidth" REAL,
            "Name" TEXT
        )"""

    cur = conn.cursor()
    cur.execute(stmt)
    with iris_file.open(newline=None, encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        stmt = "INSERT INTO iris VALUES(?, ?, ?, ?, ?)"
        # ADBC requires explicit types - no implicit str -> float conversion
        records = []
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
    cur.close()

    conn.commit()


def create_and_load_iris_postgresql(conn: "ADBCConnection", iris_file: Path) -> None:
    stmt = """CREATE TABLE iris (
            "SepalLength" DOUBLE PRECISION,
            "SepalWidth" DOUBLE PRECISION,
            "PetalLength" DOUBLE PRECISION,
            "PetalWidth" DOUBLE PRECISION,
            "Name" TEXT
        )"""
    with conn.cursor() as cur:
        cur.execute(stmt)
        with iris_file.open(newline=None, encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            stmt = "INSERT INTO iris VALUES($1, $2, $3, $4, $5)"
            # ADBC requires explicit types - no implicit str -> float conversion
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


def create_and_load_iris(conn: Union["Engine", "Connection"], iris_file: Path) -> None:
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


def create_and_load_iris_view(conn: Union[sqlite3.Connection, "Engine", "Connection", "ADBCConnection"]) -> None:
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


def types_table_metadata(dialect: str) -> "sqlalchemy.Table":
    from sqlalchemy import (
        TEXT,
        Boolean,
        Column,
        DateTime,
        Float,
        Integer,
        MetaData,
        Table,
    )

    date_type = TEXT if dialect == "sqlite" else DateTime
    bool_type = Integer if dialect == "sqlite" else Boolean
    metadata = MetaData()
    types = Table(
        "types",
        metadata,
        Column("TextCol", TEXT),
        # error: Cannot infer type argument 1 of "Column"
        Column("DateCol", date_type),  # type: ignore[misc]
        Column("IntDateCol", Integer),
        Column("IntDateOnlyCol", Integer),
        Column("FloatCol", Float),
        Column("IntCol", Integer),
        # error: Cannot infer type argument 1 of "Column"
        Column("BoolCol", bool_type),  # type: ignore[misc]
        Column("IntColWithNull", Integer),
        # error: Cannot infer type argument 1 of "Column"
        Column("BoolColWithNull", bool_type),  # type: ignore[misc]
    )
    return types


def create_and_load_types_sqlite3(
    conn: Union[sqlite3.Connection, "ADBCConnection"], types_data: List[Dict[str, Any]]
) -> None:
    stmt = """CREATE TABLE types (
                    "TextCol" TEXT,
                    "DateCol" TEXT,
                    "IntDateCol" INTEGER,
                    "IntDateOnlyCol" INTEGER,
                    "FloatCol" REAL,
                    "IntCol" INTEGER,
                    "BoolCol" INTEGER,
                    "IntColWithNull" INTEGER,
                    "BoolColWithNull" INTEGER
                )"""

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


def create_and_load_types_postgresql(
    conn: "ADBCConnection", types_data: List[Dict[str, Any]]
) -> None:
    with conn.cursor() as cur:
        stmt = """CREATE TABLE types (
                        "TextCol" TEXT,
                        "DateCol" TIMESTAMP,
                        "IntDateCol" INTEGER,
                        "IntDateOnlyCol" INTEGER,
                        "FloatCol" DOUBLE PRECISION,
                        "IntCol" INTEGER,
                        "BoolCol" BOOLEAN,
                        "IntColWithNull" INTEGER,
                        "BoolColWithNull" BOOLEAN
                    )"""
        cur.execute(stmt)

        stmt = """
                INSERT INTO types
                VALUES($1, $2::timestamp, $3, $4, $5, $6, $7, $8, $9)
                """

        cur.executemany(stmt, types_data)

    conn.commit()


def create_and_load_types(
    conn: Union["Engine", "Connection"], types_data: List[Dict[str, Any]], dialect: str
) -> None:
    from sqlalchemy import insert
    from sqlalchemy.engine import Engine

    types = types_table_metadata(dialect)

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


def create_and_load_postgres_datetz(conn: Union["Engine", "Connection"]) -> Series:
    from sqlalchemy import (
        Column,
        DateTime,
        MetaData,
        Table,
        insert,
    )
    from sqlalchemy.engine import Engine

    metadata = MetaData()
    datetz = Table("datetz", metadata, Column("DateColWithTz", DateTime(timezone=True)))
    datetz_data = [
        {
            "DateColWithTz": "2000-01-01 00:00:00-08:00",
        },
        {
            "DateColWithTz": "2000-06-01 00:00:00-07:00",
        },
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

    # "2000-01-01 00:00:00-08:00" should convert to
    # "2000-01-01 08:00:00"
    # "2000-06-01 00:00:00-07:00" should convert to
    # "2000-06-01 07:00:00"
    # GH 6415
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


def count_rows(
    conn: Union[str, sqlite3.Connection, "Engine", "Connection", "ADBCConnection"], table_name: str
) -> int:
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
def iris_path(datapath) -> Path:
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
            "FloatCol": 10.10,
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
            "FloatCol": 10.10,
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
        ("2000-01-03 00:00:00", 2**31 - 1, -1.987670),
        ("2000-01-04 00:00:00", -29, -0.0412318367011),
        ("2000-01-05 00:00:00", 20000, 0.731167677815),
        ("2000-01-06 00:00:00", -290867, 1.56762092543),
    ]
    return DataFrame(data, columns=columns)


def get_all_views(
    conn: Union[sqlite3.Connection, "Engine", "Connection", "ADBCConnection"]
) -> List[str]:
    if isinstance(conn, sqlite3.Connection):
        c = conn.execute("SELECT name FROM sqlite_master WHERE type='view'")
        return [view[0] for view in c.fetchall()]
    else:
        adbc = import_optional_dependency("adbc_driver_manager.dbapi", errors="ignore")
        if adbc and isinstance(conn, adbc.Connection):
            results = []
            info = conn.adbc_get_objects().read_all().to_pylist()
            for catalog in info:
                catalog["catalog_name"]
                for schema in catalog["catalog_db_schemas"]:
                    schema["db_schema_name"]
                    for table in schema["db_schema_tables"]:
                        if table["table_type"] == "view":
                            view_name = table["table_name"]
                            results.append(view_name)

            return results
        else:
            from sqlalchemy import inspect

            return inspect(conn).get_view_names()


def get_all_tables(
    conn: Union[sqlite3.Connection, "Engine", "Connection", "ADBCConnection"]
) -> List[str]:
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
                    for table in schema["db_schema_tables"]:
                        if table["table_type"] == "table":
                            table_name = table["table_name"]
                            results.append(table_name)

            return results
        else:
            from sqlalchemy import inspect

            return inspect(conn).get_table_names()


def drop_table(
    table_name: str,
    conn: Union[sqlite3.Connection, "Engine", "Connection", "ADBCConnection"],
) -> None:
    if isinstance(conn, sqlite3.Connection):
        conn.execute(f"DROP TABLE IF EXISTS {sql._get_valid_sqlite_name(table_name)}")
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


def drop_view(
    view_name: str,
    conn: Union[sqlite3.Connection, "Engine", "Connection", "ADBCConnection"],
) -> None:
    import sqlalchemy

    if isinstance(conn, sqlite3.Connection):
        conn.execute(f"DROP VIEW IF EXISTS {sql._get_valid_sqlite_name(view_name)}")
        conn.commit()
    else:
        adbc = import_optional_dependency("adbc_driver_manager.dbapi", errors="ignore")
        if adbc and isinstance(conn, adbc.Connection):
            with conn.cursor() as cur:
                cur.execute(f'DROP VIEW IF EXISTS "{view_name}"')
        else:
            quoted_view = conn.engine.dialect.identifier_preparer.quote_identifier(
                view_name
            )
            stmt = sqlalchemy.text(f"DROP VIEW IF EXISTS {quoted_view}")
            with conn.begin() as con:
                con.execute(stmt)  # type: ignore[union-attr]
