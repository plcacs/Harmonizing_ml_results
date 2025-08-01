#!/usr/bin/env python3
"""
A SQLAlchemy dialect for querying across Superset databases.

The dialect ``superset://`` allows users to query any table in any database that has been
configured in Superset, eg:

    > SELECT * FROM "examples.birth_names";

The syntax for tables is:

    database[[.catalog].schema].table

The dialect is built on top of Shillelagh, a framework for building DB API 2.0 libraries
and SQLAlchemy dialects based on SQLite. SQLite will parse the SQL, and pass the filters
to the adapter. The adapter builds a SQLAlchemy query object reading data from the table
and applying any filters (as well as sorting, limiting, and offsetting).

Note that no aggregation is done on the database. Aggregations and other operations like
joins and unions are done in memory, using the SQLite engine.
"""
from __future__ import annotations
import datetime
import decimal
import operator
import urllib.parse
from collections.abc import Iterator
from functools import partial, wraps
from typing import Any, Callable, Dict, Iterator as TypingIterator, List, Optional, Tuple, TypeVar, cast
from flask import current_app
from shillelagh.adapters.base import Adapter
from shillelagh.backends.apsw.dialects.base import APSWDialect
from shillelagh.exceptions import ProgrammingError
from shillelagh.fields import Boolean, Date, DateTime, Field, Float, Integer, Order, String, Time
from shillelagh.filters import Equal, Filter, Range
from shillelagh.typing import RequestedOrder, Row
from sqlalchemy import func, MetaData, Table
from sqlalchemy.engine.url import URL
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.sql import Select, select
from superset import db, feature_flag_manager, security_manager, sql_parse

F = TypeVar('F', bound=Callable[..., Any])


def check_dml(method: F) -> F:
    """
    Decorator that prevents DML against databases where it's not allowed.
    """
    @wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not self._allow_dml:
            raise ProgrammingError(f'DML not enabled in database "{self.database}"')
        return method(self, *args, **kwargs)
    return cast(F, wrapper)


def has_rowid(method: F) -> F:
    """
    Decorator that prevents updates/deletes on tables without a rowid.
    """
    @wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not self._rowid:
            raise ProgrammingError('Can only modify data in a table with a single, integer, primary key')
        return method(self, *args, **kwargs)
    return cast(F, wrapper)


class SupersetAPSWDialect(APSWDialect):
    """
    A SQLAlchemy dialect for an internal Superset engine.

    This dialect allows query to be executed across different Superset
    databases. For example, to read data from the `birth_names` table in the
    `examples` databases:

        >>> engine = create_engine('superset://')
        >>> conn = engine.connect()
        >>> results = conn.execute('SELECT * FROM "examples.birth_names"')

    Queries can also join data across different Superset databases.

    The dialect is built in top of the Shillelagh library, leveraging SQLite to
    create virtual tables on-the-fly proxying Superset tables. The
    `SupersetShillelaghAdapter` adapter is responsible for returning data when a
    Superset table is accessed.
    """
    name: str = 'superset'

    def __init__(self, allowed_dbs: Optional[List[str]] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.allowed_dbs: Optional[List[str]] = allowed_dbs

    def create_connect_args(self, url: URL) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """
        A custom Shillelagh SQLAlchemy dialect with a single adapter configured.
        """
        return (
            (),
            {
                'path': ':memory:',
                'adapters': ['superset'],
                'adapter_kwargs': {'superset': {'prefix': None, 'allowed_dbs': self.allowed_dbs}},
                'safe': True,
                'isolation_level': self.isolation_level,
            },
        )


class Duration(Field[datetime.timedelta, datetime.timedelta]):
    """
    Shillelagh field used for representing durations as `timedelta` objects.
    """
    type: str = 'DURATION'
    db_api_type: str = 'DATETIME'


class Decimal(Field[decimal.Decimal, decimal.Decimal]):
    """
    Shillelagh field used for representing decimals.
    """
    type: str = 'DECIMAL'
    db_api_type: str = 'NUMBER'


class FallbackField(Field[Any, str]):
    """
    Fallback field for unknown types; converts to string.
    """
    type: str = 'TEXT'
    db_api_type: str = 'STRING'

    def parse(self, value: Any) -> Optional[str]:
        return value if value is None else str(value)


class SupersetShillelaghAdapter(Adapter):
    """
    A Shillelagh adapter for Superset tables.

    Shillelagh adapters are responsible for fetching data from a given resource,
    allowing it to be represented as a virtual table in SQLite. This one works
    as a proxy to Superset tables.
    """
    safe: bool = True
    supports_limit: bool = True
    supports_offset: bool = True
    type_map: Dict[type, Any] = {
        bool: Boolean,
        float: Float,
        int: Integer,
        str: String,
        datetime.date: Date,
        datetime.datetime: DateTime,
        datetime.time: Time,
        datetime.timedelta: Duration,
        decimal.Decimal: Decimal,
    }

    @staticmethod
    def supports(uri: str, fast: bool = True, prefix: Optional[str] = 'superset', allowed_dbs: Optional[List[str]] = None, **kwargs: Any) -> bool:
        """
        Return if a table is supported by the adapter.

        An URL for a table has the format [prefix.]database[[.catalog].schema].table,
        eg, `superset.examples.birth_names`.

        When using the Superset SQLAlchemy and DB engine spec the prefix is dropped, so
        that tables should have the format database[[.catalog].schema].table.
        """
        parts: List[str] = [urllib.parse.unquote(part) for part in uri.split('.')]
        if prefix is not None:
            if parts.pop(0) != prefix:
                return False
        if allowed_dbs is not None and parts[0] not in allowed_dbs:
            return False
        return 2 <= len(parts) <= 4

    @staticmethod
    def parse_uri(uri: str) -> Tuple[str]:
        """
        Pass URI through unmodified.
        """
        return (uri,)

    def __init__(self, uri: str, prefix: Optional[str] = 'superset', **kwargs: Any) -> None:
        if not feature_flag_manager.is_feature_enabled('ENABLE_SUPERSET_META_DB'):
            raise ProgrammingError('Superset meta database is disabled')
        super().__init__(**kwargs)
        parts: List[str] = [urllib.parse.unquote(part) for part in uri.split('.')]
        if prefix is not None:
            if prefix != parts.pop(0):
                raise ProgrammingError('Invalid prefix')
            self.prefix: Optional[str] = prefix
        else:
            self.prefix = None
        self.database: str = parts.pop(0)
        self.table: str = parts.pop(-1)
        self.schema: Optional[str] = parts.pop(-1) if parts else None
        self.catalog: Optional[str] = parts.pop(-1) if parts else None
        self._rowid: Optional[str] = None
        self._allow_dml: bool = False
        self._set_columns()

    @classmethod
    def get_field(cls, python_type: type) -> Field[Any, Any]:
        """
        Convert a Python type into a Shillelagh field.
        """
        class_ = cls.type_map.get(python_type, FallbackField)
        return class_(filters=[Equal, Range], order=Order.ANY, exact=True)

    def _set_columns(self) -> None:
        """
        Inspect the table and get its columns.

        This is done on initialization because it's expensive.
        """
        from superset.models.core import Database
        database_obj = db.session.query(Database).filter_by(database_name=self.database).first()
        if database_obj is None:
            raise ProgrammingError(f'Database not found: {self.database}')
        self._allow_dml = database_obj.allow_dml
        table_obj = sql_parse.Table(self.table, self.schema, self.catalog)
        security_manager.raise_for_access(database=database_obj, table=table_obj)
        self.engine_context = partial(database_obj.get_sqla_engine, catalog=self.catalog, schema=self.schema)
        metadata = MetaData()
        with self.engine_context() as engine:
            try:
                self._table: Table = Table(self.table, metadata, schema=self.schema, autoload=True, autoload_with=engine)
            except NoSuchTableError as ex:
                raise ProgrammingError(f'Table does not exist: {self.table}') from ex
        primary_keys = [column for column in list(self._table.primary_key) if column.primary_key]
        if len(primary_keys) == 1 and isinstance(primary_keys[0].type.python_type, type(int)):
            self._rowid = primary_keys[0].name
        self.columns: Dict[str, Field[Any, Any]] = {column.name: self.get_field(column.type.python_type) for column in self._table.c}

    def get_columns(self) -> Dict[str, Field[Any, Any]]:
        """
        Return table columns.
        """
        return self.columns

    def _build_sql(
        self,
        bounds: Dict[str, Filter[Any]],
        order: List[Tuple[str, RequestedOrder]],
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> Select[Any]:
        """
        Build SQLAlchemy query object.
        """
        query: Select[Any] = select([self._table])
        for column_name, filter_ in bounds.items():
            column = self._table.c[column_name]
            if isinstance(filter_, Equal):
                query = query.where(column == filter_.value)
            elif isinstance(filter_, Range):
                if filter_.start is not None:
                    op = operator.ge if filter_.include_start else operator.gt
                    query = query.where(op(column, filter_.start))
                if filter_.end is not None:
                    op = operator.le if filter_.include_end else operator.lt
                    query = query.where(op(column, filter_.end))
            else:
                raise ProgrammingError(f'Invalid filter: {filter_}')
        for column_name, requested_order in order:
            column = self._table.c[column_name]
            if requested_order == Order.DESCENDING:
                column = column.desc()
            query = query.order_by(column)
        if limit is not None:
            query = query.limit(limit)
        if offset is not None:
            query = query.offset(offset)
        return query

    def get_data(
        self,
        bounds: Dict[str, Filter[Any]],
        order: List[Tuple[str, RequestedOrder]],
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        **kwargs: Any
    ) -> TypingIterator[Row]:
        """
        Return data for a `SELECT` statement.
        """
        app_limit: Optional[int] = current_app.config['SUPERSET_META_DB_LIMIT']
        if limit is None:
            limit = app_limit
        elif app_limit is not None:
            limit = min(limit, app_limit)
        query = self._build_sql(bounds, order, limit, offset)
        with self.engine_context() as engine:
            connection = engine.connect()
            rows = connection.execute(query)
            for i, row in enumerate(rows):
                data: Row = dict(zip(self.columns, row, strict=False))
                data['rowid'] = data[self._rowid] if self._rowid else i
                yield data

    @check_dml
    def insert_row(self, row: Row) -> Any:
        """
        Insert a single row.
        """
        row_id = row.pop('rowid')
        if row_id and self._rowid:
            if row.get(self._rowid) != row_id:
                raise ProgrammingError(f'Invalid rowid specified: {row_id}')
            row[self._rowid] = row_id
        if self._rowid and row[self._rowid] is None and self._table.c[self._rowid].autoincrement:
            row.pop(self._rowid)
        query = self._table.insert().values(**row)
        with self.engine_context() as engine:
            connection = engine.connect()
            result = connection.execute(query)
            if self._rowid:
                return result.inserted_primary_key[0]
            query = select([func.count()]).select_from(self._table)
            return connection.execute(query).scalar()

    @check_dml
    @has_rowid
    def delete_row(self, row_id: Any) -> None:
        """
        Delete a single row given its row ID.
        """
        query = self._table.delete().where(self._table.c[self._rowid] == row_id)
        with self.engine_context() as engine:
            connection = engine.connect()
            connection.execute(query)

    @check_dml
    @has_rowid
    def update_row(self, row_id: Any, row: Row) -> None:
        """
        Update a single row given its row ID.

        Note that the updated row might have a new row ID.
        """
        new_row_id = row.pop('rowid')
        if new_row_id:
            if row.get(self._rowid) != new_row_id:
                raise ProgrammingError(f'Invalid rowid specified: {new_row_id}')
            row[self._rowid] = new_row_id
        query = self._table.update().where(self._table.c[self._rowid] == row_id).values(**row)
        with self.engine_context() as engine:
            connection = engine.connect()
            connection.execute(query)