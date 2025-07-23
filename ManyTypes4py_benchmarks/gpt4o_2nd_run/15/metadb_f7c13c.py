from __future__ import annotations
import datetime
import decimal
import operator
import urllib.parse
from collections.abc import Iterator
from functools import partial, wraps
from typing import Any, Callable, cast, TypeVar, Optional, Dict, Tuple, List, Union, Generator
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
    @wraps(method)
    def wrapper(self, *args: Any, **kwargs: Any) -> Any:
        if not self._allow_dml:
            raise ProgrammingError(f'DML not enabled in database "{self.database}"')
        return method(self, *args, **kwargs)
    return cast(F, wrapper)

def has_rowid(method: F) -> F:
    @wraps(method)
    def wrapper(self, *args: Any, **kwargs: Any) -> Any:
        if not self._rowid:
            raise ProgrammingError('Can only modify data in a table with a single, integer, primary key')
        return method(self, *args, **kwargs)
    return cast(F, wrapper)

class Duration(Field[datetime.timedelta, datetime.timedelta]):
    type = 'DURATION'
    db_api_type = 'DATETIME'

class Decimal(Field[decimal.Decimal, decimal.Decimal]):
    type = 'DECIMAL'
    db_api_type = 'NUMBER'

class FallbackField(Field[Any, str]):
    type = 'TEXT'
    db_api_type = 'STRING'

    def parse(self, value: Any) -> str:
        return value if value is None else str(value)

class SupersetAPSWDialect(APSWDialect):
    name = 'superset'

    def __init__(self, allowed_dbs: Optional[List[str]] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.allowed_dbs = allowed_dbs

    def create_connect_args(self, url: URL) -> Tuple[Tuple, Dict[str, Any]]:
        return ((), {'path': ':memory:', 'adapters': ['superset'], 'adapter_kwargs': {'superset': {'prefix': None, 'allowed_dbs': self.allowed_dbs}}, 'safe': True, 'isolation_level': self.isolation_level})

class SupersetShillelaghAdapter(Adapter):
    safe = True
    supports_limit = True
    supports_offset = True
    type_map: Dict[Any, Field] = {bool: Boolean, float: Float, int: Integer, str: String, datetime.date: Date, datetime.datetime: DateTime, datetime.time: Time, datetime.timedelta: Duration, decimal.Decimal: Decimal}

    @staticmethod
    def supports(uri: str, fast: bool = True, prefix: str = 'superset', allowed_dbs: Optional[List[str]] = None, **kwargs: Any) -> bool:
        parts = [urllib.parse.unquote(part) for part in uri.split('.')]
        if prefix is not None:
            if parts.pop(0) != prefix:
                return False
        if allowed_dbs is not None and parts[0] not in allowed_dbs:
            return False
        return 2 <= len(parts) <= 4

    @staticmethod
    def parse_uri(uri: str) -> Tuple[str]:
        return (uri,)

    def __init__(self, uri: str, prefix: str = 'superset', **kwargs: Any) -> None:
        if not feature_flag_manager.is_feature_enabled('ENABLE_SUPERSET_META_DB'):
            raise ProgrammingError('Superset meta database is disabled')
        super().__init__(**kwargs)
        parts = [urllib.parse.unquote(part) for part in uri.split('.')]
        if prefix is not None:
            if prefix != parts.pop(0):
                raise ProgrammingError('Invalid prefix')
            self.prefix = prefix
        self.database = parts.pop(0)
        self.table = parts.pop(-1)
        self.schema = parts.pop(-1) if parts else None
        self.catalog = parts.pop(-1) if parts else None
        self._rowid: Optional[str] = None
        self._allow_dml = False
        self._set_columns()

    @classmethod
    def get_field(cls, python_type: Any) -> Field:
        class_ = cls.type_map.get(python_type, FallbackField)
        return class_(filters=[Equal, Range], order=Order.ANY, exact=True)

    def _set_columns(self) -> None:
        from superset.models.core import Database
        database = db.session.query(Database).filter_by(database_name=self.database).first()
        if database is None:
            raise ProgrammingError(f'Database not found: {self.database}')
        self._allow_dml = database.allow_dml
        table = sql_parse.Table(self.table, self.schema, self.catalog)
        security_manager.raise_for_access(database=database, table=table)
        self.engine_context = partial(database.get_sqla_engine, catalog=self.catalog, schema=self.schema)
        metadata = MetaData()
        with self.engine_context() as engine:
            try:
                self._table = Table(self.table, metadata, schema=self.schema, autoload=True, autoload_with=engine)
            except NoSuchTableError as ex:
                raise ProgrammingError(f'Table does not exist: {self.table}') from ex
        primary_keys = [column for column in list(self._table.primary_key) if column.primary_key]
        if len(primary_keys) == 1 and isinstance(primary_keys[0].type.python_type, type(int)):
            self._rowid = primary_keys[0].name
        self.columns = {column.name: self.get_field(column.type.python_type) for column in self._table.c}

    def get_columns(self) -> Dict[str, Field]:
        return self.columns

    def _build_sql(self, bounds: Dict[str, Filter], order: List[Tuple[str, RequestedOrder]], limit: Optional[int] = None, offset: Optional[int] = None) -> Select:
        query = select([self._table])
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

    def get_data(self, bounds: Dict[str, Filter], order: List[Tuple[str, RequestedOrder]], limit: Optional[int] = None, offset: Optional[int] = None, **kwargs: Any) -> Generator[Row, None, None]:
        app_limit = current_app.config['SUPERSET_META_DB_LIMIT']
        if limit is None:
            limit = app_limit
        elif app_limit is not None:
            limit = min(limit, app_limit)
        query = self._build_sql(bounds, order, limit, offset)
        with self.engine_context() as engine:
            connection = engine.connect()
            rows = connection.execute(query)
            for i, row in enumerate(rows):
                data = dict(zip(self.columns, row, strict=False))
                data['rowid'] = data[self._rowid] if self._rowid else i
                yield data

    @check_dml
    def insert_row(self, row: Dict[str, Any]) -> Union[int, None]:
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
    def delete_row(self, row_id: int) -> None:
        query = self._table.delete().where(self._table.c[self._rowid] == row_id)
        with self.engine_context() as engine:
            connection = engine.connect()
            connection.execute(query)

    @check_dml
    @has_rowid
    def update_row(self, row_id: int, row: Dict[str, Any]) -> None:
        new_row_id = row.pop('rowid')
        if new_row_id:
            if row.get(self._rowid) != new_row_id:
                raise ProgrammingError(f'Invalid rowid specified: {new_row_id}')
            row[self._rowid] = new_row_id
        query = self._table.update().where(self._table.c[self._rowid] == row_id).values(**row)
        with self.engine_context() as engine:
            connection = engine.connect()
            connection.execute(query)
