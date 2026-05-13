from __future__ import annotations
import datetime
import decimal
from collections.abc import Iterator
from functools import partial
from typing import Any, Callable, Optional, TypeVar
from flask import Flask
from shillelagh.adapters.base import Adapter
from shillelagh.backends.apsw.dialects.base import APSWDialect
from shillelagh.exceptions import ProgrammingError
from shillelagh.fields import Boolean, Date, DateTime, Field, Float, Integer, Order, String, Time
from shillelagh.filters import Equal, Filter, Range
from shillelagh.typing import RequestedOrder, Row
from sqlalchemy import MetaData, Table
from sqlalchemy.engine.url import URL
from sqlalchemy.sql import Select
from typing_extensions import Literal

F = TypeVar('F', bound=Callable[..., Any])

def check_dml(method: F) -> F: ...
def has_rowid(method: F) -> F: ...

class Duration(Field[datetime.timedelta, datetime.timedelta]):
    type = 'DURATION'
    db_api_type = 'DATETIME'

class Decimal(Field[decimal.Decimal, decimal.Decimal]):
    type = 'DECIMAL'
    db_api_type = 'NUMBER'

class FallbackField(Field[Any, str]):
    type = 'TEXT'
    db_api_type = 'STRING'
    def parse(self, value: Any) -> Optional[str]: ...

class SupersetShillelaghAdapter(Adapter):
    safe: bool = True
    supports_limit: bool = True
    supports_offset: bool = True
    type_map: dict[type, type[Field[Any, Any]]] = ...
    prefix: str
    database: str
    table: str
    schema: str | None
    catalog: str | None
    _rowid: str | None
    _allow_dml: bool
    _table: Table
    columns: dict[str, Field[Any, Any]]
    engine_context: Any
    @staticmethod
    def supports(uri: str, fast: bool = True, prefix: str = 'superset', allowed_dbs: Optional[list[str]] = None, **kwargs: Any) -> bool: ...
    @staticmethod
    def parse_uri(uri: str) -> tuple[str]: ...
    def __init__(self, uri: str, prefix: str = 'superset', **kwargs: Any) -> None: ...
    @classmethod
    def get_field(cls, python_type: type) -> Field[Any, Any]: ...
    def _set_columns(self) -> None: ...
    def get_columns(self) -> dict[str, Field[Any, Any]]: ...
    def _build_sql(self, bounds: dict[str, Filter], order: list[tuple[str, RequestedOrder]], limit: Optional[int] = None, offset: Optional[int] = None) -> Select: ...
    def get_data(self, bounds: dict[str, Filter], order: list[tuple[str, RequestedOrder]], limit: Optional[int] = None, offset: Optional[int] = None, **kwargs: Any) -> Iterator[Row]: ...
    @check_dml
    def insert_row(self, row: dict[str, Any]) -> int: ...
    @check_dml
    @has_rowid
    def delete_row(self, row_id: int) -> None: ...
    @check_dml
    @has_rowid
    def update_row(self, row_id: int, row: dict[str, Any]) -> None: ...

class SupersetAPSWDialect(APSWDialect):
    name = 'superset'
    allowed_dbs: list[str] | None
    def __init__(self, allowed_dbs: Optional[list[str]] = None, **kwargs: Any) -> None: ...
    def create_connect_args(self, url: URL) -> tuple[tuple[()], dict[str, Any]]: ...