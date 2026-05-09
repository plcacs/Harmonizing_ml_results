"""
Stub file for 'metadb_f7c13c' module
"""

from __future__ import annotations
from collections.abc import Iterator
from datetime import datetime, date, time, timedelta
from decimal import Decimal
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    overload,
)
from sqlalchemy import MetaData, Table, Select
from sqlalchemy.engine.url import URL
from shillelagh.adapters.base import Adapter
from shillelagh.fields import (
    Boolean,
    Date,
    DateTime,
    Field,
    Float,
    Integer,
    Order,
    String,
    Time,
)
from superset import Database

F = TypeVar('F', bound=Callable[..., Any])

class SupersetAPSWDialect:
    name: str
    
    def __init__(self, allowed_dbs: Optional[List[str]] = None, **kwargs: Any) -> None:
        ...
    
    def create_connect_args(self, url: URL) -> tuple[tuple[()], dict[str, Any]]:
        ...

def check_dml(method: F) -> F:
    ...

def has_rowid(method: F) -> F:
    ...

class Duration(Field[datetime.timedelta, datetime.timedelta]):
    type: str
    db_api_type: str

class Decimal(Field[Decimal, Decimal]):
    type: str
    db_api_type: str

class FallbackField(Field[Any, str]):
    type: str
    db_api_type: str

    def parse(self, value: Any) -> str:
        ...

class SupersetShillelaghAdapter(Adapter):
    safe: bool
    supports_limit: bool
    supports_offset: bool
    type_map: Dict[Type[Any], Type[Field[Any, Any]]]
    _rowid: Optional[str]
    _allow_dml: bool
    columns: Dict[str, Field[Any, Any]]
    _table: Table
    engine_context: Callable[[], Any]
    
    def __init__(self, uri: str, prefix: str = 'superset', **kwargs: Any) -> None:
        ...
    
    @classmethod
    def get_field(cls, python_type: Type[Any]) -> Field[Any, Any]:
        ...

    def _set_columns(self) -> None:
        ...

    def get_columns(self) -> Dict[str, Field[Any, Any]]:
        ...

    def _build_sql(
        self,
        bounds: Dict[str, Union[Equal, Range]],
        order: List[Tuple[str, Order]],
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Select:
        ...

    def get_data(
        self,
        bounds: Dict[str, Union[Equal, Range]],
        order: List[Tuple[str, Order]],
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[Dict[str, Any]]:
        ...

    @check_dml
    def insert_row(self, row: Dict[str, Any]) -> Optional[int]:
        ...

    @check_dml
    @has_rowid
    def delete_row(self, row_id: int) -> None:
        ...

    @check_dml
    @has_rowid
    def update_row(self, row_id: int, row: Dict[str, Any]) -> None:
        ...