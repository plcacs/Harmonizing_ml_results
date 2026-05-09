"""
Stub file for 'metadb_f7c13c' module.
"""

from __future__ import annotations
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)
from datetime import datetime, timedelta
from decimal import Decimal
from urllib.parse import unquote
from sqlalchemy import MetaData, Table, Select
from sqlalchemy.exc import NoSuchTableError
from shillelagh.adapters.base import Adapter
from shillelagh.backends.apsw.dialects.base import APSWDialect
from shillelagh.exceptions import ProgrammingError
from shillelagh.fields import Field, Boolean, Date, DateTime, Float, Integer, String, Time
from shillelagh.typing import RequestedOrder, Row

F = TypeVar('F', bound=Callable[..., Any])

class SupersetAPSWDialect(APSWDialect):
    name: str = 'superset'

    def __init__(self, allowed_dbs: Optional[Any] = None, **kwargs: Any) -> None:
        ...

    def create_connect_args(self, url: Any) -> Tuple[Tuple[()], Dict[str, Any]]:
        ...

class Duration(Field[datetime.timedelta, datetime.timedelta]):
    type: str = 'DURATION'
    db_api_type: str = 'DATETIME'

class Decimal(Field[Decimal, Decimal]):
    type: str = 'DECIMAL'
    db_api_type: str = 'NUMBER'

class FallbackField(Field[Any, str]):
    type: str = 'TEXT'
    db_api_type: str = 'STRING'

    def parse(self, value: Any) -> str:
        ...

class SupersetShillelaghAdapter(Adapter):
    supports_limit: bool = True
    supports_offset: bool = True
    type_map: Dict[Type, Type[Field]] = {
        bool: Boolean,
        float: Float,
        int: Integer,
        str: String,
        datetime.date: Date,
        datetime.datetime: DateTime,
        datetime.time: Time,
        datetime.timedelta: Duration,
        Decimal: Decimal,
    }

    def __init__(self, uri: str, prefix: str = 'superset', **kwargs: Any) -> None:
        ...

    @staticmethod
    def supports(
        uri: str,
        fast: bool = True,
        prefix: Optional[str] = None,
        allowed_dbs: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> bool:
        ...

    @staticmethod
    def parse_uri(uri: str) -> Tuple[str]:
        ...

    def _set_columns(self) -> None:
        ...

    def get_columns(self) -> Dict[str, Field]:
        ...

    def _build_sql(
        self,
        bounds: Dict[str, Any],
        order: List[RequestedOrder],
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Select:
        ...

    def get_data(
        self,
        bounds: Dict[str, Any],
        order: List[RequestedOrder],
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        **kwargs: Any,
    ) -> Generator[Row, None, None]:
        ...

    @overload
    def insert_row(self, row: Dict[str, Any]) -> int:
        ...

    @overload
    def insert_row(self, row: Dict[str, Any]) -> None:
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