"""
Stub file for 'metadb_f7c13c' module
"""

from __future__ import annotations
from collections.abc import Iterator
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
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
from sqlalchemy.sql import Select
from shillelagh.adapters.base import Adapter
from shillelagh.fields import Field
from superset.models.core import Database

F = TypeVar("F", bound=Callable[..., Any])

class SupersetAPSWDialect(APSWDialect):
    name: ClassVar[str] = "superset"

    def __init__(self, allowed_dbs: Optional[List[str]] = None, **kwargs: Any) -> None:
        ...

    def create_connect_args(self, url: str) -> Tuple[Tuple[()], Dict[str, Any]]:
        ...

class Duration(Field[datetime.timedelta, datetime.timedelta]):
    type: ClassVar[str] = "DURATION"
    db_api_type: ClassVar[str] = "DATETIME"

class Decimal(Field[Decimal, Decimal]):
    type: ClassVar[str] = "DECIMAL"
    db_api_type: ClassVar[str] = "NUMBER"

class FallbackField(Field[Any, str]):
    type: ClassVar[str] = "TEXT"
    db_api_type: ClassVar[str] = "STRING"

    def parse(self, value: Any) -> Optional[str]:
        ...

class SupersetShillelaghAdapter(Adapter):
    safe: ClassVar[bool] = True
    supports_limit: ClassVar[bool] = True
    supports_offset: ClassVar[bool] = True
    type_map: ClassVar[Dict[Type[Any], Type[Field[Any, Any]]]] = {
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

    def __init__(self, uri: str, prefix: str = "superset", **kwargs: Any) -> None:
        ...

    def _set_columns(self) -> None:
        ...

    def get_columns(self) -> Dict[str, Field[Any, Any]]:
        ...

    def _build_sql(
        self,
        bounds: Dict[str, Filter],
        order: List[Tuple[str, RequestedOrder]],
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Select:
        ...

    def get_data(
        self,
        bounds: Dict[str, Filter],
        order: List[Tuple[str, RequestedOrder]],
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[Dict[str, Any]]:
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
    def delete_row(self, row_id: Union[str, int]) -> None:
        ...

    @check_dml
    @has_rowid
    def update_row(self, row_id: Union[str, int], row: Dict[str, Any]) -> None:
        ...