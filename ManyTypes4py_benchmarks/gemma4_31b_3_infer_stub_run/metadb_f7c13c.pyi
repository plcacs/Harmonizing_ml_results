from __future__ import annotations
import datetime
import decimal
from collections.abc import Iterator, Mapping
from typing import Any, Callable, Optional, TypeVar, Union, Sequence
from shillelagh.adapters.base import Adapter
from shillelagh.backends.apsw.dialects.base import APSWDialect
from shillelagh.fields import Field, Order
from shillelagh.filters import Filter
from sqlalchemy.sql import Select
from sqlalchemy.sql.schema import Table

F = TypeVar('F', bound=Callable[..., Any])

def check_dml(method: F) -> F: ...

def has_rowid(method: F) -> F: ...

class Duration(Field[datetime.timedelta, datetime.timedelta]):
    type: str
    db_api_type: str

class Decimal(Field[decimal.Decimal, decimal.Decimal]):
    type: str
    db_api_type: str

class FallbackField(Field[Any, str]):
    type: str
    db_api_type: str
    def parse(self, value: Any) -> Optional[str]: ...

class SupersetAPSWDialect(APSWDialect):
    name: str
    def __init__(self, allowed_dbs: Optional[Sequence[str]] = None, **kwargs: Any) -> None: ...
    def create_connect_args(self, url: Any) -> tuple[tuple[Any, ...], dict[str, Any]]: ...

class SupersetShillelaghAdapter(Adapter):
    safe: bool
    supports_limit: bool
    supports_offset: bool
    type_map: dict[type, type[Field]]
    prefix: Optional[str]
    database: str
    table: str
    schema: Optional[str]
    catalog: Optional[str]
    _rowid: Optional[str]
    _allow_dml: bool
    engine_context: Callable[[], Any]
    _table: Table
    columns: dict[str, Field]

    def __init__(self, uri: str, prefix: str = 'superset', **kwargs: Any) -> None: ...
    
    @staticmethod
    def supports(uri: str, fast: bool = True, prefix: Optional[str] = 'superset', allowed_dbs: Optional[Sequence[str]] = None, **kwargs: Any) -> bool: ...
    
    @staticmethod
    def parse_uri(uri: str) -> tuple[str]: ...
    
    @classmethod
    def get_field(cls, python_type: type) -> Field: ...
    
    def _set_columns(self) -> None: ...
    
    def get_columns(self) -> dict[str, Field]: ...
    
    def _build_sql(self, bounds: Mapping[str, Filter], order: Sequence[tuple[str, Order]], limit: Optional[int] = None, offset: Optional[int] = None) -> Select: ...
    
    def get_data(self, bounds: Mapping[str, Filter], order: Sequence[tuple[str, Order]], limit: Optional[int] = None, offset: Optional[int] = None, **kwargs: Any) -> Iterator[dict[str, Any]]: ...
    
    def insert_row(self, row: dict[str, Any]) -> Any: ...
    
    def delete_row(self, row_id: Any) -> None: ...
    
    def update_row(self, row_id: Any, row: dict[str, Any]) -> None: ...