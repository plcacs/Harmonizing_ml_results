from __future__ import annotations

import datetime
import decimal
from collections.abc import Collection, Iterator, Mapping, Sequence
from typing import Any, Callable, Optional, TypeVar

from shillelagh.adapters.base import Adapter
from shillelagh.backends.apsw.dialects.base import APSWDialect
from shillelagh.fields import Field
from shillelagh.filters import Filter
from shillelagh.typing import RequestedOrder, Row
from sqlalchemy.engine.url import URL
from sqlalchemy.sql import Select

F = TypeVar("F", bound=Callable[..., Any])


def check_dml(method: F) -> F: ...
def has_rowid(method: F) -> F: ...


class SupersetAPSWDialect(APSWDialect):
    name: str = ...
    allowed_dbs: Optional[Collection[str]]

    def __init__(self, allowed_dbs: Optional[Collection[str]] = ..., **kwargs: Any) -> None: ...
    def create_connect_args(self, url: URL) -> tuple[tuple[Any, ...], dict[str, Any]]: ...


class Duration(Field[datetime.timedelta, datetime.timedelta]):
    type: str = ...
    db_api_type: str = ...


class Decimal(Field[decimal.Decimal, decimal.Decimal]):
    type: str = ...
    db_api_type: str = ...


class FallbackField(Field[Any, str]):
    type: str = ...
    db_api_type: str = ...

    def parse(self, value: Any) -> Optional[str]: ...


class SupersetShillelaghAdapter(Adapter):
    safe: bool = ...
    supports_limit: bool = ...
    supports_offset: bool = ...
    type_map: dict[type[Any], type[Field[Any, Any]]] = ...

    prefix: Optional[str]
    database: str
    table: str
    schema: Optional[str]
    catalog: Optional[str]
    _rowid: Optional[str]
    _allow_dml: bool
    columns: dict[str, Field[Any, Any]]

    @staticmethod
    def supports(
        uri: str,
        fast: bool = ...,
        prefix: Optional[str] = ...,
        allowed_dbs: Optional[Collection[str]] = ...,
        **kwargs: Any,
    ) -> bool: ...
    @staticmethod
    def parse_uri(uri: str) -> tuple[str, ...]: ...
    def __init__(self, uri: str, prefix: Optional[str] = ..., **kwargs: Any) -> None: ...
    @classmethod
    def get_field(cls, python_type: type[Any]) -> Field[Any, Any]: ...
    def get_columns(self) -> dict[str, Field[Any, Any]]: ...
    def _build_sql(
        self,
        bounds: Mapping[str, Filter],
        order: Sequence[tuple[str, RequestedOrder]],
        limit: Optional[int] = ...,
        offset: Optional[int] = ...,
    ) -> Select: ...
    def get_data(
        self,
        bounds: Mapping[str, Filter],
        order: Sequence[tuple[str, RequestedOrder]],
        limit: Optional[int] = ...,
        offset: Optional[int] = ...,
        **kwargs: Any,
    ) -> Iterator[Row]: ...
    def insert_row(self, row: dict[str, Any]) -> int: ...
    def delete_row(self, row_id: int) -> None: ...
    def update_row(self, row_id: int, row: dict[str, Any]) -> None: ...