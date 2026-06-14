from collections.abc import Collection, Iterator
from typing import Any, Callable, TypeVar
import datetime
import decimal

from shillelagh.adapters.base import Adapter
from shillelagh.backends.apsw.dialects.base import APSWDialect
from shillelagh.fields import Field, Order
from shillelagh.filters import Filter
from shillelagh.typing import Row
from sqlalchemy.engine.url import URL
from sqlalchemy.sql import Select

F = TypeVar("F", bound=Callable[..., Any])


class SupersetAPSWDialect(APSWDialect):
    name: str = "superset"

    def __init__(self, allowed_dbs: Collection[str] | None = ..., **kwargs: Any) -> None: ...
    def create_connect_args(self, url: URL) -> tuple[tuple[Any, ...], dict[str, Any]]: ...


def check_dml(method: F) -> F: ...
def has_rowid(method: F) -> F: ...


class Duration(Field[datetime.timedelta, datetime.timedelta]):
    type: str = "DURATION"
    db_api_type: str = "DATETIME"


class Decimal(Field[decimal.Decimal, decimal.Decimal]):
    type: str = "DECIMAL"
    db_api_type: str = "NUMBER"


class FallbackField(Field[Any, str]):
    type: str = "TEXT"
    db_api_type: str = "STRING"

    def parse(self, value: Any) -> str | None: ...


class SupersetShillelaghAdapter(Adapter):
    safe: bool = True
    supports_limit: bool = True
    supports_offset: bool = True
    type_map: dict[type[Any], type[Field[Any, Any]]]

    @staticmethod
    def supports(
        uri: str,
        fast: bool = ...,
        prefix: str | None = ...,
        allowed_dbs: Collection[str] | None = ...,
        **kwargs: Any,
    ) -> bool: ...
    @staticmethod
    def parse_uri(uri: str) -> tuple[str, ...]: ...
    def __init__(self, uri: str, prefix: str | None = ..., **kwargs: Any) -> None: ...
    @classmethod
    def get_field(cls, python_type: type[Any]) -> Field[Any, Any]: ...
    def _set_columns(self) -> None: ...
    def get_columns(self) -> dict[str, Field[Any, Any]]: ...
    def _build_sql(
        self,
        bounds: dict[str, Filter],
        order: list[tuple[str, Order | None]],
        limit: int | None = ...,
        offset: int | None = ...,
    ) -> Select: ...
    def get_data(
        self,
        bounds: dict[str, Filter],
        order: list[tuple[str, Order | None]],
        limit: int | None = ...,
        offset: int | None = ...,
        **kwargs: Any,
    ) -> Iterator[Row]: ...
    def insert_row(self, row: dict[str, Any]) -> int: ...
    def delete_row(self, row_id: int) -> None: ...
    def update_row(self, row_id: int, row: dict[str, Any]) -> None: ...