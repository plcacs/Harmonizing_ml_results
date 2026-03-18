```pyi
from __future__ import annotations

import logging
from datetime import datetime
from re import Pattern
from typing import Any, TYPE_CHECKING

from sqlalchemy.dialects.postgresql import DOUBLE_PRECISION, ENUM, JSON
from sqlalchemy.dialects.postgresql.base import PGInspector
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.engine.url import URL
from sqlalchemy.types import Date, DateTime, String
from superset.constants import TimeGrain
from superset.db_engine_specs.base import BaseEngineSpec, BasicParametersMixin
from superset.errors import ErrorLevel, SupersetError, SupersetErrorType
from superset.exceptions import SupersetException, SupersetSecurityException
from superset.models.sql_lab import Query
from superset.utils.core import GenericDataType

if TYPE_CHECKING:
    from superset.models.core import Database

logger: logging.Logger

CONNECTION_INVALID_USERNAME_REGEX: Pattern[str]
CONNECTION_INVALID_PASSWORD_REGEX: Pattern[str]
CONNECTION_INVALID_PASSWORD_NEEDED_REGEX: Pattern[str]
CONNECTION_INVALID_HOSTNAME_REGEX: Pattern[str]
CONNECTION_PORT_CLOSED_REGEX: Pattern[str]
CONNECTION_HOST_DOWN_REGEX: Pattern[str]
CONNECTION_UNKNOWN_DATABASE_REGEX: Pattern[str]
COLUMN_DOES_NOT_EXIST_REGEX: Pattern[str]
SYNTAX_ERROR_REGEX: Pattern[str]

def parse_options(connect_args: dict[str, Any]) -> dict[str, str]: ...

class PostgresBaseEngineSpec(BaseEngineSpec):
    engine: str
    engine_name: str
    _time_grain_expressions: dict[TimeGrain | None, str]
    custom_errors: dict[Pattern[str], tuple[str, SupersetErrorType, dict[str, list[str]]]]
    
    @classmethod
    def fetch_data(cls, cursor: Any, limit: int | None = None) -> list[Any]: ...
    
    @classmethod
    def epoch_to_dttm(cls) -> str: ...
    
    @classmethod
    def convert_dttm(cls, target_type: str, dttm: datetime, db_extra: dict[str, Any] | None = None) -> str | None: ...

class PostgresEngineSpec(BasicParametersMixin, PostgresBaseEngineSpec):
    engine: str
    engine_aliases: set[str]
    supports_dynamic_schema: bool
    supports_catalog: bool
    supports_dynamic_catalog: bool
    default_driver: str
    sqlalchemy_uri_placeholder: str
    encryption_parameters: dict[str, str]
    max_column_name_length: int
    try_remove_schema_from_table_name: bool
    column_type_mappings: tuple[tuple[Pattern[str], Any, GenericDataType], ...]
    
    @classmethod
    def get_schema_from_engine_params(cls, sqlalchemy_uri: URL, connect_args: dict[str, Any]) -> str | None: ...
    
    @classmethod
    def get_default_schema_for_query(cls, database: Database, query: Query) -> str | None: ...
    
    @classmethod
    def adjust_engine_params(cls, uri: URL, connect_args: dict[str, Any], catalog: str | None = None, schema: str | None = None) -> tuple[URL, dict[str, Any]]: ...
    
    @classmethod
    def get_default_catalog(cls, database: Database) -> str | None: ...
    
    @classmethod
    def get_prequeries(cls, database: Database, catalog: str | None = None, schema: str | None = None) -> list[str]: ...
    
    @classmethod
    def get_allow_cost_estimate(cls, extra: dict[str, Any]) -> bool: ...
    
    @classmethod
    def estimate_statement_cost(cls, database: Database, statement: str, cursor: Any) -> dict[str, float]: ...
    
    @classmethod
    def query_cost_formatter(cls, raw_cost: list[dict[str, Any]]) -> list[dict[str, str]]: ...
    
    @classmethod
    def get_catalog_names(cls, database: Database, inspector: Inspector) -> set[str]: ...
    
    @classmethod
    def get_table_names(cls, database: Database, inspector: Inspector, schema: str) -> set[str]: ...
    
    @staticmethod
    def get_extra_params(database: Database) -> dict[str, Any]: ...
    
    @classmethod
    def get_datatype(cls, type_code: Any) -> str | None: ...
    
    @classmethod
    def get_cancel_query_id(cls, cursor: Any, query: Query) -> Any: ...
    
    @classmethod
    def cancel_query(cls, cursor: Any, query: Query, cancel_query_id: Any) -> bool: ...
```