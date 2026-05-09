from __future__ import annotations
from datetime import datetime
from re import Pattern
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
from flask_babel import gettext as __
from sqlalchemy.engine.url import URL
from sqlalchemy.types import Date, DateTime
from superset.constants import TimeGrain
from superset.db_engine_specs.base import BaseEngineSpec
from superset.models.sql_lab import Query
from superset.sql.parse import SQLScript
from superset.errors import SupersetError, SupersetErrorType
from superset.exceptions import SupersetException, SupersetSecurityException
from superset.models.core import Database
from superset.utils.core import GenericDataType

CONNECTION_INVALID_USERNAME_REGEX: Pattern[str] = ...
CONNECTION_INVALID_PASSWORD_REGEX: Pattern[str] = ...
CONNECTION_INVALID_PASSWORD_NEEDED_REGEX: Pattern[str] = ...
CONNECTION_INVALID_HOSTNAME_REGEX: Pattern[str] = ...
CONNECTION_PORT_CLOSED_REGEX: Pattern[str] = ...
CONNECTION_HOST_DOWN_REGEX: Pattern[str] = ...
CONNECTION_UNKNOWN_DATABASE_REGEX: Pattern[str] = ...
COLUMN_DOES_NOT_EXIST_REGEX: Pattern[str] = ...
SYNTAX_ERROR_REGEX: Pattern[str] = ...

def parse_options(connect_args: Dict[str, Any]) -> Dict[str, str]:
    ...

class PostgresBaseEngineSpec(BaseEngineSpec):
    engine: str
    engine_name: str
    _time_grain_expressions: Dict[Optional[TimeGrain], str]
    custom_errors: Dict[Pattern[str], Tuple[str, SupersetErrorType, Dict[str, List[str]]]]

    @classmethod
    def fetch_data(cls, cursor: Any, limit: Optional[int]) -> List[Any]:
        ...

    @classmethod
    def epoch_to_dttm(cls) -> str:
        ...

    @classmethod
    def convert_dttm(cls, target_type: str, dttm: datetime, db_extra: Optional[Dict[str, Any]] = None) -> Optional[str]:
        ...

class PostgresEngineSpec(BasicParametersMixin, PostgresBaseEngineSpec):
    engine: str
    engine_aliases: Set[str]
    supports_dynamic_schema: bool
    supports_catalog: bool
    supports_dynamic_catalog: bool
    default_driver: str
    sqlalchemy_uri_placeholder: str
    encryption_parameters: Dict[str, str]
    max_column_name_length: int
    try_remove_schema_from_table_name: bool
    column_type_mappings: Tuple[Tuple[Pattern[str], Any, GenericDataType], ...]

    @classmethod
    def get_schema_from_engine_params(cls, sqlalchemy_uri: URL, connect_args: Dict[str, Any]) -> Optional[str]:
        ...

    @classmethod
    def get_default_schema_for_query(cls, database: Database, query: Query) -> str:
        ...

    @classmethod
    def adjust_engine_params(cls, uri: URL, connect_args: Dict[str, Any], catalog: Optional[str] = None, schema: Optional[str] = None) -> Tuple[URL, Dict[str, Any]]:
        ...

    @classmethod
    def get_default_catalog(cls, database: Database) -> str:
        ...

    @classmethod
    def get_prequeries(cls, database: Database, catalog: Optional[str] = None, schema: Optional[str] = None) -> List[str]:
        ...

    @classmethod
    def get_allow_cost_estimate(cls, extra: Dict[str, Any]) -> bool:
        ...

    @classmethod
    def estimate_statement_cost(cls, database: Database, statement: str, cursor: Any) -> Dict[str, float]:
        ...

    @classmethod
    def query_cost_formatter(cls, raw_cost: Any) -> List[Dict[str, str]]:
        ...

    @classmethod
    def get_catalog_names(cls, database: Database, inspector: Any) -> Set[str]:
        ...

    @classmethod
    def get_table_names(cls, database: Database, inspector: Any, schema: str) -> Set[str]:
        ...

    @staticmethod
    def get_extra_params(database: Database) -> Dict[str, Any]:
        ...

    @classmethod
    def get_datatype(cls, type_code: int) -> Optional[str]:
        ...

    @classmethod
    def get_cancel_query_id(cls, cursor: Any, query: Any) -> int:
        ...

    @classmethod
    def cancel_query(cls, cursor: Any, query: Any, cancel_query_id: int) -> bool:
        ...