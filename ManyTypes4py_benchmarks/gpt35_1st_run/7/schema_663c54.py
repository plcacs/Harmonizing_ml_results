from __future__ import annotations
from collections.abc import Iterable, Mapping
import logging
from typing import TYPE_CHECKING
from sqlalchemy import MetaData, Table
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm.attributes import InstrumentedAttribute
from ..const import SupportedDialect
from ..db_schema import DOUBLE_PRECISION_TYPE_SQL, DOUBLE_TYPE
from ..util import session_scope
if TYPE_CHECKING:
    from .. import Recorder
_LOGGER: logging.Logger = logging.getLogger(__name__)
MYSQL_ERR_INCORRECT_STRING_VALUE: int = 1366
UTF8_NAME: str = '𓆚𓃗'
PRECISE_NUMBER: float = 1.000000000000001

def _get_precision_column_types(table_object: DeclarativeBase) -> list[str]:
    ...

def validate_table_schema_supports_utf8(instance: Recorder, table_object: DeclarativeBase, columns: list[InstrumentedAttribute]) -> set[str]:
    ...

def validate_table_schema_has_correct_collation(instance: Recorder, table_object: DeclarativeBase) -> set[str]:
    ...

def _validate_table_schema_has_correct_collation(instance: Recorder, table_object: DeclarativeBase) -> set[str]:
    ...

def _validate_table_schema_supports_utf8(instance: Recorder, table_object: DeclarativeBase, columns: list[InstrumentedAttribute]) -> set[str]:
    ...

def validate_db_schema_precision(instance: Recorder, table_object: DeclarativeBase) -> set[str]:
    ...

def _validate_db_schema_precision(instance: Recorder, table_object: DeclarativeBase) -> set[str]:
    ...

def _log_schema_errors(table_object: DeclarativeBase, schema_errors: set[str]) -> None:
    ...

def _check_columns(schema_errors: set[str], stored: dict[str, float], expected: dict[str, float], columns: list[str], table_name: str, supports: str) -> None:
    ...

def correct_db_schema_utf8(instance: Recorder, table_object: DeclarativeBase, schema_errors: set[str]) -> None:
    ...

def correct_db_schema_precision(instance: Recorder, table_object: DeclarativeBase, schema_errors: set[str]) -> None:
    ...
