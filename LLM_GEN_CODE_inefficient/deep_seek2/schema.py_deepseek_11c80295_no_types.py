"""Schema repairs."""
from __future__ import annotations
from collections.abc import Iterable, Mapping
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Set, Tuple, Type
from sqlalchemy import MetaData, Table
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, Session
from sqlalchemy.orm.attributes import InstrumentedAttribute
from ..const import SupportedDialect
from ..db_schema import DOUBLE_PRECISION_TYPE_SQL, DOUBLE_TYPE
from ..util import session_scope
if TYPE_CHECKING:
    from .. import Recorder
_LOGGER = logging.getLogger(__name__)
MYSQL_ERR_INCORRECT_STRING_VALUE: int = 1366
UTF8_NAME: str = '𓆚𓃗'
PRECISE_NUMBER: float = 1.000000000000001

def _get_precision_column_types(table_object):
    """Get the column names for the columns that need to be checked for precision."""
    return [column.key for column in table_object.__table__.columns if column.type is DOUBLE_TYPE]

def validate_table_schema_supports_utf8(instance, table_object, columns):
    """Do some basic checks for common schema errors caused by manual migration."""
    schema_errors: Set[str] = set()
    if instance.dialect_name != SupportedDialect.MYSQL:
        return schema_errors
    try:
        schema_errors = _validate_table_schema_supports_utf8(instance, table_object, columns)
    except Exception:
        _LOGGER.exception('Error when validating DB schema')
    _log_schema_errors(table_object, schema_errors)
    return schema_errors

def validate_table_schema_has_correct_collation(instance, table_object):
    """Verify the table has the correct collation."""
    schema_errors: Set[str] = set()
    if instance.dialect_name != SupportedDialect.MYSQL:
        return schema_errors
    try:
        schema_errors = _validate_table_schema_has_correct_collation(instance, table_object)
    except Exception:
        _LOGGER.exception('Error when validating DB schema')
    _log_schema_errors(table_object, schema_errors)
    return schema_errors

def _validate_table_schema_has_correct_collation(instance, table_object):
    """Ensure the table has the correct collation to avoid union errors with mixed collations."""
    schema_errors: Set[str] = set()
    with session_scope(session=instance.get_session(), read_only=True) as session:
        table: str = table_object.__tablename__
        metadata_obj: MetaData = MetaData()
        reflected_table: Table = Table(table, metadata_obj, autoload_with=instance.engine)
        connection: Any = session.connection()
        dialect_kwargs: Dict[str, Any] = reflected_table.dialect_kwargs
        collate: str = dialect_kwargs.get('mysql_collate') or dialect_kwargs.get('mariadb_collate') or connection.dialect._fetch_setting(connection, 'collation_server')
        if collate and collate != 'utf8mb4_unicode_ci':
            _LOGGER.debug('Database %s collation is not utf8mb4_unicode_ci', table)
            schema_errors.add(f'{table}.utf8mb4_unicode_ci')
    return schema_errors

def _validate_table_schema_supports_utf8(instance, table_object, columns):
    """Do some basic checks for common schema errors caused by manual migration."""
    schema_errors: Set[str] = set()
    with session_scope(session=instance.get_session(), read_only=True) as session:
        db_object: Any = table_object(**{column.key: UTF8_NAME for column in columns})
        table: str = table_object.__tablename__
        session.add(db_object)
        try:
            session.flush()
        except OperationalError as err:
            if err.orig and err.orig.args[0] == MYSQL_ERR_INCORRECT_STRING_VALUE:
                _LOGGER.debug('Database %s statistics_meta does not support 4-byte UTF-8', table)
                schema_errors.add(f'{table}.4-byte UTF-8')
                return schema_errors
            raise
        finally:
            session.rollback()
    return schema_errors

def validate_db_schema_precision(instance, table_object):
    """Do some basic checks for common schema errors caused by manual migration."""
    schema_errors: Set[str] = set()
    if instance.dialect_name not in (SupportedDialect.MYSQL, SupportedDialect.POSTGRESQL):
        return schema_errors
    try:
        schema_errors = _validate_db_schema_precision(instance, table_object)
    except Exception:
        _LOGGER.exception('Error when validating DB schema')
    _log_schema_errors(table_object, schema_errors)
    return schema_errors

def _validate_db_schema_precision(instance, table_object):
    """Do some basic checks for common schema errors caused by manual migration."""
    schema_errors: Set[str] = set()
    columns: List[str] = _get_precision_column_types(table_object)
    with session_scope(session=instance.get_session(), read_only=True) as session:
        db_object: Any = table_object(**{column: PRECISE_NUMBER for column in columns})
        table: str = table_object.__tablename__
        try:
            session.add(db_object)
            session.flush()
            session.refresh(db_object)
            _check_columns(schema_errors=schema_errors, stored={column: getattr(db_object, column) for column in columns}, expected={column: PRECISE_NUMBER for column in columns}, columns=columns, table_name=table, supports='double precision')
        finally:
            session.rollback()
    return schema_errors

def _log_schema_errors(table_object, schema_errors):
    """Log schema errors."""
    if not schema_errors:
        return
    _LOGGER.debug('Detected %s schema errors: %s', table_object.__tablename__, ', '.join(sorted(schema_errors)))

def _check_columns(schema_errors, stored, expected, columns, table_name, supports):
    """Check that the columns in the table support the given feature.

    Errors are logged and added to the schema_errors set.
    """
    for column in columns:
        if stored[column] == expected[column]:
            continue
        schema_errors.add(f'{table_name}.{supports}')
        _LOGGER.error('Column %s in database table %s does not support %s (stored=%s != expected=%s)', column, table_name, supports, stored[column], expected[column])

def correct_db_schema_utf8(instance, table_object, schema_errors):
    """Correct utf8 issues detected by validate_db_schema."""
    table_name: str = table_object.__tablename__
    if f'{table_name}.4-byte UTF-8' in schema_errors or f'{table_name}.utf8mb4_unicode_ci' in schema_errors:
        from ..migration import _correct_table_character_set_and_collation
        _correct_table_character_set_and_collation(table_name, instance.get_session)

def correct_db_schema_precision(instance, table_object, schema_errors):
    """Correct precision issues detected by validate_db_schema."""
    table_name: str = table_object.__tablename__
    if f'{table_name}.double precision' in schema_errors:
        from ..migration import _modify_columns
        precision_columns: List[str] = _get_precision_column_types(table_object)
        session_maker: Callable[[], Session] = instance.get_session
        engine: Any = instance.engine
        assert engine is not None, 'Engine should be set'
        _modify_columns(session_maker, engine, table_name, [f'{column} {DOUBLE_PRECISION_TYPE_SQL}' for column in precision_columns])