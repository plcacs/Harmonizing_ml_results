"""Schema migration helpers."""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
import contextlib
from dataclasses import dataclass, replace as dataclass_replace
from datetime import datetime, timedelta
import logging
from time import time
from typing import TYPE_CHECKING, Any, cast, final, Optional, Union, Dict, List, Set, Tuple, Type
from uuid import UUID
import sqlalchemy
from sqlalchemy import ForeignKeyConstraint, MetaData, Table, func, text, update
from sqlalchemy.engine import CursorResult, Engine
from sqlalchemy.exc import DatabaseError, IntegrityError, InternalError, OperationalError, ProgrammingError, SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm.session import Session
from sqlalchemy.schema import AddConstraint, CreateTable, DropConstraint
from sqlalchemy.sql.expression import true
from sqlalchemy.sql.lambdas import StatementLambdaElement
from homeassistant.core import HomeAssistant
from homeassistant.util.enum import try_parse_enum
from homeassistant.util.ulid import ulid_at_time, ulid_to_bytes
from .auto_repairs.events.schema import correct_db_schema as events_correct_db_schema, validate_db_schema as events_validate_db_schema
from .auto_repairs.states.schema import correct_db_schema as states_correct_db_schema, validate_db_schema as states_validate_db_schema
from .auto_repairs.statistics.duplicates import delete_statistics_duplicates, delete_statistics_meta_duplicates
from .auto_repairs.statistics.schema import correct_db_schema as statistics_correct_db_schema, validate_db_schema as statistics_validate_db_schema
from .const import CONTEXT_ID_AS_BINARY_SCHEMA_VERSION, EVENT_TYPE_IDS_SCHEMA_VERSION, LEGACY_STATES_EVENT_ID_INDEX_SCHEMA_VERSION, STATES_META_SCHEMA_VERSION, SupportedDialect
from .db_schema import BIG_INTEGER_SQL, CONTEXT_ID_BIN_MAX_LENGTH, DOUBLE_PRECISION_TYPE_SQL, LEGACY_STATES_ENTITY_ID_LAST_UPDATED_TS_INDEX, LEGACY_STATES_EVENT_ID_INDEX, MYSQL_COLLATE, MYSQL_DEFAULT_CHARSET, SCHEMA_VERSION, STATISTICS_TABLES, TABLE_STATES, Base, Events, EventTypes, LegacyBase, MigrationChanges, SchemaChanges, States, StatesMeta, Statistics, StatisticsMeta, StatisticsRuns, StatisticsShortTerm
from .models import process_timestamp
from .models.time import datetime_to_timestamp_or_none
from .queries import batch_cleanup_entity_ids, delete_duplicate_short_term_statistics_row, delete_duplicate_statistics_row, find_entity_ids_to_migrate, find_event_type_to_migrate, find_events_context_ids_to_migrate, find_states_context_ids_to_migrate, find_unmigrated_short_term_statistics_rows, find_unmigrated_statistics_rows, get_migration_changes, has_entity_ids_to_migrate, has_event_type_to_migrate, has_events_context_ids_to_migrate, has_states_context_ids_to_migrate, has_used_states_entity_ids, has_used_states_event_ids, migrate_single_short_term_statistics_row_to_timestamp, migrate_single_statistics_row_to_timestamp
from .statistics import cleanup_statistics_timestamp_migration, get_start_time
from .tasks import RecorderTask
from .util import database_job_retry_wrapper, database_job_retry_wrapper_method, execute_stmt_lambda_element, get_index_by_name, retryable_database_job_method, session_scope

if TYPE_CHECKING:
    from . import Recorder

LIVE_MIGRATION_MIN_SCHEMA_VERSION: int = 42
MIGRATION_NOTE_OFFLINE: str = 'Note: this may take several hours on large databases and slow machines. Home Assistant will not start until the upgrade is completed. Please be patient and do not turn off or restart Home Assistant while the upgrade is in progress!'
MIGRATION_NOTE_MINUTES: str = 'Note: this may take several minutes on large databases and slow machines. Please be patient!'
MIGRATION_NOTE_WHILE: str = 'This will take a while; please be patient!'
_EMPTY_ENTITY_ID: str = 'missing.entity_id'
_EMPTY_EVENT_TYPE: str = 'missing_event_type'
_LOGGER: logging.Logger = logging.getLogger(__name__)

@dataclass
class _ColumnTypesForDialect:
    big_int_type: str
    timestamp_type: str
    context_bin_type: str

_MYSQL_COLUMN_TYPES: _ColumnTypesForDialect = _ColumnTypesForDialect(
    big_int_type='INTEGER(20)', 
    timestamp_type=DOUBLE_PRECISION_TYPE_SQL,
    context_bin_type=f'BLOB({CONTEXT_ID_BIN_MAX_LENGTH})'
)

_POSTGRESQL_COLUMN_TYPES: _ColumnTypesForDialect = _ColumnTypesForDialect(
    big_int_type='INTEGER',
    timestamp_type=DOUBLE_PRECISION_TYPE_SQL,
    context_bin_type='BYTEA'
)

_SQLITE_COLUMN_TYPES: _ColumnTypesForDialect = _ColumnTypesForDialect(
    big_int_type='INTEGER',
    timestamp_type='FLOAT',
    context_bin_type='BLOB'
)

_COLUMN_TYPES_FOR_DIALECT: Dict[SupportedDialect, _ColumnTypesForDialect] = {
    SupportedDialect.MYSQL: _MYSQL_COLUMN_TYPES,
    SupportedDialect.POSTGRESQL: _POSTGRESQL_COLUMN_TYPES,
    SupportedDialect.SQLITE: _SQLITE_COLUMN_TYPES
}

def _unindexable_legacy_column(instance: Any, base: Any, err: Exception) -> bool:
    """Ignore index errors on char(0) columns."""
    return bool(base == LegacyBase and isinstance(err, OperationalError) and instance.engine and (instance.engine.dialect.name == SupportedDialect.MYSQL) and isinstance(err.orig, BaseException) and err.orig.args and (err.orig.args[0] == 1167))

def raise_if_exception_missing_str(ex: Exception, match_substrs: List[str]) -> None:
    """Raise if the exception and cause do not contain the match substrs."""
    lower_ex_strs = [str(ex).lower(), str(ex.__cause__).lower()]
    for str_sub in match_substrs:
        for exc_str in lower_ex_strs:
            if exc_str and str_sub in exc_str:
                return
    raise ex

def _get_initial_schema_version(session: Session) -> Optional[int]:
    """Get the schema version the database was created with."""
    res = session.query(SchemaChanges.schema_version).order_by(SchemaChanges.change_id.asc()).first()
    return getattr(res, 'schema_version', None)

def get_initial_schema_version(session_maker: Callable[[], Session]) -> Optional[int]:
    """Get the schema version the database was created with."""
    try:
        with session_scope(session=session_maker(), read_only=True) as session:
            return _get_initial_schema_version(session)
    except Exception:
        _LOGGER.exception('Error when determining DB schema version')
        return None

def _get_current_schema_version(session: Session) -> Optional[int]:
    """Get the schema version."""
    res = session.query(SchemaChanges.schema_version).order_by(SchemaChanges.change_id.desc()).first()
    return getattr(res, 'schema_version', None)

def get_current_schema_version(session_maker: Callable[[], Session]) -> Optional[int]:
    """Get the schema version."""
    try:
        with session_scope(session=session_maker(), read_only=True) as session:
            return _get_current_schema_version(session)
    except Exception:
        _LOGGER.exception('Error when determining DB schema version')
        return None

@dataclass(frozen=True, kw_only=True)
class SchemaValidationStatus:
    """Store schema validation status."""
    current_version: Optional[int]
    initial_version: Optional[int]
    non_live_data_migration_needed: bool
    migration_needed: bool
    schema_errors: Set[str]
    start_version: Optional[int]

def _schema_is_current(current_version: Optional[int]) -> bool:
    """Check if the schema is current."""
    return current_version == SCHEMA_VERSION

def validate_db_schema(hass: HomeAssistant, instance: Any, session_maker: Callable[[], Session]) -> Optional[SchemaValidationStatus]:
    """Check if the schema is valid."""
    schema_errors: Set[str] = set()
    current_version: Optional[int] = get_current_schema_version(session_maker)
    initial_version: Optional[int] = get_initial_schema_version(session_maker)
    if current_version is None or initial_version is None:
        return None
    is_current: bool = _schema_is_current(current_version)
    if is_current:
        schema_errors = _find_schema_errors(hass, instance, session_maker)
    schema_migration_needed: bool = not is_current
    _non_live_data_migration_needed: bool = non_live_data_migration_needed(
        instance, session_maker, 
        initial_schema_version=initial_version, 
        start_schema_version=current_version
    )
    return SchemaValidationStatus(
        current_version=current_version,
        initial_version=initial_version,
        non_live_data_migration_needed=_non_live_data_migration_needed,
        migration_needed=schema_migration_needed or _non_live_data_migration_needed,
        schema_errors=schema_errors,
        start_version=current_version
    )

def _find_schema_errors(hass: HomeAssistant, instance: Any, session_maker: Callable[[], Session]) -> Set[str]:
    """Find schema errors."""
    schema_errors: Set[str] = set()
    schema_errors |= statistics_validate_db_schema(instance)
    schema_errors |= states_validate_db_schema(instance)
    schema_errors |= events_validate_db_schema(instance)
    return schema_errors

def live_migration(schema_status: SchemaValidationStatus) -> bool:
    """Check if live migration is possible."""
    return schema_status.current_version >= LIVE_MIGRATION_MIN_SCHEMA_VERSION and (not schema_status.non_live_data_migration_needed)

def pre_migrate_schema(engine: Engine) -> None:
    """Prepare for migration."""
    inspector = sqlalchemy.inspect(engine)
    if inspector.has_table('statistics_meta') and (not inspector.has_table('statistics_short_term')):
        LegacyBase.metadata.create_all(engine, (LegacyBase.metadata.tables['statistics_short_term'],))

def _migrate_schema(
    instance: Any,
    hass: HomeAssistant,
    engine: Engine,
    session_maker: Callable[[], Session],
    schema_status: SchemaValidationStatus,
    end_version: int
) -> SchemaValidationStatus:
    """Check if the schema needs to be upgraded."""
    current_version: int = schema_status.current_version
    start_version: int = schema_status.start_version
    if current_version < end_version:
        _LOGGER.warning('The database is about to upgrade from schema version %s to %s%s', 
                       current_version, end_version, 
                       f'. {MIGRATION_NOTE_OFFLINE}' if current_version < LIVE_MIGRATION_MIN_SCHEMA_VERSION else '')
        schema_status = dataclass_replace(schema_status, current_version=end_version)
    
    for version in range(current_version, end_version):
        new_version = version + 1
        _LOGGER.warning('Upgrading recorder db schema to version %s', new_version)
        _apply_update(instance, hass, engine, session_maker, new_version, start_version)
        with session_scope(session=session_maker()) as session:
            session.add(SchemaChanges(schema_version=new_version))
        _LOGGER.warning('Upgrade to version %s done', new_version)
    return schema_status

def migrate_schema_non_live(
    instance: Any,
    hass: HomeAssistant,
    engine: Engine,
    session_maker: Callable[[], Session],
    schema_status: SchemaValidationStatus
) -> SchemaValidationStatus:
    """Check if the schema needs to be upgraded."""
    end_version: int = LIVE_MIGRATION_MIN_SCHEMA_VERSION
    return _migrate_schema(instance, hass, engine, session_maker, schema_status, end_version)

def migrate_schema_live(
    instance: Any,
    hass: HomeAssistant,
    engine: Engine,
    session_maker: Callable[[], Session],
    schema_status: SchemaValidationStatus
) -> SchemaValidationStatus:
    """Check if the schema needs to be upgraded."""
    schema_status = _migrate_schema(instance, hass, engine, session_maker, schema_status, SCHEMA_VERSION)
    if (schema_errors := schema_status.schema_errors):
        _LOGGER.warning('Database is about to correct DB schema errors: %s', ', '.join(sorted(schema_errors)))
        statistics_correct_db_schema(instance, schema_errors)
        states_correct_db_schema(instance, schema_errors)
        events_correct_db_schema(instance, schema_errors)
    return schema_status

def _get_migration_changes(session: Session) -> Dict[str, int]:
    """Return migration changes as a dict."""
    migration_changes = {row[0]: row[1] for row in execute_stmt_lambda_element(session, get_migration_changes())}
    return migration_changes

def non_live_data_migration_needed(
    instance: Any,
    session_maker: Callable[[], Session],
    *,
    initial_schema_version: int,
    start_schema_version: int
) -> bool:
    """Return True if non-live data migration is needed."""
    migration_needed: bool = False
    with session_scope(session=session_maker()) as session:
        migration_changes = _get_migration_changes(session)
        for migrator_cls in NON_LIVE_DATA_MIGRATORS:
            migrator = migrator_cls(
                initial_schema_version=initial_schema_version,
                start_schema_version=start_schema_version,
                migration_changes=migration_changes
            )
            migration_needed |= migrator.needs_migrate(instance, session)
    return migration_needed

def migrate_data_non_live(
    instance: Any,
    session_maker: Callable[[], Session],
    schema_status: SchemaValidationStatus
) -> None:
    """Do non-live data migration."""
    with session_scope(session=session_maker()) as session:
        migration_changes = _get_migration_changes(session)
    for migrator_cls in NON_LIVE_DATA_MIGRATORS:
        migrator = migrator_cls(
            initial_schema_version=schema_status.initial_version,
            start_schema_version=schema_status.start_version,
            migration_changes=migration_changes
        )
        migrator.migrate_all(instance, session_maker)

def migrate_data_live(
    instance: Any,
    session_maker: Callable[[], Session],
    schema_status: SchemaValidationStatus
) -> None:
    """Queue live schema migration tasks."""
    with session_scope(session=session_maker()) as session:
        migration_changes = _get_migration_changes(session)
        for migrator_cls in LIVE_DATA_MIGRATORS:
            migrator = migrator_cls(
                initial_schema_version=schema_status.initial_version,
                start_schema_version=schema_status.start_version,
                migration_changes=migration_changes
            )
            migrator.queue_migration(instance, session)

def _create_index(
    instance: Any,
    session_maker: Callable[[], Session],
    table_name: str,
    index_name: str,
    *,
    base: Type[DeclarativeBase] = Base
) -> None:
    """Create an index for the specified table."""
    table = Table(table_name, base.metadata)
    _LOGGER.debug('Looking up index %s for table %s', index_name, table_name)
    index_list = [idx for idx in table.indexes if idx.name == index_name]
    if not index_list:
        _LOGGER.debug('The index %s no longer exists', index_name)
        return
    index = index_list[0]
    _LOGGER.debug('Creating %s index', index_name)
    _LOGGER.warning('Adding index `%s` to table `%s`. %s', index_name, table_name, MIGRATION_NOTE_MINUTES)
    with session_scope(session=session_maker()) as session:
        try:
            connection = session.connection()
            index.create(connection)
        except (InternalError, OperationalError, ProgrammingError) as err:
            if _unindexable_legacy_column(instance, base, err):
                _LOGGER.debug("Can't add legacy index %s to column %s, continuing", index_name, table_name)
                return
            raise_if_exception_missing_str(err, ['already exists', 'duplicate'])
            _LOGGER.warning('Index %s already exists on %s, continuing', index_name, table_name)
            return
    _LOGGER.warning('Finished adding index `%s` to table `%s`', index_name, table_name)

def _execute_or_collect_error(
    session_maker: Callable[[], Session],
    query: str,
    errors: List[str]
) -> bool:
    """Execute a query or collect an error."""
    with session_scope(session=session_maker()) as session:
        try:
            session.connection().execute(text(query))
        except SQLAlchemyError as err:
            errors.append(str(err))
            return False
        return True

def _drop_index(
    session_maker: Callable[[], Session],
    table_name: str,
    index_name: str,
    quiet: Optional[bool] = None
) -> None:
    """Drop an index from a specified table."""
    _LOGGER.warning('Dropping index `%s` from table `%s`. %s', index_name, table_name, MIG