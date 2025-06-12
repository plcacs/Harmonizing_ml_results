"""Schema migration helpers."""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
import contextlib
from dataclasses import dataclass, replace as dataclass_replace
from datetime import timedelta
import logging
from time import time
from typing import TYPE_CHECKING, Any, cast, final
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
LIVE_MIGRATION_MIN_SCHEMA_VERSION = 42
MIGRATION_NOTE_OFFLINE = 'Note: this may take several hours on large databases and slow machines. Home Assistant will not start until the upgrade is completed. Please be patient and do not turn off or restart Home Assistant while the upgrade is in progress!'
MIGRATION_NOTE_MINUTES = 'Note: this may take several minutes on large databases and slow machines. Please be patient!'
MIGRATION_NOTE_WHILE = 'This will take a while; please be patient!'
_EMPTY_ENTITY_ID = 'missing.entity_id'
_EMPTY_EVENT_TYPE = 'missing_event_type'
_LOGGER = logging.getLogger(__name__)

@dataclass
class _ColumnTypesForDialect:
    pass
_MYSQL_COLUMN_TYPES = _ColumnTypesForDialect(big_int_type='INTEGER(20)', timestamp_type=DOUBLE_PRECISION_TYPE_SQL, context_bin_type=f'BLOB({CONTEXT_ID_BIN_MAX_LENGTH})')
_POSTGRESQL_COLUMN_TYPES = _ColumnTypesForDialect(big_int_type='INTEGER', timestamp_type=DOUBLE_PRECISION_TYPE_SQL, context_bin_type='BYTEA')
_SQLITE_COLUMN_TYPES = _ColumnTypesForDialect(big_int_type='INTEGER', timestamp_type='FLOAT', context_bin_type='BLOB')
_COLUMN_TYPES_FOR_DIALECT = {SupportedDialect.MYSQL: _MYSQL_COLUMN_TYPES, SupportedDialect.POSTGRESQL: _POSTGRESQL_COLUMN_TYPES, SupportedDialect.SQLITE: _SQLITE_COLUMN_TYPES}

def _unindexable_legacy_column(instance, base, err):
    """Ignore index errors on char(0) columns."""
    return bool(base == LegacyBase and isinstance(err, OperationalError) and instance.engine and (instance.engine.dialect.name == SupportedDialect.MYSQL) and isinstance(err.orig, BaseException) and err.orig.args and (err.orig.args[0] == 1167))

def raise_if_exception_missing_str(ex, match_substrs):
    """Raise if the exception and cause do not contain the match substrs."""
    lower_ex_strs = [str(ex).lower(), str(ex.__cause__).lower()]
    for str_sub in match_substrs:
        for exc_str in lower_ex_strs:
            if exc_str and str_sub in exc_str:
                return
    raise ex

def _get_initial_schema_version(session):
    """Get the schema version the database was created with."""
    res = session.query(SchemaChanges.schema_version).order_by(SchemaChanges.change_id.asc()).first()
    return getattr(res, 'schema_version', None)

def get_initial_schema_version(session_maker):
    """Get the schema version the database was created with."""
    try:
        with session_scope(session=session_maker(), read_only=True) as session:
            return _get_initial_schema_version(session)
    except Exception:
        _LOGGER.exception('Error when determining DB schema version')
        return None

def _get_current_schema_version(session):
    """Get the schema version."""
    res = session.query(SchemaChanges.schema_version).order_by(SchemaChanges.change_id.desc()).first()
    return getattr(res, 'schema_version', None)

def get_current_schema_version(session_maker):
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

def _schema_is_current(current_version):
    """Check if the schema is current."""
    return current_version == SCHEMA_VERSION

def validate_db_schema(hass, instance, session_maker):
    """Check if the schema is valid.

    This checks that the schema is the current version as well as for some common schema
    errors caused by manual migration between database engines, for example importing an
    SQLite database to MariaDB.
    """
    schema_errors = set()
    current_version = get_current_schema_version(session_maker)
    initial_version = get_initial_schema_version(session_maker)
    if current_version is None or initial_version is None:
        return None
    if (is_current := _schema_is_current(current_version)):
        schema_errors = _find_schema_errors(hass, instance, session_maker)
    schema_migration_needed = not is_current
    _non_live_data_migration_needed = non_live_data_migration_needed(instance, session_maker, initial_schema_version=initial_version, start_schema_version=current_version)
    return SchemaValidationStatus(current_version=current_version, initial_version=initial_version, non_live_data_migration_needed=_non_live_data_migration_needed, migration_needed=schema_migration_needed or _non_live_data_migration_needed, schema_errors=schema_errors, start_version=current_version)

def _find_schema_errors(hass, instance, session_maker):
    """Find schema errors."""
    schema_errors = set()
    schema_errors |= statistics_validate_db_schema(instance)
    schema_errors |= states_validate_db_schema(instance)
    schema_errors |= events_validate_db_schema(instance)
    return schema_errors

def live_migration(schema_status):
    """Check if live migration is possible."""
    return schema_status.current_version >= LIVE_MIGRATION_MIN_SCHEMA_VERSION and (not schema_status.non_live_data_migration_needed)

def pre_migrate_schema(engine):
    """Prepare for migration.

    This function is called before calling Base.metadata.create_all.
    """
    inspector = sqlalchemy.inspect(engine)
    if inspector.has_table('statistics_meta') and (not inspector.has_table('statistics_short_term')):
        LegacyBase.metadata.create_all(engine, (LegacyBase.metadata.tables['statistics_short_term'],))

def _migrate_schema(instance, hass, engine, session_maker, schema_status, end_version):
    """Check if the schema needs to be upgraded."""
    current_version = schema_status.current_version
    start_version = schema_status.start_version
    if current_version < end_version:
        _LOGGER.warning('The database is about to upgrade from schema version %s to %s%s', current_version, end_version, f'. {MIGRATION_NOTE_OFFLINE}' if current_version < LIVE_MIGRATION_MIN_SCHEMA_VERSION else '')
        schema_status = dataclass_replace(schema_status, current_version=end_version)
    for version in range(current_version, end_version):
        new_version = version + 1
        _LOGGER.warning('Upgrading recorder db schema to version %s', new_version)
        _apply_update(instance, hass, engine, session_maker, new_version, start_version)
        with session_scope(session=session_maker()) as session:
            session.add(SchemaChanges(schema_version=new_version))
        _LOGGER.warning('Upgrade to version %s done', new_version)
    return schema_status

def migrate_schema_non_live(instance, hass, engine, session_maker, schema_status):
    """Check if the schema needs to be upgraded."""
    end_version = LIVE_MIGRATION_MIN_SCHEMA_VERSION
    return _migrate_schema(instance, hass, engine, session_maker, schema_status, end_version)

def migrate_schema_live(instance, hass, engine, session_maker, schema_status):
    """Check if the schema needs to be upgraded."""
    end_version = SCHEMA_VERSION
    schema_status = _migrate_schema(instance, hass, engine, session_maker, schema_status, end_version)
    if (schema_errors := schema_status.schema_errors):
        _LOGGER.warning('Database is about to correct DB schema errors: %s', ', '.join(sorted(schema_errors)))
        statistics_correct_db_schema(instance, schema_errors)
        states_correct_db_schema(instance, schema_errors)
        events_correct_db_schema(instance, schema_errors)
    return schema_status

def _get_migration_changes(session):
    """Return migration changes as a dict."""
    migration_changes = {row[0]: row[1] for row in execute_stmt_lambda_element(session, get_migration_changes())}
    return migration_changes

def non_live_data_migration_needed(instance, session_maker, *, initial_schema_version, start_schema_version):
    """Return True if non-live data migration is needed.

    :param initial_schema_version: The schema version the database was created with.
    :param start_schema_version: The schema version when starting the migration.

    This must only be called if database schema is current.
    """
    migration_needed = False
    with session_scope(session=session_maker()) as session:
        migration_changes = _get_migration_changes(session)
        for migrator_cls in NON_LIVE_DATA_MIGRATORS:
            migrator = migrator_cls(initial_schema_version=initial_schema_version, start_schema_version=start_schema_version, migration_changes=migration_changes)
            migration_needed |= migrator.needs_migrate(instance, session)
    return migration_needed

def migrate_data_non_live(instance, session_maker, schema_status):
    """Do non-live data migration.

    This must be called after non-live schema migration is completed.
    """
    with session_scope(session=session_maker()) as session:
        migration_changes = _get_migration_changes(session)
    for migrator_cls in NON_LIVE_DATA_MIGRATORS:
        migrator = migrator_cls(initial_schema_version=schema_status.initial_version, start_schema_version=schema_status.start_version, migration_changes=migration_changes)
        migrator.migrate_all(instance, session_maker)

def migrate_data_live(instance, session_maker, schema_status):
    """Queue live schema migration tasks.

    This must be called after live schema migration is completed.
    """
    with session_scope(session=session_maker()) as session:
        migration_changes = _get_migration_changes(session)
        for migrator_cls in LIVE_DATA_MIGRATORS:
            migrator = migrator_cls(initial_schema_version=schema_status.initial_version, start_schema_version=schema_status.start_version, migration_changes=migration_changes)
            migrator.queue_migration(instance, session)

def _create_index(instance, session_maker, table_name, index_name, *, base=Base):
    """Create an index for the specified table.

    The index name should match the name given for the index
    within the table definition described in the models
    """
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

def _execute_or_collect_error(session_maker, query, errors):
    """Execute a query or collect an error."""
    with session_scope(session=session_maker()) as session:
        try:
            session.connection().execute(text(query))
        except SQLAlchemyError as err:
            errors.append(str(err))
            return False
        return True

def _drop_index(session_maker, table_name, index_name, quiet=None):
    """Drop an index from a specified table.

    There is no universal way to do something like `DROP INDEX IF EXISTS`
    so we will simply execute the DROP command and ignore any exceptions

    WARNING: Due to some engines (MySQL at least) being unable to use bind
    parameters in a DROP INDEX statement (at least via SQLAlchemy), the query
    string here is generated from the method parameters without sanitizing.
    DO NOT USE THIS FUNCTION IN ANY OPERATION THAT TAKES USER INPUT.
    """
    _LOGGER.warning('Dropping index `%s` from table `%s`. %s', index_name, table_name, MIGRATION_NOTE_MINUTES)
    index_to_drop = None
    with session_scope(session=session_maker()) as session:
        index_to_drop = get_index_by_name(session, table_name, index_name)
    if index_to_drop is None:
        _LOGGER.warning('The index `%s` on table `%s` no longer exists', index_name, table_name)
        return
    errors = []
    for query in (f'DROP INDEX {index_name}', f'DROP INDEX {table_name}.{index_name}', f'DROP INDEX {index_name} ON {table_name}', f'DROP INDEX {index_to_drop}'):
        if _execute_or_collect_error(session_maker, query, errors):
            _LOGGER.warning('Finished dropping index `%s` from table `%s`', index_name, table_name)
            return
    if not quiet:
        _LOGGER.warning('Failed to drop index `%s` from table `%s`. Schema Migration will continue; this is not a critical operation: %s', index_name, table_name, errors)

def _add_columns(session_maker, table_name, columns_def):
    """Add columns to a table."""
    _LOGGER.warning('Adding columns %s to table %s. %s', ', '.join((column.split(' ')[0] for column in columns_def)), table_name, MIGRATION_NOTE_MINUTES)
    columns_def = [f'ADD {col_def}' for col_def in columns_def]
    with session_scope(session=session_maker()) as session:
        try:
            connection = session.connection()
            connection.execute(text(f'ALTER TABLE {table_name} {', '.join(columns_def)}'))
        except (InternalError, OperationalError, ProgrammingError):
            _LOGGER.info('Unable to use quick column add. Adding 1 by 1')
        else:
            return
    for column_def in columns_def:
        with session_scope(session=session_maker()) as session:
            try:
                connection = session.connection()
                connection.execute(text(f'ALTER TABLE {table_name} {column_def}'))
            except (InternalError, OperationalError, ProgrammingError) as err:
                raise_if_exception_missing_str(err, ['already exists', 'duplicate'])
                _LOGGER.warning('Column %s already exists on %s, continuing', column_def.split(' ')[1], table_name)

def _modify_columns(session_maker, engine, table_name, columns_def):
    """Modify columns in a table."""
    if engine.dialect.name == SupportedDialect.SQLITE:
        _LOGGER.debug('Skipping to modify columns %s in table %s; Modifying column length in SQLite is unnecessary, it does not impose any length restrictions', ', '.join((column.split(' ')[0] for column in columns_def)), table_name)
        return
    _LOGGER.warning('Modifying columns %s in table %s. %s', ', '.join((column.split(' ')[0] for column in columns_def)), table_name, MIGRATION_NOTE_MINUTES)
    if engine.dialect.name == SupportedDialect.POSTGRESQL:
        columns_def = [f'ALTER {column} TYPE {type_}' for column, type_ in (col_def.split(' ', 1) for col_def in columns_def)]
    elif engine.dialect.name == 'mssql':
        columns_def = [f'ALTER COLUMN {col_def}' for col_def in columns_def]
    else:
        columns_def = [f'MODIFY {col_def}' for col_def in columns_def]
    with session_scope(session=session_maker()) as session:
        try:
            connection = session.connection()
            connection.execute(text(f'ALTER TABLE {table_name} {', '.join(columns_def)}'))
        except (InternalError, OperationalError):
            _LOGGER.info('Unable to use quick column modify. Modifying 1 by 1')
        else:
            return
    for column_def in columns_def:
        with session_scope(session=session_maker()) as session:
            try:
                connection = session.connection()
                connection.execute(text(f'ALTER TABLE {table_name} {column_def}'))
            except (InternalError, OperationalError):
                _LOGGER.exception('Could not modify column %s in table %s', column_def, table_name)
                raise

def _update_states_table_with_foreign_key_options(session_maker, engine):
    """Add the options to foreign key constraints.

    This is not supported for SQLite because it does not support
    dropping constraints.
    """
    if engine.dialect.name not in (SupportedDialect.MYSQL, SupportedDialect.POSTGRESQL):
        raise RuntimeError(f'_update_states_table_with_foreign_key_options not supported for {engine.dialect.name}')
    inspector = sqlalchemy.inspect(engine)
    tmp_states_table = Table(TABLE_STATES, MetaData())
    alters = [{'old_fk': ForeignKeyConstraint((), (), name=foreign_key['name'], table=tmp_states_table), 'columns': foreign_key['constrained_columns']} for foreign_key in inspector.get_foreign_keys(TABLE_STATES) if foreign_key['name'] and (not foreign_key.get('options') or foreign_key.get('options', {}).get('ondelete') is None)]
    if not alters:
        return
    states_key_constraints = Base.metadata.tables[TABLE_STATES].foreign_key_constraints
    for alter in alters:
        with session_scope(session=session_maker()) as session:
            try:
                connection = session.connection()
                connection.execute(DropConstraint(alter['old_fk']))
                for fkc in states_key_constraints:
                    if fkc.column_keys == alter['columns']:
                        create_rule = fkc._create_rule
                        add_constraint = AddConstraint(fkc)
                        fkc._create_rule = create_rule
                        connection.execute(add_constraint)
            except (InternalError, OperationalError):
                _LOGGER.exception('Could not update foreign options in %s table', TABLE_STATES)
                raise

def _drop_foreign_key_constraints(session_maker, engine, table, column):
    """Drop foreign key constraints for a table on specific columns.

    This is not supported for SQLite because it does not support
    dropping constraints.
    """
    if engine.dialect.name not in (SupportedDialect.MYSQL, SupportedDialect.POSTGRESQL):
        raise RuntimeError(f'_drop_foreign_key_constraints not supported for {engine.dialect.name}')
    inspector = sqlalchemy.inspect(engine)
    tmp_table = Table(table, MetaData())
    drops = [ForeignKeyConstraint((), (), name=foreign_key['name'], table=tmp_table) for foreign_key in inspector.get_foreign_keys(table) if foreign_key['name'] and foreign_key['constrained_columns'] == [column]]
    for drop in drops:
        with session_scope(session=session_maker()) as session:
            try:
                connection = session.connection()
                connection.execute(DropConstraint(drop))
            except (InternalError, OperationalError):
                _LOGGER.exception('Could not drop foreign constraints in %s table on %s', TABLE_STATES, column)
                raise

def _restore_foreign_key_constraints(session_maker, engine, foreign_columns):
    """Restore foreign key constraints."""
    for table, column, foreign_table, foreign_column in foreign_columns:
        constraints = Base.metadata.tables[table].foreign_key_constraints
        for constraint in constraints:
            if constraint.column_keys == [column]:
                break
        else:
            _LOGGER.info('Did not find a matching constraint for %s.%s', table, column)
            continue
        inspector = sqlalchemy.inspect(engine)
        if any((foreign_key['name'] and foreign_key['constrained_columns'] == [column] for foreign_key in inspector.get_foreign_keys(table))):
            _LOGGER.info('The database already has a matching constraint for %s.%s', table, column)
            continue
        if TYPE_CHECKING:
            assert foreign_table is not None
            assert foreign_column is not None
        create_rule = constraint._create_rule
        add_constraint = AddConstraint(constraint)
        constraint._create_rule = create_rule
        try:
            _add_constraint(session_maker, add_constraint, table, column)
        except IntegrityError:
            _LOGGER.exception('Could not update foreign options in %s table, will delete violations and try again', table)
            _delete_foreign_key_violations(session_maker, engine, table, column, foreign_table, foreign_column)
            _add_constraint(session_maker, add_constraint, table, column)

def _add_constraint(session_maker, add_constraint, table, column):
    """Add a foreign key constraint."""
    _LOGGER.warning('Adding foreign key constraint to %s.%s. Note: this can take several minutes on large databases and slow machines. Please be patient!', table, column)
    with session_scope(session=session_maker()) as session:
        try:
            connection = session.connection()
            connection.execute(add_constraint)
        except (InternalError, OperationalError):
            _LOGGER.exception('Could not update foreign options in %s table', table)
            raise

def _delete_foreign_key_violations(session_maker, engine, table, column, foreign_table, foreign_column):
    """Remove rows which violate the constraints."""
    if engine.dialect.name not in (SupportedDialect.MYSQL, SupportedDialect.POSTGRESQL):
        raise RuntimeError(f'_delete_foreign_key_violations not supported for {engine.dialect.name}')
    _LOGGER.warning('Rows in table %s where %s references non existing %s.%s will be %s. Note: this can take several minutes on large databases and slow machines. Please be patient!', table, column, foreign_table, foreign_column, 'set to NULL' if table == foreign_table else 'deleted')
    result = None
    if table == foreign_table:
        if engine.dialect.name == SupportedDialect.MYSQL:
            while result is None or result.rowcount > 0:
                with session_scope(session=session_maker()) as session:
                    result = session.connection().execute(text(f'UPDATE {table} as t1 SET {column} = NULL WHERE (t1.{column} IS NOT NULL AND NOT EXISTS (SELECT 1 FROM (SELECT {foreign_column} from {foreign_table}) AS t2 WHERE t2.{foreign_column} = t1.{column})) LIMIT 100000;'))
        elif engine.dialect.name == SupportedDialect.POSTGRESQL:
            while result is None or result.rowcount > 0:
                with session_scope(session=session_maker()) as session:
                    result = session.connection().execute(text(f'UPDATE {table} SET {column} = NULL WHERE {column} in (SELECT {column} from {table} as t1 WHERE (t1.{column} IS NOT NULL AND NOT EXISTS (SELECT 1 FROM {foreign_table} AS t2 WHERE t2.{foreign_column} = t1.{column})) LIMIT 100000);'))
        return
    if engine.dialect.name == SupportedDialect.MYSQL:
        while result is None or result.rowcount > 0:
            with session_scope(session=session_maker()) as session:
                result = session.connection().execute(text(f'DELETE FROM {table} WHERE ({table}.{column} IS NOT NULL AND NOT EXISTS (SELECT 1 FROM {foreign_table} AS t2 WHERE t2.{foreign_column} = {table}.{column})) LIMIT 100000;'))
    elif engine.dialect.name == SupportedDialect.POSTGRESQL:
        while result is None or result.rowcount > 0:
            with session_scope(session=session_maker()) as session:
                result = session.connection().execute(text(f'DELETE FROM {table} WHERE {column} in (SELECT {column} from {table} as t1 WHERE (t1.{column} IS NOT NULL AND NOT EXISTS (SELECT 1 FROM {foreign_table} AS t2 WHERE t2.{foreign_column} = t1.{column})) LIMIT 100000);'))

@database_job_retry_wrapper('Apply migration update', 10)
def _apply_update(instance, hass, engine, session_maker, new_version, old_version):
    """Perform operations to bring schema up to date."""
    migrator_cls = _SchemaVersionMigrator.get_migrator(new_version)
    migrator_cls(instance, hass, engine, session_maker, old_version).apply_update()

class _SchemaVersionMigrator(ABC):
    """Perform operations to bring schema up to date."""
    __migrators = {}

    def __init_subclass__(cls, target_version, **kwargs):
        """Post initialisation processing."""
        super().__init_subclass__(**kwargs)
        if target_version in _SchemaVersionMigrator.__migrators:
            raise ValueError('Duplicated version')
        _SchemaVersionMigrator.__migrators[target_version] = cls

    def __init__(self, instance, hass, engine, session_maker, old_version):
        """Initialize."""
        self.instance = instance
        self.hass = hass
        self.engine = engine
        self.session_maker = session_maker
        self.old_version = old_version
        assert engine.dialect.name is not None, 'Dialect name must be set'
        dialect = try_parse_enum(SupportedDialect, engine.dialect.name)
        self.column_types = _COLUMN_TYPES_FOR_DIALECT.get(dialect, _SQLITE_COLUMN_TYPES)

    @classmethod
    def get_migrator(cls, target_version):
        """Return a migrator for a specific schema version."""
        try:
            return cls.__migrators[target_version]
        except KeyError as err:
            raise ValueError(f'No migrator for schema version {target_version}') from err

    @final
    def apply_update(self):
        """Perform operations to bring schema up to date."""
        self._apply_update()

    @abstractmethod
    def _apply_update(self):
        """Version specific update method."""

class _SchemaVersion1Migrator(_SchemaVersionMigrator, target_version=1):

    def _apply_update(self):
        """Version specific update method."""

class _SchemaVersion2Migrator(_SchemaVersionMigrator, target_version=2):

    def _apply_update(self):
        """Version specific update method."""
        _create_index(self.instance, self.session_maker, 'recorder_runs', 'ix_recorder_runs_start_end')

class _SchemaVersion3Migrator(_SchemaVersionMigrator, target_version=3):

    def _apply_update(self):
        """Version specific update method."""

class _SchemaVersion4Migrator(_SchemaVersionMigrator, target_version=4):

    def _apply_update(self):
        """Version specific update method."""
        if self.old_version == 3:
            _drop_index(self.session_maker, 'states', 'ix_states_created_domain')
        if self.old_version == 2:
            _drop_index(self.session_maker, 'states', 'ix_states_entity_id_created')
        _drop_index(self.session_maker, 'states', 'states__state_changes')
        _drop_index(self.session_maker, 'states', 'states__significant_changes')
        _drop_index(self.session_maker, 'states', 'ix_states_entity_id_created')

class _SchemaVersion5Migrator(_SchemaVersionMigrator, target_version=5):

    def _apply_update(self):
        """Version specific update method."""
        _create_index(self.instance, self.session_maker, 'states', LEGACY_STATES_EVENT_ID_INDEX)

class _SchemaVersion6Migrator(_SchemaVersionMigrator, target_version=6):

    def _apply_update(self):
        """Version specific update method."""
        _add_columns(self.session_maker, 'events', ['context_id CHARACTER(36)', 'context_user_id CHARACTER(36)'])
        _create_index(self.instance, self.session_maker, 'events', 'ix_events_context_id')
        _add_columns(self.session_maker, 'states', ['context_id CHARACTER(36)', 'context_user_id CHARACTER(36)'])
        _create_index(self.instance, self.session_maker, 'states', 'ix_states_context_id')

class _SchemaVersion7Migrator(_SchemaVersionMigrator, target_version=7):

    def _apply_update(self):
        """Version specific update method."""

class _SchemaVersion8Migrator(_SchemaVersionMigrator, target_version=8):

    def _apply_update(self):
        """Version specific update method."""
        _add_columns(self.session_maker, 'events', ['context_parent_id CHARACTER(36)'])
        _add_columns(self.session_maker, 'states', ['old_state_id INTEGER'])

class _SchemaVersion9Migrator(_SchemaVersionMigrator, target_version=9):

    def _apply_update(self):
        """Version specific update method."""
        _drop_index(self.session_maker, 'states', 'ix_states_context_user_id')
        _drop_index(self.session_maker, 'states', 'ix_states_context_parent_id')
        _drop_index(self.session_maker, 'states', 'ix_states_entity_id')
        _drop_index(self.session_maker, 'events', 'ix_events_event_type')

class _SchemaVersion10Migrator(_SchemaVersionMigrator, target_version=10):

    def _apply_update(self):
        """Version specific update method."""

class _SchemaVersion11Migrator(_SchemaVersionMigrator, target_version=11):

    def _apply_update(self):
        """Version specific update method."""
        _create_index(self.instance, self.session_maker, 'states', 'ix_states_old_state_id')
        if self.engine.dialect.name in (SupportedDialect.MYSQL, SupportedDialect.POSTGRESQL):
            _update_states_table_with_foreign_key_options(self.session_maker, self.engine)

class _SchemaVersion12Migrator(_SchemaVersionMigrator, target_version=12):

    def _apply_update(self):
        """Version specific update method."""
        if self.engine.dialect.name == SupportedDialect.MYSQL:
            _modify_columns(self.session_maker, self.engine, 'events', ['event_data LONGTEXT'])
            _modify_columns(self.session_maker, self.engine, 'states', ['attributes LONGTEXT'])

class _SchemaVersion13Migrator(_SchemaVersionMigrator, target_version=13):

    def _apply_update(self):
        """Version specific update method."""
        if self.engine.dialect.name == SupportedDialect.MYSQL:
            _modify_columns(self.session_maker, self.engine, 'events', ['time_fired DATETIME(6)', 'created DATETIME(6)'])
            _modify_columns(self.session_maker, self.engine, 'states', ['last_changed DATETIME(6)', 'last_updated DATETIME(6)', 'created DATETIME(6)'])

class _SchemaVersion14Migrator(_SchemaVersionMigrator, target_version=14):

    def _apply_update(self):
        """Version specific update method."""
        _modify_columns(self.session_maker, self.engine, 'events', ['event_type VARCHAR(64)'])

class _SchemaVersion15Migrator(_SchemaVersionMigrator, target_version=15):

    def _apply_update(self):
        """Version specific update method."""

class _SchemaVersion16Migrator(_SchemaVersionMigrator, target_version=16):

    def _apply_update(self):
        """Version specific update method."""
        if self.engine.dialect.name in (SupportedDialect.MYSQL, SupportedDialect.POSTGRESQL):
            _drop_foreign_key_constraints(self.session_maker, self.engine, TABLE_STATES, 'old_state_id')

class _SchemaVersion17Migrator(_SchemaVersionMigrator, target_version=17):

    def _apply_update(self):
        """Version specific update method."""

class _SchemaVersion18Migrator(_SchemaVersionMigrator, target_version=18):

    def _apply_update(self):
        """Version specific update method."""
        Base.metadata.drop_all(bind=self.engine, tables=[cast(Table, StatisticsShortTerm.__table__), cast(Table, Statistics.__table__), cast(Table, StatisticsMeta.__table__)])
        cast(Table, StatisticsMeta.__table__).create(self.engine)
        cast(Table, StatisticsShortTerm.__table__).create(self.engine)
        cast(Table, Statistics.__table__).create(self.engine)

class _SchemaVersion19Migrator(_SchemaVersionMigrator, target_version=19):

    def _apply_update(self):
        """Version specific update method."""
        with session_scope(session=self.session_maker()) as session:
            session.add(StatisticsRuns(start=get_start_time()))

class _SchemaVersion20Migrator(_SchemaVersionMigrator, target_version=20):

    def _apply_update(self):
        """Version specific update method."""
        if self.engine.dialect.name in [SupportedDialect.MYSQL, SupportedDialect.POSTGRESQL]:
            _modify_columns(self.session_maker, self.engine, 'statistics', [f'{column} {DOUBLE_PRECISION_TYPE_SQL}' for column in ('max', 'mean', 'min', 'state', 'sum')])

class _SchemaVersion21Migrator(_SchemaVersionMigrator, target_version=21):

    def _apply_update(self):
        """Version specific update method."""
        if self.engine.dialect.name == SupportedDialect.MYSQL:
            for table in ('events', 'states', 'statistics_meta'):
                _correct_table_character_set_and_collation(table, self.session_maker)

class _SchemaVersion22Migrator(_SchemaVersionMigrator, target_version=22):

    def _apply_update(self):
        """Version specific update method."""
        if self.engine.dialect.name == 'oracle':
            Base.metadata.drop_all(bind=self.engine, tables=[cast(Table, StatisticsShortTerm.__table__), cast(Table, Statistics.__table__), cast(Table, StatisticsMeta.__table__), cast(Table, StatisticsRuns.__table__)])
            cast(Table, StatisticsRuns.__table__).create(self.engine)
            cast(Table, StatisticsMeta.__table__).create(self.engine)
            cast(Table, StatisticsShortTerm.__table__).create(self.engine)
            cast(Table, Statistics.__table__).create(self.engine)
        with session_scope(session=self.session_maker()) as session:
            if session.query(Statistics.id).count() and (last_run_string := session.query(func.max(StatisticsRuns.start)).scalar()):
                last_run_start_time = process_timestamp(last_run_string)
                if last_run_start_time:
                    fake_start_time = last_run_start_time + timedelta(minutes=5)
                    while fake_start_time < last_run_start_time + timedelta(hours=1):
                        session.add(StatisticsRuns(start=fake_start_time))
                        fake_start_time += timedelta(minutes=5)
        with session_scope(session=self.session_maker()) as session:
            for sum_statistic in session.query(StatisticsMeta.id).filter_by(has_sum=true()):
                last_statistic = session.query(Statistics.start, Statistics.last_reset, Statistics.state, Statistics.sum).filter_by(metadata_id=sum_statistic.id).order_by(Statistics.start.desc()).first()
                if last_statistic:
                    session.add(StatisticsShortTerm(metadata_id=sum_statistic.id, start=last_statistic.start, last_reset=last_statistic.last_reset, state=last_statistic.state, sum=last_statistic.sum))

class _SchemaVersion23Migrator(_SchemaVersionMigrator, target_version=23):

    def _apply_update(self):
        """Version specific update method."""
        _add_columns(self.session_maker, 'statistics_meta', ['name VARCHAR(255)'])

class _SchemaVersion24Migrator(_SchemaVersionMigrator, target_version=24):

    def _apply_update(self):
        """Version specific update method."""

class _SchemaVersion25Migrator(_SchemaVersionMigrator, target_version=25):

    def _apply_update(self):
        """Version specific update method."""
        _add_columns(self.session_maker, 'states', [f'attributes_id {self.column_types.big_int_type}'])
        _create_index(self.instance, self.session_maker, 'states', 'ix_states_attributes_id')

class _SchemaVersion26Migrator(_SchemaVersionMigrator, target_version=26):

    def _apply_update(self):
        """Version specific update method."""
        _create_index(self.instance, self.session_maker, 'statistics_runs', 'ix_statistics_runs_start')

class _SchemaVersion27Migrator(_SchemaVersionMigrator, target_version=27):

    def _apply_update(self):
        """Version specific update method."""
        _add_columns(self.session_maker, 'events', [f'data_id {self.column_types.big_int_type}'])
        _create_index(self.instance, self.session_maker, 'events', 'ix_events_data_id')

class _SchemaVersion28Migrator(_SchemaVersionMigrator, target_version=28):

    def _apply_update(self):
        """Version specific update method."""
        _add_columns(self.session_maker, 'events', ['origin_idx INTEGER'])
        _drop_index(self.session_maker, 'events', 'ix_events_context_user_id')
        _drop_index(self.session_maker, 'events', 'ix_events_context_parent_id')
        _add_columns(self.session_maker, 'states', ['origin_idx INTEGER', 'context_id VARCHAR(36)', 'context_user_id VARCHAR(36)', 'context_parent_id VARCHAR(36)'])
        _create_index(self.instance, self.session_maker, 'states', 'ix_states_context_id')

class _SchemaVersion29Migrator(_SchemaVersionMigrator, target_version=29):

    def _apply_update(self):
        """Version specific update method."""
        _drop_index(self.session_maker, 'statistics_meta', 'ix_statistics_meta_statistic_id')
        if self.engine.dialect.name == SupportedDialect.MYSQL:
            with contextlib.suppress(SQLAlchemyError), session_scope(session=self.session_maker()) as session:
                connection = session.connection()
                connection.execute(text('ALTER TABLE statistics_meta ROW_FORMAT=DYNAMIC'))
        try:
            _create_index(self.instance, self.session_maker, 'statistics_meta', 'ix_statistics_meta_statistic_id')
        except DatabaseError:
            with session_scope(session=self.session_maker()) as session:
                delete_statistics_meta_duplicates(self.instance, session)
            _create_index(self.instance, self.session_maker, 'statistics_meta', 'ix_statistics_meta_statistic_id')

class _SchemaVersion30Migrator(_SchemaVersionMigrator, target_version=30):

    def _apply_update(self):
        """Version specific update method."""

class _SchemaVersion31Migrator(_SchemaVersionMigrator, target_version=31):

    def _apply_update(self):
        """Version specific update method."""
        _add_columns(self.session_maker, 'events', [f'time_fired_ts {self.column_types.timestamp_type}'])
        _add_columns(self.session_maker, 'states', [f'last_updated_ts {self.column_types.timestamp_type}', f'last_changed_ts {self.column_types.timestamp_type}'])
        _create_index(self.instance, self.session_maker, 'events', 'ix_events_time_fired_ts')
        _create_index(self.instance, self.session_maker, 'events', 'ix_events_event_type_time_fired_ts')
        _create_index(self.instance, self.session_maker, 'states', 'ix_states_entity_id_last_updated_ts')
        _create_index(self.instance, self.session_maker, 'states', 'ix_states_last_updated_ts')
        _migrate_columns_to_timestamp(self.instance, self.session_maker, self.engine)

class _SchemaVersion32Migrator(_SchemaVersionMigrator, target_version=32):

    def _apply_update(self):
        """Version specific update method."""
        _drop_index(self.session_maker, 'states', 'ix_states_entity_id_last_updated')
        _drop_index(self.session_maker, 'events', 'ix_events_event_type_time_fired')
        _drop_index(self.session_maker, 'states', 'ix_states_last_updated')
        _drop_index(self.session_maker, 'events', 'ix_events_time_fired')
        with session_scope(session=self.session_maker()) as session:
            assert self.instance.engine is not None, 'engine should never be None'
            _wipe_old_string_time_columns(self.instance, self.instance.engine, session)

class _SchemaVersion33Migrator(_SchemaVersionMigrator, target_version=33):

    def _apply_update(self):
        """Version specific update method."""

class _SchemaVersion34Migrator(_SchemaVersionMigrator, target_version=34):

    def _apply_update(self):
        """Version specific update method."""
        _add_columns(self.session_maker, 'statistics', [f'created_ts {self.column_types.timestamp_type}', f'start_ts {self.column_types.timestamp_type}', f'last_reset_ts {self.column_types.timestamp_type}'])
        _add_columns(self.session_maker, 'statistics_short_term', [f'created_ts {self.column_types.timestamp_type}', f'start_ts {self.column_types.timestamp_type}', f'last_reset_ts {self.column_types.timestamp_type}'])
        _create_index(self.instance, self.session_maker, 'statistics', 'ix_statistics_start_ts')
        _create_index(self.instance, self.session_maker, 'statistics', 'ix_statistics_statistic_id_start_ts')
        _create_index(self.instance, self.session_maker, 'statistics_short_term', 'ix_statistics_short_term_start_ts')
        _create_index(self.instance, self.session_maker, 'statistics_short_term', 'ix_statistics_short_term_statistic_id_start_ts')
        _migrate_statistics_columns_to_timestamp_removing_duplicates(self.hass, self.instance, self.session_maker, self.engine)

class _SchemaVersion35Migrator(_SchemaVersionMigrator, target_version=35):

    def _apply_update(self):
        """Version specific update method."""
        _drop_index(self.session_maker, 'statistics', 'ix_statistics_statistic_id_start', quiet=True)
        _drop_index(self.session_maker, 'statistics_short_term', 'ix_statistics_short_term_statistic_id_start', quiet=True)
        while not cleanup_statistics_timestamp_migration(self.instance):
            pass

class _SchemaVersion36Migrator(_SchemaVersionMigrator, target_version=36):

    def _apply_update(self):
        """Version specific update method."""
        for table in ('states', 'events'):
            _add_columns(self.session_maker, table, [f'context_id_bin {self.column_types.context_bin_type}', f'context_user_id_bin {self.column_types.context_bin_type}', f'context_parent_id_bin {self.column_types.context_bin_type}'])
        _create_index(self.instance, self.session_maker, 'events', 'ix_events_context_id_bin')
        _create_index(self.instance, self.session_maker, 'states', 'ix_states_context_id_bin')

class _SchemaVersion37Migrator(_SchemaVersionMigrator, target_version=37):

    def _apply_update(self):
        """Version specific update method."""
        _add_columns(self.session_maker, 'events', [f'event_type_id {self.column_types.big_int_type}'])
        _create_index(self.instance, self.session_maker, 'events', 'ix_events_event_type_id')
        _drop_index(self.session_maker, 'events', 'ix_events_event_type_time_fired_ts')
        _create_index(self.instance, self.session_maker, 'events', 'ix_events_event_type_id_time_fired_ts')

class _SchemaVersion38Migrator(_SchemaVersionMigrator, target_version=38):

    def _apply_update(self):
        """Version specific update method."""
        _add_columns(self.session_maker, 'states', [f'metadata_id {self.column_types.big_int_type}'])
        _create_index(self.instance, self.session_maker, 'states', 'ix_states_metadata_id')
        _create_index(self.instance, self.session_maker, 'states', 'ix_states_metadata_id_last_updated_ts')

class _SchemaVersion39Migrator(_SchemaVersionMigrator, target_version=39):

    def _apply_update(self):
        """Version specific update method."""
        _drop_index(self.session_maker, 'events', 'ix_events_event_type_time_fired_ts', quiet=True)
        _drop_index(self.session_maker, 'events', 'ix_events_event_type', quiet=True)
        _drop_index(self.session_maker, 'events', 'ix_events_event_type_time_fired', quiet=True)
        _drop_index(self.session_maker, 'events', 'ix_events_time_fired', quiet=True)
        _drop_index(self.session_maker, 'events', 'ix_events_context_user_id', quiet=True)
        _drop_index(self.session_maker, 'events', 'ix_events_context_parent_id', quiet=True)
        _drop_index(self.session_maker, 'states', 'ix_states_entity_id_last_updated', quiet=True)
        _drop_index(self.session_maker, 'states', 'ix_states_last_updated', quiet=True)
        _drop_index(self.session_maker, 'states', 'ix_states_entity_id', quiet=True)
        _drop_index(self.session_maker, 'states', 'ix_states_context_user_id', quiet=True)
        _drop_index(self.session_maker, 'states', 'ix_states_context_parent_id', quiet=True)
        _drop_index(self.session_maker, 'states', 'ix_states_created_domain', quiet=True)
        _drop_index(self.session_maker, 'states', 'ix_states_entity_id_created', quiet=True)
        _drop_index(self.session_maker, 'states', 'states__state_changes', quiet=True)
        _drop_index(self.session_maker, 'states', 'states__significant_changes', quiet=True)
        _drop_index(self.session_maker, 'states', 'ix_states_entity_id_created', quiet=True)
        _drop_index(self.session_maker, 'statistics', 'ix_statistics_statistic_id_start', quiet=True)
        _drop_index(self.session_maker, 'statistics_short_term', 'ix_statistics_short_term_statistic_id_start', quiet=True)

class _SchemaVersion40Migrator(_SchemaVersionMigrator, target_version=40):

    def _apply_update(self):
        """Version specific update method."""
        _drop_index(self.session_maker, 'events', 'ix_events_event_type_id')
        _drop_index(self.session_maker, 'states', 'ix_states_metadata_id')
        _drop_index(self.session_maker, 'statistics', 'ix_statistics_metadata_id')
        _drop_index(self.session_maker, 'statistics_short_term', 'ix_statistics_short_term_metadata_id')

class _SchemaVersion41Migrator(_SchemaVersionMigrator, target_version=41):

    def _apply_update(self):
        """Version specific update method."""
        _create_index(self.instance, self.session_maker, 'event_types', 'ix_event_types_event_type')
        _create_index(self.instance, self.session_maker, 'states_meta', 'ix_states_meta_entity_id')

class _SchemaVersion42Migrator(_SchemaVersionMigrator, target_version=42):

    def _apply_update(self):
        """Version specific update method."""
        _migrate_statistics_columns_to_timestamp_removing_duplicates(self.hass, self.instance, self.session_maker, self.engine)

class _SchemaVersion43Migrator(_SchemaVersionMigrator, target_version=43):

    def _apply_update(self):
        """Version specific update method."""
        _add_columns(self.session_maker, 'states', [f'last_reported_ts {self.column_types.timestamp_type}'])

class _SchemaVersion44Migrator(_SchemaVersionMigrator, target_version=44):

    def _apply_update(self):
        """Version specific update method."""

class _SchemaVersion45Migrator(_SchemaVersionMigrator, target_version=45):

    def _apply_update(self):
        """Version specific update method."""
FOREIGN_COLUMNS = (('events', ('data_id', 'event_type_id'), (('data_id', 'event_data', 'data_id'), ('event_type_id', 'event_types', 'event_type_id'))), ('states', ('event_id', 'old_state_id', 'attributes_id', 'metadata_id'), (('event_id', None, None), ('old_state_id', 'states', 'state_id'), ('attributes_id', 'state_attributes', 'attributes_id'), ('metadata_id', 'states_meta', 'metadata_id'))), ('statistics', ('metadata_id',), (('metadata_id', 'statistics_meta', 'id'),)), ('statistics_short_term', ('metadata_id',), (('metadata_id', 'statistics_meta', 'id'),)))

class _SchemaVersion46Migrator(_SchemaVersionMigrator, target_version=46):

    def _apply_update(self):
        """Version specific update method."""
        if self.engine.dialect.name == SupportedDialect.SQLITE:
            return
        identity_sql = 'NOT NULL AUTO_INCREMENT' if self.engine.dialect.name == SupportedDialect.MYSQL else ''
        for table, columns, _ in FOREIGN_COLUMNS:
            for column in columns:
                _drop_foreign_key_constraints(self.session_maker, self.engine, table, column)
        for table, columns, _ in FOREIGN_COLUMNS:
            _modify_columns(self.session_maker, self.engine, table, [f'{column} {BIG_INTEGER_SQL}' for column in columns])
        id_columns = (('events', 'event_id'), ('event_data', 'data_id'), ('event_types', 'event_type_id'), ('states', 'state_id'), ('state_attributes', 'attributes_id'), ('states_meta', 'metadata_id'), ('statistics', 'id'), ('statistics_short_term', 'id'), ('statistics_meta', 'id'), ('recorder_runs', 'run_id'), ('schema_changes', 'change_id'), ('statistics_runs', 'run_id'))
        for table, column in id_columns:
            _modify_columns(self.session_maker, self.engine, table, [f'{column} {BIG_INTEGER_SQL} {identity_sql}'])

class _SchemaVersion47Migrator(_SchemaVersionMigrator, target_version=47):

    def _apply_update(self):
        """Version specific update method."""
        if self.engine.dialect.name == SupportedDialect.SQLITE:
            return
        _restore_foreign_key_constraints(self.session_maker, self.engine, [(table, column, foreign_table, foreign_column) for table, _, foreign_mappings in FOREIGN_COLUMNS for column, foreign_table, foreign_column in foreign_mappings])

class _SchemaVersion48Migrator(_SchemaVersionMigrator, target_version=48):

    def _apply_update(self):
        """Version specific update method."""
        _migrate_columns_to_timestamp(self.instance, self.session_maker, self.engine)

def _migrate_statistics_columns_to_timestamp_removing_duplicates(hass, instance, session_maker, engine):
    """Migrate statistics columns to timestamp or cleanup duplicates."""
    try:
        _migrate_statistics_columns_to_timestamp(instance, session_maker, engine)
    except IntegrityError as ex:
        _LOGGER.error('Statistics table contains duplicate entries: %s; Cleaning up duplicates and trying again; %s', ex, MIGRATION_NOTE_WHILE)
        with session_scope(session=session_maker()) as session:
            delete_statistics_duplicates(instance, hass, session)
        try:
            _migrate_statistics_columns_to_timestamp(instance, session_maker, engine)
        except IntegrityError:
            _LOGGER.warning('Statistics table still contains duplicate entries after cleanup; Falling back to a one by one migration')
            _migrate_statistics_columns_to_timestamp_one_by_one(instance, session_maker)
        _LOGGER.error('Statistics migration successfully recovered after statistics table duplicate cleanup')

def _correct_table_character_set_and_collation(table, session_maker):
    """Correct issues detected by validate_db_schema."""
    _LOGGER.warning('Updating character set and collation of table %s to utf8mb4. %s', table, MIGRATION_NOTE_MINUTES)
    with contextlib.suppress(SQLAlchemyError), session_scope(session=session_maker()) as session:
        connection = session.connection()
        connection.execute(text(f'ALTER TABLE {table} CONVERT TO CHARACTER SET {MYSQL_DEFAULT_CHARSET} COLLATE {MYSQL_COLLATE}, LOCK=EXCLUSIVE'))

@database_job_retry_wrapper('Wipe old string time columns', 3)
def _wipe_old_string_time_columns(instance, engine, session):
    """Wipe old string time columns to save space."""
    if engine.dialect.name == SupportedDialect.SQLITE:
        session.execute(text('UPDATE events set time_fired=NULL;'))
        session.commit()
        session.execute(text('UPDATE states set last_updated=NULL, last_changed=NULL;'))
        session.commit()
    elif engine.dialect.name == SupportedDialect.MYSQL:
        session.execute(text('UPDATE events set time_fired=NULL LIMIT 100000;'))
        session.commit()
        session.execute(text('UPDATE states set last_updated=NULL, last_changed=NULL LIMIT 100000;'))
        session.commit()
    elif engine.dialect.name == SupportedDialect.POSTGRESQL:
        session.execute(text('UPDATE events set time_fired=NULL where event_id in (select event_id from events where time_fired_ts is NOT NULL LIMIT 100000);'))
        session.commit()
        session.execute(text('UPDATE states set last_updated=NULL, last_changed=NULL where state_id in (select state_id from states where last_updated_ts is NOT NULL LIMIT 100000);'))
        session.commit()

@database_job_retry_wrapper('Migrate columns to timestamp', 3)
def _migrate_columns_to_timestamp(instance, session_maker, engine):
    """Migrate columns to use timestamp."""
    result = None
    if engine.dialect.name == SupportedDialect.SQLITE:
        with session_scope(session=session_maker()) as session:
            connection = session.connection()
            connection.execute(text('UPDATE events set time_fired_ts=strftime("%s",time_fired) + cast(substr(time_fired,-7) AS FLOAT) WHERE time_fired_ts is NULL;'))
            connection.execute(text('UPDATE states set last_updated_ts=strftime("%s",last_updated) + cast(substr(last_updated,-7) AS FLOAT), last_changed_ts=strftime("%s",last_changed) + cast(substr(last_changed,-7) AS FLOAT)  WHERE last_updated_ts is NULL;'))
    elif engine.dialect.name == SupportedDialect.MYSQL:
        while result is None or result.rowcount > 0:
            with session_scope(session=session_maker()) as session:
                result = session.connection().execute(text('UPDATE events set time_fired_ts=IF(time_fired is NULL or UNIX_TIMESTAMP(time_fired) is NULL,0,UNIX_TIMESTAMP(time_fired)) where time_fired_ts is NULL LIMIT 100000;'))
        result = None
        while result is None or result.rowcount > 0:
            with session_scope(session=session_maker()) as session:
                result = session.connection().execute(text('UPDATE states set last_updated_ts=IF(last_updated is NULL or UNIX_TIMESTAMP(last_updated) is NULL,0,UNIX_TIMESTAMP(last_updated) ), last_changed_ts=UNIX_TIMESTAMP(last_changed) where last_updated_ts is NULL LIMIT 100000;'))
    elif engine.dialect.name == SupportedDialect.POSTGRESQL:
        while result is None or result.rowcount > 0:
            with session_scope(session=session_maker()) as session:
                result = session.connection().execute(text('UPDATE events SET time_fired_ts= (case when time_fired is NULL then 0 else EXTRACT(EPOCH FROM time_fired::timestamptz) end) WHERE event_id IN ( SELECT event_id FROM events where time_fired_ts is NULL LIMIT 100000  );'))
        result = None
        while result is None or result.rowcount > 0:
            with session_scope(session=session_maker()) as session:
                result = session.connection().execute(text('UPDATE states set last_updated_ts=(case when last_updated is NULL then 0 else EXTRACT(EPOCH FROM last_updated::timestamptz) end), last_changed_ts=EXTRACT(EPOCH FROM last_changed::timestamptz) where state_id IN ( SELECT state_id FROM states where last_updated_ts is NULL LIMIT 100000  );'))

@database_job_retry_wrapper('Migrate statistics columns to timestamp one by one', 3)
def _migrate_statistics_columns_to_timestamp_one_by_one(instance, session_maker):
    """Migrate statistics columns to use timestamp on by one.

    If something manually inserted data into the statistics table
    in the past it may have inserted duplicate rows.

    Before we had the unique index on (statistic_id, start) this
    the data could have been inserted without any errors and we
    could end up with duplicate rows that go undetected (even by
    our current duplicate cleanup code) until we try to migrate the
    data to use timestamps.

    This will migrate the data one by one to ensure we do not hit any
    duplicate rows, and remove the duplicate rows as they are found.
    """
    for find_func, migrate_func, delete_func in ((find_unmigrated_statistics_rows, migrate_single_statistics_row_to_timestamp, delete_duplicate_statistics_row), (find_unmigrated_short_term_statistics_rows, migrate_single_short_term_statistics_row_to_timestamp, delete_duplicate_short_term_statistics_row)):
        with session_scope(session=session_maker()) as session:
            while (stats := session.execute(find_func(instance.max_bind_vars)).all()):
                for statistic_id, start, created, last_reset in stats:
                    start_ts = datetime_to_timestamp_or_none(process_timestamp(start))
                    created_ts = datetime_to_timestamp_or_none(process_timestamp(created))
                    last_reset_ts = datetime_to_timestamp_or_none(process_timestamp(last_reset))
                    try:
                        session.execute(migrate_func(statistic_id, start_ts, created_ts, last_reset_ts))
                    except IntegrityError:
                        session.execute(delete_func(statistic_id))
                session.commit()

@database_job_retry_wrapper('Migrate statistics columns to timestamp', 3)
def _migrate_statistics_columns_to_timestamp(instance, session_maker, engine):
    """Migrate statistics columns to use timestamp."""
    result = None
    if engine.dialect.name == SupportedDialect.SQLITE:
        for table in STATISTICS_TABLES:
            with session_scope(session=session_maker()) as session:
                session.connection().execute(text(f"UPDATE {table} set start_ts=strftime('%s',start) + cast(substr(start,-7) AS FLOAT), created_ts=strftime('%s',created) + cast(substr(created,-7) AS FLOAT), last_reset_ts=strftime('%s',last_reset) + cast(substr(last_reset,-7) AS FLOAT) where start_ts is NULL;"))
    elif engine.dialect.name == SupportedDialect.MYSQL:
        for table in STATISTICS_TABLES:
            result = None
            while result is None or result.rowcount > 0:
                with session_scope(session=session_maker()) as session:
                    result = session.connection().execute(text(f'UPDATE {table} set start_ts=IF(start is NULL or UNIX_TIMESTAMP(start) is NULL,0,UNIX_TIMESTAMP(start) ), created_ts=UNIX_TIMESTAMP(created), last_reset_ts=UNIX_TIMESTAMP(last_reset) where start_ts is NULL LIMIT 100000;'))
    elif engine.dialect.name == SupportedDialect.POSTGRESQL:
        for table in STATISTICS_TABLES:
            result = None
            while result is None or result.rowcount > 0:
                with session_scope(session=session_maker()) as session:
                    result = session.connection().execute(text(f'UPDATE {table} set start_ts=(case when start is NULL then 0 else EXTRACT(EPOCH FROM start::timestamptz) end), created_ts=EXTRACT(EPOCH FROM created::timestamptz), last_reset_ts=EXTRACT(EPOCH FROM last_reset::timestamptz) where id IN (SELECT id FROM {table} where start_ts is NULL LIMIT 100000);'))

def _context_id_to_bytes(context_id):
    """Convert a context_id to bytes."""
    if context_id is None:
        return None
    with contextlib.suppress(ValueError):
        if len(context_id) == 26:
            return ulid_to_bytes(context_id)
        return UUID(context_id).bytes
    return None

def _generate_ulid_bytes_at_time(timestamp):
    """Generate a ulid with a specific timestamp."""
    return ulid_to_bytes(ulid_at_time(timestamp or time()))

def post_migrate_entity_ids(instance):
    """Remove old entity_id strings from states.

    We cannot do this in migrate_entity_ids since the history queries
    still need to work while the migration is in progress.
    """
    session_maker = instance.get_session
    _LOGGER.debug('Cleanup legacy entity_ids')
    with session_scope(session=session_maker()) as session:
        cursor_result = session.connection().execute(batch_cleanup_entity_ids())
        is_done = not cursor_result or cursor_result.rowcount == 0
    _LOGGER.debug('Cleanup legacy entity_ids done=%s', is_done)
    return is_done

def _initialize_database(session):
    """Initialize a new database.

    The function determines the schema version by inspecting the db structure.

    When the schema version is not present in the db, either db was just
    created with the correct schema, or this is a db created before schema
    versions were tracked. For now, we'll test if the changes for schema
    version 1 are present to make the determination. Eventually this logic
    can be removed and we can assume a new db is being created.
    """
    inspector = sqlalchemy.inspect(session.connection())
    indexes = inspector.get_indexes('events')
    for index in indexes:
        if index['column_names'] in (['time_fired'], ['time_fired_ts']):
            session.add(StatisticsRuns(start=get_start_time()))
            session.add(SchemaChanges(schema_version=SCHEMA_VERSION))
            return True
    current_version = SchemaChanges(schema_version=0)
    session.add(current_version)
    return True

def initialize_database(session_maker):
    """Initialize a new database."""
    try:
        with session_scope(session=session_maker(), read_only=True) as session:
            if _get_current_schema_version(session) is not None:
                return True
        with session_scope(session=session_maker()) as session:
            return _initialize_database(session)
    except Exception:
        _LOGGER.exception('Error when initialise database')
        return False

@dataclass(slots=True)
class MigrationTask(RecorderTask):
    """Base class for migration tasks."""
    commit_before = False

    def run(self, instance):
        """Run migration task."""
        if not self.migrator.migrate_data(instance):
            instance.queue_task(MigrationTask(self.migrator))

@dataclass(slots=True)
class CommitBeforeMigrationTask(MigrationTask):
    """Base class for migration tasks which commit first."""
    commit_before = True

@dataclass(frozen=True, kw_only=True)
class DataMigrationStatus:
    """Container for data migrator status."""

class BaseMigration(ABC):
    """Base class for migrations."""
    index_to_drop = None
    required_schema_version = 0
    migration_version = 1

    def __init__(self, *, initial_schema_version, start_schema_version, migration_changes):
        """Initialize a new BaseRunTimeMigration.

        :param initial_schema_version: The schema version the database was created with.
        :param start_schema_version: The schema version when starting the migration.
        """
        self.initial_schema_version = initial_schema_version
        self.start_schema_version = start_schema_version
        self.migration_changes = migration_changes

    @abstractmethod
    def migrate_data(self, instance, /):
        """Migrate some data, return True if migration is completed."""

    def _migrate_data(self, instance):
        """Migrate some data, returns True if migration is completed."""
        status = self.migrate_data_impl(instance)
        if status.migration_done:
            with session_scope(session=instance.get_session()) as session:
                self.migration_done(instance, session)
                _mark_migration_done(session, self.__class__)
            if self.index_to_drop is not None:
                table, index, _ = self.index_to_drop
                _drop_index(instance.get_session, table, index)
        return not status.needs_migrate

    @abstractmethod
    def migrate_data_impl(self, instance):
        """Migrate some data, return if the migration needs to run and if it is done."""

    def migration_done(self, instance, session):
        """Will be called after migrate returns True or if migration is not needed."""

    @abstractmethod
    def needs_migrate_impl(self, instance, session):
        """Return if the migration needs to run and if it is done."""

    def needs_migrate(self, instance, session):
        """Return if the migration needs to run.

        If the migration needs to run, it will return True.

        If the migration does not need to run, it will return False and
        mark the migration as done in the database if its not already
        marked as done.
        """
        if self.initial_schema_version > self.max_initial_schema_version:
            _LOGGER.debug("Data migration '%s' not needed, database created with version %s after migrator was added", self.migration_id, self.initial_schema_version)
            return False
        if self.start_schema_version < self.required_schema_version:
            _LOGGER.info("Data migration '%s' needed, schema too old", self.migration_id)
            return True
        has_needed_index = self._has_needed_index(session)
        if has_needed_index is True:
            _LOGGER.info("Data migration '%s' needed, index to drop still exists", self.migration_id)
            return True
        if self.migration_changes.get(self.migration_id, -1) >= self.migration_version:
            _LOGGER.debug("Data migration '%s' not needed, already completed", self.migration_id)
            return False
        if has_needed_index is False:
            _LOGGER.info("Data migration '%s' needed, index to drop does not exist", self.migration_id)
            return True
        needs_migrate = self.needs_migrate_impl(instance, session)
        if needs_migrate.migration_done:
            _mark_migration_done(session, self.__class__)
        _LOGGER.info("Data migration '%s' needed: %s", self.migration_id, needs_migrate.needs_migrate)
        return needs_migrate.needs_migrate

    def _has_needed_index(self, session):
        """Check if the index needed by the migration exists."""
        if self.index_to_drop is None:
            return None
        table_name, index_name, _ = self.index_to_drop
        return get_index_by_name(session, table_name, index_name) is not None

class BaseOffLineMigration(BaseMigration):
    """Base class for off line migrations."""

    def migrate_all(self, instance, session_maker):
        """Migrate all data."""
        with session_scope(session=session_maker()) as session:
            if not self.needs_migrate(instance, session):
                _LOGGER.debug("Migration not needed for '%s'", self.migration_id)
                self.migration_done(instance, session)
                return
        self._ensure_index_exists(instance)
        _LOGGER.warning("The database is about to do data migration step '%s', %s", self.migration_id, MIGRATION_NOTE_OFFLINE)
        while not self.migrate_data(instance):
            pass
        _LOGGER.warning("Data migration step '%s' completed", self.migration_id)

    @database_job_retry_wrapper_method('migrate data', 10)
    def migrate_data(self, instance):
        """Migrate some data, returns True if migration is completed."""
        return self._migrate_data(instance)

    def _ensure_index_exists(self, instance):
        """Ensure the index needed by the migration exists."""
        if not self.index_to_drop:
            return
        table_name, index_name, base = self.index_to_drop
        with session_scope(session=instance.get_session()) as session:
            if get_index_by_name(session, table_name, index_name) is not None:
                return
        _LOGGER.warning("Data migration step '%s' needs index `%s` on table `%s`, but it does not exist and will be added now", self.migration_id, index_name, table_name)
        _create_index(instance, instance.get_session, table_name, index_name, base=base)

class BaseRunTimeMigration(BaseMigration):
    """Base class for run time migrations."""
    task = MigrationTask

    def queue_migration(self, instance, session):
        """Start migration if needed."""
        if self.needs_migrate(instance, session):
            instance.queue_task(self.task(self))
        else:
            self.migration_done(instance, session)

    @retryable_database_job_method('migrate data')
    def migrate_data(self, instance):
        """Migrate some data, returns True if migration is completed."""
        return self._migrate_data(instance)

class BaseMigrationWithQuery(BaseMigration):
    """Base class for run time migrations."""

    @abstractmethod
    def needs_migrate_query(self):
        """Return the query to check if the migration needs to run."""

    def needs_migrate_impl(self, instance, session):
        """Return if the migration needs to run."""
        needs_migrate = execute_stmt_lambda_element(session, self.needs_migrate_query())
        return DataMigrationStatus(needs_migrate=bool(needs_migrate), migration_done=not needs_migrate)

class StatesContextIDMigration(BaseMigrationWithQuery, BaseOffLineMigration):
    """Migration to migrate states context_ids to binary format."""
    required_schema_version = CONTEXT_ID_AS_BINARY_SCHEMA_VERSION
    max_initial_schema_version = CONTEXT_ID_AS_BINARY_SCHEMA_VERSION - 1
    migration_id = 'state_context_id_as_binary'
    migration_version = 2
    index_to_drop = ('states', 'ix_states_context_id', LegacyBase)

    def migrate_data_impl(self, instance):
        """Migrate states context_ids to use binary format, return True if completed."""
        _to_bytes = _context_id_to_bytes
        session_maker = instance.get_session
        _LOGGER.debug('Migrating states context_ids to binary format')
        with session_scope(session=session_maker()) as session:
            if (states := session.execute(find_states_context_ids_to_migrate(instance.max_bind_vars)).all()):
                session.execute(update(States), [{'state_id': state_id, 'context_id': None, 'context_id_bin': _to_bytes(context_id) or _generate_ulid_bytes_at_time(last_updated_ts), 'context_user_id': None, 'context_user_id_bin': _to_bytes(context_user_id), 'context_parent_id': None, 'context_parent_id_bin': _to_bytes(context_parent_id)} for state_id, last_updated_ts, context_id, context_user_id, context_parent_id in states])
            is_done = not states
        _LOGGER.debug('Migrating states context_ids to binary format: done=%s', is_done)
        return DataMigrationStatus(needs_migrate=not is_done, migration_done=is_done)

    def needs_migrate_query(self):
        """Return the query to check if the migration needs to run."""
        return has_states_context_ids_to_migrate()

class EventsContextIDMigration(BaseMigrationWithQuery, BaseOffLineMigration):
    """Migration to migrate events context_ids to binary format."""
    required_schema_version = CONTEXT_ID_AS_BINARY_SCHEMA_VERSION
    max_initial_schema_version = CONTEXT_ID_AS_BINARY_SCHEMA_VERSION - 1
    migration_id = 'event_context_id_as_binary'
    migration_version = 2
    index_to_drop = ('events', 'ix_events_context_id', LegacyBase)

    def migrate_data_impl(self, instance):
        """Migrate events context_ids to use binary format, return True if completed."""
        _to_bytes = _context_id_to_bytes
        session_maker = instance.get_session
        _LOGGER.debug('Migrating context_ids to binary format')
        with session_scope(session=session_maker()) as session:
            if (events := session.execute(find_events_context_ids_to_migrate(instance.max_bind_vars)).all()):
                session.execute(update(Events), [{'event_id': event_id, 'context_id': None, 'context_id_bin': _to_bytes(context_id) or _generate_ulid_bytes_at_time(time_fired_ts), 'context_user_id': None, 'context_user_id_bin': _to_bytes(context_user_id), 'context_parent_id': None, 'context_parent_id_bin': _to_bytes(context_parent_id)} for event_id, time_fired_ts, context_id, context_user_id, context_parent_id in events])
            is_done = not events
        _LOGGER.debug('Migrating events context_ids to binary format: done=%s', is_done)
        return DataMigrationStatus(needs_migrate=not is_done, migration_done=is_done)

    def needs_migrate_query(self):
        """Return the query to check if the migration needs to run."""
        return has_events_context_ids_to_migrate()

class EventTypeIDMigration(BaseMigrationWithQuery, BaseOffLineMigration):
    """Migration to migrate event_type to event_type_ids."""
    required_schema_version = EVENT_TYPE_IDS_SCHEMA_VERSION
    max_initial_schema_version = EVENT_TYPE_IDS_SCHEMA_VERSION - 1
    migration_id = 'event_type_id_migration'

    def migrate_data_impl(self, instance):
        """Migrate event_type to event_type_ids, return True if completed."""
        session_maker = instance.get_session
        _LOGGER.debug('Migrating event_types')
        event_type_manager = instance.event_type_manager
        with session_scope(session=session_maker()) as session:
            if (events := session.execute(find_event_type_to_migrate(instance.max_bind_vars)).all()):
                event_types = {event_type for _, event_type in events}
                if None in event_types:
                    event_types.remove(None)
                    event_types.add(_EMPTY_EVENT_TYPE)
                event_type_to_id = event_type_manager.get_many(event_types, session)
                if (missing_event_types := {event_type for event_type, event_id in event_type_to_id.items() if event_id is None}):
                    missing_db_event_types = [EventTypes(event_type=event_type) for event_type in missing_event_types]
                    session.add_all(missing_db_event_types)
                    session.flush()
                    for db_event_type in missing_db_event_types:
                        assert db_event_type.event_type is not None, 'event_type should never be None'
                        event_type_to_id[db_event_type.event_type] = db_event_type.event_type_id
                        event_type_manager.clear_non_existent(db_event_type.event_type)
                session.execute(update(Events), [{'event_id': event_id, 'event_type': None, 'event_type_id': event_type_to_id[_EMPTY_EVENT_TYPE if event_type is None else event_type]} for event_id, event_type in events])
            is_done = not events
        _LOGGER.debug('Migrating event_types done=%s', is_done)
        return DataMigrationStatus(needs_migrate=not is_done, migration_done=is_done)

    def needs_migrate_query(self):
        """Check if the data is migrated."""
        return has_event_type_to_migrate()

class EntityIDMigration(BaseMigrationWithQuery, BaseOffLineMigration):
    """Migration to migrate entity_ids to states_meta."""
    required_schema_version = STATES_META_SCHEMA_VERSION
    max_initial_schema_version = STATES_META_SCHEMA_VERSION - 1
    migration_id = 'entity_id_migration'

    def migrate_data_impl(self, instance):
        """Migrate entity_ids to states_meta, return True if completed.

        We do this in two steps because we need the history queries to work
        while we are migrating.

        1. Link the states to the states_meta table
        2. Remove the entity_id column from the states table (in post_migrate_entity_ids)
        """
        _LOGGER.debug('Migrating entity_ids')
        states_meta_manager = instance.states_meta_manager
        with session_scope(session=instance.get_session()) as session:
            if (states := session.execute(find_entity_ids_to_migrate(instance.max_bind_vars)).all()):
                entity_ids = {entity_id for _, entity_id in states}
                if None in entity_ids:
                    entity_ids.remove(None)
                    entity_ids.add(_EMPTY_ENTITY_ID)
                entity_id_to_metadata_id = states_meta_manager.get_many(entity_ids, session, True)
                if (missing_entity_ids := {entity_id for entity_id, metadata_id in entity_id_to_metadata_id.items() if metadata_id is None}):
                    missing_states_metadata = [StatesMeta(entity_id=entity_id) for entity_id in missing_entity_ids]
                    session.add_all(missing_states_metadata)
                    session.flush()
                    for db_states_metadata in missing_states_metadata:
                        assert db_states_metadata.entity_id is not None, 'entity_id should never be None'
                        entity_id_to_metadata_id[db_states_metadata.entity_id] = db_states_metadata.metadata_id
                session.execute(update(States), [{'state_id': state_id, 'metadata_id': entity_id_to_metadata_id[_EMPTY_ENTITY_ID if entity_id is None else entity_id]} for state_id, entity_id in states])
            is_done = not states
        _LOGGER.debug('Migrating entity_ids done=%s', is_done)
        return DataMigrationStatus(needs_migrate=not is_done, migration_done=is_done)

    def needs_migrate_query(self):
        """Check if the data is migrated."""
        return has_entity_ids_to_migrate()

class EventIDPostMigration(BaseRunTimeMigration):
    """Migration to remove old event_id index from states."""
    migration_id = 'event_id_post_migration'
    max_initial_schema_version = LEGACY_STATES_EVENT_ID_INDEX_SCHEMA_VERSION - 1
    task = MigrationTask
    migration_version = 2

    def migrate_data_impl(self, instance):
        """Remove old event_id index from states, returns True if completed.

        We used to link states to events using the event_id column but we no
        longer store state changed events in the events table.

        If all old states have been purged and existing states are in the new
        format we can drop the index since it can take up ~10MB per 1M rows.
        """
        session_maker = instance.get_session
        _LOGGER.debug('Cleanup legacy entity_ids')
        with session_scope(session=session_maker()) as session:
            result = session.execute(has_used_states_event_ids()).scalar()
            all_gone = not result
        fk_remove_ok = False
        if all_gone:
            assert instance.engine is not None, 'engine should never be None'
            if instance.dialect_name == SupportedDialect.SQLITE:
                fk_remove_ok = rebuild_sqlite_table(session_maker, instance.engine, States)
            else:
                try:
                    _drop_foreign_key_constraints(session_maker, instance.engine, TABLE_STATES, 'event_id')
                except (InternalError, OperationalError):
                    fk_remove_ok = False
                else:
                    fk_remove_ok = True
            if fk_remove_ok:
                _drop_index(session_maker, 'states', LEGACY_STATES_EVENT_ID_INDEX)
                instance.use_legacy_events_index = False
        return DataMigrationStatus(needs_migrate=False, migration_done=fk_remove_ok)

    @staticmethod
    def _legacy_event_id_foreign_key_exists(instance):
        """Check if the legacy event_id foreign key exists."""
        engine = instance.engine
        assert engine is not None
        inspector = sqlalchemy.inspect(engine)
        return bool(next((fk for fk in inspector.get_foreign_keys(TABLE_STATES) if fk['constrained_columns'] == ['event_id']), None))

    def needs_migrate_impl(self, instance, session):
        """Return if the migration needs to run."""
        if self.start_schema_version <= LEGACY_STATES_EVENT_ID_INDEX_SCHEMA_VERSION:
            return DataMigrationStatus(needs_migrate=False, migration_done=False)
        if get_index_by_name(session, TABLE_STATES, LEGACY_STATES_EVENT_ID_INDEX) is not None or self._legacy_event_id_foreign_key_exists(instance):
            instance.use_legacy_events_index = True
            return DataMigrationStatus(needs_migrate=True, migration_done=False)
        return DataMigrationStatus(needs_migrate=False, migration_done=True)

class EntityIDPostMigration(BaseMigrationWithQuery, BaseOffLineMigration):
    """Migration to remove old entity_id strings from states.

    Introduced in HA Core 2023.4 by PR #89557.
    """
    migration_id = 'entity_id_post_migration'
    max_initial_schema_version = STATES_META_SCHEMA_VERSION - 1
    index_to_drop = (TABLE_STATES, LEGACY_STATES_ENTITY_ID_LAST_UPDATED_TS_INDEX, LegacyBase)

    def migrate_data_impl(self, instance):
        """Migrate some data, returns True if migration is completed."""
        is_done = post_migrate_entity_ids(instance)
        return DataMigrationStatus(needs_migrate=not is_done, migration_done=is_done)

    def needs_migrate_query(self):
        """Check if the data is migrated."""
        return has_used_states_entity_ids()
NON_LIVE_DATA_MIGRATORS = (StatesContextIDMigration, EventsContextIDMigration, EventTypeIDMigration, EntityIDMigration, EntityIDPostMigration)
LIVE_DATA_MIGRATORS = (EventIDPostMigration,)

def _mark_migration_done(session, migration):
    """Mark a migration as done in the database."""
    session.merge(MigrationChanges(migration_id=migration.migration_id, version=migration.migration_version))

def rebuild_sqlite_table(session_maker, engine, table):
    """Rebuild an SQLite table.

    This must only be called after all migrations are complete
    and the database is in a consistent state.

    If the table is not migrated to the current schema this
    will likely fail.
    """
    table_table = cast(Table, table.__table__)
    orig_name = table_table.name
    temp_name = f'{table_table.name}_temp_{int(time())}'
    _LOGGER.warning('Rebuilding SQLite table %s; %s', orig_name, MIGRATION_NOTE_WHILE)
    try:
        with session_scope(session=session_maker()) as session:
            session.connection().execute(text('PRAGMA foreign_keys=OFF'))
        with session_scope(session=session_maker()) as session:
            new_sql = str(CreateTable(table_table).compile(engine)).strip('\n') + ';'
            source_sql = f'CREATE TABLE {orig_name}'
            replacement_sql = f'CREATE TABLE {temp_name}'
            assert source_sql in new_sql, f'{source_sql} should be in new_sql'
            new_sql = new_sql.replace(source_sql, replacement_sql)
            session.execute(text(new_sql))
            column_names = ','.join([column.name for column in table_table.columns])
            sql = f'INSERT INTO {temp_name} SELECT {column_names} FROM {orig_name};'
            session.execute(text(sql))
            session.execute(text(f'DROP TABLE {orig_name}'))
            session.execute(text(f'ALTER TABLE {temp_name} RENAME TO {orig_name}'))
            for index in table_table.indexes:
                index.create(session.connection())
            session.execute(text('PRAGMA foreign_key_check'))
            session.commit()
    except SQLAlchemyError:
        _LOGGER.exception('Error recreating SQLite table %s', table_table.name)
        return False
    else:
        _LOGGER.warning('Rebuilding SQLite table %s finished', orig_name)
        return True
    finally:
        with session_scope(session=session_maker()) as session:
            session.connection().execute(text('PRAGMA foreign_keys=ON'))