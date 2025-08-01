#!/usr/bin/env python3
"""Test util methods."""
from contextlib import AbstractContextManager, nullcontext as does_not_raise
from datetime import UTC, datetime, timedelta
import os
from pathlib import Path
import sqlite3
import threading
from typing import Any, Callable, Iterator, List, Optional, Tuple, Type, Union
from unittest.mock import MagicMock, Mock, patch

import pytest
from sqlalchemy import lambda_stmt, text
from sqlalchemy.engine.result import ChunkedIteratorResult
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.sql.elements import TextClause
from sqlalchemy.sql.lambdas import StatementLambdaElement

from homeassistant.components import recorder
from homeassistant.components.recorder import Recorder, util
from homeassistant.components.recorder.const import DOMAIN, SQLITE_URL_PREFIX, SupportedDialect
from homeassistant.components.recorder.db_schema import RecorderRuns
from homeassistant.components.recorder.history.modern import _get_single_entity_start_time_stmt
from homeassistant.components.recorder.models import UnsupportedDialect, process_timestamp
from homeassistant.components.recorder.util import MIN_VERSION_SQLITE, RETRYABLE_MYSQL_ERRORS, database_job_retry_wrapper, end_incomplete_runs, is_second_sunday, resolve_period, retryable_database_job, retryable_database_job_method, session_scope
from homeassistant.const import EVENT_HOMEASSISTANT_STOP
from homeassistant.core import HomeAssistant
from homeassistant.helpers import issue_registry as ir
from homeassistant.util import dt as dt_util

from .common import async_wait_recording_done, corrupt_db_file, run_information_with_session
from tests.common import async_test_home_assistant
from tests.typing import RecorderInstanceContextManager, RecorderInstanceGenerator

@pytest.fixture
async def mock_recorder_before_hass(async_test_recorder: Any) -> None:
    """Set up recorder."""
    ...

@pytest.fixture
def setup_recorder(recorder_mock: Any) -> None:
    """Set up recorder."""
    ...

async def test_session_scope_not_setup(hass: HomeAssistant, setup_recorder: Any) -> None:
    """Try to create a session scope when not setup."""
    with patch.object(util.get_instance(hass), 'get_session', return_value=None), pytest.raises(RuntimeError), util.session_scope(hass=hass):
        pass

async def test_recorder_bad_execute(hass: HomeAssistant, setup_recorder: Any) -> None:
    """Bad execute, retry 3 times."""
    def to_native(validate_entity_id: bool = True) -> Any:
        """Raise exception."""
        raise SQLAlchemyError

    mck1: MagicMock = MagicMock()
    mck1.to_native = to_native
    with pytest.raises(SQLAlchemyError), patch('homeassistant.components.recorder.core.time.sleep') as e_mock:
        util.execute((mck1,), to_native=True)
    assert e_mock.call_count == 2

def test_validate_or_move_away_sqlite_database(hass: HomeAssistant, tmp_path: Path, caplog: Any) -> None:
    """Ensure a malformed sqlite database is moved away."""
    test_dir: Path = tmp_path.joinpath('test_validate_or_move_away_sqlite_database')
    test_dir.mkdir()
    test_db_file: str = f'{test_dir}/broken.db'
    dburl: str = f'{SQLITE_URL_PREFIX}{test_db_file}'
    assert util.validate_sqlite_database(test_db_file) is False
    assert os.path.exists(test_db_file) is True
    assert util.validate_or_move_away_sqlite_database(dburl) is False
    corrupt_db_file(test_db_file)
    assert util.validate_sqlite_database(dburl) is False
    assert util.validate_or_move_away_sqlite_database(dburl) is False
    assert 'corrupt or malformed' in caplog.text
    assert util.validate_sqlite_database(dburl) is False
    assert util.validate_or_move_away_sqlite_database(dburl) is True

@pytest.mark.skip_on_db_engine(['mysql', 'postgresql'])
@pytest.mark.usefixtures('skip_by_db_engine')
@pytest.mark.parametrize('persistent_database', [True])
@pytest.mark.usefixtures('hass_storage')
async def test_last_run_was_recently_clean(async_setup_recorder_instance: Callable[..., Any]) -> None:
    """Test we can check if the last recorder run was recently clean.

    This is only implemented for SQLite.
    """
    config: dict = {recorder.CONF_COMMIT_INTERVAL: 1}
    async with async_test_home_assistant() as hass:
        return_values: List[bool] = []
        real_last_run_was_recently_clean: Callable[[Any], bool] = util.last_run_was_recently_clean

        def _last_run_was_recently_clean(cursor: Any) -> bool:
            return_values.append(real_last_run_was_recently_clean(cursor))
            return return_values[-1]
        with patch('homeassistant.components.recorder.util.last_run_was_recently_clean', wraps=_last_run_was_recently_clean) as last_run_was_recently_clean_mock:
            await async_setup_recorder_instance(hass, config)
            await hass.async_block_till_done()
            last_run_was_recently_clean_mock.assert_not_called()
        hass.bus.async_fire(EVENT_HOMEASSISTANT_STOP)
        await hass.async_block_till_done()
        await hass.async_stop()
    async with async_test_home_assistant() as hass:
        with patch('homeassistant.components.recorder.util.last_run_was_recently_clean', wraps=_last_run_was_recently_clean) as last_run_was_recently_clean_mock:
            await async_setup_recorder_instance(hass, config)
            last_run_was_recently_clean_mock.assert_called_once()
            assert return_values[-1] is True
        hass.bus.async_fire(EVENT_HOMEASSISTANT_STOP)
        await hass.async_block_till_done()
        await hass.async_stop()
    thirty_min_future_time: datetime = dt_util.utcnow() + timedelta(minutes=30)
    async with async_test_home_assistant() as hass:
        with patch('homeassistant.components.recorder.util.last_run_was_recently_clean', wraps=_last_run_was_recently_clean) as last_run_was_recently_clean_mock, patch('homeassistant.components.recorder.core.dt_util.utcnow', return_value=thirty_min_future_time):
            await async_setup_recorder_instance(hass, config)
            last_run_was_recently_clean_mock.assert_called_once()
            assert return_values[-1] is False
        hass.bus.async_fire(EVENT_HOMEASSISTANT_STOP)
        await hass.async_block_till_done()
        await hass.async_stop()

@pytest.mark.parametrize('mysql_version', ['10.3.0-MariaDB', '8.0.0'])
def test_setup_connection_for_dialect_mysql(mysql_version: str) -> None:
    """Test setting up the connection for a mysql dialect."""
    instance_mock: MagicMock = MagicMock()
    execute_args: List[str] = []
    close_mock: MagicMock = MagicMock()

    def execute_mock(statement: Any) -> None:
        nonlocal execute_args
        execute_args.append(statement)

    def fetchall_mock() -> Optional[List[List[str]]]:
        nonlocal execute_args
        if execute_args[-1] == 'SELECT VERSION()':
            return [[mysql_version]]
        return None

    def _make_cursor_mock(*args: Any, **kwargs: Any) -> MagicMock:
        return MagicMock(execute=execute_mock, close=close_mock, fetchall=fetchall_mock)
    dbapi_connection: MagicMock = MagicMock(cursor=_make_cursor_mock)
    util.setup_connection_for_dialect(instance_mock, 'mysql', dbapi_connection, True)
    assert len(execute_args) == 3
    assert execute_args[0] == 'SET session wait_timeout=28800'
    assert execute_args[1] == 'SELECT VERSION()'
    assert execute_args[2] == "SET time_zone = '+00:00'"

@pytest.mark.parametrize('sqlite_version', [str(MIN_VERSION_SQLITE)])
def test_setup_connection_for_dialect_sqlite(sqlite_version: str) -> None:
    """Test setting up the connection for a sqlite dialect."""
    instance_mock: MagicMock = MagicMock()
    execute_args: List[str] = []
    close_mock: MagicMock = MagicMock()

    def execute_mock(statement: Any) -> None:
        nonlocal execute_args
        execute_args.append(statement)

    def fetchall_mock() -> Optional[List[List[str]]]:
        nonlocal execute_args
        if execute_args[-1] == 'SELECT sqlite_version()':
            return [[sqlite_version]]
        return None

    def _make_cursor_mock(*args: Any, **kwargs: Any) -> MagicMock:
        return MagicMock(execute=execute_mock, close=close_mock, fetchall=fetchall_mock)
    dbapi_connection: MagicMock = MagicMock(cursor=_make_cursor_mock)
    ret: Optional[Any] = util.setup_connection_for_dialect(instance_mock, 'sqlite', dbapi_connection, True)
    assert ret is not None
    assert len(execute_args) == 5
    assert execute_args[0] == 'PRAGMA journal_mode=WAL'
    assert execute_args[1] == 'SELECT sqlite_version()'
    assert execute_args[2] == 'PRAGMA cache_size = -16384'
    assert execute_args[3] == 'PRAGMA synchronous=NORMAL'
    assert execute_args[4] == 'PRAGMA foreign_keys=ON'
    execute_args.clear()
    ret = util.setup_connection_for_dialect(instance_mock, 'sqlite', dbapi_connection, False)
    assert ret is None
    assert len(execute_args) == 3
    assert execute_args[0] == 'PRAGMA cache_size = -16384'
    assert execute_args[1] == 'PRAGMA synchronous=NORMAL'
    assert execute_args[2] == 'PRAGMA foreign_keys=ON'

@pytest.mark.parametrize('sqlite_version', [str(MIN_VERSION_SQLITE)])
def test_setup_connection_for_dialect_sqlite_zero_commit_interval(sqlite_version: str) -> None:
    """Test setting up the connection for a sqlite dialect with a zero commit interval."""
    instance_mock: MagicMock = MagicMock(commit_interval=0)
    execute_args: List[str] = []
    close_mock: MagicMock = MagicMock()

    def execute_mock(statement: Any) -> None:
        nonlocal execute_args
        execute_args.append(statement)

    def fetchall_mock() -> Optional[List[List[str]]]:
        nonlocal execute_args
        if execute_args[-1] == 'SELECT sqlite_version()':
            return [[sqlite_version]]
        return None

    def _make_cursor_mock(*args: Any, **kwargs: Any) -> MagicMock:
        return MagicMock(execute=execute_mock, close=close_mock, fetchall=fetchall_mock)
    dbapi_connection: MagicMock = MagicMock(cursor=_make_cursor_mock)
    ret: Optional[Any] = util.setup_connection_for_dialect(instance_mock, 'sqlite', dbapi_connection, True)
    assert ret is not None
    assert len(execute_args) == 5
    assert execute_args[0] == 'PRAGMA journal_mode=WAL'
    assert execute_args[1] == 'SELECT sqlite_version()'
    assert execute_args[2] == 'PRAGMA cache_size = -16384'
    assert execute_args[3] == 'PRAGMA synchronous=FULL'
    assert execute_args[4] == 'PRAGMA foreign_keys=ON'
    execute_args.clear()
    ret = util.setup_connection_for_dialect(instance_mock, 'sqlite', dbapi_connection, False)
    assert ret is None
    assert len(execute_args) == 3
    assert execute_args[0] == 'PRAGMA cache_size = -16384'
    assert execute_args[1] == 'PRAGMA synchronous=FULL'
    assert execute_args[2] == 'PRAGMA foreign_keys=ON'

@pytest.mark.parametrize(('mysql_version', 'message'), [
    ('10.2.0-MariaDB', 'Version 10.2.0 of MariaDB is not supported; minimum supported version is 10.3.0.'),
    ('5.7.26-0ubuntu0.18.04.1', 'Version 5.7.26 of MySQL is not supported; minimum supported version is 8.0.0.'),
    ('some_random_response', 'Version some_random_response of MySQL is not supported; minimum supported version is 8.0.0.')
])
def test_fail_outdated_mysql(caplog: Any, mysql_version: str, message: str) -> None:
    """Test setting up the connection for an outdated mysql version."""
    instance_mock: MagicMock = MagicMock()
    execute_args: List[str] = []
    close_mock: MagicMock = MagicMock()

    def execute_mock(statement: Any) -> None:
        nonlocal execute_args
        execute_args.append(statement)

    def fetchall_mock() -> Optional[List[List[str]]]:
        nonlocal execute_args
        if execute_args[-1] == 'SELECT VERSION()':
            return [[mysql_version]]
        return None

    def _make_cursor_mock(*args: Any, **kwargs: Any) -> MagicMock:
        return MagicMock(execute=execute_mock, close=close_mock, fetchall=fetchall_mock)
    dbapi_connection: MagicMock = MagicMock(cursor=_make_cursor_mock)
    with pytest.raises(UnsupportedDialect):
        util.setup_connection_for_dialect(instance_mock, 'mysql', dbapi_connection, True)
    assert message in caplog.text

@pytest.mark.parametrize('mysql_version', ['10.3.0', '8.0.0'])
def test_supported_mysql(caplog: Any, mysql_version: str) -> None:
    """Test setting up the connection for a supported mysql version."""
    instance_mock: MagicMock = MagicMock()
    execute_args: List[str] = []
    close_mock: MagicMock = MagicMock()

    def execute_mock(statement: Any) -> None:
        nonlocal execute_args
        execute_args.append(statement)

    def fetchall_mock() -> Optional[List[List[str]]]:
        nonlocal execute_args
        if execute_args[-1] == 'SELECT VERSION()':
            return [[mysql_version]]
        return None

    def _make_cursor_mock(*args: Any, **kwargs: Any) -> MagicMock:
        return MagicMock(execute=execute_mock, close=close_mock, fetchall=fetchall_mock)
    dbapi_connection: MagicMock = MagicMock(cursor=_make_cursor_mock)
    util.setup_connection_for_dialect(instance_mock, 'mysql', dbapi_connection, True)
    assert 'minimum supported version' not in caplog.text

@pytest.mark.parametrize(('pgsql_version', 'message'), [
    ('11.12 (Debian 11.12-1.pgdg100+1)', 'Version 11.12 of PostgreSQL is not supported; minimum supported version is 12.0.'),
    ('9.2.10', 'Version 9.2.10 of PostgreSQL is not supported; minimum supported version is 12.0.'),
    ('unexpected', 'Version unexpected of PostgreSQL is not supported; minimum supported version is 12.0.')
])
def test_fail_outdated_pgsql(caplog: Any, pgsql_version: str, message: str) -> None:
    """Test setting up the connection for an outdated PostgreSQL version."""
    instance_mock: MagicMock = MagicMock()
    execute_args: List[str] = []
    close_mock: MagicMock = MagicMock()

    def execute_mock(statement: Any) -> None:
        nonlocal execute_args
        execute_args.append(statement)

    def fetchall_mock() -> Optional[List[List[str]]]:
        nonlocal execute_args
        if execute_args[-1] == 'SHOW server_version':
            return [[pgsql_version]]
        return None

    def _make_cursor_mock(*args: Any, **kwargs: Any) -> MagicMock:
        return MagicMock(execute=execute_mock, close=close_mock, fetchall=fetchall_mock)
    dbapi_connection: MagicMock = MagicMock(cursor=_make_cursor_mock)
    with pytest.raises(UnsupportedDialect):
        util.setup_connection_for_dialect(instance_mock, 'postgresql', dbapi_connection, True)
    assert message in caplog.text

@pytest.mark.parametrize('pgsql_version', ['14.0 (Debian 14.0-1.pgdg110+1)'])
def test_supported_pgsql(caplog: Any, pgsql_version: str) -> None:
    """Test setting up the connection for a supported PostgreSQL version."""
    instance_mock: MagicMock = MagicMock()
    execute_args: List[str] = []
    close_mock: MagicMock = MagicMock()

    def execute_mock(statement: Any) -> None:
        nonlocal execute_args
        execute_args.append(statement)

    def fetchall_mock() -> Optional[List[List[str]]]:
        nonlocal execute_args
        if execute_args[-1] == 'SHOW server_version':
            return [[pgsql_version]]
        return None

    def _make_cursor_mock(*args: Any, **kwargs: Any) -> MagicMock:
        return MagicMock(execute=execute_mock, close=close_mock, fetchall=fetchall_mock)
    dbapi_connection: MagicMock = MagicMock(cursor=_make_cursor_mock)
    database_engine: Any = util.setup_connection_for_dialect(instance_mock, 'postgresql', dbapi_connection, True)
    assert 'minimum supported version' not in caplog.text
    assert database_engine is not None
    assert database_engine.optimizer.slow_range_in_select is True

@pytest.mark.parametrize(('sqlite_version', 'message'), [
    ('3.30.0', 'Version 3.30.0 of SQLite is not supported; minimum supported version is 3.40.1.'),
    ('2.0.0', 'Version 2.0.0 of SQLite is not supported; minimum supported version is 3.40.1.')
])
def test_fail_outdated_sqlite(caplog: Any, sqlite_version: str, message: str) -> None:
    """Test setting up the connection for an outdated sqlite version."""
    instance_mock: MagicMock = MagicMock()
    execute_args: List[str] = []
    close_mock: MagicMock = MagicMock()

    def execute_mock(statement: Any) -> None:
        nonlocal execute_args
        execute_args.append(statement)

    def fetchall_mock() -> Optional[List[List[str]]]:
        nonlocal execute_args
        if execute_args[-1] == 'SELECT sqlite_version()':
            return [[sqlite_version]]
        return None

    def _make_cursor_mock(*args: Any, **kwargs: Any) -> MagicMock:
        return MagicMock(execute=execute_mock, close=close_mock, fetchall=fetchall_mock)
    dbapi_connection: MagicMock = MagicMock(cursor=_make_cursor_mock)
    with pytest.raises(UnsupportedDialect):
        util.setup_connection_for_dialect(instance_mock, 'sqlite', dbapi_connection, True)
    assert message in caplog.text

@pytest.mark.parametrize('sqlite_version', ['3.40.1', '3.41.0'])
def test_supported_sqlite(caplog: Any, sqlite_version: str) -> None:
    """Test setting up the connection for a supported sqlite version."""
    instance_mock: MagicMock = MagicMock()
    execute_args: List[str] = []
    close_mock: MagicMock = MagicMock()

    def execute_mock(statement: Any) -> None:
        nonlocal execute_args
        execute_args.append(statement)

    def fetchall_mock() -> Optional[List[List[str]]]:
        nonlocal execute_args
        if execute_args[-1] == 'SELECT sqlite_version()':
            return [[sqlite_version]]
        return None

    def _make_cursor_mock(*args: Any, **kwargs: Any) -> MagicMock:
        return MagicMock(execute=execute_mock, close=close_mock, fetchall=fetchall_mock)
    dbapi_connection: MagicMock = MagicMock(cursor=_make_cursor_mock)
    database_engine: Any = util.setup_connection_for_dialect(instance_mock, 'sqlite', dbapi_connection, True)
    assert 'minimum supported version' not in caplog.text
    assert database_engine is not None
    assert database_engine.optimizer.slow_range_in_select is False

@pytest.mark.parametrize(('dialect', 'message'), [
    ('mssql', 'Database mssql is not supported'),
    ('oracle', 'Database oracle is not supported'),
    ('some_db', 'Database some_db is not supported')
])
def test_warn_unsupported_dialect(caplog: Any, dialect: str, message: str) -> None:
    """Test setting up the connection for an outdated sqlite version."""
    instance_mock: MagicMock = MagicMock()
    dbapi_connection: MagicMock = MagicMock()
    with pytest.raises(UnsupportedDialect):
        util.setup_connection_for_dialect(instance_mock, dialect, dbapi_connection, True)
    assert message in caplog.text

@pytest.mark.parametrize(('mysql_version', 'min_version'), [
    ('10.5.16-MariaDB', '10.5.17'),
    ('10.6.8-MariaDB', '10.6.9'),
    ('10.7.1-MariaDB', '10.7.5'),
    ('10.8.0-MariaDB', '10.8.4')
])
async def test_issue_for_mariadb_with_MDEV_25020(hass: HomeAssistant, caplog: Any, mysql_version: str, min_version: str, issue_registry: Any) -> None:
    """Test we create an issue for MariaDB versions affected.

    See https://jira.mariadb.org/browse/MDEV-25020.
    """
    instance_mock: MagicMock = MagicMock()
    instance_mock.hass = hass
    execute_args: List[str] = []
    close_mock: MagicMock = MagicMock()

    def execute_mock(statement: Any) -> None:
        nonlocal execute_args
        execute_args.append(statement)

    def fetchall_mock() -> Optional[List[List[str]]]:
        nonlocal execute_args
        if execute_args[-1] == 'SELECT VERSION()':
            return [[mysql_version]]
        return None

    def _make_cursor_mock(*args: Any, **kwargs: Any) -> MagicMock:
        return MagicMock(execute=execute_mock, close=close_mock, fetchall=fetchall_mock)
    dbapi_connection: MagicMock = MagicMock(cursor=_make_cursor_mock)
    database_engine: Any = await hass.async_add_executor_job(util.setup_connection_for_dialect, instance_mock, 'mysql', dbapi_connection, True)
    await hass.async_block_till_done()
    issue = issue_registry.async_get_issue(DOMAIN, 'maria_db_range_index_regression')
    assert issue is not None
    assert issue.translation_placeholders == {'min_version': min_version}
    assert database_engine is not None
    assert database_engine.optimizer.slow_range_in_select is True

@pytest.mark.parametrize('mysql_version', ['10.5.17-MariaDB', '10.6.9-MariaDB', '10.7.5-MariaDB', '10.8.4-MariaDB', '10.9.1-MariaDB'])
async def test_no_issue_for_mariadb_with_MDEV_25020(hass: HomeAssistant, caplog: Any, mysql_version: str, issue_registry: Any) -> None:
    """Test we do not create an issue for MariaDB versions not affected.

    See https://jira.mariadb.org/browse/MDEV-25020.
    """
    instance_mock: MagicMock = MagicMock()
    instance_mock.hass = hass
    execute_args: List[str] = []
    close_mock: MagicMock = MagicMock()

    def execute_mock(statement: Any) -> None:
        nonlocal execute_args
        execute_args.append(statement)

    def fetchall_mock() -> Optional[List[List[str]]]:
        nonlocal execute_args
        if execute_args[-1] == 'SELECT VERSION()':
            return [[mysql_version]]
        return None

    def _make_cursor_mock(*args: Any, **kwargs: Any) -> MagicMock:
        return MagicMock(execute=execute_mock, close=close_mock, fetchall=fetchall_mock)
    dbapi_connection: MagicMock = MagicMock(cursor=_make_cursor_mock)
    database_engine: Any = await hass.async_add_executor_job(util.setup_connection_for_dialect, instance_mock, 'mysql', dbapi_connection, True)
    await hass.async_block_till_done()
    issue = issue_registry.async_get_issue(DOMAIN, 'maria_db_range_index_regression')
    assert issue is None
    assert database_engine is not None
    assert database_engine.optimizer.slow_range_in_select is False

@pytest.mark.skip_on_db_engine(['mysql', 'postgresql'])
@pytest.mark.usefixtures('skip_by_db_engine')
async def test_basic_sanity_check(hass: HomeAssistant, setup_recorder: Any, recorder_db_url: str) -> None:
    """Test the basic sanity checks with a missing table.

    This test is specific for SQLite.
    """
    cursor: Any = util.get_instance(hass).engine.raw_connection().cursor()
    assert util.basic_sanity_check(cursor) is True
    cursor.execute('DROP TABLE states;')
    with pytest.raises(sqlite3.DatabaseError):
        util.basic_sanity_check(cursor)

@pytest.mark.skip_on_db_engine(['mysql', 'postgresql'])
@pytest.mark.usefixtures('skip_by_db_engine')
async def test_combined_checks(hass: HomeAssistant, setup_recorder: Any, caplog: Any, recorder_db_url: str) -> None:
    """Run Checks on the open database.

    This test is specific for SQLite.
    """
    instance: Any = util.get_instance(hass)
    instance.db_retry_wait = 0
    cursor: Any = instance.engine.raw_connection().cursor()
    assert util.run_checks_on_open_db('fake_db_path', cursor) is None
    assert 'could not validate that the sqlite3 database' in caplog.text
    caplog.clear()
    with patch('homeassistant.components.recorder.util.basic_sanity_check', return_value=False):
        caplog.clear()
        assert util.run_checks_on_open_db('fake_db_path', cursor) is None
        assert 'could not validate that the sqlite3 database' in caplog.text
    with patch('homeassistant.components.recorder.util.last_run_was_recently_clean'):
        caplog.clear()
        assert util.run_checks_on_open_db('fake_db_path', cursor) is None
        assert 'restarted cleanly and passed the basic sanity check' in caplog.text
    caplog.clear()
    with patch('homeassistant.components.recorder.util.last_run_was_recently_clean', side_effect=sqlite3.DatabaseError), pytest.raises(sqlite3.DatabaseError):
        util.run_checks_on_open_db('fake_db_path', cursor)
    caplog.clear()
    with patch('homeassistant.components.recorder.util.last_run_was_recently_clean', side_effect=sqlite3.DatabaseError), pytest.raises(sqlite3.DatabaseError):
        util.run_checks_on_open_db('fake_db_path', cursor)
    cursor.execute('DROP TABLE events;')
    caplog.clear()
    with pytest.raises(sqlite3.DatabaseError):
        util.run_checks_on_open_db('fake_db_path', cursor)
    caplog.clear()
    with pytest.raises(sqlite3.DatabaseError):
        util.run_checks_on_open_db('fake_db_path', cursor)

async def test_end_incomplete_runs(hass: HomeAssistant, setup_recorder: Any, caplog: Any) -> None:
    """Ensure we can end incomplete runs."""
    with session_scope(hass=hass) as session:
        run_info: RecorderRuns = run_information_with_session(session)
        assert isinstance(run_info, RecorderRuns)
        assert run_info.closed_incorrect is False
        now: datetime = dt_util.utcnow()
        end_incomplete_runs(session, now)
        run_info = run_information_with_session(session)
        assert run_info.closed_incorrect is True
        assert process_timestamp(run_info.end) == now
        session.flush()
        later: datetime = dt_util.utcnow()
        end_incomplete_runs(session, later)
        run_info = run_information_with_session(session)
        assert process_timestamp(run_info.end) == now
    assert 'Ended unfinished session' in caplog.text

@pytest.mark.skip_on_db_engine(['mysql', 'postgresql'])
@pytest.mark.usefixtures('skip_by_db_engine')
async def test_periodic_db_cleanups(hass: HomeAssistant, setup_recorder: Any, recorder_db_url: str) -> None:
    """Test periodic db cleanups.

    This test is specific for SQLite.
    """
    with patch.object(util.get_instance(hass).engine, 'connect') as connect_mock:
        util.periodic_db_cleanups(util.get_instance(hass))
    text_obj: Any = connect_mock.return_value.__enter__.return_value.execute.mock_calls[0][1][0]
    assert isinstance(text_obj, TextClause)
    assert str(text_obj) == 'PRAGMA wal_checkpoint(TRUNCATE);'

@pytest.mark.skip_on_db_engine(['mysql', 'postgresql'])
@pytest.mark.usefixtures('skip_by_db_engine')
@pytest.mark.parametrize('persistent_database', [True])
async def test_write_lock_db(async_setup_recorder_instance: Callable[..., Any], hass: HomeAssistant, recorder_db_url: str) -> None:
    """Test database write lock.

    This is only supported for SQLite.

    Use file DB, in memory DB cannot do write locks.
    """
    config: dict = {recorder.CONF_DB_URL: recorder_db_url + '?timeout=0.1'}
    instance: Any = await async_setup_recorder_instance(hass, config)
    await hass.async_block_till_done()

    def _drop_table() -> None:
        with instance.engine.connect() as connection:
            connection.execute(text('DROP TABLE events;'))
    instance.recorder_and_worker_thread_ids.add(threading.get_ident())
    with util.write_lock_db_sqlite(instance), pytest.raises(OperationalError):
        await instance.async_add_executor_job(_drop_table)

def test_is_second_sunday() -> None:
    """Test we can find the second sunday of the month."""
    assert is_second_sunday(datetime(2022, 1, 9, 0, 0, 0, tzinfo=dt_util.UTC)) is True
    assert is_second_sunday(datetime(2022, 2, 13, 0, 0, 0, tzinfo=dt_util.UTC)) is True
    assert is_second_sunday(datetime(2022, 3, 13, 0, 0, 0, tzinfo=dt_util.UTC)) is True
    assert is_second_sunday(datetime(2022, 4, 10, 0, 0, 0, tzinfo=dt_util.UTC)) is True
    assert is_second_sunday(datetime(2022, 5, 8, 0, 0, 0, tzinfo=dt_util.UTC)) is True
    assert is_second_sunday(datetime(2022, 1, 10, 0, 0, 0, tzinfo=dt_util.UTC)) is False

def test_build_mysqldb_conv() -> None:
    """Test building the MySQLdb connect conv param."""
    mock_converters: Mock = Mock(conversions={'original': 'preserved'})
    mock_constants: Mock = Mock(FIELD_TYPE=Mock(DATETIME='DATETIME'))
    with patch.dict('sys.modules', **{'MySQLdb.constants': mock_constants, 'MySQLdb.converters': mock_converters}):
        conv = util.build_mysqldb_conv()
    assert conv['original'] == 'preserved'
    assert conv['DATETIME']('INVALID') is None
    assert conv['DATETIME']('2022-05-13T22:33:12.741') == datetime(2022, 5, 13, 22, 33, 12, 741000, tzinfo=None)

class MockExecutor:
    def __init__(self, stmt: StatementLambdaElement) -> None:
        assert isinstance(stmt, StatementLambdaElement)
        self.stmt = stmt

    def all(self) -> Any:
        global all_calls  # type: ignore
        all_calls += 1  # type: ignore
        if all_calls == 2:
            return ['mock_row']
        raise SQLAlchemyError

@patch('homeassistant.components.recorder.util.QUERY_RETRY_WAIT', 0)
async def test_execute_stmt_lambda_element(hass: HomeAssistant, setup_recorder: Any) -> None:
    """Test executing with execute_stmt_lambda_element."""
    instance: Any = recorder.get_instance(hass)
    hass.states.async_set('sensor.on', 'on')
    new_state: Any = hass.states.get('sensor.on')
    await async_wait_recording_done(hass)
    now: datetime = dt_util.utcnow()
    tomorrow: datetime = now + timedelta(days=1)
    one_week_from_now: datetime = now + timedelta(days=7)
    global all_calls
    all_calls = 0

    with session_scope(hass=hass) as session:
        metadata_id: int = instance.states_meta_manager.get('sensor.on', session, True)
        start_time_ts: float = dt_util.utcnow().timestamp()
        stmt: StatementLambdaElement = lambda_stmt(lambda: _get_single_entity_start_time_stmt(start_time_ts, metadata_id, False, False, False))
        rows: Union[List[Any], ChunkedIteratorResult] = util.execute_stmt_lambda_element(session, stmt)
        assert isinstance(rows, list)
        assert rows[0].state == new_state.state
        assert rows[0].metadata_id == metadata_id
        rows = util.execute_stmt_lambda_element(session, stmt, now, one_week_from_now)
        assert isinstance(rows, ChunkedIteratorResult)
        row = next(rows)
        assert row.state == new_state.state
        assert row.metadata_id == metadata_id
        rows = util.execute_stmt_lambda_element(session, stmt, now, one_week_from_now, orm_rows=False)
        assert not isinstance(rows, ChunkedIteratorResult)
        row = next(rows)
        assert row.state == new_state.state
        assert row.metadata_id == metadata_id
        rows = util.execute_stmt_lambda_element(session, stmt, now, tomorrow)
        assert isinstance(rows, list)
        assert rows[0].state == new_state.state
        assert rows[0].metadata_id == metadata_id
        with patch.object(session, 'execute', Mock(side_effect=lambda stmt: MockExecutor(stmt))):
            rows = util.execute_stmt_lambda_element(session, stmt, now, tomorrow)
            assert rows == ['mock_row']

@pytest.mark.parametrize(('start_time', 'periods'), [
    (datetime(2022, 10, 21, 7, 25, 50, 123, tzinfo=UTC),
     {('hour', 0): ('2022-10-21T07:00:00', '2022-10-21T08:00:00'),
      ('hour', -1): ('2022-10-21T06:00:00', '2022-10-21T07:00:00'),
      ('hour', 1): ('2022-10-21T08:00:00', '2022-10-21T09:00:00'),
      ('day', 0): ('2022-10-21T07:00:00', '2022-10-22T07:00:00'),
      ('day', -1): ('2022-10-20T07:00:00', '2022-10-21T07:00:00'),
      ('day', 1): ('2022-10-22T07:00:00', '2022-10-23T07:00:00'),
      ('week', 0): ('2022-10-17T07:00:00', '2022-10-24T07:00:00'),
      ('week', -1): ('2022-10-10T07:00:00', '2022-10-17T07:00:00'),
      ('week', 1): ('2022-10-24T07:00:00', '2022-10-31T07:00:00'),
      ('month', 0): ('2022-10-01T07:00:00', '2022-11-01T07:00:00'),
      ('month', -1): ('2022-09-01T07:00:00', '2022-10-01T07:00:00'),
      ('month', -12): ('2021-10-01T07:00:00', '2021-11-01T07:00:00'),
      ('month', 1): ('2022-11-01T07:00:00', '2022-12-01T08:00:00'),
      ('month', 2): ('2022-12-01T08:00:00', '2023-01-01T08:00:00'),
      ('month', 3): ('2023-01-01T08:00:00', '2023-02-01T08:00:00'),
      ('month', 12): ('2023-10-01T07:00:00', '2023-11-01T07:00:00'),
      ('month', 13): ('2023-11-01T07:00:00', '2023-12-01T08:00:00'),
      ('month', 14): ('2023-12-01T08:00:00', '2024-01-01T08:00:00'),
      ('year', 0): ('2022-01-01T08:00:00', '2023-01-01T08:00:00'),
      ('year', -1): ('2021-01-01T08:00:00', '2022-01-01T08:00:00'),
      ('year', 1): ('2023-01-01T08:00:00', '2024-01-01T08:00:00')}),
    (datetime(2024, 2, 28, 8, 25, 50, 123, tzinfo=UTC),
     {('hour', 0): ('2024-02-28T08:00:00', '2024-02-28T09:00:00'),
      ('hour', -1): ('2024-02-28T07:00:00', '2024-02-28T08:00:00'),
      ('hour', 1): ('2024-02-28T09:00:00', '2024-02-28T10:00:00'),
      ('day', 0): ('2024-02-28T08:00:00', '2024-02-29T08:00:00'),
      ('day', -1): ('2024-02-27T08:00:00', '2024-02-28T08:00:00'),
      ('day', 1): ('2024-02-29T08:00:00', '2024-03-01T08:00:00'),
      ('week', 0): ('2024-02-26T08:00:00', '2024-03-04T08:00:00'),
      ('week', -1): ('2024-02-19T08:00:00', '2024-02-26T08:00:00'),
      ('week', 1): ('2024-03-04T08:00:00', '2024-03-11T07:00:00'),
      ('month', 0): ('2024-02-01T08:00:00', '2024-03-01T08:00:00'),
      ('month', -1): ('2024-01-01T08:00:00', '2024-02-01T08:00:00'),
      ('month', -2): ('2023-12-01T08:00:00', '2024-01-01T08:00:00'),
      ('month', -3): ('2023-11-01T07:00:00', '2023-12-01T08:00:00'),
      ('month', -12): ('2023-02-01T08:00:00', '2023-03-01T08:00:00'),
      ('month', -13): ('2023-01-01T08:00:00', '2023-02-01T08:00:00'),
      ('month', -14): ('2022-12-01T08:00:00', '2023-01-01T08:00:00'),
      ('month', 1): ('2024-03-01T08:00:00', '2024-04-01T07:00:00'),
      ('year', 0): ('2024-01-01T08:00:00', '2025-01-01T08:00:00'),
      ('year', -1): ('2023-01-01T08:00:00', '2024-01-01T08:00:00'),
      ('year', 1): ('2025-01-01T08:00:00', '2026-01-01T08:00:00')})
])
async def test_resolve_period(hass: HomeAssistant, freezer: Any, start_time: datetime, periods: dict) -> None:
    """Test resolve_period."""
    assert hass.config.time_zone == 'US/Pacific'
    freezer.move_to(start_time)
    now: datetime = dt_util.utcnow()
    for period_def, expected_period in periods.items():
        start_t, end_t = resolve_period({'calendar': {'period': period_def[0], 'offset': period_def[1]}})
        assert start_t.isoformat() == f'{expected_period[0]}+00:00'
        assert end_t.isoformat() == f'{expected_period[1]}+00:00'
    assert resolve_period({}) == (None, None)
    assert resolve_period({'fixed_period': {'end_time': now}}) == (None, now)
    assert resolve_period({'fixed_period': {'start_time': now}}) == (now, None)
    assert resolve_period({'fixed_period': {'end_time': now, 'start_time': now}}) == (now, now)
    assert resolve_period({'rolling_window': {'duration': timedelta(hours=1, minutes=25)}}) == (now - timedelta(hours=1, minutes=25), now)
    assert resolve_period({'rolling_window': {'duration': timedelta(hours=1), 'offset': timedelta(minutes=-25)}}) == (now - timedelta(hours=1, minutes=25), now - timedelta(minutes=25))

NonRetryable: OperationalError = OperationalError(None, None, BaseException())
Retryable: OperationalError = OperationalError(None, None, BaseException(RETRYABLE_MYSQL_ERRORS[0], ''))

@pytest.mark.parametrize(('side_effect', 'dialect', 'retval', 'expected_result', 'num_calls'), [
    (None, SupportedDialect.MYSQL, None, does_not_raise(), 1),
    (ValueError, SupportedDialect.MYSQL, None, pytest.raises(ValueError), 1),
    (NonRetryable, SupportedDialect.MYSQL, None, pytest.raises(OperationalError), 1),
    (Retryable, SupportedDialect.MYSQL, None, pytest.raises(OperationalError), 5),
    (NonRetryable, SupportedDialect.SQLITE, None, pytest.raises(OperationalError), 1),
    (Retryable, SupportedDialect.SQLITE, None, pytest.raises(OperationalError), 1)
])
def test_database_job_retry_wrapper(side_effect: Any, dialect: SupportedDialect, retval: Any, expected_result: Any, num_calls: int) -> None:
    """Test database_job_retry_wrapper."""
    instance: Any = Mock()
    instance.db_retry_wait = 0
    instance.engine.dialect.name = dialect
    mock_job: MagicMock = Mock(side_effect=side_effect)

    @database_job_retry_wrapper('test', 5)
    def job(instance: Any, *args: Any, **kwargs: Any) -> Any:
        mock_job()
        return retval
    with expected_result:
        assert job(instance) == retval
    assert len(mock_job.mock_calls) == num_calls

@pytest.mark.parametrize(('side_effect', 'dialect', 'retval', 'expected_result'), [
    (None, SupportedDialect.MYSQL, False, does_not_raise()),
    (None, SupportedDialect.MYSQL, True, does_not_raise()),
    (ValueError, SupportedDialect.MYSQL, False, pytest.raises(ValueError)),
    (NonRetryable, SupportedDialect.MYSQL, True, does_not_raise()),
    (Retryable, SupportedDialect.MYSQL, False, does_not_raise()),
    (NonRetryable, SupportedDialect.SQLITE, True, does_not_raise()),
    (Retryable, SupportedDialect.SQLITE, True, does_not_raise())
])
def test_retryable_database_job(side_effect: Any, retval: Any, expected_result: Any, dialect: SupportedDialect) -> None:
    """Test retryable_database_job."""
    instance: Any = Mock()
    instance.db_retry_wait = 0
    instance.engine.dialect.name = dialect
    mock_job: MagicMock = Mock(side_effect=side_effect)

    @retryable_database_job(description='test')
    def job(instance: Any, *args: Any, **kwargs: Any) -> Any:
        mock_job()
        return retval
    with expected_result:
        assert job(instance) == retval
    assert len(mock_job.mock_calls) == 1

@pytest.mark.parametrize(('side_effect', 'dialect', 'retval', 'expected_result'), [
    (None, SupportedDialect.MYSQL, False, does_not_raise()),
    (None, SupportedDialect.MYSQL, True, does_not_raise()),
    (ValueError, SupportedDialect.MYSQL, False, pytest.raises(ValueError)),
    (NonRetryable, SupportedDialect.MYSQL, True, does_not_raise()),
    (Retryable, SupportedDialect.MYSQL, False, does_not_raise()),
    (NonRetryable, SupportedDialect.SQLITE, True, does_not_raise()),
    (Retryable, SupportedDialect.SQLITE, True, does_not_raise())
])
def test_retryable_database_job_method(side_effect: Any, retval: Any, expected_result: Any, dialect: SupportedDialect) -> None:
    """Test retryable_database_job_method."""
    instance: Any = Mock()
    instance.db_retry_wait = 0
    instance.engine.dialect.name = dialect
    mock_job: MagicMock = Mock(side_effect=side_effect)

    class Test:
        @retryable_database_job_method(description='test')
        def job(self, instance: Any, *args: Any, **kwargs: Any) -> Any:
            mock_job()
            return retval
    test = Test()
    with expected_result:
        assert test.job(instance) == retval
    assert len(mock_job.mock_calls) == 1
