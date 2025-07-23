"""Test util methods."""
from contextlib import AbstractContextManager, nullcontext as does_not_raise
from datetime import UTC, datetime, timedelta
import os
from pathlib import Path
import sqlite3
import threading
from typing import Any, Optional, Tuple, Dict, List, Union, cast
from unittest.mock import MagicMock, Mock, patch
from freezegun.api import FrozenDateTimeFactory
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

@pytest.fixture
def setup_recorder(recorder_mock: Any) -> None:
    """Set up recorder."""

async def test_session_scope_not_setup(hass: HomeAssistant, setup_recorder: Any) -> None:
    """Try to create a session scope when not setup."""
    with patch.object(util.get_instance(hass), 'get_session', return_value=None), pytest.raises(RuntimeError), util.session_scope(hass=hass):
        pass

async def test_recorder_bad_execute(hass: HomeAssistant, setup_recorder: Any) -> None:
    """Bad execute, retry 3 times."""

    def to_native(validate_entity_id: bool = True) -> None:
        """Raise exception."""
        raise SQLAlchemyError
    mck1 = MagicMock()
    mck1.to_native = to_native
    with pytest.raises(SQLAlchemyError), patch('homeassistant.components.recorder.core.time.sleep') as e_mock:
        util.execute((mck1,), to_native=True)
    assert e_mock.call_count == 2

def test_validate_or_move_away_sqlite_database(hass: HomeAssistant, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Ensure a malformed sqlite database is moved away."""
    test_dir = tmp_path.joinpath('test_validate_or_move_away_sqlite_database')
    test_dir.mkdir()
    test_db_file = f'{test_dir}/broken.db'
    dburl = f'{SQLITE_URL_PREFIX}{test_db_file}'
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
async def test_last_run_was_recently_clean(async_setup_recorder_instance: RecorderInstanceGenerator) -> None:
    """Test we can check if the last recorder run was recently clean.

    This is only implemented for SQLite.
    """
    config = {recorder.CONF_COMMIT_INTERVAL: 1}
    async with async_test_home_assistant() as hass:
        return_values: List[bool] = []
        real_last_run_was_recently_clean = util.last_run_was_recently_clean

        def _last_run_was_recently_clean(cursor: sqlite3.Cursor) -> bool:
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
    thirty_min_future_time = dt_util.utcnow() + timedelta(minutes=30)
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
    instance_mock = MagicMock()
    execute_args: List[str] = []
    close_mock = MagicMock()

    def execute_mock(statement: str) -> None:
        nonlocal execute_args
        execute_args.append(statement)

    def fetchall_mock() -> Optional[List[List[str]]]:
        nonlocal execute_args
        if execute_args[-1] == 'SELECT VERSION()':
            return [[mysql_version]]
        return None

    def _make_cursor_mock(*_: Any) -> MagicMock:
        return MagicMock(execute=execute_mock, close=close_mock, fetchall=fetchall_mock)
    dbapi_connection = MagicMock(cursor=_make_cursor_mock)
    util.setup_connection_for_dialect(instance_mock, 'mysql', dbapi_connection, True)
    assert len(execute_args) == 3
    assert execute_args[0] == 'SET session wait_timeout=28800'
    assert execute_args[1] == 'SELECT VERSION()'
    assert execute_args[2] == "SET time_zone = '+00:00'"

@pytest.mark.parametrize('sqlite_version', [str(MIN_VERSION_SQLITE)])
def test_setup_connection_for_dialect_sqlite(sqlite_version: str) -> None:
    """Test setting up the connection for a sqlite dialect."""
    instance_mock = MagicMock()
    execute_args: List[str] = []
    close_mock = MagicMock()

    def execute_mock(statement: str) -> None:
        nonlocal execute_args
        execute_args.append(statement)

    def fetchall_mock() -> Optional[List[List[str]]]:
        nonlocal execute_args
        if execute_args[-1] == 'SELECT sqlite_version()':
            return [[sqlite_version]]
        return None

    def _make_cursor_mock(*_: Any) -> MagicMock:
        return MagicMock(execute=execute_mock, close=close_mock, fetchall=fetchall_mock)
    dbapi_connection = MagicMock(cursor=_make_cursor_mock)
    assert util.setup_connection_for_dialect(instance_mock, 'sqlite', dbapi_connection, True) is not None
    assert len(execute_args) == 5
    assert execute_args[0] == 'PRAGMA journal_mode=WAL'
    assert execute_args[1] == 'SELECT sqlite_version()'
    assert execute_args[2] == 'PRAGMA cache_size = -16384'
    assert execute_args[3] == 'PRAGMA synchronous=NORMAL'
    assert execute_args[4] == 'PRAGMA foreign_keys=ON'
    execute_args = []
    assert util.setup_connection_for_dialect(instance_mock, 'sqlite', dbapi_connection, False) is None
    assert len(execute_args) == 3
    assert execute_args[0] == 'PRAGMA cache_size = -16384'
    assert execute_args[1] == 'PRAGMA synchronous=NORMAL'
    assert execute_args[2] == 'PRAGMA foreign_keys=ON'

@pytest.mark.parametrize('sqlite_version', [str(MIN_VERSION_SQLITE)])
def test_setup_connection_for_dialect_sqlite_zero_commit_interval(sqlite_version: str) -> None:
    """Test setting up the connection for a sqlite dialect with a zero commit interval."""
    instance_mock = MagicMock(commit_interval=0)
    execute_args: List[str] = []
    close_mock = MagicMock()

    def execute_mock(statement: str) -> None:
        nonlocal execute_args
        execute_args.append(statement)

    def fetchall_mock() -> Optional[List[List[str]]]:
        nonlocal execute_args
        if execute_args[-1] == 'SELECT sqlite_version()':
            return [[sqlite_version]]
        return None

    def _make_cursor_mock(*_: Any) -> MagicMock:
        return MagicMock(execute=execute_mock, close=close_mock, fetchall=fetchall_mock)
    dbapi_connection = MagicMock(cursor=_make_cursor_mock)
    assert util.setup_connection_for_dialect(instance_mock, 'sqlite', dbapi_connection, True) is not None
    assert len(execute_args) == 5
    assert execute_args[0] == 'PRAGMA journal_mode=WAL'
    assert execute_args[1] == 'SELECT sqlite_version()'
    assert execute_args[2] == 'PRAGMA cache_size = -16384'
    assert execute_args[3] == 'PRAGMA synchronous=FULL'
    assert execute_args[4] == 'PRAGMA foreign_keys=ON'
    execute_args = []
    assert util.setup_connection_for_dialect(instance_mock, 'sqlite', dbapi_connection, False) is None
    assert len(execute_args) == 3
    assert execute_args[0] == 'PRAGMA cache_size = -16384'
    assert execute_args[1] == 'PRAGMA synchronous=FULL'
    assert execute_args[2] == 'PRAGMA foreign_keys=ON'

@pytest.mark.parametrize(('mysql_version', 'message'), [('10.2.0-MariaDB', 'Version 10.2.0 of MariaDB is not supported; minimum supported version is 10.3.0.'), ('5.7.26-0ubuntu0.18.04.1', 'Version 5.7.26 of MySQL is not supported; minimum supported version is 8.0.0.'), ('some_random_response', 'Version some_random_response of MySQL is not supported; minimum supported version is 8.0.0.')])
def test_fail_outdated_mysql(caplog: pytest.LogCaptureFixture, mysql_version: str, message: str) -> None:
    """Test setting up the connection for an outdated mysql version."""
    instance_mock = MagicMock()
    execute_args: List[str] = []
    close_mock = MagicMock()

    def execute_mock(statement: str) -> None:
        nonlocal execute_args
        execute_args.append(statement)

    def fetchall_mock() -> Optional[List[List[str]]]:
        nonlocal execute_args
        if execute_args[-1] == 'SELECT VERSION()':
            return [[mysql_version]]
        return None

    def _make_cursor_mock(*_: Any) -> MagicMock:
        return MagicMock(execute=execute_mock, close=close_mock, fetchall=fetchall_mock)
    dbapi_connection = MagicMock(cursor=_make_cursor_mock)
    with pytest.raises(UnsupportedDialect):
        util.setup_connection_for_dialect(instance_mock, 'mysql', dbapi_connection, True)
    assert message in caplog.text

@pytest.mark.parametrize('mysql_version', ['10.3.0', '8.0.0'])
def test_supported_mysql(caplog: pytest.LogCaptureFixture, mysql_version: str) -> None:
    """Test setting up the connection for a supported mysql version."""
    instance_mock = MagicMock()
    execute_args: List[str] = []
    close_mock = MagicMock()

    def execute_mock(statement: str) -> None:
        nonlocal execute_args
        execute_args.append(statement)

    def fetchall_mock() -> Optional[List[List[str]]]:
        nonlocal execute_args
        if execute_args[-1] == 'SELECT VERSION()':
            return [[mysql_version]]
        return None

    def _make_cursor_mock(*_: Any) -> MagicMock:
        return MagicMock(execute=execute_mock, close=close_mock, fetchall=fetchall_mock)
    dbapi_connection = MagicMock(cursor=_make_cursor_mock)
    util.setup_connection_for_dialect(instance_mock, 'mysql', dbapi_connection, True)
    assert 'minimum supported version' not in caplog.text

@pytest.mark.parametrize(('pgsql_version', 'message'), [('11.12 (Debian 11.12-1.pgdg100+1)', 'Version 11.12 of PostgreSQL is not supported; minimum supported version is 12.0.'), ('9.2.10', 'Version 9.2.10 of PostgreSQL is not supported; minimum supported version is 12.0.'), ('unexpected', 'Version unexpected of PostgreSQL is not supported; minimum supported version is 12.0.')])
def test_fail_outdated_pgsql(caplog: pytest.LogCaptureFixture, pgsql_version: str, message: str) -> None:
    """Test setting up the connection for an outdated PostgreSQL version."""
    instance_mock = MagicMock()
    execute_args: List[str] = []
    close_mock = MagicMock()

    def execute_mock(statement: str) -> None:
        nonlocal execute_args
        execute_args.append(statement)

    def fetchall_mock() -> Optional[List[List[str]]]:
        nonlocal execute_args
        if execute_args[-1] == 'SHOW server_version':
            return [[pgsql_version]]
        return None

    def _make_cursor_mock(*_: Any) -> MagicMock:
        return MagicMock(execute=execute_mock, close=close_mock, fetchall=fetchall_mock)
    dbapi_connection = MagicMock(cursor=_make_cursor_mock)
    with pytest.raises(UnsupportedDialect):
        util.setup_connection_for_dialect(instance_mock, 'postgresql', dbapi_connection, True)
    assert message in caplog.text

@pytest.mark.parametrize('pgsql_version', ['14.0 (Debian 14.0-1.pgdg110+1)'])
def test_supported_pgsql(caplog: pytest.LogCaptureFixture, pgsql_version: str) -> None:
    """Test setting up the connection for a supported PostgreSQL version."""
    instance_mock = MagicMock()
    execute_args: List[str] = []
    close_mock = MagicMock()

    def execute_mock(statement: str) -> None:
        nonlocal execute_args
        execute_args.append(statement)

    def fetchall_mock() -> Optional[List[List[str]]]:
        nonlocal execute_args
        if execute_args[-1] == 'SHOW server_version':
            return [[pgsql_version]]
        return None

    def _make_cursor_mock(*_: Any) -> MagicMock:
        return MagicMock(execute=execute_mock, close=close_mock, fetchall=fetchall_mock)
    dbapi_connection = MagicMock(cursor=_make_cursor_mock)
    database_engine = util.setup_connection_for_dialect(instance_mock, 'postgresql', dbapi_connection, True)
    assert 'minimum supported version' not in caplog.text
   