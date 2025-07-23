"""Test data purging."""
from collections.abc import Generator
from datetime import datetime, timedelta
import json
import sqlite3
from typing import Any, Dict, List, Optional, Set, Tuple, cast
from unittest.mock import patch
from freezegun import freeze_time
import pytest
from sqlalchemy import text, update
from sqlalchemy.exc import DatabaseError, OperationalError
from sqlalchemy.orm.session import Session
from homeassistant.components.recorder import DOMAIN as RECORDER_DOMAIN, Recorder, migration
from homeassistant.components.recorder.const import SupportedDialect
from homeassistant.components.recorder.history import get_significant_states
from homeassistant.components.recorder.purge import purge_old_data
from homeassistant.components.recorder.services import SERVICE_PURGE, SERVICE_PURGE_ENTITIES
from homeassistant.components.recorder.tasks import PurgeTask
from homeassistant.components.recorder.util import session_scope
from homeassistant.const import EVENT_STATE_CHANGED
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
from .common import async_attach_db_engine, async_recorder_block_till_done, async_wait_purge_done, async_wait_recording_done, old_db_schema
from .db_schema_32 import EventData, Events, RecorderRuns, StateAttributes, States, StatisticsRuns, StatisticsShortTerm
from tests.typing import RecorderInstanceContextManager

@pytest.fixture
async def mock_recorder_before_hass(async_test_recorder: Any) -> Generator[None, None, None]:
    """Set up recorder."""
    yield

@pytest.fixture(autouse=True)
def db_schema_32() -> Generator[None, None, None]:
    """Fixture to initialize the db with the old schema 32."""
    with old_db_schema('32'):
        yield

@pytest.fixture(name='use_sqlite')
def mock_use_sqlite(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    """Pytest fixture to switch purge method."""
    with patch('homeassistant.components.recorder.core.Recorder.dialect_name', return_value=SupportedDialect.SQLITE if request.param else SupportedDialect.MYSQL):
        yield

async def test_purge_old_states(hass: HomeAssistant, recorder_mock: Recorder) -> None:
    """Test deleting old states."""
    await async_attach_db_engine(hass)
    await _add_test_states(hass)
    with session_scope(hass=hass) as session:
        states = session.query(States)
        state_attributes = session.query(StateAttributes)
        assert states.count() == 6
        assert states[0].old_state_id is None
        assert states[5].old_state_id == states[4].state_id
        assert state_attributes.count() == 3
        events = session.query(Events).filter(Events.event_type == 'state_changed')
        assert events.count() == 0
        assert 'test.recorder2' in recorder_mock.states_manager._last_committed_id
    purge_before = dt_util.utcnow() - timedelta(days=4)
    finished = purge_old_data(recorder_mock, purge_before, states_batch_size=1, events_batch_size=1, repack=False)
    assert not finished
    with session_scope(hass=hass) as session:
        states = session.query(States)
        state_attributes = session.query(StateAttributes)
        assert states.count() == 2
        assert state_attributes.count() == 1
        assert 'test.recorder2' in recorder_mock.states_manager._last_committed_id
        states_after_purge = list(session.query(States))
        state_map_by_state = {state.state: state for state in states_after_purge}
        dontpurgeme_5 = state_map_by_state['dontpurgeme_5']
        dontpurgeme_4 = state_map_by_state['dontpurgeme_4']
        assert dontpurgeme_5.old_state_id == dontpurgeme_4.state_id
        assert dontpurgeme_4.old_state_id is None
    finished = purge_old_data(recorder_mock, purge_before, repack=False)
    assert finished
    with session_scope(hass=hass) as session:
        states = session.query(States)
        state_attributes = session.query(StateAttributes)
        assert states.count() == 2
        assert state_attributes.count() == 1
        assert 'test.recorder2' in recorder_mock.states_manager._last_committed_id
    purge_before = dt_util.utcnow()
    finished = purge_old_data(recorder_mock, purge_before, states_batch_size=1, events_batch_size=1, repack=False)
    assert not finished
    with session_scope(hass=hass) as session:
        states = session.query(States)
        state_attributes = session.query(StateAttributes)
        assert states.count() == 0
        assert state_attributes.count() == 0
        assert 'test.recorder2' not in recorder_mock.states_manager._last_committed_id
    await _add_test_states(hass)
    with session_scope(hass=hass) as session:
        states = session.query(States)
        assert states.count() == 6
        assert states[0].old_state_id is None
        assert states[5].old_state_id == states[4].state_id
        events = session.query(Events).filter(Events.event_type == 'state_changed')
        assert events.count() == 0
        assert 'test.recorder2' in recorder_mock.states_manager._last_committed_id
        state_attributes = session.query(StateAttributes)
        assert state_attributes.count() == 3

@pytest.mark.skip_on_db_engine(['mysql', 'postgresql'])
@pytest.mark.usefixtures('recorder_mock', 'skip_by_db_engine')
async def test_purge_old_states_encouters_database_corruption(hass: HomeAssistant) -> None:
    """Test database image image is malformed while deleting old states.

    This test is specific for SQLite, wiping the database on error only happens
    with SQLite.
    """
    await async_attach_db_engine(hass)
    await _add_test_states(hass)
    await async_wait_recording_done(hass)
    sqlite3_exception = DatabaseError('statement', {}, [])
    sqlite3_exception.__cause__ = sqlite3.DatabaseError('not a database')
    with patch('homeassistant.components.recorder.core.move_away_broken_database') as move_away, patch('homeassistant.components.recorder.purge.purge_old_data', side_effect=sqlite3_exception):
        await hass.services.async_call(RECORDER_DOMAIN, SERVICE_PURGE, {'keep_days': 0})
        await hass.async_block_till_done()
        await async_wait_recording_done(hass)
    assert move_away.called
    with session_scope(hass=hass) as session:
        states_after_purge = session.query(States)
        assert states_after_purge.count() == 0

async def test_purge_old_states_encounters_temporary_mysql_error(hass: HomeAssistant, recorder_mock: Recorder, caplog: pytest.LogCaptureFixture) -> None:
    """Test retry on specific mysql operational errors."""
    await async_attach_db_engine(hass)
    await _add_test_states(hass)
    await async_wait_recording_done(hass)
    mysql_exception = OperationalError('statement', {}, [])
    mysql_exception.orig = Exception(1205, 'retryable')
    with patch('homeassistant.components.recorder.util.time.sleep') as sleep_mock, patch('homeassistant.components.recorder.purge._purge_old_recorder_runs', side_effect=[mysql_exception, None]), patch.object(recorder_mock.engine.dialect, 'name', 'mysql'):
        await hass.services.async_call(RECORDER_DOMAIN, SERVICE_PURGE, {'keep_days': 0})
        await hass.async_block_till_done()
        await async_wait_recording_done(hass)
        await async_wait_recording_done(hass)
    assert 'retrying' in caplog.text
    assert sleep_mock.called

@pytest.mark.usefixtures('recorder_mock')
async def test_purge_old_states_encounters_operational_error(hass: HomeAssistant, caplog: pytest.LogCaptureFixture) -> None:
    """Test error on operational errors that are not mysql does not retry."""
    await async_attach_db_engine(hass)
    await _add_test_states(hass)
    await async_wait_recording_done(hass)
    exception = OperationalError('statement', {}, [])
    with patch('homeassistant.components.recorder.purge._purge_old_recorder_runs', side_effect=exception):
        await hass.services.async_call(RECORDER_DOMAIN, SERVICE_PURGE, {'keep_days': 0})
        await hass.async_block_till_done()
        await async_wait_recording_done(hass)
        await async_wait_recording_done(hass)
    assert 'retrying' not in caplog.text
    assert 'Error executing purge' in caplog.text

async def test_purge_old_events(hass: HomeAssistant, recorder_mock: Recorder) -> None:
    """Test deleting old events."""
    await async_attach_db_engine(hass)
    await _add_test_events(hass)
    with session_scope(hass=hass) as session:
        events = session.query(Events).filter(Events.event_type.like('EVENT_TEST%'))
        assert events.count() == 6
        purge_before = dt_util.utcnow() - timedelta(days=4)
    finished = purge_old_data(recorder_mock, purge_before, repack=False, events_batch_size=1, states_batch_size=1)
    assert not finished
    with session_scope(hass=hass) as session:
        events = session.query(Events).filter(Events.event_type.like('EVENT_TEST%'))
        assert events.count() == 2
    finished = purge_old_data(recorder_mock, purge_before, repack=False, events_batch_size=1, states_batch_size=1)
    assert finished
    with session_scope(hass=hass) as session:
        events = session.query(Events).filter(Events.event_type.like('EVENT_TEST%'))
        assert events.count() == 2

async def test_purge_old_recorder_runs(hass: HomeAssistant, recorder_mock: Recorder) -> None:
    """Test deleting old recorder runs keeps current run."""
    await async_attach_db_engine(hass)
    await _add_test_recorder_runs(hass)
    with session_scope(hass=hass) as session:
        recorder_runs = session.query(RecorderRuns)
        assert recorder_runs.count() == 7
    purge_before = dt_util.utcnow()
    finished = purge_old_data(recorder_mock, purge_before, repack=False, events_batch_size=1, states_batch_size=1)
    assert not finished
    finished = purge_old_data(recorder_mock, purge_before, repack=False, events_batch_size=1, states_batch_size=1)
    assert finished
    with session_scope(hass=hass) as session:
        recorder_runs = session.query(RecorderRuns)
        assert recorder_runs.count() == 3

async def test_purge_old_statistics_runs(hass: HomeAssistant, recorder_mock: Recorder) -> None:
    """Test deleting old statistics runs keeps the latest run."""
    await async_attach_db_engine(hass)
    await _add_test_statistics_runs(hass)
    with session_scope(hass=hass) as session:
        statistics_runs = session.query(StatisticsRuns)
        assert statistics_runs.count() == 7
    purge_before = dt_util.utcnow()
    finished = purge_old_data(recorder_mock, purge_before, repack=False)
    assert not finished
    finished = purge_old_data(recorder_mock, purge_before, repack=False)
    assert finished
    with session_scope(hass=hass) as session:
        statistics_runs = session.query(StatisticsRuns)
        assert statistics_runs.count() == 1

@pytest.mark.parametrize('use_sqlite', [True, False], indirect=True)
@pytest.mark.usefixtures('recorder_mock')
async def test_purge_method(hass: HomeAssistant, caplog: pytest.LogCaptureFixture, use_sqlite: bool) -> None:
    """Test purge method."""

    def assert_recorder_runs_equal(run1: RecorderRuns, run2: RecorderRuns) -> None:
        assert run1.run_id == run2.run_id
        assert run1.start == run2.start
        assert run1.end == run2.end
        assert run1.closed_incorrect == run2.closed_incorrect
        assert run1.created == run2.created

    def assert_statistic_runs_equal(run1: StatisticsRuns, run2: StatisticsRuns) -> None:
        assert run1.run_id == run2.run_id
        assert run1.start == run2.start
        
    await async_attach_db_engine(hass)
    service_data = {'keep_days': 4}
    await _add_test_events(hass)
    await _add_test_states(hass)
    await _add_test_statistics(hass)
    await _add_test_recorder_runs(hass)
    await _add_test_statistics_runs(hass)
    await hass.async_block_till_done()
    await async_wait_recording_done(hass)
    with session_scope(hass=hass) as session:
        states = session.query(States)
        assert states.count() == 6
        events = session.query(Events).filter(Events.event_type.like('EVENT_TEST%'))
        assert events.count() == 6
        statistics = session.query(StatisticsShortTerm)
        assert statistics.count() == 6
        recorder_runs = session.query(RecorderRuns)
        assert recorder_runs.count() == 7
        runs_before_purge = recorder_runs.all()
        statistics_runs = session.query(StatisticsRuns).order_by(StatisticsRuns.run_id)
        assert statistics_runs.count() == 7
        statistic_runs_before_purge = statistics_runs.all()
        for itm in runs_before_purge:
            session.expunge(itm)
        for itm in statistic_runs_before_purge:
            session.expunge(itm)
    await hass.async_block_till_done()
    await async_wait_purge_done(hass)
    await hass.services.async_call('recorder', 'purge')
    await hass.async_block_till_done()
    await async_wait_purge_done(hass)
    with session_scope(hass=hass) as session:
        states = session.query(States)
        events = session.query(Events).filter(Events.event_type.like('EVENT_TEST%'))
        statistics = session.query(StatisticsShortTerm)
        assert states.count() == 4
        assert events.count() == 4
        assert statistics.count() == 4
    await hass.services.async_call('recorder', 'purge', service_data=service_data)
    await hass.async_block_till_done()
    await async_wait_purge_done(hass)
    with session_scope(hass=hass) as session:
        states = session.query(States)
        events = session.query(Events).filter(Events.event_type.like('EVENT_TEST%'))
        statistics = session.query(StatisticsShortTerm)
        recorder_runs = session.query(RecorderRuns)
        statistics_runs = session.query(StatisticsRuns)
        assert states.count() == 2
        assert events.count() == 2
        assert statistics.count() == 2
        runs = recorder_runs.all()
        assert_recorder_runs_equal(runs[0], runs_before_purge[0])
        assert_recorder_runs_equal(runs[1], runs_before_purge[5])
        assert_recorder_runs_equal(runs[2], runs_before_purge[6])
        runs = statistics_runs.all()
        assert_statistic_runs_equal(runs[0], statistic_runs_before_purge[0])
        assert_statistic_runs_equal(runs[1], statistic_runs_before_purge[5])
        assert_statistic_runs_equal(runs[2], statistic_runs_before_purge[6])
        assert 'EVENT_TEST_PURGE' not in (event.event_type for event in events.all())
    service_data['repack'] = True
    await hass.services.async_call('recorder', 'purge', service_data=service_data)
    await hass.async_block_till_done()
    await async_wait_purge_done(hass)
    assert 'Vacuuming SQL DB to free space' in caplog.text or 'Optimizing SQL DB to free space' in caplog.text

@pytest.mark.parametrize('use_sqlite', [True, False], indirect=True)
@pytest.mark.usefixtures('recorder_mock')
async def test_purge_edge_case(hass: HomeAssistant, use_sqlite: bool) -> None:
    """Test states and events are purged even if they occurred shortly before purge_before."""

    async def _add_db_entries(hass: HomeAssistant, timestamp: datetime) -> None:
        with session_scope(hass=hass) as session:
            session.add(Events(event_id=1001, event_type='EVENT_TEST_PURGE', event_data='{}', origin='LOCAL', time_fired_ts=timestamp.timestamp()))
