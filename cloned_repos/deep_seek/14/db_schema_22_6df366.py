"""Models for SQLAlchemy.

This file contains the model definitions for schema version 22,
used by Home Assistant Core 2021.10.0, which adds a table for
5-minute statistics.
It is used to test the schema migration logic.
"""
from __future__ import annotations
from collections.abc import Iterable
from datetime import datetime, timedelta
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union, overload
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Identity, Index, Integer, String, Text, distinct
from sqlalchemy.dialects import mysql, oracle, postgresql
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.orm.session import Session
from homeassistant.const import MAX_LENGTH_EVENT_CONTEXT_ID, MAX_LENGTH_EVENT_EVENT_TYPE, MAX_LENGTH_EVENT_ORIGIN, MAX_LENGTH_STATE_DOMAIN, MAX_LENGTH_STATE_ENTITY_ID, MAX_LENGTH_STATE_STATE
from homeassistant.core import Context, Event, EventOrigin, State, split_entity_id
from homeassistant.helpers.json import JSONEncoder
from homeassistant.util import dt as dt_util
Base = declarative_base()
SCHEMA_VERSION = 22
_LOGGER = logging.getLogger(__name__)
DB_TIMEZONE = '+00:00'
TABLE_EVENTS = 'events'
TABLE_STATES = 'states'
TABLE_RECORDER_RUNS = 'recorder_runs'
TABLE_SCHEMA_CHANGES = 'schema_changes'
TABLE_STATISTICS = 'statistics'
TABLE_STATISTICS_META = 'statistics_meta'
TABLE_STATISTICS_RUNS = 'statistics_runs'
TABLE_STATISTICS_SHORT_TERM = 'statistics_short_term'
ALL_TABLES = [TABLE_STATES, TABLE_EVENTS, TABLE_RECORDER_RUNS, TABLE_SCHEMA_CHANGES, TABLE_STATISTICS, TABLE_STATISTICS_META, TABLE_STATISTICS_RUNS, TABLE_STATISTICS_SHORT_TERM]
DATETIME_TYPE = DateTime(timezone=True).with_variant(mysql.DATETIME(timezone=True, fsp=6), 'mysql')
DOUBLE_TYPE = Float().with_variant(mysql.DOUBLE(asdecimal=False), 'mysql').with_variant(oracle.DOUBLE_PRECISION(), 'oracle').with_variant(postgresql.DOUBLE_PRECISION(), 'postgresql')

class Events(Base):
    """Event history data."""
    __table_args__ = (Index('ix_events_event_type_time_fired', 'event_type', 'time_fired'), {'mysql_default_charset': 'utf8mb4', 'mysql_collate': 'utf8mb4_unicode_ci'})
    __tablename__ = TABLE_EVENTS
    event_id: Column[int] = Column(Integer, Identity(), primary_key=True)
    event_type: Column[str] = Column(String(MAX_LENGTH_EVENT_EVENT_TYPE))
    event_data: Column[str] = Column(Text().with_variant(mysql.LONGTEXT, 'mysql'))
    origin: Column[str] = Column(String(MAX_LENGTH_EVENT_ORIGIN))
    time_fired: Column[datetime] = Column(DATETIME_TYPE, index=True)
    created: Column[datetime] = Column(DATETIME_TYPE, default=dt_util.utcnow)
    context_id: Column[str] = Column(String(MAX_LENGTH_EVENT_CONTEXT_ID), index=True)
    context_user_id: Column[str] = Column(String(MAX_LENGTH_EVENT_CONTEXT_ID), index=True)
    context_parent_id: Column[str] = Column(String(MAX_LENGTH_EVENT_CONTEXT_ID), index=True)

    def __repr__(self) -> str:
        """Return string representation of instance for debugging."""
        return f"<recorder.Events(id={self.event_id}, type='{self.event_type}', data='{self.event_data}', origin='{self.origin}', time_fired='{self.time_fired}')>"

    @staticmethod
    def from_event(event: Event, event_data: Optional[str] = None) -> Events:
        """Create an event database object from a native event."""
        return Events(event_type=event.event_type, event_data=event_data or json.dumps(event.data, cls=JSONEncoder, separators=(',', ':')), origin=str(event.origin.value), time_fired=event.time_fired, context_id=event.context.id, context_user_id=event.context.user_id, context_parent_id=event.context.parent_id)

    def to_native(self, validate_entity_id: bool = True) -> Optional[Event]:
        """Convert to a native HA Event."""
        context = Context(id=self.context_id, user_id=self.context_user_id, parent_id=self.context_parent_id)
        try:
            return Event(self.event_type, json.loads(self.event_data), EventOrigin(self.origin), process_timestamp(self.time_fired), context=context)
        except ValueError:
            _LOGGER.exception('Error converting to event: %s', self)
            return None

class States(Base):
    """State change history."""
    __table_args__ = (Index('ix_states_entity_id_last_updated', 'entity_id', 'last_updated'), {'mysql_default_charset': 'utf8mb4', 'mysql_collate': 'utf8mb4_unicode_ci'})
    __tablename__ = TABLE_STATES
    state_id: Column[int] = Column(Integer, Identity(), primary_key=True)
    domain: Column[str] = Column(String(MAX_LENGTH_STATE_DOMAIN))
    entity_id: Column[str] = Column(String(MAX_LENGTH_STATE_ENTITY_ID))
    state: Column[str] = Column(String(MAX_LENGTH_STATE_STATE))
    attributes: Column[str] = Column(Text().with_variant(mysql.LONGTEXT, 'mysql'))
    event_id: Column[int] = Column(Integer, ForeignKey('events.event_id', ondelete='CASCADE'), index=True)
    last_changed: Column[datetime] = Column(DATETIME_TYPE, default=dt_util.utcnow)
    last_updated: Column[datetime] = Column(DATETIME_TYPE, default=dt_util.utcnow, index=True)
    created: Column[datetime] = Column(DATETIME_TYPE, default=dt_util.utcnow)
    old_state_id: Column[int] = Column(Integer, ForeignKey('states.state_id'), index=True)
    event: relationship[Optional[Events]] = relationship('Events', uselist=False)
    old_state: relationship[Optional['States']] = relationship('States', remote_side=[state_id])

    def __repr__(self) -> str:
        """Return string representation of instance for debugging."""
        return f"<recorder.States(id={self.state_id}, domain='{self.domain}', entity_id='{self.entity_id}', state='{self.state}', event_id='{self.event_id}', last_updated='{self.last_updated.isoformat(sep=' ', timespec='seconds')}', old_state_id={self.old_state_id})>"

    @staticmethod
    def from_event(event: Event) -> States:
        """Create object from a state_changed event."""
        entity_id = event.data['entity_id']
        state = event.data.get('new_state')
        dbstate = States(entity_id=entity_id)
        if state is None:
            dbstate.state = ''
            dbstate.domain = split_entity_id(entity_id)[0]
            dbstate.attributes = '{}'
            dbstate.last_changed = event.time_fired
            dbstate.last_updated = event.time_fired
        else:
            dbstate.domain = state.domain
            dbstate.state = state.state
            dbstate.attributes = json.dumps(dict(state.attributes), cls=JSONEncoder, separators=(',', ':'))
            dbstate.last_changed = state.last_changed
            dbstate.last_updated = state.last_updated
        return dbstate

    def to_native(self, validate_entity_id: bool = True) -> Optional[State]:
        """Convert to an HA state object."""
        try:
            return State(self.entity_id, self.state, json.loads(self.attributes), process_timestamp(self.last_changed), process_timestamp(self.last_updated), context=Context(id=None), validate_entity_id=validate_entity_id)
        except ValueError:
            _LOGGER.exception('Error converting row to state: %s', self)
            return None

class StatisticResult(TypedDict):
    """Statistic result data class.

    Allows multiple datapoints for the same statistic_id.
    """
    statistic_id: str
    start: datetime
    mean: Optional[float]
    min: Optional[float]
    max: Optional[float]
    last_reset: Optional[datetime]
    state: Optional[float]
    sum: Optional[float]

class StatisticDataBase(TypedDict):
    """Mandatory fields for statistic data class."""
    start: datetime

class StatisticData(StatisticDataBase, total=False):
    """Statistic data class."""
    mean: float
    min: float
    max: float
    last_reset: Optional[datetime]
    state: float
    sum: float

class StatisticsBase:
    """Statistics base class."""
    id: Column[int] = Column(Integer, Identity(), primary_key=True)
    created: Column[datetime] = Column(DATETIME_TYPE, default=dt_util.utcnow)

    @declared_attr
    def metadata_id(self) -> Column[int]:
        """Define the metadata_id column for sub classes."""
        return Column(Integer, ForeignKey(f'{TABLE_STATISTICS_META}.id', ondelete='CASCADE'), index=True)
    start: Column[datetime] = Column(DATETIME_TYPE, index=True)
    mean: Column[Optional[float]] = Column(DOUBLE_TYPE)
    min: Column[Optional[float]] = Column(DOUBLE_TYPE)
    max: Column[Optional[float]] = Column(DOUBLE_TYPE)
    last_reset: Column[Optional[datetime]] = Column(DATETIME_TYPE)
    state: Column[Optional[float]] = Column(DOUBLE_TYPE)
    sum: Column[Optional[float]] = Column(DOUBLE_TYPE)

    @classmethod
    def from_stats(cls, metadata_id: int, stats: StatisticData) -> StatisticsBase:
        """Create object from a statistics."""
        return cls(metadata_id=metadata_id, **stats)

class Statistics(Base, StatisticsBase):
    """Long term statistics."""
    duration: timedelta = timedelta(hours=1)
    __table_args__ = (Index('ix_statistics_statistic_id_start', 'metadata_id', 'start'),)
    __tablename__ = TABLE_STATISTICS

class StatisticsShortTerm(Base, StatisticsBase):
    """Short term statistics."""
    duration: timedelta = timedelta(minutes=5)
    __table_args__ = (Index('ix_statistics_short_term_statistic_id_start', 'metadata_id', 'start'),)
    __tablename__ = TABLE_STATISTICS_SHORT_TERM

class StatisticMetaData(TypedDict):
    """Statistic meta data class."""
    statistic_id: str
    source: str
    unit_of_measurement: Optional[str]
    has_mean: bool
    has_sum: bool

class StatisticsMeta(Base):
    """Statistics meta data."""
    __table_args__ = ({'mysql_default_charset': 'utf8mb4', 'mysql_collate': 'utf8mb4_unicode_ci'},)
    __tablename__ = TABLE_STATISTICS_META
    id: Column[int] = Column(Integer, Identity(), primary_key=True)
    statistic_id: Column[str] = Column(String(255), index=True)
    source: Column[str] = Column(String(32))
    unit_of_measurement: Column[Optional[str]] = Column(String(255))
    has_mean: Column[bool] = Column(Boolean)
    has_sum: Column[bool] = Column(Boolean)

    @staticmethod
    def from_meta(source: str, statistic_id: str, unit_of_measurement: Optional[str], has_mean: bool, has_sum: bool) -> StatisticsMeta:
        """Create object from meta data."""
        return StatisticsMeta(source=source, statistic_id=statistic_id, unit_of_measurement=unit_of_measurement, has_mean=has_mean, has_sum=has_sum)

class RecorderRuns(Base):
    """Representation of recorder run."""
    __table_args__ = (Index('ix_recorder_runs_start_end', 'start', 'end'),)
    __tablename__ = TABLE_RECORDER_RUNS
    run_id: Column[int] = Column(Integer, Identity(), primary_key=True)
    start: Column[datetime] = Column(DateTime(timezone=True), default=dt_util.utcnow)
    end: Column[Optional[datetime]] = Column(DateTime(timezone=True))
    closed_incorrect: Column[bool] = Column(Boolean, default=False)
    created: Column[datetime] = Column(DateTime(timezone=True), default=dt_util.utcnow)

    def __repr__(self) -> str:
        """Return string representation of instance for debugging."""
        end = f"'{self.end.isoformat(sep=' ', timespec='seconds')}'" if self.end else None
        return f"<recorder.RecorderRuns(id={self.run_id}, start='{self.start.isoformat(sep=' ', timespec='seconds')}', end={end}, closed_incorrect={self.closed_incorrect}, created='{self.created.isoformat(sep=' ', timespec='seconds')}')>"

    def entity_ids(self, point_in_time: Optional[datetime] = None) -> List[str]:
        """Return the entity ids that existed in this run.

        Specify point_in_time if you want to know which existed at that point
        in time inside the run.
        """
        session = Session.object_session(self)
        assert session is not None, 'RecorderRuns need to be persisted'
        query = session.query(distinct(States.entity_id)).filter(States.last_updated >= self.start)
        if point_in_time is not None:
            query = query.filter(States.last_updated < point_in_time)
        elif self.end is not None:
            query = query.filter(States.last_updated < self.end)
        return [row[0] for row in query]

    def to_native(self, validate_entity_id: bool = True) -> RecorderRuns:
        """Return self, native format is this model."""
        return self

class SchemaChanges(Base):
    """Representation of schema version changes."""
    __tablename__ = TABLE_SCHEMA_CHANGES
    change_id: Column[int] = Column(Integer, Identity(), primary_key=True)
    schema_version: Column[int] = Column(Integer)
    changed: Column[datetime] = Column(DateTime(timezone=True), default=dt_util.utcnow)

    def __repr__(self) -> str:
        """Return string representation of instance for debugging."""
        return f"<recorder.SchemaChanges(id={self.change_id}, schema_version={self.schema_version}, changed='{self.changed.isoformat(sep=' ', timespec='seconds')}')>"

class StatisticsRuns(Base):
    """Representation of statistics run."""
    __tablename__ = TABLE_STATISTICS_RUNS
    run_id: Column[int] = Column(Integer, Identity(), primary_key=True)
    start: Column[datetime] = Column(DateTime(timezone=True))

    def __repr__(self) -> str:
        """Return string representation of instance for debugging."""
        return f"<recorder.StatisticsRuns(id={self.run_id}, start='{self.start.isoformat(sep=' ', timespec='seconds')}', )>"

@overload
def process_timestamp(ts: None) -> None: ...

@overload
def process_timestamp(ts: datetime) -> datetime: ...

def process_timestamp(ts: Optional[datetime]) -> Optional[datetime]:
    """Process a timestamp into datetime object."""
    if ts is None:
        return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=dt_util.UTC)
    return dt_util.as_utc(ts)

@overload
def process_timestamp_to_utc_isoformat(ts: None) -> None: ...

@overload
def process_timestamp_to_utc_isoformat(ts: datetime) -> str: ...

def process_timestamp_to_utc_isoformat(ts: Optional[datetime]) -> Optional[str]:
    """Process a timestamp into UTC isotime."""
    if ts is None:
        return None
    if ts.tzinfo == dt_util.UTC:
        return ts.isoformat()
    if ts.tzinfo is None:
        return f'{ts.isoformat()}{DB_TIMEZONE}'
    return ts.astimezone(dt_util.UTC).isoformat()

class LazyState(State):
    """A lazy version of core State."""
    __slots__ = ['_attributes', '_context', '_last_changed', '_last_updated', '_row']

    def __init__(self, row: States) -> None:
        """Init the lazy state."""
        self._row = row
        self.entity_id = self._row.entity_id
        self.state = self._row.state or ''
        self._attributes: Optional[Dict[str, Any]] = None
        self._last_changed: Optional[datetime] = None
        self._last_updated: Optional[datetime] = None
        self._context: Optional[Context] = None

    @property
    def attributes(self) -> Dict[str, Any]:
        """State attributes."""
        if not self._attributes:
            try:
                self._attributes = json.loads(self._row.attributes)
            except ValueError:
                _LOGGER.exception('Error converting row to state: %s', self._row)
                self._attributes = {}
        return self._attributes

    @attributes.setter
    def attributes(self, value: Dict[str, Any]) -> None:
        """Set attributes."""
        self._attributes = value

    @property
    def context(self) -> Context:
        """State context."""
        if not self._context:
            self._context = Context(id=None)
        return self._context

    @context.setter
    def context(self, value: Context) -> None:
        """Set context."""
        self._context = value