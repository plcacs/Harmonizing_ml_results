#!/usr/bin/env python3
"""Models for SQLAlchemy.

This file contains the model definitions for schema version 18,
used by Home Assistant Core 2021.7.0, which did a major refactoring
of long term statistics database models.
It is used to test the schema migration logic.
"""
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Identity, Index, Integer, String, Text, distinct
from sqlalchemy.dialects import mysql
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.orm.session import Session
from homeassistant.const import (MAX_LENGTH_EVENT_CONTEXT_ID, MAX_LENGTH_EVENT_EVENT_TYPE,
                                 MAX_LENGTH_EVENT_ORIGIN, MAX_LENGTH_STATE_DOMAIN,
                                 MAX_LENGTH_STATE_ENTITY_ID, MAX_LENGTH_STATE_STATE)
from homeassistant.core import Context, Event, EventOrigin, State, split_entity_id
from homeassistant.helpers.json import JSONEncoder
from homeassistant.util import dt as dt_util

Base = declarative_base()
SCHEMA_VERSION: int = 18
_LOGGER = logging.getLogger(__name__)
DB_TIMEZONE: str = '+00:00'
TABLE_EVENTS: str = 'events'
TABLE_STATES: str = 'states'
TABLE_RECORDER_RUNS: str = 'recorder_runs'
TABLE_SCHEMA_CHANGES: str = 'schema_changes'
TABLE_STATISTICS: str = 'statistics'
TABLE_STATISTICS_META: str = 'statistics_meta'
ALL_TABLES: List[str] = [TABLE_STATES, TABLE_EVENTS, TABLE_RECORDER_RUNS, TABLE_SCHEMA_CHANGES, TABLE_STATISTICS, TABLE_STATISTICS_META]
DATETIME_TYPE = DateTime(timezone=True).with_variant(mysql.DATETIME(timezone=True, fsp=6), 'mysql')


class Events(Base):
    """Event history data."""
    __table_args__ = (
        Index('ix_events_event_type_time_fired', 'event_type', 'time_fired'),
        {'mysql_default_charset': 'utf8mb4', 'mysql_collate': 'utf8mb4_unicode_ci'}
    )
    __tablename__ = TABLE_EVENTS
    event_id = Column(Integer, Identity(), primary_key=True)
    event_type = Column(String(MAX_LENGTH_EVENT_EVENT_TYPE))
    event_data = Column(Text().with_variant(mysql.LONGTEXT, 'mysql'))
    origin = Column(String(MAX_LENGTH_EVENT_ORIGIN))
    time_fired = Column(DATETIME_TYPE, index=True)
    created = Column(DATETIME_TYPE, default=dt_util.utcnow)
    context_id = Column(String(MAX_LENGTH_EVENT_CONTEXT_ID), index=True)
    context_user_id = Column(String(MAX_LENGTH_EVENT_CONTEXT_ID), index=True)
    context_parent_id = Column(String(MAX_LENGTH_EVENT_CONTEXT_ID), index=True)

    def __repr__(self) -> str:
        """Return string representation of instance for debugging."""
        return (f"<recorder.Events(id={self.event_id}, type='{self.event_type}', data='{self.event_data}', "
                f"origin='{self.origin}', time_fired='{self.time_fired}')>")

    @staticmethod
    def from_event(event: Event, event_data: Optional[str] = None) -> "Events":
        """Create an event database object from a native event."""
        return Events(
            event_type=event.event_type,
            event_data=event_data or json.dumps(event.data, cls=JSONEncoder),
            origin=str(event.origin.value),
            time_fired=event.time_fired,
            context_id=event.context.id,
            context_user_id=event.context.user_id,
            context_parent_id=event.context.parent_id
        )

    def to_native(self, validate_entity_id: bool = True) -> Optional[Event]:
        """Convert to a native HA Event."""
        context: Context = Context(id=self.context_id, user_id=self.context_user_id, parent_id=self.context_parent_id)
        try:
            return Event(
                self.event_type,
                json.loads(self.event_data),
                EventOrigin(self.origin),
                process_timestamp(self.time_fired),
                context=context
            )
        except ValueError:
            _LOGGER.exception('Error converting to event: %s', self)
            return None


class States(Base):
    """State change history."""
    __table_args__ = (
        Index('ix_states_entity_id_last_updated', 'entity_id', 'last_updated'),
        {'mysql_default_charset': 'utf8mb4', 'mysql_collate': 'utf8mb4_unicode_ci'}
    )
    __tablename__ = TABLE_STATES
    state_id = Column(Integer, Identity(), primary_key=True)
    domain = Column(String(MAX_LENGTH_STATE_DOMAIN))
    entity_id = Column(String(MAX_LENGTH_STATE_ENTITY_ID))
    state = Column(String(MAX_LENGTH_STATE_STATE))
    attributes = Column(Text().with_variant(mysql.LONGTEXT, 'mysql'))
    event_id = Column(Integer, ForeignKey('events.event_id', ondelete='CASCADE'), index=True)
    last_changed = Column(DATETIME_TYPE, default=dt_util.utcnow)
    last_updated = Column(DATETIME_TYPE, default=dt_util.utcnow, index=True)
    created = Column(DATETIME_TYPE, default=dt_util.utcnow)
    old_state_id = Column(Integer, ForeignKey('states.state_id'), index=True)
    event = relationship('Events', uselist=False)
    old_state = relationship('States', remote_side=[state_id])

    def __repr__(self) -> str:
        """Return string representation of instance for debugging."""
        return (f"<recorder.States(id={self.state_id}, domain='{self.domain}', entity_id='{self.entity_id}', "
                f"state='{self.state}', event_id='{self.event_id}', last_updated='{self.last_updated.isoformat(sep=' ', timespec='seconds')}', "
                f"old_state_id={self.old_state_id})>")

    @staticmethod
    def from_event(event: Event) -> "States":
        """Create object from a state_changed event."""
        entity_id: str = event.data['entity_id']
        state_obj: Optional[State] = event.data.get('new_state')
        dbstate = States(entity_id=entity_id)
        if state_obj is None:
            dbstate.state = ''
            dbstate.domain = split_entity_id(entity_id)[0]
            dbstate.attributes = '{}'
            dbstate.last_changed = event.time_fired
            dbstate.last_updated = event.time_fired
        else:
            dbstate.domain = state_obj.domain
            dbstate.state = state_obj.state
            dbstate.attributes = json.dumps(dict(state_obj.attributes), cls=JSONEncoder)
            dbstate.last_changed = state_obj.last_changed
            dbstate.last_updated = state_obj.last_updated
        return dbstate

    def to_native(self, validate_entity_id: bool = True) -> Optional[State]:
        """Convert to an HA state object."""
        try:
            return State(
                self.entity_id,
                self.state,
                json.loads(self.attributes),
                process_timestamp(self.last_changed),
                process_timestamp(self.last_updated),
                context=Context(id=None),
                validate_entity_id=validate_entity_id
            )
        except ValueError:
            _LOGGER.exception('Error converting row to state: %s', self)
            return None


class Statistics(Base):
    """Statistics."""
    __table_args__ = (Index('ix_statistics_statistic_id_start', 'metadata_id', 'start'),)
    __tablename__ = TABLE_STATISTICS
    id = Column(Integer, primary_key=True)
    created = Column(DATETIME_TYPE, default=dt_util.utcnow)
    metadata_id = Column(Integer, ForeignKey(f'{TABLE_STATISTICS_META}.id', ondelete='CASCADE'), index=True)
    start = Column(DATETIME_TYPE, index=True)
    mean = Column(Float())
    min = Column(Float())
    max = Column(Float())
    last_reset = Column(DATETIME_TYPE)
    state = Column(Float())
    sum = Column(Float())

    @staticmethod
    def from_stats(metadata_id: int, start: datetime, stats: Dict[str, Any]) -> "Statistics":
        """Create object from a statistics."""
        return Statistics(metadata_id=metadata_id, start=start, **stats)


class StatisticsMeta(Base):
    """Statistics meta data."""
    __tablename__ = TABLE_STATISTICS_META
    id = Column(Integer, primary_key=True)
    statistic_id = Column(String(255), index=True)
    source = Column(String(32))
    unit_of_measurement = Column(String(255))
    has_mean = Column(Boolean)
    has_sum = Column(Boolean)

    @staticmethod
    def from_meta(source: str, statistic_id: str, unit_of_measurement: str, has_mean: bool, has_sum: bool) -> "StatisticsMeta":
        """Create object from meta data."""
        return StatisticsMeta(
            source=source,
            statistic_id=statistic_id,
            unit_of_measurement=unit_of_measurement,
            has_mean=has_mean,
            has_sum=has_sum
        )


class RecorderRuns(Base):
    """Representation of recorder run."""
    __table_args__ = (Index('ix_recorder_runs_start_end', 'start', 'end'),)
    __tablename__ = TABLE_RECORDER_RUNS
    run_id = Column(Integer, Identity(), primary_key=True)
    start = Column(DateTime(timezone=True), default=dt_util.utcnow)
    end = Column(DateTime(timezone=True))
    closed_incorrect = Column(Boolean, default=False)
    created = Column(DateTime(timezone=True), default=dt_util.utcnow)

    def __repr__(self) -> str:
        """Return string representation of instance for debugging."""
        end_str: Optional[str] = f"'{self.end.isoformat(sep=' ', timespec='seconds')}'" if self.end else None
        return (f"<recorder.RecorderRuns(id={self.run_id}, start='{self.start.isoformat(sep=' ', timespec='seconds')}', "
                f"end={end_str}, closed_incorrect={self.closed_incorrect}, created='{self.created.isoformat(sep=' ', timespec='seconds')}')>")

    def entity_ids(self, point_in_time: Optional[datetime] = None) -> List[str]:
        """Return the entity ids that existed in this run.

        Specify point_in_time if you want to know which existed at that point
        in time inside the run.
        """
        session: Optional[Session] = Session.object_session(self)
        assert session is not None, 'RecorderRuns need to be persisted'
        query = session.query(distinct(States.entity_id)).filter(States.last_updated >= self.start)
        if point_in_time is not None:
            query = query.filter(States.last_updated < point_in_time)
        elif self.end is not None:
            query = query.filter(States.last_updated < self.end)
        return [row[0] for row in query]

    def to_native(self, validate_entity_id: bool = True) -> "RecorderRuns":
        """Return self, native format is this model."""
        return self


class SchemaChanges(Base):
    """Representation of schema version changes."""
    __tablename__ = TABLE_SCHEMA_CHANGES
    change_id = Column(Integer, Identity(), primary_key=True)
    schema_version = Column(Integer)
    changed = Column(DateTime(timezone=True), default=dt_util.utcnow)

    def __repr__(self) -> str:
        """Return string representation of instance for debugging."""
        return (f"<recorder.SchemaChanges(id={self.change_id}, schema_version={self.schema_version}, "
                f"changed='{self.changed.isoformat(sep=' ', timespec='seconds')}')>")


def process_timestamp(ts: Optional[datetime]) -> Optional[datetime]:
    """Process a timestamp into datetime object."""
    if ts is None:
        return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=dt_util.UTC)
    return dt_util.as_utc(ts)


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
        self._row: States = row
        self.entity_id: str = self._row.entity_id
        self.state: str = self._row.state or ''
        self._attributes: Optional[Dict[str, Any]] = None
        self._last_changed: Optional[datetime] = None
        self._last_updated: Optional[datetime] = None
        self._context: Optional[Context] = None

    @property
    def attributes(self) -> Dict[str, Any]:
        """State attributes."""
        if self._attributes is None:
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
        if self._context is None:
            self._context = Context(id=None)
        return self._context

    @context.setter
    def context(self, value: Context) -> None:
        """Set context."""
        self._context = value

    @property
    def last_changed(self) -> datetime:
        """Last changed datetime."""
        if self._last_changed is None:
            self._last_changed = process_timestamp(self._row.last_changed)  # type: ignore
        return self._last_changed

    @last_changed.setter
    def last_changed(self, value: datetime) -> None:
        """Set last changed datetime."""
        self._last_changed = value

    @property
    def last_updated(self) -> datetime:
        """Last updated datetime."""
        if self._last_updated is None:
            self._last_updated = process_timestamp(self._row.last_updated)  # type: ignore
        return self._last_updated

    @last_updated.setter
    def last_updated(self, value: datetime) -> None:
        """Set last updated datetime."""
        self._last_updated = value

    def as_dict(self) -> Dict[str, Any]:
        """Return a dict representation of the LazyState.

        Async friendly.

        To be used for JSON serialization.
        """
        if self._last_changed:
            last_changed_isoformat: str = self._last_changed.isoformat()
        else:
            last_changed_isoformat = process_timestamp_to_utc_isoformat(self._row.last_changed)  # type: ignore
        if self._last_updated:
            last_updated_isoformat: str = self._last_updated.isoformat()
        else:
            last_updated_isoformat = process_timestamp_to_utc_isoformat(self._row.last_updated)  # type: ignore
        return {
            'entity_id': self.entity_id,
            'state': self.state,
            'attributes': self._attributes or self.attributes,
            'last_changed': last_changed_isoformat,
            'last_updated': last_updated_isoformat
        }

    def __eq__(self, other: object) -> bool:
        """Return the comparison."""
        return (
            isinstance(other, (self.__class__, State)) and
            self.entity_id == other.entity_id and
            self.state == other.state and
            self.attributes == other.attributes
        )