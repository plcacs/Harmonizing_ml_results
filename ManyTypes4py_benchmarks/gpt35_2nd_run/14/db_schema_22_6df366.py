from __future__ import annotations
from collections.abc import Iterable
from datetime import datetime, timedelta
import json
import logging
from typing import TypedDict, overload
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Identity, Index, Integer, String, Text, distinct
from sqlalchemy.dialects import mysql, oracle, postgresql
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.orm.session import Session
from homeassistant.const import MAX_LENGTH_EVENT_CONTEXT_ID, MAX_LENGTH_EVENT_EVENT_TYPE, MAX_LENGTH_EVENT_ORIGIN, MAX_LENGTH_STATE_DOMAIN, MAX_LENGTH_STATE_ENTITY_ID, MAX_LENGTH_STATE_STATE
from homeassistant.core import Context, Event, EventOrigin, State, split_entity_id
from homeassistant.helpers.json import JSONEncoder
from homeassistant.util import dt as dt_util

Base: declarative_base
SCHEMA_VERSION: int = 22
_LOGGER: logging.Logger = logging.getLogger(__name__)
DB_TIMEZONE: str = '+00:00'
TABLE_EVENTS: str = 'events'
TABLE_STATES: str = 'states'
TABLE_RECORDER_RUNS: str = 'recorder_runs'
TABLE_SCHEMA_CHANGES: str = 'schema_changes'
TABLE_STATISTICS: str = 'statistics'
TABLE_STATISTICS_META: str = 'statistics_meta'
TABLE_STATISTICS_RUNS: str = 'statistics_runs'
TABLE_STATISTICS_SHORT_TERM: str = 'statistics_short_term'
ALL_TABLES: list[str] = [TABLE_STATES, TABLE_EVENTS, TABLE_RECORDER_RUNS, TABLE_SCHEMA_CHANGES, TABLE_STATISTICS, TABLE_STATISTICS_META, TABLE_STATISTICS_RUNS, TABLE_STATISTICS_SHORT_TERM]
DATETIME_TYPE: DateTime = DateTime(timezone=True).with_variant(mysql.DATETIME(timezone=True, fsp=6), 'mysql')
DOUBLE_TYPE: Float = Float().with_variant(mysql.DOUBLE(asdecimal=False), 'mysql').with_variant(oracle.DOUBLE_PRECISION(), 'oracle').with_variant(postgresql.DOUBLE_PRECISION(), 'postgresql')

class Events(Base):
    __table_args__: tuple = (Index('ix_events_event_type_time_fired', 'event_type', 'time_fired'), {'mysql_default_charset': 'utf8mb4', 'mysql_collate': 'utf8mb4_unicode_ci'})
    __tablename__: str = TABLE_EVENTS
    event_id: Column = Column(Integer, Identity(), primary_key=True)
    event_type: Column = Column(String(MAX_LENGTH_EVENT_EVENT_TYPE))
    event_data: Column = Column(Text().with_variant(mysql.LONGTEXT, 'mysql'))
    origin: Column = Column(String(MAX_LENGTH_EVENT_ORIGIN))
    time_fired: Column = Column(DATETIME_TYPE, index=True)
    created: Column = Column(DATETIME_TYPE, default=dt_util.utcnow)
    context_id: Column = Column(String(MAX_LENGTH_EVENT_CONTEXT_ID), index=True)
    context_user_id: Column = Column(String(MAX_LENGTH_EVENT_CONTEXT_ID), index=True)
    context_parent_id: Column = Column(String(MAX_LENGTH_EVENT_CONTEXT_ID), index=True)

    def __repr__(self) -> str:
        return f"<recorder.Events(id={self.event_id}, type='{self.event_type}', data='{self.event_data}', origin='{self.origin}', time_fired='{self.time_fired}')>"

    @staticmethod
    def from_event(event, event_data=None) -> Events:
        return Events(event_type=event.event_type, event_data=event_data or json.dumps(event.data, cls=JSONEncoder, separators=(',', ':')), origin=str(event.origin.value), time_fired=event.time_fired, context_id=event.context.id, context_user_id=event.context.user_id, context_parent_id=event.context.parent_id)

    def to_native(self, validate_entity_id=True) -> Event:
        context: Context = Context(id=self.context_id, user_id=self.context_user_id, parent_id=self.context_parent_id)
        try:
            return Event(self.event_type, json.loads(self.event_data), EventOrigin(self.origin), process_timestamp(self.time_fired), context=context)
        except ValueError:
            _LOGGER.exception('Error converting to event: %s', self)
            return None

class States(Base):
    __table_args__: tuple = (Index('ix_states_entity_id_last_updated', 'entity_id', 'last_updated'), {'mysql_default_charset': 'utf8mb4', 'mysql_collate': 'utf8mb4_unicode_ci'})
    __tablename__: str = TABLE_STATES
    state_id: Column = Column(Integer, Identity(), primary_key=True)
    domain: Column = Column(String(MAX_LENGTH_STATE_DOMAIN))
    entity_id: Column = Column(String(MAX_LENGTH_STATE_ENTITY_ID))
    state: Column = Column(String(MAX_LENGTH_STATE_STATE))
    attributes: Column = Column(Text().with_variant(mysql.LONGTEXT, 'mysql'))
    event_id: Column = Column(Integer, ForeignKey('events.event_id', ondelete='CASCADE'), index=True)
    last_changed: Column = Column(DATETIME_TYPE, default=dt_util.utcnow)
    last_updated: Column = Column(DATETIME_TYPE, default=dt_util.utcnow, index=True)
    created: Column = Column(DATETIME_TYPE, default=dt_util.utcnow)
    old_state_id: Column = Column(Integer, ForeignKey('states.state_id'), index=True)
    event: relationship = relationship('Events', uselist=False)
    old_state: relationship = relationship('States', remote_side=[state_id])

    def __repr__(self) -> str:
        return f"<recorder.States(id={self.state_id}, domain='{self.domain}', entity_id='{self.entity_id}', state='{self.state}', event_id='{self.event_id}', last_updated='{self.last_updated.isoformat(sep=' ', timespec='seconds')}', old_state_id={self.old_state_id})>"

    @staticmethod
    def from_event(event) -> States:
        entity_id: str = event.data['entity_id']
        state = event.data.get('new_state')
        dbstate: States = States(entity_id=entity_id)
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

    def to_native(self, validate_entity_id=True) -> State:
        try:
            return State(self.entity_id, self.state, json.loads(self.attributes), process_timestamp(self.last_changed), process_timestamp(self.last_updated), context=Context(id=None), validate_entity_id=validate_entity_id)
        except ValueError:
            _LOGGER.exception('Error converting row to state: %s', self)
            return None

class StatisticResult(TypedDict):
    ...

class StatisticDataBase(TypedDict):
    ...

class StatisticData(StatisticDataBase, total=False):
    ...

class StatisticsBase:
    ...

class Statistics(Base, StatisticsBase):
    ...

class StatisticsShortTerm(Base, StatisticsBase):
    ...

class StatisticMetaData(TypedDict):
    ...

class StatisticsMeta(Base):
    ...

class RecorderRuns(Base):
    ...

class SchemaChanges(Base):
    ...

class StatisticsRuns(Base):
    ...

@overload
def process_timestamp(ts) -> None:
    ...

@overload
def process_timestamp(ts) -> None:
    ...

def process_timestamp(ts) -> None:
    ...

@overload
def process_timestamp_to_utc_isoformat(ts) -> None:
    ...

@overload
def process_timestamp_to_utc_isoformat(ts) -> None:
    ...

def process_timestamp_to_utc_isoformat(ts) -> None:
    ...

class LazyState(State):
    ...
