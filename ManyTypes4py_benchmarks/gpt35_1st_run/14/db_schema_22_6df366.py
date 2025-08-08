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
SCHEMA_VERSION: int
_LOGGER: logging.Logger
DB_TIMEZONE: str
TABLE_EVENTS: str
TABLE_STATES: str
TABLE_RECORDER_RUNS: str
TABLE_SCHEMA_CHANGES: str
TABLE_STATISTICS: str
TABLE_STATISTICS_META: str
TABLE_STATISTICS_RUNS: str
TABLE_STATISTICS_SHORT_TERM: str
ALL_TABLES: list[str]
DATETIME_TYPE: DateTime
DOUBLE_TYPE: Float

class Events(Base):
    event_id: Column
    event_type: Column
    event_data: Column
    origin: Column
    time_fired: Column
    created: Column
    context_id: Column
    context_user_id: Column
    context_parent_id: Column

    def __repr__(self) -> str:
        ...

    @staticmethod
    def from_event(event, event_data=None) -> Events:
        ...

    def to_native(self, validate_entity_id=True) -> Event:
        ...

class States(Base):
    state_id: Column
    domain: Column
    entity_id: Column
    state: Column
    attributes: Column
    event_id: Column
    last_changed: Column
    last_updated: Column
    created: Column
    old_state_id: Column
    event: relationship
    old_state: relationship

    def __repr__(self) -> str:
        ...

    @staticmethod
    def from_event(event) -> States:
        ...

    def to_native(self, validate_entity_id=True) -> State:
        ...

class StatisticResult(TypedDict):
    ...

class StatisticDataBase(TypedDict):
    ...

class StatisticData(StatisticDataBase, total=False):
    ...

class StatisticsBase:
    id: Column
    created: Column
    metadata_id: Column
    start: Column
    mean: Column
    min: Column
    max: Column
    last_reset: Column
    state: Column
    sum: Column

    @declared_attr
    def metadata_id(self) -> Column:
        ...

    @classmethod
    def from_stats(cls, metadata_id, stats) -> StatisticsBase:
        ...

class Statistics(Base, StatisticsBase):
    duration: timedelta

class StatisticsShortTerm(Base, StatisticsBase):
    duration: timedelta

class StatisticMetaData(TypedDict):
    ...

class StatisticsMeta(Base):
    id: Column
    statistic_id: Column
    source: Column
    unit_of_measurement: Column
    has_mean: Column
    has_sum: Column

    @staticmethod
    def from_meta(source, statistic_id, unit_of_measurement, has_mean, has_sum) -> StatisticsMeta:
        ...

class RecorderRuns(Base):
    run_id: Column
    start: Column
    end: Column
    closed_incorrect: Column
    created: Column

    def __repr__(self) -> str:
        ...

    def entity_ids(self, point_in_time=None) -> list[str]:
        ...

    def to_native(self, validate_entity_id=True) -> RecorderRuns:
        ...

class SchemaChanges(Base):
    change_id: Column
    schema_version: Column
    changed: Column

    def __repr__(self) -> str:
        ...

class StatisticsRuns(Base):
    run_id: Column
    start: Column

    def __repr__(self) -> str:
        ...

@overload
def process_timestamp(ts) -> None:
    ...

@overload
def process_timestamp(ts) -> None:
    ...

def process_timestamp(ts) -> datetime:
    ...

@overload
def process_timestamp_to_utc_isoformat(ts) -> None:
    ...

@overload
def process_timestamp_to_utc_isoformat(ts) -> None:
    ...

def process_timestamp_to_utc_isoformat(ts) -> str:
    ...

class LazyState(State):
    __slots__: list[str]

    def __init__(self, row) -> None:
        ...

    @property
    def attributes(self) -> dict:
        ...

    @attributes.setter
    def attributes(self, value) -> None:
        ...

    @property
    def context(self) -> Context:
        ...

    @context.setter
    def context(self, value) -> None:
        ...

    @property
    def last_changed(self) -> datetime:
        ...

    @last_changed.setter
    def last_changed(self, value) -> None:
        ...

    @property
    def last_updated(self) -> datetime:
        ...

    @last_updated.setter
    def last_updated(self, value) -> None:
        ...

    def as_dict(self) -> dict:
        ...

    def __eq__(self, other) -> bool:
        ...
