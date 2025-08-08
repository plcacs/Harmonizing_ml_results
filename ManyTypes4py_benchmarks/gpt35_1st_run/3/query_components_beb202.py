import datetime
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Hashable, Iterable, Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NamedTuple, Optional, Union, cast
from uuid import UUID
import sqlalchemy as sa
from cachetools import Cache, TTLCache
from jinja2 import Environment, PackageLoader, select_autoescape
from sqlalchemy import orm
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.exc import NoResultFound
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.type_api import TypeEngine
from typing_extensions import TypeVar
from prefect.server import models, schemas
from prefect.server.database import orm_models
from prefect.server.database.dependencies import db_injector
from prefect.server.database.interface import PrefectDBInterface
from prefect.server.exceptions import FlowRunGraphTooLarge, ObjectNotFoundError
from prefect.server.schemas.graph import Edge, Graph, GraphArtifact, GraphState, Node
from prefect.server.schemas.states import StateType
from prefect.server.utilities.database import UUID as UUIDTypeDecorator
from prefect.server.utilities.database import Timestamp, bindparams_from_clause
from prefect.types._datetime import DateTime
T = TypeVar('T', covariant=True)

class FlowRunNotificationsFromQueue(NamedTuple):
    pass

class FlowRunGraphV2Node(NamedTuple):
    pass

ONE_HOUR: int = 60 * 60

jinja_env: Environment = Environment(loader=PackageLoader('prefect.server.database', package_path='sql'), autoescape=select_autoescape(), trim_blocks=True)

class BaseQueryComponents(ABC):
    _configuration_cache: TTLCache = TTLCache(maxsize=100, ttl=ONE_HOUR)

    def unique_key(self) -> tuple:
        return (self.__class__,)

    @abstractmethod
    def insert(self, obj):
        ...

    @property
    @abstractmethod
    def uses_json_strings(self):
        ...

    @abstractmethod
    def cast_to_json(self, json_obj):
        ...

    @abstractmethod
    def build_json_object(self, *args):
        ...

    @abstractmethod
    def json_arr_agg(self, json_array):
        ...

    @abstractmethod
    def make_timestamp_intervals(self, start_time, end_time, interval):
        ...

    @abstractmethod
    def set_state_id_on_inserted_flow_runs_statement(self, inserted_flow_run_ids, insert_flow_run_states):
        ...

    @abstractmethod
    async def get_flow_run_notifications_from_queue(self, session, limit):
        ...

    @db_injector
    async def queue_flow_run_notifications(self, db, session, flow_run):
        ...

    @db_injector
    def get_scheduled_flow_runs_from_work_queues(self, db, limit_per_queue=None, work_queue_ids=None, scheduled_before=None):
        ...

    @property
    @abstractmethod
    def _get_scheduled_flow_runs_from_work_pool_template_path(self):
        ...

    @db_injector
    async def get_scheduled_flow_runs_from_work_pool(self, db, session, limit=None, worker_limit=None, queue_limit=None, work_pool_ids=None, work_queue_ids=None, scheduled_before=None, scheduled_after=None, respect_queue_priorities=False):
        ...

    @db_injector
    async def read_configuration_value(self, db, session, key):
        ...

    def clear_configuration_value_cache_for_key(self, key):
        ...

    @cached_property
    def _flow_run_graph_v2_query(self):
        ...

    @abstractmethod
    def _build_flow_run_graph_v2_query(self):
        ...

    @db_injector
    async def flow_run_graph_v2(self, db, session, flow_run_id, since, max_nodes, max_artifacts):
        ...

    async def _get_flow_run_graph_artifacts(self, db, session, flow_run_id, max_artifacts):
        ...

    async def _get_flow_run_graph_states(self, session, flow_run_id):
        ...

class AsyncPostgresQueryComponents(BaseQueryComponents):

    def insert(self, obj):
        ...

    @property
    def uses_json_strings(self):
        ...

    def cast_to_json(self, json_obj):
        ...

    def build_json_object(self, *args):
        ...

    def json_arr_agg(self, json_array):
        ...

    def make_timestamp_intervals(self, start_time, end_time, interval):
        ...

    @db_injector
    def set_state_id_on_inserted_flow_runs_statement(self, db, inserted_flow_run_ids, insert_flow_run_states):
        ...

    @db_injector
    async def get_flow_run_notifications_from_queue(self, db, session, limit):
        ...

    @property
    def _get_scheduled_flow_runs_from_work_pool_template_path(self):
        ...

    @db_injector
    def _build_flow_run_graph_v2_query(self, db):
        ...

class UUIDList(sa.TypeDecorator[list[UUID]]):
    ...

class AioSqliteQueryComponents(BaseQueryComponents):

    def insert(self, obj):
        ...

    @property
    def uses_json_strings(self):
        ...

    def cast_to_json(self, json_obj):
        ...

    def build_json_object(self, *args):
        ...

    def json_arr_agg(self, json_array):
        ...

    def make_timestamp_intervals(self, start_time, end_time, interval):
        ...

    @db_injector
    def set_state_id_on_inserted_flow_runs_statement(self, db, inserted_flow_run_ids, insert_flow_run_states):
        ...

    @db_injector
    async def get_flow_run_notifications_from_queue(self, db, session, limit):
        ...

    @db_injector
    def _get_scheduled_flow_runs_join(self, db, work_queue_query, limit_per_queue, scheduled_before):
        ...

    @property
    def _get_scheduled_flow_runs_from_work_pool_template_path(self):
        ...

    @db_injector
    def _build_flow_run_graph_v2_query(self, db):
        ...
