import datetime
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Hashable, Iterable, Sequence
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
)
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

T = TypeVar("T", infer_variance=True)


class FlowRunNotificationsFromQueue(NamedTuple):
    queue_id: UUID
    flow_run_notification_policy_id: UUID
    flow_run_notification_policy_message_template: Optional[str]
    block_document_id: UUID
    flow_id: UUID
    flow_name: str
    flow_run_id: UUID
    flow_run_name: str
    flow_run_parameters: Dict[str, Any]
    flow_run_state_type: StateType
    flow_run_state_name: str
    flow_run_state_timestamp: DateTime
    flow_run_state_message: Optional[str]


class FlowRunGraphV2Node(NamedTuple):
    kind: Literal["flow-run", "task-run"]
    id: UUID
    label: str
    state_type: StateType
    start_time: DateTime
    end_time: Optional[DateTime]
    parent_ids: Optional[List[UUID]]
    child_ids: Optional[List[UUID]]
    encapsulating_ids: Optional[List[UUID]]


ONE_HOUR: int = 60 * 60


jinja_env: Environment = Environment(
    loader=PackageLoader("prefect.server.database", package_path="sql"),
    autoescape=select_autoescape(),
    trim_blocks=True,
)


class BaseQueryComponents(ABC):
    """
    Abstract base class used to inject dialect-specific SQL operations into Prefect.
    """

    _configuration_cache: ClassVar[Cache[str, Dict[str, Any]]] = TTLCache(
        maxsize=100, ttl=ONE_HOUR
    )

    def unique_key(self) -> Tuple[Hashable, ...]:
        """
        Returns a key used to determine whether to instantiate a new DB interface.
        """
        return (self.__class__,)

    # --- dialect-specific SqlAlchemy bindings

    @abstractmethod
    def insert(
        self, obj: Type[orm_models.Base]
    ) -> Union[postgresql.Insert, sqlite.Insert]:
        """dialect-specific insert statement"""

    # --- dialect-specific JSON handling

    @property
    @abstractmethod
    def uses_json_strings(self) -> bool:
        """specifies whether the configured dialect returns JSON as strings"""

    @abstractmethod
    def cast_to_json(self, json_obj: sa.ColumnElement[T]) -> sa.ColumnElement[T]:
        """casts to JSON object if necessary"""

    @abstractmethod
    def build_json_object(
        self, *args: Union[str, sa.ColumnElement[Any]]
    ) -> sa.ColumnElement[Any]:
        """builds a JSON object from sequential key-value pairs"""

    @abstractmethod
    def json_arr_agg(self, json_array: sa.ColumnElement[Any]) -> sa.ColumnElement[Any]:
        """aggregates a JSON array"""

    # --- dialect-optimized subqueries

    @abstractmethod
    def make_timestamp_intervals(
        self,
        start_time: DateTime,
        end_time: DateTime,
        interval: datetime.timedelta,
    ) -> sa.Select[Tuple[DateTime, DateTime]]: ...

    @abstractmethod
    def set_state_id_on_inserted_flow_runs_statement(
        self,
        inserted_flow_run_ids: Sequence[UUID],
        insert_flow_run_states: Iterable[Dict[str, Any]],
    ) -> sa.Update: ...

    @abstractmethod
    async def get_flow_run_notifications_from_queue(
        self, session: AsyncSession, limit: int
    ) -> Sequence[FlowRunNotificationsFromQueue]:
        """Database-specific implementation of reading notifications from the queue and deleting them"""

    @db_injector
    async def queue_flow_run_notifications(
        self,
        db: PrefectDBInterface,
        session: AsyncSession,
        flow_run: Union[schemas.core.FlowRun, orm_models.FlowRun],
    ) -> None:
        """Database-specific implementation of queueing notifications for a flow run"""

        def as_array(elems: Sequence[str]) -> sa.ColumnElement[Sequence[str]]:
            return sa.cast(postgresql.array(elems), type_=postgresql.ARRAY(sa.String()))

        if TYPE_CHECKING:
            assert flow_run.state_name is not None

        FlowRunNotificationQueue = db.FlowRunNotificationQueue
        FlowRunNotificationPolicy = db.FlowRunNotificationPolicy

        # insert a <policy, state> pair into the notification queue
        stmt = self.insert(FlowRunNotificationQueue).from_select(
            [
                FlowRunNotificationQueue.flow_run_notification_policy_id,
                FlowRunNotificationQueue.flow_run_state_id,
            ],
            # ... by selecting from any notification policy that matches the criteria
            sa.select(
                FlowRunNotificationPolicy.id,
                sa.cast(sa.literal(str(flow_run.state_id)), UUIDTypeDecorator),
            )
            .select_from(FlowRunNotificationPolicy)
            .where(
                sa.and_(
                    # the policy is active
                    FlowRunNotificationPolicy.is_active.is_(True),
                    # the policy state names aren't set or match the current state name
                    sa.or_(
                        FlowRunNotificationPolicy.state_names == [],
                        FlowRunNotificationPolicy.state_names.has_any(
                            as_array([flow_run.state_name])
                        ),
                    ),
                    # the policy tags aren't set, or the tags match the flow run tags
                    sa.or_(
                        FlowRunNotificationPolicy.tags == [],
                        FlowRunNotificationPolicy.tags.has_any(as_array(flow_run.tags)),
                    ),
                )
            ),
            # don't send python defaults as part of the insert statement, because they are
            # evaluated once per statement and create unique constraint violations on each row
            include_defaults=False,
        )
        await session.execute(stmt)

    @db_injector
    def get_scheduled_flow_runs_from_work_queues(
        self,
        db: PrefectDBInterface,
        limit_per_queue: Optional[int] = None,
        work_queue_ids: Optional[List[UUID]] = None,
        scheduled_before: Optional[DateTime] = None,
    ) -> sa.Select[Tuple[orm_models.FlowRun, UUID]]:
        """
        Returns all scheduled runs in work queues, subject to provided parameters.

        This query returns a `(orm_models.FlowRun, orm_models.WorkQueue.id)` pair; calling
        `result.all()` will return both; calling `result.scalars().unique().all()`
        will return only the flow run because it grabs the first result.
        """

        FlowRun, WorkQueue = db.FlowRun, db.WorkQueue

        # get any work queues that have a concurrency limit, and compute available
        # slots as their limit less the number of running flows
        concurrency_queues = (
            sa.select(
                WorkQueue.id,
                sa.func.greatest(
                    0,
                    WorkQueue.concurrency_limit - sa.func.count(FlowRun.id),
                ).label("available_slots"),
            )
            .select_from(WorkQueue)
            .join(
                FlowRun,
                sa.and_(
                    FlowRun.work_queue_name == WorkQueue.name,
                    FlowRun.state_type.in_(
                        (StateType.RUNNING, StateType.PENDING, StateType.CANCELLING)
                    ),
                ),
                isouter=True,
            )
            .where(WorkQueue.concurrency_limit.is_not(None))
            .group_by(WorkQueue.id)
            .cte("concurrency_queues")
        )

        # use the available slots information to generate a join
        # for all scheduled runs
        scheduled_flow_runs, join_criteria = self._get_scheduled_flow_runs_join(
            work_queue_query=concurrency_queues,
            limit_per_queue=limit_per_queue,
            scheduled_before=scheduled_before,
        )

        # starting with the work queue table, join the limited queues to get the
        # concurrency information and the scheduled flow runs to load all applicable
        # runs. this will return all the scheduled runs allowed by the parameters
        query = (
            # return a flow run and work queue id
            sa.select(
                orm.aliased(FlowRun, scheduled_flow_runs), WorkQueue.id.label("wq_id")
            )
            .select_from(WorkQueue)
            .join(
                concurrency_queues,
                WorkQueue.id == concurrency_queues.c.id,
                isouter=True,
            )
            .join(scheduled_flow_runs, join_criteria)
            .where(
                WorkQueue.is_paused.is_(False),
                WorkQueue.id.in_(work_queue_ids) if work_queue_ids else sa.true(),
            )
            .order_by(
                scheduled_flow_runs.c.next_scheduled_start_time,
                scheduled_flow_runs.c.id,
            )
        )

        return query

    @db_injector
    def _get_scheduled_flow_runs_join(
        self,
        db: PrefectDBInterface,
        work_queue_query: sa.CTE,
        limit_per_queue: Optional[int],
        scheduled_before: Optional[DateTime],
    ) -> Tuple[sa.FromClause, sa.ColumnExpressionArgument[bool]]:
        """Used by self.get_scheduled_flow_runs_from_work_queue, allowing just
        this function to be changed on a per-dialect basis"""

        FlowRun = db.FlowRun

        # precompute for readability
        scheduled_before_clause = (
            FlowRun.next_scheduled_start_time <= scheduled_before
            if scheduled_before is not None
            else sa.true()
        )

        # get scheduled flow runs with lateral join where the limit is the
        # available slots per queue
        scheduled_flow_runs = (
            sa.select(FlowRun)
            .where(
                FlowRun.work_queue_name == db.WorkQueue.name,
                FlowRun.state_type == StateType.SCHEDULED,
                scheduled_before_clause,
            )
            .with_for_update(skip_locked=True)
            # priority given to runs with earlier next_scheduled_start_time
            .order_by(FlowRun.next_scheduled_start_time)
            # if null, no limit will be applied
            .limit(sa.func.least(limit_per_queue, work_queue_query.c.available_slots))
            .lateral("scheduled_flow_runs")
        )

        # Perform a cross-join
        join_criteria = sa.true()

        return scheduled_flow_runs, join_criteria

    # -------------------------------------------------------
    # Workers
    # -------------------------------------------------------

    @property
    @abstractmethod
    def _get_scheduled_flow_runs_from_work_pool_template_path(self) -> str:
        """
        Template for the query to get scheduled flow runs from a work pool
        """

    @db_injector
    async def get_scheduled_flow_runs_from_work_pool(
        self,
        db: PrefectDBInterface,
        session: AsyncSession,
        limit: Optional[int] = None,
        worker_limit: Optional[int] = None,
        queue_limit: Optional[int] = None,
        work_pool_ids: Optional[List[UUID]] = None,
        work_queue_ids: Optional[List[UUID]] = None,
        scheduled_before: Optional[DateTime] = None,
        scheduled_after: Optional[DateTime] = None,
        respect_queue_priorities: bool = False,
    ) -> List[schemas.responses.WorkerFlowRunResponse]:
        template = jinja_env.get_template(
            self._get_scheduled_flow_runs_from_work_pool_template_path
        )

        raw_query = sa.text(
            template.render(
                work_pool_ids=work_pool_ids,
                work_queue_ids=work_queue_ids,
                respect_queue_priorities=respect_queue_priorities,
                scheduled_before=scheduled_before,
                scheduled_after=scheduled_after,
            )
        )

        bindparams: List[sa.BindParameter[Any]] = []

        if scheduled_before:
            bindparams.append(
                sa.bindparam("scheduled_before", scheduled_before, type_=Timestamp)
            )

        if scheduled_after:
            bindparams.append(
                sa.bindparam("scheduled_after", scheduled_after, type_=Timestamp)
            )

        # if work pool IDs were provided, bind them
        if work_pool_ids:
            assert all(isinstance(i, UUID) for i in work_pool_ids)
            bindparams.append(
                sa.bindparam(
                    "work_pool_ids",
                    work_pool_ids,
                    expanding=True,
                    type_=UUIDTypeDecorator,
                )
            )

        # if work queue IDs were provided, bind them
        if work_queue_ids:
            assert all(isinstance(i, UUID) for i in work_queue_ids)
            bindparams.append(
                sa.bindparam(
                    "work_queue_ids",
                    work_queue_ids,
                    expanding=True,
                    type_=UUIDTypeDecorator,
                )
            )

        query = raw_query.bindparams(
            *bindparams,
            limit=1000 if limit is None else limit,
            worker_limit=1000 if worker_limit is None else worker_limit,
            queue_limit=1000 if queue_limit is None else queue_limit,
        )

        FlowRun = db.FlowRun
        orm_query = (
            sa.select(
                sa.column("run_work_pool_id", UUIDTypeDecorator),
                sa.column("run_work_queue_id", UUIDTypeDecorator),
                FlowRun,
            )
            .from_statement(query)
            # indicate that the state relationship isn't being loaded
            .options(orm.noload(FlowRun.state))
        )

        result: sa.Result[
            Tuple[UUID, UUID, orm_models.FlowRun]
        ] = await session.execute(orm_query)

        return [
            schemas.responses.WorkerFlowRunResponse(
                work_pool_id=run_work_pool_id,
                work_queue_id=run_work_queue_id,
                flow_run=schemas.core.FlowRun.model_validate(
                    flow_run, from_attributes=True
                ),
            )
            for (run_work_pool_id, run_work_queue_id, flow_run) in result.t
        ]

    @db_injector
    async def read_configuration_value(
        self, db: PrefectDBInterface, session: AsyncSession, key: str
    ) -> Optional[Dict[str, Any]]:
        """
        Read a configuration value by key.

        Configuration values should not be changed at run time, so retrieved
        values are cached in memory.

        The main use of configurations is encrypting blocks, this speeds up nested
        block document queries.
        """
        Configuration = db.Configuration
        value = None
        try:
            value = self._configuration_cache[key]
        except KeyError:
            query = sa.select(Configuration).where(Configuration.key == key)
            if (configuration := await session.scalar(query)) is not None:
                value = self._configuration_cache[key] = configuration.value
        return value

    def clear_configuration_value_cache_for_key(self, key: str) -> None:
        """Removes a configuration key from the cache."""
        self._configuration_cache.pop(key, None)

    @cached_property
    def _flow_run_graph_v2_query(self) -> sa.Select[FlowRunGraphV2Node]:
        query = self._build_flow_run_graph_v2_query()
        param_names = set(bindparams_from_clause(query))
        required = {"flow_run_id", "max_nodes", "since"}
        assert param_names >= required, (
            "_build_flow_run_graph_v2_query result is missing required bind params: "
            f"{sorted(required - param_names)}"
        )
        return query

    @abstractmethod
    def _build_flow_run_graph_v2_query(self) -> sa.Select[FlowRunGraphV2Node]:
        """The flow run graph query, per database flavour

        The query must accept the following bind parameters:

            flow_run_id: UUID
            since: DateTime
            max_nodes: int

        """

    @db_injector
    async def flow_run_graph_v2(
        self,
        db: PrefectDBInterface,
        session: AsyncSession,
        flow_run_id: UUID,
        since: DateTime,
        max_nodes: int,
        max_artifacts: int,
    ) -> Graph:
        """Returns the query that selects all of the nodes and edges for a flow run graph (version 2)."""
        FlowRun = db.FlowRun
        result = await session.execute(
            sa.select(
                sa.func.coalesce(
                    FlowRun.start_time, FlowRun.expected_start_time, type_=Timestamp
                ),
                FlowRun.end_time,
            ).where(FlowRun.id == flow_run_id)
        )
        try:
            start_time, end_time = result.t.one()
        except NoResultFound:
            raise ObjectNotFoundError(f"Flow run {flow_run_id} not found")

        query = self._flow_run_graph_v2_query
        results = await session.execute(
            query,
            params=dict(flow_run_id=flow_run_id, since=since, max_nodes=max_nodes + 1),
        )

        graph_artifacts = await self._get_flow_run_graph_artifacts(
            db, session, flow_run_id, max_artifacts
        )
        graph_states = await self._get_flow_run_graph_states(session, flow_run_id)

        nodes: List