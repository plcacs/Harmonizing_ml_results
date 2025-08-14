#!/usr/bin/env python3
"""
Annotated Python program with type annotations.
"""

import datetime
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Hashable, Iterable, Sequence
from functools import cached_property
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
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
from sqlalchemy.sql.elements import BindParameter
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


class FlowRunNotificationsFromQueue(
    NamedTuple("FlowRunNotificationsFromQueue", [
        ("queue_id", UUID),
        ("flow_run_notification_policy_id", UUID),
        ("flow_run_notification_policy_message_template", Optional[str]),
        ("block_document_id", UUID),
        ("flow_id", UUID),
        ("flow_name", str),
        ("flow_run_id", UUID),
        ("flow_run_name", str),
        ("flow_run_parameters", Dict[str, Any]),
        ("flow_run_state_type", StateType),
        ("flow_run_state_name", str),
        ("flow_run_state_timestamp", DateTime),
        ("flow_run_state_message", Optional[str]),
    ])
):
    pass


class FlowRunGraphV2Node(
    NamedTuple("FlowRunGraphV2Node", [
        ("kind", Union[Literal["flow-run"], Literal["task-run"]]),
        ("id", UUID),
        ("label", str),
        ("state_type", StateType),
        ("start_time", DateTime),
        ("end_time", Optional[DateTime]),
        ("parent_ids", Optional[List[UUID]]),
        ("child_ids", Optional[List[UUID]]),
        ("encapsulating_ids", Optional[List[UUID]]),
    ])
):
    pass


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
    def insert(self, obj: type[orm_models.Base]) -> Union[postgresql.Insert, sqlite.Insert]:
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
    def build_json_object(self, *args: Union[str, sa.ColumnElement[Any]]) -> sa.ColumnElement[Any]:
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
    ) -> sa.Select[Tuple[DateTime, DateTime]]:
        ...

    @abstractmethod
    def set_state_id_on_inserted_flow_runs_statement(
        self,
        inserted_flow_run_ids: Sequence[UUID],
        insert_flow_run_states: Iterable[Dict[str, Any]],
    ) -> sa.Update:
        ...

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

        if hasattr(flow_run, "state_name") and flow_run.state_name is None:
            raise ValueError("flow_run.state_name must not be None")  # for type checking

        FlowRunNotificationQueue = db.FlowRunNotificationQueue
        FlowRunNotificationPolicy = db.FlowRunNotificationPolicy

        stmt = self.insert(FlowRunNotificationQueue).from_select(
            [
                FlowRunNotificationQueue.flow_run_notification_policy_id,
                FlowRunNotificationQueue.flow_run_state_id,
            ],
            sa.select(
                FlowRunNotificationPolicy.id,
                sa.cast(sa.literal(str(flow_run.state_id)), UUIDTypeDecorator),
            )
            .select_from(FlowRunNotificationPolicy)
            .where(
                sa.and_(
                    FlowRunNotificationPolicy.is_active.is_(True),
                    sa.or_(
                        FlowRunNotificationPolicy.state_names == [],
                        FlowRunNotificationPolicy.state_names.has_any(
                            as_array([flow_run.state_name])
                        ),
                    ),
                    sa.or_(
                        FlowRunNotificationPolicy.tags == [],
                        FlowRunNotificationPolicy.tags.has_any(as_array(flow_run.tags)),
                    ),
                )
            ),
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
        """
        FlowRun, WorkQueue = db.FlowRun, db.WorkQueue

        concurrency_queues: sa.CTE = (
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

        scheduled_flow_runs, join_criteria = self._get_scheduled_flow_runs_join(
            work_queue_query=concurrency_queues,
            limit_per_queue=limit_per_queue,
            scheduled_before=scheduled_before,
        )

        query: sa.Select[Tuple[orm_models.FlowRun, UUID]] = (
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
    ) -> Tuple[sa.FromClause, sa.ColumnExpression[Any]]:
        """Used by self.get_scheduled_flow_runs_from_work_queue"""
        FlowRun = db.FlowRun

        scheduled_before_clause = (
            FlowRun.next_scheduled_start_time <= scheduled_before
            if scheduled_before is not None
            else sa.true()
        )

        scheduled_flow_runs = (
            sa.select(FlowRun)
            .where(
                FlowRun.work_queue_name == db.WorkQueue.name,
                FlowRun.state_type == StateType.SCHEDULED,
                scheduled_before_clause,
            )
            .with_for_update(skip_locked=True)
            .order_by(FlowRun.next_scheduled_start_time)
            .limit(sa.func.least(limit_per_queue, work_queue_query.c.available_slots))
            .lateral("scheduled_flow_runs")
        )

        join_criteria: sa.ColumnExpression[Any] = sa.true()

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

        raw_query: sa.TextClause = sa.text(
            template.render(
                work_pool_ids=work_pool_ids,
                work_queue_ids=work_queue_ids,
                respect_queue_priorities=respect_queue_priorities,
                scheduled_before=scheduled_before,
                scheduled_after=scheduled_after,
            )
        )

        bindparams: List[BindParameter[Any]] = []

        if scheduled_before:
            bindparams.append(
                sa.bindparam("scheduled_before", scheduled_before, type_=Timestamp)
            )

        if scheduled_after:
            bindparams.append(
                sa.bindparam("scheduled_after", scheduled_after, type_=Timestamp)
            )

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
            .options(orm.noload(FlowRun.state))
        )

        result: sa.Result[Tuple[UUID, UUID, orm_models.FlowRun]] = await session.execute(orm_query)

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
        """
        Configuration = db.Configuration
        value: Optional[Dict[str, Any]] = None
        try:
            value = self._configuration_cache[key]
        except KeyError:
            query = sa.select(Configuration).where(Configuration.key == key)
            configuration = await session.scalar(query)
            if configuration is not None:
                value = self._configuration_cache[key] = configuration.value
        return value

    def clear_configuration_value_cache_for_key(self, key: str) -> None:
        """Removes a configuration key from the cache."""
        self._configuration_cache.pop(key, None)

    @cached_property
    def _flow_run_graph_v2_query(self) -> sa.Select[FlowRunGraphV2Node]:
        query: sa.Select[FlowRunGraphV2Node] = self._build_flow_run_graph_v2_query()
        param_names = set(bindparams_from_clause(query))
        required = {"flow_run_id", "max_nodes", "since"}
        assert param_names >= required, (
            "_build_flow_run_graph_v2_query result is missing required bind params: "
            f"{sorted(required - param_names)}"
        )
        return query

    @abstractmethod
    def _build_flow_run_graph_v2_query(self) -> sa.Select[FlowRunGraphV2Node]:
        """The flow run graph query, per database flavour.
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

        graph_artifacts: Dict[Optional[UUID], List[GraphArtifact]] = await self._get_flow_run_graph_artifacts(
            db, session, flow_run_id, max_artifacts
        )
        graph_states: List[GraphState] = await self._get_flow_run_graph_states(session, flow_run_id)

        nodes: List[Tuple[UUID, Node]] = []
        root_node_ids: List[UUID] = []

        for row in results.t:
            if not row.parent_ids:
                root_node_ids.append(row.id)

            nodes.append(
                (
                    row.id,
                    Node(
                        kind=row.kind,
                        id=row.id,
                        label=row.label,
                        state_type=row.state_type,
                        start_time=row.start_time,
                        end_time=row.end_time,
                        parents=[Edge(id=pid) for pid in row.parent_ids or []],
                        children=[Edge(id=cid) for cid in row.child_ids or []],
                        encapsulating=[
                            Edge(id=eid)
                            for eid in dict.fromkeys(row.encapsulating_ids or ())
                        ],
                        artifacts=graph_artifacts.get(row.id, []),
                    ),
                )
            )

            if len(nodes) > max_nodes:
                raise FlowRunGraphTooLarge(
                    f"The graph of flow run {flow_run_id} has more than {max_nodes} nodes."
                )

        return Graph(
            start_time=start_time,
            end_time=end_time,
            root_node_ids=root_node_ids,
            nodes=nodes,
            artifacts=graph_artifacts.get(None, []),
            states=graph_states,
        )

    async def _get_flow_run_graph_artifacts(
        self,
        db: PrefectDBInterface,
        session: AsyncSession,
        flow_run_id: UUID,
        max_artifacts: int,
    ) -> Dict[Optional[UUID], List[GraphArtifact]]:
        Artifact, ArtifactCollection = db.Artifact, db.ArtifactCollection

        query = (
            sa.select(Artifact, ArtifactCollection.id.label("latest_in_collection_id"))
            .where(Artifact.flow_run_id == flow_run_id, Artifact.type != "result")
            .join(
                ArtifactCollection,
                onclause=sa.and_(
                    ArtifactCollection.key == Artifact.key,
                    ArtifactCollection.latest_id == Artifact.id,
                ),
                isouter=True,
            )
            .order_by(Artifact.created.asc())
            .limit(max_artifacts)
        )

        results = await session.execute(query)

        artifacts_by_task: Dict[Optional[UUID], List[GraphArtifact]] = defaultdict(list)
        for artifact, latest_in_collection_id in results.t:
            artifacts_by_task[artifact.task_run_id].append(
                GraphArtifact(
                    id=artifact.id,
                    created=artifact.created,
                    key=artifact.key,
                    type=artifact.type,
                    data=artifact.data if artifact.type == "progress" else None,
                    is_latest=artifact.key is None or latest_in_collection_id is not None,
                )
            )

        return dict(artifacts_by_task)

    async def _get_flow_run_graph_states(
        self, session: AsyncSession, flow_run_id: UUID
    ) -> List[GraphState]:
        states = await models.flow_run_states.read_flow_run_states(session, flow_run_id)
        return [GraphState.model_validate(state, from_attributes=True) for state in states]


class AsyncPostgresQueryComponents(BaseQueryComponents):
    # --- Postgres-specific SqlAlchemy bindings

    def insert(self, obj: type[orm_models.Base]) -> postgresql.Insert:
        return postgresql.insert(obj)

    # --- Postgres-specific JSON handling

    @property
    def uses_json_strings(self) -> bool:
        return False

    def cast_to_json(self, json_obj: sa.ColumnElement[T]) -> sa.ColumnElement[T]:
        return json_obj

    def build_json_object(
        self, *args: Union[str, sa.ColumnElement[Any]]
    ) -> sa.ColumnElement[Any]:
        return sa.func.jsonb_build_object(*args)

    def json_arr_agg(self, json_array: sa.ColumnElement[Any]) -> sa.ColumnElement[Any]:
        return sa.func.jsonb_agg(json_array)

    # --- Postgres-optimized subqueries

    def make_timestamp_intervals(
        self,
        start_time: DateTime,
        end_time: DateTime,
        interval: datetime.timedelta,
    ) -> sa.Select[Tuple[DateTime, DateTime]]:
        dt = sa.func.generate_series(start_time, end_time, interval, type_=Timestamp()).column_valued("dt")
        return (
            sa.select(
                dt.label("interval_start"),
                sa.type_coerce(
                    dt + sa.bindparam("interval", interval, type_=sa.Interval()),
                    type_=Timestamp(),
                ).label("interval_end"),
            )
            .where(dt < end_time)
            .limit(500)
        )

    @db_injector
    def set_state_id_on_inserted_flow_runs_statement(
        self,
        db: PrefectDBInterface,
        inserted_flow_run_ids: Sequence[UUID],
        insert_flow_run_states: Iterable[Dict[str, Any]],
    ) -> sa.Update:
        FlowRun, FlowRunState = db.FlowRun, db.FlowRunState
        stmt = (
            sa.update(FlowRun)
            .where(
                FlowRun.id.in_(inserted_flow_run_ids),
                FlowRunState.flow_run_id == FlowRun.id,
                FlowRunState.id.in_([r["id"] for r in insert_flow_run_states]),
            )
            .values(state_id=FlowRunState.id)
            .execution_options(synchronize_session=False)
        )
        return stmt

    @db_injector
    async def get_flow_run_notifications_from_queue(
        self, db: PrefectDBInterface, session: AsyncSession, limit: int
    ) -> Sequence[FlowRunNotificationsFromQueue]:
        Flow, FlowRun = db.Flow, db.FlowRun
        FlowRunNotificationPolicy = db.FlowRunNotificationPolicy
        FlowRunNotificationQueue = db.FlowRunNotificationQueue
        FlowRunState = db.FlowRunState

        queued_notifications_ids: sa.CTE = (
            sa.select(FlowRunNotificationQueue.id)
            .select_from(FlowRunNotificationQueue)
            .order_by(FlowRunNotificationQueue.updated)
            .limit(limit)
            .with_for_update(skip_locked=True)
        ).cte("queued_notifications_ids")

        queued_notifications: sa.CTE = (
            sa.delete(FlowRunNotificationQueue)
            .returning(
                FlowRunNotificationQueue.id,
                FlowRunNotificationQueue.flow_run_notification_policy_id,
                FlowRunNotificationQueue.flow_run_state_id,
            )
            .where(FlowRunNotificationQueue.id.in_(sa.select(queued_notifications_ids)))
            .cte("queued_notifications")
        )

        notification_details_stmt: sa.Select[FlowRunNotificationsFromQueue] = (
            sa.select(
                queued_notifications.c.id.label("queue_id"),
                FlowRunNotificationPolicy.id.label("flow_run_notification_policy_id"),
                FlowRunNotificationPolicy.message_template.label("flow_run_notification_policy_message_template"),
                FlowRunNotificationPolicy.block_document_id,
                Flow.id.label("flow_id"),
                Flow.name.label("flow_name"),
                FlowRun.id.label("flow_run_id"),
                FlowRun.name.label("flow_run_name"),
                FlowRun.parameters.label("flow_run_parameters"),
                FlowRunState.type.label("flow_run_state_type"),
                FlowRunState.name.label("flow_run_state_name"),
                FlowRunState.timestamp.label("flow_run_state_timestamp"),
                FlowRunState.message.label("flow_run_state_message"),
            )
            .select_from(queued_notifications)
            .join(
                FlowRunNotificationPolicy,
                queued_notifications.c.flow_run_notification_policy_id == FlowRunNotificationPolicy.id,
            )
            .join(
                FlowRunState,
                queued_notifications.c.flow_run_state_id == FlowRunState.id,
            )
            .join(FlowRun, FlowRunState.flow_run_id == FlowRun.id)
            .join(Flow, FlowRun.flow_id == Flow.id)
        )

        result = await session.execute(notification_details_stmt)
        return result.t.fetchall()

    @property
    def _get_scheduled_flow_runs_from_work_pool_template_path(self) -> str:
        return "postgres/get-runs-from-worker-queues.sql.jinja"

    @db_injector
    def _build_flow_run_graph_v2_query(self, db: PrefectDBInterface) -> sa.Select[FlowRunGraphV2Node]:
        param_flow_run_id = sa.bindparam("flow_run_id", type_=UUIDTypeDecorator)
        param_since = sa.bindparam("since", type_=Timestamp)
        param_max_nodes = sa.bindparam("max_nodes", type_=sa.Integer)

        Flow, FlowRun, TaskRun = db.Flow, db.FlowRun, db.TaskRun
        input = sa.func.jsonb_each(TaskRun.task_inputs).table_valued("key", "value", name="input")
        argument = (
            sa.func.jsonb_array_elements(input.c.value, type_=postgresql.JSONB())
            .table_valued(sa.column("value", postgresql.JSONB()))
            .render_derived(name="argument")
        )
        edges = (
            sa.select(
                sa.case((FlowRun.id.is_not(None), "flow-run"), else_="task-run").label("kind"),
                sa.func.coalesce(FlowRun.id, TaskRun.id).label("id"),
                sa.func.coalesce(Flow.name + " / " + FlowRun.name, TaskRun.name).label("label"),
                sa.func.coalesce(FlowRun.state_type, TaskRun.state_type).label("state_type"),
                sa.func.coalesce(
                    FlowRun.start_time,
                    FlowRun.expected_start_time,
                    TaskRun.start_time,
                    TaskRun.expected_start_time,
                ).label("start_time"),
                sa.func.coalesce(
                    FlowRun.end_time,
                    TaskRun.end_time,
                    sa.case(
                        (
                            TaskRun.state_type == StateType.COMPLETED,
                            TaskRun.expected_start_time,
                        ),
                        else_=sa.null(),
                    ),
                ).label("end_time"),
                sa.cast(argument.c.value["id"].astext, type_=UUIDTypeDecorator).label("parent"),
                (input.c.key == "__parents__").label("has_encapsulating_task"),
            )
            .join_from(TaskRun, input, onclause=sa.true(), isouter=True)
            .join(argument, onclause=sa.true(), isouter=True)
            .join(FlowRun, isouter=True, onclause=FlowRun.parent_task_run_id == TaskRun.id)
            .join(Flow, isouter=True, onclause=Flow.id == FlowRun.flow_id)
            .where(
                TaskRun.flow_run_id == param_flow_run_id,
                TaskRun.state_type != StateType.PENDING,
                sa.func.coalesce(
                    FlowRun.start_time,
                    FlowRun.expected_start_time,
                    TaskRun.start_time,
                    TaskRun.expected_start_time,
                ).is_not(None),
            )
            .order_by(sa.func.coalesce(FlowRun.id, TaskRun.id))
        ).cte("edges")
        children, parents = edges.alias("children"), edges.alias("parents")
        with_encapsulating = (
            sa.select(
                children.c.id,
                sa.func.array_agg(postgresql.aggregate_order_by(parents.c.id, parents.c.start_time)).label("encapsulating_ids"),
            )
            .join(parents, onclause=parents.c.id == children.c.parent)
            .where(children.c.has_encapsulating_task.is_(True))
            .group_by(children.c.id)
        ).cte("with_encapsulating")
        with_parents = (
            sa.select(
                children.c.id,
                sa.func.array_agg(postgresql.aggregate_order_by(parents.c.id, parents.c.start_time)).label("parent_ids"),
            )
            .join(parents, onclause=parents.c.id == children.c.parent)
            .where(children.c.has_encapsulating_task.is_distinct_from(True))
            .group_by(children.c.id)
            .cte("with_parents")
        )
        with_children = (
            sa.select(
                parents.c.id,
                sa.func.array_agg(postgresql.aggregate_order_by(children.c.id, children.c.start_time)).label("child_ids"),
            )
            .join(children, onclause=children.c.parent == parents.c.id)
            .where(children.c.has_encapsulating_task.is_distinct_from(True))
            .group_by(parents.c.id)
            .cte("with_children")
        )

        graph = (
            sa.select(
                edges.c.kind,
                edges.c.id,
                edges.c.label,
                edges.c.state_type,
                edges.c.start_time,
                edges.c.end_time,
                with_parents.c.parent_ids,
                with_children.c.child_ids,
                with_encapsulating.c.encapsulating_ids,
            )
            .distinct(edges.c.id)
            .join(with_parents, isouter=True, onclause=with_parents.c.id == edges.c.id)
            .join(with_children, isouter=True, onclause=with_children.c.id == edges.c.id)
            .join(with_encapsulating, isouter=True, onclause=with_encapsulating.c.id == edges.c.id)
            .cte("nodes")
        )
        query = (
            sa.select(
                graph.c.kind,
                graph.c.id,
                graph.c.label,
                graph.c.state_type,
                graph.c.start_time,
                graph.c.end_time,
                graph.c.parent_ids,
                graph.c.child_ids,
                graph.c.encapsulating_ids,
            )
            .where(sa.or_(graph.c.end_time.is_(None), graph.c.end_time >= param_since))
            .order_by(graph.c.start_time, graph.c.end_time)
            .limit(param_max_nodes)
        )
        return cast(sa.Select[FlowRunGraphV2Node], query)


class UUIDList(sa.TypeDecorator[List[UUID]]):
    """Map a JSON list of strings back to a list of UUIDs at the result loading stage"""

    impl: Union[TypeEngine[Any], type[TypeEngine[Any]]] = sa.JSON()

    def process_result_value(
        self, value: Optional[List[Union[str, UUID]]], dialect: sa.Dialect
    ) -> Optional[List[UUID]]:
        if value is None:
            return value
        return [v if isinstance(v, UUID) else UUID(v) for v in value]


class AioSqliteQueryComponents(BaseQueryComponents):
    # --- Sqlite-specific SqlAlchemy bindings

    def insert(self, obj: type[orm_models.Base]) -> sqlite.Insert:
        return sqlite.insert(obj)

    # --- Sqlite-specific JSON handling

    @property
    def uses_json_strings(self) -> bool:
        return True

    def cast_to_json(self, json_obj: sa.ColumnElement[T]) -> sa.ColumnElement[T]:
        return sa.func.json(json_obj)

    def build_json_object(
        self, *args: Union[str, sa.ColumnElement[Any]]
    ) -> sa.ColumnElement[Any]:
        return sa.func.json_object(*args)

    def json_arr_agg(self, json_array: sa.ColumnElement[Any]) -> sa.ColumnElement[Any]:
        return sa.func.json_group_array(json_array)

    def make_timestamp_intervals(
        self,
        start_time: DateTime,
        end_time: DateTime,
        interval: datetime.timedelta,
    ) -> sa.Select[Tuple[DateTime, DateTime]]:
        start = sa.bindparam("start_time", start_time, Timestamp)
        stop = sa.bindparam("end_time", end_time - interval, Timestamp)
        step = sa.bindparam("interval", interval, sa.Interval)
        one = sa.literal(1, literal_execute=True)

        base_case = sa.select(
            start.label("interval_start"),
            sa.func.date_add(start, step).label("interval_end"),
            one.label("counter"),
        ).cte(recursive=True)
        recursive_case = sa.select(
            base_case.c.interval_end,
            sa.func.date_add(base_case.c.interval_end, step),
            base_case.c.counter + one,
        ).where(
            base_case.c.interval_start < stop,
            base_case.c.counter < 500,
        )
        cte = base_case.union_all(recursive_case)
        return sa.select(cte.c.interval_start, cte.c.interval_end)

    @db_injector
    def set_state_id_on_inserted_flow_runs_statement(
        self,
        db: PrefectDBInterface,
        inserted_flow_run_ids: Sequence[UUID],
        insert_flow_run_states: Iterable[Dict[str, Any]],
    ) -> sa.Update:
        fr_model, frs_model = db.FlowRun, db.FlowRunState
        subquery = (
            sa.select(frs_model.id)
            .where(
                frs_model.flow_run_id == fr_model.id,
                frs_model.id.in_([r["id"] for r in insert_flow_run_states]),
            )
            .limit(1)
            .scalar_subquery()
        )
        stmt = (
            sa.update(fr_model)
            .where(fr_model.id.in_(inserted_flow_run_ids))
            .values(state_id=subquery)
            .execution_options(synchronize_session=False)
        )
        return stmt

    @db_injector
    async def get_flow_run_notifications_from_queue(
        self, db: PrefectDBInterface, session: AsyncSession, limit: int
    ) -> Sequence[FlowRunNotificationsFromQueue]:
        Flow, FlowRun = db.Flow, db.FlowRun
        FlowRunNotificationPolicy = db.FlowRunNotificationPolicy
        FlowRunNotificationQueue = db.FlowRunNotificationQueue
        FlowRunState = db.FlowRunState

        notification_details_stmt: sa.Select[FlowRunNotificationsFromQueue] = (
            sa.select(
                FlowRunNotificationQueue.id.label("queue_id"),
                FlowRunNotificationPolicy.id.label("flow_run_notification_policy_id"),
                FlowRunNotificationPolicy.message_template.label("flow_run_notification_policy_message_template"),
                FlowRunNotificationPolicy.block_document_id,
                Flow.id.label("flow_id"),
                Flow.name.label("flow_name"),
                FlowRun.id.label("flow_run_id"),
                FlowRun.name.label("flow_run_name"),
                FlowRun.parameters.label("flow_run_parameters"),
                FlowRunState.type.label("flow_run_state_type"),
                FlowRunState.name.label("flow_run_state_name"),
                FlowRunState.timestamp.label("flow_run_state_timestamp"),
                FlowRunState.message.label("flow_run_state_message"),
            )
            .select_from(FlowRunNotificationQueue)
            .join(
                FlowRunNotificationPolicy,
                FlowRunNotificationQueue.flow_run_notification_policy_id == FlowRunNotificationPolicy.id,
            )
            .join(
                FlowRunState,
                FlowRunNotificationQueue.flow_run_state_id == FlowRunState.id,
            )
            .join(FlowRun, FlowRunState.flow_run_id == FlowRun.id)
            .join(Flow, FlowRun.flow_id == Flow.id)
            .order_by(FlowRunNotificationQueue.updated)
            .limit(limit)
        )

        result = await session.execute(notification_details_stmt)
        notifications: List[FlowRunNotificationsFromQueue] = result.t.fetchall()

        delete_stmt = (
            sa.delete(FlowRunNotificationQueue)
            .where(FlowRunNotificationQueue.id.in_([n.queue_id for n in notifications]))
            .execution_options(synchronize_session="fetch")
        )

        await session.execute(delete_stmt)
        return notifications

    @db_injector
    def _get_scheduled_flow_runs_join(
        self,
        db: PrefectDBInterface,
        work_queue_query: sa.CTE,
        limit_per_queue: Optional[int],
        scheduled_before: Optional[DateTime],
    ) -> Tuple[sa.FromClause, sa.ColumnExpression[Any]]:
        FlowRun = db.FlowRun
        scheduled_before_clause = (
            FlowRun.next_scheduled_start_time <= scheduled_before
            if scheduled_before is not None
            else sa.true()
        )

        scheduled_flow_runs = (
            sa.select(
                (
                    sa.func.row_number()
                    .over(
                        partition_by=[FlowRun.work_queue_name],
                        order_by=FlowRun.next_scheduled_start_time,
                    )
                    .label("rank")
                ),
                FlowRun,
            )
            .where(FlowRun.state_type == StateType.SCHEDULED, scheduled_before_clause)
            .subquery("scheduled_flow_runs")
        )

        limit_val: int = 999999 if limit_per_queue is None else limit_per_queue

        join_criteria: sa.ColumnExpression[Any] = sa.and_(
            scheduled_flow_runs.c.work_queue_name == db.WorkQueue.name,
            scheduled_flow_runs.c.rank
            <= sa.func.min(sa.func.coalesce(work_queue_query.c.available_slots, limit_val), limit_val),
        )
        return scheduled_flow_runs, join_criteria

    @property
    def _get_scheduled_flow_runs_from_work_pool_template_path(self) -> str:
        return "sqlite/get-runs-from-worker-queues.sql.jinja"

    @db_injector
    def _build_flow_run_graph_v2_query(self, db: PrefectDBInterface) -> sa.Select[FlowRunGraphV2Node]:
        param_flow_run_id = sa.bindparam("flow_run_id", type_=UUIDTypeDecorator)
        param_since = sa.bindparam("since", type_=Timestamp)
        param_max_nodes = sa.bindparam("max_nodes", type_=sa.Integer)

        Flow, FlowRun, TaskRun = db.Flow, db.FlowRun, db.TaskRun
        input = sa.func.json_each(TaskRun.task_inputs).table_valued("key", "value", name="input")
        argument = sa.func.json_each(
            input.c.value, type_=postgresql.JSON()
        ).table_valued("key", sa.column("value", postgresql.JSON()), name="argument")
        edges = (
            sa.select(
                sa.case((FlowRun.id.is_not(None), "flow-run"), else_="task-run").label("kind"),
                sa.func.coalesce(FlowRun.id, TaskRun.id).label("id"),
                sa.func.coalesce(Flow.name + " / " + FlowRun.name, TaskRun.name).label("label"),
                sa.func.coalesce(FlowRun.state_type, TaskRun.state_type).label("state_type"),
                sa.func.coalesce(
                    FlowRun.start_time,
                    FlowRun.expected_start_time,
                    TaskRun.start_time,
                    TaskRun.expected_start_time,
                ).label("start_time"),
                sa.func.coalesce(
                    FlowRun.end_time,
                    TaskRun.end_time,
                    sa.case(
                        (
                            TaskRun.state_type == StateType.COMPLETED,
                            TaskRun.expected_start_time,
                        ),
                        else_=sa.null(),
                    ),
                ).label("end_time"),
                argument.c.value["id"].astext.label("parent"),
                (input.c.key == "__parents__").label("has_encapsulating_task"),
            )
            .join_from(TaskRun, input, onclause=sa.true(), isouter=True)
            .join(argument, onclause=sa.true(), isouter=True)
            .join(FlowRun, isouter=True, onclause=FlowRun.parent_task_run_id == TaskRun.id)
            .join(Flow, isouter=True, onclause=Flow.id == FlowRun.flow_id)
            .where(
                TaskRun.flow_run_id == param_flow_run_id,
                TaskRun.state_type != StateType.PENDING,
                sa.func.coalesce(
                    FlowRun.start_time,
                    FlowRun.expected_start_time,
                    TaskRun.start_time,
                    TaskRun.expected_start_time,
                ).is_not(None),
            )
            .order_by(sa.func.coalesce(FlowRun.id, TaskRun.id))
        ).cte("edges")
        children, parents = edges.alias("children"), edges.alias("parents")
        with_encapsulating = (
            sa.select(
                children.c.id,
                sa.func.json_group_array(parents.c.id).label("encapsulating_ids"),
            )
            .join(parents, onclause=parents.c.id == children.c.parent)
            .where(children.c.has_encapsulating_task.is_(True))
            .group_by(children.c.id)
        ).cte("with_encapsulating")
        with_parents = (
            sa.select(
                children.c.id,
                sa.func.json_group_array(parents.c.id).label("parent_ids"),
            )
            .join(parents, onclause=parents.c.id == children.c.parent)
            .where(children.c.has_encapsulating_task.is_distinct_from(True))
            .group_by(children.c.id)
            .cte("with_parents")
        )
        with_children = (
            sa.select(
                parents.c.id, sa.func.json_group_array(children.c.id).label("child_ids")
            )
            .join(children, onclause=children.c.parent == parents.c.id)
            .where(children.c.has_encapsulating_task.is_distinct_from(True))
            .group_by(parents.c.id)
            .cte("with_children")
        )

        graph = (
            sa.select(
                edges.c.kind,
                edges.c.id,
                edges.c.label,
                edges.c.state_type,
                edges.c.start_time,
                edges.c.end_time,
                with_parents.c.parent_ids,
                with_children.c.child_ids,
                with_encapsulating.c.encapsulating_ids,
            )
            .distinct()
            .join(with_parents, isouter=True, onclause=with_parents.c.id == edges.c.id)
            .join(with_children, isouter=True, onclause=with_children.c.id == edges.c.id)
            .join(with_encapsulating, isouter=True, onclause=with_encapsulating.c.id == edges.c.id)
            .cte("nodes")
        )

        query = (
            sa.select(
                graph.c.kind,
                graph.c.id,
                graph.c.label,
                graph.c.state_type,
                graph.c.start_time,
                graph.c.end_time,
                sa.type_coerce(graph.c.parent_ids, UUIDList),
                sa.type_coerce(graph.c.child_ids, UUIDList),
                sa.type_coerce(graph.c.encapsulating_ids, UUIDList),
            )
            .where(sa.or_(graph.c.end_time.is_(None), graph.c.end_time >= param_since))
            .order_by(graph.c.start_time, graph.c.end_time)
            .limit(param_max_nodes)
        )
        return cast(sa.Select[FlowRunGraphV2Node], query)