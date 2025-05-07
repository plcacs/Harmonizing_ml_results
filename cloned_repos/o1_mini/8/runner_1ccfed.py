from __future__ import annotations
import asyncio
import datetime
import inspect
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Coroutine,
    Dict,
    Iterable,
    List,
    Optional,
    TypedDict,
    Union,
)
from uuid import UUID, uuid4
import anyio
import anyio.abc
from cachetools import LRUCache
from typing_extensions import Self
from prefect._internal.concurrency.api import create_call, from_async, from_sync
from prefect.client.orchestration import PrefectClient, get_client
from prefect.client.schemas.filters import (
    FlowRunFilter,
    FlowRunFilterId,
    FlowRunFilterState,
    FlowRunFilterStateName,
    FlowRunFilterStateType,
)
from prefect.client.schemas.objects import (
    ConcurrencyLimitConfig,
    State,
    StateType,
    Flow as APIFlow,
)
from prefect.events import DeploymentTriggerTypes, TriggerTypes
from prefect.events.related import tags_as_related_resources
from prefect.events.schemas.events import RelatedResource
from prefect.events.utilities import emit_event
from prefect.exceptions import Abort, ObjectNotFound
from prefect.flows import Flow, FlowStateHook, load_flow_from_flow_run
from prefect.logging.loggers import PrefectLogAdapter, flow_run_logger, get_logger
from prefect.runner.storage import RunnerStorage
from prefect.schedules import Schedule
from prefect.settings import (
    PREFECT_API_URL,
    PREFECT_RUNNER_SERVER_ENABLE,
    get_current_settings,
)
from prefect.states import Crashed, Pending, exception_to_failed_state
from prefect.types._datetime import DateTime
from prefect.types.entrypoint import EntrypointType
from prefect.utilities.asyncutils import asyncnullcontext, is_async_fn, sync_compatible
from prefect.utilities.engine import propose_state
from prefect.utilities.processutils import get_sys_executable, run_process
from prefect.utilities.services import (
    critical_service_loop,
    start_client_metrics_server,
)
from prefect.utilities.slugify import slugify

if TYPE_CHECKING:
    import concurrent.futures
    from prefect.client.schemas.objects import FlowRun
    from prefect.client.schemas.responses import DeploymentResponse
    from prefect.client.types.flexible_schedule_list import FlexibleScheduleList
    from prefect.deployments.runner import RunnerDeployment

__all__ = ["Runner"]


class ProcessMapEntry(TypedDict):
    pid: int
    flow_run: FlowRun


class Runner:
    def __init__(
        self,
        name: Optional[str] = None,
        query_seconds: Optional[int] = None,
        prefetch_seconds: int = 10,
        heartbeat_seconds: Optional[int] = None,
        limit: Optional[int] = None,
        pause_on_shutdown: bool = True,
        webserver: bool = False,
    ) -> None:
        """
        Responsible for managing the execution of remotely initiated flow runs.

        Args:
            name: The name of the runner. If not provided, a random one
                will be generated. If provided, it cannot contain '/' or '%'.
            query_seconds: The number of seconds to wait between querying for
                scheduled flow runs; defaults to `PREFECT_RUNNER_POLL_FREQUENCY`
            prefetch_seconds: The number of seconds to prefetch flow runs for.
            heartbeat_seconds: The number of seconds to wait between emitting
                flow run heartbeats. The runner will not emit heartbeats if the value is None.
                Defaults to `PREFECT_RUNNER_HEARTBEAT_FREQUENCY`.
            limit: The maximum number of flow runs this runner should be running at
            pause_on_shutdown: A boolean for whether or not to automatically pause
                deployment schedules on shutdown; defaults to `True`
            webserver: a boolean flag for whether to start a webserver for this runner
        """
        settings = get_current_settings()
        if name and ("/" in name or "%" in name):
            raise ValueError("Runner name cannot contain '/' or '%'")
        self.name: str = Path(name).stem if name is not None else f"runner-{uuid4()}"
        self._logger: logging.Logger = get_logger("runner")
        self.started: bool = False
        self.stopping: bool = False
        self.pause_on_shutdown: bool = pause_on_shutdown
        self.limit: int = limit or settings.runner.process_limit
        self.webserver: bool = webserver
        self.query_seconds: int = query_seconds or settings.runner.poll_frequency
        self._prefetch_seconds: int = prefetch_seconds
        self.heartbeat_seconds: Optional[int] = (
            heartbeat_seconds or settings.runner.heartbeat_frequency
        )
        if self.heartbeat_seconds is not None and self.heartbeat_seconds < 30:
            raise ValueError("Heartbeat must be 30 seconds or greater.")
        self._limiter: Optional[anyio.CapacityLimiter] = None
        self._client: PrefectClient = get_client()
        self._submitting_flow_run_ids: set[UUID] = set()
        self._cancelling_flow_run_ids: set[UUID] = set()
        self._scheduled_task_scopes: set[anyio.CancelScope] = set()
        self._deployment_ids: set[UUID] = set()
        self._flow_run_process_map: Dict[UUID, ProcessMapEntry] = {}
        self._tmp_dir: Path = Path(tempfile.gettempdir()) / "runner_storage" / str(uuid4())
        self._storage_objs: List[RunnerStorage] = []
        self._deployment_storage_map: Dict[UUID, RunnerStorage] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._deployment_cache: LRUCache[UUID, DeploymentResponse] = LRUCache(maxsize=100)
        self._flow_cache: LRUCache[UUID, APIFlow] = LRUCache(maxsize=100)

    @sync_compatible
    async def add_deployment(
        self, deployment: RunnerDeployment
    ) -> UUID:
        """
        Registers the deployment with the Prefect API and will monitor for work once
        the runner is started.

        Args:
            deployment: A deployment for the runner to register.
        """
        apply_coro: Coroutine[Any, Any, UUID] = deployment.apply()
        if TYPE_CHECKING:
            assert inspect.isawaitable(apply_coro)
        deployment_id: UUID = await apply_coro
        storage: Optional[RunnerStorage] = deployment.storage
        if storage is not None:
            add_storage_coro: Coroutine[Any, Any, RunnerStorage] = self._add_storage(storage)
            if TYPE_CHECKING:
                assert inspect.isawaitable(add_storage_coro)
            storage = await add_storage_coro
            self._deployment_storage_map[deployment_id] = storage
        self._deployment_ids.add(deployment_id)
        return deployment_id

    @sync_compatible
    async def add_flow(
        self,
        flow: Flow,
        name: Optional[str] = None,
        interval: Optional[Union[int, datetime.timedelta]] = None,
        cron: Optional[str] = None,
        rrule: Optional[str] = None,
        paused: Optional[bool] = None,
        schedule: Optional[Schedule] = None,
        schedules: Optional[FlexibleScheduleList] = None,
        concurrency_limit: Optional[int] = None,
        parameters: Optional[Dict[str, Any]] = None,
        triggers: Optional[List[TriggerTypes]] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        version: Optional[str] = None,
        enforce_parameter_schema: bool = True,
        entrypoint_type: EntrypointType = EntrypointType.FILE_PATH,
    ) -> UUID:
        """
        Provides a flow to the runner to be run based on the provided configuration.

        Will create a deployment for the provided flow and register the deployment
        with the runner.

        Args:
            flow: A flow for the runner to run.
            name: The name to give the created deployment. Will default to the name
                of the runner.
            interval: An interval on which to execute the current flow. Accepts either a number
                or a timedelta object. If a number is given, it will be interpreted as seconds.
            cron: A cron schedule of when to execute runs of this flow.
            rrule: An rrule schedule of when to execute runs of this flow.
            paused: Whether or not to set the created deployment as paused.
            schedule: A schedule object defining when to execute runs of this deployment.
                Used to provide additional scheduling options like `timezone` or `parameters`.
            schedules: A list of schedule objects defining when to execute runs of this flow.
                Used to define multiple schedules or additional scheduling options like `timezone`.
            concurrency_limit: The maximum number of concurrent runs of this flow to allow.
            triggers: A list of triggers that should kick of a run of this flow.
            parameters: A dictionary of default parameter values to pass to runs of this flow.
            description: A description for the created deployment. Defaults to the flow's
                description if not provided.
            tags: A list of tags to associate with the created deployment for organizational
                purposes.
            version: A version for the created deployment. Defaults to the flow's version.
            entrypoint_type: Type of entrypoint to use for the deployment. When using a module path
                entrypoint, ensure that the module will be importable in the execution environment.
        """
        api: Optional[str] = PREFECT_API_URL.value()
        if any([interval, cron, rrule, schedule, schedules]) and not api:
            self._logger.warning(
                "Cannot schedule flows on an ephemeral server; run `prefect server start` to start the scheduler."
            )
        deployment_name: str = self.name if name is None else name
        to_deployment_coro: Coroutine[Any, Any, RunnerDeployment] = flow.to_deployment(
            name=deployment_name,
            interval=interval,
            cron=cron,
            rrule=rrule,
            schedule=schedule,
            schedules=schedules,
            paused=paused,
            triggers=triggers,
            parameters=parameters,
            description=description,
            tags=tags,
            version=version,
            enforce_parameter_schema=enforce_parameter_schema,
            entrypoint_type=entrypoint_type,
            concurrency_limit=concurrency_limit,
        )
        if TYPE_CHECKING:
            assert inspect.isawaitable(to_deployment_coro)
        deployment: RunnerDeployment = await to_deployment_coro
        add_deployment_coro: Coroutine[Any, Any, UUID] = self.add_deployment(deployment)
        if TYPE_CHECKING:
            assert inspect.isawaitable(add_deployment_coro)
        return await add_deployment_coro

    @sync_compatible
    async def _add_storage(
        self, storage: RunnerStorage
    ) -> RunnerStorage:
        """
        Adds a storage object to the runner. The storage object will be used to pull
        code to the runner's working directory before the runner starts.

        Args:
            storage: The storage object to add to the runner.
        Returns:
            The updated storage object that was added to the runner.
        """
        if storage not in self._storage_objs:
            storage_copy: RunnerStorage = deepcopy(storage)
            storage_copy.set_base_path(self._tmp_dir)
            self._logger.debug(
                f"Adding storage {storage_copy!r} to runner at {str(storage_copy.destination)!r}"
            )
            self._storage_objs.append(storage_copy)
            return storage_copy
        else:
            return next(
                (s for s in self._storage_objs if s == storage), storage  # type: ignore
            )

    def handle_sigterm(self, *args: Any, **kwargs: Any) -> None:
        """
        Gracefully shuts down the runner when a SIGTERM is received.
        """
        self._logger.info("SIGTERM received, initiating graceful shutdown...")
        from_sync.call_in_loop_thread(create_call(self.stop))
        sys.exit(0)

    @sync_compatible
    async def start(
        self, run_once: bool = False, webserver: Optional[bool] = None
    ) -> None:
        """
        Starts a runner.

        The runner will begin monitoring for and executing any scheduled work for all added flows.

        Args:
            run_once: If True, the runner will through one query loop and then exit.
            webserver: a boolean for whether to start a webserver for this runner. If provided,
                overrides the default on the runner

        Examples:
            Initialize a Runner, add two flows, and serve them by starting the Runner:

            