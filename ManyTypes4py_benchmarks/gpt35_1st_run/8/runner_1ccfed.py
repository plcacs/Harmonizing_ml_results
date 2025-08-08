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
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Dict, Iterable, List, Optional, TypedDict, Union
from uuid import UUID, uuid4
import anyio
import anyio.abc
from cachetools import LRUCache
from typing_extensions import Self
from prefect._internal.concurrency.api import create_call, from_async, from_sync
from prefect.client.orchestration import PrefectClient, get_client
from prefect.client.schemas.filters import FlowRunFilter, FlowRunFilterId, FlowRunFilterState, FlowRunFilterStateName, FlowRunFilterStateType
from prefect.client.schemas.objects import ConcurrencyLimitConfig, State, StateType
from prefect.client.schemas.objects import Flow as APIFlow
from prefect.events import DeploymentTriggerTypes, TriggerTypes
from prefect.events.related import tags_as_related_resources
from prefect.events.schemas.events import RelatedResource
from prefect.events.utilities import emit_event
from prefect.exceptions import Abort, ObjectNotFound
from prefect.flows import Flow, FlowStateHook, load_flow_from_flow_run
from prefect.logging.loggers import PrefectLogAdapter, flow_run_logger, get_logger
from prefect.runner.storage import RunnerStorage
from prefect.schedules import Schedule
from prefect.settings import PREFECT_API_URL, PREFECT_RUNNER_SERVER_ENABLE, get_current_settings
from prefect.states import Crashed, Pending, exception_to_failed_state
from prefect.types._datetime import DateTime
from prefect.types.entrypoint import EntrypointType
from prefect.utilities.asyncutils import asyncnullcontext, is_async_fn, sync_compatible
from prefect.utilities.engine import propose_state
from prefect.utilities.processutils import get_sys_executable, run_process
from prefect.utilities.services import critical_service_loop, start_client_metrics_server
from prefect.utilities.slugify import slugify
if TYPE_CHECKING:
    import concurrent.futures
    from prefect.client.schemas.objects import FlowRun
    from prefect.client.schemas.responses import DeploymentResponse
    from prefect.client.types.flexible_schedule_list import FlexibleScheduleList
    from prefect.deployments.runner import RunnerDeployment
__all__ = ['Runner']

class ProcessMapEntry(TypedDict):
    pass

class Runner:

    def __init__(self, name: Optional[str] = None, query_seconds: Optional[int] = None, prefetch_seconds: int = 10, heartbeat_seconds: Optional[int] = None, limit: Optional[int] = None, pause_on_shutdown: bool = True, webserver: bool = False):
        self.name: str = Path(name).stem if name is not None else f'runner-{uuid4()}'
        self._logger: logging.Logger = get_logger('runner')
        self.started: bool = False
        self.stopping: bool = False
        self.pause_on_shutdown: bool = pause_on_shutdown
        self.limit: int = limit or get_current_settings().runner.process_limit
        self.webserver: bool = webserver
        self.query_seconds: int = query_seconds or get_current_settings().runner.poll_frequency
        self._prefetch_seconds: int = prefetch_seconds
        self.heartbeat_seconds: Optional[int] = heartbeat_seconds or get_current_settings().runner.heartbeat_frequency
        if self.heartbeat_seconds is not None and self.heartbeat_seconds < 30:
            raise ValueError('Heartbeat must be 30 seconds or greater.')
        self._limiter: Optional[anyio.CapacityLimiter] = None
        self._client: PrefectClient = get_client()
        self._submitting_flow_run_ids: set = set()
        self._cancelling_flow_run_ids: set = set()
        self._scheduled_task_scopes: set = set()
        self._deployment_ids: set = set()
        self._flow_run_process_map: dict = {}
        self._tmp_dir: Path = Path(tempfile.gettempdir()) / 'runner_storage' / str(uuid4())
        self._storage_objs: List[RunnerStorage] = []
        self._deployment_storage_map: Dict[UUID, RunnerStorage] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._deployment_cache: LRUCache = LRUCache(maxsize=100)
        self._flow_cache: LRUCache = LRUCache(maxsize=100)

    async def add_deployment(self, deployment: RunnerDeployment) -> UUID:
        ...

    async def add_flow(self, flow: Flow, name: Optional[str] = None, interval: Optional[Union[int, datetime.timedelta]] = None, cron: Optional[str] = None, rrule: Optional[str] = None, paused: Optional[bool] = None, schedule: Optional[Schedule] = None, schedules: Optional[List[Schedule]] = None, concurrency_limit: Optional[int] = None, parameters: Optional[Dict[str, Any]] = None, triggers: Optional[List[TriggerTypes]] = None, description: Optional[str] = None, tags: Optional[List[str]] = None, version: Optional[str] = None, enforce_parameter_schema: bool = True, entrypoint_type: EntrypointType = EntrypointType.FILE_PATH) -> UUID:
        ...

    async def _add_storage(self, storage: RunnerStorage) -> RunnerStorage:
        ...

    def handle_sigterm(self, *args, **kwargs) -> None:
        ...

    async def start(self, run_once: bool = False, webserver: Optional[bool] = None) -> None:
        ...

    def execute_in_background(self, func: Callable, *args, **kwargs) -> Any:
        ...

    async def cancel_all(self) -> None:
        ...

    async def stop(self) -> None:
        ...

    async def _get_and_submit_flow_runs(self) -> None:
        ...

    async def _check_for_cancelled_flow_runs(self, should_stop: Callable[[], bool] = lambda: False, on_stop: Callable[[], None] = lambda: None) -> None:
        ...

    async def _cancel_run(self, flow_run: FlowRun, state_msg: Optional[str] = None) -> None:
        ...

    async def _get_flow_and_deployment(self, flow_run: FlowRun) -> Tuple[APIFlow, Optional[RunnerDeployment]]:
        ...

    async def _emit_flow_run_heartbeats(self) -> None:
        ...

    async def _emit_flow_run_heartbeat(self, flow_run: FlowRun) -> None:
        ...

    def _event_resource(self) -> Dict[str, str]:
        ...

    def _emit_flow_run_cancelled_event(self, flow_run: FlowRun, flow: APIFlow, deployment: Optional[RunnerDeployment]) -> None:
        ...

    async def _get_scheduled_flow_runs(self) -> List[FlowRun]:
        ...

    def has_slots_available(self) -> bool:
        ...

    def _acquire_limit_slot(self, flow_run_id: UUID) -> bool:
        ...

    def _release_limit_slot(self, flow_run_id: UUID) -> None:
        ...

    async def _submit_scheduled_flow_runs(self, flow_run_response: List[FlowRun], entrypoints: Optional[List[EntrypointType]] = None) -> List[FlowRun]:
        ...

    async def _submit_run(self, flow_run: FlowRun, entrypoint: Optional[EntrypointType] = None) -> None:
        ...

    async def _submit_run_and_capture_errors(self, flow_run: FlowRun, task_status: anyio.abc.TaskStatus, entrypoint: Optional[EntrypointType] = None) -> Any:
        ...

    async def _propose_pending_state(self, flow_run: FlowRun) -> bool:
        ...

    async def _propose_failed_state(self, flow_run: FlowRun, exc: Exception) -> None:
        ...

    async def _propose_crashed_state(self, flow_run: FlowRun, message: str) -> None:
        ...

    async def _mark_flow_run_as_cancelled(self, flow_run: FlowRun, state_updates: Optional[Dict[str, Any]] = None) -> None:
        ...

    async def _schedule_task(self, __in_seconds: int, fn: Callable, *args, **kwargs) -> None:
        ...

    async def _run_on_cancellation_hooks(self, flow_run: FlowRun, state: State) -> None:
        ...

    async def _run_on_crashed_hooks(self, flow_run: FlowRun, state: State) -> None:
        ...

    async def __aenter__(self) -> Runner:
        ...

    async def __aexit__(self, *exc_info) -> None:
        ...

    def __repr__(self) -> str:
        ...
