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
    pid: Optional[int]
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
        webserver: bool = False
    ) -> None:
        settings = get_current_settings()
        if name and ('/' in name or '%' in name):
            raise ValueError("Runner name cannot contain '/' or '%'")
        self.name = Path(name).stem if name is not None else f'runner-{uuid4()}'
        self._logger = get_logger('runner')
        self.started = False
        self.stopping = False
        self.pause_on_shutdown = pause_on_shutdown
        self.limit = limit or settings.runner.process_limit
        self.webserver = webserver
        self.query_seconds = query_seconds or settings.runner.poll_frequency
        self._prefetch_seconds = prefetch_seconds
        self.heartbeat_seconds = heartbeat_seconds or settings.runner.heartbeat_frequency
        if self.heartbeat_seconds is not None and self.heartbeat_seconds < 30:
            raise ValueError('Heartbeat must be 30 seconds or greater.')
        self._limiter: Optional[anyio.CapacityLimiter] = None
        self._client: PrefectClient = get_client()
        self._submitting_flow_run_ids: set[UUID] = set()
        self._cancelling_flow_run_ids: set[UUID] = set()
        self._scheduled_task_scopes: set[anyio.CancelScope] = set()
        self._deployment_ids: set[UUID] = set()
        self._flow_run_process_map: Dict[UUID, ProcessMapEntry] = dict()
        self._tmp_dir = Path(tempfile.gettempdir()) / 'runner_storage' / str(uuid4())
        self._storage_objs: List[RunnerStorage] = []
        self._deployment_storage_map: Dict[UUID, RunnerStorage] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._deployment_cache: LRUCache[UUID, DeploymentResponse] = LRUCache(maxsize=100)
        self._flow_cache: LRUCache[UUID, APIFlow] = LRUCache(maxsize=100)

    @sync_compatible
    async def add_deployment(self, deployment: RunnerDeployment) -> UUID:
        apply_coro = deployment.apply()
        if TYPE_CHECKING:
            assert inspect.isawaitable(apply_coro)
        deployment_id = await apply_coro
        storage = deployment.storage
        if storage is not None:
            add_storage_coro = self._add_storage(storage)
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
        triggers: Optional[List[Union[TriggerTypes, DeploymentTriggerTypes]]] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        version: Optional[str] = None,
        enforce_parameter_schema: bool = True,
        entrypoint_type: EntrypointType = EntrypointType.FILE_PATH
    ) -> UUID:
        api = PREFECT_API_URL.value()
        if any([interval, cron, rrule, schedule, schedules]) and (not api):
            self._logger.warning('Cannot schedule flows on an ephemeral server; run `prefect server start` to start the scheduler.')
        name = self.name if name is None else name
        to_deployment_coro = flow.to_deployment(
            name=name,
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
            concurrency_limit=concurrency_limit
        )
        if TYPE_CHECKING:
            assert inspect.isawaitable(to_deployment_coro)
        deployment = await to_deployment_coro
        add_deployment_coro = self.add_deployment(deployment)
        if TYPE_CHECKING:
            assert inspect.isawaitable(add_deployment_coro)
        return await add_deployment_coro

    @sync_compatible
    async def _add_storage(self, storage: RunnerStorage) -> RunnerStorage:
        if storage not in self._storage_objs:
            storage_copy = deepcopy(storage)
            storage_copy.set_base_path(self._tmp_dir)
            self._logger.debug(f'Adding storage {storage_copy!r} to runner at {str(storage_copy.destination)!r}')
            self._storage_objs.append(storage_copy)
            return storage_copy
        else:
            return next((s for s in self._storage_objs if s == storage))

    def handle_sigterm(self, *args: Any, **kwargs: Any) -> None:
        self._logger.info('SIGTERM received, initiating graceful shutdown...')
        from_sync.call_in_loop_thread(create_call(self.stop))
        sys.exit(0)

    async def start(self, run_once: bool = False, webserver: Optional[bool] = None) -> None:
        from prefect.runner.server import start_webserver
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, self.handle_sigterm)
        webserver = webserver if webserver is not None else self.webserver
        if webserver or PREFECT_RUNNER_SERVER_ENABLE.value():
            server_thread = threading.Thread(name='runner-server-thread', target=partial(start_webserver, runner=self), daemon=True)
            server_thread.start()
        start_client_metrics_server()
        async with self as runner:
            async with self._loops_task_group as loops_task_group:
                for storage in self._storage_objs:
                    if storage.pull_interval:
                        loops_task_group.start_soon(partial(critical_service_loop, workload=storage.pull_code, interval=storage.pull_interval, run_once=run_once, jitter_range=0.3))
                    else:
                        loops_task_group.start_soon(storage.pull_code)
                loops_task_group.start_soon(partial(critical_service_loop, workload=runner._get_and_submit_flow_runs, interval=self.query_seconds, run_once=run_once, jitter_range=0.3))
                loops_task_group.start_soon(partial(critical_service_loop, workload=runner._check_for_cancelled_flow_runs, interval=self.query_seconds * 2, run_once=run_once, jitter_range=0.3))
                if self.heartbeat_seconds is not None:
                    loops_task_group.start_soon(partial(critical_service_loop, workload=runner._emit_flow_run_heartbeats, interval=self.heartbeat_seconds, run_once=run_once, jitter_range=0.3))

    def execute_in_background(self, func: Callable[..., Coroutine[Any, Any, Any]], *args: Any, **kwargs: Any) -> concurrent.futures.Future:
        if TYPE_CHECKING:
            assert self._loop is not None
        return asyncio.run_coroutine_threadsafe(func(*args, **kwargs), self._loop)

    async def cancel_all(self) -> None:
        runs_to_cancel: List[FlowRun] = []
        for info in self._flow_run_process_map.values():
            runs_to_cancel.append(info['flow_run'])
        if runs_to_cancel:
            for run in runs_to_cancel:
                try:
                    await self._cancel_run(run, state_msg='Runner is shutting down.')
                except Exception:
                    self._logger.exception(f'Exception encountered while cancelling {run.id}', exc_info=True)

    @sync_compatible
    async def stop(self) -> None:
        if not self.started:
            raise RuntimeError('Runner has not yet started. Please start the runner by calling .start()')
        self.started = False
        self.stopping = True
        await self.cancel_all()
        try:
            self._loops_task_group.cancel_scope.cancel()
        except Exception:
            self._logger.exception('Exception encountered while shutting down', exc_info=True)

    async def execute_flow_run(self, flow_run_id: UUID, entrypoint: Optional[str] = None) -> None:
        self.pause_on_shutdown = False
        context = self if not self.started else asyncnullcontext()
        async with context:
            if not self._acquire_limit_slot(flow_run_id):
                return
            async with anyio.create_task_group() as tg:
                with anyio.CancelScope():
                    self._submitting_flow_run_ids.add(flow_run_id)
                    flow_run = await self._client.read_flow_run(flow_run_id)
                    pid = await self._runs_task_group.start(partial(self._submit_run_and_capture_errors, flow_run=flow_run, entrypoint=entrypoint))
                    self._flow_run_process_map[flow_run.id] = ProcessMapEntry(pid=pid, flow_run=flow_run)
                    workload = partial(self._check_for_cancelled_flow_runs, should_stop=lambda: not self._flow_run_process_map, on_stop=tg.cancel_scope.cancel)
                    tg.start_soon(partial(critical_service_loop, workload=workload, interval=self.query_seconds, jitter_range=0.3))
                    if self.heartbeat_seconds is not None:
                        tg.start_soon(partial(critical_service_loop, workload=self._emit_flow_run_heartbeats, interval=self.heartbeat_seconds, jitter_range=0.3))

    def _get_flow_run_logger(self, flow_run: FlowRun) -> PrefectLogAdapter:
        return flow_run_logger(flow_run=flow_run).getChild('runner', extra={'runner_name': self.name})

    async def _run_process(self, flow_run: FlowRun, task_status: Optional[anyio.abc.TaskStatus] = None, entrypoint: Optional[str] = None) -> int:
        command = [get_sys_executable(), '-m', 'prefect.engine']
        flow_run_logger = self._get_flow_run_logger(flow_run)
        kwargs: Dict[str, Any] = {}
        if sys.platform == 'win32':
            kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
        flow_run_logger.info('Opening process...')
        env = get_current_settings().to_environment_variables(exclude_unset=True)
        env.update({**{'PREFECT__FLOW_RUN_ID': str(flow_run.id), 'PREFECT__STORAGE_BASE_PATH': str(self._tmp_dir), 'PREFECT__ENABLE_CANCELLATION_AND_CRASHED_HOOKS': 'false'}, **({'PREFECT__FLOW_ENTRYPOINT': entrypoint} if entrypoint else {})})
        env.update(**os.environ)
        storage = self._deployment_storage_map.get(flow_run.deployment_id) if flow_run.deployment_id else None
        if storage and storage.pull_interval:
            last_adhoc_pull = getattr(storage, 'last_adhoc_pull', None)
            if last_adhoc_pull is None or last_adhoc_pull < datetime.datetime.now() - datetime.timedelta(seconds=storage.pull_interval):
                self._logger.debug('Performing adhoc pull of code for flow run %s with storage %r', flow_run.id, storage)
                await storage.pull_code()
                setattr(storage, 'last_adhoc_pull', datetime.datetime.now())
        process = await run_process(command=command, stream_output=True, task_status=task_status, task_status_handler=None, env=env, cwd=storage.destination if storage else None, **kwargs)
        if process.returncode is None:
            raise RuntimeError('Process exited with None return code')
        if process.returncode:
            help_message = None
            level = logging.ERROR
            if process.returncode == -9:
                level = logging.INFO
                help_message = 'This indicates that the process exited due to a SIGKILL signal. Typically, this is either caused by manual cancellation or high memory usage causing the operating system to terminate the process.'
            if process.returncode == -15:
                level = logging.INFO
                help_message = 'This indicates that the process exited due to a SIGTERM signal. Typically, this is caused by manual cancellation.'
            elif process.returncode == 247:
                help_message = 'This indicates that the process was terminated due to high memory usage.'
            elif sys.platform == 'win32' and process.returncode == STATUS_CONTROL_C_EXIT:
                level = logging.INFO
                help_message = 'Process was terminated due to a Ctrl+C or Ctrl+Break signal. Typically, this is caused by manual cancellation.'
            flow_run_logger.log(level, f'Process for flow run {flow_run.name!r} exited with status code: {process.returncode}' + (f'; {help_message}' if help_message else ''))
        else:
            flow_run_logger.info(f'Process for flow run {flow_run.name!r} exited cleanly.')
        return process.returncode

    async def _kill_process(self, pid: int, grace_seconds: int = 30) -> None:
        if sys.platform == 'win32':
            try:
                os.kill(pid, signal.CTRL_BREAK_EVENT)
            except (ProcessLookupError, WindowsError):
                raise RuntimeError(f'Unable to kill process {pid!r}: The process was not found.')
        else:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                raise RuntimeError(f'Unable to kill process {pid!r}: The process was not found.')
            check_interval = max(grace_seconds / 10, 1)
            with anyio.move_on_after(grace_seconds):
                while True:
                    await anyio.sleep(check_interval)
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        return
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                return

    async def _pause_schedules(self) -> None:
        self._logger.info('Pausing all deployments...')
        for deployment_id in self._deployment_ids:
            await self._client.set_deployment_paused_state(deployment_id, True)
            self._logger.debug(f"Paused deployment '{deployment_id}'")
        self._logger.info('All deployments have been paused!')

    async def _get_and_submit_flow_runs(self) -> None:
        if self.stopping:
            return
        runs_response = await self._get_scheduled_flow_runs()
        self.last_polled = DateTime.now('UTC')
        return await self._submit_scheduled_flow_runs(flow_run_response=runs_response)

    async def _check_for_cancelled_flow_runs(self, should_stop: Callable[[], bool] = lambda: False, on_stop: Callable[[], None] = lambda: None) -> List[FlowRun]:
        if self.stopping:
            return []
        if not self.started:
            raise RuntimeError('Runner is not set up. Please make sure you are running this runner as an async context manager.')
        if should_stop():
            self._logger.debug('Runner has no active flow runs or deployments. Sending message to loop service that no further cancellation checks are needed.')
            on_stop()
        self._logger.debug('Checking for cancelled flow runs...')
        named_cancelling_flow_runs = await self._client.read_flow_runs(flow_run_filter=FlowRunFilter(state=FlowRunFilterState(type=FlowRunFilterStateType(any_=[StateType.CANCELLED]), name=FlowRunFilterStateName(any_=['Cancelling'])), id=FlowRunFilterId(any_=list(self._flow_run_process_map.keys() - self._cancelling_flow_run_ids))))
        typed_cancelling_flow_runs = await self._client.read_flow_runs(flow_run_filter=FlowRunFilter(state=FlowRunFilterState(type=FlowRunFilterStateType(any_=[StateType.CANCELLING])), id=FlowRunFilterId(any_=list(self._flow_run_process_map.keys() - self._cancelling_flow_run_ids))))
        cancelling_flow_runs = named_cancelling_flow_runs + typed_cancelling_flow_runs
        if cancelling_flow_runs:
            self._logger.info(f'Found {len(cancelling_flow_runs)} flow runs awaiting cancellation.')
        for flow_run in cancelling_flow_runs:
            self._cancelling_flow_run_ids.add(flow_run.id)
            self._runs_task_group.start_soon(self._cancel_run, flow_run)
        return cancelling_flow_runs

    async def _cancel_run(self, flow_run: FlowRun, state_msg: Optional[str] = None) -> None:
        run_logger = self._get_flow_run_logger(flow_run)
        process_map_entry = self._flow_run_process_map.get(flow_run.id)
        pid = process_map_entry.get('pid') if process_map_entry else None
        if not pid:
            if flow_run.state:
                await self._run_on_cancellation_hooks(flow_run, flow_run.state)
            await self._mark_flow_run_as_cancelled(flow_run, state_updates={'message': 'Could not find process ID for flow run and cancellation cannot be guaranteed.'})
            return
        try:
            await self._kill_process(pid)
        except RuntimeError as exc:
            self._logger.warning(f'{exc} Marking flow run as cancelled.')
            if flow_run.state:
                await self._run_on_cancellation_hooks(flow_run, flow_run.state)
            await self._mark_flow_run_as_cancelled(flow_run)
        except Exception:
            run_logger.exception(f"Encountered exception while killing process for flow run '{flow_run.id}'. Flow run may not be cancelled.")
            self._cancelling_flow_run_ids.remove(flow_run.id)
        else:
            if flow_run.state:
                await self._run_on_cancellation_hooks(flow_run, flow_run.state)
            await self._mark_flow_run_as_cancelled(flow_run, state_updates={'message': state_msg or 'Flow run was cancelled successfully.'})
            flow, deployment = await self._get_flow_and_deployment(flow_run)
            self._emit_flow_run_cancelled_event(flow_run=flow_run, flow=flow, deployment=deployment)
            run_logger.info(f"Cancelled flow run '{flow_run.name}'!")

    async def _get_flow_and_deployment(self, flow_run: FlowRun) -> tuple[Optional[APIFlow], Optional[DeploymentResponse]]:
        deployment = self._deployment_cache.get(flow_run.deployment_id) if flow_run.deployment_id else None
        flow = self._flow_cache.get(flow_run.flow_id)
        if not deployment and flow_run.deployment_id is not None:
            try:
                deployment = await self._client.read_deployment(flow_run.deployment_id)
                self._deployment_cache[flow_run.deployment_id] = deployment
            except ObjectNotFound:
                deployment = None
        if not flow:
            try:
                flow = await self._client.read_flow(flow_run.flow_id)
                self._flow_cache[flow_run.flow_id] = flow
            except ObjectNotFound:
                flow = None
        return (flow, deployment)

    async def _emit_flow_run_heartbeats(self) -> None:
        coros = []
        for entry in self._flow_run_process_map.values():
            coros.append(self._emit_flow_run_heartbeat(entry['flow_run']))
        await asyncio.gather(*coros)

    async def _emit_flow_run_heartbeat(self, flow_run: FlowRun) -> None:
        from prefect import __version__
        related: List[RelatedResource] = []
        tags: List[str] = []
        flow, deployment = await self._get_flow_and_deployment(flow_run)
        if deployment:
            related.append(RelatedResource({'prefect.resource.id': f'prefect.deployment.{deployment.id}', 'prefect.resource.role': 'deployment', 'prefect.resource.name': deployment.name}))
            tags.extend(deployment.tags)
        if flow:
            related.append(RelatedResource({'prefect.resource.id': f'prefect.flow.{flow.id}', 'prefect.resource.role': 'flow', 'prefect.resource.name': flow.name}))
        tags.extend(flow_run.tags)
        related = [RelatedResource.model_validate(r) for r in related]
        related += tags_as_related_resources(set(tags))
        emit_event(event='prefect.flow-run.heartbeat', resource={'prefect.resource.id': f'prefect.flow-run.{flow_run.id}', 'prefect.resource.name': flow_run.name, 'prefect.version': __version__}, related=related)

    def _event_resource(self) -> Dict[str, str]:
        from prefect import __version__
        return {'prefect.resource.id': f'prefect.runner.{slugify(self.name)}', 'prefect.resource.name': self.name, 'prefect.version': __version__}

    def _emit_flow_run_cancelled_event(self, flow_run: FlowRun, flow: Optional[APIFlow], deployment: Optional[DeploymentResponse]) -> None:
        related: List[RelatedResource] = []
        tags: List[str] = []
        if deployment:
            related.append(RelatedResource({'prefect.resource.id': f'prefect.deployment.{deployment.id}', 'prefect.resource.role': 'deployment', 'prefect.resource.name': deployment.name}))
            tags.extend(deployment.tags)
        if flow:
            related.append(RelatedResource({'prefect.resource.id': f'prefect.flow.{flow.id}', 'prefect.resource.role': 'flow', 'prefect.resource.name': flow.name}))
        related.append(RelatedResource({'prefect.resource.id': f'prefect.flow-run.{flow_run.id}', 'prefect.resource.role': 'flow-run', 'prefect.resource.name': flow_run.name}))
        tags.extend(flow_run.tags)
        related = [RelatedResource.model_validate(r) for r in related]
        related += tags_as_related_resources(set(tags))
        emit_event(event='prefect.runner.cancelled-flow-run', resource=self._event_resource(), related=related)
        self._logger.debug(f'Emitted flow run heartbeat event for {flow_run.id}')

    async def _get_scheduled_flow_runs(self) -> List[FlowRun]:
        scheduled_before = DateTime.now('utc').add(seconds=int(self._prefetch_seconds))
        self._logger.debug(f'Querying for flow runs scheduled before {scheduled_before}')
        scheduled_flow_runs = await self._client.get_scheduled_flow_runs_for_deployments(deployment_ids=list(self._deployment_ids), scheduled_before=scheduled_before)
        self._logger.debug(f'Discovered {len(scheduled_flow_runs)} scheduled_flow_runs')
        return scheduled_flow_runs

    def has_slots_available(self) -> bool:
        if not self._limiter:
            return False
        return self._limiter.available_tokens > 0

    def _acquire_limit_slot(self, flow_run_id: UUID) -> bool:
        try:
            if self._limiter:
                self._limiter.acquire_on_behalf_of_nowait(flow_run_id)
                self._logger.debug("Limit slot acquired for flow run '%s'", flow_run_id)
            return True
        except RuntimeError as exc:
            if "this borrower is already holding one of this CapacityLimiter's tokens" in str(exc):
                self._logger.warning(f"Duplicate submission of flow run '{flow_run_id}' detected. Runner will not re-submit flow run.")
                return False
            else:
                raise
        except anyio.WouldBlock:
            if TYPE_CHECKING:
                assert self._limiter is not None
            self._logger.info(f'Flow run limit reached; {self._limiter.borrowed_tokens} flow runs in progress. You can control this limit by passing a `limit` value to `serve` or adjusting the PREFECT_RUNNER_PROCESS_LIMIT setting.')
            return False

    def _release_limit_slot(self, flow_run_id: UUID) -> None:
        if self._limiter:
            self._limiter.release_on_behalf_of(flow_run_id)
            self._logger.debug("Limit slot released for flow run '%s'", flow_run_id)

    async def _submit_scheduled_flow_runs(self, flow_run_response: List[FlowRun], entrypoints: Optional[List[str]] = None) -> List[FlowRun]:
        submittable_flow_runs = sorted(flow_run_response, key=lambda run: run.next_scheduled_start_time or datetime.datetime.max)
        for i, flow_run in enumerate(submittable_flow_runs):
            if flow_run.id in self._submitting_flow_run_ids:
                continue
            if self._acquire_limit_slot(flow_run.id):
                run_logger = self._get_flow_run_logger(flow_run)
                run_logger.info(f"Runner '{self.name}' submitting flow run '{flow_run.id}'")
                self._submitting_flow_run_ids.add(flow_run.id)
                self._runs_task_group.start_soon(partial(self._submit_run, flow_run=flow_run, entrypoint=entrypoints[i] if entrypoints else None))
            else:
                break
        return list(filter(lambda run: run.id in self._submitting_flow_run_ids, submittable_flow_runs))

    async def _submit_run(self, flow_run: FlowRun, entrypoint: Optional[str] = None) -> None:
        run_logger = self._get_flow_run_logger(flow_run)
        ready_to_submit = await self._propose_pending_state(flow_run)
        if ready_to_submit:
            readiness_result = await self._runs_task_group.start(partial(self._submit_run_and_capture_errors, flow_run=flow_run, entrypoint=entrypoint))
            if readiness_result and (not isinstance(readiness_result, Exception)):
                self._flow_run_process_map[flow_run.id] = ProcessMapEntry(pid=readiness_result, flow_run=flow_run)
            if self.heartbeat_seconds is not None:
                await self._emit_flow_run_heartbeat(flow_run)
            run_logger.info(f"Completed submission of flow run '{flow_run.id}'")
        else:
            self._release_limit_slot(flow_run.id)
        self._submitting_flow_run_ids.discard(flow_run.id)

    async def _submit_run_and_capture_errors(self, flow_run: FlowRun, task_status: anyio.abc.TaskStatus, entrypoint: Optional[str] = None) -> Union[int, Exception]:
        run_logger = self._get_flow_run_logger(flow_run)
        try:
            status_code = await self._run_process(flow_run=flow_run, task_status=task_status, entrypoint=entrypoint)
        except Exception as exc:
            if task_status:
                run_logger.exception(f"Failed to start process for flow run '{flow_run.id}'.")
                task_status.started(exc)
                message = f'Flow run process could not be started:\n{exc!r}'
                await self._propose_crashed_state(flow_run, message)
            else:
                run_logger.exception(f"An error occurred while monitoring flow run '{flow_run.id}'. The flow run will not be marked as failed, but an issue may have occurred.")
            return exc
        finally:
            self._release_limit_slot(flow_run.id)
            self._flow_run_process_map.pop(flow_run.id, None)
        if status_code != 0:
            await self._propose_crashed_state(flow_run, f'Flow run process exited with non-zero status code {status_code}.')
        api_flow_run = await self._client.read_flow_run(flow_run_id=flow_run.id)
        terminal_state = api_flow_run.state
        if terminal_state and terminal_state.is_crashed():
            await self._run_on_crashed_hooks(flow_run=flow_run, state=terminal_state)
        return status_code

    async def _propose_pending_state(self, flow_run: FlowRun) -> bool:
        run_logger = self._get_flow_run_logger(flow_run)
        state = flow_run.state
        try:
            state = await propose_state(self._client, Pending(), flow_run_id=flow_run.id)
        except Abort as exc:
            run_logger.info(f"Aborted submission of flow run '{flow_run.id}'. Server sent an abort signal: {exc}")
            return False
        except Exception:
            run_logger.exception(f"Failed to update state of flow run '{flow_run.id}'")
            return False
        if not state.is_pending():
            run_logger.info(f"Aborted submission of flow run '{flow_run.id}': Server returned a non-pending state {state.type.value!r}")
            return False
        return True

    async def _propose_failed_state(self, flow_run: FlowRun, exc: Exception) -> None:
        run_logger = self._get_flow_run_logger(flow_run)
        try:
            await propose_state(self._client, await exception_to_failed_state(message='Submission failed.', exc=exc), flow_run_id=flow_run.id)
        except Abort:
            pass
        except Exception:
            run_logger.error(f"Failed to update state of flow run '{flow_run.id}'", exc_info=True)

    async def _propose_crashed_state(self, flow_run: FlowRun, message: str) -> None:
        run_logger = self._get_flow_run_logger(flow_run)
        try:
            state = await propose_state(self._client, Crashed(message=message), flow_run_id=flow_run.id)
        except Abort:
            pass
        except Exception:
            run_logger.exception(f"Failed to update state of flow run '{flow_run.id}'")
        else:
            if state.is_crashed():
                run_logger.info(f"Reported flow run '{flow_run.id}' as crashed: {message}")

    async def _mark_flow_run_as_cancelled(self, flow_run: FlowRun, state_updates: Optional[Dict[str, Any]] = None) -> None:
        state_updates = state_updates or {}
        state_updates.setdefault('name', 'Cancelled')
        state_updates.setdefault('type', StateType.CANCELLED)
        state = flow_run.state.model_copy(update=state_updates) if flow_run.state else None
        if not state:
            self._logger.warning(f'Could not find state for flow run {flow_run.id} and cancellation cannot be guaranteed.')
            return
        await self._client.set_flow_run_state(flow_run.id, state, force=True)
        await self._schedule_task(60 * 10, self._cancelling_flow_run_ids.remove, flow_run.id)

    async def _schedule_task(self, __in_seconds: int, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        async def wrapper(task_status: anyio.abc.TaskStatus) -> None:
            if self.started:
                with anyio.CancelScope() as scope:
                    self._scheduled_task_scopes.add(scope)
                    task_status.started()
                    await anyio.sleep(__in_seconds)
                self._scheduled_task_scopes.remove(scope)
            else:
                task_status.started()
            result = fn(*args, **kwargs)
            if asyncio.iscoroutine(result):
                await result
        await self._runs_task_group.start(wrapper)

    async def _run_on_cancellation_hooks(self, flow_run: FlowRun, state: State) -> None:
        if state.is_cancelling():
            try:
                flow = await load_flow_from_flow_run(flow_run, storage_base_path=str(self._tmp_dir))
                hooks = flow.on_cancellation_hooks or []
                await _run_hooks(hooks, flow_run, flow, state)
            except ObjectNotFound:
                run_logger = self._get_flow_run_logger(flow_run)
                run_logger.warning(f'Runner cannot retrieve flow to execute cancellation hooks for flow run {flow_run.id!r}.')

    async def _run_on_crashed_hooks(self, flow_run: FlowRun, state: State) -> None:
        if state.is_crashed():
            flow = await load_flow_from_flow_run(flow_run, storage_base_path=str(self._tmp_dir))
            hooks = flow.on_crashed_hooks or []
            await _run_hooks(hooks, flow_run, flow, state)

    async def __aenter__(self) -> Self:
        self._logger.debug('Starting runner...')
        self._client = get_client()
        self._tmp_dir.mkdir(parents=True)
        self._limiter = anyio.CapacityLimiter(self.limit) if self.limit else None
        if not hasattr(self, '_loop') or not self._loop:
            self._loop = asyncio.get_event_loop()
        await self._client.__aenter__()
        if not hasattr(self, '_runs_task_group') or not self._runs_task_group:
            self._runs_task_group = anyio.create_task_group()
        await self._runs_task_group.__aenter__()
        if not hasattr(self, '_loops_task_group') or not self._loops_task_group:
            self._loops_task_group = anyio.create_task_group()
        self.started = True
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        self._logger.debug('Stopping runner...')
        if self.pause_on_shutdown:
            await self._pause_schedules()
        self.started = False
        for scope in self._scheduled_task_scopes:
            scope.cancel()
        if self._runs_task_group:
            await self._runs_task_group.__aexit__(*exc_info)
        if self._client:
            await self._client.__aexit__(*exc_info)
        shutil.rmtree(str(self._tmp_dir))
        del self._runs_task_group, self._loops_task_group

    def __repr__(self) -> str:
        return f'Runner(name={self.name!r})'

if sys.platform == 'win32':
    STATUS_CONTROL_C_EXIT = 3221225786

async def _run_hooks(hooks: List[FlowStateHook], flow_run: FlowRun, flow: Flow, state: State) -> None:
    logger = flow_run_logger(flow_run, flow)
    for hook in hooks:
        try:
            logger.info(f'Running hook {hook.__name__!r} in response to entering state {state.name!r}')
            if is_async_fn(hook):
                await hook(flow=flow, flow_run=flow_run, state=state)
            else:
                await from_async.call_in_new_thread(create_call(hook, flow=flow, flow_run=flow_run, state=state))
        except Exception:
            logger.error(f'An error was encountered while running hook {hook.__name__!r}', exc_info=True)
        else:
            logger.info(f'Hook {hook.__name__!r} finished running successfully')
