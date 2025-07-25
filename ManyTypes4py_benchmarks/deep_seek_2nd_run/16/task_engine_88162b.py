from __future__ import annotations
import asyncio
import inspect
import logging
import threading
import time
from asyncio import CancelledError
from contextlib import ExitStack, asynccontextmanager, contextmanager, nullcontext
from dataclasses import dataclass, field
from functools import partial
from textwrap import dedent
from typing import (
    TYPE_CHECKING, Any, AsyncGenerator, Callable, Coroutine, Dict, Generator, 
    Generic, Iterator, List, Literal, Optional, Sequence, Set, Tuple, Type, 
    TypeVar, Union, cast, overload
)
from uuid import UUID
import anyio
from opentelemetry import trace
from typing_extensions import ParamSpec, Self
from prefect.cache_policies import CachePolicy
from prefect.client.orchestration import PrefectClient, SyncPrefectClient, get_client
from prefect.client.schemas import TaskRun
from prefect.client.schemas.objects import State, TaskRunInput
from prefect.concurrency.context import ConcurrencyContext
from prefect.concurrency.v1.asyncio import concurrency as aconcurrency
from prefect.concurrency.v1.context import ConcurrencyContext as ConcurrencyContextV1
from prefect.concurrency.v1.sync import concurrency
from prefect.context import AsyncClientContext, FlowRunContext, SyncClientContext, TaskRunContext, hydrated_context
from prefect.events.schemas.events import Event as PrefectEvent
from prefect.exceptions import Abort, Pause, PrefectException, TerminationSignal, UpstreamTaskError
from prefect.logging.loggers import get_logger, patch_print, task_run_logger
from prefect.results import ResultRecord, _format_user_supplied_storage_key, get_result_store, should_persist_result
from prefect.settings import PREFECT_DEBUG_MODE, PREFECT_TASKS_REFRESH_CACHE
from prefect.settings.context import get_current_settings
from prefect.states import AwaitingRetry, Completed, Failed, Pending, Retrying, Running, exception_to_crashed_state, exception_to_failed_state, return_value_to_state
from prefect.telemetry.run_telemetry import RunTelemetry
from prefect.transactions import IsolationLevel, Transaction, transaction
from prefect.types._datetime import DateTime, Duration
from prefect.utilities._engine import get_hook_name
from prefect.utilities.annotations import NotSet
from prefect.utilities.asyncutils import run_coro_as_sync
from prefect.utilities.callables import call_with_parameters, parameters_to_args_kwargs
from prefect.utilities.collections import visit_collection
from prefect.utilities.engine import emit_task_run_state_change_event, link_state_to_result, resolve_to_final_result
from prefect.utilities.math import clamped_poisson_interval
from prefect.utilities.timeout import timeout, timeout_async

if TYPE_CHECKING:
    from prefect.tasks import OneOrManyFutureOrResult, Task

P = ParamSpec('P')
R = TypeVar('R')
T = TypeVar('T')
BACKOFF_MAX = 10

class TaskRunTimeoutError(TimeoutError):
    """Raised when a task run exceeds its timeout."""

@dataclass
class BaseTaskRunEngine(Generic[P, R]):
    logger: logging.Logger = field(default_factory=lambda: get_logger('engine'))
    parameters: Optional[Dict[str, Any]] = None
    task_run: Optional[TaskRun] = None
    retries: int = 0
    wait_for: Optional[Any] = None
    context: Optional[Dict[str, Any]] = None
    _return_value: Any = NotSet
    _raised: Any = NotSet
    _initial_run_context: Optional[Any] = None
    _is_started: bool = False
    _task_name_set: bool = False
    _last_event: Optional[PrefectEvent] = None
    _telemetry: RunTelemetry = field(default_factory=RunTelemetry)

    def __post_init__(self) -> None:
        if self.parameters is None:
            self.parameters = {}

    @property
    def state(self) -> State:
        if not self.task_run or not self.task_run.state:
            raise ValueError('Task run is not set')
        return self.task_run.state

    def is_cancelled(self) -> bool:
        if self.context and 'cancel_event' in self.context and isinstance(self.context['cancel_event'], threading.Event):
            return self.context['cancel_event'].is_set()
        return False

    def compute_transaction_key(self) -> Optional[str]:
        key = None
        if self.task.cache_policy and isinstance(self.task.cache_policy, CachePolicy):
            flow_run_context = FlowRunContext.get()
            task_run_context = TaskRunContext.get()
            if flow_run_context:
                parameters = flow_run_context.parameters
            else:
                parameters = None
            try:
                if not task_run_context:
                    raise ValueError('Task run context is not set')
                key = self.task.cache_policy.compute_key(task_ctx=task_run_context, inputs=self.parameters or {}, flow_parameters=parameters or {})
            except Exception:
                self.logger.exception('Error encountered when computing cache key - result will not be persisted.')
                key = None
        elif self.task.result_storage_key is not None:
            key = _format_user_supplied_storage_key(self.task.result_storage_key)
        return key

    def _resolve_parameters(self) -> None:
        if not self.parameters:
            return None
        resolved_parameters: Dict[str, Any] = {}
        for parameter, value in self.parameters.items():
            try:
                resolved_parameters[parameter] = visit_collection(value, visit_fn=resolve_to_final_result, return_data=True, max_depth=-1, remove_annotations=True, context={'parameter_name': parameter})
            except UpstreamTaskError:
                raise
            except Exception as exc:
                raise PrefectException(f'Failed to resolve inputs in parameter {parameter!r}. If your parameter type is not supported, consider using the `quote` annotation to skip resolution of inputs.') from exc
        self.parameters = resolved_parameters

    def _set_custom_task_run_name(self) -> None:
        from prefect.utilities._engine import resolve_custom_task_run_name
        if not self._task_name_set and self.task.task_run_name:
            task_run_name = resolve_custom_task_run_name(task=self.task, parameters=self.parameters or {})
            self.logger.extra['task_run_name'] = task_run_name
            self.logger.debug(f'Renamed task run {self.task_run.name!r} to {task_run_name!r}')
            self.task_run.name = task_run_name
            self._task_name_set = True
            self._telemetry.update_run_name(name=task_run_name)

    def _wait_for_dependencies(self) -> None:
        if not self.wait_for:
            return
        visit_collection(self.wait_for, visit_fn=resolve_to_final_result, return_data=False, max_depth=-1, remove_annotations=True, context={'current_task_run': self.task_run, 'current_task': self.task})

    def record_terminal_state_timing(self, state: State) -> None:
        if self.task_run and self.task_run.start_time and (not self.task_run.end_time):
            self.task_run.end_time = state.timestamp
            if self.state.is_running():
                self.task_run.total_run_time += state.timestamp - self.state.timestamp

    def is_running(self) -> bool:
        """Whether or not the engine is currently running a task."""
        if (task_run := getattr(self, 'task_run', None)) is None:
            return False
        return task_run.state.is_running() or task_run.state.is_scheduled()

    def log_finished_message(self) -> None:
        if not self.task_run:
            return
        display_state = repr(self.state) if PREFECT_DEBUG_MODE else str(self.state)
        level = logging.INFO if self.state.is_completed() else logging.ERROR
        msg = f'Finished in state {display_state}'
        if self.state.is_pending() and self.state.name != 'NotReady':
            msg += '\nPlease wait for all submitted tasks to complete before exiting your flow by calling `.wait()` on the `PrefectFuture` returned from your `.submit()` calls.'
            msg += dedent('\n\n                        Example:\n\n                        from prefect import flow, task\n\n                        @task\n                        def say_hello(name):\n                            print(f"Hello, {name}!")\n\n                        @flow\n                        def example_flow():\n                            future = say_hello.submit(name="Marvin")\n                            future.wait()\n\n                        example_flow()\n                                    ')
        self.logger.log(level=level, msg=msg)

    def handle_rollback(self, txn: Transaction) -> None:
        assert self.task_run is not None
        rolled_back_state = Completed(name='RolledBack', message='Task rolled back as part of transaction')
        self._last_event = emit_task_run_state_change_event(task_run=self.task_run, initial_state=self.state, validated_state=rolled_back_state, follows=self._last_event)

@dataclass
class SyncTaskRunEngine(BaseTaskRunEngine[P, R]):
    task_run: Optional[TaskRun] = None
    _client: Optional[SyncPrefectClient] = None

    @property
    def client(self) -> SyncPrefectClient:
        if not self._is_started or self._client is None:
            raise RuntimeError('Engine has not started.')
        return self._client

    def can_retry(self, exc: Exception) -> bool:
        retry_condition = self.task.retry_condition_fn
        if not self.task_run:
            raise ValueError('Task run is not set')
        try:
            self.logger.debug(f'Running `retry_condition_fn` check {retry_condition!r} for task {self.task.name!r}')
            state = Failed(data=exc, message=f'Task run encountered unexpected exception: {repr(exc)}')
            if asyncio.iscoroutinefunction(retry_condition):
                should_retry = run_coro_as_sync(retry_condition(self.task, self.task_run, state))
            elif inspect.isfunction(retry_condition):
                should_retry = retry_condition(self.task, self.task_run, state)
            else:
                should_retry = not retry_condition
            return should_retry
        except Exception:
            self.logger.error(f"An error was encountered while running `retry_condition_fn` check '{retry_condition!r}' for task {self.task.name!r}", exc_info=True)
            return False

    def call_hooks(self, state: Optional[State] = None) -> None:
        if state is None:
            state = self.state
        task = self.task
        task_run = self.task_run
        if not task_run:
            raise ValueError('Task run is not set')
        if state.is_failed() and task.on_failure_hooks:
            hooks = task.on_failure_hooks
        elif state.is_completed() and task.on_completion_hooks:
            hooks = task.on_completion_hooks
        else:
            hooks = None
        for hook in hooks or []:
            hook_name = get_hook_name(hook)
            try:
                self.logger.info(f'Running hook {hook_name!r} in response to entering state {state.name!r}')
                result = hook(task, task_run, state)
                if asyncio.iscoroutine(result):
                    run_coro_as_sync(result)
            except Exception:
                self.logger.error(f'An error was encountered while running hook {hook_name!r}', exc_info=True)
            else:
                self.logger.info(f'Hook {hook_name!r} finished running successfully')

    def begin_run(self) -> None:
        try:
            self._resolve_parameters()
            self._set_custom_task_run_name()
            self._wait_for_dependencies()
        except UpstreamTaskError as upstream_exc:
            state = self.set_state(Pending(name='NotReady', message=str(upstream_exc)), force=self.state.is_pending())
            return
        new_state = Running()
        assert self.task_run is not None, 'Task run is not set'
        self.task_run.start_time = new_state.timestamp
        flow_run_context = FlowRunContext.get()
        if flow_run_context and flow_run_context.flow_run:
            flow_run = flow_run_context.flow_run
            self.task_run.flow_run_run_count = flow_run.run_count
        state = self.set_state(new_state)
        if state.is_completed():
            try:
                state.result(retry_result_failure=False, _sync=True)
            except Exception:
                state = self.set_state(new_state, force=True)
        backoff_count = 0
        while state.is_pending() or state.is_paused():
            if backoff_count < BACKOFF_MAX:
                backoff_count += 1
            interval = clamped_poisson_interval(average_interval=backoff_count, clamping_factor=0.3)
            time.sleep(interval)
            state = self.set_state(new_state)

    def set_state(self, state: State, force: bool = False) -> State:
        last_state = self.state
        if not self.task_run:
            raise ValueError('Task run is not set')
        self.task_run.state = new_state = state
        if last_state.timestamp == new_state.timestamp:
            new_state.timestamp += Duration(microseconds=1)
        new_state.state_details.task_run_id = self.task_run.id
        new_state.state_details.flow_run_id = self.task_run.flow_run_id
        self.task_run.state_id = new_state.id
        self.task_run.state_type = new_state.type
        self.task_run.state_name = new_state.name
        if new_state.is_running():
            self.task_run.run_count += 1
        if new_state.is_final():
            if isinstance(state.data, ResultRecord):
                result = state.data.result
            else:
                result = state.data
            link_state_to_result(state, result)
        self._last_event = emit_task_run_state_change_event(task_run=self.task_run, initial_state=last_state, validated_state=self.task_run.state, follows=self._last_event)
        self._telemetry.update_state(new_state)
        return new_state

    def result(self, raise_on_failure: bool = True) -> Any:
        if self._return_value is not NotSet:
            if isinstance(self._return_value, ResultRecord):
                return self._return_value.result
            return self._return_value
        if self._raised is not NotSet:
            if raise_on_failure:
                raise self._raised
            return self._raised

    def handle_success(self, result: Any, transaction: Transaction) -> Any:
        if self.task.cache_expiration is not None:
            expiration = DateTime.now('utc') + self.task.cache_expiration
        else:
            expiration = None
        terminal_state = run_coro_as_sync(return_value_to_state(result, result_store=get_result_store(), key=transaction.key, expiration=expiration))
        handle_rollback = partial(self.handle_rollback)
        handle_rollback.log_on_run = False
        transaction.stage(terminal_state.data, on_rollback_hooks=[handle_rollback] + self.task.on_rollback_hooks, on_commit_hooks=self.task.on_commit_hooks)
        if transaction.is_committed():
            terminal_state.name = 'Cached'
        self.record_terminal_state_timing(terminal_state)
        self.set_state(terminal_state)
        self._return_value = result
        self._telemetry.end_span_on_success()
        return result

    def handle_retry(self, exc: Exception) -> bool:
        """Handle any task run retries.

        - If the task has retries left, and the retry condition is met, set the task to retrying and return True.
        - If the task has a retry delay, place in AwaitingRetry state with a delayed scheduled time.
        - If the task has no retries left, or the retry condition is not met, return False.
        """
        if self.retries < self.task.retries and self.can_retry(exc):
            if self.task.retry_delay_seconds:
                delay = self.task.retry_delay_seconds[min(self.retries, len(self.task.retry_delay_seconds) - 1)] if isinstance(self.task.retry_delay_seconds, Sequence) else self.task.retry_delay_seconds
                new_state = AwaitingRetry(scheduled_time=DateTime.now('utc').add(seconds=delay))
            else:
                delay = None
                new_state = Retrying()
            self.logger.info('Task run failed with exception: %r - Retry %s/%s will start %s', exc, self.retries + 1, self.task.retries, str(delay) + ' second(s) from now' if delay else 'immediately')
            self.set_state(new_state, force=True)
            self.retries = self.retries + 1
            return True
        elif self.retries >= self.task.retries:
            self.logger.error('Task run failed with exception: %r - Retries are exhausted', exc, exc_info=True)
            return False
        return False

    def handle_exception(self, exc: Exception) -> None:
        self._telemetry.record_exception(exc)
        if not self.handle_retry(exc):
            state = run_coro_as_sync(exception_to_failed_state(exc, message='Task run encountered an exception', result_store=get_result_store(), write_result=True))
            self.record_terminal_state_timing(state)
            self.set_state(state)
            self._raised = exc
            self._telemetry.end_span_on_failure(state.message if state else None)

    def handle_timeout(self, exc: TimeoutError) -> None:
        if not self.handle_retry(exc):
            if isinstance(exc, TaskRunTimeoutError):
                message = f'Task run exceeded timeout of {self.task.timeout_seconds} second(s)'
            else:
                message = f'Task run failed due to timeout: {exc!r}'
            self.logger.error(message)
            state = Failed(data=exc, message=message, name='TimedOut')
            self.record_terminal_state_timing(state)
            self.set_state(state)
            self._raised = exc

   