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
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Coroutine,
    Dict,
    Generator,
    Generic,
    Literal,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
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
from prefect.states import (
    AwaitingRetry,
    Completed,
    Failed,
    Pending,
    Retrying,
    Running,
    exception_to_crashed_state,
    exception_to_failed_state,
    return_value_to_state,
)
from prefect.telemetry.run_telemetry import RunTelemetry
from prefect.transactions import IsolationLevel, Transaction, transaction
from prefect.types._datetime import DateTime, Duration
from prefect.utilities._engine import get_hook_name, emit_task_run_state_change_event, link_state_to_result, resolve_to_final_result
from prefect.utilities.annotations import NotSet
from prefect.utilities.asyncutils import run_coro_as_sync
from prefect.utilities.callables import call_with_parameters, parameters_to_args_kwargs
from prefect.utilities.collections import visit_collection
from prefect.utilities.engine import clamped_poisson_interval, timeout, timeout_async

if TYPE_CHECKING:
    from prefect.tasks import OneOrManyFutureOrResult, Task

P = ParamSpec("P")
R = TypeVar("R")

BACKOFF_MAX: int = 10


class TaskRunTimeoutError(TimeoutError):
    """Raised when a task run exceeds its timeout."""
    pass


@dataclass
class BaseTaskRunEngine(Generic[P, R]):
    logger: logging.Logger = field(default_factory=lambda: get_logger("engine"))
    parameters: Optional[Dict[str, Any]] = None
    task_run: Optional[Any] = None
    retries: int = 0
    wait_for: Optional[Any] = None
    context: Optional[Dict[str, Any]] = None
    _return_value: Any = NotSet
    _raised: Any = NotSet
    _initial_run_context: Any = None
    _is_started: bool = False
    _task_name_set: bool = False
    _last_event: Optional[Any] = None
    _telemetry: RunTelemetry = field(default_factory=RunTelemetry)

    def __post_init__(self) -> None:
        if self.parameters is None:
            self.parameters = {}

    @property
    def state(self) -> State:
        if not self.task_run or not self.task_run.state:
            raise ValueError("Task run is not set")
        return self.task_run.state

    def is_cancelled(self) -> bool:
        if self.context and "cancel_event" in self.context and isinstance(self.context["cancel_event"], threading.Event):
            return self.context["cancel_event"].is_set()
        return False

    def compute_transaction_key(self) -> Optional[Any]:
        key: Optional[Any] = None
        if self.task.cache_policy and isinstance(self.task.cache_policy, CachePolicy):
            flow_run_context = FlowRunContext.get()
            task_run_context = TaskRunContext.get()
            if flow_run_context:
                parameters = flow_run_context.parameters
            else:
                parameters = None
            try:
                if not task_run_context:
                    raise ValueError("Task run context is not set")
                key = self.task.cache_policy.compute_key(task_ctx=task_run_context, inputs=self.parameters or {}, flow_parameters=parameters or {})
            except Exception:
                self.logger.exception("Error encountered when computing cache key - result will not be persisted.")
                key = None
        elif self.task.result_storage_key is not None:
            key = _format_user_supplied_storage_key(self.task.result_storage_key)
        return key

    def _resolve_parameters(self) -> None:
        if not self.parameters:
            return
        resolved_parameters: Dict[str, Any] = {}
        for parameter, value in self.parameters.items():
            try:
                resolved_parameters[parameter] = visit_collection(
                    value, visit_fn=resolve_to_final_result, return_data=True, max_depth=-1, remove_annotations=True, context={"parameter_name": parameter}
                )
            except UpstreamTaskError:
                raise
            except Exception as exc:
                raise PrefectException(
                    f"Failed to resolve inputs in parameter {parameter!r}. If your parameter type is not supported, consider using the `quote` annotation to skip resolution of inputs."
                ) from exc
        self.parameters = resolved_parameters

    def _set_custom_task_run_name(self) -> None:
        from prefect.utilities._engine import resolve_custom_task_run_name
        if not self._task_name_set and self.task.task_run_name:
            task_run_name: str = resolve_custom_task_run_name(task=self.task, parameters=self.parameters or {})
            self.logger.extra["task_run_name"] = task_run_name
            self.logger.debug(f"Renamed task run {self.task_run.name!r} to {task_run_name!r}")
            self.task_run.name = task_run_name
            self._task_name_set = True
            self._telemetry.update_run_name(name=task_run_name)

    def _wait_for_dependencies(self) -> None:
        if not self.wait_for:
            return
        visit_collection(
            self.wait_for, visit_fn=resolve_to_final_result, return_data=False, max_depth=-1, remove_annotations=True, context={"current_task_run": self.task_run, "current_task": self.task}
        )

    def record_terminal_state_timing(self, state: State) -> None:
        if self.task_run and self.task_run.start_time and (not self.task_run.end_time):
            self.task_run.end_time = state.timestamp
            if self.state.is_running():
                self.task_run.total_run_time += state.timestamp - self.state.timestamp

    def is_running(self) -> bool:
        """Whether or not the engine is currently running a task."""
        if (task_run := getattr(self, "task_run", None)) is None:
            return False
        return task_run.state.is_running() or task_run.state.is_scheduled()

    def log_finished_message(self) -> None:
        if not self.task_run:
            return
        display_state: str = repr(self.state) if PREFECT_DEBUG_MODE else str(self.state)
        level: int = logging.INFO if self.state.is_completed() else logging.ERROR
        msg: str = f"Finished in state {display_state}"
        if self.state.is_pending() and self.state.name != "NotReady":
            msg += "\nPlease wait for all submitted tasks to complete before exiting your flow by calling `.wait()` on the `PrefectFuture` returned from your `.submit()` calls."
            msg += dedent(
                "\n\n                        Example:\n\n                        from prefect import flow, task\n\n                        @task\n                        def say_hello(name):\n                            print(f\"Hello, {name}!\")\n\n                        @flow\n                        def example_flow():\n                            future = say_hello.submit(name=\"Marvin\")\n                            future.wait()\n\n                        example_flow()\n                                    "
            )
        self.logger.log(level=level, msg=msg)

    def handle_rollback(self, txn: Transaction) -> None:
        assert self.task_run is not None
        rolled_back_state: State = Completed(name="RolledBack", message="Task rolled back as part of transaction")
        self._last_event = emit_task_run_state_change_event(
            task_run=self.task_run, initial_state=self.state, validated_state=rolled_back_state, follows=self._last_event
        )


@dataclass
class SyncTaskRunEngine(BaseTaskRunEngine[P, R]):
    task_run: Optional[Any] = None
    _client: Optional[SyncPrefectClient] = None

    @property
    def client(self) -> SyncPrefectClient:
        if not self._is_started or self._client is None:
            raise RuntimeError("Engine has not started.")
        return self._client

    def can_retry(self, exc: Exception) -> bool:
        retry_condition = self.task.retry_condition_fn
        if not self.task_run:
            raise ValueError("Task run is not set")
        try:
            self.logger.debug(f"Running `retry_condition_fn` check {retry_condition!r} for task {self.task.name!r}")
            state: State = Failed(data=exc, message=f"Task run encountered unexpected exception: {repr(exc)}")
            if asyncio.iscoroutinefunction(retry_condition):
                should_retry: bool = run_coro_as_sync(retry_condition(self.task, self.task_run, state))
            elif inspect.isfunction(retry_condition):
                should_retry = retry_condition(self.task, self.task_run, state)
            else:
                should_retry = not retry_condition
            return should_retry
        except Exception:
            self.logger.error(
                f"An error was encountered while running `retry_condition_fn` check '{retry_condition!r}' for task {self.task.name!r}",
                exc_info=True,
            )
            return False

    def call_hooks(self, state: Optional[State] = None) -> None:
        if state is None:
            state = self.state
        task = self.task
        task_run = self.task_run
        if not task_run:
            raise ValueError("Task run is not set")
        if state.is_failed() and task.on_failure_hooks:
            hooks = task.on_failure_hooks
        elif state.is_completed() and task.on_completion_hooks:
            hooks = task.on_completion_hooks
        else:
            hooks = None
        for hook in hooks or []:
            hook_name: str = get_hook_name(hook)
            try:
                self.logger.info(f"Running hook {hook_name!r} in response to entering state {state.name!r}")
                result = hook(task, task_run, state)
                if asyncio.iscoroutine(result):
                    run_coro_as_sync(result)
            except Exception:
                self.logger.error(f"An error was encountered while running hook {hook_name!r}", exc_info=True)
            else:
                self.logger.info(f"Hook {hook_name!r} finished running successfully")

    def begin_run(self) -> None:
        try:
            self._resolve_parameters()
            self._set_custom_task_run_name()
            self._wait_for_dependencies()
        except UpstreamTaskError as upstream_exc:
            state = self.set_state(Pending(name="NotReady", message=str(upstream_exc)), force=self.state.is_pending())
            return
        new_state: State = Running()
        assert self.task_run is not None, "Task run is not set"
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
        backoff_count: int = 0
        while state.is_pending() or state.is_paused():
            if backoff_count < BACKOFF_MAX:
                backoff_count += 1
            interval: float = clamped_poisson_interval(average_interval=backoff_count, clamping_factor=0.3)
            time.sleep(interval)
            state = self.set_state(new_state)

    def set_state(self, state: State, force: bool = False) -> State:
        last_state: State = self.state
        if not self.task_run:
            raise ValueError("Task run is not set")
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
        self._last_event = emit_task_run_state_change_event(
            task_run=self.task_run, initial_state=last_state, validated_state=self.task_run.state, follows=self._last_event
        )
        self._telemetry.update_state(new_state)
        return new_state

    def result(self, raise_on_failure: bool = True) -> R:
        if self._return_value is not NotSet:
            if isinstance(self._return_value, ResultRecord):
                return self._return_value.result
            return self._return_value
        if self._raised is not NotSet:
            if raise_on_failure:
                raise self._raised
            return self._raised

    def handle_success(self, result: R, transaction: Transaction) -> R:
        if self.task.cache_expiration is not None:
            expiration: DateTime = DateTime.now("utc") + self.task.cache_expiration
        else:
            expiration = None
        terminal_state: State = run_coro_as_sync(return_value_to_state(result, result_store=get_result_store(), key=transaction.key, expiration=expiration))
        handle_rollback = partial(self.handle_rollback)
        handle_rollback.log_on_run = False  # type: ignore
        transaction.stage(terminal_state.data, on_rollback_hooks=[handle_rollback] + self.task.on_rollback_hooks, on_commit_hooks=self.task.on_commit_hooks)
        if transaction.is_committed():
            terminal_state.name = "Cached"
        self.record_terminal_state_timing(terminal_state)
        self.set_state(terminal_state)
        self._return_value = result
        self._telemetry.end_span_on_success()
        return result

    def handle_retry(self, exc: Exception) -> bool:
        if self.retries < self.task.retries and self.can_retry(exc):
            if self.task.retry_delay_seconds:
                delay: Optional[Union[int, float]] = (
                    self.task.retry_delay_seconds[min(self.retries, len(self.task.retry_delay_seconds) - 1)]
                    if isinstance(self.task.retry_delay_seconds, Sequence)
                    else self.task.retry_delay_seconds
                )
                new_state: State = AwaitingRetry(scheduled_time=DateTime.now("utc").add(seconds=delay))
            else:
                delay = None
                new_state = Retrying()
            self.logger.info(
                "Task run failed with exception: %r - Retry %s/%s will start %s",
                exc,
                self.retries + 1,
                self.task.retries,
                str(delay) + " second(s) from now" if delay else "immediately",
            )
            self.set_state(new_state, force=True)
            self.retries = self.retries + 1
            return True
        elif self.retries >= self.task.retries:
            self.logger.error("Task run failed with exception: %r - Retries are exhausted", exc, exc_info=True)
            return False
        return False

    def handle_exception(self, exc: Exception) -> None:
        self._telemetry.record_exception(exc)
        if not self.handle_retry(exc):
            state: State = run_coro_as_sync(exception_to_failed_state(exc, message="Task run encountered an exception", result_store=get_result_store(), write_result=True))
            self.record_terminal_state_timing(state)
            self.set_state(state)
            self._raised = exc
            self._telemetry.end_span_on_failure(state.message if state else None)

    def handle_timeout(self, exc: Exception) -> None:
        if not self.handle_retry(exc):
            if isinstance(exc, TaskRunTimeoutError):
                message: str = f"Task run exceeded timeout of {self.task.timeout_seconds} second(s)"
            else:
                message = f"Task run failed due to timeout: {exc!r}"
            self.logger.error(message)
            state: State = Failed(data=exc, message=message, name="TimedOut")
            self.record_terminal_state_timing(state)
            self.set_state(state)
            self._raised = exc

    def handle_crash(self, exc: Exception) -> None:
        state: State = run_coro_as_sync(exception_to_crashed_state(exc))
        self.logger.error(f"Crash detected! {state.message}")
        self.logger.debug("Crash details:", exc_info=exc)
        self.record_terminal_state_timing(state)
        self.set_state(state, force=True)
        self._raised = exc
        self._telemetry.record_exception(exc)
        self._telemetry.end_span_on_failure(state.message if state else None)

    @contextmanager
    def setup_run_context(self, client: Optional[SyncPrefectClient] = None) -> Generator[Self, None, None]:
        from prefect.utilities.engine import should_log_prints

        settings = get_current_settings()
        if client is None:
            client = self.client
        if not self.task_run:
            raise ValueError("Task run is not set")
        with ExitStack() as stack:
            log_prints: bool = False
            if (log_prints := should_log_prints(self.task)):
                stack.enter_context(patch_print())
            if self.task.persist_result is not None:
                persist_result = self.task.persist_result
            elif settings.tasks.default_persist_result is not None:
                persist_result = settings.tasks.default_persist_result
            else:
                persist_result = should_persist_result()
            stack.enter_context(
                TaskRunContext(
                    task=self.task,
                    log_prints=log_prints,
                    task_run=self.task_run,
                    parameters=self.parameters,
                    result_store=get_result_store().update_for_task(self.task, _sync=True),
                    client=client,
                    persist_result=persist_result,
                )
            )
            stack.enter_context(ConcurrencyContextV1())
            stack.enter_context(ConcurrencyContext())
            self.logger = task_run_logger(task_run=self.task_run, task=self.task)
            yield self

    @contextmanager
    def initialize_run(self, task_run_id: Optional[Any] = None, dependencies: Optional[Any] = None) -> Generator[Self, None, None]:
        with hydrated_context(self.context):
            with SyncClientContext.get_or_create() as client_ctx:
                self._client = client_ctx.client
                self._is_started = True
                parent_flow_run_context = FlowRunContext.get()
                parent_task_run_context = TaskRunContext.get()
                try:
                    if not self.task_run:
                        self.task_run = run_coro_as_sync(
                            self.task.create_local_run(
                                id=task_run_id,
                                parameters=self.parameters,
                                flow_run_context=parent_flow_run_context,
                                parent_task_run_context=parent_task_run_context,
                                wait_for=self.wait_for,
                                extra_task_inputs=dependencies,
                            )
                        )
                        self._last_event = emit_task_run_state_change_event(task_run=self.task_run, initial_state=None, validated_state=self.task_run.state)
                    with self.setup_run_context():
                        self.logger.debug(f"Created task run {self.task_run.name!r} for task {self.task.name!r}")
                        self._telemetry.start_span(run=self.task_run, client=self.client, parameters=self.parameters)
                        yield self
                except TerminationSignal as exc:
                    self.handle_crash(exc)
                    raise exc
                except Exception:
                    raise
                except (Pause, Abort) as exc:
                    if isinstance(exc, Abort):
                        self.logger.error("Task run was aborted: %s", exc)
                    raise
                except GeneratorExit:
                    raise
                except BaseException as exc:
                    self.handle_crash(exc)
                    raise
                finally:
                    self.log_finished_message()
                    self._is_started = False
                    self._client = None

    async def wait_until_ready(self) -> None:
        """Waits until the scheduled time (if its the future), then enters Running."""
        scheduled_time: Optional[DateTime] = self.state.state_details.scheduled_time
        if scheduled_time:
            sleep_time: float = (scheduled_time - DateTime.now("utc")).total_seconds()
            await anyio.sleep(sleep_time if sleep_time > 0 else 0)
            new_state: State = Retrying() if self.state.name == "AwaitingRetry" else Running()
            self.set_state(new_state, force=True)

    @contextmanager
    def start(self, task_run_id: Optional[Any] = None, dependencies: Optional[Any] = None) -> Generator[Self, None, None]:
        with self.initialize_run(task_run_id=task_run_id, dependencies=dependencies):
            with trace.use_span(self._telemetry.span) if self._telemetry.span else nullcontext():
                self.begin_run()
                try:
                    yield self
                finally:
                    self.call_hooks()

    @contextmanager
    def transaction_context(self) -> Generator[Transaction, None, None]:
        overwrite: bool = self.task.refresh_cache if self.task.refresh_cache is not None else PREFECT_TASKS_REFRESH_CACHE.value()
        isolation_level: Optional[IsolationLevel] = (
            IsolationLevel(self.task.cache_policy.isolation_level)
            if self.task.cache_policy and self.task.cache_policy is not NotSet and (self.task.cache_policy.isolation_level is not None)
            else None
        )
        with transaction(
            key=self.compute_transaction_key(),
            store=get_result_store(),
            overwrite=overwrite,
            logger=self.logger,
            write_on_commit=should_persist_result(),
            isolation_level=isolation_level,
        ) as txn:
            yield txn

    @contextmanager
    def run_context(self) -> Generator[Self, None, None]:
        with self.setup_run_context():
            try:
                with timeout(seconds=self.task.timeout_seconds, timeout_exc_type=TaskRunTimeoutError):
                    self.logger.debug(f"Executing task {self.task.name!r} for task run {self.task_run.name!r}...")
                    if self.is_cancelled():
                        raise CancelledError("Task run cancelled by the task runner")
                    yield self
            except TimeoutError as exc:
                self.handle_timeout(exc)
            except Exception as exc:
                self.handle_exception(exc)

    def call_task_fn(self, transaction: Transaction) -> R:
        parameters: Dict[str, Any] = self.parameters or {}
        if transaction.is_committed():
            result: R = transaction.read()
        elif self.task_run.tags:
            with concurrency(list(self.task_run.tags), self.task_run.id):
                result = call_with_parameters(self.task.fn, parameters)
        else:
            result = call_with_parameters(self.task.fn, parameters)
        self.handle_success(result, transaction=transaction)
        return result


@dataclass
class AsyncTaskRunEngine(BaseTaskRunEngine[P, R]):
    task_run: Optional[Any] = None
    _client: Optional[PrefectClient] = None

    @property
    def client(self) -> PrefectClient:
        if not self._is_started or self._client is None:
            raise RuntimeError("Engine has not started.")
        return self._client

    async def can_retry(self, exc: Exception) -> bool:
        retry_condition = self.task.retry_condition_fn
        if not self.task_run:
            raise ValueError("Task run is not set")
        try:
            self.logger.debug(f"Running `retry_condition_fn` check {retry_condition!r} for task {self.task.name!r}")
            state: State = Failed(data=exc, message=f"Task run encountered unexpected exception: {repr(exc)}")
            if asyncio.iscoroutinefunction(retry_condition):
                should_retry: bool = await retry_condition(self.task, self.task_run, state)
            elif inspect.isfunction(retry_condition):
                should_retry = retry_condition(self.task, self.task_run, state)
            else:
                should_retry = not retry_condition
            return should_retry
        except Exception:
            self.logger.error(
                f"An error was encountered while running `retry_condition_fn` check '{retry_condition!r}' for task {self.task.name!r}",
                exc_info=True,
            )
            return False

    async def call_hooks(self, state: Optional[State] = None) -> None:
        if state is None:
            state = self.state
        task = self.task
        task_run = self.task_run
        if not task_run:
            raise ValueError("Task run is not set")
        if state.is_failed() and task.on_failure_hooks:
            hooks = task.on_failure_hooks
        elif state.is_completed() and task.on_completion_hooks:
            hooks = task.on_completion_hooks
        else:
            hooks = None
        for hook in hooks or []:
            hook_name: str = get_hook_name(hook)
            try:
                self.logger.info(f"Running hook {hook_name!r} in response to entering state {state.name!r}")
                result = hook(task, task_run, state)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                self.logger.error(f"An error was encountered while running hook {hook_name!r}", exc_info=True)
            else:
                self.logger.info(f"Hook {hook_name!r} finished running successfully")

    async def begin_run(self) -> None:
        try:
            self._resolve_parameters()
            self._set_custom_task_run_name()
            self._wait_for_dependencies()
        except UpstreamTaskError as upstream_exc:
            state: State = await self.set_state(Pending(name="NotReady", message=str(upstream_exc)), force=self.state.is_pending())
            return
        new_state: State = Running()
        self.task_run.start_time = new_state.timestamp
        flow_run_context = FlowRunContext.get()
        if flow_run_context:
            flow_run = flow_run_context.flow_run
            self.task_run.flow_run_run_count = flow_run.run_count
        state: State = await self.set_state(new_state)
        if state.is_completed():
            try:
                await state.result(retry_result_failure=False)
            except Exception:
                state = await self.set_state(new_state, force=True)
        backoff_count: int = 0
        while state.is_pending() or state.is_paused():
            if backoff_count < BACKOFF_MAX:
                backoff_count += 1
            interval: float = clamped_poisson_interval(average_interval=backoff_count, clamping_factor=0.3)
            await anyio.sleep(interval)
            state = await self.set_state(new_state)

    async def set_state(self, state: State, force: bool = False) -> State:
        last_state: State = self.state
        if not self.task_run:
            raise ValueError("Task run is not set")
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
            if isinstance(new_state.data, ResultRecord):
                result = new_state.data.result
            else:
                result = new_state.data
            link_state_to_result(new_state, result)
        self._last_event = emit_task_run_state_change_event(
            task_run=self.task_run, initial_state=last_state, validated_state=self.task_run.state, follows=self._last_event
        )
        self._telemetry.update_state(new_state)
        return new_state

    async def result(self, raise_on_failure: bool = True) -> R:
        if self._return_value is not NotSet:
            if isinstance(self._return_value, ResultRecord):
                return self._return_value.result
            return self._return_value
        if self._raised is not NotSet:
            if raise_on_failure:
                raise self._raised
            return self._raised

    async def handle_success(self, result: R, transaction: Transaction) -> R:
        if self.task.cache_expiration is not None:
            expiration: DateTime = DateTime.now("utc") + self.task.cache_expiration
        else:
            expiration = None
        terminal_state: State = await return_value_to_state(result, result_store=get_result_store(), key=transaction.key, expiration=expiration)
        handle_rollback = partial(self.handle_rollback)
        handle_rollback.log_on_run = False  # type: ignore
        transaction.stage(terminal_state.data, on_rollback_hooks=[handle_rollback] + self.task.on_rollback_hooks, on_commit_hooks=self.task.on_commit_hooks)
        if transaction.is_committed():
            terminal_state.name = "Cached"
        self.record_terminal_state_timing(terminal_state)
        await self.set_state(terminal_state)
        self._return_value = result
        self._telemetry.end_span_on_success()
        return result

    async def handle_retry(self, exc: Exception) -> bool:
        if self.retries < self.task.retries and await self.can_retry(exc):
            if self.task.retry_delay_seconds:
                delay: Optional[Union[int, float]] = (
                    self.task.retry_delay_seconds[min(self.retries, len(self.task.retry_delay_seconds) - 1)]
                    if isinstance(self.task.retry_delay_seconds, Sequence)
                    else self.task.retry_delay_seconds
                )
                new_state: State = AwaitingRetry(scheduled_time=DateTime.now("utc").add(seconds=delay))
            else:
                delay = None
                new_state = Retrying()
            self.logger.info(
                "Task run failed with exception: %r - Retry %s/%s will start %s",
                exc,
                self.retries + 1,
                self.task.retries,
                str(delay) + " second(s) from now" if delay else "immediately",
            )
            await self.set_state(new_state, force=True)
            self.retries = self.retries + 1
            return True
        elif self.retries >= self.task.retries:
            self.logger.error("Task run failed with exception: %r - Retries are exhausted", exc, exc_info=True)
            return False
        return False

    async def handle_exception(self, exc: Exception) -> None:
        self._telemetry.record_exception(exc)
        if not await self.handle_retry(exc):
            state: State = await exception_to_failed_state(exc, message="Task run encountered an exception", result_store=get_result_store())
            self.record_terminal_state_timing(state)
            await self.set_state(state)
            self._raised = exc
            self._telemetry.end_span_on_failure(state.message)

    async def handle_timeout(self, exc: Exception) -> None:
        self._telemetry.record_exception(exc)
        if not await self.handle_retry(exc):
            if isinstance(exc, TaskRunTimeoutError):
                message: str = f"Task run exceeded timeout of {self.task.timeout_seconds} second(s)"
            else:
                message = f"Task run failed due to timeout: {exc!r}"
            self.logger.error(message)
            state: State = Failed(data=exc, message=message, name="TimedOut")
            self.record_terminal_state_timing(state)
            await self.set_state(state)
            self._raised = exc
            self._telemetry.end_span_on_failure(state.message)

    async def handle_crash(self, exc: Exception) -> None:
        state: State = await exception_to_crashed_state(exc)
        self.logger.error(f"Crash detected! {state.message}")
        self.logger.debug("Crash details:", exc_info=exc)
        self.record_terminal_state_timing(state)
        await self.set_state(state, force=True)
        self._raised = exc
        self._telemetry.record_exception(exc)
        self._telemetry.end_span_on_failure(state.message)

    @asynccontextmanager
    async def setup_run_context(self, client: Optional[PrefectClient] = None) -> AsyncGenerator[Self, None]:
        from prefect.utilities.engine import should_log_prints

        settings = get_current_settings()
        if client is None:
            client = self.client
        if not self.task_run:
            raise ValueError("Task run is not set")
        with ExitStack() as stack:
            log_prints: bool = False
            if (log_prints := should_log_prints(self.task)):
                stack.enter_context(patch_print())
            if self.task.persist_result is not None:
                persist_result = self.task.persist_result
            elif settings.tasks.default_persist_result is not None:
                persist_result = settings.tasks.default_persist_result
            else:
                persist_result = should_persist_result()
            stack.enter_context(
                TaskRunContext(
                    task=self.task,
                    log_prints=log_prints,
                    task_run=self.task_run,
                    parameters=self.parameters,
                    result_store=await get_result_store().update_for_task(self.task, _sync=False),
                    client=client,
                    persist_result=persist_result,
                )
            )
            stack.enter_context(ConcurrencyContext())
            self.logger = task_run_logger(task_run=self.task_run, task=self.task)
            yield self

    @asynccontextmanager
    async def initialize_run(self, task_run_id: Optional[Any] = None, dependencies: Optional[Any] = None) -> AsyncGenerator[Self, None]:
        with hydrated_context(self.context):
            async with AsyncClientContext.get_or_create():
                self._client = get_client()
                self._is_started = True
                parent_flow_run_context = FlowRunContext.get()
                parent_task_run_context = TaskRunContext.get()
                try:
                    if not self.task_run:
                        self.task_run = await self.task.create_local_run(
                            id=task_run_id,
                            parameters=self.parameters,
                            flow_run_context=parent_flow_run_context,
                            parent_task_run_context=parent_task_run_context,
                            wait_for=self.wait_for,
                            extra_task_inputs=dependencies,
                        )
                        self._last_event = emit_task_run_state_change_event(task_run=self.task_run, initial_state=None, validated_state=self.task_run.state)
                    async with self.setup_run_context():
                        self.logger.debug(f"Created task run {self.task_run.name!r} for task {self.task.name!r}")
                        await self._telemetry.async_start_span(run=self.task_run, client=self.client, parameters=self.parameters)
                        yield self
                except TerminationSignal as exc:
                    await self.handle_crash(exc)
                    raise exc
                except Exception:
                    raise
                except (Pause, Abort) as exc:
                    if isinstance(exc, Abort):
                        self.logger.error("Task run was aborted: %s", exc)
                    raise
                except GeneratorExit:
                    raise
                except BaseException as exc:
                    await self.handle_crash(exc)
                    raise
                finally:
                    self.log_finished_message()
                    self._is_started = False
                    self._client = None

    async def wait_until_ready(self) -> None:
        """Waits until the scheduled time (if its the future), then enters Running."""
        scheduled_time: Optional[DateTime] = self.state.state_details.scheduled_time
        if scheduled_time:
            sleep_time: float = (scheduled_time - DateTime.now("utc")).total_seconds()
            await anyio.sleep(sleep_time if sleep_time > 0 else 0)
            new_state: State = Retrying() if self.state.name == "AwaitingRetry" else Running()
            await self.set_state(new_state, force=True)

    @asynccontextmanager
    async def start(self, task_run_id: Optional[Any] = None, dependencies: Optional[Any] = None) -> AsyncGenerator[Self, None]:
        async with self.initialize_run(task_run_id=task_run_id, dependencies=dependencies):
            with trace.use_span(self._telemetry.span) if self._telemetry.span else nullcontext():
                await self.begin_run()
                try:
                    yield self
                finally:
                    await self.call_hooks()

    @asynccontextmanager
    async def transaction_context(self) -> AsyncGenerator[Transaction, None]:
        overwrite: bool = self.task.refresh_cache if self.task.refresh_cache is not None else PREFECT_TASKS_REFRESH_CACHE.value()
        isolation_level: Optional[IsolationLevel] = (
            IsolationLevel(self.task.cache_policy.isolation_level)
            if self.task.cache_policy and self.task.cache_policy is not NotSet and (self.task.cache_policy.isolation_level is not None)
            else None
        )
        with transaction(
            key=self.compute_transaction_key(),
            store=get_result_store(),
            overwrite=overwrite,
            logger=self.logger,
            write_on_commit=should_persist_result(),
            isolation_level=isolation_level,
        ) as txn:
            yield txn

    @asynccontextmanager
    async def run_context(self) -> AsyncGenerator[Self, None]:
        async with self.setup_run_context():
            try:
                with timeout_async(seconds=self.task.timeout_seconds, timeout_exc_type=TaskRunTimeoutError):
                    self.logger.debug(f"Executing task {self.task.name!r} for task run {self.task_run.name!r}...")
                    if self.is_cancelled():
                        raise CancelledError("Task run cancelled by the task runner")
                    yield self
            except TimeoutError as exc:
                await self.handle_timeout(exc)
            except Exception as exc:
                await self.handle_exception(exc)

    async def call_task_fn(self, transaction: Transaction) -> R:
        parameters: Dict[str, Any] = self.parameters or {}
        if transaction.is_committed():
            result: R = transaction.read()
        elif self.task_run and self.task_run.tags:
            async with aconcurrency(list(self.task_run.tags), self.task_run.id):
                result = await call_with_parameters(self.task.fn, parameters)
        else:
            result = await call_with_parameters(self.task.fn, parameters)
        await self.handle_success(result, transaction=transaction)
        return result


def run_task_sync(
    task: Any,
    task_run_id: Optional[Any] = None,
    task_run: Optional[Any] = None,
    parameters: Optional[Dict[str, Any]] = None,
    wait_for: Optional[Any] = None,
    return_type: Literal["result", "state"] = "result",
    dependencies: Optional[Any] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Any:
    engine: SyncTaskRunEngine[P, R] = SyncTaskRunEngine(task=task, parameters=parameters, task_run=task_run, wait_for=wait_for, context=context)
    with engine.start(task_run_id=task_run_id, dependencies=dependencies):
        while engine.is_running():
            run_coro_as_sync(engine.wait_until_ready())
            with engine.run_context(), engine.transaction_context() as txn:
                engine.call_task_fn(txn)
    return engine.state if return_type == "state" else engine.result()


async def run_task_async(
    task: Any,
    task_run_id: Optional[Any] = None,
    task_run: Optional[Any] = None,
    parameters: Optional[Dict[str, Any]] = None,
    wait_for: Optional[Any] = None,
    return_type: Literal["result", "state"] = "result",
    dependencies: Optional[Any] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Any:
    engine: AsyncTaskRunEngine[P, R] = AsyncTaskRunEngine(task=task, parameters=parameters, task_run=task_run, wait_for=wait_for, context=context)
    async with engine.start(task_run_id=task_run_id, dependencies=dependencies):
        while engine.is_running():
            await engine.wait_until_ready()
            async with engine.run_context(), engine.transaction_context() as txn:
                await engine.call_task_fn(txn)
    return engine.state if return_type == "state" else await engine.result()


def run_generator_task_sync(
    task: Any,
    task_run_id: Optional[Any] = None,
    task_run: Optional[Any] = None,
    parameters: Optional[Dict[str, Any]] = None,
    wait_for: Optional[Any] = None,
    return_type: Literal["result"] = "result",
    dependencies: Optional[Any] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Any:
    if return_type != "result":
        raise ValueError("The return_type for a generator task must be 'result'")
    engine: SyncTaskRunEngine[P, R] = SyncTaskRunEngine(task=task, parameters=parameters, task_run=task_run, wait_for=wait_for, context=context)
    with engine.start(task_run_id=task_run_id, dependencies=dependencies):
        while engine.is_running():
            run_coro_as_sync(engine.wait_until_ready())
            with engine.run_context(), engine.transaction_context() as txn:
                if False and txn.is_committed():
                    txn.read()
                else:
                    call_args, call_kwargs = parameters_to_args_kwargs(task.fn, engine.parameters or {})
                    gen = task.fn(*call_args, **call_kwargs)
                    try:
                        while True:
                            gen_result = next(gen)
                            link_state_to_result(engine.state, gen_result)
                            yield gen_result
                    except StopIteration as exc:
                        engine.handle_success(exc.value, transaction=txn)
                    except GeneratorExit as exc:
                        engine.handle_success(None, transaction=txn)
                        gen.throw(exc)
    return engine.result()


async def run_generator_task_async(
    task: Any,
    task_run_id: Optional[Any] = None,
    task_run: Optional[Any] = None,
    parameters: Optional[Dict[str, Any]] = None,
    wait_for: Optional[Any] = None,
    return_type: Literal["result"] = "result",
    dependencies: Optional[Any] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Any:
    if return_type != "result":
        raise ValueError("The return_type for a generator task must be 'result'")
    engine: AsyncTaskRunEngine[P, R] = AsyncTaskRunEngine(task=task, parameters=parameters, task_run=task_run, wait_for=wait_for, context=context)
    async with engine.start(task_run_id=task_run_id, dependencies=dependencies):
        while engine.is_running():
            await engine.wait_until_ready()
            async with engine.run_context(), engine.transaction_context() as txn:
                if False and txn.is_committed():
                    txn.read()
                else:
                    call_args, call_kwargs = parameters_to_args_kwargs(task.fn, engine.parameters or {})
                    gen = task.fn(*call_args, **call_kwargs)
                    try:
                        while True:
                            gen_result = await gen.__anext__()
                            link_state_to_result(engine.state, gen_result)
                            yield gen_result
                    except (StopAsyncIteration, GeneratorExit) as exc:
                        await engine.handle_success(None, transaction=txn)
                        if isinstance(exc, GeneratorExit):
                            gen.throw(exc)
    if engine.state.is_failed():
        await engine.result()


def run_task(
    task: Any,
    task_run_id: Optional[Any] = None,
    task_run: Optional[Any] = None,
    parameters: Optional[Dict[str, Any]] = None,
    wait_for: Optional[Any] = None,
    return_type: Literal["result", "state"] = "result",
    dependencies: Optional[Any] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Runs the provided task.

    Args:
        task: The task to run
        task_run_id: The ID of the task run; if not provided, a new task run
            will be created
        task_run: The task run object; if not provided, a new task run
            will be created
        parameters: The parameters to pass to the task
        wait_for: A list of futures to wait for before running the task
        return_type: The return type to return; either "state" or "result"
        dependencies: A dictionary of task run inputs to use for dependency tracking
        context: A dictionary containing the context to use for the task run; only
            required if the task is running on in a remote environment

    Returns:
        The result of the task run
    """
    kwargs: Dict[str, Any] = dict(task=task, task_run_id=task_run_id, task_run=task_run, parameters=parameters, wait_for=wait_for, return_type=return_type, dependencies=dependencies, context=context)
    if task.isasync and task.isgenerator:
        return run_generator_task_async(**kwargs)
    elif task.isgenerator:
        return run_generator_task_sync(**kwargs)
    elif task.isasync:
        return run_task_async(**kwargs)
    else:
        return run_task_sync(**kwargs)