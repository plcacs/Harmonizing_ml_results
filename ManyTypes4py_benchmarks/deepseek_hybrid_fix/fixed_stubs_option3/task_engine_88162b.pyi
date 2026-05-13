from __future__ import annotations
import asyncio
import inspect
import logging
import threading
import time
from asyncio import CancelledError
from contextlib import AbstractAsyncContextManager, AbstractContextManager, ExitStack, asynccontextmanager, contextmanager, nullcontext
from dataclasses import dataclass, field
from functools import partial
from textwrap import dedent
from typing import Any, AsyncGenerator, AsyncIterator, Awaitable, Callable, Coroutine, Dict, Generator, Generic, Iterator, List, Literal, Optional, Protocol, Sequence, Tuple, Type, TypeVar, Union, overload
from uuid import UUID

import anyio
from opentelemetry import trace
from opentelemetry.trace import Span
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
from prefect.logging.loggers import PrefectLogAdapter, get_logger, patch_print, task_run_logger
from prefect.results import ResultRecord, ResultStore, _format_user_supplied_storage_key, get_result_store, should_persist_result
from prefect.settings import PREFECT_DEBUG_MODE, PREFECT_TASKS_REFRESH_CACHE, Settings
from prefect.settings.context import get_current_settings
from prefect.states import AwaitingRetry, Completed, Failed, Pending, Retrying, Running, State as PrefectState, exception_to_crashed_state, exception_to_failed_state, return_value_to_state
from prefect.telemetry.run_telemetry import RunTelemetry
from prefect.transactions import IsolationLevel, Transaction, transaction
from prefect.types._datetime import DateTime, Duration
from prefect.utilities._engine import get_hook_name
from prefect.utilities.annotations import NotSet
from prefect.utilities.asyncutils import run_coro_as_sync
from prefect.utilities.callables import Parameters, call_with_parameters, parameters_to_args_kwargs
from prefect.utilities.collections import visit_collection
from prefect.utilities.engine import emit_task_run_state_change_event, link_state_to_result, resolve_to_final_result
from prefect.utilities.math import clamped_poisson_interval
from prefect.utilities.timeout import timeout, timeout_async

P = ParamSpec("P")
R = TypeVar("R")
BACKOFF_MAX: int = 10


class TaskRunTimeoutError(TimeoutError):
    ...


@dataclass
class BaseTaskRunEngine(Generic[P, R]):
    task: Any
    logger: PrefectLogAdapter = field(default_factory=lambda: get_logger("engine"))
    parameters: Dict[str, Any] | None = None
    task_run: TaskRun | None = None
    retries: int = 0
    wait_for: Any = None
    context: Dict[str, Any] | None = None
    _return_value: Any = NotSet
    _raised: Any = NotSet
    _initial_run_context: Any = None
    _is_started: bool = False
    _task_name_set: bool = False
    _last_event: PrefectEvent | None = None
    _telemetry: RunTelemetry = field(default_factory=RunTelemetry)

    def __post_init__(self) -> None: ...

    @property
    def state(self) -> State: ...

    def is_cancelled(self) -> bool: ...

    def compute_transaction_key(self) -> str | None: ...

    def _resolve_parameters(self) -> None: ...

    def _set_custom_task_run_name(self) -> None: ...

    def _wait_for_dependencies(self) -> None: ...

    def record_terminal_state_timing(self, state: State) -> None: ...

    def is_running(self) -> bool: ...

    def log_finished_message(self) -> None: ...

    def handle_rollback(self, txn: Transaction) -> None: ...


@dataclass
class SyncTaskRunEngine(BaseTaskRunEngine[P, R]):
    task_run: TaskRun | None = None
    _client: SyncPrefectClient | None = None

    @property
    def client(self) -> SyncPrefectClient: ...

    def can_retry(self, exc: BaseException) -> bool: ...

    def call_hooks(self, state: State | None = None) -> None: ...

    def begin_run(self) -> None: ...

    def set_state(self, state: State, force: bool = False) -> State: ...

    def result(self, raise_on_failure: bool = True) -> R | BaseException: ...

    def handle_success(self, result: Any, transaction: Transaction) -> Any: ...

    def handle_retry(self, exc: BaseException) -> bool: ...

    def handle_exception(self, exc: BaseException) -> None: ...

    def handle_timeout(self, exc: TimeoutError) -> None: ...

    def handle_crash(self, exc: BaseException) -> None: ...

    @contextmanager
    def setup_run_context(self, client: SyncPrefectClient | None = None) -> Generator[None, None, None]: ...

    @contextmanager
    def initialize_run(self, task_run_id: UUID | None = None, dependencies: Dict[str, Any] | None = None) -> Generator[Self, None, None]: ...

    async def wait_until_ready(self) -> None: ...

    @contextmanager
    def start(self, task_run_id: UUID | None = None, dependencies: Dict[str, Any] | None = None) -> Generator[None, None, None]: ...

    @contextmanager
    def transaction_context(self) -> Generator[Transaction, None, None]: ...

    @contextmanager
    def run_context(self) -> Generator[Self, None, None]: ...

    def call_task_fn(self, transaction: Transaction) -> Any: ...


@dataclass
class AsyncTaskRunEngine(BaseTaskRunEngine[P, R]):
    task_run: TaskRun | None = None
    _client: PrefectClient | None = None

    @property
    def client(self) -> PrefectClient: ...

    async def can_retry(self, exc: BaseException) -> bool: ...

    async def call_hooks(self, state: State | None = None) -> None: ...

    async def begin_run(self) -> None: ...

    async def set_state(self, state: State, force: bool = False) -> State: ...

    async def result(self, raise_on_failure: bool = True) -> R | BaseException: ...

    async def handle_success(self, result: Any, transaction: Transaction) -> Any: ...

    async def handle_retry(self, exc: BaseException) -> bool: ...

    async def handle_exception(self, exc: BaseException) -> None: ...

    async def handle_timeout(self, exc: TimeoutError) -> None: ...

    async def handle_crash(self, exc: BaseException) -> None: ...

    @asynccontextmanager
    async def setup_run_context(self, client: PrefectClient | None = None) -> AsyncGenerator[None, None]: ...

    @asynccontextmanager
    async def initialize_run(self, task_run_id: UUID | None = None, dependencies: Dict[str, Any] | None = None) -> AsyncGenerator[Self, None]: ...

    async def wait_until_ready(self) -> None: ...

    @asynccontextmanager
    async def start(self, task_run_id: UUID | None = None, dependencies: Dict[str, Any] | None = None) -> AsyncGenerator[None, None]: ...

    @asynccontextmanager
    async def transaction_context(self) -> AsyncGenerator[Transaction, None]: ...

    @asynccontextmanager
    async def run_context(self) -> AsyncGenerator[Self, None]: ...

    async def call_task_fn(self, transaction: Transaction) -> Any: ...


def run_task_sync(
    task: Any,
    task_run_id: UUID | None = None,
    task_run: TaskRun | None = None,
    parameters: Dict[str, Any] | None = None,
    wait_for: Any = None,
    return_type: Literal["state", "result"] = "result",
    dependencies: Dict[str, Any] | None = None,
    context: Dict[str, Any] | None = None,
) -> State | R: ...


async def run_task_async(
    task: Any,
    task_run_id: UUID | None = None,
    task_run: TaskRun | None = None,
    parameters: Dict[str, Any] | None = None,
    wait_for: Any = None,
    return_type: Literal["state", "result"] = "result",
    dependencies: Dict[str, Any] | None = None,
    context: Dict[str, Any] | None = None,
) -> State | R: ...


def run_generator_task_sync(
    task: Any,
    task_run_id: UUID | None = None,
    task_run: TaskRun | None = None,
    parameters: Dict[str, Any] | None = None,
    wait_for: Any = None,
    return_type: Literal["result"] = "result",
    dependencies: Dict[str, Any] | None = None,
    context: Dict[str, Any] | None = None,
) -> Generator[R, None, None]: ...


async def run_generator_task_async(
    task: Any,
    task_run_id: UUID | None = None,
    task_run: TaskRun | None = None,
    parameters: Dict[str, Any] | None = None,
    wait_for: Any = None,
    return_type: Literal["result"] = "result",
    dependencies: Dict[str, Any] | None = None,
    context: Dict[str, Any] | None = None,
) -> AsyncGenerator[R, None]: ...


def run_task(
    task: Any,
    task_run_id: UUID | None = None,
    task_run: TaskRun | None = None,
    parameters: Dict[str, Any] | None = None,
    wait_for: Any = None,
    return_type: Literal["state", "result"] = "result",
    dependencies: Dict[str, Any] | None = None,
    context: Dict[str, Any] | None = None,
) -> State | R | Generator[R, None, None] | AsyncGenerator[R, None]: ...