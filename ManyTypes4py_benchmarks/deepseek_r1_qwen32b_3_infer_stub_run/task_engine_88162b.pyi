from __future__ import annotations
from asyncio import CancelledError
from contextlib import ExitStack, asynccontextmanager, contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Coroutine,
    Generator,
    Generic,
    Iterable,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    Dict,
    List,
    Tuple,
    Any,
    Optional,
    Union,
    Callable,
    Coroutine,
    Generator,
    Generic,
    TypeVar,
    Dict,
    Any,
    Optional,
    Union,
    Callable,
    Coroutine,
    Generator,
    Generic,
    TypeVar,
    Dict,
    Any,
    Optional,
    Union,
    Callable,
    Coroutine,
    Generator,
    Generic,
    TypeVar,
    Dict,
)
from prefect.client.schemas import TaskRun
from prefect.client.schemas.objects import State, TaskRunInput
from prefect.concurrency.context import ConcurrencyContext
from prefect.concurrency.v1.asyncio import concurrency as aconcurrency
from prefect.concurrency.v1.context import ConcurrencyContext as ConcurrencyContextV1
from prefect.concurrency.v1.sync import concurrency
from prefect.context import (
    AsyncClientContext,
    FlowRunContext,
    SyncClientContext,
    TaskRunContext,
    hydrated_context,
)
from prefect.events.schemas.events import Event as PrefectEvent
from prefect.exceptions import Abort, Pause, PrefectException, TerminationSignal, UpstreamTaskError
from prefect.logging.loggers import get_logger, patch_print, task_run_logger
from prefect.results import ResultRecord
from prefect.settings import PREFECT_DEBUG_MODE, PREFECT_TASKS_REFRESH_CACHE
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
from prefect.transactions import IsolationLevel, Transaction
from prefect.types._datetime import DateTime, Duration
from prefect.utilities.asyncutils import run_coro_as_sync
from prefect.utilities.callables import call_with_parameters, parameters_to_args_kwargs
from prefect.utilities.collections import visit_collection
from prefect.utilities.engine import emit_task_run_state_change_event, link_state_to_result, resolve_to_final_result
from prefect.utilities.math import clamped_poisson_interval
from prefect.utilities.timeout import timeout, timeout_async
import anyio
import logging
import threading
import time
from uuid import UUID

P = ParamSpec('P')
R = TypeVar('R')

class BaseTaskRunEngine(Generic[P, R]):
    logger: Any
    parameters: Dict[str, Any]
    task_run: Optional[TaskRun]
    retries: int
    wait_for: Optional[Iterable[Any]]
    context: Optional[Dict[str, Any]]
    _return_value: Any
    _raised: Any
    _initial_run_context: Any
    _is_started: bool
    _task_name_set: bool
    _last_event: Any
    _telemetry: Any

    def __post_init__(self) -> None: ...

    @property
    def state(self) -> State: ...

    def is_cancelled(self) -> bool: ...

    def compute_transaction_key(self) -> Optional[str]: ...

    def _resolve_parameters(self) -> None: ...

    def _set_custom_task_run_name(self) -> None: ...

    def _wait_for_dependencies(self) -> None: ...

    def record_terminal_state_timing(self, state: State) -> None: ...

    def is_running(self) -> bool: ...

    def log_finished_message(self) -> None: ...

    def handle_rollback(self, txn: Transaction) -> None: ...

class SyncTaskRunEngine(BaseTaskRunEngine[P, R]):
    task_run: Optional[TaskRun]
    _client: Optional[SyncPrefectClient]

    @property
    def client(self) -> SyncPrefectClient: ...

    def can_retry(self, exc: Exception) -> bool: ...

    def call_hooks(self, state: Optional[State] = None) -> None: ...

    def begin_run(self) -> None: ...

    def set_state(self, state: State, force: bool = False) -> State: ...

    def result(self, raise_on_failure: bool = True) -> R: ...

    def handle_success(self, result: R, transaction: Transaction) -> R: ...

    def handle_retry(self, exc: Exception) -> bool: ...

    def handle_exception(self, exc: Exception) -> None: ...

    def handle_timeout(self, exc: Exception) -> None: ...

    def handle_crash(self, exc: Exception) -> None: ...

    @contextmanager
    def setup_run_context(self, client: Optional[SyncPrefectClient] = None) -> Generator: ...

    @contextmanager
    def initialize_run(self, task_run_id: Optional[UUID] = None, dependencies: Optional[Dict[str, Any]] = None) -> Generator: ...

    def start(self, task_run_id: Optional[UUID] = None, dependencies: Optional[Dict[str, Any]] = None) -> Generator: ...

    @contextmanager
    def transaction_context(self) -> Generator: ...

    @contextmanager
    def run_context(self) -> Generator: ...

    def call_task_fn(self, transaction: Transaction) -> R: ...

class AsyncTaskRunEngine(BaseTaskRunEngine[P, R]):
    task_run: Optional[TaskRun]
    _client: Optional[AsyncPrefectClient]

    @property
    def client(self) -> AsyncPrefectClient: ...

    async def can_retry(self, exc: Exception) -> bool: ...

    async def call_hooks(self, state: Optional[State] = None) -> None: ...

    async def begin_run(self) -> None: ...

    async def set_state(self, state: State, force: bool = False) -> State: ...

    async def result(self, raise_on_failure: bool = True) -> R: ...

    async def handle_success(self, result: R, transaction: Transaction) -> R: ...

    async def handle_retry(self, exc: Exception) -> bool: ...

    async def handle_exception(self, exc: Exception) -> None: ...

    async def handle_timeout(self, exc: Exception) -> None: ...

    async def handle_crash(self, exc: Exception) -> None: ...

    @asynccontextmanager
    async def setup_run_context(self, client: Optional[AsyncPrefectClient] = None) -> Generator: ...

    @asynccontextmanager
    async def initialize_run(self, task_run_id: Optional[UUID] = None, dependencies: Optional[Dict[str, Any]] = None) -> Generator: ...

    @asynccontextmanager
    async def start(self, task_run_id: Optional[UUID] = None, dependencies: Optional[Dict[str, Any]] = None) -> Generator: ...

    @asynccontextmanager
    async def transaction_context(self) -> Generator: ...

    @asynccontextmanager
    async def run_context(self) -> Generator: ...

    async def call_task_fn(self, transaction: Transaction) -> R: ...

def run_task_sync(
    task: Any,
    task_run_id: Optional[UUID] = None,
    task_run: Optional[TaskRun] = None,
    parameters: Optional[Dict[str, Any]] = None,
    wait_for: Optional[Iterable[Any]] = None,
    return_type: Literal['state', 'result'] = 'result',
    dependencies: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Union[State, R]: ...

async def run_task_async(
    task: Any,
    task_run_id: Optional[UUID] = None,
    task_run: Optional[TaskRun] = None,
    parameters: Optional[Dict[str, Any]] = None,
    wait_for: Optional[Iterable[Any]] = None,
    return_type: Literal['state', 'result'] = 'result',
    dependencies: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Union[State, R]: ...

def run_generator_task_sync(
    task: Any,
    task_run_id: Optional[UUID] = None,
    task_run: Optional[TaskRun] = None,
    parameters: Optional[Dict[str, Any]] = None,
    wait_for: Optional[Iterable[Any]] = None,
    return_type: Literal['state', 'result'] = 'result',
    dependencies: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Generator[Any, None, R]: ...

async def run_generator_task_async(
    task: Any,
    task_run_id: Optional[UUID] = None,
    task_run: Optional[TaskRun] = None,
    parameters: Optional[Dict[str, Any]] = None,
    wait_for: Optional[Iterable[Any]] = None,
    return_type: Literal['state', 'result'] = 'result',
    dependencies: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Generator[Any, None, R]: ...

def run_task(
    task: Any,
    task_run_id: Optional[UUID] = None,
    task_run: Optional[TaskRun] = None,
    parameters: Optional[Dict[str, Any]] = None,
    wait_for: Optional[Iterable[Any]] = None,
    return_type: Literal['state', 'result'] = 'result',
    dependencies: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Union[State, R]: ...