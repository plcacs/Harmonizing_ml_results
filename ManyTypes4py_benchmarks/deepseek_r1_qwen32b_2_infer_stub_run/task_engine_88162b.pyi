from __future__ import annotations
from asyncio import CancelledError
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass
from datetime import datetime as DateTime
from typing import (
    Any,
    Callable,
    ClassVar,
    ContextManager,
    Coroutine,
    Dict,
    Generator,
    Generic,
    Iterable,
    List,
    Literal,
    Optional,
    ParamSpec,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)
from uuid import UUID

from prefect.client.schemas import TaskRun
from prefect.client.schemas.objects import State
from prefect.concurrency.context import ConcurrencyContext
from prefect.logging.loggers import PrefectLogger
from prefect.settings import PREFECT_DEBUG_MODE
from prefect.states import (
    AwaitingRetry,
    Completed,
    Failed,
    Pending,
    Retrying,
    Running,
)
from prefect.types import Duration
from prefect.utilities.asyncutils import AsyncGenerator

P = ParamSpec('P')
R = TypeVar('R')

class BaseTaskRunEngine(Generic[P, R]):
    logger: ClassVar[PrefectLogger]
    parameters: Optional[dict[str, Any]]
    task_run: Optional[TaskRun]
    retries: int
    wait_for: Optional[Iterable[Union[UUID, TaskRunInput]]]
    context: Optional[dict[str, Any]]
    _return_value: Any
    _raised: Any
    _initial_run_context: Any
    _is_started: bool
    _task_name_set: bool
    _last_event: Any
    _telemetry: RunTelemetry

    def __init__(self, **kwargs: Any) -> None: ...

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
    _client: Optional[PrefectClient]

    @property
    def client(self) -> PrefectClient: ...

    def can_retry(self, exc: Exception) -> bool: ...

    def call_hooks(self, state: Optional[State] = None) -> None: ...

    def begin_run(self) -> None: ...

    def set_state(self, state: State, force: bool = False) -> State: ...

    def result(self, raise_on_failure: bool = True) -> R: ...

    def handle_success(self, result: R, transaction: Transaction) -> R: ...

    def handle_retry(self, exc: Exception) -> bool: ...

    def handle_exception(self, exc: Exception) -> None: ...

    def handle_timeout(self, exc: TaskRunTimeoutError) -> None: ...

    def handle_crash(self, exc: Exception) -> None: ...

    @contextmanager
    def setup_run_context(self, client: Optional[PrefectClient] = None) -> Generator[None, None, None]: ...

    @contextmanager
    def initialize_run(self, task_run_id: Optional[UUID] = None, dependencies: Optional[dict[str, Any]] = None) -> Generator[SyncTaskRunEngine[P, R], None, None]: ...

    @contextmanager
    def transaction_context(self) -> Generator[Transaction, None, None]: ...

    @contextmanager
    def run_context(self) -> Generator[None, None, None]: ...

    def call_task_fn(self, transaction: Transaction) -> R: ...

class AsyncTaskRunEngine(BaseTaskRunEngine[P, R]):
    task_run: Optional[TaskRun]
    _client: Optional[PrefectClient]

    @property
    def client(self) -> PrefectClient: ...

    async def can_retry(self, exc: Exception) -> bool: ...

    async def call_hooks(self, state: Optional[State] = None) -> None: ...

    async def begin_run(self) -> None: ...

    async def set_state(self, state: State, force: bool = False) -> State: ...

    async def result(self, raise_on_failure: bool = True) -> R: ...

    async def handle_success(self, result: R, transaction: Transaction) -> R: ...

    async def handle_retry(self, exc: Exception) -> bool: ...

    async def handle_exception(self, exc: Exception) -> None: ...

    async def handle_timeout(self, exc: TaskRunTimeoutError) -> None: ...

    async def handle_crash(self, exc: Exception) -> None: ...

    @asynccontextmanager
    async def setup_run_context(self, client: Optional[PrefectClient] = None) -> AsyncGenerator[None, None]: ...

    @asynccontextmanager
    async def initialize_run(self, task_run_id: Optional[UUID] = None, dependencies: Optional[dict[str, Any]] = None) -> AsyncGenerator[AsyncTaskRunEngine[P, R], None]: ...

    @asynccontextmanager
    async def transaction_context(self) -> AsyncGenerator[Transaction, None]: ...

    @asynccontextmanager
    async def run_context(self) -> AsyncGenerator[None, None]: ...

    async def call_task_fn(self, transaction: Transaction) -> R: ...

def run_task_sync(
    task: Task,
    task_run_id: Optional[UUID] = None,
    task_run: Optional[TaskRun] = None,
    parameters: Optional[dict[str, Any]] = None,
    wait_for: Optional[Iterable[Union[UUID, TaskRunInput]]] = None,
    return_type: Literal['state', 'result'] = 'result',
    dependencies: Optional[dict[str, Any]] = None,
    context: Optional[dict[str, Any]] = None,
) -> Union[State, R]: ...

async def run_task_async(
    task: Task,
    task_run_id: Optional[UUID] = None,
    task_run: Optional[TaskRun] = None,
    parameters: Optional[dict[str, Any]] = None,
    wait_for: Optional[Iterable[Union[UUID, TaskRunInput]]] = None,
    return_type: Literal['state', 'result'] = 'result',
    dependencies: Optional[dict[str, Any]] = None,
    context: Optional[dict[str, Any]] = None,
) -> Union[State, R]: ...

def run_generator_task_sync(
    task: Task,
    task_run_id: Optional[UUID] = None,
    task_run: Optional[TaskRun] = None,
    parameters: Optional[dict[str, Any]] = None,
    wait_for: Optional[Iterable[Union[UUID, TaskRunInput]]] = None,
    return_type: Literal['result'] = 'result',
    dependencies: Optional[dict[str, Any]] = None,
    context: Optional[dict[str, Any]] = None,
) -> Generator[Any, None, R]: ...

async def run_generator_task_async(
    task: Task,
    task_run_id: Optional[UUID] = None,
    task_run: Optional[TaskRun] = None,
    parameters: Optional[dict[str, Any]] = None,
    wait_for: Optional[Iterable[Union[UUID, TaskRunInput]]] = None,
    return_type: Literal['result'] = 'result',
    dependencies: Optional[dict[str, Any]] = None,
    context: Optional[dict[str, Any]] = None,
) -> AsyncGenerator[Any, None]: ...

def run_task(
    task: Task,
    task_run_id: Optional[UUID] = None,
    task_run: Optional[TaskRun] = None,
    parameters: Optional[dict[str, Any]] = None,
    wait_for: Optional[Iterable[Union[UUID, TaskRunInput]]] = None,
    return_type: Literal['state', 'result'] = 'result',
    dependencies: Optional[dict[str, Any]] = None,
    context: Optional[dict[str, Any]] = None,
) -> Union[State, R]: ...