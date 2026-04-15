from __future__ import annotations
import asyncio
import threading
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from dataclasses import dataclass, field
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Dict,
    Generator,
    Generic,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)
from uuid import UUID
from opentelemetry.trace import Span
from typing_extensions import ParamSpec, Self
from prefect.cache_policies import CachePolicy
from prefect.client.orchestration import PrefectClient, SyncPrefectClient
from prefect.client.schemas import TaskRun
from prefect.client.schemas.objects import State
from prefect.concurrency.context import ConcurrencyContext
from prefect.concurrency.v1.context import ConcurrencyContext as ConcurrencyContextV1
from prefect.context import AsyncClientContext, FlowRunContext, SyncClientContext, TaskRunContext
from prefect.events.schemas.events import Event as PrefectEvent
from prefect.exceptions import Abort, Pause, PrefectException, TerminationSignal, UpstreamTaskError
from prefect.logging.loggers import PrefectLogAdapter
from prefect.results import ResultRecord, ResultStore
from prefect.settings import Settings
from prefect.states import (
    AwaitingRetry,
    Completed,
    Failed,
    Pending,
    Retrying,
    Running,
    State as PrefectState,
)
from prefect.telemetry.run_telemetry import RunTelemetry
from prefect.transactions import IsolationLevel, Transaction
from prefect.types._datetime import DateTime, Duration
from prefect.utilities.annotations import NotSet
from prefect.utilities.callables import Parameters

if TYPE_CHECKING:
    from prefect.tasks import OneOrManyFutureOrResult, Task

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")
BACKOFF_MAX: int = 10

class TaskRunTimeoutError(TimeoutError): ...

@dataclass
class BaseTaskRunEngine(Generic[P, R]):
    logger: PrefectLogAdapter = field(default_factory=lambda: ...)
    parameters: Optional[Dict[str, Any]] = None
    task_run: Optional[TaskRun] = None
    retries: int = 0
    wait_for: Optional[Any] = None
    context: Optional[Dict[str, Any]] = None
    _return_value: Union[Any, NotSet] = NotSet
    _raised: Union[BaseException, NotSet] = NotSet
    _initial_run_context: Optional[Any] = None
    _is_started: bool = False
    _task_name_set: bool = False
    _last_event: Optional[PrefectEvent] = None
    _telemetry: RunTelemetry = field(default_factory=lambda: ...)
    
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

@dataclass
class SyncTaskRunEngine(BaseTaskRunEngine[P, R]):
    task_run: Optional[TaskRun] = None
    _client: Optional[SyncPrefectClient] = None
    
    @property
    def client(self) -> SyncPrefectClient: ...
    
    def can_retry(self, exc: BaseException) -> bool: ...
    
    def call_hooks(self, state: Optional[State] = None) -> None: ...
    
    def begin_run(self) -> None: ...
    
    def set_state(self, state: State, force: bool = False) -> State: ...
    
    def result(self, raise_on_failure: bool = True) -> Union[R, BaseException]: ...
    
    def handle_success(self, result: R, transaction: Transaction) -> R: ...
    
    def handle_retry(self, exc: BaseException) -> bool: ...
    
    def handle_exception(self, exc: BaseException) -> None: ...
    
    def handle_timeout(self, exc: TimeoutError) -> None: ...
    
    def handle_crash(self, exc: BaseException) -> None: ...
    
    @contextmanager
    def setup_run_context(self, client: Optional[SyncPrefectClient] = None) -> Generator[None, None, None]: ...
    
    @contextmanager
    def initialize_run(self, task_run_id: Optional[UUID] = None, dependencies: Optional[Dict[str, Any]] = None) -> Generator[Self, None, None]: ...
    
    async def wait_until_ready(self) -> None: ...
    
    @contextmanager
    def start(self, task_run_id: Optional[UUID] = None, dependencies: Optional[Dict[str, Any]] = None) -> Generator[None, None, None]: ...
    
    @contextmanager
    def transaction_context(self) -> Generator[Transaction, None, None]: ...
    
    @contextmanager
    def run_context(self) -> Generator[Self, None, None]: ...
    
    def call_task_fn(self, transaction: Transaction) -> R: ...

@dataclass
class AsyncTaskRunEngine(BaseTaskRunEngine[P, R]):
    task_run: Optional[TaskRun] = None
    _client: Optional[PrefectClient] = None
    
    @property
    def client(self) -> PrefectClient: ...
    
    async def can_retry(self, exc: BaseException) -> bool: ...
    
    async def call_hooks(self, state: Optional[State] = None) -> None: ...
    
    async def begin_run(self) -> None: ...
    
    async def set_state(self, state: State, force: bool = False) -> State: ...
    
    async def result(self, raise_on_failure: bool = True) -> Union[R, BaseException]: ...
    
    async def handle_success(self, result: R, transaction: Transaction) -> R: ...
    
    async def handle_retry(self, exc: BaseException) -> bool: ...
    
    async def handle_exception(self, exc: BaseException) -> None: ...
    
    async def handle_timeout(self, exc: TimeoutError) -> None: ...
    
    async def handle_crash(self, exc: BaseException) -> None: ...
    
    @asynccontextmanager
    async def setup_run_context(self, client: Optional[PrefectClient] = None) -> AsyncGenerator[None, None]: ...
    
    @asynccontextmanager
    async def initialize_run(self, task_run_id: Optional[UUID] = None, dependencies: Optional[Dict[str, Any]] = None) -> AsyncGenerator[Self, None]: ...
    
    async def wait_until_ready(self) -> None: ...
    
    @asynccontextmanager
    async def start(self, task_run_id: Optional[UUID] = None, dependencies: Optional[Dict[str, Any]] = None) -> AsyncGenerator[None, None]: ...
    
    @asynccontextmanager
    async def transaction_context(self) -> AsyncGenerator[Transaction, None]: ...
    
    @asynccontextmanager
    async def run_context(self) -> AsyncGenerator[Self, None]: ...
    
    async def call_task_fn(self, transaction: Transaction) -> R: ...

def run_task_sync(
    task: Task,
    task_run_id: Optional[UUID] = None,
    task_run: Optional[TaskRun] = None,
    parameters: Optional[Dict[str, Any]] = None,
    wait_for: Optional[Any] = None,
    return_type: Literal["state", "result"] = "result",
    dependencies: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Union[State, R]: ...

async def run_task_async(
    task: Task,
    task_run_id: Optional[UUID] = None,
    task_run: Optional[TaskRun] = None,
    parameters: Optional[Dict[str, Any]] = None,
    wait_for: Optional[Any] = None,
    return_type: Literal["state", "result"] = "result",
    dependencies: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Union[State, R]: ...

def run_generator_task_sync(
    task: Task,
    task_run_id: Optional[UUID] = None,
    task_run: Optional[TaskRun] = None,
    parameters: Optional[Dict[str, Any]] = None,
    wait_for: Optional[Any] = None,
    return_type: Literal["result"] = "result",
    dependencies: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Generator[R, None, None]: ...

async def run_generator_task_async(
    task: Task,
    task_run_id: Optional[UUID] = None,
    task_run: Optional[TaskRun] = None,
    parameters: Optional[Dict[str, Any]] = None,
    wait_for: Optional[Any] = None,
    return_type: Literal["result"] = "result",
    dependencies: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> AsyncGenerator[R, None]: ...

@overload
def run_task(
    task: Task,
    task_run_id: Optional[UUID] = None,
    task_run: Optional[TaskRun] = None,
    parameters: Optional[Dict[str, Any]] = None,
    wait_for: Optional[Any] = None,
    return_type: Literal["state"] = ...,
    dependencies: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> State: ...

@overload
def run_task(
    task: Task,
    task_run_id: Optional[UUID] = None,
    task_run: Optional[TaskRun] = None,
    parameters: Optional[Dict[str, Any]] = None,
    wait_for: Optional[Any] = None,
    return_type: Literal["result"] = ...,
    dependencies: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Union[R, Generator[R, None, None], AsyncGenerator[R, None]]: ...

def run_task(
    task: Task,
    task_run_id: Optional[UUID] = None,
    task_run: Optional[TaskRun] = None,
    parameters: Optional[Dict[str, Any]] = None,
    wait_for: Optional[Any] = None,
    return_type: Literal["state", "result"] = "result",
    dependencies: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Union[State, R, Generator[R, None, None], AsyncGenerator[R, None]]: ...