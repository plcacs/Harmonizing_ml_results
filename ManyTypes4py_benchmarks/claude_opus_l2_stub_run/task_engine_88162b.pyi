from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Generator,
    Generic,
    Literal,
    Optional,
    Sequence,
    Union,
)
from uuid import UUID

from typing_extensions import ParamSpec, Self

from prefect.cache_policies import CachePolicy
from prefect.client.orchestration import PrefectClient, SyncPrefectClient
from prefect.client.schemas import TaskRun
from prefect.client.schemas.objects import State, TaskRunInput
from prefect.events.schemas.events import Event as PrefectEvent
from prefect.results import ResultRecord
from prefect.telemetry.run_telemetry import RunTelemetry
from prefect.transactions import Transaction
from prefect.utilities.annotations import NotSet

if TYPE_CHECKING:
    from prefect.tasks import OneOrManyFutureOrResult, Task

P = ParamSpec("P")
R = TypeVar("R")

# Need TypeVar import for module-level usage
from typing import TypeVar

BACKOFF_MAX: int = ...


class TaskRunTimeoutError(TimeoutError):
    """Raised when a task run exceeds its timeout."""
    ...


@dataclass
class BaseTaskRunEngine(Generic[P, R]):
    task: Task[P, R]
    logger: logging.Logger = ...
    parameters: Optional[Dict[str, Any]] = ...
    task_run: Optional[TaskRun] = ...
    retries: int = ...
    wait_for: Optional[Any] = ...
    context: Optional[Dict[str, Any]] = ...
    _return_value: Any = ...
    _raised: Any = ...
    _initial_run_context: Optional[Any] = ...
    _is_started: bool = ...
    _task_name_set: bool = ...
    _last_event: Optional[PrefectEvent] = ...
    _telemetry: RunTelemetry = ...

    def __post_init__(self) -> None: ...

    @property
    def state(self) -> State[Any]: ...

    def is_cancelled(self) -> bool: ...
    def compute_transaction_key(self) -> Optional[str]: ...
    def _resolve_parameters(self) -> None: ...
    def _set_custom_task_run_name(self) -> None: ...
    def _wait_for_dependencies(self) -> None: ...
    def record_terminal_state_timing(self, state: State[Any]) -> None: ...
    def is_running(self) -> bool: ...
    def log_finished_message(self) -> None: ...
    def handle_rollback(self, txn: Transaction) -> None: ...


@dataclass
class SyncTaskRunEngine(BaseTaskRunEngine[P, R]):
    task: Task[P, R]
    task_run: Optional[TaskRun] = ...
    _client: Optional[SyncPrefectClient] = ...

    @property
    def client(self) -> SyncPrefectClient: ...

    def can_retry(self, exc: Exception) -> bool: ...
    def call_hooks(self, state: Optional[State[Any]] = ...) -> None: ...
    def begin_run(self) -> None: ...
    def set_state(self, state: State[Any], force: bool = ...) -> State[Any]: ...
    def result(self, raise_on_failure: bool = ...) -> Any: ...
    def handle_success(self, result: Any, transaction: Transaction) -> Any: ...
    def handle_retry(self, exc: Exception) -> bool: ...
    def handle_exception(self, exc: Exception) -> None: ...
    def handle_timeout(self, exc: TimeoutError) -> None: ...
    def handle_crash(self, exc: BaseException) -> None: ...

    @contextmanager
    def setup_run_context(self, client: Optional[SyncPrefectClient] = ...) -> Generator[None, None, None]: ...

    @contextmanager
    def initialize_run(
        self,
        task_run_id: Optional[UUID] = ...,
        dependencies: Optional[Dict[str, Any]] = ...,
    ) -> Generator[Self, None, None]: ...

    async def wait_until_ready(self) -> None: ...

    @contextmanager
    def start(
        self,
        task_run_id: Optional[UUID] = ...,
        dependencies: Optional[Dict[str, Any]] = ...,
    ) -> Generator[None, None, None]: ...

    @contextmanager
    def transaction_context(self) -> Generator[Transaction, None, None]: ...

    @contextmanager
    def run_context(self) -> Generator[Self, None, None]: ...

    def call_task_fn(self, transaction: Transaction) -> Any: ...


@dataclass
class AsyncTaskRunEngine(BaseTaskRunEngine[P, R]):
    task: Task[P, R]
    task_run: Optional[TaskRun] = ...
    _client: Optional[PrefectClient] = ...

    @property
    def client(self) -> PrefectClient: ...

    async def can_retry(self, exc: Exception) -> bool: ...
    async def call_hooks(self, state: Optional[State[Any]] = ...) -> None: ...
    async def begin_run(self) -> None: ...
    async def set_state(self, state: State[Any], force: bool = ...) -> State[Any]: ...
    async def result(self, raise_on_failure: bool = ...) -> Any: ...
    async def handle_success(self, result: Any, transaction: Transaction) -> Any: ...
    async def handle_retry(self, exc: Exception) -> bool: ...
    async def handle_exception(self, exc: Exception) -> None: ...
    async def handle_timeout(self, exc: TimeoutError) -> None: ...
    async def handle_crash(self, exc: BaseException) -> None: ...

    @asynccontextmanager
    async def setup_run_context(self, client: Optional[PrefectClient] = ...) -> AsyncGenerator[None, None]: ...

    @asynccontextmanager
    async def initialize_run(
        self,
        task_run_id: Optional[UUID] = ...,
        dependencies: Optional[Dict[str, Any]] = ...,
    ) -> AsyncGenerator[Self, None]: ...

    async def wait_until_ready(self) -> None: ...

    @asynccontextmanager
    async def start(
        self,
        task_run_id: Optional[UUID] = ...,
        dependencies: Optional[Dict[str, Any]] = ...,
    ) -> AsyncGenerator[None, None]: ...

    @asynccontextmanager
    async def transaction_context(self) -> AsyncGenerator[Transaction, None]: ...

    @asynccontextmanager
    async def run_context(self) -> AsyncGenerator[Self, None]: ...

    async def call_task_fn(self, transaction: Transaction) -> Any: ...


def run_task_sync(
    task: Task[P, R],
    task_run_id: Optional[UUID] = ...,
    task_run: Optional[TaskRun] = ...,
    parameters: Optional[Dict[str, Any]] = ...,
    wait_for: Optional[Any] = ...,
    return_type: str = ...,
    dependencies: Optional[Dict[str, Any]] = ...,
    context: Optional[Dict[str, Any]] = ...,
) -> Any: ...


async def run_task_async(
    task: Task[P, R],
    task_run_id: Optional[UUID] = ...,
    task_run: Optional[TaskRun] = ...,
    parameters: Optional[Dict[str, Any]] = ...,
    wait_for: Optional[Any] = ...,
    return_type: str = ...,
    dependencies: Optional[Dict[str, Any]] = ...,
    context: Optional[Dict[str, Any]] = ...,
) -> Any: ...


def run_generator_task_sync(
    task: Task[P, R],
    task_run_id: Optional[UUID] = ...,
    task_run: Optional[TaskRun] = ...,
    parameters: Optional[Dict[str, Any]] = ...,
    wait_for: Optional[Any] = ...,
    return_type: str = ...,
    dependencies: Optional[Dict[str, Any]] = ...,
    context: Optional[Dict[str, Any]] = ...,
) -> Generator[Any, None, Any]: ...


async def run_generator_task_async(
    task: Task[P, R],
    task_run_id: Optional[UUID] = ...,
    task_run: Optional[TaskRun] = ...,
    parameters: Optional[Dict[str, Any]] = ...,
    wait_for: Optional[Any] = ...,
    return_type: str = ...,
    dependencies: Optional[Dict[str, Any]] = ...,
    context: Optional[Dict[str, Any]] = ...,
) -> AsyncGenerator[Any, None]: ...


def run_task(
    task: Task[P, R],
    task_run_id: Optional[UUID] = ...,
    task_run: Optional[TaskRun] = ...,
    parameters: Optional[Dict[str, Any]] = ...,
    wait_for: Optional[Any] = ...,
    return_type: str = ...,
    dependencies: Optional[Dict[str, Any]] = ...,
    context: Optional[Dict[str, Any]] = ...,
) -> Any: ...