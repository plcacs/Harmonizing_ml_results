from __future__ import annotations
import abc
import asyncio
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from typing import TYPE_CHECKING, Any, Coroutine, Dict, Generic, Iterable, List, Optional, overload
from typing_extensions import ParamSpec, Self, TypeVar
from prefect.client.schemas.objects import TaskRunInput
from prefect.exceptions import MappingLengthMismatch, MappingMissingIterable
from prefect.futures import PrefectConcurrentFuture, PrefectDistributedFuture, PrefectFuture, PrefectFutureList
from prefect.logging.loggers import get_logger, get_run_logger
from prefect.settings import PREFECT_TASK_RUNNER_THREAD_POOL_MAX_WORKERS
from prefect.utilities.annotations import allow_failure, quote, unmapped
from prefect.utilities.callables import collapse_variadic_parameters, explode_variadic_parameter, get_parameter_defaults
from prefect.utilities.collections import isiterable
if TYPE_CHECKING:
    import logging
    from prefect.tasks import Task
P = ParamSpec('P')
T = TypeVar('T')
R = TypeVar('R')
F = TypeVar('F', bound=PrefectFuture[Any], default=PrefectConcurrentFuture[Any])

class TaskRunner(abc.ABC, Generic[F]):
    def __init__(self) -> None:
        self.logger: logging.Logger = get_logger(f'task_runner.{self.name}')
        self._started: bool = False

    @property
    def name(self) -> str:
        return type(self).__name__.lower().replace('taskrunner', '')

    @abc.abstractmethod
    def duplicate(self) -> TaskRunner[F]:
        ...

    @overload
    @abc.abstractmethod
    def submit(self, task: Task, parameters: Dict[str, Any], wait_for: Optional[List[F]] = None, dependencies: Optional[Dict[str, Any]] = None) -> F:
        ...

    @abc.abstractmethod
    def submit(self, task: Task, parameters: Dict[str, Any], wait_for: Optional[List[F]] = None, dependencies: Optional[Dict[str, Any]] = None) -> F:
        ...

    def map(self, task: Task, parameters: Dict[str, Any], wait_for: Optional[List[F]] = None) -> PrefectFutureList[F]:
        ...

    def __enter__(self) -> TaskRunner[F]:
        ...

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        ...

class ThreadPoolTaskRunner(TaskRunner[PrefectConcurrentFuture[R]]):
    def __init__(self, max_workers: Optional[int] = None) -> None:
        ...

    def duplicate(self) -> ThreadPoolTaskRunner:
        ...

    @overload
    def submit(self, task: Task, parameters: Dict[str, Any], wait_for: Optional[List[PrefectConcurrentFuture[R]]] = None, dependencies: Optional[Dict[str, Any]] = None) -> PrefectConcurrentFuture[R]:
        ...

    def submit(self, task: Task, parameters: Dict[str, Any], wait_for: Optional[List[PrefectConcurrentFuture[R]]] = None, dependencies: Optional[Dict[str, Any]] = None) -> PrefectConcurrentFuture[R]:
        ...

    @overload
    def map(self, task: Task, parameters: Dict[str, Any], wait_for: Optional[List[PrefectConcurrentFuture[R]]] = None) -> PrefectFutureList[PrefectConcurrentFuture[R]]:
        ...

    def map(self, task: Task, parameters: Dict[str, Any], wait_for: Optional[List[PrefectConcurrentFuture[R]]] = None) -> PrefectFutureList[PrefectConcurrentFuture[R]]:
        ...

    def cancel_all(self) -> None:
        ...

    def __enter__(self) -> ThreadPoolTaskRunner:
        ...

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        ...

    def __eq__(self, value: Any) -> bool:
        ...

class PrefectTaskRunner(TaskRunner[PrefectDistributedFuture[R]]):
    def __init__(self) -> None:
        ...

    def duplicate(self) -> PrefectTaskRunner:
        ...

    @overload
    def submit(self, task: Task, parameters: Dict[str, Any], wait_for: Optional[List[PrefectDistributedFuture[R]]] = None, dependencies: Optional[Dict[str, Any]] = None) -> PrefectDistributedFuture[R]:
        ...

    def submit(self, task: Task, parameters: Dict[str, Any], wait_for: Optional[List[PrefectDistributedFuture[R]]] = None, dependencies: Optional[Dict[str, Any]] = None) -> PrefectDistributedFuture[R]:
        ...

    @overload
    def map(self, task: Task, parameters: Dict[str, Any], wait_for: Optional[List[PrefectDistributedFuture[R]]] = None) -> PrefectFutureList[PrefectDistributedFuture[R]]:
        ...

    def map(self, task: Task, parameters: Dict[str, Any], wait_for: Optional[List[PrefectDistributedFuture[R]]] = None) -> PrefectFutureList[PrefectDistributedFuture[R]]:
        ...
