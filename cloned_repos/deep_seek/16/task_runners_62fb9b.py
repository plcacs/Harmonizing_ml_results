from __future__ import annotations
import abc
import asyncio
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, Future as ConcurrentFuture
from contextvars import copy_context
from typing import (
    TYPE_CHECKING,
    Any,
    Coroutine,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    overload,
)
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
    """
    Abstract base class for task runners.
    """
    def __init__(self) -> None:
        self.logger: logging.Logger = get_logger(f'task_runner.{self.name}')
        self._started: bool = False

    @property
    def name(self) -> str:
        """The name of this task runner"""
        return type(self).__name__.lower().replace('taskrunner', '')

    @abc.abstractmethod
    def duplicate(self) -> Self:
        """Return a new instance of this task runner with the same configuration."""
        ...

    @overload
    def submit(
        self,
        task: Task[P, R],
        parameters: Dict[str, Any],
        wait_for: Optional[Iterable[PrefectFuture[Any]]] = None,
        dependencies: Optional[Dict[str, List[TaskRunInput]]] = None,
    ) -> F:
        ...

    @overload
    def submit(
        self,
        task: Task[P, R],
        parameters: Dict[str, Any],
        wait_for: Optional[Iterable[PrefectFuture[Any]]] = None,
        dependencies: Optional[Dict[str, List[TaskRunInput]]] = None,
    ) -> F:
        ...

    @abc.abstractmethod
    def submit(
        self,
        task: Task[P, R],
        parameters: Dict[str, Any],
        wait_for: Optional[Iterable[PrefectFuture[Any]]] = None,
        dependencies: Optional[Dict[str, List[TaskRunInput]]] = None,
    ) -> F:
        ...

    def map(
        self,
        task: Task[P, R],
        parameters: Dict[str, Any],
        wait_for: Optional[Iterable[PrefectFuture[Any]]] = None,
    ) -> PrefectFutureList[R]:
        """
        Submit multiple tasks to the task run engine.
        """
        if not self._started:
            raise RuntimeError('The task runner must be started before submitting work.')
        from prefect.utilities.engine import collect_task_run_inputs_sync, resolve_inputs_sync
        task_inputs: Dict[str, List[TaskRunInput]] = {k: collect_task_run_inputs_sync(v, max_depth=0) for k, v in parameters.items()}
        parameters = resolve_inputs_sync(parameters, max_depth=0)
        parameters = explode_variadic_parameter(task.fn, parameters)
        iterable_parameters: Dict[str, List[Any]] = {}
        static_parameters: Dict[str, Any] = {}
        annotated_parameters: Dict[str, Union[allow_failure, quote]] = {}
        
        for key, val in parameters.items():
            if isinstance(val, (allow_failure, quote)):
                annotated_parameters[key] = val
                val = val.unwrap()
            if isinstance(val, unmapped):
                static_parameters[key] = val.value
            elif isiterable(val):
                iterable_parameters[key] = list(val)
            else:
                static_parameters[key] = val

        if not len(iterable_parameters):
            raise MappingMissingIterable(f'No iterable parameters were received. Parameters for map must include at least one iterable. Parameters: {parameters}')
        
        iterable_parameter_lengths: Dict[str, int] = {key: len(val) for key, val in iterable_parameters.items()}
        lengths: Set[int] = set(iterable_parameter_lengths.values())
        
        if len(lengths) > 1:
            raise MappingLengthMismatch(f'Received iterable parameters with different lengths. Parameters for map must all be the same length. Got lengths: {iterable_parameter_lengths}')
        
        map_length: int = list(lengths)[0]
        futures: List[F] = []
        
        for i in range(map_length):
            call_parameters: Dict[str, Any] = {key: value[i] for key, value in iterable_parameters.items()}
            call_parameters.update({key: value for key, value in static_parameters.items()})
            
            for key, value in get_parameter_defaults(task.fn).items():
                call_parameters.setdefault(key, value)
            
            for key, annotation in annotated_parameters.items():
                call_parameters[key] = annotation.rewrap(call_parameters[key])
            
            call_parameters = collapse_variadic_parameters(task.fn, call_parameters)
            futures.append(self.submit(task=task, parameters=call_parameters, wait_for=wait_for, dependencies=task_inputs))
        
        return PrefectFutureList(futures)

    def __enter__(self) -> Self:
        if self._started:
            raise RuntimeError('This task runner is already started')
        self.logger.debug('Starting task runner')
        self._started = True
        return self

    def __exit__(self, exc_type: Optional[type], exc_value: Optional[BaseException], traceback: Optional[Any]) -> None:
        self.logger.debug('Stopping task runner')
        self._started = False

class ThreadPoolTaskRunner(TaskRunner[PrefectConcurrentFuture[R]]):
    def __init__(self, max_workers: Optional[int] = None) -> None:
        super().__init__()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._max_workers: int = PREFECT_TASK_RUNNER_THREAD_POOL_MAX_WORKERS.value() or sys.maxsize if max_workers is None else max_workers
        self._cancel_events: Dict[uuid.UUID, threading.Event] = {}

    def duplicate(self) -> Self:
        return type(self)(max_workers=self._max_workers)

    @overload
    def submit(
        self,
        task: Task[P, R],
        parameters: Dict[str, Any],
        wait_for: Optional[Iterable[PrefectFuture[Any]]] = None,
        dependencies: Optional[Dict[str, List[TaskRunInput]]] = None,
    ) -> PrefectConcurrentFuture[R]:
        ...

    @overload
    def submit(
        self,
        task: Task[P, R],
        parameters: Dict[str, Any],
        wait_for: Optional[Iterable[PrefectFuture[Any]]] = None,
        dependencies: Optional[Dict[str, List[TaskRunInput]]] = None,
    ) -> PrefectConcurrentFuture[R]:
        ...

    def submit(
        self,
        task: Task[P, R],
        parameters: Dict[str, Any],
        wait_for: Optional[Iterable[PrefectFuture[Any]]] = None,
        dependencies: Optional[Dict[str, List[TaskRunInput]]] = None,
    ) -> PrefectConcurrentFuture[R]:
        if not self._started or self._executor is None:
            raise RuntimeError('Task runner is not started')
        from prefect.context import FlowRunContext
        from prefect.task_engine import run_task_async, run_task_sync
        task_run_id: uuid.UUID = uuid.uuid4()
        cancel_event: threading.Event = threading.Event()
        self._cancel_events[task_run_id] = cancel_event
        context = copy_context()
        flow_run_ctx = FlowRunContext.get()
        
        if flow_run_ctx:
            get_run_logger(flow_run_ctx).debug(f'Submitting task {task.name} to thread pool executor...')
        else:
            self.logger.debug(f'Submitting task {task.name} to thread pool executor...')
        
        submit_kwargs: Dict[str, Any] = dict(
            task=task,
            task_run_id=task_run_id,
            parameters=parameters,
            wait_for=wait_for,
            return_type='state',
            dependencies=dependencies,
            context=dict(cancel_event=cancel_event)
        )
        
        future: ConcurrentFuture[Any]
        if task.isasync:
            future = self._executor.submit(context.run, asyncio.run, run_task_async(**submit_kwargs))
        else:
            future = self._executor.submit(context.run, run_task_sync, **submit_kwargs)
        
        prefect_future: PrefectConcurrentFuture[R] = PrefectConcurrentFuture(task_run_id=task_run_id, wrapped_future=future)
        return prefect_future

    @overload
    def map(
        self,
        task: Task[P, R],
        parameters: Dict[str, Any],
        wait_for: Optional[Iterable[PrefectFuture[Any]]] = None,
    ) -> PrefectFutureList[R]:
        ...

    @overload
    def map(
        self,
        task: Task[P, R],
        parameters: Dict[str, Any],
        wait_for: Optional[Iterable[PrefectFuture[Any]]] = None,
    ) -> PrefectFutureList[R]:
        ...

    def map(
        self,
        task: Task[P, R],
        parameters: Dict[str, Any],
        wait_for: Optional[Iterable[PrefectFuture[Any]]] = None,
    ) -> PrefectFutureList[R]:
        return super().map(task, parameters, wait_for)

    def cancel_all(self) -> None:
        for event in self._cancel_events.values():
            event.set()
            self.logger.debug('Set cancel event')
        if self._executor is not None:
            self._executor.shutdown(cancel_futures=True)
            self._executor = None

    def __enter__(self) -> Self:
        super().__enter__()
        self._executor = ThreadPoolExecutor(max_workers=self._max_workers)
        return self

    def __exit__(self, exc_type: Optional[type], exc_value: Optional[BaseException], traceback: Optional[Any]) -> None:
        self.cancel_all()
        if self._executor is not None:
            self._executor.shutdown(cancel_futures=True)
            self._executor = None
        super().__exit__(exc_type, exc_value, traceback)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, ThreadPoolTaskRunner):
            return False
        return self._max_workers == value._max_workers

ConcurrentTaskRunner = ThreadPoolTaskRunner

class PrefectTaskRunner(TaskRunner[PrefectDistributedFuture[R]]):
    def __init__(self) -> None:
        super().__init__()

    def duplicate(self) -> Self:
        return type(self)()

    @overload
    def submit(
        self,
        task: Task[P, R],
        parameters: Dict[str, Any],
        wait_for: Optional[Iterable[PrefectFuture[Any]]] = None,
        dependencies: Optional[Dict[str, List[TaskRunInput]]] = None,
    ) -> PrefectDistributedFuture[R]:
        ...

    @overload
    def submit(
        self,
        task: Task[P, R],
        parameters: Dict[str, Any],
        wait_for: Optional[Iterable[PrefectFuture[Any]]] = None,
        dependencies: Optional[Dict[str, List[TaskRunInput]]] = None,
    ) -> PrefectDistributedFuture[R]:
        ...

    def submit(
        self,
        task: Task[P, R],
        parameters: Dict[str, Any],
        wait_for: Optional[Iterable[PrefectFuture[Any]]] = None,
        dependencies: Optional[Dict[str, List[TaskRunInput]]] = None,
    ) -> PrefectDistributedFuture[R]:
        if not self._started:
            raise RuntimeError('Task runner is not started')
        from prefect.context import FlowRunContext
        flow_run_ctx = FlowRunContext.get()
        if flow_run_ctx:
            get_run_logger(flow_run_ctx).info(f'Submitting task {task.name} to for execution by a Prefect task worker...')
        else:
            self.logger.info(f'Submitting task {task.name} to for execution by a Prefect task worker...')
        return task.apply_async(kwargs=parameters, wait_for=wait_for, dependencies=dependencies)

    @overload
    def map(
        self,
        task: Task[P, R],
        parameters: Dict[str, Any],
        wait_for: Optional[Iterable[PrefectFuture[Any]]] = None,
    ) -> PrefectFutureList[R]:
        ...

    @overload
    def map(
        self,
        task: Task[P, R],
        parameters: Dict[str, Any],
        wait_for: Optional[Iterable[PrefectFuture[Any]]] = None,
    ) -> PrefectFutureList[R]:
        ...

    def map(
        self,
        task: Task[P, R],
        parameters: Dict[str, Any],
        wait_for: Optional[Iterable[PrefectFuture[Any]]] = None,
    ) -> PrefectFutureList[R]:
        return super().map(task, parameters, wait_for)
