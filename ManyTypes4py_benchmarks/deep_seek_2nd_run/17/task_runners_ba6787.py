from __future__ import annotations
import asyncio
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union, overload
from uuid import UUID, uuid4
from typing_extensions import ParamSpec
from prefect.client.schemas.objects import TaskRunInput
from prefect.context import serialize_context
from prefect.futures import PrefectFuture, PrefectFutureList, PrefectWrappedFuture
from prefect.logging.loggers import get_logger, get_run_logger
from prefect.states import State, exception_to_crashed_state
from prefect.task_engine import run_task_async, run_task_sync
from prefect.task_runners import TaskRunner
from prefect.tasks import Task
from prefect.utilities.asyncutils import run_coro_as_sync
from prefect.utilities.collections import visit_collection
from prefect.utilities.importtools import lazy_import
from prefect_ray.context import RemoteOptionsContext

if TYPE_CHECKING:
    import ray

ray = lazy_import('ray')
logger = get_logger(__name__)
P = ParamSpec('P')
T = TypeVar('T')
F = TypeVar('F', bound=PrefectFuture[Any])
R = TypeVar('R')

class PrefectRayFuture(PrefectWrappedFuture[R, 'ray.ObjectRef']):

    def wait(self, timeout: Optional[float] = None) -> None:
        try:
            result = ray.get(self.wrapped_future, timeout=timeout)
        except ray.exceptions.GetTimeoutError:
            return
        except Exception as exc:
            result = run_coro_as_sync(exception_to_crashed_state(exc))
        if isinstance(result, State):
            self._final_state = result

    def result(self, timeout: Optional[float] = None, raise_on_failure: bool = True) -> R:
        if not self._final_state:
            try:
                object_ref_result = ray.get(self.wrapped_future, timeout=timeout)
            except ray.exceptions.GetTimeoutError as exc:
                raise TimeoutError(f'Task run {self.task_run_id} did not complete within {timeout} seconds') from exc
            if isinstance(object_ref_result, State):
                self._final_state = object_ref_result
            else:
                return object_ref_result
        _result = self._final_state.result(raise_on_failure=raise_on_failure, fetch=True)
        if asyncio.iscoroutine(_result):
            _result = run_coro_as_sync(_result)
        return _result

    def add_done_callback(self, fn: Callable[[Self], None]) -> None:
        if not self._final_state:
            def call_with_self(future: Any) -> None:
                """Call the callback with self as the argument, this is necessary to ensure we remove the future from the pending set"""
                fn(self)
            self._wrapped_future._on_completed(call_with_self)
            return
        fn(self)

    def __del__(self) -> None:
        if self._final_state:
            return
        try:
            ray.get(self.wrapped_future, timeout=0)
        except ray.exceptions.GetTimeoutError:
            pass
        try:
            local_logger = get_run_logger()
        except Exception:
            local_logger = logger
        local_logger.warning('A future was garbage collected before it resolved. Please call `.wait()` or `.result()` on futures to ensure they resolve.')

class RayTaskRunner(TaskRunner[PrefectRayFuture[R]]):
    """
    A parallel task_runner that submits tasks to `ray`.
    By default, a temporary Ray cluster is created for the duration of the flow run.
    Alternatively, if you already have a `ray` instance running, you can provide
    the connection URL via the `address` kwarg.
    Args:
        address (string, optional): Address of a currently running `ray` instance; if
            one is not provided, a temporary instance will be created.
        init_kwargs (dict, optional): Additional kwargs to use when calling `ray.init`.
    Examples:
        Using a temporary local ray cluster:
        