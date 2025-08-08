from __future__ import annotations
import asyncio
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Iterable, Optional, TypeVar, overload
from uuid import UUID, uuid4
from typing_extensions import ParamSpec, Self
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

    def wait(self, timeout=None) -> None:
        ...

    def result(self, timeout=None, raise_on_failure=True) -> Any:
        ...

    def add_done_callback(self, fn: Callable[[PrefectRayFuture[R]], None]) -> None:
        ...

    def __del__(self) -> None:
        ...

class RayTaskRunner(TaskRunner[PrefectRayFuture[R]]):

    def __init__(self, address=None, init_kwargs=None) -> None:
        ...

    def duplicate(self) -> RayTaskRunner:
        ...

    def __eq__(self, other) -> bool:
        ...

    def submit(self, task, parameters, wait_for=None, dependencies=None) -> PrefectRayFuture[R]:
        ...

    def map(self, task, parameters, wait_for=None) -> Any:
        ...

    def _exchange_prefect_for_ray_futures(self, kwargs_prefect_futures) -> Tuple[Any, List['ray.ObjectRef']]:
        ...

    @staticmethod
    def _run_prefect_task(*upstream_ray_obj_refs, task, task_run_id, context, parameters, wait_for=None, dependencies=None) -> State:
        ...

    def __enter__(self) -> RayTaskRunner:
        ...

    def __exit__(self, *exc_info) -> None:
        ...
