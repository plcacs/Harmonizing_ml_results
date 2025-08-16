import asyncio
from contextlib import ExitStack
from typing import Any, Callable, Coroutine, Dict, Iterable, Optional, Set, TypeVar, Union, overload
import distributed
from typing_extensions import ParamSpec
from prefect.client.schemas.objects import State, TaskRunInput
from prefect.futures import PrefectFuture, PrefectFutureList, PrefectWrappedFuture
from prefect.logging.loggers import get_logger, get_run_logger
from prefect.task_runners import TaskRunner
from prefect.tasks import Task
from prefect.utilities.asyncutils import run_coro_as_sync
from prefect.utilities.collections import visit_collection
from prefect.utilities.importtools import from_qualified_name, to_qualified_name
from prefect_dask.client import PrefectDaskClient

logger: Any = get_logger(__name__)
P = ParamSpec('P')
T = TypeVar('T')
F = TypeVar('F', bound=PrefectFuture)
R = TypeVar('R')

class PrefectDaskFuture(PrefectWrappedFuture[R, distributed.Future]):
    def wait(self, timeout: Optional[float] = None) -> None:
        ...

    def result(self, timeout: Optional[float] = None, raise_on_failure: bool = True) -> Any:
        ...

    def __del__(self) -> None:
        ...

class DaskTaskRunner(TaskRunner):
    def __init__(self, cluster: Optional[distributed.deploy.Cluster] = None, address: Optional[str] = None, cluster_class: Optional[Union[str, Callable]] = None, cluster_kwargs: Optional[Dict[str, Any]] = None, adapt_kwargs: Optional[Dict[str, Any]] = None, client_kwargs: Optional[Dict[str, Any]] = None) -> None:
        ...

    def __eq__(self, other: Any) -> bool:
        ...

    def duplicate(self) -> 'DaskTaskRunner':
        ...

    @overload
    def submit(self, task: Task, parameters: Dict[str, Any], wait_for: Optional[Union[F, PrefectFutureList[F]]] = None, dependencies: Optional[Set[F]] = None) -> PrefectDaskFuture[R]:
        ...

    @overload
    def submit(self, task: Task, parameters: Dict[str, Any], wait_for: Optional[Union[F, PrefectFutureList[F]]] = None, dependencies: Optional[Set[F]] = None) -> PrefectDaskFuture[R]:
        ...

    def submit(self, task: Task, parameters: Dict[str, Any], wait_for: Optional[Union[F, PrefectFutureList[F]]] = None, dependencies: Optional[Set[F]] = None) -> PrefectDaskFuture[R]:
        ...

    @overload
    def map(self, task: Task, parameters: Iterable[Dict[str, Any]], wait_for: Optional[Union[F, PrefectFutureList[F]]] = None) -> PrefectFutureList[R]:
        ...

    @overload
    def map(self, task: Task, parameters: Iterable[Dict[str, Any]], wait_for: Optional[Union[F, PrefectFutureList[F]]] = None) -> PrefectFutureList[R]:
        ...

    def map(self, task: Task, parameters: Iterable[Dict[str, Any]], wait_for: Optional[Union[F, PrefectFutureList[F]]] = None) -> PrefectFutureList[R]:
        ...

    def _optimize_futures(self, expr: Any) -> Any:
        ...

    def __enter__(self) -> 'DaskTaskRunner':
        ...

    def __exit__(self, *args: Any) -> None:
        ...
