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

logger = get_logger(__name__)
P = ParamSpec('P')
T = TypeVar('T')
F = TypeVar('F', bound=PrefectFuture)
R = TypeVar('R')


class PrefectDaskFuture(PrefectWrappedFuture[R, distributed.Future]):
    """
    A Prefect future that wraps a distributed.Future. This future is used
    when the task run is submitted to a DaskTaskRunner.
    """

    def wait(self, timeout: Optional[float] = None) -> None:
        try:
            result = self._wrapped_future.result(timeout=timeout)
        except Exception:
            return
        if isinstance(result, State):
            self._final_state = result

    def result(self, timeout: Optional[float] = None, raise_on_failure: bool = True) -> R:
        if not self._final_state:
            try:
                future_result = self._wrapped_future.result(timeout=timeout)
            except distributed.TimeoutError as exc:
                raise TimeoutError(f'Task run {self.task_run_id} did not complete within {timeout} seconds') from exc
            if isinstance(future_result, State):
                self._final_state = future_result
            else:
                return future_result
        _result = self._final_state.result(raise_on_failure=raise_on_failure, fetch=True)
        if asyncio.iscoroutine(_result):
            _result = run_coro_as_sync(_result)
        return _result

    def __del__(self) -> None:
        if self._final_state or self._wrapped_future.done():
            return
        try:
            local_logger = get_run_logger()
        except Exception:
            local_logger = logger
        local_logger.warning('A future was garbage collected before it resolved. Please call `.wait()` or `.result()` on futures to ensure they resolve.')


class DaskTaskRunner(TaskRunner):
    """
    A parallel task_runner that submits tasks to the `dask.distributed` scheduler.
    By default a temporary `distributed.LocalCluster` is created (and
    subsequently torn down) within the `start()` contextmanager. To use a
    different cluster class (e.g.
    [`dask_kubernetes.KubeCluster`](https://kubernetes.dask.org/)), you can
    specify `cluster_class`/`cluster_kwargs`.

    Alternatively, if you already have a dask cluster running, you can provide
    the cluster object via the `cluster` kwarg or the address of the scheduler
    via the `address` kwarg.
    !!! warning "Multiprocessing safety"
        Note that, because the `DaskTaskRunner` uses multiprocessing, calls to flows
        in scripts must be guarded with `if __name__ == "__main__":` or warnings will
        be displayed.

    Args:
        cluster (distributed.deploy.Cluster, optional): Currently running dask cluster;
            if one is not provider (or specified via `address` kwarg), a temporary
            cluster will be created in `DaskTaskRunner.start()`. Defaults to `None`.
        address (str, optional): Address of a currently running dask
            scheduler. Defaults to `None`.
        cluster_class (Union[str, Callable[..., distributed.deploy.Cluster]], optional): The cluster class to use
            when creating a temporary dask cluster. Can be either the full
            class name (e.g. `"distributed.LocalCluster"`), or the class itself.
        cluster_kwargs (Optional[Dict[str, Any]], optional): Additional kwargs to pass to the
            `cluster_class` when creating a temporary dask cluster.
        adapt_kwargs (Optional[Dict[str, Any]], optional): Additional kwargs to pass to `cluster.adapt`
            when creating a temporary dask cluster. Note that adaptive scaling
            is only enabled if `adapt_kwargs` are provided.
        client_kwargs (Optional[Dict[str, Any]], optional): Additional kwargs to use when creating a
            [`dask.distributed.Client`](https://distributed.dask.org/en/latest/api.html#client).

    Examples:
        Using a temporary local dask cluster:
        