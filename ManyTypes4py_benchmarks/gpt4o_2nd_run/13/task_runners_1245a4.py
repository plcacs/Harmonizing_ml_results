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
    def wait(self, timeout: Optional[float] = None) -> None:
        try:
            result = self._wrapped_future.result(timeout=timeout)
        except Exception:
            return
        if isinstance(result, State):
            self._final_state = result

    def result(self, timeout: Optional[float] = None, raise_on_failure: bool = True) -> Any:
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
    def __init__(self, cluster: Optional[distributed.deploy.Cluster] = None, address: Optional[str] = None, cluster_class: Optional[Union[str, Callable[..., distributed.deploy.Cluster]]] = None, cluster_kwargs: Optional[Dict[str, Any]] = None, adapt_kwargs: Optional[Dict[str, Any]] = None, client_kwargs: Optional[Dict[str, Any]] = None) -> None:
        if address:
            if cluster or cluster_class or cluster_kwargs or adapt_kwargs:
                raise ValueError('Cannot specify `address` and `cluster`/`cluster_class`/`cluster_kwargs`/`adapt_kwargs`')
        elif cluster:
            if cluster_class or cluster_kwargs:
                raise ValueError('Cannot specify `cluster` and `cluster_class`/`cluster_kwargs`')
        elif isinstance(cluster_class, str):
            cluster_class = from_qualified_name(cluster_class)
        else:
            cluster_class = cluster_class
        cluster_kwargs = cluster_kwargs.copy() if cluster_kwargs else {}
        adapt_kwargs = adapt_kwargs.copy() if adapt_kwargs else {}
        client_kwargs = client_kwargs.copy() if client_kwargs else {}
        client_kwargs.setdefault('set_as_default', False)
        if 'asynchronous' in client_kwargs:
            raise ValueError('`client_kwargs` cannot set `asynchronous`. This option is managed by Prefect.')
        if 'asynchronous' in cluster_kwargs:
            raise ValueError('`cluster_kwargs` cannot set `asynchronous`. This option is managed by Prefect.')
        self.address = address
        self.cluster_class = cluster_class
        self.cluster_kwargs = cluster_kwargs
        self.adapt_kwargs = adapt_kwargs
        self.client_kwargs = client_kwargs
        self._client: Optional[PrefectDaskClient] = None
        self._cluster: Optional[distributed.deploy.Cluster] = cluster
        self._exit_stack = ExitStack()
        super().__init__()

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, DaskTaskRunner):
            return self.address == other.address and self.cluster_class == other.cluster_class and (self.cluster_kwargs == other.cluster_kwargs) and (self.adapt_kwargs == other.adapt_kwargs) and (self.client_kwargs == other.client_kwargs)
        else:
            return False

    def duplicate(self) -> 'DaskTaskRunner':
        return type(self)(address=self.address, cluster_class=self.cluster_class, cluster_kwargs=self.cluster_kwargs, adapt_kwargs=self.adapt_kwargs, client_kwargs=self.client_kwargs)

    @overload
    def submit(self, task: Callable[..., R], parameters: Dict[str, Any], wait_for: Optional[Iterable[PrefectFuture]] = None, dependencies: Optional[Set[PrefectFuture]] = None) -> PrefectDaskFuture[R]:
        ...

    @overload
    def submit(self, task: Callable[..., R], parameters: Dict[str, Any], wait_for: Optional[Iterable[PrefectFuture]] = None, dependencies: Optional[Set[PrefectFuture]] = None) -> PrefectDaskFuture[R]:
        ...

    def submit(self, task: Callable[..., R], parameters: Dict[str, Any], wait_for: Optional[Iterable[PrefectFuture]] = None, dependencies: Optional[Set[PrefectFuture]] = None) -> PrefectDaskFuture[R]:
        if not self._started:
            raise RuntimeError('The task runner must be started before submitting work.')
        parameters = self._optimize_futures(parameters)
        wait_for = self._optimize_futures(wait_for) if wait_for else None
        future = self._client.submit(task, parameters=parameters, wait_for=wait_for, dependencies=dependencies, return_type='state')
        return PrefectDaskFuture[R](wrapped_future=future, task_run_id=future.task_run_id)

    @overload
    def map(self, task: Callable[..., R], parameters: Iterable[Dict[str, Any]], wait_for: Optional[Iterable[PrefectFuture]] = None) -> PrefectFutureList[R]:
        ...

    @overload
    def map(self, task: Callable[..., R], parameters: Iterable[Dict[str, Any]], wait_for: Optional[Iterable[PrefectFuture]] = None) -> PrefectFutureList[R]:
        ...

    def map(self, task: Callable[..., R], parameters: Iterable[Dict[str, Any]], wait_for: Optional[Iterable[PrefectFuture]] = None) -> PrefectFutureList[R]:
        return super().map(task, parameters, wait_for)

    def _optimize_futures(self, expr: Any) -> Any:
        def visit_fn(expr: Any) -> Any:
            if isinstance(expr, PrefectDaskFuture):
                dask_future = expr.wrapped_future
                if dask_future is not None:
                    return dask_future
            return expr
        return visit_collection(expr, visit_fn=visit_fn, return_data=True)

    def __enter__(self) -> 'DaskTaskRunner':
        in_dask = False
        try:
            client = distributed.get_client()
            if client.cluster is not None:
                self._cluster = client.cluster
            elif client.scheduler is not None:
                self.address = client.scheduler.address
            else:
                raise RuntimeError('No global client found and no address provided')
            in_dask = True
        except ValueError:
            pass
        super().__enter__()
        exit_stack = self._exit_stack.__enter__()
        if self._cluster:
            self.logger.info(f'Connecting to existing Dask cluster {self._cluster}')
            self._connect_to = self._cluster
            if self.adapt_kwargs:
                self._cluster.adapt(**self.adapt_kwargs)
        elif self.address:
            self.logger.info(f'Connecting to an existing Dask cluster at {self.address}')
            self._connect_to = self.address
        else:
            self.cluster_class = self.cluster_class or distributed.LocalCluster
            self.logger.info(f'Creating a new Dask cluster with `{to_qualified_name(self.cluster_class)}`')
            self._connect_to = self._cluster = exit_stack.enter_context(self.cluster_class(**self.cluster_kwargs))
            if self.adapt_kwargs:
                maybe_coro = self._cluster.adapt(**self.adapt_kwargs)
                if asyncio.iscoroutine(maybe_coro):
                    run_coro_as_sync(maybe_coro)
        self._client = exit_stack.enter_context(PrefectDaskClient(self._connect_to, **self.client_kwargs))
        if self._client.dashboard_link and (not in_dask):
            self.logger.info(f'The Dask dashboard is available at {self._client.dashboard_link}')
        return self

    def __exit__(self, *args: Any) -> None:
        self._exit_stack.__exit__(*args)
        super().__exit__(*args)
