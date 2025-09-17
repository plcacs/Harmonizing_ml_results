#!/usr/bin/env python3
"""
Interface and implementations of the Dask Task Runner.
[Task Runners](https://docs.prefect.io/api-ref/prefect/task-runners/)
in Prefect are responsible for managing the execution of Prefect task runs.
Generally speaking, users are not expected to interact with
task runners outside of configuring and initializing them for a flow.

Example:
    ---------------------------------------------------------
    import time

    from prefect import flow, task

    @task
    def shout(number: int) -> None:
        time.sleep(0.5)
        print(f"#{number}")

    @flow
    def count_to(highest_number: int) -> None:
        for number in range(highest_number):
            shout.submit(number)

    if __name__ == "__main__":
        count_to(10)

    # outputs
    #0
    #1
    #2
    #3
    #4
    #5
    #6
    #7
    #8
    #9
    ---------------------------------------------------------

    Switching to a `DaskTaskRunner`:
    ---------------------------------------------------------
    import time

    from prefect import flow, task
    from prefect_dask import DaskTaskRunner

    @task
    def shout(number: int) -> None:
        time.sleep(0.5)
        print(f"#{number}")

    @flow(task_runner=DaskTaskRunner)
    def count_to(highest_number: int) -> None:
        for number in range(highest_number):
            shout.submit(number)

    if __name__ == "__main__":
        count_to(10)

    # outputs
    #3
    #7
    #2
    #6
    #4
    #0
    #1
    #5
    #8
    #9
    ---------------------------------------------------------
"""
import asyncio
from contextlib import ExitStack
from typing import Any, Callable, Coroutine, Dict, Iterable, Optional, Set, TypeVar, Union, overload, cast
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
P = ParamSpec("P")
T = TypeVar("T")
F = TypeVar("F", bound=PrefectFuture)
R = TypeVar("R")


class PrefectDaskFuture(PrefectWrappedFuture[R, distributed.Future]):
    """
    A Prefect future that wraps a distributed.Future. This future is used
    when the task run is submitted to a DaskTaskRunner.
    """

    def wait(self, timeout: Optional[Union[float, int]] = None) -> None:
        try:
            result: Any = self._wrapped_future.result(timeout=timeout)
        except Exception:
            return
        if isinstance(result, State):
            self._final_state = result

    def result(self, timeout: Optional[Union[float, int]] = None, raise_on_failure: bool = True) -> R:
        if not self._final_state:
            try:
                future_result: Any = self._wrapped_future.result(timeout=timeout)
            except distributed.TimeoutError as exc:
                raise TimeoutError(
                    f"Task run {self.task_run_id} did not complete within {timeout} seconds"
                ) from exc
            if isinstance(future_result, State):
                self._final_state = future_result
            else:
                return future_result  # type: ignore
        _result: Any = self._final_state.result(raise_on_failure=raise_on_failure, fetch=True)
        if asyncio.iscoroutine(_result):
            _result = run_coro_as_sync(cast(Coroutine[Any, Any, R], _result))
        return _result

    def __del__(self) -> None:
        if self._final_state or self._wrapped_future.done():
            return
        try:
            local_logger = get_run_logger()
        except Exception:
            local_logger = logger
        local_logger.warning(
            "A future was garbage collected before it resolved. Please call `.wait()` or `.result()` on futures to ensure they resolve."
        )


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
        cluster_class (str or Callable, optional): The cluster class to use
            when creating a temporary dask cluster. Can be either the full
            class name (e.g. `"distributed.LocalCluster"`), or the class itself.
        cluster_kwargs (dict, optional): Additional kwargs to pass to the
            `cluster_class` when creating a temporary dask cluster.
        adapt_kwargs (dict, optional): Additional kwargs to pass to `cluster.adapt`
            when creating a temporary dask cluster. Note that adaptive scaling
            is only enabled if `adapt_kwargs` are provided.
        client_kwargs (dict, optional): Additional kwargs to use when creating a
            [`dask.distributed.Client`](https://distributed.dask.org/en/latest/api.html#client).

    Examples:
        Using a temporary local dask cluster:
        ---------------------------------------------------------
        from prefect import flow
        from prefect_dask.task_runners import DaskTaskRunner

        @flow(task_runner=DaskTaskRunner)
        def my_flow() -> None:
            ...
        ---------------------------------------------------------

        Using a temporary cluster running elsewhere. Any Dask cluster class should
        work, here we use [dask-cloudprovider](https://cloudprovider.dask.org):
        ---------------------------------------------------------
        DaskTaskRunner(
            cluster_class="dask_cloudprovider.FargateCluster",
            cluster_kwargs={
                "image": "prefecthq/prefect:latest",
                "n_workers": 5,
            },
        )
        ---------------------------------------------------------

        Connecting to an existing dask cluster:
        ---------------------------------------------------------
        DaskTaskRunner(address="192.0.2.255:8786")
        ---------------------------------------------------------
    """

    def __init__(
        self,
        cluster: Optional[Any] = None,
        address: Optional[str] = None,
        cluster_class: Optional[Union[str, Callable]] = None,
        cluster_kwargs: Optional[Dict[str, Any]] = None,
        adapt_kwargs: Optional[Dict[str, Any]] = None,
        client_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if address:
            if cluster or cluster_class or cluster_kwargs or adapt_kwargs:
                raise ValueError(
                    "Cannot specify `address` and `cluster`/`cluster_class`/`cluster_kwargs`/`adapt_kwargs`"
                )
        elif cluster:
            if cluster_class or cluster_kwargs:
                raise ValueError("Cannot specify `cluster` and `cluster_class`/`cluster_kwargs`")
        elif isinstance(cluster_class, str):
            cluster_class = from_qualified_name(cluster_class)
        else:
            cluster_class = cluster_class
        self.address: Optional[str] = address
        self.cluster_class: Optional[Union[str, Callable]] = cluster_class
        self.cluster_kwargs: Dict[str, Any] = cluster_kwargs.copy() if cluster_kwargs else {}
        self.adapt_kwargs: Dict[str, Any] = adapt_kwargs.copy() if adapt_kwargs else {}
        self.client_kwargs: Dict[str, Any] = client_kwargs.copy() if client_kwargs else {}
        self.client_kwargs.setdefault("set_as_default", False)
        if "asynchronous" in self.client_kwargs:
            raise ValueError("`client_kwargs` cannot set `asynchronous`. This option is managed by Prefect.")
        if "asynchronous" in self.cluster_kwargs:
            raise ValueError("`cluster_kwargs` cannot set `asynchronous`. This option is managed by Prefect.")
        self._client: Optional[PrefectDaskClient] = None
        self._cluster: Optional[Any] = cluster
        self._exit_stack: ExitStack = ExitStack()
        super().__init__()

    def __eq__(self, other: Any) -> bool:
        """
        Check if an instance has the same settings as this task runner.
        """
        if isinstance(other, DaskTaskRunner):
            return (
                self.address == other.address
                and self.cluster_class == other.cluster_class
                and (self.cluster_kwargs == other.cluster_kwargs)
                and (self.adapt_kwargs == other.adapt_kwargs)
                and (self.client_kwargs == other.client_kwargs)
            )
        else:
            return False

    def duplicate(self) -> "DaskTaskRunner":
        """
        Create a new instance of the task runner with the same settings.
        """
        return type(self)(
            address=self.address,
            cluster_class=self.cluster_class,
            cluster_kwargs=self.cluster_kwargs,
            adapt_kwargs=self.adapt_kwargs,
            client_kwargs=self.client_kwargs,
        )

    @overload
    def submit(
        self,
        task: Callable[..., Any],
        parameters: Any,
        wait_for: Optional[Any] = None,
        dependencies: Optional[Any] = None,
    ) -> PrefectDaskFuture[R]:
        ...

    @overload
    def submit(
        self,
        task: Callable[..., Any],
        parameters: Any,
        wait_for: Optional[Any] = None,
        dependencies: Optional[Any] = None,
    ) -> PrefectDaskFuture[R]:
        ...

    def submit(
        self,
        task: Callable[..., Any],
        parameters: Any,
        wait_for: Optional[Any] = None,
        dependencies: Optional[Any] = None,
    ) -> PrefectDaskFuture[R]:
        if not getattr(self, "_started", False):
            raise RuntimeError("The task runner must be started before submitting work.")
        parameters = self._optimize_futures(parameters)
        wait_for = self._optimize_futures(wait_for) if wait_for else None
        future: distributed.Future = self._client.submit(
            task,
            parameters=parameters,
            wait_for=wait_for,
            dependencies=dependencies,
            return_type="state",
        )
        return PrefectDaskFuture[R](wrapped_future=future, task_run_id=future.task_run_id)

    @overload
    def map(
        self,
        task: Callable[..., Any],
        parameters: Any,
        wait_for: Optional[Any] = None,
    ) -> Any:
        ...

    @overload
    def map(
        self,
        task: Callable[..., Any],
        parameters: Any,
        wait_for: Optional[Any] = None,
    ) -> Any:
        ...

    def map(
        self, task: Callable[..., Any], parameters: Any, wait_for: Optional[Any] = None
    ) -> Any:
        return super().map(task, parameters, wait_for)

    def _optimize_futures(self, expr: Any) -> Any:
        def visit_fn(inner_expr: Any) -> Any:
            if isinstance(inner_expr, PrefectDaskFuture):
                dask_future: Optional[distributed.Future] = inner_expr.wrapped_future
                if dask_future is not None:
                    return dask_future
            return inner_expr

        return visit_collection(expr, visit_fn=visit_fn, return_data=True)

    def __enter__(self) -> "DaskTaskRunner":
        """
        Start the task runner and prep for context exit.
        - Creates a cluster if an external address is not set.
        - Creates a client to connect to the cluster.
        """
        in_dask: bool = False
        try:
            client: distributed.Client = distributed.get_client()
            if client.cluster is not None:
                self._cluster = client.cluster
            elif client.scheduler is not None:
                self.address = client.scheduler.address
            else:
                raise RuntimeError("No global client found and no address provided")
            in_dask = True
        except ValueError:
            pass
        super().__enter__()
        let_exit_stack: Any = self._exit_stack.__enter__()
        if self._cluster:
            self.logger.info(f"Connecting to existing Dask cluster {self._cluster}")
            self._connect_to: Any = self._cluster
            if self.adapt_kwargs:
                self._cluster.adapt(**self.adapt_kwargs)
        elif self.address:
            self.logger.info(f"Connecting to an existing Dask cluster at {self.address}")
            self._connect_to = self.address
        else:
            self.cluster_class = self.cluster_class or distributed.LocalCluster
            self.logger.info(f"Creating a new Dask cluster with `{to_qualified_name(self.cluster_class)}`")
            self._connect_to = self._cluster = let_exit_stack.enter_context(self.cluster_class(**self.cluster_kwargs))
            if self.adapt_kwargs:
                maybe_coro: Any = self._cluster.adapt(**self.adapt_kwargs)
                if asyncio.iscoroutine(maybe_coro):
                    run_coro_as_sync(maybe_coro)
        self._client = let_exit_stack.enter_context(PrefectDaskClient(self._connect_to, **self.client_kwargs))
        if self._client.dashboard_link and (not in_dask):
            self.logger.info(f"The Dask dashboard is available at {self._client.dashboard_link}")
        return self

    def __exit__(self, *args: Any) -> None:
        self._exit_stack.__exit__(*args)
        super().__exit__(*args)
