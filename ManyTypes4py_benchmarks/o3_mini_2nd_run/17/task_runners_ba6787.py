#!/usr/bin/env python3
"""
Interface and implementations of the Ray Task Runner.
[Task Runners](https://docs.prefect.io/latest/develop/task-runners/)
in Prefect are responsible for managing the execution of Prefect task runs.
Generally speaking, users are not expected to interact with
task runners outside of configuring and initializing them for a flow.

Example:
    >>> import time
    >>>
    >>> from prefect import flow, task
    >>>
    >>> @task
    ... def shout(number):
    ...     time.sleep(0.5)
    ...     print(f"#{number}")
    >>>
    >>> @flow
    ... def count_to(highest_number):
    ...     for number in range(highest_number):
    ...         shout.submit(number)
    >>>
    >>> if __name__ == "__main__":
    ...     count_to(10)
    ...
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

    Switching to a `RayTaskRunner`:
    >>> import time
    >>>
    >>> from prefect import flow, task
    >>> from prefect_ray import RayTaskRunner
    >>>
    >>> @task
    ... def shout(number):
    ...     time.sleep(0.5)
    ...     print(f"#{number}")
    >>>
    >>> @flow(task_runner=RayTaskRunner)
    ... def count_to(highest_number):
    ...     for number in range(highest_number):
    ...         shout.submit(number)
    >>>
    >>> if __name__ == "__main__":
    ...     count_to(10)
    ...
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
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Coroutine, Iterable, Optional, TypeVar, overload, Tuple, List
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
P = ParamSpec("P")
T = TypeVar("T")
F = TypeVar("F", bound=PrefectFuture[Any])
R = TypeVar("R")


class PrefectRayFuture(PrefectWrappedFuture[R, "ray.ObjectRef"]):
    def wait(self, timeout: Optional[float] = None) -> None:
        try:
            result: Any = ray.get(self.wrapped_future, timeout=timeout)
        except ray.exceptions.GetTimeoutError:
            return
        except Exception as exc:
            result = run_coro_as_sync(exception_to_crashed_state(exc))
        if isinstance(result, State):
            self._final_state = result

    def result(self, timeout: Optional[float] = None, raise_on_failure: bool = True) -> R:
        if not self._final_state:
            try:
                object_ref_result: Any = ray.get(self.wrapped_future, timeout=timeout)
            except ray.exceptions.GetTimeoutError as exc:
                raise TimeoutError(
                    f"Task run {self.task_run_id} did not complete within {timeout} seconds"
                ) from exc
            if isinstance(object_ref_result, State):
                self._final_state = object_ref_result
            else:
                return object_ref_result
        _result: Any = self._final_state.result(raise_on_failure=raise_on_failure, fetch=True)
        if asyncio.iscoroutine(_result):
            _result = run_coro_as_sync(_result)
        return _result

    def add_done_callback(self, fn: Callable[[PrefectRayFuture[Any]], None]) -> None:
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
        local_logger.warning(
            "A future was garbage collected before it resolved. Please call "
            "`.wait()` or `.result()` on futures to ensure they resolve."
        )


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
        >>> from prefect import flow
        >>> from prefect_ray.task_runners import RayTaskRunner
        >>>
        >>> @flow(task_runner=RayTaskRunner())
        ... def my_flow():
        ...     ...
        >>>
        Using an existing ray instance:
        >>> RayTaskRunner(address="ray://192.0.2.255:8786")
    """

    def __init__(self, address: Optional[str] = None, init_kwargs: Optional[dict] = None) -> None:
        self.address: Optional[str] = address
        self.init_kwargs: dict = init_kwargs.copy() if init_kwargs else {}
        self.init_kwargs.setdefault("namespace", "prefect")
        self._ray_context: Optional[Any] = None
        super().__init__()

    def duplicate(self) -> Self:
        """
        Return a new instance with the same settings as this one.
        """
        return type(self)(address=self.address, init_kwargs=self.init_kwargs)

    def __eq__(self, other: object) -> bool:
        """
        Check if an instance has the same settings as this task runner.
        """
        if isinstance(other, RayTaskRunner):
            return self.address == other.address and self.init_kwargs == other.init_kwargs
        else:
            return False

    @overload
    def submit(
        self,
        task: Task,
        parameters: Any,
        wait_for: Optional[Any] = None,
        dependencies: Optional[Any] = None,
    ) -> PrefectRayFuture[Any]:
        ...

    @overload
    def submit(
        self,
        task: Task,
        parameters: Any,
        wait_for: Optional[Any] = None,
        dependencies: Optional[Any] = None,
    ) -> PrefectRayFuture[Any]:
        ...

    def submit(
        self,
        task: Task,
        parameters: Any,
        wait_for: Optional[Any] = None,
        dependencies: Optional[Any] = None,
    ) -> PrefectRayFuture[Any]:
        if not self._started:
            raise RuntimeError("The task runner must be started before submitting work.")
        parameters, upstream_ray_obj_refs = self._exchange_prefect_for_ray_futures(parameters)
        task_run_id: UUID = uuid4()
        context: dict = serialize_context()
        remote_options: Optional[dict] = RemoteOptionsContext.get().current_remote_options
        if remote_options:
            ray_decorator = ray.remote(**remote_options)
        else:
            ray_decorator = ray.remote
        object_ref: "ray.ObjectRef" = ray_decorator(self._run_prefect_task).options(name=task.name).remote(
            *upstream_ray_obj_refs,
            task=task,
            task_run_id=task_run_id,
            parameters=parameters,
            wait_for=wait_for,
            dependencies=dependencies,
            context=context,
        )
        return PrefectRayFuture[R](task_run_id=task_run_id, wrapped_future=object_ref)

    @overload
    def map(
        self,
        task: Task,
        parameters: Any,
        wait_for: Optional[Any] = None,
    ) -> PrefectFutureList[Any]:
        ...

    @overload
    def map(
        self,
        task: Task,
        parameters: Any,
        wait_for: Optional[Any] = None,
    ) -> PrefectFutureList[Any]:
        ...

    def map(self, task: Task, parameters: Any, wait_for: Optional[Any] = None) -> PrefectFutureList[Any]:
        return super().map(task, parameters, wait_for)

    def _exchange_prefect_for_ray_futures(
        self, kwargs_prefect_futures: Any
    ) -> Tuple[Any, List["ray.ObjectRef"]]:
        """Exchanges Prefect futures for Ray futures."""
        upstream_ray_obj_refs: List["ray.ObjectRef"] = []

        def exchange_prefect_for_ray_future(expr: Any) -> Any:
            """Exchanges Prefect future for Ray future."""
            if isinstance(expr, PrefectRayFuture):
                ray_future: "ray.ObjectRef" = expr.wrapped_future
                upstream_ray_obj_refs.append(ray_future)
                return ray_future
            return expr

        kwargs_ray_futures: Any = visit_collection(
            kwargs_prefect_futures, visit_fn=exchange_prefect_for_ray_future, return_data=True
        )
        return (kwargs_ray_futures, upstream_ray_obj_refs)

    @staticmethod
    def _run_prefect_task(
        *upstream_ray_obj_refs: "ray.ObjectRef",
        task: Task,
        task_run_id: UUID,
        context: dict,
        parameters: Any,
        wait_for: Optional[Any] = None,
        dependencies: Optional[Any] = None,
    ) -> Any:
        """Resolves Ray futures before calling the actual Prefect task function.

        Passing upstream_ray_obj_refs directly as args enables Ray to wait for
        upstream tasks before running this remote function.
        This variable is otherwise unused as the ray object refs are also
        contained in parameters.
        """

        def resolve_ray_future(expr: Any) -> Any:
            """Resolves Ray future."""
            if isinstance(expr, ray.ObjectRef):
                return ray.get(expr)
            return expr

        parameters = visit_collection(parameters, visit_fn=resolve_ray_future, return_data=True)
        run_task_kwargs: dict[str, Any] = {
            "task": task,
            "task_run_id": task_run_id,
            "parameters": parameters,
            "wait_for": wait_for,
            "dependencies": dependencies,
            "context": context,
            "return_type": "state",
        }
        if task.isasync:
            return asyncio.run(run_task_async(**run_task_kwargs))
        else:
            return run_task_sync(**run_task_kwargs)

    def __enter__(self) -> Self:
        super().__enter__()
        if ray.is_initialized():
            self.logger.info("Local Ray instance is already initialized. Using existing local instance.")
            return self
        elif self.address and self.address != "auto":
            self.logger.info(f"Connecting to an existing Ray instance at {self.address}")
            init_args = (self.address,)
        else:
            self.logger.info("Creating a local Ray instance")
            init_args = ()
        self._ray_context = ray.init(*init_args, **self.init_kwargs)
        dashboard_url: Optional[str] = getattr(self._ray_context, "dashboard_url", None)
        nodes: List[Any] = ray.nodes()
        living_nodes: List[Any] = [node for node in nodes if node.get("alive")]
        self.logger.info(f"Using Ray cluster with {len(living_nodes)} nodes.")
        if dashboard_url:
            self.logger.info(f"The Ray UI is available at {dashboard_url}")
        return self

    def __exit__(self, *exc_info: Any) -> Optional[bool]:
        """
        Shuts down the driver/cluster.
        """
        if ray.get_runtime_context().worker.mode == 0:
            self.logger.debug("Shutting down Ray driver...")
            ray.shutdown()
        super().__exit__(*exc_info)
        return None