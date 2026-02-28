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
    """A Prefect future that wraps a Ray object ref."""

    def wait(self, timeout: Optional[float] = None) -> None:
        """Wait for the task run to complete.

        Args:
            timeout (float, optional): The timeout in seconds. Defaults to None.
        """
        try:
            result = ray.get(self.wrapped_future, timeout=timeout)
        except ray.exceptions.GetTimeoutError:
            return
        except Exception as exc:
            result = run_coro_as_sync(exception_to_crashed_state(exc))
        if isinstance(result, State):
            self._final_state = result

    def result(self, timeout: Optional[float] = None, raise_on_failure: bool = True) -> R:
        """Get the result of the task run.

        Args:
            timeout (float, optional): The timeout in seconds. Defaults to None.
            raise_on_failure (bool, optional): Whether to raise an exception if the task run fails. Defaults to True.

        Returns:
            R: The result of the task run.
        """
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

    def add_done_callback(self, fn: Callable[['PrefectRayFuture[R]', None]]) -> None:
        """Add a callback to be called when the task run is complete.

        Args:
            fn (Callable[['PrefectRayFuture[R]', None]]): The callback function.
        """
        if not self._final_state:
            def call_with_self(future):
                """Call the callback with self as the argument, this is necessary to ensure we remove the future from the pending set"""
                fn(self)
            self._wrapped_future._on_completed(call_with_self)
            return
        fn(self)

    def __del__(self) -> None:
        """Delete the future."""
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
    """A parallel task_runner that submits tasks to `ray`.

    By default, a temporary Ray cluster is created for the duration of the flow run.
    Alternatively, if you already have a `ray` instance running, you can provide
    the connection URL via the `address` kwarg.
    """

    def __init__(self, address: Optional[str] = None, init_kwargs: Optional[dict] = None) -> None:
        """Initialize the task runner.

        Args:
            address (str, optional): Address of a currently running `ray` instance; if
                one is not provided, a temporary instance will be created.
            init_kwargs (dict, optional): Additional kwargs to use when calling `ray.init`.
        """
        self.address = address
        self.init_kwargs = init_kwargs.copy() if init_kwargs else {}
        self.init_kwargs.setdefault('namespace', 'prefect')
        self._ray_context = None
        super().__init__()

    def duplicate(self) -> 'RayTaskRunner':
        """Return a new instance of with the same settings as this one."""
        return type(self)(address=self.address, init_kwargs=self.init_kwargs)

    def __eq__(self, other: object) -> bool:
        """Check if an instance has the same settings as this task runner."""
        if isinstance(other, RayTaskRunner):
            return self.address == other.address and self.init_kwargs == other.init_kwargs
        else:
            return False

    @overload
    def submit(self, task: Task, parameters: Any, wait_for: Optional[Iterable['PrefectRayFuture[R]']] = None, dependencies: Optional[Iterable['PrefectRayFuture[R]']] = None) -> PrefectRayFuture[R]:
        ...

    @overload
    def submit(self, task: Task, parameters: Any, wait_for: Optional[Iterable['PrefectRayFuture[R]']] = None, dependencies: Optional[Iterable['PrefectRayFuture[R]']] = None) -> PrefectRayFuture[R]:
        ...

    def submit(self, task: Task, parameters: Any, wait_for: Optional[Iterable['PrefectRayFuture[R]']] = None, dependencies: Optional[Iterable['PrefectRayFuture[R]']] = None) -> PrefectRayFuture[R]:
        """Submit a task to the task runner.

        Args:
            task (Task): The task to submit.
            parameters (Any): The parameters to pass to the task.
            wait_for (Iterable['PrefectRayFuture[R]', optional]: The futures to wait for before submitting the task. Defaults to None.
            dependencies (Iterable['PrefectRayFuture[R]', optional]: The futures that this task depends on. Defaults to None.

        Returns:
            PrefectRayFuture[R]: The future of the task run.
        """
        if not self._started:
            raise RuntimeError('The task runner must be started before submitting work.')
        parameters, upstream_ray_obj_refs = self._exchange_prefect_for_ray_futures(parameters)
        task_run_id = uuid4()
        context = serialize_context()
        remote_options = RemoteOptionsContext.get().current_remote_options
        if remote_options:
            ray_decorator = ray.remote(**remote_options)
        else:
            ray_decorator = ray.remote
        object_ref = ray_decorator(self._run_prefect_task).options(name=task.name).remote(*upstream_ray_obj_refs, task=task, task_run_id=task_run_id, parameters=parameters, wait_for=wait_for, dependencies=dependencies, context=context)
        return PrefectRayFuture[R](task_run_id=task_run_id, wrapped_future=object_ref)

    @overload
    def map(self, task: Task, parameters: Any, wait_for: Optional[Iterable['PrefectRayFuture[R]']] = None) -> PrefectFutureList[R]:
        ...

    @overload
    def map(self, task: Task, parameters: Any, wait_for: Optional[Iterable['PrefectRayFuture[R]']] = None) -> PrefectFutureList[R]:
        ...

    def map(self, task: Task, parameters: Any, wait_for: Optional[Iterable['PrefectRayFuture[R]']] = None) -> PrefectFutureList[R]:
        """Map a task to a list of parameters.

        Args:
            task (Task): The task to map.
            parameters (Any): The parameters to map the task to.
            wait_for (Iterable['PrefectRayFuture[R]', optional]: The futures to wait for before mapping the task. Defaults to None.

        Returns:
            PrefectFutureList[R]: The list of futures of the task runs.
        """
        return super().map(task, parameters, wait_for)

    def _exchange_prefect_for_ray_futures(self, kwargs_prefect_futures: Any) -> tuple[Any, list['ray.ObjectRef']]:
        """Exchanges Prefect futures for Ray futures.

        Args:
            kwargs_prefect_futures (Any): The parameters to exchange.

        Returns:
            tuple[Any, list['ray.ObjectRef']]: The exchanged parameters and the list of upstream Ray futures.
        """
        upstream_ray_obj_refs = []

        def exchange_prefect_for_ray_future(expr: Any) -> 'ray.ObjectRef':
            """Exchanges Prefect future for Ray future."""
            if isinstance(expr, PrefectRayFuture):
                ray_future = expr.wrapped_future
                upstream_ray_obj_refs.append(ray_future)
                return ray_future
            return expr
        kwargs_ray_futures = visit_collection(kwargs_prefect_futures, visit_fn=exchange_prefect_for_ray_future, return_data=True)
        return (kwargs_ray_futures, upstream_ray_obj_refs)

    @staticmethod
    def _run_prefect_task(*upstream_ray_obj_refs: 'ray.ObjectRef', task: Task, task_run_id: UUID, context: Any, parameters: Any, wait_for: Optional[Iterable['PrefectRayFuture[R]']] = None, dependencies: Optional[Iterable['PrefectRayFuture[R]']] = None) -> R:
        """Resolves Ray futures before calling the actual Prefect task function.

        Passing upstream_ray_obj_refs directly as args enables Ray to wait for
        upstream tasks before running this remote function.
        This variable is otherwise unused as the ray object refs are also
        contained in parameters.
        """

        def resolve_ray_future(expr: Any) -> Any:
            """Resolves Ray future."""
            if isinstance(expr, 'ray.ObjectRef'):
                return ray.get(expr)
            return expr
        parameters = visit_collection(parameters, visit_fn=resolve_ray_future, return_data=True)
        run_task_kwargs = {'task': task, 'task_run_id': task_run_id, 'parameters': parameters, 'wait_for': wait_for, 'dependencies': dependencies, 'context': context, 'return_type': 'state'}
        if task.isasync:
            return asyncio.run(run_task_async(**run_task_kwargs))
        else:
            return run_task_sync(**run_task_kwargs)

    def __enter__(self) -> 'RayTaskRunner':
        """Enter the context manager."""
        super().__enter__()
        if ray.is_initialized():
            self.logger.info('Local Ray instance is already initialized. Using existing local instance.')
            return self
        elif self.address and self.address != 'auto':
            self.logger.info(f'Connecting to an existing Ray instance at {self.address}')
            init_args = (self.address,)
        else:
            self.logger.info('Creating a local Ray instance')
            init_args = ()
        self._ray_context = ray.init(*init_args, **self.init_kwargs)
        dashboard_url = getattr(self._ray_context, 'dashboard_url', None)
        nodes = ray.nodes()
        living_nodes = [node for node in nodes if node.get('alive')]
        self.logger.info(f'Using Ray cluster with {len(living_nodes)} nodes.')
        if dashboard_url:
            self.logger.info(f'The Ray UI is available at {dashboard_url}')
        return self

    def __exit__(self, *exc_info) -> None:
        """Exit the context manager."""
        if ray.get_runtime_context().worker.mode == 0:
            self.logger.debug('Shutting down Ray driver...')
            ray.shutdown()
        super().__exit__(*exc_info)
