"""
Module containing the base workflow task class and decorator - for most use cases, using the [`@task` decorator][prefect.tasks.task] is preferred.
"""
# This file requires type-checking with pyright because mypy does not yet support PEP612
# See https://github.com/python/mypy/issues/8645

import asyncio
import datetime
import inspect
from copy import copy
from functools import partial, update_wrapper
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Coroutine,
    Generic,
    Iterable,
    NoReturn,
    Optional,
    Protocol,
    TypeVar,
    Union,
    cast,
    overload,
)
from uuid import UUID, uuid4

from typing_extensions import Literal, ParamSpec, Self, TypeAlias, TypeIs

import prefect.states
from prefect.cache_policies import DEFAULT, NO_CACHE, CachePolicy
from prefect.client.orchestration import get_client
from prefect.client.schemas import TaskRun
from prefect.client.schemas.objects import (
    StateDetails,
    TaskRunInput,
    TaskRunPolicy,
    TaskRunResult,
)
from prefect.context import (
    FlowRunContext,
    TagsContext,
    TaskRunContext,
    serialize_context,
)
from prefect.futures import PrefectDistributedFuture, PrefectFuture, PrefectFutureList
from prefect.logging.loggers import get_logger
from prefect.results import (
    ResultSerializer,
    ResultStorage,
    ResultStore,
    get_or_create_default_task_scheduling_storage,
)
from prefect.settings.context import get_current_settings
from prefect.states import Pending, Scheduled, State
from prefect.utilities.annotations import NotSet
from prefect.utilities.asyncutils import run_coro_as_sync, sync_compatible
from prefect.utilities.callables import (
    expand_mapping_parameters,
    get_call_parameters,
    raise_for_reserved_arguments,
)
from prefect.utilities.hashing import hash_objects
from prefect.utilities.importtools import to_qualified_name
from prefect.utilities.urls import url_for

if TYPE_CHECKING:
    import logging
    from prefect.client.orchestration import PrefectClient
    from prefect.context import TaskRunContext
    from prefect.transactions import Transaction

T = TypeVar("T")
R = TypeVar("R")  # The return type of the user's function
P = ParamSpec("P")  # The parameters of the task

NUM_CHARS_DYNAMIC_KEY = 8

logger: "logging.Logger" = get_logger("tasks")

FutureOrResult: TypeAlias = Union[PrefectFuture[T], T]
OneOrManyFutureOrResult: TypeAlias = Union[
    FutureOrResult[T], Iterable[FutureOrResult[T]]
]

def task_input_hash(
    context: "TaskRunContext", arguments: dict[str, Any]
) -> Optional[str]:
    """
    A task cache key implementation which hashes all inputs to the task using a JSON or
    cloudpickle serializer. If any arguments are not JSON serializable, the pickle
    serializer is used as a fallback. If cloudpickle fails, this will return a null key
    indicating that a cache key could not be generated for the given inputs.

    Arguments:
        context: the active `TaskRunContext`
        arguments: a dictionary of arguments to be passed to the underlying task

    Returns:
        a string hash if hashing succeeded, else `None`
    """
    return hash_objects(
        context.task.task_key,
        context.task.fn.__code__.co_code.hex(),
        arguments,
    )

def exponential_backoff(backoff_factor: float) -> Callable[[int], list[float]]:
    """
    A task retry backoff utility that configures exponential backoff for task retries.
    The exponential backoff design matches the urllib3 implementation.

    Arguments:
        backoff_factor: the base delay for the first retry, subsequent retries will
            increase the delay time by powers of 2.

    Returns:
        a callable that can be passed to the task constructor
    """
    def retry_backoff_callable(retries: int) -> list[float]:
        retries = min(retries, 50)
        return [backoff_factor * max(0, 2**r) for r in range(retries)]
    return retry_backoff_callable

def _infer_parent_task_runs(
    flow_run_context: Optional[FlowRunContext],
    task_run_context: Optional[TaskRunContext],
    parameters: dict[str, Any],
) -> list[TaskRunResult]:
    """
    Attempt to infer the parent task runs for this task run based on the
    provided flow run and task run contexts, as well as any parameters.
    """
    parents: list[TaskRunResult] = []

    if task_run_context:
        if not flow_run_context:
            parents.append(TaskRunResult(id=task_run_context.task_run.id))
        elif flow_run_context and task_run_context.task_run.flow_run_id == getattr(
            flow_run_context.flow_run, "id", None
        ):
            parents.append(TaskRunResult(id=task_run_context.task_run.id))

    if flow_run_context:
        for v in parameters.values():
            if isinstance(v, State):
                upstream_state = v
            elif isinstance(v, PrefectFuture):
                upstream_state = v.state
            else:
                upstream_state = flow_run_context.task_run_results.get(id(v))
            if upstream_state and upstream_state.is_running():
                parents.append(
                    TaskRunResult(id=upstream_state.state_details.task_run_id)
                )

    return parents

def _generate_task_key(fn: Callable[..., Any]) -> str:
    """Generate a task key based on the function name and source code."""
    if not hasattr(fn, "__qualname__"):
        return to_qualified_name(type(fn))

    qualname = fn.__qualname__.split(".")[-1]

    try:
        code_obj = getattr(fn, "__code__", None)
        if code_obj is None:
            code_obj = fn.__call__.__code__
    except AttributeError:
        raise AttributeError(
            f"{fn} is not a standard Python function object and could not be converted to a task."
        ) from None

    code_hash = (
        h[:NUM_CHARS_DYNAMIC_KEY] if (h := hash_objects(code_obj)) else "unknown"
    )

    return f"{qualname}-{code_hash}"

class TaskRunNameCallbackWithParameters(Protocol):
    @classmethod
    def is_callback_with_parameters(cls, callable: Callable[..., str]) -> TypeIs[Self]:
        sig = inspect.signature(callable)
        return "parameters" in sig.parameters

    def __call__(self, parameters: dict[str, Any]) -> str: ...

StateHookCallable: TypeAlias = Callable[
    ["Task[..., Any]", TaskRun, State], Union[Awaitable[None], None]
]
TaskRunNameValueOrCallable: TypeAlias = Union[
    Callable[[], str], TaskRunNameCallbackWithParameters, str
]

class Task(Generic[P, R]):
    """
    A Prefect task definition.
    """
    def __init__(
        self,
        fn: Callable[P, R],
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        version: Optional[str] = None,
        cache_policy: Union[CachePolicy, type[NotSet]] = NotSet,
        cache_key_fn: Optional[
            Callable[["TaskRunContext", dict[str, Any]], Optional[str]]
        ] = None,
        cache_expiration: Optional[datetime.timedelta] = None,
        task_run_name: Optional[TaskRunNameValueOrCallable] = None,
        retries: Optional[int] = None,
        retry_delay_seconds: Optional[
            Union[
                float,
                int,
                list[float],
                Callable[[int], list[float]],
            ]
        ] = None,
        retry_jitter_factor: Optional[float] = None,
        persist_result: Optional[bool] = None,
        result_storage: Optional[ResultStorage] = None,
        result_serializer: Optional[ResultSerializer] = None,
        result_storage_key: Optional[str] = None,
        cache_result_in_memory: bool = True,
        timeout_seconds: Union[int, float, None] = None,
        log_prints: Optional[bool] = False,
        refresh_cache: Optional[bool] = None,
        on_completion: Optional[list[StateHookCallable]] = None,
        on_failure: Optional[list[StateHookCallable]] = None,
        on_rollback: Optional[list[Callable[["Transaction"], None]]] = None,
        on_commit: Optional[list[Callable[["Transaction"], None]]] = None,
        retry_condition_fn: Optional[
            Callable[["Task[..., Any]", TaskRun, State], bool]
        ] = None,
        viz_return_value: Optional[Any] = None,
    ):
        self.description = description or inspect.getdoc(fn)
        update_wrapper(self, fn)
        self.fn = fn
        self.isasync: bool = asyncio.iscoroutinefunction(
            self.fn
        ) or inspect.isasyncgenfunction(self.fn)
        self.isgenerator: bool = inspect.isgeneratorfunction(
            self.fn
        ) or inspect.isasyncgenfunction(self.fn)
        self.name = name if name else getattr(self.fn, "__name__", type(self.fn).__name__)
        self.task_run_name = task_run_name
        self.version = version
        self.log_prints = log_prints
        self.tags = set(tags if tags else [])
        self.task_key = _generate_task_key(self.fn)
        self.cache_key_fn = cache_key_fn
        self.cache_expiration = cache_expiration
        self.refresh_cache = refresh_cache
        self.retries = retries if retries is not None else get_current_settings().tasks.default_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.retry_jitter_factor = retry_jitter_factor
        self.persist_result = persist_result
        self.result_storage = result_storage
        self.result_serializer = result_serializer
        self.result_storage_key = result_storage_key
        self.cache_result_in_memory = cache_result_in_memory
        self.timeout_seconds = float(timeout_seconds) if timeout_seconds else None
        self.on_rollback_hooks = on_rollback or []
        self.on_commit_hooks = on_commit or []
        self.on_completion_hooks = on_completion or []
        self.on_failure_hooks = on_failure or []
        self.retry_condition_fn = retry_condition_fn
        self.viz_return_value = viz_return_value

    @property
    def ismethod(self) -> bool:
        return hasattr(self.fn, "__prefect_self__")

    def __get__(self, instance: Any, owner: Any) -> "Task[P, R]":
        if instance is None:
            return self
        else:
            bound_task = copy(self)
            bound_task.fn.__prefect_self__ = instance
            return bound_task

    def with_options(
        self,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        cache_policy: Union[CachePolicy, type[NotSet]] = NotSet,
        cache_key_fn: Optional[
            Callable[["TaskRunContext", dict[str, Any]], Optional[str]]
        ] = None,
        task_run_name: Optional[
            Union[TaskRunNameValueOrCallable, type[NotSet]]
        ] = NotSet,
        cache_expiration: Optional[datetime.timedelta] = None,
        retries: Union[int, type[NotSet]] = NotSet,
        retry_delay_seconds: Union[
            float,
            int,
            list[float],
            Callable[[int], list[float]],
            type[NotSet],
        ] = NotSet,
        retry_jitter_factor: Union[float, type[NotSet]] = NotSet,
        persist_result: Union[bool, type[NotSet]] = NotSet,
        result_storage: Union[ResultStorage, type[NotSet]] = NotSet,
        result_serializer: Union[ResultSerializer, type[NotSet]] = NotSet,
        result_storage_key: Union[str, type[NotSet]] = NotSet,
        cache_result_in_memory: Optional[bool] = None,
        timeout_seconds: Union[int, float, None] = None,
        log_prints: Union[bool, type[NotSet]] = NotSet,
        refresh_cache: Union[bool, type[NotSet]] = NotSet,
        on_completion: Optional[list[StateHookCallable]] = None,
        on_failure: Optional[list[StateHookCallable]] = None,
        retry_condition_fn: Optional[
            Callable[["Task[..., Any]", TaskRun, State], bool]
        ] = None,
        viz_return_value: Optional[Any] = None,
    ) -> "Task[P, R]":
        return Task(
            fn=self.fn,
            name=name or self.name,
            description=description or self.description,
            tags=tags or copy(self.tags),
            cache_policy=cache_policy if cache_policy is not NotSet else self.cache_policy,
            cache_key_fn=cache_key_fn or self.cache_key_fn,
            cache_expiration=cache_expiration or self.cache_expiration,
            task_run_name=task_run_name if task_run_name is not NotSet else self.task_run_name,
            retries=retries if retries is not NotSet else self.retries,
            retry_delay_seconds=(
                retry_delay_seconds
                if retry_delay_seconds is not NotSet
                else self.retry_delay_seconds
            ),
            retry_jitter_factor=(
                retry_jitter_factor
                if retry_jitter_factor is not NotSet
                else self.retry_jitter_factor
            ),
            persist_result=(
                persist_result if persist_result is not NotSet else self.persist_result
            ),
            result_storage=(
                result_storage if result_storage is not NotSet else self.result_storage
            ),
            result_storage_key=(
                result_storage_key
                if result_storage_key is not NotSet
                else self.result_storage_key
            ),
            result_serializer=(
                result_serializer
                if result_serializer is not NotSet
                else self.result_serializer
            ),
            cache_result_in_memory=(
                cache_result_in_memory
                if cache_result_in_memory is not None
                else self.cache_result_in_memory
            ),
            timeout_seconds=(
                timeout_seconds if timeout_seconds is not None else self.timeout_seconds
            ),
            log_prints=(log_prints if log_prints is not NotSet else self.log_prints),
            refresh_cache=(
                refresh_cache if refresh_cache is not NotSet else self.refresh_cache
            ),
            on_completion=on_completion or self.on_completion_hooks,
            on_failure=on_failure or self.on_failure_hooks,
            retry_condition_fn=retry_condition_fn or self.retry_condition_fn,
            viz_return_value=viz_return_value or self.viz_return_value,
        )

    def on_completion(self, fn: StateHookCallable) -> StateHookCallable:
        self.on_completion_hooks.append(fn)
        return fn

    def on_failure(self, fn: StateHookCallable) -> StateHookCallable:
        self.on_failure_hooks.append(fn)
        return fn

    def on_commit(
        self, fn: Callable[["Transaction"], None]
    ) -> Callable[["Transaction"], None]:
        self.on_commit_hooks.append(fn)
        return fn

    def on_rollback(
        self, fn: Callable[["Transaction"], None]
    ) -> Callable[["Transaction"], None]:
        self.on_rollback_hooks.append(fn)
        return fn

    async def create_run(
        self,
        client: Optional["PrefectClient"] = None,
        id: Optional[UUID] = None,
        parameters: Optional[dict[str, Any]] = None,
        flow_run_context: Optional[FlowRunContext] = None,
        parent_task_run_context: Optional[TaskRunContext] = None,
        wait_for: Optional[OneOrManyFutureOrResult[Any]] = None,
        extra_task_inputs: Optional[dict[str, set[TaskRunInput]]] = None,
        deferred: bool = False,
    ) -> TaskRun:
        if flow_run_context is None:
            flow_run_context = FlowRunContext.get()
        if parent_task_run_context is None:
            parent_task_run_context = TaskRunContext.get()
        if parameters is None:
            parameters = {}
        if client is None:
            client = get_client()

        async with client:
            if not flow_run_context:
                dynamic_key = f"{self.task_key}-{str(uuid4().hex)}"
                task_run_name = self.name
            else:
                from prefect.utilities._engine import dynamic_key_for_task_run
                dynamic_key = dynamic_key_for_task_run(
                    context=flow_run_context, task=self
                )
                task_run_name = f"{self.name}-{dynamic_key}"

            if deferred:
                state = Scheduled()
                state.state_details.deferred = True
            else:
                state = Pending()

            if deferred and (parameters or wait_for):
                parameters_id = uuid4()
                state.state_details.task_parameters_id = parameters_id
                self.persist_result = True
                store = await ResultStore(
                    result_storage=await get_or_create_default_task_scheduling_storage()
                ).update_for_task(self)
                context = serialize_context()
                data: dict[str, Any] = {"context": context}
                if parameters:
                    data["parameters"] = parameters
                if wait_for:
                    data["wait_for"] = wait_for
                await store.store_parameters(parameters_id, data)

            from prefect.utilities.engine import collect_task_run_inputs_sync
           