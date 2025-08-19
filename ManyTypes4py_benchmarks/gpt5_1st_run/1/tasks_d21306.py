"""
Module containing the base workflow task class and decorator - for most use cases, using the [`@task` decorator][prefect.tasks.task] is preferred.
"""
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
from prefect.context import FlowRunContext, TagsContext, TaskRunContext, serialize_context
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
from prefect.utilities.annotations import NotSet, NotSetType
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

T = TypeVar("T")
R = TypeVar("R")
P = ParamSpec("P")

NUM_CHARS_DYNAMIC_KEY: int = 8
logger = get_logger("tasks")

FutureOrResult = Union[PrefectFuture[T], T]
OneOrManyFutureOrResult = Union[FutureOrResult[T], Iterable[FutureOrResult[T]]]


def task_input_hash(
    context: TaskRunContext, arguments: dict[str, Any]
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
        context.task.task_key, context.task.fn.__code__.co_code.hex(), arguments
    )


def exponential_backoff(
    backoff_factor: Union[int, float],
) -> Callable[[int], list[float]]:
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
        return [float(backoff_factor) * max(0, 2**r) for r in range(retries)]

    return retry_backoff_callable


def _infer_parent_task_runs(
    flow_run_context: Optional[FlowRunContext],
    task_run_context: Optional[TaskRunContext],
    parameters: dict[str, Any],
) -> list[TaskRunResult]:
    """
    Attempt to infer the parent task runs for this task run based on the
    provided flow run and task run contexts, as well as any parameters. It is
    assumed that the task run is running within those contexts.
    If any parameter comes from a running task run, that task run is considered
    a parent. This is expected to happen when task inputs are yielded from
    generator tasks.
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
                upstream_state: Optional[State[Any]] = v
            elif isinstance(v, PrefectFuture):
                upstream_state = v.state
            else:
                upstream_state = flow_run_context.task_run_results.get(id(v))
            if upstream_state and upstream_state.is_running():
                parents.append(
                    TaskRunResult(id=upstream_state.state_details.task_run_id)
                )
    return parents


def _generate_task_key(fn: Any) -> str:
    """Generate a task key based on the function name and source code.

    We may eventually want some sort of top-level namespace here to
    disambiguate tasks with the same function name in different modules,
    in a more human-readable way, while avoiding relative import problems (see #12337).

    As long as the task implementations are unique (even if named the same), we should
    not have any collisions.

    Args:
        fn: The function to generate a task key for.
    """
    if not hasattr(fn, "__qualname__"):
        return to_qualified_name(type(fn))
    qualname = fn.__qualname__.split(".")[-1]
    try:
        code_obj = getattr(fn, "__code__", None)
        if code_obj is None:
            code_obj = fn.__call__.__code__  # type: ignore[attr-defined]
    except AttributeError:
        raise AttributeError(
            f"{fn} is not a standard Python function object and could not be converted to a task."
        ) from None
    code_hash = h[:NUM_CHARS_DYNAMIC_KEY] if (h := hash_objects(code_obj)) else "unknown"
    return f"{qualname}-{code_hash}"


class TaskRunNameCallbackWithParameters(Protocol):
    @classmethod
    def is_callback_with_parameters(cls, callable: Callable[..., Any]) -> bool:
        sig = inspect.signature(callable)
        return "parameters" in sig.parameters

    def __call__(self, parameters: dict[str, Any]) -> str: ...


StateHookCallable = Callable[["Task[..., Any]", TaskRun, State[Any]], Union[Awaitable[None], None]]
TaskRunNameValueOrCallable = Union[Callable[[], str], TaskRunNameCallbackWithParameters, str]


class Task(Generic[P, R]):
    """
    A Prefect task definition.

    !!! note
        We recommend using [the `@task` decorator][prefect.tasks.task] for most use-cases.

    Wraps a function with an entrypoint to the Prefect engine. Calling this class within a flow function
    creates a new task run.

    To preserve the input and output types, we use the generic type variables P and R for "Parameters" and
    "Returns" respectively.

    Args:
        fn: The function defining the task.
        name: An optional name for the task; if not provided, the name will be inferred
            from the given function.
        description: An optional string description for the task.
        tags: An optional set of tags to be associated with runs of this task. These
            tags are combined with any tags defined by a `prefect.tags` context at
            task runtime.
        version: An optional string specifying the version of this task definition
        cache_policy: A cache policy that determines the level of caching for this task
        cache_key_fn: An optional callable that, given the task run context and call
            parameters, generates a string key; if the key matches a previous completed
            state, that state result will be restored instead of running the task again.
        cache_expiration: An optional amount of time indicating how long cached states
            for this task should be restorable; if not provided, cached states will
            never expire.
        task_run_name: An optional name to distinguish runs of this task; this name can be provided
            as a string template with the task's keyword arguments as variables,
            or a function that returns a string.
        retries: An optional number of times to retry on task run failure.
        retry_delay_seconds: Optionally configures how long to wait before retrying the
            task after failure. This is only applicable if `retries` is nonzero. This
            setting can either be a number of seconds, a list of retry delays, or a
            callable that, given the total number of retries, generates a list of retry
            delays. If a number of seconds, that delay will be applied to all retries.
            If a list, each retry will wait for the corresponding delay before retrying.
            When passing a callable or a list, the number of configured retry delays
            cannot exceed 50.
        retry_jitter_factor: An optional factor that defines the factor to which a retry
            can be jittered in order to avoid a "thundering herd".
        persist_result: A toggle indicating whether the result of this task
            should be persisted to result storage. Defaults to `None`, which
            indicates that the global default should be used (which is `True` by
            default).
        result_storage: An optional block to use to persist the result of this task.
            Defaults to the value set in the flow the task is called in.
        result_storage_key: An optional key to store the result in storage at when persisted.
            Defaults to a unique identifier.
        result_serializer: An optional serializer to use to serialize the result of this
            task for persistence. Defaults to the value set in the flow the task is
            called in.
        timeout_seconds: An optional number of seconds indicating a maximum runtime for
            the task. If the task exceeds this runtime, it will be marked as failed.
        log_prints: If set, `print` statements in the task will be redirected to the
            Prefect logger for the task run. Defaults to `None`, which indicates
            that the value from the flow should be used.
        refresh_cache: If set, cached results for the cache key are not used.
            Defaults to `None`, which indicates that a cached result from a previous
            execution with matching cache key is used.
        on_failure: An optional list of callables to run when the task enters a failed state.
        on_completion: An optional list of callables to run when the task enters a completed state.
        on_commit: An optional list of callables to run when the task's idempotency record is committed.
        on_rollback: An optional list of callables to run when the task rolls back.
        retry_condition_fn: An optional callable run when a task run returns a Failed state. Should
            return `True` if the task should continue to its retry policy (e.g. `retries=3`), and `False` if the task
            should end as failed. Defaults to `None`, indicating the task should always continue
            to its retry policy.
        viz_return_value: An optional value to return when the task dependency tree is visualized.
    """

    fn: Callable[P, R]
    description: Optional[str]
    isasync: bool
    isgenerator: bool
    name: str
    task_run_name: Optional[TaskRunNameValueOrCallable]
    version: Optional[str]
    log_prints: Optional[bool]
    tags: set[str]
    task_key: str
    cache_key_fn: Optional[Callable[[TaskRunContext, dict[str, Any]], Optional[str]]]
    cache_expiration: Optional[datetime.timedelta]
    refresh_cache: Optional[bool]
    cache_policy: Optional[CachePolicy]
    retries: int
    retry_delay_seconds: Optional[Union[int, float, list[float]]]
    retry_jitter_factor: Optional[float]
    persist_result: Optional[bool]
    result_storage: Optional[Union[ResultStorage, str]]
    result_serializer: Optional[ResultSerializer]
    result_storage_key: Optional[str]
    cache_result_in_memory: bool
    timeout_seconds: Optional[float]
    on_rollback_hooks: list[Callable[..., Any]]
    on_commit_hooks: list[Callable[..., Any]]
    on_completion_hooks: list[StateHookCallable]
    on_failure_hooks: list[StateHookCallable]
    retry_condition_fn: Optional[Callable[[State[Any]], bool]]
    viz_return_value: Any

    def __init__(
        self,
        fn: Callable[P, R],
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        version: Optional[str] = None,
        cache_policy: Union[CachePolicy, NotSetType, None] = NotSet,
        cache_key_fn: Optional[
            Callable[[TaskRunContext, dict[str, Any]], Optional[str]]
        ] = None,
        cache_expiration: Optional[datetime.timedelta] = None,
        task_run_name: Optional[TaskRunNameValueOrCallable] = None,
        retries: Optional[int] = None,
        retry_delay_seconds: Optional[
            Union[int, float, list[float], Callable[[int], list[float]]]
        ] = None,
        retry_jitter_factor: Optional[float] = None,
        persist_result: Optional[bool] = None,
        result_storage: Optional[Union[ResultStorage, str]] = None,
        result_serializer: Optional[ResultSerializer] = None,
        result_storage_key: Optional[str] = None,
        cache_result_in_memory: bool = True,
        timeout_seconds: Optional[Union[int, float]] = None,
        log_prints: Optional[bool] = False,
        refresh_cache: Optional[bool] = None,
        on_completion: Optional[Iterable[StateHookCallable]] = None,
        on_failure: Optional[Iterable[StateHookCallable]] = None,
        on_rollback: Optional[Iterable[Callable[..., Any]]] = None,
        on_commit: Optional[Iterable[Callable[..., Any]]] = None,
        retry_condition_fn: Optional[Callable[[State[Any]], bool]] = None,
        viz_return_value: Any = None,
    ) -> None:
        hook_categories = [on_completion, on_failure]
        hook_names = ["on_completion", "on_failure"]
        for hooks, hook_name in zip(hook_categories, hook_names):
            if hooks is not None:
                try:
                    hooks = list(hooks)
                except TypeError:
                    raise TypeError(
                        f"Expected iterable for '{hook_name}'; got {type(hooks).__name__} instead. Please provide a list of hooks to '{hook_name}':\n\n@task({hook_name}=[hook1, hook2])\ndef my_task():\n\tpass"
                    )
                for hook in hooks:
                    if not callable(hook):
                        raise TypeError(
                            f"Expected callables in '{hook_name}'; got {type(hook).__name__} instead. Please provide a list of hooks to '{hook_name}':\n\n@task({hook_name}=[hook1, hook2])\ndef my_task():\n\tpass"
                        )
        if not callable(fn):
            raise TypeError("'fn' must be callable")
        self.description = description or inspect.getdoc(fn)
        update_wrapper(self, fn)
        self.fn = fn
        self.isasync = asyncio.iscoroutinefunction(self.fn) or inspect.isasyncgenfunction(self.fn)
        self.isgenerator = inspect.isgeneratorfunction(self.fn) or inspect.isasyncgenfunction(self.fn)
        if not name:
            if not hasattr(self.fn, "__name__"):
                self.name = type(self.fn).__name__
            else:
                self.name = self.fn.__name__
        else:
            self.name = name
        if task_run_name is not None:
            if not isinstance(task_run_name, str) and (not callable(task_run_name)):
                raise TypeError(
                    f"Expected string or callable for 'task_run_name'; got {type(task_run_name).__name__} instead."
                )
        self.task_run_name = task_run_name
        self.version = version
        self.log_prints = log_prints
        raise_for_reserved_arguments(self.fn, ["return_state", "wait_for"])
        self.tags = set(tags if tags else [])
        self.task_key = _generate_task_key(self.fn)
        if cache_policy is not NotSet and cache_key_fn is not None:
            logger.warning(
                f"Both `cache_policy` and `cache_key_fn` are set on task {self}. `cache_key_fn` will be used."
            )
        if cache_key_fn:
            cache_policy = CachePolicy.from_cache_key_fn(cache_key_fn)
        self.cache_key_fn = cache_key_fn
        self.cache_expiration = cache_expiration
        self.refresh_cache = refresh_cache
        if persist_result is None:
            if any(
                [
                    cache_policy and cache_policy != NO_CACHE and (cache_policy != NotSet),
                    cache_key_fn is not None,
                    result_storage_key is not None,
                    result_storage is not None,
                    result_serializer is not None,
                ]
            ):
                persist_result = True
        if persist_result is False:
            self.cache_policy = None if cache_policy is None else NO_CACHE
            if cache_policy and cache_policy is not NotSet and (cache_policy != NO_CACHE):
                logger.warning("Ignoring `cache_policy` because `persist_result` is False")
        elif cache_policy is NotSet and result_storage_key is None:
            self.cache_policy = DEFAULT
        elif result_storage_key:
            self.cache_policy = None
        else:
            self.cache_policy = cast(Optional[CachePolicy], cache_policy)
        settings = get_current_settings()
        self.retries = retries if retries is not None else settings.tasks.default_retries
        if retry_delay_seconds is None:
            retry_delay_seconds = settings.tasks.default_retry_delay_seconds
        if callable(retry_delay_seconds):
            self.retry_delay_seconds = retry_delay_seconds(self.retries)
        elif not isinstance(retry_delay_seconds, (list, int, float, type(None))):
            raise TypeError(
                f"Invalid `retry_delay_seconds` provided; must be an int, float, list or callable. Received type {type(retry_delay_seconds)}"
            )
        else:
            self.retry_delay_seconds = cast(Optional[Union[int, float, list[float]]], retry_delay_seconds)
        if isinstance(self.retry_delay_seconds, list) and len(self.retry_delay_seconds) > 50:
            raise ValueError("Can not configure more than 50 retry delays per task.")
        if retry_jitter_factor is not None and retry_jitter_factor < 0:
            raise ValueError("`retry_jitter_factor` must be >= 0.")
        self.retry_jitter_factor = retry_jitter_factor
        self.persist_result = persist_result
        if result_storage and (not isinstance(result_storage, str)):
            if getattr(result_storage, "_block_document_id", None) is None:
                raise TypeError(
                    "Result storage configuration must be persisted server-side. Please call `.save()` on your block before passing it in."
                )
        self.result_storage = result_storage
        self.result_serializer = result_serializer
        self.result_storage_key = result_storage_key
        self.cache_result_in_memory = cache_result_in_memory
        self.timeout_seconds = float(timeout_seconds) if timeout_seconds else None
        self.on_rollback_hooks = list(on_rollback or [])
        self.on_commit_hooks = list(on_commit or [])
        self.on_completion_hooks = list(on_completion or [])
        self.on_failure_hooks = list(on_failure or [])
        if retry_condition_fn is not None and (not callable(retry_condition_fn)):
            raise TypeError(
                f"Expected `retry_condition_fn` to be callable, got {type(retry_condition_fn).__name__} instead."
            )
        self.retry_condition_fn = retry_condition_fn
        self.viz_return_value = viz_return_value

    @property
    def ismethod(self) -> bool:
        return hasattr(self.fn, "__prefect_self__")

    def __get__(self, instance: Any, owner: type) -> "Task[P, R]":
        """
        Implement the descriptor protocol so that the task can be used as an instance method.
        When an instance method is loaded, this method is called with the "self" instance as
        an argument. We return a copy of the task with that instance bound to the task's function.
        """
        if instance is None:
            return self
        else:
            bound_task = copy(self)
            bound_task.fn.__prefect_self__ = instance  # type: ignore[attr-defined]
            return bound_task

    def with_options(
        self,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        cache_policy: Union[CachePolicy, NotSetType, None] = NotSet,
        cache_key_fn: Optional[
            Callable[[TaskRunContext, dict[str, Any]], Optional[str]]
        ] = None,
        task_run_name: Union[TaskRunNameValueOrCallable, NotSetType] = NotSet,
        cache_expiration: Optional[datetime.timedelta] = None,
        retries: Union[int, NotSetType] = NotSet,
        retry_delay_seconds: Union[
            int, float, list[float], Callable[[int], list[float]], NotSetType
        ] = NotSet,
        retry_jitter_factor: Union[float, NotSetType] = NotSet,
        persist_result: Union[bool, NotSetType] = NotSet,
        result_storage: Union[ResultStorage, str, NotSetType] = NotSet,
        result_serializer: Union[ResultSerializer, NotSetType] = NotSet,
        result_storage_key: Union[str, NotSetType] = NotSet,
        cache_result_in_memory: Optional[bool] = None,
        timeout_seconds: Optional[Union[int, float]] = None,
        log_prints: Union[bool, None, NotSetType] = NotSet,
        refresh_cache: Union[bool, None, NotSetType] = NotSet,
        on_completion: Optional[Iterable[StateHookCallable]] = None,
        on_failure: Optional[Iterable[StateHookCallable]] = None,
        retry_condition_fn: Optional[Callable[[State[Any]], bool]] = None,
        viz_return_value: Any = None,
    ) -> "Task[P, R]":
        """
        Create a new task from the current object, updating provided options.
        """
        return Task(
            fn=self.fn,
            name=name or self.name,
            description=description or self.description,
            tags=tags or copy(self.tags),
            cache_policy=cache_policy if cache_policy is not NotSet else self.cache_policy,
            cache_key_fn=cache_key_fn or self.cache_key_fn,
            cache_expiration=cache_expiration or self.cache_expiration,
            task_run_name=self.task_run_name if task_run_name is NotSet else cast(Optional[TaskRunNameValueOrCallable], task_run_name),
            retries=self.retries if retries is NotSet else cast(int, retries),
            retry_delay_seconds=self.retry_delay_seconds
            if retry_delay_seconds is NotSet
            else cast(Union[int, float, list[float], Callable[[int], list[float]]], retry_delay_seconds),
            retry_jitter_factor=self.retry_jitter_factor
            if retry_jitter_factor is NotSet
            else cast(Optional[float], retry_jitter_factor),
            persist_result=self.persist_result if persist_result is NotSet else cast(Optional[bool], persist_result),
            result_storage=self.result_storage if result_storage is NotSet else cast(Optional[Union[ResultStorage, str]], result_storage),
            result_storage_key=self.result_storage_key if result_storage_key is NotSet else cast(Optional[str], result_storage_key),
            result_serializer=self.result_serializer if result_serializer is NotSet else cast(Optional[ResultSerializer], result_serializer),
            cache_result_in_memory=self.cache_result_in_memory
            if cache_result_in_memory is None
            else cache_result_in_memory,
            timeout_seconds=self.timeout_seconds
            if timeout_seconds is None
            else timeout_seconds,
            log_prints=self.log_prints if log_prints is NotSet else cast(Optional[bool], log_prints),
            refresh_cache=self.refresh_cache if refresh_cache is NotSet else cast(Optional[bool], refresh_cache),
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

    def on_commit(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        self.on_commit_hooks.append(fn)
        return fn

    def on_rollback(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        self.on_rollback_hooks.append(fn)
        return fn

    async def create_run(
        self,
        client: Optional["PrefectClient"] = None,
        id: Optional[UUID] = None,
        parameters: Optional[dict[str, Any]] = None,
        flow_run_context: Optional[FlowRunContext] = None,
        parent_task_run_context: Optional[TaskRunContext] = None,
        wait_for: Optional[Iterable[Any]] = None,
        extra_task_inputs: Optional[dict[str, set[TaskRunInput]]] = None,
        deferred: bool = False,
    ) -> TaskRun:
        from prefect.utilities._engine import dynamic_key_for_task_run
        from prefect.utilities.engine import collect_task_run_inputs_sync

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
                dynamic_key = dynamic_key_for_task_run(context=flow_run_context, task=self)
                task_run_name = f"{self.name}-{dynamic_key}"

            if deferred:
                state: State[Any] = Scheduled()
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

            task_inputs: dict[str, set[TaskRunInput]] = {
                k: collect_task_run_inputs_sync(v) for k, v in parameters.items()
            }

            if task_parents := _infer_parent_task_runs(
                flow_run_context=flow_run_context,
                task_run_context=parent_task_run_context,
                parameters=parameters,
            ):
                task_inputs["__parents__"] = task_parents  # type: ignore[assignment]

            if wait_for:
                task_inputs["wait_for"] = collect_task_run_inputs_sync(wait_for)  # type: ignore[assignment]

            for k, extras in (extra_task_inputs or {}).items():
                task_inputs[k] = task_inputs[k].union(extras)

            task_run = client.create_task_run(
                task=self,
                name=task_run_name,
                flow_run_id=getattr(flow_run_context.flow_run, "id", None)
                if flow_run_context and flow_run_context.flow_run
                else None,
                dynamic_key=str(dynamic_key),
                id=id,
                state=state,
                task_inputs=task_inputs,
                extra_tags=TagsContext.get().current_tags,
            )
            if inspect.isawaitable(task_run):
                task_run = await task_run
            return cast(TaskRun, task_run)

    async def create_local_run(
        self,
        client: Optional["PrefectClient"] = None,
        id: Optional[UUID] = None,
        parameters: Optional[dict[str, Any]] = None,
        flow_run_context: Optional[FlowRunContext] = None,
        parent_task_run_context: Optional[TaskRunContext] = None,
        wait_for: Optional[Iterable[Any]] = None,
        extra_task_inputs: Optional[dict[str, set[TaskRunInput]]] = None,
        deferred: bool = False,
    ) -> TaskRun:
        from prefect.utilities._engine import dynamic_key_for_task_run
        from prefect.utilities.engine import collect_task_run_inputs_sync

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
                dynamic_key = dynamic_key_for_task_run(
                    context=flow_run_context, task=self, stable=False
                )
                task_run_name = f"{self.name}-{dynamic_key[:3]}"

            if deferred:
                state: State[Any] = Scheduled()
                state.state_details.deferred = True
            else:
                state = Pending()

            if deferred and (parameters or wait_for):
                parameters_id = uuid4()
                state.state_details.task_parameters_id = parameters_id
                self.persist_result = True
                store = await ResultStore(
                    result_storage=await get_or_create_default_task_scheduling_storage()
                ).update_for_task(task)  # type: ignore[name-defined]
                context = serialize_context()
                data: dict[str, Any] = {"context": context}
                if parameters:
                    data["parameters"] = parameters
                if wait_for:
                    data["wait_for"] = wait_for
                await store.store_parameters(parameters_id, data)

            task_inputs: dict[str, set[TaskRunInput]] = {
                k: collect_task_run_inputs_sync(v) for k, v in parameters.items()
            }
            if task_parents := _infer_parent_task_runs(
                flow_run_context=flow_run_context,
                task_run_context=parent_task_run_context,
                parameters=parameters,
            ):
                task_inputs["__parents__"] = task_parents  # type: ignore[assignment]
            if wait_for:
                task_inputs["wait_for"] = collect_task_run_inputs_sync(wait_for)  # type: ignore[assignment]
            for k, extras in (extra_task_inputs or {}).items():
                task_inputs[k] = task_inputs[k].union(extras)

            flow_run_id = (
                getattr(flow_run_context.flow_run, "id", None)
                if flow_run_context and flow_run_context.flow_run
                else None
            )
            task_run_id = id or uuid4()
            state = prefect.states.Pending(
                state_details=StateDetails(
                    task_run_id=task_run_id, flow_run_id=flow_run_id
                )
            )
            task_run = TaskRun(
                id=task_run_id,
                name=task_run_name,
                flow_run_id=flow_run_id,
                task_key=self.task_key,
                dynamic_key=str(dynamic_key),
                task_version=self.version,
                empirical_policy=TaskRunPolicy(
                    retries=self.retries,
                    retry_delay=self.retry_delay_seconds,  # type: ignore[arg-type]
                    retry_jitter_factor=self.retry_jitter_factor,
                ),
                tags=list(set(self.tags).union(TagsContext.get().current_tags or [])),
                task_inputs=task_inputs or {},
                expected_start_time=state.timestamp,
                state_id=state.id,
                state_type=state.type,
                state_name=state.name,
                state=state,
                created=state.timestamp,
                updated=state.timestamp,
            )
            return task_run

    @overload
    def __call__(
        self,
        *args: P.args,
        return_state: Literal[True],
        wait_for: Optional[Iterable[Any]] = ...,
        **kwargs: P.kwargs,
    ) -> State[R]:
        ...

    @overload
    def __call__(
        self,
        *args: P.args,
        return_state: Literal[False] = ...,
        wait_for: Optional[Iterable[Any]] = ...,
        **kwargs: P.kwargs,
    ) -> R:
        ...

    def __call__(
        self,
        *args: P.args,
        return_state: bool = False,
        wait_for: Optional[Iterable[Any]] = None,
        **kwargs: P.kwargs,
    ) -> Union[R, State[R]]:
        """
        Run the task and return the result. If `return_state` is True returns
        the result is wrapped in a Prefect State which provides error handling.
        """
        from prefect.utilities.visualization import (
            get_task_viz_tracker,
            track_viz_task,
        )

        parameters = get_call_parameters(self.fn, args, kwargs)
        return_type = "state" if return_state else "result"

        task_run_tracker = get_task_viz_tracker()
        if task_run_tracker:
            return track_viz_task(
                self.isasync, self.name, parameters, self.viz_return_value
            )
        from prefect.task_engine import run_task

        return run_task(
            task=self, parameters=parameters, wait_for=wait_for, return_type=return_type
        )

    @overload
    def submit(
        self,
        *args: P.args,
        return_state: Literal[True],
        wait_for: Optional[Iterable[Any]] = ...,
        **kwargs: P.kwargs,
    ) -> State[R]:
        ...

    @overload
    def submit(
        self,
        *args: P.args,
        return_state: Literal[False] = ...,
        wait_for: Optional[Iterable[Any]] = ...,
        **kwargs: P.kwargs,
    ) -> PrefectFuture[R]:
        ...

    def submit(
        self,
        *args: P.args,
        return_state: bool = False,
        wait_for: Optional[Iterable[Any]] = None,
        **kwargs: P.kwargs,
    ) -> Union[PrefectFuture[R], State[R]]:
        """
        Submit a run of the task to the engine.
        """
        from prefect.utilities.visualization import (
            VisualizationUnsupportedError,
            get_task_viz_tracker,
        )

        parameters = get_call_parameters(self.fn, args, kwargs)

        flow_run_context = FlowRunContext.get()
        if not flow_run_context:
            raise RuntimeError(
                "Unable to determine task runner to use for submission. If you are submitting a task outside of a flow, please use `.delay` to submit the task run for deferred execution."
            )

        task_viz_tracker = get_task_viz_tracker()
        if task_viz_tracker:
            raise VisualizationUnsupportedError(
                "`task.submit()` is not currently supported by `flow.visualize()`"
            )

        task_runner = flow_run_context.task_runner
        future: PrefectFuture[R] = task_runner.submit(self, parameters, wait_for)
        if return_state:
            future.wait()
            return future.state  # type: ignore[return-value]
        else:
            return future

    @overload
    def map(
        self,
        *args: P.args,
        return_state: Literal[True],
        wait_for: Optional[Iterable[Any]] = ...,
        deferred: bool = ...,
        **kwargs: P.kwargs,
    ) -> list[State[R]]:
        ...

    @overload
    def map(
        self,
        *args: P.args,
        return_state: Literal[False] = ...,
        wait_for: Optional[Iterable[Any]] = ...,
        deferred: bool = ...,
        **kwargs: P.kwargs,
    ) -> PrefectFutureList[R]:
        ...

    def map(
        self,
        *args: P.args,
        return_state: bool = False,
        wait_for: Optional[Iterable[Any]] = None,
        deferred: bool = False,
        **kwargs: P.kwargs,
    ) -> Union[PrefectFutureList[R], list[State[R]]]:
        """
        Submit a mapped run of the task to a worker.
        """
        from prefect.task_runners import TaskRunner
        from prefect.utilities.visualization import (
            VisualizationUnsupportedError,
            get_task_viz_tracker,
        )

        parameters = get_call_parameters(self.fn, args, kwargs, apply_defaults=False)
        flow_run_context = FlowRunContext.get()

        task_viz_tracker = get_task_viz_tracker()
        if task_viz_tracker:
            raise VisualizationUnsupportedError(
                "`task.map()` is not currently supported by `flow.visualize()`"
            )

        if deferred:
            parameters_list = expand_mapping_parameters(self.fn, parameters)
            futures: PrefectFutureList[R] = PrefectFutureList(
                [self.apply_async(kwargs=parameters, wait_for=wait_for) for parameters in parameters_list]  # type: ignore[list-item]
            )
        elif (task_runner := getattr(flow_run_context, "task_runner", None)):
            assert isinstance(task_runner, TaskRunner)
            futures = task_runner.map(self, parameters, wait_for)  # type: ignore[assignment]
        else:
            raise RuntimeError(
                "Unable to determine task runner to use for mapped task runs. If you are mapping a task outside of a flow, please provide `deferred=True` to submit the mapped task runs for deferred execution."
            )

        if return_state:
            states: list[State[R]] = []
            for future in futures:
                future.wait()
                states.append(future.state)  # type: ignore[arg-type]
            return states
        else:
            return futures

    def apply_async(
        self,
        args: Optional[tuple[Any, ...]] = None,
        kwargs: Optional[dict[str, Any]] = None,
        wait_for: Optional[Iterable[Any]] = None,
        dependencies: Optional[dict[str, set[TaskRunInput]]] = None,
    ) -> PrefectDistributedFuture:
        """
        Create a pending task run for a task worker to execute.
        """
        from prefect.utilities.visualization import (
            VisualizationUnsupportedError,
            get_task_viz_tracker,
        )

        task_viz_tracker = get_task_viz_tracker()
        if task_viz_tracker:
            raise VisualizationUnsupportedError(
                "`task.apply_async()` is not currently supported by `flow.visualize()`"
            )

        args = args or ()
        kwargs = kwargs or {}

        parameters = get_call_parameters(self.fn, args, kwargs)

        task_run = run_coro_as_sync(
            self.create_run(
                parameters=parameters,
                deferred=True,
                wait_for=wait_for,
                extra_task_inputs=dependencies,
            )
        )
        from prefect.utilities.engine import emit_task_run_state_change_event

        emit_task_run_state_change_event(
            task_run=task_run, initial_state=None, validated_state=task_run.state
        )
        if (task_run_url := url_for(task_run)):
            logger.info(
                f"Created task run {task_run.name!r}. View it in the UI at {task_run_url!r}"
            )
        return PrefectDistributedFuture(task_run_id=task_run.id)

    def delay(self, *args: P.args, **kwargs: P.kwargs) -> PrefectDistributedFuture:
        """
        An alias for `apply_async` with simpler calling semantics.
        """
        return self.apply_async(args=args, kwargs=kwargs)

    @sync_compatible
    async def serve(self) -> None:
        """Serve the task using the provided task runner. This method is used to
        establish a websocket connection with the Prefect server and listen for
        submitted task runs to execute.
        """
        from prefect.task_worker import serve

        await serve(self)


@overload
def task(__fn: Callable[P, R]) -> Task[P, R]:
    ...


@overload
def task(
    __fn: None = ...,
    *,
    name: Optional[str] = ...,
    description: Optional[str] = ...,
    tags: Optional[Iterable[str]] = ...,
    version: Optional[str] = ...,
    cache_policy: Union[CachePolicy, NotSetType, None] = ...,
    cache_key_fn: Optional[Callable[[TaskRunContext, dict[str, Any]], Optional[str]]] = ...,
    cache_expiration: Optional[datetime.timedelta] = ...,
    task_run_name: Optional[TaskRunNameValueOrCallable] = ...,
    retries: Optional[int] = ...,
    retry_delay_seconds: Optional[
        Union[int, float, list[float], Callable[[int], list[float]]]
    ] = ...,
    retry_jitter_factor: Optional[float] = ...,
    persist_result: Optional[bool] = ...,
    result_storage: Optional[Union[ResultStorage, str]] = ...,
    result_storage_key: Optional[str] = ...,
    result_serializer: Optional[ResultSerializer] = ...,
    cache_result_in_memory: bool = ...,
    timeout_seconds: Optional[Union[int, float]] = ...,
    log_prints: Optional[bool] = ...,
    refresh_cache: Optional[bool] = ...,
    on_completion: Optional[Iterable[StateHookCallable]] = ...,
    on_failure: Optional[Iterable[StateHookCallable]] = ...,
    retry_condition_fn: Optional[Callable[[State[Any]], bool]] = ...,
    viz_return_value: Any = ...,
) -> Callable[[Callable[P, R]], Task[P, R]]:
    ...


def task(
    __fn: Optional[Callable[P, R]] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[Iterable[str]] = None,
    version: Optional[str] = None,
    cache_policy: Union[CachePolicy, NotSetType, None] = NotSet,
    cache_key_fn: Optional[Callable[[TaskRunContext, dict[str, Any]], Optional[str]]] = None,
    cache_expiration: Optional[datetime.timedelta] = None,
    task_run_name: Optional[TaskRunNameValueOrCallable] = None,
    retries: Optional[int] = None,
    retry_delay_seconds: Optional[
        Union[int, float, list[float], Callable[[int], list[float]]]
    ] = None,
    retry_jitter_factor: Optional[float] = None,
    persist_result: Optional[bool] = None,
    result_storage: Optional[Union[ResultStorage, str]] = None,
    result_storage_key: Optional[str] = None,
    result_serializer: Optional[ResultSerializer] = None,
    cache_result_in_memory: bool = True,
    timeout_seconds: Optional[Union[int, float]] = None,
    log_prints: Optional[bool] = None,
    refresh_cache: Optional[bool] = None,
    on_completion: Optional[Iterable[StateHookCallable]] = None,
    on_failure: Optional[Iterable[StateHookCallable]] = None,
    retry_condition_fn: Optional[Callable[[State[Any]], bool]] = None,
    viz_return_value: Any = None,
) -> Union[Task[P, R], Callable[[Callable[P, R]], Task[P, R]]]:
    """
    Decorator to designate a function as a task in a Prefect workflow.
    """
    if __fn:
        if isinstance(__fn, (classmethod, staticmethod)):
            method_decorator = type(__fn).__name__
            raise TypeError(f"@{method_decorator} should be applied on top of @task")
        return Task(
            fn=__fn,
            name=name,
            description=description,
            tags=tags,
            version=version,
            cache_policy=cache_policy,
            cache_key_fn=cache_key_fn,
            cache_expiration=cache_expiration,
            task_run_name=task_run_name,
            retries=retries,
            retry_delay_seconds=retry_delay_seconds,
            retry_jitter_factor=retry_jitter_factor,
            persist_result=persist_result,
            result_storage=result_storage,
            result_storage_key=result_storage_key,
            result_serializer=result_serializer,
            cache_result_in_memory=cache_result_in_memory,
            timeout_seconds=timeout_seconds,
            log_prints=log_prints,
            refresh_cache=refresh_cache,
            on_completion=on_completion,
            on_failure=on_failure,
            retry_condition_fn=retry_condition_fn,
            viz_return_value=viz_return_value,
        )
    else:
        return cast(
            Callable[[Callable[P, R]], Task[P, R]],
            partial(
                task,
                name=name,
                description=description,
                tags=tags,
                version=version,
                cache_policy=cache_policy,
                cache_key_fn=cache_key_fn,
                cache_expiration=cache_expiration,
                task_run_name=task_run_name,
                retries=retries,
                retry_delay_seconds=retry_delay_seconds,
                retry_jitter_factor=retry_jitter_factor,
                persist_result=persist_result,
                result_storage=result_storage,
                result_storage_key=result_storage_key,
                result_serializer=result_serializer,
                cache_result_in_memory=cache_result_in_memory,
                timeout_seconds=timeout_seconds,
                log_prints=log_prints,
                refresh_cache=refresh_cache,
                on_completion=on_completion,
                on_failure=on_failure,
                retry_condition_fn=retry_condition_fn,
                viz_return_value=viz_return_value,
            ),
        )