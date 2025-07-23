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
    Dict,
    List,
    Set,
    Tuple,
    Type,
)
from uuid import UUID, uuid4
from typing_extensions import Literal, ParamSpec, Self, TypeAlias, TypeIs
import prefect.states
from prefect.cache_policies import DEFAULT, NO_CACHE, CachePolicy
from prefect.client.orchestration import get_client
from prefect.client.schemas import TaskRun
from prefect.client.schemas.objects import StateDetails, TaskRunInput, TaskRunPolicy, TaskRunResult
from prefect.context import FlowRunContext, TagsContext, TaskRunContext, serialize_context
from prefect.futures import PrefectDistributedFuture, PrefectFuture, PrefectFutureList
from prefect.logging.loggers import get_logger
from prefect.results import ResultSerializer, ResultStorage, ResultStore, get_or_create_default_task_scheduling_storage
from prefect.settings.context import get_current_settings
from prefect.states import Pending, Scheduled, State
from prefect.utilities.annotations import NotSet
from prefect.utilities.asyncutils import run_coro_as_sync, sync_compatible
from prefect.utilities.callables import expand_mapping_parameters, get_call_parameters, raise_for_reserved_arguments
from prefect.utilities.hashing import hash_objects
from prefect.utilities.importtools import to_qualified_name
from prefect.utilities.urls import url_for

if TYPE_CHECKING:
    import logging
    from prefect.client.orchestration import PrefectClient
    from prefect.context import TaskRunContext
    from prefect.transactions import Transaction

T = TypeVar('T')
R = TypeVar('R')
P = ParamSpec('P')
NUM_CHARS_DYNAMIC_KEY = 8
logger: 'logging.Logger' = get_logger('tasks')
FutureOrResult = Union[PrefectFuture[T], T]
OneOrManyFutureOrResult = Union[FutureOrResult[T], Iterable[FutureOrResult[T]]]

def task_input_hash(context: 'TaskRunContext', arguments: Dict[str, Any]) -> Optional[str]:
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
    return hash_objects(context.task.task_key, context.task.fn.__code__.co_code.hex(), arguments)

def exponential_backoff(backoff_factor: float) -> Callable[[int], List[float]]:
    """
    A task retry backoff utility that configures exponential backoff for task retries.
    The exponential backoff design matches the urllib3 implementation.

    Arguments:
        backoff_factor: the base delay for the first retry, subsequent retries will
            increase the delay time by powers of 2.

    Returns:
        a callable that can be passed to the task constructor
    """
    def retry_backoff_callable(retries: int) -> List[float]:
        retries = min(retries, 50)
        return [backoff_factor * max(0, 2 ** r) for r in range(retries)]
    return retry_backoff_callable

def _infer_parent_task_runs(
    flow_run_context: Optional[FlowRunContext],
    task_run_context: Optional[TaskRunContext],
    parameters: Dict[str, Any]
) -> List[TaskRunResult]:
    """
    Attempt to infer the parent task runs for this task run based on the
    provided flow run and task run contexts, as well as any parameters. It is
    assumed that the task run is running within those contexts.
    If any parameter comes from a running task run, that task run is considered
    a parent. This is expected to happen when task inputs are yielded from
    generator tasks.
    """
    parents: List[TaskRunResult] = []
    if task_run_context:
        if not flow_run_context:
            parents.append(TaskRunResult(id=task_run_context.task_run.id))
        elif flow_run_context and task_run_context.task_run.flow_run_id == getattr(flow_run_context.flow_run, 'id', None):
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
                parents.append(TaskRunResult(id=upstream_state.state_details.task_run_id))
    return parents

def _generate_task_key(fn: Callable[..., Any]) -> str:
    """Generate a task key based on the function name and source code.

    We may eventually want some sort of top-level namespace here to
    disambiguate tasks with the same function name in different modules,
    in a more human-readable way, while avoiding relative import problems (see #12337).

    As long as the task implementations are unique (even if named the same), we should
    not have any collisions.

    Args:
        fn: The function to generate a task key for.
    """
    if not hasattr(fn, '__qualname__'):
        return to_qualified_name(type(fn))
    qualname = fn.__qualname__.split('.')[-1]
    try:
        code_obj = getattr(fn, '__code__', None)
        if code_obj is None:
            code_obj = fn.__call__.__code__
    except AttributeError:
        raise AttributeError(f'{fn} is not a standard Python function object and could not be converted to a task.') from None
    code_hash = h[:NUM_CHARS_DYNAMIC_KEY] if (h := hash_objects(code_obj)) else 'unknown'
    return f'{qualname}-{code_hash}'

class TaskRunNameCallbackWithParameters(Protocol):
    @classmethod
    def is_callback_with_parameters(cls, callable: Callable[..., Any]) -> bool:
        sig = inspect.signature(callable)
        return 'parameters' in sig.parameters

    def __call__(self, parameters: Dict[str, Any]) -> str:
        ...

StateHookCallable = Callable[['Task[..., Any]', TaskRun, State], Union[Awaitable[None], None]]
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
    """
    def __init__(
        self,
        fn: Callable[P, R],
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        version: Optional[str] = None,
        cache_policy: Union[CachePolicy, NotSet] = NotSet,
        cache_key_fn: Optional[Callable[['TaskRunContext', Dict[str, Any]], Optional[str]]] = None,
        cache_expiration: Optional[datetime.timedelta] = None,
        task_run_name: Optional[TaskRunNameValueOrCallable] = None,
        retries: Optional[int] = None,
        retry_delay_seconds: Optional[Union[float, List[float], Callable[[int], List[float]]]] = None,
        retry_jitter_factor: Optional[float] = None,
        persist_result: Optional[bool] = None,
        result_storage: Optional[ResultStorage] = None,
        result_serializer: Optional[ResultSerializer] = None,
        result_storage_key: Optional[str] = None,
        cache_result_in_memory: bool = True,
        timeout_seconds: Optional[float] = None,
        log_prints: bool = False,
        refresh_cache: Optional[bool] = None,
        on_completion: Optional[List[Callable[..., Any]]] = None,
        on_failure: Optional[List[Callable[..., Any]]] = None,
        on_rollback: Optional[List[Callable[..., Any]]] = None,
        on_commit: Optional[List[Callable[..., Any]]] = None,
        retry_condition_fn: Optional[Callable[..., bool]] = None,
        viz_return_value: Any = None
    ):
        hook_categories = [on_completion, on_failure]
        hook_names = ['on_completion', 'on_failure']
        for hooks, hook_name in zip(hook_categories, hook_names):
            if hooks is not None:
                try:
                    hooks = list(hooks)
                except TypeError:
                    raise TypeError(f"Expected iterable for '{hook_name}'; got {type(hooks).__name__} instead. Please provide a list of hooks to '{hook_name}':\n\n@task({hook_name}=[hook1, hook2])\ndef my_task():\n\tpass")
                for hook in hooks:
                    if not callable(hook):
                        raise TypeError(f"Expected callables in '{hook_name}'; got {type(hook).__name__} instead. Please provide a list of hooks to '{hook_name}':\n\n@task({hook_name}=[hook1, hook2])\ndef my_task():\n\tpass")
        if not callable(fn):
            raise TypeError("'fn' must be callable")
        self.description = description or inspect.getdoc(fn)
        update_wrapper(self, fn)
        self.fn = fn
        self.isasync = asyncio.iscoroutinefunction(self.fn) or inspect.isasyncgenfunction(self.fn)
        self.isgenerator = inspect.isgeneratorfunction(self.fn) or inspect.isasyncgenfunction(self.fn)
        if not name:
            if not hasattr(self.fn, '__name__'):
                self.name = type(self.fn).__name__
            else:
                self.name = self.fn.__name__
        else:
            self.name = name
        if task_run_name is not None:
            if not isinstance(task_run_name, str) and (not callable(task_run_name)):
                raise TypeError(f"Expected string or callable for 'task_run_name'; got {type(task_run_name).__name__} instead.")
        self.task_run_name = task_run_name
        self.version = version
        self.log_prints = log_prints
        raise_for_reserved_arguments(self.fn, ['return_state', 'wait_for'])
        self.tags = set(tags if tags else [])
        self.task_key = _generate_task_key(self.fn)
        if cache_policy is not NotSet and cache_key_fn is not None:
            logger.warning(f'Both `cache_policy` and `cache_key_fn` are set on task {self}. `cache_key_fn` will be used.')
        if cache_key_fn:
            cache_policy = CachePolicy.from_cache_key_fn(cache_key_fn)
        self.cache_key_fn = cache_key_fn
        self.cache_expiration = cache_expiration
        self.refresh_cache = refresh_cache
        if persist_result is None:
            if any([cache_policy and cache_policy != NO_CACHE and (cache_policy != NotSet), cache_key_fn is not None, result_storage_key is not None, result_storage is not None, result_serializer is not None]):
                persist_result = True
        if persist_result is False:
            self.cache_policy = None if cache_policy is None else NO_CACHE
            if cache_policy and cache_policy is not NotSet and (cache_policy != NO_CACHE):
                logger.warning('Ignoring `cache_policy` because `persist_result` is False')
        elif cache_policy is NotSet and result_storage_key is None:
            self.cache_policy = DEFAULT
        elif result_storage_key:
            self.cache_policy = None
        else:
            self.cache_policy = cache_policy
        settings = get_current_settings()
        self.retries = retries if retries is not None else settings.tasks.default_retries
        if retry_delay_seconds is None:
            retry_delay_seconds = settings.tasks.default_retry_delay_seconds
        if callable(retry_delay_seconds):
            self.retry_delay_seconds = retry_delay_seconds(self.retries)
        elif not isinstance(retry_delay_seconds, (list, int, float, type(None))):
            raise TypeError(f'Invalid `retry_delay_seconds` provided; must be an int, float, list or callable. Received type {type(retry_delay_seconds)}')
        else:
            self.retry_delay_seconds = retry_delay_seconds
        if isinstance(self.retry_delay_seconds, list) and len(self.retry_delay_seconds) > 50:
            raise ValueError('Can not configure more than 50 retry delays per task.')
        if retry_jitter_factor is not None and retry_jitter_factor < 0:
            raise ValueError('`retry_jitter_factor` must be >= 0.')
        self.retry_jitter_factor = retry_jitter_factor
        self.persist_result = persist_result
        if result_storage and (not isinstance(result_storage, str)):
            if getattr(result_storage, '_block_document_id', None) is None:
                raise TypeError('Result storage configuration must be persisted server-side. Please call `.save()` on your block before passing it in.')
        self.result_storage = result_storage
        self.result_serializer = result_serializer
        self.result_storage_key = result_storage_key
        self.cache_result_in_memory = cache_result_in_memory
        self.timeout_seconds = float(timeout_seconds) if timeout_seconds else None
        self.on_rollback_hooks = on_rollback or []
        self.on_commit_hooks = on_commit or []
        self.on_completion_hooks = on_completion or []
        self.on_failure_hooks = on_failure or []
        if retry_condition_fn is not None and (not callable(retry_condition_fn)):
            raise TypeError(f'Expected `retry_condition_fn` to be callable, got {type(retry_condition_fn).__name__} instead.')
        self.retry_condition_fn = retry_condition_fn
        self.viz_return_value = viz_return_value

    @property
    def ismethod(self) -> bool:
        return hasattr(self.fn, '__prefect_self__')

    def __get__(self, instance: Any, owner: Type[Any]) -> 'Task[P, R]':
        """
        Implement the descriptor protocol so that the task can be used as an instance method.
        When an instance method is loaded, this method is called with the "self" instance as
        an argument. We return a copy of the task with that instance bound to the task's function.
        """
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
        cache_policy: Union[CachePolicy, NotSet] = NotSet,
        cache_key_fn: Optional[Callable[['TaskRunContext', Dict[str, Any]], Optional[str]]] = None,
        task_run_name: Union[TaskRunNameValueOrCallable, NotSet] = NotSet,
        cache_expiration: Optional[datetime.timedelta] = None,
        retries: Union[int, NotSet] = NotSet,
        retry_delay_seconds: Union[float, List[float], Callable[[int], List[float]], NotSet] = NotSet,
        retry_jitter_factor: Union[float, NotSet] = NotSet,
        persist_result: Union[bool, NotSet] = NotSet,
        result_storage: Union[ResultStorage, NotSet] = NotSet,
        result_serializer: Union[ResultSerializer, NotSet] = NotSet,
        result_storage_key: Union[str, NotSet] = NotSet,
        cache_result_in_memory: Optional[bool] = None,
        timeout_seconds: Optional[float] = None,
        log_prints: Union[bool, NotSet] = NotSet,
        refresh_cache: Union[bool, NotSet] = NotSet,
        on_completion: Optional[List[Callable[..., Any]]] = None,
        on_failure: Optional[List[Callable[..., Any]]] = None,
        retry_condition_fn: Optional[Callable[..., bool]] = None,
        viz_return_value: Any = None
    ) -> 'Task[P, R]':
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
            cache_expiration=cache_expiration or self.cache_exp