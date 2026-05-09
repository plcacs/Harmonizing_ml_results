from typing import TYPE_CHECKING, Any, Awaitable, Callable, Coroutine, Generic, Iterable, NoReturn, Optional, Protocol, TypeVar, Union, cast, overload
from typing_extensions import Literal, ParamSpec, Self, TypeAlias, TypeIs
from prefect.utilities.annotations import NotSet

T = TypeVar('T')
R = TypeVar('R')
P = ParamSpec('P')

class TaskRunNameCallbackWithParameters(Protocol):

    @classmethod
    def is_callback_with_parameters(cls, callable: Callable[..., Any]) -> bool:
        sig = inspect.signature(callable)
        return 'parameters' in sig.parameters

    def __call__(self, parameters: dict[str, Any]) -> str: ...

StateHookCallable = Callable[['Task[..., Any]', TaskRun, State], Union[Awaitable[None], None]]
TaskRunNameValueOrCallable = Union[Callable[[], str], TaskRunNameCallbackWithParameters, str]

class Task(Generic[P, R]):
    def __init__(self, 
                 fn: Callable[P, R], 
                 name: Optional[str] = None, 
                 description: Optional[str] = None, 
                 tags: Optional[set[str]] = None, 
                 version: Optional[str] = None, 
                 cache_policy: NotSet | Literal['DEFAULT', 'NO_CACHE'] = NotSet, 
                 cache_key_fn: Optional[Callable[[TaskRunContext, dict[str, Any]], str]] = None, 
                 cache_expiration: Optional[float] = None, 
                 task_run_name: TaskRunNameValueOrCallable | NotSet = NotSet, 
                 retries: int | NotSet = NotSet, 
                 retry_delay_seconds: int | float | list[float] | Callable[[int], list[float]] | None = None, 
                 retry_jitter_factor: float | None = None, 
                 persist_result: bool | NotSet = NotSet, 
                 result_storage: str | None = None, 
                 result_serializer: str | None = None, 
                 result_storage_key: str | None = None, 
                 cache_result_in_memory: bool = True, 
                 timeout_seconds: float | None = None, 
                 log_prints: bool | NotSet = NotSet, 
                 refresh_cache: bool | NotSet = NotSet, 
                 on_completion: list[Callable[[Task[..., Any], TaskRun, State], Union[Awaitable[None], None]]] | None = None, 
                 on_failure: list[Callable[[Task[..., Any], TaskRun, State], Union[Awaitable[None], None]]] | None = None, 
                 retry_condition_fn: Callable[[Task[..., Any], TaskRun, State], bool] | None = None, 
                 viz_return_value: Any = None):
        ...

    @property
    def ismethod(self) -> bool: ...

    def __get__(self, instance: Any, owner: Any) -> Self: ...

    async def create_run(self, 
                         client: Optional[Any] = None, 
                         id: Optional[UUID] = None, 
                         parameters: Optional[dict[str, Any]] = None, 
                         flow_run_context: Optional[FlowRunContext] = None, 
                         parent_task_run_context: Optional[TaskRunContext] = None, 
                         wait_for: Optional[list[PrefectFuture[Any]]] = None, 
                         extra_task_inputs: Optional[dict[str, Any]] = None, 
                         deferred: bool = False) -> TaskRun: ...

    async def create_local_run(self, 
                               client: Optional[Any] = None, 
                               id: Optional[UUID] = None, 
                               parameters: Optional[dict[str, Any]] = None, 
                               flow_run_context: Optional[FlowRunContext] = None, 
                               parent_task_run_context: Optional[TaskRunContext] = None, 
                               wait_for: Optional[list[PrefectFuture[Any]]] = None, 
                               extra_task_inputs: Optional[dict[str, Any]] = None, 
                               deferred: bool = False) -> TaskRun: ...

    @overload
    def with_options(self, 
                     *, 
                     name: Optional[str] = ..., 
                     description: Optional[str] = ..., 
                     tags: Optional[set[str]] = ..., 
                     cache_policy: NotSet | Literal['DEFAULT', 'NO_CACHE'] = ..., 
                     cache_key_fn: Optional[Callable[[TaskRunContext, dict[str, Any]], str]] = ..., 
                     task_run_name: TaskRunNameValueOrCallable | NotSet = ..., 
                     cache_expiration: Optional[float] = ..., 
                     retries: int | NotSet = ..., 
                     retry_delay_seconds: int | float | list[float] | Callable[[int], list[float]] | None = ..., 
                     retry_jitter_factor: float | None = ..., 
                     persist_result: bool | NotSet = ..., 
                     result_storage: str | None = ..., 
                     result_serializer: str | None = ..., 
                     result_storage_key: str | None = ..., 
                     cache_result_in_memory: bool = ..., 
                     timeout_seconds: float | None = ..., 
                     log_prints: bool | NotSet = ..., 
                     refresh_cache: bool | NotSet = ..., 
                     on_completion: list[Callable[[Task[..., Any], TaskRun, State], Union[Awaitable[None], None]]] | None = ..., 
                     on_failure: list[Callable[[Task[..., Any], TaskRun, State], Union[Awaitable[None], None]]] | None = ..., 
                     retry_condition_fn: Callable[[Task[..., Any], TaskRun, State], bool] | None = ..., 
                     viz_return_value: Any = ...) -> Task[P, R]: ...

    def with_options(self, 
                     *, 
                     name: Optional[str] = None, 
                     description: Optional[str] = None, 
                     tags: Optional[set[str]] = None, 
                     cache_policy: NotSet | Literal['DEFAULT', 'NO_CACHE'] = NotSet, 
                     cache_key_fn: Optional[Callable[[TaskRunContext, dict[str, Any]], str]] = None, 
                     task_run_name: TaskRunNameValueOrCallable | NotSet = NotSet, 
                     cache_expiration: Optional[float] = None, 
                     retries: int | NotSet = NotSet, 
                     retry_delay_seconds: int | float | list[float] | Callable[[int], list[float]] | None = None, 
                     retry_jitter_factor: float | None = None, 
                     persist_result: bool | NotSet = NotSet, 
                     result_storage: str | None = None, 
                     result_serializer: str | None = None, 
                     result_storage_key: str | None = None, 
                     cache_result_in_memory: bool = True, 
                     timeout_seconds: float | None = None, 
                     log_prints: bool | NotSet = NotSet, 
                     refresh_cache: bool | NotSet = NotSet, 
                     on_completion: list[Callable[[Task[..., Any], TaskRun, State], Union[Awaitable[None], None]]] | None = None, 
                     on_failure: list[Callable[[Task[..., Any], TaskRun, State], Union[Awaitable[None], None]]] | None = None, 
                     retry_condition_fn: Callable[[Task[..., Any], TaskRun, State], bool] | None = None, 
                     viz_return_value: Any = None) -> Task[P, R]: ...

    def on_completion(self, fn: Callable[[Task[..., Any], TaskRun, State], Union[Awaitable[None], None]]) -> Callable[[Task[..., Any], TaskRun, State], Union[Awaitable[None], None]]: ...

    def on_failure(self, fn: Callable[[Task[..., Any], TaskRun, State], Union[Awaitable[None], None]]) -> Callable[[Task[..., Any], TaskRun, State], Union[Awaitable[None], None]]: ...

    def on_commit(self, fn: Callable[[Task[..., Any], TaskRun, State], Union[Awaitable[None], None]]) -> Callable[[Task[..., Any], TaskRun, State], Union[Awaitable[None], None]]: ...

    def on_rollback(self, fn: Callable[[Task[..., Any], TaskRun, State], Union[Awaitable[None], None]]) -> Callable[[Task[..., Any], TaskRun, State], Union[Awaitable[None], None]]: ...

    @overload
    def __call__(self, 
                 *args: Any, 
                 return_state: bool = False, 
                 wait_for: Optional[list[PrefectFuture[Any]]] = None, 
                 **kwargs: Any) -> PrefectFuture[R] | State[R]: ...

    @overload
    def __call__(self, 
                 *args: Any, 
                 **kwargs: Any) -> PrefectFuture[R] | State[R]: ...

    @overload
    def __call__(self, 
                 *args: Any, 
                 return_state: bool = True, 
                 wait_for: Optional[list[PrefectFuture[Any]]] = None, 
                 **kwargs: Any) -> PrefectFuture[R] | State[R]: ...

    @overload
    def __call__(self, 
                 *args: Any, 
                 return_state: bool = False, 
                 wait_for: Optional[list[PrefectFuture[Any]]] = None, 
                 **kwargs: Any) -> PrefectFuture[R] | State[R]: ...

    def __call__(self, 
                 *args: Any, 
                 return_state: bool = False, 
                 wait_for: Optional[list[PrefectFuture[Any]]] = None, 
                 **kwargs: Any) -> PrefectFuture[R] | State[R]: ...

    @overload
    def submit(self, 
               *args: Any, 
               return_state: bool = False, 
               wait_for: Optional[list[PrefectFuture[Any]]] = None, 
               **kwargs: Any) -> PrefectFuture[R] | State[R]: ...

    @overload
    def submit(self, 
               *args: Any, 
               **kwargs: Any) -> PrefectFuture[R] | State[R]: ...

    @overload
    def submit(self, 
               *args: Any, 
               return_state: bool = True, 
               wait_for: Optional[list[PrefectFuture[Any]]] = None, 
               **kwargs: Any) -> PrefectFuture[R] | State[R]: ...

    @overload
    def submit(self, 
               *args: Any, 
               return_state: bool = False, 
               wait_for: Optional[list[PrefectFuture[Any]]] = None, 
               **kwargs: Any) -> PrefectFuture[R] | State[R]: ...

    @overload
    def submit(self, 
               *args: Any, 
               return_state: bool = False, 
               wait_for: Optional[list[PrefectFuture[Any]]] = None, 
               **kwargs: Any) -> PrefectFuture[R] | State[R]: ...

    def submit(self, 
               *args: Any, 
               return_state: bool = False, 
               wait_for: Optional[list[PrefectFuture[Any]]] = None, 
               **kwargs: Any) -> PrefectFuture[R] | State[R]: ...

    @overload
    def map(self, 
            *args: Any, 
            return_state: bool = False, 
            wait_for: Optional[list[PrefectFuture[Any]]] = None, 
            deferred: bool = False, 
            **kwargs: Any) -> list[PrefectFuture[R]] | list[State[R]]: ...

    @overload
    def map(self, 
            *args: Any, 
            wait_for: Optional[list[PrefectFuture[Any]]] = None, 
            deferred: bool = False, 
            **kwargs: Any) -> list[PrefectFuture[R]] | list[State[R]]: ...

    @overload
    def map(self, 
            *args: Any, 
            return_state: bool = True, 
            wait_for: Optional[list[PrefectFuture[Any]]] = None, 
            deferred: bool = False, 
            **kwargs: Any) -> list[PrefectFuture[R]] | list[State[R]]: ...

    @overload
    def map(self, 
            *args: Any, 
            wait_for: Optional[list[PrefectFuture[Any]]] = None, 
            deferred: bool = False, 
            **kwargs: Any) -> list[PrefectFuture[R]] | list[State[R]]: ...

    @overload
    def map(self, 
            *args: Any, 
            return_state: bool = False, 
            wait_for: Optional[list[PrefectFuture[Any]]] = None, 
            deferred: bool = False, 
            **kwargs: Any) -> list[PrefectFuture[R]] | list[State[R]]: ...

    @overload
    def map(self, 
            *args: Any, 
            return_state: bool = False, 
            wait_for: Optional[list[PrefectFuture[Any]]] = None, 
            deferred: bool = False, 
            **kwargs: Any) -> list[PrefectFuture[R]] | list[State[R]]: ...

    def map(self, 
            *args: Any, 
            return_state: bool = False, 
            wait_for: Optional[list[PrefectFuture[Any]]] = None, 
            deferred: bool = False, 
            **kwargs: Any) -> list[PrefectFuture[R]] | list[State[R]]: ...

    def apply_async(self, 
                    args: Optional[tuple[Any, ...]] = None, 
                    kwargs: Optional[dict[str, Any]] = None, 
                    wait_for: Optional[list[PrefectFuture[Any]]] = None, 
                    dependencies: Optional[dict[str, Any]] = None) -> PrefectDistributedFuture[R]: ...

    def delay(self, *args: Any, **kwargs: Any) -> PrefectDistributedFuture[R]: ...

    @sync_compatible
    async def serve(self) -> None: ...

@overload
def task(__fn: Callable[P, R]) -> Task[P, R]: ...

@overload
def task(__fn: None = None, 
         *, 
         name: Optional[str] = ..., 
         description: Optional[str] = ..., 
         tags: Optional[set[str]] = ..., 
         version: Optional[str] = ..., 
         cache_policy: NotSet | Literal['DEFAULT', 'NO_CACHE'] = ..., 
         cache_key_fn: Optional[Callable[[TaskRunContext, dict[str, Any]], str]] = ..., 
         cache_expiration: Optional[float] = ..., 
         task_run_name: TaskRunNameValueOrCallable | NotSet = ..., 
         retries: int | NotSet = ..., 
         retry_delay_seconds: int | float | list[float] | Callable[[int], list[float]] | None = ..., 
         retry_jitter_factor: float | None = ..., 
         persist_result: bool | NotSet = ..., 
         result_storage: str | None = ..., 
         result_storage_key: str | None = ..., 
         result_serializer: str | None = ..., 
         cache_result_in_memory: bool = ..., 
         timeout_seconds: float | None = ..., 
         log_prints: bool | NotSet = ..., 
         refresh_cache: bool | NotSet = ..., 
         on_completion: list[Callable[[Task[..., Any], TaskRun, State], Union[Awaitable[None], None]]] | None = ..., 
         on_failure: list[Callable[[Task[..., Any], TaskRun, State], Union[Awaitable[None], None]]] | None = ..., 
         retry_condition_fn: Callable[[Task[..., Any], TaskRun, State], bool] | None = ..., 
         viz_return_value: Any = ...) -> Callable[[Callable[P, R]], Task[P, R]]: ...

@overload
def task(*, 
         name: Optional[str] = ..., 
         description: Optional[str] = ..., 
         tags: Optional[set[str]] = ..., 
         version: Optional[str] = ..., 
         cache_policy: NotSet | Literal['DEFAULT', 'NO_CACHE'] = ..., 
         cache_key_fn: Optional[Callable[[TaskRunContext, dict[str, Any]], str]] = ..., 
         cache_expiration: Optional[float] = ..., 
         task_run_name: TaskRunNameValueOrCallable | NotSet = ..., 
         retries: int | NotSet = ..., 
         retry_delay_seconds: int | float | list[float] | Callable[[int], list[float]] | None = ..., 
         retry_jitter_factor: float | None = ..., 
         persist_result: bool | NotSet = ..., 
         result_storage: str | None = ..., 
         result_storage_key: str | None = ..., 
         result_serializer: str | None = ..., 
         cache_result_in_memory: bool = ..., 
         timeout_seconds: float | None = ..., 
         log_prints: bool | NotSet = ..., 
         refresh_cache: bool | NotSet = ..., 
         on_completion: list[Callable[[Task[..., Any], TaskRun, State], Union[Awaitable[None], None]]] | None = ..., 
         on_failure: list[Callable[[Task[..., Any], TaskRun, State], Union[Awaitable[None], None]]] | None = ..., 
         retry_condition_fn: Callable[[Task[..., Any], TaskRun, State], bool] | None = ..., 
         viz_return_value: Any = ...) -> Callable[[Callable[P, R]], Task[P, R]]: ...

def task(__fn: Callable[P, R] | None = None, 
         *, 
         name: Optional[str] = None, 
         description: Optional[str] = None, 
         tags: Optional[set[str]] = None, 
         version: Optional[str] = None, 
         cache_policy: NotSet | Literal['DEFAULT', 'NO_CACHE'] = NotSet, 
         cache_key_fn: Optional[Callable[[TaskRunContext, dict[str, Any]], str]] = None, 
         cache_expiration: Optional[float] = None, 
         task_run_name: TaskRunNameValueOrCallable | NotSet = NotSet, 
         retries: int | NotSet = None, 
         retry_delay_seconds: int | float | list[float] | Callable[[int], list[float]] | None = None, 
         retry_jitter_factor: float | None = None, 
         persist_result: bool | NotSet = NotSet, 
         result_storage: str | None = None, 
         result_storage_key: str | None = None, 
         result_serializer: str | None = None, 
         cache_result_in_memory: bool = True, 
         timeout_seconds: float | None = None, 
         log_prints: bool | NotSet = NotSet, 
         refresh_cache: bool | NotSet = NotSet, 
         on_completion: list[Callable[[Task[..., Any], TaskRun, State], Union[Awaitable[None], None]]] | None = None, 
         on_failure: list[Callable[[Task[..., Any], TaskRun, State], Union[Awaitable[None], None]]] | None = None, 
         retry_condition_fn: Callable[[Task[..., Any], TaskRun, State], bool] | None = None, 
         viz_return_value: Any = None) -> Task[P, R] | Callable[[Callable[P, R]], Task[P, R]]: ...
