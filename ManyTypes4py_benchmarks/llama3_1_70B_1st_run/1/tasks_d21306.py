class TaskRunNameCallbackWithParameters(Protocol):
    @classmethod
    def is_callback_with_parameters(cls, callable: Callable) -> bool:
        sig = inspect.signature(callable)
        return 'parameters' in sig.parameters

    def __call__(self, parameters: dict) -> str:
        ...

StateHookCallable = Callable[['Task[..., Any]', TaskRun, State], Union[Awaitable[None], None]]
TaskRunNameValueOrCallable = Union[Callable[[], str], TaskRunNameCallbackWithParameters, str]

class Task(Generic[P, R]):
    def __init__(self, fn: Callable[P, R], 
                 name: str = None, 
                 description: str = None, 
                 tags: set[str] = None, 
                 version: str = None, 
                 cache_policy: NotSet = NotSet, 
                 cache_key_fn: Callable = None, 
                 cache_expiration: datetime.timedelta = None, 
                 task_run_name: TaskRunNameValueOrCallable = None, 
                 retries: int = None, 
                 retry_delay_seconds: Union[int, float, list, Callable] = None, 
                 retry_jitter_factor: float = None, 
                 persist_result: bool = None, 
                 result_storage: str = None, 
                 result_storage_key: str = None, 
                 result_serializer: str = None, 
                 cache_result_in_memory: bool = True, 
                 timeout_seconds: float = None, 
                 log_prints: bool = False, 
                 refresh_cache: bool = None, 
                 on_completion: list[Callable] = None, 
                 on_failure: list[Callable] = None, 
                 retry_condition_fn: Callable = None, 
                 viz_return_value: Any = None):
        ...

    @property
    def ismethod(self) -> bool:
        return hasattr(self.fn, '__prefect_self__')

    def __get__(self, instance: Any, owner: Any) -> 'Task':
        ...

    def with_options(self, *, 
                     name: str = None, 
                     description: str = None, 
                     tags: set[str] = None, 
                     cache_policy: NotSet = NotSet, 
                     cache_key_fn: Callable = None, 
                     task_run_name: TaskRunNameValueOrCallable = NotSet, 
                     cache_expiration: datetime.timedelta = None, 
                     retries: int = NotSet, 
                     retry_delay_seconds: Union[int, float, list, Callable] = NotSet, 
                     retry_jitter_factor: float = NotSet, 
                     persist_result: bool = NotSet, 
                     result_storage: str = NotSet, 
                     result_storage_key: str = NotSet, 
                     result_serializer: str = NotSet, 
                     cache_result_in_memory: bool = None, 
                     timeout_seconds: float = None, 
                     log_prints: bool = NotSet, 
                     refresh_cache: bool = NotSet, 
                     on_completion: list[Callable] = None, 
                     on_failure: list[Callable] = None, 
                     retry_condition_fn: Callable = None, 
                     viz_return_value: Any = None) -> 'Task':
        ...

    def on_completion(self, fn: Callable) -> Callable:
        ...

    def on_failure(self, fn: Callable) -> Callable:
        ...

    def on_commit(self, fn: Callable) -> Callable:
        ...

    def on_rollback(self, fn: Callable) -> Callable:
        ...

    async def create_run(self, client: PrefectClient = None, 
                          id: UUID = None, 
                          parameters: dict = None, 
                          flow_run_context: FlowRunContext = None, 
                          parent_task_run_context: TaskRunContext = None, 
                          wait_for: list[PrefectFuture] = None, 
                          extra_task_inputs: dict = None, 
                          deferred: bool = False) -> TaskRun:
        ...

    async def create_local_run(self, client: PrefectClient = None, 
                                id: UUID = None, 
                                parameters: dict = None, 
                                flow_run_context: FlowRunContext = None, 
                                parent_task_run_context: TaskRunContext = None, 
                                wait_for: list[PrefectFuture] = None, 
                                extra_task_inputs: dict = None, 
                                deferred: bool = False) -> TaskRun:
        ...

    @overload
    def __call__(self, *args: P.args, return_state: bool = False, wait_for: list[PrefectFuture] = None, **kwargs: P.kwargs) -> Union[PrefectFuture[R], PrefectState[R]]:
        ...

    @overload
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Union[PrefectFuture[R], PrefectState[R]]:
        ...

    def __call__(self, *args: P.args, return_state: bool = False, wait_for: list[PrefectFuture] = None, **kwargs: P.kwargs) -> Union[PrefectFuture[R], PrefectState[R]]:
        ...

    @overload
    def submit(self, *args: P.args, return_state: bool = False, wait_for: list[PrefectFuture] = None, **kwargs: P.kwargs) -> Union[PrefectFuture[R], PrefectState[R]]:
        ...

    @overload
    def submit(self, *args: P.args, **kwargs: P.kwargs) -> Union[PrefectFuture[R], PrefectState[R]]:
        ...

    def submit(self, *args: P.args, return_state: bool = False, wait_for: list[PrefectFuture] = None, **kwargs: P.kwargs) -> Union[PrefectFuture[R], PrefectState[R]]:
        ...

    @overload
    def map(self, *args: P.args, return_state: bool = False, wait_for: list[PrefectFuture] = None, deferred: bool = False, **kwargs: P.kwargs) -> list[Union[PrefectFuture[R], PrefectState[R]]]:
        ...

    @overload
    def map(self, *args: P.args, wait_for: list[PrefectFuture] = None, deferred: bool = False, **kwargs: P.kwargs) -> list[Union[PrefectFuture[R], PrefectState[R]]]:
        ...

    def map(self, *args: P.args, return_state: bool = False, wait_for: list[PrefectFuture] = None, deferred: bool = False, **kwargs: P.kwargs) -> list[Union[PrefectFuture[R], PrefectState[R]]]:
        ...

    def apply_async(self, args: tuple[P.args] = None, 
                     kwargs: dict[P.kwargs] = None, 
                     wait_for: list[PrefectFuture] = None, 
                     dependencies: dict = None) -> PrefectDistributedFuture[R]:
        ...

    def delay(self, *args: P.args, **kwargs: P.kwargs) -> PrefectDistributedFuture[R]:
        ...

    @sync_compatible
    async def serve(self) -> None:
        ...

@overload
def task(__fn: Callable[P, R]) -> Task[P, R]:
    ...

@overload
def task(__fn: Callable[P, R] = None, *, 
         name: str = None, 
         description: str = None, 
         tags: set[str] = None, 
         version: str = None, 
         cache_policy: NotSet = NotSet, 
         cache_key_fn: Callable = None, 
         cache_expiration: datetime.timedelta = None, 
         task_run_name: TaskRunNameValueOrCallable = None, 
         retries: int = None, 
         retry_delay_seconds: Union[int, float, list, Callable] = None, 
         retry_jitter_factor: float = None, 
         persist_result: bool = None, 
         result_storage: str = None, 
         result_storage_key: str = None, 
         result_serializer: str = None, 
         cache_result_in_memory: bool = True, 
         timeout_seconds: float = None, 
         log_prints: bool = False, 
         refresh_cache: bool = None, 
         on_completion: list[Callable] = None, 
         on_failure: list[Callable] = None, 
         retry_condition_fn: Callable = None, 
         viz_return_value: Any = None) -> Callable[[Callable[P, R]], Task[P, R]]:
    ...

def task(__fn: Callable[P, R] = None, *, 
         name: str = None, 
         description: str = None, 
         tags: set[str] = None, 
         version: str = None, 
         cache_policy: NotSet = NotSet, 
         cache_key_fn: Callable = None, 
         cache_expiration: datetime.timedelta = None, 
         task_run_name: TaskRunNameValueOrCallable = None, 
         retries: int = None, 
         retry_delay_seconds: Union[int, float, list, Callable] = None, 
         retry_jitter_factor: float = None, 
         persist_result: bool = None, 
         result_storage: str = None, 
         result_storage_key: str = None, 
         result_serializer: str = None, 
         cache_result_in_memory: bool = True, 
         timeout_seconds: float = None, 
         log_prints: bool = False, 
         refresh_cache: bool = None, 
         on_completion: list[Callable] = None, 
         on_failure: list[Callable] = None, 
         retry_condition_fn: Callable = None, 
         viz_return_value: Any = None) -> Union[Task[P, R], Callable[[Callable[P, R]], Task[P, R]]]:
    ...
