from typing import TypeVar, Generic, Callable, Any, ParamSpec, Literal, Optional, Union
from functools import partial
from prefect.client.schemas import TaskRun
from prefect.states import Pending
from prefect.utilities.annotations import NotSet

T = TypeVar('T')
P = ParamSpec('P')

def task(
    __fn: Optional[Callable[P, T]] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[Any] = None,
    version: Optional[str] = None,
    cache_policy: Literal[NotSet, 'DEFAULT', 'NO_CACHE'] = NotSet,
    cache_key_fn: Optional[Callable[..., str]] = None,
    cache_expiration: Optional[Any] = None,
    task_run_name: Optional[Union[str, Callable[..., str]]] = None,
    retries: Optional[int] = 0,
    retry_delay_seconds: Optional[Union[int, list, Callable[[int], list]]] = 0,
    retry_jitter_factor: Optional[float] = None,
    persist_result: Optional[bool] = None,
    result_storage: Optional[str] = None,
    result_storage_key: Optional[str] = None,
    result_serializer: Optional[str] = None,
    cache_result_in_memory: Optional[bool] = True,
    timeout_seconds: Optional[float] = None,
    log_prints: Optional[bool] = None,
    refresh_cache: Optional[bool] = None,
    on_completion: Optional[Any] = None,
    on_failure: Optional[Any] = None,
    retry_condition_fn: Optional[Callable[..., bool]] = None,
    viz_return_value: Optional[Any] = None
) -> Callable[P, Task[P, T]]:
    """Decorator to designate a function as a task in a Prefect workflow."""
    if __fn:
        if isinstance(__fn, (classmethod, staticmethod)):
            method_decorator = type(__fn).__name__
            raise TypeError(f'@{method_decorator} should be applied on top of @task')
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
            viz_return_value=viz_return_value
        )
    else:
        return partial(
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
            viz_return_value=viz_return_value
        )
