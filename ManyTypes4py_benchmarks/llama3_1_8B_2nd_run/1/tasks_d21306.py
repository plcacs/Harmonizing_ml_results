def task(__fn: Optional[Callable] = None, *, 
         name: Optional[str] = None, description: Optional[str] = None, 
         tags: Optional[Iterable[str]] = None, version: Optional[str] = None, 
         cache_policy: Optional[CachePolicy] = NotSet, cache_key_fn: Optional[Callable] = None, 
         cache_expiration: Optional[datetime.timedelta] = None, task_run_name: Optional[str] = None, 
         retries: Optional[int] = 0, retry_delay_seconds: Optional[Union[int, List[int], Callable]] = None, 
         retry_jitter_factor: Optional[float] = None, persist_result: Optional[bool] = None, 
         result_storage: Optional[str] = None, result_storage_key: Optional[str] = None, 
         result_serializer: Optional[str] = None, cache_result_in_memory: bool = True, 
         timeout_seconds: Optional[float] = None, log_prints: Optional[bool] = None, 
         refresh_cache: Optional[bool] = None, on_completion: Optional[Iterable[Callable]] = None, 
         on_failure: Optional[Iterable[Callable]] = None, retry_condition_fn: Optional[Callable] = None, 
         viz_return_value: Any = None) -> Task:
    """
    Decorator to designate a function as a task in a Prefect workflow.

    This decorator may be used for asynchronous or synchronous functions.

    Args:
        name: An optional name for the task; if not provided, the name will be inferred
            from the given function.
        description: An optional string description for the task.
        tags: An optional set of tags to be associated with runs of this task. These
            tags are combined with any tags defined by a `prefect.tags` context at
            task runtime.
        version: An optional string specifying the version of this task definition
        cache_key_fn: An optional callable that, given the task run context and call
            parameters, generates a string key; if the key matches a previous completed
            state, that state result will be restored instead of running the task again.
        cache_expiration: An optional amount of time indicating how long cached states
            for this task should be restorable; if not provided, cached states will
            never expire.
        task_run_name: An optional name to distinguish runs of this task; this name can be provided
            as a string template with the task's keyword arguments as variables,
            or a function that returns a string.
        retries: An optional number of times to retry on task run failure
        retry_delay_seconds: Optionally configures how long to wait before retrying the
            task after failure. This is only applicable if `retries` is nonzero. This
            setting can either be a number of seconds, a list of retry delays, or a
            callable that, given the total number of retries, generates a list of retry
            delays. If a number of seconds, that delay will be applied to all retries.
            If a list, each retry will wait for the corresponding delay before retrying.
            When passing a callable or a list, the number of
            configured retry delays cannot exceed 50.
        retry_jitter_factor: An optional factor that defines the factor to which a
            retry can be jittered in order to avoid a "thundering herd".
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
        retry_condition_fn: An optional callable run when a task run returns a Failed state. Should
            return `True` if the task should continue to its retry policy (e.g. `retries=3`), and `False` if the task
            should end as failed. Defaults to `None`, indicating the task should always continue
            to its retry policy.
        viz_return_value: An optional value to return when the task dependency tree is visualized.

    Returns:
        A callable `Task` object which, when called, will submit the task for execution.

    Examples:
        Define a simple task

        >>> @task
        >>> def add(x, y):
        >>>     return x + y

        Define an async task

        >>> @task
        >>> async def add(x, y):
        >>>     return x + y

        Define a task with tags and a description

        >>> @task(tags={"a", "b"}, description="This task is empty but its my first!")
        >>> def my_task():
        >>>     pass

        Define a task with a custom name

        >>> @task(name="The Ultimate Task")
        >>> def my_task():
        >>>     pass

        Define a task that retries 3 times with a 5 second delay between attempts

        >>> from random import randint
        >>>
        >>> @task(retries=3, retry_delay_seconds=5)
        >>> def my_task():
        >>>     x = randint(0, 5)
        >>>     if x >= 3:  # Make a task that fails sometimes
        >>>         raise ValueError("Retry me please!")
        >>>     return x

        Define a task that is cached for a day based on its inputs

        >>> from prefect.tasks import task_input_hash
        >>> from datetime import timedelta
        >>>
        >>> @task(cache_key_fn=task_input_hash, cache_expiration=timedelta(days=1))
        >>> def my_task():
        >>>     return "hello"
    """
    if __fn:
        if isinstance(__fn, (classmethod, staticmethod)):
            method_decorator = type(__fn).__name__
            raise TypeError(f'@{method_decorator} should be applied on top of @task')
        return Task(fn=__fn, name=name, description=description, tags=tags, version=version, 
                    cache_policy=cache_policy, cache_key_fn=cache_key_fn, cache_expiration=cache_expiration, 
                    task_run_name=task_run_name, retries=retries, retry_delay_seconds=retry_delay_seconds, 
                    retry_jitter_factor=retry_jitter_factor, persist_result=persist_result, 
                    result_storage=result_storage, result_storage_key=result_storage_key, 
                    result_serializer=result_serializer, cache_result_in_memory=cache_result_in_memory, 
                    timeout_seconds=timeout_seconds, log_prints=log_prints, refresh_cache=refresh_cache, 
                    on_completion=on_completion, on_failure=on_failure, retry_condition_fn=retry_condition_fn, 
                    viz_return_value=viz_return_value)
    else:
        return cast(Callable[[Callable[P, R]], Task[P, R]], partial(task, name=name, description=description, tags=tags, version=version, 
                                                                        cache_policy=cache_policy, cache_key_fn=cache_key_fn, cache_expiration=cache_expiration, 
                                                                        task_run_name=task_run_name, retries=retries, retry_delay_seconds=retry_delay_seconds, 
                                                                        retry_jitter_factor=retry_jitter_factor, persist_result=persist_result, 
                                                                        result_storage=result_storage, result_storage_key=result_storage_key, 
                                                                        result_serializer=result_serializer, cache_result_in_memory=cache_result_in_memory, 
                                                                        timeout_seconds=timeout_seconds, log_prints=log_prints, refresh_cache=refresh_cache, 
                                                                        on_completion=on_completion, on_failure=on_failure, retry_condition_fn=retry_condition_fn, 
                                                                        viz_return_value=viz_return_value))
