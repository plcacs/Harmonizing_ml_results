        from prefect import task
        from prefect.cache_policies import TASK_SOURCE, INPUTS
        from prefect.isolation_levels import SERIALIZABLE

        from prefect_redis import RedisLockManager

        cache_policy = (INPUTS + TASK_SOURCE).configure(
            isolation_level=SERIALIZABLE,
            lock_manager=RedisLockManager(host="my-redis-host"),
        )

        @task(cache_policy=cache_policy)
        def my_cached_task(x: int):
            return x + 42
        