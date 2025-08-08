from typing import Any, Callable, Union, Tuple
from redis.asyncio import Redis
from prefect.settings.base import PrefectBaseSettings, build_settings_config
from pydantic import Field

CacheKey: Tuple[Callable[..., Any], Tuple[Any, ...], Tuple[Tuple[str, Any], ...], Union[asyncio.AbstractEventLoop, None]]
_client_cache: dict = {}

def _running_loop() -> Union[asyncio.AbstractEventLoop, None]:
    ...

def cached(fn: Callable) -> Callable:
    ...

def close_all_cached_connections() -> None:
    ...

def get_async_redis_client(host=None, port=None, db=None, password=None, username=None, health_check_interval=None, decode_responses=True, ssl=None) -> Redis:
    ...

def async_redis_from_settings(settings, **options) -> Redis:
    ...
