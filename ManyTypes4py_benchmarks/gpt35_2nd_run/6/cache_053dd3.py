from __future__ import annotations
import inspect
import logging
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, TYPE_CHECKING, Optional
from flask import current_app as app, request
from flask_caching import Cache
from flask_caching.backends import NullCache
from werkzeug.wrappers import Response
from superset import db
from superset.extensions import cache_manager
from superset.models.cache import CacheKey
from superset.utils.hashing import md5_sha_from_dict
from superset.utils.json import json_int_dttm_ser
if TYPE_CHECKING:
    from superset.stats_logger import BaseStatsLogger

config: dict = app.config
stats_logger: BaseStatsLogger = config['STATS_LOGGER']
logger: logging.Logger = logging.getLogger(__name__)

def generate_cache_key(values_dict: dict, key_prefix: str = '') -> str:
    hash_str: str = md5_sha_from_dict(values_dict, default=json_int_dttm_ser)
    return f'{key_prefix}{hash_str}'

def set_and_log_cache(cache_instance: Cache, cache_key: str, cache_value: dict, cache_timeout: Optional[int] = None, datasource_uid: Optional[str] = None) -> None:
    if isinstance(cache_instance.cache, NullCache):
        return
    timeout: int = cache_timeout if cache_timeout is not None else app.config['CACHE_DEFAULT_TIMEOUT']
    try:
        dttm: str = datetime.utcnow().isoformat().split('.')[0]
        value: dict = {**cache_value, 'dttm': dttm}
        cache_instance.set(cache_key, value, timeout=timeout)
        stats_logger.incr('set_cache_key')
        if datasource_uid and config['STORE_CACHE_KEYS_IN_METADATA_DB']:
            ck: CacheKey = CacheKey(cache_key=cache_key, cache_timeout=cache_timeout, datasource_uid=datasource_uid)
            db.session.add(ck)
    except Exception as ex:
        logger.warning('Could not cache key %s', cache_key)
        logger.exception(ex)

ONE_YEAR: int = 365 * 24 * 60 * 60

def memoized_func(key: str, cache: Cache = cache_manager.cache) -> Callable:
    def wrap(f: Callable) -> Callable:

        def wrapped_f(*args, **kwargs) -> Any:
            should_cache: bool = kwargs.pop('cache', True)
            force: bool = kwargs.pop('force', False)
            cache_timeout: int = kwargs.pop('cache_timeout', 0)
            if not should_cache:
                return f(*args, **kwargs)
            signature: inspect.Signature = inspect.signature(f)
            bound_args: inspect.BoundArguments = signature.bind(*args, **kwargs)
            bound_args.apply_defaults()
            cache_key: str = key.format(**bound_args.arguments)
            obj: Any = cache.get(cache_key)
            if not force and obj is not None:
                return obj
            obj = f(*args, **kwargs)
            cache.set(cache_key, obj, timeout=cache_timeout)
            return obj
        return wrapped_f
    return wrap

def etag_cache(cache: Cache = cache_manager.cache, get_last_modified: Optional[Callable] = None, max_age: int = app.config['CACHE_DEFAULT_TIMEOUT'], raise_for_access: Optional[Callable] = None, skip: Optional[Callable] = None) -> Callable:
    def decorator(f: Callable) -> Callable:

        @wraps(f)
        def wrapper(*args, **kwargs) -> Response:
            if raise_for_access:
                try:
                    raise_for_access(*args, **kwargs)
                except Exception:
                    return f(*args, **kwargs)
            if request.method == 'POST' or (skip and skip(*args, **kwargs)):
                return f(*args, **kwargs)
            response: Optional[Response] = None
            try:
                key_args: list = list(args)
                key_kwargs: dict = kwargs.copy()
                key_kwargs.update(request.args)
                cache_key: str = wrapper.make_cache_key(f, *key_args, **key_kwargs)
                response = cache.get(cache_key)
            except Exception:
                if app.debug:
                    raise
                logger.exception('Exception possibly due to cache backend.')
            content_changed_time: datetime = datetime.utcnow()
            if get_last_modified:
                content_changed_time = get_last_modified(*args, **kwargs)
                if response and response.last_modified and (response.last_modified.timestamp() < content_changed_time.timestamp()):
                    response = None
            if response is None:
                response = f(*args, **kwargs)
                if get_last_modified or raise_for_access:
                    response.cache_control.no_cache = True
                else:
                    response.cache_control.public = True
                response.last_modified = content_changed_time
                expiration: int = max_age or ONE_YEAR
                response.expires = response.last_modified + timedelta(seconds=expiration)
                response.add_etag()
                try:
                    cache.set(cache_key, response, timeout=max_age)
                except Exception:
                    if app.debug:
                        raise
                    logger.exception('Exception possibly due to cache backend.')
            return response.make_conditional(request)
        wrapper.uncached = f
        wrapper.cache_timeout = max_age
        wrapper.make_cache_key = cache._memoize_make_cache_key(make_name=None, timeout=max_age)
        return wrapper
    return decorator
