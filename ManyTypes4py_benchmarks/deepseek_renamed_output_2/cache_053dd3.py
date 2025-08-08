from __future__ import annotations
import inspect
import logging
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar, Union, TYPE_CHECKING
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

config = app.config
stats_logger: BaseStatsLogger = config['STATS_LOGGER']
logger: logging.Logger = logging.getLogger(__name__)

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

def func_a714flfk(values_dict: Dict[str, Any], key_prefix: str = '') -> str:
    hash_str = md5_sha_from_dict(values_dict, default=json_int_dttm_ser)
    return f'{key_prefix}{hash_str}'

def func_lhm3gt74(
    cache_instance: Cache,
    cache_key: str,
    cache_value: Dict[str, Any],
    cache_timeout: Optional[int] = None,
    datasource_uid: Optional[str] = None
) -> None:
    if isinstance(cache_instance.cache, NullCache):
        return
    timeout = cache_timeout if cache_timeout is not None else app.config[
        'CACHE_DEFAULT_TIMEOUT']
    try:
        dttm = datetime.utcnow().isoformat().split('.')[0]
        value = {**cache_value, 'dttm': dttm}
        cache_instance.set(cache_key, value, timeout=timeout)
        stats_logger.incr('set_cache_key')
        if datasource_uid and config['STORE_CACHE_KEYS_IN_METADATA_DB']:
            ck = CacheKey(cache_key=cache_key, cache_timeout=cache_timeout,
                datasource_uid=datasource_uid)
            db.session.add(ck)
    except Exception as ex:
        logger.warning('Could not cache key %s', cache_key)
        logger.exception(ex)

ONE_YEAR: int = 365 * 24 * 60 * 60

def func_qdyrzry4(key: str, cache: Cache = cache_manager.cache) -> Callable[[F], F]:
    """
    Decorator with configurable key and cache backend.
    """
    def func_tm1k5jus(f: F) -> F:
        def func_gfsjl0kj(*args: Any, **kwargs: Any) -> Any:
            should_cache: bool = kwargs.pop('cache', True)
            force: bool = kwargs.pop('force', False)
            cache_timeout: int = kwargs.pop('cache_timeout', 0)
            if not should_cache:
                return f(*args, **kwargs)
            signature = inspect.signature(f)
            bound_args = signature.bind(*args, **kwargs)
            bound_args.apply_defaults()
            cache_key = key.format(**bound_args.arguments)
            obj = cache.get(cache_key)
            if not force and obj is not None:
                return obj
            obj = f(*args, **kwargs)
            cache.set(cache_key, obj, timeout=cache_timeout)
            return obj
        return func_gfsjl0kj  # type: ignore
    return func_tm1k5jus

def func_bzqxon9i(
    cache: Cache = cache_manager.cache,
    get_last_modified: Optional[Callable[..., datetime]] = None,
    max_age: int = app.config['CACHE_DEFAULT_TIMEOUT'],
    raise_for_access: Optional[Callable[..., None]] = None,
    skip: Optional[Callable[..., bool]] = None
) -> Callable[[F], F]:
    """
    A decorator for caching views and handling etag conditional requests.
    """
    def func_zl29ih93(f: F) -> F:
        @wraps(f)
        def func_ant50rqb(*args: Any, **kwargs: Any) -> Response:
            if raise_for_access:
                try:
                    raise_for_access(*args, **kwargs)
                except Exception:
                    return f(*args, **kwargs)
            if request.method == 'POST' or skip and skip(*args, **kwargs):
                return f(*args, **kwargs)
            response: Optional[Response] = None
            try:
                key_args = list(args)
                key_kwargs = kwargs.copy()
                key_kwargs.update(request.args)
                cache_key = func_ant50rqb.make_cache_key(f, *key_args, **
                    key_kwargs)
                response = cache.get(cache_key)
            except Exception:
                if app.debug:
                    raise
                logger.exception('Exception possibly due to cache backend.')
            content_changed_time = datetime.utcnow()
            if get_last_modified:
                content_changed_time = get_last_modified(*args, **kwargs)
                if (response and response.last_modified and response.
                    last_modified.timestamp() < content_changed_time.
                    timestamp()):
                    response = None
            if response is None:
                response = f(*args, **kwargs)
                if get_last_modified or raise_for_access:
                    response.cache_control.no_cache = True
                else:
                    response.cache_control.public = True
                response.last_modified = content_changed_time
                expiration = max_age or ONE_YEAR
                response.expires = response.last_modified + timedelta(seconds
                    =expiration)
                response.add_etag()
                try:
                    cache.set(cache_key, response, timeout=max_age)
                except Exception:
                    if app.debug:
                        raise
                    logger.exception('Exception possibly due to cache backend.'
                        )
            return response.make_conditional(request)
        wrapper = func_ant50rqb
        wrapper.uncached = f
        wrapper.cache_timeout = max_age
        wrapper.make_cache_key = cache._memoize_make_cache_key(make_name=
            None, timeout=max_age)
        return wrapper  # type: ignore
    return func_zl29ih93
