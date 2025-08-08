from __future__ import annotations
import inspect
import logging
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, TYPE_CHECKING, Dict
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

config: Dict[str, Any] = app.config
stats_logger: BaseStatsLogger = config['STATS_LOGGER']
logger: logging.Logger = logging.getLogger(__name__)

def generate_cache_key(values_dict: Dict[str, Any], key_prefix: str = '') -> str:
    hash_str: str = md5_sha_from_dict(values_dict, default=json_int_dttm_ser)
    return f'{key_prefix}{hash_str}'

def set_and_log_cache(cache_instance: Cache, cache_key: str, cache_value: Dict[str, Any], cache_timeout: int = None, datasource_uid: str = None) -> None:
    if isinstance(cache_instance.cache, NullCache):
        return
    timeout: int = cache_timeout if cache_timeout is not None else app.config['CACHE_DEFAULT_TIMEOUT']
    try:
        dttm: str = datetime.utcnow().isoformat().split('.')[0]
        value: Dict[str, Any] = {**cache_value, 'dttm': dttm}
        cache_instance.set(cache_key, value, timeout=timeout)
        stats_logger.incr('set_cache_key')
        if datasource_uid and config['STORE_CACHE_KEYS_IN_METADATA_DB']:
            ck: CacheKey = CacheKey(cache_key=cache_key, cache_timeout=cache_timeout, datasource_uid=datasource_uid)
            db.session.add(ck)
    except Exception as ex:
        logger.warning('Could not cache key %s', cache_key)
        logger.exception(ex)

def memoized_func(key: str, cache: Cache = cache_manager.cache) -> Callable[[Callable], Callable]:

def etag_cache(cache: Cache = cache_manager.cache, get_last_modified: Callable = None, max_age: int = app.config['CACHE_DEFAULT_TIMEOUT'], raise_for_access: Callable = None, skip: Callable = None) -> Callable[[Callable], Callable]:
