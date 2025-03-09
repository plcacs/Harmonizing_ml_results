from typing import Any, Dict, List, Optional, Tuple
import redis
from flask_caching.backends.rediscache import RedisCache, RedisSentinelCache
from redis.sentinel import Sentinel

class RedisCacheBackend(RedisCache):
    MAX_EVENT_COUNT: int = 100

    def __init__(self, host, port, password=None, db=0, default_timeout=300, key_prefix=None, ssl=False, ssl_certfile=None, ssl_keyfile=None, ssl_cert_reqs='required', ssl_ca_certs=None, **kwargs: Any):
        super().__init__(host=host, port=port, password=password, db=db, default_timeout=default_timeout, key_prefix=key_prefix, **kwargs)
        self._cache: redis.Redis = redis.Redis(host=host, port=port, password=password, db=db, ssl=ssl, ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile, ssl_cert_reqs=ssl_cert_reqs, ssl_ca_certs=ssl_ca_certs, **kwargs)

    def xadd(self, stream_name, event_data, event_id='*', maxlen=None):
        return self._cache.xadd(stream_name, event_data, event_id, maxlen)

    def xrange(self, stream_name, start='-', end='+', count=None):
        count = count or self.MAX_EVENT_COUNT
        return self._cache.xrange(stream_name, start, end, count)

    @classmethod
    def from_config(cls, config):
        kwargs: Dict[str, Any] = {'host': config.get('CACHE_REDIS_HOST', 'localhost'), 'port': config.get('CACHE_REDIS_PORT', 6379), 'db': config.get('CACHE_REDIS_DB', 0), 'password': config.get('CACHE_REDIS_PASSWORD', None), 'key_prefix': config.get('CACHE_KEY_PREFIX', None), 'default_timeout': config.get('CACHE_DEFAULT_TIMEOUT', 300), 'ssl': config.get('CACHE_REDIS_SSL', False), 'ssl_certfile': config.get('CACHE_REDIS_SSL_CERTFILE', None), 'ssl_keyfile': config.get('CACHE_REDIS_SSL_KEYFILE', None), 'ssl_cert_reqs': config.get('CACHE_REDIS_SSL_CERT_REQS', 'required'), 'ssl_ca_certs': config.get('CACHE_REDIS_SSL_CA_CERTS', None)}
        return cls(**kwargs)

class RedisSentinelCacheBackend(RedisSentinelCache):
    MAX_EVENT_COUNT: int = 100

    def __init__(self, sentinels, master, password=None, sentinel_password=None, db=0, default_timeout=300, key_prefix='', ssl=False, ssl_certfile=None, ssl_keyfile=None, ssl_cert_reqs='required', ssl_ca_certs=None, **kwargs: Any):
        self._sentinel: Sentinel = Sentinel(sentinels, sentinel_kwargs={'password': sentinel_password}, **{k: v for k, v in kwargs.items() if k not in ['ssl', 'ssl_certfile', 'ssl_keyfile', 'ssl_cert_reqs', 'ssl_ca_certs']})
        master_kwargs: Dict[str, Any] = {'password': password, 'ssl': ssl, 'ssl_certfile': ssl_certfile if ssl else None, 'ssl_keyfile': ssl_keyfile if ssl else None, 'ssl_cert_reqs': ssl_cert_reqs if ssl else None, 'ssl_ca_certs': ssl_ca_certs if ssl else None}
        if not ssl:
            master_kwargs = {k: v for k, v in master_kwargs.items() if not k.startswith('ssl')}
        master_kwargs = {k: v for k, v in master_kwargs.items() if v is not None}
        self._cache: redis.Redis = self._sentinel.master_for(master, **master_kwargs)
        super().__init__(host=None, port=None, password=password, db=db, default_timeout=default_timeout, key_prefix=key_prefix, **kwargs)

    def xadd(self, stream_name, event_data, event_id='*', maxlen=None):
        return self._cache.xadd(stream_name, event_data, event_id, maxlen)

    def xrange(self, stream_name, start='-', end='+', count=None):
        count = count or self.MAX_EVENT_COUNT
        return self._cache.xrange(stream_name, start, end, count)

    @classmethod
    def from_config(cls, config):
        kwargs: Dict[str, Any] = {'sentinels': config.get('CACHE_REDIS_SENTINELS', [('127.0.0.1', 26379)]), 'master': config.get('CACHE_REDIS_SENTINEL_MASTER', 'mymaster'), 'password': config.get('CACHE_REDIS_PASSWORD', None), 'sentinel_password': config.get('CACHE_REDIS_SENTINEL_PASSWORD', None), 'key_prefix': config.get('CACHE_KEY_PREFIX', ''), 'db': config.get('CACHE_REDIS_DB', 0), 'ssl': config.get('CACHE_REDIS_SSL', False), 'ssl_certfile': config.get('CACHE_REDIS_SSL_CERTFILE', None), 'ssl_keyfile': config.get('CACHE_REDIS_SSL_KEYFILE', None), 'ssl_cert_reqs': config.get('CACHE_REDIS_SSL_CERT_REQS', 'required'), 'ssl_ca_certs': config.get('CACHE_REDIS_SSL_CA_CERTS', None)}
        return cls(**kwargs)