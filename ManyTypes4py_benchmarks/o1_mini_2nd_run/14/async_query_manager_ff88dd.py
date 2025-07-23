from __future__ import annotations
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple
import jwt
from flask import Flask, Request, request, Response, session
from flask_caching.backends.base import BaseCache
from superset.async_events.cache_backend import RedisCacheBackend, RedisSentinelCacheBackend
from superset.utils import json
from superset.utils.core import get_user_id

logger = logging.getLogger(__name__)


class CacheBackendNotInitialized(Exception):
    pass


class AsyncQueryTokenException(Exception):
    pass


class UnsupportedCacheBackendError(Exception):
    pass


class AsyncQueryJobException(Exception):
    pass


def build_job_metadata(
    channel_id: str,
    job_id: str,
    user_id: Optional[int],
    **kwargs: Any
) -> Dict[str, Any]:
    return {
        'channel_id': channel_id,
        'job_id': job_id,
        'user_id': user_id,
        'status': kwargs.get('status'),
        'errors': kwargs.get('errors', []),
        'result_url': kwargs.get('result_url'),
    }


def parse_event(event_data: Tuple[bytes, Dict[bytes, bytes]]) -> Dict[str, Any]:
    event_id = event_data[0].decode('utf-8') if isinstance(event_data[0], bytes) else event_data[0]
    event_payload = event_data[1]['data'].decode('utf-8') if isinstance(event_data[1]['data'], bytes) else event_data[1]['data']
    return {'id': event_id, **json.loads(event_payload)}


def increment_id(entry_id: str) -> str:
    try:
        prefix, last = (entry_id[:-1], int(entry_id[-1]))
        return prefix + str(last + 1)
    except Exception:
        return entry_id


def get_cache_backend(config: Dict[str, Any]) -> BaseCache:
    cache_config = config.get('GLOBAL_ASYNC_QUERIES_CACHE_BACKEND', {})
    cache_type = cache_config.get('CACHE_TYPE')
    if cache_type == 'RedisCache':
        return RedisCacheBackend.from_config(cache_config)
    if cache_type == 'RedisSentinelCache':
        return RedisSentinelCacheBackend.from_config(cache_config)
    raise UnsupportedCacheBackendError('Unsupported cache backend configuration')


class AsyncQueryManager:
    MAX_EVENT_COUNT: int = 100
    STATUS_PENDING: str = 'pending'
    STATUS_RUNNING: str = 'running'
    STATUS_ERROR: str = 'error'
    STATUS_DONE: str = 'done'

    _cache: Optional[BaseCache]
    _stream_prefix: str
    _jwt_cookie_name: str
    _jwt_cookie_secure: bool
    _jwt_cookie_samesite: Optional[str]
    _jwt_cookie_domain: Optional[str]
    _jwt_secret: str
    _load_chart_data_into_cache_job: Any
    _load_explore_json_into_cache_job: Any

    def __init__(self) -> None:
        super().__init__()
        self._cache = None
        self._stream_prefix = ''
        self._jwt_cookie_name = ''
        self._jwt_cookie_secure = False
        self._jwt_cookie_samesite = None
        self._jwt_cookie_domain = None
        self._jwt_secret = ''
        self._load_chart_data_into_cache_job = None
        self._load_explore_json_into_cache_job = None

    def init_app(self, app: Flask) -> None:
        config: Dict[str, Any] = app.config
        cache_type: Optional[str] = config.get('CACHE_CONFIG', {}).get('CACHE_TYPE')
        data_cache_type: Optional[str] = config.get('DATA_CACHE_CONFIG', {}).get('CACHE_TYPE')
        if cache_type in [None, 'null'] or data_cache_type in [None, 'null']:
            raise Exception(
                '\n                Cache backends (CACHE_CONFIG, DATA_CACHE_CONFIG) must be configured\n'
                '                and non-null in order to enable async queries\n                '
            )
        self._cache = get_cache_backend(config)
        logger.debug('Using GAQ Cache backend as %s', type(self._cache).__name__)
        if len(config['GLOBAL_ASYNC_QUERIES_JWT_SECRET']) < 32:
            raise AsyncQueryTokenException('Please provide a JWT secret at least 32 bytes long')
        self._stream_prefix = config['GLOBAL_ASYNC_QUERIES_REDIS_STREAM_PREFIX']
        self._stream_limit = config['GLOBAL_ASYNC_QUERIES_REDIS_STREAM_LIMIT']
        self._stream_limit_firehose = config['GLOBAL_ASYNC_QUERIES_REDIS_STREAM_LIMIT_FIREHOSE']
        self._jwt_cookie_name = config['GLOBAL_ASYNC_QUERIES_JWT_COOKIE_NAME']
        self._jwt_cookie_secure = config['GLOBAL_ASYNC_QUERIES_JWT_COOKIE_SECURE']
        self._jwt_cookie_samesite = config['GLOBAL_ASYNC_QUERIES_JWT_COOKIE_SAMESITE']
        self._jwt_cookie_domain = config.get('GLOBAL_ASYNC_QUERIES_JWT_COOKIE_DOMAIN')
        self._jwt_secret = config['GLOBAL_ASYNC_QUERIES_JWT_SECRET']
        if config.get('GLOBAL_ASYNC_QUERIES_REGISTER_REQUEST_HANDLERS'):
            self.register_request_handlers(app)
        from superset.tasks.async_queries import load_chart_data_into_cache, load_explore_json_into_cache

        self._load_chart_data_into_cache_job = load_chart_data_into_cache
        self._load_explore_json_into_cache_job = load_explore_json_into_cache

    def register_request_handlers(self, app: Flask) -> None:

        @app.after_request
        def validate_session(response: Response) -> Response:
            user_id: Optional[int] = get_user_id()
            reset_token: bool = (
                not request.cookies.get(self._jwt_cookie_name)
                or 'async_channel_id' not in session
                or 'async_user_id' not in session
                or (user_id != session.get('async_user_id'))
            )
            if reset_token:
                async_channel_id: str = str(uuid.uuid4())
                session['async_channel_id'] = async_channel_id
                session['async_user_id'] = user_id
                sub: Optional[str] = str(user_id) if user_id else None
                token: str = jwt.encode({'channel': async_channel_id, 'sub': sub}, self._jwt_secret, algorithm='HS256')
                response.set_cookie(
                    self._jwt_cookie_name,
                    value=token,
                    httponly=True,
                    secure=self._jwt_cookie_secure,
                    domain=self._jwt_cookie_domain,
                    samesite=self._jwt_cookie_samesite,
                )
            return response

    def parse_channel_id_from_request(self, req: Request) -> str:
        token: Optional[str] = req.cookies.get(self._jwt_cookie_name)
        if not token:
            raise AsyncQueryTokenException('Token not preset')
        try:
            payload: Dict[str, Any] = jwt.decode(token, self._jwt_secret, algorithms=['HS256'])
            return payload['channel']
        except Exception as ex:
            logger.warning('Parse jwt failed', exc_info=True)
            raise AsyncQueryTokenException('Failed to parse token') from ex

    def init_job(self, channel_id: str, user_id: Optional[int]) -> Dict[str, Any]:
        job_id: str = str(uuid.uuid4())
        return build_job_metadata(channel_id, job_id, user_id, status=self.STATUS_PENDING)

    def submit_explore_json_job(
        self,
        channel_id: str,
        form_data: Any,
        response_type: str,
        force: bool = False,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        from superset import security_manager
        job_metadata: Dict[str, Any] = self.init_job(channel_id, user_id)
        guest_user = security_manager.get_current_guest_user_if_guest()
        if guest_user:
            job_data = {**job_metadata, 'guest_token': guest_user.guest_token}
        else:
            job_data = job_metadata
        self._load_explore_json_into_cache_job.delay(job_data, form_data, response_type, force)
        return job_metadata

    def submit_chart_data_job(
        self,
        channel_id: str,
        form_data: Any,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        from superset import security_manager
        job_metadata: Dict[str, Any] = self.init_job(channel_id, user_id)
        guest_user = security_manager.get_current_guest_user_if_guest()
        if guest_user:
            job_data = {**job_metadata, 'guest_token': guest_user.guest_token}
        else:
            job_data = job_metadata
        self._load_chart_data_into_cache_job.delay(job_data, form_data)
        return job_metadata

    def read_events(
        self,
        channel: str,
        last_id: Optional[str]
    ) -> List[Dict[str, Any]]:
        if not self._cache:
            raise CacheBackendNotInitialized('Cache backend not initialized')
        stream_name: str = f'{self._stream_prefix}{channel}'
        start_id: str = increment_id(last_id) if last_id else '-'
        results: List[Tuple[Any, Any]] = self._cache.xrange(stream_name, start_id, '+', self.MAX_EVENT_COUNT)
        if isinstance(self._cache, (RedisSentinelCacheBackend, RedisCacheBackend)):
            decoded_results: List[Tuple[str, Dict[str, str]]] = [
                (
                    event_id.decode('utf-8') if isinstance(event_id, bytes) else event_id,
                    {key.decode('utf-8'): value.decode('utf-8') for key, value in event_data.items()}
                )
                for event_id, event_data in results
            ]
            return [] if not decoded_results else [parse_event(event) for event in decoded_results]
        return [] if not results else [parse_event(event) for event in results]

    def update_job(
        self,
        job_metadata: Dict[str, Any],
        status: str,
        **kwargs: Any
    ) -> None:
        if not self._cache:
            raise CacheBackendNotInitialized('Cache backend not initialized')
        if 'channel_id' not in job_metadata:
            raise AsyncQueryJobException('No channel ID specified')
        if 'job_id' not in job_metadata:
            raise AsyncQueryJobException('No job ID specified')
        updates: Dict[str, Any] = {'status': status, **kwargs}
        event_data: Dict[str, Any] = {'data': json.dumps({**job_metadata, **updates})}
        full_stream_name: str = f'{self._stream_prefix}full'
        scoped_stream_name: str = f"{self._stream_prefix}{job_metadata['channel_id']}"
        logger.debug('********** logging event data to stream %s', scoped_stream_name)
        logger.debug(event_data)
        self._cache.xadd(scoped_stream_name, event_data, '*', self._stream_limit)
        self._cache.xadd(full_stream_name, event_data, '*', self._stream_limit_firehose)
