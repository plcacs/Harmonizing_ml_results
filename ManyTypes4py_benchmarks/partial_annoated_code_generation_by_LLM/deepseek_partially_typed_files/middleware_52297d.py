import cProfile
import logging
import tempfile
import time
from collections.abc import Callable, MutableMapping
from typing import Annotated, Any, Concatenate, Optional, Union, List, Dict, Tuple
from urllib.parse import urlencode, urljoin
from django.conf import settings
from django.conf.urls.i18n import is_language_prefix_patterns_used
from django.core import signals
from django.db import connection
from django.http import HttpRequest, HttpResponse, HttpResponseRedirect
from django.http.response import HttpResponseBase
from django.middleware.locale import LocaleMiddleware as DjangoLocaleMiddleware
from django.shortcuts import render
from django.utils import translation
from django.utils.cache import patch_vary_headers
from django.utils.crypto import constant_time_compare
from django.utils.deprecation import MiddlewareMixin
from django.utils.log import log_response
from django.utils.translation import gettext as _
from django_scim.middleware import SCIMAuthCheckMiddleware
from django_scim.settings import scim_settings
from sentry_sdk import set_tag
from typing_extensions import ParamSpec, override
from zerver.actions.message_summary import get_ai_requests, get_ai_time
from zerver.lib.cache import get_remote_cache_requests, get_remote_cache_time
from zerver.lib.db_connections import reset_queries
from zerver.lib.debug import maybe_tracemalloc_listen
from zerver.lib.exceptions import ErrorCode, JsonableError, MissingAuthenticationError, WebhookError
from zerver.lib.markdown import get_markdown_requests, get_markdown_time
from zerver.lib.per_request_cache import flush_per_request_caches
from zerver.lib.rate_limiter import RateLimitResult
from zerver.lib.request import RequestNotes
from zerver.lib.response import AsynchronousResponse, json_response, json_response_from_error, json_unauthorized
from zerver.lib.subdomains import get_subdomain
from zerver.lib.typed_endpoint import INTENTIONALLY_UNDOCUMENTED, ApiParamConfig, typed_endpoint
from zerver.lib.user_agent import parse_user_agent
from zerver.models import Realm
from zerver.models.realms import get_realm

ParamT = ParamSpec('ParamT')
logger: logging.Logger = logging.getLogger('zulip.requests')
slow_query_logger: logging.Logger = logging.getLogger('zulip.slow_queries')

def record_request_stop_data(log_data: Dict[str, Any]) -> None:
    log_data['time_stopped'] = time.time()
    log_data['remote_cache_time_stopped'] = get_remote_cache_time()
    log_data['remote_cache_requests_stopped'] = get_remote_cache_requests()
    log_data['markdown_time_stopped'] = get_markdown_time()
    log_data['markdown_requests_stopped'] = get_markdown_requests()
    if settings.PROFILE_ALL_REQUESTS:
        log_data['prof'].disable()

def async_request_timer_stop(request: HttpRequest) -> None:
    log_data: Optional[Dict[str, Any]] = RequestNotes.get_notes(request).log_data
    assert log_data is not None
    record_request_stop_data(log_data)

def record_request_restart_data(log_data: Dict[str, Any]) -> None:
    if settings.PROFILE_ALL_REQUESTS:
        log_data['prof'].enable()
    log_data['time_restarted'] = time.time()
    log_data['remote_cache_time_restarted'] = get_remote_cache_time()
    log_data['remote_cache_requests_restarted'] = get_remote_cache_requests()
    log_data['markdown_time_restarted'] = get_markdown_time()
    log_data['markdown_requests_restarted'] = get_markdown_requests()

def async_request_timer_restart(request: HttpRequest) -> None:
    log_data: Optional[Dict[str, Any]] = RequestNotes.get_notes(request).log_data
    assert log_data is not None
    if 'time_restarted' in log_data:
        return
    record_request_restart_data(log_data)

def record_request_start_data(log_data: Dict[str, Any]) -> None:
    if settings.PROFILE_ALL_REQUESTS:
        log_data['prof'] = cProfile.Profile()
        log_data['prof'].enable()
    reset_queries()
    log_data['time_started'] = time.time()
    log_data['remote_cache_time_start'] = get_remote_cache_time()
    log_data['remote_cache_requests_start'] = get_remote_cache_requests()
    log_data['markdown_time_start'] = get_markdown_time()
    log_data['markdown_requests_start'] = get_markdown_requests()
    log_data['ai_time_start'] = get_ai_time()
    log_data['ai_requests_start'] = get_ai_time()

def timedelta_ms(timedelta: float) -> float:
    return timedelta * 1000

def format_timedelta(timedelta: float) -> str:
    if timedelta >= 1:
        return f'{timedelta:.1f}s'
    return f'{timedelta_ms(timedelta):.0f}ms'

def is_slow_query(time_delta: float, path: str) -> bool:
    if time_delta < 1.2:
        return False
    is_exempt: bool = path == '/activity' or path.startswith(('/realm_activity/', '/user_activity/'))
    if is_exempt:
        return time_delta >= 5
    if 'webathena_kerberos' in path:
        return time_delta >= 10
    return True

def write_log_line(log_data: Dict[str, Any], path: str, method: str, remote_ip: str, requester_for_logs: str, client_name: str, client_version: Optional[str]=None, status_code: int=200, error_content: Optional[bytes]=None) -> None:
    time_delta: float = -1
    optional_orig_delta: str = ''
    if 'time_started' in log_data:
        time_delta = time.time() - log_data['time_started']
    if 'time_stopped' in log_data:
        orig_time_delta: float = time_delta
        time_delta = log_data['time_stopped'] - log_data['time_started'] + (time.time() - log_data['time_restarted'])
        optional_orig_delta = f' (lp: {format_timedelta(orig_time_delta)})'
    remote_cache_output: str = ''
    if 'remote_cache_time_start' in log_data:
        remote_cache_time_delta: float = get_remote_cache_time() - log_data['remote_cache_time_start']
        remote_cache_count_delta: int = get_remote_cache_requests() - log_data['remote_cache_requests_start']
        if 'remote_cache_requests_stopped' in log_data:
            remote_cache_time_delta += log_data['remote_cache_time_stopped'] - log_data['remote_cache_time_restarted']
            remote_cache_count_delta += log_data['remote_cache_requests_stopped'] - log_data['remote_cache_requests_restarted']
        if remote_cache_time_delta > 0.005:
            remote_cache_output = f' (mem: {format_timedelta(remote_cache_time_delta)}/{remote_cache_count_delta})'
    startup_output: str = ''
    if 'startup_time_delta' in log_data and log_data['startup_time_delta'] > 0.005:
        startup_output = ' (+start: {})'.format(format_timedelta(log_data['startup_time_delta']))
    markdown_output: str = ''
    if 'markdown_time_start' in log_data:
        markdown_time_delta: float = get_markdown_time() - log_data['markdown_time_start']
        markdown_count_delta: int = get_markdown_requests() - log_data['markdown_requests_start']
        if 'markdown_requests_stopped' in log_data:
            markdown_time_delta += log_data['markdown_time_stopped'] - log_data['markdown_time_restarted']
            markdown_count_delta += log_data['markdown_requests_stopped'] - log_data['markdown_requests_restarted']
        if markdown_time_delta > 0.005:
            markdown_output = f' (md: {format_timedelta(markdown_time_delta)}/{markdown_count_delta})'
    ai_output: str = ''
    if 'ai_time_start' in log_data:
        ai_time_delta: float = get_ai_time() - log_data['ai_time_start']
        ai_count_delta: int = get_ai_requests() - log_data['ai_requests_start']
        if ai_time_delta > 0.005:
            ai_output = f' (ai: {format_timedelta(ai_time_delta)}/{ai_count_delta})'
    db_time_output: str = ''
    queries: List[Dict[str, Any]] = connection.connection.queries if connection.connection is not None else []
    if len(queries) > 0:
        query_time: float = sum((float(query.get('time', 0)) for query in queries))
        db_time_output = f' (db: {format_timedelta(query_time)}/{len(queries)}q)'
    if 'extra' in log_data:
        extra_request_data: str = ' {}'.format(log_data['extra'])
    else:
        extra_request_data = ''
    logger_client: str
    if client_version is None:
        logger_client = f'({requester_for_logs} via {client_name})'
    else:
        logger_client = f'({requester_for_logs} via {client_name}/{client_version})'
    logger_timing: str = f'{format_timedelta(time_delta):>5}{optional_orig_delta}{remote_cache_output}{markdown_output}{ai_output}{db_time_output}{startup_output} {path}'
    logger_line: str = f'{remote_ip:<15} {method:<7} {status_code:3} {logger_timing}{extra_request_data} {logger_client}'
    if status_code in [200, 304] and method == 'GET' and path.startswith('/static'):
        logger.debug(logger_line)
    else:
        logger.info(logger_line)
    if is_slow_query(time_delta, path):
        slow_query_logger.info(logger_line)
    if settings.PROFILE_ALL_REQUESTS:
        log_data['prof'].disable()
        with tempfile.NamedTemporaryFile(prefix='profile.data.{}.{}.'.format(path.split('/')[-1], int(time_delta * 1000)), delete=False) as stats_file:
            log_data['prof'].dump_stats(stats_file.name)
    if 400 <= status_code < 500 and status_code not in [401, 404, 405]:
        error_data: str = repr(error_content)
        if len(error_data) > 200:
            error_data = '[content more than 200 characters]'
        logger.info('status=%3d, data=%s, uid=%s', status_code, error_data, requester_for_logs)

@typed_endpoint
def parse_client(request: HttpRequest, *, req_client: Annotated[Optional[str], ApiParamConfig('client', documentation_status=INTENTIONALLY_UNDOCUMENTED)]=None) -> Tuple[str, Optional[str]]:
    if req_client is not None:
        return (req_client, None)
    user_agent: Optional[Dict[str, str]] = None
    if 'User-Agent' in request.headers:
        user_agent = parse_user_agent(request.headers['User-Agent'])
    else:
        user_agent = None
    if user_agent is None:
        return ('Unspecified', None)
    client_name: str = user_agent['name']
    if client_name.startswith('Zulip'):
        return (client_name, user_agent.get('version'))
    return (client_name, None)

class LogRequests(MiddlewareMixin):

    def process_request(self, request: HttpRequest) -> None:
        maybe_tracemalloc_listen()
        request_notes: Any = RequestNotes.get_notes(request)
        if request_notes.log_data is not None:
            assert request_notes.saved_response is not None
            return
        try:
            (request_notes.client_name, request_notes.client_version) = parse_client(request)
        except JsonableError as e:
            logging.exception(e)
            request_notes.client_name = 'Unparsable'
            request_notes.client_version = None
        set_tag('client', request_notes.client_name)
        request_notes.log_data = {}
        record_request_start_data(request_notes.log_data)

    def process_view(self, request: HttpRequest, view_func: Callable[Concatenate[HttpRequest, ParamT], HttpResponseBase], args: List[object], kwargs: Dict[str, Any]) -> None:
        request_notes: Any = RequestNotes.get_notes(request)
        if request_notes.saved_response is not None:
            return
        assert request_notes.log_data is not None
        request_notes.log_data['startup_time_delta'] = time.time() - request_notes.log_data['time_started']
        record_request_start_data(request_notes.log_data)

    def process_response(self, request: HttpRequest, response: HttpResponseBase) -> HttpResponseBase:
        if isinstance(response, AsynchronousResponse):
            return response
        remote_ip: str = request.META['REMOTE_ADDR']
        request_notes: Any = RequestNotes.get_notes(request)
        requester_for_logs: Optional[str] = request_notes.requester_for_logs
        if requester_for_logs is None:
            if request_notes.remote_server is not None:
                requester_for_logs = request_notes.remote_server.format_requester_for_logs()
            elif request.user.is_authenticated:
                requester_for_logs = request.user.format_requester_for_logs()
            else:
                requester_for_logs = 'unauth@{}'.format(get_subdomain(request) or 'root')
        content: Optional[bytes] = response.content if isinstance(response, HttpResponse) else None
        assert request_notes.client_name is not None and request_notes.log_data is not None
        assert request.method is not None
        write_log_line(request_notes.log_data, request.path, request.method, remote_ip, requester_for_logs, request_notes.client_name, client_version=request_notes.client_version, status_code=response.status_code, error_content=content)
        return response

class JsonErrorHandler(MiddlewareMixin):

    def process_exception(self, request: HttpRequest, exception: Exception) -> Optional[HttpResponse]:
        if isinstance(exception, MissingAuthenticationError):
            if 'text/html' in request.headers.get('Accept', ''):
                return HttpResponseRedirect(f"{settings.HOME_NOT_LOGGED_IN}?{urlencode({'next': request.path})}")
            if request.path.startswith('/api'):
                return json_unauthorized()
            else:
                return json_unauthorized(www_authenticate='session')
        if isinstance(exception, JsonableError):
            response: HttpResponse = json_response_from_error(exception)
            if response.status_code < 500 or isinstance(exception, WebhookError):
                return response
        elif RequestNotes.get_notes(request).error_format == 'JSON' and (not settings.TEST_SUITE):
            response = json_response(res_type='error', msg=_('Internal server error'), status=500, exception=exception)
        else:
            return None
        try:
            raise exception
        except BaseException:
            signals.got_request_exception.send(sender=None, request=request)
        log_response('%s: %s', response.reason_phrase, request.path, response=response, request=request, exception=exception)
        return response

class TagRequests(MiddlewareMixin):

    def process_view(self, request: HttpRequest, view_func: Callable[Concatenate[HttpRequest, ParamT], HttpResponseBase], args: List[object], kwargs: Dict[str, Any]) -> None:
        self.process_request(request)

    def process_request(self, request: HttpRequest) -> None:
        if request.path.startswith('/api/') or request.path.startswith('/json/'):
            RequestNotes.get_notes(request).error_format = 'JSON'
        else:
            RequestNotes.get_notes(request).error_format = 'HTML'

class CsrfFailureError(JsonableError):
    http_status_code: int = 403
    code: ErrorCode = ErrorCode.CSRF_FAILED
    data_fields: List[str] = ['reason']

    def __init__(self, reason: str) -> None:
        self.reason: str = reason

    @staticmethod
    @override
    def msg_format() -> str:
        return _('CSRF error: {reason}')

def csrf_failure(request: HttpRequest, reason: str='') -> HttpResponse:
    if RequestNotes.get_notes(request).error_format == 'JSON':
        return json_response_from_error(CsrfFailureError(reason))
    else:
        return render(request, '4xx.html', context={'csrf_failure': True}, status=403)

class LocaleMiddleware(DjangoLocaleMiddleware):

    @override
    def process_response(self, request: HttpRequest, response: HttpResponseBase) -> HttpResponseBase:
        language: Optional[str] = translation.get_language()
        language_from_path: Optional[str] = translation.get_language_from_path(request.path_info)
        urlconf: Any = getattr(request, 'urlconf', settings.ROOT_URLCONF)
        (i18n_patterns_used, _) = is_language_prefix_patterns_used(urlconf)
        if not (i18n_patterns_used and language_from_path):
            patch_vary_headers(response, ('Accept-Language',))
        assert language is not None
        response.setdefault('Content-Language', language)
        set_language: Optional[str] = RequestNotes.get_notes(request).set_language
        if set_language is not None:
            response.set_cookie(settings.LANGUAGE_COOKIE_NAME, set_language, max_age=settings.LANGUAGE_COOKIE_AGE, path=settings.LANGUAGE_COOKIE_PATH, domain=settings.LANGUAGE_COOKIE_DOMAIN, secure=settings.LANGUAGE_COOKIE_SECURE, httponly=settings.LANGUAGE_COOKIE_HTTPONLY, samesite=settings.LANGUAGE_COOKIE_SAMESITE)
        return response

class RateLimitMiddleware(MiddlewareMixin):

    def set_response_headers(self