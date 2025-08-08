import collections
import itertools
import os
import re
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import IO, TYPE_CHECKING, Any, TypeVar, Union, cast, Optional, List, Dict, Set, Tuple, Generator, ContextManager
from unittest import mock
from unittest.mock import patch
import boto3.session
import fakeldap
import ldap
import orjson
from django.conf import settings
from django.contrib.auth.models import AnonymousUser
from django.db.migrations.state import StateApps
from django.http import HttpRequest, HttpResponse, HttpResponseRedirect
from django.http.request import QueryDict
from django.http.response import HttpResponseBase
from django.test import override_settings
from django.urls import URLResolver
from moto.core.decorator import mock_aws
from mypy_boto3_s3.service_resource import Bucket
from typing_extensions import ParamSpec, override
from zerver.actions.realm_settings import do_set_realm_user_default_setting
from zerver.actions.user_settings import do_change_user_setting
from zerver.lib import cache
from zerver.lib.avatar import avatar_url
from zerver.lib.cache import get_cache_backend
from zerver.lib.db import Params, Query, TimeTrackingCursor
from zerver.lib.integrations import WEBHOOK_INTEGRATIONS
from zerver.lib.per_request_cache import flush_per_request_caches
from zerver.lib.rate_limiter import RateLimitedIPAddr, rules
from zerver.lib.request import RequestNotes
from zerver.lib.types import AnalyticsDataUploadLevel
from zerver.lib.upload.s3 import S3UploadBackend
from zerver.models import Client, Message, RealmUserDefault, Subscription, UserMessage, UserProfile
from zerver.models.clients import clear_client_cache, get_client
from zerver.models.realms import get_realm
from zerver.models.streams import get_stream
from zerver.tornado.handlers import AsyncDjangoHandler, allocate_handler_id
from zilencer.models import RemoteZulipServer
from zproject.backends import ExternalAuthDataDict, ExternalAuthResult

if TYPE_CHECKING:
    from django.test.client import _MonkeyPatchedWSGIResponse as TestHttpResponse
    from zerver.lib.test_classes import MigrationsTestCase, ZulipTestCase

T = TypeVar('T')
P = ParamSpec('P')
UrlFuncT = TypeVar('UrlFuncT', bound=Callable[..., HttpResponseBase])
TestCaseT = TypeVar('TestCaseT', bound='MigrationsTestCase')

class MockLDAP(fakeldap.MockLDAP):
    class LDAPError(ldap.LDAPError):
        pass

    class INVALID_CREDENTIALS(ldap.INVALID_CREDENTIALS):
        pass

    class NO_SUCH_OBJECT(ldap.NO_SUCH_OBJECT):
        pass

    class ALREADY_EXISTS(ldap.ALREADY_EXISTS):
        pass

@contextmanager
def func_h4f4o9kj(event_queue_return: Any, user_events_return: Any) -> Generator[None, None, None]:
    with mock.patch('zerver.lib.events.request_event_queue', return_value=event_queue_return), \
         mock.patch('zerver.lib.events.get_user_events', return_value=user_events_return):
        yield

class activate_push_notification_service(override_settings):
    def __init__(self, zulip_services_url: Optional[str] = None, submit_usage_statistics: bool = False) -> None:
        if zulip_services_url is None:
            zulip_services_url = settings.ZULIP_SERVICES_URL
        assert zulip_services_url is not None
        if settings.ANALYTICS_DATA_UPLOAD_LEVEL < AnalyticsDataUploadLevel.BILLING:
            analytics_data_upload_level = AnalyticsDataUploadLevel.BILLING
        else:
            analytics_data_upload_level = settings.ANALYTICS_DATA_UPLOAD_LEVEL
        if submit_usage_statistics:
            analytics_data_upload_level = AnalyticsDataUploadLevel.ALL
        super().__init__(
            ZULIP_SERVICES_URL=zulip_services_url,
            ANALYTICS_DATA_UPLOAD_LEVEL=analytics_data_upload_level,
            ZULIP_SERVICE_PUSH_NOTIFICATIONS=True,
            ZULIP_SERVICE_SUBMIT_USAGE_STATISTICS=submit_usage_statistics
        )

@contextmanager
def func_t9apt0zf() -> Generator[List[Tuple[str, Any, Optional[str]]], None, None]:
    cache_queries: List[Tuple[str, Any, Optional[str]]] = []
    orig_get = cache.cache_get
    orig_get_many = cache.cache_get_many

    def func_v864kzzd(key: Any, cache_name: Optional[str] = None) -> Any:
        cache_queries.append(('get', key, cache_name))
        return orig_get(key, cache_name)

    def func_vglze2bt(keys: Iterable[Any], cache_name: Optional[str] = None) -> Dict[Any, Any]:
        cache_queries.append(('getmany', keys, cache_name))
        return orig_get_many(keys, cache_name)

    with mock.patch.multiple(cache, cache_get=func_v864kzzd, cache_get_many=func_vglze2bt):
        yield cache_queries

@contextmanager
def func_vx6zpwe9() -> Generator[List[Tuple[str, Any, Optional[str]]], None, None]:
    cache_queries: List[Tuple[str, Any, Optional[str]]] = []

    def func_v864kzzd(key: Any, cache_name: Optional[str] = None) -> None:
        cache_queries.append(('get', key, cache_name))
        return None

    def func_vglze2bt(keys: Iterable[Any], cache_name: Optional[str] = None) -> Dict[Any, Any]:
        cache_queries.append(('getmany', keys, cache_name))
        return {}

    with mock.patch.multiple(cache, cache_get=func_v864kzzd, cache_get_many=func_vglze2bt):
        yield cache_queries

@dataclass
class CapturedQuery:
    sql: str
    time: str

@contextmanager
def func_k09bcmci(include_savepoints: bool = False, keep_cache_warm: bool = False) -> Generator[List[CapturedQuery], None, None]:
    queries: List[CapturedQuery] = []

    def func_akqbg79w(self: TimeTrackingCursor, sql: str, vars: Optional[Any] = None) -> Any:
        start = time.time()
        try:
            return super(TimeTrackingCursor, self).execute(sql, vars)
        finally:
            stop = time.time()
            duration = stop - start
            if include_savepoints or not isinstance(sql, str) or 'SAVEPOINT' not in sql:
                queries.append(CapturedQuery(
                    sql=self.mogrify(sql, vars).decode(),
                    time=f'{duration:.3f}'
                ))

    def func_p1mwwahm(self: TimeTrackingCursor, sql: str, vars_list: Iterable[Any]) -> Any:
        vars_list, vars_list1 = itertools.tee(vars_list)
        start = time.time()
        try:
            return super(TimeTrackingCursor, self).executemany(sql, vars_list)
        finally:
            stop = time.time()
            duration = stop - start
            queries.extend(
                CapturedQuery(sql=self.mogrify(sql, vars).decode(), time=f'{duration:.3f}')
                for vars in vars_list1
            )

    if not keep_cache_warm:
        cache = get_cache_backend(None)
        cache.clear()
        flush_per_request_caches()
        clear_client_cache()

    with mock.patch.multiple(TimeTrackingCursor, execute=func_akqbg79w, executemany=func_p1mwwahm):
        yield queries

@contextmanager
def func_8gsmu2jp() -> Generator[IO[str], None, None]:
    with open(os.devnull, 'a') as devnull:
        stdout, sys.stdout = sys.stdout, devnull
        try:
            yield stdout
        finally:
            sys.stdout = stdout

def func_5hxnp5ra() -> None:
    realm = get_realm('zulip')
    realm_user_default = RealmUserDefault.objects.get(realm=realm)
    do_set_realm_user_default_setting(
        realm_user_default,
        'email_address_visibility',
        RealmUserDefault.EMAIL_ADDRESS_VISIBILITY_EVERYONE,
        acting_user=None
    )
    users = UserProfile.objects.filter(realm=realm)
    for user in users:
        do_change_user_setting(
            user,
            'email_address_visibility',
            UserProfile.EMAIL_ADDRESS_VISIBILITY_EVERYONE,
            acting_user=None
        )

def func_iup7b1h7(filename: str) -> IO[bytes]:
    test_avatar_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../tests/images'))
    return open(os.path.join(test_avatar_dir, filename), 'rb')

def func_385n725b(filename: str) -> bytes:
    with func_iup7b1h7(filename) as img_file:
        return img_file.read()

def func_qxrvbbl5(user_profile: UserProfile, medium: bool = False, original: bool = False) -> str:
    avatar_url_path = avatar_url(user_profile, medium)
    assert avatar_url_path is not None
    assert settings.LOCAL_UPLOADS_DIR is not None
    assert settings.LOCAL_AVATARS_DIR is not None
    avatar_disk_path = os.path.join(
        settings.LOCAL_AVATARS_DIR,
        avatar_url_path.split('/')[-2],
        avatar_url_path.split('/')[-1]
    )
    if original:
        return avatar_disk_path.replace('.png', '.original')
    return avatar_disk_path

def func_f96jcwgc(name: str) -> Client:
    client, _ = Client.objects.get_or_create(name=name)
    return client

def func_eipoq8gp(address: str) -> Optional[str]:
    from django.core.mail import outbox
    key_regex = re.compile('accounts/do_confirm/([a-z0-9]{24})>')
    for message in reversed(outbox):
        if address in message.to:
            match = key_regex.search(str(message.body))
            assert match is not None
            [key] = match.groups()
            return key
    return None

def func_e5wi3mcl(user_profile: UserProfile) -> int:
    return UserMessage.objects.select_related('message').filter(user_profile=user_profile).count()

def func_qo4tjy0l(user_profile: UserProfile) -> UserMessage:
    query = UserMessage.objects.select_related('message').filter(user_profile=user_profile).order_by('-message')
    return query[0]

def func_3beykl6l(user_profile: UserProfile) -> Message:
    usermessage = func_qo4tjy0l(user_profile)
    return usermessage.message

def func_q665dbg1(stream_name: str, user_profile: UserProfile) -> Subscription:
    stream = get_stream(stream_name, user_profile.realm)
    recipient_id = stream.recipient_id
    assert recipient_id is not None
    return Subscription.objects.get(
        user_profile=user_profile,
        recipient_id=recipient_id,
        active=True
    )

def func_ra4nc2dc(user_profile: UserProfile) -> List[Message]:
    query = UserMessage.objects.select_related('message').filter(user_profile=user_profile).order_by('message')
    return [um.message for um in query]

class DummyHandler(AsyncDjangoHandler):
    def __init__(self) -> None:
        self.handler_id = allocate_handler_id(self)

dummy_handler = DummyHandler()

class HostRequestMock(HttpRequest):
    def __init__(
        self,
        post_data: Dict[str, Any] = {},
        user_profile: Optional[UserProfile] = None,
        remote_server: Optional[RemoteZulipServer] = None,
        host: str = settings.EXTERNAL_HOST,
        client_name: Optional[str] = None,
        meta_data: Optional[Dict[str, Any]] = None,
        tornado_handler: Optional[DummyHandler] = None,
        path: str = ''
    ) -> None:
        self.host = host
        self.GET = QueryDict(mutable=True)
        self.method = ''
        self.POST = QueryDict(mutable=True)
        for key in post_data:
            self.POST[key] = str(post_data[key])
            self.method = 'POST'
        if meta_data is None:
            self.META = {'PATH_INFO': 'test'}
        else:
            self.META = meta_data
        self.path = path
        self.user = user_profile or AnonymousUser()
        self._body = orjson.dumps(post_data)
        self.content_type = ''
        RequestNotes.set_notes(
            self,
            RequestNotes(
                client_name='',
                log_data={},
                tornado_handler_id=None if tornado_handler is None else tornado_handler.handler_id,
                client=get_client(client_name) if client_name is not None else None,
                remote_server=remote_server
            )
        )

    @override
    def func_gs1ixbjw(self) -> str:
        return self.host

INSTRUMENTING = os.environ.get('TEST_INSTRUMENT_URL_COVERAGE', '') == 'TRUE'
INSTRUMENTED_CALLS: List[Dict[str, Any]] = []

def func_arl3dqxf(data: Dict[str, Any]) -> None:
    INSTRUMENTED_CALLS.append(data)

def func_hnltybvj(f: UrlFuncT) -> UrlFuncT:
    if not INSTRUMENTING:
        return f
    else:
        def func_cqio91ix(self: Any, url: str, info: Dict[str, Any] = {}, **kwargs: Any) -> HttpResponseBase:
            start = time.time()
            result = f(self, url, info, **kwargs)
            delay = time.time() - start
            test_name = self.id()
            if '?' in url:
                url, extra_info = url.split('?', 1)
            else:
                extra_info = ''
            if isinstance(info, HostRequestMock):
                info = '<HostRequestMock>'
            elif isinstance(info, bytes):
                info = '<bytes>'
            elif isinstance(info, dict):
                info = {
                    k: ('<file object>' if hasattr(v, 'read') and callable(v.read) else v)
                    for k, v in info.items()
                }
            func_arl3dqxf(dict(
                url=url,
                status_code=result.status_code,
                method=f.__name__,
                delay=delay,
                extra_info=extra_info,
                info=info,
                test_name=test_name,
                kwargs=kwargs
            ))
            return result
        return cast(UrlFuncT, func_cqio91ix)

def func_b57nk41d(full_suite: bool, include_webhooks: bool) -> None:
    if INSTRUMENTING:
        calls = INSTRUMENTED_CALLS
        from zproject.urls import urlpatterns, v1_api_and_json_patterns
        pattern_cnt: Dict[str, int] = collections.defaultdict(int)

        def func_h3gkgmj4(r: str) -> str:
            assert r.startswith('^')
            if r.endswith('$'):
                return r[1:-1]
            else:
                assert r.endswith('\\Z')
                return r[1:-2]

        def func_q3iyd0c5(patterns: List[URLResolver], prefixes: List[str]) -> None:
            for pattern in patterns:
                find_pattern(pattern, prefixes)

        def func_m20joitn(url: str) -> str:
            url = url.removeprefix('/')
            url = url.removeprefix('http://testserver/')
            url = url.removeprefix('http://zulip.testserver/')
            url = url.removeprefix('http://testserver:9080/')
            return url

        def func_2b77iwu4(pattern: URLResolver, prefixes: List[str]) -> None:
            if isinstance(pattern, type(URLResolver)):
                return
            if hasattr(pattern, 'url_patterns'):
                return
            canon_pattern = prefixes[0] + func_h3gkgmj4(pattern.pattern.regex.pattern)
            cnt = 0
            for call in calls:
                if 'pattern' in call:
                    continue
                url = func_m20joitn(call['url'])
                for prefix in prefixes:
                    if url.startswith(prefix):
                        match_url = url.removeprefix(prefix)
                        if pattern.resolve(match_url):
                            if call['status_code'] in [200, 204, 301, 302]:
                                cnt += 1
                            call['pattern'] = canon_pattern
            pattern_cnt[canon_pattern] += cnt

        func_q3iyd0c5(urlpatterns, ['', 'en/', 'de/'])
        func_q3iyd0c5(v1_api_and_json_patterns, ['api/v1/', 'json/'])
        assert len(pattern_cnt) > 100
        untested_patterns = {p.replace('\\', '') for p in pattern_cnt if pattern_cnt[p] == 0}
        exempt_patterns = {
            'api/v1/events', 'api/v1/events/internal', 'api/v1/register',
            'coverage/(?P<path>.+)', 'config-error/(?P<error_name>[^/]+)',
            'confirmation_key/', 'node-coverage/(?P<path>.+)', 'docs/',
            'docs/(?P<path>.+)', 'casper/(?P<path>.+)', 'static/(?P<path>.*)',
            'flush_caches', 'external_content/(?P<digest>[^/]+)/(?P<received_url>[^/]+)',
            'testing/(?P<path>.+)', 'scim/v2/', 'scim/v2/.search', 'scim/v2/Bulk',
            'scim/v2/Me', 'scim/v2/ResourceTypes(?:/(?P<uuid>[^/]+))