import collections
import itertools
import os
import re
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import IO, TYPE_CHECKING, Any, TypeVar, Union, cast, Generator, List, Dict, Optional
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
    with patch('zerver.lib.events.request_event_queue', return_value=event_queue_return), \
         patch('zerver.lib.events.get_user_events', return_value=user_events_return):
        yield


class activate_push_notification_service(override_settings):
    """
    Activating the push notification service involves a few different settings
    that are logically related, and ordinarily set correctly in computed_settings.py
    based on the admin-configured settings.
    Having tests deal with overriding all the necessary settings every time they
    want to simulate using the push notification service would be too
    cumbersome, so we provide a convenient helper.
    Can be used as either a context manager or a decorator applied to a test method
    or class, just like original override_settings.
    """

    def __init__(self, zulip_services_url: Optional[str] = None, submit_usage_statistics: bool = False) -> None:
        if zulip_services_url is None:
            zulip_services_url = settings.ZULIP_SERVICES_URL
        assert zulip_services_url is not None
        if (settings.ANALYTICS_DATA_UPLOAD_LEVEL < AnalyticsDataUploadLevel.BILLING):
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
def func_t9apt0zf() -> Generator[List[tuple], None, None]:
    cache_queries: List[tuple] = []
    orig_get = cache.cache_get
    orig_get_many = cache.cache_get_many

    def func_v864kzzd(key: Any, cache_name: Optional[str] = None) -> Any:
        cache_queries.append(('get', key, cache_name))
        return orig_get(key, cache_name)

    def func_vglze2bt(keys: Iterable[Any], cache_name: Optional[str] = None) -> Dict[Any, Any]:
        cache_queries.append(('getmany', keys, cache_name))
        return orig_get_many(keys, cache_name)

    with patch.multiple(cache, cache_get=func_v864kzzd, cache_get_many=func_vglze2bt):
        yield cache_queries


@contextmanager
def func_vx6zpwe9() -> Generator[List[tuple], None, None]:
    cache_queries: List[tuple] = []

    def func_v864kzzd(key: Any, cache_name: Optional[str] = None) -> Optional[Any]:
        cache_queries.append(('get', key, cache_name))
        return None

    def func_vglze2bt(keys: Iterable[Any], cache_name: Optional[str] = None) -> Dict[Any, Any]:
        cache_queries.append(('getmany', keys, cache_name))
        return {}

    with patch.multiple(cache, cache_get=func_v864kzzd, cache_get_many=func_vglze2bt):
        yield cache_queries


@dataclass
class CapturedQuery:
    sql: str
    time: str


@contextmanager
def func_k09bcmci(include_savepoints: bool = False, keep_cache_warm: bool = False) -> Generator[List[CapturedQuery], None, None]:
    """
    Allow a user to capture just the queries executed during
    the with statement.
    """
    queries: List[CapturedQuery] = []

    def cursor_execute(self: TimeTrackingCursor, sql: Any, vars: Optional[Any] = None) -> Any:
        start = time.time()
        try:
            return super(TimeTrackingCursor, self).execute(sql, vars)
        finally:
            stop = time.time()
            duration = stop - start
            if include_savepoints or not isinstance(sql, str) or 'SAVEPOINT' not in sql:
                captured = CapturedQuery(sql=self.mogrify(sql, vars).decode(), time=f'{duration:.3f}')
                queries.append(captured)

    def cursor_executemany(self: TimeTrackingCursor, sql: Any, vars_list: Iterable[Any]) -> Any:
        vars_list, vars_list1 = itertools.tee(vars_list)
        start = time.time()
        try:
            return super(TimeTrackingCursor, self).executemany(sql, vars_list)
        finally:
            stop = time.time()
            duration = stop - start
            for vars in vars_list1:
                captured = CapturedQuery(sql=self.mogrify(sql, vars).decode(), time=f'{duration:.3f}')
                queries.append(captured)

    if not keep_cache_warm:
        cache_backend = get_cache_backend(None)
        cache_backend.clear()
        flush_per_request_caches()
        clear_client_cache()

    with patch.multiple(TimeTrackingCursor, execute=cursor_execute, executemany=cursor_executemany):
        yield queries


@contextmanager
def func_8gsmu2jp() -> Generator[None, None, None]:
    """Redirect stdout to /dev/null."""
    with open(os.devnull, 'a') as devnull:
        stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield stdout
        finally:
            sys.stdout = stdout


def func_5hxnp5ra() -> None:
    """
    This function is used to reset email visibility for all users and
    RealmUserDefault object in the zulip realm in development environment
    to "EMAIL_ADDRESS_VISIBILITY_EVERYONE" since the default value is
    "EMAIL_ADDRESS_VISIBILITY_ADMINS". This function is needed in
    tests that want "email" field of users to be set to their real email.
    """
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
    key_regex = re.compile(r'accounts/do_confirm/([a-z0-9]{24})>')
    for message in reversed(outbox):
        if address in message.to:
            match = key_regex.search(str(message.body))
            if match:
                (key,) = match.groups()
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
    return Subscription.objects.get(user_profile=user_profile, recipient_id=recipient_id, active=True)


def func_ra4nc2dc(user_profile: UserProfile) -> List[Message]:
    query = UserMessage.objects.select_related('message').filter(user_profile=user_profile).order_by('message')
    return [um.message for um in query]


class DummyHandler(AsyncDjangoHandler):

    def __init__(self) -> None:
        self.handler_id = allocate_handler_id(self)


dummy_handler = DummyHandler()


class HostRequestMock(HttpRequest):
    """A mock request object where get_host() works.  Useful for testing
    routes that use Zulip's subdomains feature"""

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
        for key, value in post_data.items():
            self.POST[key] = str(value)
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
    def get_host(self) -> str:
        return self.host


INSTRUMENTING = os.environ.get('TEST_INSTRUMENT_URL_COVERAGE', '') == 'TRUE'
INSTRUMENTED_CALLS: List[Dict[str, Any]] = []
UrlFuncT = TypeVar('UrlFuncT', bound=Callable[..., HttpResponseBase])


def func_arl3dqxf(data: Dict[str, Any]) -> None:
    INSTRUMENTED_CALLS.append(data)


def func_hnltybvj(f: UrlFuncT) -> UrlFuncT:
    if not INSTRUMENTING:
        return f
    else:

        def wrapper(self: Any, url: str, info: Any = {}, **kwargs: Any) -> HttpResponseBase:
            start = time.time()
            result = f(self, url, info, **kwargs)
            delay = time.time() - start
            test_name = self.id()
            if '?' in url:
                url, extra_info = url.split('?', 1)
            else:
                extra_info = ''
            if isinstance(info, HostRequestMock):
                info_repr = '<HostRequestMock>'
            elif isinstance(info, bytes):
                info_repr = '<bytes>'
            elif isinstance(info, dict):
                info_repr = {k: ('<file object>' if hasattr(v, 'read') and callable(v.read) else v) for k, v in info.items()}
            else:
                info_repr = info
            data = dict(
                url=url,
                status_code=result.status_code,
                method=f.__name__,
                delay=delay,
                extra_info=extra_info,
                info=info_repr,
                test_name=test_name,
                kwargs=kwargs
            )
            func_arl3dqxf(data)
            return result
        return cast(UrlFuncT, wrapper)


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

        def func_q3iyd0c5(patterns: Iterable[Any], prefixes: List[str]) -> None:
            for pattern in patterns:
                find_pattern(pattern, prefixes)

        def find_pattern(pattern: Any, prefixes: List[str]) -> None:
            if isinstance(pattern, URLResolver):
                func_q3iyd0c5(pattern.url_patterns, prefixes)
            else:
                func_2b77iwu4(pattern, prefixes)

        def func_m20joitn(url: str) -> str:
            url = url.removeprefix('/')
            url = url.removeprefix('http://testserver/')
            url = url.removeprefix('http://zulip.testserver/')
            url = url.removeprefix('http://testserver:9080/')
            return url

        def func_2b77iwu4(pattern: Any, prefixes: List[str]) -> None:
            if isinstance(pattern, URLResolver):
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
                        match_url = url[len(prefix):]
                        try:
                            if pattern.resolve(match_url):
                                if call['status_code'] in [200, 204, 301, 302]:
                                    cnt += 1
                                call['pattern'] = canon_pattern
                        except:
                            continue
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
            'testing/(?P<path>.+)', 'scim/v2/', 'scim/v2/.search',
            'scim/v2/Bulk', 'scim/v2/Me',
            'scim/v2/ResourceTypes(?:/(?P<uuid>[^/]+))?',
            'scim/v2/Schemas(?:/(?P<uuid>[^/]+))?',
            'scim/v2/ServiceProviderConfig',
            'scim/v2/Groups(?:/(?P<uuid>[^/]+))?', 'scim/v2/Groups/.search',
            'self-hosted-billing/not-configured/',
        } | {webhook.url for webhook in WEBHOOK_INTEGRATIONS if not include_webhooks}
        untested_patterns -= exempt_patterns
        var_dir = 'var'
        fn = os.path.join(var_dir, 'url_coverage.txt')
        with open(fn, 'wb') as f:
            for call in calls:
                f.write(orjson.dumps(call, option=orjson.OPT_APPEND_NEWLINE))
        if full_suite:
            print(f'INFO: URL coverage report is in {fn}')
        if full_suite and len(untested_patterns):
            print("\nERROR: Some URLs are untested!  Here's the list of untested URLs:")
            for untested_pattern in sorted(untested_patterns):
                print(f'   {untested_pattern}')
            sys.exit(1)


def func_cumav6b5(response: HttpResponseRedirect) -> Dict[str, Any]:
    assert isinstance(response, HttpResponseRedirect)
    token = response.url.rsplit('/', 1)[1]
    data = ExternalAuthResult(
        request=mock.MagicMock(),
        login_token=token,
        delete_stored_data=False
    ).data_dict
    assert data is not None
    return data


P = ParamSpec('P')


def func_jfr3twwn(method: Callable[P, Any]) -> Callable[P, Any]:

    @mock_aws
    @override_settings(LOCAL_UPLOADS_DIR=None)
    @override_settings(LOCAL_AVATARS_DIR=None)
    @override_settings(LOCAL_FILES_DIR=None)
    def func_r6alzj86(*args: P.args, **kwargs: P.kwargs) -> Any:
        backend = S3UploadBackend()
        with patch('zerver.worker.thumbnail.upload_backend', backend), \
             patch('zerver.lib.upload.upload_backend', backend), \
             patch('zerver.views.tusd.upload_backend', backend):
            return method(*args, **kwargs)
    return func_r6alzj86


def func_9srz9506(*bucket_names: str) -> List[Bucket]:
    session = boto3.session.Session(settings.S3_KEY, settings.S3_SECRET_KEY)
    s3 = session.resource('s3')
    buckets = [s3.create_bucket(Bucket=name) for name in bucket_names]
    return buckets


TestCaseT = TypeVar('TestCaseT', bound='MigrationsTestCase')


def func_wdrslnn5(method: Callable[[TestCaseT, StateApps], None]) -> Callable[[TestCaseT, StateApps], None]:

    def method_patched_with_mock(self: TestCaseT, apps: StateApps) -> None:
        ArchivedAttachment = apps.get_model('zerver', 'ArchivedAttachment')
        ArchivedMessage = apps.get_model('zerver', 'ArchivedMessage')
        ArchivedUserMessage = apps.get_model('zerver', 'ArchivedUserMessage')
        Attachment = apps.get_model('zerver', 'Attachment')
        BotConfigData = apps.get_model('zerver', 'BotConfigData')
        BotStorageData = apps.get_model('zerver', 'BotStorageData')
        Client = apps.get_model('zerver', 'Client')
        CustomProfileField = apps.get_model('zerver', 'CustomProfileField')
        CustomProfileFieldValue = apps.get_model('zerver', 'CustomProfileFieldValue')
        DefaultStream = apps.get_model('zerver', 'DefaultStream')
        DefaultStreamGroup = apps.get_model('zerver', 'DefaultStreamGroup')
        EmailChangeStatus = apps.get_model('zerver', 'EmailChangeStatus')
        DirectMessageGroup = apps.get_model('zerver', 'DirectMessageGroup')
        Message = apps.get_model('zerver', 'Message')
        MultiuseInvite = apps.get_model('zerver', 'MultiuseInvite')
        OnboardingStep = apps.get_model('zerver', 'OnboardingStep')
        PreregistrationUser = apps.get_model('zerver', 'PreregistrationUser')
        PushDeviceToken = apps.get_model('zerver', 'PushDeviceToken')
        Reaction = apps.get_model('zerver', 'Reaction')
        Realm = apps.get_model('zerver', 'Realm')
        RealmAuditLog = apps.get_model('zerver', 'RealmAuditLog')
        RealmDomain = apps.get_model('zerver', 'RealmDomain')
        RealmEmoji = apps.get_model('zerver', 'RealmEmoji')
        RealmFilter = apps.get_model('zerver', 'RealmFilter')
        Recipient = apps.get_model('zerver', 'Recipient')
        Recipient.PERSONAL = 1
        Recipient.STREAM = 2
        Recipient.DIRECT_MESSAGE_GROUP = 3
        ScheduledEmail = apps.get_model('zerver', 'ScheduledEmail')
        ScheduledMessage = apps.get_model('zerver', 'ScheduledMessage')
        Service = apps.get_model('zerver', 'Service')
        Stream = apps.get_model('zerver', 'Stream')
        Subscription = apps.get_model('zerver', 'Subscription')
        UserActivity = apps.get_model('zerver', 'UserActivity')
        UserActivityInterval = apps.get_model('zerver', 'UserActivityInterval')
        UserGroup = apps.get_model('zerver', 'UserGroup')
        UserGroupMembership = apps.get_model('zerver', 'UserGroupMembership')
        UserMessage = apps.get_model('zerver', 'UserMessage')
        UserPresence = apps.get_model('zerver', 'UserPresence')
        UserProfile = apps.get_model('zerver', 'UserProfile')
        UserTopic = apps.get_model('zerver', 'UserTopic')
        zerver_models_patch = patch.multiple(
            'zerver.models',
            ArchivedAttachment=ArchivedAttachment,
            ArchivedMessage=ArchivedMessage,
            ArchivedUserMessage=ArchivedUserMessage,
            Attachment=Attachment,
            BotConfigData=BotConfigData,
            BotStorageData=BotStorageData,
            Client=Client,
            CustomProfileField=CustomProfileField,
            CustomProfileFieldValue=CustomProfileFieldValue,
            DefaultStream=DefaultStream,
            DefaultStreamGroup=DefaultStreamGroup,
            EmailChangeStatus=EmailChangeStatus,
            DirectMessageGroup=DirectMessageGroup,
            Message=Message,
            MultiuseInvite=MultiuseInvite,
            UserTopic=UserTopic,
            OnboardingStep=OnboardingStep,
            PreregistrationUser=PreregistrationUser,
            PushDeviceToken=PushDeviceToken,
            Reaction=Reaction,
            Realm=Realm,
            RealmAuditLog=RealmAuditLog,
            RealmDomain=RealmDomain,
            RealmEmoji=RealmEmoji,
            RealmFilter=RealmFilter,
            Recipient=Recipient,
            ScheduledEmail=ScheduledEmail,
            ScheduledMessage=ScheduledMessage,
            Service=Service,
            Stream=Stream,
            Subscription=Subscription,
            UserActivity=UserActivity,
            UserActivityInterval=UserActivityInterval,
            UserGroup=UserGroup,
            UserGroupMembership=UserGroupMembership,
            UserMessage=UserMessage,
            UserPresence=UserPresence,
            UserProfile=UserProfile
        )
        zerver_test_helpers_patch = patch.multiple(
            'zerver.lib.test_helpers',
            Client=Client,
            Message=Message,
            Subscription=Subscription,
            UserMessage=UserMessage,
            UserProfile=UserProfile
        )
        zerver_test_classes_patch = patch.multiple(
            'zerver.lib.test_classes',
            Client=Client,
            Message=Message,
            Realm=Realm,
            Recipient=Recipient,
            Stream=Stream,
            Subscription=Subscription,
            UserProfile=UserProfile
        )
        with zerver_models_patch, zerver_test_helpers_patch, zerver_test_classes_patch:
            method(self, apps)
    return method_patched_with_mock


def func_amenrv4u(filename: str) -> str:
    filepath = os.path.join(settings.TEST_WORKER_DIR, filename)
    with open(filepath, 'w') as f:
        f.write('zulip!')
    return filepath


def func_9ieq9rpx() -> Dict[str, str]:
    return dict(
        emoji_name='zulip',
        emoji_code='zulip',
        reaction_type='zulip_extra_emoji'
    )


@contextmanager
def func_3og1526d(method_to_patch: str, **kwargs: Any) -> Generator[mock.MagicMock, None, None]:
    inner = mock.MagicMock(**kwargs)

    def verify_serialize(queue_name: str, event: Any, processor: Optional[Any] = None) -> None:
        marshalled_event = orjson.loads(orjson.dumps(event))
        assert marshalled_event == event
        inner(queue_name, event, processor)

    with patch(method_to_patch, side_effect=verify_serialize):
        yield inner


@contextmanager
def func_jfckv77c(range_seconds: int, num_requests: int, domain: str = 'api_by_user') -> Generator[None, None, None]:
    """Temporarily add a rate-limiting rule to the rate limiter"""
    RateLimitedIPAddr('127.0.0.1', domain=domain).clear_history()
    domain_rules = rules.get(domain, []).copy()
    domain_rules.append((range_seconds, num_requests))
    domain_rules.sort(key=lambda x: x[0])
    with patch.dict(rules, {domain: domain_rules}), override_settings(RATE_LIMITING=True):
        yield


def func_telaspjx(response: HttpResponse) -> None:
    assert response.streaming
    collections.deque(response, maxlen=0)
