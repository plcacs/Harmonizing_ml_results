import collections
import itertools
import os
import re
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import IO, TYPE_CHECKING, Any, Dict, List, Optional, TypeVar, Union, cast

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
from zerver.models import (
    Client,
    Message,
    RealmUserDefault,
    Subscription,
    UserMessage,
    UserProfile,
)
from zerver.models.clients import clear_client_cache, get_client
from zerver.models.realms import get_realm
from zerver.models.streams import get_stream
from zerver.tornado.handlers import AsyncDjangoHandler, allocate_handler_id
from zilencer.models import RemoteZulipServer
from zproject.backends import ExternalAuthDataDict, ExternalAuthResult

if TYPE_CHECKING:
    from django.test.client import _MonkeyPatchedWSGIResponse as TestHttpResponse
    from zerver.lib.test_classes import MigrationsTestCase, ZulipTestCase


@dataclass
class CapturedQuery:
    sql: str
    time: str


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
def stub_event_queue_user_events(event_queue_return: Any, user_events_return: Any) -> Iterator[None]:
    with mock.patch(
        'zerver.lib.events.request_event_queue',
        return_value=event_queue_return
    ), mock.patch(
        'zerver.lib.events.get_user_events',
        return_value=user_events_return
    ):
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
def cache_tries_captured() -> Iterator[List[Tuple[str, Any, Optional[str]]]]:
    cache_queries: List[Tuple[str, Any, Optional[str]]] = []
    orig_get = cache.cache_get
    orig_get_many = cache.cache_get_many

    def my_cache_get(key: Any, cache_name: Optional[str] = None) -> Any:
        cache_queries.append(('get', key, cache_name))
        return orig_get(key, cache_name)

    def my_cache_get_many(keys: Iterable[Any], cache_name: Optional[str] = None) -> Mapping[Any, Any]:
        cache_queries.append(('getmany', keys, cache_name))
        return orig_get_many(keys, cache_name)

    with mock.patch.multiple(cache, cache_get=my_cache_get, cache_get_many=my_cache_get_many):
        yield cache_queries


@contextmanager
def simulated_empty_cache() -> Iterator[List[Tuple[str, Any, Optional[str]]]]:
    cache_queries: List[Tuple[str, Any, Optional[str]]] = []

    def my_cache_get(key: Any, cache_name: Optional[str] = None) -> Any:
        cache_queries.append(('get', key, cache_name))
        return None

    def my_cache_get_many(keys: Iterable[Any], cache_name: Optional[str] = None) -> Mapping[Any, Any]:
        cache_queries.append(('getmany', keys, cache_name))
        return {}

    with mock.patch.multiple(cache, cache_get=my_cache_get, cache_get_many=my_cache_get_many):
        yield cache_queries


TestCaseT = TypeVar('TestCaseT', bound='MigrationsTestCase')


@contextmanager
def queries_captured(include_savepoints: bool = False, keep_cache_warm: bool = False) -> Iterator[List[CapturedQuery]]:
    """
    Allow a user to capture just the queries executed during
    the with statement.
    """
    queries: List[CapturedQuery] = []

    def cursor_execute(self: TimeTrackingCursor, sql: str, vars: Optional[Any] = None) -> Any:
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

    def cursor_executemany(self, sql: str, vars_list: Iterable[Any]) -> Any:
        vars_list1, _ = itertools.tee(vars_list)
        start = time.time()
        try:
            return super(TimeTrackingCursor, self).executemany(sql, vars_list)
        finally:
            stop = time.time()
            duration = stop - start
            queries.extend(
                CapturedQuery(
                    sql=self.mogrify(sql, vars).decode(),
                    time=f'{duration:.3f}'
                )
                for vars in vars_list1
            )

    if not keep_cache_warm:
        cache_backend = get_cache_backend(None)
        cache_backend.clear()
        flush_per_request_caches()
        clear_client_cache()

    with mock.patch.multiple(TimeTrackingCursor, execute=cursor_execute, executemany=cursor_executemany):
        yield queries


@contextmanager
def stdout_suppressed() -> Iterator[IO[Any]]:
    """Redirect stdout to /dev/null."""
    with open(os.devnull, 'a') as devnull:
        stdout: IO[Any]
        stdout, sys.stdout = sys.stdout, devnull
        try:
            yield stdout
        finally:
            sys.stdout = stdout


def reset_email_visibility_to_everyone_in_zulip_realm() -> None:
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


def get_test_image_file(filename: str) -> IO[bytes]:
    test_avatar_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../tests/images'))
    return open(os.path.join(test_avatar_dir, filename), 'rb')


def read_test_image_file(filename: str) -> bytes:
    with get_test_image_file(filename) as img_file:
        return img_file.read()


def avatar_disk_path(user_profile: UserProfile, medium: bool = False, original: bool = False) -> str:
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


def make_client(name: str) -> Client:
    client, _ = Client.objects.get_or_create(name=name)
    return client


def find_key_by_email(address: str) -> Optional[str]:
    from django.core.mail import outbox
    key_regex = re.compile('accounts/do_confirm/([a-z0-9]{24})>')
    for message in reversed(outbox):
        if address in message.to:
            match = key_regex.search(str(message.body))
            assert match is not None
            [key] = match.groups()
            return key
    return None


def message_stream_count(user_profile: UserProfile) -> int:
    return UserMessage.objects.select_related('message').filter(user_profile=user_profile).count()


def most_recent_usermessage(user_profile: UserProfile) -> UserMessage:
    query = UserMessage.objects.select_related('message').filter(user_profile=user_profile).order_by('-message')
    return query[0]


def most_recent_message(user_profile: UserProfile) -> Message:
    usermessage = most_recent_usermessage(user_profile)
    return usermessage.message


def get_subscription(stream_name: str, user_profile: UserProfile) -> Subscription:
    stream = get_stream(stream_name, user_profile.realm)
    recipient_id = stream.recipient_id
    assert recipient_id is not None
    return Subscription.objects.get(user_profile=user_profile, recipient_id=recipient_id, active=True)


def get_user_messages(user_profile: UserProfile) -> List[Message]:
    query = UserMessage.objects.select_related('message').filter(user_profile=user_profile).order_by('message')
    return [um.message for um in query]


class DummyHandler(AsyncDjangoHandler):

    def __init__(self) -> None:
        self.handler_id: int = allocate_handler_id(self)


dummy_handler: DummyHandler = DummyHandler()


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
    def get_host(self) -> str:
        return self.host


INSTRUMENTING: bool = os.environ.get('TEST_INSTRUMENT_URL_COVERAGE', '') == 'TRUE'
INSTRUMENTED_CALLS: List[Dict[str, Any]] = []
UrlFuncT = TypeVar('UrlFuncT', bound=Callable[..., HttpResponseBase])
P = ParamSpec('P')


def append_instrumentation_data(data: Dict[str, Any]) -> None:
    INSTRUMENTED_CALLS.append(data)


def instrument_url(f: UrlFuncT) -> UrlFuncT:
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
                info = '<HostRequestMock>'
            elif isinstance(info, bytes):
                info = '<bytes>'
            elif isinstance(info, dict):
                info = {k: '<file object>' if hasattr(v, 'read') and callable(v.read) else v for k, v in info.items()}
            append_instrumentation_data(dict(
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

        return cast(UrlFuncT, wrapper)


def write_instrumentation_reports(full_suite: bool, include_webhooks: bool) -> None:
    if INSTRUMENTING:
        calls = INSTRUMENTED_CALLS
        from zproject.urls import urlpatterns, v1_api_and_json_patterns
        pattern_cnt: Dict[str, int] = collections.defaultdict(int)

        def re_strip(r: str) -> str:
            assert r.startswith('^')
            if r.endswith('$'):
                return r[1:-1]
            else:
                assert r.endswith('\\Z')
                return r[1:-2]

        def find_patterns(patterns: Iterable[Any], prefixes: List[str]) -> None:
            for pattern in patterns:
                find_pattern(pattern, prefixes)

        def cleanup_url(url: str) -> str:
            url = url.removeprefix('/')
            url = url.removeprefix('http://testserver/')
            url = url.removeprefix('http://zulip.testserver/')
            url = url.removeprefix('http://testserver:9080/')
            return url

        def find_pattern(pattern: Any, prefixes: List[str]) -> None:
            if isinstance(pattern, type(URLResolver)):
                return
            if hasattr(pattern, 'url_patterns'):
                return
            canon_pattern = prefixes[0] + re_strip(pattern.pattern.regex.pattern)
            cnt = 0
            for call in calls:
                if 'pattern' in call:
                    continue
                url = cleanup_url(call['url'])
                for prefix in prefixes:
                    if url.startswith(prefix):
                        match_url = url.removeprefix(prefix)
                        if pattern.resolve(match_url):
                            if call['status_code'] in [200, 204, 301, 302]:
                                cnt += 1
                            call['pattern'] = canon_pattern
            pattern_cnt[canon_pattern] += cnt

        find_patterns(urlpatterns, ['', 'en/', 'de/'])
        find_patterns(v1_api_and_json_patterns, ['api/v1/', 'json/'])
        assert len(pattern_cnt) > 100
        untested_patterns = {p.replace('\\', '') for p in pattern_cnt if pattern_cnt[p] == 0}
        exempt_patterns = {
            'api/v1/events',
            'api/v1/events/internal',
            'api/v1/register',
            'coverage/(?P<path>.+)',
            'config-error/(?P<error_name>[^/]+)',
            'confirmation_key/',
            'node-coverage/(?P<path>.+)',
            'docs/',
            'docs/(?P<path>.+)',
            'casper/(?P<path>.+)',
            'static/(?P<path>.*)',
            'flush_caches',
            'external_content/(?P<digest>[^/]+)/(?P<received_url>[^/]+)',
            'testing/(?P<path>.+)',
            'scim/v2/',
            'scim/v2/.search',
            'scim/v2/Bulk',
            'scim/v2/Me',
            'scim/v2/ResourceTypes(?:/(?P<uuid>[^/]+))?',
            'scim/v2/Schemas(?:/(?P<uuid>[^/]+))?',
            'scim/v2/ServiceProviderConfig',
            'scim/v2/Groups(?:/(?P<uuid>[^/]+))?',
            'scim/v2/Groups/.search',
            'self-hosted-billing/not-configured/',
            *(webhook.url for webhook in WEBHOOK_INTEGRATIONS if not include_webhooks)
        }
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


def load_subdomain_token(response: HttpResponseRedirect) -> Dict[str, Any]:
    assert isinstance(response, HttpResponseRedirect)
    token = response.url.rsplit('/', 1)[1]
    data = ExternalAuthResult(
        request=mock.MagicMock(),
        login_token=token,
        delete_stored_data=False
    ).data_dict
    assert data is not None
    return data


def use_s3_backend(method: Callable[..., Any]) -> Callable[..., Any]:
    @mock_aws
    @override_settings(LOCAL_UPLOADS_DIR=None)
    @override_settings(LOCAL_AVATARS_DIR=None)
    @override_settings(LOCAL_FILES_DIR=None)
    def new_method(*args: Any, **kwargs: Any) -> Any:
        backend = S3UploadBackend()
        with mock.patch('zerver.worker.thumbnail.upload_backend', backend), \
             mock.patch('zerver.lib.upload.upload_backend', backend), \
             mock.patch('zerver.views.tusd.upload_backend', backend):
            return method(*args, **kwargs)
    return new_method


def create_s3_buckets(*bucket_names: str) -> List[Bucket]:
    session = boto3.session.Session(settings.S3_KEY, settings.S3_SECRET_KEY)
    s3 = session.resource('s3')
    buckets: List[Bucket] = [s3.create_bucket(Bucket=name) for name in bucket_names]
    return buckets


def use_db_models(method: Callable[[TestCaseT, StateApps], Any]) -> Callable[[TestCaseT, StateApps], Any]:
    def method_patched_with_mock(self: TestCaseT, apps: StateApps) -> Any:
        ArchivedAttachment = apps.get_model('zerver', 'ArchivedAttachment')
        ArchivedMessage = apps.get_model('zerver', 'ArchivedMessage')
        ArchivedUserMessage = apps.get_model('zerver', 'ArchivedUserMessage')
        Attachment = apps.get_model('zerver', 'Attachment')
        BotConfigData = apps.get_model('zerver', 'BotConfigData')
        BotStorageData = apps.get_model('zerver', 'BotStorageData')
        ClientModel = apps.get_model('zerver', 'Client')
        CustomProfileField = apps.get_model('zerver', 'CustomProfileField')
        CustomProfileFieldValue = apps.get_model('zerver', 'CustomProfileFieldValue')
        DefaultStream = apps.get_model('zerver', 'DefaultStream')
        DefaultStreamGroup = apps.get_model('zerver', 'DefaultStreamGroup')
        EmailChangeStatus = apps.get_model('zerver', 'EmailChangeStatus')
        DirectMessageGroup = apps.get_model('zerver', 'DirectMessageGroup')
        MessageModel = apps.get_model('zerver', 'Message')
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
        StreamModel = apps.get_model('zerver', 'Stream')
        SubscriptionModel = apps.get_model('zerver', 'Subscription')
        UserActivity = apps.get_model('zerver', 'UserActivity')
        UserActivityInterval = apps.get_model('zerver', 'UserActivityInterval')
        UserGroup = apps.get_model('zerver', 'UserGroup')
        UserGroupMembership = apps.get_model('zerver', 'UserGroupMembership')
        UserMessageModel = apps.get_model('zerver', 'UserMessage')
        UserPresence = apps.get_model('zerver', 'UserPresence')
        UserProfileModel = apps.get_model('zerver', 'UserProfile')
        UserTopic = apps.get_model('zerver', 'UserTopic')
        zerver_models_patch = mock.patch.multiple(
            'zerver.models',
            ArchivedAttachment=ArchivedAttachment,
            ArchivedMessage=ArchivedMessage,
            ArchivedUserMessage=ArchivedUserMessage,
            Attachment=Attachment,
            BotConfigData=BotConfigData,
            BotStorageData=BotStorageData,
            Client=ClientModel,
            CustomProfileField=CustomProfileField,
            CustomProfileFieldValue=CustomProfileFieldValue,
            DefaultStream=DefaultStream,
            DefaultStreamGroup=DefaultStreamGroup,
            EmailChangeStatus=EmailChangeStatus,
            DirectMessageGroup=DirectMessageGroup,
            Message=MessageModel,
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
            Stream=StreamModel,
            Subscription=SubscriptionModel,
            UserActivity=UserActivity,
            UserActivityInterval=UserActivityInterval,
            UserGroup=UserGroup,
            UserGroupMembership=UserGroupMembership,
            UserMessage=UserMessageModel,
            UserPresence=UserPresence,
            UserProfile=UserProfileModel
        )
        zerver_test_helpers_patch = mock.patch.multiple(
            'zerver.lib.test_helpers',
            Client=ClientModel,
            Message=MessageModel,
            Subscription=SubscriptionModel,
            UserMessage=UserMessageModel,
            UserProfile=UserProfileModel
        )
        zerver_test_classes_patch = mock.patch.multiple(
            'zerver.lib.test_classes',
            Client=ClientModel,
            Message=MessageModel,
            Realm=Realm,
            Recipient=Recipient,
            Stream=StreamModel,
            Subscription=SubscriptionModel,
            UserProfile=UserProfileModel
        )
        with zerver_models_patch, zerver_test_helpers_patch, zerver_test_classes_patch:
            method(self, apps)
    return method_patched_with_mock


def create_dummy_file(filename: str) -> str:
    filepath = os.path.join(settings.TEST_WORKER_DIR, filename)
    with open(filepath, 'w') as f:
        f.write('zulip!')
    return filepath


def zulip_reaction_info() -> Dict[str, str]:
    return dict(
        emoji_name='zulip',
        emoji_code='zulip',
        reaction_type='zulip_extra_emoji'
    )


@contextmanager
def mock_queue_publish(method_to_patch: str, **kwargs: Any) -> Iterator[mock.MagicMock]:
    inner = mock.MagicMock(**kwargs)

    def verify_serialize(queue_name: str, event: Any, processor: Optional[Any] = None) -> None:
        marshalled_event = orjson.loads(orjson.dumps(event))
        assert marshalled_event == event
        inner(queue_name, event, processor)

    with mock.patch(method_to_patch, side_effect=verify_serialize):
        yield inner


@contextmanager
def ratelimit_rule(range_seconds: int, num_requests: int, domain: str = 'api_by_user') -> Iterator[None]:
    """Temporarily add a rate-limiting rule to the rate limiter"""
    RateLimitedIPAddr('127.0.0.1', domain=domain).clear_history()
    domain_rules = rules.get(domain, []).copy()
    domain_rules.append((range_seconds, num_requests))
    domain_rules.sort(key=lambda x: x[0])
    with patch.dict(rules, {domain: domain_rules}), override_settings(RATE_LIMITING=True):
        yield


def consume_response(response: HttpResponseBase) -> None:
    assert response.streaming
    collections.deque(response, maxlen=0)
