import base64
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Union, cast, Dict, List, Optional, Set, Tuple, TypeVar, Type, Generator, ContextManager
from unittest import TestResult, mock, skipUnless
from urllib.parse import parse_qs, quote, urlencode
import lxml.html
import orjson
import responses
from django.apps import apps
from django.conf import settings
from django.core.files.uploadedfile import UploadedFile
from django.core.mail import EmailMessage
from django.core.signals import got_request_exception
from django.db import connection, transaction
from django.db.migrations.executor import MigrationExecutor
from django.db.migrations.state import StateApps
from django.db.models import QuerySet
from django.db.utils import IntegrityError
from django.http import HttpRequest, HttpResponse, HttpResponseBase
from django.http.response import ResponseHeaders
from django.test import Client as TestClient
from django.test import SimpleTestCase, TestCase, TransactionTestCase
from django.test.client import BOUNDARY, MULTIPART_CONTENT, ClientHandler, encode_multipart
from django.test.testcases import SerializeMixin
from django.urls import resolve
from django.utils import translation
from django.utils.module_loading import import_string
from django.utils.timezone import now as timezone_now
from fakeldap import MockLDAP
from openapi_core.contrib.django import DjangoOpenAPIRequest, DjangoOpenAPIResponse
from requests import PreparedRequest
from two_factor.plugins.phonenumber.models import PhoneDevice
from typing_extensions import override
from corporate.models import Customer, CustomerPlan, LicenseLedger
from zerver.actions.message_send import check_send_message, check_send_stream_message
from zerver.actions.realm_settings import do_change_realm_permission_group_setting
from zerver.actions.streams import bulk_add_subscriptions, bulk_remove_subscriptions
from zerver.decorator import do_two_factor_login
from zerver.lib.cache import bounce_key_prefix_for_testing
from zerver.lib.email_notifications import MissedMessageData, handle_missedmessage_emails
from zerver.lib.initial_password import initial_password
from zerver.lib.mdiff import diff_strings
from zerver.lib.message import access_message
from zerver.lib.notification_data import UserMessageNotificationsData
from zerver.lib.per_request_cache import flush_per_request_caches
from zerver.lib.redis_utils import bounce_redis_key_prefix_for_testing
from zerver.lib.response import MutableJsonResponse
from zerver.lib.sessions import get_session_dict_user
from zerver.lib.soft_deactivation import do_soft_deactivate_users
from zerver.lib.stream_subscription import get_subscribed_stream_ids_for_user
from zerver.lib.streams import create_stream_if_needed, get_default_value_for_history_public_to_subscribers, get_default_values_for_stream_permission_group_settings
from zerver.lib.subscription_info import gather_subscriptions
from zerver.lib.test_console_output import ExtraConsoleOutputFinder, ExtraConsoleOutputInTestError, tee_stderr_and_find_extra_console_output, tee_stdout_and_find_extra_console_output
from zerver.lib.test_helpers import cache_tries_captured, find_key_by_email, get_test_image_file, instrument_url, queries_captured
from zerver.lib.thumbnail import ThumbnailFormat
from zerver.lib.topic import RESOLVED_TOPIC_PREFIX, filter_by_topic_name_via_message
from zerver.lib.upload import upload_message_attachment_from_request
from zerver.lib.user_groups import get_system_user_group_for_user
from zerver.lib.webhooks.common import check_send_webhook_message, get_fixture_http_headers, standardize_headers
from zerver.models import Client, Message, NamedUserGroup, PushDeviceToken, Reaction, Realm, RealmEmoji, Recipient, Stream, Subscription, UserGroup, UserGroupMembership, UserMessage, UserProfile, UserStatus
from zerver.models.realms import clear_supported_auth_backends_cache, get_realm
from zerver.models.streams import get_realm_stream, get_stream
from zerver.models.users import get_system_bot, get_user, get_user_by_delivery_email
from zerver.openapi.openapi import validate_test_request, validate_test_response
from zerver.tornado.event_queue import clear_client_event_queues_for_testing

if settings.ZILENCER_ENABLED:
    from zilencer.models import RemoteZulipServer, get_remote_server_by_uuid

if TYPE_CHECKING:
    from django.test.client import _MonkeyPatchedWSGIResponse as TestHttpResponse

T = TypeVar('T')

class EmptyResponseError(Exception):
    pass

class UploadSerializeMixin(SerializeMixin):
    lockfile: str = 'var/upload_lock'

    @classmethod
    @override
    def setUpClass(cls) -> None:
        if not os.path.exists(cls.lockfile):
            with open(cls.lockfile, 'w'):
                pass
        super().setUpClass()

class ZulipClientHandler(ClientHandler):
    @override
    def get_response(self, request: HttpRequest) -> HttpResponseBase:
        got_exception = False

        def on_exception(**kwargs: Any) -> None:
            nonlocal got_exception
            if kwargs['request'] is request:
                got_exception = True
        request.body
        got_request_exception.connect(on_exception)
        try:
            response = super().get_response(request)
        finally:
            got_request_exception.disconnect(on_exception)
        if not got_exception and request.method != 'OPTIONS' and isinstance(response, HttpResponse) and (not (response.status_code == 302 and response.headers['Location'].startswith('/login/'))):
            openapi_request = DjangoOpenAPIRequest(request)
            openapi_response = DjangoOpenAPIResponse(response)
            response_validated = validate_test_response(openapi_request, openapi_response)
            if response_validated:
                validate_test_request(openapi_request, str(response.status_code), request.META.get('intentionally_undocumented', False))
        return response

class ZulipTestClient(TestClient):
    def __init__(self) -> None:
        super().__init__()
        self.handler = ZulipClientHandler(enforce_csrf_checks=False)

class ZulipTestCaseMixin(SimpleTestCase):
    maxDiff: Optional[int] = None
    expected_console_output: Optional[str] = None
    client_class: Type[TestClient] = ZulipTestClient
    DEFAULT_SUBDOMAIN: str = 'zulip'
    TOKENIZED_NOREPLY_REGEX: str = settings.TOKENIZED_NOREPLY_EMAIL_ADDRESS.format(token='[a-z0-9_]{24}')
    INVALID_STREAM_ID: int = 999999

    example_user_map: Dict[str, str] = {
        'hamlet': 'hamlet@zulip.com',
        'cordelia': 'cordelia@zulip.com',
        'iago': 'iago@zulip.com',
        'prospero': 'prospero@zulip.com',
        'othello': 'othello@zulip.com',
        'AARON': 'AARON@zulip.com',
        'aaron': 'aaron@zulip.com',
        'ZOE': 'ZOE@zulip.com',
        'polonius': 'polonius@zulip.com',
        'desdemona': 'desdemona@zulip.com',
        'shiva': 'shiva@zulip.com',
        'webhook_bot': 'webhook-bot@zulip.com',
        'outgoing_webhook_bot': 'outgoing-webhook@zulip.com',
        'default_bot': 'default-bot@zulip.com'
    }
    mit_user_map: Dict[str, str] = {
        'sipbtest': 'sipbtest@mit.edu',
        'starnine': 'starnine@mit.edu',
        'espuser': 'espuser@mit.edu'
    }
    lear_user_map: Dict[str, str] = {
        'cordelia': 'cordelia@zulip.com',
        'king': 'king@lear.org'
    }
    nonreg_user_map: Dict[str, str] = {
        'test': 'test@zulip.com',
        'test1': 'test1@zulip.com',
        'alice': 'alice@zulip.com',
        'newuser': 'newuser@zulip.com',
        'bob': 'bob@zulip.com',
        'cordelia': 'cordelia@zulip.com',
        'newguy': 'newguy@zulip.com',
        'me': 'me@zulip.com'
    }
    example_user_ldap_username_map: Dict[str, str] = {
        'hamlet': 'hamlet',
        'cordelia': 'cordelia',
        'aaron': 'letham'
    }
    API_KEYS: Dict[str, str] = {}

    @override
    def setUp(self) -> None:
        super().setUp()
        self.API_KEYS = {}
        test_name = self.id()
        bounce_key_prefix_for_testing(test_name)
        bounce_redis_key_prefix_for_testing(test_name)

    @override
    def tearDown(self) -> None:
        super().tearDown()
        clear_client_event_queues_for_testing()
        clear_supported_auth_backends_cache()
        flush_per_request_caches()
        translation.activate(settings.LANGUAGE_CODE)
        assert settings.LOCAL_UPLOADS_DIR is not None
        if os.path.exists(settings.LOCAL_UPLOADS_DIR):
            shutil.rmtree(settings.LOCAL_UPLOADS_DIR)
        if hasattr(self, 'mock_ldap') and hasattr(self, 'mock_initialize'):
            if self.mock_ldap is not None:
                self.mock_ldap.reset()
            self.mock_initialize.stop()

    def get_user_from_email(self, email: str, realm: Realm) -> UserProfile:
        return get_user(email, realm)

    @override
    def run(self, result: Optional[TestResult] = None) -> Optional[TestResult]:
        if not settings.BAN_CONSOLE_OUTPUT and self.expected_console_output is None:
            return super().run(result)
        extra_output_finder = ExtraConsoleOutputFinder()
        with tee_stderr_and_find_extra_console_output(extra_output_finder), tee_stdout_and_find_extra_console_output(extra_output_finder):
            test_result = super().run(result)
        if extra_output_finder.full_extra_output and (test_result is None or test_result.wasSuccessful()):
            extra_output = extra_output_finder.full_extra_output.decode(errors='replace')
            if self.expected_console_output is not None:
                self.assertEqual(extra_output, self.expected_console_output)
                return test_result
            exception_message = f'\n---- UNEXPECTED CONSOLE OUTPUT DETECTED ----\n\nTo ensure that we never miss important error output/warnings,\nwe require test-backend to have clean console output.\n\nThis message usually is triggered by forgotten debugging print()\nstatements or new logging statements.  For the latter, you can\nuse `with self.assertLogs()` to capture and verify the log output;\nuse `git grep assertLogs` to see dozens of correct examples.\n\nYou should be able to quickly reproduce this failure with:\n\n./tools/test-backend --ban-console-output {self.id()}\n\nOutput:\n{extra_output}\n--------------------------------------------\n'
            raise ExtraConsoleOutputInTestError(exception_message)
        return test_result

    @override
    def assertEqual(self, first: Any, second: Any, msg: str = '') -> None:
        if isinstance(first, str) and isinstance(second, str):
            if first != second:
                raise AssertionError('Actual and expected outputs do not match; showing diff.\n' + diff_strings(first, second) + str(msg))
        else:
            super().assertEqual(first, second, msg)

    def set_http_headers(self, extra: Dict[str, Any], skip_user_agent: bool = False) -> None:
        if 'subdomain' in extra:
            assert isinstance(extra['subdomain'], str)
            extra['HTTP_HOST'] = Realm.host_for_subdomain(extra['subdomain'])
            del extra['subdomain']
        elif 'HTTP_HOST' not in extra:
            extra['HTTP_HOST'] = Realm.host_for_subdomain(self.DEFAULT_SUBDOMAIN)
        if 'HTTP_AUTHORIZATION' in extra:
            default_user_agent = 'ZulipMobile/26.22.145 (iOS 10.3.1)'
        else:
            default_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36'
        if skip_user_agent:
            assert 'HTTP_USER_AGENT' not in extra
        elif 'HTTP_USER_AGENT' not in extra:
            extra['HTTP_USER_AGENT'] = default_user_agent

    @instrument_url
    def client_patch(self, url: str, info: Dict[str, Any] = {}, skip_user_agent: bool = False, follow: bool = False, secure: bool = False, intentionally_undocumented: bool = False, headers: Optional[Dict[str, str]] = None, **extra: Any) -> HttpResponse:
        encoded = urlencode(info)
        extra['content_type'] = 'application/x-www-form-urlencoded'
        django_client = self.client
        self.set_http_headers(extra, skip_user_agent)
        return django_client.patch(url, encoded, follow=follow, secure=secure, headers=headers, intentionally_undocumented=intentionally_undocumented, **extra)

    @instrument_url
    def client_patch_multipart(self, url: str, info: Dict[str, Any] = {}, skip_user_agent: bool = False, follow: bool = False, secure: bool = False, headers: Optional[Dict[str, str]] = None, intentionally_undocumented: bool = False, **extra: Any) -> HttpResponse:
        encoded = encode_multipart(BOUNDARY, dict(info))
        django_client = self.client
        self.set_http_headers(extra, skip_user_agent)
        return django_client.patch(url, encoded, content_type=MULTIPART_CONTENT, follow=follow, secure=secure, headers=headers, intentionally_undocumented=intentionally_undocumented, **extra)

    def json_patch(self, url: str, payload: Dict[str, Any] = {}, skip_user_agent: bool = False, follow: bool = False, secure: bool = False, **extra: Any) -> HttpResponse:
        data = orjson.dumps(payload)
        django_client = self.client
        self.set_http_headers(extra, skip_user_agent)
        return django_client.patch(url, data=data, content_type='application/json', follow=follow, secure=secure, headers=None, **extra)

    @instrument_url
    def client_put(self, url: str, info: Dict[str, Any] = {}, skip_user_agent: bool = False, follow: bool = False, secure: bool = False, headers: Optional[Dict[str, str]] = None, **extra: Any) -> HttpResponse:
        encoded = urlencode(info)
        extra['content_type'] = 'application/x-www-form-urlencoded'
        django_client = self.client
        self.set_http_headers(extra, skip_user_agent)
        return django_client.put(url, encoded, follow=follow, secure=secure, headers=headers, **extra)

    def json_put(self, url: str, payload: Dict[str, Any] = {}, skip_user_agent: bool = False, follow: bool = False, secure: bool = False, headers: Optional[Dict[str, str]] = None, **extra: Any) -> HttpResponse:
        data = orjson.dumps(payload)
        django_client = self.client
        self.set_http_headers(extra, skip_user_agent)
        return django_client.put(url, data=data, content_type='application/json', follow=follow, secure=secure, headers=headers, **extra)

    @instrument_url
    def client_delete(self, url: str, info: Dict[str, Any] = {}, skip_user_agent: bool = False, follow: bool = False, secure: bool = False, headers: Optional[Dict[str, str]] = None, intentionally_undocumented: bool = False, **extra: Any) -> HttpResponse:
        encoded = urlencode(info)
        extra['content_type'] = 'application/x-www-form-urlencoded'
        django_client = self.client
        self.set_http_headers(extra, skip_user_agent)
        return django_client.delete(url, encoded, follow=follow, secure=secure, headers={'Content-Type': 'application/x-www-form-urlencoded', **(headers or {})}, intentionally_undocumented=intentionally_undocumented, **extra)

    @instrument_url
    def client_options(self, url: str, info: Dict[str, Any] = {}, skip_user_agent: bool = False, follow: bool = False, secure: bool = False, headers: Optional[Dict[str, str]] = None, **extra: Any) -> HttpResponse:
        django_client = self.client
        self.set_http_headers(extra, skip_user_agent)
        return django_client.options(url, dict(info), follow=follow, secure=secure, headers=headers, **extra)

    @instrument_url
    def client_head(self, url: str, info: Dict[str, Any] = {}, skip_user_agent: bool = False, follow: bool = False, secure: bool = False, headers: Optional[Dict[str, str]] = None, **extra: Any) -> HttpResponse:
        django_client = self.client
        self.set_http_headers(extra, skip_user_agent)
        return django_client.head(url, info, follow=follow, secure=secure, headers=headers, **extra)

    @instrument_url
    def client_post(self, url: str, info: Dict[str, Any] = {}, skip_user_agent: bool = False, follow: bool = False, secure: bool