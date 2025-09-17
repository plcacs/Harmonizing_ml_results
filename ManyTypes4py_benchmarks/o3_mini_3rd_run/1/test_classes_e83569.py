#!/usr/bin/env python3
import base64
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, Iterator as TypingIterator, List, Mapping as TypingMapping, Optional, Union, cast, Type
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


class EmptyResponseError(Exception):
    pass


class UploadSerializeMixin(SerializeMixin):
    """
    We cannot use override_settings to change upload directory because
    because settings.LOCAL_UPLOADS_DIR is used in URL pattern and URLs
    are compiled only once. Otherwise using a different upload directory
    for conflicting test cases would have provided better performance
    while providing the required isolation.
    """
    lockfile: str = 'var/upload_lock'

    @classmethod
    @override
    def setUpClass(cls: Type["UploadSerializeMixin"]) -> None:
        if not os.path.exists(cls.lockfile):
            with open(cls.lockfile, 'w'):
                pass
        super().setUpClass()


class ZulipClientHandler(ClientHandler):

    @override
    def get_response(self, request: HttpRequest) -> HttpResponseBase:
        got_exception: bool = False

        def on_exception(**kwargs: Any) -> None:
            nonlocal got_exception
            if kwargs['request'] is request:
                got_exception = True

        request.body  # type: ignore
        got_request_exception.connect(on_exception)
        try:
            response: HttpResponseBase = super().get_response(request)
        finally:
            got_request_exception.disconnect(on_exception)
        if (not got_exception and request.method != 'OPTIONS' and
                isinstance(response, HttpResponse) and
                not (response.status_code == 302 and response.headers['Location'].startswith('/login/'))):
            openapi_request: DjangoOpenAPIRequest = DjangoOpenAPIRequest(request)
            openapi_response: DjangoOpenAPIResponse = DjangoOpenAPIResponse(response)
            response_validated: bool = validate_test_response(openapi_request, openapi_response)
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
    client_class: type = ZulipTestClient
    DEFAULT_SUBDOMAIN: str = 'zulip'
    TOKENIZED_NOREPLY_REGEX: str = settings.TOKENIZED_NOREPLY_EMAIL_ADDRESS.format(token='[a-z0-9_]{24}')

    @override
    def setUp(self) -> None:
        super().setUp()
        self.API_KEYS: Dict[str, str] = {}
        test_name: str = self.id()
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
        extra_output_finder: ExtraConsoleOutputFinder = ExtraConsoleOutputFinder()
        with tee_stderr_and_find_extra_console_output(extra_output_finder), tee_stdout_and_find_extra_console_output(extra_output_finder):
            test_result: Optional[TestResult] = super().run(result)
        if extra_output_finder.full_extra_output and (test_result is None or test_result.wasSuccessful()):
            extra_output: str = extra_output_finder.full_extra_output.decode(errors='replace')
            if self.expected_console_output is not None:
                self.assertEqual(extra_output, self.expected_console_output)
                return test_result
            exception_message: str = (f'\n---- UNEXPECTED CONSOLE OUTPUT DETECTED ----\n\n'
                                      f'To ensure that we never miss important error output/warnings,\n'
                                      f'we require test-backend to have clean console output.\n\n'
                                      f'This message usually is triggered by forgotten debugging print()\n'
                                      f'statements or new logging statements.  For the latter, you can\n'
                                      f'use `with self.assertLogs()` to capture and verify the log output;\n'
                                      f'use `git grep assertLogs` to see dozens of correct examples.\n\n'
                                      f'You should be able to quickly reproduce this failure with:\n\n'
                                      f'./tools/test-backend --ban-console-output {self.id()}\n\n'
                                      f'Output:\n{extra_output}\n--------------------------------------------\n')
            raise ExtraConsoleOutputInTestError(exception_message)
        return test_result

    def set_http_headers(self, extra: Dict[str, Any], skip_user_agent: bool = False) -> None:
        if 'subdomain' in extra:
            assert isinstance(extra['subdomain'], str)
            extra['HTTP_HOST'] = Realm.host_for_subdomain(extra['subdomain'])
            del extra['subdomain']
        elif 'HTTP_HOST' not in extra:
            extra['HTTP_HOST'] = Realm.host_for_subdomain(self.DEFAULT_SUBDOMAIN)
        if 'HTTP_AUTHORIZATION' in extra:
            default_user_agent: str = 'ZulipMobile/26.22.145 (iOS 10.3.1)'
        else:
            default_user_agent = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                                  'Chrome/79.0.3945.130 Safari/537.36')
        if skip_user_agent:
            assert 'HTTP_USER_AGENT' not in extra
        elif 'HTTP_USER_AGENT' not in extra:
            extra['HTTP_USER_AGENT'] = default_user_agent

    @instrument_url
    def client_patch(self, url: str, info: Mapping[str, Any] = {}, skip_user_agent: bool = False,
                     follow: bool = False, secure: bool = False, intentionally_undocumented: bool = False,
                     headers: Optional[Dict[str, Any]] = None, **extra: Any) -> HttpResponseBase:
        encoded: str = urlencode(info)
        extra['content_type'] = 'application/x-www-form-urlencoded'
        django_client: TestClient = self.client
        self.set_http_headers(extra, skip_user_agent)
        return django_client.patch(url, encoded, follow=follow, secure=secure,
                                     headers=headers, intentionally_undocumented=intentionally_undocumented, **extra)

    @instrument_url
    def client_patch_multipart(self, url: str, info: Mapping[str, Any] = {}, skip_user_agent: bool = False,
                               follow: bool = False, secure: bool = False, headers: Optional[Dict[str, Any]] = None,
                               intentionally_undocumented: bool = False, **extra: Any) -> HttpResponseBase:
        encoded: bytes = encode_multipart(BOUNDARY, dict(info))
        django_client: TestClient = self.client
        self.set_http_headers(extra, skip_user_agent)
        return django_client.patch(url, encoded, content_type=MULTIPART_CONTENT, follow=follow, secure=secure,
                                     headers=headers, intentionally_undocumented=intentionally_undocumented, **extra)

    def json_patch(self, url: str, payload: Mapping[str, Any] = {}, skip_user_agent: bool = False,
                   follow: bool = False, secure: bool = False, **extra: Any) -> HttpResponseBase:
        data: bytes = orjson.dumps(payload)
        django_client: TestClient = self.client
        self.set_http_headers(extra, skip_user_agent)
        return django_client.patch(url, data=data, content_type='application/json',
                                     follow=follow, secure=secure, headers=None, **extra)

    @instrument_url
    def client_put(self, url: str, info: Mapping[str, Any] = {}, skip_user_agent: bool = False,
                   follow: bool = False, secure: bool = False, headers: Optional[Dict[str, Any]] = None,
                   **extra: Any) -> HttpResponseBase:
        encoded: str = urlencode(info)
        extra['content_type'] = 'application/x-www-form-urlencoded'
        django_client: TestClient = self.client
        self.set_http_headers(extra, skip_user_agent)
        return django_client.put(url, encoded, follow=follow, secure=secure, headers=headers, **extra)

    def json_put(self, url: str, payload: Mapping[str, Any] = {}, skip_user_agent: bool = False,
                 follow: bool = False, secure: bool = False, headers: Optional[Dict[str, Any]] = None,
                 **extra: Any) -> HttpResponseBase:
        data: bytes = orjson.dumps(payload)
        django_client: TestClient = self.client
        self.set_http_headers(extra, skip_user_agent)
        return django_client.put(url, data=data, content_type='application/json', follow=follow, secure=secure,
                                   headers=headers, **extra)

    @instrument_url
    def client_delete(self, url: str, info: Mapping[str, Any] = {}, skip_user_agent: bool = False,
                      follow: bool = False, secure: bool = False, headers: Optional[Dict[str, Any]] = None,
                      intentionally_undocumented: bool = False, **extra: Any) -> HttpResponseBase:
        encoded: str = urlencode(info)
        extra['content_type'] = 'application/x-www-form-urlencoded'
        django_client: TestClient = self.client
        self.set_http_headers(extra, skip_user_agent)
        hdrs: Dict[str, Any] = {'Content-Type': 'application/x-www-form-urlencoded', **(headers or {})}
        return django_client.delete(url, encoded, follow=follow, secure=secure,
                                      headers=hdrs, intentionally_undocumented=intentionally_undocumented, **extra)

    @instrument_url
    def client_options(self, url: str, info: Mapping[str, Any] = {}, skip_user_agent: bool = False,
                       follow: bool = False, secure: bool = False, headers: Optional[Dict[str, Any]] = None,
                       **extra: Any) -> HttpResponseBase:
        django_client: TestClient = self.client
        self.set_http_headers(extra, skip_user_agent)
        return django_client.options(url, dict(info), follow=follow, secure=secure, headers=headers, **extra)

    @instrument_url
    def client_head(self, url: str, info: Mapping[str, Any] = {}, skip_user_agent: bool = False,
                    follow: bool = False, secure: bool = False, headers: Optional[Dict[str, Any]] = None,
                    **extra: Any) -> HttpResponseBase:
        django_client: TestClient = self.client
        self.set_http_headers(extra, skip_user_agent)
        return django_client.head(url, info, follow=follow, secure=secure, headers=headers, **extra)

    @instrument_url
    def client_post(self, url: str, info: Any = {}, skip_user_agent: bool = False,
                    follow: bool = False, secure: bool = False, headers: Optional[Dict[str, Any]] = None,
                    intentionally_undocumented: bool = False, content_type: Optional[str] = None,
                    **extra: Any) -> HttpResponseBase:
        django_client: TestClient = self.client
        self.set_http_headers(extra, skip_user_agent)
        encoded: Any = info
        if content_type is None:
            if isinstance(info, dict) and (not any((hasattr(value, 'read') and callable(value.read) for value in info.values()))):
                content_type = 'application/x-www-form-urlencoded'
                encoded = urlencode(info, doseq=True)
            else:
                content_type = MULTIPART_CONTENT
        elif content_type.startswith('multipart/form-data'):
            content_type = MULTIPART_CONTENT
        hdrs: Dict[str, Any] = {'Content-Type': content_type, **(headers or {})}
        return django_client.post(url, encoded, follow=follow, secure=secure,
                                    headers=hdrs, content_type=content_type, intentionally_undocumented=intentionally_undocumented, **extra)

    @instrument_url
    def client_post_request(self, url: str, req: HttpRequest) -> Any:
        match = resolve(url)
        return match.func(req)

    @instrument_url
    def client_get(self, url: str, info: Mapping[str, Any] = {}, skip_user_agent: bool = False,
                   follow: bool = False, secure: bool = False, headers: Optional[Dict[str, Any]] = None,
                   intentionally_undocumented: bool = False, **extra: Any) -> HttpResponseBase:
        django_client: TestClient = self.client
        self.set_http_headers(extra, skip_user_agent)
        return django_client.get(url, info, follow=follow, secure=secure,
                                   headers=headers, intentionally_undocumented=intentionally_undocumented, **extra)

    example_user_map: Dict[str, str] = dict(hamlet='hamlet@zulip.com', cordelia='cordelia@zulip.com',
                                            iago='iago@zulip.com', prospero='prospero@zulip.com',
                                            othello='othello@zulip.com', AARON='AARON@zulip.com',
                                            aaron='aaron@zulip.com', ZOE='ZOE@zulip.com',
                                            polonius='polonius@zulip.com', desdemona='desdemona@zulip.com',
                                            shiva='shiva@zulip.com', webhook_bot='webhook-bot@zulip.com',
                                            outgoing_webhook_bot='outgoing-webhook@zulip.com',
                                            default_bot='default-bot@zulip.com')
    mit_user_map: Dict[str, str] = dict(sipbtest='sipbtest@mit.edu', starnine='starnine@mit.edu', espuser='espuser@mit.edu')
    lear_user_map: Dict[str, str] = dict(cordelia='cordelia@zulip.com', king='king@lear.org')
    nonreg_user_map: Dict[str, str] = dict(test='test@zulip.com', test1='test1@zulip.com',
                                           alice='alice@zulip.com', newuser='newuser@zulip.com',
                                           bob='bob@zulip.com', cordelia='cordelia@zulip.com',
                                           newguy='newguy@zulip.com', me='me@zulip.com')
    example_user_ldap_username_map: Dict[str, str] = dict(hamlet='hamlet', cordelia='cordelia', aaron='letham')

    def nonreg_user(self, name: str) -> UserProfile:
        email: str = self.nonreg_user_map[name]
        return get_user_by_delivery_email(email, get_realm('zulip'))

    def example_user(self, name: str) -> UserProfile:
        email: str = self.example_user_map[name]
        return get_user_by_delivery_email(email, get_realm('zulip'))

    def mit_user(self, name: str) -> UserProfile:
        email: str = self.mit_user_map[name]
        return self.get_user_from_email(email, get_realm('zephyr'))

    def lear_user(self, name: str) -> UserProfile:
        email: str = self.lear_user_map[name]
        return self.get_user_from_email(email, get_realm('lear'))

    def nonreg_email(self, name: str) -> str:
        return self.nonreg_user_map[name]

    def example_email(self, name: str) -> str:
        return self.example_user_map[name]

    def mit_email(self, name: str) -> str:
        return self.mit_user_map[name]

    def notification_bot(self, realm: Realm) -> UserProfile:
        return get_system_bot(settings.NOTIFICATION_BOT, realm.id)

    def create_test_bot(self, short_name: str, user_profile: UserProfile, full_name: str = 'Foo Bot', **extras: Any) -> UserProfile:
        self.login_user(user_profile)
        bot_info: Dict[str, Any] = {'short_name': short_name, 'full_name': full_name}
        bot_info.update(extras)
        result = self.client_post('/json/bots', bot_info)
        self.assert_json_success(result)
        bot_email: str = f'{short_name}-bot@zulip.testserver'
        bot_profile: UserProfile = self.get_user_from_email(bot_email, user_profile.realm)
        return bot_profile

    def fail_to_create_test_bot(self, short_name: str, user_profile: UserProfile, full_name: str = 'Foo Bot', *,
                                 assert_json_error_msg: str, **extras: Any) -> None:
        self.login_user(user_profile)
        bot_info: Dict[str, Any] = {'short_name': short_name, 'full_name': full_name}
        bot_info.update(extras)
        result = self.client_post('/json/bots', bot_info)
        self.assert_json_error(result, assert_json_error_msg)

    def _get_page_params(self, result: HttpResponse) -> Any:
        """Helper for parsing page_params after fetching the web app's home view."""
        doc: lxml.html.HtmlElement = lxml.html.document_fromstring(result.content)
        div: lxml.html.HtmlElement = cast(lxml.html.HtmlElement, doc.get_element_by_id('page-params'))
        assert div is not None
        page_params_json: Optional[str] = div.get('data-params')
        assert page_params_json is not None
        page_params: Any = orjson.loads(page_params_json)
        return page_params

    def _get_sentry_params(self, response: HttpResponse) -> Optional[Any]:
        doc: lxml.html.HtmlElement = lxml.html.document_fromstring(response.content)
        try:
            script: lxml.html.HtmlElement = cast(lxml.html.HtmlElement, doc.get_element_by_id('sentry-params'))
        except KeyError:
            return None
        assert script is not None and script.text is not None
        return orjson.loads(script.text)

    def check_rendered_logged_in_app(self, result: HttpResponse) -> None:
        self.assertEqual(result.status_code, 200)
        page_params: Any = self._get_page_params(result)
        self.assertEqual(page_params['is_spectator'], False)

    def login_with_return(self, email: str, password: Optional[str] = None, **extra: Any) -> HttpResponseBase:
        if password is None:
            password = initial_password(email)
        result: HttpResponseBase = self.client_post('/accounts/login/', {'username': email, 'password': password},
                                                     skip_user_agent=False, follow=False, secure=False,
                                                     headers=None, intentionally_undocumented=False, **extra)
        self.assertNotEqual(result.status_code, 500)
        return result

    def login(self, name: str) -> None:
        assert '@' not in name, 'use login_by_email for email logins'
        user: UserProfile = self.example_user(name)
        self.login_user(user)

    def login_by_email(self, email: str, password: str) -> None:
        realm: Realm = get_realm('zulip')
        request: HttpRequest = HttpRequest()
        request.session = self.client.session
        self.assertTrue(self.client.login(request=request, username=email, password=password, realm=realm))

    def assert_login_failure(self, email: str, password: str) -> None:
        realm: Realm = get_realm('zulip')
        request: HttpRequest = HttpRequest()
        request.session = self.client.session
        self.assertFalse(self.client.login(request=request, username=email, password=password, realm=realm))

    def login_user(self, user_profile: UserProfile) -> None:
        email: str = user_profile.delivery_email
        realm: Realm = user_profile.realm
        password: str = initial_password(email)
        request: HttpRequest = HttpRequest()
        request.session = self.client.session
        self.assertTrue(self.client.login(request=request, username=email, password=password, realm=realm))

    def login_2fa(self, user_profile: UserProfile) -> None:
        request: HttpRequest = HttpRequest()
        request.session = self.client.session
        request.user = user_profile
        do_two_factor_login(request, user_profile)
        request.session.save()

    def logout(self) -> None:
        self.client.logout()

    def register(self, email: str, password: str, subdomain: str = DEFAULT_SUBDOMAIN) -> None:
        response: HttpResponseBase = self.client_post('/accounts/home/', {'email': email}, subdomain=subdomain)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response['Location'], f'/accounts/send_confirm/?email={quote(email)}')
        response = self.submit_reg_form_for_user(email, password, subdomain=subdomain)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response['Location'], f'http://{Realm.host_for_subdomain(subdomain)}/')

    def submit_reg_form_for_user(self, email: str, password: Optional[str], realm_name: str = 'Zulip Test',
                                 realm_subdomain: str = 'zuliptest', from_confirmation: str = '',
                                 full_name: Optional[str] = None, timezone: str = '',
                                 realm_in_root_domain: Optional[bool] = None,
                                 default_stream_groups: List[Any] = [], source_realm_id: str = '',
                                 key: Optional[str] = None, realm_type: int = Realm.ORG_TYPES['business']['id'],
                                 realm_default_language: str = 'en', enable_marketing_emails: Optional[bool] = None,
                                 email_address_visibility: Optional[int] = None, is_demo_organization: bool = False,
                                 **extra: Any) -> HttpResponseBase:
        if full_name is None:
            full_name = email.replace('@', '_')
        payload: Dict[str, Any] = {'full_name': full_name, 'realm_name': realm_name, 'realm_subdomain': realm_subdomain,
                                   'realm_type': realm_type, 'realm_default_language': realm_default_language,
                                   'key': key if key is not None else find_key_by_email(email),
                                   'timezone': timezone, 'terms': True, 'from_confirmation': from_confirmation,
                                   'default_stream_group': default_stream_groups,
                                   'source_realm_id': source_realm_id, 'is_demo_organization': is_demo_organization,
                                   'how_realm_creator_found_zulip': 'other',
                                   'how_realm_creator_found_zulip_extra_context': 'I found it on the internet.'}
        if enable_marketing_emails is not None:
            payload['enable_marketing_emails'] = enable_marketing_emails
        if email_address_visibility is not None:
            payload['email_address_visibility'] = email_address_visibility
        if password is not None:
            payload['password'] = password
        if realm_in_root_domain is not None:
            payload['realm_in_root_domain'] = realm_in_root_domain
        return self.client_post('/accounts/register/', payload,
                                skip_user_agent=False, follow=False, secure=False,
                                headers=None, intentionally_undocumented=False, **extra)

    def submit_realm_creation_form(self, email: str, *, realm_subdomain: str, realm_name: str,
                                   realm_type: int = Realm.ORG_TYPES['business']['id'],
                                   realm_default_language: str = 'en',
                                   realm_in_root_domain: Optional[bool] = None) -> HttpResponseBase:
        payload: Dict[str, Any] = {'email': email, 'realm_name': realm_name, 'realm_type': realm_type,
                                   'realm_default_language': realm_default_language, 'realm_subdomain': realm_subdomain}
        if realm_in_root_domain is not None:
            payload['realm_in_root_domain'] = realm_in_root_domain
        return self.client_post('/new/', payload)

    def get_confirmation_url_from_outbox(self, email_address: str, *, url_pattern: Optional[str] = None,
                                         email_subject_contains: Optional[str] = None,
                                         email_body_contains: Optional[str] = None) -> str:
        from django.core.mail import outbox
        if url_pattern is None:
            url_pattern = settings.EXTERNAL_HOST + '(\\S+)>'
        for message in reversed(outbox):
            if any((addr == email_address or addr.endswith(f' <{email_address}>') for addr in message.to)):
                match = re.search(url_pattern, str(message.body))
                assert match is not None
                if email_subject_contains:
                    self.assertIn(email_subject_contains, message.subject)
                if email_body_contains:
                    self.assertIn(email_body_contains, message.body)
                [confirmation_url] = match.groups()
                return confirmation_url
        raise AssertionError("Couldn't find a confirmation email.")

    def encode_uuid(self, uuid: str) -> str:
        if uuid in self.API_KEYS:
            api_key: str = self.API_KEYS[uuid]
        else:
            api_key = get_remote_server_by_uuid(uuid).api_key
            self.API_KEYS[uuid] = api_key
        return self.encode_credentials(uuid, api_key)

    def encode_user(self, user: UserProfile) -> str:
        email: str = user.delivery_email
        api_key: str = user.api_key
        return self.encode_credentials(email, api_key)

    def encode_email(self, email: str, realm: str = 'zulip') -> str:
        assert '@' in email
        user: UserProfile = get_user_by_delivery_email(email, get_realm(realm))
        api_key: str = user.api_key
        return self.encode_credentials(email, api_key)

    def encode_credentials(self, identifier: str, api_key: str) -> str:
        credentials: str = f'{identifier}:{api_key}'
        return 'Basic ' + base64.b64encode(credentials.encode()).decode()

    def uuid_get(self, identifier: str, url: str, info: Mapping[str, Any] = {}, **extra: Any) -> HttpResponseBase:
        extra['HTTP_AUTHORIZATION'] = self.encode_uuid(identifier)
        return self.client_get(url, info, skip_user_agent=False, follow=False, secure=False,
                               headers=None, intentionally_undocumented=False, **extra)

    def uuid_post(self, identifier: str, url: str, info: Mapping[str, Any] = {}, **extra: Any) -> HttpResponseBase:
        extra['HTTP_AUTHORIZATION'] = self.encode_uuid(identifier)
        return self.client_post(url, info, skip_user_agent=False, follow=False, secure=False,
                                headers=None, intentionally_undocumented=False, **extra)

    def api_get(self, user: UserProfile, url: str, info: Mapping[str, Any] = {}, **extra: Any) -> HttpResponseBase:
        extra['HTTP_AUTHORIZATION'] = self.encode_user(user)
        return self.client_get(url, info, skip_user_agent=False, follow=False, secure=False,
                               headers=None, intentionally_undocumented=False, **extra)

    def api_post(self, user: UserProfile, url: str, info: Mapping[str, Any] = {},
                 intentionally_undocumented: bool = False, **extra: Any) -> HttpResponseBase:
        extra['HTTP_AUTHORIZATION'] = self.encode_user(user)
        return self.client_post(url, info, skip_user_agent=False, follow=False, secure=False,
                                headers=None, intentionally_undocumented=intentionally_undocumented, **extra)

    def api_patch(self, user: UserProfile, url: str, info: Mapping[str, Any] = {}, **extra: Any) -> HttpResponseBase:
        extra['HTTP_AUTHORIZATION'] = self.encode_user(user)
        return self.client_patch(url, info, skip_user_agent=False, follow=False, secure=False,
                                 headers=None, intentionally_undocumented=False, **extra)

    def api_delete(self, user: UserProfile, url: str, info: Mapping[str, Any] = {}, **extra: Any) -> HttpResponseBase:
        extra['HTTP_AUTHORIZATION'] = self.encode_user(user)
        return self.client_delete(url, info, skip_user_agent=False, follow=False, secure=False,
                                  headers=None, intentionally_undocumented=False, **extra)

    def get_streams(self, user_profile: UserProfile) -> List[str]:
        return list(Stream.objects.filter(id__in=get_subscribed_stream_ids_for_user(user_profile)).values_list('name', flat=True))

    def send_personal_message(self, from_user: UserProfile, to_user: UserProfile, content: str = 'test content', *,
                              read_by_sender: bool = True) -> int:
        recipient_list: List[int] = [to_user.id]
        sending_client, _ = Client.objects.get_or_create(name='test suite')
        sent_message_result = check_send_message(from_user, sending_client, 'private', recipient_list, None, content,
                                                 read_by_sender=read_by_sender)
        return sent_message_result.message_id

    def send_group_direct_message(self, from_user: UserProfile, to_users: List[UserProfile], content: str = 'test content', *,
                                  read_by_sender: bool = True) -> int:
        to_user_ids: List[int] = [u.id for u in to_users]
        assert len(to_user_ids) >= 2
        sending_client, _ = Client.objects.get_or_create(name='test suite')
        sent_message_result = check_send_message(from_user, sending_client, 'private', to_user_ids, None, content,
                                                 read_by_sender=read_by_sender)
        return sent_message_result.message_id

    def send_stream_message(self, sender: UserProfile, stream_name: str, content: str = 'test content',
                            topic_name: str = 'test', recipient_realm: Optional[Realm] = None, *,
                            allow_unsubscribed_sender: bool = False, read_by_sender: bool = True) -> int:
        sending_client, _ = Client.objects.get_or_create(name='test suite')
        message_id: int = check_send_stream_message(sender=sender, client=sending_client, stream_name=stream_name,
                                                     topic_name=topic_name, body=content, realm=recipient_realm,
                                                     read_by_sender=read_by_sender)
        if (not UserMessage.objects.filter(user_profile=sender, message_id=message_id).exists() and
            (not sender.is_bot) and (not allow_unsubscribed_sender)):
            raise AssertionError(f'\n    It appears that the sender did not get a UserMessage row, which is\n'
                                 f'    almost certainly an artificial symptom that in your test setup you\n'
                                 f'    have decided to send a message to a stream without the sender being\n'
                                 f'    subscribed.\n\n'
                                 f'    Please do self.subscribe(<user for {sender.full_name}>, {stream_name!r}) first.\n\n'
                                 f'    Or choose a stream that the user is already subscribed to:\n\n'
                                 f'{self.subscribed_stream_name_list(sender)}\n        ')
        return message_id

    def get_messages_response(self, anchor: int = 1, num_before: int = 100, num_after: int = 100,
                              use_first_unread_anchor: bool = False, include_anchor: bool = True) -> Any:
        post_params: Dict[str, Any] = {
            'anchor': anchor,
            'num_before': num_before,
            'num_after': num_after,
            'use_first_unread_anchor': orjson.dumps(use_first_unread_anchor).decode(),
            'include_anchor': orjson.dumps(include_anchor).decode()
        }
        result: HttpResponseBase = self.client_get('/json/messages', dict(post_params))
        data: Any = result.json()
        return data

    def get_messages(self, anchor: int = 1, num_before: int = 100, num_after: int = 100,
                     use_first_unread_anchor: bool = False) -> Any:
        data: Any = self.get_messages_response(anchor, num_before, num_after, use_first_unread_anchor)
        return data['messages']

    def users_subscribed_to_stream(self, stream_name: str, realm: Realm) -> List[UserProfile]:
        stream: Stream = Stream.objects.get(name=stream_name, realm=realm)
        recipient: Recipient = Recipient.objects.get(type_id=stream.id, type=Recipient.STREAM)
        subscriptions: QuerySet = Subscription.objects.filter(recipient=recipient, active=True)
        return [subscription.user_profile for subscription in subscriptions]

    def not_long_term_idle_subscriber_ids(self, stream_name: str, realm: Realm) -> Any:
        stream: Stream = Stream.objects.get(name=stream_name, realm=realm)
        recipient: Recipient = Recipient.objects.get(type_id=stream.id, type=Recipient.STREAM)
        subscriptions: QuerySet = Subscription.objects.filter(recipient=recipient, active=True, is_user_active=True).exclude(user_profile__long_term_idle=True)
        user_profile_ids = set(subscriptions.values_list('user_profile_id', flat=True))
        return user_profile_ids

    def assert_json_success(self, result: HttpResponseBase, *, ignored_parameters: Optional[List[Any]] = None) -> Any:
        try:
            json: Any = orjson.loads(result.content)
        except orjson.JSONDecodeError:
            json = {'msg': 'Error parsing JSON in response!'}
        try:
            self.assertEqual(result.status_code, 200, json['msg'])
            self.assertEqual(json.get('result'), 'success')
            self.assertIn('msg', json)
            self.assertNotEqual(json['msg'], 'Error parsing JSON in response!')
            if ignored_parameters is None:
                self.assertNotIn('ignored_parameters_unsupported', json)
            else:
                self.assertIn('ignored_parameters_unsupported', json)
                self.assert_length(json['ignored_parameters_unsupported'], len(ignored_parameters))
                for param in ignored_parameters:
                    self.assertTrue(param in json['ignored_parameters_unsupported'])
        except AssertionError as e:
            if isinstance(result, MutableJsonResponse):
                raise e from result.exception
            raise
        return json

    def get_json_error(self, result: HttpResponseBase, status_code: int = 400) -> str:
        try:
            json: Any = orjson.loads(result.content)
        except orjson.JSONDecodeError:
            json = {'msg': 'Error parsing JSON in response!'}
        self.assertEqual(result.status_code, status_code, msg=json.get('msg'))
        self.assertEqual(json.get('result'), 'error')
        return json['msg']

    def assert_json_error(self, result: HttpResponseBase, msg: str, status_code: int = 400) -> None:
        try:
            self.assertEqual(self.get_json_error(result, status_code=status_code), msg)
        except AssertionError as e:
            if isinstance(result, MutableJsonResponse):
                raise e from result.exception
            raise

    def assert_length(self, items: Sequence[Any], count: int) -> None:
        actual_count: int = len(items)
        if actual_count != count:
            print('\nITEMS:\n')
            for item in items:
                print(item)
            print(f'\nexpected length: {count}\nactual length: {actual_count}')
            raise AssertionError(f'{type(items)} is of unexpected size!')

    @contextmanager
    def assert_memcached_count(self, count: int) -> TypingIterator[Any]:
        with cache_tries_captured() as cache_tries:
            yield
        self.assert_length(cache_tries, count)

    @contextmanager
    def assert_database_query_count(self, count: int, include_savepoints: bool = False,
                                    keep_cache_warm: bool = False) -> TypingIterator[Any]:
        with queries_captured(include_savepoints=include_savepoints, keep_cache_warm=keep_cache_warm) as queries:
            yield
        actual_count: int = len(queries)
        if actual_count != count:
            print('\nITEMS:\n')
            for index, query in enumerate(queries):
                print(f'#{index + 1}\nsql: {query.sql}\ntime: {query.time}\n')
            print(f'expected count: {count}\nactual count: {actual_count}')
            raise AssertionError(f"\n    {count} queries expected, got {actual_count}.\n    This is a performance-critical code path, where we check\n"
                                 f"    the number of database queries used in order to avoid accidental regressions.\n"
                                 f"    If an unnecessary query was removed or the new query is necessary, you should\n"
                                 f"    update this test, and explain what queries we added/removed in the pull request\n"
                                 f"    and why any new queries can't be avoided.")

    def assert_json_error_contains(self, result: HttpResponseBase, msg_substring: str, status_code: int = 400) -> None:
        try:
            self.assertIn(msg_substring, self.get_json_error(result, status_code=status_code))
        except AssertionError as e:
            if isinstance(result, MutableJsonResponse):
                raise e from result.exception
            raise

    def assert_in_response(self, substring: str, response: HttpResponse) -> None:
        self.assertIn(substring, response.content.decode())

    def assert_in_success_response(self, substrings: List[str], response: HttpResponse) -> None:
        self.assertEqual(response.status_code, 200)
        decoded: str = response.content.decode()
        for substring in substrings:
            self.assertIn(substring, decoded)

    def assert_not_in_success_response(self, substrings: List[str], response: HttpResponse) -> None:
        self.assertEqual(response.status_code, 200)
        decoded: str = response.content.decode()
        for substring in substrings:
            self.assertNotIn(substring, decoded)

    def assert_logged_in_user_id(self, user_id: Optional[int]) -> None:
        self.assertEqual(get_session_dict_user(self.client.session), user_id)

    def assert_message_stream_name(self, message: Message, stream_name: str) -> None:
        self.assertEqual(message.recipient.type, Recipient.STREAM)
        stream_id: int = message.recipient.type_id
        stream: Stream = Stream.objects.get(id=stream_id)
        self.assertEqual(stream.recipient_id, message.recipient_id)
        self.assertEqual(stream.name, stream_name)

    def webhook_fixture_data(self, type: str, action: str, file_type: str = 'json') -> str:
        fn: str = os.path.join(os.path.dirname(__file__), f'../webhooks/{type}/fixtures/{action}.{file_type}')
        with open(fn) as f:
            return f.read()

    def fixture_file_name(self, file_name: str, type: str = '') -> str:
        return os.path.join(os.path.dirname(__file__), f'../tests/fixtures/{type}/{file_name}')

    def fixture_data(self, file_name: str, type: str = '') -> str:
        fn: str = self.fixture_file_name(file_name, type)
        with open(fn) as f:
            return f.read()

    def make_stream(self, stream_name: str, realm: Optional[Realm] = None, invite_only: bool = False,
                    is_web_public: bool = False, history_public_to_subscribers: Optional[bool] = None) -> Stream:
        if realm is None:
            realm = get_realm('zulip')
        history_public_to_subscribers = get_default_value_for_history_public_to_subscribers(realm, invite_only, history_public_to_subscribers)
        try:
            stream: Stream = Stream.objects.create(realm=realm, name=stream_name, invite_only=invite_only,
                                                   is_web_public=is_web_public, history_public_to_subscribers=history_public_to_subscribers,
                                                   **get_default_values_for_stream_permission_group_settings(realm))
        except IntegrityError:
            raise Exception(f'\n                {stream_name} already exists\n\n                Please call make_stream with a stream name\n                that is not already in use.')
        recipient: Recipient = Recipient.objects.create(type_id=stream.id, type=Recipient.STREAM)
        stream.recipient = recipient
        stream.save(update_fields=['recipient'])
        return stream

    INVALID_STREAM_ID: int = 999999

    def get_stream_id(self, name: str, realm: Optional[Realm] = None) -> int:
        if not realm:
            realm = get_realm('zulip')
        try:
            stream: Stream = get_realm_stream(name, realm.id)
        except Stream.DoesNotExist:
            return self.INVALID_STREAM_ID
        return stream.id

    def subscribe(self, user_profile: UserProfile, stream_name: str, invite_only: bool = False,
                  is_web_public: bool = False) -> Stream:
        realm: Realm = user_profile.realm
        try:
            stream: Stream = get_stream(stream_name, user_profile.realm)
        except Stream.DoesNotExist:
            stream, _ = create_stream_if_needed(realm, stream_name, invite_only=invite_only,
                                                is_web_public=is_web_public, acting_user=user_profile)
        bulk_add_subscriptions(realm, [stream], [user_profile], acting_user=None)
        return stream

    def unsubscribe(self, user_profile: UserProfile, stream_name: str) -> None:
        realm: Realm = user_profile.realm
        stream: Stream = get_stream(stream_name, user_profile.realm)
        bulk_remove_subscriptions(realm, [user_profile], [stream], acting_user=None)

    def subscribe_via_post(self, user: UserProfile, subscriptions_raw: List[Union[str, Dict[str, Any]]],
                           extra_post_data: Dict[str, Any] = {}, invite_only: bool = False,
                           is_web_public: bool = False, allow_fail: bool = False, **extra: Any) -> HttpResponseBase:
        subscriptions: List[Dict[str, Any]] = []
        for entry in subscriptions_raw:
            if isinstance(entry, str):
                subscriptions.append({'name': entry})
            else:
                subscriptions.append(entry)
        post_data: Dict[str, Any] = {'subscriptions': orjson.dumps(subscriptions).decode(),
                                     'is_web_public': orjson.dumps(is_web_public).decode(),
                                     'invite_only': orjson.dumps(invite_only).decode()}
        post_data.update(extra_post_data)
        with transaction.atomic(savepoint=True):
            result: HttpResponseBase = self.api_post(user, '/api/v1/users/me/subscriptions', post_data,
                                                     intentionally_undocumented=False, **extra)
        if not allow_fail:
            self.assert_json_success(result)
        return result

    def subscribed_stream_name_list(self, user: UserProfile) -> str:
        subscribed_streams: List[Dict[str, Any]] = gather_subscriptions(user)[0]
        return ''.join(sorted((f'        * {stream["name"]}\n' for stream in subscribed_streams)))

    def check_user_subscribed_only_to_streams(self, user_name: str, streams: List[Stream]) -> None:
        stream_names = {stream.name for stream in streams}
        subscribed_streams: List[Dict[str, Any]] = gather_subscriptions(self.nonreg_user(user_name))[0]
        self.assertEqual(stream_names, {stream['name'] for stream in subscribed_streams})

    def resolve_topic_containing_message(self, acting_user: UserProfile, target_message_id: int, **extra: Any) -> HttpResponseBase:
        message: Message = access_message(acting_user, target_message_id)
        return self.api_patch(acting_user, f'/api/v1/messages/{target_message_id}',
                              {'topic': RESOLVED_TOPIC_PREFIX + message.topic_name(), 'propagate_mode': 'change_all'}, **extra)

    def send_webhook_payload(self, user_profile: UserProfile, url: str, payload: Union[str, Dict[str, Any]], **extra: Any) -> Message:
        prior_msg: Message = self.get_last_message()
        result: HttpResponseBase = self.client_post(url, payload, skip_user_agent=False, follow=False,
                                                    secure=False, headers=None, intentionally_undocumented=False, **extra)
        self.assert_json_success(result)
        msg: Message = self.get_last_message()
        if msg.id == prior_msg.id:
            raise EmptyResponseError('\n                Your test code called an endpoint that did\n                not write any new messages.  It is probably\n                broken (but still returns 200 due to exception\n                handling).\n\n                One possible gotcha is that you forgot to\n                subscribe the test user to the stream that\n                the webhook sends to.\n                ')
        self.assertEqual(msg.sender.email, user_profile.email)
        return msg

    def get_last_message(self) -> Message:
        return Message.objects.latest('id')

    def get_second_to_last_message(self) -> Message:
        return Message.objects.all().order_by('-id')[1]

    @contextmanager
    def simulated_markdown_failure(self) -> TypingIterator[None]:
        with mock.patch('zerver.lib.markdown.unsafe_timeout', side_effect=subprocess.CalledProcessError(1, [])), self.assertLogs(level='ERROR'):
            yield

    def create_default_device(self, user_profile: UserProfile, number: str = '+12125550100') -> None:
        phone_device: PhoneDevice = PhoneDevice(user=user_profile, name='default', confirmed=True, number=number, key='abcd', method='sms')
        phone_device.save()

    def rm_tree(self, path: str) -> None:
        if os.path.exists(path):
            shutil.rmtree(path)

    def make_import_output_dir(self, exported_from: str) -> str:
        output_dir: str = tempfile.mkdtemp(dir=settings.TEST_WORKER_DIR, prefix='test-' + exported_from + '-import-')
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def get_set(self, data: List[Dict[str, Any]], field: str) -> set:
        values = {r[field] for r in data}
        return values

    def find_by_id(self, data: List[Dict[str, Any]], db_id: int) -> Dict[str, Any]:
        [r] = (r for r in data if r['id'] == db_id)
        return r

    def init_default_ldap_database(self) -> None:
        directory: Dict[str, Any] = orjson.loads(self.fixture_data('directory.json', type='ldap'))
        for attrs in directory.values():
            if 'uid' in attrs:
                attrs['userPassword'] = [self.ldap_password(attrs['uid'][0])]
            for attr, value in attrs.items():
                if isinstance(value, str) and value.startswith('file:'):
                    with open(value.removeprefix('file:'), 'rb') as f:
                        attrs[attr] = [f.read()]
        ldap_patcher = mock.patch('django_auth_ldap.config.ldap.initialize')
        self.mock_initialize = ldap_patcher.start()
        self.mock_ldap = MockLDAP(directory)
        self.mock_initialize.return_value = self.mock_ldap

    def change_ldap_user_attr(self, username: str, attr_name: str, attr_value: str, binary: bool = False) -> None:
        dn: str = f'uid={username},ou=users,dc=zulip,dc=com'
        if binary:
            with open(attr_value, 'rb') as f:
                data = f.read()
        else:
            data = attr_value
        self.mock_ldap.directory[dn][attr_name] = [data]

    def remove_ldap_user_attr(self, username: str, attr_name: str) -> None:
        dn: str = f'uid={username},ou=users,dc=zulip,dc=com'
        self.mock_ldap.directory[dn].pop(attr_name, None)

    def ldap_username(self, username: str) -> str:
        return self.example_user_ldap_username_map[username]

    def ldap_password(self, uid: str) -> str:
        return f'{uid}_ldap_password'

    def email_display_from(self, email_message: EmailMessage) -> str:
        return email_message.extra_headers.get('From', email_message.from_email)

    def email_envelope_from(self, email_message: EmailMessage) -> str:
        return email_message.from_email

    def subscribe_realm_to_manual_license_management_plan(self, realm: Realm, licenses: int, licenses_at_next_renewal: int, billing_schedule: str) -> Any:
        customer, _ = Customer.objects.get_or_create(realm=realm)
        plan: CustomerPlan = CustomerPlan.objects.create(customer=customer, automanage_licenses=False,
                                                         billing_cycle_anchor=timezone_now(), billing_schedule=billing_schedule,
                                                         tier=CustomerPlan.TIER_CLOUD_STANDARD)
        ledger: LicenseLedger = LicenseLedger.objects.create(plan=plan, is_renewal=True, event_time=timezone_now(),
                                                               licenses=licenses, licenses_at_next_renewal=licenses_at_next_renewal)
        realm.plan_type = Realm.PLAN_TYPE_STANDARD
        realm.save(update_fields=['plan_type'])
        return (plan, ledger)

    def subscribe_realm_to_monthly_plan_on_manual_license_management(self, realm: Realm, licenses: int, licenses_at_next_renewal: int) -> Any:
        return self.subscribe_realm_to_manual_license_management_plan(realm, licenses, licenses_at_next_renewal, CustomerPlan.BILLING_SCHEDULE_MONTHLY)

    def create_user_notifications_data_object(self, *, user_id: int, **kwargs: Any) -> UserMessageNotificationsData:
        return UserMessageNotificationsData(user_id=user_id,
                                              online_push_enabled=kwargs.get('online_push_enabled', False),
                                              dm_email_notify=kwargs.get('dm_email_notify', False),
                                              dm_push_notify=kwargs.get('dm_push_notify', False),
                                              mention_email_notify=kwargs.get('mention_email_notify', False),
                                              mention_push_notify=kwargs.get('mention_push_notify', False),
                                              topic_wildcard_mention_email_notify=kwargs.get('topic_wildcard_mention_email_notify', False),
                                              topic_wildcard_mention_push_notify=kwargs.get('topic_wildcard_mention_push_notify', False),
                                              stream_wildcard_mention_email_notify=kwargs.get('stream_wildcard_mention_email_notify', False),
                                              stream_wildcard_mention_push_notify=kwargs.get('stream_wildcard_mention_push_notify', False),
                                              stream_email_notify=kwargs.get('stream_email_notify', False),
                                              stream_push_notify=kwargs.get('stream_push_notify', False),
                                              followed_topic_email_notify=kwargs.get('followed_topic_email_notify', False),
                                              followed_topic_push_notify=kwargs.get('followed_topic_push_notify', False),
                                              topic_wildcard_mention_in_followed_topic_email_notify=kwargs.get('topic_wildcard_mention_in_followed_topic_email_notify', False),
                                              topic_wildcard_mention_in_followed_topic_push_notify=kwargs.get('topic_wildcard_mention_in_followed_topic_push_notify', False),
                                              stream_wildcard_mention_in_followed_topic_email_notify=kwargs.get('stream_wildcard_mention_in_followed_topic_email_notify', False),
                                              stream_wildcard_mention_in_followed_topic_push_notify=kwargs.get('stream_wildcard_mention_in_followed_topic_push_notify', False),
                                              sender_is_muted=kwargs.get('sender_is_muted', False),
                                              disable_external_notifications=kwargs.get('disable_external_notifications', False))

    def get_maybe_enqueue_notifications_parameters(self, *, message_id: int, user_id: int, acting_user_id: int, **kwargs: Any) -> Dict[str, Any]:
        user_notifications_data: UserMessageNotificationsData = self.create_user_notifications_data_object(user_id=user_id, **kwargs)
        return dict(user_notifications_data=user_notifications_data,
                    message_id=message_id,
                    acting_user_id=acting_user_id,
                    mentioned_user_group_id=kwargs.get('mentioned_user_group_id'),
                    idle=kwargs.get('idle', True),
                    already_notified=kwargs.get('already_notified', {'email_notified': False, 'push_notified': False}))

    def verify_emoji_code_foreign_keys(self) -> None:
        dct: Dict[int, RealmEmoji] = {}
        for realm_emoji in RealmEmoji.objects.all():
            dct[realm_emoji.id] = realm_emoji
        if not dct:
            raise AssertionError('test needs RealmEmoji rows')
        count: int = 0
        for reaction in Reaction.objects.filter(reaction_type=Reaction.REALM_EMOJI):
            realm_emoji_id: int = int(reaction.emoji_code)
            assert realm_emoji_id in dct
            self.assertEqual(dct[realm_emoji_id].name, reaction.emoji_name)
            self.assertEqual(dct[realm_emoji_id].realm_id, reaction.user_profile.realm_id)
            count += 1
        for user_status in UserStatus.objects.filter(reaction_type=UserStatus.REALM_EMOJI):
            realm_emoji_id = int(user_status.emoji_code)
            assert realm_emoji_id in dct
            self.assertEqual(dct[realm_emoji_id].name, user_status.emoji_name)
            self.assertEqual(dct[realm_emoji_id].realm_id, user_status.user_profile.realm_id)
            count += 1
        if count == 0:
            raise AssertionError('test is meaningless without any pertinent rows')

    def check_user_added_in_system_group(self, user: UserProfile) -> None:
        user_group: UserGroup = get_system_user_group_for_user(user)
        self.assertTrue(UserGroupMembership.objects.filter(user_profile=user, user_group=user_group).exists())

    def _assert_long_term_idle(self, user: UserProfile) -> None:
        if not user.long_term_idle:
            raise AssertionError('\n                We expect you to explicitly call self.soft_deactivate_user\n                if your user is not already soft-deactivated.\n            ')

    def expect_soft_reactivation(self, user: UserProfile, action: Callable[[], Any]) -> None:
        self._assert_long_term_idle(user)
        action()
        user.refresh_from_db()
        self.assertEqual(user.long_term_idle, False)

    def expect_to_stay_long_term_idle(self, user: UserProfile, action: Callable[[], Any]) -> None:
        self._assert_long_term_idle(user)
        action()
        user.refresh_from_db()
        self.assertEqual(user.long_term_idle, True)

    def soft_deactivate_user(self, user: UserProfile) -> None:
        do_soft_deactivate_users([user])
        assert user.long_term_idle

    def set_up_db_for_testing_user_access(self) -> None:
        polonius: UserProfile = self.example_user('polonius')
        hamlet: UserProfile = self.example_user('hamlet')
        othello: UserProfile = self.example_user('othello')
        iago: UserProfile = self.example_user('iago')
        prospero: UserProfile = self.example_user('prospero')
        aaron: UserProfile = self.example_user('aaron')
        zoe: UserProfile = self.example_user('ZOE')
        shiva: UserProfile = self.example_user('shiva')
        realm: Realm = get_realm('zulip')
        self.unsubscribe(polonius, 'Verona')
        self.make_stream('test_stream1')
        self.make_stream('test_stream2', invite_only=True)
        self.subscribe(othello, 'test_stream1')
        self.send_stream_message(othello, 'test_stream1', content='test message', topic_name='test')
        self.unsubscribe(othello, 'test_stream1')
        self.subscribe(polonius, 'test_stream1')
        self.subscribe(polonius, 'test_stream2')
        self.subscribe(hamlet, 'test_stream1')
        self.subscribe(iago, 'test_stream2')
        self.send_personal_message(polonius, prospero)
        self.send_personal_message(shiva, polonius)
        self.send_group_direct_message(aaron, [polonius, zoe])
        members_group: NamedUserGroup = NamedUserGroup.objects.get(name='role:members', realm=realm)
        do_change_realm_permission_group_setting(realm, 'can_access_all_users_group', members_group, acting_user=None)

    def create_or_update_anonymous_group_for_setting(self, direct_members: List[UserProfile], direct_subgroups: List[UserGroup],
                                                     existing_setting_group: Optional[UserGroup] = None) -> UserGroup:
        realm: Realm = get_realm('zulip')
        if existing_setting_group is not None:
            existing_setting_group.direct_members.set(direct_members)
            existing_setting_group.direct_subgroups.set(direct_subgroups)
            return existing_setting_group
        user_group: UserGroup = UserGroup.objects.create(realm=realm)
        user_group.direct_members.set(direct_members)
        user_group.direct_subgroups.set(direct_subgroups)
        return user_group

    @contextmanager
    def thumbnail_formats(self, *thumbnail_formats: ThumbnailFormat) -> TypingIterator[None]:
        with mock.patch('zerver.lib.thumbnail.THUMBNAIL_OUTPUT_FORMATS', thumbnail_formats), \
             mock.patch('zerver.views.upload.THUMBNAIL_OUTPUT_FORMATS', thumbnail_formats):
            yield

    def create_attachment_helper(self, user: UserProfile) -> str:
        with tempfile.NamedTemporaryFile() as attach_file:
            attach_file.write(b'Hello, World!')
            attach_file.flush()
            with open(attach_file.name, 'rb') as fp:
                file_path: str = upload_message_attachment_from_request(UploadedFile(fp), user)[0]
                return file_path


class ZulipTestCase(ZulipTestCaseMixin, TestCase):

    @contextmanager
    def capture_send_event_calls(self, expected_num_events: int) -> TypingIterator[List[Any]]:
        lst: List[Any] = []
        with mock.patch('zerver.tornado.event_queue.process_notification', lst.append), self.captureOnCommitCallbacks(execute=True):
            yield lst
        self.assert_length(lst, expected_num_events)

    @override
    def send_personal_message(self, from_user: UserProfile, to_user: UserProfile, content: str = 'test content', *,
                              read_by_sender: bool = True, skip_capture_on_commit_callbacks: bool = False) -> int:
        if skip_capture_on_commit_callbacks:
            message_id: int = super().send_personal_message(from_user, to_user, content, read_by_sender=read_by_sender)
        else:
            with self.captureOnCommitCallbacks(execute=True):
                message_id = super().send_personal_message(from_user, to_user, content, read_by_sender=read_by_sender)
        return message_id

    @override
    def send_group_direct_message(self, from_user: UserProfile, to_users: List[UserProfile], content: str = 'test content', *,
                                  read_by_sender: bool = True, skip_capture_on_commit_callbacks: bool = False) -> int:
        if skip_capture_on_commit_callbacks:
            message_id: int = super().send_group_direct_message(from_user, to_users, content, read_by_sender=read_by_sender)
        else:
            with self.captureOnCommitCallbacks(execute=True):
                message_id = super().send_group_direct_message(from_user, to_users, content, read_by_sender=read_by_sender)
        return message_id

    @override
    def send_stream_message(self, sender: UserProfile, stream_name: str, content: str = 'test content',
                            topic_name: str = 'test', recipient_realm: Optional[Realm] = None, *,
                            allow_unsubscribed_sender: bool = False, read_by_sender: bool = True,
                            skip_capture_on_commit_callbacks: bool = False) -> int:
        if skip_capture_on_commit_callbacks:
            message_id: int = super().send_stream_message(sender, stream_name, content, topic_name, recipient_realm,
                                                          allow_unsubscribed_sender=allow_unsubscribed_sender, read_by_sender=read_by_sender)
        else:
            with self.captureOnCommitCallbacks(execute=True):
                message_id = super().send_stream_message(sender, stream_name, content, topic_name, recipient_realm,
                                                          allow_unsubscribed_sender=allow_unsubscribed_sender, read_by_sender=read_by_sender)
        return message_id

    def upload_image(self, image_name: str) -> str:
        with get_test_image_file(image_name) as image_file:
            response: Any = self.assert_json_success(self.client_post('/json/user_uploads', {'file': image_file}))
            import re
            return re.sub('/user_uploads/', '', response['url'])

    def upload_and_thumbnail_image(self, image_name: str) -> str:
        with self.captureOnCommitCallbacks(execute=True):
            return self.upload_image(image_name)

    def handle_missedmessage_emails(self, user_profile_id: int, message_ids: List[int]) -> None:
        with self.captureOnCommitCallbacks(execute=True):
            handle_missedmessage_emails(user_profile_id, message_ids)


def get_row_ids_in_all_tables() -> TypingIterator[tuple[str, set]]:
    all_models = apps.get_models(include_auto_created=True)
    ignored_tables = {'django_session'}
    for model in all_models:
        table_name: str = model._meta.db_table
        if table_name in ignored_tables:
            continue
        ids = model._default_manager.all().values_list('id', flat=True)
        yield (table_name, set(ids))


class ZulipTransactionTestCase(ZulipTestCaseMixin, TransactionTestCase):
    """
    The default Django TestCase wraps each test in a transaction. This
    is invaluable for being able to rollback the transaction and thus
    efficiently do many tests containing database changes, but it
    prevents testing certain transaction-related races and locking
    bugs.

    This test class is intended to be used (sparingly!) for tests that
    need to verify transaction related behavior, like locking with
    select_for_update or transaction.atomic(durable=True).

    Unlike ZulipTestCase, ZulipTransactionTestCase has the following traits:
    1. Does not offer isolation between tests by wrapping them inside an atomic transaction.
    2. Changes are committed to the current worker's test database, so side effects carry on.

    All ZulipTransactionTestCase tests must be carefully written to
    avoid side effects on the database; while Django runs
    TransactionTestCase after all normal TestCase tests on a given
    test worker to avoid pollution, they can break other
    ZulipTransactionTestCase tests if they leak state.
    """

    @override
    def setUp(self) -> None:
        super().setUp()
        self.models_ids_set: Dict[str, set] = dict(get_row_ids_in_all_tables())

    @override
    def tearDown(self) -> None:
        super().tearDown()
        for table_name, ids in get_row_ids_in_all_tables():
            self.assertSetEqual(self.models_ids_set[table_name], ids, f'{table_name} got a different set of ids after this test')

    def _fixture_teardown(self) -> None:
        pass


class WebhookTestCase(ZulipTestCase):
    CHANNEL_NAME: Optional[str] = None
    TEST_USER_EMAIL: str = 'webhook-bot@zulip.com'
    WEBHOOK_DIR_NAME: Optional[str] = None
    VIEW_FUNCTION_NAME: Optional[str] = None

    @property
    def test_user(self) -> UserProfile:
        return self.get_user_from_email(self.TEST_USER_EMAIL, get_realm('zulip'))

    @override
    def setUp(self) -> None:
        super().setUp()
        self.url: str = self.build_webhook_url()
        if self.WEBHOOK_DIR_NAME is not None:
            if self.VIEW_FUNCTION_NAME is None:
                function = import_string(f'zerver.webhooks.{self.WEBHOOK_DIR_NAME}.view.api_{self.WEBHOOK_DIR_NAME}_webhook')
            else:
                function = import_string(f'zerver.webhooks.{self.WEBHOOK_DIR_NAME}.view.{self.VIEW_FUNCTION_NAME}')
            all_event_types = None
            if hasattr(function, '_all_event_types'):
                all_event_types = function._all_event_types
            if all_event_types is None:
                return

            def side_effect(*args: Any, **kwargs: Any) -> None:
                complete_event_type = kwargs.get('complete_event_type') if len(args) < 5 else args[4]
                if complete_event_type is not None and all_event_types is not None and (complete_event_type not in all_event_types):
                    raise Exception(f'\nError: This test triggered a message using the event "{complete_event_type}", which was not properly\nregistered via the @webhook_view(..., event_types=[...]). These registrations are important for Zulip\nself-documenting the supported event types for this integration.\n\nYou can fix this by adding "{complete_event_type}" to ALL_EVENT_TYPES for this webhook.\n'.strip())
                check_send_webhook_message(*args, **kwargs)
            self.patch = mock.patch(f'zerver.webhooks.{self.WEBHOOK_DIR_NAME}.view.check_send_webhook_message', side_effect=side_effect)
            self.patch.start()
            self.addCleanup(self.patch.stop)

    def api_channel_message(self, user: UserProfile, fixture_name: str, expected_topic: Optional[str] = None,
                            expected_message: Optional[str] = None, content_type: str = 'application/json',
                            expect_noop: bool = False, **extra: Any) -> None:
        extra['HTTP_AUTHORIZATION'] = self.encode_user(user)
        self.check_webhook(fixture_name, expected_topic, expected_message, content_type, expect_noop, **extra)

    def check_webhook(self, fixture_name: str, expected_topic_name: Optional[str] = None,
                      expected_message: Optional[str] = None, content_type: str = 'application/json',
                      expect_noop: bool = False, **extra: Any) -> None:
        assert self.CHANNEL_NAME is not None
        self.subscribe(self.test_user, self.CHANNEL_NAME)
        payload: Union[str, Dict[str, Any]] = self.get_payload(fixture_name)
        if content_type is not None:
            extra['content_type'] = content_type
        if self.WEBHOOK_DIR_NAME is not None:
            headers: Dict[str, Any] = get_fixture_http_headers(self.WEBHOOK_DIR_NAME, fixture_name)
            headers = standardize_headers(headers)
            extra.update(headers)
        try:
            msg: Message = self.send_webhook_payload(self.test_user, self.url, payload, **extra)
        except EmptyResponseError:
            if expect_noop:
                return
            else:
                raise AssertionError('No message was sent. Pass expect_noop=True if this is intentional.')
        if expect_noop:
            raise Exception('\nWhile no message is expected given expect_noop=True,\nyour test code triggered an endpoint that did write\none or more new messages.\n'.strip())
        assert expected_message is not None and expected_topic_name is not None
        self.assert_channel_message(message=msg, channel_name=self.CHANNEL_NAME, topic_name=expected_topic_name, content=expected_message)

    def assert_channel_message(self, message: Message, channel_name: str, topic_name: str, content: str) -> None:
        self.assert_message_stream_name(message, channel_name)
        self.assertEqual(message.topic_name(), topic_name)
        self.assertEqual(message.content, content)

    def send_and_test_private_message(self, fixture_name: str, expected_message: str, content_type: str = 'application/json',
                                      *, sender: Optional[UserProfile] = None, **extra: Any) -> Message:
        payload: Union[str, Dict[str, Any]] = self.get_payload(fixture_name)
        extra['content_type'] = content_type
        if self.WEBHOOK_DIR_NAME is not None:
            headers: Dict[str, Any] = get_fixture_http_headers(self.WEBHOOK_DIR_NAME, fixture_name)
            headers = standardize_headers(headers)
            extra.update(headers)
        if sender is None:
            sender = self.test_user
        msg: Message = self.send_webhook_payload(sender, self.url, payload, **extra)
        self.assertEqual(msg.content, expected_message)
        return msg

    def build_webhook_url(self, *args: Any, **kwargs: Any) -> str:
        url: str = self.URL_TEMPLATE  # type: ignore
        if url.find('api_key') >= 0:
            api_key: str = self.test_user.api_key
            url = self.URL_TEMPLATE.format(api_key=api_key, stream=self.CHANNEL_NAME)  # type: ignore
        else:
            url = self.URL_TEMPLATE.format(stream=self.CHANNEL_NAME)  # type: ignore
        has_arguments: bool = bool(kwargs or args)
        if has_arguments and url.find('?') == -1:
            url = f'{url}?'
        else:
            url = f'{url}&'
        for key, value in kwargs.items():
            url = f'{url}{key}={value}&'
        for arg in args:
            url = f'{url}{arg}&'
        return url[:-1] if has_arguments else url

    def get_payload(self, fixture_name: str) -> Union[str, Dict[str, Any]]:
        return self.get_body(fixture_name)

    def get_body(self, fixture_name: str) -> str:
        assert self.WEBHOOK_DIR_NAME is not None
        body: str = self.webhook_fixture_data(self.WEBHOOK_DIR_NAME, fixture_name)
        orjson.loads(body)
        return body


class MigrationsTestCase(ZulipTransactionTestCase):
    @property
    def app(self) -> str:
        app_config = apps.get_containing_app_config(type(self).__module__)
        assert app_config is not None
        return app_config.name

    migrate_from: Optional[str] = None
    migrate_to: Optional[str] = None

    @override
    def setUp(self) -> None:
        super().setUp()
        assert self.migrate_from and self.migrate_to, f"TestCase '{type(self).__name__}' must define migrate_from and migrate_to properties"
        migrate_from = [(self.app, self.migrate_from)]
        migrate_to = [(self.app, self.migrate_to)]
        executor = MigrationExecutor(connection)
        old_apps: StateApps = executor.loader.project_state(migrate_from).apps
        executor.migrate(migrate_from)
        self.setUpBeforeMigration(old_apps)
        executor = MigrationExecutor(connection)
        executor.loader.build_graph()
        executor.migrate(migrate_to)
        self.apps = executor.loader.project_state(migrate_to).apps

    def setUpBeforeMigration(self, apps: StateApps) -> None:
        pass


def get_topic_messages(user_profile: UserProfile, stream: Stream, topic_name: str) -> List[Message]:
    query: QuerySet = UserMessage.objects.filter(user_profile=user_profile, message__recipient=stream.recipient).order_by('id')
    return [um.message for um in filter_by_topic_name_via_message(query, topic_name)]


@skipUnless(settings.ZILENCER_ENABLED, 'requires zilencer')
class BouncerTestCase(ZulipTestCase):

    @override
    def setUp(self) -> None:
        self.server_uuid: str = '6cde5f7a-1f7e-4978-9716-49f69ebfc9fe'
        self.server = RemoteZulipServer.objects.all().latest('id')
        self.server.uuid = self.server_uuid
        self.server.hostname = 'demo.example.com'
        self.server.save()
        super().setUp()

    @override
    def tearDown(self) -> None:
        RemoteZulipServer.objects.filter(uuid=self.server_uuid).delete()
        super().tearDown()

    def request_callback(self, request: PreparedRequest) -> tuple[int, ResponseHeaders, bytes]:
        kwargs: Dict[str, Any] = {}
        if isinstance(request.body, bytes):
            data: Any = orjson.loads(request.body)
            kwargs = dict(content_type='application/json')
        else:
            assert isinstance(request.body, (str, type(None)))
            params = parse_qs(request.body) if request.body is not None else {}
            data = {k: v[0] for k, v in params.items()}
        assert request.url is not None
        assert settings.ZULIP_SERVICES_URL is not None
        local_url: str = request.url.replace(settings.ZULIP_SERVICES_URL, '')
        if request.method == 'POST':
            result: HttpResponseBase = self.uuid_post(self.server_uuid, local_url, data, subdomain='', **kwargs)
        elif request.method == 'GET':
            result = self.uuid_get(self.server_uuid, local_url, data, subdomain='', **kwargs)
        else:
            raise Exception("Unsupported method")
        return (result.status_code, result.headers, result.content)

    def add_mock_response(self) -> None:
        assert settings.ZULIP_SERVICES_URL is not None
        COMPILED_URL = re.compile(settings.ZULIP_SERVICES_URL + '.*')
        responses.add_callback(responses.POST, COMPILED_URL, callback=self.request_callback)
        responses.add_callback(responses.GET, COMPILED_URL, callback=self.request_callback)

    def get_generic_payload(self, method: str = 'register') -> Dict[str, Any]:
        user_id: int = 10
        token: str = '111222'
        token_kind: str = PushDeviceToken.FCM
        return {'user_id': user_id, 'token': token, 'token_kind': token_kind}