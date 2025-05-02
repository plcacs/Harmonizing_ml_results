import base64
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Union, cast
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
    lockfile = 'var/upload_lock'

    @classmethod
    @override
    def setUpClass(cls):
        if not os.path.exists(cls.lockfile):
            with open(cls.lockfile, 'w'):
                pass
        super().setUpClass()

class ZulipClientHandler(ClientHandler):

    @override
    def get_response(self, request):
        got_exception = False

        def on_exception(**kwargs):
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

    def __init__(self):
        super().__init__()
        self.handler = ZulipClientHandler(enforce_csrf_checks=False)

class ZulipTestCaseMixin(SimpleTestCase):
    maxDiff = None
    expected_console_output = None
    client_class = ZulipTestClient

    @override
    def setUp(self):
        super().setUp()
        self.API_KEYS = {}
        test_name = self.id()
        bounce_key_prefix_for_testing(test_name)
        bounce_redis_key_prefix_for_testing(test_name)

    @override
    def tearDown(self):
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

    def get_user_from_email(self, email, realm):
        return get_user(email, realm)

    @override
    def run(self, result=None):
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
    "\n    WRAPPER_COMMENT:\n\n    We wrap calls to self.client.{patch,put,get,post,delete} for various\n    reasons.  Some of this has to do with fixing encodings before calling\n    into the Django code.  Some of this has to do with providing a future\n    path for instrumentation.  Some of it's just consistency.\n\n    The linter will prevent direct calls to self.client.foo, so the wrapper\n    functions have to fake out the linter by using a local variable called\n    django_client to fool the regex.\n    "
    DEFAULT_SUBDOMAIN = 'zulip'
    TOKENIZED_NOREPLY_REGEX = settings.TOKENIZED_NOREPLY_EMAIL_ADDRESS.format(token='[a-z0-9_]{24}')

    @override
    def assertEqual(self, first, second, msg=''):
        if isinstance(first, str) and isinstance(second, str):
            if first != second:
                raise AssertionError('Actual and expected outputs do not match; showing diff.\n' + diff_strings(first, second) + str(msg))
        else:
            super().assertEqual(first, second, msg)

    def set_http_headers(self, extra, skip_user_agent=False):
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
    def client_patch(self, url, info={}, skip_user_agent=False, follow=False, secure=False, intentionally_undocumented=False, headers=None, **extra):
        """
        We need to urlencode, since Django's function won't do it for us.
        """
        encoded = urlencode(info)
        extra['content_type'] = 'application/x-www-form-urlencoded'
        django_client = self.client
        self.set_http_headers(extra, skip_user_agent)
        return django_client.patch(url, encoded, follow=follow, secure=secure, headers=headers, intentionally_undocumented=intentionally_undocumented, **extra)

    @instrument_url
    def client_patch_multipart(self, url, info={}, skip_user_agent=False, follow=False, secure=False, headers=None, intentionally_undocumented=False, **extra):
        """
        Use this for patch requests that have file uploads or
        that need some sort of multi-part content.  In the future
        Django's test client may become a bit more flexible,
        so we can hopefully eliminate this.  (When you post
        with the Django test client, it deals with MULTIPART_CONTENT
        automatically, but not patch.)
        """
        encoded = encode_multipart(BOUNDARY, dict(info))
        django_client = self.client
        self.set_http_headers(extra, skip_user_agent)
        return django_client.patch(url, encoded, content_type=MULTIPART_CONTENT, follow=follow, secure=secure, headers=headers, intentionally_undocumented=intentionally_undocumented, **extra)

    def json_patch(self, url, payload={}, skip_user_agent=False, follow=False, secure=False, **extra):
        data = orjson.dumps(payload)
        django_client = self.client
        self.set_http_headers(extra, skip_user_agent)
        return django_client.patch(url, data=data, content_type='application/json', follow=follow, secure=secure, headers=None, **extra)

    @instrument_url
    def client_put(self, url, info={}, skip_user_agent=False, follow=False, secure=False, headers=None, **extra):
        encoded = urlencode(info)
        extra['content_type'] = 'application/x-www-form-urlencoded'
        django_client = self.client
        self.set_http_headers(extra, skip_user_agent)
        return django_client.put(url, encoded, follow=follow, secure=secure, headers=headers, **extra)

    def json_put(self, url, payload={}, skip_user_agent=False, follow=False, secure=False, headers=None, **extra):
        data = orjson.dumps(payload)
        django_client = self.client
        self.set_http_headers(extra, skip_user_agent)
        return django_client.put(url, data=data, content_type='application/json', follow=follow, secure=secure, headers=headers, **extra)

    @instrument_url
    def client_delete(self, url, info={}, skip_user_agent=False, follow=False, secure=False, headers=None, intentionally_undocumented=False, **extra):
        encoded = urlencode(info)
        extra['content_type'] = 'application/x-www-form-urlencoded'
        django_client = self.client
        self.set_http_headers(extra, skip_user_agent)
        return django_client.delete(url, encoded, follow=follow, secure=secure, headers={'Content-Type': 'application/x-www-form-urlencoded', **(headers or {})}, intentionally_undocumented=intentionally_undocumented, **extra)

    @instrument_url
    def client_options(self, url, info={}, skip_user_agent=False, follow=False, secure=False, headers=None, **extra):
        django_client = self.client
        self.set_http_headers(extra, skip_user_agent)
        return django_client.options(url, dict(info), follow=follow, secure=secure, headers=headers, **extra)

    @instrument_url
    def client_head(self, url, info={}, skip_user_agent=False, follow=False, secure=False, headers=None, **extra):
        django_client = self.client
        self.set_http_headers(extra, skip_user_agent)
        return django_client.head(url, info, follow=follow, secure=secure, headers=headers, **extra)

    @instrument_url
    def client_post(self, url, info={}, skip_user_agent=False, follow=False, secure=False, headers=None, intentionally_undocumented=False, content_type=None, **extra):
        django_client = self.client
        self.set_http_headers(extra, skip_user_agent)
        encoded = info
        if content_type is None:
            if isinstance(info, dict) and (not any((hasattr(value, 'read') and callable(value.read) for value in info.values()))):
                content_type = 'application/x-www-form-urlencoded'
                encoded = urlencode(info, doseq=True)
            else:
                content_type = MULTIPART_CONTENT
        elif content_type.startswith('multipart/form-data'):
            content_type = MULTIPART_CONTENT
        return django_client.post(url, encoded, follow=follow, secure=secure, headers={'Content-Type': content_type, **(headers or {})}, content_type=content_type, intentionally_undocumented=intentionally_undocumented, **extra)

    @instrument_url
    def client_post_request(self, url, req):
        """
        We simulate hitting an endpoint here, although we
        actually resolve the URL manually and hit the view
        directly.  We have this helper method to allow our
        instrumentation to work for /notify_tornado and
        future similar methods that require doing funny
        things to a request object.
        """
        match = resolve(url)
        return match.func(req)

    @instrument_url
    def client_get(self, url, info={}, skip_user_agent=False, follow=False, secure=False, headers=None, intentionally_undocumented=False, **extra):
        django_client = self.client
        self.set_http_headers(extra, skip_user_agent)
        return django_client.get(url, info, follow=follow, secure=secure, headers=headers, intentionally_undocumented=intentionally_undocumented, **extra)
    example_user_map = dict(hamlet='hamlet@zulip.com', cordelia='cordelia@zulip.com', iago='iago@zulip.com', prospero='prospero@zulip.com', othello='othello@zulip.com', AARON='AARON@zulip.com', aaron='aaron@zulip.com', ZOE='ZOE@zulip.com', polonius='polonius@zulip.com', desdemona='desdemona@zulip.com', shiva='shiva@zulip.com', webhook_bot='webhook-bot@zulip.com', outgoing_webhook_bot='outgoing-webhook@zulip.com', default_bot='default-bot@zulip.com')
    mit_user_map = dict(sipbtest='sipbtest@mit.edu', starnine='starnine@mit.edu', espuser='espuser@mit.edu')
    lear_user_map = dict(cordelia='cordelia@zulip.com', king='king@lear.org')
    nonreg_user_map = dict(test='test@zulip.com', test1='test1@zulip.com', alice='alice@zulip.com', newuser='newuser@zulip.com', bob='bob@zulip.com', cordelia='cordelia@zulip.com', newguy='newguy@zulip.com', me='me@zulip.com')
    example_user_ldap_username_map = dict(hamlet='hamlet', cordelia='cordelia', aaron='letham')

    def nonreg_user(self, name):
        email = self.nonreg_user_map[name]
        return get_user_by_delivery_email(email, get_realm('zulip'))

    def example_user(self, name):
        email = self.example_user_map[name]
        return get_user_by_delivery_email(email, get_realm('zulip'))

    def mit_user(self, name):
        email = self.mit_user_map[name]
        return self.get_user_from_email(email, get_realm('zephyr'))

    def lear_user(self, name):
        email = self.lear_user_map[name]
        return self.get_user_from_email(email, get_realm('lear'))

    def nonreg_email(self, name):
        return self.nonreg_user_map[name]

    def example_email(self, name):
        return self.example_user_map[name]

    def mit_email(self, name):
        return self.mit_user_map[name]

    def notification_bot(self, realm):
        return get_system_bot(settings.NOTIFICATION_BOT, realm.id)

    def create_test_bot(self, short_name, user_profile, full_name='Foo Bot', **extras):
        self.login_user(user_profile)
        bot_info = {'short_name': short_name, 'full_name': full_name}
        bot_info.update(extras)
        result = self.client_post('/json/bots', bot_info)
        self.assert_json_success(result)
        bot_email = f'{short_name}-bot@zulip.testserver'
        bot_profile = self.get_user_from_email(bot_email, user_profile.realm)
        return bot_profile

    def fail_to_create_test_bot(self, short_name, user_profile, full_name='Foo Bot', *, assert_json_error_msg, **extras):
        self.login_user(user_profile)
        bot_info = {'short_name': short_name, 'full_name': full_name}
        bot_info.update(extras)
        result = self.client_post('/json/bots', bot_info)
        self.assert_json_error(result, assert_json_error_msg)

    def _get_page_params(self, result):
        """Helper for parsing page_params after fetching the web app's home view."""
        doc = lxml.html.document_fromstring(result.content)
        div = cast(lxml.html.HtmlMixin, doc).get_element_by_id('page-params')
        assert div is not None
        page_params_json = div.get('data-params')
        assert page_params_json is not None
        page_params = orjson.loads(page_params_json)
        return page_params

    def _get_sentry_params(self, response):
        doc = lxml.html.document_fromstring(response.content)
        try:
            script = cast(lxml.html.HtmlMixin, doc).get_element_by_id('sentry-params')
        except KeyError:
            return None
        assert script is not None and script.text is not None
        return orjson.loads(script.text)

    def check_rendered_logged_in_app(self, result):
        """Verifies that a visit of / was a 200 that rendered page_params
        and not for a (logged-out) spectator."""
        self.assertEqual(result.status_code, 200)
        page_params = self._get_page_params(result)
        self.assertEqual(page_params['is_spectator'], False)

    def login_with_return(self, email, password=None, **extra):
        if password is None:
            password = initial_password(email)
        result = self.client_post('/accounts/login/', {'username': email, 'password': password}, skip_user_agent=False, follow=False, secure=False, headers=None, intentionally_undocumented=False, **extra)
        self.assertNotEqual(result.status_code, 500)
        return result

    def login(self, name):
        """
        Use this for really simple tests where you just need
        to be logged in as some user, but don't need the actual
        user object for anything else.  Try to use 'hamlet' for
        non-admins and 'iago' for admins:

            self.login('hamlet')

        Try to use 'cordelia' or 'othello' as "other" users.
        """
        assert '@' not in name, 'use login_by_email for email logins'
        user = self.example_user(name)
        self.login_user(user)

    def login_by_email(self, email, password):
        realm = get_realm('zulip')
        request = HttpRequest()
        request.session = self.client.session
        self.assertTrue(self.client.login(request=request, username=email, password=password, realm=realm))

    def assert_login_failure(self, email, password):
        realm = get_realm('zulip')
        request = HttpRequest()
        request.session = self.client.session
        self.assertFalse(self.client.login(request=request, username=email, password=password, realm=realm))

    def login_user(self, user_profile):
        email = user_profile.delivery_email
        realm = user_profile.realm
        password = initial_password(email)
        request = HttpRequest()
        request.session = self.client.session
        self.assertTrue(self.client.login(request=request, username=email, password=password, realm=realm))

    def login_2fa(self, user_profile):
        """
        We need this function to call request.session.save().
        do_two_factor_login doesn't save session; in normal request-response
        cycle this doesn't matter because middleware will save the session
        when it finds it dirty; however,in tests we will have to do that
        explicitly.
        """
        request = HttpRequest()
        request.session = self.client.session
        request.user = user_profile
        do_two_factor_login(request, user_profile)
        request.session.save()

    def logout(self):
        self.client.logout()

    def register(self, email, password, subdomain=DEFAULT_SUBDOMAIN):
        response = self.client_post('/accounts/home/', {'email': email}, subdomain=subdomain)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response['Location'], f'/accounts/send_confirm/?email={quote(email)}')
        response = self.submit_reg_form_for_user(email, password, subdomain=subdomain)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response['Location'], f'http://{Realm.host_for_subdomain(subdomain)}/')

    def submit_reg_form_for_user(self, email, password, realm_name='Zulip Test', realm_subdomain='zuliptest', from_confirmation='', full_name=None, timezone='', realm_in_root_domain=None, default_stream_groups=[], source_realm_id='', key=None, realm_type=Realm.ORG_TYPES['business']['id'], realm_default_language='en', enable_marketing_emails=None, email_address_visibility=None, is_demo_organization=False, **extra):
        """
        Stage two of the two-step registration process.

        If things are working correctly the account should be fully
        registered after this call.

        You can pass the HTTP_HOST variable for subdomains via extra.
        """
        if full_name is None:
            full_name = email.replace('@', '_')
        payload = {'full_name': full_name, 'realm_name': realm_name, 'realm_subdomain': realm_subdomain, 'realm_type': realm_type, 'realm_default_language': realm_default_language, 'key': key if key is not None else find_key_by_email(email), 'timezone': timezone, 'terms': True, 'from_confirmation': from_confirmation, 'default_stream_group': default_stream_groups, 'source_realm_id': source_realm_id, 'is_demo_organization': is_demo_organization, 'how_realm_creator_found_zulip': 'other', 'how_realm_creator_found_zulip_extra_context': 'I found it on the internet.'}
        if enable_marketing_emails is not None:
            payload['enable_marketing_emails'] = enable_marketing_emails
        if email_address_visibility is not None:
            payload['email_address_visibility'] = email_address_visibility
        if password is not None:
            payload['password'] = password
        if realm_in_root_domain is not None:
            payload['realm_in_root_domain'] = realm_in_root_domain
        return self.client_post('/accounts/register/', payload, skip_user_agent=False, follow=False, secure=False, headers=None, intentionally_undocumented=False, **extra)

    def submit_realm_creation_form(self, email, *, realm_subdomain, realm_name, realm_type=Realm.ORG_TYPES['business']['id'], realm_default_language='en', realm_in_root_domain=None):
        payload = {'email': email, 'realm_name': realm_name, 'realm_type': realm_type, 'realm_default_language': realm_default_language, 'realm_subdomain': realm_subdomain}
        if realm_in_root_domain is not None:
            payload['realm_in_root_domain'] = realm_in_root_domain
        return self.client_post('/new/', payload)

    def get_confirmation_url_from_outbox(self, email_address, *, url_pattern=None, email_subject_contains=None, email_body_contains=None):
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

    def encode_uuid(self, uuid):
        """
        identifier: Can be an email or a remote server uuid.
        """
        if uuid in self.API_KEYS:
            api_key = self.API_KEYS[uuid]
        else:
            api_key = get_remote_server_by_uuid(uuid).api_key
            self.API_KEYS[uuid] = api_key
        return self.encode_credentials(uuid, api_key)

    def encode_user(self, user):
        email = user.delivery_email
        api_key = user.api_key
        return self.encode_credentials(email, api_key)

    def encode_email(self, email, realm='zulip'):
        assert '@' in email
        user = get_user_by_delivery_email(email, get_realm(realm))
        api_key = user.api_key
        return self.encode_credentials(email, api_key)

    def encode_credentials(self, identifier, api_key):
        """
        identifier: Can be an email or a remote server uuid.
        """
        credentials = f'{identifier}:{api_key}'
        return 'Basic ' + base64.b64encode(credentials.encode()).decode()

    def uuid_get(self, identifier, url, info={}, **extra):
        extra['HTTP_AUTHORIZATION'] = self.encode_uuid(identifier)
        return self.client_get(url, info, skip_user_agent=False, follow=False, secure=False, headers=None, intentionally_undocumented=False, **extra)

    def uuid_post(self, identifier, url, info={}, **extra):
        extra['HTTP_AUTHORIZATION'] = self.encode_uuid(identifier)
        return self.client_post(url, info, skip_user_agent=False, follow=False, secure=False, headers=None, intentionally_undocumented=False, **extra)

    def api_get(self, user, url, info={}, **extra):
        extra['HTTP_AUTHORIZATION'] = self.encode_user(user)
        return self.client_get(url, info, skip_user_agent=False, follow=False, secure=False, headers=None, intentionally_undocumented=False, **extra)

    def api_post(self, user, url, info={}, intentionally_undocumented=False, **extra):
        extra['HTTP_AUTHORIZATION'] = self.encode_user(user)
        return self.client_post(url, info, skip_user_agent=False, follow=False, secure=False, headers=None, intentionally_undocumented=intentionally_undocumented, **extra)

    def api_patch(self, user, url, info={}, **extra):
        extra['HTTP_AUTHORIZATION'] = self.encode_user(user)
        return self.client_patch(url, info, skip_user_agent=False, follow=False, secure=False, headers=None, intentionally_undocumented=False, **extra)

    def api_delete(self, user, url, info={}, **extra):
        extra['HTTP_AUTHORIZATION'] = self.encode_user(user)
        return self.client_delete(url, info, skip_user_agent=False, follow=False, secure=False, headers=None, intentionally_undocumented=False, **extra)

    def get_streams(self, user_profile):
        """
        Helper function to get the active stream names for a user
        """
        return list(Stream.objects.filter(id__in=get_subscribed_stream_ids_for_user(user_profile)).values_list('name', flat=True))

    def send_personal_message(self, from_user, to_user, content='test content', *, read_by_sender=True):
        recipient_list = [to_user.id]
        sending_client, _ = Client.objects.get_or_create(name='test suite')
        sent_message_result = check_send_message(from_user, sending_client, 'private', recipient_list, None, content, read_by_sender=read_by_sender)
        return sent_message_result.message_id

    def send_group_direct_message(self, from_user, to_users, content='test content', *, read_by_sender=True):
        to_user_ids = [u.id for u in to_users]
        assert len(to_user_ids) >= 2
        sending_client, _ = Client.objects.get_or_create(name='test suite')
        sent_message_result = check_send_message(from_user, sending_client, 'private', to_user_ids, None, content, read_by_sender=read_by_sender)
        return sent_message_result.message_id

    def send_stream_message(self, sender, stream_name, content='test content', topic_name='test', recipient_realm=None, *, allow_unsubscribed_sender=False, read_by_sender=True):
        sending_client, _ = Client.objects.get_or_create(name='test suite')
        message_id = check_send_stream_message(sender=sender, client=sending_client, stream_name=stream_name, topic_name=topic_name, body=content, realm=recipient_realm, read_by_sender=read_by_sender)
        if not UserMessage.objects.filter(user_profile=sender, message_id=message_id).exists() and (not sender.is_bot) and (not allow_unsubscribed_sender):
            raise AssertionError(f'\n    It appears that the sender did not get a UserMessage row, which is\n    almost certainly an artificial symptom that in your test setup you\n    have decided to send a message to a stream without the sender being\n    subscribed.\n\n    Please do self.subscribe(<user for {sender.full_name}>, {stream_name!r}) first.\n\n    Or choose a stream that the user is already subscribed to:\n\n{self.subscribed_stream_name_list(sender)}\n        ')
        return message_id

    def get_messages_response(self, anchor=1, num_before=100, num_after=100, use_first_unread_anchor=False, include_anchor=True):
        post_params = {'anchor': anchor, 'num_before': num_before, 'num_after': num_after, 'use_first_unread_anchor': orjson.dumps(use_first_unread_anchor).decode(), 'include_anchor': orjson.dumps(include_anchor).decode()}
        result = self.client_get('/json/messages', dict(post_params))
        data = result.json()
        return data

    def get_messages(self, anchor=1, num_before=100, num_after=100, use_first_unread_anchor=False):
        data = self.get_messages_response(anchor, num_before, num_after, use_first_unread_anchor)
        return data['messages']

    def users_subscribed_to_stream(self, stream_name, realm):
        stream = Stream.objects.get(name=stream_name, realm=realm)
        recipient = Recipient.objects.get(type_id=stream.id, type=Recipient.STREAM)
        subscriptions = Subscription.objects.filter(recipient=recipient, active=True)
        return [subscription.user_profile for subscription in subscriptions]

    def not_long_term_idle_subscriber_ids(self, stream_name, realm):
        stream = Stream.objects.get(name=stream_name, realm=realm)
        recipient = Recipient.objects.get(type_id=stream.id, type=Recipient.STREAM)
        subscriptions = Subscription.objects.filter(recipient=recipient, active=True, is_user_active=True).exclude(user_profile__long_term_idle=True)
        user_profile_ids = set(subscriptions.values_list('user_profile_id', flat=True))
        return user_profile_ids

    def assert_json_success(self, result, *, ignored_parameters=None):
        """
        Successful POSTs return a 200 and JSON of the form {"result": "success",
        "msg": ""}.
        """
        try:
            json = orjson.loads(result.content)
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

    def get_json_error(self, result, status_code=400):
        try:
            json = orjson.loads(result.content)
        except orjson.JSONDecodeError:
            json = {'msg': 'Error parsing JSON in response!'}
        self.assertEqual(result.status_code, status_code, msg=json.get('msg'))
        self.assertEqual(json.get('result'), 'error')
        return json['msg']

    def assert_json_error(self, result, msg, status_code=400):
        """
        Invalid POSTs return an error status code and JSON of the form
        {"result": "error", "msg": "reason"}.
        """
        try:
            self.assertEqual(self.get_json_error(result, status_code=status_code), msg)
        except AssertionError as e:
            if isinstance(result, MutableJsonResponse):
                raise e from result.exception
            raise

    def assert_length(self, items, count):
        actual_count = len(items)
        if actual_count != count:
            print('\nITEMS:\n')
            for item in items:
                print(item)
            print(f'\nexpected length: {count}\nactual length: {actual_count}')
            raise AssertionError(f'{type(items)} is of unexpected size!')

    @contextmanager
    def assert_memcached_count(self, count):
        with cache_tries_captured() as cache_tries:
            yield
        self.assert_length(cache_tries, count)

    @contextmanager
    def assert_database_query_count(self, count, include_savepoints=False, keep_cache_warm=False):
        """
        This captures the queries executed and check the total number of queries.
        Useful when minimizing unnecessary roundtrips to the database is important.
        """
        with queries_captured(include_savepoints=include_savepoints, keep_cache_warm=keep_cache_warm) as queries:
            yield
        actual_count = len(queries)
        if actual_count != count:
            print('\nITEMS:\n')
            for index, query in enumerate(queries):
                print(f'#{index + 1}\nsql: {query.sql}\ntime: {query.time}\n')
            print(f'expected count: {count}\nactual count: {actual_count}')
            raise AssertionError(f"\n    {count} queries expected, got {actual_count}.\n    This is a performance-critical code path, where we check\n    the number of database queries used in order to avoid accidental regressions.\n    If an unnecessary query was removed or the new query is necessary, you should\n    update this test, and explain what queries we added/removed in the pull request\n    and why any new queries can't be avoided.")

    def assert_json_error_contains(self, result, msg_substring, status_code=400):
        try:
            self.assertIn(msg_substring, self.get_json_error(result, status_code=status_code))
        except AssertionError as e:
            if isinstance(result, MutableJsonResponse):
                raise e from result.exception
            raise

    def assert_in_response(self, substring, response):
        self.assertIn(substring, response.content.decode())

    def assert_in_success_response(self, substrings, response):
        self.assertEqual(response.status_code, 200)
        decoded = response.content.decode()
        for substring in substrings:
            self.assertIn(substring, decoded)

    def assert_not_in_success_response(self, substrings, response):
        self.assertEqual(response.status_code, 200)
        decoded = response.content.decode()
        for substring in substrings:
            self.assertNotIn(substring, decoded)

    def assert_logged_in_user_id(self, user_id):
        """
        Verifies the user currently logged in for the test client has the provided user_id.
        Pass None to verify no user is logged in.
        """
        self.assertEqual(get_session_dict_user(self.client.session), user_id)

    def assert_message_stream_name(self, message, stream_name):
        self.assertEqual(message.recipient.type, Recipient.STREAM)
        stream_id = message.recipient.type_id
        stream = Stream.objects.get(id=stream_id)
        self.assertEqual(stream.recipient_id, message.recipient_id)
        self.assertEqual(stream.name, stream_name)

    def webhook_fixture_data(self, type, action, file_type='json'):
        fn = os.path.join(os.path.dirname(__file__), f'../webhooks/{type}/fixtures/{action}.{file_type}')
        with open(fn) as f:
            return f.read()

    def fixture_file_name(self, file_name, type=''):
        return os.path.join(os.path.dirname(__file__), f'../tests/fixtures/{type}/{file_name}')

    def fixture_data(self, file_name, type=''):
        fn = self.fixture_file_name(file_name, type)
        with open(fn) as f:
            return f.read()

    def make_stream(self, stream_name, realm=None, invite_only=False, is_web_public=False, history_public_to_subscribers=None):
        if realm is None:
            realm = get_realm('zulip')
        history_public_to_subscribers = get_default_value_for_history_public_to_subscribers(realm, invite_only, history_public_to_subscribers)
        try:
            stream = Stream.objects.create(realm=realm, name=stream_name, invite_only=invite_only, is_web_public=is_web_public, history_public_to_subscribers=history_public_to_subscribers, **get_default_values_for_stream_permission_group_settings(realm))
        except IntegrityError:
            raise Exception(f'\n                {stream_name} already exists\n\n                Please call make_stream with a stream name\n                that is not already in use.')
        recipient = Recipient.objects.create(type_id=stream.id, type=Recipient.STREAM)
        stream.recipient = recipient
        stream.save(update_fields=['recipient'])
        return stream
    INVALID_STREAM_ID = 999999

    def get_stream_id(self, name, realm=None):
        if not realm:
            realm = get_realm('zulip')
        try:
            stream = get_realm_stream(name, realm.id)
        except Stream.DoesNotExist:
            return self.INVALID_STREAM_ID
        return stream.id

    def subscribe(self, user_profile, stream_name, invite_only=False, is_web_public=False):
        realm = user_profile.realm
        try:
            stream = get_stream(stream_name, user_profile.realm)
        except Stream.DoesNotExist:
            stream, from_stream_creation = create_stream_if_needed(realm, stream_name, invite_only=invite_only, is_web_public=is_web_public, acting_user=user_profile)
        bulk_add_subscriptions(realm, [stream], [user_profile], acting_user=None)
        return stream

    def unsubscribe(self, user_profile, stream_name):
        realm = user_profile.realm
        stream = get_stream(stream_name, user_profile.realm)
        bulk_remove_subscriptions(realm, [user_profile], [stream], acting_user=None)

    def subscribe_via_post(self, user, subscriptions_raw, extra_post_data={}, invite_only=False, is_web_public=False, allow_fail=False, **extra):
        subscriptions = []
        for entry in subscriptions_raw:
            if isinstance(entry, str):
                subscriptions.append({'name': entry})
            else:
                subscriptions.append(entry)
        post_data = {'subscriptions': orjson.dumps(subscriptions).decode(), 'is_web_public': orjson.dumps(is_web_public).decode(), 'invite_only': orjson.dumps(invite_only).decode()}
        post_data.update(extra_post_data)
        with transaction.atomic(savepoint=True):
            result = self.api_post(user, '/api/v1/users/me/subscriptions', post_data, intentionally_undocumented=False, **extra)
        if not allow_fail:
            self.assert_json_success(result)
        return result

    def subscribed_stream_name_list(self, user):
        subscribed_streams = gather_subscriptions(user)[0]
        return ''.join(sorted((f'        * {stream['name']}\n' for stream in subscribed_streams)))

    def check_user_subscribed_only_to_streams(self, user_name, streams):
        stream_names = {stream.name for stream in streams}
        subscribed_streams = gather_subscriptions(self.nonreg_user(user_name))[0]
        self.assertEqual(stream_names, {stream['name'] for stream in subscribed_streams})

    def resolve_topic_containing_message(self, acting_user, target_message_id, **extra):
        """
        Mark all messages within the topic associated with message `target_message_id` as resolved.
        """
        message = access_message(acting_user, target_message_id)
        return self.api_patch(acting_user, f'/api/v1/messages/{target_message_id}', {'topic': RESOLVED_TOPIC_PREFIX + message.topic_name(), 'propagate_mode': 'change_all'}, **extra)

    def send_webhook_payload(self, user_profile, url, payload, **extra):
        """
        Send a webhook payload to the server, and verify that the
        post is successful.

        This is a pretty low-level function.  For most use cases
        see the helpers that call this function, which do additional
        checks.

        Occasionally tests will call this directly, for unique
        situations like having multiple messages go to a stream,
        where the other helper functions are a bit too rigid,
        and you'll want the test itself do various assertions.
        Even in those cases, you're often better to simply
        call client_post and assert_json_success.

        If the caller expects a message to be sent to a stream,
        the caller should make sure the user is subscribed.
        """
        prior_msg = self.get_last_message()
        result = self.client_post(url, payload, skip_user_agent=False, follow=False, secure=False, headers=None, intentionally_undocumented=False, **extra)
        self.assert_json_success(result)
        msg = self.get_last_message()
        if msg.id == prior_msg.id:
            raise EmptyResponseError('\n                Your test code called an endpoint that did\n                not write any new messages.  It is probably\n                broken (but still returns 200 due to exception\n                handling).\n\n                One possible gotcha is that you forgot to\n                subscribe the test user to the stream that\n                the webhook sends to.\n                ')
        self.assertEqual(msg.sender.email, user_profile.email)
        return msg

    def get_last_message(self):
        return Message.objects.latest('id')

    def get_second_to_last_message(self):
        return Message.objects.all().order_by('-id')[1]

    @contextmanager
    def simulated_markdown_failure(self):
        """
        This raises a failure inside of the try/except block of
        markdown.__init__.do_convert.
        """
        with mock.patch('zerver.lib.markdown.unsafe_timeout', side_effect=subprocess.CalledProcessError(1, [])), self.assertLogs(level='ERROR'):
            yield

    def create_default_device(self, user_profile, number='+12125550100'):
        phone_device = PhoneDevice(user=user_profile, name='default', confirmed=True, number=number, key='abcd', method='sms')
        phone_device.save()

    def rm_tree(self, path):
        if os.path.exists(path):
            shutil.rmtree(path)

    def make_import_output_dir(self, exported_from):
        output_dir = tempfile.mkdtemp(dir=settings.TEST_WORKER_DIR, prefix='test-' + exported_from + '-import-')
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def get_set(self, data, field):
        values = {r[field] for r in data}
        return values

    def find_by_id(self, data, db_id):
        [r] = (r for r in data if r['id'] == db_id)
        return r

    def init_default_ldap_database(self):
        """
        Takes care of the mock_ldap setup, loads
        a directory from zerver/tests/fixtures/ldap/directory.json with various entries
        to be used by tests.
        If a test wants to specify its own directory, it can just replace
        self.mock_ldap.directory with its own content, but in most cases it should be
        enough to use change_user_attr to make simple modifications to the pre-loaded
        directory. If new user entries are needed to test for some additional unusual
        scenario, it's most likely best to add that to directory.json.
        """
        directory = orjson.loads(self.fixture_data('directory.json', type='ldap'))
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

    def change_ldap_user_attr(self, username, attr_name, attr_value, binary=False):
        """
        Method for changing the value of an attribute of a user entry in the mock
        directory. Use option binary=True if you want binary data to be loaded
        into the attribute from a file specified at attr_value. This changes
        the attribute only for the specific test function that calls this method,
        and is isolated from other tests.
        """
        dn = f'uid={username},ou=users,dc=zulip,dc=com'
        if binary:
            with open(attr_value, 'rb') as f:
                data = f.read()
        else:
            data = attr_value
        self.mock_ldap.directory[dn][attr_name] = [data]

    def remove_ldap_user_attr(self, username, attr_name):
        """
        Method for removing the value of an attribute of a user entry in the mock
        directory. This changes the attribute only for the specific test function
        that calls this method, and is isolated from other tests.
        """
        dn = f'uid={username},ou=users,dc=zulip,dc=com'
        self.mock_ldap.directory[dn].pop(attr_name, None)

    def ldap_username(self, username):
        """
        Maps Zulip username to the name of the corresponding LDAP user
        in our test directory at zerver/tests/fixtures/ldap/directory.json,
        if the LDAP user exists.
        """
        return self.example_user_ldap_username_map[username]

    def ldap_password(self, uid):
        return f'{uid}_ldap_password'

    def email_display_from(self, email_message):
        """
        Returns the email address that will show in email clients as the
        "From" field.
        """
        return email_message.extra_headers.get('From', email_message.from_email)

    def email_envelope_from(self, email_message):
        """
        Returns the email address that will be used if the email bounces.
        """
        return email_message.from_email

    def subscribe_realm_to_manual_license_management_plan(self, realm, licenses, licenses_at_next_renewal, billing_schedule):
        customer, _ = Customer.objects.get_or_create(realm=realm)
        plan = CustomerPlan.objects.create(customer=customer, automanage_licenses=False, billing_cycle_anchor=timezone_now(), billing_schedule=billing_schedule, tier=CustomerPlan.TIER_CLOUD_STANDARD)
        ledger = LicenseLedger.objects.create(plan=plan, is_renewal=True, event_time=timezone_now(), licenses=licenses, licenses_at_next_renewal=licenses_at_next_renewal)
        realm.plan_type = Realm.PLAN_TYPE_STANDARD
        realm.save(update_fields=['plan_type'])
        return (plan, ledger)

    def subscribe_realm_to_monthly_plan_on_manual_license_management(self, realm, licenses, licenses_at_next_renewal):
        return self.subscribe_realm_to_manual_license_management_plan(realm, licenses, licenses_at_next_renewal, CustomerPlan.BILLING_SCHEDULE_MONTHLY)

    def create_user_notifications_data_object(self, *, user_id, **kwargs):
        return UserMessageNotificationsData(user_id=user_id, online_push_enabled=kwargs.get('online_push_enabled', False), dm_email_notify=kwargs.get('dm_email_notify', False), dm_push_notify=kwargs.get('dm_push_notify', False), mention_email_notify=kwargs.get('mention_email_notify', False), mention_push_notify=kwargs.get('mention_push_notify', False), topic_wildcard_mention_email_notify=kwargs.get('topic_wildcard_mention_email_notify', False), topic_wildcard_mention_push_notify=kwargs.get('topic_wildcard_mention_push_notify', False), stream_wildcard_mention_email_notify=kwargs.get('stream_wildcard_mention_email_notify', False), stream_wildcard_mention_push_notify=kwargs.get('stream_wildcard_mention_push_notify', False), stream_email_notify=kwargs.get('stream_email_notify', False), stream_push_notify=kwargs.get('stream_push_notify', False), followed_topic_email_notify=kwargs.get('followed_topic_email_notify', False), followed_topic_push_notify=kwargs.get('followed_topic_push_notify', False), topic_wildcard_mention_in_followed_topic_email_notify=kwargs.get('topic_wildcard_mention_in_followed_topic_email_notify', False), topic_wildcard_mention_in_followed_topic_push_notify=kwargs.get('topic_wildcard_mention_in_followed_topic_push_notify', False), stream_wildcard_mention_in_followed_topic_email_notify=kwargs.get('stream_wildcard_mention_in_followed_topic_email_notify', False), stream_wildcard_mention_in_followed_topic_push_notify=kwargs.get('stream_wildcard_mention_in_followed_topic_push_notify', False), sender_is_muted=kwargs.get('sender_is_muted', False), disable_external_notifications=kwargs.get('disable_external_notifications', False))

    def get_maybe_enqueue_notifications_parameters(self, *, message_id, user_id, acting_user_id, **kwargs):
        """
        Returns a dictionary with the passed parameters, after filling up the
        missing data with default values, for testing what was passed to the
        `maybe_enqueue_notifications` method.
        """
        user_notifications_data = self.create_user_notifications_data_object(user_id=user_id, **kwargs)
        return dict(user_notifications_data=user_notifications_data, message_id=message_id, acting_user_id=acting_user_id, mentioned_user_group_id=kwargs.get('mentioned_user_group_id'), idle=kwargs.get('idle', True), already_notified=kwargs.get('already_notified', {'email_notified': False, 'push_notified': False}))

    def verify_emoji_code_foreign_keys(self):
        """
        DB tables that refer to RealmEmoji use int(emoji_code) as the
        foreign key. Those tables tend to de-normalize emoji_name due
        to our inheritance-based setup. This helper makes sure those
        invariants are intact, which is particularly tricky during
        the import/export process (or during conversions from things
        like Slack/RocketChat/MatterMost/etc.).
        """
        dct = {}
        for realm_emoji in RealmEmoji.objects.all():
            dct[realm_emoji.id] = realm_emoji
        if not dct:
            raise AssertionError('test needs RealmEmoji rows')
        count = 0
        for reaction in Reaction.objects.filter(reaction_type=Reaction.REALM_EMOJI):
            realm_emoji_id = int(reaction.emoji_code)
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

    def check_user_added_in_system_group(self, user):
        user_group = get_system_user_group_for_user(user)
        self.assertTrue(UserGroupMembership.objects.filter(user_profile=user, user_group=user_group).exists())

    def _assert_long_term_idle(self, user):
        if not user.long_term_idle:
            raise AssertionError('\n                We expect you to explicitly call self.soft_deactivate_user\n                if your user is not already soft-deactivated.\n            ')

    def expect_soft_reactivation(self, user, action):
        self._assert_long_term_idle(user)
        action()
        user.refresh_from_db()
        self.assertEqual(user.long_term_idle, False)

    def expect_to_stay_long_term_idle(self, user, action):
        self._assert_long_term_idle(user)
        action()
        user.refresh_from_db()
        self.assertEqual(user.long_term_idle, True)

    def soft_deactivate_user(self, user):
        do_soft_deactivate_users([user])
        assert user.long_term_idle

    def set_up_db_for_testing_user_access(self):
        polonius = self.example_user('polonius')
        hamlet = self.example_user('hamlet')
        othello = self.example_user('othello')
        iago = self.example_user('iago')
        prospero = self.example_user('prospero')
        aaron = self.example_user('aaron')
        zoe = self.example_user('ZOE')
        shiva = self.example_user('shiva')
        realm = get_realm('zulip')
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
        members_group = NamedUserGroup.objects.get(name='role:members', realm=realm)
        do_change_realm_permission_group_setting(realm, 'can_access_all_users_group', members_group, acting_user=None)

    def create_or_update_anonymous_group_for_setting(self, direct_members, direct_subgroups, existing_setting_group=None):
        realm = get_realm('zulip')
        if existing_setting_group is not None:
            existing_setting_group.direct_members.set(direct_members)
            existing_setting_group.direct_subgroups.set(direct_subgroups)
            return existing_setting_group
        user_group = UserGroup.objects.create(realm=realm)
        user_group.direct_members.set(direct_members)
        user_group.direct_subgroups.set(direct_subgroups)
        return user_group

    @contextmanager
    def thumbnail_formats(self, *thumbnail_formats):
        with mock.patch('zerver.lib.thumbnail.THUMBNAIL_OUTPUT_FORMATS', thumbnail_formats), mock.patch('zerver.views.upload.THUMBNAIL_OUTPUT_FORMATS', thumbnail_formats):
            yield

    def create_attachment_helper(self, user):
        with tempfile.NamedTemporaryFile() as attach_file:
            attach_file.write(b'Hello, World!')
            attach_file.flush()
            with open(attach_file.name, 'rb') as fp:
                file_path = upload_message_attachment_from_request(UploadedFile(fp), user)[0]
                return file_path

class ZulipTestCase(ZulipTestCaseMixin, TestCase):

    @contextmanager
    def capture_send_event_calls(self, expected_num_events):
        lst = []
        with mock.patch('zerver.tornado.event_queue.process_notification', lst.append), self.captureOnCommitCallbacks(execute=True):
            yield lst
        self.assert_length(lst, expected_num_events)

    @override
    def send_personal_message(self, from_user, to_user, content='test content', *, read_by_sender=True, skip_capture_on_commit_callbacks=False):
        """This function is a wrapper on 'send_personal_message',
        defined in 'ZulipTestCaseMixin' with an extra parameter
        'skip_capture_on_commit_callbacks'.

        It should be set to 'True' when making a call with either
        'verify_action' or 'capture_send_event_calls' as context manager
        because they already have 'self.captureOnCommitCallbacks'
        (See the comment in 'capture_send_event_calls').

        For all other cases, we should call 'send_personal_message' with
        'self.captureOnCommitCallbacks' for 'send_event_on_commit' or/and
        'queue_event_on_commit' to work.
        """
        if skip_capture_on_commit_callbacks:
            message_id = super().send_personal_message(from_user, to_user, content, read_by_sender=read_by_sender)
        else:
            with self.captureOnCommitCallbacks(execute=True):
                message_id = super().send_personal_message(from_user, to_user, content, read_by_sender=read_by_sender)
        return message_id

    @override
    def send_group_direct_message(self, from_user, to_users, content='test content', *, read_by_sender=True, skip_capture_on_commit_callbacks=False):
        """This function is a wrapper on 'send_group_direct_message',
        defined in 'ZulipTestCaseMixin' with an extra parameter
        'skip_capture_on_commit_callbacks'.

        It should be set to 'True' when making a call with either
        'verify_action' or 'capture_send_event_calls' as context manager
        because they already have 'self.captureOnCommitCallbacks'
        (See the comment in 'capture_send_event_calls').

        For all other cases, we should call 'send_group_direct_message' with
        'self.captureOnCommitCallbacks' for 'send_event_on_commit' or/and
        'queue_event_on_commit' to work.
        """
        if skip_capture_on_commit_callbacks:
            message_id = super().send_group_direct_message(from_user, to_users, content, read_by_sender=read_by_sender)
        else:
            with self.captureOnCommitCallbacks(execute=True):
                message_id = super().send_group_direct_message(from_user, to_users, content, read_by_sender=read_by_sender)
        return message_id

    @override
    def send_stream_message(self, sender, stream_name, content='test content', topic_name='test', recipient_realm=None, *, allow_unsubscribed_sender=False, read_by_sender=True, skip_capture_on_commit_callbacks=False):
        """This function is a wrapper on 'send_stream_message',
        defined in 'ZulipTestCaseMixin' with an extra parameter
        'skip_capture_on_commit_callbacks'.

        It should be set to 'True' when making a call with either
        'verify_action' or 'capture_send_event_calls' as context manager
        because they already have 'self.captureOnCommitCallbacks'
        (See the comment in 'capture_send_event_calls').

        For all other cases, we should call 'send_stream_message' with
        'self.captureOnCommitCallbacks' for 'send_event_on_commit' or/and
        'queue_event_on_commit' to work.
        """
        if skip_capture_on_commit_callbacks:
            message_id = super().send_stream_message(sender, stream_name, content, topic_name, recipient_realm, allow_unsubscribed_sender=allow_unsubscribed_sender, read_by_sender=read_by_sender)
        else:
            with self.captureOnCommitCallbacks(execute=True):
                message_id = super().send_stream_message(sender, stream_name, content, topic_name, recipient_realm, allow_unsubscribed_sender=allow_unsubscribed_sender, read_by_sender=read_by_sender)
        return message_id

    def upload_image(self, image_name):
        with get_test_image_file(image_name) as image_file:
            response = self.assert_json_success(self.client_post('/json/user_uploads', {'file': image_file}))
            return re.sub('/user_uploads/', '', response['url'])

    def upload_and_thumbnail_image(self, image_name):
        with self.captureOnCommitCallbacks(execute=True):
            return self.upload_image(image_name)

    def handle_missedmessage_emails(self, user_profile_id, message_ids):
        with self.captureOnCommitCallbacks(execute=True):
            handle_missedmessage_emails(user_profile_id, message_ids)

def get_row_ids_in_all_tables():
    all_models = apps.get_models(include_auto_created=True)
    ignored_tables = {'django_session'}
    for model in all_models:
        table_name = model._meta.db_table
        if table_name in ignored_tables:
            continue
        ids = model._default_manager.all().values_list('id', flat=True)
        yield (table_name, set(ids))

class ZulipTransactionTestCase(ZulipTestCaseMixin, TransactionTestCase):
    """The default Django TestCase wraps each test in a transaction. This
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
    def setUp(self):
        super().setUp()
        self.models_ids_set = dict(get_row_ids_in_all_tables())

    @override
    def tearDown(self):
        """Verifies that the test did not adjust the set of rows in the test
        database. This is a sanity check to help ensure that tests
        using this class do not have unintended side effects on the
        test database.
        """
        super().tearDown()
        for table_name, ids in get_row_ids_in_all_tables():
            self.assertSetEqual(self.models_ids_set[table_name], ids, f'{table_name} got a different set of ids after this test')

    def _fixture_teardown(self):
        """We override the default _fixture_teardown method defined on
        TransactionTestCase, so that the test database does not get
        flushed/deleted after each test using this class.
        """

class WebhookTestCase(ZulipTestCase):
    """Shared test class for all incoming webhooks tests.

    Used by configuring the below class attributes, and calling
    send_and_test_message in individual tests.

    * Tests can override build_webhook_url if the webhook requires a
      different URL format.

    * Tests can override get_body for cases where there is no
      available fixture file.

    * Tests should specify WEBHOOK_DIR_NAME to enforce that all event
      types are declared in the @webhook_view decorator. This is
      important for ensuring we document all fully supported event types.
    """
    CHANNEL_NAME = None
    TEST_USER_EMAIL = 'webhook-bot@zulip.com'
    WEBHOOK_DIR_NAME = None
    VIEW_FUNCTION_NAME = None

    @property
    def test_user(self):
        return self.get_user_from_email(self.TEST_USER_EMAIL, get_realm('zulip'))

    @override
    def setUp(self):
        super().setUp()
        self.url = self.build_webhook_url()
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

            def side_effect(*args, **kwargs):
                complete_event_type = kwargs.get('complete_event_type') if len(args) < 5 else args[4]
                if complete_event_type is not None and all_event_types is not None and (complete_event_type not in all_event_types):
                    raise Exception(f'\nError: This test triggered a message using the event "{complete_event_type}", which was not properly\nregistered via the @webhook_view(..., event_types=[...]). These registrations are important for Zulip\nself-documenting the supported event types for this integration.\n\nYou can fix this by adding "{complete_event_type}" to ALL_EVENT_TYPES for this webhook.\n'.strip())
                check_send_webhook_message(*args, **kwargs)
            self.patch = mock.patch(f'zerver.webhooks.{self.WEBHOOK_DIR_NAME}.view.check_send_webhook_message', side_effect=side_effect)
            self.patch.start()
            self.addCleanup(self.patch.stop)

    def api_channel_message(self, user, fixture_name, expected_topic=None, expected_message=None, content_type='application/json', expect_noop=False, **extra):
        extra['HTTP_AUTHORIZATION'] = self.encode_user(user)
        self.check_webhook(fixture_name, expected_topic, expected_message, content_type, expect_noop, **extra)

    def check_webhook(self, fixture_name, expected_topic_name=None, expected_message=None, content_type='application/json', expect_noop=False, **extra):
        """
        check_webhook is the main way to test "normal" webhooks that
        work by receiving a payload from a third party and then writing
        some message to a Zulip stream.

        We use `fixture_name` to find the payload data in of our test
        fixtures.  Then we verify that a message gets sent to a stream:

            self.CHANNEL_NAME: stream name
            expected_topic_name: topic name
            expected_message: content

        We simulate the delivery of the payload with `content_type`,
        and you can pass other headers via `extra`.

        For the rare cases of webhooks actually sending direct messages,
        see send_and_test_private_message.

        When no message is expected to be sent, set `expect_noop` to True.
        """
        assert self.CHANNEL_NAME is not None
        self.subscribe(self.test_user, self.CHANNEL_NAME)
        payload = self.get_payload(fixture_name)
        if content_type is not None:
            extra['content_type'] = content_type
        if self.WEBHOOK_DIR_NAME is not None:
            headers = get_fixture_http_headers(self.WEBHOOK_DIR_NAME, fixture_name)
            headers = standardize_headers(headers)
            extra.update(headers)
        try:
            msg = self.send_webhook_payload(self.test_user, self.url, payload, **extra)
        except EmptyResponseError:
            if expect_noop:
                return
            else:
                raise AssertionError('No message was sent. Pass expect_noop=True if this is intentional.')
        if expect_noop:
            raise Exception('\nWhile no message is expected given expect_noop=True,\nyour test code triggered an endpoint that did write\none or more new messages.\n'.strip())
        assert expected_message is not None and expected_topic_name is not None
        self.assert_channel_message(message=msg, channel_name=self.CHANNEL_NAME, topic_name=expected_topic_name, content=expected_message)

    def assert_channel_message(self, message, channel_name, topic_name, content):
        self.assert_message_stream_name(message, channel_name)
        self.assertEqual(message.topic_name(), topic_name)
        self.assertEqual(message.content, content)

    def send_and_test_private_message(self, fixture_name, expected_message, content_type='application/json', *, sender=None, **extra):
        """
        For the rare cases that you are testing a webhook that sends
        direct messages, use this function.

        Most webhooks send to streams, and you will want to look at
        check_webhook.
        """
        payload = self.get_payload(fixture_name)
        extra['content_type'] = content_type
        if self.WEBHOOK_DIR_NAME is not None:
            headers = get_fixture_http_headers(self.WEBHOOK_DIR_NAME, fixture_name)
            headers = standardize_headers(headers)
            extra.update(headers)
        if sender is None:
            sender = self.test_user
        msg = self.send_webhook_payload(sender, self.url, payload, **extra)
        self.assertEqual(msg.content, expected_message)
        return msg

    def build_webhook_url(self, *args, **kwargs):
        url = self.URL_TEMPLATE
        if url.find('api_key') >= 0:
            api_key = self.test_user.api_key
            url = self.URL_TEMPLATE.format(api_key=api_key, stream=self.CHANNEL_NAME)
        else:
            url = self.URL_TEMPLATE.format(stream=self.CHANNEL_NAME)
        has_arguments = kwargs or args
        if has_arguments and url.find('?') == -1:
            url = f'{url}?'
        else:
            url = f'{url}&'
        for key, value in kwargs.items():
            url = f'{url}{key}={value}&'
        for arg in args:
            url = f'{url}{arg}&'
        return url[:-1] if has_arguments else url

    def get_payload(self, fixture_name):
        """
        Generally webhooks that override this should return dicts."""
        return self.get_body(fixture_name)

    def get_body(self, fixture_name):
        assert self.WEBHOOK_DIR_NAME is not None
        body = self.webhook_fixture_data(self.WEBHOOK_DIR_NAME, fixture_name)
        orjson.loads(body)
        return body

class MigrationsTestCase(ZulipTransactionTestCase):
    """
    Test class for database migrations inspired by this blog post:
       https://www.caktusgroup.com/blog/2016/02/02/writing-unit-tests-django-migrations/
    Documented at https://zulip.readthedocs.io/en/latest/subsystems/schema-migrations.html
    """

    @property
    def app(self):
        app_config = apps.get_containing_app_config(type(self).__module__)
        assert app_config is not None
        return app_config.name
    migrate_from = None
    migrate_to = None

    @override
    def setUp(self):
        super().setUp()
        assert self.migrate_from and self.migrate_to, f"TestCase '{type(self).__name__}' must define migrate_from and migrate_to properties"
        migrate_from = [(self.app, self.migrate_from)]
        migrate_to = [(self.app, self.migrate_to)]
        executor = MigrationExecutor(connection)
        old_apps = executor.loader.project_state(migrate_from).apps
        executor.migrate(migrate_from)
        self.setUpBeforeMigration(old_apps)
        executor = MigrationExecutor(connection)
        executor.loader.build_graph()
        executor.migrate(migrate_to)
        self.apps = executor.loader.project_state(migrate_to).apps

    def setUpBeforeMigration(self, apps):
        pass

def get_topic_messages(user_profile, stream, topic_name):
    query = UserMessage.objects.filter(user_profile=user_profile, message__recipient=stream.recipient).order_by('id')
    return [um.message for um in filter_by_topic_name_via_message(query, topic_name)]

@skipUnless(settings.ZILENCER_ENABLED, 'requires zilencer')
class BouncerTestCase(ZulipTestCase):

    @override
    def setUp(self):
        self.server_uuid = '6cde5f7a-1f7e-4978-9716-49f69ebfc9fe'
        self.server = RemoteZulipServer.objects.all().latest('id')
        self.server.uuid = self.server_uuid
        self.server.hostname = 'demo.example.com'
        self.server.save()
        super().setUp()

    @override
    def tearDown(self):
        RemoteZulipServer.objects.filter(uuid=self.server_uuid).delete()
        super().tearDown()

    def request_callback(self, request):
        kwargs = {}
        if isinstance(request.body, bytes):
            data = orjson.loads(request.body)
            kwargs = dict(content_type='application/json')
        else:
            assert isinstance(request.body, str) or request.body is None
            params = parse_qs(request.body)
            data = {k: v[0] for k, v in params.items()}
        assert request.url is not None
        assert settings.ZULIP_SERVICES_URL is not None
        local_url = request.url.replace(settings.ZULIP_SERVICES_URL, '')
        if request.method == 'POST':
            result = self.uuid_post(self.server_uuid, local_url, data, subdomain='', **kwargs)
        elif request.method == 'GET':
            result = self.uuid_get(self.server_uuid, local_url, data, subdomain='')
        return (result.status_code, result.headers, result.content)

    def add_mock_response(self):
        assert settings.ZULIP_SERVICES_URL is not None
        COMPILED_URL = re.compile(settings.ZULIP_SERVICES_URL + '.*')
        responses.add_callback(responses.POST, COMPILED_URL, callback=self.request_callback)
        responses.add_callback(responses.GET, COMPILED_URL, callback=self.request_callback)

    def get_generic_payload(self, method='register'):
        user_id = 10
        token = '111222'
        token_kind = PushDeviceToken.FCM
        return {'user_id': user_id, 'token': token, 'token_kind': token_kind}