import base64
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Union, cast, Optional, Dict, List, Set, Tuple, TypeVar
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
from zerver.lib.streams import (
    create_stream_if_needed,
    get_default_value_for_history_public_to_subscribers,
    get_default_values_for_stream_permission_group_settings,
)
from zerver.lib.subscription_info import gather_subscriptions
from zerver.lib.test_console_output import (
    ExtraConsoleOutputFinder,
    ExtraConsoleOutputInTestError,
    tee_stderr_and_find_extra_console_output,
    tee_stdout_and_find_extra_console_output,
)
from zerver.lib.test_helpers import (
    cache_tries_captured,
    find_key_by_email,
    get_test_image_file,
    instrument_url,
    queries_captured,
)
from zerver.lib.thumbnail import ThumbnailFormat
from zerver.lib.topic import RESOLVED_TOPIC_PREFIX, filter_by_topic_name_via_message
from zerver.lib.upload import upload_message_attachment_from_request
from zerver.lib.user_groups import get_system_user_group_for_user
from zerver.lib.webhooks.common import (
    check_send_webhook_message,
    get_fixture_http_headers,
    standardize_headers,
)
from zerver.models import (
    Client,
    Message,
    NamedUserGroup,
    PushDeviceToken,
    Reaction,
    Realm,
    RealmEmoji,
    Recipient,
    Stream,
    Subscription,
    UserGroup,
    UserGroupMembership,
    UserMessage,
    UserProfile,
    UserStatus,
)
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

    lockfile: str = "var/upload_lock"

    @classmethod
    @override
    def setUpClass(cls: Any) -> None:
        if not os.path.exists(cls.lockfile):
            with open(cls.lockfile, "w"):  # nocoverage - rare locking case
                pass

        super().setUpClass()


class ZulipClientHandler(ClientHandler):
    @override
    def get_response(self, request: HttpRequest) -> HttpResponseBase:
        got_exception = False

        def on_exception(**kwargs: object) -> None:
            nonlocal got_exception
            if kwargs["request"] is request:
                got_exception = True

        request.body  # noqa: B018 # prevents RawPostDataException
        got_request_exception.connect(on_exception)
        try:
            response = super().get_response(request)
        finally:
            got_request_exception.disconnect(on_exception)

        if (
            not got_exception  # Django will reraise this exception
            and request.method != "OPTIONS"
            and isinstance(response, HttpResponse)
            and not (
                response.status_code == 302 and response.headers["Location"].startswith("/login/")
            )
        ):
            openapi_request = DjangoOpenAPIRequest(request)
            openapi_response = DjangoOpenAPIResponse(response)
            response_validated = validate_test_response(openapi_request, openapi_response)
            if response_validated:
                validate_test_request(
                    openapi_request,
                    str(response.status_code),
                    request.META.get("intentionally_undocumented", False),
                )
        return response


class ZulipTestClient(TestClient):
    def __init__(self) -> None:
        super().__init__()
        self.handler = ZulipClientHandler(enforce_csrf_checks=False)


class ZulipTestCaseMixin(SimpleTestCase):
    # Ensure that the test system just shows us diffs
    maxDiff: Optional[int] = None
    # This bypasses BAN_CONSOLE_OUTPUT for the test case when set.
    # Override this to verify if the given extra console output matches the
    # expectation.
    expected_console_output: Optional[str] = None
    client_class = ZulipTestClient

    @override
    def setUp(self) -> None:
        super().setUp()
        self.API_KEYS: Dict[str, str] = {}

        test_name = self.id()
        bounce_key_prefix_for_testing(test_name)
        bounce_redis_key_prefix_for_testing(test_name)

    @override
    def tearDown(self) -> None:
        super().tearDown()
        # Important: we need to clear event queues to avoid leaking data to future tests.
        clear_client_event_queues_for_testing()
        clear_supported_auth_backends_cache()
        flush_per_request_caches()
        translation.activate(settings.LANGUAGE_CODE)

        # Clean up local uploads directory after tests:
        assert settings.LOCAL_UPLOADS_DIR is not None
        if os.path.exists(settings.LOCAL_UPLOADS_DIR):
            shutil.rmtree(settings.LOCAL_UPLOADS_DIR)

        # Clean up after using fakeldap in LDAP tests:
        if hasattr(self, "mock_ldap") and hasattr(self, "mock_initialize"):
            if self.mock_ldap is not None:
                self.mock_ldap.reset()
            self.mock_initialize.stop()

    def get_user_from_email(self, email: str, realm: Realm) -> UserProfile:
        return get_user(email, realm)

    @override
    def run(self, result: Optional[TestResult] = None) -> Optional[TestResult]:  # nocoverage
        if not settings.BAN_CONSOLE_OUTPUT and self.expected_console_output is None:
            return super().run(result)
        extra_output_finder = ExtraConsoleOutputFinder()
        with (
            tee_stderr_and_find_extra_console_output(extra_output_finder),
            tee_stdout_and_find_extra_console_output(extra_output_finder),
        ):
            test_result = super().run(result)
        if extra_output_finder.full_extra_output and (
            test_result is None or test_result.wasSuccessful()
        ):
            extra_output = extra_output_finder.full_extra_output.decode(errors="replace")
            if self.expected_console_output is not None:
                self.assertEqual(extra_output, self.expected_console_output)
                return test_result

            exception_message = f"""
---- UNEXPECTED CONSOLE OUTPUT DETECTED ----

To ensure that we never miss important error output/warnings,
we require test-backend to have clean console output.

This message usually is triggered by forgotten debugging print()
statements or new logging statements.  For the latter, you can
use `with self.assertLogs()` to capture and verify the log output;
use `git grep assertLogs` to see dozens of correct examples.

You should be able to quickly reproduce this failure with:

./tools/test-backend --ban-console-output {self.id()}

Output:
{extra_output}
--------------------------------------------
"""
            raise ExtraConsoleOutputInTestError(exception_message)
        return test_result

    """
    WRAPPER_COMMENT:

    We wrap calls to self.client.{patch,put,get,post,delete} for various
    reasons.  Some of this has to do with fixing encodings before calling
    into the Django code.  Some of this has to do with providing a future
    path for instrumentation.  Some of it's just consistency.

    The linter will prevent direct calls to self.client.foo, so the wrapper
    functions have to fake out the linter by using a local variable called
    django_client to fool the regex.
    """
    DEFAULT_SUBDOMAIN: str = "zulip"
    TOKENIZED_NOREPLY_REGEX: str = settings.TOKENIZED_NOREPLY_EMAIL_ADDRESS.format(
        token=r"[a-z0-9_]{24}"
    )

    @override
    def assertEqual(self, first: Any, second: Any, msg: Any = "") -> None:
        if isinstance(first, str) and isinstance(second, str):
            if first != second:
                raise AssertionError(
                    "Actual and expected outputs do not match; showing diff.\n"
                    + diff_strings(first, second)
                    + str(msg)
                )
        else:
            super().assertEqual(first, second, msg)

    def set_http_headers(self, extra: Dict[str, str], skip_user_agent: bool = False) -> None:
        if "subdomain" in extra:
            assert isinstance(extra["subdomain"], str)
            extra["HTTP_HOST"] = Realm.host_for_subdomain(extra["subdomain"])
            del extra["subdomain"]
        elif "HTTP_HOST" not in extra:
            extra["HTTP_HOST"] = Realm.host_for_subdomain(self.DEFAULT_SUBDOMAIN)

        # set User-Agent
        if "HTTP_AUTHORIZATION" in extra:
            # An API request; use mobile as the default user agent
            default_user_agent = "ZulipMobile/26.22.145 (iOS 10.3.1)"
        else:
            # A web app request; use a browser User-Agent string.
            default_user_agent = (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
                " AppleWebKit/537.36 (KHTML, like Gecko)"
                " Chrome/79.0.3945.130 Safari/537.36"
            )
        if skip_user_agent:
            # Provide a way to disable setting User-Agent if desired.
            assert "HTTP_USER_AGENT" not in extra
        elif "HTTP_USER_AGENT" not in extra:
            extra["HTTP_USER_AGENT"] = default_user_agent

    @instrument_url
    def client_patch(
        self,
        url: str,
        info: Mapping[str, Any] = {},
        skip_user_agent: bool = False,
        follow: bool = False,
        secure: bool = False,
        intentionally_undocumented: bool = False,
        headers: Optional[Mapping[str, Any]] = None,
        **extra: str,
    ) -> "TestHttpResponse":
        """
        We need to urlencode, since Django's function won't do it for us.
        """
        encoded = urlencode(info)
        extra["content_type"] = "application/x-www-form-urlencoded"
        django_client = self.client  # see WRAPPER_COMMENT
        self.set_http_headers(extra, skip_user_agent)
        return django_client.patch(
            url,
            encoded,
            follow=follow,
            secure=secure,
            headers=headers,
            intentionally_undocumented=intentionally_undocumented,
            **extra,
        )

    @instrument_url
    def client_patch_multipart(
        self,
        url: str,
        info: Mapping[str, Any] = {},
        skip_user_agent: bool = False,
        follow: bool = False,
        secure: bool = False,
        headers: Optional[Mapping[str, Any]] = None,
        intentionally_undocumented: bool = False,
        **extra: str,
    ) -> "TestHttpResponse":
        """
        Use this for patch requests that have file uploads or
        that need some sort of multi-part content.  In the future
        Django's test client may become a bit more flexible,
        so we can hopefully eliminate this.  (When you post
        with the Django test client, it deals with MULTIPART_CONTENT
        automatically, but not patch.)
        """
        encoded = encode_multipart(BOUNDARY, dict(info))
        django_client = self.client  # see WRAPPER_COMMENT
        self.set_http_headers(extra, skip_user_agent)
        return django_client.patch(
            url,
            encoded,
            content_type=MULTIPART_CONTENT,
            follow=follow,
            secure=secure,
            headers=headers,
            intentionally_undocumented=intentionally_undocumented,
            **extra,
        )

    def json_patch(
        self,
        url: str,
        payload: Mapping[str, Any] = {},
        skip_user_agent: bool = False,
        follow: bool = False,
        secure: bool = False,
        **extra: str,
    ) -> "TestHttpResponse":
        data = orjson.dumps(payload)
        django_client = self.client  # see WRAPPER_COMMENT
        self.set_http_headers(extra, skip_user_agent)
        return django_client.patch(
            url,
            data=data,
            content_type="application/json",
            follow=follow,
            secure=secure,
            headers=None,
            **extra,
        )

    @instrument_url
    def client_put(
        self,
        url: str,
        info: Mapping[str, Any] = {},
        skip_user_agent: bool = False,
        follow: bool = False,
        secure: bool = False,
        headers: Optional[Mapping[str, Any]] = None,
        **extra: str,
    ) -> "TestHttpResponse":
        encoded = urlencode(info)
        extra["content_type"] = "application/x-www-form-urlencoded"
        django_client = self.client  # see WRAPPER_COMMENT
        self.set_http_headers(extra, skip_user_agent)
        return django_client.put(
            url, encoded, follow=follow, secure=secure, headers=headers, **extra
        )

    def json_put(
        self,
        url: str,
        payload: Mapping[str, Any] = {},
        skip_user_agent: bool = False,
        follow: bool = False,
        secure: bool = False,
        headers: Optional[Mapping[str, Any]] = None,
        **extra: str,
    ) -> "TestHttpResponse":
        data = orjson.dumps(payload)
        django_client = self.client  # see WRAPPER_COMMENT
        self.set_http_headers(extra, skip_user_agent)
        return django_client.put(
            url,
            data=data,
            content_type="application/json",
            follow=follow,
            secure=secure,
            headers=headers,
            **extra,
        )

    @instrument_url
    def client_delete(
        self,
        url: str,
        info: Mapping[str, Any] = {},
        skip_user_agent: bool = False,
        follow: bool = False,
        secure: bool = False,
        headers: Optional[Mapping[str, Any]] = None,
        intentionally_undocumented: bool = False,
        **extra: str,
    ) -> "TestHttpResponse":
        encoded = urlencode(info)
        extra["content_type"] = "application/x-www-form-urlencoded"
        django_client = self.client  # see WRAPPER_COMMENT
        self.set_http_headers(extra, skip_user_agent)
        return django_client.delete(
            url,
            encoded,
            follow=follow,
            secure=secure,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",  # https://code.djangoproject.com/ticket/33230
                **(headers or {}),
            },
            intentionally_undocumented=intentionally_undocumented,
            **extra,
        )

    @instrument_url
    def client_options(
        self,
        url: str,
        info: Mapping[str, Any] = {},
        skip_user_agent: bool = False,
        follow: bool = False,
        secure: bool = False,
        headers: Optional[Mapping[str