import collections
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import IO, Any, TypeVar, Union
from unittest import mock

import fakeldap
import ldap
from django.http import HttpRequest, HttpResponse, HttpResponseRedirect
from django.http.request import QueryDict
from django.http.response import HttpResponseBase
from django.test import override_settings
from mypy_boto3_s3.service_resource import Bucket
from typing_extensions import ParamSpec, override

from zerver.models import Client, Message, Subscription, UserMessage, UserProfile
from zerver.tornado.handlers import AsyncDjangoHandler
from zilencer.models import RemoteZulipServer
from zproject.backends import ExternalAuthDataDict

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from django.test.client import _MonkeyPatchedWSGIResponse as TestHttpResponse
    from zerver.lib.test_classes import MigrationsTestCase, ZulipTestCase
    from django.db.migrations.state import StateApps


class MockLDAP(fakeldap.MockLDAP):
    class LDAPError(ldap.LDAPError): ...
    class INVALID_CREDENTIALS(ldap.INVALID_CREDENTIALS): ...
    class NO_SUCH_OBJECT(ldap.NO_SUCH_OBJECT): ...
    class ALREADY_EXISTS(ldap.ALREADY_EXISTS): ...


@contextmanager
def stub_event_queue_user_events(
    event_queue_return: Any, user_events_return: Any
) -> Iterator[None]: ...


class activate_push_notification_service(override_settings):
    def __init__(
        self,
        zulip_services_url: str | None = ...,
        submit_usage_statistics: bool = ...,
    ) -> None: ...


@dataclass
class CapturedQuery:
    sql: str
    time: str


@contextmanager
def cache_tries_captured() -> Iterator[list[tuple[str, Any, str | None]]]: ...

@contextmanager
def simulated_empty_cache() -> Iterator[list[tuple[str, Any, str | None]]]: ...

@contextmanager
def queries_captured(
    include_savepoints: bool = ..., keep_cache_warm: bool = ...
) -> Iterator[list[CapturedQuery]]: ...

@contextmanager
def stdout_suppressed() -> Iterator[Any]: ...

def reset_email_visibility_to_everyone_in_zulip_realm() -> None: ...
def get_test_image_file(filename: str) -> IO[bytes]: ...
def read_test_image_file(filename: str) -> bytes: ...
def avatar_disk_path(
    user_profile: UserProfile, medium: bool = ..., original: bool = ...
) -> str: ...
def make_client(name: str) -> Client: ...
def find_key_by_email(address: str) -> str | None: ...
def message_stream_count(user_profile: UserProfile) -> int: ...
def most_recent_usermessage(user_profile: UserProfile) -> UserMessage: ...
def most_recent_message(user_profile: UserProfile) -> Message: ...
def get_subscription(stream_name: str, user_profile: UserProfile) -> Subscription: ...
def get_user_messages(user_profile: UserProfile) -> list[Message]: ...


class DummyHandler(AsyncDjangoHandler):
    def __init__(self) -> None: ...

dummy_handler: DummyHandler


class HostRequestMock(HttpRequest):
    host: str
    GET: QueryDict
    POST: QueryDict
    META: dict[str, Any]
    method: str
    path: str
    user: UserProfile | Any
    content_type: str

    def __init__(
        self,
        post_data: dict[str, Any] = ...,
        user_profile: UserProfile | None = ...,
        remote_server: RemoteZulipServer | None = ...,
        host: str = ...,
        client_name: str | None = ...,
        meta_data: dict[str, Any] | None = ...,
        tornado_handler: DummyHandler | None = ...,
        path: str = ...,
    ) -> None: ...

    @override
    def get_host(self) -> str: ...


INSTRUMENTING: bool
INSTRUMENTED_CALLS: list[dict[str, Any]]
UrlFuncT = TypeVar("UrlFuncT", bound=Callable[..., HttpResponseBase])

def append_instrumentation_data(data: dict[str, Any]) -> None: ...
def instrument_url(f: UrlFuncT) -> UrlFuncT: ...
def write_instrumentation_reports(full_suite: bool, include_webhooks: bool) -> None: ...
def load_subdomain_token(response: HttpResponse) -> ExternalAuthDataDict: ...

P = ParamSpec("P")

def use_s3_backend(method: Callable[P, Any]) -> Callable[P, Any]: ...
def create_s3_buckets(*bucket_names: str) -> list[Bucket]: ...

TestCaseT = TypeVar("TestCaseT", bound="MigrationsTestCase")

def use_db_models(
    method: Callable[..., Any],
) -> Callable[..., Any]: ...
def create_dummy_file(filename: str) -> str: ...
def zulip_reaction_info() -> dict[str, str]: ...

@contextmanager
def mock_queue_publish(
    method_to_patch: str, **kwargs: Any
) -> Iterator[mock.MagicMock]: ...

@contextmanager
def ratelimit_rule(
    range_seconds: int, num_requests: int, domain: str = ...
) -> Iterator[None]: ...

def consume_response(response: HttpResponseBase) -> None: ...