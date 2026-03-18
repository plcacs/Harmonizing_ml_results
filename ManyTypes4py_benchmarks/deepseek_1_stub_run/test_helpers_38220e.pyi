```python
import collections.abc
import os
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import IO, Any, TypeVar, Union, overload
from unittest.mock import MagicMock, Mock
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

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

_T = TypeVar("_T")
_P = ParamSpec("_P")
_TestCaseT = TypeVar("_TestCaseT", bound="MigrationsTestCase")
_UrlFuncT = TypeVar("_UrlFuncT", bound=Callable[..., HttpResponseBase])

class MockLDAP(fakeldap.MockLDAP):
    class LDAPError(ldap.LDAPError): ...
    class INVALID_CREDENTIALS(ldap.INVALID_CREDENTIALS): ...
    class NO_SUCH_OBJECT(ldap.NO_SUCH_OBJECT): ...
    class ALREADY_EXISTS(ldap.ALREADY_EXISTS): ...

def stub_event_queue_user_events(
    event_queue_return: Any, user_events_return: Any
) -> AbstractContextManager[None]: ...

class activate_push_notification_service(override_settings):
    def __init__(
        self,
        zulip_services_url: str | None = ...,
        submit_usage_statistics: bool = ...,
    ) -> None: ...

def cache_tries_captured() -> AbstractContextManager[list[Any]]: ...

def simulated_empty_cache() -> AbstractContextManager[list[Any]]: ...

@dataclass
class CapturedQuery:
    sql: str
    time: str

def queries_captured(
    include_savepoints: bool = ...,
    keep_cache_warm: bool = ...,
) -> AbstractContextManager[list[CapturedQuery]]: ...

def stdout_suppressed() -> AbstractContextManager[IO[Any]]: ...

def reset_email_visibility_to_everyone_in_zulip_realm() -> None: ...

def get_test_image_file(filename: str) -> IO[bytes]: ...

def read_test_image_file(filename: str) -> bytes: ...

def avatar_disk_path(
    user_profile: UserProfile,
    medium: bool = ...,
    original: bool = ...,
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

dummy_handler: DummyHandler = ...

class HostRequestMock(HttpRequest):
    def __init__(
        self,
        post_data: dict[str, Any] = ...,
        user_profile: UserProfile | None = ...,
        remote_server: RemoteZulipServer | None = ...,
        host: str = ...,
        client_name: str | None = ...,
        meta_data: dict[str, Any] | None = ...,
        tornado_handler: Any | None = ...,
        path: str = ...,
    ) -> None: ...
    @override
    def get_host(self) -> str: ...

INSTRUMENTING: bool = ...
INSTRUMENTED_CALLS: list[dict[str, Any]] = ...

def append_instrumentation_data(data: dict[str, Any]) -> None: ...

def instrument_url(f: _UrlFuncT) -> _UrlFuncT: ...

def write_instrumentation_reports(full_suite: bool, include_webhooks: bool) -> None: ...

def load_subdomain_token(response: HttpResponseRedirect) -> ExternalAuthDataDict: ...

def use_s3_backend(
    method: Callable[_P, _T]
) -> Callable[_P, _T]: ...

def create_s3_buckets(*bucket_names: str) -> list[Bucket]: ...

def use_db_models(
    method: Callable[..., Any]
) -> Callable[..., Any]: ...

def create_dummy_file(filename: str) -> str: ...

def zulip_reaction_info() -> dict[str, str]: ...

def mock_queue_publish(
    method_to_patch: str, **kwargs: Any
) -> AbstractContextManager[Mock]: ...

def ratelimit_rule(
    range_seconds: int,
    num_requests: int,
    domain: str = ...,
) -> AbstractContextManager[None]: ...

def consume_response(response: HttpResponse) -> None: ...
```