from __future__ import annotations
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import (
    Any,
    AnyStr,
    Callable,
    ContextManager,
    Dict,
    IO,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    ParamSpec,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)
from unittest.mock import MagicMock
from zerver.lib.types import AnalyticsDataUploadLevel
from zerver.models import (
    Client,
    Message,
    RealmUserDefault,
    Subscription,
    UserMessage,
    UserProfile,
)
from django.conf import settings
from django.http import (
    HttpRequest,
    HttpResponse,
    HttpResponseRedirect,
    HttpResponseBase,
    QueryDict,
)
from django.test.client import _MonkeyPatchedWSGIResponse as TestHttpResponse
from django.test import override_settings
from django.urls import URLResolver
from moto.core.decorator import mock_aws
from mypy_boto3_s3.service_resource import Bucket
from zproject.backends import ExternalAuthDataDict, ExternalAuthResult
from zerver.lib.cache import cache_get, cache_get_many
from zerver.lib.request import RequestNotes
from zerver.tornado.handlers import AsyncDjangoHandler
from zilencer.models import RemoteZulipServer
from zerver.lib.integrations import WEBHOOK_INTEGRATIONS

P = ParamSpec("P")
TestCaseT = TypeVar("TestCaseT", bound="MigrationsTestCase")

class MockLDAP:
    class LDAPError(Exception):
        pass

    class INVALID_CREDENTIALS(Exception):
        pass

    class NO_SUCH_OBJECT(Exception):
        pass

    class ALREADY_EXISTS(Exception):
        pass

class CapturedQuery:
    sql: str
    time: str

class DummyHandler(AsyncDjangoHandler):
    pass

dummy_handler: DummyHandler

class HostRequestMock(HttpRequest):
    def __init__(
        self,
        post_data: Dict[str, str] = ...,
        user_profile: Optional[UserProfile] = ...,
        remote_server: Optional[RemoteZulipServer] = ...,
        host: str = ...,
        client_name: Optional[str] = ...,
        meta_data: Optional[Dict[str, Any]] = ...,
        tornado_handler: Optional[DummyHandler] = ...,
        path: str = ...,
    ):
        ...

    def get_host(self) -> str:
        ...

INSTRUMENTING: bool
INSTRUMENTED_CALLS: List[Dict[str, Any]]
UrlFuncT = TypeVar("UrlFuncT", bound=Callable[..., HttpResponseBase])

def append_instrumentation_data(data: Dict[str, Any]) -> None:
    ...

def instrument_url(f: Callable[P, HttpResponseBase]) -> Callable[P, HttpResponseBase]:
    ...

def write_instrumentation_reports(full_suite: bool, include_webhooks: bool) -> None:
    ...

def load_subdomain_token(response: HttpResponseRedirect) -> ExternalAuthDataDict:
    ...

def use_s3_backend(method: Callable[..., Any]) -> Callable[..., Any]:
    ...

def create_s3_buckets(*bucket_names: str) -> List[Bucket]:
    ...

def use_db_models(method: Callable[..., Any]) -> Callable[..., Any]:
    ...

def create_dummy_file(filename: str) -> str:
    ...

def zulip_reaction_info() -> Dict[str, str]:
    ...

@contextmanager
def mock_queue_publish(method_to_patch: str, **kwargs: Any) -> Iterator[MagicMock]:
    ...

@contextmanager
def ratelimit_rule(range_seconds: int, num_requests: int, domain: str = ...) -> Iterator[None]:
    ...

def consume_response(response: HttpResponse) -> None:
    ...

def reset_email_visibility_to_everyone_in_zulip_realm() -> None:
    ...

def get_test_image_file(filename: str) -> IO[bytes]:
    ...

def read_test_image_file(filename: str) -> bytes:
    ...

def avatar_disk_path(user_profile: UserProfile, medium: bool = ..., original: bool = ...) -> str:
    ...

def make_client(name: str) -> Client:
    ...

def find_key_by_email(address: str) -> Optional[str]:
    ...

def message_stream_count(user_profile: UserProfile) -> int:
    ...

def most_recent_usermessage(user_profile: UserProfile) -> UserMessage:
    ...

def most_recent_message(user_profile: UserProfile) -> Message:
    ...

def get_subscription(stream_name: str, user_profile: UserProfile) -> Subscription:
    ...

def get_user_messages(user_profile: UserProfile) -> List[Message]:
    ...

@contextmanager
def stub_event_queue_user_events(
    event_queue_return: Any,
    user_events_return: Any,
) -> Iterator[None]:
    ...

class activate_push_notification_service(override_settings):
    def __init__(
        self,
        zulip_services_url: Optional[str] = ...,
        submit_usage_statistics: bool = ...,
    ) -> None:
        ...

@contextmanager
def cache_tries_captured() -> Iterator[List[Tuple[str, Any, Optional[str]]]]:
    ...

@contextmanager
def simulated_empty_cache() -> Iterator[List[Tuple[str, Any, Optional[str]]]]:
    ...

@contextmanager
def queries_captured(
    include_savepoints: bool = ...,
    keep_cache_warm: bool = ...,
) -> Iterator[List[CapturedQuery]]:
    ...

@contextmanager
def stdout_suppressed() -> Iterator[sys.stdout]:
    ...