from __future__ import annotations
from collections.abc import Iterator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import (
    Any,
    Callable,
    cast,
    ContextManager,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    ParamSpec,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)
from unittest.mock import MagicMock
from django.http import (
    HttpResponseRedirect,
    HttpResponse,
    HttpRequest,
    HttpResponseBase,
    QueryDict,
)
from django.test.client import _MonkeyPatchedWSGIResponse
from django.db.models import Model
from zerver.models import (
    Client,
    Message,
    RealmUserDefault,
    Subscription,
    UserMessage,
    UserProfile,
    Stream,
    Recipient,
)
from zerver.lib.events import (
    EventQueue,
    UserEvents,
)
from zerver.lib.types import AnalyticsDataUploadLevel
from zerver.lib.test_classes import MigrationsTestCase, ZulipTestCase
from zproject.backends import ExternalAuthDataDict, ExternalAuthResult
from zerver.tornado.handlers import AsyncDjangoHandler
from moto.core.decorator import mock_aws
from mypy_boto3_s3.service_resource import Bucket
from zilencer.models import RemoteZulipServer
from zerver.lib.avatar import avatar_url
from zerver.lib.cache import cache_get, cache_get_many
from zerver.lib.db import Query, TimeTrackingCursor
from zerver.lib.integrations import WEBHOOK_INTEGRATIONS
from zerver.lib.request import RequestNotes
from zerver.lib.types import AnalyticsDataUploadLevel
from zerver.lib.upload.s3 import S3UploadBackend
from zerver.models.clients import Client
from zerver.models.realms import Realm
from zerver.models.streams import Stream
from typing_extensions import ParamSpec, override
from zerver.lib.events import request_event_queue, get_user_events

class MockLDAP:
    class LDAPError(Exception): ...
    class INVALID_CREDENTIALS(Exception): ...
    class NO_SUCH_OBJECT(Exception): ...
    class ALREADY_EXISTS(Exception): ...

@contextmanager
def stub_event_queue_user_events(event_queue_return: EventQueue, user_events_return: UserEvents) -> ContextManager[None]:
    ...

class activate_push_notification_service(override_settings):
    def __init__(self, zulip_services_url: str = ..., submit_usage_statistics: bool = ...) -> None: ...

@contextmanager
def cache_tries_captured() -> Iterator[List[Tuple[str, Any, Optional[str]]]]:
    ...

@contextmanager
def simulated_empty_cache() -> Iterator[List[Tuple[str, Any, Optional[str]]]]:
    ...

@dataclass
class CapturedQuery:
    sql: str
    time: str

@contextmanager
def queries_captured(include_savepoints: bool = ..., keep_cache_warm: bool = ...) -> Iterator[List[CapturedQuery]]:
    ...

@contextmanager
def stdout_suppressed() -> Iterator[Any]:
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

class DummyHandler(AsyncDjangoHandler):
    def __init__(self) -> None:
        ...

dummy_handler: DummyHandler = ...

class HostRequestMock(HttpRequest):
    def __init__(self, post_data: Dict[str, str] = ..., user_profile: Optional[UserProfile] = ..., remote_server: Optional[RemoteZulipServer] = ..., host: str = ..., client_name: Optional[str] = ..., meta_data: Optional[Dict[str, Any]] = ..., tornado_handler: Optional[AsyncDjangoHandler] = ..., path: str = ...) -> None:
        ...

    @override
    def get_host(self) -> str:
        ...

INSTRUMENTING: bool = ...
INSTRUMENTED_CALLS: List[Dict[str, Any]] = ...
UrlFuncT = TypeVar('UrlFuncT', bound=Callable[..., HttpResponseBase])

def append_instrumentation_data(data: Dict[str, Any]) -> None:
    ...

def instrument_url(f: Callable[..., HttpResponseBase]) -> UrlFuncT:
    ...

def write_instrumentation_reports(full_suite: bool, include_webhooks: bool) -> None:
    ...

def load_subdomain_token(response: HttpResponseRedirect) -> ExternalAuthResult:
    ...

P = ParamSpec('P')

def use_s3_backend(method: Callable[P, Any]) -> Callable[P, Any]:
    ...

def create_s3_buckets(*bucket_names: str) -> List[Bucket]:
    ...

TestCaseT = TypeVar('TestCaseT', bound=MigrationsTestCase)

def use_db_models(method: Callable[[TestCaseT, StateApps], Any]) -> Callable[[TestCaseT, StateApps], Any]:
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