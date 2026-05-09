import collections
import itertools
import os
import re
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    Any,
    IO,
    Optional,
    TypeVar,
    Union,
    cast,
    Dict,
    List,
    Tuple,
    AnyStr,
    Iterable as TIterable,
    Iterator as TIterator,
    Optional as TOptional,
    Callable as TCallable,
    Sequence,
)
from unittest import mock
from unittest.mock import MagicMock
from zerver.models import (
    Client,
    Message,
    RealmUserDefault,
    Subscription,
    UserMessage,
    UserProfile,
)
from zerver.models.clients import get_client
from zerver.models.streams import get_stream
from zerver.tornado.handlers import AsyncDjangoHandler
from zerver.lib.events import request_event_queue, get_user_events
from zerver.lib.cache import cache_get, cache_get_many
from zerver.lib.db import TimeTrackingCursor
from zerver.lib.upload.s3 import S3UploadBackend
from boto3.session import Session
from mypy_boto3_s3.service_resource import Bucket
from zerver.lib.test_classes import MigrationsTestCase
from zerver.lib.types import AnalyticsDataUploadLevel
from django.conf import settings
from django.contrib.auth.models import AnonymousUser
from django.http import HttpRequest, HttpResponseRedirect, HttpResponseBase
from django.test.client import _MonkeyPatchedWSGIResponse
from zilencer.models import RemoteZulipServer
from zerver.actions.realm_settings import do_set_realm_user_default_setting
from zerver.actions.user_settings import do_change_user_setting

P = TypeVar("P")
TestCaseT = TypeVar("TestCaseT", bound=MigrationsTestCase)

class MockLDAP:
    class LDAPError(Exception):
        pass

    class INVALID_CREDENTIALS(Exception):
        pass

    class NO_SUCH_OBJECT(Exception):
        pass

    class ALREADY_EXISTS(Exception):
        pass

@contextmanager
def stub_event_queue_user_events(event_queue_return: Any, user_events_return: Any) -> Iterator[None]:
    ...

class activate_push_notification_service:
    def __init__(self, zulip_services_url: str = ..., submit_usage_statistics: bool = ...) -> None:
        ...

@contextmanager
def cache_tries_captured() -> Iterator[List[Tuple[str, Any, TOptional[str]]]]:
    ...

@contextmanager
def simulated_empty_cache() -> Iterator[List[Tuple[str, Any, TOptional[str]]]]:
    ...

@dataclass
class CapturedQuery:
    sql: str
    time: str

@contextmanager
def queries_captured(include_savepoints: bool = ..., keep_cache_warm: bool = ...) -> Iterator[List[CapturedQuery]]:
    ...

@contextmanager
def stdout_suppressed() -> Iterator[IO[Any]]:
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

def find_key_by_email(address: str) -> TOptional[str]:
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
    def __init__(self, post_data: Dict[str, str] = ..., user_profile: TOptional[UserProfile] = ..., remote_server: TOptional[RemoteZulipServer] = ..., host: str = ..., client_name: TOptional[str] = ..., meta_data: TOptional[Dict[str, Any]] = ..., tornado_handler: TOptional[AsyncDjangoHandler] = ..., path: str = ...) -> None:
        ...

    def get_host(self) -> str:
        ...

INSTRUMENTING: bool = ...
INSTRUMENTED_CALLS: List[Dict[str, Any]] = ...
UrlFuncT = TypeVar("UrlFuncT", bound=Callable[..., HttpResponseBase])

def append_instrumentation_data(data: Dict[str, Any]) -> None:
    ...

def instrument_url(f: UrlFuncT) -> UrlFuncT:
    ...

def write_instrumentation_reports(full_suite: bool, include_webhooks: bool) -> None:
    ...

def load_subdomain_token(response: HttpResponseRedirect) -> Dict[str, Any]:
    ...

def use_s3_backend(method: Callable[P, Any]) -> Callable[P, Any]:
    ...

def create_s3_buckets(*bucket_names: str) -> List[Bucket]:
    ...

def use_db_models(method: Callable[[MigrationsTestCase, StateApps], Any]) -> Callable[[MigrationsTestCase, StateApps], Any]:
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