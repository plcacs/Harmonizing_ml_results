```python
import collections.abc
import sys
from _typeshed import Incomplete
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import IO, Any, ClassVar, ContextManager, Optional, TypeVar, Union, overload
from unittest.mock import MagicMock, Mock
from django.conf import settings
from django.contrib.auth.models import AnonymousUser
from django.http import HttpRequest, HttpResponse, HttpResponseRedirect
from django.http.request import QueryDict
from django.http.response import HttpResponseBase
from django.test import override_settings
from moto.core.decorator import mock_aws
from mypy_boto3_s3.service_resource import Bucket
from typing_extensions import ParamSpec, override
from zerver.lib.types import AnalyticsDataUploadLevel
from zerver.models import Client, Message, RealmUserDefault, Subscription, UserMessage, UserProfile
from zilencer.models import RemoteZulipServer
from zproject.backends import ExternalAuthDataDict, ExternalAuthResult

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

_T = TypeVar("_T")
_P = ParamSpec("_P")
TestCaseT = TypeVar("TestCaseT", bound="MigrationsTestCase")
UrlFuncT = TypeVar("UrlFuncT", bound=Callable[..., HttpResponseBase])

INSTRUMENTING: bool = ...
INSTRUMENTED_CALLS: list[dict[str, Any]] = ...

class MockLDAP:
    class LDAPError: ...
    class INVALID_CREDENTIALS: ...
    class NO_SUCH_OBJECT: ...
    class ALREADY_EXISTS: ...

def stub_event_queue_user_events(
    event_queue_return: Any, user_events_return: Any
) -> ContextManager[None]: ...

class activate_push_notification_service(override_settings):
    def __init__(
        self,
        zulip_services_url: Optional[str] = ...,
        submit_usage_statistics: bool = ...,
    ) -> None: ...

def cache_tries_captured() -> ContextManager[list[tuple[str, Any, Optional[str]]]]: ...

def simulated_empty_cache() -> ContextManager[list[tuple[str, Any, Optional[str]]]]: ...

@dataclass
class CapturedQuery:
    sql: str
    time: str

def queries_captured(
    include_savepoints: bool = ...,
    keep_cache_warm: bool = ...,
) -> ContextManager[list[CapturedQuery]]: ...

def stdout_suppressed() -> ContextManager[IO[Any]]: ...

def reset_email_visibility_to_everyone_in_zulip_realm() -> None: ...

def get_test_image_file(filename: str) -> IO[bytes]: ...

def read_test_image_file(filename: str) -> bytes: ...

def avatar_disk_path(
    user_profile: UserProfile,
    medium: bool = ...,
    original: bool = ...,
) -> str: ...

def make_client(name: str) -> Client: ...

def find_key_by_email(address: str) -> Optional[str]: ...

def message_stream_count(user_profile: UserProfile) -> int: ...

def most_recent_usermessage(user_profile: UserProfile) -> UserMessage: ...

def most_recent_message(user_profile: UserProfile) -> Message: ...

def get_subscription(stream_name: str, user_profile: UserProfile) -> Subscription: ...

def get_user_messages(user_profile: UserProfile) -> list[Message]: ...

class DummyHandler:
    handler_id: int
    def __init__(self) -> None: ...

dummy_handler: DummyHandler = ...

class HostRequestMock(HttpRequest):
    host: str
    GET: QueryDict
    method: str
    POST: QueryDict
    META: dict[str, Any]
    path: str
    user: Union[UserProfile, AnonymousUser]
    _body: bytes
    content_type: str
    
    def __init__(
        self,
        post_data: dict[str, Any] = ...,
        user_profile: Optional[UserProfile] = ...,
        remote_server: Optional[RemoteZulipServer] = ...,
        host: str = ...,
        client_name: Optional[str] = ...,
        meta_data: Optional[dict[str, Any]] = ...,
        tornado_handler: Optional[Any] = ...,
        path: str = ...,
    ) -> None: ...
    
    @override
    def get_host(self) -> str: ...

def append_instrumentation_data(data: dict[str, Any]) -> None: ...

def instrument_url(f: UrlFuncT) -> UrlFuncT: ...

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
) -> ContextManager[MagicMock]: ...

def ratelimit_rule(
    range_seconds: int,
    num_requests: int,
    domain: str = ...,
) -> ContextManager[None]: ...

def consume_response(response: HttpResponse) -> None: ...
```