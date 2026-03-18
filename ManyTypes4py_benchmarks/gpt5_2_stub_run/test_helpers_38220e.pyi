from typing import Any, Callable, Iterable, Iterator, Mapping, TypeVar, IO, Optional, List, Dict, Generator
from contextlib import contextmanager
from django.http import HttpRequest, HttpResponseRedirect, HttpResponseBase
from django.http.request import QueryDict
from django.test import override_settings
from mypy_boto3_s3.service_resource import Bucket
from typing_extensions import ParamSpec
from unittest import mock
import fakeldap
import ldap
from zerver.tornado.handlers import AsyncDjangoHandler
from zerver.models import Client, Message, Subscription, UserMessage, UserProfile
from zproject.backends import ExternalAuthDataDict

class MockLDAP(fakeldap.MockLDAP):
    class LDAPError(ldap.LDAPError): ...
    class INVALID_CREDENTIALS(ldap.INVALID_CREDENTIALS): ...
    class NO_SUCH_OBJECT(ldap.NO_SUCH_OBJECT): ...
    class ALREADY_EXISTS(ldap.ALREADY_EXISTS): ...

@contextmanager
def stub_event_queue_user_events(event_queue_return: Any, user_events_return: Any) -> Generator[None, None, None]: ...

class activate_push_notification_service(override_settings):
    def __init__(self, zulip_services_url: Optional[str] = ..., submit_usage_statistics: bool = ...) -> None: ...

@contextmanager
def cache_tries_captured() -> Generator[List[Any], None, None]: ...

@contextmanager
def simulated_empty_cache() -> Generator[List[Any], None, None]: ...

class CapturedQuery:
    sql: Any
    time: Any
    ...

@contextmanager
def queries_captured(include_savepoints: bool = ..., keep_cache_warm: bool = ...) -> Generator[List[CapturedQuery], None, None]: ...

@contextmanager
def stdout_suppressed() -> Generator[IO[str], None, None]: ...

def reset_email_visibility_to_everyone_in_zulip_realm() -> None: ...

def get_test_image_file(filename: str) -> IO[bytes]: ...

def read_test_image_file(filename: str) -> bytes: ...

def avatar_disk_path(user_profile: UserProfile, medium: bool = ..., original: bool = ...) -> str: ...

def make_client(name: str) -> Client: ...

def find_key_by_email(address: str) -> Optional[str]: ...

def message_stream_count(user_profile: UserProfile) -> int: ...

def most_recent_usermessage(user_profile: UserProfile) -> UserMessage: ...

def most_recent_message(user_profile: UserProfile) -> Message: ...

def get_subscription(stream_name: str, user_profile: UserProfile) -> Subscription: ...

def get_user_messages(user_profile: UserProfile) -> List[Message]: ...

class DummyHandler(AsyncDjangoHandler):
    handler_id: Any
    def __init__(self) -> None: ...

dummy_handler: DummyHandler

class HostRequestMock(HttpRequest):
    host: str
    GET: QueryDict
    method: str
    POST: QueryDict
    META: Dict[str, Any]
    path: str
    user: Any
    _body: bytes
    content_type: str
    def __init__(
        self,
        post_data: Dict[str, Any] = ...,
        user_profile: Any = ...,
        remote_server: Any = ...,
        host: str = ...,
        client_name: Optional[str] = ...,
        meta_data: Optional[Dict[str, Any]] = ...,
        tornado_handler: Any = ...,
        path: str = ...
    ) -> None: ...
    def get_host(self) -> str: ...

INSTRUMENTING: bool
INSTRUMENTED_CALLS: List[Dict[str, Any]]

UrlFuncT = TypeVar("UrlFuncT", bound=Callable[..., HttpResponseBase])

def append_instrumentation_data(data: Any) -> None: ...

def instrument_url(f: UrlFuncT) -> UrlFuncT: ...

def write_instrumentation_reports(full_suite: bool, include_webhooks: bool) -> None: ...

def load_subdomain_token(response: HttpResponseRedirect) -> ExternalAuthDataDict: ...

P = ParamSpec("P")

def use_s3_backend(method: Callable[P, Any]) -> Callable[P, Any]: ...

def create_s3_buckets(*bucket_names: str) -> List[Bucket]: ...

TestCaseT = TypeVar("TestCaseT", bound="MigrationsTestCase")

def use_db_models(method: Callable[..., Any]) -> Callable[..., Any]: ...

def create_dummy_file(filename: str) -> str: ...

def zulip_reaction_info() -> Dict[str, str]: ...

@contextmanager
def mock_queue_publish(method_to_patch: str, **kwargs: Any) -> Generator[mock.MagicMock, None, None]: ...

@contextmanager
def ratelimit_rule(range_seconds: int, num_requests: int, domain: str = ...) -> Generator[None, None, None]: ...

def consume_response(response: Iterable[Any]) -> None: ...