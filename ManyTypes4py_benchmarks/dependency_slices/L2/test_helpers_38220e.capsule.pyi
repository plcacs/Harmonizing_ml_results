from typing import Any

# === Third-party dependency: boto3.session ===
class Session:
    def __init__(self, aws_access_key_id = ..., aws_secret_access_key = ..., aws_session_token = ..., region_name = ..., botocore_session = ..., profile_name = ..., aws_account_id = ...) -> Any: ...
    def resource(self, service_name, region_name = ..., api_version = ..., use_ssl = ..., verify = ..., endpoint_url = ..., aws_access_key_id = ..., aws_secret_access_key = ..., aws_session_token = ..., config = ...) -> Any: ...

# === Third-party dependency: django.conf ===
settings: LazySettings

# === Third-party dependency: django.contrib.auth.models ===
class AnonymousUser: ...

# === Third-party dependency: django.core.mail ===
# Used symbols: outbox

# === Third-party dependency: django.db.migrations.state ===
class StateApps(Apps): ...

# === Third-party dependency: django.http ===
# Used symbols: HttpRequest, HttpResponseRedirect

# === Third-party dependency: django.http.request ===
class QueryDict(MultiValueDict):
    def __init__(self, query_string = ..., mutable = ..., encoding = ...) -> Any: ...

# === Third-party dependency: django.http.response ===
class HttpResponseBase: ...

# === Third-party dependency: django.test ===
# Used symbols: override_settings

# === Third-party dependency: django.urls ===
# Used symbols: URLResolver

# === Unresolved dependency: fakeldap ===
# Used unresolved symbols: MockLDAP

# === Unresolved dependency: ldap ===
# Used unresolved symbols: ALREADY_EXISTS, INVALID_CREDENTIALS, LDAPError, NO_SUCH_OBJECT

# === Third-party dependency: moto.core.decorator ===
def mock_aws(func: Callable[P, T]) -> Callable[P, T]: ...
def mock_aws(func: None = ..., config: DefaultConfig | None = ...) -> MockAWS: ...
def mock_aws(func: Callable[P, T] | None = ..., config: DefaultConfig | None = ...) -> MockAWS | Callable[P, T]: ...

# === Third-party dependency: mypy_boto3_s3.service_resource ===
class Bucket(ServiceResource): ...

# === Third-party dependency: orjson ===
# Used symbols: OPT_APPEND_NEWLINE, dumps, loads

# === Internal dependency: zerver.actions.realm_settings ===
def do_set_realm_user_default_setting(realm_user_default: RealmUserDefault, name: str, value: Any, *, acting_user: UserProfile | None) -> None: ...

# === Internal dependency: zerver.actions.user_settings ===
def do_change_user_setting(user_profile: UserProfile, setting_name: str, setting_value: bool | str | int, *, acting_user: UserProfile | None) -> None: ...

# === Internal dependency: zerver.lib.avatar ===
def avatar_url(user_profile: UserProfile, medium: bool = ..., client_gravatar: bool = ...) -> str | None: ...

# === Internal dependency: zerver.lib.cache ===
def get_cache_backend(cache_name: str | None) -> BaseCache: ...
def cache_get(key: str, cache_name: str | None = ...) -> Any: ...
def cache_get_many(keys: list[str], cache_name: str | None = ...) -> dict[str, Any]: ...

# === Internal dependency: zerver.lib.db ===
class TimeTrackingCursor(cursor): ...

# === Internal dependency: zerver.lib.integrations ===
WEBHOOK_INTEGRATIONS: list[WebhookIntegration]

# === Internal dependency: zerver.lib.per_request_cache ===
def flush_per_request_caches() -> None: ...

# === Internal dependency: zerver.lib.rate_limiter ===
rules: dict[str, list[tuple[int, int]]]
class RateLimitedIPAddr(RateLimitedObject): ...

# === Internal dependency: zerver.lib.request ===
class RequestNotes(BaseNotes[HttpRequest, 'RequestNotes']): ...

# === Internal dependency: zerver.lib.types ===
class AnalyticsDataUploadLevel(IntEnum): ...

# === Internal dependency: zerver.lib.upload.s3 ===
class S3UploadBackend(ZulipUploadBackend):
    def __init__(self) -> None: ...

# === Internal dependency: zerver.models ===
# re-export: from zerver.models.clients import Client as Client
# re-export: from zerver.models.messages import Message as Message
# re-export: from zerver.models.messages import UserMessage as UserMessage
# re-export: from zerver.models.streams import Subscription as Subscription
# re-export: from zerver.models.users import RealmUserDefault as RealmUserDefault
# re-export: from zerver.models.users import UserProfile as UserProfile

# === Internal dependency: zerver.models.clients ===
def clear_client_cache() -> None: ...
def get_client(name: str) -> Client: ...

# === Internal dependency: zerver.models.realms ===
def get_realm(string_id: str) -> Realm: ...

# === Internal dependency: zerver.models.streams ===
def get_stream(stream_name: str, realm: Realm) -> Stream: ...

# === Unresolved dependency: zerver.tornado.handlers ===
# Used unresolved symbols: AsyncDjangoHandler, allocate_handler_id

# === Unresolved dependency: zproject.backends ===
# Used unresolved symbols: ExternalAuthDataDict, ExternalAuthResult

# === Unresolved dependency: zproject.urls ===
# Used unresolved symbols: urlpatterns, v1_api_and_json_patterns