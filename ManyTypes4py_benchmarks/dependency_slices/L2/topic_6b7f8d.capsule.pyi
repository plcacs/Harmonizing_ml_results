from typing import Any

# === Third-party dependency: django.db ===
connection: ConnectionProxy

# === Third-party dependency: django.db.models ===
# Used symbols: F, Func, JSONField, Q, QuerySet, Subquery, TextField, Value

# === Third-party dependency: django.db.models.functions ===
# Used symbols: Cast

# === Third-party dependency: django.utils.connection ===
class ConnectionProxy: ...

# === Third-party dependency: django.utils.translation ===
def gettext(message) -> Any: ...
class override(ContextDecorator): ...

# === Third-party dependency: orjson ===
# Used symbols: dumps, loads

# === Internal dependency: zerver.lib.message ===
def bulk_access_stream_messages_query(user_profile: UserProfile, messages: QuerySet[Message], stream: Stream) -> QuerySet[Message]: ...

# === Internal dependency: zerver.lib.types ===
class EditHistoryEvent(TypedDict): ...

# === Internal dependency: zerver.lib.utils ===
def assert_is_not_none(value: T | None) -> T: ...

# === Internal dependency: zerver.models ===
# re-export: from zerver.models.messages import Message as Message
# re-export: from zerver.models.messages import Reaction as Reaction
# re-export: from zerver.models.messages import UserMessage as UserMessage
# re-export: from zerver.models.users import UserProfile as UserProfile