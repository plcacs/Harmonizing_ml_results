from typing import Any

# === Third-party dependency: django.conf ===
settings: LazySettings

# === Third-party dependency: django.db ===
connection: ConnectionProxy

# === Third-party dependency: django.utils.connection ===
class ConnectionProxy: ...

# === Third-party dependency: django.utils.timezone ===
def now() -> Any: ...

# === Third-party dependency: psycopg2.sql ===
class SQL(Composable):
    def __init__(self, string) -> Any: ...
class Identifier(Composable): ...
class Literal(Composable):
    ...

# === Internal dependency: zerver.lib.logging_util ===
def log_to_file(logger: Logger, filename: str, log_format: str = ...) -> None: ...

# === Internal dependency: zerver.lib.request ===
class RequestVariableConversionError(JsonableError): ...

# === Internal dependency: zerver.models ===
# re-export: from zerver.models.messages import ArchivedAttachment as ArchivedAttachment
# re-export: from zerver.models.messages import ArchivedReaction as ArchivedReaction
# re-export: from zerver.models.messages import ArchivedSubMessage as ArchivedSubMessage
# re-export: from zerver.models.messages import ArchivedUserMessage as ArchivedUserMessage
# re-export: from zerver.models.messages import ArchiveTransaction as ArchiveTransaction
# re-export: from zerver.models.messages import Attachment as Attachment
# re-export: from zerver.models.messages import Message as Message
# re-export: from zerver.models.messages import Reaction as Reaction
# re-export: from zerver.models.messages import SubMessage as SubMessage
# re-export: from zerver.models.messages import UserMessage as UserMessage
# re-export: from zerver.models.realms import Realm as Realm
# re-export: from zerver.models.recipients import Recipient as Recipient
# re-export: from zerver.models.streams import Stream as Stream