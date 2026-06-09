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
def log_to_file(logger, filename, log_format=...): ...

# === Internal dependency: zerver.lib.request ===
class RequestVariableConversionError(JsonableError): ...

# === Internal dependency: zerver.models ===
from zerver.models.messages import ArchivedAttachment as ArchivedAttachment
from zerver.models.messages import ArchivedReaction as ArchivedReaction
from zerver.models.messages import ArchivedSubMessage as ArchivedSubMessage
from zerver.models.messages import ArchivedUserMessage as ArchivedUserMessage
from zerver.models.messages import ArchiveTransaction as ArchiveTransaction
from zerver.models.messages import Attachment as Attachment
from zerver.models.messages import Message as Message
from zerver.models.messages import Reaction as Reaction
from zerver.models.messages import SubMessage as SubMessage
from zerver.models.messages import UserMessage as UserMessage
from zerver.models.realms import Realm as Realm
from zerver.models.recipients import Recipient as Recipient
from zerver.models.streams import Stream as Stream