from typing import Any

# === Internal dependency: faust.types ===
# re-export: from .app import AppT
# re-export: from .channels import ChannelT
# re-export: from .events import EventT
# re-export: from .models import ModelArg
# re-export: from .serializers import SchemaT
# re-export: from .streams import StreamT
# re-export: from .tuples import FutureMessage
# re-export: from .tuples import Message
# re-export: from .tuples import PendingMessage
# re-export: from .tuples import RecordMetadata
# re-export: from .tuples import TP

# === Internal dependency: faust.types.core ===
OpenHeadersArg: Any

# === Internal dependency: faust.types.tuples ===
def _PendingMessage_to_Message(p: PendingMessage) -> 'Message': ...

# === Third-party dependency: mode ===
# Used symbols: get_logger, want_seconds

# === Third-party dependency: mode.utils.futures ===
class stampede: ...
async def maybe_async(res: Any) -> Any: ...

# === Third-party dependency: mode.utils.queues ===
class ThrowableQueue(FlowControlQueue): ...