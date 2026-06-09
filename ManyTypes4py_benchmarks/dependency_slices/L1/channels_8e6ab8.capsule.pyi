from typing import Any

# === Internal dependency: faust.types ===
from .app import AppT
from .channels import ChannelT
from .events import EventT
from .models import ModelArg
from .serializers import SchemaT
from .streams import StreamT
from .tuples import FutureMessage
from .tuples import Message
from .tuples import PendingMessage
from .tuples import RecordMetadata
from .tuples import TP

# === Internal dependency: faust.types.core ===
OpenHeaders = Union[List[Tuple[str, bytes]], MutableMapping[str, bytes]]
OpenHeadersArg = Optional[OpenHeaders]

# === Internal dependency: faust.types.tuples ===
def _PendingMessage_to_Message(p): ...

# === Third-party dependency: mode ===
# Used symbols: get_logger, want_seconds

# === Third-party dependency: mode.utils.futures ===
class stampede: ...
async def maybe_async(res: Any) -> Any: ...

# === Third-party dependency: mode.utils.queues ===
class ThrowableQueue(FlowControlQueue): ...