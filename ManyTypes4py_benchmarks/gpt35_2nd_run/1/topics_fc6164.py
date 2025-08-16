import abc
import asyncio
from typing import Any, Mapping, Optional, Pattern, Sequence, Set, Union
from mode import Seconds
from mode.utils.queues import ThrowableQueue
from .channels import ChannelT
from .codecs import CodecArg
from .tuples import TP

if typing.TYPE_CHECKING:
    from .app import AppT as _AppT
    from .models import ModelArg as _ModelArg
    from .serializers import SchemaT as _SchemaT
else:
    class _AppT:
        ...

    class _ModelArg:
        ...

    class _SchemaT:
        ...

__all__: Sequence[str] = ['TopicT']

class TopicT(ChannelT):
    has_prefix: bool = False

    @abc.abstractmethod
    def __init__(self, app: '_AppT', *, topics: Optional[Sequence[str]] = None, pattern: Optional[Pattern[str]] = None, schema: Optional['_SchemaT'] = None, key_type: Optional[Any] = None, value_type: Optional[Any] = None, is_iterator: bool = False, partitions: Optional[int] = None, retention: Optional[Seconds] = None, compacting: Optional[bool] = None, deleting: Optional[bool] = None, replicas: Optional[int] = None, acks: bool = True, internal: bool = False, config: Optional[Mapping[str, Any]] = None, queue: Optional[ThrowableQueue] = None, key_serializer: Optional[CodecArg] = None, value_serializer: Optional[CodecArg] = None, maxsize: Optional[int] = None, root: Optional[str] = None, active_partitions: Optional[Set[int]] = None, allow_empty: bool = False, has_prefix: bool = False, loop: Optional[asyncio.AbstractEventLoop] = None):
        ...

    @property
    @abc.abstractmethod
    def pattern(self) -> Pattern[str]:
        ...

    @pattern.setter
    def pattern(self, pattern: Pattern[str]):
        ...

    @property
    @abc.abstractmethod
    def partitions(self) -> int:
        ...

    @partitions.setter
    def partitions(self, partitions: int):
        ...

    @abc.abstractmethod
    def derive(self, **kwargs: Any):
        ...

    @abc.abstractmethod
    def derive_topic(self, *, topics: Optional[Sequence[str]] = None, schema: Optional['_SchemaT'] = None, key_type: Optional[Any] = None, value_type: Optional[Any] = None, partitions: Optional[int] = None, retention: Optional[Seconds] = None, compacting: Optional[bool] = None, deleting: Optional[bool] = None, internal: bool = False, config: Optional[Mapping[str, Any]] = None, prefix: str = '', suffix: str = '', **kwargs: Any):
        ...
