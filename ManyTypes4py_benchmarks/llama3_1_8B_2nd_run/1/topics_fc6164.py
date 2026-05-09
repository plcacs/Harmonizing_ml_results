import abc
import asyncio
import typing
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
__all__ = ['TopicT']

class TopicT(ChannelT):
    has_prefix: bool = False

    @abc.abstractmethod
    def __init__(self, app: _AppT, *, topics: Optional[Sequence[str]] = None, pattern: Optional[Pattern[str]] = None, schema: Optional[_SchemaT] = None, key_type: Optional[CodecArg] = None, value_type: Optional[CodecArg] = None, is_iterator: bool = False, partitions: Optional[Mapping[str, int]] = None, retention: Optional[Seconds] = None, compacting: Optional[bool] = None, deleting: Optional[bool] = None, replicas: Optional[int] = None, acks: bool = True, internal: bool = False, config: Optional[Any] = None, queue: Optional[ThrowableQueue] = None, key_serializer: Optional[CodecArg] = None, value_serializer: Optional[CodecArg] = None, maxsize: Optional[int] = None, root: Optional[Any] = None, active_partitions: Optional[Mapping[str, int]] = None, allow_empty: bool = False, has_prefix: bool = False, loop: Optional[asyncio.AbstractEventLoop] = None):
        ...

    @property
    @abc.abstractmethod
    def pattern(self) -> Optional[Pattern[str]]:
        ...

    @pattern.setter
    @abc.abstractmethod
    def pattern(self, pattern: Optional[Pattern[str]]) -> None:
        ...

    @property
    @abc.abstractmethod
    def partitions(self) -> Optional[Mapping[str, int]]:
        ...

    @partitions.setter
    @abc.abstractmethod
    def partitions(self, partitions: Optional[Mapping[str, int]]) -> None:
        ...

    @abc.abstractmethod
    def derive(self, **kwargs: Any) -> '_TopicT':
        ...

    @abc.abstractmethod
    def derive_topic(self, *, topics: Optional[Sequence[str]] = None, schema: Optional[_SchemaT] = None, key_type: Optional[CodecArg] = None, value_type: Optional[CodecArg] = None, partitions: Optional[Mapping[str, int]] = None, retention: Optional[Seconds] = None, compacting: Optional[bool] = None, deleting: Optional[bool] = None, internal: bool = False, config: Optional[Any] = None, prefix: str = '', suffix: str = '', **kwargs: Any) -> '_TopicT':
        ...
