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
    class _AppT: ...             # noqa
    class _ModelArg: ...         # noqa
    class _SchemaT: ...          # noqa

__all__ = ['TopicT']


class TopicT(ChannelT):

    topics: Sequence[str]
    retention: Optional[Seconds]
    compacting: Optional[bool]
    deleting: Optional[bool]
    replicas: Optional[int]
    config: Optional[Mapping[str, Any]]
    acks: bool
    internal: bool
    has_prefix: bool = False
    active_partitions: Optional[Set[TP]]

    @abc.abstractmethod
    def __init__(self,
                 app: _AppT,
                 *,
                 topics: Sequence[str] = None,
                 pattern: Union[str, Pattern] = None,
                 schema: _SchemaT = None,
                 key_type: _ModelArg = None,
                 value_type: _ModelArg = None,
                 is_iterator: bool = False,
                 partitions: int = None,
                 retention: Seconds = None,
                 compacting: bool = None,
                 deleting: bool = None,
                 replicas: int = None,
                 acks: bool = True,
                 internal: bool = False,
                 config: Mapping[str, Any] = None,
                 queue: ThrowableQueue = None,
                 key_serializer: CodecArg = None,
                 value_serializer: CodecArg = None,
                 maxsize: int = None,
                 root: ChannelT = None,
                 active_partitions: Set[TP] = None,
                 allow_empty: bool = False,
                 has_prefix: bool = False,
                 loop: asyncio.AbstractEventLoop = None) -> None:
        ...

    @property
    @abc.abstractmethod
    def pattern(self) -> Optional[Pattern]:
        ...

    @pattern.setter
    def pattern(self, pattern: Union[str, Pattern]) -> None:
        ...

    @property
    @abc.abstractmethod
    def partitions(self) -> Optional[int]:
        ...

    @partitions.setter
    def partitions(self, partitions: int) -> None:
        ...

    @abc.abstractmethod
    def derive(self, **kwargs: Any) -> ChannelT:
        ...

    @abc.abstractmethod
    def derive_topic(self,
                     *,
                     topics: Sequence[str] = None,
                     schema: _SchemaT = None,
                     key_type: _ModelArg = None,
                     value_type: _ModelArg = None,
                     partitions: int = None,
                     retention: Seconds = None,
                     compacting: bool = None,
                     deleting: bool = None,
                     internal: bool = False,
                     config: Mapping[str, Any] = None,
                     prefix: str = '',
                     suffix: str = '',
                     **kwargs: Any) -> 'TopicT':
        ...
