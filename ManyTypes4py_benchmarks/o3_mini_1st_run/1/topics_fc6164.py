import abc
import asyncio
import re
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
    def __init__(
        self,
        app: "_AppT",
        *,
        topics: Optional[Union[str, Sequence[str]]] = None,
        pattern: Optional[Union[str, Pattern]] = None,
        schema: Optional["_SchemaT"] = None,
        key_type: Optional[Any] = None,
        value_type: Optional[Any] = None,
        is_iterator: bool = False,
        partitions: Optional[Union[int, Sequence[int]]] = None,
        retention: Optional[Seconds] = None,
        compacting: Optional[bool] = None,
        deleting: Optional[bool] = None,
        replicas: Optional[int] = None,
        acks: bool = True,
        internal: bool = False,
        config: Optional[Mapping[str, Any]] = None,
        queue: Optional[ThrowableQueue] = None,
        key_serializer: Optional[CodecArg] = None,
        value_serializer: Optional[CodecArg] = None,
        maxsize: Optional[int] = None,
        root: Optional[Any] = None,
        active_partitions: Optional[Set[Any]] = None,
        allow_empty: bool = False,
        has_prefix: bool = False,
        loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> None:
        ...

    @property
    @abc.abstractmethod
    def pattern(self) -> Optional[Union[str, Pattern]]:
        ...

    @pattern.setter
    def pattern(self, pattern: Optional[Union[str, Pattern]]) -> None:
        ...

    @property
    @abc.abstractmethod
    def partitions(self) -> Optional[Union[int, Sequence[int]]]:
        ...

    @partitions.setter
    def partitions(self, partitions: Optional[Union[int, Sequence[int]]]) -> None:
        ...

    @abc.abstractmethod
    def derive(self, **kwargs: Any) -> "TopicT":
        ...

    @abc.abstractmethod
    def derive_topic(
        self,
        *,
        topics: Optional[Union[str, Sequence[str]]] = None,
        schema: Optional["_SchemaT"] = None,
        key_type: Optional[Any] = None,
        value_type: Optional[Any] = None,
        partitions: Optional[Union[int, Sequence[int]]] = None,
        retention: Optional[Seconds] = None,
        compacting: Optional[bool] = None,
        deleting: Optional[bool] = None,
        internal: bool = False,
        config: Optional[Mapping[str, Any]] = None,
        prefix: str = '',
        suffix: str = '',
        **kwargs: Any
    ) -> "TopicT":
        ...