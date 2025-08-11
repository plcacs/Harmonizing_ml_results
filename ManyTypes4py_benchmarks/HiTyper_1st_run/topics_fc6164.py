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
    has_prefix = False

    @abc.abstractmethod
    def __init__(self, app: Union[types.topics.ChannelT, bool, models.ModelArg], *, topics: Union[None, types.topics.ChannelT, bool, models.ModelArg]=None, pattern: Union[None, types.topics.ChannelT, bool, models.ModelArg]=None, schema: Union[None, types.topics.ChannelT, bool, models.ModelArg]=None, key_type: Union[None, types.topics.ChannelT, bool, models.ModelArg]=None, value_type: Union[None, types.topics.ChannelT, bool, models.ModelArg]=None, is_iterator: bool=False, partitions: Union[None, types.topics.ChannelT, bool, models.ModelArg]=None, retention: Union[None, types.topics.ChannelT, bool, models.ModelArg]=None, compacting: Union[None, types.topics.ChannelT, bool, models.ModelArg]=None, deleting: Union[None, types.topics.ChannelT, bool, models.ModelArg]=None, replicas: Union[None, types.topics.ChannelT, bool, models.ModelArg]=None, acks: bool=True, internal: bool=False, config: Union[None, types.topics.ChannelT, bool, models.ModelArg]=None, queue: Union[None, types.topics.ChannelT, bool, models.ModelArg]=None, key_serializer: Union[None, types.topics.ChannelT, bool, models.ModelArg]=None, value_serializer: Union[None, types.topics.ChannelT, bool, models.ModelArg]=None, maxsize: Union[None, types.topics.ChannelT, bool, models.ModelArg]=None, root: Union[None, types.topics.ChannelT, bool, models.ModelArg]=None, active_partitions: Union[None, types.topics.ChannelT, bool, models.ModelArg]=None, allow_empty: bool=False, has_prefix: bool=False, loop: Union[None, types.topics.ChannelT, bool, models.ModelArg]=None) -> None:
        ...

    @property
    @abc.abstractmethod
    def pattern(self) -> None:
        ...

    @pattern.setter
    def pattern(self, pattern) -> None:
        ...

    @property
    @abc.abstractmethod
    def partitions(self) -> None:
        ...

    @partitions.setter
    def partitions(self, partitions) -> None:
        ...

    @abc.abstractmethod
    def derive(self, **kwargs) -> None:
        ...

    @abc.abstractmethod
    def derive_topic(self, *, topics: Union[None, bool, str, models.ModelArg]=None, schema: Union[None, bool, str, models.ModelArg]=None, key_type: Union[None, bool, str, models.ModelArg]=None, value_type: Union[None, bool, str, models.ModelArg]=None, partitions: Union[None, bool, str, models.ModelArg]=None, retention: Union[None, bool, str, models.ModelArg]=None, compacting: Union[None, bool, str, models.ModelArg]=None, deleting: Union[None, bool, str, models.ModelArg]=None, internal: bool=False, config: Union[None, bool, str, models.ModelArg]=None, prefix: typing.Text='', suffix: typing.Text='', **kwargs) -> None:
        ...